"""Experiment: JEV as a gate in front of a top-1 RAG.

Claim under test: a top-1 RAG is usually fine; JEV can detect the cases where the top-1 chunk
is not enough and fix them, at a small added cost/latency on the common (fast) path.

Systems (same index, embeddings, generator and answer prompt):
  top1      FAISS top-1 -> answer.                                        (baseline)
  jev_gate  RAG.ask_gate(): JEV judges each neighbor (one call per step) and stops as soon
            as the kept chunks suffice.                                    (proposed)
  sim_gate  same idea with cosine similarity instead of JEV: fast path if top-1 similarity
            >= tau_fast, else every top-N neighbor with similarity >= tau_keep.
  topk_eq   plain top-k with k = mean chunks jev_gate examines on the tuning set.
  oracle    the gold chunks as context (upper bound; unanswerable -> empty context).

Phases:
  1. tuning (on --tune-questions, or on the evaluated questions if none is given):
     tau_fast / tau_keep from similarities only; the JEV sufficiency threshold from one JEV
     call on each top-1; k_eq from running the JEV gate. Saved to results/<prefix>_tuning.json
     and reused on later runs (pass --retune to redo it).
  2. evaluation: systems run one after another per question (clean latency); the judge
     grades in the background. Each finished question is appended to
     results/<prefix>_partial.jsonl, so an interrupted run resumes where it stopped.
  3. report: results/<prefix>_results.json and results/<prefix>_report.md.

Examples:
  python gate_experiment.py                                   # small base (data.jsonl)
  python gate_experiment.py --data data/squad/corpus.jsonl --lang en --cache \\
      --questions data/squad/questions_test.json \\
      --tune-questions data/squad/questions_tune.json --prefix squad --dry-run
"""

import argparse
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import rag
from eval import JUDGE_MIN_INTERVAL_S, JUDGE_MODEL, POINTWISE_PROMPT, fmt, judge, retrieval_metrics
from rag import RAG, _context

HERE = Path(__file__).parent
OUT_DIR = HERE / "results"
NEIGHBORS = 5  # neighbors available to every gate
SEED = 42
BOOTSTRAP = 5000
ARMS = ["top1", "jev_gate", "sim_gate", "topk_eq", "oracle"]
JEV_GRID = [round(0.1 * i, 1) for i in range(1, 10)]
MIN_ANSWER_MATCH = 4  # shortest accepted answer used for string matching (avoids "2" matching anything)

# USD per token (OpenRouter list prices when this was written) and per JEV call, for --dry-run only
PRICES = {
    "judge_in": 2e-6,
    "judge_out": 10e-6,
    "llm_in": 0.4e-6,
    "llm_out": 1.6e-6,
    "jev_call": 0.00002,
}


# ---------- ground truth ----------


def answer_in(texts: list[str], answers: list[str]) -> bool:
    return any(a.lower() in t.lower() for t in texts for a in answers if len(a) >= MIN_ANSWER_MATCH)


def truth_top1_enough(item: dict, top1: dict) -> bool:
    """The top-1 chunk is enough when the question is answerable and the chunk is its gold
    chunk or contains one of its accepted answers (SQuAD facts repeat across paragraphs)."""
    if not item["gold"]:
        return False
    return top1["id"] in item["gold"] or answer_in([top1["text"]], item.get("answers", []))


# ---------- phase 1: tuning ----------


def tune(system: RAG, items: list[dict], source: str) -> dict:
    with ThreadPoolExecutor(max_workers=8) as pool:
        ranked = dict(zip([q["id"] for q in items], pool.map(lambda q: system.search(q["question"], k=NEIGHBORS), items)))
    truth = {q["id"]: truth_top1_enough(q, ranked[q["id"]][0]) for q in items}

    # similarity gate: tau_fast maximizes decision accuracy, tau_keep maximizes micro-F1 of kept chunks
    top1 = [(ranked[q["id"]][0]["similarity"], truth[q["id"]]) for q in items]
    candidates = sorted({s for s, _ in top1}) + [1.01]
    tau_fast = max(candidates, key=lambda t: sum((s >= t) == ok for s, ok in top1))
    pairs = [(h["similarity"], h["id"] in q["gold"]) for q in items for h in ranked[q["id"]]]
    n_gold = sum(len(q["gold"]) for q in items)

    def f1(t: float) -> float:
        tp = sum(1 for s, g in pairs if s >= t and g)
        kept = sum(1 for s, _ in pairs if s >= t)
        return 2 * tp / (kept + n_gold) if kept + n_gold else 0.0

    tau_keep = max(sorted({s for s, _ in pairs}) + [1.01], key=f1)

    # JEV threshold: sufficiency of the top-1 alone, one JEV call per question
    with ThreadPoolExecutor(max_workers=8) as pool:
        suf = list(pool.map(lambda q: system.gate_step(q["question"], [], ranked[q["id"]][0])[1], items))
    grid = []
    for t in JEV_GRID:
        says = [s >= t for s in suf]
        ok = [truth[q["id"]] for q in items]
        grid.append(
            {
                "threshold": t,
                "accuracy": sum(a == b for a, b in zip(says, ok)) / len(items),
                "false_pos": sum(a and not b for a, b in zip(says, ok)),
                "false_neg": sum(b and not a for a, b in zip(says, ok)),
            }
        )
    # best accuracy; ties go to the higher threshold (fewer bad chunks approved)
    best = max(grid, key=lambda g: (g["accuracy"], g["threshold"]))

    # retrieval budget of the JEV gate at that threshold
    with ThreadPoolExecutor(max_workers=8) as pool:
        examined = list(
            pool.map(
                lambda q: len(system.gate_select(q["question"], NEIGHBORS, best["threshold"])["steps"]), items
            )
        )
    return {
        "source": source,
        "n": len(items),
        "tau_fast": tau_fast,
        "tau_fast_accuracy": sum((s >= tau_fast) == ok for s, ok in top1) / len(items),
        "tau_keep": tau_keep,
        "tau_keep_f1": f1(tau_keep),
        "jev_threshold": best["threshold"],
        "jev_grid": grid,
        "jev_mean_examined": sum(examined) / len(items),
        "k_eq": max(1, round(sum(examined) / len(items))),
    }


# ---------- phase 2: systems and grading ----------


def run_arm(system: RAG, arm: str, item: dict, params: dict) -> dict:
    q = item["question"]
    start = time.perf_counter()
    out: dict = {}
    if arm == "top1":
        chunks = system.search(q, k=1)
        out = {"examined": 1}
    elif arm == "jev_gate":
        res = system.ask_gate(q, NEIGHBORS, params["jev_threshold"])
        chunks = res["kept"]
        first = res["steps"][0]["sufficiency"] >= params["jev_threshold"]
        out = {
            "examined": len(res["steps"]),
            "fast_path": first,
            "says_top1_enough": first,
            "steps": [
                {"id": s["id"], "relevance": round(s["relevance"], 2), "sufficiency": round(s["sufficiency"], 2)}
                for s in res["steps"]
            ],
            "answer": res["answer"],
        }
    elif arm == "sim_gate":
        hits = system.search(q, k=NEIGHBORS)
        enough = hits[0]["similarity"] >= params["tau_fast"]
        chunks = hits[:1] if enough else [h for h in hits if h["similarity"] >= params["tau_keep"]]
        out = {"examined": 1 if enough else NEIGHBORS, "fast_path": enough, "says_top1_enough": enough}
    elif arm == "topk_eq":
        chunks = system.search(q, k=params["k_eq"])
        out = {"examined": params["k_eq"]}
    elif arm == "oracle":
        chunks = [d for d in system.docs if d["id"] in item["gold"]]
        out = {"examined": len(chunks)}
    if "answer" not in out:
        out["answer"] = system.answer(q, chunks)
    out["latency_s"] = round(time.perf_counter() - start, 2)
    out["context_ids"] = [c["id"] for c in chunks]
    out["context_chars"] = sum(len(c["text"]) for c in chunks)
    out["answer_in_context"] = answer_in([c["text"] for c in chunks], item.get("answers", [])) if item["gold"] else None
    out["context"] = chunks
    return out


def grade(runs: dict, item: dict) -> None:
    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        futures = {
            arm: pool.submit(
                judge,
                POINTWISE_PROMPT.format(
                    question=item["question"],
                    reference=item["reference"],
                    context=_context(run["context"]) or "(vazio)",
                    answer=run["answer"],
                ),
            )
            for arm, run in runs.items()
        }
        for arm, run in runs.items():
            j = futures[arm].result()
            run["judge"] = j
            run["correct"] = j["correct_parts"] == j["total_parts"] and not j["hallucination"]
            run["retrieval"] = retrieval_metrics(run["context_ids"], item["gold"])
            del run["context"]


# ---------- statistics ----------


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_ci(a: list[bool], b: list[bool], rng: random.Random) -> tuple[float, float]:
    """95% CI of accuracy(b) - accuracy(a), resampling questions."""
    n = len(a)
    diffs = []
    for _ in range(BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP) - 1]


def confusion(results: list[dict], arm: str) -> dict:
    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in results:
        says, truth = r["runs"][arm]["says_top1_enough"], r["truth_top1_enough"]
        cm[("t" if says == truth else "f") + ("p" if says else "n")] += 1
    return cm


# ---------- phase 3: report ----------


def report(results: list[dict], meta: dict) -> str:
    n = len(results)
    rng = random.Random(SEED)
    params = meta["params"]
    col = {arm: [r["runs"][arm] for r in results] for arm in ARMS}

    def mean(arm: str, key, rows: list[dict] | None = None) -> float:
        xs = [r["runs"][arm] for r in rows] if rows is not None else col[arm]
        return sum(key(x) for x in xs) / len(xs) if xs else float("nan")

    tuned_on = (
        f"num conjunto separado de {params['n']} perguntas (`{params['source']}`)"
        if params["source"] != meta["questions"]
        else "nestas mesmas perguntas (vantagem para os controles; numa base pública, use um conjunto separado)"
    )
    lines = [
        "# Experimento: JEV como portão na frente de um RAG top-1",
        "",
        f"Gerado em {meta['date']}. Base: `{meta['data']}` ({meta['n_docs']} trechos, idioma `{meta['lang']}`). "
        f"Perguntas: `{meta['questions']}` ({n}). Juiz: `{meta['judge_model']}`. "
        f"Gerador: `{meta['llm_model']}`. Embeddings: `{meta['embed_model']}`. Uma execução.",
        "",
        "**Hipótese:** um RAG top-1 costuma acertar; o JEV detecta os casos em que o trecho do top-1 "
        "não basta e os corrige, com custo e latência pequenos no caso comum.",
        "",
        "- **top1:** o trecho mais parecido e resposta direta (baseline).",
        f"- **jev_gate:** o JEV julga cada vizinho numa única chamada (relevância + suficiência) e para "
        f"quando os trechos mantidos bastam (suficiência >= {params['jev_threshold']}); até {NEIGHBORS} "
        f"vizinhos; desiste após {meta['gate_early_stop']} trechos seguidos com relevância "
        f"< {meta['gate_near_zero']}.",
        f"- **sim_gate:** o mesmo portão usando similaridade de cosseno: top-1 se similaridade >= "
        f"{params['tau_fast']:.3f}, senão os vizinhos com similaridade >= {params['tau_keep']:.3f}.",
        f"- **topk_eq:** top-{params['k_eq']}, o número médio de trechos que o jev_gate examinou no ajuste "
        f"({params['jev_mean_examined']:.2f}).",
        "- **oracle:** os trechos gold como contexto (teto).",
        "",
        f"Limiares ajustados {tuned_on}.",
        "",
        "## 1. O portão sabe quando o top-1 não basta?",
        "",
        "Verdade: o top-1 basta quando a pergunta tem resposta e o trecho é o gold ou contém uma resposta "
        f"aceita. Casos em que o top-1 basta: {sum(r['truth_top1_enough'] for r in results)} de {n}.",
        "",
        "| Portão | Acurácia | Aprovou um top-1 ruim (falso positivo) | Barrou um top-1 bom (falso negativo) | Acertos (VP / VN) |",
        "|---|---|---|---|---|",
    ]
    for arm in ("jev_gate", "sim_gate"):
        cm = confusion(results, arm)
        lines.append(
            f"| {arm} | {(cm['tp'] + cm['tn']) / n:.0%} | {cm['fp']} | {cm['fn']} | {cm['tp']} / {cm['tn']} |"
        )
    lines += [
        "",
        "O falso positivo é o erro caro: o portão aprova um trecho ruim e o sistema responde errado ou "
        "incompleto. O falso negativo só custa latência (examina trechos a mais sem necessidade).",
        "",
        "## 2. Qualidade e custo",
        "",
        "| Métrica | " + " | ".join(ARMS) + " |",
        "|---" * (len(ARMS) + 1) + "|",
        "| Respostas corretas e completas | " + " | ".join(f"{mean(a, lambda x: x['correct']):.0%}" for a in ARMS) + " |",
        "| Nota do juiz (1 a 5) | " + " | ".join(fmt(mean(a, lambda x: x["judge"]["score"])) for a in ARMS) + " |",
        "| Alucinação | " + " | ".join(f"{mean(a, lambda x: x['judge']['hallucination']):.0%}" for a in ARMS) + " |",
        "| Trechos examinados (média) | " + " | ".join(fmt(mean(a, lambda x: x["examined"])) for a in ARMS) + " |",
        "| Trechos enviados ao LLM (média) | "
        + " | ".join(fmt(mean(a, lambda x: x["retrieval"]["context_size"])) for a in ARMS)
        + " |",
        "| Tokens de contexto (média, ≈ caracteres/4) | "
        + " | ".join(f"{mean(a, lambda x: x['context_chars']) / 4:.0f}" for a in ARMS)
        + " |",
        "| Trechos não-gold enviados (média) | "
        + " | ".join(fmt(mean(a, lambda x: x["retrieval"]["noise_chunks"])) for a in ARMS)
        + " |",
        "| Latência p50 (s) | " + " | ".join(fmt(percentile([x["latency_s"] for x in col[a]], 50)) for a in ARMS) + " |",
        "| Latência p95 (s) | " + " | ".join(fmt(percentile([x["latency_s"] for x in col[a]], 95)) for a in ARMS) + " |",
        "| Caminho rápido (1 trecho) | - | "
        + f"{mean('jev_gate', lambda x: x['fast_path']):.0%} | {mean('sim_gate', lambda x: x['fast_path']):.0%} | - | - |",
        "",
        "### Por categoria (respostas corretas)",
        "",
        "| Categoria | n | " + " | ".join(ARMS) + " |",
        "|---|---" + "|---" * len(ARMS) + "|",
    ]
    for cat in dict.fromkeys(r["category"] for r in results):
        rows = [r for r in results if r["category"] == cat]
        lines.append(
            f"| {cat} | {len(rows)} | "
            + " | ".join(f"{mean(a, lambda x: x['correct'], rows):.0%}" for a in ARMS)
            + " |"
        )

    lines += [
        "",
        "## 3. Comparação pareada",
        "",
        "Correta = o juiz considera todas as partes corretas e não há alucinação.",
        "",
        "### Contra o top1",
        "",
        "| Sistema | Corrigiu um erro do top1 | Estragou um acerto do top1 | McNemar p | Diferença de acurácia (IC 95%) |",
        "|---|---|---|---|---|",
    ]
    base = [x["correct"] for x in col["top1"]]
    for arm in ARMS[1:]:
        other = [x["correct"] for x in col[arm]]
        b = sum(1 for x, y in zip(base, other) if not x and y)
        c = sum(1 for x, y in zip(base, other) if x and not y)
        lo, hi = bootstrap_ci(base, other, rng)
        diff = (sum(other) - sum(base)) / n
        lines.append(f"| {arm} | {b} | {c} | {mcnemar_p(b, c):.3f} | {diff:+.0%} ({lo:+.0%} a {hi:+.0%}) |")

    lines += [
        "",
        "### jev_gate contra os controles",
        "",
        "Mostra se o ganho vem do JEV ou apenas de examinar mais trechos (topk_eq) ou de um portão "
        "barato (sim_gate).",
        "",
        "| Controle | jev_gate acerta e controle erra | controle acerta e jev_gate erra | McNemar p | "
        "Diferença de acurácia (IC 95%) | Tokens de contexto (jev / controle) | Latência p50 (jev / controle) |",
        "|---|---|---|---|---|---|---|",
    ]
    gate = [x["correct"] for x in col["jev_gate"]]
    for arm in ("sim_gate", "topk_eq"):
        ctrl = [x["correct"] for x in col[arm]]
        b = sum(1 for x, y in zip(ctrl, gate) if not x and y)
        c = sum(1 for x, y in zip(ctrl, gate) if x and not y)
        lo, hi = bootstrap_ci(ctrl, gate, rng)
        lines.append(
            f"| {arm} | {b} | {c} | {mcnemar_p(b, c):.3f} | {(sum(gate) - sum(ctrl)) / n:+.0%} "
            f"({lo:+.0%} a {hi:+.0%}) | {mean('jev_gate', lambda x: x['context_chars']) / 4:.0f} / "
            f"{mean(arm, lambda x: x['context_chars']) / 4:.0f} | "
            f"{fmt(percentile([x['latency_s'] for x in col['jev_gate']], 50))} / "
            f"{fmt(percentile([x['latency_s'] for x in col[arm]], 50))} |"
        )

    fast = [r for r in results if r["runs"]["jev_gate"]["fast_path"]]
    slow = [r for r in results if not r["runs"]["jev_gate"]["fast_path"]]

    def split_row(label: str, rows: list[dict]) -> str:
        if not rows:
            return f"| {label} | 0 | - | - | - | - | - |"
        m = len(rows)
        acc_t = sum(r["runs"]["top1"]["correct"] for r in rows) / m
        acc_j = sum(r["runs"]["jev_gate"]["correct"] for r in rows) / m
        lat_t = sum(r["runs"]["top1"]["latency_s"] for r in rows) / m
        lat_j = sum(r["runs"]["jev_gate"]["latency_s"] for r in rows) / m
        return f"| {label} | {m} | {acc_t:.0%} | {acc_j:.0%} | {lat_t:.2f} | {lat_j:.2f} | {lat_j - lat_t:+.2f} |"

    lines += [
        "",
        "## 4. Caso comum x caso raro",
        "",
        "Separação pelo caminho que o jev_gate tomou.",
        "",
        "| Caminho | n | Acurácia top1 | Acurácia jev_gate | Latência top1 (s) | Latência jev_gate (s) | Custo extra (s) |",
        "|---|---|---|---|---|---|---|",
        split_row("rápido (top-1 aprovado)", fast),
        split_row("lento (top-1 barrado)", slow),
        split_row("total", results),
        "",
        "## 5. Ajuste do limiar do JEV",
        "",
        f"Suficiência do top-1 sozinho, uma chamada ao JEV por pergunta de ajuste ({params['n']} perguntas). "
        "Escolhido: maior acurácia; empate vai para o limiar mais alto (menos falsos positivos).",
        "",
        "| Limiar | Acurácia | Falsos positivos | Falsos negativos |",
        "|---|---|---|---|",
        *[
            f"| {g['threshold']}{' ←' if g['threshold'] == params['jev_threshold'] else ''} | "
            f"{g['accuracy']:.0%} | {g['false_pos']} | {g['false_neg']} |"
            for g in params["jev_grid"]
        ],
        "",
        "## 6. Por pergunta",
        "",
        "| # | Categoria | top-1 basta? | Portão JEV | Passos JEV (relevância/suficiência) | " + " | ".join(ARMS) + " |",
        "|---|---|---|---|---" + "|---" * len(ARMS) + "|",
    ]
    for r in results:
        g = r["runs"]["jev_gate"]
        steps = " ".join(f"{s['id']} ({s['relevance']}/{s['sufficiency']})" for s in g["steps"])
        marks = " | ".join("✅" if r["runs"][a]["correct"] else "❌" for a in ARMS)
        lines.append(
            f"| {r['id']} | {r['category']} | {'sim' if r['truth_top1_enough'] else 'não'} | "
            f"{'aprova' if g['says_top1_enough'] else 'barra'} | {steps} | {marks} |"
        )

    if n <= 50:  # full answers only for small runs; large runs keep them in the JSON
        lines += ["", "## 7. Respostas", ""]
        for r in results:
            lines += [f"### {r['id']}: {r['question']}", "", f"Referência: {r['reference']}", ""]
            for arm in ARMS:
                run = r["runs"][arm]
                lines += [
                    f"**{arm}** ({'correta' if run['correct'] else 'incorreta'}, nota {run['judge']['score']}, "
                    f"contexto: {', '.join(run['context_ids']) or 'vazio'}, {run['latency_s']} s)",
                    "",
                    f"> {run['answer']}".replace("\n", "\n> "),
                    "",
                    f"Juiz: {run['judge']['justification']}",
                    "",
                ]

    lines += [
        "",
        "## Limitações",
        "",
        f"- {n} perguntas, uma única execução.",
        "- O juiz é um LLM e pode errar.",
        "- A verdade de \"o top-1 basta\" usa o trecho gold ou a presença de uma resposta aceita no texto; "
        "respostas com menos de 4 caracteres não entram na comparação de texto.",
        "- Com poucas perguntas, a latência p95 é praticamente o valor máximo e fica sensível a uma única "
        "chamada lenta ou repetida pela rede.",
        "- A latência depende da rede e do OpenRouter no momento da execução.",
    ]
    return "\n".join(lines) + "\n"


# ---------- dry run ----------


def dry_run(items: list[dict], tune_items: list[dict], system_docs: list[dict], params: dict | None) -> None:
    """Estimate calls, time and cost without calling any API."""
    n = len(items)
    avg_chunk = sum(len(d["text"]) for d in system_docs) / len(system_docs) / 4  # tokens
    k_eq = params["k_eq"] if params else 2
    # chunks per arm: top1, jev_gate (~1.3), sim_gate (~1.2), topk_eq, oracle (~1)
    ctx = avg_chunk * (1 + 1.3 + 1.2 + k_eq + 1)
    gen_calls = 5 * n
    judge_calls = 5 * n
    jev_calls = 2.5 * n + (0 if params else 3.5 * len(tune_items))
    judge_in = judge_calls * 550 + n * ctx  # template + reference + answer, plus contexts
    cost = (
        judge_in * PRICES["judge_in"]
        + judge_calls * 150 * PRICES["judge_out"]
        + (gen_calls * 150 + n * ctx) * PRICES["llm_in"]
        + gen_calls * 80 * PRICES["llm_out"]
        + jev_calls * PRICES["jev_call"]
    )
    minutes = max(judge_calls * JUDGE_MIN_INTERVAL_S, n * 12) / 60
    print("Estimativa (sem chamar nenhuma API):")
    print(f"  perguntas avaliadas: {n}   perguntas de ajuste: {len(tune_items)}   trechos na base: {len(system_docs)}")
    print(f"  ajuste: {'reaproveitado de execução anterior' if params else f'~{int(3.5 * len(tune_items))} chamadas ao JEV + embeddings'}")
    print(f"  chamadas: juiz {judge_calls}, gerador ~{gen_calls}, JEV ~{int(jev_calls)}")
    print(f"  tempo: ~{minutes:.0f} min (juiz limitado a 1 chamada a cada {JUDGE_MIN_INTERVAL_S} s)")
    print(f"  custo: ~US$ {cost:.2f} (preços de tabela; confira o saldo da chave antes)")


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.jsonl")
    parser.add_argument("--questions", default="eval_questions.json")
    parser.add_argument("--tune-questions", help="separate question set for tuning thresholds")
    parser.add_argument("--lang", default="pt", choices=sorted(rag.PROMPTS))
    parser.add_argument("--cache", action="store_true", help="cache corpus embeddings on disk")
    parser.add_argument("--limit", type=int, help="evaluate only the first N questions")
    parser.add_argument("--prefix", default="gate")
    parser.add_argument("--retune", action="store_true", help="ignore saved tuning and redo it")
    parser.add_argument("--dry-run", action="store_true", help="estimate calls, time and cost, then exit")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from saved results")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    final_path = OUT_DIR / f"{args.prefix}_results.json"
    tuning_path = OUT_DIR / f"{args.prefix}_tuning.json"
    partial_path = OUT_DIR / f"{args.prefix}_partial.jsonl"

    if args.report_only:
        saved = json.loads(final_path.read_text(encoding="utf-8"))
        (OUT_DIR / f"{args.prefix}_report.md").write_text(report(saved["results"], saved["meta"]), encoding="utf-8")
        print(f"Relatório refeito: {OUT_DIR / f'{args.prefix}_report.md'}")
        return

    items = json.loads((HERE / args.questions).read_text(encoding="utf-8"))[: args.limit]
    tune_source = args.tune_questions or args.questions
    tune_items = json.loads((HERE / tune_source).read_text(encoding="utf-8"))
    params = None if args.retune or not tuning_path.exists() else json.loads(tuning_path.read_text(encoding="utf-8"))

    if args.dry_run:
        docs = [json.loads(line) for line in (HERE / args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
        dry_run(items, tune_items, docs, params)
        return

    system = RAG(HERE / args.data, lang=args.lang, cache=args.cache)

    if params is None:
        print(f"Ajustando limiares em {tune_source} ({len(tune_items)} perguntas)...", flush=True)
        params = tune(system, tune_items, tune_source)
        tuning_path.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Limiares: JEV {params['jev_threshold']}, sim tau_fast {params['tau_fast']:.3f} "
        f"tau_keep {params['tau_keep']:.3f}, k_eq {params['k_eq']}",
        flush=True,
    )

    done = {}
    if partial_path.exists():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["id"]] = row
        print(f"Retomando: {len(done)} perguntas já avaliadas.", flush=True)

    write_lock = threading.Lock()

    def finish(row: dict) -> None:
        runs = row["runs"]
        grade(runs, row)
        with write_lock:
            with partial_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[row["id"]] = row
            marks = "  ".join(f"{a} {'ok' if runs[a]['correct'] else '--'}" for a in ARMS)
            print(f"[{len(done)}/{len(items)}] {row['id']}  {marks}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as grader:  # judge runs while the next question is answered
        futures = []
        for item in items:
            if item["id"] in done:
                continue
            for f in futures:  # stop at the first grading failure (e.g. out of credit); progress is saved
                if f.done() and f.exception():
                    raise f.exception()
            runs ={arm: run_arm(system, arm, item, params) for arm in ARMS}
            truth = truth_top1_enough(item, runs["top1"]["context"][0])
            futures.append(grader.submit(finish, {**item, "truth_top1_enough": truth, "runs": runs}))
        for f in futures:
            f.result()  # surface grading errors

    results = [done[q["id"]] for q in items]
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": args.data,
        "questions": args.questions,
        "lang": args.lang,
        "n_docs": len(system.docs),
        "judge_model": JUDGE_MODEL,
        "llm_model": rag.LLM_MODEL,
        "embed_model": rag.EMBED_MODEL,
        "neighbors": NEIGHBORS,
        "min_relevance": rag.MIN_RELEVANCE,
        "gate_early_stop": rag.GATE_EARLY_STOP,
        "gate_near_zero": rag.GATE_NEAR_ZERO,
        "params": params,
    }
    final_path.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / f"{args.prefix}_report.md").write_text(report(results, meta), encoding="utf-8")
    partial_path.unlink()
    print(f"\nSalvo em {final_path} e {OUT_DIR / f'{args.prefix}_report.md'}")


if __name__ == "__main__":
    main()
