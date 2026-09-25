"""End-to-end experiment: does the context policy change hallucination in the final answer?

Uses the candidates and scores saved by rerank_experiment.py (no new retrieval, JEV or Cohere
calls). For each test question, four context policies decide what reaches the generator:

  top3         the 3 most similar chunks by embedding (common practice)
  cohere_top3  the 3 best chunks by Cohere rerank-v4.0-pro
  cohere_cut   every chunk with Cohere score >= cut (cut tuned on the tuning questions)
  jev_cut      every chunk with JEV relevance >= cut (cut tuned on the tuning questions)

When a policy keeps no chunk, the generator is NOT called and a fixed "not found" message is
returned. The generator (gpt-4.1-mini) answers from the kept chunks; an LLM judge
(claude-sonnet-5) grades each answer against the SQuAD reference. Generator cost and tokens
are the real values reported by OpenRouter.

Usage: python e2e_experiment.py [--limit N] [--dry-run] [--report-only]
Writes results/e2e_results.json and results/e2e_report.md (resumable via e2e_partial.jsonl).
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import meter
import rag
from eval import JUDGE_MIN_INTERVAL_S, JUDGE_MODEL, POINTWISE_PROMPT, JudgeRefused, judge
from rag import RAG, _context
from rerank_experiment import DATA, TEST, best_threshold, bootstrap_ci, mcnemar_p, percentile

HERE = Path(__file__).parent
OUT_DIR = HERE / "results"
ARMS = ["top3", "cohere_top3", "cohere_cut", "jev_cut"]
COHERE_PRICE_PER_SEARCH = 2.50 / 1000  # list price (third-party sources), see the article draft
SEED = 42


def select(arm: str, row: dict, cuts: dict) -> list[str]:
    """Chunk ids a policy sends to the generator, in the order they are sent."""
    cands, scores = row["candidates"], row["scores"]
    if arm == "top3":
        return cands[:3]
    method = "cohere" if arm.startswith("cohere") else "jev"
    order = sorted(range(len(cands)), key=lambda i: -scores[method][i])
    if arm == "cohere_top3":
        return [cands[i] for i in order[:3]]
    return [cands[i] for i in order if scores[method][i] >= cuts[method]]


def run_question(system: RAG, row: dict, item: dict, cuts: dict) -> dict:
    by_id = {d["id"]: d for d in system.docs}
    runs = {}
    for arm in ARMS:
        ids = select(arm, row, cuts)
        chunks = [by_id[i] for i in ids]
        with meter.metered() as m:
            start = time.perf_counter()
            answer = system.answer(item["question"], chunks)  # fixed message, no call, when empty
            latency = time.perf_counter() - start
        runs[arm] = {
            "context_ids": ids,
            "context": chunks,
            "answer": answer,
            "llm_called": bool(chunks),
            "llm_cost": m["cost"]["llm"],
            "llm_tokens_in": m["tokens_in"]["llm"],
            "llm_tokens_out": m["tokens_out"]["llm"],
            "llm_latency_s": round(latency, 3),
            "has_useful_chunk": any(lbl for c, lbl in zip(row["candidates"], row["labels"]) if c in ids),
        }
    return runs


def grade(item: dict, runs: dict) -> None:
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
            try:
                j = futures[arm].result()
            except JudgeRefused:  # the question is left out of the analysis (see report)
                run["judge"], run["correct"] = None, None
            else:
                run["judge"] = j
                run["correct"] = j["correct_parts"] == j["total_parts"] and not j["hallucination"]
            del run["context"]


# ---------- report ----------


def report(results: list[dict], meta: dict) -> str:
    rng = random.Random(SEED)
    refused = [r for r in results if any(r["runs"][a]["judge"] is None for a in ARMS)]
    results = [r for r in results if r not in refused]
    n = len(results)
    ans = [r for r in results if r["category"] == "answerable"]
    una = [r for r in results if r["category"] == "unanswerable"]

    def rate(rows, arm, fn):
        return sum(fn(r["runs"][arm]) for r in rows) / len(rows) if rows else float("nan")

    def cost_per_1000(arm):
        llm = sum(r["runs"][arm]["llm_cost"] for r in results) / n
        extra = 0.0
        if arm.startswith("cohere"):
            extra = COHERE_PRICE_PER_SEARCH
        elif arm == "jev_cut":
            extra = sum(r["jev_cost"] for r in results) / n
        return llm * 1000, extra * 1000

    lines = [
        "# Experimento ponta a ponta: a política de contexto muda a alucinação?",
        "",
        f"Gerado em {meta['date']}. {n} perguntas de teste do SQuAD 2.0 ({len(ans)} com resposta, "
        f"{len(una)} sem resposta), com os mesmos 10 candidatos por pergunta do experimento de rerank. "
        f"Gerador: `{meta['llm_model']}`. Juiz: `{meta['judge_model']}`. Uma execução.",
        "",
        f"{len(refused)} pergunta(s) excluída(s) porque o filtro de segurança do juiz se recusou a avaliar "
        f"alguma das respostas: {', '.join(r['id'] for r in refused) or 'nenhuma'}.",
        "",
        "- **top3:** os 3 trechos mais parecidos pelo embedding (prática comum).",
        "- **cohere_top3:** os 3 melhores segundo o Cohere `rerank-v4.0-pro`.",
        f"- **cohere_cut:** todos os trechos com nota Cohere >= {meta['cuts']['cohere']:.3f}.",
        f"- **jev_cut:** todos os trechos com relevância JEV >= {meta['cuts']['jev']:.2f}.",
        "",
        "Os cortes foram escolhidos nas 100 perguntas de ajuste (maior F1), sem olhar as de teste. "
        "Quando uma política não mantém nenhum trecho, o gerador **não é chamado** e o sistema devolve "
        "uma mensagem fixa de \"não encontrado\".",
        "",
        "Correta = o juiz considera todas as partes corretas e não há alucinação. Alucinação = o juiz "
        "marca que a resposta afirma algo fora do contexto recebido ou que contradiz a referência.",
        "",
        "## 1. Resultado principal",
        "",
        "| Métrica | " + " | ".join(ARMS) + " |",
        "|---" * (len(ARMS) + 1) + "|",
        "| Respostas corretas (todas) | " + " | ".join(f"{rate(results, a, lambda x: x['correct']):.0%}" for a in ARMS) + " |",
        "| **Alucinação (todas)** | " + " | ".join(f"**{rate(results, a, lambda x: x['judge']['hallucination']):.0%}**" for a in ARMS) + " |",
        f"| Corretas, perguntas com resposta (n={len(ans)}) | "
        + " | ".join(f"{rate(ans, a, lambda x: x['correct']):.0%}" for a in ARMS) + " |",
        f"| Corretas, perguntas sem resposta (n={len(una)}) | "
        + " | ".join(f"{rate(una, a, lambda x: x['correct']):.0%}" for a in ARMS) + " |",
        f"| **Alucinação nas perguntas sem resposta** (n={len(una)}) | "
        + " | ".join(f"**{rate(una, a, lambda x: x['judge']['hallucination']):.0%}**" for a in ARMS) + " |",
        f"| \"Não sei\" indevido (perguntas com resposta) | "
        + " | ".join(f"{rate(ans, a, lambda x: x['judge']['abstained'] and not x['correct']):.0%}" for a in ARMS) + " |",
        "| Perguntas sem chamada ao gerador | " + " | ".join(f"{rate(results, a, lambda x: not x['llm_called']):.0%}" for a in ARMS) + " |",
        "",
        "## 2. Comparação pareada com o jev_cut",
        "",
        "| Contra | Métrica | jev_cut melhor | outro melhor | McNemar p | Diferença jev_cut − outro (IC 95%) |",
        "|---|---|---|---|---|---|",
    ]
    for other in ARMS[:-1]:
        for label, key, better_is_true in (("corretas", "correct", True), ("alucinação", "hallucination", False)):
            get = (lambda x, k=key: x[k]) if key == "correct" else (lambda x: x["judge"]["hallucination"])
            j = [get(r["runs"]["jev_cut"]) for r in results]
            o = [get(r["runs"][other]) for r in results]
            if better_is_true:
                jb = sum(1 for a, b in zip(j, o) if a and not b)
                ob = sum(1 for a, b in zip(j, o) if b and not a)
            else:
                jb = sum(1 for a, b in zip(j, o) if b and not a)
                ob = sum(1 for a, b in zip(j, o) if a and not b)
            lo, hi = bootstrap_ci(o, j, rng)
            lines.append(
                f"| {other} | {label} | {jb} | {ob} | {mcnemar_p(jb, ob):.3f} | "
                f"{(sum(j) - sum(o)) / n:+.1%} ({lo:+.1%} a {hi:+.1%}) |"
            )

    lines += [
        "",
        "## 3. Custo e contexto",
        "",
        "| Métrica | " + " | ".join(ARMS) + " |",
        "|---" * (len(ARMS) + 1) + "|",
        "| Trechos enviados ao gerador (média) | "
        + " | ".join(f"{sum(len(r['runs'][a]['context_ids']) for r in results) / n:.2f}" for a in ARMS) + " |",
        "| Tokens de entrada do gerador (média, medido) | "
        + " | ".join(f"{sum(r['runs'][a]['llm_tokens_in'] for r in results) / n:.0f}" for a in ARMS) + " |",
        "| Gerador, US$ a cada mil perguntas (medido) | "
        + " | ".join(f"{cost_per_1000(a)[0]:.3f}" for a in ARMS) + " |",
        "| Seleção de contexto, US$ a cada mil perguntas | "
        + " | ".join(["0"] + [f"{cost_per_1000(a)[1]:.3f}" for a in ARMS[1:]]) + " |",
        "| **Total, US$ a cada mil perguntas** | "
        + " | ".join(f"**{sum(cost_per_1000(a)):.3f}**" for a in ARMS) + " |",
        "| Latência do gerador p50 (s) | "
        + " | ".join(f"{percentile([r['runs'][a]['llm_latency_s'] for r in results], 50):.2f}" for a in ARMS) + " |",
        "",
        "Seleção de contexto: JEV medido (`usage.cost`, 11 chamadas por pergunta no experimento de rerank); "
        f"Cohere a preço de tabela (US$ {COHERE_PRICE_PER_SEARCH * 1000:.2f} a cada mil buscas). A latência "
        "do gerador inclui 0 s quando ele não é chamado.",
        "",
        "## 4. Por que as políticas erram",
        "",
        "Perguntas com resposta em que a política enviou ao menos um trecho útil (o gerador tinha como acertar):",
        "",
        "| | " + " | ".join(ARMS) + " |",
        "|---" * (len(ARMS) + 1) + "|",
        "| Com trecho útil no contexto | " + " | ".join(f"{rate(ans, a, lambda x: x['has_useful_chunk']):.0%}" for a in ARMS) + " |",
        "| Corretas quando havia trecho útil | "
        + " | ".join(
            f"{sum(r['runs'][a]['correct'] for r in ans if r['runs'][a]['has_useful_chunk']) / max(1, sum(r['runs'][a]['has_useful_chunk'] for r in ans)):.0%}"
            for a in ARMS
        ) + " |",
        "",
        "## Limitações",
        "",
        f"- {n} perguntas de uma base (SQuAD 2.0), uma execução; {len(una)} perguntas sem resposta.",
        "- O juiz é um LLM e pode errar; a alucinação é a marcação dele.",
        "- O custo da Cohere é preço de tabela de terceiros; o do JEV e o do gerador são medidos.",
        "- Latência só do gerador: a seleção de contexto (JEV ~0,7 s, Cohere ~1 s) não está somada.",
    ]
    return "\n".join(lines) + "\n"


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    final, md, partial = OUT_DIR / "e2e_results.json", OUT_DIR / "e2e_report.md", OUT_DIR / "e2e_partial.jsonl"
    if args.report_only:
        saved = json.loads(final.read_text(encoding="utf-8"))
        md.write_text(report(saved["results"], saved["meta"]), encoding="utf-8")
        print(f"Relatório refeito: {md}")
        return

    saved = json.loads((OUT_DIR / "rerank_results.json").read_text(encoding="utf-8"))
    rows = saved["rows"]
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"][: args.limit]
    cuts = {m: best_threshold(tune, m) for m in ("jev", "cohere")}
    items = {q["id"]: q for q in json.loads((HERE / TEST).read_text(encoding="utf-8"))}

    if args.dry_run:
        calls = len(test) * len(ARMS)
        print(f"{len(test)} perguntas x {len(ARMS)} políticas: até {calls} chamadas ao gerador e {calls} ao juiz "
              f"(~{calls * JUDGE_MIN_INTERVAL_S / 60:.0f} min, ~US$ {calls * 0.0028 + calls * 0.0003:.2f}). "
              f"Cortes: JEV {cuts['jev']}, Cohere {cuts['cohere']:.3f}")
        return

    system = RAG(HERE / DATA, lang="en", cache=True)  # cached corpus embeddings: no API cost
    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
        print(f"Retomando: {len(done)} perguntas já avaliadas.", flush=True)

    lock = threading.Lock()

    def finish(result: dict) -> None:
        grade(result, result["runs"])
        with lock:
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            done[result["id"]] = result
            def mark(run):
                if run["judge"] is None:
                    return "RECUSA"
                return "ok" if run["correct"] else ("ALUC" if run["judge"]["hallucination"] else "--")

            marks = "  ".join(f"{a} {mark(result['runs'][a])}" for a in ARMS)
            print(f"[{len(done)}/{len(test)}] {result['category'][:5]}  {marks}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as grader:
        futures = []
        for row in test:
            if row["id"] in done:
                continue
            for f in futures:
                if f.done() and f.exception():
                    raise f.exception()
            item = items[row["id"]]
            runs = run_question(system, row, item, cuts)
            result = {
                "id": row["id"], "category": item["category"], "question": item["question"],
                "reference": item["reference"], "jev_cost": row["jev_cost"], "runs": runs,
            }
            futures.append(grader.submit(finish, result))
        for f in futures:
            f.result()

    results = [done[r["id"]] for r in test]
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "llm_model": rag.LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "cuts": cuts,
        "cohere_price_per_search": COHERE_PRICE_PER_SEARCH,
    }
    final.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    md.write_text(report(results, meta), encoding="utf-8")
    partial.unlink()
    print(f"\nSalvo em {final} e {md}")


if __name__ == "__main__":
    main()
