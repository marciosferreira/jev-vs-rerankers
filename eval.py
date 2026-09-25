"""Compare plain top-k RAG against JEV-based RAG, using an LLM as judge.

Experiments (python eval.py <name>, default k1):
  k1   - top1:    FAISS top-1 -> LLM answer (single shot)
         top5:    FAISS top-5 -> LLM answer (reference)
         jev_inc: RAG.ask_incremental() - neighbors one at a time (up to 5), JEV relevance
                  per chunk, stop as soon as JEV sufficiency passes
  iter - top5 vs jev_iter: RAG.ask() (JEV filter + sufficiency + LLM-rephrased search rounds)
All systems share the same index, embedding model, generator model and answer prompt.

The judge grades each answer on its own (pointwise) and compares pairs of answers blind,
in random order (pairwise). Retrieval is also scored objectively against gold chunk ids.

Writes results/<prefix>_results.json (everything) and results/<prefix>_report.md (summary).
"""

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import rag
from rag import RAG, _context, _post

JUDGE_MODEL = "anthropic/claude-sonnet-5"
JUDGE_MIN_INTERVAL_S = 3.2  # OpenRouter caps new accounts at 20 requests/min for this model
SEED = 42

HERE = Path(__file__).parent
OUT_DIR = HERE / "results"

POINTWISE_PROMPT = """Você é um avaliador rigoroso de sistemas de perguntas e respostas (RAG).

O sistema só pode usar a base de conhecimento. A resposta de referência diz o que a base permite \
responder; quando a base não tem a informação, a resposta correta é dizer que não sabe.

Pergunta: {question}

Resposta de referência: {reference}

Contexto que o sistema recebeu:
{context}

Resposta do sistema:
{answer}

Regras:
- Julgue correção e completude comparando com a RESPOSTA DE REFERÊNCIA, não com o contexto. Se a \
referência traz a informação de uma parte, a base tem essa informação, e dizer "não sei" sobre essa \
parte conta como ERRADO, mesmo que o contexto recebido pelo sistema não a contenha (isso é uma falha \
de recuperação do sistema).
- Dizer "não sei" só é correto para partes que a referência diz não estarem na base.
- Use o contexto apenas para decidir se houve alucinação.

Avalie e responda APENAS com um objeto JSON, sem texto fora dele:
{{
  "total_parts": <número de partes/perguntas independentes contidas na pergunta>,
  "correct_parts": <quantas partes a resposta tratou corretamente, segundo as regras acima>,
  "hallucination": <true se a resposta afirma algo que não está no contexto recebido ou que \
contradiz a referência>,
  "abstained": <true se a resposta diz que não sabe ou que não encontrou alguma parte>,
  "score": <nota de 1 a 5: 5 = correta e completa, 4 = correta com pequena falha, \
3 = parcialmente correta, 2 = majoritariamente errada ou incompleta, 1 = errada ou inventada>,
  "justification": "<uma ou duas frases>"
}}"""

PAIRWISE_PROMPT = """Você compara duas respostas de sistemas de perguntas e respostas (RAG).

O sistema só pode usar a base de conhecimento. A resposta de referência diz o que a base permite \
responder; quando a base não tem a informação, a resposta correta é dizer que não sabe.

Pergunta: {question}

Resposta de referência: {reference}

Resposta A:
{a}

Resposta B:
{b}

Qual resposta é melhor, considerando correção, completude e ausência de informação inventada? \
Detalhes corretos além do que foi perguntado NÃO tornam uma resposta melhor. Se as duas respondem \
corretamente o que foi perguntado (ou as duas dizem corretamente que não sabem), responda "empate". \
Responda APENAS com um objeto JSON:
{{"winner": "A" | "B" | "empate", "reason": "<uma frase>"}}"""


class JudgeRefused(RuntimeError):
    """The judge model's content filter refused to grade this prompt."""


_judge_lock = threading.Lock()
_judge_last = 0.0


def judge(prompt: str) -> dict:
    global _judge_last
    for attempt in range(3):
        with _judge_lock:  # space calls out to respect the judge's per-minute rate limit
            wait = _judge_last + JUDGE_MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            _judge_last = time.monotonic()
        resp = _post(
            "/chat/completions",
            {
                "model": JUDGE_MODEL,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        choice = resp["choices"][0]
        if choice.get("finish_reason") == "content_filter":  # deterministic: retrying won't help
            raise JudgeRefused("judge's safety filter blocked this evaluation")
        text = choice["message"].get("content") or ""
        try:
            return json.loads(text[text.index("{") : text.rindex("}") + 1])
        except ValueError:  # empty reply or no valid JSON: ask again
            if attempt == 2:
                raise RuntimeError(f"Judge returned no valid JSON: {text[:200]!r}")


def run_topk(k: int):
    def run(system: RAG, question: str) -> dict:
        chunks = system.search(question, k=k)
        return {"answer": system.answer(question, chunks), "context": chunks, "examined": k}

    return run


def run_jev_iter(system: RAG, question: str) -> dict:
    result = system.ask(question)
    return {
        "answer": result["answer"],
        "context": result["kept"],
        "examined": sum(len(r["ranked"]) for r in result["rounds"]),
        "rounds": len(result["rounds"]),
        "sufficiency": [round(r["sufficiency"], 2) for r in result["rounds"]],
    }


def run_jev_inc(system: RAG, question: str) -> dict:
    result = system.ask_incremental(question)
    return {
        "answer": result["answer"],
        "context": result["kept"],
        "examined": len(result["steps"]),
        "steps": [
            {
                "id": s["hit"]["id"],
                "relevance": round(s["hit"]["relevance"], 2),
                "sufficiency": None if s["sufficiency"] is None else round(s["sufficiency"], 2),
            }
            for s in result["steps"]
        ],
    }


EXPERIMENTS = {
    "k1": {
        "prefix": "eval_k1",
        "title": "RAG top-1 x RAG top-5 x JEV incremental (1 trecho por vez, até 5)",
        "systems": {"top1": run_topk(1), "top5": run_topk(5), "jev_inc": run_jev_inc},
        "pairs": [("jev_inc", "top1"), ("jev_inc", "top5")],
        "descriptions": {
            "top1": "FAISS top-1 por similaridade, resposta direta, sem filtro.",
            "top5": "FAISS top-5 por similaridade, resposta direta, sem filtro (referência).",
            "jev_inc": f"examina os vizinhos um de cada vez, até {rag.INCREMENTAL_STEPS}; o JEV "
            f"avalia a relevância de cada um (>= {rag.MIN_RELEVANCE}) e a busca para quando a "
            f"suficiência dos trechos mantidos chega a {rag.MIN_SUFFICIENCY}.",
        },
    },
    "iter": {
        "prefix": "eval",
        "title": "RAG top-5 x RAG com JEV iterativo",
        "systems": {"top5": run_topk(5), "jev_iter": run_jev_iter},
        "pairs": [("jev_iter", "top5")],
        "descriptions": {
            "top5": "FAISS top-5 por similaridade, resposta direta, sem filtro.",
            "jev_iter": f"filtro de relevância do JEV (>= {rag.MIN_RELEVANCE}), teste de "
            f"suficiência (>= {rag.MIN_SUFFICIENCY}) e busca iterativa com consultas reformuladas "
            f"(até {rag.MAX_ROUNDS} rodadas, top-{rag.TOP_K} por consulta).",
        },
    },
}


def retrieval_metrics(context_ids: list[str], gold: list[str]) -> dict:
    hits = set(context_ids) & set(gold)
    return {
        "gold_recall": len(hits) / len(gold) if gold else None,
        "noise_chunks": len(set(context_ids) - set(gold)),
        "context_size": len(context_ids),
    }


def evaluate(system: RAG, item: dict, exp: dict, rng: random.Random) -> dict:
    q, ref = item["question"], item["reference"]
    runs = {}
    for name, runner in exp["systems"].items():
        start = time.perf_counter()
        runs[name] = runner(system, q)
        runs[name]["latency_s"] = round(time.perf_counter() - start, 2)
        runs[name]["context_ids"] = [c["id"] for c in runs[name]["context"]]

    # judge calls for one question are independent, so run them together
    orders = [(x, y) if rng.random() < 0.5 else (y, x) for x, y in exp["pairs"]]  # blind order
    with ThreadPoolExecutor(max_workers=8) as pool:
        grades = {
            name: pool.submit(
                judge,
                POINTWISE_PROMPT.format(
                    question=q,
                    reference=ref,
                    context=_context(run["context"]) or "(vazio)",
                    answer=run["answer"],
                ),
            )
            for name, run in runs.items()
        }
        pairs = [
            pool.submit(
                judge,
                PAIRWISE_PROMPT.format(question=q, reference=ref, a=runs[a]["answer"], b=runs[b]["answer"]),
            )
            for a, b in orders
        ]
        for name, run in runs.items():
            run["judge"] = grades[name].result()
            run["retrieval"] = retrieval_metrics(run["context_ids"], item["gold"])
            del run["context"]
        pairwise = []
        for (x, y), (a, b), fut in zip(exp["pairs"], orders, pairs):
            verdict = fut.result()
            pairwise.append(
                {
                    "pair": f"{x} x {y}",
                    "winner": {"A": a, "B": b}.get(verdict["winner"], "empate"),
                    "order": [a, b],
                    "reason": verdict["reason"],
                }
            )
    return {**item, "runs": runs, "pairwise": pairwise}


def summarize(rows: list[dict]) -> dict:
    recalls = [r["retrieval"]["gold_recall"] for r in rows if r["retrieval"]["gold_recall"] is not None]
    n = len(rows)
    return {
        "n": n,
        "score": sum(r["judge"]["score"] for r in rows) / n,
        "completeness": sum(r["judge"]["correct_parts"] / r["judge"]["total_parts"] for r in rows) / n,
        "hallucination_rate": sum(r["judge"]["hallucination"] for r in rows) / n,
        "gold_recall": sum(recalls) / len(recalls) if recalls else None,
        "noise_chunks": sum(r["retrieval"]["noise_chunks"] for r in rows) / n,
        "context_size": sum(r["retrieval"]["context_size"] for r in rows) / n,
        "examined": sum(r["examined"] for r in rows) / n,
        "latency_s": sum(r["latency_s"] for r in rows) / n,
    }


def fmt(value, pct: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def yn(flag: bool) -> str:
    return "sim" if flag else "não"


def report(results: list[dict], exp: dict, meta: dict) -> str:
    names = list(exp["systems"])
    pair_names = [f"{x} x {y}" for x, y in exp["pairs"]]
    overall = {n: summarize([r["runs"][n] for r in results]) for n in names}
    unanswerable = [r for r in results if not r["gold"]]
    head = "| Métrica | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"

    def row(label: str, key: str, pct: bool = False) -> str:
        return f"| {label} | " + " | ".join(fmt(overall[n][key], pct) for n in names) + " |"

    lines = [
        f"# Avaliação: {exp['title']}",
        "",
        f"Gerado em {meta['date']}. Juiz: `{meta['judge_model']}`. Gerador (todos): `{meta['llm_model']}`. "
        f"Embeddings (todos): `{meta['embed_model']}`. {len(results)} perguntas, uma execução.",
        "",
        *[f"- **{n}:** {exp['descriptions'][n]}" for n in names],
        "",
        "## Resultado geral",
        "",
        head,
        sep,
        row("Nota do juiz (1 a 5)", "score"),
        row("Completude (partes corretas)", "completeness", True),
        row("Taxa de alucinação", "hallucination_rate", True),
        "| Abstenção correta (perguntas sem resposta) | "
        + " | ".join(
            f"{sum(r['runs'][n]['judge']['abstained'] for r in unanswerable)}/{len(unanswerable)}"
            for n in names
        )
        + " |",
        row("Recall dos trechos gold", "gold_recall", True),
        row("Trechos examinados (média)", "examined"),
        row("Trechos enviados ao LLM (média)", "context_size"),
        row("Trechos não-gold enviados ao LLM (média)", "noise_chunks"),
        row("Latência média (s)", "latency_s"),
        "",
        "## Comparação direta (às cegas)",
        "",
        "| Par | Vitórias do 1º | Vitórias do 2º | Empates |",
        "|---|---|---|---|",
    ]
    for (x, y), pname in zip(exp["pairs"], pair_names):
        verdicts = [p["winner"] for r in results for p in r["pairwise"] if p["pair"] == pname]
        lines.append(
            f"| {pname} | {x}: {verdicts.count(x)} | {y}: {verdicts.count(y)} | {verdicts.count('empate')} |"
        )

    lines += [
        "",
        "## Por categoria (nota média do juiz / completude)",
        "",
        "| Categoria | n | " + " | ".join(names) + " |",
        "|---|---" + "|---" * len(names) + "|",
    ]
    for cat in dict.fromkeys(r["category"] for r in results):
        subset = [r for r in results if r["category"] == cat]
        cells = []
        for n in names:
            s = summarize([r["runs"][n] for r in subset])
            cells.append(f"{fmt(s['score'])} / {fmt(s['completeness'], True)}")
        lines.append(f"| {cat} | {len(subset)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Por pergunta",
        "",
        "| # | Pergunta | "
        + " | ".join(f"Nota {n}" for n in names)
        + " | "
        + " | ".join(f"Recall {n}" for n in names)
        + " | "
        + " | ".join(pair_names)
        + " |",
        "|---|---" + "|---" * (2 * len(names) + len(pair_names)) + "|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r['question']} | "
            + " | ".join(str(r["runs"][n]["judge"]["score"]) for n in names)
            + " | "
            + " | ".join(fmt(r["runs"][n]["retrieval"]["gold_recall"], True) for n in names)
            + " | "
            + " | ".join(p["winner"] for p in r["pairwise"])
            + " |"
        )

    lines += ["", "## Respostas e justificativas do juiz", ""]
    for r in results:
        lines += [f"### {r['id']}: {r['question']}", "", f"Referência: {r['reference']}", ""]
        for n in names:
            run = r["runs"][n]
            extra = ""
            if "steps" in run:
                extra = "; passos: " + ", ".join(
                    f"{s['id']} (rel {s['relevance']}"
                    + ("" if s["sufficiency"] is None else f", suf {s['sufficiency']}")
                    + ")"
                    for s in run["steps"]
                )
            lines += [
                f"**{n}** (nota {run['judge']['score']}, alucinou: {yn(run['judge']['hallucination'])}, "
                f"contexto: {', '.join(run['context_ids']) or 'vazio'}{extra})",
                "",
                f"> {run['answer']}".replace("\n", "\n> "),
                "",
                f"Juiz: {run['judge']['justification']}",
                "",
            ]
        for p in r["pairwise"]:
            lines += [f"Comparação {p['pair']}: **{p['winner']}**. {p['reason']}", ""]

    lines += [
        "## Limitações desta avaliação",
        "",
        f"- {len(results)} perguntas e uma única execução: diferenças pequenas podem ser ruído.",
        "- Base de conhecimento de 20 trechos, escrita para este teste.",
        "- O juiz é um LLM e pode errar. A comparação direta usa ordem aleatória para reduzir viés de posição.",
        "- Recall e trechos não-gold são objetivos (comparados com os ids gold de cada pergunta) e não dependem do juiz.",
    ]
    return "\n".join(lines) + "\n"


def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "k1"
    exp = EXPERIMENTS[exp_name]
    questions = json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))
    system = RAG(HERE / "data.jsonl")
    rng = random.Random(SEED)

    results = []
    for item in questions:
        print(f"{item['id']} {item['question']}", flush=True)
        results.append(evaluate(system, item, exp, rng))
        r = results[-1]
        scores = "  ".join(f"{n} {run['judge']['score']}" for n, run in r["runs"].items())
        verdicts = "  ".join(f"[{p['pair']}: {p['winner']}]" for p in r["pairwise"])
        print(f"    {scores}  {verdicts}", flush=True)

    meta = {
        "experiment": exp_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "judge_model": JUDGE_MODEL,
        "llm_model": rag.LLM_MODEL,
        "embed_model": rag.EMBED_MODEL,
        "top_k": rag.TOP_K,
        "incremental_steps": rag.INCREMENTAL_STEPS,
        "min_relevance": rag.MIN_RELEVANCE,
        "min_sufficiency": rag.MIN_SUFFICIENCY,
        "max_rounds": rag.MAX_ROUNDS,
        "seed": SEED,
    }
    summary = {n: summarize([r["runs"][n] for r in results]) for n in exp["systems"]}

    OUT_DIR.mkdir(exist_ok=True)
    json_path = OUT_DIR / f"{exp['prefix']}_results.json"
    md_path = OUT_DIR / f"{exp['prefix']}_report.md"
    json_path.write_text(
        json.dumps({"meta": meta, "summary": summary, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(report(results, exp, meta), encoding="utf-8")
    print(f"\nSalvo em {json_path} e {md_path}")


if __name__ == "__main__":
    main()
