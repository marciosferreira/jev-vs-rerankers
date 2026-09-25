"""Retrieval-level experiment: can JEV tell "answers the question" from "merely similar"?

For each question, the embedding retriever brings the top-K chunks. Every chunk is then scored
by four methods, and the scores are compared against the ground truth (does the chunk contain
the answer?). No generator and no judge are involved.

  cosine   embedding similarity (what a plain RAG ranks by)
  jev      JEV relevance: "the passage contains information that helps answer the question"
  minilm   cross-encoder reranker Xenova/ms-marco-MiniLM-L-12-v2 (small, fast, local)
  bge      cross-encoder reranker BAAI/bge-reranker-base (larger, local)

JEV also answers one question no reranker can: whether the top-N chunks, together, contain the
answer (set sufficiency). It is compared with each method's max chunk score as a detector of
"is the answer among the candidates at all?".

A chunk "answers" a question when it is the gold chunk or contains an accepted answer
(>= 4 characters). Unanswerable questions have no answering chunk.

Thresholds for the fixed-cut comparison are chosen on the tuning questions and applied to the
test questions. Per-question results go to results/<prefix>_partial.jsonl (resumable).

Usage:
  python rerank_experiment.py [--limit N] [--dry-run] [--report-only] [--prefix rerank]
"""

import argparse
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import meter
from rag import RAG

HERE = Path(__file__).parent
OUT_DIR = HERE / "results"
DATA = "data/squad/corpus.jsonl"
TEST = "data/squad/questions_test.json"
TUNE = "data/squad/questions_tune.json"
K = 10  # candidates per question from the embedding retriever
SET_N = 5  # chunks in the JEV set-sufficiency question
RERANKERS = {"minilm": "Xenova/ms-marco-MiniLM-L-12-v2", "bge": "BAAI/bge-reranker-base"}
# newer rerankers (sentence-transformers + PyTorch), added to saved results with --add
EXTRA_RERANKERS = {
    "bge_m3": "BAAI/bge-reranker-v2-m3",
    "qwen3": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
    "cohere": "cohere/rerank-v4.0-pro",  # API; needs COHERE_API_KEY in .env
}
COHERE_MIN_INTERVAL_S = 6.5  # trial keys allow ~10 rerank calls per minute
QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
METHODS = ["cosine", "jev", *RERANKERS]
MIN_ANSWER_MATCH = 4
SEED = 42
BOOTSTRAP = 5000


def answers(item: dict, chunk: dict) -> bool:
    if not item["gold"]:
        return False
    if chunk["id"] in item["gold"]:
        return True
    text = chunk["text"].lower()
    return any(a.lower() in text for a in item.get("answers", []) if len(a) >= MIN_ANSWER_MATCH)


# ---------- scoring ----------


def score_question(system: RAG, rerankers: dict, item: dict, split: str) -> dict:
    q = item["question"]
    hits = system.search(q, k=K)
    labels = [answers(item, h) for h in hits]
    row = {
        "id": item["id"],
        "split": split,
        "category": item["category"],
        "question": q,
        "candidates": [h["id"] for h in hits],
        "labels": labels,
        "scores": {"cosine": [h["similarity"] for h in hits]},
        "latency_s": {"cosine": 0.0},
    }

    # JEV: the K relevance calls and the set-sufficiency call, all in parallel
    start = time.perf_counter()
    with meter.metered() as m, ThreadPoolExecutor(max_workers=K + 1) as pool:
        rel = [pool.submit(meter.bind(system.relevance), q, h) for h in hits]
        suf = pool.submit(meter.bind(system.sufficiency), q, hits[:SET_N])
        row["scores"]["jev"] = [f.result() for f in rel]
        row["jev_set_sufficiency"] = suf.result()
    row["latency_s"]["jev"] = round(time.perf_counter() - start, 3)
    row["jev_cost"] = meter.total(m)
    row["jev_missing_cost"] = m["missing_cost"]

    docs = [h["text"] for h in hits]
    for name, model in rerankers.items():
        start = time.perf_counter()
        row["scores"][name] = [float(s) for s in model.rerank(q, docs)]
        row["latency_s"][name] = round(time.perf_counter() - start, 3)
    return row


# ---------- metrics ----------


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability that a random positive outscores a random negative (ties count half)."""
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank of the tie block
        rank_sum += avg_rank * sum(lbl for _, lbl in ranked[i:j])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def top1(row: dict, method: str) -> bool:
    scores = row["scores"][method]
    return row["labels"][max(range(len(scores)), key=scores.__getitem__)]


def mrr(row: dict, method: str) -> float:
    order = sorted(range(len(row["labels"])), key=lambda i: -row["scores"][method][i])
    for rank, i in enumerate(order, 1):
        if row["labels"][i]:
            return 1 / rank
    return 0.0


def best_threshold(rows: list[dict], method: str) -> float:
    """Cut that maximizes F1 of 'keep chunk' decisions over the given rows."""
    pairs = [(s, lbl) for r in rows for s, lbl in zip(r["scores"][method], r["labels"])]
    n_pos = sum(lbl for _, lbl in pairs)
    best, best_f1 = math.inf, -1.0
    for t in sorted({s for s, _ in pairs}):
        tp = sum(1 for s, lbl in pairs if s >= t and lbl)
        kept = sum(1 for s, _ in pairs if s >= t)
        f1 = 2 * tp / (kept + n_pos) if kept + n_pos else 0.0
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def cut_metrics(rows: list[dict], method: str, t: float) -> dict:
    tp = kept = pos = served = abstain_ok = n_ans = n_none = 0
    for r in rows:
        keep = [s >= t for s in r["scores"][method]]
        tp += sum(k and lbl for k, lbl in zip(keep, r["labels"]))
        kept += sum(keep)
        pos += sum(r["labels"])
        if any(r["labels"]):
            n_ans += 1
            served += any(k and lbl for k, lbl in zip(keep, r["labels"]))
        else:
            n_none += 1
            abstain_ok += not any(keep)
    return {
        "precision": tp / kept if kept else 0.0,
        "recall": tp / pos if pos else 0.0,
        "chunks_kept": kept / len(rows),
        "served": served / n_ans if n_ans else None,  # an answering chunk survived the cut
        "abstain_ok": abstain_ok / n_none if n_none else None,  # nothing kept when nothing answers
    }


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def bootstrap_ci(a: list[bool], b: list[bool], rng: random.Random) -> tuple[float, float]:
    n = len(a)
    diffs = sorted(sum(b[i] - a[i] for i in idx) / n for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(BOOTSTRAP)))
    return diffs[int(0.025 * BOOTSTRAP)], diffs[int(0.975 * BOOTSTRAP) - 1]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


# ---------- report ----------


def report(rows: list[dict], meta: dict) -> str:
    rng = random.Random(SEED)
    METHODS = ["cosine", "jev", *meta["rerankers"]]
    tune = [r for r in rows if r["split"] == "tune"]
    test = [r for r in rows if r["split"] == "test"]
    recoverable = [r for r in test if any(r["labels"])]
    answerable = [r for r in test if r["category"] == "answerable"]
    no_answer = [r for r in test if not any(r["labels"])]
    traps = [r for r in recoverable if not top1(r, "cosine")]
    n = len(test)

    def pct(x):
        return "-" if x is None else f"{x:.0%}"

    lines = [
        "# Experimento: o JEV distingue \"responde\" de \"só parece\"?",
        "",
        f"Gerado em {meta['date']}. Base: `{meta['data']}` ({meta['n_docs']} trechos). "
        f"Perguntas de teste: {n} ({len(answerable)} com resposta, {n - len(answerable)} sem resposta); "
        f"perguntas de ajuste (só para escolher os cortes): {len(tune)}. "
        f"O embedding traz {K} candidatos por pergunta; cada candidato recebe uma nota de cada método.",
        "",
        "- **cosine:** similaridade do embedding (`openai/text-embedding-3-small`), o que um RAG comum usa.",
        "- **jev:** relevância do JEV (`typesafe/jev-1.13`): \"o trecho contém informação que ajuda a responder a pergunta\".",
        *[f"- **{name}:** reranker `{model}` (local)." for name, model in meta["rerankers"].items()],
        "",
        "Um trecho \"responde\" quando é o trecho gold ou contém uma das respostas aceitas.",
        "",
        "## 1. Quem põe em 1º lugar um trecho que responde?",
        "",
        f"Considerando as {len(recoverable)} perguntas de teste em que algum dos {K} candidatos responde "
        f"(nas outras {n - len(recoverable)}, nenhum método pode acertar).",
        "",
        "| Método | 1º lugar responde | MRR | Armadilhas corrigidas |",
        "|---|---|---|---|",
    ]
    for m in METHODS:
        p1 = sum(top1(r, m) for r in recoverable) / len(recoverable)
        mr = sum(mrr(r, m) for r in recoverable) / len(recoverable)
        fixed = f"{sum(top1(r, m) for r in traps)}/{len(traps)}" if m != "cosine" else "-"
        lines.append(f"| {m} | {p1:.0%} | {mr:.2f} | {fixed} |")
    lines += [
        "",
        f"**Armadilhas:** perguntas em que o trecho mais parecido (1º do cosine) **não** responde, mas outro "
        f"candidato responde. Foram {len(traps)} de {len(recoverable)} ({len(traps) / len(recoverable):.0%}). "
        "A coluna mostra em quantas delas o método pôs um trecho que responde em 1º lugar.",
        "",
        "**MRR:** média de 1/posição do primeiro trecho que responde (1 = sempre em 1º; 0.5 = em média em 2º).",
        "",
        "### JEV contra cada método (1º lugar responde, pareado)",
        "",
        "| Contra | JEV acerta e o outro erra | O outro acerta e JEV erra | McNemar p | Diferença (IC 95%) |",
        "|---|---|---|---|---|",
    ]
    jev = [top1(r, "jev") for r in recoverable]
    for m in METHODS:
        if m == "jev":
            continue
        other = [top1(r, m) for r in recoverable]
        b = sum(1 for x, y in zip(other, jev) if not x and y)
        c = sum(1 for x, y in zip(other, jev) if x and not y)
        lo, hi = bootstrap_ci(other, jev, rng)
        lines.append(
            f"| {m} | {b} | {c} | {mcnemar_p(b, c):.3f} | {(sum(jev) - sum(other)) / len(jev):+.0%} ({lo:+.0%} a {hi:+.0%}) |"
        )

    lines += [
        "",
        "## 2. A nota separa \"responde\" de \"só parece\" com um corte fixo?",
        "",
        "Na prática, um corte fixo decide quais trechos vão para o LLM. Aqui o corte de cada método foi "
        "escolhido nas perguntas de ajuste (maior F1) e aplicado nas de teste.",
        "",
        "| Método | Separação (AUC) | Corte | Precisão | Recall | Trechos mantidos por pergunta | "
        "Perguntas com um trecho que responde mantido | Perguntas sem resposta em que nada passou |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in METHODS:
        pos = [s for r in test for s, lbl in zip(r["scores"][m], r["labels"]) if lbl]
        neg = [s for r in test for s, lbl in zip(r["scores"][m], r["labels"]) if not lbl]
        t = best_threshold(tune or test, m)
        c = cut_metrics(test, m, t)
        lines.append(
            f"| {m} | {auc(pos, neg):.3f} | {t:.3g} | {c['precision']:.0%} | {c['recall']:.0%} | "
            f"{c['chunks_kept']:.2f} | {pct(c['served'])} | {pct(c['abstain_ok'])} |"
        )
    lines += [
        "",
        "**AUC:** chance de um trecho que responde ter nota maior que um que não responde, comparando "
        "trechos de perguntas diferentes (0.5 = sorte; 1.0 = separação perfeita). É o que importa para um "
        "corte fixo funcionar em qualquer pergunta.",
        "",
        "## 3. O conjunto de candidatos tem a resposta? (o que só o JEV faz)",
        "",
        f"Antes de chamar o LLM, dá para saber se a resposta está entre os candidatos? Das {n} perguntas de teste, "
        f"{len(recoverable)} têm a resposta entre os {K} candidatos e {len(no_answer)} não têm. "
        "Os rerankers só podem usar a maior nota individual; o JEV também pode avaliar o conjunto "
        f"(\"os {SET_N} primeiros trechos, juntos, respondem a pergunta?\").",
        "",
        "| Sinal | Separação (AUC) |",
        "|---|---|",
    ]
    for m in METHODS:
        pos = [max(r["scores"][m]) for r in recoverable]
        neg = [max(r["scores"][m]) for r in no_answer]
        lines.append(f"| maior nota do {m} | {auc(pos, neg):.3f} |")
    in_top = [r for r in test if any(r["labels"][:SET_N])]
    out_top = [r for r in test if not any(r["labels"][:SET_N])]
    lines.append(
        f"| JEV, suficiência do conjunto (top-{SET_N}) | "
        f"{auc([r['jev_set_sufficiency'] for r in in_top], [r['jev_set_sufficiency'] for r in out_top]):.3f} |"
    )
    lines += [
        "",
        f"Para a suficiência do conjunto, \"tem a resposta\" significa que algum dos {SET_N} primeiros candidatos "
        "responde.",
        "",
        "## 4. Custo e latência para avaliar os candidatos",
        "",
        "| Método | Latência p50 por pergunta (s) | Latência p95 (s) | Custo por pergunta | Onde roda |",
        "|---|---|---|---|---|",
        "| cosine | 0 (já calculado na busca) | 0 | 0 | junto com a busca |",
    ]
    jev_cost = sum(r["jev_cost"] for r in test) / n
    lines.append(
        f"| jev | {percentile([r['latency_s']['jev'] for r in test], 50):.2f} | "
        f"{percentile([r['latency_s']['jev'] for r in test], 95):.2f} | US$ {jev_cost:.6f} (medido) | "
        f"API, {K + 1} chamadas em paralelo |"
    )
    for m in meta["rerankers"]:
        api = m == "cohere"
        measured = meta.get("latency_measured", {}).get(m)
        if measured:  # dedicated measurement, without rate-limit waits
            lat = (f"{percentile(measured, 50):.2f} | {percentile(measured, 95):.2f} "
                   f"(medida à parte, {len(measured)} chamadas)")
        elif m in meta.get("latency_invalid", {}):
            lat = f"não medida: {meta['latency_invalid'][m]} | -"
        else:
            lat = (f"{percentile([r['latency_s'][m] for r in test], 50):.2f} | "
                   f"{percentile([r['latency_s'][m] for r in test], 95):.2f}")
        cost = "cobrado por busca (aqui, cota de teste gratuita)" if api else "0 (CPU local)"
        where = "API, 1 chamada com os 10 trechos" if api else "este computador, sem GPU"
        lines.append(f"| {m} | {lat} | {cost} | {where} |")
    missing = sum(r["jev_missing_cost"] for r in test)
    lines += [
        "",
        f"Custo do JEV medido pelo `usage.cost` do OpenRouter; {missing} chamadas vieram sem custo informado. "
        f"Por mil perguntas: US$ {jev_cost * 1000:.2f}.",
        "",
        "## Limitações",
        "",
        "- Uma base (SQuAD 2.0, Wikipédia em inglês) e uma execução. O comportamento pode mudar em outros domínios.",
        "- Os rerankers leem no máximo ~512 tokens por par; parágrafos maiores são truncados.",
        "- A latência dos rerankers depende da CPU desta máquina; com GPU, cairia bastante. A do JEV depende da rede.",
        "- A verdade usa o trecho gold ou a presença de uma resposta aceita no texto, o que pode errar em casos raros.",
    ]
    return "\n".join(lines) + "\n"


# ---------- adding a reranker to saved results ----------


def add_reranker(name: str, final: Path, md: Path) -> None:
    """Score the saved candidates with one of EXTRA_RERANKERS, without re-running JEV or the
    retriever. Scores go to results/<prefix>_<name>_scores.jsonl (resumable, kept)."""
    saved =json.loads(final.read_text(encoding="utf-8"))
    rows = saved["rows"]
    docs = {d["id"]: d["text"] for d in map(json.loads, (HERE / DATA).read_text(encoding="utf-8").splitlines())}
    # kept permanently: re-running only merges these scores again (no model/API calls)
    scores_path = final.with_name(final.name.replace("_results.json", f"_{name}_scores.jsonl"))
    done = {}
    if scores_path.exists():
        for line in scores_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["id"], r["split"])] = r
        print(f"Retomando {name}: {len(done)} perguntas já avaliadas.", flush=True)

    model_id = EXTRA_RERANKERS[name]
    if len(done) < len(rows):
        score = cohere_scorer(model_id) if name == "cohere" else local_scorer(name, model_id)
    for i, row in enumerate(rows, 1):
        key = (row["id"], row["split"])
        if key in done:
            continue
        texts = [docs[c] for c in row["candidates"]]
        scores, latency = score(row["question"], texts)
        done[key] = {"id": row["id"], "split": row["split"], "scores": scores, "latency_s": round(latency, 3)}
        with scores_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(done[key]) + "\n")
        ok = "✓" if any(row["labels"]) and row["labels"][max(range(K), key=scores.__getitem__)] else "·"
        print(f"[{i}/{len(rows)}] {row['split']} {row['id']}  {name}{ok}  {done[key]['latency_s']:.1f}s", flush=True)

    # re-read before writing, so rerankers added by other runs in the meantime are kept
    saved = json.loads(final.read_text(encoding="utf-8"))
    for row in saved["rows"]:
        row["scores"][name] = done[(row["id"], row["split"])]["scores"]
        row["latency_s"][name] = done[(row["id"], row["split"])]["latency_s"]
    saved["meta"]["rerankers"][name] = model_id
    final.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    md.write_text(report(saved["rows"], saved["meta"]), encoding="utf-8")
    print(f"\n{name} acrescentado. Relatório refeito: {md}")


def local_scorer(name: str, model_id: str):
    import warnings

    from sentence_transformers import CrossEncoder

    warnings.filterwarnings("ignore")
    model = CrossEncoder(model_id, max_length=512)

    def score(question: str, texts: list[str]) -> list[float]:
        if name == "qwen3":
            pairs = [(f"{QWEN3_PREFIX}<Query>: {question}\n", f"<Document>: {t}{QWEN3_SUFFIX}") for t in texts]
        else:
            pairs = [(question, t) for t in texts]
        start = time.perf_counter()
        # single-logit CrossEncoders already apply a sigmoid, so scores are 0-1
        scores = [float(s) for s in model.predict(pairs, batch_size=K)]
        return scores, time.perf_counter() - start

    return score


def cohere_scorer(model_id: str):
    """Cohere Rerank API (v2). Trial keys are rate limited, so calls are spaced out and
    429s are retried after a pause."""
    key = env_key("COHERE_API_KEY")
    model = model_id.removeprefix("cohere/")
    last = [0.0]

    def score(question: str, texts: list[str]) -> list[float]:
        for attempt in range(6):
            wait = last[0] + COHERE_MIN_INTERVAL_S - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
            req = urllib.request.Request(
                "https://api.cohere.com/v2/rerank",
                data=json.dumps({"model": model, "query": question, "documents": texts}).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
            try:
                start = time.perf_counter()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    results = json.load(resp)["results"]
                latency = time.perf_counter() - start
                scores = [0.0] * len(texts)
                for r in results:
                    scores[r["index"]] = r["relevance_score"]
                return scores, latency
            except urllib.error.HTTPError as e:
                body = e.read().decode()
                if e.code != 429 and e.code < 500 or attempt == 5:
                    raise RuntimeError(f"Cohere rerank failed ({e.code}): {body}") from e
                print(f"  Cohere {e.code}, esperando 60 s: {body[:150]}", flush=True)
                time.sleep(60)
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(10)

    return score


def env_key(name: str) -> str:
    """Read a key from the environment or from .env (case-insensitive name)."""
    if os.environ.get(name):
        return os.environ[name]
    for line in (HERE / ".env").read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip().lower() == name.lower():
            return v.strip().strip('"').strip("'")
    raise RuntimeError(f"{name} não encontrada no ambiente nem no .env")


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="use only the first N test (and N/3 tuning) questions")
    parser.add_argument("--prefix", default="rerank")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--add", choices=sorted(EXTRA_RERANKERS), help="add a newer reranker to saved results")
    parser.add_argument("--cohere-latency", type=int, metavar="N", help="measure Cohere API latency on N questions")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    partial = OUT_DIR / f"{args.prefix}_partial.jsonl"
    final = OUT_DIR / f"{args.prefix}_results.json"
    md = OUT_DIR / f"{args.prefix}_report.md"

    if args.add:
        add_reranker(args.add, final, md)
        return

    if args.cohere_latency:
        # latency of the Cohere API alone (the scorer spaces calls out but times only the request)
        saved = json.loads(final.read_text(encoding="utf-8"))
        docs = {d["id"]: d["text"] for d in map(json.loads, (HERE / DATA).read_text(encoding="utf-8").splitlines())}
        score = cohere_scorer(EXTRA_RERANKERS["cohere"])
        test_rows = [r for r in saved["rows"] if r["split"] == "test"][: args.cohere_latency]
        times = []
        for i, row in enumerate(test_rows, 1):
            _, latency = score(row["question"], [docs[c] for c in row["candidates"]])
            times.append(round(latency, 3))
            print(f"[{i}/{len(test_rows)}] {latency:.2f}s", flush=True)
        saved = json.loads(final.read_text(encoding="utf-8"))  # re-read: other runs may have written
        saved["meta"].setdefault("latency_measured", {})["cohere"] = times
        saved["meta"].get("latency_invalid", {}).pop("cohere", None)
        final.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        md.write_text(report(saved["rows"], saved["meta"]), encoding="utf-8")
        print(f"p50 {percentile(times, 50):.2f}s  p95 {percentile(times, 95):.2f}s. Relatório refeito: {md}")
        return

    if args.report_only:
        saved = json.loads(final.read_text(encoding="utf-8"))
        md.write_text(report(saved["rows"], saved["meta"]), encoding="utf-8")
        print(f"Relatório refeito: {md}")
        return

    test = json.loads((HERE / TEST).read_text(encoding="utf-8"))
    tune = json.loads((HERE / TUNE).read_text(encoding="utf-8"))
    if args.limit:
        test, tune = test[: args.limit], tune[: max(1, args.limit // 3)]
    work = [(q, "tune") for q in tune] + [(q, "test") for q in test]

    if args.dry_run:
        calls = len(work) * (K + 1)
        print(f"Perguntas: {len(test)} de teste + {len(tune)} de ajuste. Chamadas ao JEV: {calls} "
              f"(~US$ {calls * 0.000025:.2f}). Rerankers locais: {len(work) * K * len(RERANKERS)} pares.")
        return

    done = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[(row["id"], row["split"])] = row
        print(f"Retomando: {len(done)} perguntas já avaliadas.", flush=True)

    from fastembed.rerank.cross_encoder import TextCrossEncoder

    print("Carregando rerankers (o primeiro uso baixa os modelos)...", flush=True)
    rerankers = {name: TextCrossEncoder(model_name=model) for name, model in RERANKERS.items()}
    system = RAG(HERE / DATA, lang="en", cache=True)

    for i, (item, split) in enumerate(work, 1):
        if (item["id"], split) in done:
            continue
        row = score_question(system, rerankers, item, split)
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[(item["id"], split)] = row
        best = {m: "✓" if any(row["labels"]) and top1(row, m) else "·" for m in METHODS}
        print(f"[{i}/{len(work)}] {split} {item['id']}  " + " ".join(f"{m}{best[m]}" for m in METHODS), flush=True)

    rows = [done[(q["id"], s)] for q, s in work]
    meta = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": DATA,
        "n_docs": len(system.docs),
        "k": K,
        "set_n": SET_N,
        "rerankers": RERANKERS,
    }
    final.write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    md.write_text(report(rows, meta), encoding="utf-8")
    partial.unlink()
    print(f"\nSalvo em {final} e {md}")


if __name__ == "__main__":
    main()
