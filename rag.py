"""Test RAG: JSONL -> in-memory FAISS -> similarity top-k -> JEV relevance filter -> LLM answer.

Retrieval is iterative: while JEV says the kept chunks are not enough to fully answer the
question, the LLM proposes new search queries (rephrasings / sub-questions) and another
round of search + JEV filtering runs, up to MAX_ROUNDS.

Usage: python rag.py "sua pergunta"
"""

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np

import meter
from jev import api_key, decide, noul

BASE_URL = "https://openrouter.ai/api/v1"
EMBED_MODEL = "openai/text-embedding-3-small"
LLM_MODEL = "openai/gpt-4.1-mini"

TOP_K = 5  # candidates per search query (small so a 20-chunk base leaves room for later rounds)
MIN_RELEVANCE = 0.3  # keep every chunk JEV rates at or above this
MAX_CHUNKS = 15  # context-size cap only, not a ranking cutoff
MAX_ROUNDS = 3  # search rounds before giving up
MIN_SUFFICIENCY = 0.5  # JEV probability that the kept chunks fully answer the question
MAX_NEW_QUERIES = 3  # search queries the LLM may propose per extra round
INCREMENTAL_STEPS = 5  # ask_incremental: max neighbors examined, one at a time
GATE_STEPS = 5  # ask_gate: max neighbors examined
GATE_EARLY_STOP = 3  # ask_gate: give up after this many consecutive near-zero chunks...
GATE_NEAR_ZERO = 0.1  # ...where near-zero means relevance below this

EMBED_BATCH = 256  # texts per embeddings request

# Prompts and labels per knowledge-base language (JEV instructions, state labels, answer prompt).
PROMPTS = {
    "pt": {
        "relevance": "O trecho contém informação que ajuda a responder a pergunta",
        "sufficiency": "Os trechos, juntos, contêm toda a informação necessária para responder "
        "completamente a pergunta, incluindo todas as suas partes",
        "gate_relevance": "O trecho novo contém informação que ajuda a responder a pergunta",
        "gate_sufficiency": "Os trechos já aceitos junto com o trecho novo contêm toda a informação "
        "necessária para responder completamente a pergunta, incluindo todas as suas partes",
        "question": "Pergunta",
        "chunk": "Trecho",
        "chunks": "Trechos",
        "accepted": "Trechos já aceitos",
        "new_chunk": "Trecho novo",
        "none": "(nenhum)",
        "context": "Contexto",
        "answer_system": "Responda em português usando apenas o contexto fornecido. "
        "Cite os ids dos trechos usados entre colchetes. "
        "Se o contexto não bastar para alguma parte da pergunta, diga que não sabe essa parte.",
        "no_chunks": "Nenhum trecho relevante encontrado para responder a pergunta.",
    },
    "en": {
        "relevance": "The passage contains information that helps answer the question",
        "sufficiency": "The passages, together, contain all the information needed to fully "
        "answer the question, including all of its parts",
        "gate_relevance": "The new passage contains information that helps answer the question",
        "gate_sufficiency": "The already accepted passages together with the new passage contain "
        "all the information needed to fully answer the question, including all of its parts",
        "question": "Question",
        "chunk": "Passage",
        "chunks": "Passages",
        "accepted": "Already accepted passages",
        "new_chunk": "New passage",
        "none": "(none)",
        "context": "Context",
        "answer_system": "Answer in English using only the provided context. "
        "Cite the ids of the passages you used in square brackets. "
        "If the context is not enough for some part of the question, say you don't know that part.",
        "no_chunks": "No relevant passage was found to answer the question.",
    },
}

def _post(path: str, payload: dict) -> dict:
    kind = "embed" if path == "/embeddings" else "llm"
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps({**payload, "usage": {"include": True}}).encode(),
        headers={"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.load(resp)
            meter.record(kind, result.get("usage"))
            return result
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504, 520, 521, 522, 523, 524) or attempt == 4:
                raise RuntimeError(f"OpenRouter {path} failed ({e.code}): {e.read().decode()}") from e
            if e.code == 429:
                time.sleep(20 * (attempt + 1))  # rate limits reset per minute
                continue
        except OSError:  # timeouts, DNS failures, connection resets
            if attempt == 4:
                raise
        time.sleep(min(30, 2 ** (attempt + 1)))


def _chat(system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    resp = _post("/chat/completions", {"model": LLM_MODEL, "messages": messages})
    return resp["choices"][0]["message"]["content"]


def _context(chunks: list[dict]) -> str:
    return "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts (in batches) and L2-normalize, so inner product == cosine similarity."""
    rows = []
    for i in range(0, len(texts), EMBED_BATCH):
        data = _post("/embeddings", {"model": EMBED_MODEL, "input": texts[i : i + EMBED_BATCH]})["data"]
        rows += [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]
    vectors = np.array(rows, "float32")
    faiss.normalize_L2(vectors)
    return vectors


def _cached_embed(texts: list[str], jsonl_path: Path) -> np.ndarray:
    """Corpus embeddings cached next to the JSONL, keyed by model and content hash."""
    digest = hashlib.sha256((EMBED_MODEL + "\n" + "\n".join(texts)).encode()).hexdigest()[:16]
    cache = jsonl_path.with_name(f"{jsonl_path.stem}.{digest}.npy")
    if cache.exists():
        return np.load(cache)
    vectors = embed(texts)
    np.save(cache, vectors)
    return vectors


class RAG:
    def __init__(self, jsonl_path: str | Path, lang: str = "pt", cache: bool = False):
        jsonl_path = Path(jsonl_path)
        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        self.docs = [json.loads(line) for line in lines if line.strip()]
        self.lang = lang
        self.p = PROMPTS[lang]
        texts = [d["text"] for d in self.docs]
        vectors = _cached_embed(texts, jsonl_path) if cache else embed(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = TOP_K, exclude: set[str] = frozenset()) -> list[dict]:
        """Nearest chunks by cosine similarity, skipping ids already seen."""
        n = min(k + len(exclude), self.index.ntotal)
        sims, ids = self.index.search(embed([query]), n)
        hits = [{**self.docs[i], "similarity": float(s)} for s, i in zip(sims[0], ids[0])]
        return [h for h in hits if h["id"] not in exclude][:k]

    def relevance(self, query: str, hit: dict) -> float:
        """JEV probability that one chunk helps answer the query."""
        p = self.p
        return noul(f"{p['question']}: {query}\n\n{p['chunk']}: {hit['text']}", p["relevance"])

    def rerank(self, query: str, hits: list[dict]) -> list[dict]:
        """JEV probability that each chunk helps answer the query (calls run in parallel)."""
        if not hits:
            return []
        with ThreadPoolExecutor(max_workers=20) as pool:
            probs = list(pool.map(lambda h: self.relevance(query, h), hits))
        ranked = [{**h, "relevance": p} for h, p in zip(hits, probs)]
        return sorted(ranked, key=lambda h: h["relevance"], reverse=True)

    def sufficiency(self, query: str, chunks: list[dict]) -> float:
        """JEV probability that the chunks, together, fully answer the query."""
        if not chunks:
            return 0.0
        p = self.p
        return noul(f"{p['question']}: {query}\n\n{p['chunks']}:\n{_context(chunks)}", p["sufficiency"])

    def next_queries(self, query: str, kept: list[dict], tried: list[str]) -> list[str]:
        """Ask the LLM for new search queries targeting what is still missing."""
        found = _context(kept) if kept else "(nenhum)"
        text = _chat(
            "Você gera consultas para um buscador semântico. Responda apenas com as "
            f"consultas, no máximo {MAX_NEW_QUERIES}, uma por linha, sem numeração.",
            f"Pergunta do usuário: {query}\n\n"
            f"Trechos relevantes já encontrados:\n{found}\n\n"
            f"Consultas já feitas:\n" + "\n".join(tried) + "\n\n"
            "Gere novas consultas para buscar a informação que ainda falta. Se a pergunta "
            "tiver várias partes, faça uma consulta para cada parte ainda não respondida. "
            "Use palavras diferentes das consultas já feitas.",
        )
        queries = [q.strip(" -•\t") for q in text.splitlines()]
        return [q for q in queries if q and q not in tried][:MAX_NEW_QUERIES]

    def answer(self, query: str, chunks: list[dict]) -> str:
        p = self.p
        if not chunks:
            return p["no_chunks"]
        return _chat(
            p["answer_system"],
            f"{p['context']}:\n{_context(chunks)}\n\n{p['question']}: {query}",
        )

    def ask(self, query: str) -> dict:
        seen: set[str] = set()
        kept: list[dict] = []
        tried: list[str] = []
        rounds = []
        queries = [query]

        for n in range(1, MAX_ROUNDS + 1):
            candidates = []
            for q in queries:
                hits = self.search(q, exclude=seen)
                seen.update(h["id"] for h in hits)
                candidates += hits
            tried += queries

            # relevance is always judged against the user's question, not the rephrased query
            ranked = self.rerank(query, candidates)
            kept += [h for h in ranked if h["relevance"] >= MIN_RELEVANCE]
            kept.sort(key=lambda h: h["relevance"], reverse=True)
            kept = kept[:MAX_CHUNKS]

            score = self.sufficiency(query, kept)
            rounds.append({"round": n, "queries": queries, "ranked": ranked, "sufficiency": score})
            if score >= MIN_SUFFICIENCY or n == MAX_ROUNDS or len(seen) == self.index.ntotal:
                break
            queries = self.next_queries(query, kept, tried)
            if not queries:
                break

        return {"rounds": rounds, "kept": kept, "answer": self.answer(query, kept)}

    def ask_incremental(self, query: str, max_steps: int = INCREMENTAL_STEPS) -> dict:
        """Examine nearest neighbors one at a time, stopping as soon as JEV says the kept
        chunks suffice. Sufficiency is only re-checked when a new chunk is kept."""
        kept: list[dict] = []
        steps = []
        # one search ranks all neighbors; walking the list equals fetching one vector at a time
        for hit in self.search(query, k=max_steps):
            hit = {**hit, "relevance": self.relevance(query, hit)}
            score = None
            if hit["relevance"] >= MIN_RELEVANCE:
                kept.append(hit)
                score = self.sufficiency(query, kept)
            steps.append({"hit": hit, "sufficiency": score})
            if score is not None and score >= MIN_SUFFICIENCY:
                break
        return {"steps": steps, "kept": kept, "answer": self.answer(query, kept)}

    def gate_step(self, query: str, kept: list[dict], hit: dict) -> tuple[float, float]:
        """One JEV call judging a new chunk: (relevance of the chunk, sufficiency of kept + chunk)."""
        p = self.p
        state = (
            f"{p['question']}: {query}\n\n"
            f"{p['accepted']}:\n{_context(kept) or p['none']}\n\n"
            f"{p['new_chunk']}: [{hit['id']}] {hit['text']}"
        )
        answers = decide(
            state,
            {
                "rel": {"type": "noul", "instructions": p["gate_relevance"]},
                "suf": {"type": "noul", "instructions": p["gate_sufficiency"]},
            },
        )["answers"]
        return answers["rel"]["noul"], answers["suf"]["noul"]

    def gate_select(
        self, query: str, max_steps: int = GATE_STEPS, min_sufficiency: float | None = None
    ) -> dict:
        """JEV as a gate in front of a top-1 RAG (retrieval only, no answer).

        Fast path: the top-1 chunk is judged sufficient, at the cost of one JEV call.
        Otherwise the next neighbors are examined one at a time (one JEV call each) until the
        kept chunks suffice, max_steps is reached, or GATE_EARLY_STOP consecutive chunks are
        near-irrelevant (likely unanswerable).
        """
        threshold = MIN_SUFFICIENCY if min_sufficiency is None else min_sufficiency
        kept: list[dict] = []
        steps = []
        near_zero = 0
        for hit in self.search(query, k=max_steps):
            rel, suf = self.gate_step(query, kept, hit)
            hit = {**hit, "relevance": rel}
            if rel >= MIN_RELEVANCE:
                kept.append(hit)
            steps.append({"id": hit["id"], "relevance": rel, "sufficiency": suf})
            near_zero = near_zero + 1 if rel < GATE_NEAR_ZERO else 0
            if suf >= threshold or near_zero >= GATE_EARLY_STOP:
                break
        return {"steps": steps, "kept": kept}

    def ask_gate(
        self, query: str, max_steps: int = GATE_STEPS, min_sufficiency: float | None = None
    ) -> dict:
        """gate_select() followed by the answer."""
        result = self.gate_select(query, max_steps, min_sufficiency)
        return {**result, "answer": self.answer(query, result["kept"])}


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Quando o Cristo Redentor foi inaugurado e qual a sua altura?"
    rag = RAG(Path(__file__).with_name("data.jsonl"))
    print(f"{len(rag.docs)} trechos indexados no FAISS ({rag.index.d} dimensões)")
    print(f"Pergunta: {query}")

    result = rag.ask(query)
    for r in result["rounds"]:
        print(f"\n--- Rodada {r['round']} ---")
        for q in r["queries"]:
            print(f"  busca: {q}")
        for h in r["ranked"]:
            mark = "OK " if h["relevance"] >= MIN_RELEVANCE else " x "
            print(f"  {mark} jev {h['relevance']:.2f}  sim {h['similarity']:.3f}  {h['id']}")
        verdict = "basta" if r["sufficiency"] >= MIN_SUFFICIENCY else "não basta"
        print(f"  suficiência: {r['sufficiency']:.2f} ({verdict})")

    print(f"\nTrechos usados: {', '.join(h['id'] for h in result['kept']) or 'nenhum'}")
    print(f"\nResposta ({LLM_MODEL}):\n{result['answer']}")


if __name__ == "__main__":
    main()
