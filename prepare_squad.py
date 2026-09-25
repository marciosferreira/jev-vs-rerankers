"""Download SQuAD 2.0 (dev) and convert it to this project's format.

Outputs (in data/squad/):
  corpus.jsonl          every dev paragraph as a chunk {"id", "title", "text"}
  questions_tune.json   questions for tuning thresholds (never used in the final report)
  questions_test.json   questions for the experiment
Question format matches eval_questions.json:
  {"id", "category": "answerable" | "unanswerable", "question", "gold": [chunk id] or [],
   "reference", "answers", "source_paragraph"}

SQuAD 2.0 unanswerable questions were written to look answerable from a specific paragraph;
that paragraph (source_paragraph) is a deliberate distractor, so their gold list is empty.

No API calls. Usage:
  python prepare_squad.py [--n-test 300] [--n-tune 100] [--unanswerable-frac 0.2] [--seed 42]
"""

import argparse
import json
import random
import urllib.request
from pathlib import Path

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
OUT = Path(__file__).parent / "data" / "squad"
UNANSWERABLE_REFERENCE = (
    "The knowledge base does not contain this information. "
    "The correct response is to say it does not know."
)


def download() -> dict:
    raw = OUT / "dev-v2.0.json"
    if not raw.exists():
        print(f"Baixando {URL} ...")
        urllib.request.urlretrieve(URL, raw)
    return json.loads(raw.read_text(encoding="utf-8"))


def convert(squad: dict) -> tuple[list[dict], list[dict]]:
    corpus, questions = [], []
    for article in squad["data"]:
        title = article["title"].replace("_", " ")
        for paragraph in article["paragraphs"]:
            pid = f"p{len(corpus):04d}"
            # the title is prepended, as most real RAG pipelines keep the document title with the chunk
            corpus.append({"id": pid, "title": title, "text": f"{title}: {paragraph['context']}"})
            for qa in paragraph["qas"]:
                answers = list(dict.fromkeys(a["text"].strip() for a in qa["answers"]))
                unanswerable = qa.get("is_impossible", False)
                questions.append(
                    {
                        "id": qa["id"],
                        "category": "unanswerable" if unanswerable else "answerable",
                        "question": qa["question"].strip(),
                        "gold": [] if unanswerable else [pid],
                        "reference": UNANSWERABLE_REFERENCE
                        if unanswerable
                        else "Accepted answers (any one is correct): " + " | ".join(answers),
                        "answers": answers,
                        "source_paragraph": pid,
                    }
                )
    return corpus, questions


def sample(questions: list[dict], n: int, unanswerable_frac: float, rng: random.Random) -> list[dict]:
    pools = {
        cat: [q for q in questions if q["category"] == cat] for cat in ("answerable", "unanswerable")
    }
    n_un = round(n * unanswerable_frac)
    picked = rng.sample(pools["unanswerable"], n_un) + rng.sample(pools["answerable"], n - n_un)
    rng.shuffle(picked)
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--n-tune", type=int, default=100)
    parser.add_argument(
        "--unanswerable-frac",
        type=float,
        default=0.2,
        help="share of unanswerable questions (SQuAD dev has ~50%%; real traffic usually has fewer)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    corpus, questions = convert(download())
    # drop very short questions, which are often unintelligible without their paragraph
    questions = [q for q in questions if len(q["question"].split()) >= 4]

    rng = random.Random(args.seed)
    picked = sample(questions, args.n_test + args.n_tune, args.unanswerable_frac, rng)
    tune, test = picked[: args.n_tune], picked[args.n_tune :]

    with (OUT / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    for name, items in (("questions_tune.json", tune), ("questions_test.json", test)):
        (OUT / name).write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")

    def describe(items: list[dict]) -> str:
        un = sum(q["category"] == "unanswerable" for q in items)
        return f"{len(items)} perguntas ({len(items) - un} com resposta, {un} sem resposta)"

    print(f"Corpus: {len(corpus)} parágrafos -> {OUT / 'corpus.jsonl'}")
    print(f"Ajuste: {describe(tune)} -> {OUT / 'questions_tune.json'}")
    print(f"Teste:  {describe(test)} -> {OUT / 'questions_test.json'}")


if __name__ == "__main__":
    main()
