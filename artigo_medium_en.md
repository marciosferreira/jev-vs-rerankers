# JEV Gating Beats Top-k: 8× Fewer RAG Hallucinations at 1/7 the Cost

## Rerankers are great at ordering passages. They can't say "none of these answer the question." A decision model can.

> **A note before you start.** This article isn't meant to pass as my own writing. Its purpose is to share an experiment that tested an idea of mine, and whose results, within the limits described below, support it. To get the results out quickly, the text was drafted by AI (Claude Opus 5.5) and reviewed and edited by me. The experiment code was also written with it, under my direction. The idea, the thesis and the decisions are mine, and every number comes from the code and results published in the repository. More details in "How this was made", at the end.
>
> **Disclosure:** I have no affiliation with, and received no funding or credits from, TypeSafe, Cohere, Anthropic, OpenAI or OpenRouter. All API usage was paid for by me (Cohere via its free trial key).

---

## TL;DR

We ran 299 SQuAD 2.0 questions through four RAG context policies and had an LLM judge grade every answer.

- **Sending the LLM the top 3 passages**, ranked either by embeddings or by **Cohere Rerank 4 Pro**, produced **23–24 hallucinated answers** and **46–48 confidently wrong answers** in total.
- **Letting JEV decide what the LLM sees** cut that to **3 hallucinations** and **13 confidently wrong answers**. JEV is a decision model that returns a yes/no probability. We kept every passage above a cut-off and skipped the LLM entirely when nothing passed. Overall accuracy was a statistical tie (83.6% vs 81.9%).
- **Total cost** (passage scoring + generation) was **$0.39 per 1,000 questions, vs $2.84 with Cohere top-3.** That is ~7× cheaper, and the passage-scoring step alone is ~10× cheaper.
- **The trade-off:** more "I don't know" answers on questions that *did* have an answer (15% vs 2%).

---

## The problem: "similar" is not "answers the question"

A standard RAG pipeline embeds your documents, retrieves the *k* passages most similar to the question, optionally reranks them, and sends the top few to the LLM. This has two weak points:

1. **Embeddings measure topic similarity, not usefulness.** A passage about *who designed* a monument looks very similar to *"when was the monument inaugurated?"*, but doesn't answer it. In our test, in **15% of the questions that had a useful candidate**, the top embedding hit did not contain the answer, even though another retrieved passage did.
2. **Nothing in the pipeline can say "there is no answer here."** Top-*k* always sends *k* passages. When the answer isn't in your documents, the LLM gets plausible but irrelevant context, and it tends to "help" by answering anyway.

Rerankers fix the first problem, not the second. They are trained to **order** passages and are almost always used as "give me the top *k*", so there is always a "best" passage to send, even when every candidate is useless.

## The idea: let a decision model gate the context

For each retrieved passage, ask *"does this passage help answer the question, yes or no?"*. Send the LLM **only the passages that pass**. If none pass, **don't call the LLM**: return a fixed *"I couldn't find this in the documents I searched."*

We used **[JEV](https://docs.typesafe.ai/introduction/quickstart)** (`typesafe/jev-1.13`, by TypeSafe, available through OpenRouter). JEV doesn't generate text. You send it a text and typed questions (yes/no, multiple choice or scale), and it returns probabilities. It is billed per token; in our test one call cost about $0.000025. For each passage we asked:

> *"The passage contains information that helps answer the question"*

The property that matters: **JEV's probabilities were comparable across questions, so a single cut-off transferred well from the tuning questions to the test questions.** Reranker scores are designed to compare passages *of the same query*, and the same kind of fixed cut-off transferred much worse (details below). Both cut-offs, JEV's and Cohere's, were tuned the same way.

## The experiment

- **Knowledge base:** all **1,204 paragraphs** of the SQuAD 2.0 dev set (English Wikipedia).
- **Questions:** 300 test questions, **246 answerable and 54 unanswerable**. SQuAD 2.0's unanswerable questions were written to *look* answerable from a specific paragraph, so they are deliberate traps. One answerable question was excluded because the judge's safety filter refused to grade it (it was about immune evasion), which leaves **299 questions: 245 answerable, 54 unanswerable**.
- **Retrieval:** `text-embedding-3-small` and FAISS, **10 candidates per question**, identical for every method.
- **Cut-offs:** tuned on a separate set of 100 questions (best F1 at keeping useful passages), then frozen. JEV: 0.97. Cohere: 0.891.
- **Generator:** `gpt-4.1-mini`, told to answer only from the context and to say it doesn't know otherwise.
- **Judge:** `claude-sonnet-5`, which graded every answer against SQuAD's reference.
- **Costs:** measured from the usage OpenRouter reports for each call. Cohere at list price.

**How the judge's grades are defined:**

- **Correct:** every part of the question is answered correctly. For an unanswerable question, the only correct answer is "I don't know".
- **Hallucination:** the answer states something that is not in the context the LLM received, or that contradicts the reference.
- **Confidently wrong:** the answer is not correct and does not say "I don't know". This includes hallucinations, and also answering an unanswerable question from a plausible passage. The judge doesn't flag that second case as a hallucination, because the claim is in the context, but for the user it's just as wrong.
- **Incorrect "I don't know":** the answer says it doesn't know but isn't correct. This is either a false abstention (the answer existed) or, in a few cases, an abstention mixed with an unsupported claim.

Every answer falls in exactly one of three groups: correct, confidently wrong, or incorrect "I don't know". Hallucinations are a subset of the confidently wrong answers.

**The four context policies:**

![The four context policies: top3, cohere_top3, cohere_cut and jev_cut](https://raw.githubusercontent.com/marciosferreira/jev-vs-rerankers/main/figures/fig1_policies.png)
*The four context policies compared.*

## Results

### Wrong answers and accuracy (299 questions)

![Hallucinated answers: top3 24, cohere_top3 23, cohere_cut 7, jev_cut 3. Confidently wrong answers: 46, 48, 24, 13. Correct answers: 81.3%, 81.9%, 79.3%, 83.6%](https://raw.githubusercontent.com/marciosferreira/jev-vs-rerankers/main/figures/fig2_results.png)
*End-to-end results on 299 SQuAD 2.0 questions, graded by an LLM judge.*

- **Against Cohere top-3:** in the questions where only one of the two hallucinated, JEV came out ahead **21 to 1**. For confidently wrong answers the count was **38 to 3**. Both have McNemar p < 0.001. Overall accuracy was a statistical tie.
- **Against Cohere with a tuned cut-off:** JEV had fewer hallucinations (3 vs 7, not significant) and fewer confidently wrong answers (13 vs 24; p = 0.05, 95% CI of the difference −7.0 to −0.3 points). That result is borderline, not conclusive.
- **Where the errors come from:** when the context contained a useful passage, the generator answered correctly **97–99% of the time under every policy**. The errors come from what reaches the LLM, not from the LLM.

### Cost and latency

![Total cost per 1,000 questions: top3 $0.341, cohere_top3 $2.841, cohere_cut $2.650, jev_cut $0.389](https://raw.githubusercontent.com/marciosferreira/jev-vs-rerankers/main/figures/fig3_cost.png)
*Cost per 1,000 questions: LLM generation plus passage scoring.*

Cohere at list price, about $2.50 per 1,000 searches (one search is one query with up to 100 documents).

- **Against Cohere top-3:** ~7× cheaper overall.
- **Against plain top-3:** the JEV gate adds **14%** to the cost and cuts hallucinations **8×** (24 → 3).
- **Tokens:** JEV sends the LLM **200 input tokens per question on average, vs 605 for top-3**. That average includes the 27% of questions where the LLM isn't called; when it is called, the average is 276. The more expensive your generator, the more this saves.

**Latency.** Scoring 10 passages took **0.70 s** at the median with JEV (1.62 s at the 95th percentile), against **0.54 s** with Cohere (1.75 s). They're roughly equivalent, and the LLM call (~2 s) dominates end-to-end time.

The JEV figures come from **11 calls per question**, sent in parallel: 10 relevance calls, one per passage, plus one question about the whole set that the jev_cut policy doesn't use. The policy only needs the 10, so its real cost is ~9% lower than shown. Cohere scores all passages in one call. So JEV makes ~10× more requests, which matters for rate limits.

## Why it works: ordering vs deciding

We also scored all 10 candidates of every question with six methods and compared the scores with the ground truth, with no LLM involved.

![Top-1 passage answers: Cohere Rerank 4 Pro and bge-reranker-v2-m3 97%, JEV 95%. Detects no useful passage (AUC): JEV 0.95, Cohere 0.85. Withholds useless context at 85% of questions served: JEV 87%, Cohere 61%, bge-reranker-v2-m3 52%, embedding 31%](https://raw.githubusercontent.com/marciosferreira/jev-vs-rerankers/main/figures/fig4_ordering_vs_deciding.png)
*Retrieval-level comparison on all 300 test questions, no LLM involved.*

Two different sets of "no answer" questions appear in this article:

- **54 unanswerable questions**, where the answer isn't in the knowledge base at all;
- **61 questions with no useful passage among the 10 candidates**: the 54 above, plus 7 where the answer is in the base but retrieval didn't bring it.

This retrieval-level evaluation doesn't use the judge, so it covers all 300 test questions: 239 with a useful candidate, which the "ordering" row uses, and 61 without one, which the "deciding" rows use.

- **Ordering:** the state-of-the-art rerankers are 2 points ahead of JEV. That gap isn't statistically significant.
- **Deciding:** JEV is clearly ahead. The fairest comparison sets each method's cut-off so that the same share of questions (85%) still gets a useful passage. At that point JEV withholds useless context in **87%** of the 61 cases, Cohere in **61%**, bge-reranker-v2-m3 in 52% and the embedding in 31%. The gap holds at 90% and 80% as well. With the cut-offs tuned on the separate question set, the numbers are 87% for JEV and 59% for Cohere (+28 points, 95% CI +13 to +43).

The likely reason is how each model is trained. A reranker learns *"A is better than B for this query"*, so its scores aren't meant to be compared across queries. JEV answers a yes/no question with a probability, which is what a fixed "send it or not" rule needs.

## How to use it

```python
import requests
from concurrent.futures import ThreadPoolExecutor

JEV_URL = "https://openrouter.ai/api/alpha/decisions"   # JEV isn't served on /chat/completions
INSTRUCTION = "The passage contains information that helps answer the question"

def relevance(question: str, passage: str) -> float:
    r = requests.post(
        JEV_URL,
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
        json={
            "model": "typesafe/jev-1.13",
            "state": f"Question: {question}\n\nPassage: {passage}",
            "questions": {"rel": {"type": "noul", "instructions": INSTRUCTION}},
        },
        timeout=60,
    )
    r.raise_for_status()   # surfaces 429s instead of a confusing KeyError
    return r.json()["answers"]["rel"]["noul"]   # probability of "yes", 0 to 1

def answer(question: str, candidates: list[str], cutoff: float = 0.97) -> str:
    if not candidates:
        return "I couldn't find this in the documents I searched."
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:   # parallel: ~0.7 s for 10
        probs = list(pool.map(lambda p: relevance(question, p), candidates))
    context = [p for p, prob in zip(candidates, probs) if prob >= cutoff]
    if not context:
        return "I couldn't find this in the documents I searched."   # no LLM call
    return call_your_llm(question, context)
```

Practical notes:

- **Return a fixed message** instead of asking the LLM to say it doesn't know. It's deterministic, and it's free.
- **Say "I couldn't find it in the documents I searched", not "it's not in the documents".** In 7 of the 61 cases with no useful candidate, the answer *was* in the base but retrieval missed it. A second search with a rephrased query before giving up would recover some of these.
- **Tune the cut-off on your own data**, to your own risk tolerance. A lower cut-off means fewer false "I don't know"s and more risk of a wrong answer.
- **Watch your rate limits.** Scoring 10 passages is 10 requests. Retry with backoff on 429s.
- **If you also want the best ordering**, let a reranker order and JEV decide, and pay for both.

## Caveats

- **One dataset**: SQuAD 2.0, English Wikipedia, short (~200-token) paragraphs. Other domains, languages and chunk sizes may behave differently.
- **One run.** The generator, the judge and JEV are all non-deterministic, so repeated runs would give somewhat different numbers. We didn't measure that spread.
- **54 unanswerable questions.** The improvement over top-3 (with or without Cohere) is highly significant. The comparison with a tuned Cohere cut-off is borderline.
- **The judge is an LLM**, and its labels were not reviewed by humans.
- **Fewer wrong answers cost some recall:** 15% "I don't know" on answerable questions, vs 2% for Cohere top-3.
- **Cohere pricing** is the list price reported by third-party sources. Cohere's own pricing page lists dedicated-instance prices.
- **Not every reranker was tested.** Qwen3-Reranker, Voyage and Jina were left out.

## Bottom line

If your application can live with an occasional "I don't know" but not with a confident wrong answer (legal, health, support, compliance), don't let a reranker's top-*k* decide what your LLM reads. Rerankers are excellent at ordering. A decision model is better at the question that actually prevents wrong answers: *is there anything here worth sending at all?*

## How this was made

This study was built together with an AI assistant, and I want to be specific about who did what.

- **The question and the thesis are mine.** I started from a practical doubt: can a yes/no decision model tell apart passages that answer a question from passages that only look similar, better than a reranker? The claims in this article are the ones I decided to make.
- **The code was written by Claude Opus 5.5** (Anthropic), running in Claude Code, from my instructions: the RAG pipeline, the experiments, the statistics and the figures.
- **The method was a dialogue.** Several methodological choices were proposed by the AI and then accepted, changed or rejected by me. These included the similarity-threshold and top-*k* controls, testing against state-of-the-art rerankers, tuning cut-offs on a separate question set, and measuring real costs. Other choices came from me, such as putting the gate in front of a top-1 RAG, the focus on hallucination and cost, and which rerankers to test and in what order.
- **The text was drafted with Claude Opus 5.5** and reviewed and edited by me.
- **Every number comes from the published code and results**, not from the AI's text. Anyone can rerun the experiments from the repository.
- **A note on vendors:** the code was written, and the answers graded, by Anthropic models (Claude Opus 5.5 and `claude-sonnet-5`). No Anthropic product was among the systems compared.

---

*Code, data preparation and full reports: [github.com/marciosferreira/jev-vs-rerankers](https://github.com/marciosferreira/jev-vs-rerankers). Total API spend for all experiments: about $7 on OpenRouter, plus Cohere's free trial.*
