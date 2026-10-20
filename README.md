# Research Assistant with Citations (RAG)

A question-answering assistant over a small set of arXiv papers. It answers **only from the papers**, cites the **paper and page** for every claim, and says "Not found in the provided papers." when the answer isn't there.

Built as a learning project and designed to run at **$0**: embeddings and reranking run locally, and only answer generation uses the Gemini free tier.

## How it works

```
arXiv PDFs -> page text -> 300-word chunks (50 overlap) -> local embeddings (bge-small)
          -> Chroma vector store
question -> retrieve 30 chunks -> cross-encoder rerank -> relevance gate -> top 5 -> Gemini -> answer with [n] citations
```

| Part | Choice |
|---|---|
| Papers | 9 RAG-related arXiv papers (IDs below), 443 chunks |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, CPU) |
| Vector store | Chroma |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) |
| Relevance gate | Refuses without calling the LLM if the best reranker score is below -4.0 |
| Generator | `gemini-3.5-flash-lite` (free tier), answers cached on disk |
| Environment | Kaggle notebook (tested on Python 3.13) |

**Papers:** 2005.11401 (RAG), 2004.04906 (DPR), 2212.10496 (HyDE), 2307.03172 (Lost in the Middle), 2309.15217 (RAGAS), 2310.11511 (Self-RAG), 2312.10997 (RAG survey), 2401.15884 (Corrective RAG), 2210.03629 (ReAct). The notebook downloads them; they are not stored in this repo.

## Evaluation

I wrote 18 test questions: 13 answerable (11 from one paper, 2 needing several) and 5 that should be refused. Every change below was measured on the same 18 questions.

**Final configuration:**

| Metric | Result |
|---|---|
| Right paper cited (answerable) | 13 / 13 |
| Wrongly refused (answerable) | 0 / 13 |
| Correctly refused (off-topic) | 5 / 5 |
| Answers graded by hand against reference hints | **11 correct, 2 partial, 0 wrong (of 13)** |

The first three are automatic checks. They don't prove an answer is *correct*, which is why I also graded the 13 answers by hand. The two partial answers are listed under limitations. Per-question answers are in `results/results_final.json`.

Scores can move by a question between runs, because Gemini may word an answer differently.

### What I tried and what I learned

| Setup | Wrongly refused | Correct refusals | Cited hit | What happened |
|---|---|---|---|---|
| Baseline: top 8 by embedding similarity | 1 | 5 | 12 | "Two components of RAG" was refused: six RAG Survey chunks outranked the original paper |
| + Cross-encoder rerank (top 5 of 30) | 0 | 4 | 12 | Fixed that question, but "capital of France" got a vague answer |
| + Stricter refusal prompt | 0 | 4 | 12 | No change: the small model ignored it |
| + Relevance gate | 0 | 5 | 12 | Off-topic questions refused deterministically, without an API call |
| + Prompt requiring `[n]` on every sentence (**final**) | 0 | 5 | 13 | One answer had no citation until the prompt demanded it |
| Tried: prompt asking for fuller answers | 4 | 5 | 9 | Fixed one answer, broke four others. **Reverted** |

Takeaways:

- Retrieval was the weak link. The right paper was always among the results, but the right *passage* was not always near the top. Reranking fixed that.
- A numeric check (the gate) was more reliable than asking a small model to refuse politely.
- A prompt change that fixes one question can break others. The fixed test set caught it.

## Known limitations

- The gate threshold (-4.0) was chosen from the same 18 questions it is scored on, so the 5/5 refusal result is partly by construction.
- Questions on adjacent topics (for example LoRA, attention) score above the gate, so only the prompt stops them. Both refused in testing, but that is two examples.
- 13 hand-graded answers is a small sample: one question moves the score by about 8 points.
- The ReAct answer is correct and cited but does not say reasoning and acting are *interleaved*. I left it as a known partial after the fix attempt broke other questions.
- The HyDE answer is correct but incomplete: it says a hypothetical document is generated and embedded, but not that the embedding is used to retrieve real documents.
- The retrieval metric checks that the right paper was retrieved, not that the right passage was.

## Run it

1. Open `research_assistant_rag.ipynb` in a Kaggle notebook (File -> Import Notebook).
2. Settings -> **Internet On**.
3. Get a free Gemini API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and add it under Add-ons -> **Secrets**. The label must match `SECRET_NAME` in the config cell (default: `Gemini API Key`).
4. Run All. The first run takes about 5-10 minutes.

To run elsewhere, `pip install -r requirements.txt` and replace the `kaggle_secrets` lines with your own way of loading the key (for example an environment variable). Don't commit your key.

Free-tier models and limits change often. If `GEN_MODEL` stops working, list the models your key can use (see the commented loop in the config cell) and switch.

## What I'd do next

- Hybrid search (BM25 plus embeddings) and chunk-size experiments.
- Replace hand grading with an LLM judge, then measure how often it agrees with human grades.
- A larger test set, and re-tuning the gate threshold on questions it was not tuned on.

## Repo contents

- `research_assistant_rag.ipynb`: the full pipeline and evaluation
- `results/results_final.json`: per-question answers and automatic check results from the final run
- `requirements.txt`
- `.gitignore`
