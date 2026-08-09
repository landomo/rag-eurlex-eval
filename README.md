# rag-eurlex-eval

A retrieval-augmented generation system over EU regulatory text (the **AI Act** and the
**GDPR**), built as a controlled experiment rather than a demo. Nine pipeline
configurations are evaluated on a shared gold set with [Ragas](https://docs.ragas.io),
so every design choice — chunking strategy, dense vs. hybrid retrieval, cross-encoder
reranking — is backed by a number instead of an opinion.

> **Status:** the harness is complete and tested. The results table below is empty until
> you run the grid yourself — see [Reproducing the results](#reproducing-the-results).
> Nothing in this repo reports a number that wasn't measured.

---

## Why this corpus

EU regulations are a deliberately unfriendly retrieval target:

- **Long-range dependencies.** Article 6 of the AI Act defines high-risk systems by
  reference to Annex III; Article 22 of the GDPR only makes sense alongside Article 4's
  definitions. Naive chunking severs these links.
- **Legalese defeats pure semantics.** "Legitimate interest", "DPIA", "Article 22" are
  exact tokens. Dense retrieval alone routinely misses them, which is the case for
  hybrid search.
- **Hallucination is expensive.** A confidently wrong answer about a compliance
  obligation is worse than no answer. This is why the pipeline is built to abstain, and
  why faithfulness is the headline metric.
- **The structure is machine-readable.** Articles, recitals, and annexes give a natural
  chunk boundary to test structure-aware chunking against.

---

## Architecture

```
EUR-Lex (CELEX 32024R1689, 32016R0679)
  │
  ├─ ingest.py ......... HTML → normalised text → segmented into
  │                      Recitals / Article N / Annex N (data/processed/sections.jsonl)
  │
  ├─ chunking.py ....... 4 strategies, each producing its own document set
  │      fixed_512            token windows, structure-blind          (baseline)
  │      recursive_1000       char-recursive w/ legal separators
  │      semantic             embedding-breakpoint chunking
  │      structural_article   one chunk per Article, heading-prefixed
  │
  ├─ index.py .......... text-embedding-3-small → Chroma (cosine),
  │                      one persistent collection per strategy
  │
  ├─ retrieval.py ...... dense      Chroma top-k
  │                      hybrid     BM25 + dense, reciprocal-rank fusion (0.4 / 0.6)
  │                      + rerank   fetch 20 → FlashRank cross-encoder → top 5
  │
  ├─ pipeline.py ....... strict-grounding prompt, gpt-4o-mini, forced abstention
  │
  ├─ testset.py ........ 28 hand-written questions + ~45 LLM-generated,
  │                      each with a reference answer and reference_contexts
  │
  └─ evaluate.py ....... Ragas: faithfulness, answer relevancy, context precision,
                         context recall, factual correctness, noise sensitivity
```

### Stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` | 1536-dim, cheap enough to re-embed the corpus four times |
| Vector store | Chroma, persistent, cosine | No server to run; one collection per chunking strategy keeps runs isolated |
| Sparse retrieval | BM25 (`rank_bm25`) | Legal queries carry exact statutory tokens |
| Fusion | LangChain `EnsembleRetriever`, weights 0.4 BM25 / 0.6 dense | Reciprocal-rank fusion; lexical weight is deliberately non-trivial |
| Reranker | FlashRank `ms-marco-MiniLM-L-12-v2` | CPU cross-encoder, no extra API key, ~50 MB |
| Generator | `gpt-4o-mini`, temperature 0 | Cost; the experiment is about retrieval, so the generator is held constant |
| Judge | `gpt-4o-mini` via Ragas | Same model across all runs, so judge bias is a constant, not a confound |

---

## Experiment design

**Stage 1** — full factorial: 4 chunking strategies × 2 retrieval modes = 8 runs.
**Stage 2** — cross-encoder reranking applied to the Stage 1 winner = 1 run.

Everything except the variable under test is frozen: same gold set, same generator,
same judge, same `top_k=5`, temperature 0 throughout. Reranking over-retrieves
`fetch_k=20` and compresses back to 5, so the generator's context budget is unchanged
and any delta is attributable to *ordering*, not to seeing more text.

### The gold set

Two halves, on purpose:

- **28 hand-written questions** (`config/seed_questions.yaml`), tagged by category:
  single-article lookup, multi-hop, cross-regulation, lexical, and **negative**
  (questions the corpus genuinely cannot answer — these test abstention, and a system
  that scores well on recall while confidently answering them is broken).
- **~45 LLM-generated questions** grounded in randomly sampled Articles, for breadth.

Reference answers are drafted from the *source Articles*, never through the retrieval
system under test — otherwise the gold set is circular and every metric is inflated.
Seed items are written with `needs_review: true`; a gold set nobody read is not a gold set.

### Metrics, and what each one is actually for

| Metric | Question it answers | What moves it |
|---|---|---|
| **Faithfulness** | Are the answer's claims entailed by the retrieved context? | Prompt strictness, context quality. The hallucination metric. |
| **Answer relevancy** | Does the answer address the question asked? | Generation, mostly constant here |
| **Context precision** | Of the chunks retrieved, how many were useful? | **Reranking.** Punishes over-retrieval. |
| **Context recall** | Did retrieval surface everything the reference needed? | **Chunking.** Punishes severed cross-references. |
| **Factual correctness** | End-to-end answer quality vs. reference | Everything |
| **Noise sensitivity** | How often irrelevant context leaks wrong claims in? | Precision/recall trade-off. *Lower is better.* |

Precision and recall are the retrieval-side pair; faithfulness and relevancy are the
generation-side pair. Reporting only one side is how RAG benchmarks mislead — a system
that retrieves 20 chunks will look faithful and score terribly on precision.

---

## Results

<!-- Paste results/RESULTS.md here after running the grid. -->

_Not yet run._ Execute `make all` and paste the generated `results/RESULTS.md` table
here. `results/summary.csv` holds the same data machine-readably, and
`results/runs/*.json` keeps every per-question score for error analysis.

| chunker | retrieval | rerank | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Factual Correctness | Noise Sensitivity |
|---|---|---|---|---|---|---|---|---|
| _fill from results/RESULTS.md_ | | | | | | | | |

**Questions worth answering in this section once you have numbers:**

1. Which chunker won on **context recall**, and by how much over the `fixed_512` baseline?
2. Did hybrid beat dense, and was the gain concentrated in the lexical-category questions?
3. Did reranking move **context precision** without costing recall?
4. On the negative questions, how often did the system correctly abstain?

Point 4 is the one interviewers remember, and `results/runs/*.json` contains the
per-question responses needed to compute it.

---

## Reproducing the results

```bash
git clone <this repo> && cd rag-eurlex-eval
make install                      # venv + pinned deps
cp .env.example .env              # add your OPENAI_API_KEY

make ingest                       # download + segment the AI Act and GDPR
make index                        # chunk 4 ways, embed, build Chroma collections
make testset                      # build the gold set (LLM-assisted, ~$0.20)
make run                          # the 9-run grid
make report                       # → results/RESULTS.md
```

Or all at once: `make all`.

**Cost and time.** The dominant cost is the Ragas judge: six LLM metrics × ~73 questions
× 9 runs. With `gpt-4o-mini` expect roughly **$3–6** and **60–90 minutes** end to end.
Sanity-check the wiring first with `python scripts/04_run_experiments.py --limit 5`.

Runs are cached by `run_id` — re-running skips completed configurations. Use `--force`
to override.

**Offline tests** (no API key, no network):

```bash
make test
```

These cover segmentation, all four chunkers, dense/hybrid/reranked retrieval wiring, and
prompt assembly against a synthetic regulation fixture. Two tests self-skip when
`tiktoken` and the FlashRank weights haven't been downloaded yet.

---

## Design notes

**The ablation is not rigged.** `fixed_512`, `recursive_1000`, and `semantic` operate on
the flat regulation text and never receive section metadata — only
`structural_article` sees the legal hierarchy. There is a test asserting this
(`test_flat_chunkers_do_not_leak_section_ids`), because it would be easy to accidentally
hand every strategy the structure and then "discover" that structure helps.

**Abstention is a feature.** The system prompt requires a fixed refusal string when the
context is insufficient. This depresses answer relevancy on the negative questions and
that is correct behaviour — a compliance assistant that guesses is a liability.

**Semantic chunks are capped.** `SemanticChunker` occasionally emits multi-thousand-token
chunks, which would hand that strategy a larger context budget than the others. Chunks
are capped at 3,000 characters so context size stays comparable across strategies.

**Structural chunks are self-describing.** Each carries its heading (`GDPR - Article 17:
Right to erasure`) as a prefix, so a fragment retrieved in isolation still identifies
itself — and the generator has something concrete to cite.

**Dependency pinning is deliberate.** Ragas 0.4.x imports
`langchain_community.chat_models.vertexai`, which was removed in langchain-community
0.4. Ragas `0.3.9` on the langchain `0.3` line is the last combination that installs and
imports cleanly; `requirements.txt` pins it with a comment explaining why.

---

## Known limitations

- **Single judge model.** All metrics come from `gpt-4o-mini`. LLM-as-judge scores are
  noisy and correlated with the judge's own biases. Re-running with a second judge
  (`gpt-4o`, or Claude) and reporting agreement would materially strengthen the claims.
- **No confidence intervals.** Each configuration is evaluated once. Bootstrapping over
  the per-question scores in `results/runs/*.json` would give error bars and is a
  worthwhile next step — several of the deltas may not survive them.
- **Reference answers are LLM-drafted.** They are grounded in source Articles rather than
  the retrieval system, but they are not lawyer-reviewed. Treat absolute scores as
  relative signals, not as legal ground truth.
- **English only, single snapshot.** Regulations are consolidated documents that change;
  `data/raw/` caches whatever EUR-Lex served on ingestion day.
- **No latency benchmarking.** Wall-clock time per run is recorded, but retrieval latency
  is not isolated from generation.

---

## Repository layout

```
config/seed_questions.yaml   hand-written evaluation questions, by category
src/ragbench/
  config.py                  paths, models, experiment grid
  ingest.py                  EUR-Lex download + legal segmentation
  chunking.py                the four chunking strategies
  index.py                   Chroma index construction
  retrieval.py               dense / hybrid / reranked retrievers
  pipeline.py                the RAG chain and its grounding prompt
  testset.py                 gold set construction
  evaluate.py                Ragas harness
  report.py                  ablation table generation
scripts/01..05               the runnable stages
tests/                       offline tests, synthetic regulation fixture
results/                     per-run JSON, summary.csv, RESULTS.md
```

## Licence

MIT. The EU AI Act and GDPR texts are © European Union,
[https://eur-lex.europa.eu](https://eur-lex.europa.eu), reused under the
[EUR-Lex reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
They are not redistributed in this repository; `scripts/01_ingest.py` fetches them.
