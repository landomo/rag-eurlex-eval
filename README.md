# rag-eurlex-eval

A retrieval-augmented generation system over EU regulatory text (the **AI Act** and the
**GDPR**), built as a controlled experiment rather than a demo and evaluated with
[Ragas](https://docs.ragas.io).

Two hypotheses were tested under controlled conditions. **Both came back negative**, and
one earlier result had to be thrown out after a defect was found in the retrieval
configuration that had produced it. The interesting content of this repo is therefore the
measurement discipline — what was validated before it was trusted, what was discarded, and
what it all cost — rather than a table of wins.

> **Status:** evaluated. Two controlled comparisons were run on a ~$13 budget; the full
> 4×2 grid was not affordable and is not claimed. Every number below was measured.
> Full analysis, caveats and cost breakdown: **[docs/EVALUATION.md](docs/EVALUATION.md)**.

---

## Why this corpus

EU regulations are a deliberately unfriendly retrieval target:

- **Long-range dependencies.** Article 6 of the AI Act defines high-risk systems by
  reference to Annex III; Article 22 of the GDPR only makes sense alongside Article 4's
  definitions. Naive chunking severs these links.
- **Legalese may defeat pure semantics.** "Legitimate interest", "DPIA", "Article 22" are
  exact tokens that dense embeddings can blur. That was the case for hybrid search, and
  it is the hypothesis this project tested — [it did not hold](#b--retrieval-mode-structural-chunking-27-hand-written-questions).
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
  ├─ index.py .......... embeddings → Chroma (cosine),
  │                      one persistent collection per strategy
  │
  ├─ retrieval.py ...... dense      Chroma top-k
  │                      hybrid     BM25 + dense, RRF (0.5 / 0.5), truncated to k
  │                      + rerank   fetch 20 → FlashRank cross-encoder → top 5
  │
  ├─ pipeline.py ....... strict-grounding prompt, Claude, forced abstention
  │
  ├─ testset.py ........ 27 hand-written questions + 43 LLM-generated, each with a
  │                      reference answer, reference_contexts and a category
  │
  └─ evaluate.py ....... Ragas: faithfulness, context precision, context recall,
                         factual correctness  (answer relevancy excluded - NaN with Claude)
```

### Stack

| Layer | Choice | Why |
|---|---|---|
| Generator | Claude Haiku 4.5, temperature 0 | The experiment is about retrieval, so the generator is held constant across all runs |
| Judge | Claude Haiku 4.5 via Ragas | Same model in every run, so judge bias is a constant rather than a confound |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed, local | Anthropic serves no embeddings API. ONNX Runtime, no PyTorch, no second API key — and it reuses the runtime FlashRank already needs |
| Vector store | Chroma, persistent, cosine | No server to run; one collection per chunking strategy keeps runs isolated |
| Sparse retrieval | BM25 (`rank_bm25`) | Legal queries carry exact statutory tokens |
| Fusion | LangChain `EnsembleRetriever`, weights **0.5 / 0.5**, output truncated to k | Equal weights are forced by the RRF arithmetic: at 0.4/0.6 the lexical arm can never enter the top-k and hybrid silently becomes dense ([why](docs/EVALUATION.md#a-second-defect-found-by-arithmetic-rather-than-by-running)) |
| Reranker | FlashRank `ms-marco-MiniLM-L-12-v2` | CPU cross-encoder, no extra API key, ~50 MB |

Every one of these is swappable from `.env` without touching code — `RAGBENCH_LLM_PROVIDER`
accepts `anthropic` or `openai`, `RAGBENCH_EMBED_PROVIDER` accepts `local`, `openai` or
`voyage`. All vendor-specific code lives in `src/ragbench/providers.py`; nothing else in
the codebase imports an SDK directly, and there are tests asserting that.

---

## Experiment design

**Designed:** full factorial, 4 chunking strategies × 2 retrieval modes = 8 runs, plus
cross-encoder reranking on the winner.

**Actually run:** two controlled comparisons (chunk size; retrieval mode). The full grid
cost ~$13 to evaluate and the budget did not stretch. Each comparison varies one factor
with everything else frozen, so each is internally valid — but they use different gold
sets and do not compose into a grid. See [docs/EVALUATION.md](docs/EVALUATION.md) for
what that does and does not license.

Everything except the variable under test is frozen: same gold set, same generator, same
judge, same `top_k=5`, temperature 0 throughout. Both retrieval arms are truncated to
exactly k — without that, `EnsembleRetriever` returns the union of both retrievers and
hybrid silently receives more context than dense, which is what invalidated the first
hybrid result.

One confound survives and is reported rather than hidden: holding **k** constant is not
the same as holding **context budget** constant, so comparing chunkers at fixed k partly
measures chunk size. Comparison A is affected by this.

### The gold set

Two halves, on purpose:

- **27 hand-written questions** (`config/seed_questions.yaml`), tagged by category:
  single-article lookup, multi-hop, cross-regulation, lexical, and **negative**
  (questions the corpus genuinely cannot answer — these test abstention, and a system
  that scores well on recall while confidently answering them is broken).
- **43 LLM-generated questions** grounded in randomly sampled Articles, for breadth.
  Comparison B uses the hand-written 27 only: the generated majority are single-Article
  lookups that every configuration handles, so they dilute signal and consume budget.

Reference answers are drafted from the *source Articles*, never through the retrieval
system under test — otherwise the gold set is circular and every metric is inflated.
Seed items are written with `needs_review: true`; a gold set nobody read is not a gold set.

### Metrics, and what each one is actually for

| Metric | Question it answers | What moves it |
|---|---|---|
| **Faithfulness** | Are the answer's claims entailed by the retrieved context? | Prompt strictness, context quality. The hallucination metric. |
| ~~Answer relevancy~~ | Does the answer address the question asked? | **Not used** — returns NaN on 100% of samples with a Claude judge |
| **Context precision** | Of the chunks retrieved, how many were useful? | **Reranking.** Punishes over-retrieval. |
| **Context recall** | Did retrieval surface everything the reference needed? | **Chunking.** Punishes severed cross-references. |
| **Factual correctness** | End-to-end answer quality vs. reference | Everything |
| ~~Noise sensitivity~~ | How often irrelevant context leaks wrong claims in? | **Not used** — 20/25 samples only, and outside the `core` set |

Precision and recall are the retrieval-side pair; faithfulness and factual correctness are
the generation-side pair. Reporting only one side is how RAG benchmarks mislead — a system
that retrieves 20 chunks will look faithful and score terribly on precision. Which is
exactly the trap the discarded hybrid run fell into.

---

## Results

Two controlled comparisons, each varying one factor. **They use different gold sets and
are not comparable to each other.** Full analysis in [docs/EVALUATION.md](docs/EVALUATION.md).

### A — chunk size, dense retrieval, 70 questions

| chunker | mean chunk | faithfulness | ctx precision | ctx recall | factual correctness |
|---|---|---|---|---|---|
| `fixed_512` | 2,735 chars | 0.915 | **0.552** | **0.712** | 0.434 |
| `recursive_1000` | 676 chars | **0.965** | 0.355 | 0.528 | 0.426 |

At fixed k, **chunk size dominates retrieval scores**. `fixed_512` wins recall by +0.184
not because its boundaries are better — it is structure-blind token windowing — but
because five large chunks deliver 4× more text than five small ones. Anyone comparing
chunkers at fixed k is partly measuring chunk size. Faithfulness moves the other way:
less context, fewer chances to make unsupported claims.

### B — retrieval mode, structural chunking, 27 hand-written questions

| retrieval | faithfulness | ctx precision | ctx recall |
|---|---|---|---|
| dense | 0.942 | **0.519** | **0.703** |
| hybrid (BM25 + dense) | **0.960** | 0.485 | 0.665 |

**Hybrid did not beat dense** given an equal context budget, and lost hardest on the
`lexical` questions built for it to win (recall 0.333 vs 0.544, n=3). This contradicts the
hypothesis the project was built to test.

An earlier run showed hybrid winning by +0.118 recall. It was discarded:
`EnsembleRetriever` returns the *union* of both retrievers, so hybrid was fed 8.3 chunks
against dense's 5 — scoring higher for retrieving more. A second defect was found by
arithmetic: with RRF weights 0.4/0.6 and `c=60`, BM25's best result (0.4/61) scores below
dense's fifth (0.6/65), so after truncation hybrid returns *exactly* the dense results and
the experiment compares a configuration against itself. Both are fixed and asserted in tests.

### Also measured

- Abstention: **2/2** on deliberately unanswerable questions, both configurations.
- `cross_reg` context precision is **0.25–0.27** everywhere — cross-regulation questions
  are the pipeline's clear weakness, and nothing tested fixes it.
- `answer_relevancy` returns NaN on **100%** of samples with a Claude judge and is
  excluded from every metric set — see [docs/METRIC_SUPPORT.md](docs/METRIC_SUPPORT.md).

### Not measured

`semantic` chunking, cross-encoder reranking, and `structural_article` against the other
chunkers. The budget ran out; these are absent rather than assumed.

### Cost

| | cost |
|---|---|
| Serving one answer | **$0.0043** |
| Evaluating that same answer (`core`, 8 judge calls) | **$0.0224** |

**Evaluation costs 5.3× the answer it grades**, multiplied by every configuration.
`context_precision` alone spends one judge call per retrieved chunk. Serving 1,000 real
users would cost less than measuring six configurations once. Breakdown, levers and the
23% error in my own estimate: [docs/EVALUATION.md](docs/EVALUATION.md#5-cost-analysis).

## Reproducing the results

```bash
git clone https://github.com/landomo/rag-eurlex-eval && cd rag-eurlex-eval
make install                      # venv + pinned deps
cp .env.example .env              # add ANTHROPIC_API_KEY — that is the only key needed

make ingest                       # download + segment the AI Act and GDPR
make index                        # chunk 4 ways, embed locally, build Chroma collections
make testset                      # build the gold set (LLM-assisted)

.venv/bin/python scripts/06_check_testset.py             # free: validate the gold set
.venv/bin/python scripts/00_diagnose_metrics.py          # ~$0.02: validate the metrics
.venv/bin/python scripts/07_estimate_cost.py --budget 5  # price the plan in dollars

.venv/bin/python scripts/04_run_experiments.py --metrics core --origin seed
make report                       # → results/RESULTS.md
```

Or all at once: `make all`.

**Time and cost.** Embeddings and reranking run locally and are free; the spend is
entirely Claude calls, and it is dominated by evaluation rather than generation — see the
[cost analysis](docs/EVALUATION.md#5-cost-analysis). Price any plan in dollars first:

```bash
.venv/bin/python scripts/07_estimate_cost.py --budget 5
```

Measured reference points: **$0.0043** to serve one answer, **$0.0224** to evaluate it,
**$1.94** for two configurations over 27 questions. A full 4×2 grid on 70 questions costs
roughly **$13**.

Validate before you spend. `scripts/06_check_testset.py` checks the gold set for free and
`scripts/00_diagnose_metrics.py` verifies every metric returns real numbers for about
$0.02 — that one would have saved a third of this project's budget had it existed sooner.

Runs are cached by configuration **and question count**, so stopping mid-grid loses
nothing and resuming repays nothing. Use `--force` to override.

The first `make index` downloads ~130 MB of ONNX embedding weights; the first reranked run
downloads the FlashRank cross-encoder. Both then run offline.

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

**One key, by design.** The obvious way to build this is OpenAI end to end. Anthropic
serves no embeddings API, so using Claude means solving embeddings separately — the easy
answer is a second vendor and a second key, the better answer is running them locally.
fastembed does that in ONNX with no PyTorch, on the runtime FlashRank already pulls in.

**Dependency pinning is deliberate.** Ragas 0.4.x imports
`langchain_community.chat_models.vertexai`, which was removed in langchain-community
0.4. Ragas `0.3.9` on the langchain `0.3` line is the last combination that installs and
imports cleanly. `instructor` and `eval_type_backport` are pinned too: ragas pulls
`instructor` in unpinned, and on Python 3.9 — which is what macOS ships — it cannot parse
its own `str | Path` annotations without the backport. Every pin was resolved against
Python 3.9 including transitive dependencies.

---

## Known limitations

- **Partial grid.** Two controlled comparisons, not a 4×2 ablation. `semantic` chunking,
  reranking, and `structural_article` versus the other chunkers were never run.
- **Single run per configuration, no confidence intervals.** The recall gaps in
  Comparison A are large relative to plausible judge noise; the faithfulness gaps are not,
  and are flagged as suggestive only in the report.
- **Small per-category counts.** The `lexical` finding rests on 3 questions and the
  abstention check on 2. Directional, not conclusive.
- **Comparison A is confounded by chunk size.** Holding k constant rather than context
  budget constant means chunker comparisons partly measure chunk length. Diagnosed, not
  fixed — fixing it requires a re-run.
- **Single judge model.** All scores come from one Claude model, and LLM-judge scores
  carry that model's biases. `RAGBENCH_JUDGE_MODEL` and `RAGBENCH_LLM_PROVIDER` make the
  cross-check a config change; it was not affordable here.
- **Reference answers are LLM-drafted** from source Articles — grounded, categorised, and
  spot-checked for correct Article citations, but not lawyer-reviewed.
- **`answer_relevancy` unavailable** with a Claude judge. Four metrics, not six.
- **English only, single snapshot** of two consolidated regulations.

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
  providers.py               LLM + embedding provider abstraction
  evaluate.py                Ragas harness
  report.py                  ablation table generation
scripts/00..07               diagnose, ingest, index, testset, run, report,
                            check gold set, estimate cost
tests/                       offline tests, synthetic regulation fixture
results/                     per-run JSON (per-question scores), summary.csv,
                            by_category.csv, RESULTS.md
docs/EVALUATION.md           the evaluation report: findings, caveats, costs
docs/METRIC_SUPPORT.md       which Ragas metrics survive a Claude judge
```

## Licence

MIT. The EU AI Act and GDPR texts are © European Union,
[https://eur-lex.europa.eu](https://eur-lex.europa.eu), reused under the
[EUR-Lex reuse policy](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html).
They are not redistributed in this repository; `scripts/01_ingest.py` fetches them.
