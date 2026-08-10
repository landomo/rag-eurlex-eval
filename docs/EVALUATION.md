# Evaluation report

RAG over the EU AI Act and GDPR, evaluated with Ragas. This document states what was
measured, what was not, what the numbers support, and what they cost.

The headline is that **two of the three hypotheses this project set out to test did not
survive contact with a controlled measurement**, and one measurement had to be discarded
entirely because of a defect in the retrieval configuration. Those are the useful parts.

---

## 1. Scope, stated up front

This is **not** the full 4×2 ablation the harness supports. The evaluation budget ran out
first. What exists is two controlled comparisons, each varying one factor:

| Comparison | Varies | Held constant | Gold set |
|---|---|---|---|
| **A — chunk size** | `fixed_512` vs `recursive_1000` | dense retrieval, k=5, judge, prompt | 70 questions |
| **B — retrieval mode** | dense vs hybrid | `structural_article` chunking, k=5, judge, prompt | 27 questions |

**Scores from A and B are not comparable to each other.** They use different gold sets, of
different sizes and different difficulty mixes. Each comparison is internally valid; the
two are not a grid.

Not measured: `semantic` chunking, cross-encoder reranking, `structural_article` against
the other chunkers, and any configuration on both gold sets.

---

## 2. Comparison A — chunk size drives recall at fixed k

70-question gold set (27 hand-written + 43 generated), dense retrieval, k=5.

| chunker | mean chunk | context supplied | faithfulness | ctx precision | ctx recall | factual correctness |
|---|---|---|---|---|---|---|
| `fixed_512` | 2,735 chars | ~13,700 chars | 0.915 | **0.552** | **0.712** | 0.434 |
| `recursive_1000` | 676 chars | ~3,400 chars | **0.965** | 0.355 | 0.528 | 0.426 |

**Finding: at fixed k, chunk size dominates retrieval scores.** `fixed_512` beats
`recursive_1000` by **+0.184 context recall** and **+0.197 context precision** — not
because its boundaries are smarter (they are strictly worse; it is structure-blind token
windowing) but because five 2,735-character chunks deliver four times more text than five
676-character ones.

This is a measurement artifact of holding k constant rather than holding *context budget*
constant, and it is easy to mistake for a chunking-quality result. Anyone comparing
chunkers at fixed k is partly measuring chunk size.

**Faithfulness moves the opposite way** (0.965 vs 0.915). Less retrieved context means
fewer opportunities to make unsupported claims. Smaller chunks trade recall for
groundedness — a real trade-off, not a free win in either direction.

### Caveats

- Single run per configuration, no confidence intervals. The recall gap (0.184) is large
  relative to plausible judge noise; the faithfulness gap (0.05) is not, and should be
  treated as suggestive only.
- 43 of the 70 questions are LLM-generated and skew toward single-Article lookups.

---

## 3. Comparison B — hybrid retrieval did not beat dense

27 hand-written questions, `structural_article` chunking, k=5 enforced on both arms,
RRF weights 0.5/0.5.

| retrieval | faithfulness | ctx precision | ctx recall |
|---|---|---|---|
| dense | 0.942 | **0.519** | **0.703** |
| hybrid (BM25 + dense) | **0.960** | 0.485 | 0.665 |

**Finding: given an equal context budget, hybrid retrieval was slightly worse than dense
on both retrieval metrics** (−0.034 precision, −0.038 recall).

This contradicts the project's stated hypothesis. The case for hybrid on statutory text is
that legal queries carry exact tokens — "Article 22", "DPIA", "pseudonymisation" — that
dense embeddings blur. That reasoning is sound and it is why the experiment was worth
running. It just is not what the measurement shows here.

The per-category data makes the failure sharper rather than softer:

| category | n | dense ctx recall | hybrid ctx recall |
|---|---|---|---|
| lookup | 12 | 0.789 | 0.752 |
| multi_hop | 6 | 0.667 | **0.698** |
| cross_reg | 4 | **0.722** | 0.564 |
| lexical | 3 | **0.544** | 0.333 |
| negative | 2 | — | — |

Hybrid loses hardest on `lexical` — precisely the category constructed for it to win.
With n=3 that is nowhere near conclusive, but it is the opposite of the predicted
direction, and "the effect I expected went the other way on the questions designed to
detect it" is worth more than a confirmed prior.

### Why the earlier, larger hybrid gain was not real

An earlier run showed hybrid beating dense by **+0.118 recall** on `fixed_512`. That run
is excluded from this report. `EnsembleRetriever` returns the *union* of both retrievers'
results rather than the top k: dense returned exactly 5 chunks, hybrid returned up to 10
and averaged 8.3. Hybrid was scoring higher because it fed the generator 66% more text.

That defect also made hybrid ~66% more expensive per question, so it was inflating both
the score and the bill.

### A second defect, found by arithmetic rather than by running

With the output correctly capped at k, the original RRF weights made the comparison
meaningless in a different way. `EnsembleRetriever` scores `weight / (c + rank)` with
`c = 60`. At weights 0.4 BM25 / 0.6 dense:

```
BM25  rank 1 = 0.4 / 61 = 0.00656
dense rank 5 = 0.6 / 65 = 0.00923
```

Every dense result outranks every BM25 result. After truncation to k=5, the hybrid arm
returns exactly the dense arm's results — the experiment would have compared a
configuration against itself and reported a null effect as a finding.

Weights are now 0.5/0.5, where BM25 rank 1 (0.00820) beats dense rank 5 (0.00769) and the
lists genuinely interleave. `tests/test_pipeline.py` asserts this property against the
arithmetic, not against the config value, so it cannot silently regress.

**Comparison B was run after both fixes. Comparison A is dense-only and unaffected.**

---

## 4. Abstention

Both `structural_article` configurations abstained on **2/2** deliberately unanswerable
questions ("corporate tax rate for AI companies in Ireland", "what the Digital Markets Act
says about gatekeeper interoperability"), returning the mandated refusal rather than
inventing an answer.

Two questions is a token check, not evidence of calibration. But abstention is a designed
behaviour here — the system prompt requires a fixed refusal string when context is
insufficient — and it is measurable rather than assumed. On a compliance corpus a
confident wrong answer is worse than no answer.

Note that `cross_reg` context precision sits at 0.25–0.27 across every configuration.
Questions spanning both regulations are the pipeline's clear weak point, and no
configuration tested fixes it.

---

## 5. Cost analysis

Measured against Claude Haiku 4.5 list pricing ($1/M input, $5/M output).

### Serving vs evaluating

| | tokens | cost |
|---|---|---|
| **Serve one answer** (1 generation call) | 2,764 in / 300 out | **$0.0043** |
| **Evaluate that same answer** (`core`, 8 judge calls) | 14,658 in / 1,850 out | **$0.0224** |

**Evaluation costs 5.3× more than the answer it grades**, and that multiplies by every
configuration in the grid. Serving 1,000 real user questions over this corpus would cost
$4.26 — less than measuring six configurations once.

Where the evaluation money goes:

| metric | calls per question | why |
|---|---|---|
| `context_precision` | 5 | one judge call **per retrieved chunk** |
| `faithfulness` | 2 | extract claims, then entail each against the full context |
| `context_recall` | 1 | compare reference against the full context |

Four of those eight calls carry the entire retrieved context, so context length is paid
for repeatedly. Chunking strategy therefore drives evaluation cost as well as quality:
`fixed_512` costs **$0.0342** per question per configuration against `structural_article`'s
**$0.0276** — 24% more, purely because its chunks are bigger.

### Estimate vs actual

The final run was predicted at $1.50 and cost **$1.94** — the model was **23% low**
(correction factor ×1.30). Estimates in `scripts/07_estimate_cost.py` now carry that
correction. Earlier phases were not measured directly; total project spend was
approximately $13.

### What would actually reduce this

| lever | saving | cost to validity |
|---|---|---|
| Fewer questions | linear | wider confidence intervals; the honest lever |
| Fewer configurations | linear | fewer conclusions |
| Non-LLM retrieval metrics | ~60% | **high** — measured r=0.42/0.35 against LLM-judged versions (see below) |
| Prompt caching | up to 90% on input | none, but Ragas does not use it |
| Batch API | 50% | none, but Ragas does not support it |
| Smaller k | linear in context | changes what is being measured |

Ragas' `NonLLMContextPrecisionWithReference` and `NonLLMContextRecall` score retrieved
chunks against `reference_contexts` by string similarity at **zero API cost**. Verified on
68 real samples: they run, but correlate only **r=0.42** (precision) and **r=0.35**
(recall) with their LLM-judged counterparts. They capture surface overlap, not semantic
coverage. Shipped as the `budget` metric set for screening many configurations cheaply —
**not** as a substitute, and their numbers should never be reported as if they were the
LLM metrics.

The single largest structural saving would be prompt caching: four of eight judge calls
re-send identical context. Ragas does not expose it.

---

## 6. Instrument validation

Before trusting any metric, the metrics themselves were checked against 25 real samples.

| metric | real values | status |
|---|---|---|
| `faithfulness` | 25/25 | usable |
| `llm_context_precision_with_reference` | 25/25 | usable |
| `context_recall` | 25/25 | usable |
| `factual_correctness` | 25/25 | usable |
| `noise_sensitivity` | 20/25 | partial |
| `answer_relevancy` | **0/25** | **broken with a Claude judge** |

`answer_relevancy` returned NaN on **every** sample — including non-abstaining ones — while
still billing 3 judge calls each. Ragas asks the judge for free-text JSON and sets
`np.nan` when parsing fails; Claude's output never matched the expected schema. It is
excluded from every metric set. Full detail in [`METRIC_SUPPORT.md`](METRIC_SUPPORT.md),
including why `llm_factory(provider="anthropic")` does not fix it.

A metric that silently returns NaN is worse than a missing one: it looks like data.

---

## 7. Recommendations

**For this pipeline**

1. **Compare chunkers at equal context budget, not equal k.** Fix total characters
   supplied to the generator and vary k per strategy. Comparison A is currently
   confounded by chunk size, and that confound is invisible in the scores.
2. **Fix `cross_reg` before anything else.** Context precision of 0.25 on
   cross-regulation questions is the largest single weakness, and no configuration tested
   changed it. Likely needs query decomposition — retrieve per-regulation, then merge —
   rather than better chunking.
3. **Do not adopt hybrid retrieval on this corpus** on current evidence. It cost more and
   scored slightly worse. Re-test with a larger `lexical` set (n=3 is far too small) before
   concluding either way.
4. **Test reranking next.** It is the untested lever most likely to move context
   precision, which is the weaker of the two retrieval metrics everywhere.

**For anyone running Ragas with a non-OpenAI judge**

1. Validate every metric on one question before running a grid. `scripts/00_diagnose_metrics.py`
   does this for ~$0.02 and would have saved roughly a third of this project's spend.
2. Never rank runs across different gold-set sizes. Cache keys must include the question
   count — they did not here, and a 5-question smoke run silently occupied the full run's
   cache slot, which would have put 5-question scores into the results table labelled as 70.
3. Read the retrieved contexts, not just the scores. Both hybrid defects were invisible in
   the aggregates and obvious in the raw output.
4. Price in dollars before running, not calls. "6,930 calls" reads like scale; it is a bill.

**On experimental design under a budget**

The full 4×2 grid on 70 questions would have cost ~$13. Two controlled comparisons on
27 hand-written questions cost ~$4 and support clearer claims, because the hand-written
questions are categorised and adversarial while the generated majority were single-Article
lookups every configuration handled. Fewer, better questions beat more, weaker ones — and
that should have been the design from the start rather than a retreat from it.

---

## 8. Reproducing

```bash
make install
cp .env.example .env          # ANTHROPIC_API_KEY
make ingest && make index
.venv/bin/python scripts/03_make_testset.py
.venv/bin/python scripts/06_check_testset.py            # free, validates the gold set
.venv/bin/python scripts/00_diagnose_metrics.py         # ~$0.02, validates the metrics
.venv/bin/python scripts/07_estimate_cost.py --budget 5 # price the plan in dollars
.venv/bin/python scripts/04_run_experiments.py --metrics core --origin seed
make report
```

Raw per-question scores for every run are in `results/runs/*.json`. Per-category data is
in `results/by_category.csv`. The five-question smoke runs in `results/runs_smoke_n5/` are
retained for provenance and are **not** part of any reported result.
