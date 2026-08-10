# Metric support with a Claude judge

Ragas metrics are LLM-judged, and not all of them survive a change of judge model.
This was measured, not assumed: 25 real samples across five retrieval
configurations, Claude Haiku 4.5 as judge, Ragas 0.3.9.

| Metric | Real values | Status | In which set |
|---|---|---|---|
| `faithfulness` | 25/25 | usable | core, standard, full |
| `llm_context_precision_with_reference` | 25/25 | usable | core, standard, full |
| `context_recall` | 25/25 | usable | core, standard, full |
| `factual_correctness` | 25/25 | usable | standard, full |
| `noise_sensitivity` | 20/25 | partial | full |
| `answer_relevancy` | **0/25** | **broken** | none |

## Why `answer_relevancy` fails

It prompts the judge to emit JSON, parses the text, and on parse failure sets
`np.nan`:

```python
if all(q == "" for q in gen_questions):
    logger.warning("Invalid JSON response. Expected dictionary with key 'question'")
    score = np.nan
```

Claude's output never matched the expected schema, so every sample returned NaN —
while still costing 3 judge calls each. A column of NaN you paid for is worse than
a metric you chose not to measure, so it is excluded from every metric set.

This is distinct from the *noncommittal* path in the same function, which returns
**0**, not NaN. Abstentions score zero; parse failures score NaN. Ours were parse
failures — NaN appeared on non-abstaining answers too.

## What did not work

`ragas.llms.llm_factory(provider="anthropic")` looked like the fix: it enforces
schemas via Instructor tool-calling instead of parsing free text. It is not
compatible with these metrics:

```
AttributeError: 'InstructorLLM' object has no attribute 'agenerate_prompt'
```

Ragas 0.3.9's classic metrics call the LangChain interface on whatever LLM they
are handed; `llm_factory` targets Ragas' newer experimental API. The type
signature of `PydanticPrompt.generate_multiple` accepts `InstructorBaseRagasLLM`,
which is misleading — the runtime dispatch does not.

## Consequence for the results

The project reports two retrieval-side metrics (context precision, context recall)
and two generation-side metrics (faithfulness, factual correctness). That supports
every claim made: chunking strategy is judged on recall, reranking on precision,
generation quality on faithfulness and factual correctness.

To obtain `answer_relevancy`, set `RAGBENCH_LLM_PROVIDER=openai` and re-run — the
provider abstraction makes that a config change, not a code change.
