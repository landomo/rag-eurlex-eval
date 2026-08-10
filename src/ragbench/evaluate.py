"""Ragas evaluation harness.

Metrics and why each one is here:

  Faithfulness                    - are the claims in the answer entailed by the
                                    retrieved context? This is the hallucination
                                    metric and the one worth putting on a CV.
  ResponseRelevancy               - does the answer actually address the question?
  LLMContextPrecisionWithReference- of the retrieved chunks, how many were useful?
                                    Punishes over-retrieval; this is what reranking
                                    is supposed to move.
  LLMContextRecall                - did retrieval surface everything the reference
                                    answer needed? This is what chunking moves.
  FactualCorrectness              - end-to-end answer quality against the reference.
  NoiseSensitivity                - how often irrelevant retrieved context leaks
                                    incorrect claims into the answer.

Precision/recall are the retrieval-side pair; faithfulness/relevancy are the
generation-side pair. Reporting only one side is how RAG benchmarks mislead.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .config import RUNS, RunSpec
from .providers import describe
from .pipeline import RagPipeline
from .testset import TestItem


# Approximate judge LLM calls per sample, measured against Ragas 0.3.9 internals.
# Used only to warn you what a run will cost before it starts.
METRIC_CALL_COST = {
    "faithfulness": 2,
    "llm_context_precision_with_reference": 5,   # one call per retrieved chunk
    "context_recall": 1,
    "answer_relevancy": 3,                       # strictness=3
    "factual_correctness": 2,
    "noise_sensitivity": 5,                      # one call per retrieved chunk
}

# "core" is the defensible minimum: one generation-side metric and both
# retrieval-side metrics. It is ~2.5x cheaper than "full" and covers every
# claim the experiment actually makes.
# Which metrics actually return numbers, measured over 25 real samples with a
# Claude judge on Ragas 0.3.9 (see docs/METRIC_SUPPORT.md):
#
#   faithfulness                          25/25  usable
#   llm_context_precision_with_reference  25/25  usable
#   context_recall                        25/25  usable
#   factual_correctness                   25/25  usable
#   noise_sensitivity                     20/25  partial
#   answer_relevancy                       0/25  BROKEN - excluded from every set
#
# answer_relevancy asks the judge for free-text JSON and parses it; Claude's
# output never matched the expected schema, so it returned NaN while still
# billing 3 calls per sample. Paying for a column of NaN is worse than not
# measuring it, so it is not in any set. Use an OpenAI judge if you want it.
METRIC_SETS = {
    "core": ["faithfulness", "llm_context_precision_with_reference", "context_recall"],
    "standard": [
        "faithfulness",
        "llm_context_precision_with_reference",
        "context_recall",
        "factual_correctness",
    ],
    "full": [
        "faithfulness",
        "llm_context_precision_with_reference",
        "context_recall",
        "factual_correctness",
        "noise_sensitivity",
    ],
}


def build_metrics(which: str = "core", llm=None):
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NoiseSensitivity,
        ResponseRelevancy,
    )

    by_name = {
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "factual_correctness": FactualCorrectness,
        "noise_sensitivity": NoiseSensitivity,
    }
    if which not in METRIC_SETS:
        raise SystemExit(f"Unknown metric set {which!r}. Options: {list(METRIC_SETS)}")
    # Attach the judge explicitly rather than relying on evaluate() to inject it,
    # which matters for the Instructor-backed LLM.
    return [by_name[n](llm=llm) if llm is not None else by_name[n]() for n in METRIC_SETS[which]]


def estimate_calls(metric_set: str, n_questions: int, n_configs: int) -> dict:
    per_sample = sum(METRIC_CALL_COST[m] for m in METRIC_SETS[metric_set])
    judge = per_sample * n_questions * n_configs
    gen = n_questions * n_configs
    return {"generation_calls": gen, "judge_calls": judge, "total": gen + judge}


def judge_components():
    """The judge is identical across every run, so judge bias is a constant, not a confound.

    Backend note: ragas 0.3.9's classic metrics call the LangChain interface
    (`agenerate_prompt`) on whatever LLM they are given, so `llm_factory`'s
    Instructor-backed LLM raises AttributeError on every metric. llm_factory is
    for Ragas' newer experimental API, not these metrics. The LangChain wrapper
    is therefore the only working backend here; RAGBENCH_JUDGE_BACKEND=instructor
    is retained only so the incompatibility can be reproduced.
    """
    import os

    from ragas.embeddings import LangchainEmbeddingsWrapper

    from .config import MODELS
    from .providers import get_embeddings

    backend = os.getenv("RAGBENCH_JUDGE_BACKEND", "langchain").lower()
    emb = LangchainEmbeddingsWrapper(get_embeddings())

    if backend == "instructor" and MODELS.llm_provider in {"anthropic", "openai"}:
        from ragas.llms import llm_factory

        try:
            if MODELS.llm_provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic()
            else:
                import openai

                client = openai.OpenAI()
            return llm_factory(MODELS.judge, provider=MODELS.llm_provider, client=client), emb
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Could not initialise the Instructor judge backend: {exc}\n\n"
                "On Python 3.9 this is usually the missing type-annotation backport:\n"
                "    .venv/bin/pip install eval_type_backport==0.4.0 instructor==1.15.4\n\n"
                "Or fall back to the LangChain judge (note: answer_relevancy and\n"
                "noise_sensitivity return NaN with Claude on that path):\n"
                "    RAGBENCH_JUDGE_BACKEND=langchain .venv/bin/python scripts/00_diagnose_metrics.py"
            ) from exc

    from ragas.llms import LangchainLLMWrapper

    from .providers import get_chat_llm

    return LangchainLLMWrapper(get_chat_llm("judge", temperature=0.0)), emb


def collect_predictions(pipeline: RagPipeline, testset: list[TestItem], verbose: bool = True) -> list[dict]:
    rows = []
    for i, item in enumerate(testset, 1):
        result = pipeline(item.user_input)
        rows.append(
            {
                "user_input": item.user_input,
                "retrieved_contexts": result.contexts,
                "response": result.answer,
                "reference": item.reference,
                "_id": item.id,
                "_origin": item.origin,
                "_category": getattr(item, "category", "generated"),
                "_citations": result.citations,
            }
        )
        if verbose and i % 10 == 0:
            print(f"      {i}/{len(testset)} answered")
    return rows


def score(rows: list[dict], run_id: str, seconds: float | None = None,
          metric_set: str = "core") -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    llm, emb = judge_components()
    payload = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
    ]
    dataset = EvaluationDataset.from_list(payload)

    result = evaluate(
        dataset=dataset,
        metrics=build_metrics(metric_set, llm=llm),
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=8, timeout=180, max_retries=5),
        show_progress=True,
    )

    df = result.to_pandas()
    for r, (_, prow) in zip(rows, df.iterrows()):
        prow_d = prow.to_dict()
        r["_scores"] = {
            k: (None if _isnan(v) else float(v))
            for k, v in prow_d.items()
            if isinstance(v, (int, float))
        }

    aggregate = {}
    for col in df.columns:
        if df[col].dtype.kind in "fi":
            aggregate[col] = round(float(df[col].mean(skipna=True)), 4)

    return {
        "run_id": run_id,
        "metric_set": metric_set,
        "n_samples": len(rows),
        "seconds": round(seconds, 1) if seconds else None,
        "models": describe(),
        "aggregate": aggregate,
        "samples": rows,
    }


def _isnan(v) -> bool:
    try:
        return v != v
    except Exception:  # noqa: BLE001
        return False


def run_path(run_id: str) -> Path:
    return RUNS / f"{run_id}.json"


def already_done(run_id: str) -> bool:
    return run_path(run_id).exists()


def save_run(payload: dict) -> Path:
    p = run_path(payload["run_id"])
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def evaluate_spec(
    spec: RunSpec,
    testset: list[TestItem],
    skip_existing: bool = True,
    metric_set: str = "core",
) -> dict:
    from .retrieval import build_retriever

    key = spec.run_key(len(testset))
    if skip_existing and already_done(key):
        print(f"  = {key}: cached, skipping")
        return json.loads(run_path(key).read_text(encoding="utf-8"))

    print(f"  > {key}")
    t0 = time.time()
    retriever = build_retriever(
        spec.chunker, mode=spec.mode, top_k=spec.top_k, rerank=spec.rerank
    )
    rows = collect_predictions(RagPipeline(retriever), testset)
    payload = score(rows, key, seconds=time.time() - t0, metric_set=metric_set)
    payload["spec"] = asdict(spec)
    save_run(payload)
    agg = payload["aggregate"]
    print(
        "    faithfulness={:.3f} ctx_precision={:.3f} ctx_recall={:.3f}".format(
            agg.get("faithfulness", float("nan")),
            agg.get("llm_context_precision_with_reference", float("nan")),
            agg.get("context_recall", float("nan")),
        )
    )
    return payload
