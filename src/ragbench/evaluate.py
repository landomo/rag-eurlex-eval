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


def build_metrics(judge_llm=None, judge_embeddings=None):
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NoiseSensitivity,
        ResponseRelevancy,
    )

    return [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        FactualCorrectness(),
        NoiseSensitivity(),
    ]


def judge_components():
    """The judge is identical across every run, so judge bias is a constant, not a confound."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from .providers import get_chat_llm, get_embeddings

    return (
        LangchainLLMWrapper(get_chat_llm("judge", temperature=0.0)),
        LangchainEmbeddingsWrapper(get_embeddings()),
    )


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
                "_citations": result.citations,
            }
        )
        if verbose and i % 10 == 0:
            print(f"      {i}/{len(testset)} answered")
    return rows


def score(rows: list[dict], run_id: str, seconds: float | None = None) -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    llm, emb = judge_components()
    payload = [
        {k: v for k, v in r.items() if not k.startswith("_")} for r in rows
    ]
    dataset = EvaluationDataset.from_list(payload)

    result = evaluate(
        dataset=dataset,
        metrics=build_metrics(),
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


def evaluate_spec(spec: RunSpec, testset: list[TestItem], skip_existing: bool = True) -> dict:
    from .retrieval import build_retriever

    if skip_existing and already_done(spec.run_id):
        print(f"  = {spec.run_id}: cached, skipping")
        return json.loads(run_path(spec.run_id).read_text(encoding="utf-8"))

    print(f"  > {spec.run_id}")
    t0 = time.time()
    retriever = build_retriever(
        spec.chunker, mode=spec.mode, top_k=spec.top_k, rerank=spec.rerank
    )
    rows = collect_predictions(RagPipeline(retriever), testset)
    payload = score(rows, spec.run_id, seconds=time.time() - t0)
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
