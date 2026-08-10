"""Retriever construction: dense, hybrid (BM25 + dense), and optional cross-encoder rerank.

Dense  : Chroma, cosine similarity, top-k.
Hybrid : EnsembleRetriever doing reciprocal-rank fusion over BM25 and dense.
         Lexical weight is meaningful here because legal queries carry exact tokens
         ("Article 22", "legitimate interest", "DPIA").
Rerank : over-retrieve fetch_k, then re-score with a FlashRank cross-encoder and
         keep top-k. Runs on CPU, no extra API key.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from pydantic import Field

from .config import HYBRID_WEIGHTS, RERANK_FETCH_K, TOP_K
from .index import load_chunks, open_index


class _TopK(BaseRetriever):
    """Truncate a retriever's output to k documents.

    EnsembleRetriever fuses ranked lists with RRF but returns the *union*, so with
    two retrievers at k=5 it can return up to 10 documents. Left unbounded that
    makes hybrid look better than dense partly because it simply supplies more
    text to the generator - and costs proportionally more per question. The
    comparison is only meaningful if both arms hand over the same budget.
    """

    inner: BaseRetriever = Field(...)
    k: int = Field(default=TOP_K)

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.inner.invoke(query)[: self.k]

    async def _aget_relevant_documents(self, query: str, *, run_manager=None):
        return (await self.inner.ainvoke(query))[: self.k]


def build_retriever(
    strategy: str,
    mode: str = "dense",
    top_k: int = TOP_K,
    rerank: bool = False,
    fetch_k: int = RERANK_FETCH_K,
    embeddings: Embeddings | None = None,
    vectorstore=None,
) -> BaseRetriever:
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"mode must be 'dense' or 'hybrid', got {mode!r}")

    store = vectorstore if vectorstore is not None else open_index(strategy, embeddings)
    k = fetch_k if rerank else top_k

    dense = store.as_retriever(search_kwargs={"k": k})

    if mode == "dense":
        base: BaseRetriever = dense
    else:
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever

        bm25 = BM25Retriever.from_documents(load_chunks(strategy))
        bm25.k = k
        base = _TopK(
            inner=EnsembleRetriever(retrievers=[bm25, dense], weights=list(HYBRID_WEIGHTS)),
            k=k,
        )

    if not rerank:
        return base

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain_community.document_compressors import FlashrankRerank

    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_k)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)
