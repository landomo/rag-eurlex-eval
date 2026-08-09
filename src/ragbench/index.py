"""Build one persistent Chroma collection per chunking strategy.

Chunks are also written to disk as JSONL so the BM25 half of the hybrid retriever
can be rebuilt cheaply without re-chunking or re-embedding.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import INDEXES
from .chunking import chunk, stats
from .ingest import Section, load_sections


def _dir(strategy: str) -> Path:
    return INDEXES / strategy


def chunks_path(strategy: str) -> Path:
    return _dir(strategy) / "chunks.jsonl"


def get_embeddings() -> Embeddings:
    from .providers import get_embeddings as _get

    return _get()


def save_chunks(strategy: str, docs: list[Document]) -> None:
    p = chunks_path(strategy)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(
                json.dumps({"text": d.page_content, "metadata": d.metadata}, ensure_ascii=False)
                + "\n"
            )


def load_chunks(strategy: str) -> list[Document]:
    p = chunks_path(strategy)
    if not p.exists():
        raise SystemExit(f"No chunks for {strategy!r}. Run: python scripts/02_build_indexes.py")
    with p.open(encoding="utf-8") as fh:
        return [
            Document(page_content=r["text"], metadata=r["metadata"])
            for r in (json.loads(line) for line in fh if line.strip())
        ]


def build_index(
    strategy: str,
    sections: list[Section] | None = None,
    embeddings: Embeddings | None = None,
    rebuild: bool = False,
    batch_size: int = 256,
):
    """Chunk -> embed -> persist. Returns (vectorstore, chunk stats)."""
    from langchain_chroma import Chroma

    sections = sections or load_sections()
    embeddings = embeddings or get_embeddings()
    target = _dir(strategy)

    if rebuild and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    if not chunks_path(strategy).exists() or rebuild:
        docs = chunk(strategy, sections)
        save_chunks(strategy, docs)
    else:
        docs = load_chunks(strategy)

    store = Chroma(
        collection_name=strategy,
        embedding_function=embeddings,
        persist_directory=str(target / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )

    existing = store._collection.count()
    if existing == 0:
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            store.add_documents(batch, ids=[f"{strategy}-{i + j}" for j in range(len(batch))])
    return store, stats(docs)


def open_index(strategy: str, embeddings: Embeddings | None = None):
    from langchain_chroma import Chroma

    target = _dir(strategy) / "chroma"
    if not target.exists():
        raise SystemExit(f"No index for {strategy!r}. Run: python scripts/02_build_indexes.py")
    return Chroma(
        collection_name=strategy,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(target),
    )
