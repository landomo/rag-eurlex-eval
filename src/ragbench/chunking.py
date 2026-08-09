"""The four chunking strategies under comparison.

Each strategy is a function: list[Section] -> list[langchain Document].

  fixed_512           naive token windows, structure-blind. The baseline everyone ships first.
  recursive_1000      character-recursive with legal separators (Article/paragraph markers).
  semantic            embedding-based breakpoints (LangChain SemanticChunker).
  structural_article  one chunk per Article, oversized articles split on paragraph numbers.

fixed/recursive/semantic run over the *flat* regulation text so the comparison is
honest: only `structural_article` gets to see the legal hierarchy.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter

from .ingest import Section

# Splits at "1." / "2." style paragraph markers used throughout EU regulations.
PARA_RE = re.compile(r"\n(?=\d{1,2}\.\s)")


def _flat_documents(sections: Iterable[Section]) -> list[Document]:
    """Rebuild one Document per regulation - what a structure-blind pipeline sees."""
    buckets: dict[str, list[Section]] = {}
    for s in sections:
        buckets.setdefault(s.source, []).append(s)
    docs = []
    for source, secs in buckets.items():
        text = "\n\n".join(s.text for s in secs)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": source, "short_name": secs[0].short_name},
            )
        )
    return docs


def _tag(docs: list[Document], strategy: str) -> list[Document]:
    """Attach strategy + a best-effort citation so answers can cite something."""
    out = []
    for i, d in enumerate(docs):
        meta = dict(d.metadata)
        meta["strategy"] = strategy
        meta["chunk_index"] = i
        if "label" not in meta:
            m = re.search(r"Article\s+(\d+[a-z]?)", d.page_content)
            meta["label"] = f"Article {m.group(1)}" if m else "unlabelled"
        meta["citation"] = f"{meta.get('short_name', meta.get('source', '?'))}, {meta['label']}"
        meta["n_chars"] = len(d.page_content)
        out.append(Document(page_content=d.page_content, metadata=meta))
    return out


# --------------------------------------------------------------------------- strategies

def fixed_512(sections: list[Section]) -> list[Document]:
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=512, chunk_overlap=64
    )
    return _tag(splitter.split_documents(_flat_documents(sections)), "fixed_512")


def recursive_1000(sections: list[Section]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\nArticle ", "\nCHAPTER ", "\nSECTION ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return _tag(splitter.split_documents(_flat_documents(sections)), "recursive_1000")


def semantic(sections: list[Section]) -> list[Document]:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    from .config import MODELS

    splitter = SemanticChunker(
        OpenAIEmbeddings(model=MODELS.embedding),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    docs = splitter.split_documents(_flat_documents(sections))
    # SemanticChunker can emit very long chunks; cap them so context windows stay
    # comparable across strategies.
    capper = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    docs = capper.split_documents(docs)
    return _tag(docs, "semantic")


def structural_article(sections: list[Section]) -> list[Document]:
    """One chunk per Article, carrying its own citation. Oversized articles are
    split on numbered paragraphs, and every piece is prefixed with its heading so
    the fragment stays self-describing when retrieved in isolation."""
    max_chars = 2600
    docs: list[Document] = []
    for s in sections:
        header = f"{s.short_name} - {s.label}"
        if s.heading:
            header += f": {s.heading}"
        base_meta = {
            "source": s.source,
            "short_name": s.short_name,
            "label": s.label,
            "kind": s.kind,
            "section_id": s.id,
        }
        if len(s.text) <= max_chars:
            docs.append(Document(page_content=f"{header}\n\n{s.text}", metadata=dict(base_meta)))
            continue

        parts = [p.strip() for p in PARA_RE.split(s.text) if p.strip()]
        buf = ""
        for part in parts:
            if len(buf) + len(part) > max_chars and buf:
                docs.append(Document(page_content=f"{header}\n\n{buf}", metadata=dict(base_meta)))
                buf = part
            else:
                buf = f"{buf}\n{part}" if buf else part
        if buf:
            docs.append(Document(page_content=f"{header}\n\n{buf}", metadata=dict(base_meta)))
    return _tag(docs, "structural_article")


STRATEGIES: dict[str, Callable[[list[Section]], list[Document]]] = {
    "fixed_512": fixed_512,
    "recursive_1000": recursive_1000,
    "semantic": semantic,
    "structural_article": structural_article,
}


def chunk(strategy: str, sections: list[Section]) -> list[Document]:
    if strategy not in STRATEGIES:
        raise KeyError(f"Unknown strategy {strategy!r}. Options: {list(STRATEGIES)}")
    return STRATEGIES[strategy](sections)


def stats(docs: list[Document]) -> dict:
    lens = [len(d.page_content) for d in docs]
    lens.sort()
    n = len(lens)
    return {
        "n_chunks": n,
        "chars_mean": round(sum(lens) / n, 1) if n else 0,
        "chars_p50": lens[n // 2] if n else 0,
        "chars_p95": lens[int(n * 0.95)] if n else 0,
        "chars_max": lens[-1] if n else 0,
    }
