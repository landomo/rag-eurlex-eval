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

# EU regulations number article paragraphs "1." and recitals "(1)". Match both,
# or the Recitals block never splits and becomes one enormous chunk.
PARA_RE = re.compile(r"\n(?=(?:\d{1,2}\.|\(\d{1,3}\))\s)")


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
    """Embedding-breakpoint chunking.

    The unit of comparison is the PARAGRAPH, not the sentence. LangChain's default
    splits on sentence boundaries, which on this corpus means ~8,000 embedding calls
    and many minutes on CPU with no output. EU regulations are drafted in numbered
    paragraphs that already are the semantic unit, so splitting there cuts the work
    by roughly 4x and arguably measures the right boundaries. Still structure-blind:
    it sees paragraph breaks in flat text, not Article numbers.
    """
    import time

    from langchain_experimental.text_splitter import SemanticChunker

    from .providers import get_embeddings

    docs_in = _flat_documents(sections)
    units = sum(len(re.split(r"\n\s*\n", d.page_content)) for d in docs_in)
    print(
        f"    semantic: embedding ~{units:,} paragraph units on CPU "
        f"(this is the slow strategy; expect a few minutes)",
        flush=True,
    )

    splitter = SemanticChunker(
        get_embeddings(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
        sentence_split_regex=r"\n\s*\n",
    )

    out: list[Document] = []
    for i, d in enumerate(docs_in, 1):
        t0 = time.time()
        out.extend(splitter.split_documents([d]))
        print(
            f"      [{i}/{len(docs_in)}] {d.metadata.get('short_name', '?')}: "
            f"{len(out)} chunks so far ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # SemanticChunker can emit very long chunks; cap them so context windows stay
    # comparable across strategies.
    capper = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    return _tag(capper.split_documents(out), "semantic")


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

        # Backstop: if a "paragraph" is still oversized (unnumbered prose, tables,
        # an annex), split it by characters. Without this, one unmatched pattern
        # silently produces a chunk too large for any context window.
        expanded: list[str] = []
        hard = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=100)
        for part in parts:
            expanded.extend([part] if len(part) <= max_chars else hard.split_text(part))

        buf = ""
        for part in expanded:
            if len(buf) + len(part) > max_chars and buf:
                docs.append(Document(page_content=f"{header}\n\n{buf}", metadata=dict(base_meta)))
                buf = part
            else:
                buf = f"{buf}\n{part}" if buf else part
        if buf:
            docs.append(Document(page_content=f"{header}\n\n{buf}", metadata=dict(base_meta)))

    oversized = [d for d in docs if len(d.page_content) > max_chars * 1.5]
    if oversized:
        raise AssertionError(
            f"structural_article produced {len(oversized)} oversized chunks "
            f"(largest {max(len(d.page_content) for d in oversized):,} chars). "
            "This would overflow the generator context window."
        )
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
