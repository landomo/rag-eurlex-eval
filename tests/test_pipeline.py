"""Offline structural tests. No network, no API key.

They verify the wiring - segmentation, all four chunkers, dense/hybrid/reranked
retrieval, prompt assembly - using deterministic fake embeddings and a stub LLM.
What they cannot verify is answer quality; that is what the Ragas run is for.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from fixtures import FAKE_REGULATION
from ragbench import chunking, ingest
from ragbench.config import Regulation
from ragbench.pipeline import SYSTEM_PROMPT, format_context

REG = Regulation(key="test_reg", celex="32099R0001", short_name="Widget Act", long_name="Test")


@pytest.fixture(scope="module")
def sections():
    return ingest.segment(FAKE_REGULATION, REG)


def test_segmentation_finds_articles_and_preamble(sections):
    labels = {s.label for s in sections}
    assert "Recitals" in labels
    assert {"Article 1", "Article 3", "Article 6"} <= labels
    assert any(s.kind == "annex" for s in sections)


def test_article_bodies_are_not_truncated(sections):
    art3 = next(s for s in sections if s.label == "Article 3")
    assert "subliminal techniques" in art3.text
    assert "scientific research" in art3.text


def test_headings_are_captured(sections):
    art6 = next(s for s in sections if s.label == "Article 6")
    assert "Penalties" in art6.heading


@pytest.mark.parametrize("strategy", ["recursive_1000", "structural_article"])
def test_chunkers_produce_tagged_documents(strategy, sections):
    docs = chunking.chunk(strategy, sections)
    assert docs, f"{strategy} produced no chunks"
    for d in docs:
        assert d.page_content.strip()
        assert d.metadata["strategy"] == strategy
        assert "citation" in d.metadata
    st = chunking.stats(docs)
    assert st["n_chunks"] == len(docs)


def test_fixed_token_chunker(sections):
    try:
        docs = chunking.chunk("fixed_512", sections)
    except Exception as exc:  # tiktoken needs to download its BPE file once
        pytest.skip(f"tiktoken unavailable offline: {exc}")
    assert docs
    assert all(d.metadata["strategy"] == "fixed_512" for d in docs)


def test_structural_chunks_carry_article_labels(sections):
    docs = chunking.chunk("structural_article", sections)
    labels = {d.metadata["label"] for d in docs}
    assert "Article 4" in labels
    art4 = next(d for d in docs if d.metadata["label"] == "Article 4")
    # Every structural chunk is prefixed with its own heading so it stays
    # self-describing when retrieved out of context.
    assert art4.page_content.startswith("Widget Act - Article 4")


def test_flat_chunkers_do_not_leak_section_ids(sections):
    docs = chunking.chunk("recursive_1000", sections)
    assert all("section_id" not in d.metadata for d in docs), (
        "structure-blind chunkers must not receive section metadata, "
        "or the ablation is rigged"
    )


# ------------------------------------------------------------------ retrieval

@pytest.fixture(scope="module")
def indexed(tmp_path_factory, sections):
    from langchain_chroma import Chroma

    docs = chunking.chunk("structural_article", sections)
    emb = DeterministicFakeEmbedding(size=64)
    store = Chroma.from_documents(
        docs,
        embedding=emb,
        collection_name="test_structural",
        persist_directory=str(tmp_path_factory.mktemp("chroma")),
    )
    return store, docs


def test_dense_retriever_returns_k(indexed):
    from ragbench.retrieval import build_retriever

    store, _ = indexed
    r = build_retriever("structural_article", mode="dense", top_k=3, vectorstore=store)
    out = r.invoke("What are the penalties for prohibited widget practices?")
    assert len(out) == 3
    assert all(isinstance(d, Document) for d in out)


def test_hybrid_retriever_merges_lexical_and_dense(indexed, monkeypatch):
    from ragbench import retrieval

    store, docs = indexed
    monkeypatch.setattr(retrieval, "load_chunks", lambda s: docs)
    r = retrieval.build_retriever("structural_article", mode="hybrid", top_k=4, vectorstore=store)
    out = r.invoke("administrative fines total worldwide annual turnover")
    assert out
    # BM25 should surface the penalties article for this very lexical query.
    assert any("Article 6" in d.metadata.get("label", "") for d in out)


def test_reranked_retriever_narrows_to_top_k(indexed, monkeypatch):
    pytest.importorskip("flashrank")
    from ragbench import retrieval

    store, docs = indexed
    monkeypatch.setattr(retrieval, "load_chunks", lambda s: docs)
    try:
        r = retrieval.build_retriever(
            "structural_article", mode="hybrid", top_k=2, rerank=True, fetch_k=8, vectorstore=store
        )
        out = r.invoke("When is a widget system considered high-risk?")
    except Exception as exc:  # first use downloads the cross-encoder weights
        pytest.skip(f"flashrank model unavailable offline: {exc}")
    assert len(out) <= 2


def test_invalid_mode_rejected(indexed):
    from ragbench.retrieval import build_retriever

    store, _ = indexed
    with pytest.raises(ValueError):
        build_retriever("structural_article", mode="sparse", vectorstore=store)


# ------------------------------------------------------------------ prompt

def test_context_formatting_numbers_and_cites():
    docs = [
        Document(page_content="body one", metadata={"citation": "Widget Act, Article 1"}),
        Document(page_content="body two", metadata={"citation": "Widget Act, Article 2"}),
    ]
    ctx = format_context(docs)
    assert "[1] Widget Act, Article 1" in ctx
    assert "[2] Widget Act, Article 2" in ctx


def test_system_prompt_mandates_abstention():
    assert "does not contain enough information" in SYSTEM_PROMPT
    assert "Never rely on prior knowledge" in SYSTEM_PROMPT


# ------------------------------------------------------------------ providers

def test_default_provider_wiring_is_anthropic_plus_local(monkeypatch):
    """Defaults must need exactly one key. If this breaks, the README lies."""
    import importlib

    for var in ["RAGBENCH_LLM_PROVIDER", "RAGBENCH_EMBED_PROVIDER",
                "RAGBENCH_GEN_MODEL", "RAGBENCH_JUDGE_MODEL", "RAGBENCH_EMBED_MODEL"]:
        monkeypatch.delenv(var, raising=False)

    from ragbench import config

    importlib.reload(config)
    assert config.MODELS.llm_provider == "anthropic"
    assert config.MODELS.embed_provider == "local"
    assert config.MODELS.generator.startswith("claude-")
    assert config.MODELS.embedding == "BAAI/bge-small-en-v1.5"


def test_openai_provider_selects_openai_model_defaults(monkeypatch):
    import importlib

    monkeypatch.setenv("RAGBENCH_LLM_PROVIDER", "openai")
    monkeypatch.setenv("RAGBENCH_EMBED_PROVIDER", "openai")
    for var in ["RAGBENCH_GEN_MODEL", "RAGBENCH_JUDGE_MODEL", "RAGBENCH_EMBED_MODEL"]:
        monkeypatch.delenv(var, raising=False)

    from ragbench import config

    importlib.reload(config)
    assert config.MODELS.generator == "gpt-4o-mini"
    assert config.MODELS.embedding == "text-embedding-3-small"
    importlib.reload(config)


def test_missing_key_fails_fast_with_a_useful_message(monkeypatch):
    """Must hold whether or not the developer has a populated .env on disk.

    config.py calls load_dotenv() at import, so the env var has to be removed
    AFTER the reload - otherwise .env silently puts the key back and the test
    passes for the wrong reason.
    """
    import importlib

    monkeypatch.setenv("RAGBENCH_LLM_PROVIDER", "anthropic")

    from ragbench import config, providers

    importlib.reload(config)
    importlib.reload(providers)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc:
        providers.get_chat_llm("generator")
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_unknown_provider_is_rejected(monkeypatch):
    import importlib

    monkeypatch.setenv("RAGBENCH_LLM_PROVIDER", "cohere")
    from ragbench import config, providers

    importlib.reload(config)
    importlib.reload(providers)
    with pytest.raises(SystemExit):
        providers.get_chat_llm("generator")


def test_run_key_includes_question_count():
    """A --limit smoke run must not occupy the full run's cache slot.

    This exact collision silently produced a results table built from 5-question
    runs while claiming to be the full gold set.
    """
    from ragbench.config import RunSpec

    spec = RunSpec("structural_article", "hybrid")
    assert spec.run_key(5) != spec.run_key(57)
    assert "n5" in spec.run_key(5) and "n57" in spec.run_key(57)
    assert RunSpec("a", "dense").run_key(10) != RunSpec("a", "dense", rerank=True).run_key(10)


def test_metric_sets_are_coherent():
    from ragbench.evaluate import METRIC_CALL_COST, METRIC_SETS, estimate_calls

    assert set(METRIC_SETS["core"]) < set(METRIC_SETS["full"])
    assert all(m in METRIC_CALL_COST for m in METRIC_SETS["full"])
    cheap = estimate_calls("core", 57, 9)["total"]
    dear = estimate_calls("full", 57, 9)["total"]
    assert cheap < dear
