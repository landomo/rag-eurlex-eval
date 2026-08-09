"""Provider abstraction for LLMs and embeddings.

The experiment is about retrieval, so the generator and judge are held constant
across runs - but which vendor supplies them is a config choice, not a rewrite.

  LLM        anthropic (default) | openai
  Embeddings local (default)     | openai | voyage

Anthropic serves no embeddings API, so the default embedding backend is local:
fastembed runs BAAI/bge-small-en-v1.5 through ONNX Runtime - no PyTorch, no second
API key, and it reuses the runtime FlashRank already needs for reranking. That
keeps the whole pipeline runnable from a single ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import MODELS


class ProviderError(SystemExit):
    pass


def _require(var: str, provider: str) -> str:
    val = os.getenv(var)
    if not val:
        raise ProviderError(
            f"{var} is not set, but the {provider!r} provider needs it.\n"
            f"Copy .env.example to .env and add it. Never commit .env - it is gitignored."
        )
    return val


def get_chat_llm(role: str = "generator", temperature: float = 0.0) -> BaseChatModel:
    """role is 'generator' (answers questions) or 'judge' (scores them)."""
    model = MODELS.generator if role == "generator" else MODELS.judge
    provider = MODELS.llm_provider

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        _require("ANTHROPIC_API_KEY", provider)
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=2048)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        _require("OPENAI_API_KEY", provider)
        return ChatOpenAI(model=model, temperature=temperature)

    raise ProviderError(f"Unknown RAGBENCH_LLM_PROVIDER={provider!r}. Use 'anthropic' or 'openai'.")


def get_embeddings() -> Embeddings:
    provider = MODELS.embed_provider

    if provider == "local":
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

        # Downloads ~130 MB of ONNX weights on first use, then runs offline.
        return FastEmbedEmbeddings(model_name=MODELS.embedding)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        _require("OPENAI_API_KEY", provider)
        return OpenAIEmbeddings(model=MODELS.embedding)

    if provider == "voyage":
        from langchain_voyageai import VoyageAIEmbeddings

        _require("VOYAGE_API_KEY", provider)
        return VoyageAIEmbeddings(model=MODELS.embedding)

    raise ProviderError(
        f"Unknown RAGBENCH_EMBED_PROVIDER={provider!r}. Use 'local', 'openai' or 'voyage'."
    )


def describe() -> dict:
    return {
        "llm_provider": MODELS.llm_provider,
        "generator": MODELS.generator,
        "judge": MODELS.judge,
        "embed_provider": MODELS.embed_provider,
        "embedding": MODELS.embedding,
    }


def preflight() -> None:
    """Fail fast with a useful message rather than 400ing on question 1 of 73."""
    get_chat_llm("generator")
    get_chat_llm("judge")
    get_embeddings()
