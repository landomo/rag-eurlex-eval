"""Central configuration. Everything reads from here so experiments stay reproducible."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INDEXES = DATA / "indexes"
RESULTS = ROOT / "results"
RUNS = RESULTS / "runs"

for _p in (RAW, PROCESSED, INDEXES, RESULTS, RUNS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- corpus

@dataclass(frozen=True)
class Regulation:
    key: str
    celex: str
    short_name: str
    long_name: str

    @property
    def url(self) -> str:
        # EUR-Lex serves the consolidated English HTML at a stable CELEX endpoint.
        return f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{self.celex}"

    @property
    def raw_path(self) -> Path:
        return RAW / f"{self.key}.txt"


CORPUS: list[Regulation] = [
    Regulation(
        key="ai_act",
        celex="32024R1689",
        short_name="EU AI Act",
        long_name="Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence",
    ),
    Regulation(
        key="gdpr",
        celex="32016R0679",
        short_name="GDPR",
        long_name="Regulation (EU) 2016/679 on the protection of natural persons with regard to the processing of personal data",
    ),
]

SECTIONS_PATH = PROCESSED / "sections.jsonl"
TESTSET_PATH = PROCESSED / "testset.jsonl"

# --------------------------------------------------------------------------- models

# Sensible per-provider defaults, so switching provider does not require also
# knowing that provider's model names.
_LLM_DEFAULTS = {
    "anthropic": {"generator": "claude-haiku-4-5-20251001", "judge": "claude-haiku-4-5-20251001"},
    "openai": {"generator": "gpt-4o-mini", "judge": "gpt-4o-mini"},
}
_EMBED_DEFAULTS = {
    "local": "BAAI/bge-small-en-v1.5",
    "openai": "text-embedding-3-small",
    "voyage": "voyage-3",
}

_LLM_PROVIDER = os.getenv("RAGBENCH_LLM_PROVIDER", "anthropic").lower()
_EMBED_PROVIDER = os.getenv("RAGBENCH_EMBED_PROVIDER", "local").lower()


@dataclass(frozen=True)
class Models:
    llm_provider: str = _LLM_PROVIDER
    embed_provider: str = _EMBED_PROVIDER
    generator: str = os.getenv(
        "RAGBENCH_GEN_MODEL",
        _LLM_DEFAULTS.get(_LLM_PROVIDER, _LLM_DEFAULTS["anthropic"])["generator"],
    )
    judge: str = os.getenv(
        "RAGBENCH_JUDGE_MODEL",
        _LLM_DEFAULTS.get(_LLM_PROVIDER, _LLM_DEFAULTS["anthropic"])["judge"],
    )
    embedding: str = os.getenv(
        "RAGBENCH_EMBED_MODEL", _EMBED_DEFAULTS.get(_EMBED_PROVIDER, _EMBED_DEFAULTS["local"])
    )


MODELS = Models()

TOP_K = int(os.getenv("RAGBENCH_TOP_K", "5"))
RERANK_FETCH_K = int(os.getenv("RAGBENCH_RERANK_FETCH_K", "20"))

# --------------------------------------------------------------------------- grid

CHUNKERS = ["fixed_512", "recursive_1000", "semantic", "structural_article"]
RETRIEVAL_MODES = ["dense", "hybrid"]

# Hybrid weighting: BM25 first, dense second. Lexical weight is deliberately
# non-trivial because legal queries carry exact tokens ("Article 22", "DPIA").
HYBRID_WEIGHTS: tuple[float, float] = (0.4, 0.6)


@dataclass(frozen=True)
class RunSpec:
    chunker: str
    mode: str
    rerank: bool = False
    top_k: int = TOP_K

    @property
    def run_id(self) -> str:
        return f"{self.chunker}__{self.mode}{'__rerank' if self.rerank else ''}"

    def run_key(self, n_questions: int) -> str:
        """Cache key INCLUDING the question count.

        Without n, a `--limit 5` smoke run writes the same filename as the full
        run, the full run skips it as cached, and the results table silently
        reports 5-question scores as if they were the whole gold set.
        """
        return f"{self.run_id}__n{n_questions}"


def default_grid() -> list[RunSpec]:
    """4 chunkers x 2 retrieval modes. Reranking is added separately, on the winner."""
    return [RunSpec(c, m) for c in CHUNKERS for m in RETRIEVAL_MODES]


def require_api_key() -> None:
    """Validate credentials and model wiring before spending an hour of compute."""
    from .providers import preflight

    preflight()
