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

@dataclass(frozen=True)
class Models:
    generator: str = os.getenv("RAGBENCH_GEN_MODEL", "gpt-4o-mini")
    judge: str = os.getenv("RAGBENCH_JUDGE_MODEL", "gpt-4o-mini")
    embedding: str = os.getenv("RAGBENCH_EMBED_MODEL", "text-embedding-3-small")


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


def default_grid() -> list[RunSpec]:
    """4 chunkers x 2 retrieval modes. Reranking is added separately, on the winner."""
    return [RunSpec(c, m) for c in CHUNKERS for m in RETRIEVAL_MODES]


def require_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key
