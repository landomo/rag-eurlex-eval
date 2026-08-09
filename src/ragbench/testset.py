"""Build the evaluation gold set.

Two halves, on purpose:

  seed      hand-written questions (config/seed_questions.yaml). These encode the
            queries a real user of a compliance assistant would ask, including
            multi-hop and cross-regulation ones that synthetic generators rarely
            produce. Reference answers are drafted by an LLM given the *full*
            source Articles - never via the retrieval system, so the gold set is
            not circular - and flagged for human review.

  generated LLM-generated question/reference pairs grounded in randomly sampled
            Articles, for breadth.

Every record carries `reference_contexts`, which is what lets Ragas compute
context recall and reference-based context precision.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field

from .config import MODELS, TESTSET_PATH
from .ingest import Section, load_sections

QA_SYSTEM = """You write evaluation data for a legal retrieval system. \
Given an excerpt of an EU regulation, produce a question a compliance professional \
would realistically ask, and a reference answer drawn strictly from the excerpt.

Rules:
- The answer must be fully supported by the excerpt. Invent nothing.
- The question must be answerable without seeing the excerpt (name the concept, not "this article").
- Prefer questions about obligations, thresholds, exemptions, definitions and deadlines.
- Return strict JSON: {"question": "...", "reference": "..."}"""

SEED_ANSWER_SYSTEM = """You produce reference answers for evaluating a legal retrieval system.
Answer the question using ONLY the supplied excerpts from EU regulations.
Cite the Article numbers you relied on. If the excerpts genuinely do not answer the
question, reply exactly: INSUFFICIENT.
Return strict JSON: {"reference": "..."}"""


@dataclass
class TestItem:
    id: str
    user_input: str
    reference: str
    reference_contexts: list[str]
    origin: str                      # "seed" | "generated"
    section_ids: list[str] = field(default_factory=list)
    needs_review: bool = False


def _json_from(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(m.group(0))


def _llm(model: str | None = None, temperature: float = 0.0):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or MODELS.judge, temperature=temperature)


# --------------------------------------------------------------------------- generated

def generate_from_sections(
    sections: list[Section], n: int = 45, seed: int = 13, min_chars: int = 900
) -> list[TestItem]:
    rng = random.Random(seed)
    pool = [s for s in sections if s.kind == "article" and len(s.text) >= min_chars]
    rng.shuffle(pool)
    llm = _llm()
    items: list[TestItem] = []

    for s in pool:
        if len(items) >= n:
            break
        excerpt = s.text[:6000]
        try:
            out = llm.invoke(
                [("system", QA_SYSTEM), ("human", f"{s.short_name} - {s.label}\n\n{excerpt}")]
            )
            data = _json_from(out.content)
        except Exception as exc:  # noqa: BLE001 - one bad sample shouldn't kill the run
            print(f"    skip {s.id}: {exc}")
            continue
        items.append(
            TestItem(
                id=f"gen-{len(items):03d}",
                user_input=data["question"].strip(),
                reference=data["reference"].strip(),
                reference_contexts=[excerpt],
                origin="generated",
                section_ids=[s.id],
            )
        )
    return items


# --------------------------------------------------------------------------- seed

def _match_sections(question: str, sections: list[Section], k: int = 4) -> list[Section]:
    """Cheap lexical match to pull the Articles a seed question is about.

    BM25 over section text - no embeddings, no retrieval system, so the gold set
    stays independent of the pipeline under test.
    """
    from rank_bm25 import BM25Okapi

    corpus = [re.findall(r"\w+", s.text.lower()) for s in sections]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(re.findall(r"\w+", question.lower()))
    ranked = sorted(range(len(sections)), key=lambda i: scores[i], reverse=True)[:k]
    return [sections[i] for i in ranked]


def build_seed_items(seed_questions: list[str], sections: list[Section]) -> list[TestItem]:
    llm = _llm()
    items: list[TestItem] = []
    for i, q in enumerate(seed_questions):
        matched = _match_sections(q, sections)
        contexts = [f"{s.short_name} - {s.label}\n{s.text[:5000]}" for s in matched]
        try:
            out = llm.invoke(
                [
                    ("system", SEED_ANSWER_SYSTEM),
                    ("human", "Excerpts:\n\n" + "\n\n---\n\n".join(contexts) + f"\n\nQuestion: {q}"),
                ]
            )
            reference = _json_from(out.content)["reference"].strip()
        except Exception as exc:  # noqa: BLE001
            print(f"    skip seed {i}: {exc}")
            continue
        if reference.upper().startswith("INSUFFICIENT"):
            print(f"    seed {i} unanswerable from matched sections - flagged: {q[:60]}")
            continue
        items.append(
            TestItem(
                id=f"seed-{i:03d}",
                user_input=q,
                reference=reference,
                reference_contexts=contexts,
                origin="seed",
                section_ids=[s.id for s in matched],
                needs_review=True,
            )
        )
    return items


# --------------------------------------------------------------------------- io

def save(items: list[TestItem]) -> None:
    with TESTSET_PATH.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")


def load() -> list[TestItem]:
    if not TESTSET_PATH.exists():
        raise SystemExit("No testset. Run: python scripts/03_make_testset.py")
    with TESTSET_PATH.open(encoding="utf-8") as fh:
        return [TestItem(**json.loads(line)) for line in fh if line.strip()]
