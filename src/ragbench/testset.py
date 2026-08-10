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

from .config import TESTSET_PATH
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

- Answer as fully as the excerpts allow. A partial but accurate answer is useful and
  expected; multi-part questions rarely have every part in one Article.
- Cite the Article numbers you relied on, e.g. "(GDPR, Article 28)".
- Do NOT hedge about the excerpts being incomplete. Never write "the excerpts do not
  provide", "insufficient basis", or similar. State what the law says, nothing else.
- Only if the excerpts contain NOTHING relevant to the question at all, reply with the
  single word INSUFFICIENT and nothing else.

Return strict JSON: {"reference": "..."}"""

# The model does not always obey "reply exactly INSUFFICIENT" - it hedges in prose.
# A hedged reference is worse than a dropped one: it silently becomes ground truth
# that says "this cannot be answered", which penalises every configuration equally
# and measures nothing.
REFUSAL_MARKERS = (
    "insufficient",
    "do not provide sufficient",
    "does not provide sufficient",
    "do not contain sufficient",
    "does not contain sufficient",
    "do not contain enough",
    "does not contain enough",
    "not provide a comprehensive",
    "cannot be answered",
    "cannot answer",
    "excerpts do not",
    "excerpts provided do not",
    "supplied excerpts do not",
    "no relevant information",
)


def _is_refusal(text: str) -> bool:
    head = text.strip().lower()[:400]
    return any(m in head for m in REFUSAL_MARKERS)


ABSTENTION_REFERENCE = (
    "The provided context does not contain enough information to answer this question. "
    "This topic is not addressed by the EU AI Act or the GDPR."
)


@dataclass
class TestItem:
    id: str
    user_input: str
    reference: str
    reference_contexts: list[str]
    origin: str                      # "seed" | "generated"
    category: str = "generated"      # lookup | multi_hop | cross_reg | lexical | negative
    section_ids: list[str] = field(default_factory=list)
    needs_review: bool = False


def _json_from(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in model output: {text[:200]}")
    return json.loads(m.group(0))


def _llm(temperature: float = 0.0):
    from .providers import get_chat_llm

    return get_chat_llm("judge", temperature=temperature)


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

        # Same trap as the seed path: a hedged "the excerpt does not provide..."
        # becomes ground truth asserting the question is unanswerable.
        if _is_refusal(data.get("reference", "")):
            print(f"    skip {s.id}: hedged reference, not usable as ground truth")
            continue

        items.append(
            TestItem(
                id=f"gen-{len(items):03d}",
                user_input=data["question"].strip(),
                reference=data["reference"].strip(),
                reference_contexts=[excerpt],
                origin="generated",
                category="generated",
                section_ids=[s.id],
            )
        )
    return items


# --------------------------------------------------------------------------- seed

def _match_sections(question: str, sections: list[Section], k: int = 8) -> list[Section]:
    """Cheap lexical match to pull the Articles a seed question is about.

    BM25 over section text - no embeddings, no retrieval system, so the gold set
    stays independent of the pipeline under test.
    """
    from rank_bm25 import BM25Okapi

    # Index heading + label alongside body text: multi-hop questions often name the
    # concept ("data protection impact assessment") that appears in the Article
    # heading but only obliquely in its body.
    corpus = [
        re.findall(r"\w+", f"{s.label} {s.heading} {s.heading} {s.text}".lower())
        for s in sections
    ]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(re.findall(r"\w+", question.lower()))
    ranked = sorted(range(len(sections)), key=lambda i: scores[i], reverse=True)[:k]
    return [sections[i] for i in ranked]


def build_seed_items(seeds: list[dict], sections: list[Section]) -> list[TestItem]:
    """Draft a reference answer for each hand-written question.

    Negatives never touch the LLM: they are unanswerable by construction, so their
    reference IS the refusal. Keeping them in the gold set is the whole point -
    it makes over-confident answering measurable.

    For the rest, matching escalates (8 sections, then 16) before giving up, because
    multi-hop and cross-regulation questions are exactly the valuable ones and also
    exactly the ones a narrow lexical match misses.
    """
    llm = _llm()
    items: list[TestItem] = []
    unresolved: list[str] = []

    for i, entry in enumerate(seeds):
        q = entry["q"] if isinstance(entry, dict) else str(entry)
        category = entry.get("category", "lookup") if isinstance(entry, dict) else "lookup"

        if category == "negative":
            items.append(
                TestItem(
                    id=f"seed-{i:03d}",
                    user_input=q,
                    reference=ABSTENTION_REFERENCE,
                    reference_contexts=[],
                    origin="seed",
                    category=category,
                    needs_review=False,
                )
            )
            continue

        reference, matched = "", []
        for k in (8, 16):
            matched = _match_sections(q, sections, k=k)
            contexts = [f"{s.short_name} - {s.label}\n{s.text[:9000]}" for s in matched]
            try:
                out = llm.invoke(
                    [
                        ("system", SEED_ANSWER_SYSTEM),
                        (
                            "human",
                            "Excerpts:\n\n"
                            + "\n\n---\n\n".join(contexts)
                            + f"\n\nQuestion: {q}",
                        ),
                    ]
                )
                candidate = _json_from(out.content)["reference"].strip()
            except Exception as exc:  # noqa: BLE001
                print(f"    seed {i} ({category}) call failed at k={k}: {exc}")
                continue
            if not _is_refusal(candidate):
                reference = candidate
                break
            print(f"    seed {i} ({category}): refusal at k={k}, escalating")

        if not reference:
            unresolved.append(f"[{category}] {q}")
            continue

        items.append(
            TestItem(
                id=f"seed-{i:03d}",
                user_input=q,
                reference=reference,
                reference_contexts=[
                    f"{s.short_name} - {s.label}\n{s.text[:9000]}" for s in matched
                ],
                origin="seed",
                category=category,
                section_ids=[s.id for s in matched],
                needs_review=True,
            )
        )

    if unresolved:
        print(f"\n    {len(unresolved)} seed questions could not be grounded even at k=16:")
        for u in unresolved:
            print(f"      - {u[:100]}")
        print("    These are dropped. If many are multi_hop/cross_reg, the matcher needs work.")
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
