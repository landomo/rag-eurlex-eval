"""Download and normalise the EUR-Lex corpus, then segment it into legal sections.

Two outputs:
  data/raw/<key>.txt        normalised plain text of each regulation
  data/processed/sections.jsonl   one record per Recital block / Article

The section split is what the `structural_article` chunker consumes. The other
chunkers deliberately work on the *flat* document text, so the ablation actually
measures whether structure-awareness buys anything.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import CORPUS, SECTIONS_PATH, Regulation

UA = "ragbench/0.1 (research; contact: your-email@example.com)"

# "Article 5" / "Article 6a" headings sitting on their own line.
ARTICLE_RE = re.compile(r"^\s*Article\s+(\d+[a-z]?)\s*$", re.MULTILINE)
ANNEX_RE = re.compile(r"^\s*ANNEX\s+([IVXLC]+|\d+)\s*$", re.MULTILINE)


@dataclass
class Section:
    id: str
    source: str          # regulation key
    short_name: str
    kind: str            # "preamble" | "article" | "annex"
    label: str           # "Article 6" / "Recitals" / "Annex III"
    heading: str         # article title where recoverable
    text: str

    @property
    def citation(self) -> str:
        return f"{self.short_name}, {self.label}"


# --------------------------------------------------------------------------- fetch

def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def fetch(reg: Regulation, force: bool = False) -> str:
    """Fetch one regulation. Falls back to a manually placed file if the network fails.

    EUR-Lex is a public, open-data source; be polite and cache locally (we do).
    """
    if reg.raw_path.exists() and not force:
        return reg.raw_path.read_text(encoding="utf-8")

    import urllib.error
    import urllib.request

    req = urllib.request.Request(reg.url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover
        raise SystemExit(
            f"\nCould not download {reg.short_name} from EUR-Lex ({exc}).\n"
            f"Manual fallback:\n"
            f"  1. Open {reg.url}\n"
            f"  2. Save the page text as {reg.raw_path}\n"
            f"  3. Re-run this script.\n"
        ) from exc

    text = _html_to_text(html)
    if len(text) < 20_000:
        raise SystemExit(
            f"{reg.short_name}: downloaded page looks too short ({len(text)} chars). "
            f"EUR-Lex may have changed its layout - check {reg.url} manually."
        )
    reg.raw_path.write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------------- segment

def _heading_after(text: str, end: int) -> str:
    """First non-empty line *after* an 'Article N' heading is usually its title.

    `end` must be the end offset of the heading match, not its start - the regex
    consumes leading whitespace, so slicing from the start would return the
    heading itself.
    """
    for line in text[end : end + 400].splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:200]
    return ""


def segment(text: str, reg: Regulation) -> list[Section]:
    sections: list[Section] = []
    matches = list(ARTICLE_RE.finditer(text))

    if not matches:
        raise SystemExit(
            f"{reg.short_name}: no 'Article N' headings found - the text layout is "
            "unexpected. Inspect data/raw/ and adjust ARTICLE_RE."
        )

    # Everything before Article 1 is the preamble (citations + recitals).
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(
            Section(
                id=f"{reg.key}:preamble",
                source=reg.key,
                short_name=reg.short_name,
                kind="preamble",
                label="Recitals",
                heading="Recitals and preamble",
                text=preamble,
            )
        )

    annex_start = None
    annex_match = ANNEX_RE.search(text, matches[-1].end())
    if annex_match:
        annex_start = annex_match.start()

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else (annex_start or len(text))
        body = text[start:end].strip()
        if len(body) < 60:  # table-of-contents echo, not the real article
            continue
        num = m.group(1)
        sections.append(
            Section(
                id=f"{reg.key}:art{num}",
                source=reg.key,
                short_name=reg.short_name,
                kind="article",
                label=f"Article {num}",
                heading=_heading_after(text, m.end()),
                text=body,
            )
        )

    if annex_start is not None:
        annexes = list(ANNEX_RE.finditer(text, annex_start))
        for i, m in enumerate(annexes):
            start = m.start()
            end = annexes[i + 1].start() if i + 1 < len(annexes) else len(text)
            body = text[start:end].strip()
            if len(body) < 100:  # table-of-contents echo, not the real annex
                continue
            sections.append(
                Section(
                    id=f"{reg.key}:annex{m.group(1)}",
                    source=reg.key,
                    short_name=reg.short_name,
                    kind="annex",
                    label=f"Annex {m.group(1)}",
                    heading="",
                    text=body,
                )
            )

    # EUR-Lex pages repeat the article list in a table of contents; the real body is
    # always the longest instance of a given label. Keep the longest per id.
    best: dict[str, Section] = {}
    for s in sections:
        if s.id not in best or len(s.text) > len(best[s.id].text):
            best[s.id] = s
    return list(best.values())


def build(force: bool = False) -> list[Section]:
    all_sections: list[Section] = []
    for reg in CORPUS:
        text = fetch(reg, force=force)
        secs = segment(text, reg)
        print(f"  {reg.short_name:<10} {len(text):>9,} chars -> {len(secs):>4} sections")
        all_sections.extend(secs)

    with SECTIONS_PATH.open("w", encoding="utf-8") as fh:
        for s in all_sections:
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return all_sections


def load_sections() -> list[Section]:
    if not SECTIONS_PATH.exists():
        raise SystemExit("No sections found. Run: python scripts/01_ingest.py")
    with SECTIONS_PATH.open(encoding="utf-8") as fh:
        return [Section(**json.loads(line)) for line in fh if line.strip()]
