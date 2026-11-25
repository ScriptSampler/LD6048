"""Emit a Harvard reference list (alphabetical) for the keys cited in the draft,
from references/references.json, and check that every in-text citation in
report/draft.md has an entry and every entry is cited.

    python -m src.reference_list            # prints the list for the keys in CITED
    python -m src.reference_list --check    # cross-checks draft.md citations

Keys map to references/verified/<key>.json. Legal sources are formatted from the
legal items' primary_sources (selected by index) because one legal item may hold
several documents.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFS = json.loads((ROOT / "references" / "references.json").read_text(encoding="utf-8"))
BY_KEY = {r["key"]: r for r in REFS["references"]}
LEGAL = {l["key"]: l for l in REFS["legal"]}


def harvard(key: str) -> str:
    r = BY_KEY[key]
    s = r.get("harvard_reference", "").strip()
    if key == "ibm_telco_dataset":
        s = s.split("\n")[0].strip()
    return s


def legal_source(key: str, idx: int) -> str:
    return LEGAL[key]["primary_sources"][idx]["harvard_reference"].strip()


def author_year_from_harvard(s: str) -> str:
    """Return e.g. 'Ascarza (2018)' for matching in-text citations."""
    m = re.match(r"^(.*?)\s*\((\d{4}[a-z]?|no date)\)", s)
    return f"{m.group(1).strip()} ({m.group(2)})" if m else s[:40]


if __name__ == "__main__":
    if "--check" in sys.argv:
        draft = (ROOT / "report" / "draft.md").read_text(encoding="utf-8")
        body, _, refsec = draft.partition("\n# References")
        refsec = refsec.split("\n# ", 1)[0]  # stop at the next top-level heading (appendices)
        entries = [l.strip() for l in refsec.split("\n") if l.strip() and not l.startswith("#") and not l.startswith("<!--") and not l.startswith("{-}")]
        # crude author-year extraction from entries
        problems = []
        years_in_body = set(re.findall(r"\((?:[^()]*?,\s*)?(\d{4}[a-z]?)\)", body))
        for e in entries:
            m = re.match(r"^([A-Za-z][^(]*?)(?:\s*\(([A-Z]{2,6})\))?\s*\((\d{4}[a-z]?|no date)\)", e)
            if not m:
                # Legislation: match the short title (text before ', c.', ', SI' or ' (') in the body
                title = re.split(r",\s*c\.|,\s*SI\b|\s*\(|\.\s", e.replace("*", ""))[0].strip().rstrip(",")
                abbrs = re.findall(r"\(([A-Z][A-Z ]{1,12})\)", e)
                if title and title not in body and not any(a in body for a in abbrs):
                    problems.append(f"Legislation entry not cited in body: {e[:80]}")
                continue
            author, abbr, year = m.group(1).strip(), m.group(2), m.group(3)
            first_author = re.split(r",| and | et al", author)[0].strip()
            surname = first_author.split()[0].strip("'")
            candidates = [surname] + ([abbr] if abbr else [])
            ok = any(re.search(re.escape(c) + r".{0,80}?" + re.escape(year), body, flags=re.S) for c in candidates)
            if not ok:
                problems.append(f"Entry not cited in body: {e[:90]}")
        # Reverse check: every (Author, year) or Author (year) citation in the body has an entry
        cited = set(re.findall(r"([A-Z][A-Za-z'\-]+(?: et al\.)?)[,]?\s*\(?(\d{4}[a-z]?|no date)\)?", body))
        entry_text = " ".join(entries)
        for name, year in sorted(cited):
            name = name.replace("'s", "")
            if name in ("Figure", "Table", "Part", "Article", "Articles", "Snippet", "Appendix", "Act", "Regulations", "February", "LD", "Since", "SI"):
                continue
            if not re.search(re.escape(name.replace(" et al.", "")) + r".{0,120}?\(" + re.escape(year) + r"\)", entry_text, flags=re.S):
                problems.append(f"Citation without entry: {name} ({year})")
        print(f"{len(entries)} reference entries; {len(problems)} problems")
        for p in problems:
            print("  ", p)
    else:
        for k in sys.argv[1:]:
            print(harvard(k))
