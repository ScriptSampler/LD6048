"""Extract every number from the main text of report/draft.md (from the Introduction to the
reference list) and match it to a value in results/ (allowing for rounding, percentage scaling,
complements, sign and monthly or annual rescaling). Writes results/check_numbers.json listing
matched and unmatched numbers.

Skipped: the title page, declaration and contents (they come before the Introduction), code blocks,
author-year citations (other text inside the same brackets is kept), years, Figure/Table/Part/
Snippet/Article/section numbers, page ranges, and the reference list and appendices. Every other
number must match a value in results/, or failing that be a whole number on the ALLOWED_CONSTANTS
list (brief and design constants) or a literature figure listed in report/cited_numbers.json. Files in
results/ that the pipeline does not compute from the data (this check's own output, hand-recorded
reviews, run records) and per-row files (split indices, predictions) are not searched.

This is an existence check. With thousands of values in results/, any two-decimal value in [0, 1],
any one-decimal percentage and almost any integer up to 100 match something, so a slip in such a
number passes; the check catches numbers that match nothing saved, such as most larger figures.
"""
from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RESULTS, ROOT
from .io_utils import save_json

DRAFT = ROOT / "report" / "draft.md"

# Whole numbers from the brief and the design rather than the data (word budgets, marks, counting numbers,
# seed, split, folds, CI level, cost model, search draws, the 14 coefficients of Figure 8, the brief's 11
# blank charges and 22 duplicates), accepted when no results file holds them. Data-derived values are
# deliberately not listed, so they must match results/; years and article numbers are stripped before
# matching.
ALLOWED_CONSTANTS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 20, 22, 25, 40, 42, 50, 60, 80, 95, 100,
                     400, 500, 700, 1000, 3000}

# Not searched: this check's own previous output (an unmatched number would match itself on the next
# run), hand-recorded files, run records that hold no results (stage timings, the machine, the report's
# own word counts), and per-row files whose thousands of values would match almost anything
NOT_SEARCHED = {"check_numbers.json", "reproducibility_check.json", "run_log.json", "environment.json",
                "word_counts.json", "split_indices.json", "oof_predictions.csv", "test_predictions.csv"}


def searched(p: Path) -> bool:
    return p.name not in NOT_SEARCHED and not p.name.startswith("rubric_review")


def result_files() -> list[Path]:
    return sorted(p for ext in ("*.json", "*.csv") for p in RESULTS.rglob(ext) if searched(p))


def collect_result_values() -> set[float]:
    return set().union(*map(file_values, result_files()))


def file_values(p: Path) -> set[float]:
    """Every finite number in one JSON or CSV results file."""
    vals: set[float] = set()

    def add(v):
        # numbers only: a numeric string (e.g. an earlier report's unmatched number) is not a result
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            return
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return
        vals.add(f)

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            add(o)

    if p.suffix == ".json":
        try:
            walk(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
        return vals
    try:
        df = pd.read_csv(p)
    except Exception:
        return vals
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        for v in s.values:
            add(v)
    return vals


def expand(vals: set[float]) -> dict[str, set[float]]:
    """Rounded representations at several precisions, plus x100 (percent), monthly and annual
    rescaling, sign, and complements (1-x) for values in [0, 1] (rates). Each value is rounded by
    Python's round(), which works on the binary value; the value itself, its negative and its
    percentage are also rounded half-up on their decimal form, so 64.425 gives 64.42 and 64.43."""
    out: dict[str, set[float]] = {}
    for v in vals:
        cands = [v, v * 100, v / 12, v * 12, -v] + ([1 - v, (1 - v) * 100] if 0 <= v <= 1 else [])
        for i, cand in enumerate(cands):
            half_up = i in (0, 1, 4) and abs(cand) < 1e15  # Decimal.quantize fails on huge values
            dec = Decimal(repr(cand)) if half_up else None
            for dp in (0, 1, 2, 3, 4):
                out.setdefault(f"{round(cand, dp):.{dp}f}", set()).add(v)
                if half_up:
                    out.setdefault(f"{dec.quantize(Decimal(1).scaleb(-dp), ROUND_HALF_UP):.{dp}f}", set()).add(v)
    return out


def _drop_citations(m: re.Match) -> str:
    """Remove the author-year citations from one bracket, keeping any other text in it, so that
    '(Cramér's V 0.27; Barocas, Hardt and Narayanan, 2023)' keeps its number."""
    kept = [s for s in m.group(1).split(";") if not re.fullmatch(r"[^\d]*,\s*(?:\d{4}[a-z]?|no date)\s*", s)]
    return "(" + ";".join(kept) + ")" if kept else " "


def numbers_in_draft(text: str) -> list[tuple[str, str]]:
    # Main text only: from the Introduction (after the title page, declaration and contents) to the
    # reference list; then drop code blocks, citations "(Author, 2018)" and "et al. (2020)", labels, years
    start = re.search(r"^#\s*Introduction\b", text, flags=re.M)
    body = re.split(r"\n#+\s*(References|Reference list)[^\n]*\n", text[start.start() if start else 0:])[0]
    body = re.sub(r"```.*?```", " ", body, flags=re.S)
    body = re.sub(r"\(([^()]*?\d{4}[^()]*)\)", _drop_citations, body)  # parenthetical citations
    body = re.sub(r"\b(?:et al\.|and [A-Z][a-z]+|[A-Z][A-Za-z'\-]+)\s*\(\d{4}[a-z]?\)", " ", body)  # narrative citations
    body = re.sub(r"\bSI\s*\d{4}/\d+", " ", body)  # statutory instrument numbers
    body = re.sub(r"\b(Figure|Fig\.|Table|Part|Appendix|Snippet|Article|Articles|Art\.|Arts|Section|s\.|reg\.|regulation|Chapter)\s*\d+[A-Za-z]?(\s*[-–]\s*\d+)?", " ", body)
    body = re.sub(r"\b(19[89]\d|20[0-3]\d)\b", " ", body)  # years
    body = re.sub(r"\bpp?\.\s*\d+\s*[-–]\s*\d+", " ", body)
    body = re.sub(r"\b\d+(?:\.\d+)?\s*(pt|px|dpi|GB|MB)\b", " ", body)
    nums = []
    for m in re.finditer(r"(?<![\w.])[-−]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-−]?\d+(?:\.\d+)?", body):
        s = m.group(0)
        ctx = body[max(0, m.start() - 40): m.end() + 25].replace("\n", " ")
        nums.append((s, ctx))
    return nums


def cited_numbers() -> dict[str, list[dict]]:
    """Literature figures from report/cited_numbers.json, by number (one number can have several sources)."""
    p = ROOT / "report" / "cited_numbers.json"
    out: dict[str, list[dict]] = {}
    if p.exists():
        for it in json.loads(p.read_text(encoding="utf-8"))["items"]:
            out.setdefault(it["number"], []).append(it)
    return out


def run() -> dict:
    text = DRAFT.read_text(encoding="utf-8")
    vals = collect_result_values()
    table = expand(vals)
    cited = cited_numbers()
    matched, unmatched = [], []
    for s, ctx in numbers_in_draft(text):
        clean = s.replace(",", "").replace("−", "-")
        try:
            f = float(clean)
        except ValueError:
            continue
        dp = len(clean.split(".")[1]) if "." in clean else 0
        key = f"{round(f, dp):.{dp}f}"
        if key in table:
            matched.append({"number": s, "context": ctx, "source": "results/"})
        elif f in ALLOWED_CONSTANTS and dp == 0:
            matched.append({"number": s, "context": ctx, "source": "allowed constant"})
        elif clean in cited:
            matched.append({"number": s, "context": ctx, "source": "cited: " + cited[clean][0]["source"]})
        else:
            unmatched.append({"number": s, "context": ctx})
    out = {"n_numbers": len(matched) + len(unmatched), "n_matched": len(matched), "n_unmatched": len(unmatched),
           "unmatched": unmatched}
    save_json(out, "check_numbers.json")
    print(f"check_numbers: {len(matched)} matched, {len(unmatched)} unmatched")
    for u in unmatched:
        print("  UNMATCHED", u["number"], "|", u["context"])
    return out


if __name__ == "__main__":
    run()
