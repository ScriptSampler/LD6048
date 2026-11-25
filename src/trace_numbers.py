"""Regenerate the number table of TRACEABILITY.md: every number in the main text of report/draft.md
with the results/ files that hold a matching value, using the same extraction and matching as
src/check_numbers.py. A literature figure is credited to its source when its report/cited_numbers.json
entry's 'match' phrase appears in the number's context; a number that only the allow-list of brief and
design constants accepts is marked as such. The hand-written sections above the table are kept.

    python -m src.trace_numbers
"""
from __future__ import annotations

from .check_numbers import (ALLOWED_CONSTANTS, DRAFT, cited_numbers, expand, file_values, numbers_in_draft,
                            result_files)
from .config import ROOT

TRACE = ROOT / "TRACEABILITY.md"
HEADING = "## Numbers in the draft body"


def run() -> int:
    keys = {p.name: set(expand(file_values(p))) for p in result_files()}
    cited = cited_numbers()
    rows = []
    for s, ctx in numbers_in_draft(DRAFT.read_text(encoding="utf-8")):
        clean = s.replace(",", "").replace("−", "-")
        dp = len(clean.split(".")[1]) if "." in clean else 0
        key = f"{round(float(clean), dp):.{dp}f}"
        # derived.json (values computed for the text) first, then smaller files: a match in a file with few
        # values is less likely to be a coincidence
        files = sorted((name for name, k in keys.items() if key in k),
                       key=lambda name: (name != "derived.json", len(keys[name])))
        cite = next((it for it in cited.get(clean, []) if it.get("match") and it["match"] in ctx), None)
        if cite:  # a quoted literature figure, even if a results file happens to hold the same value
            source = "cited: " + cite["source"]
        elif files:
            source = ", ".join(files[:4]) + (f" (+{len(files) - 4} more)" if len(files) > 4 else "")
        elif float(clean) in ALLOWED_CONSTANTS and dp == 0:
            source = "allow-listed constant (no value in results/)"
        elif clean in cited:
            source = "cited: " + cited[clean][0]["source"]
        else:
            source = "UNMATCHED"
        rows.append(f"| {s} | {ctx.replace('|', '/')} | {source} |")
    kept = TRACE.read_text(encoding="utf-8").split(HEADING)[0]
    TRACE.write_text(kept + f"{HEADING} ({len(rows)})\n\n| Number | Context | Source |\n|---|---|---|\n"
                     + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"trace_numbers: {len(rows)} numbers written to {TRACE.name}")
    return len(rows)


if __name__ == "__main__":
    run()
