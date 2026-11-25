"""Compare two full runs of `python run_all.py` and write results/reproducibility_check.json.

Copy results/ and figures/ aside after the first run, run again, then:

    python -m src.compare_runs <first run's results folder> <first run's figures folder>

Files are compared byte for byte; timing files are skipped, and CSV files that differ only in timing
columns are listed separately. derived.json is compared key by key, ignoring timing keys. Figure PNGs
are compared byte for byte (SVGs carry a creation date, so they always differ).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from .config import FIGURES, RESULTS

TIMING_FILES = ["check_numbers.json", "compute_footprint.csv", "compute_footprint_total.json", "derived.json",
                "environment.json", "reproducibility_check.json", "run_log.json", "word_counts.json"]
TIMING_COLS = {"seconds", "fit_seconds", "fit_seconds_mean"}
TIMING_KEYS = {"co2e_grams", "total_modelling_minutes", "fit_time_ratio_rf_vs_lr", "fit_time_ratio_xgb_vs_lr",
               "nested_cv_seconds_rf", "nested_cv_seconds_lr", "fit_seconds_best_tuned"}
# examiner records kept in results/ but written by no pipeline stage
STATIC = {"rubric_review_round1.json", "rubric_review_round2.json"}


def same_but_timing(a: Path, b: Path) -> bool:
    x, y = pd.read_csv(a), pd.read_csv(b)
    if list(x.columns) != list(y.columns):
        return False
    keep = [c for c in x.columns if c not in TIMING_COLS]
    return x[keep].equals(y[keep])


def run(first_results: Path, first_figures: Path) -> dict:
    identical, static, different, timing_only, missing = [], [], [], [], []
    for p in sorted(RESULTS.glob("*")):
        if p.is_dir() or p.name in TIMING_FILES:
            continue
        q = first_results / p.name
        if not q.exists():
            missing.append(p.name)
        elif q.read_bytes() == p.read_bytes():
            (static if p.name in STATIC else identical).append(p.name)
        elif p.suffix == ".csv" and same_but_timing(q, p):
            timing_only.append(p.name)
        else:
            different.append(p.name)
    da = json.loads((first_results / "derived.json").read_text(encoding="utf-8"))
    db = json.loads((RESULTS / "derived.json").read_text(encoding="utf-8"))
    derived_diff = {k: [da.get(k), db.get(k)] for k in sorted(set(da) | set(db))
                    if k not in TIMING_KEYS and da.get(k) != db.get(k)}
    pngs = sorted(p.name for p in FIGURES.glob("*.png"))
    fig_diff = [n for n in pngs if not (first_figures / n).exists()
                or (first_figures / n).read_bytes() != (FIGURES / n).read_bytes()]
    out = {
        "identical": identical, "static_records_identical": static, "different": different,
        "different_only_in_timing_columns": timing_only, "timing_files_skipped": TIMING_FILES,
        "missing_in_first_run": missing, "derived_non_timing_differences": derived_diff,
        "figures_png_identical": len(pngs) - len(fig_diff), "figures_png_different": fig_diff, "n_figures_png": len(pngs),
        "summary": (f"{len(identical)} regenerated files identical (plus {len(static)} static records), {len(timing_only)} differ "
                    f"only in timing columns, {len(different)} different, {len(missing)} missing in the first run; derived.json "
                    + ("differs only in timing keys" if not derived_diff else f"differs in {len(derived_diff)} non-timing keys")
                    + f"; {len(pngs) - len(fig_diff)} of {len(pngs)} figure PNGs identical"),
    }
    (RESULTS / "reproducibility_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(run(Path(sys.argv[1]), Path(sys.argv[2]))["summary"])
