"""Rebuild every table, figure and number from the raw CSV in one command:

    python run_all.py            # full run (about 11 minutes with 16 logical processors)
    python run_all.py --quick    # smoke test with tiny search budgets (NOT for the report)

Stages: manifest -> clean -> EDA -> models -> evaluate -> explain -> fairness ->
derived numbers -> environment record -> report build (build_report.py) -> number check ->
TRACEABILITY.md number table.

A full run overwrites report/draft.md and report/draft.docx, so finish the submission copy in
Word separately (build_report.py moves aside a draft.docx that was edited since its last build).
Smoke tests (--quick or LD6048_QUICK=1) write to ld6048_quick in the system temp folder, or to
LD6048_OUT_ROOT if set, and neither they nor out-of-tree runs rebuild the report.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny search budgets for smoke tests")
    ap.add_argument("--skip-models", action="store_true",
                    help="rerun EDA and derived numbers and rebuild the report; models untouched")
    args = ap.parse_args()
    if args.quick:
        os.environ["LD6048_QUICK"] = "1"
    sys.path.insert(0, str(ROOT))
    from src import clean as clean_mod
    from src import derived, eda, evaluate, explain, fairness, load, models
    from src.config import OUT_ROOT, QUICK, RESULTS, SEED
    from src.io_utils import save_json

    t_all = time.time()
    log = {"seed": SEED, "quick": QUICK, "stages": []}

    def stage(name, fn):
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        out = fn()
        log["stages"].append({"stage": name, "seconds": round(time.time() - t0, 1)})
        return out

    raw, manifest = stage("manifest", load.build_manifest)
    df, levels = stage("clean", lambda: clean_mod.clean(raw))
    checks = clean_mod.consistency_checks(df)
    save_json({"levels": levels, "consistency": checks}, "cleaning_checks.json")
    stage("eda", lambda: eda.run(df))
    if not args.skip_models:
        stage("models", lambda: models.run(df))
        stage("evaluate", evaluate.run)
        stage("explain", explain.run)
        stage("fairness", fairness.run)
    stage("derived", derived.run)  # reads saved results only, so it also refreshes after --skip-models
    # Environment record
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    save_json({"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
               "processor": platform.processor(), "cpu_count": os.cpu_count(), "packages": freeze}, "environment.json")
    log["total_seconds"] = round(time.time() - t_all, 1)
    save_json(log, "run_log.json")
    # Rebuild the report from the template, then check its numbers; never from smoke-test results
    # (QUICK, set by --quick or LD6048_QUICK=1) or from an out-of-tree run, whose results/ the report does not read
    if QUICK or OUT_ROOT.resolve() != ROOT.resolve():
        print("\nReport not rebuilt: smoke-test or out-of-tree run.")
    else:
        subprocess.run([sys.executable, str(ROOT / "build_report.py")], check=True)
        from src import check_numbers, trace_numbers
        if check_numbers.run()["n_unmatched"]:
            sys.exit("Numbers in report/draft.md match nothing in results/ (see results/check_numbers.json)")
        trace_numbers.run()  # TRACEABILITY.md's number table
    print(f"\nDone in {log['total_seconds'] / 60:.1f} min. Results in {RESULTS}")


if __name__ == "__main__":
    main()
