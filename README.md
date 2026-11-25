# LD6048 Programming using AI: customer churn prediction (Telco Customer Churn)

Reproducible analysis and report for the LD6048 assessment (Northumbria University).
One command regenerates every analysis result, figure and data table from the raw CSV and rebuilds the
report (`report/draft.md`, `report/draft.docx`); the remaining numbers in the text were transcribed, and
`src/check_numbers.py` checks that each number in the main text (code and citations excluded) matches a value
saved in `results/`, a constant on its allow-list or a literature figure listed in `report/cited_numbers.json`.

`report/draft.md` is the report text as the pipeline builds it; the file name is historical. GitHub's preview of it
shows the figure captions without the images, because the image links are resolved from the repository root when
the Word file is built; the images are in `figures/`. Its title page and declaration page hold placeholders, which
are filled in only in the copy finished for submission (step 6).

Third-party material (the dataset, two adapted functions and quoted public-sector text) is listed with its licences
in `THIRD_PARTY_NOTICES.md`.

## Reproduction

Run every command from the repository root.

1. **Python 3.11** and a virtual environment.

   Windows (PowerShell or Command Prompt):

   ```
   py -3.11 -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

   macOS or Linux:

   ```
   python3.11 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

   The later steps are written for Windows. On macOS and Linux write `.venv/bin/python` wherever a step says
   `.venv\Scripts\python`; in Git Bash on Windows write `.venv/Scripts/python`.

   Platforms: tested only on Windows x64 with Python 3.11.9 (`results/environment.json`). The pinned versions do
   not install on Intel Macs, where shap 0.51.0 needs older numba and llvmlite than the pinned 0.67.0 and 0.49.0.
   On macOS, XGBoost needs the OpenMP runtime first (`brew install libomp`); on Linux, pip also downloads the
   unpinned `nvidia-nccl-cu12` that XGBoost 3.2.0 requires there.

2. **Data.** The raw file is included: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (970,457 bytes). It is IBM's
   sample file, copied unchanged from IBM's public repository <https://github.com/IBM/telco-customer-churn-on-icp4d>
   (`data/Telco-Customer-Churn.csv`, Apache License 2.0; see `THIRD_PARTY_NOTICES.md`); `results/data_manifest.json`
   records where it was obtained and its SHA-256. The same table is listed on Kaggle at
   <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>. Kaggle reports 977,501 bytes for its copy, one
   byte more per line (consistent with Windows line endings), so a Kaggle download will not match the recorded
   SHA-256 even though it should hold the same table. Do not use IBM's extended version; its Churn Score / Churn
   Reason columns leak the target. `src/load.py` stops if the file differs from the expected fingerprint
   (7,043 x 21, 1,869 churners, 11 blank TotalCharges).

3. **Run everything** (seed 42; about 11 minutes on a machine with 16 logical processors):

   ```
   .venv\Scripts\python run_all.py
   ```

   A full run ends by rebuilding the report (step 4) and checking its numbers (step 5), and stops with an
   error if a number in the draft matches nothing in `results/`. `run_all.py --quick` (or `LD6048_QUICK=1`) is a
   smoke test with tiny search budgets: it writes to `ld6048_quick` in the system temp folder, or to
   `LD6048_OUT_ROOT` if that is set, never to `results/` or `figures/`, and like any out-of-tree run it does not
   rebuild the report. `build_report.py` also refuses to build from smoke-test results found in `results/`.
   `run_all.py --skip-models` reuses `data/processed/`, which only a full run creates, so in a fresh clone run the
   full pipeline first (if it was run too early, `git restore results figures` undoes it).

   A rerun changes tracked files by design: the creation date in every `figures/*.svg`, the timing columns and
   timing files in `results/`, and the timing values written into `report/draft.md` and `TRACEABILITY.md` (minutes,
   g CO2e, the fit-time ratio and the fit-seconds column of Table C1). Two runs on the same machine gave otherwise
   identical results (`results/reproducibility_check.json`). The committed results come from Windows x64, Python
   3.11.9 and 16 logical processors. On another processor, operating system or core count low-order digits may
   differ; if a number written into the text then matches nothing in `results/`, the run ends with an error after
   all results are saved. The committed text files were written on Windows (CRLF) and are stored byte for byte, so
   after a run on macOS or Linux Git lists every regenerated text file as modified; `git diff --ignore-cr-at-eol`
   shows the real changes.

4. **Rebuild only the report** (for example after editing `report/draft_template.md`). `build_report.py` renders the template (prose) into `report/draft.md`, filling the data tables from `results/` (`src/make_tables.py`), some computed values from `results/derived.json` and the reference list from `report/references.md`, then builds `report/draft.docx` with pandoc and records both word counts in `results/word_counts.json`:

   ```
   .venv\Scripts\python build_report.py
   ```

   Every build (and so every full run) overwrites `report/draft.md` and writes the Word files `report/draft.docx`
   and `report/reference.docx`, which are build outputs and are not in the repository;
   page layout and styles are set in `make_reference_docx()`. If `draft.docx` was changed after its last build,
   for example edited in Word, the build first moves that version to `report/draft.edited-<date-time>.docx`;
   if Word has `draft.docx` open, the build stops and asks you to close it.

5. **Verify.** `.venv\Scripts\python -m src.check_numbers` checks that every number from the Introduction to the end of Part 5, outside code and citations, matches a value in `results/` after rounding, or failing that is one of the brief and design constants on its allow-list or a literature figure listed in `report/cited_numbers.json`; it is an existence check: it cannot tell whether a matching number describes the right quantity, and small numbers match some saved value by chance. `.venv\Scripts\python -m src.reference_list --check` cross-checks citations against the reference list (the count it prints is one more than the 62 entries, because it also counts the `{-}` marker of the heading), and `.venv\Scripts\python -m src.trace_numbers` regenerates the number table of `TRACEABILITY.md` (a file listed there holds a matching value, which is not always the number's source). A full run does the number check and the traceability table itself; run the citation check by hand.

6. **Submission copy (author only).** The Word file for submission is finished by hand in a separate copy, never in
   `report/draft.docx`, `report/draft.md` or `report/draft_template.md`: the contents are updated (select the table
   and press F9, or References > Update Table) and the title-page details are filled in. That copy holds personal
   details and is not in this repository.

## Layout

| Path | Contents |
|---|---|
| `src/config.py` | seed, paths, column groups, CV budgets, illustrative cost model, palette |
| `src/load.py` | raw load, SHA-256 manifest, checks against the expected fingerprint |
| `src/clean.py` | deterministic cleaning (types, whitespace, target coding), decision log |
| `src/features.py` | feature engineering and leak-free `ColumnTransformer` / `Pipeline` builders |
| `src/eda.py` | Part 2 analysis: Wilson CIs, Cramér's V, mutual information, Mann-Whitney, Kaplan-Meier, Cox, segment heatmaps, opportunity segments, figures 1-3 |
| `src/models.py` | split, pre-processing ablations (training split only; XGBoost retuned for each engineered-feature set), VIF, XGBoost's best inner-CV PR-AUC by depth, nested 5x2 CV with inner randomised search, final tuning, OOF predictions, calibration choice, soft-voting ensemble |
| `src/evaluate.py` | thresholds (F1, F2, expected profit) chosen on OOF predictions, bootstrap CIs, corrected resampled t-test, DeLong, McNemar, calibration, error analysis, learning curve, figures 4-7 |
| `src/explain.py` | odds ratios with bootstrap CIs (scoring model, and an interpretation model refitted without `MonthlyCharges`, with its VIFs), permutation importance, SHAP, agreement between views, figure 8 |
| `src/fairness.py` | fairlearn `MetricFrame` audit by SeniorCitizen, gender, Partner, Dependents, figure 9; the selected model refitted without gender, and without gender and senior status (Table C7) |
| `src/derived.py` | quantities derived from other results (break-even save rates, oracle share, fit-time ratios, the dependents' and partners' recall-parity thresholds and costs, the dependents' threshold scan, paired-bootstrap profit gaps between models, per-customer expected-value targeting against the single threshold, the model's gain over the best segment rule with a paired bootstrap interval, corrected t-tests of the engineered-feature gains, calibration gaps) |
| `src/metrics.py` | metric helpers: point metrics, bootstrap CIs, DeLong, corrected resampled t-test, calibration summaries, expected-profit curve |
| `src/figstyle.py`, `src/io_utils.py` | the figure theme, and helpers for writing results consistently |
| `src/make_tables.py` | renders Tables 2, 3, 5–7 and every appendix table from `results/` (Tables 1 and 4 are written in the template) |
| `src/check_numbers.py` | checks that each number in the main text exists in `results/` (fallbacks: an allow-list of brief and design constants, and literature numbers in `report/cited_numbers.json`) |
| `src/compare_runs.py` | compares two full runs and writes `results/reproducibility_check.json` |
| `src/trace_numbers.py` | regenerates `TRACEABILITY.md`'s table of every number in the main text and the results files that hold it |
| `src/reference_list.py` | cross-checks in-text citations against the reference list |
| `src/build_bibliography.py` | assembles `references/annotated_bibliography.md` and `references/references.json` from `references/verified/` |
| `tests/make_synthetic.py` | writes a synthetic file with the raw data's layout for smoke tests (to `synthetic/`); nothing from it is used in the report |
| `run_all.py` | runs every stage in order, then the report build, the number check and the `TRACEABILITY.md` table |
| `build_report.py` | renders the template, counts words (Word-style and strict), builds the docx with pandoc and sets its pagination (captions kept with tables, rows never split) |
| `data/raw/` | the raw CSV (included) |
| `data/processed/` | train/test split and fitted pipelines, created by `run_all.py`; not committed |
| `results/` | every computed number the report uses (JSON/CSV) |
| `figures/` | PNG (300 dpi) and SVG |
| `report/` | `draft_template.md` (edit this), `draft.md` (rebuilt by every build), `references.md`, `cited_numbers.json`; the Word files `draft.docx` and `reference.docx` are built locally and not committed |
| `references/` | `annotated_bibliography.md`, `references.json`, `verified/*.json` (machine-verified metadata) |
| `TRACEABILITY.md` | every number in the main text with the results files that hold a matching value (regenerated by every full run) |
| `requirements.txt` | the pinned package versions |
| `THIRD_PARTY_NOTICES.md`, `LICENSES/` | third-party material and its licences |

Not in this repository: the built Word files, the copy finished for submission, the university's assignment brief
and the author's working notes and records.

## Design rules

- Split first (stratified 80/20, seed 42). Every fitted step (imputation, scaling, encoding,
  resampling, calibration) lives inside a scikit-learn / imbalanced-learn `Pipeline`; every
  data-driven modelling decision is made on the training split by cross-validation.
- The hold-out test set plays no part in any choice: it is used only after model selection, for the final
  evaluation and the post-selection analyses (permutation importance, SHAP, error analysis, fairness audit).
- Figures use one theme and the colour-blind-safe Okabe-Ito palette.
