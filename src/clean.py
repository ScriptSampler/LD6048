"""Deterministic cleaning that learns nothing from the data (so it is safe to
apply before the train/test split): type fixes, whitespace, target coding.

Anything that is *fitted* (imputation statistics, scaling, encoding, resampling)
lives inside the modelling Pipeline in features.py, never here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (ADDONS, ALL_CATEGORICAL, ID_COL, MULTILINES, NUMERIC,
                     TARGET)
from .io_utils import save_csv

# Pre-processing decision log: issue -> action -> justification -> alternative
# rejected -> evidence. Evidence fields that depend on CV ablations are filled in
# by models.py, which appends rows with the measured numbers.
DECISION_LOG: list[dict] = []


def log_decision(issue, action, justification, alternative, evidence):
    DECISION_LOG.append({"issue": issue, "action": action, "justification": justification,
                         "alternative_rejected": alternative, "evidence": evidence})


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges arrives as text; 11 rows are blank (tenure 0, never billed)."""
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    return df


def clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Return an analysis-ready frame with Churn as 0/1. No fitted operations."""
    df = df_raw.copy()
    # 1. Strip whitespace from every text column and check spellings.
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].str.strip()
    # 2. TotalCharges to numeric; blanks become NaN here and 0 inside the pipeline.
    df = fix_total_charges(df)
    n_blank = int(df["TotalCharges"].isna().sum())
    log_decision(
        "TotalCharges stored as text; %d blank values" % n_blank,
        "Convert with pd.to_numeric(errors='coerce'); impute 0 inside the Pipeline (SimpleImputer, constant)",
        "Blanks are structurally missing: tenure = 0, so the customer has not yet been billed; 0 is the true value",
        "Drop the rows (loses a coherent tenure-0 group, all non-churners); mean/median imputation (assigns ~$1,400 of charges to unbilled customers)",
        "All %d blanks have tenure == 0 and Churn == No (data_manifest.json)" % n_blank,
    )
    # 3. Target to 0/1.
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    # 4. SeniorCitizen is already 0/1; keep it numeric but treat it as categorical
    #    in EDA. Make it int for safety.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    log_decision(
        "SeniorCitizen coded 0/1 while other binaries are Yes/No",
        "Keep as integer 0/1 (equivalent to a Yes/No dummy); label as 'Senior citizen' in figures",
        "A 0/1 integer is already the encoding a one-hot binary would produce",
        "Recode to Yes/No then one-hot (identical design matrix, extra step)",
        "Values {0,1} only (data_manifest.json)",
    )
    # 5. Category spellings: assert the expected level sets.
    levels = {c: sorted(df[c].astype(str).unique().tolist()) for c in ALL_CATEGORICAL}
    return df, levels


def consistency_checks(df: pd.DataFrame) -> dict:
    """TotalCharges vs tenure x MonthlyCharges; IQR outlier screen."""
    out = {}
    expected = df["tenure"] * df["MonthlyCharges"]
    ratio = (df["TotalCharges"] / expected).replace([np.inf, -np.inf], np.nan)
    out["ratio_median"] = float(ratio.median())
    out["ratio_q05"] = float(ratio.quantile(0.05))
    out["ratio_q95"] = float(ratio.quantile(0.95))
    out["ratio_min"] = float(ratio.min())
    out["ratio_max"] = float(ratio.max())
    out["n_ratio_outside_0.8_1.2"] = int(((ratio < 0.8) | (ratio > 1.2)).sum())
    out["r_tenure_totalcharges"] = float(df[["tenure", "TotalCharges"]].dropna().corr().iloc[0, 1])
    out["abs_diff_median"] = float((df["TotalCharges"] - expected).abs().median())
    for c in NUMERIC:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out[f"{c}_iqr_outliers"] = int(((s < lo) | (s > hi)).sum())
        out[f"{c}_min"], out[f"{c}_max"] = float(s.min()), float(s.max())
        out[f"{c}_skew"] = float(s.skew())
    out["n_attribute_duplicates"] = int(df.drop(columns=[ID_COL]).duplicated().sum())
    out["n_full_duplicates_incl_id"] = int(df.duplicated().sum())
    return out


def write_decision_log() -> None:
    save_csv(pd.DataFrame(DECISION_LOG), "preprocessing_log.csv")


if __name__ == "__main__":
    from .load import build_manifest
    raw, _ = build_manifest()
    df, levels = clean(raw)
    print(consistency_checks(df))
    print(levels)
