"""Load the raw Kaggle file, fingerprint it, and assert the facts stated in the
brief. Stops with a clear message if the file differs from what was expected."""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from .config import (EXPECTED, ID_COL, RAW_FILE, SKIP_FACT_CHECK, TARGET)
from .io_utils import save_json


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw() -> pd.DataFrame:
    if not RAW_FILE.exists():
        sys.exit(f"Raw file not found: {RAW_FILE}\n"
                 "Download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from "
                 "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
                 "and place it in data/raw/.")
    # Read TotalCharges as text so the 11 blanks are visible, not silently NaN.
    df = pd.read_csv(RAW_FILE, dtype={"TotalCharges": str})
    return df


def fact_check(df: pd.DataFrame) -> dict:
    """Recompute the facts listed in the brief and compare."""
    tc_blank = df["TotalCharges"].str.strip().eq("")
    dup = df.drop(columns=[ID_COL]).duplicated(keep="first")
    tc_num = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    ratio = (tc_num / (df["tenure"] * df["MonthlyCharges"])).replace([np.inf, -np.inf], np.nan)
    facts = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "customerID_unique": bool(df[ID_COL].is_unique),
        "n_churn_yes": int((df[TARGET] == "Yes").sum()),
        "churn_rate": float((df[TARGET] == "Yes").mean()),
        "n_blank_totalcharges": int(tc_blank.sum()),
        "blank_totalcharges_all_tenure0": bool((df.loc[tc_blank, "tenure"] == 0).all()),
        "blank_totalcharges_all_churn_no": bool((df.loc[tc_blank, TARGET] == "No").all()),
        "n_tenure0": int((df["tenure"] == 0).sum()),
        "seniorcitizen_values": sorted(df["SeniorCitizen"].unique().tolist()),
        "n_no_internet": int((df["InternetService"] == "No").sum()),
        "n_no_internet_service_label": int((df["OnlineSecurity"] == "No internet service").sum()),
        "n_no_phone": int((df["PhoneService"] == "No").sum()),
        "n_no_phone_service_label": int((df["MultipleLines"] == "No phone service").sum()),
        "n_attribute_duplicates": int(dup.sum()),
        "totalcharges_ratio_median": float(ratio.median()),
        "r_tenure_totalcharges": float(np.corrcoef(df["tenure"], tc_num.fillna(0))[0, 1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    mismatches = {k: (facts[k], v) for k, v in EXPECTED.items() if facts.get(k) != v}
    if not facts["customerID_unique"]:
        mismatches["customerID_unique"] = (False, True)
    if not facts["blank_totalcharges_all_tenure0"] or not facts["blank_totalcharges_all_churn_no"]:
        mismatches["blank_totalcharges_structure"] = ("differs", "tenure==0 & Churn==No")
    facts["mismatches"] = mismatches
    return facts


def build_manifest() -> tuple[pd.DataFrame, dict]:
    df = load_raw()
    facts = fact_check(df)
    manifest = {
        "file": RAW_FILE.name,
        "source": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        "obtained_from": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv "
                         "(IBM's public mirror of the same 21-column sample file; verified against the brief's fingerprint below)",
        "sha256": sha256(RAW_FILE),
        "bytes": RAW_FILE.stat().st_size,
        "columns": df.columns.tolist(),
        "facts": facts,
    }
    save_json(manifest, "data_manifest.json")
    if facts["mismatches"] and not SKIP_FACT_CHECK:
        sys.exit("Data facts differ from the brief; stopping. Mismatches "
                 f"(observed, expected): {facts['mismatches']}")
    return df, manifest


if __name__ == "__main__":
    _, m = build_manifest()
    print({k: v for k, v in m["facts"].items() if k != "dtypes"})
