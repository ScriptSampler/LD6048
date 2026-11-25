"""Generate a SYNTHETIC Telco-shaped CSV for smoke-testing the pipeline only.

It mimics the schema and quirks of WA_Fn-UseC_-Telco-Customer-Churn.csv (text
TotalCharges with blanks at tenure 0, 'No internet service' labels, 0/1
SeniorCitizen) so that the code paths can be exercised before the real file is
available. NOTHING produced from this file may be used in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 7043, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.59, 0.41]),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                     "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21]),
    })
    df["customerID"] = [f"{i:04d}-{c}" for i, c in enumerate(df["customerID"])]
    df["MultipleLines"] = np.where(df["PhoneService"] == "No", "No phone service",
                                   rng.choice(["Yes", "No"], n))
    for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[c] = np.where(df["InternetService"] == "No", "No internet service", rng.choice(["Yes", "No"], n))
    base = 20 + np.where(df["InternetService"] == "Fiber optic", 50, np.where(df["InternetService"] == "DSL", 25, 0))
    addons = sum((df[c] == "Yes").astype(int) for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"])
    df["MonthlyCharges"] = np.round(base + 5 * addons + rng.normal(0, 3, n), 2).clip(18, 120)
    tc = df["tenure"] * df["MonthlyCharges"] * rng.uniform(0.95, 1.05, n)
    # churn probability: month-to-month, fibre, short tenure, e-cheque raise it
    logit = -2.2 + 1.6 * (df["Contract"] == "Month-to-month") + 0.7 * (df["InternetService"] == "Fiber optic") \
        - 0.03 * df["tenure"] + 0.5 * (df["PaymentMethod"] == "Electronic check") + 0.3 * df["SeniorCitizen"] \
        - 0.5 * (df["OnlineSecurity"] == "Yes") - 0.4 * (df["TechSupport"] == "Yes")
    p = 1 / (1 + np.exp(-logit))
    churn = rng.uniform(size=n) < p
    df["Churn"] = np.where(churn, "Yes", "No")
    # tenure-0 rows: blank TotalCharges and Churn No
    zero = df["tenure"] == 0
    df.loc[zero, "Churn"] = "No"
    tc_str = pd.Series(np.round(tc, 2).astype(str), index=df.index)
    tc_str[zero] = " "
    df["TotalCharges"] = tc_str
    cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
    return df[cols]


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("synthetic")
    out.mkdir(parents=True, exist_ok=True)
    make().to_csv(out / "WA_Fn-UseC_-Telco-Customer-Churn.csv", index=False)
    print("wrote", out / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
