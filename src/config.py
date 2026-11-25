"""Central configuration: paths, seed, column groups, palette, cost model.

Everything that another module might otherwise hard-code lives here so that the
whole pipeline is reproducible from one place.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

SEED = 42

# QUICK=1 shrinks search sizes for smoke tests; results are NOT for the report.
QUICK = os.environ.get("LD6048_QUICK", "0") == "1"

# ---------------------------------------------------------------------------
# Paths. LD6048_OUT_ROOT lets smoke tests write to a scratch directory so that
# synthetic data can never pollute results/ or figures/; a smoke test without it
# writes to ld6048_quick in the system temp folder, never into the repository.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(os.environ.get("LD6048_OUT_ROOT") or (Path(tempfile.gettempdir()) / "ld6048_quick" if QUICK else ROOT))
DATA_RAW = Path(os.environ.get("LD6048_DATA_RAW", ROOT / "data" / "raw"))
DATA_PROCESSED = OUT_ROOT / "data" / "processed"
RESULTS = OUT_ROOT / "results"
FIGURES = OUT_ROOT / "figures"
MODELS_DIR = DATA_PROCESSED / "models"
for _p in (DATA_PROCESSED, RESULTS, FIGURES, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

RAW_FILE = DATA_RAW / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# ---------------------------------------------------------------------------
# Expected facts about the raw file (from the brief / prompt). load.py asserts
# them and stops if they differ.
# ---------------------------------------------------------------------------
EXPECTED = {
    "n_rows": 7043,
    "n_cols": 21,
    "n_churn_yes": 1869,
    "n_blank_totalcharges": 11,
    "n_no_internet": 1526,
    "n_no_phone": 682,
    "n_attribute_duplicates": 22,
}
# The synthetic smoke-test file will not satisfy these; skip the assertion then.
SKIP_FACT_CHECK = os.environ.get("LD6048_SKIP_FACT_CHECK", "0") == "1"

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
TARGET = "Churn"
ID_COL = "customerID"
BINARY_YN = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
          "StreamingTV", "StreamingMovies"]
PROTECTION = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
MULTILINES = "MultipleLines"
NOMINAL = ["gender", "InternetService", "Contract", "PaymentMethod"]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
SENIOR = "SeniorCitizen"
SENSITIVE = ["SeniorCitizen", "gender", "Partner", "Dependents"]
ALL_CATEGORICAL = ["gender", "SeniorCitizen"] + BINARY_YN + [MULTILINES] + \
    ["InternetService"] + ADDONS + ["Contract", "PaperlessBilling", "PaymentMethod"]
ALL_CATEGORICAL = list(dict.fromkeys(ALL_CATEGORICAL))  # dedupe, keep order

# Reference levels for logistic-regression dummy coding (odds ratios are
# interpreted relative to these).
REFERENCE_LEVELS = {
    "gender": "Female",
    "InternetService": "DSL",
    "Contract": "Month-to-month",
    "PaymentMethod": "Bank transfer (automatic)",
    "contract_x_internet": "Month-to-month | DSL",
}

# ---------------------------------------------------------------------------
# Cross-validation / search budgets
# ---------------------------------------------------------------------------
OUTER_SPLITS = 5
OUTER_REPEATS = 1 if QUICK else 2
INNER_SPLITS = 3 if QUICK else 5
N_ITER = {"lr": 3 if QUICK else 25, "dt": 3 if QUICK else 25,
          "rf": 2 if QUICK else 20, "xgb": 2 if QUICK else 100,
          "mlp": 2 if QUICK else 60}
N_BOOT = 50 if QUICK else 1000
ABLATION_REPEATS = 1 if QUICK else 2
TUNING_METRIC = "average_precision"   # PR-AUC: ranking churners under imbalance
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Illustrative retention cost model (clearly labelled as illustrative in the
# report). Monetary units are those of the dataset (MonthlyCharges).
# ---------------------------------------------------------------------------
COST_MODEL = {
    "horizon_months": 12,      # value horizon for a saved customer
    "margin": 0.40,            # contribution margin on recurring revenue
    "offer_cost": 60.0,        # cost of the retention offer when accepted
    "contact_cost": 5.0,       # cost of making contact (call/agent time)
    "save_rate": 0.30,         # share of contacted would-be churners retained
    "leakage": 1.00,           # share of contacted non-churners who take the offer anyway
}
SENSITIVITY_SAVE_RATES = [0.15, 0.30, 0.45, 0.60]
SENSITIVITY_OFFER_COSTS = [30.0, 60.0, 90.0, 120.0]

# ---------------------------------------------------------------------------
# Okabe-Ito colour-blind-safe palette (fixed order; never cycled beyond 8)
# ---------------------------------------------------------------------------
OKABE_ITO = {
    "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "purple": "#CC79A7", "black": "#000000", "grey": "#999999",
}
PALETTE = [OKABE_ITO[k] for k in ("blue", "orange", "green", "vermillion",
                                  "purple", "skyblue", "yellow", "black")]
CHURN_COLOURS = {0: OKABE_ITO["blue"], 1: OKABE_ITO["vermillion"]}
MODEL_ORDER = ["Majority", "LogReg", "DecisionTree", "RandomForest", "XGBoost", "MLP"]
MODEL_LABELS = {"Majority": "Majority class", "LogReg": "Logistic regression",
                "DecisionTree": "Decision tree", "RandomForest": "Random forest",
                "XGBoost": "XGBoost", "MLP": "MLP", "Voting": "Soft-voting ensemble"}
MODEL_COLOURS = dict(zip(MODEL_ORDER + ["Voting"], PALETTE[:7]))
