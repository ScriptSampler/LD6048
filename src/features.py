"""Feature engineering and the ColumnTransformer / Pipeline builders.

Every fitted step (imputation, scaling, encoding, resampling) is inside the
Pipeline so that cross-validation and the final hold-out evaluation are
leak-free. The FeatureEngineer transformer is deterministic (row-wise) but is
kept inside the Pipeline too, so ablations can switch it on and off.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

from .config import (ADDONS, BINARY_YN, MULTILINES, NOMINAL, NUMERIC,
                     PROTECTION, REFERENCE_LEVELS)

CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
ENGINEERED = ["n_addons", "any_protection", "auto_pay", "avg_monthly_spend",
              "price_change", "contract_x_internet"]
# Candidates per model family. n_addons, any_protection, auto_pay and
# price_change are exact linear combinations of columns already in the linear
# design (add-on dummies, payment dummies, MonthlyCharges - avg spend), so they
# cannot add information to a logistic regression and would make its design
# matrix rank-deficient; they are tested for the tree/MLP family only.
LINEAR_CANDIDATES = ["avg_monthly_spend", "contract_x_internet"]
TREE_CANDIDATES = ["n_addons", "any_protection", "auto_pay", "avg_monthly_spend", "price_change"]


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features (no statistics learned from other rows)."""
    X = X.copy()
    X["n_addons"] = (X[ADDONS] == "Yes").sum(axis=1)
    X["any_protection"] = ((X[PROTECTION] == "Yes").any(axis=1)).astype(int)
    X["auto_pay"] = X["PaymentMethod"].str.contains("automatic").astype(int)
    tc = X["TotalCharges"].fillna(0.0)
    X["avg_monthly_spend"] = np.where(X["tenure"] > 0, tc / X["tenure"].clip(lower=1),
                                      X["MonthlyCharges"])
    X["price_change"] = X["MonthlyCharges"] - X["avg_monthly_spend"]
    X["contract_x_internet"] = X["Contract"] + " | " + X["InternetService"]
    return X


def log1p_column(A, idx: int):
    """log1p one column of a numeric array (module-level so pipelines pickle)."""
    A = np.asarray(A, dtype=float).copy()
    A[:, idx] = np.log1p(np.clip(A[:, idx], 0, None))
    return A


def collapse_redundant(X: pd.DataFrame) -> pd.DataFrame:
    """'No internet service' / 'No phone service' duplicate InternetService /
    PhoneService; collapse them to 'No' so the dummies are not redundant."""
    X = X.copy()
    for c in ADDONS:
        X[c] = X[c].replace({"No internet service": "No"})
    X[MULTILINES] = X[MULTILINES].replace({"No phone service": "No"})
    return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Pipeline-friendly wrapper so ablations can toggle each engineered
    feature and the redundant-level collapse."""

    def __init__(self, features=None, collapse=True):
        self.features = features
        self.collapse = collapse

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if self.collapse:
            X = collapse_redundant(X)
        feats = ENGINEERED if self.features is None else list(self.features)
        if feats:
            X = engineer_features(X)
            drop = [f for f in ENGINEERED if f not in feats]
            X = X.drop(columns=drop)
        return X

    def get_feature_names_out(self, input_features=None):
        feats = ENGINEERED if self.features is None else list(self.features)
        return np.asarray(list(input_features) + feats, dtype=object)


def build_preprocessor(features=None, model_kind="linear", contract_ordinal=False,
                       log_totalcharges=False, scale=True, collapse=True):
    """Build the ColumnTransformer for a given feature set.

    model_kind='linear' -> reference-category dummy coding (drop one level per
    nominal feature) so odds ratios are interpretable and the design matrix has
    full rank; 'tree' -> drop='if_binary' (trees are indifferent to redundancy).
    """
    feats = ENGINEERED if features is None else list(features)
    numeric = list(NUMERIC) + [f for f in ("n_addons", "avg_monthly_spend", "price_change") if f in feats]
    binary = list(BINARY_YN) + [MULTILINES] + list(ADDONS) + \
        [f for f in ("any_protection", "auto_pay") if f in feats] + ["SeniorCitizen"]
    nominal = [c for c in NOMINAL if not (contract_ordinal and c == "Contract")]
    if "contract_x_internet" in feats:
        if model_kind == "linear":
            # Cell-means coding: the nine Contract x InternetService cells replace
            # the two main effects so that the design matrix stays full rank.
            nominal = [c for c in nominal if c not in ("Contract", "InternetService")]
        nominal = nominal + ["contract_x_internet"]

    # Numeric: constant-0 imputation for TotalCharges blanks (tenure 0), then
    # optional log1p for the skewed cumulative charges, then scaling.
    # Numeric: impute the 11 structural blanks with 0, optionally log1p(TotalCharges), then scale.
    num_steps = [("impute", SimpleImputer(strategy="constant", fill_value=0.0))]
    if log_totalcharges:
        num_steps.append(("log_tc", FunctionTransformer(
            log1p_column, feature_names_out="one-to-one",
            kw_args={"idx": numeric.index("TotalCharges")})))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    if not collapse:
        # Add-ons still hold 'No internet service' -> three levels -> treat as nominal.
        binary = [c for c in binary if c not in ADDONS and c != MULTILINES]
        nominal = nominal + list(ADDONS) + [MULTILINES]

    # Nominal: reference-category coding for the linear model (full-rank design, odds ratios
    # against a named baseline); drop='if_binary' for trees, which are indifferent to redundancy.
    if model_kind == "linear" and all(c in REFERENCE_LEVELS for c in nominal):
        drop = np.array([REFERENCE_LEVELS[c] for c in nominal], dtype=object)
    else:
        drop = "first" if model_kind == "linear" else "if_binary"
    nom_enc = OneHotEncoder(drop=drop, sparse_output=False, handle_unknown="ignore")

    transformers = [
        ("num", num_pipe, numeric),
        ("bin", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"), binary),
        ("nom", nom_enc, nominal),
    ]
    if contract_ordinal:
        transformers.append(("ord", Pipeline([
            ("enc", OrdinalEncoder(categories=[CONTRACT_ORDER])),
            ("scale", StandardScaler() if scale else "passthrough")]), ["Contract"]))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct


def make_pipeline(estimator, features=None, model_kind="linear", sampler=None,
                  contract_ordinal=False, log_totalcharges=False, scale=True, collapse=True):
    """Full leak-free pipeline: engineer -> encode/scale -> [resample] -> model."""
    steps = [
        ("engineer", FeatureEngineer(features=features, collapse=collapse)),
        ("prep", build_preprocessor(features, model_kind, contract_ordinal,
                                    log_totalcharges, scale, collapse)),
    ]
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps.append(("sample", sampler))
        steps.append(("model", estimator))
        return ImbPipeline(steps)
    steps.append(("model", estimator))
    return Pipeline(steps)


def feature_names(pipe) -> list[str]:
    """Human-readable feature names after the ColumnTransformer."""
    return list(pipe.named_steps["prep"].get_feature_names_out())
