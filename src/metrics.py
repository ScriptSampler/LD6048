"""Metric helpers: point metrics, bootstrap CIs, DeLong, corrected resampled
t-test, calibration summaries, expected-profit curve."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

from .config import COST_MODEL, SEED

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier",
                "mcc", "lift_top_decile", "specificity"]


def top_decile_lift(y: np.ndarray, p: np.ndarray, frac: float = 0.10) -> float:
    """Churn rate among the top decile by score divided by the base rate.
    Ties at the decile boundary are included in full, so a constant scorer
    (every customer tied) gives exactly 1.0 rather than an arbitrary subset."""
    n = len(y)
    if y.mean() == 0:
        return np.nan
    k = max(1, int(np.ceil(frac * n)))
    p_sorted = np.sort(p)[::-1]
    cut = p_sorted[k - 1]
    top = p >= cut
    return float(y[top].mean() / y.mean())


def compute_metrics(y, p, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    out = {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier_score_loss(y, p),
        "mcc": matthews_corrcoef(y, yhat),
        "lift_top_decile": top_decile_lift(y, p),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        "threshold": thr,
    }
    return {k: float(v) for k, v in out.items()}


def bootstrap_metrics(y, p, thr: float, n_boot: int, seed: int = SEED,
                      alpha: float = 0.05) -> dict:
    """Percentile bootstrap CIs for every metric (resampling test rows)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    rows = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        rows.append(compute_metrics(y[idx], p[idx], thr))
    b = pd.DataFrame(rows)
    point = compute_metrics(y, p, thr)
    out = {}
    for m in METRIC_NAMES:
        out[m] = point[m]
        out[f"{m}_lo"] = float(b[m].quantile(alpha / 2))
        out[f"{m}_hi"] = float(b[m].quantile(1 - alpha / 2))
    out["threshold"] = thr
    return out


def paired_bootstrap_diff(y, p_a, p_b, metric_fn, n_boot: int, seed: int = SEED) -> dict:
    rng = np.random.RandomState(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric_fn(y[idx], p_a[idx]) - metric_fn(y[idx], p_b[idx]))
    diffs = np.asarray(diffs)
    return {"diff": float(metric_fn(y, p_a) - metric_fn(y, p_b)),
            "ci_lo": float(np.quantile(diffs, 0.025)), "ci_hi": float(np.quantile(diffs, 0.975)),
            "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}


# ---------------------------------------------------------------------------
# DeLong test (Sun & Xu 2014 fast implementation)
#
# _compute_midrank and _fast_delong are adapted from compare_auc_delong_xu.py in
# https://github.com/yandexdataschool/roc_comparison (MIT License, Copyright (c) 2021
# Nikita Kazeev), which states that it was adapted from https://github.com/Netflix/vmaf
# (BSD-2-Clause-Patent, Copyright (c) 2020 Netflix, Inc.). Licence texts are in LICENSES/;
# see THIRD_PARTY_NOTICES.md.
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_test(y, p_a, p_b) -> dict:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    label_1_count = int(y.sum())
    preds = np.vstack([np.asarray(p_a)[order], np.asarray(p_b)[order]])
    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if var > 0 else np.nan
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff),
            "se": float(np.sqrt(var)) if var > 0 else np.nan, "z": float(z), "p": float(p)}


# ---------------------------------------------------------------------------
# Corrected resampled t-test (Nadeau & Bengio 2003)
# ---------------------------------------------------------------------------
def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int, n_test: int) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    k = len(d)
    mean = d.mean()
    var = d.var(ddof=1)
    corr_var = var * (1.0 / k + n_test / n_train)
    t = mean / np.sqrt(corr_var) if corr_var > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=k - 1) if corr_var > 0 else np.nan
    return {"mean_diff": float(mean), "sd_diff": float(np.sqrt(var)), "t": float(t),
            "df": k - 1, "p": float(p), "n_folds": k}


# ---------------------------------------------------------------------------
# Calibration summaries
# ---------------------------------------------------------------------------
def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = 0.0, 1.0 + 1e-9
    idx = np.clip(np.digitize(p, bins[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration: logit(y) ~ a + b * logit(p). b=1, a=0 is ideal."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lp, np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins[0], bins[-1] = 0.0, 1.0 + 1e-9
    idx = np.clip(np.digitize(p, bins[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        k, n = int(y[m].sum()), int(m.sum())
        from statsmodels.stats.proportion import proportion_confint
        lo, hi = proportion_confint(k, n, method="wilson")
        rows.append({"bin": b, "n": n, "mean_pred": float(p[m].mean()), "obs_rate": k / n,
                     "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Expected profit of a retention campaign under the illustrative cost model
# ---------------------------------------------------------------------------
def campaign_profit(y, p, monthly, thr: float, cm: dict = COST_MODEL) -> dict:
    """Contact everyone with p >= thr. A contacted churner is saved with
    probability save_rate, yielding margin * horizon * monthly minus the offer;
    a contacted non-churner takes the offer with probability leakage (pure
    cost). Contact cost is paid for everyone contacted."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float); m = np.asarray(monthly, float)
    contact = p >= thr
    value = cm["margin"] * cm["horizon_months"] * m
    gain_churn = cm["save_rate"] * (value - cm["offer_cost"]) - cm["contact_cost"]
    cost_stay = -(cm["leakage"] * cm["offer_cost"] + cm["contact_cost"])
    profit = np.where(y == 1, gain_churn, cost_stay) * contact
    return {"threshold": float(thr), "n_contacted": int(contact.sum()),
            "share_contacted": float(contact.mean()), "profit": float(profit.sum()),
            "profit_per_customer": float(profit.sum() / len(y)),
            "churners_contacted": int((contact & (y == 1)).sum()),
            "recall": float((contact & (y == 1)).sum() / max(1, y.sum())),
            "precision": float((contact & (y == 1)).sum() / max(1, contact.sum()))}


def profit_curve(y, p, monthly, cm: dict = COST_MODEL, grid=None) -> pd.DataFrame:
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    return pd.DataFrame([campaign_profit(y, p, monthly, t, cm) for t in grid])


def best_threshold(y, p, kind: str = "f1", monthly=None, cm: dict = COST_MODEL) -> tuple[float, float]:
    grid = np.linspace(0.02, 0.98, 97)
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    best_t, best_v = 0.5, -np.inf
    for t in grid:
        yhat = (p >= t).astype(int)
        if kind == "f1":
            v = f1_score(y, yhat, zero_division=0)
        elif kind == "f2":
            from sklearn.metrics import fbeta_score
            v = fbeta_score(y, yhat, beta=2, zero_division=0)
        elif kind == "profit":
            v = campaign_profit(y, p, monthly, t, cm)["profit"]
        else:
            raise ValueError(kind)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v
