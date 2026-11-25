"""Fairness audit with fairlearn MetricFrame on the hold-out test set at the
operating threshold: recall (TPR), false-positive rate, precision, selection
rate, calibration-in-the-large and Brier by SeniorCitizen, gender, Partner and
Dependents, with group sizes and bootstrap CIs; demographic-parity and
equalised-odds differences. Feeds Part 5.

Outputs: results/fairness_metrics.csv, fairness_summary.csv, fairness_blind.csv; figure fig09.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from fairlearn.metrics import (MetricFrame, demographic_parity_difference, equal_opportunity_difference,
                               equalized_odds_difference, false_positive_rate, selection_rate,
                               true_positive_rate)
from sklearn.metrics import precision_score

from .config import (DATA_PROCESSED, MODEL_COLOURS, MODEL_LABELS, OKABE_ITO, QUICK, RESULTS,
                     SEED, SENSITIVE)
from .figstyle import COL_W, panel_label, save_fig, setup_style
from .io_utils import load_json, save_csv

N_BOOT_FAIR = 50 if QUICK else 500
GROUP_LABELS = {"SeniorCitizen": {0: "Not senior", 1: "Senior"}, "gender": {"Female": "Female", "Male": "Male"},
                "Partner": {"No": "No partner", "Yes": "Partner"}, "Dependents": {"No": "No dependents", "Yes": "Dependents"}}


def _metrics(y, yhat, p):
    return {"recall": float(true_positive_rate(y, yhat)) if y.sum() else np.nan,
            "fpr": float(false_positive_rate(y, yhat)) if (y == 0).sum() else np.nan,
            "precision": float(precision_score(y, yhat, zero_division=0)),
            "selection_rate": float(selection_rate(y, yhat)),
            "observed_rate": float(y.mean()), "mean_pred": float(p.mean()),
            "brier": float(np.mean((p - y) ** 2)), "n": int(len(y))}


def audit(model: str, test: pd.DataFrame, X_te: pd.DataFrame, thr: float):
    y = test["y"].values; p = test[f"p_{model}"].values; yhat = (p >= thr).astype(int)
    rng = np.random.RandomState(SEED)
    rows, summ = [], []
    for attr in SENSITIVE:
        s = X_te[attr].values
        groups = sorted(pd.unique(s), key=str)
        point = {g: _metrics(y[s == g], yhat[s == g], p[s == g]) for g in groups}
        boots = {g: [] for g in groups}
        n = len(y)
        for _ in range(N_BOOT_FAIR):
            idx = rng.randint(0, n, n)
            yb, hb, pb, sb = y[idx], yhat[idx], p[idx], s[idx]
            for g in groups:
                m = sb == g
                if m.sum() > 0 and yb[m].sum() > 0 and (yb[m] == 0).sum() > 0:
                    boots[g].append(_metrics(yb[m], hb[m], pb[m]))
        for g in groups:
            b = pd.DataFrame(boots[g])
            for metric in ["recall", "fpr", "precision", "selection_rate", "observed_rate", "mean_pred", "brier"]:
                rows.append({"model": model, "attribute": attr, "group": GROUP_LABELS[attr].get(g, str(g)),
                             "n": point[g]["n"], "metric": metric, "value": point[g][metric],
                             "ci_lo": float(b[metric].quantile(0.025)) if len(b) else np.nan,
                             "ci_hi": float(b[metric].quantile(0.975)) if len(b) else np.nan})
        mf = MetricFrame(metrics={"recall": true_positive_rate, "fpr": false_positive_rate, "selection_rate": selection_rate},
                         y_true=y, y_pred=yhat, sensitive_features=s)
        summ.append({"model": model, "attribute": attr, "threshold": thr,
                     "demographic_parity_difference": float(demographic_parity_difference(y, yhat, sensitive_features=s)),
                     "equalized_odds_difference": float(equalized_odds_difference(y, yhat, sensitive_features=s)),
                     "equal_opportunity_difference": float(equal_opportunity_difference(y, yhat, sensitive_features=s)),
                     "recall_difference": float(mf.difference()["recall"]),
                     "fpr_difference": float(mf.difference()["fpr"]),
                     "selection_rate_ratio": float(mf.ratio()["selection_rate"]),
                     "base_rate_difference": float(abs(np.diff([point[g]["observed_rate"] for g in groups])).max()) if len(groups) == 2 else np.nan,
                     "groups": " | ".join(f"{GROUP_LABELS[attr].get(g, g)} (n={point[g]['n']})" for g in groups)})
    return pd.DataFrame(rows), pd.DataFrame(summ)


def blind_variants(model: str) -> pd.DataFrame:
    """'Aware for audit, blind for scoring' (Part 5): the selected pipeline refitted with gender, and then
    with gender and senior status, held constant, which removes them (a constant column carries no
    information). Each variant, the full model included, gets its own profit-optimal threshold from
    5-fold out-of-fold training predictions and is then scored on the test set, so the three differ only
    in their inputs. The audit attributes still come from the unmodified test data.
    Output: results/fairness_blind.csv (one row per variant)."""
    import joblib
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from .config import MODELS_DIR
    from .metrics import best_threshold, campaign_profit
    X_tr = pd.read_csv(DATA_PROCESSED / "train.csv", index_col=0); y_tr = X_tr.pop("Churn")
    X_te = pd.read_csv(DATA_PROCESSED / "test.csv", index_col=0); y = X_te.pop("Churn").values
    pipe = joblib.load(MODELS_DIR / f"{model}.joblib")["final"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for variant, held in (("all inputs", {}), ("gender removed", {"gender": "Female"}),
                          ("gender and senior status removed", {"gender": "Female", "SeniorCitizen": 0})):
        Xa, Xb = X_tr.assign(**held), X_te.assign(**held)
        p_oof = cross_val_predict(clone(pipe), Xa, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr, _ = best_threshold(y_tr.values, p_oof, "profit", monthly=Xa["MonthlyCharges"].values)
        p = clone(pipe).fit(Xa, y_tr).predict_proba(Xb)[:, 1]
        yhat = (p >= thr).astype(int)
        row = {"variant": variant, "threshold": float(thr), "test_pr_auc": float(average_precision_score(y, p)),
               "test_profit": campaign_profit(y, p, X_te["MonthlyCharges"].values, thr)["profit"]}
        for attr in SENSITIVE:
            s = X_te[attr].values
            for g in sorted(pd.unique(s), key=str):
                mt = _metrics(y[s == g], yhat[s == g], p[s == g])
                for metric in ("recall", "fpr", "selection_rate"):
                    row[f"{attr}: {GROUP_LABELS[attr].get(g, g)}: {metric}"] = mt[metric]
        rows.append(row)
    out = pd.DataFrame(rows)
    save_csv(out, "fairness_blind.csv")
    return out


def fig_fairness(metrics: pd.DataFrame, model: str):
    import matplotlib.pyplot as plt
    d = metrics[metrics.model == model]
    fig, axes = plt.subplots(1, 4, figsize=(COL_W * 1.5, 3.3), sharey=True)
    show = [("recall", "Recall"), ("fpr", "False-positive rate"), ("selection_rate", "Selection rate"), ("precision", "Precision"), ("observed_rate", "Base rate")]
    for ax, attr, letter in zip(axes, SENSITIVE, "abcd"):
        dd = d[d.attribute == attr]
        groups = list(dict.fromkeys(dd["group"]))
        colours = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["green"]]
        for gi, g in enumerate(groups):
            gg = dd[dd.group == g].set_index("metric")
            xs = np.arange(len(show)) + (gi - (len(groups) - 1) / 2) * 0.28
            vals = [gg.loc[m, "value"] for m, _ in show]
            lo = [max(0.0, gg.loc[m, "value"] - gg.loc[m, "ci_lo"]) for m, _ in show]
            hi = [max(0.0, gg.loc[m, "ci_hi"] - gg.loc[m, "value"]) for m, _ in show]
            n = int(gg["n"].iloc[0])
            ax.errorbar(xs, vals, yerr=[lo, hi], fmt="o", color=colours[gi], capsize=2, lw=1.2, label=f"{g} (n={n:,})")
        ax.set_xticks(range(len(show))); ax.set_xticklabels([s for _, s in show], fontsize=7.5, rotation=25, ha="right")
        ax.set_title({"SeniorCitizen": "Senior citizen", "gender": "Gender"}.get(attr, attr), fontsize=9.5)
        ax.legend(fontsize=7, loc="upper left", frameon=True, framealpha=0.85, edgecolor="none")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        panel_label(ax, letter)
    axes[0].set_ylabel("Rate (test set, 95% bootstrap CI)")
    fig.suptitle(f"Group fairness audit, {MODEL_LABELS[model]} at the operating threshold", fontsize=10)
    save_fig(fig, "fig09_fairness")


def run():
    setup_style()
    test = pd.read_csv(RESULTS / "test_predictions.csv")
    X_te = pd.read_csv(DATA_PROCESSED / "test.csv", index_col=0).loc[test["row"].values]
    thr = load_json("thresholds.json")
    best = load_json("model_ranking.json")["best_single_model"]
    allm, alls = [], []
    for model in dict.fromkeys([best, "LogReg"]):
        m, s = audit(model, test, X_te, thr[model]["operating"])
        allm.append(m); alls.append(s)
    metrics = pd.concat(allm, ignore_index=True); summary = pd.concat(alls, ignore_index=True)
    save_csv(metrics, "fairness_metrics.csv")
    save_csv(summary, "fairness_summary.csv")
    fig_fairness(metrics, best)
    blind_variants(best)
    print(summary[["model", "attribute", "demographic_parity_difference", "equalized_odds_difference", "base_rate_difference"]].round(3).to_string(index=False))


if __name__ == "__main__":
    run()
