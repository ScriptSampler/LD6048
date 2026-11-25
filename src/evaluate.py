"""Part 4 evaluation: operating thresholds (chosen on OOF training predictions),
expected-profit curve and sensitivity, test metrics with bootstrap CIs,
statistical comparisons (corrected resampled t-test, DeLong, paired bootstrap,
McNemar), calibration, error analysis, learning curve, and figures 4-7.

Reads results/ written by models.py; never refits on the test set.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, learning_curve
from statsmodels.stats.contingency_tables import mcnemar

from .config import (COST_MODEL, DATA_PROCESSED, MODELS_DIR, MODEL_COLOURS, MODEL_LABELS,
                     MODEL_ORDER, N_BOOT, OKABE_ITO, QUICK, RESULTS, SEED,
                     SENSITIVITY_OFFER_COSTS, SENSITIVITY_SAVE_RATES)
from .eda import tenure_band
from .figstyle import COL_W, panel_label, save_fig, setup_style
from .io_utils import load_json, save_csv, save_json
from .metrics import (best_threshold, bootstrap_metrics, calibration_slope_intercept,
                      campaign_profit, compute_metrics, corrected_resampled_ttest, delong_test,
                      expected_calibration_error, paired_bootstrap_diff, profit_curve,
                      reliability_table)

ALL_MODELS = MODEL_ORDER + ["Voting"]
SHORT_NAMES = {"LogReg": "LR", "RandomForest": "RF", "XGBoost": "XGB", "MLP": "MLP", "DecisionTree": "Tree"}
# the forest's vermillion is close to logistic regression's orange, so it is also drawn dash-dot
LINESTYLE = {"RandomForest": "-."}


def _load():
    oof = pd.read_csv(RESULTS / "oof_predictions.csv")
    test = pd.read_csv(RESULTS / "test_predictions.csv")
    cv = pd.read_csv(RESULTS / "nested_cv_scores.csv")
    vcv = pd.read_csv(RESULTS / "voting_outer_cv_scores.csv")
    split = load_json("split_summary.json")
    return oof, test, cv, vcv, split


# ---------------------------------------------------------------------------
def choose_thresholds(oof: pd.DataFrame) -> dict:
    y, m = oof["y"].values, oof["MonthlyCharges"].values
    out = {}
    for name in ALL_MODELS:
        p = oof[f"p_{name}_final"].values
        t_f1, v_f1 = best_threshold(y, p, "f1")
        t_f2, v_f2 = best_threshold(y, p, "f2")
        t_pr, v_pr = best_threshold(y, p, "profit", monthly=m)
        out[name] = {"f1_opt": t_f1, "f1_at_opt": v_f1, "f2_opt": t_f2, "f2_at_opt": v_f2,
                     "profit_opt": t_pr, "profit_at_opt_oof": v_pr, "default": 0.5,
                     "operating": t_pr}
    out["_cost_model"] = COST_MODEL
    out["_note"] = "All thresholds chosen on out-of-fold predictions from the training split; 'operating' = expected-profit-optimal under the illustrative cost model."
    save_json(out, "thresholds.json")
    return out


def profit_analysis(oof: pd.DataFrame, test: pd.DataFrame, thr: dict, best: str) -> None:
    y, m = oof["y"].values, oof["MonthlyCharges"].values
    curves = []
    for name in ALL_MODELS:
        c = profit_curve(y, oof[f"p_{name}_final"].values, m)
        c.insert(0, "model", name)
        curves.append(c)
    save_csv(pd.concat(curves, ignore_index=True), "profit_curve_oof.csv")
    # Test profit at the OOF-chosen thresholds, plus reference policies
    yt, mt = test["y"].values, test["MonthlyCharges"].values
    rows = []
    for name in ALL_MODELS:
        p = test[f"p_{name}"].values
        for kind in ["operating", "f1_opt", "default"]:
            r = campaign_profit(yt, p, mt, thr[name][kind])
            r.update({"model": name, "threshold_kind": kind})
            rows.append(r)
    r = campaign_profit(yt, np.ones_like(yt, dtype=float), mt, 0.5); r.update({"model": "Contact everyone", "threshold_kind": "all"}); rows.append(r)
    r = campaign_profit(yt, np.zeros_like(yt, dtype=float), mt, 0.5); r.update({"model": "Contact no one", "threshold_kind": "none"}); rows.append(r)
    # Oracle upper bound: contact exactly the churners
    r = campaign_profit(yt, yt.astype(float), mt, 0.5); r.update({"model": "Oracle (perfect knowledge)", "threshold_kind": "oracle"}); rows.append(r)
    save_csv(pd.DataFrame(rows), "profit_test.csv", float_format="%.10g")  # profits to the cent: whole dollars round correctly
    # Sensitivity: best model, grid over save rate x offer cost; threshold re-chosen on OOF each time
    sens = []
    for sr in SENSITIVITY_SAVE_RATES:
        for oc in SENSITIVITY_OFFER_COSTS:
            cm = {**COST_MODEL, "save_rate": sr, "offer_cost": oc}
            t, _ = best_threshold(y, oof[f"p_{best}_final"].values, "profit", monthly=m, cm=cm)
            rt = campaign_profit(yt, test[f"p_{best}"].values, mt, t, cm)
            rl = campaign_profit(yt, test["p_LogReg"].values, mt, best_threshold(y, oof["p_LogReg_final"].values, "profit", monthly=m, cm=cm)[0], cm)
            sens.append({"save_rate": sr, "offer_cost": oc, "threshold_best": t, "profit_test_best": rt["profit"],
                         "share_contacted_best": rt["share_contacted"], "profit_test_logreg": rl["profit"],
                         "profit_per_customer_best": rt["profit_per_customer"]})
    save_csv(pd.DataFrame(sens), "profit_sensitivity.csv")


# ---------------------------------------------------------------------------
def test_metrics(test: pd.DataFrame, thr: dict) -> pd.DataFrame:
    y = test["y"].values
    rows = []
    for name in ALL_MODELS:
        p = test[f"p_{name}"].values
        for kind in ["default", "operating"]:
            r = bootstrap_metrics(y, p, thr[name][kind], N_BOOT)
            r.update({"model": name, "threshold_kind": kind})
            rows.append(r)
    df = pd.DataFrame(rows)
    save_csv(df, "test_metrics.csv")
    return df


def rank_models(cv: pd.DataFrame, vcv: pd.DataFrame) -> dict:
    tuned = cv[cv["variant"] == "tuned"].groupby("model")[["pr_auc", "roc_auc", "brier"]].agg(["mean", "std"])
    tuned.columns = [f"{a}_{b}" for a, b in tuned.columns]
    tuned = tuned.reset_index()
    ranking = tuned.sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False)
    best_single = ranking.iloc[0]["model"]
    out = {"ranking_by_cv_pr_auc": ranking["model"].tolist(),
           "best_single_model": best_single,
           "cv_pr_auc": ranking.set_index("model")["pr_auc_mean"].to_dict(),
           "note": "Best model chosen on nested-CV PR-AUC (training split only), never on the test set."}
    save_json(out, "model_ranking.json")
    return out


def stat_tests(cv: pd.DataFrame, vcv: pd.DataFrame, test: pd.DataFrame, thr: dict, split: dict, best: str) -> None:
    n_tr, n_te_fold = split["n_train"], split["n_train"] / 5  # outer folds: 4/5 train, 1/5 validation
    tuned = cv[cv["variant"] == "tuned"].sort_values(["model", "repeat", "fold"])
    lr = tuned[tuned["model"] == "LogReg"]
    rows = []
    for name in ALL_MODELS:
        if name == "LogReg":
            continue
        a = vcv.sort_values(["repeat", "fold"]) if name == "Voting" else tuned[tuned["model"] == name]
        for metric in ["pr_auc", "roc_auc", "brier", "f1"]:
            r = corrected_resampled_ttest(a[metric].values, lr[metric].values, n_tr * 0.8, n_te_fold)
            r.update({"comparison": f"{name} vs LogReg", "metric": metric, "test": "corrected resampled t-test (Nadeau-Bengio)", "source": "outer CV folds (5x2)"})
            rows.append(r)
    # Voting vs best single model
    a = vcv.sort_values(["repeat", "fold"]); b = tuned[tuned["model"] == best]
    for metric in ["pr_auc", "roc_auc"]:
        r = corrected_resampled_ttest(a[metric].values, b[metric].values, n_tr * 0.8, n_te_fold)
        r.update({"comparison": f"Voting vs {best}", "metric": metric, "test": "corrected resampled t-test (Nadeau-Bengio)", "source": "outer CV folds (5x2)"})
        rows.append(r)
    # Default vs tuned per model
    for name in MODEL_ORDER[1:]:
        d = cv[(cv["model"] == name) & (cv["variant"] == "default")].sort_values(["repeat", "fold"])
        t = cv[(cv["model"] == name) & (cv["variant"] == "tuned")].sort_values(["repeat", "fold"])
        r = corrected_resampled_ttest(t["pr_auc"].values, d["pr_auc"].values, n_tr * 0.8, n_te_fold)
        r.update({"comparison": f"{name} tuned vs default", "metric": "pr_auc", "test": "corrected resampled t-test (Nadeau-Bengio)", "source": "outer CV folds (5x2)"})
        rows.append(r)
    save_csv(pd.DataFrame(rows), "stat_tests_cv.csv")

    # Test-set tests
    y = test["y"].values
    rows = []
    for name in ALL_MODELS:
        if name == "LogReg":
            continue
        pa, pb = test[f"p_{name}"].values, test["p_LogReg"].values
        d = delong_test(y, pa, pb)
        rows.append({"comparison": f"{name} vs LogReg", "metric": "roc_auc", "test": "DeLong", **d})
        pb_ = paired_bootstrap_diff(y, pa, pb, average_precision_score, N_BOOT)
        rows.append({"comparison": f"{name} vs LogReg", "metric": "pr_auc", "test": "paired bootstrap", **pb_})
        pb2 = paired_bootstrap_diff(y, pa, pb, roc_auc_score, N_BOOT)
        rows.append({"comparison": f"{name} vs LogReg", "metric": "roc_auc", "test": "paired bootstrap", **pb2})
        for kind in ["operating", "default"]:
            ya = (pa >= thr[name][kind]).astype(int); yb = (pb >= thr["LogReg"][kind]).astype(int)
            ca, cb = (ya == y), (yb == y)
            tab = [[int((ca & cb).sum()), int((ca & ~cb).sum())], [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
            m = mcnemar(tab, exact=False, correction=True)
            rows.append({"comparison": f"{name} vs LogReg", "metric": f"accuracy@{kind}", "test": "McNemar",
                         "statistic": float(m.statistic), "p": float(m.pvalue), "b_only_a_correct": tab[0][1], "b_only_b_correct": tab[1][0]})
    save_csv(pd.DataFrame(rows), "stat_tests_test.csv")


# ---------------------------------------------------------------------------
def calibration(oof: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rel = []
    for name in ALL_MODELS:
        for part, df, col in [("oof", oof, f"p_{name}_final"), ("oof_uncalibrated", oof, f"p_{name}"),
                              ("test", test, f"p_{name}"), ("test_uncalibrated", test, f"p_{name}_uncal")]:
            if col not in df:
                continue
            y, p = df["y"].values, df[col].values
            slope, intercept = calibration_slope_intercept(y, p) if name != "Majority" else (np.nan, np.nan)
            rows.append({"model": name, "part": part, "brier": float(np.mean((p - y) ** 2)),
                         "ece": expected_calibration_error(y, p), "slope": slope, "intercept": intercept,
                         "mean_pred": float(p.mean()), "observed": float(y.mean())})
            if part in ("oof", "test") and name != "Majority":
                t = reliability_table(y, p)
                t.insert(0, "part", part); t.insert(0, "model", name)
                rel.append(t)
    save_csv(pd.concat(rel, ignore_index=True), "reliability_tables.csv")
    df = pd.DataFrame(rows)
    save_csv(df, "calibration_metrics.csv")
    return df


# ---------------------------------------------------------------------------
def error_analysis(test: pd.DataFrame, thr: dict, best: str) -> None:
    X_te = pd.read_csv(DATA_PROCESSED / "test.csv", index_col=0)
    X_te = X_te.loc[test["row"].values] if "row" in test else X_te
    y = test["y"].values
    p = test[f"p_{best}"].values
    t = thr[best]["operating"]
    yhat = (p >= t).astype(int)
    d = X_te.copy()
    d["y"], d["p"], d["yhat"] = y, p, yhat
    d["outcome"] = np.select([(d.y == 1) & (d.yhat == 1), (d.y == 1) & (d.yhat == 0), (d.y == 0) & (d.yhat == 1)],
                             ["TP", "FN", "FP"], default="TN")
    d["tenure_band"] = tenure_band(d["tenure"])
    rows = []
    for col in ["Contract", "InternetService", "PaymentMethod", "tenure_band", "SeniorCitizen", "PaperlessBilling"]:
        for lvl, g in d.groupby(col):
            churn = g[g.y == 1]; stay = g[g.y == 0]
            rows.append({"feature": col, "level": str(lvl), "n": len(g), "churners": len(churn),
                         "fn_rate_among_churners": float((churn.outcome == "FN").mean()) if len(churn) else np.nan,
                         "fp_rate_among_stayers": float((stay.outcome == "FP").mean()) if len(stay) else np.nan,
                         "share_of_all_FN": float((g.outcome == "FN").sum() / max(1, (d.outcome == "FN").sum())),
                         "share_of_all_FP": float((g.outcome == "FP").sum() / max(1, (d.outcome == "FP").sum())),
                         "mean_p_churners": float(churn.p.mean()) if len(churn) else np.nan})
    save_csv(pd.DataFrame(rows), "error_analysis.csv")
    summary = {"model": best, "threshold": t, "n_test": int(len(d)),
               "TP": int((d.outcome == "TP").sum()), "FN": int((d.outcome == "FN").sum()),
               "FP": int((d.outcome == "FP").sum()), "TN": int((d.outcome == "TN").sum()),
               "fn_mean_p": float(d.loc[d.outcome == "FN", "p"].mean()),
               "fn_share_p_below_0.2": float((d.loc[d.outcome == "FN", "p"] < 0.2).mean()),
               "fn_share_two_year_or_one_year": float(d.loc[d.outcome == "FN", "Contract"].isin(["One year", "Two year"]).mean()),
               "fn_share_tenure_gt_24": float((d.loc[d.outcome == "FN", "tenure"] > 24).mean()),
               "fp_share_month_to_month": float((d.loc[d.outcome == "FP", "Contract"] == "Month-to-month").mean()),
               "fp_share_fibre": float((d.loc[d.outcome == "FP", "InternetService"] == "Fiber optic").mean()),
               "fp_mean_tenure": float(d.loc[d.outcome == "FP", "tenure"].mean()),
               "fn_mean_tenure": float(d.loc[d.outcome == "FN", "tenure"].mean())}
    save_json(summary, "error_analysis_summary.json")


def learning_curve_best(best: str) -> pd.DataFrame:
    X_tr = pd.read_csv(DATA_PROCESSED / "train.csv", index_col=0)
    y = X_tr.pop("Churn")
    fitted = joblib.load(MODELS_DIR / f"{best}.joblib")
    pipe = fitted["tuned"]
    sizes = np.linspace(0.1, 1.0, 5 if QUICK else 8)
    rows = []
    for scoring in ["average_precision", "roc_auc"]:
        ts, tr_s, va_s = learning_curve(pipe, X_tr, y, train_sizes=sizes, scoring=scoring,
                                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1,
                                        random_state=SEED, shuffle=True)
        for i, n in enumerate(ts):
            rows.append({"metric": scoring, "n_train": int(n), "train_mean": tr_s[i].mean(), "train_sd": tr_s[i].std(ddof=1),
                         "val_mean": va_s[i].mean(), "val_sd": va_s[i].std(ddof=1)})
    df = pd.DataFrame(rows)
    df["model"] = best
    save_csv(df, "learning_curve.csv")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_ablation():
    import matplotlib.pyplot as plt
    abl = pd.read_csv(RESULTS / "ablation_preprocessing.csv")
    imb = pd.read_csv(RESULTS / "ablation_imbalance.csv")
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 1.2, 3.5), gridspec_kw={"width_ratios": [1.25, 1.05]})
    ax = axes[0]
    fe = abl[abl["ablation"] == "feature_engineering"]
    folds = pd.read_csv(RESULTS / "ablation_folds.csv")
    folds = folds[folds["ablation"] == "feature_engineering"]
    rows, labels, ypos = [], [], []
    y = 0
    for name in ["XGBoost", "LogReg"]:
        sub = fe[fe.model == name]
        base = sub[sub.variant == "none"]
        if base.empty:
            continue
        b = base["pr_auc_mean"].iloc[0]
        base_f = folds[(folds.model == name) & (folds.variant == "none")].sort_values("fold")["pr_auc"].values
        variants = [v for v in sub["variant"] if v != "none"]
        for v in variants:
            r = sub[sub.variant == v].iloc[0]
            vf = folds[(folds.model == name) & (folds.variant == v)].sort_values("fold")["pr_auc"].values
            paired_sd = float(np.std(vf - base_f, ddof=1)) if len(vf) == len(base_f) and len(vf) > 1 else r["pr_auc_sd"]
            rows.append((name, y, r["pr_auc_mean"] - b, paired_sd))
            labels.append(v.replace("+", "+ ").replace("_", " ").replace("x internet", "× internet"))
            ypos.append(y)
            y += 1
        # models.run_ablations tunes XGBoost for each feature set; logistic regression keeps C = 1
        group = "XGBoost, tuned for each feature set" if name == "XGBoost" else MODEL_LABELS[name]
        ax.text(0.01, y - 0.55, group, fontsize=8, fontweight="bold", va="center", ha="left",
                transform=ax.get_yaxis_transform(), color=MODEL_COLOURS[name],  # x in axes units, clear of the spine
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6), zorder=5)  # masks the zero line behind it
        y += 1.2
    for name, yy_, val, sd in rows:
        ax.errorbar(val, yy_, xerr=sd, fmt="o", color=MODEL_COLOURS[name], capsize=2, lw=1.2, zorder=6)  # above the label boxes
    ax.axvline(0, color=OKABE_ITO["grey"], lw=1)
    # ticks every 0.005 with three decimals, so the labels stay apart at this panel width
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if abs(v) < 1e-12 else f"{v:+.3f}".replace("-", "−")))
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, y - 0.5)
    ax.set_xlabel("Change in CV PR-AUC vs no engineered features\n(± SD of paired fold differences)")
    ax.grid(axis="y", visible=False)
    panel_label(ax, "a")
    ax = axes[1]
    lr = imb[imb.model == "LogReg"].set_index("variant")
    opts = ["none", "class_weight_balanced", "smote", "threshold_moving"]
    labels = ["None", "Class weight", "SMOTE", "Threshold moving"]
    xx = np.arange(len(opts))
    ax.bar(xx - 0.2, lr.loc[opts, "pr_auc_mean"], 0.38, yerr=lr.loc[opts, "pr_auc_sd"], color=OKABE_ITO["blue"], label="PR-AUC", capsize=2)
    rec = [lr.loc[o, "recall_moved_mean"] if (o == "threshold_moving" and "recall_moved_mean" in lr) else lr.loc[o, "recall_mean"] for o in opts]
    ax.bar(xx + 0.2, rec, 0.38, color=OKABE_ITO["orange"], label="Recall (at 0.5; threshold moving:\nper-fold F1-optimal)")
    short = ["None", "Class\nweight", "SMOTE", "Threshold\nmoving"]
    ax.set_xticks(xx); ax.set_xticklabels([f"{l}\n({lr.loc[o, 'brier_mean']:.3f})" for l, o in zip(short, opts)], fontsize=7.5)
    ax.set_xlabel("Option (Brier score in brackets)", fontsize=8.5)
    ax.set_ylabel("Score (logistic regression, 5-fold CV × 2)")
    # headroom above 1 keeps the two-line legend clear of the tallest recall bar
    ax.set_ylim(0, 1.13); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.legend(loc="upper left", fontsize=8); ax.grid(axis="x", visible=False)
    panel_label(ax, "b")
    save_fig(fig, "fig04_ablation")


def fig_cv_distributions(cv: pd.DataFrame, vcv: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 1.15, 3.3), sharey=False)
    rng = np.random.RandomState(SEED)
    for ax, metric, letter in zip(axes, ["pr_auc", "roc_auc"], "ab"):
        for i, name in enumerate(MODEL_ORDER + ["Voting"]):
            if name == "Voting":
                d = None; t = vcv[metric].values
            else:
                d = cv[(cv.model == name) & (cv.variant == "default")][metric].values
                t = cv[(cv.model == name) & (cv.variant == "tuned")][metric].values
            if d is not None:
                ax.scatter(i - 0.17 + rng.uniform(-0.05, 0.05, len(d)), d, s=14, color=OKABE_ITO["grey"], alpha=0.8, label="Default (grey)" if i == 1 else None, zorder=3)
                ax.plot([i - 0.28, i - 0.06], [d.mean()] * 2, color="#444444", lw=1.5)
            ax.scatter(i + 0.17 + rng.uniform(-0.05, 0.05, len(t)), t, s=14, color=MODEL_COLOURS[name], alpha=0.9, zorder=3,
                       label="Tuned (model colour)" if i == 1 else None)
            ax.plot([i + 0.06, i + 0.28], [t.mean()] * 2, color=MODEL_COLOURS[name], lw=1.5)
        ax.set_xticks(range(len(MODEL_ORDER) + 1))
        ax.set_xticklabels(["Dummy", "LR", "Tree", "RF", "XGB", "MLP", "Voting"], fontsize=8.5)
        ax.set_ylabel("PR-AUC (average precision)" if metric == "pr_auc" else "ROC-AUC")
        ax.grid(axis="x", visible=False)
        if metric == "pr_auc":
            ax.legend(loc="lower right", fontsize=8)
        panel_label(ax, letter)
    fig.suptitle("Outer-fold scores from nested cross-validation (5-fold, repeated twice; bar = fold mean)", fontsize=10)
    save_fig(fig, "fig05_cv_distributions")


def fig_roc_pr_cal(test: pd.DataFrame, show=("LogReg", "RandomForest", "XGBoost", "MLP", "DecisionTree")):
    import matplotlib.pyplot as plt
    y = test["y"].values
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 1.3, 3.3))
    ax = axes[0]
    for name in show:
        p = test[f"p_{name}"].values
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLOURS[name], label=f"{MODEL_LABELS[name]} ({roc_auc_score(y, p):.3f})", lw=1.5)
    ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], ls="--", lw=1)
    ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate (recall)")
    # an opaque legend box, so the chance line does not run through its text (the theme has no frames)
    ax.legend(fontsize=7.5, loc="lower right", title="ROC-AUC", title_fontsize=8, frameon=True, framealpha=1.0, edgecolor="none")
    panel_label(ax, "a")
    ax = axes[1]
    for name in show:
        p = test[f"p_{name}"].values
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=MODEL_COLOURS[name], label=f"{SHORT_NAMES[name]} {average_precision_score(y, p):.3f}", lw=1.5)
    ax.axhline(y.mean(), color=OKABE_ITO["grey"], ls="--", lw=1)
    ax.text(0.98, 0.03, f"dashed line: prevalence {y.mean():.1%}", fontsize=7.5, color="#555555", ha="right", va="bottom",
            transform=ax.transAxes)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    # short names (as on Figure 5's axis), in the colours and order of panel (a): a narrow legend fits the empty top right
    ax.legend(fontsize=7, loc="upper right", title="PR-AUC", title_fontsize=7.5, frameon=True, framealpha=1.0,
              edgecolor="none", handlelength=1.4, labelspacing=0.25, borderpad=0.3, borderaxespad=0.3)
    panel_label(ax, "b")
    ax = axes[2]
    for name in ("LogReg", "RandomForest", "XGBoost", "MLP"):
        t = reliability_table(y, test[f"p_{name}"].values)
        ax.errorbar(t["mean_pred"], t["obs_rate"], yerr=[np.clip(t["obs_rate"] - t["ci_lo"], 0, None), np.clip(t["ci_hi"] - t["obs_rate"], 0, None)],
                    fmt="o-", color=MODEL_COLOURS[name], label=MODEL_LABELS[name], lw=1.2, ms=4, capsize=2)
    ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], ls="--", lw=1)
    ax.set_xlabel("Mean predicted probability (decile bins)"); ax.set_ylabel("Observed churn rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7.5, loc="upper left")
    panel_label(ax, "c")
    fig.suptitle("Hold-out test set: ROC, precision-recall and reliability (95% Wilson CIs in panel c)", fontsize=10)
    save_fig(fig, "fig06_roc_pr_calibration")


def fig_operating(oof: pd.DataFrame, test: pd.DataFrame, thr: dict, best: str):
    import matplotlib.pyplot as plt
    y = test["y"].values; p = test[f"p_{best}"].values
    t = thr[best]["operating"]
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 1.35, 3.4), gridspec_kw={"width_ratios": [0.8, 1.25, 1.1], "wspace": 0.35})
    # (a) confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y, (p >= t).astype(int))
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", OKABE_ITO["blue"]])
    ax.imshow(cm, cmap=seq, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm.sum() * 100:.1f}%)", ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted\nstay", "Predicted\nchurn"], fontsize=8.5)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nstay", "Actual\nchurn"], fontsize=8.5)
    ax.set_xlabel(f"Threshold {t:.2f}", fontsize=8.5)
    ax.grid(False)
    panel_label(ax, "a")
    # (b) profit vs threshold (OOF) for best and LogReg
    ax = axes[1]
    pc = pd.read_csv(RESULTS / "profit_curve_oof.csv")
    for name in dict.fromkeys([best, "LogReg", "RandomForest"]):
        c = pc[pc.model == name]
        ax.plot(c["threshold"], c["profit_per_customer"], color=MODEL_COLOURS[name], label=MODEL_LABELS[name], lw=1.5,
                ls=LINESTYLE.get(name, "-"))
    ax.axvline(t, color=OKABE_ITO["grey"], ls="--", lw=1)
    ax.text(t + 0.02, 0.30, f"chosen\nthreshold {t:.2f}", fontsize=7.5, color="#555555", va="center", ha="left",
            transform=ax.get_xaxis_transform())  # beside the line, below the curves, not across either
    ax.axhline(0, color=OKABE_ITO["grey"], lw=0.8)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected profit per customer ($, OOF)")
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=1.0, edgecolor="none")  # clear of both curves
    panel_label(ax, "b")
    # (c) cumulative gains
    ax = axes[2]
    for name in dict.fromkeys([best, "LogReg", "RandomForest"]):
        pp = test[f"p_{name}"].values
        order = np.argsort(-pp)
        gains = np.cumsum(y[order]) / y.sum()
        frac = np.arange(1, len(y) + 1) / len(y)
        ax.plot(frac, gains, color=MODEL_COLOURS[name], label=MODEL_LABELS[name], lw=1.5, ls=LINESTYLE.get(name, "-"))
    ax.plot([0, 1], [0, 1], color=OKABE_ITO["grey"], ls="--", lw=1, label="Random")
    prev = y.mean()
    ax.plot([0, prev, 1], [0, 1, 1], color=OKABE_ITO["black"], ls=":", lw=1, label="Perfect")
    ax.axvline(0.1, color=OKABE_ITO["grey"], lw=0.8)
    ax.set_xlabel("Share of customers contacted")
    ax.set_ylabel("Share of churners captured")
    ax.legend(fontsize=7.5, loc="lower right")
    panel_label(ax, "c")
    fig.suptitle("Operating point (test set), profit-threshold curve (training OOF) and cumulative gains", fontsize=10)
    save_fig(fig, "fig07_operating_point")


def fig_learning_curve(lc: pd.DataFrame, best: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(COL_W * 0.75, 3.2))
    d = lc[lc.metric == "average_precision"]
    ax.plot(d["n_train"], d["train_mean"], "o-", color=OKABE_ITO["grey"], label="Training folds")
    ax.fill_between(d["n_train"], d["train_mean"] - d["train_sd"], d["train_mean"] + d["train_sd"], color=OKABE_ITO["grey"], alpha=0.2)
    ax.plot(d["n_train"], d["val_mean"], "o-", color=MODEL_COLOURS[best], label="Validation folds")
    ax.fill_between(d["n_train"], d["val_mean"] - d["val_sd"], d["val_mean"] + d["val_sd"], color=MODEL_COLOURS[best], alpha=0.2)
    ax.set_xlabel("Training-set size (customers)"); ax.set_ylabel("PR-AUC (5-fold CV, mean ± SD)")
    ax.legend(loc="lower right")
    ax.set_title(f"Learning curve, {MODEL_LABELS[best]}", fontsize=10)
    save_fig(fig, "figS3_learning_curve")


def fig_sensitivity(best: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    s = pd.read_csv(RESULTS / "profit_sensitivity.csv")
    piv = s.pivot(index="save_rate", columns="offer_cost", values="profit_per_customer_best")
    thr = s.pivot(index="save_rate", columns="offer_cost", values="threshold_best")
    div = LinearSegmentedColormap.from_list("div", [OKABE_ITO["vermillion"], "#F2F2F2", OKABE_ITO["blue"]])
    vmax = np.nanmax(np.abs(piv.values))
    fig, ax = plt.subplots(figsize=(COL_W * 0.7, 3.0))
    im = ax.imshow(piv.values, cmap=div, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"${c:.0f}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{r:.0%}" for r in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"${piv.values[i, j]:.1f}\nthr {thr.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_xlabel("Offer cost"); ax.set_ylabel("Save rate among contacted churners")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Test profit per customer ($)")
    ax.set_title(f"Profit sensitivity, {MODEL_LABELS[best]}", fontsize=10)
    save_fig(fig, "figS4_profit_sensitivity")


# ---------------------------------------------------------------------------
def run():
    setup_style()
    oof, test, cv, vcv, split = _load()
    thr = choose_thresholds(oof)
    ranking = rank_models(cv, vcv)
    best = ranking["best_single_model"]
    profit_analysis(oof, test, thr, best)
    tm = test_metrics(test, thr)
    stat_tests(cv, vcv, test, thr, split, best)
    calibration(oof, test)
    error_analysis(test, thr, best)
    lc = learning_curve_best(best)
    fig_ablation()
    fig_cv_distributions(cv, vcv)
    fig_roc_pr_cal(test)
    fig_operating(oof, test, thr, best)
    fig_learning_curve(lc, best)
    fig_sensitivity(best)
    print("Best model (nested-CV PR-AUC):", best)
    print(tm[tm.threshold_kind == "operating"][["model", "roc_auc", "pr_auc", "f1", "recall", "precision", "brier"]].round(3).to_string(index=False))
    return best


if __name__ == "__main__":
    run()
