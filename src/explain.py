"""Explainability: logistic-regression odds ratios (bootstrap CIs), permutation
importance on the test set (Strobl et al., 2007), SHAP for the gradient-boosting
model (Lundberg and Lee, 2017), and where the linear and tree views agree.

Outputs: results/lr_odds_ratios.csv (scoring model), lr_odds_ratios_interp.csv and vif_interp.csv
(interpretation model without monthly charges, shown in Figure 8a and Tables C2-C3), perm_importance.csv,
shap_mean_abs.csv, importance_agreement.json; figures fig08 and figS5.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from .config import (DATA_PROCESSED, MODELS_DIR, MODEL_COLOURS, MODEL_LABELS, OKABE_ITO, QUICK,
                     RESULTS, SEED)
from .figstyle import COL_W, panel_label, save_fig, setup_style
from .io_utils import load_json, save_csv, save_json

N_BOOT_COEF = 30 if QUICK else 200
N_PERM = 5 if QUICK else 20

PRETTY = {"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
          "avg_monthly_spend": "Avg monthly spend", "price_change": "Price change", "n_addons": "No. of add-ons",
          "any_protection": "Any protection add-on", "auto_pay": "Automatic payment", "SeniorCitizen": "Senior citizen",
          "contract_x_internet": "Contract × Internet"}


def _load():
    X_tr = pd.read_csv(DATA_PROCESSED / "train.csv", index_col=0); y_tr = X_tr.pop("Churn")
    X_te = pd.read_csv(DATA_PROCESSED / "test.csv", index_col=0); y_te = X_te.pop("Churn")
    best = load_json("model_ranking.json")["best_single_model"]
    fitted = {n: joblib.load(MODELS_DIR / f"{n}.joblib") for n in ["LogReg", "XGBoost", "RandomForest", best]}
    return X_tr, y_tr, X_te, y_te, best, fitted


def _split(name: str) -> tuple[str, str | None]:
    """Return (original feature, level) for a transformed column name."""
    for k in sorted(PRETTY, key=len, reverse=True):
        if name == k:
            return k, None
        if name.startswith(k + "_"):
            return k, name[len(k) + 1:]
    if "_" in name:
        base, lvl = name.split("_", 1)
        return base, lvl
    return name, None


def _pretty(name: str) -> str:
    # e.g. 'Contract_Two year' -> 'Contract: Two year'; 'OnlineSecurity_Yes' -> 'Online security: Yes'
    # (the tables' labels, so Figure 8 and Table C3 read like Table C2)
    from .make_tables import PRETTY_FEAT
    base, lvl = _split(name)
    import re as _re
    label = PRETTY.get(base) or PRETTY_FEAT.get(base) or _re.sub(r"(?<=[a-z])(?=[A-Z])", " ", base).capitalize()
    if base == "SeniorCitizen" and lvl is not None:
        lvl = {"1": "Yes", "0": "No"}.get(lvl, lvl)
    return f"{label}: {lvl}" if lvl is not None else label


def original_feature(name: str) -> str:
    return _split(name)[0]


# ---------------------------------------------------------------------------
def lr_odds_ratios(X_tr, y_tr, fitted) -> pd.DataFrame:
    pipe = fitted["LogReg"]["tuned"]
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    coef = pipe.named_steps["model"].coef_[0]
    rng = np.random.RandomState(SEED)
    boots = []
    n = len(X_tr)
    for _ in range(N_BOOT_COEF):
        idx = rng.randint(0, n, n)
        m = clone(pipe).fit(X_tr.iloc[idx], y_tr.iloc[idx])
        boots.append(m.named_steps["model"].coef_[0])
    boots = np.asarray(boots)
    numeric = set(pipe.named_steps["prep"].transformers_[0][2])
    df = pd.DataFrame({"feature": names, "coef": coef, "odds_ratio": np.exp(coef),
                       "or_ci_lo": np.exp(np.quantile(boots, 0.025, axis=0)),
                       "or_ci_hi": np.exp(np.quantile(boots, 0.975, axis=0)),
                       "unit": ["per +1 SD" if f in numeric else "vs reference level" for f in names]})
    df["label"] = df["feature"].map(_pretty)
    if load_json("chosen_config.json").get("log_totalcharges"):  # the linear pipeline models log1p(TotalCharges)
        df.loc[df["feature"] == "TotalCharges", "label"] = "Total charges (log)"
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df["C"] = pipe.named_steps["model"].C
    df["intercept"] = pipe.named_steps["model"].intercept_[0]
    save_csv(df, "lr_odds_ratios.csv")
    return df


def lr_interpretation(X_tr, y_tr, fitted, drop=("MonthlyCharges",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Odds ratios for reading, not for scoring: the tuned logistic pipeline refitted with MonthlyCharges
    held constant, which removes it (its standardised column is all zeros, so its coefficient stays
    zero). The monthly charge is almost a sum of the service dummies (VIF 860 in the scoring model), so
    with it in the model single coefficients cannot be read one at a time. Returns the odds ratios with
    bootstrap CIs and the VIFs of the remaining design matrix (training split)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    pipe = fitted["LogReg"]["tuned"]
    Xc = X_tr.copy()
    for c in drop:
        Xc[c] = 0.0
    m = clone(pipe).fit(Xc, y_tr)
    names = list(m.named_steps["prep"].get_feature_names_out())
    keep = [i for i, n in enumerate(names) if n not in drop]
    rng = np.random.RandomState(SEED)
    boots = []
    n = len(Xc)
    for _ in range(N_BOOT_COEF):
        idx = rng.randint(0, n, n)
        boots.append(clone(pipe).fit(Xc.iloc[idx], y_tr.iloc[idx]).named_steps["model"].coef_[0][keep])
    boots = np.asarray(boots)
    coef = m.named_steps["model"].coef_[0][keep]
    numeric = set(pipe.named_steps["prep"].transformers_[0][2])
    feats = [names[i] for i in keep]
    df = pd.DataFrame({"feature": feats, "coef": coef, "odds_ratio": np.exp(coef),
                       "or_ci_lo": np.exp(np.quantile(boots, 0.025, axis=0)),
                       "or_ci_hi": np.exp(np.quantile(boots, 0.975, axis=0)),
                       "unit": ["per +1 SD" if f in numeric else "vs reference level" for f in feats]})
    df["label"] = df["feature"].map(_pretty)
    if load_json("chosen_config.json").get("log_totalcharges"):
        df.loc[df["feature"] == "TotalCharges", "label"] = "Total charges (log)"
    df["abs_coef"] = df["coef"].abs()
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df["C"] = m.named_steps["model"].C
    df["dropped"] = ", ".join(drop)
    save_csv(df, "lr_odds_ratios_interp.csv")
    Z = np.asarray(m[:-1].transform(Xc), float)[:, keep]
    Zc = np.column_stack([np.ones(len(Z)), Z])
    vif = pd.DataFrame({"feature": feats, "vif": [variance_inflation_factor(Zc, i + 1) for i in range(Z.shape[1])]})
    vif = vif.sort_values("vif", ascending=False).reset_index(drop=True)
    save_csv(vif, "vif_interp.csv")
    return df, vif


def perm_importance(X_te, y_te, fitted, best) -> pd.DataFrame:
    rows = []
    for name in dict.fromkeys([best, "LogReg", "XGBoost"]):
        est = fitted[name]["final"]
        r = permutation_importance(est, X_te, y_te, scoring="average_precision", n_repeats=N_PERM,
                                   random_state=SEED, n_jobs=-1)
        for i, c in enumerate(X_te.columns):
            rows.append({"model": name, "feature": c, "importance_mean": float(r.importances_mean[i]),
                         "importance_sd": float(r.importances_std[i])})
    df = pd.DataFrame(rows).sort_values(["model", "importance_mean"], ascending=[True, False])
    save_csv(df, "perm_importance.csv")
    return df


def shap_values(X_te, fitted, model_name="XGBoost"):
    import shap
    pipe = fitted[model_name]["tuned"]
    Z = np.asarray(pipe[:-1].transform(X_te), dtype=float)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Z)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0),
                             "mean_shap": sv.mean(axis=0)})
    mean_abs["label"] = mean_abs["feature"].map(_pretty)
    mean_abs["original_feature"] = mean_abs["feature"].map(original_feature)
    mean_abs = mean_abs.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    mean_abs["model"] = model_name
    save_csv(mean_abs, "shap_mean_abs.csv")
    return sv, Z, names, mean_abs


def agreement(lr: pd.DataFrame, perm: pd.DataFrame, shap_tab: pd.DataFrame, best: str) -> dict:
    lr_by_feat = lr.assign(orig=lr["feature"].map(original_feature)).groupby("orig")["abs_coef"].max()
    pm = perm[perm.model == best].set_index("feature")["importance_mean"]
    sh = shap_tab.groupby("original_feature")["mean_abs_shap"].sum()
    common = sorted(set(lr_by_feat.index) & set(pm.index))
    rho_lr_perm = stats.spearmanr(lr_by_feat.loc[common], pm.loc[common])
    common2 = sorted(set(lr_by_feat.index) & set(sh.index))
    rho_lr_shap = stats.spearmanr(lr_by_feat.loc[common2], sh.loc[common2])
    out = {"best_model": best,
           "top5_lr": lr_by_feat.sort_values(ascending=False).head(5).index.tolist(),
           "top5_perm_best": pm.sort_values(ascending=False).head(5).index.tolist(),
           "top5_shap_xgb": sh.sort_values(ascending=False).head(5).index.tolist(),
           "spearman_lr_vs_perm": float(rho_lr_perm.statistic), "spearman_lr_vs_perm_p": float(rho_lr_perm.pvalue),
           "spearman_lr_vs_shap": float(rho_lr_shap.statistic), "spearman_lr_vs_shap_p": float(rho_lr_shap.pvalue),
           "n_features_compared": len(common)}
    save_json(out, "importance_agreement.json")
    return out


# ---------------------------------------------------------------------------
def _beeswarm(ax, fig, sv, Z, names, top_k=12):
    """Beeswarm of SHAP values for the top_k features by mean |SHAP|; point
    colour = standardised feature value (blue low, vermillion high)."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("okabe_div", [OKABE_ITO["blue"], "#DDDDDD", OKABE_ITO["vermillion"]])
    order = np.argsort(-np.abs(sv).mean(axis=0))[:top_k][::-1]
    rng = np.random.RandomState(SEED)
    sc = None
    for row, j in enumerate(order):
        v = sv[:, j]
        x = Z[:, j]
        # rank-scale feature values to [0, 1] so binary and continuous columns share a colour scale
        r = stats.rankdata(x) / len(x)
        # vertical jitter proportional to local density (simple binning)
        bins = np.linspace(v.min(), v.max() + 1e-9, 40)
        idx = np.digitize(v, bins)
        jitter = np.zeros_like(v)
        for b in np.unique(idx):
            m = idx == b
            k = m.sum()
            jitter[m] = (rng.permutation(k) - k / 2) / max(k, 1) * min(0.8, 0.02 * k)
        sc = ax.scatter(v, row + jitter, c=r, cmap=cmap, norm=Normalize(0, 1), s=5, alpha=0.7, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty(names[j]) for j in order], fontsize=8)
    ax.axvline(0, color=OKABE_ITO["grey"], lw=0.8)
    ax.set_xlabel("SHAP value (log-odds of churn)")
    ax.grid(axis="y", visible=False)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, shrink=0.4, anchor=(0.0, 0.5), ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"], fontsize=7.5)
    cb.set_label("Feature value (rank)", fontsize=7.5)


def fig_explain(lr: pd.DataFrame, perm: pd.DataFrame, sv, Z, names, best: str):
    import matplotlib.pyplot as plt
    import shap
    fig = plt.figure(figsize=(COL_W * 1.35, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.35, 1.0])
    # (a) odds ratios (top 14 by |coef|) of the interpretation model (lr_interpretation)
    ax = fig.add_subplot(gs[0])
    top = lr.head(14).iloc[::-1]
    yy = np.arange(len(top))
    ax.errorbar(top["odds_ratio"], yy, xerr=[top["odds_ratio"] - top["or_ci_lo"], top["or_ci_hi"] - top["odds_ratio"]],
                fmt="o", color=MODEL_COLOURS["LogReg"], capsize=2, lw=1.2)
    ax.axvline(1, color=OKABE_ITO["grey"], lw=1)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ticks = [t for t in [0.1, 0.25, 1, 4, 8] if top["or_ci_lo"].min() * 0.8 <= t <= top["or_ci_hi"].max() * 1.25]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter()); ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax.set_yticks(yy); ax.set_yticklabels(top["label"], fontsize=8)
    ax.set_xlabel("Odds ratio, model without monthly charges\n(log scale, 95% bootstrap CI)")
    ax.set_title("Logistic regression", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "a")
    # (b) SHAP beeswarm (XGBoost), drawn directly so it obeys the figure theme
    ax = fig.add_subplot(gs[1])
    _beeswarm(ax, fig, sv, Z, names, top_k=12)
    ax.set_title("XGBoost SHAP (test set)", fontsize=9.5)
    panel_label(ax, "b")
    # (c) permutation importance for the best model
    ax = fig.add_subplot(gs[2])
    pm = perm[perm.model == best].head(12).iloc[::-1]
    yy = np.arange(len(pm))
    ax.barh(yy, pm["importance_mean"], xerr=pm["importance_sd"], color=MODEL_COLOURS[best], capsize=2, height=0.6)
    ax.set_yticks(yy); ax.set_yticklabels([("Total charges (log in model)" if f == "TotalCharges" and best == "LogReg" else _pretty(f)) for f in pm["feature"]], fontsize=8)
    ax.set_xlabel("Drop in test PR-AUC when permuted")
    ax.set_title(f"Permutation importance, {MODEL_LABELS[best]}", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    panel_label(ax, "c")
    save_fig(fig, "fig08_explainability")


def fig_shap_dependence(sv, Z, names):
    import matplotlib.pyplot as plt
    feats = [f for f in ["tenure", "MonthlyCharges", "TotalCharges"] if f in names][:3]
    fig, axes = plt.subplots(1, len(feats), figsize=(COL_W * 1.1, 3.0))
    axes = np.atleast_1d(axes)
    for ax, f, letter in zip(axes, feats, "abc"):
        j = names.index(f)
        ax.scatter(Z[:, j], sv[:, j], s=6, alpha=0.4, color=MODEL_COLOURS["XGBoost"])
        ax.axhline(0, color=OKABE_ITO["grey"], lw=0.8)
        ax.set_xlabel(f"{PRETTY.get(f, f)} (standardised)")
        ax.set_ylabel("SHAP value (log-odds)")
        panel_label(ax, letter)
    fig.suptitle("SHAP dependence, XGBoost (test set)", fontsize=10)
    save_fig(fig, "figS5_shap_dependence")


def run():
    setup_style()
    X_tr, y_tr, X_te, y_te, best, fitted = _load()
    lr = lr_odds_ratios(X_tr, y_tr, fitted)
    lr_read, _ = lr_interpretation(X_tr, y_tr, fitted)
    perm = perm_importance(X_te, y_te, fitted, best)
    sv, Z, names, shap_tab = shap_values(X_te, fitted, "XGBoost")
    agr = agreement(lr, perm, shap_tab, best)
    fig_explain(lr_read, perm, sv, Z, names, best)
    fig_shap_dependence(sv, Z, names)
    print("Top LR:", agr["top5_lr"], "| top perm:", agr["top5_perm_best"], "| rho:", round(agr["spearman_lr_vs_perm"], 2))


if __name__ == "__main__":
    run()
