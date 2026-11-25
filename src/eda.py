"""Part 2: data analysis and opportunity identification.

Descriptive EDA may use the full dataset (the brief's Part 2 is descriptive);
every modelling decision is made on the training split in models.py.
Outputs: results/eda_*.json|csv and figures fig01-fig03.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
from statsmodels.stats.proportion import proportion_confint

from .config import (ADDONS, ALL_CATEGORICAL, CHURN_COLOURS, ID_COL, NUMERIC,
                     OKABE_ITO, PALETTE, SEED, TARGET)
from .features import engineer_features
from .figstyle import COL_W, panel_label, save_fig, setup_style
from .io_utils import save_csv, save_json

TENURE_BANDS = [(-1, 6, "0-6"), (6, 12, "7-12"), (12, 24, "13-24"), (24, 48, "25-48"),
                (48, 72, "49-72")]


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def tenure_band(t: pd.Series) -> pd.Series:
    return pd.cut(t, bins=[b[0] for b in TENURE_BANDS] + [TENURE_BANDS[-1][1]],
                  labels=[b[2] for b in TENURE_BANDS]).astype(str)


def rate_table(df: pd.DataFrame, col: str, target: str = TARGET) -> pd.DataFrame:
    g = df.groupby(col, observed=True)[target].agg(["size", "sum", "mean"]).reset_index()
    g.columns = ["level", "n", "churners", "churn_rate"]
    ci = g.apply(lambda r: wilson(int(r["churners"]), int(r["n"])), axis=1, result_type="expand")
    g["ci_lo"], g["ci_hi"] = ci[0], ci[1]
    g.insert(0, "feature", col)
    g["level"] = g["level"].astype(str)
    return g


def cramers_v(ct: np.ndarray) -> tuple[float, float, float, int]:
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    n = ct.sum()
    k = min(ct.shape) - 1
    v = np.sqrt(chi2 / (n * k)) if k > 0 else np.nan
    return float(v), float(chi2), float(p), int(dof)


def associations(df: pd.DataFrame) -> pd.DataFrame:
    y = df[TARGET].values
    h_y = stats.entropy(np.bincount(y) / len(y), base=2)
    rows = []
    for c in ALL_CATEGORICAL:
        ct = pd.crosstab(df[c], df[TARGET]).values
        v, chi2, p, dof = cramers_v(ct)
        mi = mutual_info_score(df[c].astype(str), y) / np.log(2)  # bits
        rows.append({"feature": c, "type": "categorical", "n_levels": int(df[c].nunique()),
                     "cramers_v": v, "chi2": chi2, "dof": dof, "p_value": p,
                     "mutual_info_bits": mi, "mi_normalised": mi / h_y,
                     "rank_biserial": np.nan, "point_biserial_r": np.nan,
                     "mann_whitney_U": np.nan,
                     "median_churn": np.nan, "median_stay": np.nan})
    rng = np.random.RandomState(SEED)
    for c in NUMERIC:
        s = df[[c, TARGET]].dropna()
        a, b = s.loc[s[TARGET] == 1, c], s.loc[s[TARGET] == 0, c]
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rb = 1 - 2 * U / (len(a) * len(b))          # rank-biserial (churn vs stay)
        r_pb = stats.pointbiserialr(s[TARGET], s[c]).statistic
        mi = mutual_info_classif(s[[c]].values, s[TARGET].values, random_state=SEED)[0] / np.log(2)
        rows.append({"feature": c, "type": "numeric", "n_levels": np.nan,
                     "cramers_v": np.nan, "chi2": np.nan, "dof": np.nan, "p_value": float(p),
                     "mutual_info_bits": float(mi), "mi_normalised": float(mi / h_y),
                     "rank_biserial": float(-rb), "point_biserial_r": float(r_pb),
                     "mann_whitney_U": float(U),
                     "median_churn": float(a.median()), "median_stay": float(b.median())})
    out = pd.DataFrame(rows)
    # A single comparable effect size: Cramer's V for categoricals, |point-biserial| for numerics.
    out["effect_for_ranking"] = out["cramers_v"].fillna(out["point_biserial_r"].abs())
    return out.sort_values("effect_for_ranking", ascending=False).reset_index(drop=True)


def km_by_contract(df: pd.DataFrame) -> dict:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    out = {"groups": {}}
    res = multivariate_logrank_test(df["tenure"], df["Contract"], df[TARGET])
    out["logrank_stat"] = float(res.test_statistic)
    out["logrank_p"] = float(res.p_value)
    out["logrank_dof"] = int(res.degrees_of_freedom) if hasattr(res, "degrees_of_freedom") else 2
    fitters = {}
    for lvl, g in df.groupby("Contract"):
        kmf = KaplanMeierFitter(label=lvl)
        kmf.fit(g["tenure"], event_observed=g[TARGET])
        fitters[lvl] = kmf
        sf = kmf.survival_function_
        pts = {}
        for t in (12, 24, 36, 48, 60):
            pts[str(t)] = float(kmf.predict(t))
        med = kmf.median_survival_time_
        out["groups"][lvl] = {"n": int(len(g)), "events": int(g[TARGET].sum()),
                              "survival_at": pts,
                              "median_survival": None if np.isinf(med) else float(med)}
    return out, fitters


def cox_model(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive Cox PH on tenure with a compact covariate set (full data)."""
    from lifelines import CoxPHFitter
    cols = ["tenure", TARGET, "Contract", "InternetService", "PaymentMethod",
            "MonthlyCharges", "SeniorCitizen", "Partner", "Dependents",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]
    d = df[cols].copy()
    for c in ["Partner", "Dependents", "PaperlessBilling"]:
        d[c] = (d[c] == "Yes").astype(int)
    for c in ["OnlineSecurity", "TechSupport"]:
        d[c] = (d[c] == "Yes").astype(int)
    d = pd.get_dummies(d, columns=["Contract", "InternetService", "PaymentMethod"],
                       drop_first=True, dtype=float)
    d.columns = [c.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") for c in d.columns]
    d["MonthlyCharges"] = d["MonthlyCharges"] / 10.0   # HR per +$10
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(d, duration_col="tenure", event_col=TARGET)
    s = cph.summary.reset_index().rename(columns={"covariate": "covariate"})
    s["concordance"] = cph.concordance_index_
    return s[["covariate", "coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p", "concordance"]]


def stratified_payment_by_contract(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Does the electronic-cheque effect persist within each contract type?"""
    rows = []
    for contract, g in df.groupby("Contract"):
        t = rate_table(g, "PaymentMethod")
        t.insert(0, "Contract", contract)
        rows.append(t)
    tab = pd.concat(rows, ignore_index=True)
    # Mantel-Haenszel: electronic cheque vs other, stratified by contract.
    from statsmodels.stats.contingency_tables import StratifiedTable
    tables = []
    for contract, g in df.groupby("Contract"):
        ec = (g["PaymentMethod"] == "Electronic check").astype(int)
        ct = pd.crosstab(ec, g[TARGET]).reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
        tables.append(ct)
    st = StratifiedTable(tables)
    crude = pd.crosstab((df["PaymentMethod"] == "Electronic check").astype(int), df[TARGET]) \
        .reindex(index=[1, 0], columns=[1, 0], fill_value=0).values
    crude_or = (crude[0, 0] * crude[1, 1]) / (crude[0, 1] * crude[1, 0])
    mh = {"crude_odds_ratio_echeck": float(crude_or),
          "mh_pooled_odds_ratio_echeck": float(st.oddsratio_pooled),
          "mh_or_ci": [float(x) for x in st.oddsratio_pooled_confint()],
          "homogeneity_p": float(st.test_equal_odds().pvalue)}
    return tab, mh


def segment_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The four opportunity segments of Table 3 (also the simple targeting rules in derived.py)."""
    m2m = df["Contract"] == "Month-to-month"
    fib = df["InternetService"] == "Fiber optic"
    return {
        "S1 Month-to-month + fibre": m2m & fib,
        "S2 Month-to-month, tenure <= 12": m2m & (df["tenure"] <= 12),
        "S3 Month-to-month + electronic cheque": m2m & (df["PaymentMethod"] == "Electronic check"),
        "S4 Fibre without security or tech support": fib & (df["OnlineSecurity"] == "No") & (df["TechSupport"] == "No"),
    }


def segments(df: pd.DataFrame) -> pd.DataFrame:
    defs = {**segment_masks(df), "All customers": pd.Series(True, index=df.index)}
    total_mrr = df["MonthlyCharges"].sum()
    churn_mrr = df.loc[df[TARGET] == 1, "MonthlyCharges"].sum()
    rows = []
    for name, mask in defs.items():
        g = df[mask]
        k, n = int(g[TARGET].sum()), int(len(g))
        lo, hi = wilson(k, n)
        rows.append({"segment": name, "n": n, "share_of_customers": n / len(df),
                     "churners": k, "churn_rate": k / n, "ci_lo": lo, "ci_hi": hi,
                     "mrr": float(g["MonthlyCharges"].sum()),
                     "mrr_share": float(g["MonthlyCharges"].sum() / total_mrr),
                     "mrr_at_risk": float(g.loc[g[TARGET] == 1, "MonthlyCharges"].sum()),
                     "share_of_all_churner_mrr": float(g.loc[g[TARGET] == 1, "MonthlyCharges"].sum() / churn_mrr),
                     "share_of_all_churners": k / int(df[TARGET].sum()),
                     "mean_monthly_charge": float(g["MonthlyCharges"].mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_churn_by_feature(rates: pd.DataFrame, overall: float) -> None:
    feats = ["Contract", "InternetService", "PaymentMethod", "tenure_band",
             "OnlineSecurity", "TechSupport", "SeniorCitizen", "PaperlessBilling"]
    titles = {"tenure_band": "Tenure (months)", "SeniorCitizen": "Senior citizen",
              "InternetService": "Internet service", "PaymentMethod": "Payment",
              "OnlineSecurity": "Online security", "TechSupport": "Tech support",
              "PaperlessBilling": "Paperless billing"}
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(COL_W * 1.3, 5.0), sharex=True)
    xmax = float(min(100, rates.loc[rates["feature"].isin(feats), "ci_hi"].max() * 100 + 8))
    for ax, f, letter in zip(axes.ravel(), feats, "abcdefgh"):
        t = rates[rates["feature"] == f].copy()
        if f == "tenure_band":
            order = [b[2] for b in TENURE_BANDS]
            t = t.set_index("level").loc[order].reset_index()
        elif f == "SeniorCitizen":
            t["level"] = t["level"].map({"0": "No", "1": "Yes"})
        elif f == "PaymentMethod":
            t["level"] = t["level"].str.replace(" (automatic)", "\n(auto)", regex=False)
        y = np.arange(len(t))[::-1]
        ax.errorbar(t["churn_rate"] * 100, y, xerr=[np.clip((t["churn_rate"] - t["ci_lo"]) * 100, 0, None),
                                                    np.clip((t["ci_hi"] - t["churn_rate"]) * 100, 0, None)],
                    fmt="o", ms=2.5, color=OKABE_ITO["blue"], ecolor=OKABE_ITO["blue"], capsize=3, lw=1.2)
        ax.axvline(overall * 100, color=OKABE_ITO["grey"], ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{lvl}\n(n={n:,})" for lvl, n in zip(t["level"], t["n"])], fontsize=8)
        ax.set_ylim(-0.6, len(t) - 0.4)
        ax.set_title(f"({letter}) {titles.get(f, f)}", fontsize=9.5, loc="left")
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("Churn rate (%)")
    fig.suptitle("Churn rate by feature level with 95% Wilson CIs (dashed line = overall rate)",
                 fontsize=10)
    save_fig(fig, "fig01_churn_by_feature")


def fig_km(fitters: dict, km: dict) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(COL_W * 0.8, 3.6))
    colours = {"Month-to-month": OKABE_ITO["vermillion"], "One year": OKABE_ITO["orange"],
               "Two year": OKABE_ITO["blue"]}
    for lvl in ["Month-to-month", "One year", "Two year"]:
        if lvl in fitters:
            fitters[lvl].plot_survival_function(ax=ax, ci_show=True, color=colours[lvl],
                                                ci_alpha=0.15, label=f"{lvl} (n={km['groups'][lvl]['n']:,})")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Proportion retained (Kaplan-Meier)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 72)
    ax.legend(loc="lower left")
    ptxt = "p < 0.001" if km["logrank_p"] < 0.001 else f"p = {km['logrank_p']:.2g}"
    ax.set_title(f"Retention by contract type (log-rank {ptxt})", fontsize=10)
    save_fig(fig, "fig02_km_contract")


def fig_segments(df: pd.DataFrame, heat: pd.DataFrame, counts: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq", ["#FFFFFF", OKABE_ITO["vermillion"]])
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 1.3, 3.5), gridspec_kw={"width_ratios": [1.25, 1.1, 1]})
    # (a) heatmap Contract x InternetService
    ax = axes[0]
    im = ax.imshow(heat.values * 100, cmap=seq, vmin=0, vmax=60, aspect="auto")
    ax.set_xticks(range(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(heat.shape[0]))
    ax.set_yticklabels(heat.index, fontsize=8.5)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}%\nn={int(counts.values[i, j]):,}", ha="center", va="center",
                    fontsize=7, color="black" if v < 40 else "white")
    ax.set_xlabel("Internet service")
    ax.set_ylabel("Contract")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Churn rate (%)", fontsize=8.5)
    panel_label(ax, "a")
    # (b) hexbin tenure x MonthlyCharges coloured by churn rate
    ax = axes[1]
    hb = ax.hexbin(df["tenure"], df["MonthlyCharges"], C=df[TARGET], reduce_C_function=np.mean,
                   gridsize=18, cmap=seq, mincnt=5, vmin=0, vmax=0.8, linewidths=0.2, edgecolors="#BBBBBB")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly charges ($)")
    ax.grid(False)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Churn rate (cells with n >= 5)", fontsize=8.5)
    panel_label(ax, "b")
    # (c) ECDF of tenure by churn
    ax = axes[2]
    for lab, colour, name in [(0, CHURN_COLOURS[0], "Stayed"), (1, CHURN_COLOURS[1], "Churned")]:
        s = np.sort(df.loc[df[TARGET] == lab, "tenure"].values)
        ax.step(s, np.arange(1, len(s) + 1) / len(s), where="post", color=colour, label=f"{name} ({len(s):,})")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Cumulative share of customers")
    ax.set_xlim(0, 72)
    for line in ax.get_lines():  # label the curves directly: the panel is too narrow for a legend box
        lab = line.get_label()
        if lab.startswith("Stayed"):
            ax.text(40, 0.28, lab, color=line.get_color(), fontsize=8, ha="left", va="top")
        elif lab.startswith("Churned"):
            ax.text(6, 0.93, lab, color=line.get_color(), fontsize=8, ha="left", va="bottom")
    panel_label(ax, "c")
    fig.suptitle("Two-way segments, tenure-price surface and tenure distributions", fontsize=10)
    save_fig(fig, "fig03_segments")


def fig_numeric(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(COL_W * 1.1, 3.0))
    for ax, c, letter in zip(axes, NUMERIC, "abc"):
        data = [df.loc[df[TARGET] == 0, c].dropna().values, df.loc[df[TARGET] == 1, c].dropna().values]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True, widths=0.8)
        for body, colour in zip(parts["bodies"], [CHURN_COLOURS[0], CHURN_COLOURS[1]]):
            body.set_facecolor(colour)
            body.set_alpha(0.6)
        for k in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[k].set_color("#333333")
            parts[k].set_linewidth(0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Stayed", "Churned"])
        unit = " (months)" if c == "tenure" else " ($)"
        ax.set_ylabel({"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges"}[c] + unit)
        panel_label(ax, letter)
    fig.suptitle("Numeric features by churn status (violin, median marked)", fontsize=10)
    save_fig(fig, "figS1_numeric_distributions")


def fig_correlation(corr: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list("div", [OKABE_ITO["blue"], "#F2F2F2", OKABE_ITO["vermillion"]])
    pretty = {"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges", "n_addons": "Add-on count", "SeniorCitizen": "Senior citizen"}
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(corr.values, cmap=div, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels([pretty.get(c, c) for c in corr.columns], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels([pretty.get(c, c) for c in corr.index], fontsize=8.5)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    ax.set_title("Spearman correlations (full data)", fontsize=10)
    save_fig(fig, "figS2_correlation")


# ----------------------------------------------------------------------------
def run(df: pd.DataFrame) -> dict:
    setup_style()
    df = df.copy()
    df["tenure_band"] = tenure_band(df["tenure"])
    n, k = len(df), int(df[TARGET].sum())
    lo, hi = wilson(k, n)
    total_mrr = float(df["MonthlyCharges"].sum())
    churn_mrr = float(df.loc[df[TARGET] == 1, "MonthlyCharges"].sum())
    overall = {
        "n": n, "churners": k, "churn_rate": k / n, "ci_lo": lo, "ci_hi": hi,
        "majority_accuracy": 1 - k / n,
        "total_mrr": total_mrr, "churner_mrr": churn_mrr, "churner_mrr_share": churn_mrr / total_mrr,
        "annualised_churner_revenue": churn_mrr * 12,
        "mean_monthly_charge_churn": float(df.loc[df[TARGET] == 1, "MonthlyCharges"].mean()),
        "mean_monthly_charge_stay": float(df.loc[df[TARGET] == 0, "MonthlyCharges"].mean()),
        "median_tenure_churn": float(df.loc[df[TARGET] == 1, "tenure"].median()),
        "median_tenure_stay": float(df.loc[df[TARGET] == 0, "tenure"].median()),
        "n_tenure0": int((df["tenure"] == 0).sum()),
    }
    # Churn rate by every level of every categorical (+ tenure band)
    rates = pd.concat([rate_table(df, c) for c in ALL_CATEGORICAL + ["tenure_band"]], ignore_index=True)
    save_csv(rates, "churn_by_category.csv")
    # Numeric summary and categorical frequencies
    save_csv(df[NUMERIC].describe().T.reset_index().rename(columns={"index": "feature"}), "numeric_summary.csv")
    # Associations
    assoc = associations(df)
    save_csv(assoc, "associations.csv")
    # Correlations (Spearman) incl. engineered n_addons
    eng = engineer_features(df)
    corr_cols = ["tenure", "MonthlyCharges", "TotalCharges", "n_addons", "SeniorCitizen", TARGET]
    corr = eng[corr_cols].corr(method="spearman")
    save_csv(corr.reset_index().rename(columns={"index": "feature"}), "correlations.csv")
    overall["pearson_tenure_totalcharges"] = float(df[["tenure", "TotalCharges"]].dropna().corr().iloc[0, 1])
    overall["pearson_monthly_totalcharges"] = float(df[["MonthlyCharges", "TotalCharges"]].dropna().corr().iloc[0, 1])
    overall["spearman_tenure_totalcharges"] = float(corr.loc["tenure", "TotalCharges"])
    # Kaplan-Meier + log-rank, Cox
    km, fitters = km_by_contract(df)
    save_json(km, "km_contract.json")
    try:
        cox = cox_model(df)
        save_csv(cox, "cox_summary.csv")
    except Exception as e:  # pragma: no cover
        save_json({"error": str(e)}, "cox_summary_error.json")
    # Two-way heatmap
    heat = df.pivot_table(index="Contract", columns="InternetService", values=TARGET, aggfunc="mean")
    counts = df.pivot_table(index="Contract", columns="InternetService", values=TARGET, aggfunc="size")
    h = heat.stack().rename("churn_rate").reset_index()
    h["n"] = counts.stack().values
    save_csv(h, "segment_heatmap_contract_internet.csv")
    heat2 = df.pivot_table(index="Contract", columns="PaymentMethod", values=TARGET, aggfunc="mean")
    counts2 = df.pivot_table(index="Contract", columns="PaymentMethod", values=TARGET, aggfunc="size")
    h2 = heat2.stack().rename("churn_rate").reset_index()
    h2["n"] = counts2.stack().values
    save_csv(h2, "segment_heatmap_contract_payment.csv")
    # Confounding check
    strat, mh = stratified_payment_by_contract(df)
    save_csv(strat, "stratified_paymentmethod_by_contract.csv")
    save_json(mh, "mantel_haenszel_echeck.json")
    # Proxy associations between sensitive attributes and other features (Cramer's V)
    prox = {}
    for a in ["SeniorCitizen", "gender", "Partner", "Dependents"]:
        for b in ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling"]:
            v, chi2, p, dof = cramers_v(pd.crosstab(df[a], df[b]).values)
            prox[f"{a}__{b}"] = {"cramers_v": v, "p": p}
    save_json(prox, "proxy_associations.json")
    # Opportunity segments
    seg = segments(df)
    save_csv(seg, "segments.csv")
    save_json(overall, "eda_overall.json")
    # Figures
    fig_churn_by_feature(rates, overall["churn_rate"])
    fig_km(fitters, km)
    fig_segments(df, heat, counts)
    fig_numeric(df)
    fig_correlation(corr.drop(index=TARGET, columns=TARGET))
    return overall


if __name__ == "__main__":
    from .clean import clean
    from .load import build_manifest
    raw, _ = build_manifest()
    d, _ = clean(raw)
    print(run(d))
