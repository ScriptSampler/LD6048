"""Render the report's data tables as Markdown directly from results/, so that
no number is transcribed by hand. build_report.py substitutes the placeholders
{{T2}}, {{T3}}, {{T5A}}, {{T5B}}, {{T6}}, {{T7}}, {{B1}}, {{B2}}, {{C1}}..{{C7}} in
report/draft_template.md."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import INNER_SPLITS, MODEL_LABELS, RESULTS
from .io_utils import load_json

PRETTY_FEAT = {"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges", "gender": "Gender",
               "SeniorCitizen": "Senior citizen", "Partner": "Partner", "Dependents": "Dependents", "PhoneService": "Phone service",
               "MultipleLines": "Multiple lines", "InternetService": "Internet service", "OnlineSecurity": "Online security",
               "OnlineBackup": "Online backup", "DeviceProtection": "Device protection", "TechSupport": "Tech support",
               "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "Contract": "Contract",
               "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method", "tenure_band": "Tenure band (months)"}


def pf(name: str) -> str:
    return PRETTY_FEAT.get(name, name)
ORDER = ["Majority", "LogReg", "DecisionTree", "RandomForest", "XGBoost", "MLP", "Voting"]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


NBSP = " "  # joins "p = 0.15", "n = 222" and "[0.50, 0.60]" so Word does not break inside them


def num(x, spec=".3f", sign=False) -> str:
    """A number as printed in the tables: a minus sign (U+2212), as in the text, and no negative zero."""
    x = float(f"{x:{spec.replace(',', '')}}") + 0.0  # round as printed, then turn -0.0 into 0.0
    return f"{x:{'+' if sign and x else ''}{spec}}".replace("-", "−")


def ci(lo, hi, spec=".3f", sign=False) -> str:
    return f"[{num(lo, spec, sign)},{NBSP}{num(hi, spec, sign)}]"


def p_text(p) -> str:
    """'(p = 0.15)', '(p = 0.003)' or '(p < 0.001)', without line breaks inside."""
    rel = f"<{NBSP}0.001" if p < 0.001 else f"={NBSP}{p:.2f}" if p >= 0.01 else f"={NBSP}{p:.3f}"
    return f"(p{NBSP}{rel})"


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else num(x, f".{d}f")


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def t2_associations() -> str:
    a = pd.read_csv(RESULTS / "associations.csv")
    rows = []
    for _, r in a.iterrows():
        if r["type"] == "categorical":
            eff = f"V = {r['cramers_v']:.2f}"
        else:
            eff = f"r = {num(r['point_biserial_r'], '.2f')} (rank-biserial {num(r['rank_biserial'], '.2f')})"
        if r["p_value"] >= 0.001:
            eff += " " + p_text(r["p_value"])
        rows.append([pf(r["feature"]), r["type"], eff, f"{r['mutual_info_bits']:.3f}"])
    return md_table(["Feature", "Type", "Effect size", "Mutual information (bits)"], rows)


def t3_segments() -> str:
    s = pd.read_csv(RESULTS / "segments.csv")
    levers = {"S1": "Contract migration; review fibre pricing and service quality",
              "S2": "Early-tenure onboarding and check-ins",
              "S3": "Auto-pay migration incentives",
              "S4": "Security and support bundles"}
    rows = []
    for _, r in s.iterrows():
        if r["segment"].startswith("All"):
            continue
        key = r["segment"].split()[0]
        rows.append([r["segment"].replace("<=", "≤"), f"{int(r['n']):,} ({pct(r['share_of_customers'])})",
                     f"{pct(r['churn_rate'])} {ci(r['ci_lo'] * 100, r['ci_hi'] * 100, '.1f')}",
                     pct(r["share_of_all_churner_mrr"]), levers[key]])
    return md_table(["Segment", "Customers", f"Churn rate (95%{NBSP}CI)", "Share of churner MRR", "Retention lever"], rows)


def t5a_cv() -> str:
    """Table 5a: selection evidence on the training split (nested CV and out-of-fold expected profit)."""
    cv = pd.read_csv(RESULTS / "nested_cv_summary.csv")
    v = pd.read_csv(RESULTS / "voting_outer_cv_scores.csv")
    thr = load_json("thresholds.json")
    gaps = pd.read_csv(RESULTS / "profit_gap_bootstrap.csv").query("split == 'oof'").set_index("comparison")
    rows = []
    for m in ORDER:
        if m == "Voting":
            pr, prs, roc, rocs = v.pr_auc.mean(), v.pr_auc.std(ddof=1), v.roc_auc.mean(), v.roc_auc.std(ddof=1)
        else:
            r = cv[(cv.model == m) & (cv.variant == "tuned")].iloc[0]
            pr, prs, roc, rocs = r.pr_auc_mean, r.pr_auc_std, r.roc_auc_mean, r.roc_auc_std
        g = gaps.loc[f"{m} vs LogReg"] if f"{m} vs LogReg" in gaps.index else None
        rows.append([MODEL_LABELS[m], f"{pr:.3f} ± {prs:.3f}", f"{roc:.3f} ± {rocs:.3f}",
                     "n/a" if m == "Majority" else f"{thr[m]['operating']:.2f}",
                     num(thr[m]["profit_at_opt_oof"], ",.0f"),
                     "" if g is None else f"{num(g.profit_diff, ',.0f', True)} {ci(g.ci_lo, g.ci_hi, ',.0f', True)}"])
    return md_table(["Model", "PR-AUC (mean ± SD)", "ROC-AUC (mean ± SD)", "Profit-optimal threshold",
                     "Expected profit ($)", "Gap to logistic regression ($) [95% CI]"], rows)


def t5b_test() -> str:
    """Table 5b: the single hold-out evaluation, each model at its out-of-fold profit-optimal threshold."""
    tm = pd.read_csv(RESULTS / "test_metrics.csv")
    pt = pd.read_csv(RESULTS / "profit_test.csv")
    rows = []
    for m in ORDER:
        t = tm[(tm.model == m) & (tm.threshold_kind == "operating")].iloc[0]
        profit = pt[(pt.model == m) & (pt.threshold_kind == "operating")]["profit"].iloc[0]
        rows.append([MODEL_LABELS[m], f"{t.accuracy:.3f}", f"{t.precision:.3f}", f"{t.recall:.3f}", f"{t.f1:.3f}",
                     f"{t.roc_auc:.3f} {ci(t.roc_auc_lo, t.roc_auc_hi)}",
                     f"{t.pr_auc:.3f} {ci(t.pr_auc_lo, t.pr_auc_hi)}",
                     f"{t.brier:.3f}", f"{t.lift_top_decile:.2f}", num(profit, ",.0f")])
    return md_table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC [95% CI]", "PR-AUC [95% CI]",
                     "Brier", "Top-decile lift", "Expected profit ($)"], rows)


def t6_tests() -> str:
    cvt = pd.read_csv(RESULTS / "stat_tests_cv.csv")
    tt = pd.read_csv(RESULTS / "stat_tests_test.csv")
    gaps = pd.read_csv(RESULTS / "profit_gap_bootstrap.csv").query("split == 'test'").set_index("comparison")

    def gap(comp):
        g = gaps.loc[comp]
        return f"{num(g.profit_diff, ',.0f', True)} {ci(g.ci_lo, g.ci_hi, ',.0f', True)}"
    rows = []
    for m in ["DecisionTree", "RandomForest", "XGBoost", "MLP", "Voting"]:
        comp = f"{m} vs LogReg"
        c_pr = cvt[(cvt.comparison == comp) & (cvt.metric == "pr_auc")].iloc[0]
        c_roc = cvt[(cvt.comparison == comp) & (cvt.metric == "roc_auc")].iloc[0]
        dl = tt[(tt.comparison == comp) & (tt.test == "DeLong")].iloc[0]
        pb = tt[(tt.comparison == comp) & (tt.test == "paired bootstrap") & (tt.metric == "pr_auc")].iloc[0]
        mc = tt[(tt.comparison == comp) & (tt.metric == "accuracy@operating")].iloc[0]
        rows.append([f"{MODEL_LABELS[m]} vs logistic regression",
                     f"{num(c_pr.mean_diff, '.3f', True)} {p_text(c_pr.p)}",
                     f"{num(c_roc.mean_diff, '.4f', True)} {p_text(c_roc.p)}",
                     f"{num(dl['diff'], '.4f', True)} {p_text(dl.p)}",
                     f"{num(pb['diff'], '.3f', True)} {ci(pb.ci_lo, pb.ci_hi, '.3f', True)}",
                     f"{mc.p:.2f}", gap(comp)])
    rows.append(["Logistic regression: F1 vs profit threshold", "", "", "", "", "",
                 gap("LogReg at F1-optimal vs operating threshold")])
    tune = []
    for m in ["LogReg", "DecisionTree", "RandomForest", "XGBoost", "MLP"]:
        r = cvt[cvt.comparison == f"{m} tuned vs default"].iloc[0]
        tune.append([f"{MODEL_LABELS[m]}: tuned vs default", f"{num(r.mean_diff, '.3f', True)} {p_text(r.p)}", "", "", "", "", ""])
    # hyphenated metric names first, so Word breaks the headers at spaces; the caption names the tests
    return md_table(["Comparison", "ΔPR-AUC (CV)", "ΔROC-AUC (CV)", "ΔROC-AUC (test, DeLong)", "ΔPR-AUC (test) [95% CI]",
                     "McNemar p", "Δprofit (test), $ [95% CI]"], rows + tune)


def t7_fairness() -> str:
    fm = pd.read_csv(RESULTS / "fairness_metrics.csv")
    fs = pd.read_csv(RESULTS / "fairness_summary.csv")
    best = load_json("model_ranking.json")["best_single_model"]
    fm, fs = fm[fm.model == best], fs[fs.model == best]
    rows = []
    for attr in ["SeniorCitizen", "gender", "Partner", "Dependents"]:
        s = fs[fs.attribute == attr].iloc[0]
        groups = list(dict.fromkeys(fm[fm.attribute == attr]["group"]))
        for i, g in enumerate(groups):
            gg = fm[(fm.attribute == attr) & (fm.group == g)].set_index("metric")
            n = int(gg["n"].iloc[0])

            def cell(m):
                return f"{gg.loc[m, 'value']:.2f} {ci(gg.loc[m, 'ci_lo'], gg.loc[m, 'ci_hi'], '.2f')}"
            rows.append([pf(attr) if i == 0 else "", f"{g} (n{NBSP}={NBSP}{n:,})", f"{gg.loc['observed_rate', 'value']:.2f}",
                         f"{gg.loc['mean_pred', 'value']:.2f}",
                         cell("recall"), cell("fpr"), cell("selection_rate"), f"{gg.loc['precision', 'value']:.2f}",
                         f"{s.demographic_parity_difference:.2f}" if i == 0 else "",
                         f"{s.equalized_odds_difference:.2f}" if i == 0 else ""])
    # short headers leave the interval columns room for "[0.50, 0.60]" on one line; the caption spells them out
    return md_table(["Attribute", "Group", "Base rate", "Mean predicted", "Recall", "FPR", "Selection rate", "Precision",
                     "DP diff.", "EO diff."], rows)


def support(v: dict) -> str:
    """Sampled range of a scipy.stats distribution saved by models._space_to_json, written as an
    interval rather than in scipy's call syntax (uniform(loc, scale) samples [loc, loc + scale];
    randint(low, high) samples the integers low to high - 1)."""
    d, args = v["distribution"], v["args"]
    if v.get("kwds") or len(args) != 2:
        raise ValueError(f"Table B1 shows two positional arguments only, not {args} {v.get('kwds') or ''}")
    a, b = args
    if d == "uniform":
        return f"U[{plain(a)}, {plain(round(a + b, 12))}]"
    if d == "loguniform":
        return f"log-U[{plain(a)}, {plain(b)}]"
    if d == "randint":
        return f"integers {int(a)}–{int(b) - 1}"
    raise ValueError(f"no interval rule for distribution {d}")


def plain(x):
    """Plain decimal for every number in Tables B1 and B2: 1e-05 -> "0.00001", 100.0 -> "100"."""
    return np.format_float_positional(x, trim="-") if isinstance(x, (float, np.floating)) else str(x)


def b1_search_spaces() -> str:
    sp = load_json("search_spaces.json")
    rows = []
    for m in ["LogReg", "DecisionTree", "RandomForest", "XGBoost", "MLP"]:
        for k, v in sp[m].items():
            desc = "choice of " + ", ".join(map(plain, v)) if isinstance(v, list) else support(v)
            rows.append([MODEL_LABELS[m], k.replace("model__", ""), desc])
    return md_table(["Model", "Hyperparameter", "Search distribution"], rows)


def b2_best_params() -> str:
    bp = load_json("best_params.json")
    # search budgets as the searches ran: inner fits per outer fold (recorded by models.run) / inner folds
    fits = pd.read_csv(RESULTS / "nested_cv_scores.csv").query("variant == 'tuned'").groupby("model")["n_inner_fits"].unique()
    rows = []
    for m in ["LogReg", "DecisionTree", "RandomForest", "XGBoost", "MLP"]:
        if m not in fits.index or len(fits[m]) != 1 or pd.isna(fits[m][0]):
            raise ValueError(f"{m}: nested_cv_scores.csv has no single inner-fit count ({list(fits.get(m, []))})")
        n_iter, rem = divmod(int(fits[m][0]), INNER_SPLITS)
        if rem:
            raise ValueError(f"{m}: inner fits {list(fits[m])} do not match {INNER_SPLITS} inner folds")
        params = {k.replace("model__", ""): (plain(float(f"{v:.4g}")) if isinstance(v, float) else v)
                  for k, v in bp[m].items() if not k.startswith("_")}
        rows.append([MODEL_LABELS[m], str(n_iter), "; ".join(f"{k} = {v}" for k, v in params.items()), f"{bp[m]['_inner_cv_pr_auc']:.3f}"])
    return md_table(["Model", f"Configurations searched (× {INNER_SPLITS} inner folds)", "Selected configuration (full training split)",
                     "Inner-CV PR-AUC"], rows)


def c1_nested_full() -> str:
    cv = pd.read_csv(RESULTS / "nested_cv_summary.csv")
    rows = []
    for m in ["Majority", "LogReg", "DecisionTree", "RandomForest", "XGBoost", "MLP"]:
        for var in ["default", "tuned"]:
            r = cv[(cv.model == m) & (cv.variant == var)].iloc[0]
            rows.append([MODEL_LABELS[m], var, f"{r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f}", f"{r.roc_auc_mean:.3f} ± {r.roc_auc_std:.3f}",
                         f"{r.brier_mean:.3f}", f"{r.f1_mean:.3f}", f"{r.recall_mean:.3f}", f"{r.precision_mean:.3f}",
                         f"{r.accuracy_mean:.3f}", num(r.mcc_mean), num(r.train_val_gap_pr_auc), f"{r.fit_seconds_mean:.2f}"])
    return md_table(["Model", "Variant", "PR-AUC", "ROC-AUC", "Brier", "F1 (0.5)", "Recall (0.5)", "Precision (0.5)", "Accuracy (0.5)",
                     "MCC (0.5)", "PR-AUC gap", "Fit seconds"], rows)


def c2_vif() -> str:
    v = pd.read_csv(RESULTS / "vif.csv").head(14)
    log_tc = load_json("chosen_config.json").get("log_totalcharges")  # the linear design holds log1p(TotalCharges)

    def label(feat):
        if feat == "TotalCharges" and log_tc:
            return "Total charges (log)"
        return pf(feat.partition("_")[0]) + (": " + feat.partition("_")[2] if "_" in feat else "")
    vi = pd.read_csv(RESULTS / "vif_interp.csv").set_index("feature")["vif"]  # explain.lr_interpretation
    return md_table(["Feature (linear design)", "VIF, scoring model", "VIF without monthly charges"],
                    [[label(r.feature), f"{r.vif:.1f}", f"{vi[r.feature]:.1f}" if r.feature in vi.index else "removed"]
                     for r in v.itertuples()])


def c3_odds() -> str:
    o = pd.read_csv(RESULTS / "lr_odds_ratios_interp.csv")  # the interpretation model (explain.lr_interpretation)
    return md_table(["Feature", "Coefficient", "Odds ratio", "95% bootstrap CI", "Unit"],
                    [[r.label if isinstance(r.label, str) else r.feature, num(r.coef, ".3f", True), f"{r.odds_ratio:.3f}", ci(r.or_ci_lo, r.or_ci_hi), r.unit] for r in o.itertuples()])


def c4_calibration() -> str:
    c = pd.read_csv(RESULTS / "calibration_metrics.csv")
    ch = pd.read_csv(RESULTS / "calibration_choice.csv").set_index("model")
    rows = []
    for m in ORDER:
        chosen = ch.loc[m, "chosen"] if m in ch.index else "none"
        parts = ["oof_uncalibrated", "oof", "test_uncalibrated", "test"] if chosen != "none" else ["oof", "test"]
        names = ({"oof": "OOF (as used)", "oof_uncalibrated": "OOF, before calibration", "test": "test (as used)",
                  "test_uncalibrated": "test, before calibration"} if chosen != "none" else {"oof": "OOF", "test": "test"})
        for part in parts:
            r = c[(c.model == m) & (c.part == part)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([MODEL_LABELS[m], names[part], f"{r.brier:.4f}", f"{r.ece:.4f}", f(r.slope), f(r.intercept), chosen])
    return md_table(["Model", "Predictions", "Brier", "ECE", "Calibration slope", "Intercept", "Calibration applied"], rows)


def c5_profit() -> str:
    p = pd.read_csv(RESULTS / "profit_test.csv")
    rules = pd.read_csv(RESULTS / "rule_baselines.csv").query("split == 'test' and rule.str.startswith('S')", engine="python")
    kinds = {"operating": "Profit-optimal (OOF)", "f1_opt": "F1-optimal (OOF)", "default": "Default 0.5",
             "all": "Contact everyone", "none": "Contact no one", "oracle": "Contact the churners"}
    n = len(pd.read_csv(RESULTS / "test_predictions.csv", usecols=["y"]))
    rows = []
    for r in p.itertuples():
        # the majority class and the three benchmark rules (everyone, no one, oracle) use no threshold
        no_thr = r.model == "Majority" or r.threshold_kind in ("all", "none", "oracle")
        rows.append([r.model if r.model not in MODEL_LABELS else MODEL_LABELS[r.model], kinds[r.threshold_kind],
                     "n/a" if no_thr else f"{r.threshold:.2f}",
                     f"{r.n_contacted:,}", pct(r.share_contacted), f"{r.recall:.3f}", f"{r.precision:.3f}", num(r.profit, ",.0f"), num(r.profit_per_customer, ".2f")])
    # derived.ev_targeting: the selected model, contacting each customer whose expected profit is positive
    best = load_json("model_ranking.json")["best_single_model"]
    for r in pd.read_csv(RESULTS / "ev_targeting.csv").query("split == 'test'").itertuples():
        rows.append([MODEL_LABELS[best], "Positive expected profit", "per customer", f"{r.n_contacted:,}", pct(r.share_contacted),
                     f"{r.recall:.3f}", f"{r.precision:.3f}", num(r.profit, ",.0f"), num(r.profit / n, ".2f")])
    for r in rules.itertuples():  # derived.rule_baselines: contact every customer in one Table 3 segment
        rows.append([f"Segment rule {r.rule.split()[0]}", r.rule.split(" ", 1)[1].replace("<=", "≤"), "n/a",
                     f"{r.n_contacted:,}", pct(r.share_contacted), f"{r.recall:.3f}", f"{r.precision:.3f}",
                     num(r.profit, ",.0f"), num(r.profit / n, ".2f")])
    return md_table(["Model or rule", "Contact rule", "Threshold", "Contacted", "Share contacted", "Recall", "Precision",
                     "Expected profit ($)", "Profit per customer ($)"], rows)


def c6_errors() -> str:
    e = pd.read_csv(RESULTS / "error_analysis.csv")
    e = e[e.feature.isin(["Contract", "InternetService", "PaymentMethod", "tenure_band"])].copy()
    forder = {"Contract": 0, "InternetService": 1, "PaymentMethod": 2, "tenure_band": 3}
    border = {"0-6": 0, "7-12": 1, "13-24": 2, "25-48": 3, "49-72": 4}
    e["_f"] = e.feature.map(forder)
    e["_b"] = [border.get(str(l), 0) if f == "tenure_band" else 0 for f, l in zip(e.feature, e.level)]
    e = e.sort_values(["_f", "_b", "level"])
    return md_table(["Feature", "Level", "Test customers", "Churners", "Churners missed", "Stayers flagged", "Mean predicted probability (churners)"],
                    [[pf(r.feature), r.level, f"{int(r.n):,}", f"{int(r.churners):,}", pct(r.fn_rate_among_churners), pct(r.fp_rate_among_stayers), f"{r.mean_p_churners:.2f}"] for r in e.itertuples()])


def c7_blind() -> str:
    """Table C7: the selected model refitted without gender, and without gender and senior status
    (fairness.blind_variants), each at its own out-of-fold profit-optimal threshold, on the test set."""
    b = pd.read_csv(RESULTS / "fairness_blind.csv")

    def pair(r, a, x, y, metric):
        return f"{r[f'{a}: {x}: {metric}']:.2f} vs {r[f'{a}: {y}: {metric}']:.2f}"
    rows = [[r["variant"].capitalize(), f"{r['threshold']:.2f}", f"{r['test_pr_auc']:.3f}", num(r["test_profit"], ",.0f"),
             pair(r, "SeniorCitizen", "Senior", "Not senior", "selection_rate"), pair(r, "SeniorCitizen", "Senior", "Not senior", "fpr"),
             pair(r, "gender", "Male", "Female", "selection_rate"), pair(r, "Dependents", "Dependents", "No dependents", "recall")]
            for _, r in b.iterrows()]
    return md_table(["Inputs", "Threshold (OOF)", "PR-AUC (test)", "Expected profit, test ($)", "Selection rate, senior vs not",
                     "False-positive rate, senior vs not", "Selection rate, male vs female", "Recall, dependents vs not"], rows)


def all_tables() -> dict[str, str]:
    return {"T2": t2_associations(), "T3": t3_segments(), "T5A": t5a_cv(), "T5B": t5b_test(), "T6": t6_tests(),
            "T7": t7_fairness(), "B1": b1_search_spaces(), "B2": b2_best_params(), "C1": c1_nested_full(), "C2": c2_vif(),
            "C3": c3_odds(), "C4": c4_calibration(), "C5": c5_profit(), "C6": c6_errors(), "C7": c7_blind()}


if __name__ == "__main__":
    for k, v in all_tables().items():
        print(f"\n### {k}\n{v}")
