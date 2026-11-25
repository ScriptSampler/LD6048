"""Derived quantities quoted in the report and DEFENCE_NOTES.md (ratios, shares, differences,
break-even save rates, the dependents' parity threshold, per-customer expected-value targeting), so
that numbers computed for the text are saved in results/ rather than worked out by hand. Written to
results/derived.json (some fill report placeholders directly), with the dependents' threshold scan in
results/dependents_threshold_scan.csv, the expected-value rule in results/ev_targeting.csv and the
engineered-feature tests in results/feature_tests.csv."""
from __future__ import annotations

import json
import numpy as np

import pandas as pd

from .config import RESULTS
from .io_utils import load_json, save_csv, save_json


def recall_parity(d: dict, best: str, column: str, group: str, other: str, scan_file: str | None = None) -> None:
    """Recall parity for customers with column == 'Yes', whose churners the model misses more often: their
    own threshold is the first 0.01 step down from the operating threshold at which their out-of-fold
    recall reaches that of the other customers. It is chosen on the out-of-fold training predictions only
    (the column and operating threshold behind thresholds.json); the hold-out test set plays no part in
    the choice and is used to report results after it. Keys are written to d as '<group>_recall_parity_*'."""
    key = f"{group}_recall_parity"
    try:
        from .metrics import campaign_profit
        thr = load_json("thresholds.json")[best]["operating"]

        def load_split(pred_file, x_file, col):
            pr = pd.read_csv(RESULTS / pred_file)
            X = pd.read_csv(RESULTS.parent / "data" / "processed" / x_file, index_col=0).loc[pr["row"].values]
            return pr["y"].values, pr[col].values, pr["MonthlyCharges"].values, X[column].values == "Yes"

        def share(hit, base):
            return float(hit.sum() / max(1, base.sum()))

        def contact(p, grp, t_grp):
            return np.where(grp, p >= t_grp, p >= thr)

        # out-of-fold column p_<best>_final carries the same calibration as the test column p_<best>
        yo, po, mo, g_o = load_split("oof_predictions.csv", "train.csv", f"p_{best}_final")
        target = share((po >= thr) & (yo == 1) & ~g_o, (yo == 1) & ~g_o)
        t_grp = next((k / 100 for k in range(round(thr * 100), 1, -1)  # 0.01 steps down from thr, exact
                      if share((po >= k / 100) & (yo == 1) & g_o, (yo == 1) & g_o) >= target), None)
        if t_grp is None:
            raise ValueError(f"no threshold reaches out-of-fold recall parity for the {column} group")
        # campaign_profit uses scores only through p >= thr, so a 0/1 contact vector at 0.5 is exact
        d[f"{key}_threshold"] = t_grp
        d[f"{key}_target_recall_oof"] = target
        d[f"{key}_profit_change_oof"] = (campaign_profit(yo, contact(po, g_o, t_grp).astype(float), mo, 0.5)["profit"]
                                         - campaign_profit(yo, po, mo, thr)["profit"])
        d[f"{key}_oof_recall_{group}"] = share((po >= t_grp) & (yo == 1) & g_o, (yo == 1) & g_o)
        yt, pt_, mt, g_t = load_split("test_predictions.csv", "test.csv", f"p_{best}")
        c = contact(pt_, g_t, t_grp)
        adj = campaign_profit(yt, c.astype(float), mt, 0.5)
        d[f"{key}_profit_change"] = adj["profit"] - campaign_profit(yt, pt_, mt, thr)["profit"]
        d[f"{key}_test_n_contacted"] = adj["n_contacted"]
        for g, mask in ((group, g_t), (other, ~g_t)):
            d[f"{key}_test_recall_{g}"] = share(c & (yt == 1) & mask, (yt == 1) & mask)
            d[f"{key}_test_fpr_{g}"] = share(c & (yt == 0) & mask, (yt == 0) & mask)
        if scan_file:
            # Every 0.01 step for the group from the operating threshold down to the parity threshold:
            # how the out-of-fold and test profit and the group's test miss rate move along the way
            scan = []
            for k in range(round(thr * 100), round(t_grp * 100) - 1, -1):
                co, ct = contact(po, g_o, k / 100), contact(pt_, g_t, k / 100)
                scan.append({f"{group}_threshold": k / 100,
                             f"oof_recall_{group}": share(co & (yo == 1) & g_o, (yo == 1) & g_o),
                             "oof_profit_change": (campaign_profit(yo, co.astype(float), mo, 0.5)["profit"]
                                                   - campaign_profit(yo, po, mo, thr)["profit"]),
                             "test_profit_change": (campaign_profit(yt, ct.astype(float), mt, 0.5)["profit"]
                                                    - campaign_profit(yt, pt_, mt, thr)["profit"]),
                             f"test_miss_rate_{group}": 1 - share(ct & (yt == 1) & g_t, (yt == 1) & g_t)})
            save_csv(pd.DataFrame(scan), scan_file)
    except Exception as e:
        d[f"{key}_error"] = str(e)


def _split_frame(best: str, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Attributes, labels, scores (the column behind thresholds.json) and monthly charges for the training
    split's out-of-fold predictions ('oof') or the test set ('test'), in prediction-file order."""
    pred_file, x_file, col = (("oof_predictions.csv", "train.csv", f"p_{best}_final") if split == "oof"
                              else ("test_predictions.csv", "test.csv", f"p_{best}"))
    pr = pd.read_csv(RESULTS / pred_file)
    X = pd.read_csv(RESULTS.parent / "data" / "processed" / x_file, index_col=0).loc[pr["row"].values]
    return X, pr["y"].values, pr[col].values, pr["MonthlyCharges"].values


def rule_baselines(d: dict, best: str) -> None:
    """Simple targeting rules as a benchmark for the model (Part 4): contact every customer in one of
    Table 3's segments. The best rule is chosen on the training split (its expected profit over the
    training customers, against the model's out-of-fold profit) and then scored once on the test set,
    like the model; the model's gain over it gets a paired bootstrap 95% interval (the same resampled
    customers for both). Written to results/rule_baselines.csv and derived.json (best_rule_*,
    model_gain_over_best_rule_*)."""
    try:
        from .config import N_BOOT, SEED
        from .eda import segment_masks
        from .metrics import campaign_profit
        thr = load_json("thresholds.json")[best]["operating"]
        rows, contacts = [], {}
        for split in ("oof", "test"):
            X, y, p, m = _split_frame(best, split)
            for name, mask in [*segment_masks(X).items(), (f"{best} at its operating threshold", pd.Series(p >= thr))]:
                contacts[(split, name)] = (y, mask.values.astype(float), m)
                r = campaign_profit(y, mask.values.astype(float), m, 0.5)
                rows.append({"split": split, "rule": name, **{k: r[k] for k in
                             ("n_contacted", "share_contacted", "recall", "precision", "profit")}})
        df = pd.DataFrame(rows)
        save_csv(df, "rule_baselines.csv", float_format="%.10g")
        rules = df[df.rule.str.startswith("S")]
        best_rule = rules[rules.split == "oof"].sort_values("profit").iloc[-1]["rule"]
        model = df[~df.rule.str.startswith("S")].set_index("split")["profit"]
        rule = rules[rules.rule == best_rule].set_index("split")
        d["best_rule"] = best_rule
        d["best_rule_profit_test"] = float(rule.loc["test", "profit"])
        d["best_rule_n_contacted_test"] = int(rule.loc["test", "n_contacted"])
        d["model_gain_over_best_rule_test"] = float(model["test"] - rule.loc["test", "profit"])
        d["model_gain_over_best_rule_oof"] = float(model["oof"] - rule.loc["oof", "profit"])
        rng = np.random.RandomState(SEED)
        for split in ("oof", "test"):
            y, c_model, m = contacts[(split, f"{best} at its operating threshold")]
            c_rule = contacts[(split, best_rule)][1]
            boots = []
            for _ in range(N_BOOT):
                i = rng.randint(0, len(y), len(y))
                boots.append(campaign_profit(y[i], c_model[i], m[i], 0.5)["profit"] - campaign_profit(y[i], c_rule[i], m[i], 0.5)["profit"])
            d[f"model_gain_over_best_rule_{split}_ci_lo"] = float(np.quantile(boots, 0.025))
            d[f"model_gain_over_best_rule_{split}_ci_hi"] = float(np.quantile(boots, 0.975))
    except Exception as e:
        d["rule_baselines_error"] = str(e)


def combined_parity(d: dict, best: str) -> None:
    """Both group thresholds applied together (Part 5): customers with dependents at the dependents'
    parity threshold, other customers with a partner at the partners' threshold, everyone else at the
    operating threshold. Records the change in expected profit and contacts, and the miss and
    false-positive rates of the groups the audit discusses, out of fold and on test."""
    try:
        from .metrics import campaign_profit
        thr = load_json("thresholds.json")[best]["operating"]
        t_dep, t_par = d["dependents_recall_parity_threshold"], d["partner_recall_parity_threshold"]
        for split in ("oof", "test"):
            X, y, p, m = _split_frame(best, split)
            dep, par = X["Dependents"].values == "Yes", X["Partner"].values == "Yes"
            sen = X["SeniorCitizen"].values == 1
            c = p >= np.where(dep, t_dep, np.where(par, t_par, thr))
            new, old = campaign_profit(y, c.astype(float), m, 0.5), campaign_profit(y, p, m, thr)
            d[f"combined_parity_profit_change_{split}"] = new["profit"] - old["profit"]
            d[f"combined_parity_n_contacted_{split}"] = new["n_contacted"]
            for g, mask in (("dependents", dep), ("partner", par), ("senior", sen), ("not_senior", ~sen)):
                d[f"combined_parity_{split}_miss_rate_{g}"] = float(1 - (c & (y == 1) & mask).sum() / max(1, ((y == 1) & mask).sum()))
                d[f"combined_parity_{split}_fpr_{g}"] = float((c & (y == 0) & mask).sum() / max(1, ((y == 0) & mask).sum()))
    except Exception as e:
        d["combined_parity_error"] = str(e)


def feature_tests(d: dict) -> None:
    """Part 3's engineered-feature tests: each variant's PR-AUC gain over no engineered features on the
    same 5 x 2 training folds, with the corrected resampled t-test. models.run_ablations decides each
    single candidate by this test (chosen_config.json); this file adds the all-candidates and chosen
    variants and the t statistics. Written to results/feature_tests.csv and derived.json (feature_test_*)."""
    try:
        from .metrics import corrected_resampled_ttest
        n = load_json("split_summary.json")["n_train"]
        f = pd.read_csv(RESULTS / "ablation_folds.csv").query("ablation == 'feature_engineering'")
        rows = []
        for (model, variant), g in f.groupby(["model", "variant"], sort=False):
            if variant == "none":
                continue
            base = f[(f.model == model) & (f.variant == "none")].sort_values("fold")["pr_auc"].values
            t = corrected_resampled_ttest(g.sort_values("fold")["pr_auc"].values, base, 0.8 * n, 0.2 * n)
            rows.append({"model": model, "variant": variant, **t})
            d[f"feature_test_{model}_{variant.lstrip('+')}_gain"], d[f"feature_test_{model}_{variant.lstrip('+')}_p"] = t["mean_diff"], t["p"]
        save_csv(pd.DataFrame(rows), "feature_tests.csv", float_format="%.10g")
    except Exception as e:
        d["feature_tests_error"] = str(e)


def ev_targeting(d: dict, best: str) -> None:
    """Per-customer expected-value targeting (Part 4): contact a customer when the expected profit of
    contacting them is positive under the cost model of metrics.campaign_profit, given their own
    predicted probability p and monthly charge: p * [save_rate * (value - offer) - contact] >
    (1 - p) * (leakage * offer + contact), with value = margin * horizon * monthly charge. No threshold
    is tuned, so the rule takes nothing from the out-of-fold predictions; it is compared with the
    operating threshold on both splits, with paired bootstrap 95% intervals (the same resampled
    customers for both rules). Written to results/ev_targeting.csv and derived.json (ev_rule_*)."""
    try:
        from .config import N_BOOT, SEED
        from .metrics import campaign_profit
        tj = load_json("thresholds.json")
        thr, cm = tj[best]["operating"], tj["_cost_model"]
        loss = cm.get("leakage", 1.0) * cm["offer_cost"] + cm["contact_cost"]
        rng = np.random.RandomState(SEED)
        rows = []
        for split in ("oof", "test"):
            _, y, p, m = _split_frame(best, split)
            gain = cm["save_rate"] * (cm["margin"] * cm["horizon_months"] * m - cm["offer_cost"]) - cm["contact_cost"]
            ev = (p * gain - (1 - p) * loss > 0).astype(float)
            # campaign_profit uses scores only through p >= thr, so a 0/1 contact vector at 0.5 is exact
            new, old = campaign_profit(y, ev, m, 0.5), campaign_profit(y, p, m, thr)
            idx = [rng.randint(0, len(y), len(y)) for _ in range(N_BOOT)]
            boots = np.array([campaign_profit(y[i], ev[i], m[i], 0.5)["profit"] - campaign_profit(y[i], p[i], m[i], thr)["profit"]
                              for i in idx])
            rows.append({"split": split, "n_customers": len(y), "n_contacted": new["n_contacted"],
                         "share_contacted": new["share_contacted"], "recall": new["recall"], "precision": new["precision"],
                         "profit": new["profit"], "n_contacted_operating": old["n_contacted"], "profit_operating": old["profit"],
                         "profit_diff": new["profit"] - old["profit"], "ci_lo": float(np.quantile(boots, 0.025)),
                         "ci_hi": float(np.quantile(boots, 0.975)), "share_resamples_above_zero": float((boots > 0).mean())})
            d[f"ev_rule_profit_{split}"] = new["profit"]
            d[f"ev_rule_n_contacted_{split}"] = new["n_contacted"]
            d[f"ev_rule_gain_{split}"] = rows[-1]["profit_diff"]
            d[f"ev_rule_gain_{split}_ci_lo"], d[f"ev_rule_gain_{split}_ci_hi"] = rows[-1]["ci_lo"], rows[-1]["ci_hi"]
        save_csv(pd.DataFrame(rows), "ev_targeting.csv", float_format="%.10g")
        # below this monthly charge even a certain churner has negative expected profit (save rate x net value
        # of a save < contact cost), so no probability justifies an offer
        d["ev_rule_min_monthly_charge"] = (cm["offer_cost"] + cm["contact_cost"] / cm["save_rate"]) / (cm["margin"] * cm["horizon_months"])
    except Exception as e:
        d["ev_targeting_error"] = str(e)


def profit_gaps(best: str) -> pd.DataFrame:
    """Expected-profit differences between each model and the selected one, each at its own operating
    threshold (and the selected model at its F1-optimal threshold), on the out-of-fold training
    predictions and on the test set, with paired bootstrap 95% intervals (the same resampled customers
    for both models). Written to results/profit_gap_bootstrap.csv."""
    from .config import N_BOOT, SEED
    from .metrics import campaign_profit
    thr = load_json("thresholds.json")
    rng = np.random.RandomState(SEED)
    rows = []
    for split, file, col in (("oof", "oof_predictions.csv", "p_{}_final"), ("test", "test_predictions.csv", "p_{}")):
        df = pd.read_csv(RESULTS / file)
        y, m = df["y"].values, df["MonthlyCharges"].values
        base_p, base_t = df[col.format(best)].values, thr[best]["operating"]
        comps = [(f"{name} vs {best}", df[col.format(name)].values, thr[name]["operating"])
                 for name in ("XGBoost", "RandomForest", "MLP", "DecisionTree", "Voting") if name != best]
        comps.append((f"{best} at F1-optimal vs operating threshold", base_p, thr[best]["f1_opt"]))
        idx = [rng.randint(0, len(y), len(y)) for _ in range(N_BOOT)]
        for name, p, t in comps:
            diff = campaign_profit(y, p, m, t)["profit"] - campaign_profit(y, base_p, m, base_t)["profit"]
            boots = np.array([campaign_profit(y[i], p[i], m[i], t)["profit"] - campaign_profit(y[i], base_p[i], m[i], base_t)["profit"]
                              for i in idx])
            rows.append({"split": split, "comparison": name, "n_customers": len(y), "profit_diff": diff,
                         "profit_diff_per_customer": diff / len(y), "ci_lo": float(np.quantile(boots, 0.025)),
                         "ci_hi": float(np.quantile(boots, 0.975)), "share_resamples_above_zero": float((boots > 0).mean())})
    out = pd.DataFrame(rows)
    save_csv(out, "profit_gap_bootstrap.csv")
    return out


def run() -> dict:
    d: dict = {}
    eda = load_json("eda_overall.json")
    d["annualised_churner_revenue_millions"] = eda["annualised_churner_revenue"] / 1e6
    # Profit: share of the oracle captured by the chosen model at the operating threshold
    pt = pd.read_csv(RESULTS / "profit_test.csv")
    best = load_json("model_ranking.json")["best_single_model"]
    oracle = float(pt.loc[pt.model.str.startswith("Oracle"), "profit"].iloc[0])
    best_op = float(pt.loc[(pt.model == best) & (pt.threshold_kind == "operating"), "profit"].iloc[0])
    d["oracle_profit_share_best_operating"] = best_op / oracle
    for m in ["LogReg", "XGBoost", "RandomForest", "MLP", "DecisionTree", "Voting"]:
        d[f"profit_gap_vs_best_{m}"] = float(pt.loc[(pt.model == m) & (pt.threshold_kind == "operating"), "profit"].iloc[0]) - best_op
    d["profit_default_vs_operating_best"] = best_op - float(pt.loc[(pt.model == best) & (pt.threshold_kind == "default"), "profit"].iloc[0])
    d["profit_f1_vs_operating_best"] = best_op - float(pt.loc[(pt.model == best) & (pt.threshold_kind == "f1_opt"), "profit"].iloc[0])
    # Fairness ratios (best model, operating threshold)
    f = pd.read_csv(RESULTS / "fairness_metrics.csv")
    f = f[f.model == best]

    def val(attr, group, metric):
        return float(f[(f.attribute == attr) & (f.group == group) & (f.metric == metric)]["value"].iloc[0])
    d["senior_selection_rate_ratio"] = val("SeniorCitizen", "Senior", "selection_rate") / val("SeniorCitizen", "Not senior", "selection_rate")
    d["senior_fpr_ratio"] = val("SeniorCitizen", "Senior", "fpr") / val("SeniorCitizen", "Not senior", "fpr")
    d["senior_base_rate_ratio"] = val("SeniorCitizen", "Senior", "observed_rate") / val("SeniorCitizen", "Not senior", "observed_rate")
    d["miss_rate_dependents"] = 1 - val("Dependents", "Dependents", "recall")
    d["miss_rate_no_dependents"] = 1 - val("Dependents", "No dependents", "recall")
    d["miss_rate_partner"] = 1 - val("Partner", "Partner", "recall")
    d["miss_rate_no_partner"] = 1 - val("Partner", "No partner", "recall")
    # Compute footprint
    fp = pd.read_csv(RESULTS / "compute_footprint.csv")
    tot = load_json("compute_footprint_total.json")
    d["co2e_grams"] = (tot.get("codecarbon_kg_co2e") or 0) * 1000
    d["total_modelling_minutes"] = tot["total_modelling_seconds"] / 60
    cv = pd.read_csv(RESULTS / "nested_cv_summary.csv")
    t = cv[cv.variant == "tuned"].set_index("model")["fit_seconds_mean"]
    d["fit_time_ratio_rf_vs_lr"] = float(t["RandomForest"] / t["LogReg"])
    d["fit_time_ratio_xgb_vs_lr"] = float(t["XGBoost"] / t["LogReg"])
    d["nested_cv_seconds_rf"] = float(fp[(fp.model == "RandomForest") & (fp.stage == "nested_cv")]["seconds"].iloc[0])
    d["nested_cv_seconds_lr"] = float(fp[(fp.model == "LogReg") & (fp.stage == "nested_cv")]["seconds"].iloc[0])
    # Error analysis: share of churners missed overall
    e = load_json("error_analysis_summary.json")
    d["fn_share_of_churners"] = e["FN"] / (e["FN"] + e["TP"])
    d["fp_share_of_stayers"] = e["FP"] / (e["FP"] + e["TN"])
    d["contacted_test"] = e["TP"] + e["FP"]
    # Learning curve: gap at full size and gain from 1/3 to full
    lc = pd.read_csv(RESULTS / "learning_curve.csv")
    ap = lc[lc.metric == "average_precision"].sort_values("n_train")
    d["learning_curve_gap_full"] = float(ap["train_mean"].iloc[-1] - ap["val_mean"].iloc[-1])
    d["learning_curve_val_gain_last_half"] = float(ap["val_mean"].iloc[-1] - ap["val_mean"].iloc[len(ap) // 2])
    # Model comparison spreads (tuned CV)
    top4 = cv[(cv.variant == "tuned") & cv.model.isin(["LogReg", "XGBoost", "RandomForest", "MLP"])]
    d["cv_pr_auc_spread_top4"] = float(top4["pr_auc_mean"].max() - top4["pr_auc_mean"].min())
    d["cv_roc_auc_spread_top4"] = float(top4["roc_auc_mean"].max() - top4["roc_auc_mean"].min())
    # Calibration: MLP uncalibrated over-prediction
    cm = pd.read_csv(RESULTS / "calibration_metrics.csv")
    r = cm[(cm.model == "MLP") & (cm.part == "test_uncalibrated")].iloc[0]
    d["mlp_uncal_overprediction"] = float(r["mean_pred"] - r["observed"])
    # Churn-rate ratios quoted in Part 2
    r = pd.read_csv(RESULTS / "churn_by_category.csv")

    def rate(feat, lvl):
        return float(r[(r.feature == feat) & (r.level.astype(str) == lvl)]["churn_rate"].iloc[0])
    d["rate_ratio_contract"] = rate("Contract", "Month-to-month") / rate("Contract", "Two year")
    d["rate_ratio_tenure"] = rate("tenure_band", "0-6") / rate("tenure_band", "49-72")
    d["rate_ratio_payment"] = rate("PaymentMethod", "Electronic check") / rate("PaymentMethod", "Credit card (automatic)")
    d["rate_ratio_internet"] = rate("InternetService", "Fiber optic") / rate("InternetService", "DSL")
    # Profit curve flatness (OOF, best model): threshold range within 95 % of the maximum
    pc = pd.read_csv(RESULTS / "profit_curve_oof.csv")
    pcb = pc[pc.model == best]
    flat = pcb[pcb.profit >= 0.95 * pcb.profit.max()]["threshold"]
    d["profit_flat95_lo"], d["profit_flat95_hi"] = float(flat.min()), float(flat.max())
    # Equivalence: bootstrap CI bounds of the PR-AUC gap to the best model among the strong single models
    st = pd.read_csv(RESULTS / "stat_tests_test.csv")
    pb = st[(st.test == "paired bootstrap") & (st.metric == "pr_auc") & st.comparison.str.startswith(("XGBoost", "RandomForest", "MLP"))]
    d["pr_auc_gap_ci_lo_min"], d["pr_auc_gap_ci_hi_max"] = float(pb.ci_lo.min()), float(pb.ci_hi.max())
    # Odds-ratio attenuation after stratification (Part 2)
    mh = load_json("mantel_haenszel_echeck.json")
    d["echeck_or_attenuation_ratio"] = mh["mh_pooled_odds_ratio_echeck"] / mh["crude_odds_ratio_echeck"]
    # Learning-curve endpoints quoted in Part 4
    d["lc_val_at_half"], d["lc_n_at_half"] = float(ap["val_mean"].iloc[len(ap) // 2]), int(ap["n_train"].iloc[len(ap) // 2])
    d["lc_val_at_full"], d["lc_n_at_full"] = float(ap["val_mean"].iloc[-1]), int(ap["n_train"].iloc[-1])
    # Fit time of the chosen model (tuned, per outer fold)
    d["fit_seconds_best_tuned"] = float(t[best])
    # Proxy associations (Cramer's V) between sensitive attributes and contract / payment
    try:
        prox = load_json("proxy_associations.json")
        for k, vv in prox.items():
            d[f"cramers_v_{k}"] = vv["cramers_v"]
    except Exception:
        pass
    # Recall parity for the two groups whose churners the model misses most (Part 5): customers with
    # dependents and customers with a partner, each judged by the same rule
    recall_parity(d, best, "Dependents", "dependents", "no_dependents", "dependents_threshold_scan.csv")
    recall_parity(d, best, "Partner", "partner", "no_partner")
    combined_parity(d, best)
    # The best simple segment rule as a benchmark for the model (Part 4)
    rule_baselines(d, best)
    # Per-customer expected-value targeting against the single operating threshold (Part 4)
    ev_targeting(d, best)
    # How far each engineered feature's gain stands above fold noise (Part 3)
    feature_tests(d)
    # Expected-profit gaps between models, with paired bootstrap intervals (Part 4)
    try:
        profit_gaps(best)
    except Exception as e:
        d["profit_gaps_error"] = str(e)
    # Ex-ante feasibility: save rate at which contacting a whole segment breaks even under the cost model
    # of metrics.campaign_profit (Part 4, Table C5). Every contact costs contact_cost, each contacted stayer
    # takes the offer with probability leakage, and each saved churner is worth margin * horizon times that
    # churner's own monthly charge, so the value term uses the churners' mean charge (mrr_at_risk / churners)
    try:
        cm = load_json("thresholds.json")["_cost_model"]
        seg = pd.read_csv(RESULTS / "segments.csv")
        for _, row in seg.iterrows():
            n, k = float(row["n"]), float(row["churners"])
            r, m = k / n, float(row["mrr_at_risk"]) / k
            value = cm["margin"] * cm["horizon_months"] * m
            key = "all" if str(row["segment"]).startswith("All") else str(row["segment"]).split()[0]
            d[f"breakeven_save_rate_{key}"] = ((cm["contact_cost"] + cm.get("leakage", 1.0) * cm["offer_cost"] * (1 - r))
                                               / (r * (value - cm["offer_cost"])))
        # the same for the test split alone, which Part 4's contact-everyone loss is computed on
        tp = pd.read_csv(RESULTS / "test_predictions.csv")
        r, m = float(tp["y"].mean()), float(tp.loc[tp["y"] == 1, "MonthlyCharges"].mean())
        d["breakeven_save_rate_test_split"] = ((cm["contact_cost"] + cm.get("leakage", 1.0) * cm["offer_cost"] * (1 - r))
                                               / (r * (cm["margin"] * cm["horizon_months"] * m - cm["offer_cost"])))
    except Exception as e:
        d["breakeven_error"] = str(e)
    # Search budgets (draws per model), within-group calibration gap, coefficient counts for Figure 8
    try:
        from .config import N_ITER
        for k, v in N_ITER.items():
            d[f"n_iter_{k}"] = int(v)
    except Exception:
        pass
    try:
        fm = pd.read_csv(RESULTS / "fairness_metrics.csv")
        fm = fm[fm.model == best]
        o = fm[fm.metric == "observed_rate"].set_index(["attribute", "group"])["value"]
        mp = fm[fm.metric == "mean_pred"].set_index(["attribute", "group"])["value"]
        d["max_within_group_calibration_gap"] = float((o - mp).abs().max())
    except Exception as e:
        d["calibration_gap_error"] = str(e)
    try:
        d["n_lr_coefficients"] = int(len(pd.read_csv(RESULTS / "lr_odds_ratios.csv")))
        d["n_fig8_coefficients_shown"] = 14  # explain.py plots the 14 largest |coef|
    except Exception as e:
        d["coef_count_error"] = str(e)
    # Reference list statistics
    try:
        refs = json.loads((RESULTS.parent / "references" / "references.json").read_text(encoding="utf-8"))
        d["n_verified_references"] = len(refs["references"])
        d["n_verified_legal"] = len(refs["legal"])
    except Exception:
        pass
    save_json(d, "derived.json")
    return d


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
