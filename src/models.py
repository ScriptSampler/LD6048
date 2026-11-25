"""Part 3 ablations (on the training split only) and Part 4 modelling:
split -> ablations -> nested CV (default vs tuned) -> final tuning on the full
training set -> OOF predictions for threshold/calibration decisions -> one
evaluation on the hold-out test set.

Outputs (results/): split_summary.json, ablation_*.csv, feature_tests_xgb_params.json,
chosen_config.json, vif.csv, search_spaces.json, nested_cv_scores.csv, nested_cv_summary.csv, xgb_depth_profile.csv,
best_params.json, oof_predictions.csv, test_predictions.csv,
compute_footprint.csv, calibration_choice.csv.
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import clean as clean_mod
from .config import (ABLATION_REPEATS, DATA_PROCESSED, ID_COL, INNER_SPLITS, MODELS_DIR,
                     MODEL_ORDER, N_ITER, OUTER_REPEATS, OUTER_SPLITS, QUICK, SEED,
                     TARGET, TEST_SIZE, TUNING_METRIC)
from .features import LINEAR_CANDIDATES, TREE_CANDIDATES, make_pipeline
from .io_utils import save_csv, save_json
from .metrics import compute_metrics, expected_calibration_error

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_KIND = {"Majority": "tree", "LogReg": "linear", "DecisionTree": "tree",
              "RandomForest": "tree", "XGBoost": "tree", "MLP": "linear"}


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split(df: pd.DataFrame):
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=SEED)
    ids = df[ID_COL]
    save_json({"train_index": X_tr.index.tolist(), "test_index": X_te.index.tolist(),
               "train_ids": ids.loc[X_tr.index].tolist(), "test_ids": ids.loc[X_te.index].tolist()},
              "split_indices.json")
    save_json({"n_train": int(len(X_tr)), "n_test": int(len(X_te)),
               "churn_rate_train": float(y_tr.mean()), "churn_rate_test": float(y_te.mean()),
               "churners_train": int(y_tr.sum()), "churners_test": int(y_te.sum()),
               "test_size": TEST_SIZE, "seed": SEED, "stratified": True},
              "split_summary.json")
    X_tr.assign(Churn=y_tr).to_csv(DATA_PROCESSED / "train.csv", index=True)
    X_te.assign(Churn=y_te).to_csv(DATA_PROCESSED / "test.csv", index=True)
    return X_tr, X_te, y_tr, y_te


# ---------------------------------------------------------------------------
# Estimators and search spaces
# ---------------------------------------------------------------------------
def base_estimator(name: str, n_jobs: int = 1, class_weight=None):
    if name == "Majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "LogReg":
        return LogisticRegression(max_iter=5000, solver="lbfgs", C=1.0, class_weight=class_weight,
                                  random_state=SEED)
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=SEED, class_weight=class_weight)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=n_jobs,
                                      class_weight=class_weight)
    if name == "XGBoost":
        # class_weight is handled by scale_pos_weight, set by the caller.
        return XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, random_state=SEED,
                             n_jobs=n_jobs, tree_method="hist", eval_metric="logloss", verbosity=0)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=300,
                             random_state=SEED, n_iter_no_change=15)
    raise ValueError(name)


def search_space(name: str) -> dict:
    if name == "LogReg":
        return {"model__C": loguniform(1e-3, 1e2)}
    if name == "DecisionTree":
        return {"model__max_depth": randint(2, 12), "model__min_samples_leaf": randint(5, 200),
                "model__ccp_alpha": loguniform(1e-5, 1e-2), "model__criterion": ["gini", "entropy"]}
    if name == "RandomForest":
        return {"model__n_estimators": [200, 400, 600], "model__max_depth": [None, 6, 10, 16],
                "model__min_samples_leaf": randint(1, 30), "model__max_features": ["sqrt", 0.3, 0.5]}
    if name == "XGBoost":
        # Widened after a 25-draw search chose depth 2 and gamma 1, its bounds: stumps (depth 1) to depth 8,
        # gamma to 5, the L2 penalty over five decades and L1 over four; trees and learning rate are searched jointly.
        # Per-fold choices still reach depth 1 and 8; xgb_depth_profile.csv shows how flat the optimum is.
        return {"model__n_estimators": randint(50, 1001), "model__learning_rate": loguniform(0.005, 0.3),
                "model__max_depth": randint(1, 9), "model__min_child_weight": loguniform(1.0, 100.0),
                "model__subsample": uniform(0.5, 0.5), "model__colsample_bytree": uniform(0.3, 0.7),
                "model__reg_lambda": loguniform(0.001, 100.0), "model__reg_alpha": loguniform(0.001, 10.0),
                "model__gamma": uniform(0.0, 5.0)}
    if name == "MLP":
        return {"model__hidden_layer_sizes": [(8,), (16,), (32,), (64,), (128,), (256,), (32, 16), (64, 32), (128, 64)],
                "model__activation": ["relu", "tanh"], "model__alpha": loguniform(1e-5, 10.0),
                "model__learning_rate_init": loguniform(1e-4, 3e-2), "model__batch_size": [32, 64, 128, 256]}
    return {}


def _space_to_json(space: dict) -> dict:
    out = {}
    for k, v in space.items():
        if isinstance(v, list):
            out[k] = [str(x) if isinstance(x, tuple) else x for x in v]
        else:
            d = v.dist.name if hasattr(v, "dist") else type(v).__name__
            args = [float(a) for a in v.args] if hasattr(v, "args") else []
            kw = {kk: float(vv) for kk, vv in v.kwds.items()} if hasattr(v, "kwds") else {}
            out[k] = {"distribution": d, "args": args, "kwds": kw}
    return out


def build(name: str, cfg: dict, n_jobs: int = 1, class_weight=None, sampler=None, params=None):
    est = base_estimator(name, n_jobs=n_jobs, class_weight=class_weight)
    kind = MODEL_KIND[name]
    if "features" in cfg:
        feats = cfg["features"]
    else:
        feats = cfg["features_linear"] if name == "LogReg" else cfg["features_tree"]
    pipe = make_pipeline(est, features=feats, model_kind=kind, sampler=sampler,
                         contract_ordinal=cfg.get("contract_ordinal", False) and kind == "linear",
                         log_totalcharges=cfg.get("log_totalcharges", False) and kind == "linear",
                         scale=True, collapse=cfg.get("collapse", True))
    if params:
        pipe.set_params(**params)
    return pipe


# ---------------------------------------------------------------------------
# Cross-validated scoring helper (mean +/- SD across folds)
# ---------------------------------------------------------------------------
def cv_scores(pipe, X, y, n_repeats=ABLATION_REPEATS, thr=0.5, extra_thr_search=False):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    rows = []
    for i, (tr, va) in enumerate(cv.split(X, y)):
        m = clone(pipe)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        r = compute_metrics(y.iloc[va], p, thr)
        if extra_thr_search:
            # threshold moving: F1-optimal threshold chosen on the *training* part
            # of this fold via inner OOF predictions, applied to the validation part.
            inner = cross_val_predict(clone(pipe), X.iloc[tr], y.iloc[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                      method="predict_proba", n_jobs=-1)[:, 1]
            from .metrics import best_threshold
            t_inner, _ = best_threshold(y.iloc[tr], inner, "f1")
            r2 = compute_metrics(y.iloc[va], p, t_inner)
            r["recall_moved"], r["precision_moved"], r["f1_moved"], r["threshold_moved"] = r2["recall"], r2["precision"], r2["f1"], t_inner
        r["ece"] = expected_calibration_error(y.iloc[va], p)
        r["fold"] = i
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: dict) -> dict:
    out = dict(label)
    for m in ["pr_auc", "roc_auc", "brier", "f1", "recall", "precision", "accuracy", "ece",
              "recall_moved", "precision_moved", "f1_moved"]:
        if m in df:
            out[f"{m}_mean"] = float(df[m].mean())
            out[f"{m}_sd"] = float(df[m].std(ddof=1))
    return out


# ---------------------------------------------------------------------------
# Ablations (training split only)
# ---------------------------------------------------------------------------
def run_ablations(X_tr, y_tr) -> dict:
    base_cfg = {"features": [], "collapse": True, "contract_ordinal": False, "log_totalcharges": False}
    rows = []
    folds: dict[tuple, pd.DataFrame] = {}
    n_tr_fold, n_va_fold = 0.8 * len(X_tr), 0.2 * len(X_tr)

    def record(tag, variant, name, sc, seconds=np.nan):
        rows.append(summarise(sc, {"ablation": tag, "variant": variant, "model": name, "seconds": seconds}))
        folds[(tag, variant, name)] = sc

    def run(tag, variant, name, cfg, sampler=None, class_weight=None, extra_thr=False, pipe=None):
        if pipe is None:
            pipe = build(name, cfg, n_jobs=-1, class_weight=class_weight, sampler=sampler)
        if name == "XGBoost" and class_weight == "balanced":
            pipe.set_params(model__scale_pos_weight=float((y_tr == 0).sum() / (y_tr == 1).sum()))
        t0 = time.time()
        sc = cv_scores(pipe, X_tr, y_tr, extra_thr_search=extra_thr)
        record(tag, variant, name, sc, time.time() - t0)
        return sc

    def paired(tag, model, a, b, metric="pr_auc"):
        """Corrected resampled t-test (Nadeau-Bengio) of variant b minus variant a on the same folds."""
        from .metrics import corrected_resampled_ttest
        sa, sb = folds[(tag, a, model)].sort_values("fold"), folds[(tag, b, model)].sort_values("fold")
        return corrected_resampled_ttest(sb[metric].values, sa[metric].values, n_tr_fold, n_va_fold)

    # 1. Imputation of TotalCharges blanks: constant 0 vs median (both in-pipeline)
    for name in ["LogReg", "XGBoost"]:
        run("imputation", "constant_0", name, base_cfg)
        pipe_med = build(name, base_cfg, n_jobs=-1)
        pipe_med.set_params(prep__num__impute__strategy="median")
        run("imputation", "median", name, base_cfg, pipe=pipe_med)
    # 2. Redundant levels: collapse vs keep
    for name in ["LogReg", "XGBoost"]:
        run("redundant_levels", "collapse_to_No", name, base_cfg)
        run("redundant_levels", "keep_three_levels", name, {**base_cfg, "collapse": False})
    # 3. Contract encoding (LR only): one-hot vs ordinal
    run("contract_encoding", "one_hot", "LogReg", base_cfg)
    run("contract_encoding", "ordinal", "LogReg", {**base_cfg, "contract_ordinal": True})
    # 4. TotalCharges transform (LR, MLP): standardise vs log1p+standardise
    for name in ["LogReg", "MLP"]:
        run("totalcharges_transform", "standardise", name, base_cfg)
        run("totalcharges_transform", "log1p_standardise", name, {**base_cfg, "log_totalcharges": True})
    # 5. Multicollinearity: keep TotalCharges / drop it / replace with avg_monthly_spend (LR)
    run("collinearity", "keep_totalcharges", "LogReg", base_cfg)
    pipe_drop = build("LogReg", base_cfg, n_jobs=-1)
    # drop TotalCharges by restricting the numeric column list
    ct = pipe_drop.named_steps["prep"]
    ct.transformers = [(n, t, [c for c in cols if c != "TotalCharges"] if n == "num" else cols)
                       for n, t, cols in ct.transformers]
    run("collinearity", "drop_totalcharges", "LogReg", base_cfg, pipe=pipe_drop)
    pipe_avg = build("LogReg", {**base_cfg, "features": ["avg_monthly_spend"]}, n_jobs=-1)
    ct = pipe_avg.named_steps["prep"]
    ct.transformers = [(n, t, [c for c in cols if c != "TotalCharges"] if n == "num" else cols)
                       for n, t, cols in ct.transformers]
    run("collinearity", "replace_with_avg_monthly_spend", "LogReg", base_cfg, pipe=pipe_avg)
    # the base logistic pipeline without MonthlyCharges, the column the interpretation model
    # (explain.lr_interpretation) leaves out because its VIF is extreme
    pipe_mc = build("LogReg", base_cfg, n_jobs=-1)
    ct = pipe_mc.named_steps["prep"]
    ct.transformers = [(n, t, [c for c in cols if c != "MonthlyCharges"] if n == "num" else cols)
                       for n, t, cols in ct.transformers]
    run("collinearity", "drop_monthlycharges", "LogReg", base_cfg, pipe=pipe_mc)
    # 6. Feature engineering: none, each candidate, all candidates (per family). XGBoost's defaults
    # (depth 6, learning rate 0.3) overfit and would penalise any added column, so XGBoost is tuned for
    # each feature set by Part 4's randomised search on the training split (fixed seed: every feature set
    # meets the same configurations). Logistic regression keeps C = 1, close to its tuned value.
    cands = {"LogReg": LINEAR_CANDIDATES, "XGBoost": TREE_CANDIDATES}
    fe_params = {}

    def fe_run(name, variant, feats):
        cfg = {**base_cfg, "features": feats}
        pipe = None
        if name == "XGBoost":
            search = RandomizedSearchCV(build(name, cfg, n_jobs=1), search_space(name), n_iter=N_ITER[_key(name)],
                                        scoring=TUNING_METRIC, cv=StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED),
                                        n_jobs=-1, random_state=SEED, refit=False).fit(X_tr, y_tr)
            # scored single-threaded like the search: with subsample < 1, XGBoost's row sampling
            # depends on the thread count, so all cores would make the result machine-dependent
            pipe = build(name, cfg, n_jobs=1, params=search.best_params_)
            fe_params[variant] = {**{k.replace("model__", ""): v for k, v in search.best_params_.items()},
                                  "_inner_cv_pr_auc": float(search.best_score_)}
        run("feature_engineering", variant, name, cfg, pipe=pipe)

    for name in ["LogReg", "XGBoost"]:
        fe_run(name, "none", [])
        for f in cands[name]:
            fe_run(name, f"+{f}", [f])
        fe_run(name, "all_candidates", cands[name])
    abl = pd.DataFrame(rows)

    # Decide per family by the rule used for every other choice (Part 3): keep a candidate only if it
    # beats 'none' on the same folds by the corrected resampled t-test (mean PR-AUC gain > 0 and p < 0.05).
    gains, gain_p = {}, {}
    for name in ["LogReg", "XGBoost"]:
        for f in cands[name]:
            t = paired("feature_engineering", name, "none", f"+{f}")
            gains[f"{name}:{f}"], gain_p[f"{name}:{f}"] = float(t["mean_diff"]), float(t["p"])
    kept_linear = [f for f in LINEAR_CANDIDATES if gains[f"LogReg:{f}"] > 0 and gain_p[f"LogReg:{f}"] < 0.05]
    kept_tree = [f for f in TREE_CANDIDATES if gains[f"XGBoost:{f}"] > 0 and gain_p[f"XGBoost:{f}"] < 0.05]
    # Verify the chosen combination against 'none'
    fe_run("LogReg", "chosen", kept_linear)
    fe_run("XGBoost", "chosen", kept_tree)
    abl = pd.DataFrame(rows)
    save_csv(abl, "ablation_preprocessing.csv")
    save_json(fe_params, "feature_tests_xgb_params.json")
    kept = {"linear": kept_linear, "tree": kept_tree}

    def pick(tag, model, a, b):
        """Keep the principled default a unless alternative b is better on the same
        folds by a corrected resampled t-test (mean gain > 0 and p < 0.05)."""
        t = paired(tag, model, a, b)
        choose_b = (t["mean_diff"] > 0) and (t["p"] < 0.05)
        return (b if choose_b else a), float(t["mean_diff"]), float(t["p"])

    collapse_choice, d_col, p_col = pick("redundant_levels", "LogReg", "collapse_to_No", "keep_three_levels")
    contract_choice, d_con, p_con = pick("contract_encoding", "LogReg", "one_hot", "ordinal")
    log_choice, d_log, p_log = pick("totalcharges_transform", "LogReg", "standardise", "log1p_standardise")
    log_mlp = paired("totalcharges_transform", "MLP", "standardise", "log1p_standardise")
    imp_choice, d_imp, p_imp = pick("imputation", "LogReg", "constant_0", "median")
    col_s = abl[abl["ablation"] == "collinearity"].set_index("variant")["pr_auc_mean"]
    chosen = {
        "features_linear": kept_linear, "features_tree": kept_tree, "feature_gains_pr_auc": gains,
        "feature_gain_p_values": gain_p,
        "collapse": collapse_choice == "collapse_to_No",
        "contract_ordinal": contract_choice == "ordinal",
        "log_totalcharges": log_choice == "log1p_standardise",
        "imputation": imp_choice,
        "collinearity_variant_scores": col_s.to_dict(),
        "decision_rule": "Keep the principled default unless the alternative is better on the same CV folds by a corrected resampled t-test (mean PR-AUC gain > 0 and p < 0.05); the same test decides each engineered feature against no engineered features for its model family (XGBoost tuned for each feature set on the training split).",
        "deltas": {"collapse": d_col, "contract_ordinal": d_con, "log_totalcharges": d_log, "imputation_median": d_imp,
                   "log_totalcharges_mlp": float(log_mlp["mean_diff"])},
        "p_values": {"collapse": p_col, "contract_ordinal": p_con, "log_totalcharges": p_log, "imputation_median": p_imp,
                     "log_totalcharges_mlp": float(log_mlp["p"])},
    }
    # TotalCharges: keep unless dropping/replacing is significantly better
    best_alt = col_s[["drop_totalcharges", "replace_with_avg_monthly_spend"]].idxmax()
    t_tc = paired("collinearity", "LogReg", "keep_totalcharges", best_alt)
    chosen["drop_totalcharges"] = bool(t_tc["mean_diff"] > 0 and t_tc["p"] < 0.05)
    chosen["collinearity_best_alternative"] = best_alt
    chosen["p_values"]["collinearity"] = float(t_tc["p"])
    # MonthlyCharges left out of the base logistic pipeline: its cost in PR-AUC, same folds
    t_mc = paired("collinearity", "LogReg", "keep_totalcharges", "drop_monthlycharges")
    chosen["deltas"]["drop_monthlycharges"] = float(t_mc["mean_diff"])
    chosen["p_values"]["drop_monthlycharges"] = float(t_mc["p"])
    save_csv(pd.concat([sc.assign(ablation=k[0], variant=k[1], model=k[2]) for k, sc in folds.items()], ignore_index=True),
             "ablation_folds.csv", float_format="%.10g")  # full precision: derived.feature_tests re-tests these folds

    # 7. Class imbalance: none / class_weight / SMOTE / threshold moving (LR and XGB)
    from imblearn.over_sampling import SMOTE
    cfg_now = {"features_linear": kept_linear, "features_tree": kept_tree, "collapse": chosen["collapse"],
               "contract_ordinal": chosen["contract_ordinal"], "log_totalcharges": chosen["log_totalcharges"]}
    rows2 = []
    for name in ["LogReg", "XGBoost"]:
        for variant in ["none", "class_weight_balanced", "smote", "threshold_moving"]:
            cw = "balanced" if variant == "class_weight_balanced" else None
            sampler = SMOTE(random_state=SEED) if variant == "smote" else None
            pipe = build(name, cfg_now, n_jobs=-1, class_weight=cw, sampler=sampler)
            if name == "XGBoost" and cw == "balanced":
                pipe.set_params(model__scale_pos_weight=float((y_tr == 0).sum() / (y_tr == 1).sum()))
            t0 = time.time()
            sc = cv_scores(pipe, X_tr, y_tr, extra_thr_search=(variant == "threshold_moving"))
            r = summarise(sc, {"ablation": "imbalance", "variant": variant, "model": name, "seconds": time.time() - t0})
            if variant == "threshold_moving":
                r["threshold_mean"] = float(sc["threshold_moved"].mean())
            rows2.append(r)
            folds[("imbalance", variant, name)] = sc
    imb = pd.DataFrame(rows2)
    save_csv(imb, "ablation_imbalance.csv")
    save_csv(pd.concat([sc.assign(ablation=k[0], variant=k[1], model=k[2]) for k, sc in folds.items() if k[0] == "imbalance"],
                       ignore_index=True), "ablation_imbalance_folds.csv")
    # Rule: discrimination (PR-AUC) first; adopt weighting/resampling only if it beats 'none'
    # on the same folds (corrected t-test, p < 0.05); otherwise keep calibrated
    # probabilities and move the threshold instead.
    lr = imb[imb["model"] == "LogReg"].set_index("variant")
    tests = {v: paired("imbalance", "LogReg", "none", v) for v in ["class_weight_balanced", "smote"]}
    better = [v for v, t in tests.items() if t["mean_diff"] > 0 and t["p"] < 0.05]
    chosen["imbalance"] = better[0] if better else "none_plus_threshold_moving"
    chosen["imbalance_tests_LogReg"] = {v: {"mean_diff_pr_auc": float(t["mean_diff"]), "p": float(t["p"])} for v, t in tests.items()}
    chosen["imbalance_brier_tests_LogReg"] = {v: {k: float(x) for k, x in paired("imbalance", "LogReg", "none", v, "brier").items()} for v in ["class_weight_balanced", "smote"]}
    chosen["imbalance_table_LogReg"] = lr[["pr_auc_mean", "pr_auc_sd", "brier_mean", "recall_mean", "recall_moved_mean" if "recall_moved_mean" in lr else "recall_mean"]].to_dict()
    save_json(chosen, "chosen_config.json")

    # Preprocessing decision-log rows with measured evidence
    def fmt(tag, model, v):
        s = abl[(abl["ablation"] == tag) & (abl["model"] == model) & (abl["variant"] == v)].iloc[0]
        return f"{v}: PR-AUC {s['pr_auc_mean']:.3f} ± {s['pr_auc_sd']:.3f}"
    clean_mod.log_decision(
        "22 attribute-level duplicate rows (different customerIDs)",
        "Keep", "Distinct customers with identical attributes; removing them would bias the sample towards rarer profiles",
        "Drop as duplicates", "Rows duplicate on all 20 attributes but have unique customerID (data_manifest.json)")
    clean_mod.log_decision(
        "customerID", "Drop before modelling", "Identifier with no predictive signal; a privacy risk if retained in model artefacts",
        "Keep as a feature", "Unique per row (data_manifest.json)")
    clean_mod.log_decision(
        "'No internet service' / 'No phone service' levels duplicate InternetService / PhoneService",
        f"{'Collapse to No' if chosen['collapse'] else 'Keep three levels'}",
        "Collapsing removes perfectly redundant dummies (avoids the dummy-variable trap for the linear model); trees are indifferent",
        "Keep three levels (redundant columns)",
        f"5x{ABLATION_REPEATS} CV, LogReg: {fmt('redundant_levels', 'LogReg', 'collapse_to_No')} vs {fmt('redundant_levels', 'LogReg', 'keep_three_levels')}")
    clean_mod.log_decision(
        "Nominal encoding", "One-hot (reference-category coding for the linear model; drop='if_binary' for trees); Contract one-hot" if not chosen["contract_ordinal"] else "Contract ordinal for LR",
        "Reference coding gives full-rank design and interpretable odds ratios; ordinal Contract assumes a linear effect",
        "Label-encode nominal features (imposes false order)",
        f"LogReg: {fmt('contract_encoding', 'LogReg', 'one_hot')} vs {fmt('contract_encoding', 'LogReg', 'ordinal')}")
    clean_mod.log_decision(
        "Scaling of numeric features", "StandardScaler inside the pipeline" + (" + log1p(TotalCharges)" if chosen["log_totalcharges"] else ""),
        "L2-regularised LR and the MLP are scale-sensitive; trees are not but scaling is harmless",
        "No scaling / log1p transform of TotalCharges",
        f"LogReg: {fmt('totalcharges_transform', 'LogReg', 'standardise')} vs {fmt('totalcharges_transform', 'LogReg', 'log1p_standardise')}")
    clean_mod.log_decision(
        "Multicollinearity (tenure, MonthlyCharges, TotalCharges)",
        "Drop TotalCharges" if chosen["drop_totalcharges"] else "Keep TotalCharges (L2 penalty stabilises coefficients)",
        "VIF reported in vif.csv; prediction is unaffected by collinearity but coefficient interpretation is",
        "Replace with average monthly spend / drop",
        "LogReg PR-AUC: " + ", ".join(f"{k} {v:.3f}" for k, v in col_s.items()))
    clean_mod.log_decision(
        "Feature engineering", f"Linear model keeps {kept_linear or 'none'}; tree/MLP models keep {kept_tree or 'none'}",
        "Only candidates that beat the raw attributes on the same folds are kept (mean gain > 0, corrected t-test p < 0.05; XGBoost tuned for each feature set); count/flag features are exact linear combinations of existing dummies and so are tested for tree models only",
        "Keep all candidates regardless of evidence",
        "PR-AUC gains: " + ", ".join(f"{k} {v:+.4f}" for k, v in gains.items()))
    clean_mod.log_decision(
        "Class imbalance (26.5 % positive)", chosen["imbalance"].replace("_", " "),
        "Resampling/weighting did not improve PR-AUC beyond fold noise and damages calibration; moving the threshold keeps calibrated probabilities",
        "SMOTE / class_weight='balanced'",
        "LogReg PR-AUC / Brier: " + ", ".join(f"{v} {lr.loc[v, 'pr_auc_mean']:.3f}/{lr.loc[v, 'brier_mean']:.3f}" for v in lr.index))
    clean_mod.write_decision_log()
    return chosen


# ---------------------------------------------------------------------------
# VIF on the logistic-regression design matrix (training split)
# ---------------------------------------------------------------------------
def compute_vif(X_tr, y_tr, cfg):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    pipe = build("LogReg", cfg)
    pipe.fit(X_tr, y_tr)
    Z = pipe[:-1].transform(X_tr)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    Z = np.asarray(Z, float)
    Zc = np.column_stack([np.ones(len(Z)), Z])
    vifs = [variance_inflation_factor(Zc, i + 1) for i in range(Z.shape[1])]
    df = pd.DataFrame({"feature": names, "vif": vifs}).sort_values("vif", ascending=False)
    save_csv(df, "vif.csv")
    return df


# ---------------------------------------------------------------------------
# Nested cross-validation: default vs tuned
# ---------------------------------------------------------------------------
def nested_cv(X_tr, y_tr, cfg) -> pd.DataFrame:
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    rows, footprint = [], []
    spaces = {}
    for name in MODEL_ORDER:
        space = search_space(name)
        spaces[name] = _space_to_json(space)
        t_model = time.time()
        for i, (tr, va) in enumerate(outer.split(X_tr, y_tr)):
            rep, fold = divmod(i, OUTER_SPLITS)
            Xa, ya, Xb, yb = X_tr.iloc[tr], y_tr.iloc[tr], X_tr.iloc[va], y_tr.iloc[va]
            # default
            t0 = time.time()
            pipe = build(name, cfg, n_jobs=-1)
            pipe.fit(Xa, ya)
            pv = pipe.predict_proba(Xb)[:, 1]
            pt = pipe.predict_proba(Xa)[:, 1]
            r = compute_metrics(yb, pv, 0.5)
            r.update({"model": name, "variant": "default", "repeat": rep, "fold": fold,
                      "fit_seconds": time.time() - t0, "train_pr_auc": compute_metrics(ya, pt)["pr_auc"],
                      "train_roc_auc": compute_metrics(ya, pt)["roc_auc"], "best_params": "{}"})
            rows.append(r)
            # tuned
            t0 = time.time()
            if space:
                pipe = build(name, cfg, n_jobs=1)  # parallelism lives in RandomizedSearchCV
                search = RandomizedSearchCV(pipe, space, n_iter=N_ITER[_key(name)], scoring=TUNING_METRIC,
                                            cv=inner, n_jobs=-1, random_state=SEED, refit=True, error_score="raise")
                search.fit(Xa, ya)
                best = search.best_estimator_
                bp = json.dumps({k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in search.best_params_.items()})
                n_inner_fits = N_ITER[_key(name)] * INNER_SPLITS
            else:
                best, bp, n_inner_fits = pipe, "{}", 0
            pv = best.predict_proba(Xb)[:, 1]
            pt = best.predict_proba(Xa)[:, 1]
            r = compute_metrics(yb, pv, 0.5)
            r.update({"model": name, "variant": "tuned", "repeat": rep, "fold": fold,
                      "fit_seconds": time.time() - t0, "train_pr_auc": compute_metrics(ya, pt)["pr_auc"],
                      "train_roc_auc": compute_metrics(ya, pt)["roc_auc"], "best_params": bp,
                      "n_inner_fits": n_inner_fits})
            rows.append(r)
            print(f"  nested {name} rep{rep} fold{fold}: default PR-AUC {rows[-2]['pr_auc']:.3f} | tuned {r['pr_auc']:.3f} ({r['fit_seconds']:.0f}s)", flush=True)
        footprint.append({"model": name, "stage": "nested_cv", "seconds": time.time() - t_model})
    df = pd.DataFrame(rows)
    save_csv(df, "nested_cv_scores.csv")
    save_json(spaces, "search_spaces.json")
    # summary mean +/- SD
    summ = df.groupby(["model", "variant"]).agg(
        **{f"{m}_{s}": (m, s) for m in ["pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy", "brier", "mcc", "lift_top_decile"] for s in ["mean", "std"]},
        train_pr_auc_mean=("train_pr_auc", "mean"), train_roc_auc_mean=("train_roc_auc", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"), n_folds=("fold", "size")).reset_index()
    summ["train_val_gap_pr_auc"] = summ["train_pr_auc_mean"] - summ["pr_auc_mean"]
    summ["train_val_gap_roc_auc"] = summ["train_roc_auc_mean"] - summ["roc_auc_mean"]
    save_csv(summ, "nested_cv_summary.csv")
    return df, pd.DataFrame(footprint)


def _key(name):
    return {"LogReg": "lr", "DecisionTree": "dt", "RandomForest": "rf", "XGBoost": "xgb", "MLP": "mlp"}.get(name, "lr")


# ---------------------------------------------------------------------------
# Final tuning on the full training set, OOF predictions, test predictions
# ---------------------------------------------------------------------------
def final_models(X_tr, y_tr, X_te, y_te, cfg):
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_params, footprint, cal_rows = {}, [], []
    oof = pd.DataFrame({"y": y_tr.values, "MonthlyCharges": X_tr["MonthlyCharges"].values}, index=X_tr.index)
    test = pd.DataFrame({"y": y_te.values, "MonthlyCharges": X_te["MonthlyCharges"].values}, index=X_te.index)
    fitted = {}
    for name in MODEL_ORDER:
        t0 = time.time()
        space = search_space(name)
        if space:
            pipe = build(name, cfg, n_jobs=1)
            search = RandomizedSearchCV(pipe, space, n_iter=N_ITER[_key(name)], scoring=TUNING_METRIC, cv=inner,
                                        n_jobs=-1, random_state=SEED, refit=True)
            search.fit(X_tr, y_tr)
            best = search.best_estimator_
            best_params[name] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in search.best_params_.items()}
            best_params[name]["_inner_cv_pr_auc"] = float(search.best_score_)
            if name == "XGBoost":
                # best inner-CV PR-AUC at each depth: how flat the optimum is across the searched range
                res = pd.DataFrame({"max_depth": search.cv_results_["param_model__max_depth"].astype(int),
                                    "pr_auc": search.cv_results_["mean_test_score"]})
                save_csv(res.groupby("max_depth")["pr_auc"].agg(best_inner_cv_pr_auc="max", n_configurations="size")
                         .reset_index(), "xgb_depth_profile.csv", float_format="%.10g")
        else:
            best = build(name, cfg, n_jobs=-1).fit(X_tr, y_tr)
            best_params[name] = {}
        tune_s = time.time() - t0
        # OOF predictions with the tuned configuration (for thresholds / calibration decisions)
        t1 = time.time()
        oof[f"p_{name}"] = cross_val_predict(clone(best), X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=-1)[:, 1]
        # Calibration choice on OOF: none / sigmoid / isotonic (fitted inside CV)
        cal_choice, cal_scores = "none", {"none": float(np.mean((oof[f"p_{name}"] - y_tr.values) ** 2))}
        if name != "Majority":
            for method in ["sigmoid", "isotonic"]:
                cal = CalibratedClassifierCV(clone(best), method=method, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), ensemble=True)
                pc = cross_val_predict(cal, X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=-1)[:, 1]
                oof[f"p_{name}_{method}"] = pc
                cal_scores[method] = float(np.mean((pc - y_tr.values) ** 2))
            best_method = min(cal_scores, key=cal_scores.get)
            # Only switch if the improvement is material (> 0.001 Brier)
            if best_method != "none" and cal_scores["none"] - cal_scores[best_method] > 0.001:
                cal_choice = best_method
        cal_rows.append({"model": name, "brier_oof_none": cal_scores["none"],
                         "brier_oof_sigmoid": cal_scores.get("sigmoid", np.nan),
                         "brier_oof_isotonic": cal_scores.get("isotonic", np.nan),
                         "ece_oof_none": expected_calibration_error(y_tr, oof[f"p_{name}"]),
                         "ece_oof_sigmoid": expected_calibration_error(y_tr, oof[f"p_{name}_sigmoid"]) if f"p_{name}_sigmoid" in oof else np.nan,
                         "ece_oof_isotonic": expected_calibration_error(y_tr, oof[f"p_{name}_isotonic"]) if f"p_{name}_isotonic" in oof else np.nan,
                         "chosen": cal_choice})
        # Final estimator (calibrated if chosen), fitted on the full training set
        if cal_choice != "none":
            final = CalibratedClassifierCV(clone(best), method=cal_choice, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), ensemble=True)
            final.fit(X_tr, y_tr)
            oof[f"p_{name}_final"] = oof[f"p_{name}_{cal_choice}"]
        else:
            final = best
            oof[f"p_{name}_final"] = oof[f"p_{name}"]
        test[f"p_{name}"] = final.predict_proba(X_te)[:, 1]
        test[f"p_{name}_uncal"] = best.predict_proba(X_te)[:, 1]
        fitted[name] = {"tuned": best, "final": final, "calibration": cal_choice}
        joblib.dump(fitted[name], MODELS_DIR / f"{name}.joblib")
        footprint.append({"model": name, "stage": "final_tuning", "seconds": tune_s})
        footprint.append({"model": name, "stage": "oof_and_calibration", "seconds": time.time() - t1})
        print(f"  final {name}: tuned in {tune_s:.0f}s, calibration={cal_choice}, test ROC-AUC {compute_metrics(y_te, test[f'p_{name}'])['roc_auc']:.3f}", flush=True)
    # Soft-voting ensemble of logistic regression, the forest and XGBoost with tuned params (a fixed design choice)
    t0 = time.time()
    vote = VotingClassifier([("lr", clone(fitted["LogReg"]["tuned"])), ("rf", clone(fitted["RandomForest"]["tuned"])),
                             ("xgb", clone(fitted["XGBoost"]["tuned"]))], voting="soft", n_jobs=1)
    oof["p_Voting"] = cross_val_predict(clone(vote), X_tr, y_tr, cv=oof_cv, method="predict_proba", n_jobs=-1)[:, 1]
    oof["p_Voting_final"] = oof["p_Voting"]
    vote.fit(X_tr, y_tr)
    test["p_Voting"] = vote.predict_proba(X_te)[:, 1]
    test["p_Voting_uncal"] = test["p_Voting"]
    fitted["Voting"] = {"tuned": vote, "final": vote, "calibration": "none"}
    joblib.dump(fitted["Voting"], MODELS_DIR / "Voting.joblib")
    footprint.append({"model": "Voting", "stage": "final_fit_and_oof", "seconds": time.time() - t0})
    # Voting on the same repeated outer folds (params fixed from full-train tuning; noted as slightly optimistic)
    outer = RepeatedStratifiedKFold(n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=SEED)
    vrows = []
    for i, (tr, va) in enumerate(outer.split(X_tr, y_tr)):
        rep, fold = divmod(i, OUTER_SPLITS)
        m = clone(vote).fit(X_tr.iloc[tr], y_tr.iloc[tr])
        r = compute_metrics(y_tr.iloc[va], m.predict_proba(X_tr.iloc[va])[:, 1], 0.5)
        r.update({"model": "Voting", "variant": "tuned_fixed", "repeat": rep, "fold": fold})
        vrows.append(r)
    save_csv(pd.DataFrame(vrows), "voting_outer_cv_scores.csv")
    save_json(best_params, "best_params.json")
    save_csv(pd.DataFrame(cal_rows), "calibration_choice.csv")
    save_csv(oof.reset_index().rename(columns={"index": "row"}), "oof_predictions.csv")
    save_csv(test.reset_index().rename(columns={"index": "row"}), "test_predictions.csv")
    return fitted, pd.DataFrame(footprint)


# ---------------------------------------------------------------------------
def run(df: pd.DataFrame):
    tracker = None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="GBR", log_level="error", save_to_file=False)
        tracker.start()
    except Exception:
        tracker = None
    t_all = time.time()
    X_tr, X_te, y_tr, y_te = split(df)
    print("Ablations...", flush=True)
    t0 = time.time()
    cfg = run_ablations(X_tr, y_tr)
    abl_s = time.time() - t0
    print("Chosen config:", {k: cfg[k] for k in ("features_linear", "features_tree", "collapse", "contract_ordinal", "log_totalcharges", "imbalance", "drop_totalcharges")}, flush=True)
    if cfg["drop_totalcharges"]:
        # Not expected; handled by dropping the column from X before modelling.
        X_tr = X_tr.drop(columns=["TotalCharges"]); X_te = X_te.drop(columns=["TotalCharges"])
    compute_vif(X_tr, y_tr, cfg)
    print("Nested CV...", flush=True)
    _, fp1 = nested_cv(X_tr, y_tr, cfg)
    print("Final models...", flush=True)
    fitted, fp2 = final_models(X_tr, y_tr, X_te, y_te, cfg)
    fp = pd.concat([pd.DataFrame([{"model": "all", "stage": "ablations", "seconds": abl_s}]), fp1, fp2], ignore_index=True)
    emissions = None
    if tracker is not None:
        try:
            emissions = tracker.stop()
        except Exception:
            emissions = None
    fp_total = {"total_modelling_seconds": time.time() - t_all, "codecarbon_kg_co2e": emissions,
                "quick_mode": QUICK}
    save_csv(fp, "compute_footprint.csv")
    save_json(fp_total, "compute_footprint_total.json")
    return fitted


if __name__ == "__main__":
    from .clean import clean
    from .load import build_manifest
    raw, _ = build_manifest()
    d, _ = clean(raw)
    run(d)
