"""
Modeling infrastructure for the AMIA IMvigor210 reanalysis.
Implements Sections 4-5 of analysis_spec.md: ridge logistic regression,
repeated nested CV, fold-safe preprocessing, AUPRC as primary metric.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV,
)
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

MODELS = Path("results/models")
PREDICTIONS = Path("results/predictions")
TABLES = Path("results/tables")

N_OUTER_SPLITS = 5
N_REPEATS = 10
SEED = 2026
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Columns that are numeric dtype but should be treated as ordinal/categorical
# (per the audit note: ECOG is stored as {0,1,2} but is not a plain continuum).
CATEGORICAL_OVERRIDE = {"Baseline ECOG Score"}

FEATURE_BLOCKS = {
    "C": ["Sex", "Baseline ECOG Score", "Tobacco Use History", "Met Disease Status",
          "Received platinum", "Intravesical BCG administered"],
    "B": ["IC Level", "TC Level", "FMOne mutation burden per MB"],
    "R": ["Neoantigen burden per MB", "Immune phenotype"],
    "K": ["CD8_T_effector", "Immune_Checkpoint", "APM", "NK_cells",
          "Cytotoxic_lymphocytes", "Monocytic_lineage", "Fibroblasts", "Endothelial_cells"],
}


def get_feature_columns(*block_names):
    cols = []
    for b in block_names:
        cols += FEATURE_BLOCKS[b]
    return cols


def generate_or_load_cv_splits(y, path=None):
    """
    Generate repeated stratified CV splits once and persist them, or load
    existing ones. Every model/feature-set combination must reuse the same
    splits — do not regenerate per model.
    """
    path = path or (MODELS / "cv_splits.pkl")
    if path.exists():
        with open(path, "rb") as f:
            splits = pickle.load(f)
        print(f"Loaded existing CV splits from {path} ({len(splits)} train/test pairs). Not regenerated.")
        return splits

    rskf = RepeatedStratifiedKFold(
        n_splits=N_OUTER_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    splits = list(rskf.split(np.zeros(len(y)), y))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Generated {len(splits)} CV splits ({N_REPEATS} repeats x {N_OUTER_SPLITS} folds), saved to {path}.")
    return splits


def build_pipeline(cohort, feature_columns):
    """
    Fold-safe preprocessing + ridge logistic regression.
    Numeric: median impute (+ missingness indicator) -> standardize.
    Categorical: constant-impute as 'Missing' -> one-hot (unknown categories tolerated).
    """
    numeric_cols = [
        c for c in feature_columns
        if c not in CATEGORICAL_OVERRIDE and pd.api.types.is_numeric_dtype(cohort[c])
    ]
    categorical_cols = [c for c in feature_columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000, class_weight=None
        )),
    ])
    return pipe


def calibration_slope_intercept(y_true, y_pred_proba):
    """Fit logit(y) ~ logit(p_hat); returns (slope, intercept)."""
    eps = 1e-6
    p = np.clip(y_pred_proba, eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    # C=np.inf with an l2 penalty is the version-stable way to get an
    # effectively unpenalized fit (penalty=None was deprecated in newer sklearn).
    lr = LogisticRegression(penalty="l2", C=np.inf, solver="lbfgs", max_iter=2000)
    lr.fit(logit_p, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def run_nested_cv(cohort, feature_columns, y, splits, block_label):
    """
    Runs repeated nested CV for one feature-set. Inner CV (5-fold) tunes the
    ridge C hyperparameter by AUPRC; outer CV produces held-out predictions.
    Returns (oof_predictions_df, per_repeat_metrics_df).
    """
    pipe = build_pipeline(cohort, feature_columns)
    param_grid = {"clf__C": C_GRID}
    X = cohort[feature_columns]

    records = []
    for i, (train_idx, test_idx) in enumerate(splits):
        repeat_num = i // N_OUTER_SPLITS
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + repeat_num)
        gs = GridSearchCV(pipe, param_grid, scoring="average_precision", cv=inner_cv, n_jobs=-1)
        gs.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = gs.predict_proba(X.iloc[test_idx])[:, 1]

        for pos, p in zip(test_idx, proba):
            records.append({
                "block": block_label,
                "repeat": repeat_num,
                "patient_index": X.index[pos],
                "y_true": int(y.iloc[pos]),
                "y_pred_proba": float(p),
                "best_C": gs.best_params_["clf__C"],
            })
        print(f"  [{block_label}] repeat {repeat_num}, fold {i % N_OUTER_SPLITS} done "
              f"(best_C={gs.best_params_['clf__C']})")

    oof_df = pd.DataFrame(records)

    metric_rows = []
    for repeat, g in oof_df.groupby("repeat"):
        auprc = average_precision_score(g["y_true"], g["y_pred_proba"])
        auroc = roc_auc_score(g["y_true"], g["y_pred_proba"])
        brier = brier_score_loss(g["y_true"], g["y_pred_proba"])
        slope, intercept = calibration_slope_intercept(g["y_true"].values, g["y_pred_proba"].values)
        metric_rows.append({
            "block": block_label, "repeat": repeat, "AUPRC": auprc, "AUROC": auroc,
            "Brier": brier, "calibration_slope": slope, "calibration_intercept": intercept,
        })
    metrics_df = pd.DataFrame(metric_rows)

    return oof_df, metrics_df
