"""
Feature importance for the primary model, via standardized ridge logistic
regression coefficients -- not SHAP or permutation importance (see
discussion: for a linear model, SHAP is a per-patient rescaling of the same
coefficients with no new information; permutation importance is known to
behave unreliably under feature correlation, which K's 8 deliberately-
uncorrelation-filtered signatures exhibit by design).

Fits C+B and C+B+K on the FULL feature blocks used throughout the primary
internal analysis (NOT the reduced/harmonized external-validation blocks
in external_validation.py -- these are a separate, unrelated pair of
models). Both fit on all 298 IMvigor210 response-evaluable patients, same
"freeze" pattern (inner-CV hyperparameter selection + final fit on
everything) used for external validation, but purely to extract
coefficients here -- no external cohort involved.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import modeling as mdl


def freeze_model(cohort, feature_columns, y, seed=mdl.SEED):
    """Inner-CV hyperparameter selection + final fit on ALL provided patients."""
    pipe = mdl.build_pipeline(cohort, feature_columns)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, {"clf__C": mdl.C_GRID}, scoring="average_precision", cv=inner_cv, n_jobs=-1)
    gs.fit(cohort[feature_columns], y)
    return gs.best_estimator_


def clean_feature_name(name):
    """Strips ColumnTransformer prefixes ('num__', 'cat__') for readability."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def extract_named_coefficients(fitted_pipeline):
    """
    Returns a Series of standardized coefficients, indexed by cleaned
    feature name, sorted by absolute magnitude descending.
    """
    preprocessor = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    raw_names = preprocessor.get_feature_names_out()
    clean_names = [clean_feature_name(n) for n in raw_names]
    coefs = pd.Series(clf.coef_.ravel(), index=clean_names)
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)


def build_comparison_table(coefs_cb, coefs_cbk):
    """
    Outer-joins the two coefficient sets on feature name. Features unique to
    C+B+K (the 8 K signatures, plus any missingness-indicator columns that
    only appear because of a fold/model-specific missing pattern) show NaN
    for C+B. Shift = coefficient in C+B+K minus coefficient in C+B, only
    meaningful for features present in both.
    """
    df = pd.DataFrame({"coef_CB": coefs_cb, "coef_CBK": coefs_cbk})
    df["shift_CBK_minus_CB"] = df["coef_CBK"] - df["coef_CB"]
    df["abs_shift"] = df["shift_CBK_minus_CB"].abs()
    return df.sort_values("abs_shift", ascending=False, na_position="last")
