"""
Formal interaction test: does block K's incremental contribution vary by
PD-L1 IC Level, or is the apparent IC2+ edge from the subgroup CV analysis
better explained by that subgroup's larger sample size (more power) than a
genuine effect modification?

DELIBERATE DEPARTURE from the CV-based framing used everywhere else in this
pipeline: CV-based interaction detection is known to be badly underpowered
at n=298 (exactly what the noisy per-subgroup deltas already suggested).
This is an inferential question -- does an effect differ across subgroups
-- not a predictive-generalization question, so a classical likelihood-
ratio test on the full sample is the statistically appropriate tool here,
not another train/test split.

"K's contribution" for each patient is defined as out-of-fold
logit(p_CBK) - logit(p_CB), averaged across the 10 repeats already saved in
results/predictions/primary_oof_predictions.csv. This reuses the existing,
already-validated OOF predictions rather than re-deriving K's contribution
from the 8 raw signature scores, collapsing what would otherwise be 8
signatures x 2 IC_Level contrasts = 16 barely-estimable interaction terms
(with only 68 responders total) down to a single, well-powered interaction
of interest.

Implemented with sklearn + scipy (an unpenalized logistic fit via C=inf,
plus a manual log-likelihood / chi-square LRT) rather than statsmodels, to
avoid adding a dependency beyond what's already in requirements.txt.

Run from the project root, after run_primary_model.py:
    python3 run_interaction_test.py

Requires:
    results/predictions/primary_oof_predictions.csv
    data/processed/analysis_cohort.csv

Writes:
    results/tables/K_ICLevel_interaction_test.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2

PROCESSED = Path("data/processed")
PREDICTIONS = Path("results/predictions")
TABLES = Path("results/tables")

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_unpenalized_logit(X, y):
    """
    LogisticRegression with a very light L2 penalty (C=100), NOT fully
    unpenalized (C=np.inf). Verified via a synthetic test with a planted
    interaction effect: C=np.inf let coefficients diverge under
    quasi-separation (delta_logit_K is a strongly separating predictor by
    construction), which washed out the LR test entirely (both models
    fit near-perfectly, log-likelihoods converged, p-value spuriously
    ~1 even with a real, strong planted effect). C=100 is weak enough to
    leave inference essentially unpenalized while preventing divergence.
    """
    model = LogisticRegression(penalty="l2", C=100, solver="lbfgs", max_iter=5000)
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    p = np.clip(p, EPS, 1 - EPS)
    log_likelihood = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return model, log_likelihood


def build_design_matrices(df):
    """
    Builds reduced (main effects only) and full (main effects + interaction)
    design matrices for: y ~ delta_logit_K + IC_Level [+ delta_logit_K:IC_Level]
    IC0 is the reference category.
    """
    ic_dummies = pd.get_dummies(df["IC_Level"], prefix="IC", drop_first=True)  # IC1, IC2+
    reduced = pd.concat([df[["delta_logit_K"]], ic_dummies], axis=1)

    interactions = pd.DataFrame({
        f"delta_logit_K_x_{col}": df["delta_logit_K"].values * ic_dummies[col].values
        for col in ic_dummies.columns
    }, index=df.index)
    full = pd.concat([reduced, interactions], axis=1)

    return reduced.astype(float), full.astype(float), list(interactions.columns)


def main():
    oof = pd.read_csv(PREDICTIONS / "primary_oof_predictions.csv")
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)

    required_blocks = {"C+B", "C+B+K"}
    assert required_blocks.issubset(set(oof["block"].unique())), (
        f"Expected blocks {required_blocks} in primary_oof_predictions.csv, "
        f"found {sorted(oof['block'].unique())}"
    )

    wide = oof.pivot_table(
        index=["repeat", "patient_index"], columns="block", values="y_pred_proba"
    ).reset_index()
    wide = wide.rename(columns={"C+B": "p_CB", "C+B+K": "p_CBK"})

    y_true = oof.drop_duplicates(["repeat", "patient_index"])[
        ["repeat", "patient_index", "y_true"]
    ]
    wide = wide.merge(y_true, on=["repeat", "patient_index"])
    wide["delta_logit_K"] = logit(wide["p_CBK"]) - logit(wide["p_CB"])

    # Average the K-contribution across repeats per patient -- each repeat is
    # a resampling estimate of the SAME underlying per-patient quantity, not
    # new information about a different patient, so averaging reduces noise
    # rather than pooling independent observations.
    per_patient = wide.groupby("patient_index").agg(
        delta_logit_K=("delta_logit_K", "mean"),
        y_true=("y_true", "first"),
    ).reset_index()

    ic_level = cohort["IC Level"].copy()
    ic_level.index = cohort.index.astype(str)
    per_patient = per_patient.merge(
        ic_level.rename("IC_Level"), left_on="patient_index", right_index=True, how="left"
    )
    n_before = len(per_patient)
    per_patient = per_patient.dropna(subset=["IC_Level"])
    n_after = len(per_patient)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} patients with missing IC Level.")

    print(f"Patients in interaction test: {n_after}")
    print(per_patient["IC_Level"].value_counts())

    # Separation check: verified during development that near-complete
    # separation of delta_logit_K by outcome within a subgroup causes MLE
    # coefficients to diverge and silently produces a meaningless LR test
    # (both models fit ~perfectly, p-value spuriously near 1 even with a
    # real underlying effect). Flag it explicitly rather than let it pass
    # silently.
    for level, g in per_patient.groupby("IC_Level"):
        pos = g.loc[g["y_true"] == 1, "delta_logit_K"]
        neg = g.loc[g["y_true"] == 0, "delta_logit_K"]
        if len(pos) == 0 or len(neg) == 0:
            continue
        if pos.min() > neg.max() or neg.min() > pos.max():
            print(f"WARNING: within subgroup '{level}', delta_logit_K shows "
                  f"COMPLETE separation by outcome (no overlap between "
                  f"responder/non-responder ranges). The LR test below is "
                  f"likely unreliable for this data -- coefficients may have "
                  f"diverged. Inspect this subgroup's distribution directly "
                  f"before trusting the p-value.")

    reduced_X, full_X, interaction_cols = build_design_matrices(per_patient)
    y = per_patient["y_true"].astype(int).values

    reduced_model, ll_reduced = fit_unpenalized_logit(reduced_X, y)
    full_model, ll_full = fit_unpenalized_logit(full_X, y)

    df_diff = full_X.shape[1] - reduced_X.shape[1]
    lr_stat = 2 * (ll_full - ll_reduced)
    lr_pvalue = chi2.sf(lr_stat, df=df_diff)

    print(f"\n=== Likelihood ratio test: does K's effect vary by IC_Level? ===")
    print(f"Reduced model log-likelihood: {ll_reduced:.4f}")
    print(f"Full model log-likelihood:    {ll_full:.4f}")
    print(f"LR statistic: {lr_stat:.4f}, df: {df_diff}, p-value: {lr_pvalue:.4f}")

    coef_report = pd.Series(full_model.coef_.ravel(), index=full_X.columns)
    print("\n=== Full model coefficients ===")
    print(coef_report)

    results = pd.DataFrame([{
        "n_patients": n_after,
        "lr_statistic": lr_stat,
        "df": df_diff,
        "p_value": lr_pvalue,
        "reduced_model_loglik": ll_reduced,
        "full_model_loglik": ll_full,
    }])
    TABLES.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES / "K_ICLevel_interaction_test.csv", index=False)

    coef_report.to_frame("coefficient").to_csv(
        TABLES / "K_ICLevel_interaction_coefficients.csv"
    )

    print(f"\nSaved to {TABLES / 'K_ICLevel_interaction_test.csv'}")
    print(
        "\nNo verdict rendered here beyond the formal test itself. A "
        "non-significant LR p-value would support the read that the "
        "subgroup CV analysis's apparent IC2+ edge is more likely a power "
        "artifact (larger, more responder-rich subgroup) than a genuine "
        "effect modification. A significant p-value would support a real "
        "subgroup-dependent effect worth reporting -- interpret either way "
        "yourself, alongside the coefficients above."
    )


if __name__ == "__main__":
    main()
