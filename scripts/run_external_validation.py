"""
External transportability analysis for GSE176307.

Models are developed exclusively in IMvigor210 and evaluated once in the
independent GSE176307 cohort. No model fitting, preprocessing estimation,
or hyperparameter tuning is performed using external outcomes.

External comparisons:
    Level 1: C     vs. C + K
    Level 2: C + B vs. C + B + K   (primary)

For external transportability:
    - C contains harmonized clinical predictors.
    - B contains harmonized biomarkers.
    - K scores are recomputed in BOTH cohorts using exactly the same
      marker-gene subset available in both datasets.

Performance:
    - AUPRC
    - AUROC
    - Brier score

Uncertainty:
    - Patient-level bootstrap 95% CIs for each model metric
    - Paired patient-level bootstrap 95% CIs for:
        delta AUPRC
        delta AUROC
        delta Brier

Delta is always:
    metric(+K) - metric(baseline)

Therefore:
    positive delta AUPRC = improvement
    positive delta AUROC = improvement
    negative delta Brier = improvement
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "src"),
)

import modeling as mdl
import preprocessing as pp
import external_validation as ev


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
EXTERNAL = Path("data/external")
TABLES = Path("results/tables")
PREDICTIONS = Path("results/predictions")


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 2026


# =====================================================================
# MODEL FITTING
# =====================================================================

def freeze_model(
    cohort,
    feature_columns,
    y,
    seed=mdl.SEED,
):
    """
    Select ridge regularization strength by five-fold CV using
    IMvigor210 only, then refit the selected pipeline on the full
    IMvigor210 development cohort.
    """

    pipe = mdl.build_pipeline(
        cohort,
        feature_columns,
    )

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    gs = GridSearchCV(
        estimator=pipe,
        param_grid={
            "clf__C": mdl.C_GRID,
        },
        scoring="average_precision",
        cv=inner_cv,
        n_jobs=-1,
    )

    gs.fit(
        cohort[feature_columns],
        y,
    )

    print(
        f"  Frozen model: "
        f"{len(feature_columns)} predictors, "
        f"best_C={gs.best_params_['clf__C']}"
    )

    return gs.best_estimator_


# =====================================================================
# EXTERNAL PREDICTION
# =====================================================================

def evaluate_once(
    fitted_pipe,
    feature_columns,
    external_cohort,
    y_ext,
):
    """
    Apply a frozen IMvigor210 model once to the external cohort.
    """

    proba = fitted_pipe.predict_proba(
        external_cohort[feature_columns]
    )[:, 1]

    return pd.DataFrame(
        {
            "y_true": y_ext.to_numpy(),
            "y_pred_proba": proba,
        },
        index=external_cohort.index,
    )


# =====================================================================
# POINT ESTIMATES
# =====================================================================

def calculate_metrics(preds_df):
    """
    Calculate performance on the complete external cohort.
    """

    y = preds_df["y_true"].to_numpy()
    p = preds_df["y_pred_proba"].to_numpy()

    return {
        "AUPRC": average_precision_score(
            y,
            p,
        ),
        "AUROC": roc_auc_score(
            y,
            p,
        ),
        "Brier": brier_score_loss(
            y,
            p,
        ),
    }


# =====================================================================
# BOOTSTRAP CIs FOR INDIVIDUAL MODELS
# =====================================================================

def bootstrap_metrics(
    preds_df,
    n_boot=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
):
    """
    Patient-level bootstrap 95% confidence intervals for AUPRC,
    AUROC, and Brier score.

    The model is not retrained during bootstrapping.
    """

    rng = np.random.default_rng(seed)

    y_true = preds_df[
        "y_true"
    ].to_numpy()

    y_pred = preds_df[
        "y_pred_proba"
    ].to_numpy()

    n = len(y_true)

    auprcs = []
    aurocs = []
    briers = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yt = y_true[idx]
        yp = y_pred[idx]

        # AUPRC and AUROC require both outcome classes.
        if np.unique(yt).size < 2:
            continue

        auprcs.append(
            average_precision_score(
                yt,
                yp,
            )
        )

        aurocs.append(
            roc_auc_score(
                yt,
                yp,
            )
        )

        briers.append(
            brier_score_loss(
                yt,
                yp,
            )
        )

    auprcs = np.asarray(auprcs)
    aurocs = np.asarray(aurocs)
    briers = np.asarray(briers)

    return {
        "AUPRC_CI_low":
            np.percentile(
                auprcs,
                2.5,
            ),

        "AUPRC_CI_high":
            np.percentile(
                auprcs,
                97.5,
            ),

        "AUROC_CI_low":
            np.percentile(
                aurocs,
                2.5,
            ),

        "AUROC_CI_high":
            np.percentile(
                aurocs,
                97.5,
            ),

        "Brier_CI_low":
            np.percentile(
                briers,
                2.5,
            ),

        "Brier_CI_high":
            np.percentile(
                briers,
                97.5,
            ),

        "n_valid_bootstrap_resamples":
            len(auprcs),
    }


# =====================================================================
# PAIRED BOOTSTRAP FOR DELTAS
# =====================================================================

def bootstrap_paired_delta(
    preds_baseline,
    preds_plusK,
    n_boot=N_BOOTSTRAP,
    seed=BOOTSTRAP_SEED,
):
    """
    Paired patient-level bootstrap comparing +K with its corresponding
    baseline model.

    The same patient indices are resampled for both models during every
    bootstrap draw.

    Delta = metric(+K) - metric(baseline)

    Thus:
        positive delta AUPRC = improvement
        positive delta AUROC = improvement
        negative delta Brier = improvement
    """

    if not preds_baseline.index.equals(
        preds_plusK.index
    ):
        raise ValueError(
            "Baseline and +K prediction frames "
            "must contain the same patients in "
            "the same order."
        )

    y_baseline = preds_baseline[
        "y_true"
    ].to_numpy()

    y_plusK = preds_plusK[
        "y_true"
    ].to_numpy()

    if not np.array_equal(
        y_baseline,
        y_plusK,
    ):
        raise ValueError(
            "Outcome labels differ between "
            "baseline and +K prediction frames."
        )

    y_true = y_baseline

    p_baseline = preds_baseline[
        "y_pred_proba"
    ].to_numpy()

    p_plusK = preds_plusK[
        "y_pred_proba"
    ].to_numpy()

    # -------------------------------------------------------------
    # Observed point estimates on all external patients
    # -------------------------------------------------------------

    baseline_auprc = (
        average_precision_score(
            y_true,
            p_baseline,
        )
    )

    plusK_auprc = (
        average_precision_score(
            y_true,
            p_plusK,
        )
    )

    baseline_auroc = (
        roc_auc_score(
            y_true,
            p_baseline,
        )
    )

    plusK_auroc = (
        roc_auc_score(
            y_true,
            p_plusK,
        )
    )

    baseline_brier = (
        brier_score_loss(
            y_true,
            p_baseline,
        )
    )

    plusK_brier = (
        brier_score_loss(
            y_true,
            p_plusK,
        )
    )

    delta_auprc_point = (
        plusK_auprc
        - baseline_auprc
    )

    delta_auroc_point = (
        plusK_auroc
        - baseline_auroc
    )

    delta_brier_point = (
        plusK_brier
        - baseline_brier
    )

    # -------------------------------------------------------------
    # Paired bootstrap
    # -------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    n = len(y_true)

    delta_auprcs = []
    delta_aurocs = []
    delta_briers = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yt = y_true[idx]

        pb = p_baseline[idx]
        pk = p_plusK[idx]

        if np.unique(yt).size < 2:
            continue

        baseline_boot_auprc = (
            average_precision_score(
                yt,
                pb,
            )
        )

        plusK_boot_auprc = (
            average_precision_score(
                yt,
                pk,
            )
        )

        baseline_boot_auroc = (
            roc_auc_score(
                yt,
                pb,
            )
        )

        plusK_boot_auroc = (
            roc_auc_score(
                yt,
                pk,
            )
        )

        baseline_boot_brier = (
            brier_score_loss(
                yt,
                pb,
            )
        )

        plusK_boot_brier = (
            brier_score_loss(
                yt,
                pk,
            )
        )

        delta_auprcs.append(
            plusK_boot_auprc
            - baseline_boot_auprc
        )

        delta_aurocs.append(
            plusK_boot_auroc
            - baseline_boot_auroc
        )

        delta_briers.append(
            plusK_boot_brier
            - baseline_boot_brier
        )

    delta_auprcs = np.asarray(
        delta_auprcs
    )

    delta_aurocs = np.asarray(
        delta_aurocs
    )

    delta_briers = np.asarray(
        delta_briers
    )

    return {
        "delta_AUPRC":
            delta_auprc_point,

        "delta_AUPRC_CI_low":
            np.percentile(
                delta_auprcs,
                2.5,
            ),

        "delta_AUPRC_CI_high":
            np.percentile(
                delta_auprcs,
                97.5,
            ),

        "delta_AUROC":
            delta_auroc_point,

        "delta_AUROC_CI_low":
            np.percentile(
                delta_aurocs,
                2.5,
            ),

        "delta_AUROC_CI_high":
            np.percentile(
                delta_aurocs,
                97.5,
            ),

        "delta_Brier":
            delta_brier_point,

        "delta_Brier_CI_low":
            np.percentile(
                delta_briers,
                2.5,
            ),

        "delta_Brier_CI_high":
            np.percentile(
                delta_briers,
                97.5,
            ),

        "pct_delta_AUPRC_positive":
            float(
                (
                    delta_auprcs
                    > 0
                ).mean()
            ),

        "pct_delta_AUROC_positive":
            float(
                (
                    delta_aurocs
                    > 0
                ).mean()
            ),

        "pct_delta_Brier_negative":
            float(
                (
                    delta_briers
                    < 0
                ).mean()
            ),

        "n_valid_bootstrap_resamples":
            len(
                delta_auprcs
            ),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():

    TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREDICTIONS.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =============================================================
    # Load IMvigor210
    # =============================================================

    print(
        "=== Loading and preparing "
        "IMvigor210 (training cohort) ==="
    )

    analysis_cohort = pd.read_csv(
        PROCESSED
        / "analysis_cohort.csv",
        index_col=0,
    )

    raw = pp.load_raw()

    _, log2_tpm_entrez = (
        pp.compute_tpm(
            raw[
                "expression_counts"
            ],
            raw[
                "feature_metadata"
            ],
        )
    )

    # =============================================================
    # Load GSE176307
    # =============================================================

    print(
        "\n=== Loading GSE176307 "
        "expression data ==="
    )

    gse_tpm = pd.read_csv(
        EXTERNAL
        / "GSE176307_salmon_tpm_gene_matrix.tsv",
        sep="\t",
        index_col=0,
    )

    gse_log2tpm = np.log2(
        gse_tpm + 1
    )

    # =============================================================
    # Define K signatures
    # =============================================================

    sig_symbols = (
        ev.load_signature_gene_symbols(
            raw[
                "mariathasan_signatures"
            ],
            raw[
                "mcp_genes"
            ],
        )
    )

    # =============================================================
    # Compute harmonized K scores
    # =============================================================

    print(
        "\n=== Computing harmonized "
        "knowledge-guided scores ==="
    )

    (
        imvigor_k_scores,
        gse_k_scores,
        signature_coverage,
    ) = (
        ev.compute_harmonized_signature_scores(
            imvigor_log2tpm_entrez=
                log2_tpm_entrez,

            gse_log2tpm_symbol=
                gse_log2tpm,

            sig_symbols=
                sig_symbols,

            feature_metadata=
                raw[
                    "feature_metadata"
                ],
        )
    )

    signature_coverage.to_csv(
        TABLES
        / "external_validation_signature_coverage.csv",
        index=False,
    )

    print(
        "\nHarmonized signature coverage:"
    )

    print(
        signature_coverage[
            [
                "signature",
                "n_requested",
                "n_common",
                "coverage_fraction",
            ]
        ].to_string(
            index=False
        )
    )

    # =============================================================
    # Construct harmonized IMvigor210 cohort
    # =============================================================

    imvigor = (
        ev.build_imvigor_harmonized_cohort(
            analysis_cohort=
                analysis_cohort,

            log2_tpm_entrez=
                log2_tpm_entrez,

            harmonized_k_scores=
                imvigor_k_scores,
        )
    )

    y_imvigor = (
        imvigor[
            "binaryResponse"
        ]
        == "CR/PR"
    ).astype(int)

    print(
        "\nIMvigor210 harmonized "
        f"development cohort: "
        f"{len(imvigor)} patients, "
        f"{y_imvigor.sum()} responders "
        f"({100 * y_imvigor.mean():.1f}%)"
    )

    # =============================================================
    # Load and harmonize GSE176307 clinical data
    # =============================================================

    print(
        "\n=== Loading and harmonizing "
        "GSE176307 clinical data ==="
    )

    clinical = pd.read_csv(
        EXTERNAL
        / "gse176307_clinical.csv"
    )

    sample_key = pd.read_csv(
        EXTERNAL
        / (
            "GSE176307_BACI_Omniseq_"
            "Sample_Name_Key_submitted_GEO_v2.csv"
        )
    )

    gse_clinical = (
        ev.harmonize_gse176307_clinical(
            clinical,
            sample_key,
        )
    )

    # =============================================================
    # Add CD274 to GSE176307
    # =============================================================

    if "CD274" not in gse_log2tpm.index:
        raise ValueError(
            "CD274 was not found in the "
            "GSE176307 expression matrix."
        )

    gse_cd274 = (
        gse_log2tpm
        .loc["CD274"]
        .rename(
            "CD274"
        )
    )

    # Normalize indices for safe joining
    gse_clinical.index = (
        gse_clinical.index
        .astype(str)
    )

    gse_k_scores.index = (
        gse_k_scores.index
        .astype(str)
    )

    gse_cd274.index = (
        gse_cd274.index
        .astype(str)
    )

    # =============================================================
    # Assemble external cohort
    # =============================================================

    gse = (
        gse_clinical
        .join(
            gse_k_scores,
            how="left",
        )
        .join(
            gse_cd274,
            how="left",
        )
    )

    gse_eval = (
        gse[
            gse[
                "binaryResponse"
            ].notna()
        ]
        .copy()
    )

    y_gse = (
        gse_eval[
            "binaryResponse"
        ]
        == "CR/PR"
    ).astype(int)

    print(
        "\nGSE176307 harmonized "
        f"external cohort: "
        f"{len(gse_eval)} patients, "
        f"{y_gse.sum()} responders "
        f"({100 * y_gse.mean():.1f}%)"
    )

    # =============================================================
    # Validate predictor availability
    # =============================================================

    k_columns = list(
        mdl.FEATURE_BLOCKS["K"]
    )

    missing_imvigor_k = [
        c
        for c in k_columns
        if c not in imvigor.columns
    ]

    missing_gse_k = [
        c
        for c in k_columns
        if c not in gse_eval.columns
    ]

    if missing_imvigor_k:
        raise ValueError(
            "Missing harmonized K columns "
            "in IMvigor210: "
            f"{missing_imvigor_k}"
        )

    if missing_gse_k:
        raise ValueError(
            "Missing harmonized K columns "
            "in GSE176307: "
            f"{missing_gse_k}"
        )

    # =============================================================
    # Define external comparisons
    # =============================================================

    levels = {

        "Level1": {
            "baseline_cols":
                ev.EXT_C,

            "plusK_cols":
                ev.EXT_C
                + k_columns,

            "baseline_label":
                "C",

            "plusK_label":
                "C+K",
        },

        "Level2_primary": {
            "baseline_cols":
                ev.EXT_C
                + ev.EXT_B,

            "plusK_cols":
                ev.EXT_C
                + ev.EXT_B
                + k_columns,

            "baseline_label":
                "C+B",

            "plusK_label":
                "C+B+K",
        },
    }

    all_predictions = []
    all_performance = []
    all_bootstrap = []
    all_deltas = []

    # =============================================================
    # External evaluation
    # =============================================================

    for (
        level_name,
        spec,
    ) in levels.items():

        baseline_cols = (
            spec[
                "baseline_cols"
            ]
        )

        plusK_cols = (
            spec[
                "plusK_cols"
            ]
        )

        baseline_label = (
            spec[
                "baseline_label"
            ]
        )

        plusK_label = (
            spec[
                "plusK_label"
            ]
        )

        print(
            f"\n=== {level_name}: "
            f"{baseline_label} vs. "
            f"{plusK_label} ==="
        )

        # ---------------------------------------------------------
        # Fit/freeze using IMvigor210
        # ---------------------------------------------------------

        model_baseline = freeze_model(
            imvigor,
            baseline_cols,
            y_imvigor,
        )

        model_plusK = freeze_model(
            imvigor,
            plusK_cols,
            y_imvigor,
        )

        # ---------------------------------------------------------
        # Evaluate once externally
        # ---------------------------------------------------------

        preds_baseline = evaluate_once(
            model_baseline,
            baseline_cols,
            gse_eval,
            y_gse,
        )

        preds_plusK = evaluate_once(
            model_plusK,
            plusK_cols,
            gse_eval,
            y_gse,
        )

        preds_baseline[
            "level"
        ] = level_name

        preds_baseline[
            "block"
        ] = baseline_label

        preds_plusK[
            "level"
        ] = level_name

        preds_plusK[
            "block"
        ] = plusK_label

        all_predictions.extend(
            [
                preds_baseline,
                preds_plusK,
            ]
        )

        # ---------------------------------------------------------
        # Individual model performance
        # ---------------------------------------------------------

        for (
            label,
            preds,
        ) in [
            (
                baseline_label,
                preds_baseline,
            ),
            (
                plusK_label,
                preds_plusK,
            ),
        ]:

            point = (
                calculate_metrics(
                    preds
                )
            )

            all_performance.append(
                {
                    "level":
                        level_name,

                    "block":
                        label,

                    "n":
                        len(preds),

                    **point,
                }
            )

            boot = (
                bootstrap_metrics(
                    preds
                )
            )

            boot[
                "level"
            ] = level_name

            boot[
                "block"
            ] = label

            all_bootstrap.append(
                boot
            )

        # ---------------------------------------------------------
        # Paired delta analysis
        # ---------------------------------------------------------

        delta = (
            bootstrap_paired_delta(
                preds_baseline,
                preds_plusK,
            )
        )

        delta[
            "level"
        ] = level_name

        delta[
            "comparison"
        ] = (
            f"{plusK_label} "
            f"minus "
            f"{baseline_label}"
        )

        all_deltas.append(
            delta
        )

        print(
            f"\n  Delta AUPRC: "
            f"{delta['delta_AUPRC']:.4f} "
            f"("
            f"{delta['delta_AUPRC_CI_low']:.4f}, "
            f"{delta['delta_AUPRC_CI_high']:.4f}"
            f")"
        )

        print(
            f"  Delta AUROC: "
            f"{delta['delta_AUROC']:.4f} "
            f"("
            f"{delta['delta_AUROC_CI_low']:.4f}, "
            f"{delta['delta_AUROC_CI_high']:.4f}"
            f")"
        )

        print(
            f"  Delta Brier: "
            f"{delta['delta_Brier']:.4f} "
            f"("
            f"{delta['delta_Brier_CI_low']:.4f}, "
            f"{delta['delta_Brier_CI_high']:.4f}"
            f")"
        )

    # =============================================================
    # Save predictions
    # =============================================================

    predictions_df = pd.concat(
        all_predictions,
        axis=0,
    )

    predictions_df.to_csv(
        PREDICTIONS
        / "external_validation_predictions.csv"
    )

    # =============================================================
    # Save point estimates
    # =============================================================

    performance_df = pd.DataFrame(
        all_performance
    )

    performance_df.to_csv(
        TABLES
        / "external_validation_performance.csv",
        index=False,
    )

    # =============================================================
    # Save individual bootstrap CIs
    # =============================================================

    bootstrap_df = pd.DataFrame(
        all_bootstrap
    )

    bootstrap_df.to_csv(
        TABLES
        / "external_validation_bootstrap_ci.csv",
        index=False,
    )

    # =============================================================
    # Save paired delta bootstrap CIs
    # =============================================================

    delta_df = pd.DataFrame(
        all_deltas
    )

    delta_df.to_csv(
        TABLES
        / "external_validation_delta_bootstrap_ci.csv",
        index=False,
    )

    # =============================================================
    # Console output
    # =============================================================

    print(
        "\n========================================"
    )

    print(
        "EXTERNAL POINT ESTIMATES"
    )

    print(
        "========================================"
    )

    print(
        performance_df.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "INDIVIDUAL MODEL BOOTSTRAP 95% CIs"
    )

    print(
        "========================================"
    )

    print(
        bootstrap_df.to_string(
            index=False
        )
    )

    print(
        "\n========================================"
    )

    print(
        "PAIRED DELTA BOOTSTRAP 95% CIs"
    )

    print(
        "========================================"
    )

    print(
        delta_df.to_string(
            index=False
        )
    )

    print(
        "\nExternal performance point estimates "
        "were calculated on the complete "
        "external cohort. Patient-level "
        "bootstrap resampling was used for "
        "95% confidence intervals. Paired "
        "bootstrap resampling used identical "
        "patient resamples for the baseline "
        "and +K models."
    )

    print(
        "\nPositive delta AUPRC and delta AUROC "
        "indicate improvement with Block K; "
        "negative delta Brier indicates "
        "improved probabilistic accuracy."
    )

    print(
        "\nLevel 3 "
        "(C+B+R vs. C+B+R+K) was not "
        "evaluated because Block R is "
        "unavailable in GSE176307."
    )


if __name__ == "__main__":
    main()