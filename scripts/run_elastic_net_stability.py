"""
Step 5 driver: data-driven elastic net feature selection + stability
analysis, evaluated at Level 2 (C+B vs. C+B+ElasticNet) — the direct
counterpart to block K's C+B vs. C+B+K comparison.

COMPUTE WARNING: this is much slower than the ridge-only scripts. Elastic
net with its own inner grid search, refit on every outer fold, on a
several-thousand-gene matrix, takes meaningfully longer than anything
you've run so far. N_REPEATS_TO_USE below defaults to a SUBSET of your full
10 repeats to keep this tractable on a first pass — raise it if you have
the time/compute for the full 10.

Run from the project root, after run_primary_model.py:
    python3 run_elastic_net_stability.py

Requires:
    results/models/cv_splits.pkl
    data/processed/analysis_cohort.csv
    data/processed/expression_log2TPM.csv

Writes:
    results/predictions/elastic_net_oof_predictions.csv
    results/tables/elastic_net_performance.csv
    results/tables/elastic_net_convergence.csv
    results/tables/elastic_net_gene_selections_raw.csv
    results/tables/elastic_net_genes_per_fold.csv
    results/models/elastic_net_background_genes.pkl
    results/tables/elastic_net_gene_selection_frequency.csv
    results/tables/elastic_net_sign_consistency.csv
    results/tables/elastic_net_jaccard_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import modeling as mdl
import elastic_net as enet

PROCESSED = Path("data/processed")
PREDICTIONS = Path("results/predictions")
TABLES = Path("results/tables")
MODELS = Path("results/models")

# Number of the 10 repeats to actually run (each repeat = 5 outer folds).
# Set to 10 to use everything, if you have the time/compute.
N_REPEATS_TO_USE = 10


def main():
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)
    cohort_eval = cohort[cohort["binaryResponse"].notna()].copy()
    y = (cohort_eval["binaryResponse"] == "CR/PR").astype(int)

    splits_path = MODELS / "cv_splits.pkl"
    assert splits_path.exists(), (
        "results/models/cv_splits.pkl not found — run run_primary_model.py first. "
        "This must reuse the exact same splits as everything else."
    )
    all_splits = mdl.generate_or_load_cv_splits(y, path=splits_path)
    splits = all_splits[: N_REPEATS_TO_USE * mdl.N_OUTER_SPLITS]
    print(f"Using {len(splits)} of {len(all_splits)} total outer fits "
          f"({N_REPEATS_TO_USE} of {mdl.N_REPEATS} repeats) for compute tractability.")

    log2_tpm = pd.read_csv(PROCESSED / "expression_log2TPM.csv", index_col=0)
    log2_tpm.index = log2_tpm.index.astype(int)
    log2_tpm.columns = log2_tpm.columns.astype(str)

    cohort_eval.index = cohort_eval.index.astype(str)
    missing = set(cohort_eval.index) - set(log2_tpm.columns)
    assert not missing, f"{len(missing)} cohort sample IDs not found in log2_tpm columns."

    print("\nRunning elastic net selection + evaluation across folds "
          "(this will take a while)...\n")
    oof_df, metrics_df, selection_df, convergence_df, background_genes = enet.run_elastic_net_arm(
        cohort_eval, log2_tpm, y, splits
    )

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(PREDICTIONS / "elastic_net_oof_predictions.csv", index=False)
    metrics_df.to_csv(TABLES / "elastic_net_performance.csv", index=False)
    convergence_df.to_csv(TABLES / "elastic_net_convergence.csv", index=False)

    # Raw per-fold gene selections (repeat, fold, gene_entrez, coef_sign) --
    # this was previously computed and discarded, only feeding the
    # aggregated tables below. Saved explicitly now because it's the
    # required input for any future fold-level analysis (e.g. pathway
    # enrichment per fold), which the aggregated tables cannot reconstruct.
    selection_df.to_csv(TABLES / "elastic_net_gene_selections_raw.csv", index=False)

    # Clean per-fold gene COUNT summary, derived from the raw file above.
    genes_per_fold = (
        selection_df.groupby(["repeat", "fold"]).size()
        .rename("n_genes_selected").reset_index()
    )
    genes_per_fold.to_csv(TABLES / "elastic_net_genes_per_fold.csv", index=False)

    # Per-fold background gene sets (post variance-floor prefilter) -- the
    # correct background/universe for enrichment testing on that fold's
    # selected genes. Saved as pickle (dict of (repeat,fold) -> gene list),
    # not CSV, since this is an intermediate computational artifact (50
    # folds x ~10-15k genes each) rather than something for manual review.
    import pickle
    with open(MODELS / "elastic_net_background_genes.pkl", "wb") as f:
        pickle.dump(background_genes, f)

    freq, jaccard_summary, sign_consistency = enet.compute_stability_metrics(
        selection_df, n_folds_total=len(splits)
    )
    freq.to_csv(TABLES / "elastic_net_gene_selection_frequency.csv", index=False)
    sign_consistency.to_csv(TABLES / "elastic_net_sign_consistency.csv", index=False)
    jaccard_summary.to_csv(TABLES / "elastic_net_jaccard_summary.csv")

    print(f"\nSaved:")
    print(f"  {PREDICTIONS / 'elastic_net_oof_predictions.csv'} ({len(oof_df)} rows)")
    print(f"  {TABLES / 'elastic_net_performance.csv'} ({len(metrics_df)} repeats)")
    print(f"  {TABLES / 'elastic_net_gene_selections_raw.csv'} "
          f"({len(selection_df)} gene-selection records -- required input for "
          f"any future fold-level analysis, e.g. pathway enrichment per fold)")
    print(f"  {TABLES / 'elastic_net_genes_per_fold.csv'} "
          f"({len(genes_per_fold)} rows -- one per (repeat, fold), with "
          f"n_genes_selected)")
    print(f"  {MODELS / 'elastic_net_background_genes.pkl'} "
          f"(per-fold variance-filtered background gene sets, for use as "
          f"the enrichment-test universe -- NOT the full gene panel)")
    print(f"  {TABLES / 'elastic_net_convergence.csv'} "
          f"({(~convergence_df['converged']).sum()} of {len(convergence_df)} folds "
          f"did not converge)")
    print(f"  {TABLES / 'elastic_net_gene_selection_frequency.csv'} "
          f"({freq.shape[0]} unique genes selected at least once)")
    print(f"  {TABLES / 'elastic_net_sign_consistency.csv'}")
    print(f"  {TABLES / 'elastic_net_jaccard_summary.csv'}")
    print("\nNo interpretation rendered here. Compare elastic_net_performance.csv's "
          "AUPRC against primary_model_performance.csv's AUPRC_CBK (same repeats "
          "only, since N_REPEATS_TO_USE may be a subset), and weigh both against "
          "the selection-frequency/Jaccard/sign-consistency stability metrics.")


if __name__ == "__main__":
    main()
