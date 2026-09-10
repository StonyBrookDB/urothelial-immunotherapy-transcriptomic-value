"""
Diagnostic: with folds selecting hundreds of genes (not the ~10-20 originally
assumed), the gene-level Jaccard needs to be compared against the correct
chance baseline for THIS scale, not eyeballed in isolation. This computes
that baseline from your actual background pool sizes and selection counts.

Run from the project root, after run_elastic_net_stability.py:
    python3 check_gene_selection_scale.py

Requires:
    results/models/elastic_net_background_genes.pkl
    results/tables/elastic_net_genes_per_fold.csv
    results/tables/elastic_net_jaccard_summary.csv
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = Path("results/models")
TABLES = Path("results/tables")


def main():
    with open(MODELS / "elastic_net_background_genes.pkl", "rb") as f:
        background_genes = pickle.load(f)

    bg_sizes = pd.Series({k: len(v) for k, v in background_genes.items()})
    print("=== Background (variance-floor-filtered) pool size per fold ===")
    print(f"min/median/max: {bg_sizes.min()} / {bg_sizes.median():.0f} / {bg_sizes.max()}")
    print(f"(This is how many genes were even ELIGIBLE for selection in each fold.)\n")

    genes_per_fold = pd.read_csv(TABLES / "elastic_net_genes_per_fold.csv")
    print("=== Actual genes selected per fold ===")
    print(genes_per_fold["n_genes_selected"].describe())

    mean_k = genes_per_fold["n_genes_selected"].mean()
    mean_N = bg_sizes.mean()
    print(f"\nMean selected: {mean_k:.0f}, mean background pool: {mean_N:.0f} "
          f"({100*mean_k/mean_N:.1f}% of eligible genes selected on average)")

    # Expected Jaccard for two independent random draws of size k from a
    # population of size N (approximation, treating draws as independent
    # random subsets -- close enough for a sanity-check baseline)
    expected_overlap = mean_k * mean_k / mean_N
    expected_union = 2 * mean_k - expected_overlap
    expected_random_jaccard = expected_overlap / expected_union if expected_union > 0 else float("nan")

    print(f"\n=== Chance-baseline Jaccard for random {mean_k:.0f}-of-{mean_N:.0f} draws ===")
    print(f"Expected Jaccard under pure randomness: {expected_random_jaccard:.4f}")

    jaccard_path = TABLES / "elastic_net_jaccard_summary.csv"
    if jaccard_path.exists():
        observed = pd.read_csv(jaccard_path, index_col=0).iloc[:, 0]
        observed_mean = observed.get("mean")
        print(f"Observed gene-level Jaccard mean: {observed_mean:.4f}")
        print(f"Ratio (observed / chance baseline): {observed_mean / expected_random_jaccard:.2f}x")
        print(
            "\nNo interpretation forced here. A ratio near 1.0 means the "
            "observed overlap is indistinguishable from what random ~"
            f"{mean_k:.0f}-gene draws would produce by chance -- i.e. "
            "genuinely unstable selection, not just 'low but real' overlap. "
            "A ratio well above 1.0 means there IS real structure/"
            "reproducibility in what's being selected, even though it isn't "
            "a compact sparse panel."
        )
    else:
        print(f"\n({jaccard_path} not found -- run this after "
              f"run_elastic_net_stability.py's full pipeline completes.)")


if __name__ == "__main__":
    main()
