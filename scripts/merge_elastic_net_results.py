"""
Consolidates the elastic net arm's scattered output files into one view:
  - Performance comparison: AUPRC_CB, AUPRC_CBK, AUPRC_ElasticNet, both deltas, per repeat
  - Convergence summary: how many folds converged, any non-converged flagged
  - Top genes by selection frequency, with sign consistency attached
  - Jaccard summary (fold-to-fold selection overlap)

Run from the project root, after run_elastic_net_stability.py completes:
    python3 merge_elastic_net_results.py

Reads (all from results/tables/, all optional except elastic_net_performance.csv):
    primary_model_performance.csv
    elastic_net_performance.csv
    elastic_net_convergence.csv
    elastic_net_gene_selection_frequency.csv
    elastic_net_sign_consistency.csv
    elastic_net_jaccard_summary.csv

Writes:
    results/tables/final_comparison_CBK_vs_ElasticNet.csv
    results/tables/elastic_net_top_genes.csv
"""

from pathlib import Path
import pandas as pd

TABLES = Path("results/tables")


def build_performance_comparison():
    enet_path = TABLES / "elastic_net_performance.csv"
    if not enet_path.exists():
        print(f"({enet_path} not found — nothing to compare yet)")
        return None
    enet = pd.read_csv(enet_path).rename(columns={"AUPRC": "AUPRC_ElasticNet",
                                                    "AUROC": "AUROC_ElasticNet",
                                                    "Brier": "Brier_ElasticNet"})
    enet = enet.drop(columns=["block"], errors="ignore")

    primary_path = TABLES / "primary_model_performance.csv"
    if not primary_path.exists():
        print(f"({primary_path} not found — showing ElasticNet arm alone, no C+B/C+B+K comparison)")
        return enet

    primary = pd.read_csv(primary_path)
    keep = ["repeat", "AUPRC_CB", "AUPRC_CBK", "AUROC_CB", "AUROC_CBK", "Brier_CB", "Brier_CBK"]
    merged = enet.merge(primary[keep], on="repeat", how="left")

    merged["delta_AUPRC_CBK_vs_CB"] = merged["AUPRC_CBK"] - merged["AUPRC_CB"]
    merged["delta_AUPRC_ElasticNet_vs_CB"] = merged["AUPRC_ElasticNet"] - merged["AUPRC_CB"]

    n_repeats = len(merged)
    matched = merged["AUPRC_CB"].notna().sum()
    if matched < n_repeats:
        print(f"WARNING: only {matched}/{n_repeats} elastic net repeats have a matching "
              f"primary-run repeat number — comparison is incomplete for the rest.")

    cols = ["repeat", "AUPRC_CB", "AUPRC_CBK", "AUPRC_ElasticNet",
            "delta_AUPRC_CBK_vs_CB", "delta_AUPRC_ElasticNet_vs_CB",
            "AUROC_CB", "AUROC_CBK", "AUROC_ElasticNet",
            "Brier_CB", "Brier_CBK", "Brier_ElasticNet"]
    merged = merged[[c for c in cols if c in merged.columns]]

    merged.to_csv(TABLES / "final_comparison_CBK_vs_ElasticNet.csv", index=False)

    print("=== Performance comparison (per repeat) ===")
    print(merged.to_string(index=False))
    print()
    print("=== Summary across repeats ===")
    summary_cols = [c for c in merged.columns if c.startswith("delta_") or c.startswith("AUPRC")]
    print(merged[summary_cols].agg(["mean", "median", "std"]).T)
    print(f"\nSaved to {TABLES / 'final_comparison_CBK_vs_ElasticNet.csv'}")
    return merged


def summarize_convergence():
    path = TABLES / "elastic_net_convergence.csv"
    if not path.exists():
        print(f"\n({path} not found — skipped)")
        return None
    conv = pd.read_csv(path)
    n_total = len(conv)
    n_converged = conv["converged"].sum()
    print(f"\n=== Convergence ===")
    print(f"{n_converged}/{n_total} folds converged.")
    if n_converged < n_total:
        print("Non-converged folds (do not trust these folds' gene selections):")
        print(conv[~conv["converged"]].to_string(index=False))
    print(f"n_iter range: {conv['n_iter'].min()}-{conv['n_iter'].max()} "
          f"(max_iter cap = {conv['max_iter'].iloc[0]})")
    return conv


def build_top_genes():
    freq_path = TABLES / "elastic_net_gene_selection_frequency.csv"
    if not freq_path.exists():
        print(f"\n({freq_path} not found — skipped)")
        return None
    freq = pd.read_csv(freq_path)

    sign_path = TABLES / "elastic_net_sign_consistency.csv"
    if sign_path.exists():
        sign = pd.read_csv(sign_path)
        freq = freq.merge(sign, on="gene_entrez", how="left")

    freq = freq.sort_values("selection_frequency", ascending=False)
    freq.to_csv(TABLES / "elastic_net_top_genes.csv", index=False)

    print(f"\n=== Top genes by selection frequency ===")
    print(freq.head(20).to_string(index=False))
    print(f"\nSaved to {TABLES / 'elastic_net_top_genes.csv'}")
    return freq


def print_jaccard():
    path = TABLES / "elastic_net_jaccard_summary.csv"
    if not path.exists():
        print(f"\n({path} not found — skipped)")
        return
    jac = pd.read_csv(path, index_col=0)
    print(f"\n=== Jaccard summary (fold-to-fold gene selection overlap) ===")
    print(jac.to_string())


def main():
    build_performance_comparison()
    summarize_convergence()
    build_top_genes()
    print_jaccard()
    print(
        "\nNo verdict rendered here by design. This consolidates the files — "
        "the read on what the numbers mean (stability vs. performance vs. "
        "convergence, taken together) is yours."
    )


if __name__ == "__main__":
    main()
