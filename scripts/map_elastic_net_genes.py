"""
Maps every gene in elastic_net_gene_selection_frequency.csv to its symbol
(gene_entrez -> symbol, via feature_metadata.csv, the same lookup used
throughout this project), cross-checked against coefficient sign
consistency -- a gene selected in every fold but with an inconsistent
coefficient sign across those folds is a materially weaker finding than
one that's both universally selected AND directionally stable, so both
are reported together, not just the frequency number alone.

Saves the FULL mapped gene list (all selection frequencies), and prints a
separate summary specifically for the 100%-frequency subset, since that's
usually the headline number even though the full list is what gets saved.

Run from the project root:
    python3 map_elastic_net_genes.py

Requires:
    results/tables/elastic_net_gene_selection_frequency.csv
    results/tables/elastic_net_sign_consistency.csv
    data/raw/feature_metadata.csv

Writes:
    results/tables/elastic_net_genes_mapped.csv   (ALL genes, all frequencies)
"""

from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
TABLES = Path("results/tables")


def main():
    freq = pd.read_csv(TABLES / "elastic_net_gene_selection_frequency.csv")
    sign = pd.read_csv(TABLES / "elastic_net_sign_consistency.csv")
    fm = pd.read_csv(RAW / "feature_metadata.csv")

    # Same ambiguous-symbol handling used everywhere else in this project:
    # drop symbols that map to more than one Entrez ID rather than guess.
    fm_valid = fm.dropna(subset=["symbol"])
    counts = fm_valid["symbol"].value_counts()
    ambiguous_symbols = set(counts[counts > 1].index)
    entrez_to_symbol = dict(zip(fm_valid["entrez_id"], fm_valid["symbol"]))
    entrez_to_symbol = {
        e: s for e, s in entrez_to_symbol.items() if s not in ambiguous_symbols
    }

    merged = freq.merge(sign, on="gene_entrez", how="left")
    merged["symbol"] = merged["gene_entrez"].map(entrez_to_symbol)
    merged = merged.sort_values(
        ["selection_frequency", "sign_consistency"], ascending=[False, False]
    )

    n_total = len(merged)
    n_unmapped = merged["symbol"].isna().sum()
    print(f"{n_total} genes total in the selection frequency file.")
    if n_unmapped > 0:
        pct = 100 * n_unmapped / n_total
        print(f"WARNING: {n_unmapped} ({pct:.1f}%) could not be mapped to a symbol "
              f"(not found in feature_metadata.csv, or symbol was ambiguous). "
              f"Their gene_entrez IDs are retained in the output but symbol is blank.")

    TABLES.mkdir(parents=True, exist_ok=True)
    out_path = TABLES / "elastic_net_genes_mapped.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved full mapped gene list ({n_total} genes) to {out_path}")

    # Separate summary for the 100%-frequency subset specifically
    top = merged[merged["selection_frequency"] >= 0.999]
    print(f"\n=== {len(top)} genes selected in 100% of folds ===")
    print(top[["gene_entrez", "symbol", "times_selected", "selection_frequency", "sign_consistency"]]
          .to_string(index=False))
    n_also_sign_stable = (top["sign_consistency"] >= 0.999).sum()
    print(f"\n{n_also_sign_stable} of {len(top)} are ALSO 100% sign-consistent "
          f"(selected every fold, same coefficient direction every time).")


if __name__ == "__main__":
    main()
