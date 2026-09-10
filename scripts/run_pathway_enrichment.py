"""
Run Hallmark pathway enrichment and pathway-level stability analysis
for fold-specific elastic-net gene selections.

Primary analysis:
    FDR-significant Hallmark pathways in each outer fold.

Exploratory analysis:
    recurrence of the top-5 pathways by raw enrichment p-value.

This script intentionally reports gene-level and pathway-level stability
separately. Their raw Jaccard values are not treated as directly comparable
because the candidate universes differ substantially.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).parent.parent / "src")
)

import pathway_enrichment as pe


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
TABLES = Path("results/tables")
MODELS = Path("results/models")


# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

ALPHA = 0.05
TOP_K = 5

MONTE_CARLO_SIMULATIONS = 100000
MONTE_CARLO_SEED = 2026


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLES.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # Load Hallmark pathways
    # ==============================================================

    print(
        "\n=== Parsing Hallmark pathway definitions ==="
    )

    hallmark_path = (
        RAW
        / "h.all.v7.0.symbols.gmt"
    )

    if not hallmark_path.exists():
        raise FileNotFoundError(
            hallmark_path
        )

    pathways = pe.parse_gmt(
        hallmark_path
    )

    print(
        f"Loaded {len(pathways)} Hallmark pathways."
    )

    # ==============================================================
    # Convert Hallmark genes to Entrez IDs
    # ==============================================================

    print(
        "\n=== Resolving Hallmark genes to Entrez IDs ==="
    )

    feature_metadata = pd.read_csv(
        RAW / "feature_metadata.csv",
        index_col=0,
    )

    (
        pathway_entrez,
        manifest,
    ) = pe.resolve_pathways_to_entrez(
        pathways,
        feature_metadata,
    )

    manifest.to_csv(
        PROCESSED
        / "hallmark_manifest.csv",
        index=False,
    )

    print(
        f"Resolved {len(pathway_entrez)} pathways."
    )

    print(
        "Genes resolved per pathway "
        f"(min / median / max): "
        f"{manifest['n_resolved_entrez'].min()} / "
        f"{manifest['n_resolved_entrez'].median():.0f} / "
        f"{manifest['n_resolved_entrez'].max()}"
    )

    # ==============================================================
    # Load elastic-net selections and candidate-gene backgrounds
    # ==============================================================

    selections_path = (
        TABLES
        / "elastic_net_gene_selections_raw.csv"
    )

    background_path = (
        MODELS
        / "elastic_net_background_genes.pkl"
    )

    if not selections_path.exists():
        raise FileNotFoundError(
            f"{selections_path} not found. "
            "Run the elastic-net stability analysis first."
        )

    if not background_path.exists():
        raise FileNotFoundError(
            f"{background_path} not found. "
            "Run the elastic-net stability analysis first."
        )

    selection_df = pd.read_csv(
        selections_path
    )

    with open(
        background_path,
        "rb",
    ) as f:

        background_genes = pickle.load(
            f
        )

    required_selection_columns = {
        "repeat",
        "fold",
        "gene_entrez",
    }

    missing = (
        required_selection_columns
        - set(selection_df.columns)
    )

    if missing:
        raise ValueError(
            f"Selection file missing required columns: {missing}"
        )

    n_folds_total = (
        selection_df[
            ["repeat", "fold"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"\nOuter folds represented: {n_folds_total}"
    )

    # ==============================================================
    # Primary pathway enrichment
    # ==============================================================

    print(
        "\n=== Running fold-specific Hallmark enrichment ==="
    )

    enrichment_df = (
        pe.run_pathway_enrichment(
            selection_df=selection_df,
            background_genes=background_genes,
            pathway_entrez=pathway_entrez,
            alpha=ALPHA,
        )
    )

    enrichment_df.to_csv(
        TABLES
        / "pathway_enrichment_per_fold.csv",
        index=False,
    )

    n_tests = len(
        enrichment_df
    )

    n_significant = int(
        enrichment_df[
            "significant"
        ].sum()
    )

    print(
        f"Completed {n_tests} pathway-fold tests."
    )

    print(
        f"FDR-significant pathway-fold results "
        f"(q < {ALPHA}): {n_significant}"
    )

    # ==============================================================
    # FDR-significant pathway recurrence
    # ==============================================================

    print(
        "\n=== FDR-significant pathway stability ==="
    )

    (
        pathway_frequency,
        pathway_jaccard,
        fold_summary,
    ) = pe.compute_pathway_stability(
        enrichment_df
    )

    pathway_frequency.to_csv(
        TABLES
        / "pathway_selection_frequency.csv",
        index=False,
    )

    pathway_jaccard.to_csv(
        TABLES
        / "pathway_jaccard_summary.csv"
    )

    fold_summary.to_csv(
        TABLES
        / "pathway_fdr_fold_summary.csv"
    )

    print(
        "\nFold-level FDR summary:"
    )

    print(
        fold_summary.to_string()
    )

    print(
        "\nMost frequently FDR-significant pathways:"
    )

    print(
        pathway_frequency.head(
            15
        ).to_string(
            index=False
        )
    )

    if len(
        pathway_jaccard
    ) > 0:

        print(
            "\nFDR-significant pathway-set Jaccard:"
        )

        print(
            pathway_jaccard.to_string()
        )

    else:

        print(
            "\nNo valid FDR-significant pathway-set "
            "Jaccard comparisons were available."
        )

    # ==============================================================
    # Exploratory top-5 recurrence
    # ==============================================================

    print(
        f"\n=== Exploratory top-{TOP_K} pathway recurrence ==="
    )

    (
        topk_frequency,
        top1_frequency,
        topk_jaccard,
    ) = pe.compute_top_pathway_stability(
        enrichment_df,
        top_k=TOP_K,
    )

    topk_frequency.to_csv(
        TABLES
        / f"pathway_top{TOP_K}_frequency.csv",
        index=False,
    )

    top1_frequency.to_csv(
        TABLES
        / "pathway_top1_frequency.csv",
        index=False,
    )

    topk_jaccard.to_csv(
        TABLES
        / f"pathway_top{TOP_K}_jaccard_summary.csv"
    )

    print(
        f"\nMost frequent top-{TOP_K} pathways:"
    )

    print(
        topk_frequency.head(
            15
        ).to_string(
            index=False
        )
    )

    print(
        "\nMost frequent top-ranked pathways:"
    )

    print(
        top1_frequency.head(
            15
        ).to_string(
            index=False
        )
    )

    print(
        f"\nObserved top-{TOP_K} pathway-set Jaccard:"
    )

    print(
        topk_jaccard.to_string()
    )

    # ==============================================================
    # Monte Carlo random top-5 reference
    # ==============================================================

    print(
        "\n=== Monte Carlo chance reference ==="
    )

    n_pathways = (
        enrichment_df[
            "pathway"
        ].nunique()
    )

    random_reference = (
        pe.monte_carlo_random_topk_jaccard(
            n_pathways=n_pathways,
            top_k=TOP_K,
            n_sim=MONTE_CARLO_SIMULATIONS,
            seed=MONTE_CARLO_SEED,
        )
    )

    random_reference_df = pd.DataFrame(
        [random_reference]
    )

    random_reference_df.to_csv(
        TABLES
        / f"pathway_top{TOP_K}_random_jaccard_reference.csv",
        index=False,
    )

    print(
        pd.Series(
            random_reference
        ).to_string()
    )

    # ==============================================================
    # Gene-level stability: report separately
    # ==============================================================

    gene_jaccard_path = (
        TABLES
        / "elastic_net_jaccard_summary.csv"
    )

    if gene_jaccard_path.exists():

        print(
            "\n=== Gene-level stability (reported separately) ==="
        )

        gene_jaccard_df = pd.read_csv(
            gene_jaccard_path,
            index_col=0,
        )

        print(
            gene_jaccard_df.to_string()
        )

    else:

        print(
            "\nGene-level Jaccard summary not found; "
            "skipping display."
        )

    # ==============================================================
    # Interpretation hierarchy
    # ==============================================================

    print(
        "\n=== Interpretation hierarchy ==="
    )

    print(
        "1. FDR-significant pathway recurrence is the primary "
        "pathway result."
    )

    print(
        f"2. Top-{TOP_K} recurrence is exploratory and describes "
        "recurrent subthreshold enrichment patterns."
    )

    print(
        f"3. The random top-{TOP_K} Jaccard distribution is a "
        "descriptive chance reference, not a formal hypothesis test."
    )

    print(
        "4. Gene-level and pathway-level Jaccard values should not "
        "be directly compared as if they were on the same scale."
    )

    print(
        "\nPathway enrichment analysis complete."
    )


if __name__ == "__main__":
    main()