"""
Pathway enrichment and pathway-level stability analysis for genes selected
by elastic-net models across outer cross-validation folds.

Primary pathway analysis:
    - MSigDB Hallmark pathways
    - hypergeometric overrepresentation test within each outer fold
    - fold-specific candidate-gene universe
    - Benjamini-Hochberg FDR correction within each fold
    - FDR q < 0.05 defines significant pathway enrichment

Exploratory analysis:
    - top-5 pathways ranked by raw enrichment p-value within each fold
    - recurrence frequency across folds
    - pairwise Jaccard overlap of top-5 sets
    - Monte Carlo chance reference for random 5-of-50 pathway sets

Important:
    The top-5 analysis is descriptive and exploratory. It does not replace
    the FDR-significant pathway analysis.
"""

import numpy as np
import pandas as pd

from scipy.stats import hypergeom


# ---------------------------------------------------------------------
# GMT parsing
# ---------------------------------------------------------------------

def parse_gmt(path):
    """
    Parse a GMT file.

    Returns
    -------
    dict
        pathway_name -> list of HUGO gene symbols
    """
    pathways = {}

    with open(path) as f:

        for line in f:

            parts = (
                line
                .rstrip("\n")
                .split("\t")
            )

            if len(parts) < 3:
                continue

            name = parts[0]
            genes = parts[2:]

            pathways[name] = genes

    return pathways


# ---------------------------------------------------------------------
# Symbol -> Entrez conversion
# ---------------------------------------------------------------------

def resolve_pathways_to_entrez(
    pathways,
    feature_metadata,
):
    """
    Convert pathway gene symbols to Entrez IDs using feature_metadata.

    Symbols mapping to more than one Entrez ID are excluded rather than
    resolved arbitrarily.

    Returns
    -------
    pathway_entrez : dict
        pathway -> set of Entrez IDs

    manifest : DataFrame
        mapping coverage for each pathway
    """
    required = {
        "symbol",
        "entrez_id",
    }

    missing = required - set(
        feature_metadata.columns
    )

    if missing:
        raise ValueError(
            f"feature_metadata missing required columns: {missing}"
        )

    fm = feature_metadata.copy()

    fm = fm.dropna(
        subset=[
            "symbol",
            "entrez_id",
        ]
    )

    fm["symbol"] = (
        fm["symbol"]
        .astype(str)
        .str.strip()
    )

    fm["entrez_id"] = pd.to_numeric(
        fm["entrez_id"],
        errors="coerce",
    )

    fm = fm.dropna(
        subset=["entrez_id"]
    )

    fm["entrez_id"] = (
        fm["entrez_id"]
        .astype(int)
    )

    # Identify symbols mapping to >1 distinct Entrez ID
    symbol_counts = (
        fm.groupby("symbol")["entrez_id"]
        .nunique()
    )

    ambiguous_symbols = set(
        symbol_counts[
            symbol_counts > 1
        ].index
    )

    fm_unique = fm[
        ~fm["symbol"].isin(
            ambiguous_symbols
        )
    ].drop_duplicates(
        subset=[
            "symbol",
            "entrez_id",
        ]
    )

    symbol_to_entrez = dict(
        zip(
            fm_unique["symbol"],
            fm_unique["entrez_id"],
        )
    )

    pathway_entrez = {}
    manifest_rows = []

    for pathway_name, genes in pathways.items():

        requested = {
            str(g).strip()
            for g in genes
            if pd.notna(g)
        }

        resolved = {
            symbol_to_entrez[g]
            for g in requested
            if g in symbol_to_entrez
        }

        pathway_entrez[
            pathway_name
        ] = resolved

        manifest_rows.append({
            "pathway": pathway_name,
            "n_genes_gmt": len(requested),
            "n_resolved_entrez": len(resolved),
            "coverage_fraction": (
                len(resolved)
                / len(requested)
                if len(requested) > 0
                else np.nan
            ),
        })

    manifest = pd.DataFrame(
        manifest_rows
    )

    return (
        pathway_entrez,
        manifest,
    )


# ---------------------------------------------------------------------
# Benjamini-Hochberg correction
# ---------------------------------------------------------------------

def bh_correct(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvals = np.asarray(
        pvals,
        dtype=float,
    )

    if len(pvals) == 0:
        return np.array([])

    n = len(pvals)

    order = np.argsort(
        pvals
    )

    ranked = pvals[
        order
    ]

    adjusted = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    output = np.empty(
        n,
        dtype=float,
    )

    output[
        order
    ] = adjusted

    return output


# ---------------------------------------------------------------------
# Fold-level enrichment
# ---------------------------------------------------------------------

def test_fold_enrichment(
    selected_genes,
    background_genes,
    pathway_entrez,
):
    """
    Test Hallmark pathway overrepresentation for one outer fold.

    Parameters
    ----------
    selected_genes
        Genes selected by elastic net in this fold.

    background_genes
        Genes eligible for elastic-net selection after this fold's
        unsupervised filtering.

    pathway_entrez
        Hallmark pathway definitions converted to Entrez IDs.

    Returns
    -------
    DataFrame
        One row per pathway.
    """
    background = {
        int(g)
        for g in background_genes
    }

    selected = {
        int(g)
        for g in selected_genes
    }

    # Elastic net can only select genes in its candidate universe.
    selected = (
        selected
        & background
    )

    N = len(background)
    n = len(selected)

    rows = []

    for pathway, members in pathway_entrez.items():

        members = set(
            members
        )

        # Pathway genes eligible in this fold
        eligible_pathway_genes = (
            members
            & background
        )

        # Selected genes belonging to pathway
        selected_pathway_genes = (
            selected
            & eligible_pathway_genes
        )

        K = len(
            eligible_pathway_genes
        )

        k = len(
            selected_pathway_genes
        )

        if (
            N == 0
            or n == 0
            or K == 0
        ):
            p_value = 1.0

        else:
            # P(X >= k)
            p_value = float(
                hypergeom.sf(
                    k - 1,
                    N,
                    K,
                    n,
                )
            )

        rows.append({
            "pathway": pathway,
            "k_selected_in_pathway": k,
            "K_pathway_in_background": K,
            "n_selected_total": n,
            "N_background_total": N,
            "p_value": p_value,
        })

    result = pd.DataFrame(
        rows
    )

    result["p_adj_BH"] = bh_correct(
        result["p_value"].values
    )

    return result


# ---------------------------------------------------------------------
# Run enrichment across all folds
# ---------------------------------------------------------------------

def run_pathway_enrichment(
    selection_df,
    background_genes,
    pathway_entrez,
    alpha=0.05,
):
    """
    Run pathway enrichment independently for every outer CV fold.
    """
    fold_keys = (
        selection_df[
            ["repeat", "fold"]
        ]
        .drop_duplicates()
        .sort_values(
            ["repeat", "fold"]
        )
    )

    all_results = []

    for _, row in fold_keys.iterrows():

        repeat = int(
            row["repeat"]
        )

        fold = int(
            row["fold"]
        )

        key = (
            repeat,
            fold,
        )

        if key not in background_genes:
            raise KeyError(
                f"No candidate-gene background found for "
                f"repeat={repeat}, fold={fold}."
            )

        selected = (
            selection_df.loc[
                (
                    selection_df["repeat"]
                    == repeat
                )
                &
                (
                    selection_df["fold"]
                    == fold
                ),
                "gene_entrez",
            ]
            .astype(int)
            .tolist()
        )

        fold_result = test_fold_enrichment(
            selected_genes=selected,
            background_genes=background_genes[key],
            pathway_entrez=pathway_entrez,
        )

        fold_result["repeat"] = repeat
        fold_result["fold"] = fold

        all_results.append(
            fold_result
        )

    enrichment_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    enrichment_df["significant"] = (
        enrichment_df["p_adj_BH"]
        < alpha
    )

    return enrichment_df


# ---------------------------------------------------------------------
# FDR-significant pathway stability
# ---------------------------------------------------------------------

def compute_pathway_stability(
    enrichment_df,
):
    """
    Quantify recurrence of FDR-significant pathways across ALL outer folds.

    Every outer fold is represented, including folds with no significant
    pathways.

    Pairwise Jaccard handling:
        - nonempty vs. nonempty: ordinary Jaccard
        - empty vs. nonempty: Jaccard = 0
        - empty vs. empty: undefined and excluded

    Returns
    -------
    frequency_df
        FDR-significant frequency for every pathway.

    jaccard_summary
        Summary of valid pairwise pathway-set Jaccard values.

    fold_summary
        Number/proportion of folds with at least one significant pathway.
    """
    fold_key_df = (
        enrichment_df[
            ["repeat", "fold"]
        ]
        .drop_duplicates()
        .sort_values(
            ["repeat", "fold"]
        )
    )

    fold_keys = list(
        fold_key_df.itertuples(
            index=False,
            name=None,
        )
    )

    # Initialize every fold with an empty pathway set
    fold_sets = {
        key: set()
        for key in fold_keys
    }

    significant = enrichment_df[
        enrichment_df["significant"]
    ]

    for (
        repeat,
        fold,
    ), group in significant.groupby(
        ["repeat", "fold"]
    ):

        fold_sets[
            (int(repeat), int(fold))
        ] = set(
            group["pathway"]
        )

    n_folds_total = len(
        fold_sets
    )

    # --------------------------------------------------------------
    # Pathway frequency
    # --------------------------------------------------------------

    all_pathways = sorted(
        enrichment_df[
            "pathway"
        ].unique()
    )

    significant_counts = (
        significant
        .groupby("pathway")
        .size()
        .reindex(
            all_pathways,
            fill_value=0,
        )
    )

    frequency_df = pd.DataFrame({
        "pathway": all_pathways,
        "times_significant": (
            significant_counts.values
        ),
    })

    frequency_df[
        "selection_frequency"
    ] = (
        frequency_df[
            "times_significant"
        ]
        / n_folds_total
    )

    frequency_df = frequency_df.sort_values(
        [
            "selection_frequency",
            "pathway",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # --------------------------------------------------------------
    # Fold summary
    # --------------------------------------------------------------

    n_folds_with_sig = sum(
        len(pathways) > 0
        for pathways in fold_sets.values()
    )

    n_folds_without_sig = (
        n_folds_total
        - n_folds_with_sig
    )

    # --------------------------------------------------------------
    # Pairwise Jaccard
    # --------------------------------------------------------------

    jaccards = []

    n_empty_empty_pairs = 0
    n_valid_pairs = 0

    keys = list(
        fold_sets.keys()
    )

    for i in range(
        len(keys)
    ):

        for j in range(
            i + 1,
            len(keys),
        ):

            a = fold_sets[
                keys[i]
            ]

            b = fold_sets[
                keys[j]
            ]

            union = (
                a | b
            )

            if len(union) == 0:
                # Both folds have no significant pathways.
                # Mutual absence is not counted as biological agreement.
                n_empty_empty_pairs += 1
                continue

            intersection = (
                a & b
            )

            jaccard = (
                len(intersection)
                / len(union)
            )

            jaccards.append(
                jaccard
            )

            n_valid_pairs += 1

    if len(jaccards) > 0:

        jaccard_summary = (
            pd.Series(
                jaccards,
                name="jaccard",
            )
            .describe()
        )

    else:

        jaccard_summary = pd.Series(
            dtype=float,
            name="jaccard",
        )

    fold_summary = pd.Series({
        "n_folds_total":
            n_folds_total,

        "n_folds_with_FDR_significant_pathway":
            n_folds_with_sig,

        "n_folds_without_FDR_significant_pathway":
            n_folds_without_sig,

        "proportion_folds_with_FDR_significant_pathway":
            (
                n_folds_with_sig
                / n_folds_total
            ),

        "n_valid_jaccard_pairs":
            n_valid_pairs,

        "n_empty_empty_pairs_excluded":
            n_empty_empty_pairs,
    })

    return (
        frequency_df,
        jaccard_summary,
        fold_summary,
    )


# ---------------------------------------------------------------------
# Exploratory top-k pathway recurrence
# ---------------------------------------------------------------------

def compute_top_pathway_stability(
    enrichment_df,
    top_k=5,
):
    """
    Exploratory rank-based pathway recurrence analysis.

    For each outer fold:
        1. rank all pathways by raw enrichment p-value;
        2. retain the top-k pathways;
        3. calculate recurrence frequencies;
        4. calculate pairwise Jaccard overlap.

    Statistical significance is NOT required for this analysis.
    """
    top_sets = {}
    top1_choices = {}

    grouped = enrichment_df.groupby(
        ["repeat", "fold"]
    )

    for (
        repeat,
        fold,
    ), group in grouped:

        ranked = group.sort_values(
            [
                "p_value",
                "pathway",
            ]
        )

        top_pathways = list(
            ranked[
                "pathway"
            ].head(
                top_k
            )
        )

        key = (
            int(repeat),
            int(fold),
        )

        top_sets[
            key
        ] = set(
            top_pathways
        )

        top1_choices[
            key
        ] = (
            top_pathways[0]
        )

    n_folds_total = len(
        top_sets
    )

    all_pathways = sorted(
        enrichment_df[
            "pathway"
        ].unique()
    )

    # --------------------------------------------------------------
    # Top-k frequency
    # --------------------------------------------------------------

    topk_counts = {
        p: 0
        for p in all_pathways
    }

    for pathway_set in top_sets.values():

        for pathway in pathway_set:
            topk_counts[
                pathway
            ] += 1

    topk_frequency = pd.DataFrame([
        {
            "pathway": pathway,
            "times_in_topK": count,
            "frequency_in_topK": (
                count
                / n_folds_total
            ),
        }
        for pathway, count in topk_counts.items()
    ])

    topk_frequency = topk_frequency.sort_values(
        [
            "frequency_in_topK",
            "pathway",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # --------------------------------------------------------------
    # Top-1 frequency
    # --------------------------------------------------------------

    top1_counts = {
        p: 0
        for p in all_pathways
    }

    for pathway in top1_choices.values():
        top1_counts[
            pathway
        ] += 1

    top1_frequency = pd.DataFrame([
        {
            "pathway": pathway,
            "times_top1": count,
            "frequency_top1": (
                count
                / n_folds_total
            ),
        }
        for pathway, count in top1_counts.items()
    ])

    top1_frequency = top1_frequency.sort_values(
        [
            "frequency_top1",
            "pathway",
        ],
        ascending=[
            False,
            True,
        ],
    )

    # --------------------------------------------------------------
    # Pairwise top-k Jaccard
    # --------------------------------------------------------------

    keys = list(
        top_sets.keys()
    )

    jaccards = []

    for i in range(
        len(keys)
    ):

        for j in range(
            i + 1,
            len(keys),
        ):

            a = top_sets[
                keys[i]
            ]

            b = top_sets[
                keys[j]
            ]

            union = (
                a | b
            )

            jaccards.append(
                len(a & b)
                / len(union)
            )

    jaccard_summary = (
        pd.Series(
            jaccards,
            name=f"jaccard_top{top_k}",
        )
        .describe()
    )

    return (
        topk_frequency,
        top1_frequency,
        jaccard_summary,
    )


# ---------------------------------------------------------------------
# Monte Carlo random top-k reference
# ---------------------------------------------------------------------

def monte_carlo_random_topk_jaccard(
    n_pathways,
    top_k,
    n_sim=100000,
    seed=2026,
):
    """
    Generate a descriptive chance reference for pairwise Jaccard overlap
    between two random top-k pathway sets.

    This is NOT a formal hypothesis test because biological pathways are
    not statistically independent.
    """
    if top_k > n_pathways:
        raise ValueError(
            "top_k cannot exceed n_pathways."
        )

    rng = np.random.default_rng(
        seed
    )

    universe = np.arange(
        n_pathways
    )

    jaccards = np.empty(
        n_sim,
        dtype=float,
    )

    for i in range(
        n_sim
    ):

        a = set(
            rng.choice(
                universe,
                size=top_k,
                replace=False,
            )
        )

        b = set(
            rng.choice(
                universe,
                size=top_k,
                replace=False,
            )
        )

        jaccards[i] = (
            len(a & b)
            / len(a | b)
        )

    return {
        "n_pathways":
            n_pathways,

        "top_k":
            top_k,

        "n_simulations":
            n_sim,

        "null_mean_jaccard":
            float(
                np.mean(
                    jaccards
                )
            ),

        "null_median_jaccard":
            float(
                np.median(
                    jaccards
                )
            ),

        "null_2.5pct":
            float(
                np.percentile(
                    jaccards,
                    2.5,
                )
            ),

        "null_97.5pct":
            float(
                np.percentile(
                    jaccards,
                    97.5,
                )
            ),
    }