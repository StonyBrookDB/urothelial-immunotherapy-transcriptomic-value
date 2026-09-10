"""
External validation harmonization for GSE176307 (UNC-108/BACI).

The external analysis uses a harmonized predictor set that is available
in both IMvigor210 and GSE176307.

Clinical block:
    - Sex
    - ECOG performance status
    - Tobacco use history

Biomarker block:
    - TMB
    - CD274 (PD-L1) mRNA expression

Transcriptomic block:
    - 8 knowledge-guided gene-set scores

Important harmonization decisions:
    - "Light" smoking in GSE176307 is grouped with CURRENT smoking.
    - ECOG is represented as 0, 1, or >=2 in BOTH cohorts.
    - Knowledge-guided gene-set scores are computed using exactly the
      same marker-gene subset in both cohorts.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Harmonized feature blocks
# ---------------------------------------------------------------------

EXT_C = [
    "Sex",
    "Baseline ECOG Score",
    "Tobacco Use History",
]

EXT_B = [
    "FMOne mutation burden per MB",
    "CD274",
]

CD274_ENTREZ = 29126


TOBACCO_MAP = {
    "Never": "NEVER",
    "Former": "PREVIOUS",
    "Current": "CURRENT",
    "Light": "CURRENT",
}


# ---------------------------------------------------------------------
# Clinical harmonization
# ---------------------------------------------------------------------

def harmonize_ecog(series):
    """
    Harmonize ECOG across cohorts as:
        0 -> 0
        1 -> 1
        2 or greater -> 2

    Missing or non-numeric values remain NaN.
    """
    ecog = pd.to_numeric(series, errors="coerce")

    # Preserve missing values while collapsing >=2 to 2
    ecog = ecog.where(ecog.isna(), ecog.clip(upper=2))

    return ecog


def harmonize_gse176307_clinical(clinical, sample_key):
    """
    Harmonize GSE176307 clinical variables to the IMvigor210 coding scheme.

    Returns a DataFrame indexed by RNA-seq sample ID.

    Additional audit columns are retained but are NOT used as model predictors:
        _original_ecog
        _original_smoking
        _external_ecog3
        _external_light_smoker
    """
    clinical = clinical.copy()
    sample_key = sample_key.copy()

    clinical["Sample ID"] = (
        clinical["sample_title"]
        .astype(str)
        .str.replace("Patient sample ", "", regex=False)
        .str.strip()
    )

    merged = clinical.merge(
        sample_key,
        on="Sample ID",
        how="inner",
        validate="one_to_one",
    )

    n_before = len(clinical)
    n_after = len(merged)

    if n_after < n_before:
        print(
            f"NOTE: {n_before - n_after} clinical patient(s) had no "
            f"RNA-seq mapping and were excluded."
        )

    rna_id_col = "Omniseq_RS_ID (RNAseq)"

    if merged[rna_id_col].duplicated().any():
        raise ValueError(
            "Duplicate RNA-seq sample IDs found after clinical/sample-key merge."
        )

    harmonized = pd.DataFrame(
        index=merged[rna_id_col].astype(str).values
    )

    harmonized.index.name = "sample_id"

    # Sex
    harmonized["Sex"] = merged["gender"].values

    # --------------------------------------------------------------
    # ECOG
    # --------------------------------------------------------------
    raw_ecog = pd.to_numeric(
        merged["ecog"],
        errors="coerce",
    )

    harmonized["_original_ecog"] = raw_ecog.values
    harmonized["_external_ecog3"] = (raw_ecog == 3).values

    harmonized["Baseline ECOG Score"] = (
        harmonize_ecog(raw_ecog).values
    )

    # --------------------------------------------------------------
    # Tobacco
    # --------------------------------------------------------------
    raw_smoking = (
        merged["smoking status"]
        .astype("string")
        .str.strip()
    )

    harmonized["_original_smoking"] = raw_smoking.values
    harmonized["_external_light_smoker"] = (
        raw_smoking == "Light"
    ).values

    harmonized["Tobacco Use History"] = (
        raw_smoking.map(TOBACCO_MAP).values
    )

    # Warn if an unexpected smoking category was encountered
    unexpected = sorted(
        set(raw_smoking.dropna().unique()) - set(TOBACCO_MAP.keys())
    )

    if unexpected:
        print(
            "WARNING: Unmapped external smoking categories found: "
            + ", ".join(unexpected)
        )

    # --------------------------------------------------------------
    # TMB
    # --------------------------------------------------------------
    harmonized["FMOne mutation burden per MB"] = pd.to_numeric(
        merged["tmb"],
        errors="coerce",
    ).values

    # --------------------------------------------------------------
    # Response
    # --------------------------------------------------------------
    harmonized["io.response"] = merged["io.response"].values

    binary = pd.Series(
        np.nan,
        index=harmonized.index,
        dtype=object,
    )

    response_values = harmonized["io.response"]

    binary.loc[
        response_values.isin(["CR", "PR"])
    ] = "CR/PR"

    binary.loc[
        response_values.isin(["SD", "PD"])
    ] = "SD/PD"

    harmonized["binaryResponse"] = binary

    assert harmonized.index.is_unique

    return harmonized


# ---------------------------------------------------------------------
# Signature definitions
# ---------------------------------------------------------------------

def load_signature_gene_symbols(
    mariathasan_signatures,
    mcp_genes,
):
    """
    Reconstruct the eight knowledge-guided marker-gene sets.

    Returns
    -------
    dict
        signature_name -> set of HUGO gene symbols
    """
    primary_mariathasan = [
        "CD 8 T effector",
        "Immune Checkpoint",
        "APM",
    ]

    primary_mcp = [
        "NK cells",
        "Cytotoxic lymphocytes",
        "Monocytic lineage",
        "Fibroblasts",
        "Endothelial cells",
    ]

    display_names = {
        "CD 8 T effector": "CD8_T_effector",
        "Immune Checkpoint": "Immune_Checkpoint",
        "APM": "APM",
        "NK cells": "NK_cells",
        "Cytotoxic lymphocytes": "Cytotoxic_lymphocytes",
        "Monocytic lineage": "Monocytic_lineage",
        "Fibroblasts": "Fibroblasts",
        "Endothelial cells": "Endothelial_cells",
    }

    sig_symbols = {}

    for label in primary_mariathasan:
        genes = (
            mariathasan_signatures.loc[
                mariathasan_signatures["signature"] == label,
                "gene",
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        sig_symbols[display_names[label]] = set(genes)

    for population in primary_mcp:
        genes = (
            mcp_genes.loc[
                mcp_genes["Cell population"] == population,
                "HUGO symbols",
            ]
            .dropna()
            .astype(str)
            .str.strip()
        )

        sig_symbols[display_names[population]] = set(genes)

    return sig_symbols


# ---------------------------------------------------------------------
# Gene mapping
# ---------------------------------------------------------------------

def build_unique_symbol_to_entrez(feature_metadata):
    """
    Construct a HUGO-symbol -> Entrez-ID mapping using only unambiguous symbols.

    Symbols mapping to more than one Entrez ID are excluded rather than
    resolved arbitrarily.
    """
    required = {"symbol", "entrez_id"}

    missing = required - set(feature_metadata.columns)

    if missing:
        raise ValueError(
            f"feature_metadata is missing required columns: {missing}"
        )

    fm = feature_metadata.copy()

    fm = fm.dropna(
        subset=["symbol", "entrez_id"]
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

    fm["entrez_id"] = fm["entrez_id"].astype(int)

    # Count distinct Entrez IDs per symbol
    n_ids = (
        fm.groupby("symbol")["entrez_id"]
        .nunique()
    )

    ambiguous_symbols = set(
        n_ids[n_ids > 1].index
    )

    fm = fm[
        ~fm["symbol"].isin(ambiguous_symbols)
    ]

    # Remove exact duplicate symbol/Entrez rows
    fm = fm.drop_duplicates(
        subset=["symbol", "entrez_id"]
    )

    symbol_to_entrez = dict(
        zip(
            fm["symbol"],
            fm["entrez_id"],
        )
    )

    return symbol_to_entrez


# ---------------------------------------------------------------------
# Harmonized transcriptomic scores
# ---------------------------------------------------------------------

def compute_harmonized_signature_scores(
    imvigor_log2tpm_entrez,
    gse_log2tpm_symbol,
    sig_symbols,
    feature_metadata,
):
    """
    Compute each knowledge-guided score using EXACTLY the same marker genes
    in IMvigor210 and GSE176307.

    For each signature, genes must:
        1. have an unambiguous symbol -> Entrez mapping in IMvigor210;
        2. be present in the IMvigor210 expression matrix;
        3. be present in the GSE176307 expression matrix.

    The score is the mean log2(TPM + 1) expression across the common genes.

    Returns
    -------
    imvigor_scores : DataFrame
        samples x signatures

    gse_scores : DataFrame
        samples x signatures

    coverage : DataFrame
        signature-level gene coverage manifest
    """
    imvigor_expr = imvigor_log2tpm_entrez.copy()
    gse_expr = gse_log2tpm_symbol.copy()

    if not imvigor_expr.columns.is_unique:
        raise ValueError(
            "Duplicate IMvigor210 sample IDs found in expression matrix."
        )

    if not gse_expr.columns.is_unique:
        raise ValueError(
            "Duplicate GSE176307 sample IDs found in expression matrix."
        )

    # Normalize external symbol labels
    gse_expr.index = (
        gse_expr.index
        .astype(str)
        .str.strip()
    )

    if not gse_expr.index.is_unique:
        duplicates = (
            gse_expr.index[
                gse_expr.index.duplicated()
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate gene symbols found in GSE176307 expression matrix. "
            f"Examples: {duplicates[:10]}"
        )

    # Normalize IMvigor Entrez index
    imvigor_entrez = pd.to_numeric(
        imvigor_expr.index,
        errors="coerce",
    )

    if np.isnan(imvigor_entrez).any():
        raise ValueError(
            "Non-numeric Entrez IDs found in IMvigor210 expression matrix."
        )

    imvigor_expr.index = imvigor_entrez.astype(int)

    if not imvigor_expr.index.is_unique:
        raise ValueError(
            "Duplicate Entrez IDs found in IMvigor210 expression matrix."
        )

    symbol_to_entrez = build_unique_symbol_to_entrez(
        feature_metadata
    )

    imvigor_scores = {}
    gse_scores = {}
    coverage_rows = []

    for signature, requested_genes in sig_symbols.items():

        requested_genes = {
            str(g).strip()
            for g in requested_genes
            if pd.notna(g)
        }

        common_genes = []

        for symbol in sorted(requested_genes):

            if symbol not in symbol_to_entrez:
                continue

            entrez = symbol_to_entrez[symbol]

            if entrez not in imvigor_expr.index:
                continue

            if symbol not in gse_expr.index:
                continue

            common_genes.append(symbol)

        if len(common_genes) == 0:
            raise ValueError(
                f"{signature}: zero marker genes available in both cohorts."
            )

        imvigor_entrez_ids = [
            symbol_to_entrez[g]
            for g in common_genes
        ]

        imvigor_scores[signature] = (
            imvigor_expr
            .loc[imvigor_entrez_ids]
            .mean(axis=0)
        )

        gse_scores[signature] = (
            gse_expr
            .loc[common_genes]
            .mean(axis=0)
        )

        coverage_rows.append({
            "signature": signature,
            "n_requested": len(requested_genes),
            "n_common": len(common_genes),
            "coverage_fraction": (
                len(common_genes)
                / len(requested_genes)
            ),
            "genes_used": ";".join(common_genes),
        })

    imvigor_scores = pd.DataFrame(
        imvigor_scores
    )

    gse_scores = pd.DataFrame(
        gse_scores
    )

    coverage = pd.DataFrame(
        coverage_rows
    )

    return (
        imvigor_scores,
        gse_scores,
        coverage,
    )


# ---------------------------------------------------------------------
# CD274
# ---------------------------------------------------------------------

def compute_cd274_imvigor(log2_tpm_entrez):
    """
    Extract CD274 (PD-L1 mRNA) from IMvigor210.
    """
    expr = log2_tpm_entrez.copy()

    numeric_index = pd.to_numeric(
        expr.index,
        errors="coerce",
    )

    if np.isnan(numeric_index).any():
        raise ValueError(
            "Non-numeric Entrez IDs found in IMvigor210 expression matrix."
        )

    expr.index = numeric_index.astype(int)

    if CD274_ENTREZ not in expr.index:
        raise ValueError(
            f"CD274 (Entrez {CD274_ENTREZ}) not found in IMvigor210."
        )

    return expr.loc[
        CD274_ENTREZ
    ].rename("CD274")


# ---------------------------------------------------------------------
# IMvigor harmonized training cohort
# ---------------------------------------------------------------------

def build_imvigor_harmonized_cohort(
    analysis_cohort,
    log2_tpm_entrez,
    harmonized_k_scores,
):
    """
    Assemble the IMvigor210 development cohort for external transportability.

    The pre-existing K columns in analysis_cohort are replaced by scores
    recomputed from the gene subset shared with GSE176307.
    """
    cohort = analysis_cohort[
        analysis_cohort["binaryResponse"].notna()
    ].copy()

    cohort.index = cohort.index.astype(str)

    if not cohort.index.is_unique:
        raise ValueError(
            "Duplicate IMvigor210 patient/sample IDs in analysis_cohort."
        )

    # Harmonize ECOG identically to the external cohort
    cohort["Baseline ECOG Score"] = harmonize_ecog(
        cohort["Baseline ECOG Score"]
    )

    # CD274 mRNA
    cd274 = compute_cd274_imvigor(
        log2_tpm_entrez
    )

    cd274.index = cd274.index.astype(str)

    # Harmonized K scores
    k_scores = harmonized_k_scores.copy()
    k_scores.index = k_scores.index.astype(str)

    if not k_scores.index.is_unique:
        raise ValueError(
            "Duplicate IMvigor210 sample IDs in harmonized K scores."
        )

    # Add CD274
    cohort = cohort.join(
        cd274,
        how="left",
    )

    # Replace original K columns with harmonized versions
    cohort = cohort.drop(
        columns=[
            c
            for c in k_scores.columns
            if c in cohort.columns
        ],
        errors="ignore",
    )

    cohort = cohort.join(
        k_scores,
        how="left",
    )

    if cohort["CD274"].isna().any():
        raise ValueError(
            "Missing IMvigor210 CD274 values after sample alignment."
        )

    if cohort[k_scores.columns].isna().any().any():
        raise ValueError(
            "Missing harmonized K scores in IMvigor210 after sample alignment."
        )

    return cohort