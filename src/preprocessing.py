"""
Data audit and expression preprocessing for the AMIA IMvigor210 reanalysis.

All functions operate on relative paths from the project root and never
write into data/raw/. Raw files are read-only inputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
TABLES = Path("results/tables")


def load_raw():
    """Load the four raw source files. Returns a dict of DataFrames."""
    sample_metadata = pd.read_csv(RAW / "sample_metadata.csv", index_col=0)
    expression_counts = pd.read_csv(RAW / "expression_counts.csv", index_col=0)
    feature_metadata = pd.read_csv(RAW / "feature_metadata.csv", index_col=0)
    mariathasan_signatures = pd.read_csv(RAW / "human_gene_signatures_long.csv")
    mcp_genes = pd.read_csv(RAW / "genes.txt", sep="\t")
    return {
        "sample_metadata": sample_metadata,
        "expression_counts": expression_counts,
        "feature_metadata": feature_metadata,
        "mariathasan_signatures": mariathasan_signatures,
        "mcp_genes": mcp_genes,
    }


def sample_identity_audit(sample_metadata, expression_counts):
    """
    Verify sample/patient alignment between metadata and expression data.
    Saves results/tables/sample_identity_audit.csv and returns a summary dict.
    """
    n_samples_meta = sample_metadata.shape[0]
    n_samples_expr = expression_counts.shape[1]
    ids_match = set(sample_metadata.index) == set(expression_counts.columns)

    n_unique_patients = sample_metadata["ANONPT_ID"].nunique()
    dup_patients = sample_metadata[
        sample_metadata["ANONPT_ID"].duplicated(keep=False)
    ].sort_values("ANONPT_ID")

    response_counts = sample_metadata["Best Confirmed Overall Response"].value_counts(
        dropna=False
    )
    binary_counts = sample_metadata["binaryResponse"].value_counts(dropna=False)

    summary_rows = [
        {"check": "n_samples_metadata", "value": n_samples_meta},
        {"check": "n_samples_expression_columns", "value": n_samples_expr},
        {"check": "sample_ids_match", "value": ids_match},
        {"check": "n_unique_patients (ANONPT_ID)", "value": n_unique_patients},
        {"check": "n_duplicate_patient_specimens", "value": len(dup_patients)},
        {"check": "expected_responders_298cohort", "value": "68/298 = 22.8%"},
        {"check": "observed_CR", "value": int(response_counts.get("CR", 0))},
        {"check": "observed_PR", "value": int(response_counts.get("PR", 0))},
        {"check": "observed_SD", "value": int(response_counts.get("SD", 0))},
        {"check": "observed_PD", "value": int(response_counts.get("PD", 0))},
        {"check": "observed_NE", "value": int(response_counts.get("NE", 0))},
        {"check": "observed_responders_binary (CR/PR)", "value": int(binary_counts.get("CR/PR", 0))},
        {"check": "observed_nonresponders_binary (SD/PD)", "value": int(binary_counts.get("SD/PD", 0))},
    ]
    summary = pd.DataFrame(summary_rows)

    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "sample_identity_audit.csv", index=False)

    if len(dup_patients) > 0:
        dup_patients.to_csv(TABLES / "duplicate_patient_specimens.csv")

    print("=== Sample identity audit ===")
    print(summary.to_string(index=False))
    if len(dup_patients) > 0:
        print("\nDuplicate patient specimens (flagged, not auto-resolved):")
        cols = [c for c in ["ANONPT_ID", "Best Confirmed Overall Response", "os", "censOS"]
                if c in dup_patients.columns]
        print(dup_patients[cols])

    assert ids_match, "Sample IDs in metadata and expression matrix do not match — STOP."

    return {
        "n_samples": n_samples_meta,
        "n_unique_patients": n_unique_patients,
        "n_duplicate_specimens": len(dup_patients),
        "response_counts": response_counts.to_dict(),
    }


def gene_annotation_audit(feature_metadata, expression_counts):
    """
    Verify gene identifiers and lengths are usable for TPM computation.
    """
    n_genes_meta = feature_metadata.shape[0]
    index_matches_entrez = (feature_metadata.index == feature_metadata["entrez_id"]).all()
    expr_genes_covered = feature_metadata.index.isin(expression_counts.index).sum()

    missing_length = (feature_metadata["length"].isna()).sum()
    nonpositive_length = (feature_metadata["length"] <= 0).sum()
    dup_symbols = feature_metadata["symbol"].dropna().duplicated().sum()
    dup_entrez = feature_metadata["entrez_id"].duplicated().sum()

    audit = pd.DataFrame([
        {"check": "n_genes_feature_metadata", "value": n_genes_meta},
        {"check": "index_equals_entrez_id", "value": index_matches_entrez},
        {"check": "n_genes_present_in_expression_counts", "value": int(expr_genes_covered)},
        {"check": "missing_length", "value": int(missing_length)},
        {"check": "nonpositive_length", "value": int(nonpositive_length)},
        {"check": "duplicated_nonnull_symbols", "value": int(dup_symbols)},
        {"check": "duplicated_entrez_ids", "value": int(dup_entrez)},
    ])
    TABLES.mkdir(parents=True, exist_ok=True)
    audit.to_csv(TABLES / "gene_annotation_audit.csv", index=False)

    print("\n=== Gene annotation audit ===")
    print(audit.to_string(index=False))
    print(
        "\nDecision: key all gene lookups (expression, signatures) on Entrez ID, "
        "never on symbol string, because of the duplicated-symbol / symbol-drift issue."
    )

    assert missing_length == 0 and nonpositive_length == 0, (
        "Missing or non-positive gene lengths present — resolve before TPM computation."
    )

    return audit


def compute_tpm(expression_counts, feature_metadata):
    """
    Compute TPM from raw counts using gene length, keyed on Entrez ID.
    RPK = counts / (length_kb); TPM = RPK / sum(RPK) * 1e6, per sample.
    """
    common_genes = expression_counts.index.intersection(feature_metadata.index)
    counts = expression_counts.loc[common_genes]
    lengths_kb = feature_metadata.loc[common_genes, "length"] / 1000.0

    rpk = counts.div(lengths_kb, axis=0)
    scaling_factors = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scaling_factors, axis=1)

    # QC
    col_sums = tpm.sum(axis=0)
    assert np.allclose(col_sums, 1e6, rtol=1e-3), "TPM columns do not sum to ~1e6."
    assert (tpm >= 0).all().all(), "Negative TPM values present."
    assert np.isfinite(tpm.values).all(), "Non-finite TPM values present."

    log2_tpm = np.log2(tpm + 1)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    tpm.to_csv(PROCESSED / "expression_TPM.csv")
    log2_tpm.to_csv(PROCESSED / "expression_log2TPM.csv")

    print(f"\n=== TPM computation ===")
    print(f"Genes used: {len(common_genes)} (of {expression_counts.shape[0]} in counts matrix)")
    print(f"Samples: {tpm.shape[1]}")
    print(f"Column sum check (should be ~1,000,000): min={col_sums.min():.1f}, max={col_sums.max():.1f}")

    return tpm, log2_tpm


def build_feature_data_dictionary(sample_metadata):
    """
    Document every candidate clinical/biomarker predictor: type, missingness,
    planned block assignment. Saved as results/tables/feature_data_dictionary.csv.
    """
    clinical_vars = {
        "Sex": "Clinical",
        "Baseline ECOG Score": "Clinical",
        "Tobacco Use History": "Clinical",
        "Met Disease Status": "Clinical",
        "Received platinum": "Clinical",
        "Intravesical BCG administered": "Clinical",
    }
    biomarker_vars = {
        "IC Level": "Standard biomarker",
        "TC Level": "Standard biomarker",
        "FMOne mutation burden per MB": "Standard biomarker",
    }
    enriched_vars = {
        "Neoantigen burden per MB": "Enriched (R)",
        "Immune phenotype": "Enriched (R)",
    }
    excluded_vars = {
        "TCGA Subtype": "EXCLUDED (molecular subtype)",
        "Lund": "EXCLUDED (molecular subtype)",
        "Lund2": "EXCLUDED (molecular subtype)",
        "sizeFactor": "EXCLUDED (expression-derived)",
    }
    # Not used in any block, kept for completeness / documentation only
    not_used_vars = {
        "Race": "Not in locked spec — documented, not modeled",
        "Enrollment IC": "Not in locked spec — documented, not modeled",
        "Sample age": "Not in locked spec — documented, not modeled",
        "Tissue": "Not in locked spec — documented, not modeled",
        "Sample collected pre-platinum": "Not in locked spec — documented, not modeled",
    }

    candidates = {**clinical_vars, **biomarker_vars, **enriched_vars, **excluded_vars, **not_used_vars}
    rows = []
    for col, block in candidates.items():
        if col not in sample_metadata.columns:
            rows.append({
                "variable": col, "block": block, "present_in_data": False,
                "dtype": None, "missing_pct": None, "n_categories_or_range": None,
            })
            continue
        s = sample_metadata[col]
        present = True
        missing_pct = round(100 * s.isna().mean(), 1)
        if pd.api.types.is_numeric_dtype(s):
            dtype = "continuous"
            desc = f"min={s.min():.2f}, max={s.max():.2f}, median={s.median():.2f}"
        else:
            dtype = "categorical"
            desc = f"{s.dropna().nunique()} categories: {sorted(s.dropna().unique().tolist())[:6]}"
        rows.append({
            "variable": col, "block": block, "present_in_data": present,
            "dtype": dtype, "missing_pct": missing_pct, "n_categories_or_range": desc,
        })

    fdd = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    fdd.to_csv(TABLES / "feature_data_dictionary.csv", index=False)

    print("\n=== Feature data dictionary ===")
    print(fdd.to_string(index=False))
    return fdd
