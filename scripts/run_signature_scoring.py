"""
Step 2 driver: signature manifest, signature scores, analysis cohort table.

Run from the project root, after run_data_audit.py has completed:
    python3 run_signature_scoring.py

Requires data/processed/expression_log2TPM.csv to exist (from step 1).

Writes:
    data/processed/signature_manifest.csv
    data/processed/signature_scores.csv
    data/processed/analysis_cohort.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import preprocessing as pp
import signatures as sig

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def build_analysis_cohort(sample_metadata, scores_df):
    """
    Merge patient ID, response label, and locked C/B/R columns with the
    signature scores. Does NOT impute, scale, or drop rows — that happens
    inside CV folds in the modeling step, not here.
    """
    locked_columns = [
        "ANONPT_ID",
        "Best Confirmed Overall Response",
        "binaryResponse",
        # Clinical (C)
        "Sex", "Baseline ECOG Score", "Tobacco Use History",
        "Met Disease Status", "Received platinum", "Intravesical BCG administered",
        # Standard biomarker (B)
        "IC Level", "TC Level", "FMOne mutation burden per MB",
        # Enriched (R)
        "Neoantigen burden per MB", "Immune phenotype",
    ]
    missing_cols = [c for c in locked_columns if c not in sample_metadata.columns]
    assert not missing_cols, f"Expected columns missing from sample_metadata: {missing_cols}"

    cohort = sample_metadata[locked_columns].copy()
    cohort = cohort.join(scores_df, how="left")

    n_missing_scores = cohort[scores_df.columns].isna().any(axis=1).sum()
    assert n_missing_scores == 0, (
        f"{n_missing_scores} samples have no signature score — sample ID mismatch "
        f"between sample_metadata and signature_scores. Investigate before proceeding."
    )

    cohort.to_csv(PROCESSED / "analysis_cohort.csv")
    print(f"\n=== Analysis cohort assembled ===")
    print(f"Rows: {cohort.shape[0]}, columns: {cohort.shape[1]}")
    print(f"Response-evaluable (binaryResponse not null): {cohort['binaryResponse'].notna().sum()}")
    return cohort


def main():
    print("Loading raw + processed data...")
    data = pp.load_raw()
    log2_tpm = pd.read_csv(PROCESSED / "expression_log2TPM.csv", index_col=0)
    log2_tpm.index = log2_tpm.index.astype(int)  # Entrez IDs read back as int, not str

    print("\nBuilding signature gene map + manifest...")
    gene_map, manifest = sig.build_signature_gene_map(
        data["mariathasan_signatures"], data["mcp_genes"], data["feature_metadata"]
    )

    print("\nComputing signature scores...")
    scores_df = sig.compute_signature_scores(log2_tpm, gene_map)

    print("\nAssembling analysis cohort table...")
    build_analysis_cohort(data["sample_metadata"], scores_df)

    print("\nDone. Review data/processed/signature_manifest.csv and analysis_cohort.csv "
          "before proceeding to primary modeling.")


if __name__ == "__main__":
    main()
