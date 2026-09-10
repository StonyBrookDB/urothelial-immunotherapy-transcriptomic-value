"""
Step: feature importance via standardized ridge coefficients, comparing
C+B and C+B+K -- separate from, and using the FULL feature blocks unlike,
the harmonized models in run_external_validation.py.

Run from the project root:
    python3 run_feature_importance.py

Requires:
    data/processed/analysis_cohort.csv

Writes:
    results/tables/feature_importance_CB.csv
    results/tables/feature_importance_CBK.csv
    results/tables/feature_importance_comparison.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import modeling as mdl
import feature_importance as fi

PROCESSED = Path("data/processed")
TABLES = Path("results/tables")


def main():
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)
    cohort_eval = cohort[cohort["binaryResponse"].notna()].copy()
    y = (cohort_eval["binaryResponse"] == "CR/PR").astype(int)
    print(f"Fitting on {len(cohort_eval)} patients, {y.sum()} responders")

    cb_cols = mdl.get_feature_columns("C", "B")
    cbk_cols = mdl.get_feature_columns("C", "B", "K")

    print("\nFreezing C+B (full feature block)...")
    model_cb = fi.freeze_model(cohort_eval, cb_cols, y)
    coefs_cb = fi.extract_named_coefficients(model_cb)

    print("Freezing C+B+K (full feature block)...")
    model_cbk = fi.freeze_model(cohort_eval, cbk_cols, y)
    coefs_cbk = fi.extract_named_coefficients(model_cbk)

    TABLES.mkdir(parents=True, exist_ok=True)
    coefs_cb.rename("coefficient").to_csv(TABLES / "feature_importance_CB.csv")
    coefs_cbk.rename("coefficient").to_csv(TABLES / "feature_importance_CBK.csv")

    print("\n=== C+B: top standardized coefficients ===")
    print(coefs_cb.head(15).to_string())
    print("\n=== C+B+K: top standardized coefficients ===")
    print(coefs_cbk.head(15).to_string())

    comparison = fi.build_comparison_table(coefs_cb, coefs_cbk)
    comparison.to_csv(TABLES / "feature_importance_comparison.csv")
    print("\n=== Comparison: features with the largest coefficient shift when K is added ===")
    print(comparison.head(15).to_string())

    print(
        "\nNo interpretation rendered here. A shrinking coefficient for a "
        "biomarker feature (e.g. TMB, PD-L1) when K is added is CONSISTENT "
        "WITH redundancy, not proof of it -- correlated-feature variance "
        "reallocation can produce the same pattern. Read alongside the "
        "redundancy gradient result, not in isolation."
    )


if __name__ == "__main__":
    main()
