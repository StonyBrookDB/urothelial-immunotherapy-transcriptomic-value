"""
CR/PR vs. PD sensitivity check (drop SD) -- analysis_spec.md Section 6, item 4.

Reruns ONLY the primary comparison, C+B vs. C+B+K, restricted to the
cleaner, more polarized outcome definition (excluding SD -- stable disease,
a clinically ambiguous middle category). Tests whether the primary null
result is an artifact of how ambiguous SD patients were coded.

Deliberately does NOT rerun the redundancy gradient, elastic net stability,
subgroup, or interaction analyses -- those test different, orthogonal
design choices (biomarker richness, feature selection method, PD-L1
strata), not the SD/PD outcome boundary. Rerunning everything under an
alternate cohort definition would answer a much broader question than a
sensitivity check is meant to.

Run from the project root, after run_primary_model.py:
    python3 run_sensitivity_CRPR_vs_PD.py

Requires:
    data/processed/analysis_cohort.csv

Writes:
    results/models/cv_splits_CRPR_vs_PD.pkl
    results/tables/sensitivity_CRPR_vs_PD_performance.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import modeling as mdl

PROCESSED = Path("data/processed")
TABLES = Path("results/tables")
MODELS = Path("results/models")


def main():
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)

    # Restrict to CR/PR and PD only -- drop SD (and NE, already excluded
    # from binaryResponse everywhere else).
    sensitivity_cohort = cohort[
        cohort["Best Confirmed Overall Response"].isin(["CR", "PR", "PD"])
    ].copy()
    y = (sensitivity_cohort["binaryResponse"] == "CR/PR").astype(int)

    print(f"Sensitivity cohort (CR/PR vs. PD only): {len(sensitivity_cohort)} "
          f"patients, {y.sum()} responders ({100 * y.mean():.1f}%)")
    print(f"(Primary cohort for comparison: 298 patients, 68 responders, 22.8%)")

    # Fresh CV splits -- cannot reuse cv_splits.pkl, which was built for the
    # 298-patient cohort's row indices. Saved separately for reproducibility.
    splits_path = MODELS / "cv_splits_CRPR_vs_PD.pkl"
    splits = mdl.generate_or_load_cv_splits(y, path=splits_path)

    cb_cols = mdl.get_feature_columns("C", "B")
    cbk_cols = mdl.get_feature_columns("C", "B", "K")

    print("\nRunning nested CV: C+B ...")
    _, metrics_cb = mdl.run_nested_cv(sensitivity_cohort, cb_cols, y, splits, "C+B")

    print("\nRunning nested CV: C+B+K ...")
    _, metrics_cbk = mdl.run_nested_cv(sensitivity_cohort, cbk_cols, y, splits, "C+B+K")

    merged = metrics_cb.merge(metrics_cbk, on="repeat", suffixes=("_CB", "_CBK"))
    merged["delta_AUPRC"] = merged["AUPRC_CBK"] - merged["AUPRC_CB"]

    TABLES.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLES / "sensitivity_CRPR_vs_PD_performance.csv", index=False)

    print(f"\nSaved to {TABLES / 'sensitivity_CRPR_vs_PD_performance.csv'}")
    print(merged[["repeat", "AUPRC_CB", "AUPRC_CBK", "delta_AUPRC"]].to_string(index=False))
    print(
        "\nNo interpretation rendered here. Compare this delta_AUPRC "
        "distribution (mean/spread) against the primary run's yourself -- "
        "if broadly similar, the primary null result is not an artifact of "
        "how SD patients were coded. Do NOT adopt this as a new primary "
        "endpoint regardless of what the numbers show, per the locked spec."
    )


if __name__ == "__main__":
    main()
