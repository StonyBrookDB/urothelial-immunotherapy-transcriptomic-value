"""
Step 6 driver: subgroup redundancy analysis -- does block K's incremental
value concentrate within a specific IC Level subgroup (e.g. IC1, the
intermediate/ambiguous PD-L1 category) rather than being uniform across
the whole cohort?

Run from the project root:
    python3 run_subgroup_redundancy.py

Requires:
    data/processed/analysis_cohort.csv

Writes:
    results/tables/subgroup_redundancy_raw.csv
    results/tables/subgroup_redundancy_deltas.csv
    results/tables/subgroup_redundancy_summary.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import modeling as mdl
import subgroup

PROCESSED = Path("data/processed")
TABLES = Path("results/tables")

SUBGROUP_COLUMN = "IC Level"


def main():
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)
    cohort_eval = cohort[cohort["binaryResponse"].notna()].copy()

    print(f"Running subgroup redundancy analysis on '{SUBGROUP_COLUMN}'...")
    print(cohort_eval.groupby(SUBGROUP_COLUMN)["binaryResponse"].value_counts())
    print()

    metrics_df, skipped = subgroup.run_subgroup_redundancy(cohort_eval, SUBGROUP_COLUMN)

    TABLES.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLES / "subgroup_redundancy_raw.csv", index=False)

    deltas = subgroup.compute_subgroup_deltas(metrics_df)
    deltas.to_csv(TABLES / "subgroup_redundancy_deltas.csv", index=False)

    summary_rows = []
    for level, g in deltas.groupby("subgroup"):
        s = g["delta_AUPRC"]
        summary_rows.append({
            "subgroup": level, "n_subgroup": g["n_subgroup"].iloc[0],
            "n_responders": g["n_responders"].iloc[0],
            "mean_delta_AUPRC": s.mean(), "median_delta_AUPRC": s.median(),
            "sd_delta_AUPRC": s.std(), "min": s.min(), "max": s.max(),
            "n_positive": int((s > 0).sum()), "n_negative": int((s < 0).sum()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "subgroup_redundancy_summary.csv", index=False)

    print(f"\nSaved:")
    print(f"  {TABLES / 'subgroup_redundancy_raw.csv'}")
    print(f"  {TABLES / 'subgroup_redundancy_deltas.csv'}")
    print(f"  {TABLES / 'subgroup_redundancy_summary.csv'}")
    if skipped:
        print(f"\nSkipped subgroups (too few responders): {skipped}")
    print()
    print(summary.to_string(index=False))
    print(
        "\nNo interpretation rendered here. Compare mean_delta_AUPRC across "
        "subgroups yourself -- and weigh all of this with more caution than "
        "the full-cohort results, given the much smaller per-subgroup "
        "sample sizes (higher variance is expected here by design)."
    )


if __name__ == "__main__":
    main()
