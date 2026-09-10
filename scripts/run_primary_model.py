"""
Step 3 driver: primary model comparison — C+B vs. C+B+K.
This is the run the Aug 29 go/no-go decision is based on (analysis_spec.md Section 5).

Run from the project root, after run_signature_scoring.py has completed:
    python3 run_primary_model.py

Requires data/processed/analysis_cohort.csv to exist (from step 2).

Writes:
    results/models/cv_splits.pkl              (generated once, reused every future run)
    results/predictions/primary_oof_predictions.csv
    results/tables/primary_model_performance.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import modeling as mdl

PROCESSED = Path("data/processed")
PREDICTIONS = Path("results/predictions")
TABLES = Path("results/tables")


def main():
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)

    cohort_eval = cohort[cohort["binaryResponse"].notna()].copy()
    y = (cohort_eval["binaryResponse"] == "CR/PR").astype(int)
    print(f"Response-evaluable cohort: {len(cohort_eval)} patients, "
          f"{y.sum()} responders ({100 * y.mean():.1f}%)")

    splits = mdl.generate_or_load_cv_splits(y)

    cb_cols = mdl.get_feature_columns("C", "B")
    cbk_cols = mdl.get_feature_columns("C", "B", "K")

    print("\nRunning nested CV: C+B ...")
    oof_cb, metrics_cb = mdl.run_nested_cv(cohort_eval, cb_cols, y, splits, "C+B")

    print("\nRunning nested CV: C+B+K ...")
    oof_cbk, metrics_cbk = mdl.run_nested_cv(cohort_eval, cbk_cols, y, splits, "C+B+K")

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    oof_all = pd.concat([oof_cb, oof_cbk], ignore_index=True)
    oof_all.to_csv(PREDICTIONS / "primary_oof_predictions.csv", index=False)

    merged = metrics_cb.merge(
        metrics_cbk, on="repeat", suffixes=("_CB", "_CBK")
    )
    merged["delta_AUPRC"] = merged["AUPRC_CBK"] - merged["AUPRC_CB"]

    TABLES.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLES / "primary_model_performance.csv", index=False)

    print(f"\nSaved:")
    print(f"  {PREDICTIONS / 'primary_oof_predictions.csv'} ({len(oof_all)} rows)")
    print(f"  {TABLES / 'primary_model_performance.csv'} ({len(merged)} repeats)")
    print("\nReview delta_AUPRC distribution across repeats yourself before making "
          "the go/no-go call — this script deliberately does not interpret it for you.")


if __name__ == "__main__":
    main()
