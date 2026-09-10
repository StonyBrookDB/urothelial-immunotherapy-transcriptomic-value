"""
Component ablation within K (drop-one-out) -- analysis_spec.md Section 15B.

Determines which of K's 8 signatures is actually carrying whatever
incremental signal the block contributes, by dropping each one at a time
from the full C+B+K feature set and comparing to the full-block AUPRC
(reused from primary_model_performance.csv, not recomputed).

Interpretation caveat: because the 8 K signatures are correlated, drop-one-
out deltas can understate a signature's true importance if a correlated
signature compensates for it. Read alongside ridge coefficient stability
(mean/sign across repeated folds), not in isolation.

Run from the project root, after run_primary_model.py:
    python3 run_component_ablation.py

Requires:
    results/models/cv_splits.pkl
    results/tables/primary_model_performance.csv
    data/processed/analysis_cohort.csv

Writes:
    results/tables/component_ablation_K_raw.csv
    results/tables/component_ablation_K.csv
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
    cohort_eval = cohort[cohort["binaryResponse"].notna()].copy()
    y = (cohort_eval["binaryResponse"] == "CR/PR").astype(int)

    splits_path = MODELS / "cv_splits.pkl"
    assert splits_path.exists(), (
        "results/models/cv_splits.pkl not found — run run_primary_model.py first. "
        "This must reuse the exact same splits as everything else."
    )
    splits = mdl.generate_or_load_cv_splits(y, path=splits_path)

    primary_path = TABLES / "primary_model_performance.csv"
    assert primary_path.exists(), (
        "results/tables/primary_model_performance.csv not found — run "
        "run_primary_model.py first. Full C+B+K performance is reused from "
        "that run, not recomputed here."
    )
    primary = pd.read_csv(primary_path)[["repeat", "AUPRC_CBK"]]

    k_signatures = mdl.FEATURE_BLOCKS["K"]
    all_metrics = []

    for dropped in k_signatures:
        reduced_k = [s for s in k_signatures if s != dropped]
        cols = mdl.get_feature_columns("C", "B") + reduced_k
        print(f"\nRunning C+B+K minus '{dropped}' ...")
        _, metrics_df = mdl.run_nested_cv(
            cohort_eval, cols, y, splits, f"C+B+K minus {dropped}"
        )
        metrics_df["dropped_signature"] = dropped
        all_metrics.append(metrics_df)

    ablation_raw = pd.concat(all_metrics, ignore_index=True)
    merged = ablation_raw.merge(primary, on="repeat")
    merged["delta_AUPRC_drop"] = merged["AUPRC_CBK"] - merged["AUPRC"]

    TABLES.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TABLES / "component_ablation_K_raw.csv", index=False)

    summary = (
        merged.groupby("dropped_signature")["delta_AUPRC_drop"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    summary.to_csv(TABLES / "component_ablation_K.csv", index=False)

    print(f"\nSaved:")
    print(f"  {TABLES / 'component_ablation_K_raw.csv'}")
    print(f"  {TABLES / 'component_ablation_K.csv'}")
    print()
    print(summary.to_string(index=False))
    print(
        "\nNo interpretation rendered here. A large positive mean "
        "delta_AUPRC_drop means dropping that signature hurt performance "
        "(i.e., it was contributing). Read this alongside ridge coefficient "
        "stability, not in isolation -- correlated signatures can mask each "
        "other's individual contribution here."
    )


if __name__ == "__main__":
    main()
