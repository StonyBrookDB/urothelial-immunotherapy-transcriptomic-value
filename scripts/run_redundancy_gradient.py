"""
Step 4 driver: information-redundancy gradient (analysis_spec.md Section 6, item 1).

Tests block K against three baselines of increasing biomarker richness:
    Level 1: C        vs. C+K
    Level 2: C+B       vs. C+B+K   (reused from run_primary_model.py, not recomputed)
    Level 3: C+B+R     vs. C+B+R+K

Run from the project root, after run_primary_model.py has completed:
    python3 run_redundancy_gradient.py

Requires:
    results/models/cv_splits.pkl               (from run_primary_model.py)
    results/tables/primary_model_performance.csv (from run_primary_model.py)
    data/processed/analysis_cohort.csv

Writes:
    results/predictions/redundancy_oof_predictions.csv
    results/tables/redundancy_gradient_raw_performance.csv
    results/tables/redundancy_gradient.csv
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import modeling as mdl

PROCESSED = Path("data/processed")
PREDICTIONS = Path("results/predictions")
TABLES = Path("results/tables")
MODELS = Path("results/models")


def main():
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)
    cohort_eval = cohort[cohort["binaryResponse"].notna()].copy()
    y = (cohort_eval["binaryResponse"] == "CR/PR").astype(int)

    splits_path = MODELS / "cv_splits.pkl"
    assert splits_path.exists(), (
        "results/models/cv_splits.pkl not found — run run_primary_model.py first. "
        "The redundancy gradient must reuse those exact splits, not new ones."
    )
    splits = mdl.generate_or_load_cv_splits(y, path=splits_path)

    primary_perf_path = TABLES / "primary_model_performance.csv"
    assert primary_perf_path.exists(), (
        "results/tables/primary_model_performance.csv not found — run run_primary_model.py "
        "first. Level 2 (C+B vs. C+B+K) is reused from that run, not recomputed here."
    )
    primary_perf = pd.read_csv(primary_perf_path)

    feature_sets = {
        "C": mdl.get_feature_columns("C"),
        "C+K": mdl.get_feature_columns("C", "K"),
        "C+B+R": mdl.get_feature_columns("C", "B", "R"),
        "C+B+R+K": mdl.get_feature_columns("C", "B", "R", "K"),
    }

    all_oof, all_metrics = [], []
    for label, cols in feature_sets.items():
        print(f"\nRunning nested CV: {label} ...")
        oof_df, metrics_df = mdl.run_nested_cv(cohort_eval, cols, y, splits, label)
        all_oof.append(oof_df)
        all_metrics.append(metrics_df)

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(PREDICTIONS / "redundancy_oof_predictions.csv", index=False)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(TABLES / "redundancy_gradient_raw_performance.csv", index=False)

    def pull(block, colname):
        return metrics_all[metrics_all["block"] == block][["repeat", "AUPRC"]].rename(
            columns={"AUPRC": colname}
        )

    level1 = pull("C", "AUPRC_C")
    level1_k = pull("C+K", "AUPRC_CK")
    level2 = primary_perf[["repeat", "AUPRC_CB", "AUPRC_CBK"]]
    level3 = pull("C+B+R", "AUPRC_CBR")
    level3_k = pull("C+B+R+K", "AUPRC_CBRK")

    gradient = (
        level1.merge(level1_k, on="repeat")
        .merge(level2, on="repeat")
        .merge(level3, on="repeat")
        .merge(level3_k, on="repeat")
    )

    gradient["delta_AUPRC_level1_C"] = gradient["AUPRC_CK"] - gradient["AUPRC_C"]
    gradient["delta_AUPRC_level2_CB"] = gradient["AUPRC_CBK"] - gradient["AUPRC_CB"]
    gradient["delta_AUPRC_level3_CBR"] = gradient["AUPRC_CBRK"] - gradient["AUPRC_CBR"]

    gradient.to_csv(TABLES / "redundancy_gradient.csv", index=False)

    print(f"\nSaved:")
    print(f"  {PREDICTIONS / 'redundancy_oof_predictions.csv'} ({len(oof_all)} rows)")
    print(f"  {TABLES / 'redundancy_gradient_raw_performance.csv'} ({len(metrics_all)} rows)")
    print(f"  {TABLES / 'redundancy_gradient.csv'} ({len(gradient)} repeats x 3 levels)")
    print(
        "\nNo interpretation rendered here. Compare delta_AUPRC_level1_C -> "
        "level2_CB -> level3_CBR yourself to see whether/where transcriptomic "
        "value gets absorbed by richer biomarker baselines, per "
        "analysis_spec.md Section 6 item 1."
    )


if __name__ == "__main__":
    main()
