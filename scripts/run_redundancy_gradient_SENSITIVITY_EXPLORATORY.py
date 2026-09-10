"""
*** POST-HOC / EXPLORATORY -- NOT part of the prespecified analysis plan ***

This script exists only because the CR/PR-vs-PD sensitivity check (itself
prespecified) turned up a significant result where the primary analysis
did not. It extends that single specific finding -- does excluding SD
change WHERE K's value gets absorbed across biomarker richness, not just
WHETHER K helps -- by rerunning the redundancy gradient on the SD-excluded
cohort.

This is deliberately NOT accompanied by reruns of elastic net stability,
subgroup analysis, or the interaction test on this cohort. Doing so would
mean rebuilding the paper's evidence base around whichever cohort
definition happened to look more favorable after the fact, which is a real
credibility problem, not a hypothetical one -- see the discussion that
prompted this script. Report results from this file explicitly labeled as
post-hoc/exploratory in the manuscript, not folded into the main results
as if prespecified.

Levels tested (same structure as the primary redundancy gradient):
    Level 1: C        vs. C+K
    Level 2: C+B       vs. C+B+K   (reused from run_sensitivity_CRPR_vs_PD.py,
                                     not recomputed -- same cv_splits reused)
    Level 3: C+B+R     vs. C+B+R+K

Run from the project root, after run_sensitivity_CRPR_vs_PD.py:
    python3 run_redundancy_gradient_SENSITIVITY_EXPLORATORY.py

Requires:
    results/models/cv_splits_CRPR_vs_PD.pkl
    results/tables/sensitivity_CRPR_vs_PD_performance.csv
    data/processed/analysis_cohort.csv

Writes (all filenames flagged EXPLORATORY -- do not treat as primary output):
    results/predictions/redundancy_gradient_SENSITIVITY_EXPLORATORY_oof.csv
    results/tables/redundancy_gradient_SENSITIVITY_EXPLORATORY_raw.csv
    results/tables/redundancy_gradient_SENSITIVITY_EXPLORATORY.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import modeling as mdl

PROCESSED = Path("data/processed")
PREDICTIONS = Path("results/predictions")
TABLES = Path("results/tables")
MODELS = Path("results/models")
FIGURES = Path("results/figures")

LEVEL_COLS = ["delta_AUPRC_level1_C", "delta_AUPRC_level2_CB", "delta_AUPRC_level3_CBR"]
LEVEL_LABELS = {
    "delta_AUPRC_level1_C": "Level 1\n(C vs C+K)",
    "delta_AUPRC_level2_CB": "Level 2\n(C+B vs C+B+K)",
    "delta_AUPRC_level3_CBR": "Level 3\n(C+B+R vs C+B+R+K)",
}


def summarize_gradient(df, source_label):
    """One row per level: mean/median/sd/min/max/sign-split."""
    rows = []
    for col in LEVEL_COLS:
        s = df[col]
        rows.append({
            "source": source_label, "level": col,
            "mean": s.mean(), "median": s.median(), "sd": s.std(),
            "min": s.min(), "max": s.max(),
            "n_positive": int((s > 0).sum()), "n_negative": int((s < 0).sum()),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("POST-HOC / EXPLORATORY ANALYSIS -- prompted by the sensitivity")
    print("finding, not part of the prespecified plan. Label accordingly")
    print("in the manuscript.")
    print("=" * 70)

    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)
    sensitivity_cohort = cohort[
        cohort["Best Confirmed Overall Response"].isin(["CR", "PR", "PD"])
    ].copy()
    y = (sensitivity_cohort["binaryResponse"] == "CR/PR").astype(int)
    print(f"\nSD-excluded cohort: {len(sensitivity_cohort)} patients, "
          f"{y.sum()} responders ({100 * y.mean():.1f}%)")

    splits_path = MODELS / "cv_splits_CRPR_vs_PD.pkl"
    assert splits_path.exists(), (
        "results/models/cv_splits_CRPR_vs_PD.pkl not found — run "
        "run_sensitivity_CRPR_vs_PD.py first. This must reuse those exact "
        "splits, not new ones."
    )
    splits = mdl.generate_or_load_cv_splits(y, path=splits_path)

    sensitivity_perf_path = TABLES / "sensitivity_CRPR_vs_PD_performance.csv"
    assert sensitivity_perf_path.exists(), (
        "results/tables/sensitivity_CRPR_vs_PD_performance.csv not found — "
        "run run_sensitivity_CRPR_vs_PD.py first. Level 2 (C+B vs. C+B+K) "
        "is reused from that run, not recomputed here."
    )
    sensitivity_perf = pd.read_csv(sensitivity_perf_path)

    feature_sets = {
        "C": mdl.get_feature_columns("C"),
        "C+K": mdl.get_feature_columns("C", "K"),
        "C+B+R": mdl.get_feature_columns("C", "B", "R"),
        "C+B+R+K": mdl.get_feature_columns("C", "B", "R", "K"),
    }

    all_oof, all_metrics = [], []
    for label, cols in feature_sets.items():
        print(f"\nRunning nested CV: {label} (SD-excluded cohort) ...")
        oof_df, metrics_df = mdl.run_nested_cv(sensitivity_cohort, cols, y, splits, label)
        all_oof.append(oof_df)
        all_metrics.append(metrics_df)

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    oof_all = pd.concat(all_oof, ignore_index=True)
    oof_all.to_csv(PREDICTIONS / "redundancy_gradient_SENSITIVITY_EXPLORATORY_oof.csv", index=False)

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(TABLES / "redundancy_gradient_SENSITIVITY_EXPLORATORY_raw.csv", index=False)

    def pull(block, colname):
        return metrics_all[metrics_all["block"] == block][["repeat", "AUPRC"]].rename(
            columns={"AUPRC": colname}
        )

    level1 = pull("C", "AUPRC_C")
    level1_k = pull("C+K", "AUPRC_CK")
    level2 = sensitivity_perf[["repeat", "AUPRC_CB", "AUPRC_CBK"]]
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

    gradient.to_csv(TABLES / "redundancy_gradient_SENSITIVITY_EXPLORATORY.csv", index=False)

    print(f"\nSaved (all flagged EXPLORATORY):")
    print(f"  {PREDICTIONS / 'redundancy_gradient_SENSITIVITY_EXPLORATORY_oof.csv'} ({len(oof_all)} rows)")
    print(f"  {TABLES / 'redundancy_gradient_SENSITIVITY_EXPLORATORY_raw.csv'} ({len(metrics_all)} rows)")
    print(f"  {TABLES / 'redundancy_gradient_SENSITIVITY_EXPLORATORY.csv'} ({len(gradient)} repeats x 3 levels)")
    print(
        "\nNo interpretation rendered on the standalone numbers above. See "
        "the original-vs-exploratory comparison below, which is the "
        "actual point of this script."
    )

    # --- Comparison against the ORIGINAL (SD-included) redundancy gradient ---
    # This comparison, not the exploratory numbers in isolation, is the
    # actual scientific content of this script: does excluding SD change
    # WHERE K's value gets absorbed across the gradient, not just whether
    # Level 2 crosses significance (already known from the sensitivity check).
    original_path = TABLES / "redundancy_gradient.csv"
    if not original_path.exists():
        print(
            f"\n({original_path} not found -- run run_redundancy_gradient.py "
            f"first to get the original SD-included gradient for comparison. "
            f"Exploratory-only results above are saved, but the comparison "
            f"this script is meant to produce was skipped.)"
        )
        return

    original_df = pd.read_csv(original_path)
    original_summary = summarize_gradient(original_df, "Original (SD included)")
    exploratory_summary = summarize_gradient(gradient, "Exploratory (SD excluded)")

    combined_summary = pd.concat([original_summary, exploratory_summary], ignore_index=True)
    combined_path = TABLES / "redundancy_gradient_ORIGINAL_vs_EXPLORATORY_summary.csv"
    combined_summary.to_csv(combined_path, index=False)

    print(f"\n=== Original (SD included) vs. exploratory (SD excluded) ===")
    print(combined_summary.to_string(index=False))
    print(f"\nSaved to {combined_path}")

    # Grouped boxplot: 3 levels x 2 cohort definitions, side by side per level
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    positions_orig = np.arange(len(LEVEL_COLS)) * 2.5
    positions_expl = positions_orig + 0.9

    data_orig = [original_df[c].values for c in LEVEL_COLS]
    data_expl = [gradient[c].values for c in LEVEL_COLS]

    bp1 = ax.boxplot(data_orig, positions=positions_orig, widths=0.7, patch_artist=True,
                      boxprops=dict(facecolor="lightblue"), showmeans=True)
    bp2 = ax.boxplot(data_expl, positions=positions_expl, widths=0.7, patch_artist=True,
                      boxprops=dict(facecolor="lightsalmon"), showmeans=True)

    rng = np.random.default_rng(0)
    for pos, vals in zip(positions_orig, data_orig):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, alpha=0.6,
                    color="tab:blue", zorder=3, s=15)
    for pos, vals in zip(positions_expl, data_expl):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, alpha=0.6,
                    color="tab:red", zorder=3, s=15)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(positions_orig + 0.45)
    ax.set_xticklabels([LEVEL_LABELS[c] for c in LEVEL_COLS])
    ax.set_ylabel("delta AUPRC")
    ax.set_title(
        "Redundancy gradient: original (SD included, prespecified) vs.\n"
        "exploratory post-hoc (SD excluded) -- EXPLORATORY, not primary"
    )
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["Original (SD included)", "Exploratory (SD excluded)"], loc="best")
    plt.tight_layout()
    fig_path = FIGURES / "redundancy_gradient_ORIGINAL_vs_EXPLORATORY.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved comparison figure to {fig_path}")

    print(
        "\nRead this figure/table for whether the GRADIENT SHAPE changes, "
        "not just whether Level 2 crosses significance (already known). "
        "Specifically: does Level 1's gap look similar in both, and does "
        "Level 2's jump appear gradually across levels or abruptly, in "
        "each cohort definition."
    )


if __name__ == "__main__":
    main()
