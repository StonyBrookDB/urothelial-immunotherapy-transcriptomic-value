"""
Compares the primary analysis (SD coded as non-responder) against the
CR/PR vs. PD sensitivity check (SD excluded), to assess whether the
primary null result is an artifact of how SD patients were coded.

Run from the project root, after both run_primary_model.py and
run_sensitivity_CRPR_vs_PD.py:
    python3 compare_primary_sensitivity.py

Reads:
    results/tables/primary_model_performance.csv
    results/tables/sensitivity_CRPR_vs_PD_performance.csv

Writes:
    results/tables/primary_vs_sensitivity_comparison.csv
    results/tables/primary_vs_sensitivity_summary.csv
    results/figures/primary_vs_sensitivity_delta_AUPRC.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TABLES = Path("results/tables")
FIGURES = Path("results/figures")

PRIMARY_LABEL = "Primary (SD included)"
SENSITIVITY_LABEL = "Sensitivity (SD excluded)"


def load_and_label(path, label):
    df = pd.read_csv(path)
    keep = df[["repeat", "AUPRC_CB", "AUPRC_CBK", "delta_AUPRC"]].copy()
    keep["analysis"] = label
    return keep


def main():
    primary = load_and_label(TABLES / "primary_model_performance.csv", PRIMARY_LABEL)
    sensitivity = load_and_label(
        TABLES / "sensitivity_CRPR_vs_PD_performance.csv", SENSITIVITY_LABEL
    )

    combined = pd.concat([primary, sensitivity], ignore_index=True)
    combined = combined[["analysis", "repeat", "AUPRC_CB", "AUPRC_CBK", "delta_AUPRC"]]

    TABLES.mkdir(parents=True, exist_ok=True)
    combined.to_csv(TABLES / "primary_vs_sensitivity_comparison.csv", index=False)

    summary = (
        combined.groupby("analysis")["delta_AUPRC"]
        .agg(["mean", "median", "std", "min", "max"])
        .reset_index()
    )
    summary["n_positive"] = combined.groupby("analysis")["delta_AUPRC"].apply(
        lambda s: (s > 0).sum()
    ).values
    summary["n_negative"] = combined.groupby("analysis")["delta_AUPRC"].apply(
        lambda s: (s < 0).sum()
    ).values
    summary.to_csv(TABLES / "primary_vs_sensitivity_summary.csv", index=False)

    print(combined.to_string(index=False))
    print()
    print(summary.to_string(index=False))

    # Paired box + jittered-strip comparison of delta_AUPRC distributions
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    labels = [PRIMARY_LABEL, SENSITIVITY_LABEL]
    data_by_label = [
        combined.loc[combined["analysis"] == lab, "delta_AUPRC"].values for lab in labels
    ]

    ax.boxplot(data_by_label, tick_labels=labels, showmeans=True, widths=0.5)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data_by_label, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, alpha=0.6,
                    color="tab:blue", zorder=3)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("delta AUPRC (C+B+K minus C+B)")
    ax.set_title("Primary vs. sensitivity: incremental value of K per repeat")
    plt.tight_layout()
    fig_path = FIGURES / "primary_vs_sensitivity_delta_AUPRC.png"
    plt.savefig(fig_path, dpi=150)
    print(f"\nSaved figure to {fig_path}")

    print(
        "\nNo interpretation rendered here. If the two distributions "
        "overlap substantially (similar mean/spread, similar sign split), "
        "that supports the primary null result NOT being an artifact of "
        "how SD patients were coded. This does not change the primary "
        "endpoint regardless of what the figure shows."
    )


if __name__ == "__main__":
    main()
