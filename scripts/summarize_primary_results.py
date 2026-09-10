"""
Summarizes model performance tables into the distributional statistics
analysis_spec.md Sections 5/6/13 call for (mean/median/SD/quartiles/range
across repeats — never a naive test treating folds as independent).

Handles two tables, either or both if present:
    results/tables/primary_model_performance.csv   -> delta_AUPRC (Level 2, C+B vs C+B+K)
    results/tables/redundancy_gradient.csv          -> delta_AUPRC at all 3 levels

Run from the project root, after run_primary_model.py and/or
run_redundancy_gradient.py:
    python3 summarize_primary_results.py

Writes:
    results/tables/primary_model_summary.csv       (if primary table found)
    results/tables/redundancy_gradient_summary.csv (if redundancy table found)
"""

from pathlib import Path
import pandas as pd

TABLES = Path("results/tables")


def summarize_columns(df, delta_cols, context_cols=()):
    """
    Generic summarizer: one row per column in delta_cols + context_cols.
    n_repeats_positive/negative only computed for columns in delta_cols
    (i.e., actual delta/effect-size columns, not raw AUPRC/AUROC/Brier).
    """
    rows = []
    for m in list(delta_cols) + list(context_cols):
        s = df[m]
        is_delta = m in delta_cols
        rows.append({
            "metric": m,
            "mean": s.mean(),
            "median": s.median(),
            "sd": s.std(),
            "q25": s.quantile(0.25),
            "q75": s.quantile(0.75),
            "min": s.min(),
            "max": s.max(),
            "n_repeats_positive": (s > 0).sum() if is_delta else None,
            "n_repeats_negative": (s < 0).sum() if is_delta else None,
        })
    return pd.DataFrame(rows)


def summarize_primary():
    path = TABLES / "primary_model_performance.csv"
    if not path.exists():
        return None
    perf = pd.read_csv(path)
    summary = summarize_columns(
        perf,
        delta_cols=["delta_AUPRC"],
        context_cols=["AUPRC_CB", "AUPRC_CBK", "AUROC_CB", "AUROC_CBK",
                       "Brier_CB", "Brier_CBK"],
    )
    out_path = TABLES / "primary_model_summary.csv"
    summary.to_csv(out_path, index=False)
    print("=== Primary model (Level 2: C+B vs. C+B+K) ===")
    print(summary.to_string(index=False))
    print(f"Saved to {out_path}\n")
    return summary


def summarize_redundancy_gradient():
    path = TABLES / "redundancy_gradient.csv"
    if not path.exists():
        return None
    grad = pd.read_csv(path)
    delta_cols = [
        "delta_AUPRC_level1_C",
        "delta_AUPRC_level2_CB",
        "delta_AUPRC_level3_CBR",
    ]
    summary = summarize_columns(grad, delta_cols=delta_cols)
    out_path = TABLES / "redundancy_gradient_summary.csv"
    summary.to_csv(out_path, index=False)
    print("=== Redundancy gradient (Levels 1-3) ===")
    print(summary.to_string(index=False))
    print(f"Saved to {out_path}\n")
    return summary


def main():
    found_any = False

    if summarize_primary() is not None:
        found_any = True
    else:
        print(f"({TABLES / 'primary_model_performance.csv'} not found — skipped)\n")

    if summarize_redundancy_gradient() is not None:
        found_any = True
    else:
        print(f"({TABLES / 'redundancy_gradient.csv'} not found — skipped)\n")

    if not found_any:
        print("Neither table found. Run run_primary_model.py and/or "
              "run_redundancy_gradient.py first.")
        return

    print(
        "No verdict rendered here by design. Compare each delta column's "
        "mean/median against its spread (sd, q25-q75, min-max) yourself, per "
        "analysis_spec.md Section 5 (stable positive / stable null / unstable). "
        "For the redundancy gradient specifically, compare how the three levels "
        "move relative to each other (e.g. shrinking toward zero as the baseline "
        "gets richer), not just each level in isolation."
    )


if __name__ == "__main__":
    main()