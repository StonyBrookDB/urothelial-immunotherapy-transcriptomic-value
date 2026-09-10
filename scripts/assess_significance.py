"""
Formal significance test on delta_AUPRC across repeats — the correct unit of
analysis (10 independent-ish repeat-level values), not the 50 individual
outer folds, per analysis_spec.md Section 13's warning against naive
fold-level t-tests.

Runs on whichever of these tables exist:
    results/tables/primary_model_performance.csv            -> "Primary"
    results/tables/sensitivity_CRPR_vs_PD_performance.csv   -> "Sensitivity (CR/PR vs PD)"

Run from the project root, after run_primary_model.py and/or
run_sensitivity_CRPR_vs_PD.py:
    python3 assess_significance.py

Writes:
    results/tables/primary_significance_test.csv       (if primary table found)
    results/tables/sensitivity_significance_test.csv   (if sensitivity table found)
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

TABLES = Path("results/tables")


def run_significance_test(delta, label):
    """
    Paired Wilcoxon signed-rank test, paired t-test, and a naive 95% CI on
    the mean, for a 1-D array of repeat-level delta_AUPRC values.
    """
    n = len(delta)
    mean = delta.mean()
    sd = delta.std(ddof=1)
    naive_se = sd / np.sqrt(n)

    # Wilcoxon: tests whether delta is symmetric around 0. Non-parametric,
    # appropriate for n=10 without assuming normality. Equivalent to
    # wilcoxon(AUPRC_treatment, AUPRC_baseline).
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(delta)

    # Paired t-test on the repeat-level values -- NOT the naive fold-level
    # t-test the spec warns against; repeats, not folds, are the correct
    # unit here. Still assumes rough normality of delta, a stronger
    # assumption than Wilcoxon's.
    ttest_stat, ttest_p = stats.ttest_1samp(delta, popmean=0)

    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean - t_crit * naive_se
    ci_high = mean + t_crit * naive_se

    return pd.DataFrame([{
        "analysis": label,
        "n_repeats": n,
        "mean_delta_AUPRC": mean,
        "sd_delta_AUPRC": sd,
        "naive_SE (sd/sqrt(n))": naive_se,
        "95pct_CI_low": ci_low,
        "95pct_CI_high": ci_high,
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p_value": wilcoxon_p,
        "paired_ttest_statistic": ttest_stat,
        "paired_ttest_p_value": ttest_p,
    }])


def run_for_file(path, label, out_name):
    if not path.exists():
        print(f"({path} not found — skipped)\n")
        return None
    perf = pd.read_csv(path)
    results = run_significance_test(perf["delta_AUPRC"].values, label)

    out_path = TABLES / out_name
    results.to_csv(out_path, index=False)
    print(f"=== {label} ===")
    print(results.to_string(index=False))
    print(f"Saved to {out_path}\n")
    return results


def main():
    found_any = False

    if run_for_file(
        TABLES / "primary_model_performance.csv", "Primary", "primary_significance_test.csv"
    ) is not None:
        found_any = True

    if run_for_file(
        TABLES / "sensitivity_CRPR_vs_PD_performance.csv",
        "Sensitivity (CR/PR vs PD)", "sensitivity_significance_test.csv",
    ) is not None:
        found_any = True

    if not found_any:
        print("Neither table found. Run run_primary_model.py and/or "
              "run_sensitivity_CRPR_vs_PD.py first.")
        return

    print(
        "Caveat for both: naive_SE and the CI treat the repeats as fully "
        "independent. They are not entirely independent (same patients "
        "resampled each time), so the true SE is somewhat larger than shown "
        "-- treat this as an optimistic lower bound on uncertainty, not an "
        "exact figure."
    )


if __name__ == "__main__":
    main()
