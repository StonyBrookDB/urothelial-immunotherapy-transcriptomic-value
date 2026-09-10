"""
Direct test of the mechanism behind the redundancy gradient and coefficient
shift results: are the 8 K signatures actually correlated with the B block
biomarkers (TMB, IC Level, TC Level)? Everything said so far about "K's
information being absorbed by B" has been an inference from downstream
symptoms (AUPRC shift, coefficient shift) -- this checks the thing itself.

Spearman rank correlation used throughout (not Pearson): IC Level and TC
Level are ordinal (mapped to 0/1/2), not continuous, and TMB is right-
skewed, so rank correlation is the more appropriate and defensible choice
for this mixed-type comparison.

Run from the project root:
    python3 check_K_B_correlation.py

Requires:
    data/processed/analysis_cohort.csv

Writes:
    results/tables/K_B_correlation_matrix.csv
    results/tables/K_B_correlation_pvalues.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROCESSED = Path("data/processed")
TABLES = Path("results/tables")

K_SIGNATURES = ["CD8_T_effector", "Immune_Checkpoint", "APM", "NK_cells",
                 "Cytotoxic_lymphocytes", "Monocytic_lineage", "Fibroblasts",
                 "Endothelial_cells"]

IC_TC_MAP = {"IC0": 0, "IC1": 1, "IC2+": 2, "TC0": 0, "TC1": 1, "TC2+": 2}


def bh_correct(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def main():
    cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)
    c = cohort[cohort["binaryResponse"].notna()].copy()

    c["TMB"] = c["FMOne mutation burden per MB"]
    c["IC_Level_ordinal"] = c["IC Level"].map(IC_TC_MAP)
    c["TC_Level_ordinal"] = c["TC Level"].map(IC_TC_MAP)
    b_features = ["TMB", "IC_Level_ordinal", "TC_Level_ordinal"]

    print(f"n patients: {len(c)}")
    for b in b_features:
        print(f"  {b}: {c[b].notna().sum()} non-missing")

    corr_matrix = pd.DataFrame(index=K_SIGNATURES, columns=b_features, dtype=float)
    pval_matrix = pd.DataFrame(index=K_SIGNATURES, columns=b_features, dtype=float)

    for k in K_SIGNATURES:
        for b in b_features:
            paired = c[[k, b]].dropna()
            rho, p = spearmanr(paired[k], paired[b])
            corr_matrix.loc[k, b] = rho
            pval_matrix.loc[k, b] = p

    # BH correction across all 24 (K x B) tests
    flat_p = pval_matrix.values.flatten()
    flat_q = bh_correct(flat_p)
    qval_matrix = pd.DataFrame(
        flat_q.reshape(pval_matrix.shape), index=pval_matrix.index, columns=pval_matrix.columns
    )

    TABLES.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(TABLES / "K_B_correlation_matrix.csv")
    pval_matrix.to_csv(TABLES / "K_B_correlation_pvalues.csv")
    qval_matrix.to_csv(TABLES / "K_B_correlation_qvalues.csv")

    print("\n=== Spearman correlation: K signatures (rows) x B biomarkers (columns) ===")
    print(corr_matrix.round(3).to_string())

    print("\n=== BH-corrected q-values (24 tests) ===")
    print(qval_matrix.round(4).to_string())

    print("\n=== Significant correlations (q < 0.05) ===")
    sig_mask = qval_matrix < 0.05
    any_sig = False
    for k in K_SIGNATURES:
        for b in b_features:
            if sig_mask.loc[k, b]:
                any_sig = True
                print(f"  {k} x {b}: rho={corr_matrix.loc[k,b]:.3f}, q={qval_matrix.loc[k,b]:.4f}")
    if not any_sig:
        print("  (none)")

    print(
        "\nNo interpretation forced here. Strong, significant correlations "
        "between specific K signatures and specific B features are direct "
        "mechanistic evidence for redundancy. Weak/absent correlations "
        "would mean the AUPRC and coefficient-shift patterns need a "
        "different explanation than simple pairwise overlap."
    )


if __name__ == "__main__":
    main()
