"""
Table 1: baseline characteristics for IMvigor210 (training/development
cohort) ONLY, split into three columns: Total, Responders (CR/PR), and
Non-responders (SD/PD). Response category counts and response rate are
NOT included as rows here -- stated in prose in Methods instead, and
would otherwise be redundant with the column structure itself.

GSE176307/UNC-108 (external validation) is deliberately not a column in
this table -- roughly a third of a combined table's rows would be "Not
available" for one cohort or the other (PD-L1 IHC, this metastatic site
scheme, etc. don't exist in GSE176307). GSE176307's cohort description
belongs in prose in External Validation Results instead.

Run from the project root:
    python3 build_table1.py

Requires:
    data/processed/analysis_cohort.csv

Writes:
    results/tables/table1_baseline_characteristics.csv
    results/tables/table1_baseline_characteristics.md
"""

from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")
TABLES = Path("results/tables")


def cat_stats(series, prefix, order):
    """Returns {prefix__category: value_string, ...} plus a Missing entry."""
    n_total = len(series)
    n_missing = series.isna().sum()
    counts = series.value_counts(dropna=True)
    out = {}
    for cat in order:
        n = int(counts.get(cat, 0))
        pct = 100 * n / (n_total - n_missing) if (n_total - n_missing) > 0 else float("nan")
        out[f"{prefix}__{cat}"] = f"{n} ({pct:.1f}%)"
    # Always include a Missing entry, even when zero, so every group's
    # column shows an explicit "0 (0.0%)" rather than a blank cell when
    # another group in the same row has nonzero missingness.
    out[f"{prefix}__Missing"] = f"{n_missing} ({100*n_missing/n_total:.1f}%)" if n_total > 0 else "0 (0.0%)"
    return out


def cont_stats(series, label):
    """Returns {label: median-IQR string, label__Missing: ...} for one group."""
    n_total = len(series)
    n_missing = series.isna().sum()
    s = series.dropna()
    out = {}
    if len(s) == 0:
        out[label] = "no data"
        out[f"{label}__Missing"] = f"{n_missing} ({100*n_missing/n_total:.1f}%)" if n_total > 0 else "0 (0.0%)"
        return out
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    out[label] = f"{med:.2f} (IQR {q1:.2f}\u2013{q3:.2f})"
    # Always include Missing, even when zero -- see cat_stats for rationale.
    out[f"{label}__Missing"] = f"{n_missing} ({100*n_missing/n_total:.1f}%)" if n_total > 0 else "0 (0.0%)"
    return out


def summarize_group(c):
    """Computes all stats for one patient subset (total, responders, or non-responders)."""
    stats = {"N": f"{len(c)}"}
    stats.update(cat_stats(c["Sex"], "Sex", ["M", "F"]))

    ecog_display = c["Baseline ECOG Score"].astype("Int64").astype(str).replace("<NA>", pd.NA)
    stats.update(cat_stats(ecog_display, "ECOG", ["0", "1", "2"]))
    stats.update(cat_stats(c["Tobacco Use History"], "Tobacco", ["NEVER", "PREVIOUS", "CURRENT"]))

    stats.update(cont_stats(c["FMOne mutation burden per MB"], "TMB, median mut/Mb (IQR)"))

    n = len(c)
    stats["Treatment"] = f"Atezolizumab, {n} (100%)"

    stats.update(cat_stats(c["IC Level"], "IC", ["IC0", "IC1", "IC2+"]))
    stats.update(cat_stats(c["TC Level"], "TC", ["TC0", "TC1", "TC2+"]))
    stats.update(cat_stats(c["Met Disease Status"], "MetSite", ["LN Only", "Visceral", "Liver"]))
    stats.update(cont_stats(c["Neoantigen burden per MB"], "Neoantigen burden, median per MB (IQR)"))
    stats.update(cat_stats(c["Immune phenotype"], "ImmunePheno", ["inflamed", "excluded", "desert"]))
    return stats


def build_rows(stats_total, stats_resp, stats_nonresp):
    rows = []

    def header(text):
        rows.append((text, "", "", ""))

    def item(key, display=None):
        display = display or f"    {key}"
        rows.append((
            display,
            stats_total.get(key, ""),
            stats_resp.get(key, ""),
            stats_nonresp.get(key, ""),
        ))

    item("N")
    header("Sex, n (%)")
    for cat in ["M", "F"]:
        item(f"Sex__{cat}", f"    {cat}")
    header("ECOG, n (%)")
    for cat in ["0", "1", "2"]:
        item(f"ECOG__{cat}", f"    {cat}")
    header("Tobacco/smoking history, n (%)")
    for cat in ["NEVER", "PREVIOUS", "CURRENT"]:
        item(f"Tobacco__{cat}", f"    {cat}")
    item("TMB, median mut/Mb (IQR)")
    if any("TMB, median mut/Mb (IQR)__Missing" in d for d in (stats_total, stats_resp, stats_nonresp)):
        item("TMB, median mut/Mb (IQR)__Missing", "    Missing")
    item("Treatment")
    header("PD-L1 IC level (IHC, SP142), n (%)")
    for cat in ["IC0", "IC1", "IC2+"]:
        item(f"IC__{cat}", f"    {cat}")
    if any("IC__Missing" in d for d in (stats_total, stats_resp, stats_nonresp)):
        item("IC__Missing", "    Missing")
    header("PD-L1 TC level (IHC, SP142), n (%)")
    for cat in ["TC0", "TC1", "TC2+"]:
        item(f"TC__{cat}", f"    {cat}")
    if any("TC__Missing" in d for d in (stats_total, stats_resp, stats_nonresp)):
        item("TC__Missing", "    Missing")
    header("Metastatic site (liver/visceral/LN only), n (%)")
    for cat in ["LN Only", "Visceral", "Liver"]:
        item(f"MetSite__{cat}", f"    {cat}")
    if any("MetSite__Missing" in d for d in (stats_total, stats_resp, stats_nonresp)):
        item("MetSite__Missing", "    Missing")
    item("Neoantigen burden, median per MB (IQR)")
    if any("Neoantigen burden, median per MB (IQR)__Missing" in d for d in (stats_total, stats_resp, stats_nonresp)):
        item("Neoantigen burden, median per MB (IQR)__Missing", "    Missing")
    header("Immune phenotype, n (%)")
    for cat in ["inflamed", "excluded", "desert"]:
        item(f"ImmunePheno__{cat}", f"    {cat}")
    if any("ImmunePheno__Missing" in d for d in (stats_total, stats_resp, stats_nonresp)):
        item("ImmunePheno__Missing", "    Missing")

    return pd.DataFrame(rows, columns=[
        "Characteristic", "Total", "Responders (CR/PR)", "Non-responders (SD/PD)"
    ])


def main():
    analysis_cohort = pd.read_csv(PROCESSED / "analysis_cohort.csv", index_col=0)
    c = analysis_cohort[analysis_cohort["binaryResponse"].notna()].copy()
    responders = c[c["binaryResponse"] == "CR/PR"]
    nonresponders = c[c["binaryResponse"] == "SD/PD"]

    stats_total = summarize_group(c)
    stats_resp = summarize_group(responders)
    stats_nonresp = summarize_group(nonresponders)
    table = build_rows(stats_total, stats_resp, stats_nonresp)

    TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES / "table1_baseline_characteristics.csv"
    table.to_csv(csv_path, index=False)

    md_path = TABLES / "table1_baseline_characteristics.md"
    with open(md_path, "w") as f:
        f.write("## Table 1. Baseline characteristics, IMvigor210 (training cohort)\n\n")
        f.write(f"Total n={len(c)}, Responders (CR/PR) n={len(responders)}, "
                f"Non-responders (SD/PD) n={len(nonresponders)}.\n\n")
        f.write(table.to_markdown(index=False))
        f.write(
            "\n\n*GSE176307/UNC-108 (external validation cohort) is described "
            "in prose in External Validation Results, since it does not share "
            "several of these fields (PD-L1 IHC, this metastatic site scheme) "
            "with IMvigor210.*\n"
        )

    print(f"Total n={len(c)}, Responders n={len(responders)}, Non-responders n={len(nonresponders)}\n")
    print(table.to_string(index=False))
    print(f"\nSaved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()