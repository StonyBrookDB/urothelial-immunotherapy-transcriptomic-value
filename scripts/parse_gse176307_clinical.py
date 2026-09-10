"""
Parses GSE176307's series matrix clinical characteristics into a clean
per-patient table.

CRITICAL: !Sample_characteristics_ch1 rows are NOT reliably aligned by row
position across samples. Confirmed on this file: 90 samples split into 2
distinct field-orderings (62 vs 28 samples), almost certainly because
samples with a conditional field (e.g. "histology.other.description",
present only for non-standard histology) have every subsequent field
shifted by one position relative to samples without it. Parsing by row
index would silently assign the wrong value to the wrong field for a
whole subset of patients. This parser keys every value by its "key: value"
label instead, which is robust to the misalignment regardless of row order.

Run from the project root:
    python3 parse_gse176307_clinical.py

Requires:
    data/external/GSE176307_series_matrix.txt

Writes:
    data/external/gse176307_clinical.csv
"""

from pathlib import Path
import pandas as pd

RAW = Path("data/external")


def parse_series_matrix_characteristics(path):
    """
    Returns a DataFrame: one row per sample, one column per characteristic
    field actually present in the file (ragged fields become NaN for
    samples that don't have them, not misaligned).
    """
    char_rows = []
    sample_titles = None
    geo_accessions = None

    with open(path) as f:
        for line in f:
            if line.startswith("!Sample_title"):
                sample_titles = [
                    v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]
                ]
            elif line.startswith("!Sample_geo_accession"):
                geo_accessions = [
                    v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]
                ]
            elif line.startswith("!Sample_characteristics_ch1"):
                char_rows.append(line.rstrip("\n").split("\t")[1:])

    assert sample_titles is not None, "Sample_title row not found in series matrix."
    assert geo_accessions is not None, "Sample_geo_accession row not found."
    n_samples = len(sample_titles)
    for row in char_rows:
        assert len(row) == n_samples, (
            f"Characteristic row has {len(row)} entries, expected {n_samples} "
            f"-- series matrix may be malformed."
        )

    # Build one dict per sample, keyed by field name parsed from "key: value"
    per_sample_dicts = [dict() for _ in range(n_samples)]
    for row in char_rows:
        for i, val in enumerate(row):
            val = val.strip().strip('"')
            if ":" not in val:
                continue  # blank/padding entries, e.g. trailing empty characteristics row
            key, value = val.split(":", 1)
            key = key.strip()
            value = value.strip()
            # If this key already exists for this sample (shouldn't happen, but
            # guard against silent overwrite if it does), keep the first value
            # and flag it rather than silently dropping data.
            if key in per_sample_dicts[i]:
                raise ValueError(
                    f"Duplicate field '{key}' encountered for sample index {i} "
                    f"({sample_titles[i]}) -- parsing assumption violated, "
                    f"investigate before trusting this output."
                )
            per_sample_dicts[i][key] = value

    df = pd.DataFrame(per_sample_dicts)
    df.insert(0, "geo_accession", geo_accessions)
    df.insert(0, "sample_title", sample_titles)
    return df


def main():
    matrix_path = RAW / "GSE176307_series_matrix.txt"
    assert matrix_path.exists(), f"{matrix_path} not found."

    clinical = parse_series_matrix_characteristics(matrix_path)

    print(f"Parsed {len(clinical)} samples, {clinical.shape[1]} columns.")
    print("\nColumns found:")
    print(clinical.columns.tolist())

    print("\nMissingness per field (samples where field wasn't present at all):")
    print(clinical.isna().sum().sort_values(ascending=False))

    if "io.response" in clinical.columns:
        print("\nio.response value counts:")
        print(clinical["io.response"].value_counts(dropna=False))

    if "io.therapy" in clinical.columns:
        print("\nio.therapy value counts (drug heterogeneity check):")
        print(clinical["io.therapy"].value_counts(dropna=False))
        if "io.response" in clinical.columns:
            atezo = clinical[clinical["io.therapy"] == "Atezolizumab"]
            print(f"\nAtezolizumab-only subset: n={len(atezo)}")
            print(atezo["io.response"].value_counts(dropna=False))

    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / "gse176307_clinical.csv"
    clinical.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
