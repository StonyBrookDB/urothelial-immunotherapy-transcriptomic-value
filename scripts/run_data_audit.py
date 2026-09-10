"""
Step 1 driver: data audit + TPM generation.

Run from the project root:
    python3 run_data_audit.py

Requires data/raw/{sample_metadata.csv, expression_counts.csv, feature_metadata.csv,
human_gene_signatures_long.csv, genes.txt} to be present.

Writes:
    results/tables/sample_identity_audit.csv
    results/tables/duplicate_patient_specimens.csv   (only if duplicates found)
    results/tables/gene_annotation_audit.csv
    results/tables/feature_data_dictionary.csv
    data/processed/expression_TPM.csv
    data/processed/expression_log2TPM.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import preprocessing as pp


def main():
    print("Loading raw data...")
    data = pp.load_raw()

    print("\nRunning sample identity audit...")
    pp.sample_identity_audit(data["sample_metadata"], data["expression_counts"])

    print("\nRunning gene annotation audit...")
    pp.gene_annotation_audit(data["feature_metadata"], data["expression_counts"])

    print("\nComputing TPM...")
    pp.compute_tpm(data["expression_counts"], data["feature_metadata"])

    print("\nBuilding feature data dictionary...")
    pp.build_feature_data_dictionary(data["sample_metadata"])

    print("\nDone. Review results/tables/ before proceeding to signature scoring.")


if __name__ == "__main__":
    main()
