"""
Signature manifest construction and score computation for the AMIA IMvigor210
reanalysis. Implements Section 3 of analysis_spec.md.

All gene lookups are keyed on Entrez ID, never on symbol string, per the
decision documented in the gene annotation audit (src/preprocessing.py).
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED = Path("data/processed")
TABLES = Path("results/tables")

# Signatures pulled from the Mariathasan 2018 bundled signature file
# (data/raw/human_gene_signatures_long.csv), by their internal label.
MARIATHASAN_PRIMARY = ["CD 8 T effector", "Immune Checkpoint", "APM"]
MARIATHASAN_SECONDARY = ["gene19"]  # = Pan-F TBRS, kept secondary (circular w.r.t. IMvigor210)

# Cell populations pulled from the MCP-counter gene list (data/raw/genes.txt),
# restricted to the 5 populations used in the original report / locked spec.
MCP_POPULATIONS_PRIMARY = [
    "NK cells",
    "Cytotoxic lymphocytes",
    "Monocytic lineage",
    "Fibroblasts",
    "Endothelial cells",
]

DISPLAY_NAMES = {
    "CD 8 T effector": "CD8_T_effector",
    "Immune Checkpoint": "Immune_Checkpoint",
    "APM": "APM",
    "gene19": "Pan_F_TBRS",
    "NK cells": "NK_cells",
    "Cytotoxic lymphocytes": "Cytotoxic_lymphocytes",
    "Monocytic lineage": "Monocytic_lineage",
    "Fibroblasts": "Fibroblasts",
    "Endothelial cells": "Endothelial_cells",
}


def symbol_to_entrez_map(feature_metadata):
    """
    Build a symbol -> entrez_id lookup from feature_metadata, keeping only
    symbols that map to exactly one Entrez ID (the one duplicated symbol
    found in the gene annotation audit is dropped here and logged, not
    silently resolved by picking one).
    """
    fm = feature_metadata.dropna(subset=["symbol"])
    counts = fm["symbol"].value_counts()
    ambiguous = counts[counts > 1].index.tolist()
    unambiguous = fm[~fm["symbol"].isin(ambiguous)]
    mapping = dict(zip(unambiguous["symbol"], unambiguous["entrez_id"]))
    return mapping, ambiguous


def build_signature_gene_map(mariathasan_signatures, mcp_genes, feature_metadata):
    """
    Resolve every signature to a list of Entrez IDs actually present in the
    expression data, and build the manifest documenting coverage.

    Returns:
        gene_map: dict of display_name -> list of entrez_id (available genes only)
        manifest: DataFrame, one row per signature
    """
    sym2entrez, ambiguous_symbols = symbol_to_entrez_map(feature_metadata)
    available_entrez = set(feature_metadata.index)

    manifest_rows = []
    gene_map = {}

    # --- Mariathasan-bundled signatures (symbol-based source file) ---
    for label in MARIATHASAN_PRIMARY + MARIATHASAN_SECONDARY:
        symbols = mariathasan_signatures.loc[
            mariathasan_signatures["signature"] == label, "gene"
        ].tolist()
        entrez_ids = []
        missing = []
        for sym in symbols:
            if sym in ambiguous_symbols:
                missing.append(f"{sym} (ambiguous symbol, excluded)")
                continue
            eid = sym2entrez.get(sym)
            if eid is None:
                missing.append(f"{sym} (symbol not found in feature_metadata)")
                continue
            if eid not in available_entrez:
                missing.append(f"{sym} (entrez {eid} not in expression data)")
                continue
            entrez_ids.append(eid)

        display = DISPLAY_NAMES[label]
        gene_map[display] = entrez_ids
        manifest_rows.append({
            "signature": display,
            "source": "Mariathasan et al. 2018 (bundled)",
            "block": "Primary K" if label in MARIATHASAN_PRIMARY else "Secondary (circular)",
            "independent_of_imvigor": label != "gene19",
            "n_genes_required": len(symbols),
            "n_genes_available": len(entrez_ids),
            "missing_genes": "; ".join(missing) if missing else "",
        })

    # --- MCP-counter cell population signatures (genes.txt, has entrez already) ---
    for pop in MCP_POPULATIONS_PRIMARY:
        rows = mcp_genes[mcp_genes["Cell population"] == pop]
        entrez_ids = []
        missing = []
        for _, r in rows.iterrows():
            eid = r["ENTREZID"]
            if pd.isna(eid):
                missing.append(f"{r['HUGO symbols']} (no Entrez ID in source file)")
                continue
            eid = int(eid)
            if eid not in available_entrez:
                missing.append(f"{r['HUGO symbols']} (entrez {eid} not in expression data)")
                continue
            entrez_ids.append(eid)

        display = DISPLAY_NAMES[pop]
        gene_map[display] = entrez_ids
        manifest_rows.append({
            "signature": display,
            "source": "Becht et al. 2016 MCP-counter (official gene-level marker list)",
            "block": "Primary K",
            "independent_of_imvigor": True,
            "n_genes_required": len(rows),
            "n_genes_available": len(entrez_ids),
            "missing_genes": "; ".join(missing) if missing else "",
        })

    manifest = pd.DataFrame(manifest_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(PROCESSED / "signature_manifest.csv", index=False)

    print("=== Signature manifest ===")
    print(manifest.to_string(index=False))

    for display, ids in gene_map.items():
        assert len(ids) > 0, f"Signature {display} has zero available genes — cannot compute a score."

    return gene_map, manifest


def compute_signature_scores(log2_tpm, gene_map):
    """
    Score = mean of log2(TPM+1) across each signature's available marker
    genes, per sample. This matches the MCP-counter paper's own definition
    (log2 average / geometric-mean expression) and is applied identically
    to the Mariathasan-bundled signatures for consistency.

    log2_tpm: genes (Entrez ID index) x samples DataFrame.
    gene_map: dict of signature_name -> list of entrez_id.

    Returns a samples x signatures DataFrame.
    """
    scores = {}
    for name, entrez_ids in gene_map.items():
        present = [g for g in entrez_ids if g in log2_tpm.index]
        assert len(present) == len(entrez_ids), (
            f"Signature {name}: gene_map claims genes present in expression data "
            f"that are missing from the log2_tpm matrix passed in — check inputs match."
        )
        scores[name] = log2_tpm.loc[present].mean(axis=0)

    scores_df = pd.DataFrame(scores)
    scores_df.index.name = "sample_id"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(PROCESSED / "signature_scores.csv")

    print(f"\n=== Signature scores computed ===")
    print(f"Samples: {scores_df.shape[0]}, signatures: {scores_df.shape[1]}")
    print(scores_df.describe().T[["mean", "std", "min", "max"]])

    return scores_df
