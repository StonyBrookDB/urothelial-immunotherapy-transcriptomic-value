"""
Data-driven elastic net gene selection + stability analysis, evaluated at
Level 2 (C+B vs. C+B+ElasticNet) — the design agreed on as a middle path
between the original top-5000/MWU-screen approach and full rigor:

  - Prefilter: near-zero-variance floor (unsupervised), not an arbitrary
    top-N cutoff.
  - l1_ratio grid skewed toward sparsity (0.5-1.0), so "selected" means
    genuinely nonzero, not small-but-nonzero.
  - Final evaluation reuses the SAME ridge classifier as everything else
    (via modeling.build_pipeline) on C+B + whichever genes elastic net
    selected on that fold's training data — this isolates "how good is the
    data-driven feature *selection*" from "which algorithm did the final
    fit," matching the primary design's principle of holding the model
    family fixed to compare information content, not algorithms.
  - Stability (selection frequency, Jaccard, sign consistency) is measured
    across the existing CV-fold structure — an approximation to
    Meinshausen & Bühlmann (2010) stability selection, NOT the full
    multi-subsample/regularization-path version. State this explicitly as
    a limitation in the manuscript, don't present it as the genuine article.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

import modeling as mdl

VARIANCE_FLOOR = 0.05         # near-zero-variance floor on log2(TPM+1) scale
L1_RATIO_GRID = [0.5, 0.7, 0.9, 1.0]   # skewed toward sparsity, by design
ENET_C_GRID = [0.01, 0.1]      # dropped C=1.0: weak regularization converged poorly and
                                # produced non-sparse (~1000+ gene) selections, defeating
                                # the point of a sparsity-focused stability analysis
ENET_INNER_FOLDS = 3                    # reduced from 5 — compute cost, documented trade-off
ENET_MAX_ITER = 8000            # raised from 3000 after observing non-convergence
ENET_TOL = 1e-3                 # loosened from sklearn default (1e-4) to make convergence
                                 # at this feature count practically achievable within
                                 # ENET_MAX_ITER — documented precision/speed trade-off


def variance_prefilter_train_genes(log2_tpm, train_sample_ids, floor=VARIANCE_FLOOR):
    """
    Unsupervised prefilter, computed on TRAINING samples only (fold-safe).
    Drops near-zero-variance genes rather than an arbitrary top-N cutoff.
    """
    X_train = log2_tpm[train_sample_ids].T  # samples x genes
    vt = VarianceThreshold(threshold=floor)
    vt.fit(X_train)
    return log2_tpm.index[vt.get_support()]


def select_genes_elastic_net(X_train_genes, y_train, seed):
    """
    Elastic-net logistic regression with an inner grid search over
    (l1_ratio, C), selected by AUPRC. Returns genes with nonzero
    coefficients at the best hyperparameters, their coefficient signs, and
    whether the winning fit actually converged (do not trust selections
    from a non-converged fit -- the coefficients are an unfinished
    optimization, not a stable answer).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_genes)

    inner_cv = StratifiedKFold(n_splits=ENET_INNER_FOLDS, shuffle=True, random_state=seed)
    param_grid = {"l1_ratio": L1_RATIO_GRID, "C": ENET_C_GRID}
    base = LogisticRegression(
        penalty="elasticnet", solver="saga", max_iter=ENET_MAX_ITER, tol=ENET_TOL,
        class_weight=None,
    )
    gs = GridSearchCV(base, param_grid, scoring="average_precision", cv=inner_cv, n_jobs=-1)
    gs.fit(X_scaled, y_train)

    coefs = gs.best_estimator_.coef_.ravel()
    selected_mask = coefs != 0
    selected_genes = X_train_genes.columns[selected_mask]
    signs = pd.Series(np.sign(coefs[selected_mask]), index=selected_genes)

    n_iter = int(np.asarray(gs.best_estimator_.n_iter_).max())
    converged = n_iter < ENET_MAX_ITER

    return selected_genes, signs, gs.best_params_, converged, n_iter


def run_elastic_net_arm(cohort, log2_tpm, y, splits, seed=None):
    """
    For each outer fold: prefilter + select genes on training samples only,
    then fit the standard ridge pipeline (C+B + selected genes) and predict
    on the held-out fold. Mirrors how block K is evaluated, except the gene
    set is fold-specific rather than fixed.

    cohort/y must be positionally aligned with the splits (same row order
    used to generate cv_splits.pkl).

    Returns: oof_df, metrics_df, selection_df, convergence_df, background_genes
        background_genes: dict of (repeat, fold) -> list of gene_entrez ints
        that survived the variance-floor prefilter on that fold's training
        data. This is the correct background/universe for any downstream
        enrichment test on that fold's selected genes -- NOT the full gene
        panel, which would bias enrichment results toward genes that were
        never even eligible for selection.
    """
    seed = seed if seed is not None else mdl.SEED
    cb_cols = mdl.get_feature_columns("C", "B")

    sample_ids = cohort.index.astype(str)
    log2_tpm = log2_tpm.copy()
    log2_tpm.columns = log2_tpm.columns.astype(str)

    oof_records, selection_records, convergence_records = [], [], []
    background_genes = {}

    progress = tqdm(
        list(enumerate(splits)), total=len(splits), desc="ElasticNet folds", unit="fold"
    )
    for i, (train_idx, test_idx) in progress:
        repeat_num = i // mdl.N_OUTER_SPLITS
        fold_num = i % mdl.N_OUTER_SPLITS
        progress.set_postfix(repeat=repeat_num, fold=fold_num)
        train_ids = sample_ids[train_idx]
        test_ids = sample_ids[test_idx]
        y_train = y.iloc[train_idx]

        kept_genes = variance_prefilter_train_genes(log2_tpm, train_ids)
        background_genes[(repeat_num, fold_num)] = list(kept_genes)
        X_genes_train = log2_tpm.loc[kept_genes, train_ids].T  # samples x genes
        X_genes_test = log2_tpm.loc[kept_genes, test_ids].T

        selected_genes, signs, best_params, converged, n_iter = select_genes_elastic_net(
            X_genes_train, y_train, seed=seed + repeat_num
        )
        convergence_records.append({
            "repeat": repeat_num, "fold": fold_num, "converged": converged,
            "n_iter": n_iter, "max_iter": ENET_MAX_ITER, "best_params": best_params,
        })
        if not converged:
            tqdm.write(f"  [ElasticNet] WARNING: repeat {repeat_num} fold {fold_num} "
                       f"did NOT converge ({n_iter}/{ENET_MAX_ITER} iterations) — "
                       f"treat this fold's gene selection with caution.")

        for g in selected_genes:
            selection_records.append({
                "repeat": repeat_num, "fold": fold_num,
                "gene_entrez": g, "coef_sign": int(signs[g]),
            })

        fold_train = cohort.iloc[train_idx].copy()
        fold_test = cohort.iloc[test_idx].copy()

        if len(selected_genes) == 0:
            tqdm.write(f"  [ElasticNet] repeat {repeat_num} fold {fold_num}: "
                       f"0 genes selected (params={best_params}) — falling back to C+B only.")
            final_cols = cb_cols
        else:
            # Gene columns are integer Entrez IDs; C/B columns are strings.
            # sklearn's ColumnTransformer rejects mixed-type column names, so
            # gene columns get a string alias here. Original Entrez IDs are
            # preserved in selection_records for the stability analysis.
            #
            # Built via a single pd.concat rather than per-column assignment
            # in a loop -- repeated frame.insert on a wide DataFrame triggers
            # pandas' fragmentation warning and is genuinely slower.
            gene_col_names = [f"gene_{g}" for g in selected_genes]
            final_cols = cb_cols + gene_col_names

            train_gene_block = X_genes_train[selected_genes].copy()
            train_gene_block.columns = gene_col_names
            train_gene_block.index = fold_train.index
            fold_train = pd.concat([fold_train, train_gene_block], axis=1)

            test_gene_block = X_genes_test[selected_genes].copy()
            test_gene_block.columns = gene_col_names
            test_gene_block.index = fold_test.index
            fold_test = pd.concat([fold_test, test_gene_block], axis=1)

        pipe = mdl.build_pipeline(fold_train, final_cols)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat_num)
        gs_final = GridSearchCV(pipe, {"clf__C": mdl.C_GRID}, scoring="average_precision",
                                 cv=inner_cv, n_jobs=-1)
        gs_final.fit(fold_train[final_cols], y_train)
        proba = gs_final.predict_proba(fold_test[final_cols])[:, 1]

        for pos, p in zip(test_idx, proba):
            oof_records.append({
                "block": "C+B+ElasticNet", "repeat": repeat_num,
                "patient_index": cohort.index[pos], "y_true": int(y.iloc[pos]),
                "y_pred_proba": float(p), "n_genes_selected": len(selected_genes),
            })

        progress.set_postfix(repeat=repeat_num, fold=fold_num, n_genes=len(selected_genes))
        tqdm.write(f"  [ElasticNet] repeat {repeat_num} fold {fold_num} done "
                   f"({len(selected_genes)} genes selected, enet_params={best_params})")

    oof_df = pd.DataFrame(oof_records)
    selection_df = pd.DataFrame(selection_records)
    convergence_df = pd.DataFrame(convergence_records)

    metric_rows = []
    for repeat, g in oof_df.groupby("repeat"):
        metric_rows.append({
            "block": "C+B+ElasticNet", "repeat": repeat,
            "AUPRC": average_precision_score(g["y_true"], g["y_pred_proba"]),
            "AUROC": roc_auc_score(g["y_true"], g["y_pred_proba"]),
            "Brier": brier_score_loss(g["y_true"], g["y_pred_proba"]),
        })
    metrics_df = pd.DataFrame(metric_rows)

    n_nonconverged = (~convergence_df["converged"]).sum()
    if n_nonconverged > 0:
        print(f"\n{n_nonconverged} of {len(convergence_df)} folds did not converge — "
              f"see convergence_df / elastic_net_convergence.csv before trusting "
              f"stability metrics from those folds.")

    return oof_df, metrics_df, selection_df, convergence_df, background_genes


def compute_stability_metrics(selection_df, n_folds_total):
    """
    Selection frequency per gene, pairwise Jaccard between fold-level
    selections, and sign-consistency among genes selected more than once.
    """
    freq = (
        selection_df.groupby("gene_entrez").size().rename("times_selected").reset_index()
    )
    freq["selection_frequency"] = freq["times_selected"] / n_folds_total
    freq = freq.sort_values("selection_frequency", ascending=False)

    fold_sets = selection_df.groupby(["repeat", "fold"])["gene_entrez"].apply(set)
    jaccards = []
    keys = list(fold_sets.index)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = fold_sets.iloc[i], fold_sets.iloc[j]
            union = a | b
            jaccards.append(len(a & b) / len(union) if union else np.nan)
    jaccard_summary = pd.Series(jaccards, name="jaccard").describe()

    sign_consistency = (
        selection_df.groupby("gene_entrez")["coef_sign"]
        .apply(lambda s: (s == s.mode().iloc[0]).mean() if len(s) > 1 else np.nan)
        .rename("sign_consistency")
        .reset_index()
    )

    return freq, jaccard_summary, sign_consistency
