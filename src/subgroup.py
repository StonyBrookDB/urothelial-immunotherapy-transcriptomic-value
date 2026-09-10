"""
Subgroup redundancy analysis: does block K's incremental value over C+B
concentrate in a specific PD-L1 IC Level subgroup (e.g. IC1, the
intermediate/ambiguous category) rather than being uniform across the
cohort? Tests a more clinically pointed question than the full-cohort
redundancy gradient: does transcriptomics help most exactly where the
clinical biomarker call is least clear-cut.

Uses the SAME ridge pipeline (modeling.build_pipeline) and C grid as the
primary analysis. CV splits are generated FRESH per subgroup (cannot reuse
cv_splits.pkl, built on the full 298-patient cohort's row order) because
each subgroup is a distinct, much smaller patient subset.

Small-sample caveat: subgroup sizes here (roughly 80-115 patients, 13-35
responders) are meaningfully smaller than the full cohort. Inner CV folds
are reduced from 5 to 3 to avoid degenerate splits with very few
responders per training fold -- a documented stability trade-off, not the
primary analysis's design. Results here should be read with more caution
than the full-cohort analyses; higher variance is expected.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

import modeling as mdl

SUBGROUP_INNER_FOLDS = 3           # reduced from 5 -- small subgroup sizes
MIN_RESPONDERS_PER_SUBGROUP = 10   # below this, CV is too unstable to be meaningful


def run_subgroup_redundancy(cohort_eval, subgroup_col, seed=None):
    """
    For each non-null level of subgroup_col, runs C+B vs. C+B+K nested CV
    WITHIN that subgroup only. Returns a combined per-repeat performance
    table (with a 'subgroup' column) and a dict of skipped subgroups (too
    few responders to run CV meaningfully).
    """
    seed = seed if seed is not None else mdl.SEED
    cb_cols = mdl.get_feature_columns("C", "B")
    cbk_cols = mdl.get_feature_columns("C", "B", "K")

    all_metrics = []
    skipped = {}

    levels = sorted(cohort_eval[subgroup_col].dropna().unique())
    for level in levels:
        sub = cohort_eval[cohort_eval[subgroup_col] == level].copy()
        y_sub = (sub["binaryResponse"] == "CR/PR").astype(int)
        n_resp = int(y_sub.sum())

        if n_resp < MIN_RESPONDERS_PER_SUBGROUP:
            skipped[level] = {"n": len(sub), "n_responders": n_resp}
            print(f"  [Subgroup {level}] SKIPPED -- only {n_resp} responders "
                  f"(n={len(sub)} total), below MIN_RESPONDERS_PER_SUBGROUP="
                  f"{MIN_RESPONDERS_PER_SUBGROUP}.")
            continue

        print(f"  [Subgroup {level}] n={len(sub)}, responders={n_resp} "
              f"({100 * n_resp / len(sub):.1f}%)")

        rskf = RepeatedStratifiedKFold(
            n_splits=mdl.N_OUTER_SPLITS, n_repeats=mdl.N_REPEATS, random_state=seed
        )
        splits = list(rskf.split(np.zeros(len(y_sub)), y_sub))

        for block_label, cols in [("C+B", cb_cols), ("C+B+K", cbk_cols)]:
            oof_records = []
            for i, (train_idx, test_idx) in enumerate(splits):
                repeat_num = i // mdl.N_OUTER_SPLITS
                inner_cv = StratifiedKFold(
                    n_splits=SUBGROUP_INNER_FOLDS, shuffle=True,
                    random_state=seed + repeat_num,
                )
                pipe = mdl.build_pipeline(sub, cols)
                gs = GridSearchCV(
                    pipe, {"clf__C": mdl.C_GRID}, scoring="average_precision",
                    cv=inner_cv, n_jobs=-1,
                )
                gs.fit(sub[cols].iloc[train_idx], y_sub.iloc[train_idx])
                proba = gs.predict_proba(sub[cols].iloc[test_idx])[:, 1]
                for pos, p in zip(test_idx, proba):
                    oof_records.append({
                        "repeat": repeat_num,
                        "y_true": int(y_sub.iloc[pos]),
                        "y_pred_proba": float(p),
                    })

            oof_df = pd.DataFrame(oof_records)
            for repeat, g in oof_df.groupby("repeat"):
                all_metrics.append({
                    "subgroup": level, "block": block_label, "repeat": repeat,
                    "n_subgroup": len(sub), "n_responders": n_resp,
                    "AUPRC": average_precision_score(g["y_true"], g["y_pred_proba"]),
                    "AUROC": roc_auc_score(g["y_true"], g["y_pred_proba"]),
                    "Brier": brier_score_loss(g["y_true"], g["y_pred_proba"]),
                })
            print(f"    [{level}] {block_label} done.")

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df, skipped


def compute_subgroup_deltas(metrics_df):
    """One row per (subgroup, repeat), with delta_AUPRC = CBK - CB."""
    wide = metrics_df.pivot_table(
        index=["subgroup", "repeat", "n_subgroup", "n_responders"],
        columns="block", values="AUPRC",
    ).reset_index()
    wide["delta_AUPRC"] = wide["C+B+K"] - wide["C+B"]
    wide = wide.rename(columns={"C+B": "AUPRC_CB", "C+B+K": "AUPRC_CBK"})
    return wide
