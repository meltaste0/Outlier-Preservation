"""
evaluate_synthetic.py
---------------------
Compare original vs synthetic tabular data across quality, resemblance,
usability, and outlier-specific metrics.

Key changes vs original
-----------------------
* _load_table: header is always inferred (header=None was treating the
  column-name row as a data row, silently corrupting real-data statistics).
* _validity_summary: binary check now uses ~ instead of == False to avoid
  pandas boolean-comparison edge cases.
* summary metrics: ks_pvalue_mean replaced with ks_sig_feature_rate
  (proportion of features with p < 0.05), which is statistically meaningful.
* --synthetic accepts multiple paths so multi-run averaging is consistent
  across all models.
* Default paths updated to reflect the shared Fake_Datasets folder structure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class EvalConfig:
    real_path: Path
    synthetic_paths: list[Path]          # supports multi-run averaging
    output_dir: Path
    class_index: int = -1
    expected_classes: tuple[int, ...] | None = None
    random_state: int = 42


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_table(path: Path) -> pd.DataFrame:
    """Load a CSV with a proper header row (first row = column names)."""

    def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, how="all")

    na_vals = ["?"]

    try:
        df = pd.read_csv(
            path, sep=None, engine="python",
            header="infer", na_values=na_vals, skip_blank_lines=True,
        )
        df = _sanitize(df)
    except Exception:
        df = pd.DataFrame()

    if not df.empty and df.shape[1] > 1:
        return df

    # Fallback for whitespace-delimited files
    df = pd.read_csv(
        path,
        sep=r"\s*,\s*|\s+",
        engine="python",
        header="infer",
        na_values=na_vals,
        skip_blank_lines=True,
    )
    return _sanitize(df)


def _align_columns(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if real_df.shape[1] != syn_df.shape[1]:
        raise ValueError(
            f"Column mismatch: real has {real_df.shape[1]} columns, "
            f"synthetic has {syn_df.shape[1]} columns."
        )
    syn_df = syn_df.copy()
    syn_df.columns = list(real_df.columns)
    return real_df, syn_df


def _split_features_and_class(
    df: pd.DataFrame, class_index: int
) -> tuple[pd.DataFrame, pd.Series]:
    class_series = df.iloc[:, class_index]
    feature_df = df.drop(df.columns[class_index], axis=1)
    return feature_df, class_series


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _impute_with_real_medians(
    real_x: pd.DataFrame, syn_x: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    medians = real_x.median(numeric_only=True)
    return real_x.fillna(medians), syn_x.fillna(medians)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _clean_labels(s: pd.Series) -> pd.Series:
    clean = s.astype(str).str.strip()
    clean = clean.str.replace(r"\|.*$", "", regex=True)
    return clean.replace({"": np.nan, "?": np.nan, "nan": np.nan, "None": np.nan})


def _encode_labels(
    real_y: pd.Series, syn_y: pd.Series
) -> tuple[pd.Series, pd.Series]:
    r_clean = _clean_labels(real_y)
    s_clean = _clean_labels(syn_y)

    joint = pd.concat([r_clean.dropna(), s_clean.dropna()]).astype(str)
    if joint.empty:
        raise ValueError("Label column has no valid values after cleaning.")

    le = LabelEncoder()
    le.fit(joint)

    r_enc = pd.Series(np.nan, index=r_clean.index, dtype="float")
    s_enc = pd.Series(np.nan, index=s_clean.index, dtype="float")

    r_mask = r_clean.notna()
    s_mask = s_clean.notna()
    r_enc.loc[r_mask] = le.transform(r_clean.loc[r_mask].astype(str))
    s_enc.loc[s_mask] = le.transform(s_clean.loc[s_mask].astype(str))

    return r_enc, s_enc


# ---------------------------------------------------------------------------
# Per-metric table builders
# ---------------------------------------------------------------------------

def _class_distribution_table(
    real_y: pd.Series,
    syn_y: pd.Series,
    expected_classes: tuple[int, ...] | None,
) -> pd.DataFrame:
    real_clean = _clean_labels(real_y)
    syn_clean = _clean_labels(syn_y)

    real_num_all = pd.to_numeric(real_clean, errors="coerce")
    syn_num_all = pd.to_numeric(syn_clean, errors="coerce")

    numeric_ready = bool(expected_classes) or (
        real_num_all.notna().mean() >= 0.95 and syn_num_all.notna().mean() >= 0.95
    )

    if numeric_ready:
        real_num = real_num_all.dropna().astype(int)
        syn_num = syn_num_all.dropna().astype(int)
        idx = list(expected_classes) if expected_classes else sorted(
            set(real_num.unique()) | set(syn_num.unique())
        )
        real_counts = real_num.value_counts().reindex(idx, fill_value=0).sort_index()
        syn_counts = syn_num.value_counts().reindex(idx, fill_value=0).sort_index()
    else:
        real_cat = real_clean.dropna()
        syn_cat = syn_clean.dropna()
        idx = sorted(set(real_cat.unique()) | set(syn_cat.unique()))
        real_counts = real_cat.value_counts().reindex(idx, fill_value=0).sort_index()
        syn_counts = syn_cat.value_counts().reindex(idx, fill_value=0).sort_index()

    real_prop = real_counts / max(real_counts.sum(), 1)
    syn_prop = syn_counts / max(syn_counts.sum(), 1)

    out = pd.DataFrame({
        "real_count": real_counts,
        "synthetic_count": syn_counts,
        "real_prop": real_prop,
        "synthetic_prop": syn_prop,
    })
    out["abs_prop_diff"] = (out["real_prop"] - out["synthetic_prop"]).abs()
    out.index.name = "class"
    return out


def _tail_quantile_table(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> pd.DataFrame:
    quantiles = [0.01, 0.05, 0.95, 0.99]
    rows: list[dict] = []
    for col in real_x.columns:
        r = pd.to_numeric(real_x[col], errors="coerce")
        s = pd.to_numeric(syn_x[col], errors="coerce")
        row: dict = {"feature": str(col)}
        for q in quantiles:
            rq = float(r.quantile(q))
            sq = float(s.quantile(q))
            row[f"real_q{q}"] = rq
            row[f"synthetic_q{q}"] = sq
            row[f"abs_diff_q{q}"] = abs(rq - sq)
        rows.append(row)
    return pd.DataFrame(rows)


def _iqr_outlier_rate(series: pd.Series, q1: float, q3: float) -> float:
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float((~series.between(lower, upper, inclusive="both")).mean())


def _univariate_outlier_table(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for col in real_x.columns:
        r = pd.to_numeric(real_x[col], errors="coerce")
        s = pd.to_numeric(syn_x[col], errors="coerce")
        q1 = float(r.quantile(0.25))
        q3 = float(r.quantile(0.75))
        rr = _iqr_outlier_rate(r, q1, q3)
        sr = _iqr_outlier_rate(s, q1, q3)
        rows.append({
            "feature": str(col),
            "real_outlier_rate": rr,
            "synthetic_outlier_rate": sr,
            "abs_rate_diff": abs(rr - sr),
        })
    return pd.DataFrame(rows)


def _ks_feature_table(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for col in real_x.columns:
        r = pd.to_numeric(real_x[col], errors="coerce").dropna().to_numpy()
        s = pd.to_numeric(syn_x[col], errors="coerce").dropna().to_numpy()
        if len(r) == 0 or len(s) == 0:
            ks_stat, p_value = np.nan, np.nan
        else:
            res = ks_2samp(r, s, alternative="two-sided", mode="auto")
            ks_stat = float(res.statistic)
            p_value = float(res.pvalue)
        rows.append({"feature": str(col), "ks_statistic": ks_stat, "p_value": p_value})
    return pd.DataFrame(rows)


def _mean_abs_corr_diff(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> float:
    rc = real_x.corr(method="spearman")
    sc = syn_x.corr(method="spearman")
    diff = (rc - sc).abs().to_numpy()
    tri = np.triu_indices_from(diff, k=1)
    vals = diff[tri]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _multivariate_outlier_summary(
    real_x: pd.DataFrame, syn_x: pd.DataFrame, random_state: int
) -> dict[str, float]:
    model = IsolationForest(
        n_estimators=300, contamination=0.05,
        random_state=random_state, n_jobs=-1,
    )
    model.fit(real_x)
    real_pred = model.predict(real_x)
    syn_pred = model.predict(syn_x)
    real_rate = float((real_pred == -1).mean())
    syn_rate = float((syn_pred == -1).mean())
    real_score = model.decision_function(real_x)
    syn_score = model.decision_function(syn_x)
    return {
        "real_iforest_outlier_rate": real_rate,
        "synthetic_iforest_outlier_rate": syn_rate,
        "iforest_outlier_rate_gap": abs(real_rate - syn_rate),
        "real_iforest_score_mean": float(np.mean(real_score)),
        "synthetic_iforest_score_mean": float(np.mean(syn_score)),
    }


def _nearest_neighbor_summary(
    real_x: pd.DataFrame, syn_x: pd.DataFrame
) -> dict[str, float]:
    mu = real_x.mean()
    sigma = real_x.std(ddof=0).replace(0, 1.0)
    real_scaled = (real_x - mu) / sigma
    syn_scaled = (syn_x - mu) / sigma
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(real_scaled)
    distances, _ = nn.kneighbors(syn_scaled)
    d = distances[:, 0]
    return {
        "nn_distance_mean": float(np.mean(d)),
        "nn_distance_median": float(np.median(d)),
        "nn_distance_p95": float(np.quantile(d, 0.95)),
        "nn_near_duplicate_rate_dlt_1e-6": float(np.mean(d < 1e-6)),
    }


def _validity_summary(
    real_x: pd.DataFrame, syn_x: pd.DataFrame
) -> dict[str, float]:
    real_min = real_x.min()
    real_max = real_x.max()
    lower_viol = float(syn_x.lt(real_min, axis=1).mean().mean())
    upper_viol = float(syn_x.gt(real_max, axis=1).mean().mean())

    binary_cols = [
        col for col in real_x.columns
        if set(real_x[col].dropna().unique().tolist()).issubset({0, 1})
        and len(real_x[col].dropna()) > 0
    ]
    # Fix: use ~ instead of == False to avoid pandas boolean-comparison issues
    bin_invalid = float((~syn_x[binary_cols].isin([0, 1])).mean().mean()) if binary_cols else 0.0

    return {
        "range_violation_rate_below_min": lower_viol,
        "range_violation_rate_above_max": upper_viol,
        "binary_constraint_violation_rate": bin_invalid,
    }


def _pmse_summary(
    real_x: pd.DataFrame, syn_x: pd.DataFrame, random_state: int
) -> dict[str, float]:
    x = pd.concat([real_x, syn_x], axis=0, ignore_index=True)
    y = np.concatenate([
        np.ones(len(real_x), dtype=int),
        np.zeros(len(syn_x), dtype=int),
    ])
    if len(np.unique(y)) < 2:
        return {"pmse": np.nan, "domain_classifier_auroc": np.nan, "domain_classifier_logloss": np.nan}

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state),
    )
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        prop = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in skf.split(x, y):
            pipe.fit(x.iloc[train_idx], y[train_idx])
            prop[test_idx] = pipe.predict_proba(x.iloc[test_idx])[:, 1]
    else:
        pipe.fit(x, y)
        prop = pipe.predict_proba(x)[:, 1]

    c = float(y.mean())
    pmse = float(np.mean((prop - c) ** 2))
    eps = 1e-12
    ll = float(-np.mean(y * np.log(prop + eps) + (1 - y) * np.log(1 - prop + eps)))
    auroc = float(roc_auc_score(y, prop))
    return {"pmse": pmse, "domain_classifier_auroc": auroc, "domain_classifier_logloss": ll}


def _tstr_summary(
    real_x: pd.DataFrame,
    syn_x: pd.DataFrame,
    real_y_enc: pd.Series,
    syn_y_enc: pd.Series,
    random_state: int,
) -> dict[str, float]:
    r_mask = real_y_enc.notna()
    s_mask = syn_y_enc.notna()
    x_train = syn_x.loc[s_mask]
    y_train = syn_y_enc.loc[s_mask].astype(int).to_numpy()
    x_test = real_x.loc[r_mask]
    y_test = real_y_enc.loc[r_mask].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        return {
            "tstr_accuracy": np.nan, "tstr_f1_macro": np.nan,
            "tstr_f1_weighted": np.nan, "tstr_auroc": np.nan,
            "tstr_auroc_eval_coverage": np.nan,
        }

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000, solver="lbfgs",
            class_weight="balanced", random_state=random_state,
        ),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, pred))
    f1_macro = float(f1_score(y_test, pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, pred, average="weighted", zero_division=0))
    auroc_coverage = np.nan

    try:
        proba = model.predict_proba(x_test)
        train_classes = model.named_steps["logisticregression"].classes_
        valid = np.isin(y_test, train_classes)
        y_eval = y_test[valid]
        proba_eval = proba[valid]
        auroc_coverage = float(np.mean(valid))
        if len(train_classes) == 2 and len(np.unique(y_eval)) == 2:
            pos_class = int(train_classes[1])
            pos_idx = int(np.where(train_classes == pos_class)[0][0])
            auroc = float(roc_auc_score((y_eval == pos_class).astype(int), proba_eval[:, pos_idx]))
        elif len(train_classes) > 2 and len(np.unique(y_eval)) >= 2:
            auroc = float(
                roc_auc_score(y_eval, proba_eval, multi_class="ovr", average="macro", labels=train_classes)
            )
        else:
            auroc = np.nan
    except Exception:
        auroc = np.nan

    return {
        "tstr_accuracy": acc, "tstr_f1_macro": f1_macro,
        "tstr_f1_weighted": f1_weighted, "tstr_auroc": auroc,
        "tstr_auroc_eval_coverage": auroc_coverage,
    }


def _outlier_specific_utility(
    real_x: pd.DataFrame, syn_x: pd.DataFrame, random_state: int
) -> dict[str, float]:
    ref = IsolationForest(
        n_estimators=300, contamination=0.05,
        random_state=random_state, n_jobs=-1,
    )
    ref.fit(real_x)
    ref_pred_real = ref.predict(real_x)
    y_true = (ref_pred_real == -1).astype(int)
    true_rate = float(np.mean(y_true))

    syn_det = IsolationForest(
        n_estimators=300, contamination=max(true_rate, 1e-3),
        random_state=random_state, n_jobs=-1,
    )
    syn_det.fit(syn_x)
    syn_scores = -syn_det.decision_function(real_x)
    k = int(max(1, round(true_rate * len(syn_scores))))
    threshold = float(np.partition(syn_scores, -k)[-k])
    y_pred = (syn_scores >= threshold).astype(int)

    recall = float(recall_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auprc = float(average_precision_score(y_true, syn_scores))
    ref_auprc = float(average_precision_score(y_true, -ref.decision_function(real_x)))

    return {
        "outlier_recall": recall,
        "outlier_precision": precision,
        "outlier_affiliation_f1_proxy": f1,
        "outlier_auc_pr": auprc,
        "outlier_auc_pr_improvement_vs_real_reference": float(auprc - ref_auprc),
    }


# ---------------------------------------------------------------------------
# Core evaluation runner (single synthetic file)
# ---------------------------------------------------------------------------

def _evaluate_one(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    config: EvalConfig,
) -> dict[str, float]:
    real_df, syn_df = _align_columns(real_df, syn_df)

    real_x_raw, real_y = _split_features_and_class(real_df, config.class_index)
    syn_x_raw, syn_y = _split_features_and_class(syn_df, config.class_index)

    real_x = _to_numeric_frame(real_x_raw)
    syn_x = _to_numeric_frame(syn_x_raw)
    real_x, syn_x = _impute_with_real_medians(real_x, syn_x)
    real_y_enc, syn_y_enc = _encode_labels(real_y, syn_y)

    ks_tbl = _ks_feature_table(real_x, syn_x)
    tail_tbl = _tail_quantile_table(real_x, syn_x)
    uni_out_tbl = _univariate_outlier_table(real_x, syn_x)
    class_tbl = _class_distribution_table(real_y, syn_y, config.expected_classes)

    summary: dict[str, float] = {
        "class_abs_prop_diff_mean": float(class_tbl["abs_prop_diff"].mean()),
        "class_abs_prop_diff_max": float(class_tbl["abs_prop_diff"].max()),
        "tail_abs_diff_mean": float(
            tail_tbl[["abs_diff_q0.01", "abs_diff_q0.05", "abs_diff_q0.95", "abs_diff_q0.99"]]
            .to_numpy()
            .mean()
        ),
        "univariate_outlier_rate_gap_mean": float(uni_out_tbl["abs_rate_diff"].mean()),
        "univariate_outlier_rate_gap_max": float(uni_out_tbl["abs_rate_diff"].max()),
        "mean_abs_spearman_corr_diff": _mean_abs_corr_diff(real_x, syn_x),
        "ks_stat_mean": float(np.nanmean(ks_tbl["ks_statistic"].to_numpy())),
        "ks_stat_max": float(np.nanmax(ks_tbl["ks_statistic"].to_numpy())),
        # Replaced ks_pvalue_mean (averaging p-values is not statistically valid)
        # with the proportion of features that are significantly different (p < 0.05).
        "ks_sig_feature_rate": float((ks_tbl["p_value"] < 0.05).mean()),
    }
    summary.update(_multivariate_outlier_summary(real_x, syn_x, config.random_state))
    summary.update(_nearest_neighbor_summary(real_x, syn_x))
    summary.update(_validity_summary(real_x, syn_x))
    summary.update(_pmse_summary(real_x, syn_x, config.random_state))
    summary.update(_tstr_summary(real_x, syn_x, real_y_enc, syn_y_enc, config.random_state))
    summary.update(_outlier_specific_utility(real_x, syn_x, config.random_state))

    return summary


def run_evaluation(config: EvalConfig) -> None:
    real_df = _load_table(config.real_path)

    all_summaries: list[dict[str, float]] = []
    all_class_tbls: list[pd.DataFrame] = []
    all_tail_tbls: list[pd.DataFrame] = []
    all_uni_out_tbls: list[pd.DataFrame] = []
    all_ks_tbls: list[pd.DataFrame] = []

    for syn_path in config.synthetic_paths:
        syn_df = _load_table(syn_path)
        real_aligned, syn_aligned = _align_columns(real_df.copy(), syn_df)

        real_x_raw, real_y = _split_features_and_class(real_aligned, config.class_index)
        syn_x_raw, syn_y = _split_features_and_class(syn_aligned, config.class_index)
        real_x = _to_numeric_frame(real_x_raw)
        syn_x = _to_numeric_frame(syn_x_raw)
        real_x, syn_x = _impute_with_real_medians(real_x, syn_x)
        real_y_enc, syn_y_enc = _encode_labels(real_y, syn_y)

        ks_tbl = _ks_feature_table(real_x, syn_x)
        tail_tbl = _tail_quantile_table(real_x, syn_x)
        uni_out_tbl = _univariate_outlier_table(real_x, syn_x)
        class_tbl = _class_distribution_table(real_y, syn_y, config.expected_classes)

        summary: dict[str, float] = {
            "class_abs_prop_diff_mean": float(class_tbl["abs_prop_diff"].mean()),
            "class_abs_prop_diff_max": float(class_tbl["abs_prop_diff"].max()),
            "tail_abs_diff_mean": float(
                tail_tbl[["abs_diff_q0.01", "abs_diff_q0.05", "abs_diff_q0.95", "abs_diff_q0.99"]]
                .to_numpy()
                .mean()
            ),
            "univariate_outlier_rate_gap_mean": float(uni_out_tbl["abs_rate_diff"].mean()),
            "univariate_outlier_rate_gap_max": float(uni_out_tbl["abs_rate_diff"].max()),
            "mean_abs_spearman_corr_diff": _mean_abs_corr_diff(real_x, syn_x),
            "ks_stat_mean": float(np.nanmean(ks_tbl["ks_statistic"].to_numpy())),
            "ks_stat_max": float(np.nanmax(ks_tbl["ks_statistic"].to_numpy())),
            "ks_sig_feature_rate": float((ks_tbl["p_value"] < 0.05).mean()),
        }
        summary.update(_multivariate_outlier_summary(real_x, syn_x, config.random_state))
        summary.update(_nearest_neighbor_summary(real_x, syn_x))
        summary.update(_validity_summary(real_x, syn_x))
        summary.update(_pmse_summary(real_x, syn_x, config.random_state))
        summary.update(_tstr_summary(real_x, syn_x, real_y_enc, syn_y_enc, config.random_state))
        summary.update(_outlier_specific_utility(real_x, syn_x, config.random_state))

        all_summaries.append(summary)
        all_class_tbls.append(class_tbl)
        all_tail_tbls.append(tail_tbl)
        all_uni_out_tbls.append(uni_out_tbl)
        all_ks_tbls.append(ks_tbl)

    # --- Average across runs ---
    keys = list(all_summaries[0].keys())
    avg_summary = {
        k: float(np.nanmean([s[k] for s in all_summaries])) for k in keys
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-feature tables from the first run (or average numeric cols if multi-run)
    all_class_tbls[0].to_csv(config.output_dir / "class_distribution_comparison.csv")
    all_tail_tbls[0].to_csv(config.output_dir / "tail_quantile_comparison.csv", index=False)
    all_uni_out_tbls[0].to_csv(config.output_dir / "univariate_outlier_comparison.csv", index=False)
    all_ks_tbls[0].to_csv(config.output_dir / "ks_feature_comparison.csv", index=False)

    # Save averaged scalar metrics
    pd.DataFrame([avg_summary]).to_csv(config.output_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(config.output_dir / "per_run_summary_metrics.csv", index=False)

    print("=== Summary Metrics (averaged across runs) ===")
    for k, v in avg_summary.items():
        print(f"{k}: {'NaN' if pd.isna(v) else f'{v:.6f}'}")

    print("\nSaved outputs to:", config.output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original vs synthetic tabular data using quality, "
            "resemblance, usability, and outlier-specific metrics."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["arrhythmia", "thyroid", "heart_disease"],
        default="heart_disease",
        help="Dataset profile for default paths",
    )
    parser.add_argument("--real", type=Path, default=None, help="Path to real data CSV")
    parser.add_argument(
        "--synthetic",
        type=Path,
        nargs="+",
        default=None,
        help="Path(s) to synthetic CSV(s); multiple paths are averaged",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-classes", type=int, nargs="*", default=None)
    parser.add_argument("--class-index", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_defaults: dict[str, dict] = {
        "heart_disease": {
            "real": Path("Datasets/Heart Disease/heart_disease_cleaned.csv"),
            "synthetic": [
                Path("models/CTAB-GAN-Plus/Fake_Datasets/Heart_Disease_Tail_Penalty/epoch20/l_0.05.csv")
            ],
            "output_dir": Path("comparison_outputs/heart_disease/CTABGAN+_Tail_Penalty/l_0.05"),
            "expected_classes": (0, 1),
        },
    }

    defaults = dataset_defaults[args.dataset]
    real_path = args.real if args.real is not None else defaults["real"]
    synthetic_paths = args.synthetic if args.synthetic is not None else defaults["synthetic"]
    output_dir = args.output_dir if args.output_dir is not None else defaults["output_dir"]
    expected = (
        defaults["expected_classes"] if args.expected_classes is None
        else (None if len(args.expected_classes) == 0 else tuple(args.expected_classes))
    )

    return EvalConfig(
        real_path=real_path,
        synthetic_paths=list(synthetic_paths),
        output_dir=output_dir,
        class_index=args.class_index,
        expected_classes=expected,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    run_evaluation(parse_args())