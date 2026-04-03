from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors


@dataclass
class EvalConfig:
	real_path: Path
	synthetic_path: Path
	output_dir: Path
	class_index: int = -1
	expected_classes: tuple[int, ...] | None = None


def _load_table(path: Path, is_real: bool) -> pd.DataFrame:
	def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
		# Some whitespace-delimited files produce trailing all-NaN columns; remove them.
		return df.dropna(axis=1, how="all")

	na_vals = ["?"]
	header = None if is_real else "infer"

	# First attempt: infer delimiter to support comma- and whitespace-separated files.
	try:
		df = pd.read_csv(path, sep=None, engine="python", header=header, na_values=na_vals, skip_blank_lines=True)
		df = _sanitize(df)
	except Exception:
		df = pd.DataFrame()

	if not df.empty and df.shape[1] > 1:
		return df

	# Fallback: split on commas (with optional spaces) or runs of whitespace.
	df = pd.read_csv(
		path,
		sep=r"\s*,\s*|\s+",
		engine="python",
		header=header,
		na_values=na_vals,
		skip_blank_lines=True,
	)
	return _sanitize(df)


def _align_columns(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	if real_df.shape[1] != syn_df.shape[1]:
		raise ValueError(
			f"Column mismatch: real has {real_df.shape[1]} columns, synthetic has {syn_df.shape[1]} columns."
		)

	# Align synthetic columns to real indices so downstream operations are consistent.
	syn_df = syn_df.copy()
	syn_df.columns = list(real_df.columns)
	return real_df, syn_df


def _split_features_and_class(df: pd.DataFrame, class_index: int) -> tuple[pd.DataFrame, pd.Series]:
	class_series = df.iloc[:, class_index]
	feature_df = df.drop(df.columns[class_index], axis=1)
	return feature_df, class_series


def _to_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
	return df.apply(pd.to_numeric, errors="coerce")


def _impute_with_real_medians(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	medians = real_x.median(numeric_only=True)
	real_filled = real_x.fillna(medians)
	syn_filled = syn_x.fillna(medians)
	return real_filled, syn_filled


def _class_distribution_table(
	real_y: pd.Series,
	syn_y: pd.Series,
	expected_classes: tuple[int, ...] | None,
) -> pd.DataFrame:
	def _clean_labels(s: pd.Series) -> pd.Series:
		clean = s.astype(str).str.strip()
		clean = clean.str.replace(r"\|.*$", "", regex=True)
		clean = clean.replace({"": np.nan, "?": np.nan, "nan": np.nan, "None": np.nan})
		return clean

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

		if expected_classes:
			idx = list(expected_classes)
		else:
			idx = sorted(set(real_num.unique()) | set(syn_num.unique()))

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

	out = pd.DataFrame(
		{
			"real_count": real_counts,
			"synthetic_count": syn_counts,
			"real_prop": real_prop,
			"synthetic_prop": syn_prop,
		}
	)
	out["abs_prop_diff"] = (out["real_prop"] - out["synthetic_prop"]).abs()
	out.index.name = "class"
	return out


def _tail_quantile_table(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> pd.DataFrame:
	quantiles = [0.01, 0.05, 0.95, 0.99]
	rows: list[dict[str, float | str]] = []

	for col in real_x.columns:
		r = pd.to_numeric(real_x[col], errors="coerce")
		s = pd.to_numeric(syn_x[col], errors="coerce")

		row: dict[str, float | str] = {"feature": str(col)}
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
	mask = series.between(lower, upper, inclusive="both")
	return float((~mask).mean())


def _univariate_outlier_table(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, float | str]] = []
	for col in real_x.columns:
		r = pd.to_numeric(real_x[col], errors="coerce")
		s = pd.to_numeric(syn_x[col], errors="coerce")

		q1 = float(r.quantile(0.25))
		q3 = float(r.quantile(0.75))
		rr = _iqr_outlier_rate(r, q1, q3)
		sr = _iqr_outlier_rate(s, q1, q3)
		rows.append(
			{
				"feature": str(col),
				"real_outlier_rate": rr,
				"synthetic_outlier_rate": sr,
				"abs_rate_diff": abs(rr - sr),
			}
		)

	return pd.DataFrame(rows)


def _mean_abs_corr_diff(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> float:
	rc = real_x.corr(method="spearman")
	sc = syn_x.corr(method="spearman")
	diff = (rc - sc).abs().to_numpy()
	tri = np.triu_indices_from(diff, k=1)
	vals = diff[tri]
	vals = vals[~np.isnan(vals)]
	if len(vals) == 0:
		return 0.0
	return float(vals.mean())


def _multivariate_outlier_summary(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> dict[str, float]:
	model = IsolationForest(
		n_estimators=300,
		contamination=0.05,
		random_state=42,
		n_jobs=-1,
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


def _nearest_neighbor_summary(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> dict[str, float]:
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


def _validity_summary(real_x: pd.DataFrame, syn_x: pd.DataFrame) -> dict[str, float]:
	real_min = real_x.min()
	real_max = real_x.max()

	lower_viol = (syn_x.lt(real_min, axis=1)).mean().mean()
	upper_viol = (syn_x.gt(real_max, axis=1)).mean().mean()

	binary_cols = []
	for col in real_x.columns:
		uniq = set(pd.Series(real_x[col]).dropna().unique().tolist())
		if uniq.issubset({0, 1}) and len(uniq) > 0:
			binary_cols.append(col)

	if binary_cols:
		bin_invalid = (syn_x[binary_cols].isin([0, 1]) == False).mean().mean()
	else:
		bin_invalid = 0.0

	return {
		"range_violation_rate_below_min": float(lower_viol),
		"range_violation_rate_above_max": float(upper_viol),
		"binary_constraint_violation_rate": float(bin_invalid),
	}


def run_evaluation(config: EvalConfig) -> None:
	real_df = _load_table(config.real_path, is_real=True)
	syn_df = _load_table(config.synthetic_path, is_real=False)
	real_df, syn_df = _align_columns(real_df, syn_df)

	real_x_raw, real_y = _split_features_and_class(real_df, class_index=config.class_index)
	syn_x_raw, syn_y = _split_features_and_class(syn_df, class_index=config.class_index)

	real_x = _to_numeric_frame(real_x_raw)
	syn_x = _to_numeric_frame(syn_x_raw)
	real_x, syn_x = _impute_with_real_medians(real_x, syn_x)

	class_tbl = _class_distribution_table(real_y, syn_y, expected_classes=config.expected_classes)
	tail_tbl = _tail_quantile_table(real_x, syn_x)
	uni_out_tbl = _univariate_outlier_table(real_x, syn_x)

	summary = {
		"class_abs_prop_diff_mean": float(class_tbl["abs_prop_diff"].mean()),
		"class_abs_prop_diff_max": float(class_tbl["abs_prop_diff"].max()),
		"tail_abs_diff_mean": float(
			tail_tbl[["abs_diff_q0.01", "abs_diff_q0.05", "abs_diff_q0.95", "abs_diff_q0.99"]].to_numpy().mean()
		),
		"univariate_outlier_rate_gap_mean": float(uni_out_tbl["abs_rate_diff"].mean()),
		"univariate_outlier_rate_gap_max": float(uni_out_tbl["abs_rate_diff"].max()),
		"mean_abs_spearman_corr_diff": _mean_abs_corr_diff(real_x, syn_x),
	}
	summary.update(_multivariate_outlier_summary(real_x, syn_x))
	summary.update(_nearest_neighbor_summary(real_x, syn_x))
	summary.update(_validity_summary(real_x, syn_x))

	config.output_dir.mkdir(parents=True, exist_ok=True)
	class_tbl.to_csv(config.output_dir / "class_distribution_comparison.csv")
	tail_tbl.to_csv(config.output_dir / "tail_quantile_comparison.csv", index=False)
	uni_out_tbl.to_csv(config.output_dir / "univariate_outlier_comparison.csv", index=False)
	pd.DataFrame([summary]).to_csv(config.output_dir / "summary_metrics.csv", index=False)

	print("=== Summary Metrics ===")
	for k, v in summary.items():
		print(f"{k}: {v:.6f}")

	print("\nSaved outputs:")
	print(config.output_dir / "summary_metrics.csv")
	print(config.output_dir / "class_distribution_comparison.csv")
	print(config.output_dir / "tail_quantile_comparison.csv")
	print(config.output_dir / "univariate_outlier_comparison.csv")


def parse_args() -> EvalConfig:
	parser = argparse.ArgumentParser(description="Compare original vs synthetic tabular data for outlier-generation quality.")
	parser.add_argument(
		"--dataset",
		choices=["arrhythmia", "thyroid", "heart_disease"],
		default="arrhythmia",
		help="Dataset profile used for defaults (real/synthetic paths, output dir, expected classes)",
	)
	parser.add_argument("--real", type=Path, default=None, help="Path to original data file")
	parser.add_argument("--synthetic", type=Path, default=None, help="Path to synthetic data file")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory to save output metric tables",
	)
	parser.add_argument(
		"--expected-classes",
		type=int,
		nargs="*",
		default=None,
		help="Expected class labels for class-distribution comparison",
	)
	parser.add_argument("--class-index", type=int, default=-1, help="Index of class/label column")

	args = parser.parse_args()
	dataset_defaults = {
		# "arrhythmia": {
		# 	"real": Path("Arrhythmia/arrhythmia.data"),
		# 	"synthetic": Path("synthetic_arrhythmia.csv"),
		# 	"output_dir": Path("comparison_outputs/arrhythmia"),
		# 	"expected_classes": tuple(range(1, 17)),
		# },
		# "thyroid": {
		# 	"real": Path("Thyroid/ann-train.data"),
		# 	"synthetic": Path("synthetic_thyroid.csv"),
		# 	"output_dir": Path("comparison_outputs/thyroid"),
		# 	"expected_classes": (1, 2, 3),
		# },
		"heart_disease": {
			"real": Path("Datasets\\Heart Disease\\heart_disease_cleaned.csv"),
			"synthetic": Path("models\\CTGAN\\Fake_Datasets\\synthetic_heart_disease_ctgan.csv"),
			"output_dir": Path("comparison_outputs\\heart_disease\\CTGAN"),
			"expected_classes": (0, 1),
		},
	}
	defaults = dataset_defaults[args.dataset]

	real_path = args.real if args.real is not None else defaults["real"]
	synthetic_path = args.synthetic if args.synthetic is not None else defaults["synthetic"]
	output_dir = args.output_dir if args.output_dir is not None else defaults["output_dir"]

	if args.expected_classes is None:
		expected = defaults["expected_classes"]
	elif len(args.expected_classes) == 0:
		expected = None
	else:
		expected = tuple(args.expected_classes)

	return EvalConfig(
		real_path=real_path,
		synthetic_path=synthetic_path,
		output_dir=output_dir,
		class_index=args.class_index,
		expected_classes=expected,
	)


if __name__ == "__main__":
	run_evaluation(parse_args())
