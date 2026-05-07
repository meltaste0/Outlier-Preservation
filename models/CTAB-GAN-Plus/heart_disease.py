from pathlib import Path
import argparse
import glob

import numpy as np
import pandas as pd

from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics


def prepare_heart_disease_csv(
	source_csv: Path,
	target_csv: Path,
	label_col: str,
	max_rows: int = 0,
	seed: int = 42,
) -> None:
	"""Reorder columns so the label is last (required by evaluation helpers)."""
	df = pd.read_csv(source_csv)
	if label_col not in df.columns:
		raise ValueError(f"Label column '{label_col}' not found in {source_csv}")

	feature_cols = [col for col in df.columns if col != label_col]
	reordered = df[feature_cols + [label_col]]
	if max_rows and max_rows > 0 and len(reordered) > max_rows:
		reordered = reordered.sample(n=max_rows, random_state=seed)
	target_csv.parent.mkdir(parents=True, exist_ok=True)
	reordered.to_csv(target_csv, index=False)


def run_experiment(
	num_exp: int = 1,
	epochs: int = 10,
	batch_size: int = 128,
	max_rows: int = 20000,
	skip_eval: bool = False,
	privacy_data_percent: int = 5,
) -> None:
	script_dir = Path(__file__).resolve().parent
	project_root = script_dir.parent.parent

	dataset_filename = "heart_disease_health_indicators_BRFSS2015.csv"
	candidates = [
		project_root / "Datasets" / "Heart Disease" / dataset_filename,
		project_root / "Heart Disease" / dataset_filename,
		project_root / dataset_filename,
	]
	raw_source_path = next((p for p in candidates if p.exists()), project_root / "Heart Disease" / dataset_filename)

	real_path = script_dir / "Real_Datasets" / "Heart_Disease.csv"
	fake_dir = script_dir / "Fake_Datasets" / "Heart_Disease_Tail_Penalty"
	fake_dir.mkdir(parents=True, exist_ok=True)

	label_col = "HeartDiseaseorAttack"
	prepare_heart_disease_csv(raw_source_path, real_path, label_col, max_rows=max_rows)

	categorical_columns = [
		"HighBP",
		"HighChol",
		"CholCheck",
		"Smoker",
		"Stroke",
		"Diabetes",
		"PhysActivity",
		"Fruits",
		"Veggies",
		"HvyAlcoholConsump",
		"AnyHealthcare",
		"NoDocbcCost",
		"GenHlth",
		"DiffWalk",
		"Sex",
		"Age",
		"Education",
		"Income",
		"HeartDiseaseorAttack",
	]

	integer_columns = [
		"BMI",
		"MentHlth",
		"PhysHlth",
	]

	synthesizer = CTABGAN(
		raw_csv_path=str(real_path),
		test_ratio=0.20,
		categorical_columns=categorical_columns,
		log_columns=[],
		mixed_columns={},
		general_columns=[],
		non_categorical_columns=[],
		integer_columns=integer_columns,
		problem_type={"Classification": label_col},
	)
	synthesizer.synthesizer.epochs = epochs
	synthesizer.synthesizer.batch_size = batch_size

	print(
		f"Config -> rows={len(pd.read_csv(real_path))}, epochs={epochs}, "
		f"batch_size={batch_size}, runs={num_exp}"
	)

	for i in range(num_exp):
		print(f"[CTAB-GAN+] Training run {i + 1}/{num_exp}...")
		synthesizer.fit()
		syn = synthesizer.generate_samples()
		out_path = fake_dir / f"Heart_Disease_fake_{i}.csv"
		syn.to_csv(out_path, index=False)
		print(f"Saved: {out_path}")

	fake_paths = glob.glob(str(fake_dir / "*.csv"))
	if not fake_paths:
		raise RuntimeError("No synthetic CSVs were generated.")

	if skip_eval:
		print("Skipping utility/stat/privacy evaluation (--skip-eval enabled).")
		return

	model_dict = {"Classification": ["lr", "dt", "rf", "mlp", "svm"]}
	result_mat = get_utility_metrics(str(real_path), fake_paths, "MinMax", model_dict, test_ratio=0.20)
	result_df = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
	result_df.index = list(model_dict.values())[0]

	stat_res_avg = []
	for fake_path in fake_paths:
		stat_res_avg.append(stat_sim(str(real_path), fake_path, categorical_columns))

	stat_columns = [
		"Average WD (Continuous Columns)",
		"Average JSD (Categorical Columns)",
		"Correlation Distance",
	]
	stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)

	priv_res_avg = []
	for fake_path in fake_paths:
		priv_res_avg.append(
			privacy_metrics(str(real_path), fake_path, data_percent=privacy_data_percent)
		)

	privacy_columns = [
		"DCR between Real and Fake (5th perc)",
		"DCR within Real (5th perc)",
		"DCR within Fake (5th perc)",
		"NNDR between Real and Fake (5th perc)",
		"NNDR within Real (5th perc)",
		"NNDR within Fake (5th perc)",
	]
	privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1, 6), columns=privacy_columns)

	print("\nUtility metric deltas (real - synthetic):")
	print(result_df)
	print("\nStatistical similarity:")
	print(stat_results)
	print("\nPrivacy metrics:")
	print(privacy_results)

	outputs_dir = script_dir / "Fake_Datasets" / "Heart_Disease" / "metrics"
	outputs_dir.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(outputs_dir / "utility_metrics.csv")
	stat_results.to_csv(outputs_dir / "stat_similarity.csv", index=False)
	privacy_results.to_csv(outputs_dir / "privacy_metrics.csv", index=False)
	print(f"\nSaved metrics under: {outputs_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run CTAB-GAN+ on the Heart Disease dataset.")
	parser.add_argument("--num-exp", type=int, default=1, help="Number of train/generate runs (default: 1)")
	parser.add_argument("--epochs", type=int, default=10, help="Training epochs per run (default: 10)")
	parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
	parser.add_argument(
		"--max-rows",
		type=int,
		default=20000,
		help="Optional row cap for faster runs; 0 uses full dataset (default: 20000)",
	)
	parser.add_argument(
		"--skip-eval",
		action="store_true",
		help="Skip utility/stat/privacy metrics and only train + generate samples.",
	)
	parser.add_argument(
		"--privacy-data-percent",
		type=int,
		default=5,
		help="Percent of data used inside privacy metrics (default: 5).",
	)
	args = parser.parse_args()
	run_experiment(
		num_exp=args.num_exp,
		epochs=args.epochs,
		batch_size=args.batch_size,
		max_rows=args.max_rows,
		skip_eval=args.skip_eval,
		privacy_data_percent=args.privacy_data_percent,
	)