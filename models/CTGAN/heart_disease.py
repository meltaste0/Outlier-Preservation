from pathlib import Path
import argparse

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer


def load_heart_disease(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
	df = pd.read_csv(path)
	label = "HeartDiseaseorAttack"
	feature_cols = [col for col in df.columns if col != label]
	df = df[feature_cols + [label]]

	if max_rows > 0 and len(df) > max_rows:
		df = df.sample(n=max_rows, random_state=seed)

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

	for col in categorical_columns:
		if col in df.columns:
			df[col] = df[col].astype(str)

	return df


def main() -> None:
	parser = argparse.ArgumentParser(description="Run CTGAN on Heart Disease dataset.")
	parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
	parser.add_argument("--max-rows", type=int, default=20000, help="Row cap for faster runs; 0 for full data")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("synthetic_heart_disease_ctgan.csv"),
		help="Path to synthetic output CSV",
	)
	args = parser.parse_args()

	workspace_root = Path(__file__).resolve().parents[2]
	source_path = workspace_root / "Datasets" / "Heart Disease" / "heart_disease_health_indicators_BRFSS2015.csv"

	data = load_heart_disease(source_path, max_rows=args.max_rows, seed=args.seed)
	print(f"CTGAN config -> rows={len(data)}, epochs={args.epochs}")

	metadata = Metadata.detect_from_dataframe(data)
	synthesizer = CTGANSynthesizer(
		metadata,
		epochs=args.epochs,
		enforce_rounding=True,
		verbose=True,
	)
	synthesizer.fit(data)

	synthetic_data = synthesizer.sample(num_rows=len(data))
	args.output.parent.mkdir(parents=True, exist_ok=True)
	synthetic_data.to_csv(args.output, index=False)
	print(f"Saved synthetic data to: {args.output}")


if __name__ == "__main__":
	main()
