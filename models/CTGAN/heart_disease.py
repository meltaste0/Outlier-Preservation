# from pathlib import Path
# import argparse

# import pandas as pd
# from sdv.metadata import Metadata
# from sdv.single_table import CTGANSynthesizer


# def load_heart_disease(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
# 	df = pd.read_csv(path)
# 	label = "HeartDiseaseorAttack"
# 	feature_cols = [col for col in df.columns if col != label]
# 	df = df[feature_cols + [label]]

# 	if max_rows > 0 and len(df) > max_rows:
# 		df = df.sample(n=max_rows, random_state=seed)

# 	categorical_columns = [
# 		"HighBP",
# 		"HighChol",
# 		"CholCheck",
# 		"Smoker",
# 		"Stroke",
# 		"Diabetes",
# 		"PhysActivity",
# 		"Fruits",
# 		"Veggies",
# 		"HvyAlcoholConsump",
# 		"AnyHealthcare",
# 		"NoDocbcCost",
# 		"GenHlth",
# 		"DiffWalk",
# 		"Sex",
# 		"Age",
# 		"Education",
# 		"Income",
# 		"HeartDiseaseorAttack",
# 	]

# 	for col in categorical_columns:
# 		if col in df.columns:
# 			df[col] = df[col].astype(str)

# 	return df


# def main() -> None:
# 	parser = argparse.ArgumentParser(description="Run CTGAN on Heart Disease dataset.")
# 	parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
# 	parser.add_argument("--max-rows", type=int, default=20000, help="Row cap for faster runs; 0 for full data")
# 	parser.add_argument("--seed", type=int, default=42, help="Random seed")
# 	parser.add_argument(
# 		"--output",
# 		type=Path,
# 		default=Path("synthetic_heart_disease_ctgan.csv"),
# 		help="Path to synthetic output CSV",
# 	)
# 	args = parser.parse_args()

# 	workspace_root = Path(__file__).resolve().parents[2]
# 	source_path = workspace_root / "Datasets" / "Heart Disease" / "heart_disease_health_indicators_BRFSS2015.csv"

# 	data = load_heart_disease(source_path, max_rows=args.max_rows, seed=args.seed)
# 	print(f"CTGAN config -> rows={len(data)}, epochs={args.epochs}")

# 	metadata = Metadata.detect_from_dataframe(data)
# 	synthesizer = CTGANSynthesizer(
# 		metadata,
# 		epochs=args.epochs,
# 		enforce_rounding=True,
# 		verbose=True,
# 	)
# 	synthesizer.fit(data)

# 	synthetic_data = synthesizer.sample(num_rows=len(data))
# 	args.output.parent.mkdir(parents=True, exist_ok=True)
# 	synthetic_data.to_csv(args.output, index=False)
# 	print(f"Saved synthetic data to: {args.output}")


# if __name__ == "__main__":
# 	main()
"""
ctgan_heart_disease.py
----------------------
Train CTGAN (via SDV) on the Heart Disease dataset and save synthetic samples.

Key changes vs original
-----------------------
* Shared preprocessing via preprocessing.py — identical column handling
  across all models.
* Explicit random seeds (Python, NumPy, PyTorch) for reproducibility.
* Train/test split (test_ratio=0.20) matching CTABGAN+ so all models
  train on the same 80 % of the data.
* Consistent output folder structure (Fake_Datasets/Heart_Disease/).
* batch_size exposed as a CLI argument.
"""

from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import torch
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sklearn.model_selection import train_test_split

from preprocessing import (
    CATEGORICAL_COLUMNS,
    LABEL_COL,
    find_raw_dataset,
    prepare_heart_disease,
)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTGAN on Heart Disease dataset.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size (default: 500)")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=20000,
        help="Row cap for faster runs; 0 for full dataset (default: 20000)",
    )
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Held-out fraction (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to synthetic output CSV (default: Fake_Datasets/Heart_Disease/synthetic_heart_disease_ctgan.csv)",
    )
    args = parser.parse_args()

    set_global_seeds(args.seed)

    script_dir = Path(__file__).resolve().parent
    source_path = find_raw_dataset(script_dir)

    real_path = script_dir / "Real_Datasets" / "Heart_Disease.csv"
    output_path = args.output or (
        script_dir / "Fake_Datasets" / "Heart_Disease" / "synthetic_heart_disease_ctgan.csv"
    )

    # --- Load & prepare (label last, cast categoricals to str for SDV) ---
    data = prepare_heart_disease(
        source_csv=source_path,
        target_csv=real_path,
        max_rows=args.max_rows,
        seed=args.seed,
        cast_categoricals_to_str=True,
    )

    # --- Train/test split — consistent with CTABGAN+ (test_ratio=0.20) ---
    train_data, _ = train_test_split(
        data,
        test_size=args.test_ratio,
        random_state=args.seed,
        stratify=data[LABEL_COL],
    )
    train_data = train_data.reset_index(drop=True)

    print(
        f"CTGAN config -> total rows={len(data)}, train rows={len(train_data)}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}"
    )

    metadata = Metadata.detect_from_dataframe(train_data)

    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=args.epochs,
        batch_size=args.batch_size,
        enforce_rounding=True,
        verbose=True,
    )
    synthesizer.fit(train_data)

    # Generate as many rows as the full dataset for a fair comparison
    synthetic_data = synthesizer.sample(num_rows=len(data))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(output_path, index=False)
    print(f"Saved synthetic data to: {output_path}")


if __name__ == "__main__":
    main()