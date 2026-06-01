
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics


LABEL_COL = "thyroid_class"
CATEGORICAL_COLUMNS = [
    "symptom_1",
    "symptom_2",
    "symptom_3",
    "symptom_4",
    "symptom_5",
    "symptom_6",
    "symptom_7",
    "symptom_8",
    "symptom_9",
    "symptom_10",
    "symptom_11",
    "symptom_12",
    "symptom_13",
    "symptom_14",
    "symptom_15",
    LABEL_COL,
]
CONTINUOUS_COLUMNS = [
    "age",
    "TSH",
    "T3",
    "T4",
    "T3_uptake",
    "FTI",
]


def prepare_thyroid_dataset(
    source_csv: Path,
    target_csv: Path,
    max_rows: int = 0,
    seed: int = 42,
) -> None:

    #prepare thyroid dataset by reordering columns so the label is last.
    
    df = pd.read_csv(source_csv, header=None)
    
    column_names = CONTINUOUS_COLUMNS[:1] + CATEGORICAL_COLUMNS[:-1] + CONTINUOUS_COLUMNS[1:] + [LABEL_COL]
    if len(df.columns) != len(column_names):
 # if columns don't match just use sequential names
        column_names = [f"feature_{i}" for i in range(len(df.columns) - 1)] + [LABEL_COL]
    
    df.columns = column_names
    
    feature_cols = [col for col in df.columns if col != LABEL_COL]
    reordered = df[feature_cols + [LABEL_COL]]
    
    if max_rows and max_rows > 0 and len(reordered) > max_rows:
        reordered = reordered.sample(n=max_rows, random_state=seed)
    
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    reordered.to_csv(target_csv, index=False)


def run_experiment(
    num_exp: int = 1,
    epochs: int = 50,
    batch_size: int = 128,
    max_rows: int = 0,
    skip_eval: bool = False,
    privacy_data_percent: int = 5,
) -> None:
    """Train CTAB-GAN+ on thyroid data and optionally evaluate."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    # Find thyroid dataset
    candidates = [
        project_root / "Datasets" / "Thyroid" / "thyroid_binary_combined.csv",
        project_root / "Thyroid" / "thyroid_binary_combined.csv",
        project_root / "thyroid_binary_combined.csv",
    ]
    source_path = next((p for p in candidates if p.exists()), None)
    if source_path is None:
        raise FileNotFoundError(
            f"Thyroid dataset not found in any of: {[str(p) for p in candidates]}"
        )
    
    real_path = script_dir / "Real_Datasets" / "Thyroid.csv"
    fake_dir = script_dir / "Fake_Datasets" / "Thyroid_epoch50"
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    prepare_thyroid_dataset(
        source_csv=source_path,
        target_csv=real_path,
        max_rows=max_rows,
        seed=42,
    )
    
    synthesizer = CTABGAN(
        raw_csv_path=str(real_path),
        test_ratio=0.20,
        categorical_columns=CATEGORICAL_COLUMNS,
        log_columns=[],
        mixed_columns={},
        general_columns=[],
        non_categorical_columns=[],
        integer_columns=[],
        problem_type={"Classification": LABEL_COL},
    )
    
    synthesizer.synthesizer.epochs = epochs
    synthesizer.synthesizer.batch_size = batch_size
    
    total_rows = len(pd.read_csv(real_path))
    print(
        f"Config -> rows={total_rows}, epochs={epochs}, "
        f"batch_size={batch_size}, runs={num_exp}"
    )
    
    fake_paths: list[str] = []
    for i in range(num_exp):
        print(f"[CTAB-GAN+] Training run {i + 1}/{num_exp}...")
        synthesizer.fit()
        syn = synthesizer.generate_samples()
        out_path = fake_dir / f"Thyroid_fake_{i}.csv"
        syn.to_csv(out_path, index=False)
        fake_paths.append(str(out_path))
        print(f"Saved: {out_path}")
    
    if not fake_paths:
        raise RuntimeError("No synthetic CSVs were generated.")
    
    if skip_eval:
        print("Skipping utility/stat/privacy evaluation (--skip-eval enabled).")
        return
    
    # --- Utility metrics ---
    model_dict = {"Classification": ["lr", "dt", "rf", "mlp", "svm"]}
    result_mat = get_utility_metrics(str(real_path), fake_paths, "MinMax", model_dict, test_ratio=0.20)
    result_df = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
    result_df.index = list(model_dict.values())[0]
    
    stat_res_avg = [stat_sim(str(real_path), fp, CATEGORICAL_COLUMNS) for fp in fake_paths]
    stat_columns = [
        "Average WD (Continuous Columns)",
        "Average JSD (Categorical Columns)",
        "Correlation Distance",
    ]
    stat_results = pd.DataFrame(
        np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns
    )
    
    priv_res_avg = [
        privacy_metrics(str(real_path), fp, data_percent=privacy_data_percent)
        for fp in fake_paths
    ]
    privacy_columns = [
        "DCR between Real and Fake (5th perc)",
        "DCR within Real (5th perc)",
        "DCR within Fake (5th perc)",
        "NNDR between Real and Fake (5th perc)",
        "NNDR within Real (5th perc)",
        "NNDR within Fake (5th perc)",
    ]
    privacy_results = pd.DataFrame(
        np.array(priv_res_avg).mean(axis=0).reshape(1, 6), columns=privacy_columns
    )
    
    print("\nUtility metric deltas (real - synthetic):")
    print(result_df)
    print("\nStatistical similarity:")
    print(stat_results)
    print("\nPrivacy metrics:")
    print(privacy_results)
    
    outputs_dir = script_dir / "Fake_Datasets" / "Thyroid" / "metrics"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(outputs_dir / "utility_metrics.csv")
    stat_results.to_csv(outputs_dir / "stat_similarity.csv", index=False)
    privacy_results.to_csv(outputs_dir / "privacy_metrics.csv", index=False)
    print(f"\nSaved metrics under: {outputs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CTAB-GAN+ on the Thyroid dataset.")
    parser.add_argument(
        "--num-exp", type=int, default=1, help="Number of train/generate runs (default: 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Training epochs per run (default: 20)"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for faster runs; 0 uses full dataset (default: 0)",
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
