"""
CTAB-GAN+ training script for Thyroid dataset.
Target column: 'thyroid_class' (Normal vs Abnormal)
"""

from pathlib import Path
import pandas as pd

from model.ctabgan import CTABGAN


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REAL_CSV = PROJECT_ROOT / "Datasets" / "Thyroid" / "thyroid_binary_combined.csv"
OUTPUT_CSV = SCRIPT_DIR / "Fake_Datasets" / "Thyroid" / "synthetic_thyroid_ctabgan.csv"

CATEGORICAL_COLS = [
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
    "thyroid_class",
]

INTEGER_COLS = ["age", "TSH", "T3", "T4", "T3_uptake", "FTI"]
LABEL_COL = "thyroid_class"


def main():
    df = pd.read_csv(REAL_CSV, header=None)

    column_names = [
        "age",
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
        "TSH",
        "T3",
        "T4",
        "T3_uptake",
        "FTI",
        LABEL_COL,
    ]

    if df.shape[1] != len(column_names):
        raise ValueError(
            f"Expected {len(column_names)} columns in thyroid CSV, got {df.shape[1]}."
        )

    df.columns = column_names

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Columns are: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != LABEL_COL]
    df = df[feature_cols + [LABEL_COL]]

    tmp_csv = SCRIPT_DIR / "Real_Datasets" / "Thyroid.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_csv, index=False)

    print(f"Training CTAB-GAN+ on Thyroid: rows={len(df)}, epochs=50, batch_size=128")
    print(f"Class distribution:\n{df[LABEL_COL].value_counts()}")

    synthesizer = CTABGAN(
        raw_csv_path=str(tmp_csv),
        test_ratio=0.20,
        categorical_columns=CATEGORICAL_COLS,
        log_columns=[],
        mixed_columns={},
        general_columns=[],
        non_categorical_columns=[],
        integer_columns=INTEGER_COLS,
        problem_type={"Classification": LABEL_COL},
    )
    synthesizer.synthesizer.epochs = 50
    synthesizer.synthesizer.batch_size = 128
    print(synthesizer.synthesizer.tail_penalty_lambda)

    synthesizer.fit()
    synthetic = synthesizer.generate_samples()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved synthetic data to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
