"""
CTAB-GAN+ training script for Breast Cancer (SEER) dataset.
Target column: 'A Stage' (Regional vs Distant)
"""

from pathlib import Path
import pandas as pd
from model.ctabgan import CTABGAN

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
REAL_CSV = PROJECT_ROOT / "Datasets" / "Breast Cancer" / "breast_cancer_cleaned.csv"
OUTPUT_CSV = SCRIPT_DIR / "Fake_Datasets" / "Breast_Cancer" / "synthetic_breast_cancer_ctabgan.csv"

CATEGORICAL_COLS = [
    "Race", "Marital Status", "T Stage ", "N Stage", "6th Stage",
    "differentiate", "Grade", "Estrogen Status", "Progesterone Status",
    "Status", "A Stage"
]

INTEGER_COLS = [
    "Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"
]

LABEL_COL = "A Stage"


def main():
    df = pd.read_csv(REAL_CSV)

    # Verify label exists
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found. Columns are: {list(df.columns)}")

    # Reorder so label is last
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    df = df[feature_cols + [LABEL_COL]]

    # Save reordered version
    tmp_csv = SCRIPT_DIR / "Real_Datasets" / "Breast_Cancer.csv"
    tmp_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_csv, index=False)

    print(f"Training CTAB-GAN+ on Breast Cancer: rows={len(df)}, epochs=50, batch_size=128")
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
        problem_type={"Classification": LABEL_COL}
    )
    synthesizer.synthesizer.epochs = 50
    synthesizer.synthesizer.batch_size = 128


    synthesizer.fit()
    synthetic = synthesizer.generate_samples()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved synthetic data to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()