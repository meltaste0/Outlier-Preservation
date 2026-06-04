"""
CTGAN training script for Breast Cancer (SEER) dataset.
Loads breast_cancer_cleaned.csv, trains CTGAN, saves synthetic output.
Target column is 'A_Stage' (Regional=0, Distant=1) as the outlier class.
"""

from pathlib import Path
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


COLUMN_NAMES = [
    "Age",
    "Race",
    "Marital_Status",
    "T_Stage",
    "N_Stage",
    "6th_Stage",
    "Differentiate",
    "Grade",
    "Tumor_Size",
    "Estrogen_Status",
    "Progesterone_Status",
    "Regional_Node_Examined",
    "Regional_Node_Positive",
    "Survival_Months",
    "Status",
    "A_Stage",          # target: regional (0) / distant (1)
]

CATEGORICAL_COLS = [
    "Race",
    "Marital_Status",
    "T_Stage",
    "N_Stage",
    "6th_Stage",
    "Differentiate",
    "Grade",
    "Estrogen_Status",
    "Progesterone_Status",
    "Status",
    "A_Stage",         # target is categorical
]

NUMERIC_COLS = [
    "Age",
    "Tumor_Size",
    "Regional_Node_Examined",
    "Regional_Node_Positive",
    "Survival_Months",
]

TARGET_COL = "A_Stage"


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    data_path = project_root / "Datasets" / "Breast Cancer" / "breast_cancer_cleaned.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"Loading: {data_path}")
    data = pd.read_csv(data_path)
    
    data.columns = COLUMN_NAMES
    
    data[TARGET_COL] = data[TARGET_COL].astype(str)
    
    for col in CATEGORICAL_COLS:
        if col in data.columns:
            data[col] = data[col].astype(str)
    
    for col in NUMERIC_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna()
    
    print(f"Data shape after cleaning: {data.shape}")
    print(f"Class distribution:\n{data[TARGET_COL].value_counts()}")
    
    metadata = Metadata.detect_from_dataframe(data)
    
    print("\nTraining CTGAN...")
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=100,
        batch_size=500,
        verbose=True,
    )
    synthesizer.fit(data)
    
    print("\nGenerating synthetic samples...")
    synthetic_data = synthesizer.sample(num_rows=len(data))
    
    output_path = script_dir / "Fake_Datasets" / "Breast_Cancer" / "synthetic_breast_cancer_ctgan_100epochs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(output_path, index=False)
    
    print(f"\nSynthetic data saved: {output_path}")
    print(f"  Rows: {len(synthetic_data)}")


if __name__ == "__main__":
    main()