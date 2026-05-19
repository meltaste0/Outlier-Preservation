"""
CTGAN training script for Heart Disease (BRFSS) dataset.
Loads heart_disease_cleaned.csv, trains CTGAN, saves synthetic output.
Target column is 'HeartDiseaseorAttack' (0/1).
"""

from pathlib import Path
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


# Categorical columns (as per your cleaned CSV)
CATEGORICAL_COLS = [
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
    "HeartDiseaseorAttack",   # target
]

TARGET_COL = "HeartDiseaseorAttack"


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    data_path = project_root / "Datasets" / "Heart Disease" / "heart_disease_cleaned.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"Loading: {data_path}")
    data = pd.read_csv(data_path)
    
    # Convert categorical columns to string (required by SDV)
    for col in CATEGORICAL_COLS:
        if col in data.columns:
            data[col] = data[col].astype(str)
    
    print(f"Data shape: {data.shape}")
    print(f"Target distribution:\n{data[TARGET_COL].value_counts()}")
    
    # Detect metadata
    metadata = Metadata.detect_from_dataframe(data)
    
    # Train CTGAN
    print("\nTraining CTGAN...")
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=50,
        batch_size=500,
        verbose=True,
    )
    synthesizer.fit(data)
    
    # Generate synthetic data (same number of rows as original)
    print("\nGenerating synthetic samples...")
    synthetic_data = synthesizer.sample(num_rows=len(data))
    
    # Save output
    output_path = script_dir / "Fake_Datasets" / "Heart_Disease" / "synthetic_heart_disease_ctgan.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(output_path, index=False)
    
    print(f"\nSynthetic data saved: {output_path}")
    print(f"  Rows: {len(synthetic_data)}")


if __name__ == "__main__":
    main()