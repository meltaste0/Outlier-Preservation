"""
CTGAN training script for Thyroid dataset.
Simple and minimal — loads thyroid_binary_combined.csv, trains CTGAN, saves synthetic output.
"""

from pathlib import Path
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata


def main():
    # Locate dataset
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    data_path = project_root / "Datasets" / "Thyroid" / "thyroid_binary_combined.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    print(f"Loading: {data_path}")
    data = pd.read_csv(data_path, header=None)
    
    # Thyroid: 21 features + 1 target class
    feature_names = [f'Feature_{i}' for i in range(21)] + ['Class']
    data.columns = feature_names
    data['Class'] = data['Class'].astype(str)
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
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
    
    # Generate synthetic data
    print("\nGenerating synthetic samples...")
    num_real = len(data)
    synthetic_data = synthesizer.sample(num_rows=num_real)
    
    # Save output
    output_path = script_dir / "Fake_Datasets" / "Thyroid" / "synthetic_thyroid.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_data.to_csv(output_path, index=False)
    
    print(f"\nSynthetic data saved: {output_path}")
    print(f"  Rows: {len(synthetic_data)}")


if __name__ == "__main__":
    main()