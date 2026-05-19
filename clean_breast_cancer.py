from pathlib import Path

import pandas as pd


def clean_breast_cancer_dataset(
    source_csv: Path,
    target_csv: Path,
    label_col: str = "Status",
    max_rows: int = 0,
    seed: int = 42,
) -> None:
    """
    Clean and prepare the Breast Cancer SEER dataset.
    
    - Loads the raw CSV
    - Reorders columns so the target label is last
    - Optionally caps rows for faster processing
    - Handles any missing values
    """
    df = pd.read_csv(source_csv)
    
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {source_csv}")
    
    # Reorder so label is last
    feature_cols = [col for col in df.columns if col != label_col]
    reordered = df[feature_cols + [label_col]]
    
    # Handle missing values: drop rows with NaN
    reordered = reordered.dropna()
    
    # Cap rows if specified
    if max_rows and max_rows > 0 and len(reordered) > max_rows:
        reordered = reordered.sample(n=max_rows, random_state=seed)
    
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    reordered.to_csv(target_csv, index=False)
    
    print(f"Cleaned dataset: {len(reordered)} rows, {len(reordered.columns)} columns")
    print(f"Columns: {list(reordered.columns)}")


def main() -> None:
    """Create the canonical cleaned Breast Cancer CSV used by the GAN scripts."""
    script_path = Path(__file__)
    
    # Find source dataset
    candidates = [
        script_path.parent / "Datasets" / "Breast Cancer" / "Breast_Cancer.csv",
        script_path.parent / "Breast Cancer" / "Breast_Cancer.csv",
        script_path.parent / "Breast_Cancer.csv",
    ]
    source_csv = next((p for p in candidates if p.exists()), None)
    
    if source_csv is None:
        raise FileNotFoundError(
            f"Breast Cancer dataset not found in any of: {[str(p) for p in candidates]}"
        )
    
    target_csv = script_path.parent / "Datasets" / "Breast Cancer" / "breast_cancer_cleaned.csv"
    
    clean_breast_cancer_dataset(
        source_csv=source_csv,
        target_csv=target_csv,
        label_col="A Stage",
        max_rows=0,
        seed=42,
    )
    
    print(f"Saved cleaned Breast Cancer dataset to: {target_csv}")


if __name__ == "__main__":
    main()
