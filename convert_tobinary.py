"""
Convert multi-class datasets to binary classification:
- Arrhythmia: Class 1 (normal) vs Classes 2-16 (arrhythmia)
- Thyroid: Class 3 (normal) vs Classes 1-2 (abnormal)
"""

import pandas as pd
from pathlib import Path


def convert_arrhythmia_to_binary():
    """
    Convert arrhythmia dataset from 16-class to binary classification.
    - Class 0: Normal (original class 1)
    - Class 1: Arrhythmia (original classes 2-16)
    """
    print("=" * 60)
    print("Converting Arrhythmia dataset to binary classification...")
    print("=" * 60)
    
    # Path to raw arrhythmia data
    arrhythmia_path = Path("Datasets/Arrhythmia/arrhythmia.data")
    
    if not arrhythmia_path.exists():
        print(f"Error: {arrhythmia_path} not found")
        return
    
    # Read the data (no header, all columns are numeric)
    df = pd.read_csv(arrhythmia_path, header=None)
    
    print(f"Original shape: {df.shape}")
    print(f"Original class distribution:\n{df.iloc[:, -1].value_counts().sort_index()}")
    
    # Replace '?' with NaN and drop rows with missing values
    df = df.replace('?', pd.NA).dropna()
    
    print(f"Shape after removing missing values: {df.shape}")
    
    # Convert to numeric (in case there are string values)
    # Last column is the class label
    last_col_idx = df.shape[1] - 1
    df.iloc[:, last_col_idx] = pd.to_numeric(df.iloc[:, last_col_idx], errors='coerce')
    
    # Convert to binary: Class 1 (normal) -> 0, Classes 2-16 (arrhythmia) -> 1
    binary_labels = df.iloc[:, last_col_idx].apply(lambda x: 0 if x == 1 else 1)
    df.iloc[:, last_col_idx] = binary_labels
    
    print(f"Binary class distribution:\n{df.iloc[:, -1].value_counts().sort_index()}")
    print(f"- Class 0 (Normal): {(df.iloc[:, -1] == 0).sum()}")
    print(f"- Class 1 (Arrhythmia): {(df.iloc[:, -1] == 1).sum()}")
    
    # Save to binary CSV file
    output_path = Path("Datasets/Arrhythmia/arrhythmia_binary.csv")
    df.to_csv(output_path, index=False, header=False)
    print(f"\nBinary arrhythmia data saved to: {output_path}")
    print()


def convert_thyroid_to_binary():
    """
    Convert thyroid dataset from 3-class to binary classification.
    - Class 0: Normal (original class 3)
    - Class 1: Abnormal (original classes 1-2: hyperfunction and subnormal)
    Combines train and test sets into a single binary dataset.
    """
    print("=" * 60)
    print("Converting Thyroid dataset to binary classification...")
    print("=" * 60)
    
    # Process both train and test sets
    thyroid_files = [
        ("Datasets/Thyroid/ann-train.data", "Datasets/Thyroid/ann-train_binary.csv"),
        ("Datasets/Thyroid/ann-test.data", "Datasets/Thyroid/ann-test_binary.csv"),
    ]
    
    combined_df = None
    
    for input_file, output_file in thyroid_files:
        thyroid_path = Path(input_file)
        
        if not thyroid_path.exists():
            print(f"Warning: {thyroid_path} not found, skipping...")
            continue
        
        print(f"\nProcessing: {input_file}")
        
        # Read the data (no header, space-delimited)
        df = pd.read_csv(thyroid_path, header=None, sep=r'\s+')
        
        print(f"  Original shape: {df.shape}")
        print(f"  Original class distribution:\n{df.iloc[:, -1].value_counts().sort_index()}")
        
        # Last column is the class label
        last_col_idx = df.shape[1] - 1
        
        # Convert to numeric
        df.iloc[:, last_col_idx] = pd.to_numeric(df.iloc[:, last_col_idx], errors='coerce')
        
        # Convert to binary: Class 3 (normal) -> 0, Classes 1-2 (abnormal) -> 1
        binary_labels = df.iloc[:, last_col_idx].apply(lambda x: 0 if x == 3 else 1)
        df.iloc[:, last_col_idx] = binary_labels
        
        print(f"  Binary class distribution:\n{df.iloc[:, -1].value_counts().sort_index()}")
        print(f"  - Class 0 (Normal): {(df.iloc[:, -1] == 0).sum()}")
        print(f"  - Class 1 (Abnormal): {(df.iloc[:, -1] == 1).sum()}")
        
        # Save individual binary file (comma-separated CSV)
        output_path = Path(output_file)
        df.to_csv(output_path, index=False, header=False)
        print(f"  Binary thyroid data saved to: {output_path}")
        
        # Append to combined dataframe
        if combined_df is None:
            combined_df = df.copy()
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Save combined binary file
    if combined_df is not None:
        combined_path = Path("Datasets/Thyroid/thyroid_binary_combined.csv")
        combined_df.to_csv(combined_path, index=False, header=False)
        
        print(f"\n{'=' * 60}")
        print("Combined Thyroid Binary Dataset:")
        print(f"{'=' * 60}")
        print(f"Total shape: {combined_df.shape}")
        print(f"Combined class distribution:\n{combined_df.iloc[:, -1].value_counts().sort_index()}")
        print(f"- Class 0 (Normal): {(combined_df.iloc[:, -1] == 0).sum()}")
        print(f"- Class 1 (Abnormal): {(combined_df.iloc[:, -1] == 1).sum()}")
        print(f"Combined binary thyroid data saved to: {combined_path}")
    
    print()


def main():
    """Convert both arrhythmia and thyroid datasets to binary classification."""
    print("\n" + "=" * 60)
    print("BINARY DATASET CONVERSION")
    print("=" * 60 + "\n")
    
    convert_arrhythmia_to_binary()
    convert_thyroid_to_binary()
    
    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
