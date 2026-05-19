"""
preprocessing file for 
"""

from pathlib import Path

import pandas as pd

LABEL_COL = "HeartDiseaseorAttack"

CATEGORICAL_COLUMNS: list[str] = [
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

INTEGER_COLUMNS: list[str] = [
    "BMI",
    "MentHlth",
    "PhysHlth",
]

DATASET_FILENAME = "heart_disease_health_indicators_BRFSS2015.csv"


def find_raw_dataset(script_path: Path) -> Path:
    """Search the project tree for the raw CSV and return its path."""
    project_root = script_path.resolve().parent.parent
    candidates = [
        project_root / "Datasets" / "Heart Disease" / DATASET_FILENAME,
        project_root / "Heart Disease" / DATASET_FILENAME,
        project_root / DATASET_FILENAME,
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"Could not locate '{DATASET_FILENAME}' in any of: {candidates}"
        )
    return found


def prepare_heart_disease(
    source_csv: Path,
    target_csv: Path,
    max_rows: int = 0,
    seed: int = 42,
    cast_categoricals_to_str: bool = False,
) -> pd.DataFrame:
    """
    Load, reorder (label last), optionally cap rows, and save.

    Parameters
    ----------
    source_csv:
        Path to the raw BRFSS 2015 CSV.
    target_csv:
        Destination path for the cleaned CSV written to disk.
    max_rows:
        If > 0 and the dataset is larger, sample this many rows.
    seed:
        Random seed used for sampling.
    cast_categoricals_to_str:
        Set True for SDV-based models (CTGAN) which require string
        dtype for categorical columns; leave False for CTABGAN+.

    Returns
    -------
    The prepared DataFrame (identical to what is written to target_csv).
    """
    df = pd.read_csv(source_csv)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found in {source_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    # Put label last — required by CTABGAN+ evaluation helpers
    feature_cols = [c for c in df.columns if c != LABEL_COL]
    df = df[feature_cols + [LABEL_COL]]

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    if cast_categoricals_to_str:
        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(str)

    target_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_csv, index=False)

    return df