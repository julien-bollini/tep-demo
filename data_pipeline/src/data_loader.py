import pandas as pd
import os
from pathlib import Path

# Dynamic root detection (tep-demo/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Environment variable for containerization with local fallback
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "tep-csv"))

def optimize_memory(df):
    """
    Optimizes DataFrame memory consumption by downcasting data types.
    """
    if 'faultNumber' in df.columns:
        df['faultNumber'] = df['faultNumber'].astype('int8')

    for col in ['simulationRun', 'sample']:
        if col in df.columns:
            df[col] = df[col].astype('int16')

    # Convert float64 to float32 for faster training and lower memory footprint
    sensor_columns = df.select_dtypes(include=['float64']).columns
    df[sensor_columns] = df[sensor_columns].astype('float32')
    return df

def load_dataset(file_name):
    """
    Loads a CSV file from the raw data directory and applies memory optimization.
    """
    target_file_path = Path(RAW_DATA_PATH) / file_name
    if not target_file_path.exists():
        raise FileNotFoundError(f"❌ Missing file: {target_file_path}")

    df = pd.read_csv(target_file_path)
    return optimize_memory(df)

def split_X_y(df, drop_metadata=True):
    """
    Separates features (X) from the target label (y).
    Optionally removes metadata columns like simulation run and sample index.
    """
    y = df['faultNumber']
    columns_to_remove = ['faultNumber']

    if drop_metadata:
        columns_to_remove.extend(['simulationRun', 'sample'])

    # Safely drop columns only if they exist in the current DataFrame
    X = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    return X, y
