import pandas as pd
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = os.getenv("DATA_PATH", BASE_DIR / "data" / "raw" / "tep-csv")

def optimize_memory(df):
    # Optimise la consommation mémoire du DataFrame en convertissant les types de données.
    if 'faultNumber' in df.columns:
        df['faultNumber'] = df['faultNumber'].astype('int8')

    for col in ['simulationRun', 'sample']:
        if col in df.columns:
            df[col] = df[col].astype('int16')

    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

def load_dataset(file_name):
    full_path = Path(DEFAULT_DATA_PATH) / file_name
    if not full_path.exists():
        raise FileNotFoundError(f"Fichier absent : {full_path}")

    df = pd.read_csv(full_path)
    return optimize_memory(df)

def get_X_y(df, drop_meta=True):
    y = df['faultNumber']
    cols_to_drop = ['faultNumber']
    if drop_meta:
        cols_to_drop.extend(['simulationRun', 'sample'])

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return X, y
