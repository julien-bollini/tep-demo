import pandas as pd
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "/app/data/raw/tep-csv"))

def optimize_memory(df):
    # Optimise la consommation mémoire du DataFrame en convertissant les types de données.
    if 'faultNumber' in df.columns:
        df['faultNumber'] = df['faultNumber'].astype('int8')

    for col in ['simulationRun', 'sample']:
        if col in df.columns:
            df[col] = df[col].astype('int16')

    sensor_columns = df.select_dtypes(include=['float64']).columns
    df[sensor_columns] = df[sensor_columns].astype('float32')
    return df

def load_dataset(file_name):
    target_file_path = Path(RAW_DATA_PATH) / file_name
    if not target_file_path.exists():
        raise FileNotFoundError(f"Fichier absent : {target_file_path}")

    df = pd.read_csv(target_file_path)
    return optimize_memory(df)

def split_X_y(df, drop_metadata=True):
    y = df['faultNumber']
    columns_to_remove = ['faultNumber']
    if drop_metadata:
        columns_to_remove.extend(['simulationRun', 'sample'])

    X = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
    return X, y
