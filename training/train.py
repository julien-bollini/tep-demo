import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION AND PATHS
# ==========================================
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
# Adding data_pipeline to sys.path to allow imports from src
sys.path.append(str(root_path / "data_pipeline"))

from src.data_loader import split_X_y

# Environment variables for Docker compatibility with local fallbacks
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_PATH", root_path / "data" / "processed"))
MODEL_DIR = Path(os.getenv("MODEL_PATH", root_path / "data" / "models"))
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "tep_master_cleaned.csv"

# ==========================================
# SPLIT UTILITIES
# ==========================================
def prepare_datasets(df, retention_rate=0.5, test_size=0.2):
    """
    Downsamples the data and performs a run-aware split to prevent data leakage.
    """
    # Create a unique ID for each simulation run
    df['run_id'] = df['faultNumber'].astype(str) + "_" + df['simulationRun'].astype(str)
    unique_runs = df['run_id'].unique()

    # Downsampling logic
    n_keep = int(len(unique_runs) * retention_rate)
    kept_runs = np.random.choice(unique_runs, n_keep, replace=False)
    df = df[df['run_id'].isin(kept_runs)]

    # Split logic: training on one set of runs, testing on another
    split_idx = int(len(kept_runs) * (1 - test_size))
    train_runs = kept_runs[:split_idx]

    df_train = df[df['run_id'].isin(train_runs)].drop(columns=['run_id'])
    df_test = df[~df['run_id'].isin(train_runs)].drop(columns=['run_id'])

    return df_train, df_test

# ==========================================
# TRAINING PIPELINE
# ==========================================
def run_training():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not CLEANED_DATA_FILE.exists():
        print(f"❌ Error: {CLEANED_DATA_FILE} not found.")
        return
    # Force float32 during read to keep memory footprint low
    print(f"✔️ Loading datasets")
    df = pd.read_csv(
        CLEANED_DATA_FILE,
        engine='c',
        low_memory=True
    )

    # Ensure all float columns are float32
    float_cols = df.select_dtypes(include=['float']).columns
    df[float_cols] = df[float_cols].astype('float32')

    # Apply downsampling
    df_train, df_test = prepare_datasets(df)

    # Clean up original df from memory to free space
    del df

    print(f"✔️ Training data ready. Shape: {df_train.shape}")

    # 1. Export test set for the evaluation script
    df_test.to_csv(PROCESSED_DATA_DIR / "test_set.csv", index=False)

    # 2. Binary Detector Training (Normal vs. Faulty)
    print("✔️ Training Fault Detector")
    X_train, _ = split_X_y(df_train, drop_metadata=True)
    y_train_binary = (df_train['faultNumber'] > 0).astype(int)

    detector = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42))
    ])
    detector.fit(X_train, y_train_binary)
    joblib.dump(detector, MODEL_DIR / "tep_detector.pkl")

    # 3. Multi-class Diagnostician Training (Fault Identification)
    print("✔️ Training Fault Diagnostician")
    df_faulty = df_train[df_train['faultNumber'] > 0]
    X_diag, y_diag = split_X_y(df_faulty, drop_metadata=True)

    diagnostician = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42))
    ])
    diagnostician.fit(X_diag, y_diag)
    joblib.dump(diagnostician, MODEL_DIR / "tep_diagnostician.pkl")

    print(f"✔️ Training complete. Models and test_set.csv are ready")

if __name__ == "__main__":
    run_training()
