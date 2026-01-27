import pandas as pd
import numpy as np
from pathlib import Path
from src.config import PARQUET_DATA_PATH, PROCESSED_DATA_PATH


class DataLoader:
    """Charge et prépare les données pour l'entraînement"""

    METADATA_COLUMNS = ['faultNumber', 'simulationRun', 'sample']
    TARGET_COLUMN = 'faultNumber'

    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else PARQUET_DATA_PATH

    def load_parquet(self, filename=None):
        """Charge un ou tous les fichiers Parquet"""
        if filename:
            files = [self.data_path / filename]
        else:
            files = list(self.data_path.glob("*.parquet"))

        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)

        print(f"✔️ Loaded {len(files)} file(s) - Shape: {df.shape}")
        return df

    def split_X_y(self, df, drop_metadata=True):
        """Sépare features (X) et target (y)"""
        y = df[self.TARGET_COLUMN]
        to_drop = self.METADATA_COLUMNS if drop_metadata else [self.TARGET_COLUMN]
        X = df.drop(columns=[c for c in to_drop if c in df.columns])
        return X, y

    def train_test_split(self, df, retention_rate=0.02, test_size=0.2):
        """
        Split train/test en évitant le data leakage (run-aware split)

        Args:
            df: DataFrame source
            retention_rate: % de runs à conserver (downsampling)
            test_size: % pour le test set

        Returns:
            tuple: (df_train, df_test)
        """
        # ID unique par simulation run
        df['run_id'] = df['faultNumber'].astype(str) + "_" + df['simulationRun'].astype(str)
        unique_runs = df['run_id'].unique()

        # Downsampling
        n_keep = int(len(unique_runs) * retention_rate)
        kept_runs = np.random.choice(unique_runs, n_keep, replace=False)
        df = df[df['run_id'].isin(kept_runs)]

        # Split run-aware
        split_idx = int(len(kept_runs) * (1 - test_size))
        train_runs = kept_runs[:split_idx]

        df_train = df[df['run_id'].isin(train_runs)].drop(columns=['run_id'])
        df_test = df[~df['run_id'].isin(train_runs)].drop(columns=['run_id'])

        print(f"✔️ Train shape: {df_train.shape} | Test shape: {df_test.shape}")
        return df_train, df_test

    def save_test_set(self, df_test):
        """Sauvegarde le test set pour évaluation ultérieure"""
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DATA_PATH / "test_set.parquet"
        df_test.to_parquet(output_path, index=False)
        print(f"✔️ Test set saved: {output_path}")
