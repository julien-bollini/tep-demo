import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle
from src.config import (
    RAW_PARQUET_DIR,
    SUBSETS_DIR,
    FINAL_SPLIT_DIR,
    MERGED_FILE_PATH,
    DEFAULT_N_SIMULATIONS,
    DEFAULT_TEST_SIZE,
    RANDOM_SEED
)

class DataLoader:
    """Orchestrates data ingestion, subsampling, and leakage-proof splitting strategies.

        This service manages the transition between Silver and Gold layers, ensuring
        experimental integrity by treating simulation runs as atomic units during
        the train/test split.
    """

    def __init__(self, data_path: str | Path | None = None) -> None:
        """Initializes the DataLoader context for artifact resolution.

            Args:
                data_path (str | Path | None): Root directory for Parquet artifacts.
                    Defaults to RAW_PARQUET_DIR.
        """
        # Logical path resolution for the data source
        self.data_path: Path = Path(data_path) if data_path else RAW_PARQUET_DIR

    def load_data(self, n_simulations: int = DEFAULT_N_SIMULATIONS) -> pd.DataFrame:
        """Loads data from cache or generates a new subset from the master record.

        Args:
            n_simulations (int | None): Quota of simulation runs to retain per fault.
                If None, the full dataset is loaded.

        Returns:
            pd.DataFrame: Processed dataset with a 'unique_run_id' for unit-testing tracking.

        Raises:
            FileNotFoundError: If the master Silver artifact is missing.
        """
        subset_path: Path = SUBSETS_DIR / f"TEP_subset_N{n_simulations}.parquet"

        if subset_path.exists():
            print(f"⚡ Ingesting cached subset: {subset_path.name}")
            df: pd.DataFrame = pd.read_parquet(subset_path)
        else:
            print("📖 Generating fresh subset from Gold Master record...")
            if not MERGED_FILE_PATH.exists():
                raise FileNotFoundError(f"❌ Master artifact missing at: {MERGED_FILE_PATH}")

            df: pd.DataFrame = pd.read_parquet(MERGED_FILE_PATH)
            if n_simulations:
                df = self._subsample_by_run(df, n_simulations)
                # Persist the subset to minimize expensive I/O in future iterations
                df.to_parquet(subset_path, index=False)

        # Generate a composite key to ensure grouping integrity during train/test split
        df['unique_run_id'] = df['faultNumber'].astype(str) + "_" + df['simulationRun'].astype(str)
        return df

    def split_by_run(
        self,
        df: pd.DataFrame,
        test_size: float = DEFAULT_TEST_SIZE
    ) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
        """Executes a stratified split based on simulation runs (Group-wise splitting).

        Ensures that all observations from a single simulation run are either in
        the training set or the test set, but never both.

        Args:
            df (pd.DataFrame): Source dataframe containing 'unique_run_id'.
            test_size (float): Proportion of runs allocated to the test set.

        Returns:
            tuple: A nested structure ((X_train, y_train), (X_test, y_test)).
        """
        unique_runs: np.ndarray = shuffle(df['unique_run_id'].unique(), random_state=RANDOM_SEED)
        split_idx: int = int(len(unique_runs) * (1 - test_size))

        train_runs: np.ndarray = unique_runs[:split_idx]

        # Sectorization into training and evaluation sets
        df_train: pd.DataFrame = df[df['unique_run_id'].isin(train_runs)].copy()
        df_test: pd.DataFrame = df[~df['unique_run_id'].isin(train_runs)].copy()

        return self._finalize_split(df_train), self._finalize_split(df_test)

    def _finalize_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """De-couples features from labels and prunes metadata.

        Args:
            df (pd.DataFrame): Raw split segment.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple of (Features, Target).
        """
        metadata: list[str] = ['faultNumber', 'simulationRun', 'sample', 'unique_run_id']
        cols_to_drop: list[str] = [c for c in metadata if c in df.columns]

        X: pd.DataFrame = df.drop(columns=cols_to_drop)
        y: pd.Series = df['faultNumber']
        return X, y

    def _subsample_by_run(self, df: pd.DataFrame, n_simulations: int) -> pd.DataFrame:
        """Filters the dataset to preserve a fixed quota of simulations per fault class.

        Args:
            df (pd.DataFrame): The full master dataset.
            n_simulations (int): Number of unique runs to keep per faultNumber.

        Returns:
            pd.DataFrame: A downsampled dataframe.
        """
        print(f"📉 Downsampling to {n_simulations} simulations per fault class...")
        mask: pd.Series = df.groupby('faultNumber')['simulationRun'].transform(
            lambda x: x.isin(np.unique(x)[:n_simulations])
        )
        return df[mask].reset_index(drop=True)

    def save_test_set(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Serializes the final test set to the Gold Layer for model evaluation.

        Args:
            X_test (pd.DataFrame): Evaluation features.
            y_test (pd.Series): Evaluation ground truth labels.
        """
        output_path: Path = FINAL_SPLIT_DIR / "test_set_final.parquet"

        df_test: pd.DataFrame = X_test.copy()
        df_test['target'] = y_test.values

        # Final archival of the Gold Standard test set
        df_test.to_parquet(output_path, index=False)
        print(f"✅ Gold Standard test set archived: {output_path}")
