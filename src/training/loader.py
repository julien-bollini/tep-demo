import pandas as pd
import numpy as np
from pathlib import Path
from src.config import (
    PARQUET_DATA_PATH,
    PROCESSED_DATA_PATH,
    DEFAULT_N_SIMULATIONS,
    MERGED_FILE_PATH,
    N_SIMU_FILE_PATH
)

class DataLoader:
    """Charge et prépare les données pour l'entraînement"""

    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else PARQUET_DATA_PATH

    def load_data(self, file_path=MERGED_FILE_PATH, n_simulations=DEFAULT_N_SIMULATIONS):
        """
        Charge les données et applique un sous-échantillonnage si demandé.
        """
        # Utilise le chemin passé en argument ou celui par défaut
        path = Path(file_path) if file_path else self.data_path

        if path is None or not path.exists():
            raise FileNotFoundError(f"❌ Chemin invalide : {path}")

        print(f"📖 Chargement de : {path.name}...")
        df = pd.read_parquet(path)

        # Si n_simulations est précisé, on réduit le dataset
        if n_simulations is not None:
            df = self._subsample_by_run(df, n_simulations)

        # Save the final merged file
        df.to_parquet(N_SIMU_FILE_PATH, engine="pyarrow", index=False)
        print(f"✅ Merging completed and saved to: {N_SIMU_FILE_PATH.name}")

        # Séparation classique X (features) et y (target)
        # On exclut les colonnes de métadonnées pour l'entraînement
        metadata_cols = ['faultNumber', 'simulationRun', 'sample']
        X = df.drop(columns=metadata_cols)
        y = df['faultNumber']

        return X, y

    def _subsample_by_run(self, df, n_simulations=DEFAULT_N_SIMULATIONS):
        """
        Logique interne pour filtrer par simulationRun.
        """
        print(f"📉 Réduction à {n_simulations} simulations par type de défaut...")

        # Ajout de include_groups=False pour éviter le Warning
        return (
            df.groupby('faultNumber')
            .apply(
                lambda x: x[x['simulationRun'].isin(x['simulationRun'].unique()[:n_simulations])],
                include_groups=False
            )
            .reset_index(level=0) # On remet 'faultNumber' qui a été déplacé dans l'index
            .reset_index(drop=True)
        )

    def save_test_set(self, df_test):
        """Sauvegarde le test set pour évaluation ultérieure"""
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        output_path = PROCESSED_DATA_PATH / "test_set.parquet"
        df_test.to_parquet(output_path, index=False)
        print(f"✔️ Test set saved: {output_path}")
