import shutil
import kagglehub
import yaml
#from pathlib import Path
from src.config import (
    RAW_DATA_PATH,
    CACHE_CONFIG_PATH,
)


class DataDownloader:
    """Télécharge les données depuis Kaggle"""

    def __init__(self, dataset_name="afrniomelo/tep-csv"):
        self.dataset_name = dataset_name

    def download(self):
        """Télécharge le dataset et sauvegarde le cache"""
        if RAW_DATA_PATH.exists():
            print(f"✔️ Data already exists in {RAW_DATA_PATH}")
            return str(RAW_DATA_PATH)

        print("✔️ Downloading dataset from Kaggle")
        temp_download_path = kagglehub.dataset_download(self.dataset_name)
        print(f"Temporary download path: {temp_download_path}")

        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp_download_path, RAW_DATA_PATH, dirs_exist_ok=True)

        self._save_cache_info(temp_download_path)

        print("✔️ Download completed successfully")
        return str(RAW_DATA_PATH)

    def _save_cache_info(self, cache_path):
        """Sauvegarde l'emplacement du cache"""
        cache_info = {
            "kaggle_cache_path": str(cache_path),
            "data_path": str(RAW_DATA_PATH)
        }

        CACHE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_CONFIG_PATH, "w") as f:
            yaml.dump(cache_info, f, default_flow_style=False)

        print(f"✔️ Cache info saved to {CACHE_CONFIG_PATH}")
