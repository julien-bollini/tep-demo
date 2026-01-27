import shutil
import kagglehub
import yaml
from src.config import (
    RAW_DATA_PATH,
    CACHE_CONFIG_PATH,
)


class DataDownloader:
    """Downloads data from Kaggle"""

    def __init__(self, dataset_name="afrniomelo/tep-csv"):
        """Initializes the DataDownloader with a dataset name.

        Args:
            dataset_name (str): The Kaggle dataset name to download.
        """
        self.dataset_name = dataset_name

    def download(self):
        """Downloads the dataset and saves the cache.

        Returns:
            str: Path to the downloaded data.

        Raises:
            Exception: If the dataset already exists.
        """
        if RAW_DATA_PATH.exists():
            print("✅ Data already exists")
            return str(RAW_DATA_PATH)

        print("✅ Downloading dataset from Kaggle")
        temp_download_path = kagglehub.dataset_download(self.dataset_name)
        print(f"Temporary download path: {temp_download_path}")

        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(temp_download_path, RAW_DATA_PATH, dirs_exist_ok=True)

        self._save_cache_info(temp_download_path)

        print("✅ Download completed successfully")
        return str(RAW_DATA_PATH)

    def _save_cache_info(self, cache_path):
        """Saves the cache location.

        Args:
            cache_path (str): Path to the cached dataset.
        """
        cache_info = {
            "kaggle_cache_path": str(cache_path),
            "data_path": str(RAW_DATA_PATH),
        }

        CACHE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_CONFIG_PATH, "w") as f:
            yaml.dump(cache_info, f, default_flow_style=False)

        print(f"✅ Cache info saved to {CACHE_CONFIG_PATH}")
