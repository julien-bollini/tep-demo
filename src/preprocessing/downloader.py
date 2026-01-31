import shutil
import kagglehub
import yaml
from pathlib import Path
from src.config import (
    RAW_DATA_PATH,
    RAW_CSV_FILES,
    CACHE_CONFIG_PATH,
)


class DataDownloader:
    """Manages the automated acquisition of raw datasets from Kaggle.

    This class enforces idempotency by verifying the structural integrity of the
    Bronze Layer (raw data) artifact-by-artifact before initiating any network I/O.
    """

    def __init__(self, dataset_name: str = "afrniomelo/tep-csv") -> None:
        """Initializes the downloader with a target repository identifier.

        Args:
            dataset_name (str): The unique identifier for the Kaggle dataset.
        """
        self.dataset_name: str = dataset_name

    def download(self) -> str:
        """Triggers the dataset ingestion process and persists metadata.

        Validates the presence of all required CSV files. If any artifact is missing
        or the Bronze Layer is incomplete, the full dataset is resynchronized
        from the upstream source to ensure state consistency.

        Returns:
            str: The resolved string representation of the local data path.

        Raises:
            shutil.Error: If the directory synchronization or copy operation fails.
            OSError: If filesystem permissions prevent data persistence.
        """
        print("\n▶ STEP 1: Ingest Raw Data (Bronze Layer)")

        # Verify Bronze Layer integrity via granular artifact checking
        if RAW_DATA_PATH.exists():
            files_status: list[bool] = []

            for csv_name in RAW_CSV_FILES:
                file_exists: bool = (RAW_DATA_PATH / csv_name).exists()
                if file_exists:
                    print(f"⏩ Already cached: {csv_name}")
                else:
                    # Alerting: explicit log to explain why idempotency gate is failing
                    print(f"⚠️ Missing artifact: {csv_name}")
                files_status.append(file_exists)

            # Idempotency Gate: Bypass ingestion only if 100% of artifacts are present
            if all(files_status):
                print("✅ Bronze Layer is fully synchronized")
                return str(RAW_DATA_PATH)

        print(f"🚀 Initiating secure ingestion for dataset: {self.dataset_name}")

        # Fetch remote artifacts (KaggleHub abstracts local caching and versioning)
        temp_download_path: str = kagglehub.dataset_download(self.dataset_name)
        print(f"📦 Upstream artifact cached at: {temp_download_path}")

        # Ensure destination hierarchy exists before state synchronization
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Synchronize data: Restores missing files and enforces target state consistency
        shutil.copytree(temp_download_path, RAW_DATA_PATH, dirs_exist_ok=True)

        # Persist ingestion metadata for lifecycle audit and traceability
        self._save_cache_info(temp_download_path)

        print("✅ Ingestion cycle completed successfully")
        return str(RAW_DATA_PATH)

    def _save_cache_info(self, cache_path: str | Path) -> None:
        """Persists ingestion metadata to a YAML manifest for traceability.

        Args:
            cache_path (str | Path): The temporary location where Kaggle stored the artifact.

        Returns:
            None
        """
        # Metadata manifest structure for auditability
        cache_info: dict[str, str] = {
            "kaggle_cache_path": str(cache_path),
            "data_path": str(RAW_DATA_PATH),
        }

        # Create config directory if not present for state persistence
        CACHE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(CACHE_CONFIG_PATH, "w") as f:
            yaml.dump(cache_info, f, default_flow_style=False)

        print(f"📝 Metadata manifest updated at: {CACHE_CONFIG_PATH}")
