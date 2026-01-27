from pathlib import Path
import os

# === Base Directories ===
BASE_DIR = Path(__file__).resolve().parent.parent  # Navigate to the project root
DATA_DIR = Path(os.getenv("DATA_PATH", BASE_DIR / "data"))

# === Data Directories ===
RAW_DATA_PATH = DATA_DIR / "raw" / "tep-csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
PARQUET_DATA_PATH = DATA_DIR / "processed" / "parquet"

# === Configuration ===
CONFIG_DIR = BASE_DIR / "config"
CACHE_CONFIG_PATH = CONFIG_DIR / "cache.yaml"

# === Other Directories ===
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# === Optimisation des types de données ===  # ← NOUVEAU
OPTIMIZED_DTYPES = {"faultNumber": "int8", "simulationRun": "int16", "sample": "int16"}
for i in range(1, 42):
    OPTIMIZED_DTYPES[f"xmeas_{i}"] = "float32"
for i in range(1, 12):
    OPTIMIZED_DTYPES[f"xmv_{i}"] = "float32"
