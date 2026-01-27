from pathlib import Path
import os

# === Model Optimisation ===
DEFAULT_N_SIMULATIONS = 10 # None = All simulations

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

# === Optimisation des types de données ===
OPTIMIZED_DTYPES = {"faultNumber": "int8", "simulationRun": "int16", "sample": "int16"}
for i in range(1, 42):
    OPTIMIZED_DTYPES[f"xmeas_{i}"] = "float32"
for i in range(1, 12):
    OPTIMIZED_DTYPES[f"xmv_{i}"] = "float32"

# === Parquet File Paths ===
FAULTY_PARQUET_PATH = PARQUET_DATA_PATH / "TEP_Faulty_Training.parquet"
NORMAL_PARQUET_PATH = PARQUET_DATA_PATH / "TEP_FaultFree_Training.parquet"
MERGED_FILE_PATH = PARQUET_DATA_PATH / "TEP_Faulty_and_Normal_Merged.parquet"
N_SIMU_FILE_PATH = PARQUET_DATA_PATH / "TEP_N_Simulations.parquet"

RAW_CSV_FILES = [
    "TEP_FaultFree_Testing.csv",
    "TEP_FaultFree_Training.csv",
    "TEP_Faulty_Testing.csv",
    "TEP_Faulty_Training.csv"
]

PARQUET_FILES = [
    "TEP_FaultFree_Testing.parquet",
    "TEP_FaultFree_Training.parquet",
    "TEP_Faulty_Testing.parquet",
    "TEP_Faulty_Training.parquet"
]
