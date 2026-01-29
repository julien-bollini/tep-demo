from pathlib import Path
import os

# =============================================================================
# GLOBAL PIPELINE CONFIGURATION
# =============================================================================

# Number of simulations to retain for development (set to None for full dataset)
DEFAULT_N_SIMULATIONS = 10

# =============================================================================
# PROJECT FILESYSTEM ORCHESTRATION
# =============================================================================

# Root directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_PATH", BASE_DIR / "data"))

# Configuration and state persistence
CONFIG_DIR = BASE_DIR / "config"
CACHE_CONFIG_PATH = CONFIG_DIR / "cache.yaml"

# Metadata and artifact storage
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# =============================================================================
# DATA LIFECYCLE PATHS (MEDALLION ARCHITECTURE)
# =============================================================================

# Bronze Layer: Original CSV files from source
RAW_DATA_PATH = DATA_DIR / "raw" / "tep-csv"

# Silver Layer: Cleaned Parquet files and consolidated master records
PROCESSED_DATA_PATH = DATA_DIR / "processed"
RAW_PARQUET_DIR = PROCESSED_DATA_PATH / "parquet"
MERGED_FILE_PATH = RAW_PARQUET_DIR / "TEP_Faulty_and_Normal_Merged.parquet"

# Gold Layer: Model-ready subsets and final evaluation splits
SUBSETS_DIR = PROCESSED_DATA_PATH / "subsets"
FINAL_SPLIT_DIR = PROCESSED_DATA_PATH / "final_split"
FINAL_TEST_SET_PATH = FINAL_SPLIT_DIR / "test_set_final.parquet"

# =============================================================================
# NOMENCLATURE & REFRENCES
# =============================================================================

# Base filenames for training sets
FAULTY_TRAIN_FILENAME = "TEP_Faulty_Training.parquet"
NORMAL_TRAIN_FILENAME = "TEP_FaultFree_Training.parquet"

FAULTY_PARQUET_PATH = RAW_PARQUET_DIR / FAULTY_TRAIN_FILENAME
NORMAL_PARQUET_PATH = RAW_PARQUET_DIR / NORMAL_TRAIN_FILENAME

# =============================================================================
# SCHEMA & TYPE OPTIMIZATION
# =============================================================================

# Optimized schema to minimize RAM usage during large-scale data ingestion
OPTIMIZED_DTYPES = {
    "faultNumber": "int8",
    "simulationRun": "int16",
    "sample": "int16"
}

# Sensor measurements: xmeas_1 to xmeas_41 (mapped to float32)
for i in range(1, 42):
    OPTIMIZED_DTYPES[f"xmeas_{i}"] = "float32"

# Manipulated variables: xmv_1 to xmv_11 (mapped to float32)
for i in range(1, 12):
    OPTIMIZED_DTYPES[f"xmv_{i}"] = "float32"

# =============================================================================
# ETL REGISTRY
# =============================================================================

RAW_CSV_FILES = [
    "TEP_FaultFree_Testing.csv", "TEP_FaultFree_Training.csv",
    "TEP_Faulty_Testing.csv", "TEP_Faulty_Training.csv"
]

# =============================================================================
# INFRASTRUCTURE INITIALIZATION
# =============================================================================

def initialize_filesystem():
    """Ensures necessary project structure exists before execution."""
    required_dirs = [
        CONFIG_DIR,
        RAW_PARQUET_DIR,
        SUBSETS_DIR,
        FINAL_SPLIT_DIR,
        MODELS_DIR,
        LOGS_DIR
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)

# Execute initialization on module load
initialize_filesystem()
