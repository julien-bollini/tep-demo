from pathlib import Path
import os

# =============================================================================
# INFRASTRUCTURE: FILESYSTEM RESOLUTION
# =============================================================================

# Root directory discovery: Locates the 'tep-demo' base folder
# Optimized using __file__ to ensure path resolution works across different environments
PROJ_ROOT: Path = Path(__file__).resolve().parent.parent

# Data storage root: Support for Environment Variable override (Cloud-native readiness)
DATA_DIR: Path = Path(os.getenv("DATA_PATH", PROJ_ROOT / "data"))

# Configuration and state persistence
CONFIG_DIR: Path = PROJ_ROOT / "config"
CACHE_CONFIG_PATH: Path = CONFIG_DIR / "cache.yaml"

# Logging and Artifact storage
LOGS_DIR: Path = PROJ_ROOT / "logs"
MODEL_DIR: Path = PROJ_ROOT / "models"

# =============================================================================
# DATA LIFECYCLE: MEDALLION ARCHITECTURE PATHS
# =============================================================================

# BRONZE LAYER: Raw source ingestion (Original CSV files)
RAW_DATA_PATH: Path = DATA_DIR / "raw" / "tep-csv"
RAW_CSV_FILES: list[str] = [
    "TEP_FaultFree_Testing.csv",
    "TEP_FaultFree_Training.csv",
    "TEP_Faulty_Testing.csv",
    "TEP_Faulty_Training.csv"
]

# SILVER LAYER: Sanitized Parquet records and consolidated master datasets
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed"
RAW_PARQUET_DIR: Path = PROCESSED_DATA_PATH / "parquet"
CROPPED_DATA_PATH: Path = DATA_DIR / "processed" / "cropped"

# File references for merged master records
FAULTY_TRAIN_FILENAME: str = "TEP_Faulty_Training.parquet"
NORMAL_TRAIN_FILENAME: str = "TEP_FaultFree_Training.parquet"
MERGED_FILE_PATH: Path = RAW_PARQUET_DIR / "TEP_Faulty_and_Normal_Merged.parquet"

# GOLD LAYER: Model-ready subsets and final evaluation hold-out splits
SUBSETS_DIR: Path = PROCESSED_DATA_PATH / "subsets"
FINAL_SPLIT_DIR: Path = PROCESSED_DATA_PATH / "final_split"
FINAL_TEST_SET_PATH: Path = FINAL_SPLIT_DIR / "test_set_final.parquet"

# =============================================================================
# GLOBAL PIPELINE PARAMETERS
# =============================================================================

# Reproducibility: Static seed for all stochastic processes (CV, RF, etc.)
RANDOM_SEED: int = 42

# Workload control: Limit simulations for rapid prototyping (None for full-scale)
DEFAULT_N_SIMULATIONS: int = None     # None for full-scale training (500 runs)

# Validation strategy: Standard hold-out ratio
DEFAULT_TEST_SIZE: float = 0.2      # 20% for testing, 80% for training

# =============================================================================
# MODEL ARCHITECTURE & HYPERPARAMETERS
# =============================================================================

# Serialization references
MODEL_DETECT_NAME: str = "detector_pipeline.joblib"
MODEL_DIAG_NAME: str = "diagnostician_pipeline.joblib"

# Phase 1: Detector (Binary Anomaly Detection)
# Focused on high sensitivity to distinguish normal vs. abnormal states
DETECTOR_PARAMS = {
    "n_estimators": 50,
    "max_depth": 10,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_SEED
}

# Phase 2: Diagnostician (Multiclass Fault Classification)
# Focused on granular fault identification using faulty-only samples
DIAGNOSTICIAN_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_SEED
}

# =============================================================================
# MEMORY & SCHEMA OPTIMIZATION
# =============================================================================

# Downcasting schema to minimize RAM footprint during large-scale ingestion
# Using float32 instead of float64 to reduce memory usage by 50%
OPTIMIZED_DTYPES: dict[str, str] = {
    "faultNumber": "int8",
    "simulationRun": "int16",
    "sample": "int16"
}

# Sensor measurements (xmeas_1 to xmeas_41)
for i in range(1, 42):
    OPTIMIZED_DTYPES[f"xmeas_{i}"] = "float32"

# Manipulated variables (xmv_1 to xmv_11)
for i in range(1, 12):
    OPTIMIZED_DTYPES[f"xmv_{i}"] = "float32"

# =============================================================================
# INFRASTRUCTURE INITIALIZATION
# =============================================================================

def initialize_filesystem() -> None:
    """
    Ensures the project workspace structure exists before runtime.
    Creates missing directories for artifacts, logs, and processed data.
    """
    target_dirs: list[Path] = [
        CONFIG_DIR,
        RAW_PARQUET_DIR,
        SUBSETS_DIR,
        FINAL_SPLIT_DIR,
        MODEL_DIR,
        LOGS_DIR
    ]
    for directory in target_dirs:
        directory.mkdir(parents=True, exist_ok=True)

# Auto-initialize project structure on module import
initialize_filesystem()
