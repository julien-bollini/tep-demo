import pandas as pd
import os
from pathlib import Path
from src.data_loader import load_dataset

# Fallback path calculation relative to project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# PRIORITY: Use Docker-injected environment variable, otherwise use local fallback
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", PROJECT_ROOT / "data" / "processed"))

def run_preprocessing():
    """
    Merges Normal + Faulty datasets, optimizes types, and saves a consolidated clean CSV.
    Note: Scaling is delegated to the training pipeline to prevent data leakage.
    """
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Loading faulty data
    print("✔️ Loading 'Faulty' training data")
    df_faulty = load_dataset("TEP_Faulty_Training.csv")

    # 2. Loading normal (fault-free) data
    print("✔️ Loading 'Normal' training data")
    try:
        df_normal = load_dataset("TEP_FaultFree_Training.csv")
    except FileNotFoundError:
        # Fallback for alternative naming conventions
        df_normal = load_dataset("TEP_Normal_Training.csv")

    # 3. Labeling and Consolidation
    # Assign label 0 for normal operation
    df_normal['faultNumber'] = 0
    df_combined = pd.concat([df_normal, df_faulty], axis=0, ignore_index=True)

    print(f"✔️ Consolidation complete: {df_combined.shape[0]} rows merged")

    # 4. Memory Optimization
    print("✔️ Optimizing types: converting float64 to float32")
    # Sélectionne toutes les colonnes float64
    float64_cols = df_combined.select_dtypes(include=['float64']).columns
    # Conversion sur place pour économiser de la RAM
    df_combined[float64_cols] = df_combined[float64_cols].astype('float32')

    # 5. Saving the master cleaned dataset
    output_file = PROCESSED_DATA_PATH / "tep_master_cleaned.csv"
    df_combined.to_csv(output_file, index=False)

    print(f"✔️ Master training file saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    run_preprocessing()
