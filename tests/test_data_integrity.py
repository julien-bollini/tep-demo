import pandas as pd
import sys
from src.config import MERGED_FILE_PATH

def check_merged_data() -> None:
    """Performs a structural integrity audit on the merged TEP dataset.

    This utility validates the existence of the Silver Layer artifact,
    inspects its dimensions, and verifies the absence of null values
    to ensure it is ready for the training pipeline.

    Returns:
        None: The function logs results to stdout and handles exits via sys.exit.
    """
    # Verify artifact existence before loading
    if not MERGED_FILE_PATH.exists():
        print(f"❌ Critical Failure: Artifact not found at {MERGED_FILE_PATH}")
        sys.exit(1)

    # Load optimized parquet artifact
    df: pd.DataFrame = pd.read_parquet(MERGED_FILE_PATH)

    print("=== Data Integrity Check ===")
    print(f"📍 Path: {MERGED_FILE_PATH}")
    print(f"📊 Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")

    # Analyze class distribution for target leakage or imbalance
    print("\n🧐 Class Distribution (faultNumber):")
    print(df['faultNumber'].value_counts())

    # Assess missing data density
    nan_count: int = int(df.isna().sum().sum())
    print(f"\n❓ Missing values: {nan_count}")

    # Final validation gate
    if nan_count == 0:
        print("✅ Data is clean and production-ready!")
    else:
        print("⚠️ Warning: Data contains null values. Preprocessing required.")
        sys.exit(1)

if __name__ == "__main__":
    check_merged_data()
