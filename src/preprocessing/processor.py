import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from pathlib import Path
from typing import Final
from src.training.loader import DataLoader
from src.config import (
    RAW_DATA_PATH,
    RAW_PARQUET_DIR,
    OPTIMIZED_DTYPES,
    RAW_CSV_FILES,
    MERGED_FILE_PATH,
    FAULTY_TRAIN_FILENAME,
    NORMAL_TRAIN_FILENAME,
    CROPPED_DATA_PATH
)

class DataProcessor:
    """Orchestrates data transformation and consolidation within the Silver Layer.

    This service manages the lifecycle of raw-to-optimized data conversion,
    implementing memory-efficient ingestion and consistent windowing for the TEP dataset.
    """

    def __init__(self) -> None:
        """Initializes the processor with infrastructure-aware configurations."""
        self.loader: DataLoader = DataLoader()
        # Toggle for cache invalidation via environment variable
        self.force_mode: Final[bool] = os.getenv("FORCE_REPROCESS", "false").lower() == "true"

    def crop_and_reindex_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies temporal windowing and resets sample indexing.

        Filters observations to the operational window (140-639) and
        normalizes the time index to a 1-500 range for model consistency.
        """
        start_s, end_s, shift = 140, 639, 139

        if 'sample' not in df.columns:
            return df

        # Vectorized slicing for high-performance processing
        df_cropped: pd.DataFrame = df[(df['sample'] >= start_s) & (df['sample'] <= end_s)].copy()
        df_cropped['sample'] = df_cropped['sample'] - shift

        # Data Quality Check: Ensuring 500 samples per simulation run
        points: int = len(df_cropped['sample'].unique())
        if points != 500:
            print(f"⚠️ Integrity Warning: Expected 500 points, detected {points}.")

        return df_cropped

    def convert_csv_to_parquet(self, input_csv: str, output_parquet: str) -> None:
        """
        Converts large CSV files to Parquet format using a memory-efficient
        chunking strategy. This prevents Out-Of-Memory (OOM) errors on
        resource-constrained environments like GitHub Actions or 8GB laptops.
        """
        print(f"📦 Processing: {os.path.basename(input_csv)}")

        # Define chunk size to maintain a low memory footprint (~500MB RAM)
        chunk_size = 100000

        # Initialize the CSV stream iterator
        reader = pd.read_csv(input_csv, chunksize=chunk_size)

        writer = None

        try:
            for i, chunk in enumerate(reader):
                # Convert Pandas DataFrame to PyArrow Table for high-performance I/O
                table = pa.Table.from_pandas(chunk)

                # Initialize the ParquetWriter with the schema from the first chunk
                if writer is None:
                    writer = pq.ParquetWriter(output_parquet, table.schema, compression='snappy')

                # Stream the chunk directly to the disk
                writer.write_table(table)

                # Logging progress for observability
                if (i + 1) % 5 == 0:
                    print(f"   Processed { (i + 1) * chunk_size } rows...")

        except Exception as e:
            print(f"❌ Error during streaming conversion: {e}")
            raise

        finally:
            # Ensure the writer is properly closed to finalize the Parquet file metadata
            if writer:
                writer.close()

        print(f"✅ Successfully persisted: {output_parquet}")

    def merge_faulty_and_normal_data(self) -> pd.DataFrame:
        """Consolidates discrete training sets into a unified Master Silver record.

        Handles class labeling for baseline data and ensures vertical concatenation
        integrity before persisting the Gold Master record.
        """
        if MERGED_FILE_PATH.exists() and not self.force_mode:
            print(f"✅ Master record detected: {MERGED_FILE_PATH.name}")
            return pd.read_parquet(MERGED_FILE_PATH)

        faulty_path: Final[Path] = RAW_PARQUET_DIR / FAULTY_TRAIN_FILENAME
        normal_path: Final[Path] = RAW_PARQUET_DIR / NORMAL_TRAIN_FILENAME

        print(f"📖 Merging artifacts: {faulty_path.name} + {normal_path.name}")
        faulty_df: pd.DataFrame = pd.read_parquet(faulty_path)
        normal_df: pd.DataFrame = pd.read_parquet(normal_path)

        if normal_df.empty and faulty_df.empty:
            print("❌ Critical: Source dataframes are empty.")
            return pd.DataFrame()

        # Harmonization: Assigning class 0 to normal operation data
        if "faultNumber" not in normal_df.columns:
            normal_df["faultNumber"] = 0

        # Master dataset generation (Silver Level)
        merged_df: pd.DataFrame = pd.concat([normal_df, faulty_df], axis=0, ignore_index=True)
        merged_df.to_parquet(MERGED_FILE_PATH, engine="pyarrow", index=False)

        print(f"✅ Consolidated record saved: {MERGED_FILE_PATH.name}")
        return merged_df

    def process_silver_layer(self) -> None:
        """Orchestrates the cropping and reindexing workflow for testing data.

        Prepares the 'Testing' slice specifically for Dashboard visualization
        and model evaluation (Gold Standard).
        """
        CROPPED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        testing_files: list[str] = [f for f in RAW_CSV_FILES if "Testing" in f]
        processed_count: int = 0

        for file_name in testing_files:
            parquet_name: str = file_name.replace(".csv", ".parquet")
            output_path: Path = CROPPED_DATA_PATH / parquet_name
            source_parquet: Path = RAW_PARQUET_DIR / parquet_name

            if output_path.exists() and not self.force_mode:
                print(f"⏩ Skipping (already cropped): {parquet_name}")
                continue

            print(f"✂️ Processing slice for: {parquet_name}...")

            # Priority: Use optimized Parquet if available, fallback to CSV
            if source_parquet.exists():
                df = pd.read_parquet(source_parquet)
            else:
                df = pd.read_csv(RAW_DATA_PATH / file_name, dtype=OPTIMIZED_DTYPES)

            df_final: pd.DataFrame = self.crop_and_reindex_samples(df)
            df_final.to_parquet(output_path, engine="pyarrow", index=False)
            print(f"✅ Ready for Dashboard: {parquet_name}")
            processed_count += 1

        if processed_count == 0:
            print("✅ Cropped testing data is already up to date.")
