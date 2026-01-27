import pandas as pd
from pathlib import Path
from src.config import (
    RAW_DATA_PATH,
    PARQUET_DATA_PATH,
    OPTIMIZED_DTYPES,
    FAULTY_PARQUET_PATH,
    NORMAL_PARQUET_PATH,
    RAW_CSV_FILES,
)


class DataProcessor:
    """Handles the data processing steps of the pipeline."""

    def convert_csv_to_parquet(self):
        """Converts missing CSV files to Parquet with optimized data types.

        This method ensures that the destination directory exists, iterates over
        the CSV files defined in your configuration, and converts them to Parquet
        format only if they do not already exist.
        """
        # Ensure the destination directory exists
        PARQUET_DATA_PATH.mkdir(parents=True, exist_ok=True)

        converted_count = 0

        # Iterate over the CSV files defined in your configuration
        for csv_name in RAW_CSV_FILES:
            csv_path = RAW_DATA_PATH / csv_name
            # Generate the corresponding Parquet file name (e.g., .csv -> .parquet)
            parquet_name = csv_name.replace(".csv", ".parquet")
            parquet_path = PARQUET_DATA_PATH / parquet_name

            # Step 1: Check if the source CSV file exists
            if not csv_path.exists():
                print(f"⚠️ Source file not found: {csv_name}")
                continue

            # Step 2: Check if the Parquet file already exists
            if parquet_path.exists():
                print(f"⏩ Already converted: {parquet_name}")
                continue

            # Step 3: Convert if necessary
            try:
                df = pd.read_csv(csv_path, dtype=OPTIMIZED_DTYPES)
                df.to_parquet(parquet_path, engine="pyarrow", index=False)
                print(f"✔️ {csv_name} → {parquet_name}")
                converted_count += 1
            except Exception as e:
                print(f"❌ Error during conversion of {csv_name}: {e}")

        if converted_count == 0:
            print("✅ All files are already up to date")
        else:
            print(f"🏁 Conversion completed: {converted_count} new file(s) created.")


    def load_dataset(self, file_path):
        """Loads a Parquet file and returns it as a DataFrame.

        Args:
            file_path (Path): The path to the Parquet file to load.

        Returns:
            DataFrame: The loaded DataFrame.
        """
        return pd.read_parquet(file_path)


    def merge_faulty_and_normal_data(self):
        """
        Merges the 'Faulty' and 'Normal' Parquet files only if the merged file
        does not already exist.

        Returns:
            DataFrame: The merged DataFrame.
        """
        merged_file_path = PARQUET_DATA_PATH / "faulty_and_normal_merged.parquet"

        # Step 1: Check if the merged file already exists
        if merged_file_path.exists():
            print(f"✅ The merged file already exists: {merged_file_path.name}")
            # Optionally, load and return the existing file if needed
            return pd.read_parquet(merged_file_path)

        # Step 2: Load and check the source files
        faulty_df = self.load_dataset(FAULTY_PARQUET_PATH)
        normal_df = self.load_dataset(NORMAL_PARQUET_PATH)

        if normal_df.empty and faulty_df.empty:
            print("❌ Error: No source data to merge.")
            return pd.DataFrame()

        # Step 3: Merge and save
        # Add the faultNumber column (0 for normal data)
        if not normal_df.empty:
            normal_df["faultNumber"] = 0

        merged_df = pd.concat([normal_df, faulty_df], axis=0, ignore_index=True)

        # Save the final merged file
        merged_df.to_parquet(merged_file_path, engine="pyarrow", index=False)
        print(f"✅ Merging completed and saved to: {merged_file_path.name}")

        return merged_df
