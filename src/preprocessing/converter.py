import pandas as pd
#from pathlib import Path
from src.config import (
    RAW_DATA_PATH,
    PARQUET_DATA_PATH,
    OPTIMIZED_DTYPES
)

class DataConverter:

    def convert_csv_to_parquet(self):
        """Convertit tous les CSV en Parquet avec types optimisés"""

        if PARQUET_DATA_PATH.exists():
            print(f"Data already exists in {PARQUET_DATA_PATH}")
            return str(RAW_DATA_PATH)

        PARQUET_DATA_PATH.mkdir(parents=True, exist_ok=True)
        csv_files = list(RAW_DATA_PATH.glob("*.csv"))

        for csv_path in csv_files:
            df = pd.read_csv(csv_path, dtype=OPTIMIZED_DTYPES)
            parquet_path = PARQUET_DATA_PATH / f"{csv_path.stem}.parquet"
            df.to_parquet(parquet_path, engine="pyarrow", index=False)
            print(f"✔️ {csv_path.name} → {parquet_path.name}")

        print(f"✔️ Converted {len(csv_files)} files")
        return csv_files
