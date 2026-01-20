import os
import shutil
from pathlib import Path
import kagglehub

def run_download():
    # 1. Dynamic root detection (tep-demo/)
    # Resolves the path relative to data_pipeline/download_data.py
    base_dir = Path(__file__).resolve().parent.parent.parent

    # 2. Path configuration (Docker environment variable priority, then local fallback)
    raw_data_path = Path(os.getenv("DATA_PATH", base_dir / "data")) / "raw" / "tep-csv"

    # 3. Check if data already exists
    if raw_data_path.exists():
        print(f"✔️ Data already exists in {raw_data_path}")
        return

    # 4. Download execution
    print("✔️ Downloading dataset from Kaggle")
    temp_download_path = kagglehub.dataset_download("afrniomelo/tep-csv")

    # 5. Move files to final destination
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temp_download_path, raw_data_path, dirs_exist_ok=True)

    # 6. Cache cleanup
    if Path(temp_download_path).exists():
        shutil.rmtree(temp_download_path)
        print("✔️ Temporary cache cleared")

    print(f"✔️ Download completed successfully")

if __name__ == "__main__":
    run_download()
