from src.download_data import run_download
from src.preprocessing import run_preprocessing

def start_pipeline():
    print("1. Downloading data from Kaggle")
    run_download()

    print("\n2. Data preprocessing")
    run_preprocessing()

    print("✔️ Pipeline completed successfully")

if __name__ == "__main__":
    start_pipeline()
