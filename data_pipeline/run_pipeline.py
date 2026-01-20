# Dans data_pipeline/run_pipeline.py
from src.download_data import run_download
from src.preprocessing import run_preprocessing

def start_pipeline():
    print("1. Téléchargement des données de Kaggle")
    run_download()

    print("\n2. Preprocessing des données")
    run_preprocessing()

    print("✔️ Pipeline terminé avec succès")

if __name__ == "__main__":
    start_pipeline()
