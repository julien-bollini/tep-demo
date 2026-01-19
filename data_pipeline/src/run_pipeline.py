from download_data import run_download
from preprocessing import run_preprocessing

def start_pipeline():
    # Étape 1 : Téléchargement des données brutes depuis Kaggle
    # Cette étape vérifie si les fichiers sont déjà présents localement.
    print("1. Téléchargement des données de Kaggle")
    run_download()

    # Étape 2 : Nettoyage, Optimisation et Fusion
    # On regroupe Normal + Faulty dans un seul fichier maître (Master CSV).
    # On laisse les données "brutes" (pas de scaling) car le scaler sera
    # géré par le Pipeline Scikit-Learn dans l'entraînement.
    print("\n2. Preprocessing des données")
    run_preprocessing()

    print("✔️ Pipeline terminé avec succès")

if __name__ == "__main__":
    start_pipeline()
