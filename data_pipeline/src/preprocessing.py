import pandas as pd
import os
from pathlib import Path
from src.data_loader import load_dataset

# Calcul de secours (fallback)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# PRIORITÉ à la variable d'environnement injectée par Docker
PROCESSED_DATA_PATH = Path(os.getenv("PROCESSED_DATA_PATH", PROJECT_ROOT / "data" / "processed"))

def run_preprocessing():
    """
    Fusionne Normal + Faulty, optimise les types et sauvegarde un CSV propre.
    Le scaling est délégué au script d'entraînement.
    """
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des données en panne
    print("✔️ Chargement des données 'Faulty'")
    df_faulty = load_dataset("TEP_Faulty_Training.csv")

    # 2. Chargement des données normales
    print("✔️ Chargement des données 'Normal'")
    try:
        df_normal = load_dataset("TEP_FaultFree_Training.csv")
    except FileNotFoundError:
        df_normal = load_dataset("TEP_Normal_Training.csv")

    # 3. Étiquetage et Fusion
    df_normal['faultNumber'] = 0 # Label pour le cas sain
    df_combined = pd.concat([df_normal, df_faulty], axis=0, ignore_index=True)

    print(f"✔️ Fusion terminée : {df_combined.shape[0]} lignes")

    # 4. Sauvegarde du fichier consolidé
    output_file = PROCESSED_DATA_PATH / "tep_master_cleaned.csv"
    df_combined.to_csv(output_file, index=False)

    print(f"✔️ Fichier d'entraînement sauvegardé")
    return output_file

if __name__ == "__main__":
    run_preprocessing()
