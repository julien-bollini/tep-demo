# data_pipeline/src/run_pipeline.py
"""Exécute le pipeline complet."""

from download_data import run_download
#from clean_data import clean_tep_data
#from process_data import process_tep_data

if __name__ == "__main__":
    # Étape 1
    run_download()

    # Étape 2
 #   clean_tep_data()

    # Étape 3
 #   process_tep_data()

    print("✔️ Pipeline terminé avec succès")
