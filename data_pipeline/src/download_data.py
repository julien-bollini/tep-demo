import os
import shutil
from pathlib import Path
import kagglehub

def run_download():
    # 1. Détection dynamique de la racine (tep-demo/)
    # On part de data_pipeline/download_data.py et on remonte d'un niveau
    base_dir = Path(__file__).resolve().parent.parent.parent

    # 2. Configuration du chemin (Priorité à la variable Docker, sinon local)
    data_raw_path = Path(os.getenv("DATA_PATH", base_dir / "data")) / "raw" / "tep-csv"

    # 3. Vérification de présence
    if data_raw_path.exists():
        print(f"✔️ Données déjà présentes")
        return

    # 4. Action de téléchargement
    print("✔️ Téléchargement en cours...")
    tmp_path = kagglehub.dataset_download("afrniomelo/tep-csv")

    # 5. Déplacement vers la destination finale
    data_raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(tmp_path, data_raw_path, dirs_exist_ok=True)

    # 6. Nettoyage du cache
    if Path(tmp_path).exists():
        shutil.rmtree(tmp_path)
        print("✔️ Cache supprimé. Espace libéré.")

    print(f"✔️ Téléchargement terminé")

if __name__ == "__main__":
    run_download()
