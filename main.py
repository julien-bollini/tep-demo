import sys
from pathlib import Path

from src.preprocessing.downloader import DataDownloader
from src.preprocessing.processor import DataProcessor
from src.training.loader import DataLoader


class MLPipeline:
    """Orchestrates the ML pipeline."""

    def __init__(self):
        """Initializes the components of the pipeline."""
        self.downloader = DataDownloader()
        self.processor = DataProcessor()
        self.loader = DataLoader()

    def preprocess(self) -> int:
        """Runs the ETL (Extract, Transform, Load) part of the pipeline.

        Returns:
            int: Exit status code (0 for success).

        Raises:
            Exception: If an error occurs during preprocessing.
        """
        print("\n" + "="*70)
        print("PIPELINE TEP FAULT DETECTION")
        print("="*70)

        print("\n" + "="*70)
        print("🚀 STARTING PREPROCESSING")
        print("="*70)

        # --- Step 1 ---
        print("\n▶ STEP 1: Download CSV TEP")
        self.downloader.download()

        # --- Step 2 ---
        print("\n▶ STEP 2: Convert CSV to Parquet")
        self.processor.convert_csv_to_parquet()

        # --- Step 3 ---
        print("\n▶ STEP 3: Merge Datasets")
        self.processor.merge_faulty_and_normal_data()

        print("\n" + "="*70)
        print("✅ PREPROCESSING COMPLETED")
        print("="*70)
        return 0

    def train(self) -> int:
        """Placeholder for the training phase."""
        print("\n" + "="*50)
        print("🧠 STARTING MODEL TRAINING")
        print("="*50)

        # --- Step 1 & 2 : Chargement et Downsampling ---
        # Grâce à ta config, load_data va charger MERGED_FILE_PATH,
        # réduire les simulations, et sauvegarder N_SIMU_FILE_PATH automatiquement.
        print("\n▶ STEP 1 & 2: Chargement et Downsampling")
        X, y = self.loader.load_data()

        print(f"✅ Données prêtes : {X.shape[0]} lignes chargées.")
        print(f"📊 Features : {X.shape[1]} variables capteurs.")

        # Prochaine étape : Split Train/Test
        # print("\n▶ STEP 3: Split Train/Test")

        return 0

        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETED")
        print("="*70 + "\n")
        return 0


def main():
    """Main function to orchestrate the execution of the pipeline."""
    # Initialize pipeline
    pipeline = MLPipeline()

    try:
        # Execute preprocessing steps
        pipeline.preprocess()
        pipeline.train()
        return 0
    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
