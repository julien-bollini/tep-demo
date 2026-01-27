import sys
import argparse
#from pathlib import Path

from src.preprocessing.downloader import DataDownloader
from src.preprocessing.converter import DataConverter



class MLPipeline:
    """Orchestrateur du pipeline ML"""

    def __init__(self):
        """Initialise les composants du pipeline"""
        self.downloader = DataDownloader()
        self.converter = DataConverter()

    def preprocess(self) -> int:

            # Step 1: Download
            print("Step 1/2: Downloading data from Kaggle")
            self.downloader.download()

            # Step 2: Convert
            print("Step 2/2: Converting CSV to Parquet")
            self.converter.convert_csv_to_parquet()

            print("✔️ Preprocessing completed successfully")
            return 0

def main():
    # Initialize pipeline
    pipeline = MLPipeline()

    # Execute preprocessing steps
    pipeline.preprocess()

# Assurez-vous d'ajouter cette ligne pour exécuter le script principal
if __name__ == "__main__":
    main()
