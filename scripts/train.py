#!/usr/bin/env python3
"""
Point d'entrée pour le conteneur Docker de training.
Ce script est appelé par le Dockerfile.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import MLPipeline

if __name__ == "__main__":
    pipeline = MLPipeline()
    exit_code = pipeline.train()
    sys.exit(exit_code)
