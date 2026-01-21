import pandas as pd
import pytest
from data_pipeline.src.data_loader import split_X_y

def test_split_X_y_basic():
    # Préparation d'un DataFrame de test
    df = pd.DataFrame({
        'faultNumber': [0, 1, 2],
        'simulationRun': [1, 1, 1],
        'sample': [1, 2, 3],
        'xmeas_1': [0.1, 0.2, 0.3],
        'xmv_1': [1.0, 1.1, 1.2]
    })

    # Exécution avec drop_metadata=True (comportement par défaut)
    X, y = split_X_y(df, drop_metadata=True)

    # Vérifications
    assert len(y) == 3
    assert 'faultNumber' not in X.columns
    assert 'simulationRun' not in X.columns
    assert 'sample' not in X.columns
    assert list(X.columns) == ['xmeas_1', 'xmv_1']

def test_split_X_y_keep_metadata():
    df = pd.DataFrame({
        'faultNumber': [0],
        'simulationRun': [1],
        'xmeas_1': [0.1]
    })

    # Exécution en gardant les métadonnées
    X, y = split_X_y(df, drop_metadata=False)

    assert 'simulationRun' in X.columns
    assert 'faultNumber' not in X.columns
