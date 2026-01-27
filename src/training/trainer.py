import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_PATH
from src.data_loader import DataLoader


class ModelTrainer:
    """Entraîne les modèles de détection et diagnostic"""

    def __init__(self):
        self.loader = DataLoader()
        MODEL_PATH.mkdir(parents=True, exist_ok=True)
        self.detector = None
        self.diagnostician = None

    def train_detector(self, df_train):
        """
        Entraîne le détecteur binaire (Normal vs Faulty)

        Args:
            df_train: DataFrame d'entraînement
        """
        print("✔️ Training Fault Detector")

        X_train, _ = self.loader.split_X_y(df_train, drop_metadata=True)
        y_train_binary = (df_train['faultNumber'] > 0).astype(int)

        self.detector = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            ))
        ])

        self.detector.fit(X_train, y_train_binary)

        # Sauvegarde
        model_path = MODEL_PATH / "tep_detector.pkl"
        joblib.dump(self.detector, model_path)
        print(f"✔️ Detector saved: {model_path}")

    def train_diagnostician(self, df_train):
        """
        Entraîne le diagnosticien multi-classe (identification de la panne)

        Args:
            df_train: DataFrame d'entraînement
        """
        print("✔️ Training Fault Diagnostician")

        # Seulement les données avec pannes
        df_faulty = df_train[df_train['faultNumber'] > 0]
        X_diag, y_diag = self.loader.split_X_y(df_faulty, drop_metadata=True)

        self.diagnostician = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                n_jobs=-1,
                random_state=42
            ))
        ])

        self.diagnostician.fit(X_diag, y_diag)

        # Sauvegarde
        model_path = MODEL_PATH / "tep_diagnostician.pkl"
        joblib.dump(self.diagnostician, model_path)
        print(f"✔️ Diagnostician saved: {model_path}")

    def train_all(self, retention_rate=0.02, test_size=0.2):
        """Pipeline d'entraînement complet"""

        # 1. Chargement
        df = self.loader.load_parquet()

        # 2. Split train/test
        df_train, df_test = self.loader.train_test_split(
            df,
            retention_rate=retention_rate,
            test_size=test_size
        )

        # 3. Sauvegarde test set
        self.loader.save_test_set(df_test)

        # 4. Entraînement des modèles
        self.train_detector(df_train)
        self.train_diagnostician(df_train)

        print("✔️ Training complete")
