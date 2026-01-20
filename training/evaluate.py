import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report
import sys
import os

# ==========================================
# CONFIGURATION AND PATHS
# ==========================================
current_dir = Path(__file__).resolve().parent
root_path = current_dir.parent
# Adding data_pipeline to sys.path to allow imports from src
sys.path.append(str(root_path / "data_pipeline"))

from src.data_loader import split_X_y

# Environment variables for directory paths with local fallbacks
MODEL_DIR = Path(os.getenv("MODEL_PATH", root_path / "data" / "models"))
# A Senior DevOps practice: evaluating on the specific test set generated during training
TEST_DATA_FILE = Path(os.getenv("PROCESSED_DATA_PATH", root_path / "data" / "processed")) / "test_set.csv"

def generate_audit():
    """
    Loads saved models and evaluates performance on the unseen test dataset.
    Generates a detailed classification report and exports it to a text file.
    """
    print("✔️ tarting model performance audit...")

    # 1. Verification and Loading
    if not TEST_DATA_FILE.exists():
        print(f"❌ Error: Test set not found at {TEST_DATA_FILE}. Please run train.py first.")
        return

    print("✔️ Loading models and test data...")
    try:
        detector = joblib.load(MODEL_DIR / "tep_detector.pkl")
        diagnostician = joblib.load(MODEL_DIR / "tep_diagnostician.pkl")
    except FileNotFoundError as e:
        print(f"❌ Error: Model files missing in {MODEL_DIR}. {e}")
        return

    df_test = pd.read_csv(TEST_DATA_FILE)

    # 2. Inference Logic (Cascaded Architecture)
    print("✔️ Running inference on test set...")
    X_test, y_true = split_X_y(df_test, drop_metadata=True)

    # Step 1: Detect if a fault is present
    fault_detected = detector.predict(X_test)
    # Step 2: Identify the specific type of fault
    fault_type = diagnostician.predict(X_test)

    # Final prediction: If detector says Normal (0), label is 0.
    # Otherwise, take the diagnostician's specific fault class.
    y_pred = [diag if det == 1 else 0 for det, diag in zip(fault_detected, fault_type)]

    # 3. Performance Reporting
    # zero_division=0: Suppresses warnings when a class is never predicted
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal"] + [f"Fault {i}" for i in range(1, 21)],
        zero_division=0
    )

    print("\n" + "="*60)
    print("✔️ DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60)

    # 4. Artifact Exportation
    audit_log_path = MODEL_DIR / "audit_report.txt"
    with open(audit_log_path, "w") as f:
        f.write(report)

    print(f"\n✔️ Audit report successfully exported to: {audit_log_path}")

if __name__ == "__main__":
    generate_audit()
