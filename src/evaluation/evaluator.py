import pandas as pd
import joblib
import json
import os
from sklearn.metrics import classification_report
from src.config import MODEL_DIR, MODEL_DETECT_NAME, MODEL_DIAG_NAME, FINAL_SPLIT_DIR

class ModelEvaluator:
    """
    Orchestrates the evaluation of cascaded TEP models and manages performance artifacts.

    This class handles the conditional execution of metrics computation based on
    artifact availability or environmental overrides (FORCE_REPROCESS). It ensures
    that performance data is persisted as JSON for auditability and dashboarding.
    """
    def __init__(self) -> None:
        """Initializes paths and environmental state for evaluation."""
        self.metrics_path = MODEL_DIR / "metrics.json"
        self.test_set_path = FINAL_SPLIT_DIR / "test_set_final.parquet"
        self.force_mode = os.getenv("FORCE_REPROCESS", "false").lower() == "true"

    def run_evaluation(self) -> None:
        """
        Main entry point for the evaluation lifecycle.

        Orchestrates the workflow: checks for existing metrics artifacts,
        recomputes if necessary or requested, and ensures the results
        are printed to the standard output.
        """
        # Condition: Invalidate cache if force_mode is True or artifact is missing
        if not self.metrics_path.exists() or self.force_mode:
            self._compute_metrics()
        else:
            print(f"⏩ Metrics file found at {self.metrics_path}. Loading for display...")

        self._display_results()

    def _compute_metrics(self) -> None:
        """
        Executes inference across the cascaded model architecture and saves scores.

        Loads serialized model artifacts (Detector and Diagnostician), runs a
        two-stage prediction pipeline, and serializes the classification report
        into a JSON artifact.

        Returns:
            None

        Raises:
            FileNotFoundError: If model artifacts or test datasets are missing.
        """
        print("📊 Computing metrics (Cascaded Models)...")

        if not self.test_set_path.exists():
            print(f"❌ Error: Test set missing at {self.test_set_path}")
            return

        # Load artifacts into memory
        detector = joblib.load(MODEL_DIR / MODEL_DETECT_NAME)
        diagnostician = joblib.load(MODEL_DIR / MODEL_DIAG_NAME)
        df_test = pd.read_parquet(self.test_set_path)

        # Standard Feature/Target split
        X_test = df_test.drop(columns=['target'])
        y_test = df_test['target']

        # Stage 1: Detection (Binary/Multi-class base)
        y_pred_detect = detector.predict(X_test)
        final_preds = y_pred_detect.copy()
        mask_anomaly = (y_pred_detect >= 1)

        # Stage 2: Diagnosis (Refine predictions for detected anomalies)
        if mask_anomaly.any():
            final_preds[mask_anomaly] = diagnostician.predict(X_test[mask_anomaly])

        # Generate serialized report
        report = classification_report(y_test, final_preds, output_dict=True)
        with open(self.metrics_path, "w") as f:
            json.dump(report, f, indent=4)
        print("✅ Metrics computed and saved.")

    def _display_results(self) -> None:
        """
        Parses metric artifacts and renders a terminal-optimized dashboard.

        Transforms the JSON report into a formatted DataFrame, filtering
        for specific fault classes to provide a clear operational view.

        Returns:
            None
        """
        if not self.metrics_path.exists():
            print("❌ No metrics to display. Run with FORCE=true first.")
            return

        with open(self.metrics_path, "r") as f:
            report = json.load(f)

        df_report = pd.DataFrame(report).transpose()
        fault_metrics = df_report.loc[df_report.index.str.isdigit()].copy()
        fault_metrics.index = "Fault " + fault_metrics.index

        print("\n" + "═"*60)
        print("📊 TEP PERFORMANCE DASHBOARD".center(60))
        print("═"*60)
        print(fault_metrics[['precision', 'recall', 'f1-score']].to_string(
            formatters={
                'precision': '{:,.2%}'.format,
                'recall': '{:,.2%}'.format,
                'f1-score': '{:,.2%}'.format
            }
        ))
        print("─"*60)
        print(f"⭐ GLOBAL ACCURACY: {report['accuracy']:.2%}".center(60))
        print("═"*60 + "\n")
