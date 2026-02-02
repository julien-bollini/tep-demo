import sys
import os
import pandas as pd
import argparse
from src.preprocessing.downloader import DataDownloader
from src.preprocessing.processor import DataProcessor
from src.training.loader import DataLoader
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

class MLPipeline:
    """Orchestrates the end-to-end Machine Learning pipeline.

    This controller manages the transition between data engineering,
    model training, and performance validation, ensuring data integrity
    and artifact versioning across the MLOps lifecycle.
    """

    def __init__(self) -> None:
        """Initializes pipeline components with specific domain responsibilities."""
        self.downloader: DataDownloader = DataDownloader()
        self.processor: DataProcessor = DataProcessor()
        self.loader: DataLoader = DataLoader()
        self.trainer: ModelTrainer = ModelTrainer()
        self.evaluator: ModelEvaluator = ModelEvaluator()

    def _speak(self, message: str) -> None:
        """Helper to provide audio feedback on macOS."""
        if sys.platform == "darwin":
            os.system(f'say -v Samantha "{message}" &')

    def preprocess(self) -> None:
        """Executes the ETL (Extract, Transform, Load) phase.

        Handles raw data ingestion, conversion to high-performance formats (Parquet),
        and multi-layer processing (Bronze to Silver).

        Returns:
            int: Exit code (0 for success, 1 for failure).
        """
        print("\n" + "="*70)
        print("🚀 STARTING PREPROCESSING")
        print("="*70)

        # Step 1: Ingestion from remote/external source
        print("\n▶ STEP 1: Download CSV TEP")
        self.downloader.download()

        # Step 2: Optimization - Parquet provides better compression and schema enforcement
        print("\n▶ STEP 2: Convert CSV to Parquet")
        self.processor.convert_csv_to_parquet()

        # Step 3: Silver layer refinement (Applying business logic filters)
        print("▶ STEP 3: Processing Silver Layer")
        self.processor.process_silver_layer()

        # Step 4: Final Feature/Target set creation
        print("\n▶ STEP 4: Merge Datasets")
        self.processor.merge_faulty_and_normal_data()

        print("\n✅ PREPROCESSING COMPLETED")
        self._speak("Preprocessing finished")
        return 0

    def train(self, force: bool = False) -> int:
        """Orchestrates data preparation, splitting, and model training.

        Implements intelligent loading with caching and strict temporal/run
        separation to prevent data leakage in cascaded models.

        Args:
            force (bool): If True, bypasses cache and forces re-training of all models.
                         Defaults to False.

        Returns:
            int: Exit code (0 for success, 1 for failure).
        """
        print("\n" + "="*70)
        print("🧠 STARTING DATA PREPARATION & TRAINING")
        print("="*70)

        # Step 1: Loading from processed Parquet storage (IO efficient)
        print("\n▶ STEP 1: Loading Data (with cache check)")
        df: pd.DataFrame = self.loader.load_data()

        # Step 2: MLOps Guardrail - Splitting by 'Run' ID to ensure validation robustness
        print("▶ STEP 2: Splitting Data by Run (Avoid Leakage)")
        split_data: tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]] = self.loader.split_by_run(df)
        (X_train, y_train), (X_test, y_test) = split_data

        print("✅ Data split completed:")
        print(f"   - Train set: {X_train.shape[0]} samples")
        print(f"   - Test set:  {X_test.shape[0]} samples")
        print(f"📊 Features: {X_train.shape[1]} sensors")

        # Step 3: Artifact Versioning - Archiving the test set for reproducible evaluation
        print("\n▶ STEP 3: Archiving Test Set")
        self.loader.save_test_set(X_test, y_test)

        # Step 4: Model fitting with idempotent logic
        print("\n▶ STEP 4: Model Training (Cascaded Models)")
        df_train_ready: pd.DataFrame = pd.concat([X_train, y_train], axis=1)

        # 'force' parameter controls artifact overwriting in the trainer module
        self.trainer.train_cascaded_models(df_train_ready, force=force)

        print("\n✅ TRAINING STAGE COMPLETED")
        self._speak("Models are trained")
        return 0

    def evaluate(self) -> int:
        """Executes performance validation on unseen data.

        Aggregates metrics and generates final validation artifacts (JSON/Plots)
        for deployment readiness checks.

        Returns:
            int: Exit code (0 for success, 1 for failure).
        """
        print("\n" + "="*70)
        print("📊 STARTING MODEL EVALUATION")
        print("="*70)

        self.evaluator.run_evaluation()

        print("\n✅ EVALUATION COMPLETED")
        return 0

def main() -> int:
    """Main entry point for the pipeline CLI.

    Parses execution flags and orchestrates specific lifecycle stages based on
    infrastructure requirements (e.g., Jenkins/GitHub Actions steps).

    Returns:
        int: System exit code (0 for success, 1 for failure).
    """
    # Define CLI interface for infrastructure automation
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="TEP Machine Learning Pipeline - Production Orchestrator"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["preprocess", "train", "evaluate", "all"],
        default="all",
        help="Target pipeline stage to execute"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass model persistence and force full training"
    )

    args: argparse.Namespace = parser.parse_args()
    pipeline: MLPipeline = MLPipeline()

    try:
        # State-based execution logic to prevent redundant processing in CI/CD
        if args.step == "preprocess":
            return pipeline.preprocess()

        elif args.step == "train":
            exit_code: int = pipeline.train(force=args.force)
            # Automatic validation triggered only on forced training for immediate feedback
            if args.force:
                print("\n💡 Force flag detected: Triggering immediate post-training evaluation...")
                pipeline.evaluate()
            return exit_code

        elif args.step == "evaluate":
            return pipeline.evaluate()

        else:
            # Sequential execution for standard deployment or local testing
            pipeline.preprocess()
            pipeline.train(force=args.force)
            return pipeline.evaluate()

    except Exception as e:
        # Standard error reporting for monitoring tools (Sentry/Datadog)
        print(f"\n❌ Pipeline CRITICAL FAILURE: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # OS-level exit code signaling for container orchestrator Docker
    sys.exit(main())
