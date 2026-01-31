import pandas as pd
import joblib
import os

# =============================================================================
# INFRASTRUCTURE CONFIGURATION
# =============================================================================

# Docker-internal paths for artifact persistence
PATH_INPUT = "data/processed/parquet/TEP_Faulty_Testing.parquet"
PATH_OUTPUT = "data/processed/parquet/TEP_pred.parquet"
PATH_MODELS = "models/"

def run_targeted_validation() -> None:
    """Executes inference on a specific data slice for model validation.

    Filters the testing set for Simulation #1 (Faults 1-20) and runs the
    cascaded model pipeline to generate prediction benchmarks.

    Returns:
        None: Results are serialized to the Silver/Gold storage layer.
    """
    print("🚀 Extracting Simulation #1 for faults 1 to 20...")

    # 1. DATA INGESTION
    df_full = pd.read_parquet(PATH_INPUT)

    # 2. TARGETED SLICING
    # Isolating Simulation 1 to validate model consistency across fault types
    df = df_full[
        (df_full['simulationRun'] == 1) &
        (df_full['faultNumber'] > 0) &
        (df_full['faultNumber'] <= 20)
    ].copy()

    print(f"✅ Slice resolved: {df['faultNumber'].nunique()} faults, {len(df)} rows.")

    # 3. MODEL ORCHESTRATION
    # Loading serialized pipelines from the model repository
    try:
        detector = joblib.load(os.path.join(PATH_MODELS, "detector_pipeline.joblib"))
        diagnostician = joblib.load(os.path.join(PATH_MODELS, "diagnostician_pipeline.joblib"))
    except Exception as e:
        print(f"❌ Model artifact error : {e}")
        return

    # 4. INFERENCE PIPELINE
    # Identifying feature vector (xmeas + xmv)
    sensor_cols = [c for c in df.columns if c.startswith('xmeas_') or c.startswith('xmv_')]
    X = df[sensor_cols]

    print("🧠 Generating cascaded predictions...")
    df['Detector'] = detector.predict(X)
    df['Faults_pred'] = diagnostician.predict(X)

    # Calculate deviation from ground truth (Delta 0 = Accuracy 100%)
    df['Delta'] = df['Faults_pred'] - df['faultNumber']

    # 5. PERSISTENCE
    df.to_parquet(PATH_OUTPUT, index=False)

    # =========================================================================
    # ANALYTICS RECAP
    # =========================================================================
    print("\n--- 📊 PERFORMANCE SNAPSHOT BY CLASS ---")
    summary = df.groupby('faultNumber').agg({
        'Detector': 'mean',
        'Faults_pred': lambda x: (x == x.name).mean()
    }).rename(columns={'Faults_pred': 'Accuracy_Diag', 'Detector': 'Detection_Rate'})
    print(summary)

    print(f"\n💾 Validation artifact saved: {PATH_OUTPUT}")

if __name__ == "__main__":
    run_targeted_validation()
