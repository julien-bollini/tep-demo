import joblib
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from typing import Any
from src.api.schemas import SensorData, PredictionResponse, HealthResponse
from src.config import (
    MODEL_DIR,
    MODEL_DETECT_NAME,
    MODEL_DIAG_NAME
)


# --- Global Artifact Repository ---
MODELS: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles API startup and shutdown events.

    Loads ML artifacts into memory on startup and ensures they are available
    for inference without re-reading files for each request.
    """
    try:
        detector_path = MODEL_DIR / MODEL_DETECT_NAME
        diag_path = MODEL_DIR / MODEL_DIAG_NAME

        MODELS["detector"] = joblib.load(detector_path)
        MODELS["diagnostician"] = joblib.load(diag_path)
        print(f"✅ Models successfully loaded from {MODEL_DIR}")
    except Exception as e:
        print(f"❌ CRITICAL: Could not load model artifacts: {e}")

    yield
    # Resource teardown on shutdown
    MODELS.clear()
    print("🛑 Model resources deallocated.")

# --- API Configuration ---
app = FastAPI(
    title="TEP Reactor Monitoring API",
    description="Professional Real-time Inference API for Chemical Reactor Fault Diagnosis.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict[str, Any]:
    """Provides a cloud-ready health check endpoint.

    Returns:
        dict: Status of the service and model availability.
    """
    return {
        "status": "healthy",
        "detector_ready": "detector" in MODELS,
        "diagnostician_ready": "diagnostician" in MODELS
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: SensorData) -> dict[str, Any]:
    """
        Standardized health check for load balancers and orchestrators.

        Checks the availability of the web service and the integrity of
        memory-mapped ML models.

        Returns:
            dict[str, Any]: Health status and readiness of individual models.
        """
    if not MODELS.get("detector") or not MODELS.get("diagnostician"):
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not initialized. Check server logs."
        )

    try:
        # Data preparation: Mapping payload to tabular format
        input_df: pd.DataFrame = pd.DataFrame([payload.sensors])

        # Feature Engineering: Enforce strict column ordering (X-meas 1-41 followed by X-mv 1-11)
        # This alignment is strictly required for scikit-learn pipeline compatibility
        sensor_cols = [f'xmeas_{i}' for i in range(1, 42)] + [f'xmv_{i}' for i in range(1, 12)]
        input_df = input_df[sensor_cols]

        # Stage 1: Execute Anomaly Detection
        prediction_array = MODELS["detector"].predict(input_df)
        is_anomaly: bool = bool(int(prediction_array[0]))

        # Response payload initialization
        response: dict[str, Any] = {
            "is_anomaly": is_anomaly,
            "fault_code": 0,
            "status": "Normal Operation"
        }

        # Stage 2: Conditional Fault Diagnosis
        if is_anomaly:
            fault_prediction = MODELS["diagnostician"].predict(input_df)
            fault_code: int = int(fault_prediction[0])
            response["fault_code"] = fault_code
            response["status"] = f"Anomalous state detected: Fault {fault_code}"

        return response

    except Exception as e:
        # Log the error for observability and return a 500 status
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference failure")

if __name__ == "__main__":
    import uvicorn
    # Local development entrypoint
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
