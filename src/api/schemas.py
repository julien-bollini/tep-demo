from pydantic import BaseModel, Field
from typing import Dict

class SensorData(BaseModel):
    """
    Schema for Tennessee Eastman Process (TEP) input features.

    Validates the telemetry payload ensuring all 52 process variables
    (41 measured, 11 manipulated) are present for inference.
    """
    sensors: Dict[str, float] = Field(
        ...,
        example={
            "xmeas_1": 0.25038,
            "xmeas_2": 3674.0,
            "xmv_11": 17.373
        }
    )

    class Config:
        schema_extra = {
            "description": "The payload must contain all sensors required by the RandomForest pipelines."
        }

class PredictionResponse(BaseModel):
    """
    Standardized response format for the inference API.
    """
    is_anomaly: bool
    fault_code: int
    status: str
    confidence: float | None = None

class HealthResponse(BaseModel):
    """Health check status schema."""
    status: str
    detector_ready: bool
    diagnostician_ready: bool
