import requests

BASE_URL = "http://127.0.0.1:8000"

def test_health_endpoint():
    """Verifies API liveness and model artifact readiness.

    Ensures the service is reachable and that both cascaded models
    are correctly loaded into memory.

    Raises:
        AssertionError: If the health check fails or models are not ready.
    """
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["detector_ready"] is True
    assert data["diagnostician_ready"] is True

def test_predict_normal_operation():
    """Validates the inference pipeline with a baseline feature vector.

    Tests a 'Normal Operation' scenario where all sensors provide
    baseline values to verify correct schema handling and model response.

    Raises:
        AssertionError: If the prediction contract is violated.
    """
    # Constructing a zeroed feature vector for schema validation
    payload = {
        "sensors": {f"xmeas_{i}": 0.0 for i in range(1, 42)}
    }
    # Inject manipulated variables (xmv) into the feature set
    for i in range(1, 12):
        payload["sensors"][f"xmv_{i}"] = 0.0

    response = requests.post(f"{BASE_URL}/predict", json=payload)

    assert response.status_code == 200
    result = response.json()
    # Ensure the response adheres to the expected inference schema
    assert "is_anomaly" in result
    assert "fault_code" in result
    assert isinstance(result["is_anomaly"], bool)

def test_predict_invalid_data():
    """Ensures robust input validation and proper error signaling.

    Verifies that the API correctly triggers a 422 Unprocessable Entity
    status when receiving malformed JSON, as enforced by Pydantic schemas.

    Raises:
        AssertionError: If the API fails to reject invalid data.
    """
    payload = {"donnees_invalides": 123}
    response = requests.post(f"{BASE_URL}/predict", json=payload)

    # 422 is the standard HTTP status for Pydantic validation failures
    assert response.status_code == 422
