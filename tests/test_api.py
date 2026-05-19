import pytest
import time
from fastapi.testclient import TestClient
from src.api.server import app, cache
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.unit
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.unit
def test_regime_endpoint(client, tmp_path):
    # Mock the REGIME_DIR to use our tmp_path
    with patch("src.api.server.REGIME_DIR", str(tmp_path)):
        import pandas as pd
        df = pd.DataFrame({"regime": [2, 2, 1]})
        df.to_parquet(tmp_path / "SPY_regime.parquet")
        
        response = client.get("/regime/SPY")
        assert response.status_code == 200
        data = response.json()
        assert data["regime_label"] == "Neutral"
        assert "latency_ms" in data

@pytest.mark.unit
def test_amis_endpoint(client):
    with patch("src.api.server.fusion.compute_amis", return_value=75.5):
        response = client.get("/amis/SPY")
        assert response.status_code == 200
        data = response.json()
        assert data["amis"] == 75.5

@pytest.mark.unit
def test_optimize_endpoint(client, tmp_path):
    """Test the /optimize endpoint with mocked data and risk checks."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    import pandas as pd
    import numpy as np
    
    # Create dummy price data
    for s in ["SPY", "VIX"]:
        df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=pd.date_range("2024-01-01", periods=3))
        df.to_parquet(data_dir / f"{s}_ohlcv.parquet")

    with patch("src.api.server.DATA_DIR", str(data_dir)), \
         patch("src.api.server.solver.mean_variance", return_value=np.array([0.6, 0.4])), \
         patch("src.api.server.risk_guard.validate_order", return_value={"status": "APPROVED"}):
        
        response = client.get("/optimize?symbols=SPY,VIX")
        assert response.status_code == 200
        data = response.json()
        assert "portfolio" in data
        assert "weights" in data["portfolio"]
        assert data["portfolio"]["weights"]["SPY"] == 0.6
        assert "execution" in data
        assert len(data["execution"]["orders"]) == 2

@pytest.mark.performance
def test_api_latency(client):
    """Elite Fix 7: Isolated performance test for API latency."""
    start = time.perf_counter()
    client.get("/health")
    latency = (time.perf_counter() - start) * 1000
    assert latency < 50 # Health check should be very fast
