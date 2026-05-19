import pytest
from unittest.mock import patch, MagicMock
from src.state.state_manager import StateManager
from src.api.server import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def state_manager_fail():
    """Returns a state manager with Redis failed."""
    with patch("redis.Redis") as mock:
        mock.return_value.ping.side_effect = ConnectionError("Redis is Down")
        sm = StateManager()
        return sm

@pytest.mark.unit
def test_redis_failure_fallback(state_manager_fail):
    """Elite Fix 5: Ensure Redis failure falls back to memory store."""
    sm = state_manager_fail
    assert sm.mode == "memory"
    
    # Try setting and getting
    sm.set("failure_test", "fallback_value")
    result = sm.get("failure_test")
    assert result == "fallback_value" # Fallback works

@pytest.mark.unit
def test_amis_api_fallback_on_failure(client):
    """Elite Fix 5: Ensure API returns something safe even if a component fails."""
    # Mocking AMISFusion to raise an exception
    with patch("src.api.server.fusion.compute_amis") as mock_fusion:
        mock_fusion.side_effect = Exception("System Crash")
        response = client.get("/amis/AAPL")
        
        assert response.status_code == 500
        assert "AMIS computation failed" in response.json()["detail"]


@pytest.mark.unit
def test_risk_guard_missing_config_safety():
    """Elite Fix 5: Ensure Risk Guard uses safe defaults if config is missing."""
    from src.execution.risk_guard import RiskGuard
    # Try loading with non-existent config
    guard = RiskGuard(config_path="non_existent.yaml")
    # Should not crash, but use safe defaults (e.g. 10% positions)
    assert guard.config.get("kill_switch_drawdown") == 0.05
    assert guard.config.get("max_position_per_asset") == 0.1
    
    # Validate with high drawdown -> should reject based on defaults
    result = guard.validate_order("SPY", 0.05, 0.06, 10)
    assert result["status"] == "REJECTED"
