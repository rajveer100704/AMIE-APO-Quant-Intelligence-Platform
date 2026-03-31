import pytest
import os
import yaml
from src.execution.risk_guard import RiskGuard

@pytest.fixture
def temp_risk_config(tmp_path):
    config = {
        "risk_limits": {
            "kill_switch_drawdown": 0.05,
            "max_position_per_asset": 0.1,
            "max_slippage_bps": 50
        },
        "execution_policy": {
            "default_mode": "DRY"
        }
    }
    config_path = tmp_path / "risk_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)

@pytest.mark.unit
def test_risk_guard_kill_switch(temp_risk_config):
    guard = RiskGuard(config_path=temp_risk_config)
    # High drawdown should trigger kill switch
    result = guard.validate_order("SPY", 0.05, current_drawdown=0.06, expected_slippage_bps=10)
    assert result["status"] == "REJECTED"
    assert "Kill Switch" in result["reason"]

@pytest.mark.unit
def test_risk_guard_position_limit(temp_risk_config):
    guard = RiskGuard(config_path=temp_risk_config)
    # Exceeding 10% limit should reject
    result = guard.validate_order("SPY", 0.15, current_drawdown=0.01, expected_slippage_bps=10)
    assert result["status"] == "REJECTED"
    assert "Position Limit" in result["reason"]

@pytest.mark.unit
def test_risk_guard_slippage_limit(temp_risk_config):
    guard = RiskGuard(config_path=temp_risk_config)
    # Exceeding 50bps slippage should reject
    result = guard.validate_order("SPY", 0.05, current_drawdown=0.01, expected_slippage_bps=60)
    assert result["status"] == "REJECTED"
    assert "Slippage Violation" in result["reason"]

@pytest.mark.unit
def test_risk_guard_dry_run_approval(temp_risk_config):
    guard = RiskGuard(config_path=temp_risk_config)
    # Valid order in DRY mode should be APPROVED with dry_run=True
    result = guard.validate_order("SPY", 0.05, current_drawdown=0.01, expected_slippage_bps=10)
    assert result["status"] == "APPROVED"
    assert result.get("dry_run") is True
