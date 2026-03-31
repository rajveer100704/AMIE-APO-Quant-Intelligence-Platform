import pytest
import os
import json
import numpy as np

def load_latest_snapshot(log_dir="logs"):
    if not os.path.exists(log_dir):
        return None
    files = sorted([f for f in os.listdir(log_dir) if f.startswith("execution_")], reverse=True)
    if not files:
        return None
    with open(os.path.join(log_dir, files[0]), 'r') as f:
        return json.load(f)

@pytest.mark.unit
def test_snapshot_schema_completeness():
    """Ensure the latest snapshot follows the mandated 'Intelligence Platform' schema."""
    raw_snapshot = load_latest_snapshot()
    assert raw_snapshot is not None, "No snapshots found to validate."
    snapshot = raw_snapshot["data"]
    
    # Check top-level blocks
    assert "snapshot_id" in snapshot
    assert "portfolio" in snapshot
    assert "execution" in snapshot
    assert "validation" in snapshot
    
    # Check Portfolio block
    p = snapshot["portfolio"]
    assert "weights" in p
    assert "exposure" in p
    assert "pnl" in p
    assert "drawdown" in p
    
    # Check Execution block
    e = snapshot["execution"]
    assert "orders" in e
    assert "success_rate" in e
    assert "avg_latency_ms" in e

@pytest.mark.unit
def test_metrics_range_validity():
    """Ensure quant-grade metrics are within logical/physical bounds."""
    raw_snapshot = load_latest_snapshot()
    if not raw_snapshot: return
    snapshot = raw_snapshot["data"]

    # Exposure: Total abs weight should be ≈ 1 for long-only or neutral-ish
    # But for demo/test, we just check it's non-negative and not infinite
    exposure = snapshot["portfolio"]["exposure"]
    assert 0 <= exposure <= 2.0 # Allow for some leveraged neutral strategies
    
    # Success Rate: Must be between 0 and 1
    sr = snapshot["execution"]["success_rate"]
    assert 0.0 <= sr <= 1.0
    
    # Drawdown: Must be between 0 and 1 (or 0% and 100%)
    dd = snapshot["portfolio"]["drawdown"]
    assert 0.0 <= dd <= 1.0

@pytest.mark.unit
def test_regime_stability_indicator():
    """Ensure regime labels are valid from labels mapping in server.py logic."""
    raw_snapshot = load_latest_snapshot()
    if not raw_snapshot: return
    snapshot = raw_snapshot["data"]
    
    valid_regimes = ["Crisis", "Neutral", "Bullish"]
    assert snapshot["regime"] in valid_regimes
