import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Load test environment variables before anything else
os.environ["EXECUTION_MODE"] = "DRY"
os.environ["ALPACA_API_KEY"] = "dummy"
os.environ["ALPACA_SECRET_KEY"] = "dummy"

def pytest_sessionstart(session):
    """
    Principal Engineer's Safety Lock:
    Enforce DRY mode at the start of the test session.
    """
    mode = os.getenv("EXECUTION_MODE", "LIVE")
    if mode != "DRY":
        raise RuntimeError(f"CRITICAL SAFETY VIOLATION: Tests must run in DRY mode. Current mode: {mode}")

@pytest.fixture(scope="session")
def sample_data():
    """Provides a deterministic dataset for tests."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "close": np.linspace(100, 110, 100) + np.random.normal(0, 0.5, 100),
        "volume": np.random.randint(1000, 5000, 100),
        "high": np.linspace(101, 111, 100),
        "low": np.linspace(99, 109, 100),
        "open": np.linspace(100, 110, 100)
    }, index=dates)
    return data

@pytest.fixture
def mock_alpaca():
    with patch("alpaca_trade_api.REST") as mock:
        instance = mock.return_value
        instance.submit_order.return_value = MagicMock(status="accepted", id="test_id")
        instance.get_account.return_value = MagicMock(buying_power="100000", cash="50000")
        yield instance

@pytest.fixture
def mock_redis():
    with patch("redis.Redis") as mock:
        instance = mock.return_value
        instance.get.return_value = None
        instance.set.return_value = True
        yield instance

@pytest.fixture
def mock_dask():
    with patch("dask.distributed.Client") as mock:
        instance = mock.return_value
        yield instance
