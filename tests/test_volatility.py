import pytest
import numpy as np
import pandas as pd
import os
from src.volatility.garch_model import MarketVolatility
from src.volatility.hawkes_process import MarketHawkes
from src.volatility.monte_carlo import MonteCarloSimulator
from src.volatility.engine import VolatilityEngine

@pytest.fixture
def dummy_vol_data(tmp_path):
    """Create dummy returns and OHLCV data for volatility tests."""
    n = 500
    returns = np.random.normal(0, 0.01, n)
    df = pd.DataFrame({
        "Close": 100 * (1 + np.cumsum(returns)),
        "High": 101 * (1 + np.cumsum(returns)),
        "Low": 99 * (1 + np.cumsum(returns)),
        "Volume": 1000 * np.random.randn(n)
    }, index=pd.date_range("2023-01-01", periods=n, freq="D"))
    
    data_dir = tmp_path / "data_vol"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY_ohlcv.parquet")
    
    return returns, df, str(data_dir)

@pytest.mark.unit
def test_garch_fit_forecast(dummy_vol_data):
    returns, df, data_dir = dummy_vol_data
    vol_model = MarketVolatility()
    vol_model.fit(returns)
    forecast = vol_model.forecast(5)
    assert len(forecast) == 5
    assert np.all(forecast > 0)

@pytest.mark.unit
def test_hawkes_jumps():
    hawkes = MarketHawkes()
    jumps = hawkes.generate_jumps(100)
    assert len(jumps) == 100

@pytest.mark.unit
def test_monte_carlo_parallel():
    sim = MonteCarloSimulator(n_paths=1000, n_steps=10)
    results = sim.run_simulation(100, 0, 0.2)
    assert "cvaR_99" in results
    assert "paths" in results
    assert len(results["paths"]) == 1000

@pytest.mark.integration
def test_volatility_engine_end_to_end(dummy_vol_data, tmp_path):
    returns, df, data_dir = dummy_vol_data
    output_dir = tmp_path / "output_vol"
    output_dir.mkdir()
    
    engine = VolatilityEngine(data_dir=data_dir, output_dir=str(output_dir))
    metrics = engine.run_stress_test("SPY", n_paths=500, n_steps=10)
    
    assert "vaR_99" in metrics
    assert metrics["symbol"] == "SPY"
    assert os.path.exists(os.path.join(str(output_dir), "SPY_stress.parquet"))
