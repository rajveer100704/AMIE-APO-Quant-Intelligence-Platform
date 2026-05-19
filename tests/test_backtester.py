import pytest
import numpy as np
import pandas as pd
from src.optimizer.backtester import HighFrequencyBacktester

@pytest.mark.unit
def test_backtester_initialization():
    backtester = HighFrequencyBacktester(initial_capital=500000)
    assert backtester.capital == 500000
    assert backtester.holdings == {}
    assert backtester.history == []

@pytest.mark.unit
def test_backtester_calculate_performance():
    backtester = HighFrequencyBacktester()
    
    # Create a dummy equity curve
    dates = pd.date_range("2024-01-01", periods=10, freq="min")
    equity_curve = pd.Series([1000000, 1001000, 1002000, 999000, 998000, 1003000, 1004000, 1005000, 1002000, 1010000], index=dates)
    
    metrics = backtester.calculate_performance(equity_curve)
    
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "calmar" in metrics
    assert "total_return" in metrics
    assert metrics["total_return"] == pytest.approx(0.01)

@pytest.mark.unit
def test_run_backtest_stub():
    # Call run_backtest to cover it (it contains stubs but we should ensure it runs without crashing)
    backtester = HighFrequencyBacktester()
    
    dates = pd.date_range("2024-01-01", periods=3, freq="min")
    ohlcv_dfs = {
        "SPY": pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)
    }
    weights_series = pd.DataFrame({"SPY": [1.0, 1.0, 1.0]}, index=dates)
    
    # Run the backtest (should not throw exceptions)
    backtester.run_backtest(ohlcv_dfs, weights_series)
