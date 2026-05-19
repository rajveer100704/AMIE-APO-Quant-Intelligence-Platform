import pytest
import numpy as np
import pandas as pd
import os
from src.liquidity.metrics import LiquidityMetrics
from src.liquidity.impact_model import MarketImpactModel
from src.liquidity.engine import LiquidityEngine

@pytest.fixture
def dummy_liq_data(tmp_path):
    """Create dummy OHLCV and L2 data for liquidity tests."""
    n = 100
    ohlcv_df = pd.DataFrame({
        "Close": 100 + np.cumsum(np.random.normal(0, 0.1, n)),
        "Volume": 1000 + 500 * np.random.randn(n)
    }, index=pd.date_range("2023-01-01", periods=n, freq="min"))
    
    l2_data = {}
    for i in range(10):
        l2_data[f"bid_price_{i}"] = 100 - 0.01 * (i + 1)
        l2_data[f"bid_size_{i}"] = 100 + 10 * i
        l2_data[f"ask_price_{i}"] = 100 + 0.01 * (i + 1)
        l2_data[f"ask_size_{i}"] = 100 + 10 * i
        
    l2_df = pd.DataFrame([l2_data] * n, index=ohlcv_df.index)
    
    data_dir = tmp_path / "data_liq"
    data_dir.mkdir()
    ohlcv_df.to_parquet(data_dir / "SPY_ohlcv.parquet")
    l2_df.to_parquet(data_dir / "SPY_l2.parquet")
    
    return ohlcv_df, l2_df, str(data_dir)

@pytest.mark.unit
def test_calculate_amihud(dummy_liq_data):
    ohlcv, l2, data_dir = dummy_liq_data
    res = LiquidityMetrics.calculate_amihud(ohlcv, window=5)
    assert len(res) == len(ohlcv)
    assert res.notna().sum() > 0

@pytest.mark.unit
def test_calculate_spreads(dummy_liq_data):
    ohlcv, l2, data_dir = dummy_liq_data
    res = LiquidityMetrics.calculate_spreads(l2)
    assert len(res) == len(ohlcv)
    assert np.all(res > 0)

@pytest.mark.unit
def test_impact_slippage():
    model = MarketImpactModel()
    bps, impact = model.estimate_slippage(1e6, 1e8, 0.2, 450)
    assert bps > 0
    assert impact > 0

    # Test fit_kyle_lambda with valid dataframe
    df = pd.DataFrame({
        "Close": [100.0, 101.0, 102.0, 101.5],
        "Volume": [1000.0, 1200.0, 1100.0, 950.0]
    })
    lambd = model.fit_kyle_lambda(df)
    assert lambd != 1.0e-6 # should be updated from default

    # Test fit_kyle_lambda with insufficient data
    df_short = pd.DataFrame({
        "Close": [100.0],
        "Volume": [1000.0]
    })
    lambd_short = model.fit_kyle_lambda(df_short)
    assert lambd_short == lambd # should fall back to previous fitted value


@pytest.mark.integration
def test_liquidity_engine_end_to_end(dummy_liq_data, tmp_path):
    ohlcv, l2, data_dir = dummy_liq_data
    output_dir = tmp_path / "output_liq"
    output_dir.mkdir()
    
    engine = LiquidityEngine(data_dir=data_dir, output_dir=str(output_dir))
    summary = engine.run_liquidity_analysis("SPY", trade_notional=1000000)
    
    assert summary["symbol"] == "SPY"
    assert "slippage_bps" in summary
    assert os.path.exists(os.path.join(str(output_dir), "SPY_liquidity.parquet"))
