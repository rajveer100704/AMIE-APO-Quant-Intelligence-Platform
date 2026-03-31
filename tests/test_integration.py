import pytest
import numpy as np
import pandas as pd
import os
from src.data_ingestion.pipeline import DataPipeline
from src.optimizer.engine import APOEngine
from src.execution.risk_guard import RiskGuard
from src.execution.order_manager import OrderManager

@pytest.mark.integration
def test_full_pipeline_ingestion_to_order(tmp_path, mock_alpaca):
    """
    Elite Fix 10: Integration test asserting full flow correctness.
    """
    data_dir = str(tmp_path / "data")
    processed_dir = str(tmp_path / "processed")
    portfolio_dir = str(tmp_path / "portfolio")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(portfolio_dir, exist_ok=True)

    # 1. Pipeline Ingestion (Mocked Data)
    symbols = ["SPY", "VIX"]
    pipeline = DataPipeline(symbols, processed_dir)
    # Patch yfinance fetching
    with patch("yfinance.download") as mock_yf:
        df = pd.DataFrame({
            "Open": [100.0, 101.0, 102.0],
            "High": [101.0, 102.0, 103.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [100.0, 101.0, 102.0],
            "Volume": [1000, 1100, 1200]
        }, index=pd.date_range("2024-01-01", periods=3))
        mock_yf.return_value = df
        pipeline.run_ingestion(period="1d", interval="1h")
        
    # 2. Simulation of Intermediate Steps (Regime, Vol, Liq files)
    # For a true full integration, we would run those processors, but we'll mock the files they output for speed
    regime_dir = str(tmp_path / "regime")
    vol_dir = str(tmp_path / "vol")
    liq_dir = str(tmp_path / "liq")
    for d in [regime_dir, vol_dir, liq_dir]:
        os.makedirs(d, exist_ok=True)
        for s in symbols:
            if "regime" in d: pd.DataFrame({"regime": [2]*3}).to_parquet(os.path.join(d, f"{s}_regime.parquet"))
            if "vol" in d: pd.DataFrame({"cvaR_99": [-0.01]*3}).to_parquet(os.path.join(d, f"{s}_stress.parquet"))
            if "liq" in d: pd.DataFrame({"quoted_spread": [0.0001]*3}).to_parquet(os.path.join(d, f"{s}_liquidity.parquet"))

    # 3. APO Engine: AMIS -> Optimizer
    engine = APOEngine(data_dir=processed_dir, output_dir=portfolio_dir)
    engine.fusion.data_dirs = {"regime": regime_dir, "volatility": vol_dir, "liquidity": liq_dir}
    
    # Assert correctness
    alloc = engine.run_rebalancing(symbols=symbols)
    
    # Assert AMIS scores within range
    for s in symbols:
        assert 0 <= alloc[f"{s}_amis"] <= 100
        assert alloc[f"{s}_amis"] >= 0 # Elite Fix
        
    # Assert total weight is 1.0
    total_weight = sum([alloc[s] for s in symbols])
    assert abs(total_weight - 1.0) < 1e-6 # Elite Fix
    
    # 4. Risk Guard & Order Manager
    guard = RiskGuard()
    manager = OrderManager()
    
    orders = []
    for s in symbols:
        weight = alloc[s]
        risk_res = guard.validate_order(s, weight, 0.01, 10)
        if risk_res["status"] == "APPROVED":
            order = manager.create_order(s, weight)
            orders.append(order)
            
    assert len(orders) == len(symbols) # Should be approved given the low-risk setup
    assert all(o["status"] == "PENDING" for o in orders) # Elite Fix
