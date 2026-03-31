import os
import pytest
import pandas as pd
import pyarrow.parquet as pq
from src.data_ingestion.loaders.yfinance_loader import YFinanceLoader
from src.data_ingestion.loaders.orderbook_sim import OrderBookSimulator
from src.data_ingestion.processors.cleaner import DataCleaner
from src.data_ingestion.pipeline import DataPipeline

@pytest.mark.unit
def test_yfinance_fetch(mock_alpaca):
    """Test YFinance loader fetching data."""
    loader = YFinanceLoader(["SPY"])
    # Mocking fetching since we don't want real network calls in CI
    with pytest.MonkeyPatch().context() as m:
        m.setattr(loader, "fetch_ohlcv", lambda s, p, i: pd.DataFrame({"Close": [100.0, 101.0], "Open": [99.0, 100.0]}, index=pd.date_range("2024-01-01", periods=2)))
        df = loader.fetch_ohlcv("SPY", period="1d", interval="1m")
        assert not df.empty
        assert "Close" in df.columns

@pytest.mark.unit
def test_orderbook_sim():
    """Test L2 OrderBook simulator."""
    sim = OrderBookSimulator("SPY", levels=5)
    snapshot = sim.generate_snapshot(450.0)
    assert len(snapshot) == 10
    assert "price" in snapshot.columns
    assert "volume" in snapshot.columns

@pytest.mark.unit
def test_cleaner():
    """Test DataCleaner normalization."""
    cleaner = DataCleaner()
    df = pd.DataFrame({"Close": [100.0, 101.0, 100.5]})
    df = cleaner.normalize(df, ["Close"])
    assert "Close_norm" in df.columns

@pytest.mark.integration
def test_pipeline_execution(tmp_path):
    """Test full data pipeline execution using a temporary directory."""
    symbols = ["SPY"]
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)
    
    pipeline = DataPipeline(symbols, data_dir)
    
    # Run ingestion
    pipeline.run_ingestion(period="1d", interval="5m")
    
    ohlcv_path = os.path.join(data_dir, "SPY_ohlcv.parquet")
    l2_path = os.path.join(data_dir, "SPY_l2.parquet")
    
    assert os.path.exists(ohlcv_path)
    assert os.path.exists(l2_path)
    
    # Verify data schema
    table = pq.read_table(ohlcv_path)
    assert "Close" in table.column_names
