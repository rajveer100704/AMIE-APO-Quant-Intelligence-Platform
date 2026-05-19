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
    loader = YFinanceLoader(["SPY", "VIX"])
    
    # Test fetch_batch
    with pytest.MonkeyPatch().context() as m:
        m.setattr(loader, "fetch_ohlcv", lambda symbol, period="1mo", interval="1m": pd.DataFrame({"Close": [100.0, 101.0]}))
        data = loader.fetch_batch(period="1d", interval="1m")
        assert len(data) == 2
        assert "SPY" in data
        assert "VIX" in data
        
    # Test fetch_ohlcv exception handling
    from unittest.mock import patch
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.side_effect = Exception("Ticker error")
        df_empty = loader.fetch_ohlcv("SPY")
        assert df_empty.empty

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
    """Test DataCleaner normalization, cleaning, and outlier detection."""
    cleaner_z = DataCleaner(normalization="z-score")
    cleaner_mm = DataCleaner(normalization="min-max")
    
    # Test empty DataFrame
    df_empty = pd.DataFrame()
    assert cleaner_z.clean(df_empty).empty
    assert cleaner_z.normalize(df_empty, ["Close"]).empty
    assert cleaner_z.detect_outliers(df_empty, "Close").empty
    
    # Test columns is None or missing
    df_valid = pd.DataFrame({"Close": [100.0, 101.0, 100.5]})
    assert cleaner_z.normalize(df_valid, None).equals(df_valid)
    assert cleaner_z.normalize(df_valid, ["MissingColumn"]).equals(df_valid)
    assert cleaner_z.detect_outliers(df_valid, "MissingColumn").equals(df_valid)
    
    # Test clean handling of Inf and NaN
    df_dirty = pd.DataFrame({"Close": [100.0, float('inf'), 102.0, float('nan'), 101.0]})
    df_cleaned = cleaner_z.clean(df_dirty)
    assert not df_cleaned.isna().any().any()
    assert not (df_cleaned == float('inf')).any().any()
    
    # Test z-score with non-zero std and zero std
    df_z = pd.DataFrame({"Close": [100.0, 102.0, 101.0]})
    df_z = cleaner_z.normalize(df_z, ["Close"])
    assert "Close_norm" in df_z.columns
    
    df_zero_std = pd.DataFrame({"Close": [100.0, 100.0, 100.0]})
    df_zero_std = cleaner_z.normalize(df_zero_std, ["Close"])
    assert "Close_norm" in df_zero_std.columns
    assert (df_zero_std["Close_norm"] == 0).all()
    
    # Test min-max normalization
    df_mm = pd.DataFrame({"Close": [10.0, 20.0, 15.0]})
    df_mm = cleaner_mm.normalize(df_mm, ["Close"])
    assert "Close_norm" in df_mm.columns
    assert df_mm["Close_norm"].iloc[0] == 0.0
    assert df_mm["Close_norm"].iloc[1] == 1.0
    
    df_mm_zero = pd.DataFrame({"Close": [10.0, 10.0]})
    df_mm_zero = cleaner_mm.normalize(df_mm_zero, ["Close"])
    assert (df_mm_zero["Close_norm"] == 0).all()
    
    # Test outlier detection
    df_outliers = pd.DataFrame({"Close": [10.0, 11.0, 10.5, 10.2, 50.0]}) # 50.0 is an outlier
    df_outliers = cleaner_z.detect_outliers(df_outliers, "Close", threshold=1.5)
    assert "Close_is_outlier" in df_outliers.columns
    assert df_outliers["Close_is_outlier"].iloc[4] == True


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
