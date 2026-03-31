import unittest
import pandas as pd
import os
import pyarrow.parquet as pq
from src.data_ingestion.loaders.yfinance_loader import YFinanceLoader
from src.data_ingestion.loaders.orderbook_sim import OrderBookSimulator
from src.data_ingestion.processors.cleaner import DataCleaner
from src.data_ingestion.pipeline import DataPipeline

class TestDataLayer(unittest.TestCase):

    def setUp(self):
        self.symbols = ["SPY"]
        self.data_dir = "data/test_processed"
        self.pipeline = DataPipeline(self.symbols, self.data_dir)

    def test_yfinance_fetch(self):
        loader = YFinanceLoader(["SPY"])
        df = loader.fetch_ohlcv("SPY", period="1d", interval="1m")
        self.assertFalse(df.empty, "OHLCV data should not be empty")
        self.assertIn("Close", df.columns)

    def test_orderbook_sim(self):
        sim = OrderBookSimulator("SPY", levels=5)
        snapshot = sim.generate_snapshot(450.0)
        self.assertEqual(len(snapshot), 10, "L2 snapshot should have 10 rows (5 bids, 5 asks)")
        self.assertIn("price", snapshot.columns)
        self.assertIn("volume", snapshot.columns)

    def test_cleaner(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({"Close": [100.0, 101.0, 100.5]})
        df = cleaner.normalize(df, ["Close"])
        self.assertIn("Close_norm", df.columns)

    def test_pipeline_execution(self):
        self.pipeline.run_ingestion(period="1d", interval="5m")
        ohlcv_path = os.path.join(self.data_dir, "SPY_ohlcv.parquet")
        l2_path = os.path.join(self.data_dir, "SPY_l2.parquet")
        self.assertTrue(os.path.exists(ohlcv_path), "OHLCV Parquet file should exist")
        self.assertTrue(os.path.exists(l2_path), "L2 Parquet file should exist")

        # Verify data schema
        table = pq.read_table(ohlcv_path)
        self.assertIn("Close", table.column_names)

    def tearDown(self):
        # Cleanup test data
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                os.remove(os.path.join(self.data_dir, file))
            os.rmdir(self.data_dir)

if __name__ == "__main__":
    unittest.main()
