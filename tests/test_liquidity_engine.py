import unittest
import numpy as np
import pandas as pd
import os
from src.liquidity.metrics import LiquidityMetrics
from src.liquidity.impact_model import MarketImpactModel
from src.liquidity.engine import LiquidityEngine

class TestLiquidityEngine(unittest.TestCase):

    def setUp(self):
        # Create dummy OHLCV
        self.n = 100
        self.ohlcv_df = pd.DataFrame({
            "Close": 100 + np.cumsum(np.random.normal(0, 0.1, self.n)),
            "Volume": 1000 + 500 * np.random.randn(self.n)
        }, index=pd.date_range("2023-01-01", periods=self.n, freq="min"))
        
        # Create dummy L2
        # levels=10
        l2_data = {}
        for i in range(10):
            l2_data[f"bid_price_{i}"] = 100 - 0.01 * (i + 1)
            l2_data[f"bid_size_{i}"] = 100 + 10 * i
            l2_data[f"ask_price_{i}"] = 100 + 0.01 * (i + 1)
            l2_data[f"ask_size_{i}"] = 100 + 10 * i
            
        self.l2_df = pd.DataFrame([l2_data] * self.n, index=self.ohlcv_df.index)
        
        os.makedirs("data/test_processed_liq", exist_ok=True)
        self.ohlcv_df.to_parquet("data/test_processed_liq/SPY_ohlcv.parquet")
        self.l2_df.to_parquet("data/test_processed_liq/SPY_l2.parquet")

    def test_calculate_amihud(self):
        res = LiquidityMetrics.calculate_amihud(self.ohlcv_df, window=5)
        self.assertEqual(len(res), self.n)
        self.assertTrue(res.notna().sum() > 0)

    def test_calculate_spreads(self):
        res = LiquidityMetrics.calculate_spreads(self.l2_df)
        self.assertEqual(len(res), self.n)
        self.assertTrue(np.all(res > 0))

    def test_impact_slippage(self):
        model = MarketImpactModel()
        bps, impact = model.estimate_slippage(1e6, 1e8, 0.2, 450)
        self.assertTrue(bps > 0)
        self.assertTrue(impact > 0)

    def test_liquidity_engine_end_to_end(self):
        engine = LiquidityEngine(data_dir="data/test_processed_liq", output_dir="data/test_liq")
        summary = engine.run_liquidity_analysis("SPY", trade_notional=1000000)
        self.assertEqual(summary["symbol"], "SPY")
        self.assertIn("slippage_bps", summary)
        self.assertTrue(os.path.exists("data/test_liq/SPY_liquidity.parquet"))

    def tearDown(self):
        # Cleanup
        for path in ["data/test_processed_liq", "data/test_liq"]:
            if os.path.exists(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                os.rmdir(path)

if __name__ == "__main__":
    unittest.main()
