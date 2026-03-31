import unittest
import numpy as np
import pandas as pd
import os
from src.volatility.garch_model import MarketVolatility
from src.volatility.hawkes_process import MarketHawkes
from src.volatility.monte_carlo import MonteCarloSimulator
from src.volatility.engine import VolatilityEngine

class TestVolatilityEngine(unittest.TestCase):

    def setUp(self):
        # Create dummy returns
        self.n = 500
        self.returns = np.random.normal(0, 0.01, self.n)
        self.dummy_df = pd.DataFrame({
            "Close": 100 * (1 + np.cumsum(self.returns)),
            "High": 101 * (1 + np.cumsum(self.returns)),
            "Low": 99 * (1 + np.cumsum(self.returns)),
            "Volume": 1000 * np.random.randn(self.n)
        }, index=pd.date_range("2023-01-01", periods=self.n, freq="D"))
        
        os.makedirs("data/test_processed_vol", exist_ok=True)
        self.dummy_df.to_parquet("data/test_processed_vol/SPY_ohlcv.parquet")

    def test_garch_fit_forecast(self):
        vol_model = MarketVolatility()
        vol_model.fit(self.returns)
        forecast = vol_model.forecast(5)
        self.assertTrue(len(forecast) == 5)
        self.assertTrue(np.all(forecast > 0))

    def test_hawkes_jumps(self):
        hawkes = MarketHawkes()
        jumps = hawkes.generate_jumps(100)
        self.assertEqual(len(jumps), 100)

    def test_monte_carlo_parallel(self):
        sim = MonteCarloSimulator(n_paths=1000, n_steps=10)
        results = sim.run_simulation(100, 0, 0.2)
        self.assertIn("cvaR_99", results)
        self.assertIn("paths", results)
        self.assertEqual(len(results["paths"]), 1000)

    def test_volatility_engine_end_to_end(self):
        engine = VolatilityEngine(data_dir="data/test_processed_vol", output_dir="data/test_vol")
        metrics = engine.run_stress_test("SPY", n_paths=500, n_steps=10)
        self.assertIn("vaR_99", metrics)
        self.assertEqual(metrics["symbol"], "SPY")
        self.assertTrue(os.path.exists("data/test_vol/SPY_stress.parquet"))

    def tearDown(self):
        # Cleanup
        for path in ["data/test_processed_vol", "data/test_vol"]:
            if os.path.exists(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                os.rmdir(path)

if __name__ == "__main__":
    unittest.main()
