import unittest
import numpy as np
import pandas as pd
import os
from src.optimizer.amis_fusion import AMISFusion
from src.optimizer.solver import PortfolioSolver
from src.optimizer.engine import APOEngine

class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        # Create dummy data for all phases
        self.n = 100
        self.symbols = ["SPY", "TLT"]
        
        # 1. Regime
        os.makedirs("data/test_regime", exist_ok=True)
        pd.DataFrame({"regime": [2]*self.n}).to_parquet("data/test_regime/SPY_regime.parquet")
        pd.DataFrame({"regime": [1]*self.n}).to_parquet("data/test_regime/TLT_regime.parquet")
        
        # 2. Volatility
        os.makedirs("data/test_vol", exist_ok=True)
        pd.DataFrame({"cvaR_99": [-0.01]*self.n}).to_parquet("data/test_vol/SPY_stress.parquet")
        pd.DataFrame({"cvaR_99": [-0.03]*self.n}).to_parquet("data/test_vol/TLT_stress.parquet")
        
        # 3. Liquidity
        os.makedirs("data/test_liq", exist_ok=True)
        pd.DataFrame({"quoted_spread": [0.0001]*self.n}).to_parquet("data/test_liq/SPY_liquidity.parquet")
        pd.DataFrame({"quoted_spread": [0.0005]*self.n}).to_parquet("data/test_liq/TLT_liquidity.parquet")
        
        # 4. OHLCV (Prices)
        os.makedirs("data/test_processed_apo", exist_ok=True)
        for s in self.symbols:
            rets = np.random.normal(0, 0.01, self.n)
            pd.DataFrame({"Close": 100 * (1 + np.cumsum(rets))}).to_parquet(f"data/test_processed_apo/{s}_ohlcv.parquet")

    def test_amis_fusion(self):
        fusion = AMISFusion(data_dirs={
            "regime": "data/test_regime",
            "volatility": "data/test_vol",
            "liquidity": "data/test_liq"
        })
        score = fusion.compute_amis("SPY")
        self.assertTrue(0 <= score <= 100)
        # SPY should have higher score than TLT given the setup
        spy_score = fusion.compute_amis("SPY")
        tlt_score = fusion.compute_amis("TLT")
        self.assertTrue(spy_score > tlt_score)

    def test_portfolio_solver(self):
        solver = PortfolioSolver()
        n_assets = 2
        np.random.seed(42)
        returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 2)), columns=["A", "B"])
        cov = returns.cov().values
        amis = {"A": 100, "B": 0}
        weights = solver.optimize(returns, cov, amis)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)
        self.assertTrue(all(w >= -1e-7 for w in weights))

    def test_apo_engine_end_to_end(self):
        # We need to monkey-patch or configure the directories
        engine = APOEngine(data_dir="data/test_processed_apo", output_dir="data/test_portfolio")
        engine.fusion.data_dirs = {
            "regime": "data/test_regime",
            "volatility": "data/test_vol",
            "liquidity": "data/test_liq"
        }
        alloc = engine.run_rebalancing(symbols=self.symbols)
        self.assertIn("SPY", alloc)
        self.assertIn("TLT", alloc)
        self.assertTrue(os.path.exists("data/test_portfolio/latest_allocation.parquet"))

    def tearDown(self):
        # Cleanup
        for path in ["data/test_regime", "data/test_vol", "data/test_liq", "data/test_processed_apo", "data/test_portfolio"]:
            if os.path.exists(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                os.rmdir(path)

if __name__ == "__main__":
    unittest.main()
