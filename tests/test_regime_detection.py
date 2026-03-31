import unittest
import pandas as pd
import numpy as np
import os
from src.regime_detection.hmm_model import MarketHMM
from src.regime_detection.ms_var_model import MarketMSVAR
from src.regime_detection.processor import RegimeProcessor

class TestRegimeDetection(unittest.TestCase):

    def setUp(self):
        # Create dummy data with 3 regimes (Low, Mid, High vol)
        n = 300
        volatilities = [0.01, 0.05, 0.1]
        returns = []
        for vol in volatilities:
            returns.extend(np.random.normal(0, vol, n // 3))
        
        self.dummy_returns = np.array(returns).reshape(-1, 1)
        self.dummy_df = pd.DataFrame({
            "Close": 100 * (1 + np.cumsum(returns)),
            "High": 101 * (1 + np.cumsum(returns)),
            "Low": 99 * (1 + np.cumsum(returns)),
            "Volume": 1000 * np.random.randn(n)
        })
        # Save dummy data for processor
        os.makedirs("data/test_processed_regime", exist_ok=True)
        self.dummy_df.to_parquet("data/test_processed_regime/SPY_ohlcv.parquet")

    def test_hmm_fit_predict(self):
        hmm_model = MarketHMM(n_components=3)
        hmm_model.fit(self.dummy_returns)
        states = hmm_model.predict(self.dummy_returns)
        self.assertEqual(len(states), len(self.dummy_returns))
        # Transition matrix should be stochastic
        trans_mat = hmm_model.get_transition_matrix()
        self.assertTrue(np.allclose(trans_mat.sum(axis=1), 1.0))

    def test_msvar_fit_predict(self):
        msvar_model = MarketMSVAR(k_regimes=2)
        msvar_model.fit(pd.Series(self.dummy_returns.flatten()))
        probas = msvar_model.predict_regimes()
        # MS-AR(1) loses the first observation due to lag
        self.assertEqual(len(probas), len(self.dummy_returns) - 1)

    def test_processor_end_to_end(self):
        processor = RegimeProcessor(data_dir="data/test_processed_regime", output_dir="data/test_regime")
        df = processor.detect_regimes("SPY")
        self.assertIn("regime", df.columns)
        self.assertIn("regime_prob_0", df.columns)
        self.assertEqual(len(df), len(self.dummy_df))

    def tearDown(self):
        # Cleanup
        for path in ["data/test_processed_regime", "data/test_regime"]:
            if os.path.exists(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
                os.rmdir(path)

if __name__ == "__main__":
    unittest.main()
