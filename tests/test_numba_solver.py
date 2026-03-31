import unittest
import numpy as np
from src.optimizer.numba_solver import NumbaPortfolioSolver

class TestNumbaSolver(unittest.TestCase):

    def setUp(self):
        self.n = 3
        self.solver = NumbaPortfolioSolver(risk_aversion=3.0)
        np.random.seed(42)
        self.returns = np.random.normal(0, 0.01, (200, self.n))
        self.cov = np.cov(self.returns.T)
        self.expected = np.mean(self.returns, axis=0)

    def test_mean_variance(self):
        w = self.solver.mean_variance(self.expected, self.cov)
        self.assertAlmostEqual(np.sum(w), 1.0, places=4)
        self.assertTrue(all(wi >= -1e-6 for wi in w))

    def test_mean_cvar(self):
        w = self.solver.mean_cvar(self.returns, alpha=0.05)
        self.assertAlmostEqual(np.sum(w), 1.0, places=4)

    def test_risk_parity(self):
        w = self.solver.risk_parity(self.cov)
        self.assertAlmostEqual(np.sum(w), 1.0, places=4)
        # Risk parity: weights should be roughly equal for similar-vol assets
        self.assertTrue(max(w) - min(w) < 0.5)

    def test_black_litterman(self):
        mkt_w = np.ones(self.n) / self.n
        P = np.eye(self.n)
        Q = np.array([0.02, -0.01, 0.01])
        w = self.solver.black_litterman(self.cov, mkt_w, P, Q)
        self.assertAlmostEqual(np.sum(w), 1.0, places=4)

if __name__ == "__main__":
    unittest.main()
