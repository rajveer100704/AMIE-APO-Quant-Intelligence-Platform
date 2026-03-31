import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils.logger import logger, log_execution_time

class PortfolioSolver:
    """Adaptive Portfolio Optimizer using SciPy for Mean-Variance optimization."""

    def __init__(self, risk_aversion=3.0, transaction_cost=0.0010):
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost
        logger.info(f"PortfolioSolver initialized with λ={risk_aversion}, Cost={transaction_cost}bps.")

    @log_execution_time
    def optimize(self, returns_df, cov_matrix, amis_scores, current_weights=None):
        """Solves the portfolio optimization problem."""
        n = len(returns_df.columns)
        # Expected returns = returns * AMIS (Scale returns by AMIS/100 as a proxy for conviction)
        # In a real system, AMIS could be used as a Black-Litterman prior
        expected_returns = returns_df.mean() * (pd.Series(amis_scores) / 50.0) # AMIS 50 is neutral
        
        # Objective: Maximize Return - λ/2 * Risk - Transaction Costs
        def objective(w):
            port_return = np.dot(w, expected_returns)
            port_risk = 0.5 * self.risk_aversion * np.dot(w.T, np.dot(cov_matrix, w))
            
            # Transaction costs: Σ |w_i - current_w_i| * cost
            # (If absolute weighting is not possible, we assume rebalancing from zero)
            tc = 0
            if current_weights is not None:
                tc = np.sum(np.abs(w - current_weights)) * self.transaction_cost
                
            return -(port_return - port_risk - tc) # Negative for minimization

        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        # Bounds: Long-only (0 to 1)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        
        # Initial guess: Equal weights
        init_w = np.array([1.0/n] * n)
        
        res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not res.success:
            logger.error(f"Optimization failed: {res.message}")
            return init_w
            
        logger.info(f"Optimization successful. Expected Portfolio Return: { -res.fun:.4f}")
        return res.x

if __name__ == "__main__":
    solver = PortfolioSolver()
    # Test with 2 assets
    rets = pd.DataFrame(np.random.normal(0, 0.01, (100, 2)), columns=["SPY", "TLT"])
    cov = rets.cov().values
    amis = {"SPY": 80, "TLT": 20}
    w = solver.optimize(rets, cov, amis)
    print(f"Optimal weights: {w}")
