import os
import pandas as pd
import numpy as np
from src.optimizer.amis_fusion import AMISFusion
from src.optimizer.solver import PortfolioSolver
from src.utils.logger import logger, log_execution_time

class APOEngine:
    """Adaptive Portfolio Optimizer (APO) orchestrator."""

    def __init__(self, data_dir="data/processed", output_dir="data/portfolio"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.fusion = AMISFusion()
        self.solver = PortfolioSolver()
        logger.info(f"APOEngine initialized. Reading from {data_dir}, writing to {output_dir}")

    @log_execution_time
    def run_rebalancing(self, symbols=None):
        """Runs the AMIS fusion and rebalancing for the portfolio."""
        symbols = symbols or ["SPY", "VIX", "^TNX"]
        logger.info(f"Running APO rebalancing for: {symbols}")
        
        # 1. Compute AMIS for each symbol
        amis_scores = {s: self.fusion.compute_amis(s) for s in symbols}
        
        # 2. Get returns and covariance
        symbol_data = {}
        for s in symbols:
            path = os.path.join(self.data_dir, f"{s}_ohlcv.parquet")
            if os.path.exists(path):
                symbol_data[s] = pd.read_parquet(path)["Close"]
        
        if not symbol_data:
            logger.error("No symbol data found for optimization.")
            return
            
        returns_df = pd.DataFrame(symbol_data).pct_change().dropna()
        cov_matrix = returns_df.cov().values
        
        # 3. Solve for optimal weights
        # We only optimize symbols we have data for
        active_symbols = returns_df.columns.tolist()
        active_amis = {s: amis_scores[s] for s in active_symbols}
        
        weights = self.solver.optimize(returns_df, cov_matrix, active_amis)
        
        # 4. Store results
        allocation = pd.DataFrame([weights], columns=active_symbols)
        # Store AMIS alongside
        for s in active_symbols:
            allocation[f"{s}_amis"] = amis_scores[s]
            
        output_path = os.path.join(self.output_dir, "latest_allocation.parquet")
        allocation.to_parquet(output_path)
        
        logger.info(f"Stored latest portfolio allocation at {output_path}")
        logger.info(f"Final Weights: {dict(zip(active_symbols, weights))}")
        
        return allocation.to_dict(orient='records')[0]

if __name__ == "__main__":
    # Test with dummy data environment
    engine = APOEngine()
    try:
        alloc = engine.run_rebalancing()
        print(f"Optimal Allocation: {alloc}")
    except Exception as e:
        logger.error(f"Error in testing APOEngine: {e}")
