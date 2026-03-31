import os
import pandas as pd
import numpy as np
from src.volatility.garch_model import MarketVolatility
from src.volatility.hawkes_process import MarketHawkes
from src.volatility.monte_carlo import MonteCarloSimulator
from src.utils.logger import logger, log_execution_time

class VolatilityEngine:
    """Orchestrates volatility shock propagation and stress testing."""

    def __init__(self, data_dir="data/processed", output_dir="data/volatility"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        logger.info(f"VolatilityEngine initialized. Reading from {data_dir}, writing to {output_dir}")

    @log_execution_time
    def run_stress_test(self, symbol="SPY", n_paths=10000, n_steps=30):
        """Runs the volatility modeling and stress testing for a symbol."""
        logger.info(f"Running stress test for {symbol}...")
        
        input_path = os.path.join(self.data_dir, f"{symbol}_ohlcv.parquet")
        if not os.path.exists(input_path):
            logger.error(f"Input data not found for {symbol} at {input_path}")
            return

        # Load data
        df = pd.read_parquet(input_path)
        returns = df["Close"].pct_change().dropna()

        # 1. Fit GARCH for conditional volatility
        vol_model = MarketVolatility()
        vol_model.fit(returns)
        cond_vol = vol_model.get_conditional_volatility()
        current_vol = cond_vol[-1] if cond_vol is not None else returns.std()

        # 2. Setup Hawkes jump engine
        hawkes = MarketHawkes() # Baseline parameters
        
        # 3. Simulate Monte Carlo
        sim = MonteCarloSimulator(n_paths=n_paths, n_steps=n_steps)
        # Assuming drift mu=0 for stress testing
        results = sim.run_simulation(df["Close"].iloc[-1], 0, current_vol, jump_engine=hawkes)

        # 4. Store metrics
        metrics = {
            "symbol": symbol,
            "vaR_99": results["vaR_99"],
            "cvaR_99": results["cvaR_99"],
            "cond_vol": current_vol,
            "mean_shock_price": results["mean_price"]
        }
        
        metrics_df = pd.DataFrame([metrics])
        output_path = os.path.join(self.output_dir, f"{symbol}_stress.parquet")
        metrics_df.to_parquet(output_path)
        
        logger.info(f"Stored stress metrics for {symbol} at {output_path}")
        return metrics

if __name__ == "__main__":
    engine = VolatilityEngine()
    # Assume data from Phase 1 exists
    try:
        metrics = engine.run_stress_test("SPY", n_paths=1000)
        print(f"Stress Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error in testing VolatilityEngine: {e}")
