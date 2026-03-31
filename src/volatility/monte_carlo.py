import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src.utils.logger import logger, log_execution_time

class MonteCarloSimulator:
    """Parallelized Monte Carlo simulator for volatility shock propagation."""

    def __init__(self, n_paths=10000, n_steps=100, n_jobs=-1):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.n_jobs = n_jobs
        logger.info(f"MonteCarloSimulator initialized for {n_paths} paths, {n_steps} steps, using {n_jobs} jobs.")

    def simulate_path(self, start_price, mu, vol, jumps=None):
        """Simulates a single price path with drift, volatility, and optional jumps."""
        # Arithmetic Brownian Motion with Jumps
        dt = 1/252 # Annualized step
        returns = np.random.normal(mu * dt, vol * np.sqrt(dt), self.n_steps)
        
        if jumps is not None:
            # Add Hawkes jumps if provided
            returns += jumps[:self.n_steps]
            
        prices = start_price * np.exp(np.cumsum(returns))
        return prices[-1]

    @log_execution_time
    def run_simulation(self, start_price, mu, vol, jump_engine=None):
        """Runs the parallel simulation across multiple paths."""
        logger.info(f"Starting Monte Carlo simulation for {self.n_paths} paths...")
        
        # Pre-generate jumps if jump_engine is provided
        # (Though in a real Hawkes process, jumps might depend on the specific path)
        # For simplicity in this demo, we generate a representative jump series
        jumps = None
        if jump_engine:
            jumps = jump_engine.generate_jumps(self.n_steps)

        # Parallel run
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.simulate_path)(start_price, mu, vol, jumps) 
            for _ in range(self.n_paths)
        )
        
        results = np.array(results)
        
        # Calculate Risk Metrics
        final_returns = (results - start_price) / start_price
        cvaR_99 = np.mean(final_returns[final_returns <= np.percentile(final_returns, 1)])
        vaR_99 = np.percentile(final_returns, 1)

        logger.info(f"Simulation complete. VaR(99%): {vaR_99:.4f}, CVaR(99%): {cvaR_99:.4f}")
        
        return {
            "mean_price": np.mean(results),
            "std_price": np.std(results),
            "vaR_99": vaR_99,
            "cvaR_99": cvaR_99,
            "paths": results
        }

if __name__ == "__main__":
    sim = MonteCarloSimulator(n_paths=5000)
    res = sim.run_simulation(100, 0.05, 0.2)
    print(res["cvaR_99"])
