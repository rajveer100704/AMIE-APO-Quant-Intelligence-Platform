import numpy as np
import pandas as pd
from src.utils.logger import logger, log_execution_time

class MarketHawkes:
    """Hawkes process for self-exciting jump diffusion modeling."""

    def __init__(self, mu=0.01, alpha=0.5, beta=1.0):
        self.mu = mu        # Baseline intensity
        self.alpha = alpha  # Jump size (shocks)
        self.beta = beta    # Decay rate
        logger.info(f"MarketHawkes initialized with mu={mu}, alpha={alpha}, beta={beta}")

    @log_execution_time
    def simulate_intensity(self, events, times):
        """Simulates intensity lambda(t) based on historic events with fallback."""
        try:
            # Events: magnitude of shocks
            # Times: time of shocks (indices or relative timestamps)
            def lambda_t(t, events, times):
                intensity = self.mu
                shocks = events[times < t]
                stimes = times[times < t]
                if len(shocks) > 0:
                    intensity += np.sum(shocks * self.alpha * np.exp(-self.beta * (t - stimes)))
                return intensity

            return np.array([lambda_t(t, events, times) for t in range(len(times))])
        except Exception as e:
            logger.error("Hawkes intensity simulation failed, using fallback", error=str(e))
            # Fallback: Constant baseline intensity
            return np.full(len(times), self.mu)

    @log_execution_time
    def generate_jumps(self, n_steps, threshold=0.95):
        """Generates a series of jumps with fallback logic."""
        try:
            jumps = np.zeros(n_steps)
            intensity = self.mu
            
            for t in range(1, n_steps):
                # Intensity evolves
                intensity = self.mu + (intensity - self.mu) * np.exp(-self.beta)
                
                # Probability of a jump at time t
                if np.random.rand() < (1 - np.exp(-intensity)):
                    jump_magnitude = np.random.pareto(3) * 0.05
                    jumps[t] = jump_magnitude
                    intensity += jump_magnitude * self.alpha
                    
            return jumps
        except Exception as e:
            logger.error("Hawkes jump generation failed, using fallback", error=str(e))
            # Fallback: Random normal jumps (white noise)
            return np.random.normal(0, 0.01, n_steps)

if __name__ == "__main__":
    hawkes = MarketHawkes()
    jumps = hawkes.generate_jumps(1000)
    print(f"Total jumps: {np.count_nonzero(jumps)}")
    print(f"Max jump: {np.max(jumps)}")
