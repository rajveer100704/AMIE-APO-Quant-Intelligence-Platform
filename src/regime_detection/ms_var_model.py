import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from src.utils.logger import logger, log_execution_time

class MarketMSVAR:
    """Markov-Switching Autoregression for regime detection."""

    def __init__(self, k_regimes=3, order=1, switching_variance=True):
        self.k_regimes = k_regimes
        self.order = order
        self.switching_variance = switching_variance
        self.model = None
        self.results = None
        logger.info(f"MarketMSVAR initialized with {k_regimes} regimes and AR({order}).")

    @log_execution_time
    def fit(self, data):
        """Fits the Markov-Switching model on the provided time series."""
        logger.info(f"Fitting MS-AR on {len(data)} samples...")
        try:
            # We use MarkovAutoregression as a proxy for MS-VAR if using multiple series 
            # or just MS-AR for single series.
            self.model = MarkovAutoregression(
                data, 
                k_regimes=self.k_regimes, 
                order=self.order, 
                switching_variance=self.switching_variance
            )
            self.results = self.model.fit(iter=1000, disp=False)
            logger.info("MS-AR fitting complete.")
        except Exception as e:
            logger.error(f"Error fitting MS-AR: {e}")
            raise

    @log_execution_time
    def predict_regimes(self):
        """Returns the smoothed probabilities of the regimes."""
        if self.results is None:
            raise ValueError("Model must be fitted before prediction.")
        
        # Smoothed probabilities
        return self.results.smoothed_marginal_probabilities

    def get_transition_matrix(self):
        """Returns the transition matrix."""
        if self.results is None:
            return None
        # statsmodels stores transition params in regime_transition
        return self.results.regime_transition


if __name__ == "__main__":
    # Test with random walk + regime shifts
    n = 500
    data = np.zeros(n)
    for i in range(1, n):
        if i < 200:
            data[i] = 0.5 * data[i-1] + np.random.normal(0, 1)
        elif i < 400:
            data[i] = -0.5 * data[i-1] + np.random.normal(0, 5) # High vol regime
        else:
            data[i] = 0.8 * data[i-1] + np.random.normal(0, 0.5)
            
    model = MarketMSVAR(k_regimes=3)
    model.fit(pd.Series(data))
    probas = model.predict_regimes()
    print(f"Probabilities Head:\n{probas.head()}")
    print(f"Transition Matrix:\n{model.get_transition_matrix()}")
