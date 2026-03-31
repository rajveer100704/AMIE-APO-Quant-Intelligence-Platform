import numpy as np
import pandas as pd
from arch import arch_model
from src.utils.logger import logger, log_execution_time

class MarketVolatility:
    """GARCH/EGARCH models for conditional volatility forecasting."""

    def __init__(self, model_type="GARCH", p=1, q=1, dist="normal"):
        self.model_type = model_type
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.results = None
        logger.info(f"MarketVolatility initialized with {model_type}({p},{q}) and {dist} distribution.")

    @log_execution_time
    def fit(self, returns):
        """Fits the GARCH model on the provided returns."""
        # Scale returns to improve convergence
        scaled_returns = returns * 100
        logger.info(f"Fitting {self.model_type} on {len(returns)} samples...")
        
        try:
            self.model = arch_model(
                scaled_returns, 
                vol=self.model_type, 
                p=self.p, 
                q=self.q, 
                dist=self.dist,
                rescale=False
            )
            self.results = self.model.fit(disp="off")
            logger.info(f"Volatility fitting complete. Log-likelihood: {self.results.loglikelihood:.2f}")
        except Exception as e:
            logger.error(f"Error fitting volatility model: {e}")
            raise

    @log_execution_time
    def forecast(self, horizon=5):
        """Forecasts future volatility over the given horizon."""
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting.")
        
        forecasts = self.results.forecast(horizon=horizon)
        # Rescale back
        variance = forecasts.variance.values[-1] / (100**2)
        return np.sqrt(variance)

    def get_conditional_volatility(self):
        """Returns the in-sample conditional volatility."""
        if self.results is None:
            return None
        return self.results.conditional_volatility / 100

if __name__ == "__main__":
    # Test with random returns
    returns = np.random.normal(0, 0.01, 1000)
    vol_model = MarketVolatility()
    vol_model.fit(returns)
    forecast = vol_model.forecast(5)
    print(f"Forecasted Volatility: {forecast}")
