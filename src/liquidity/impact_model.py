import numpy as np
import pandas as pd
from src.utils.logger import logger, log_execution_time

class MarketImpactModel:
    """Models price impact and estimated slippage for varying trade sizes."""

    def __init__(self, lambda_kyle=None):
        self.lambda_kyle = lambda_kyle or 1.0e-6 # Default small lambda
        logger.info(f"MarketImpactModel initialized with lambda={self.lambda_kyle}")

    @log_execution_time
    def estimate_slippage(self, order_size, avg_daily_volume, vol, price):
        """Estimates slippage using a square-root impact model."""
        # Slippage = Y * Vol * sqrt(OrderSize / ADV)
        # Y is usually between 0.1 and 1.0 (coefficient of impact)
        y = 0.5
        adv_ratio = order_size / avg_daily_volume
        estimated_slippage_bps = y * (vol * 100) * np.sqrt(adv_ratio)
        
        # Convert to price impact
        price_impact = price * (estimated_slippage_bps / 10000)
        return estimated_slippage_bps, price_impact

    @log_execution_time
    def fit_kyle_lambda(self, df):
        """Fits Kyle's Lambda (price change vs. order flow imbalance)."""
        # Since we simulate book data, we use Volume as a proxy for order flow.
        # DeltaPrice = Lambda * (Volume * Sign(Return))
        delta_p = df["Close"].diff()
        # Proxy flow: Volume * Sign of Return
        flow = df["Volume"] * np.sign(df["Close"].diff())
        
        # Simple OLS without intercept
        valid = (delta_p.notna()) & (flow.notna()) & (flow != 0)
        if valid.sum() < 2:
            return self.lambda_kyle
            
        lambd = np.dot(delta_p[valid], flow[valid]) / np.dot(flow[valid], flow[valid])
        self.lambda_kyle = lambd
        logger.info(f"Fitted Kyle's Lambda: {self.lambda_kyle:.2e}")
        return self.lambda_kyle

if __name__ == "__main__":
    model = MarketImpactModel()
    # 1M SPY trade given 100M ADV
    bps, impact = model.estimate_slippage(1e6, 1e8, 0.2, 450)
    print(f"Estimated Slippage (bps): {bps:.2f}")
