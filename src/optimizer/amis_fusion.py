import os
import pandas as pd
import numpy as np
from src.utils.logger import logger, log_execution_time

class AMISFusion:
    """Fuses Regime, Volatility, and Liquidity signals into the Adaptive Market Intelligence Score."""

    def __init__(self, data_dirs=None):
        self.data_dirs = data_dirs or {
            "regime": "data/regime",
            "volatility": "data/volatility",
            "liquidity": "data/liquidity"
        }
        logger.info(f"AMISFusion initialized with directories: {self.data_dirs}")

    @log_execution_time
    def compute_amis(self, symbol="SPY"):
        """Computes the AMIS (0-100) for a given symbol."""
        logger.info(f"Computing AMIS for {symbol}...")
        
        # 1. Load Regime (0=Crisis, 1=Neutral, 2=Bullish)
        regime_path = os.path.join(self.data_dirs["regime"], f"{symbol}_regime.parquet")
        # 2. Load Volatility (CVaR/VaR)
        vol_path = os.path.join(self.data_dirs["volatility"], f"{symbol}_stress.parquet")
        # 3. Load Liquidity (Slippage/Spread)
        liq_path = os.path.join(self.data_dirs["liquidity"], f"{symbol}_liquidity.parquet")

        # Fallback values if files are missing
        regime_score = 50
        vol_score = 50
        liq_score = 50

        if os.path.exists(regime_path):
            regime_df = pd.read_parquet(regime_path)
            last_state = regime_df["regime"].iloc[-1]
            regime_score = {0: 0, 1: 50, 2: 100}.get(last_state, 50)

        if os.path.exists(vol_path):
            vol_df = pd.read_parquet(vol_path)
            cvar = vol_df["cvaR_99"].iloc[-1]
            # Normalize CVaR: Assume -0.05 (5% drop) is 0 score, 0 is 100 score
            vol_score = np.clip(100 * (1 + (cvar / 0.05)), 0, 100)

        if os.path.exists(liq_path):
            liq_df = pd.read_parquet(liq_path)
            spread = liq_df["quoted_spread"].iloc[-1]
            # Normalize Spread: Assume 20bps is 0 score, 1bp is 100 score
            liq_score = np.clip(100 * (1 - (spread / 0.0020)), 0, 100)

        # Final Fusion: 40% Regime, 30% Vol, 30% Liq
        amis = (0.40 * regime_score) + (0.30 * vol_score) + (0.30 * liq_score)
        
        # Guard against NaN
        if np.isnan(amis):
            amis = 50.0
        
        logger.info(f"AMIS for {symbol}: {amis:.2f} (R:{regime_score:.0f}, V:{vol_score:.0f}, L:{liq_score:.0f})")
        return float(amis)

if __name__ == "__main__":
    fusion = AMISFusion()
    # Test with defaults
    print(f"Test AMIS: {fusion.compute_amis('SPY')}")
