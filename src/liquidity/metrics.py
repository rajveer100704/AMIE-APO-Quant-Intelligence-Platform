import numpy as np
import pandas as pd
from src.utils.logger import logger, log_execution_time

class LiquidityMetrics:
    """Core liquidity indicators including Amihud, Spreads, and Book Depth."""

    @staticmethod
    def calculate_amihud(df, window=20):
        """Calculates the Amihud (2002) Illiquidity ratio."""
        # |Return| / (Price * Volume)
        returns = df["Close"].pct_change().abs()
        dollar_volume = df["Close"] * df["Volume"]
        illiquidity = returns / dollar_volume
        return illiquidity.rolling(window=window).mean()

    @staticmethod
    def calculate_spreads(l2_df):
        """Calculates Quoted and Effective Spreads from L2 book data."""
        # Quoted Spread = (Ask1 - Bid1) / Mid
        mid = (l2_df["ask_price_0"] + l2_df["bid_price_0"]) / 2
        quoted_spread = (l2_df["ask_price_0"] - l2_df["bid_price_0"]) / mid
        
        # In a real environment, Effective Spread = 2 * |Trade - Mid| / Mid
        # Since we simulate book data, we focus on the Quoted Spread as a baseline.
        return quoted_spread

    @staticmethod
    def calculate_book_depth(l2_df, levels=10):
        """Calculates total depth available on both sides of the book."""
        bid_cols = [f"bid_size_{i}" for i in range(levels)]
        ask_cols = [f"ask_size_{i}" for i in range(levels)]
        
        bid_depth = l2_df[bid_cols].sum(axis=1)
        ask_depth = l2_df[ask_cols].sum(axis=1)
        
        return bid_depth, ask_depth

if __name__ == "__main__":
    # Test with dummy data
    df = pd.DataFrame({
        "Close": [100, 101, 100.5],
        "Volume": [1000, 1500, 1200]
    })
    print(LiquidityMetrics.calculate_amihud(df, window=1))
