import os
import pandas as pd
import numpy as np
from src.liquidity.metrics import LiquidityMetrics
from src.liquidity.impact_model import MarketImpactModel
from src.utils.logger import logger, log_execution_time

class LiquidityEngine:
    """Orchestrates liquidity Heatmap generation and execution cost analysis."""

    def __init__(self, data_dir="data/processed", output_dir="data/liquidity"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        logger.info(f"LiquidityEngine initialized. Reading from {data_dir}, writing to {output_dir}")

    @log_execution_time
    def run_liquidity_analysis(self, symbol="SPY", trade_notional=1000000):
        """Generates the liquidity heatmap and execution impact metrics for a symbol."""
        logger.info(f"Running liquidity analysis for {symbol}...")
        
        ohlcv_path = os.path.join(self.data_dir, f"{symbol}_ohlcv.parquet")
        l2_path = os.path.join(self.data_dir, f"{symbol}_l2.parquet")
        
        if not os.path.exists(ohlcv_path) or not os.path.exists(l2_path):
            logger.error(f"Input data not found for {symbol}")
            return

        # Load data
        df = pd.read_parquet(ohlcv_path)
        l2_df = pd.read_parquet(l2_path)

        # 1. Metrics Calculation
        amihud = LiquidityMetrics.calculate_amihud(df)
        spreads = LiquidityMetrics.calculate_spreads(l2_df)
        bid_depth, ask_depth = LiquidityMetrics.calculate_book_depth(l2_df)
        
        # 2. Impact Modeling
        impact_model = MarketImpactModel()
        current_price = df["Close"].iloc[-1]
        # ADV Proxy from Volume
        adv = df["Volume"].mean()
        # Volatility Proxy
        vol = df["Close"].pct_change().std()
        
        bps, impact = impact_model.estimate_slippage(trade_notional, adv, vol, current_price)

        # 3. Create Heatmap Signal (Proxying Heatmap as rolling illiquidity)
        # In a real UI this would be 2D (Price level vs Time), but here we save raw metrics.
        results_df = pd.DataFrame({
            "amihud": amihud,
            "quoted_spread": spreads,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
        })
        
        output_path = os.path.join(self.output_dir, f"{symbol}_liquidity.parquet")
        results_df.to_parquet(output_path)
        
        summary = {
            "symbol": symbol,
            "slippage_bps": bps,
            "price_impact": impact,
            "mean_spread": spreads.mean(),
            "mean_depth": (bid_depth.mean() + ask_depth.mean()) / 2
        }
        
        logger.info(f"Stored liquidity results for {symbol} at {output_path}")
        return summary

if __name__ == "__main__":
    engine = LiquidityEngine()
    try:
        summary = engine.run_liquidity_analysis("SPY")
        print(f"Liquidity Summary: {summary}")
    except Exception as e:
        logger.error(f"Error in testing LiquidityEngine: {e}")
