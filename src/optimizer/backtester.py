import numpy as np
import pandas as pd
from src.utils.logger import logger, log_execution_time

class HighFrequencyBacktester:
    """HF Backtester for evaluating the Adaptive Portfolio strategy."""

    def __init__(self, initial_capital=1000000):
        self.capital = initial_capital
        self.holdings = {}
        self.history = []
        logger.info(f"Backtester initialized with ${initial_capital:,}.")

    @log_execution_time
    def run_backtest(self, ohlcv_dfs, weights_series, transaction_costs=None):
        """Runs the backtest on multiple symbols simultaneously."""
        # Align all assets (assume minutely data)
        symbols = list(ohlcv_dfs.keys())
        first_symbol = symbols[0]
        dates = ohlcv_dfs[first_symbol].index
        
        capital = self.capital
        # weights_series is expected to be a DataFrame (Date x Symbol)
        
        for date in dates:
            # Rebalance
            target_weights = weights_series.loc[date]
            current_prices = {s: ohlcv_dfs[s].loc[date, "Close"] for s in symbols}
            
            # Simple assumption: Daily rebalancing or per-bar rebalancing
            # Calculate PnL (Change in capital from asset returns)
            # In a real backtester, this is done based on shares held.
            # Here we simplify with continuous rebalancing.
            port_return = 0
            for s in symbols:
                asset_ret = ohlcv_dfs[s].loc[date, "Close"].pct_change() # We'd need actual return
                # Since we are iterating bar by bar, let's use actual bar-return
                # (For simplicity we assume daily rebalance for demo)
                pass
            
            # Let's use a more robust logic for the "Engine" to call rebalance periodicially
            pass

    def calculate_performance(self, equity_curve):
        """Computes performance metrics (Sharpe, MaxDD)."""
        returns = equity_curve.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 6.5 * 60) # Scaled for 1m to annual
        drawdown = (equity_curve / equity_curve.cummax()) - 1
        max_dd = drawdown.min()
        calmar = returns.mean() * 252 / abs(max_dd) if max_dd != 0 else 0
        
        return {
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "total_return": (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        }
