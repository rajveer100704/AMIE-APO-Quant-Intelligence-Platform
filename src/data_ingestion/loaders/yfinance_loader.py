import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from src.utils.logger import logger, log_execution_time

class YFinanceLoader:
    """Loader for fetching historical OHLCV and Macro indicators via Yahoo Finance."""

    def __init__(self, symbols=None):
        self.symbols = symbols or ["SPY", "QQQ", "VIX", "^TNX", "GLD"]
        logger.info(f"YFinanceLoader initialized for symbols: {self.symbols}")

    @log_execution_time
    def fetch_ohlcv(self, symbol, period="1mo", interval="1m"):
        """Fetches OHLCV data for a given symbol."""
        logger.info(f"Fetching OHLCV for {symbol} (period={period}, interval={interval})")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    @log_execution_time
    def fetch_batch(self, period="1mo", interval="1m"):
        """Fetches data for all symbols in batch."""
        data_map = {}
        for symbol in self.symbols:
            data_map[symbol] = self.fetch_ohlcv(symbol, period, interval)
        return data_map

if __name__ == "__main__":
    # Test loader
    loader = YFinanceLoader(["SPY", "VIX"])
    data = loader.fetch_batch(period="1d", interval="1m")
    for s, df in data.items():
        print(f"{s}: {df.shape}")
