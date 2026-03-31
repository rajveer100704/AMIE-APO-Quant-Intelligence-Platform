import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.data_ingestion.loaders.yfinance_loader import YFinanceLoader
from src.data_ingestion.loaders.orderbook_sim import OrderBookSimulator
from src.data_ingestion.processors.cleaner import DataCleaner
from src.regime_detection.processor import RegimeProcessor
from src.volatility.engine import VolatilityEngine
from src.liquidity.engine import LiquidityEngine
from src.optimizer.engine import APOEngine
from src.utils.logger import logger, log_execution_time

# Correcting imports if needed by using absolute paths or proper package structure
# For simplicity in this script, assuming they are in path

class DataPipeline:
    """Orchestrates data ingestion, cleaning, and storage."""

    def __init__(self, symbols=None, data_dir="data/processed", regime_dir="data/regime", vol_dir="data/volatility", liq_dir="data/liquidity", port_dir="data/portfolio"):
        self.symbols = symbols or ["SPY", "VIX", "^TNX"]
        self.data_dir = data_dir
        self.regime_dir = regime_dir
        self.vol_dir = vol_dir
        self.liq_dir = liq_dir
        self.port_dir = port_dir
        self.yf_loader = YFinanceLoader(self.symbols)
        self.orderbook_sim = OrderBookSimulator(levels=10)
        self.cleaner = DataCleaner()
        self.regime_processor = RegimeProcessor(data_dir=self.data_dir, output_dir=self.regime_dir)
        self.vol_engine = VolatilityEngine(data_dir=self.data_dir, output_dir=self.vol_dir)
        self.liq_engine = LiquidityEngine(data_dir=self.data_dir, output_dir=self.liq_dir)
        self.apo_engine = APOEngine(data_dir=self.data_dir, output_dir=self.port_dir)

        for path in [self.data_dir, self.regime_dir, self.vol_dir, self.liq_dir, self.port_dir]:
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"Created directory: {path}")

    @log_execution_time
    def run_ingestion(self, period="5d", interval="1m", detect_regimes=True, run_stress_tests=True, run_liquidity_analysis=True, optimize_portfolio=True):
        """Runs the end-to-end ingestion pipeline and updates the portfolio."""
        logger.info(f"Starting data ingestion for symbols: {self.symbols}")

        for symbol in self.symbols:
            logger.info(f"Processing {symbol}")

            # 1. Fetch OHLCV
            df = self.yf_loader.fetch_ohlcv(symbol, period, interval)
            if df.empty:
                continue

            # 2. Clean OHLCV
            df = self.cleaner.clean(df)
            df = self.cleaner.normalize(df, ["Close", "Volume"])

            # 3. Generate Simulated Order Book (L2) for the symbol
            # We use the Close prices as base prices for simulation
            ob_df = self.orderbook_sim.generate_series(df["Close"].values, df.index[0].to_pydatetime())

            # 4. Store OHLCV in Parquet
            ohlcv_path = os.path.join(self.data_dir, f"{symbol}_ohlcv.parquet")
            table = pa.Table.from_pandas(df)
            pq.write_table(table, ohlcv_path)
            logger.info(f"Stored {symbol} OHLCV at {ohlcv_path}")

            # 5. Store L2 in Parquet
            l2_path = os.path.join(self.data_dir, f"{symbol}_l2.parquet")
            l2_table = pa.Table.from_pandas(ob_df)
            pq.write_table(l2_table, l2_path)
            logger.info(f"Stored {symbol} L2 at {l2_path}")

            # 6. Optional Regime Detection
            if detect_regimes:
                self.regime_processor.detect_regimes(symbol)
            
            # 7. Optional Volatility Stress Test
            if run_stress_tests:
                self.vol_engine.run_stress_test(symbol)
            
            # 8. Optional Liquidity Analysis
            if run_liquidity_analysis:
                self.liq_engine.run_liquidity_analysis(symbol)

        # 9. Final Portfolio Optimization (APO)
        if optimize_portfolio:
            self.apo_engine.run_rebalancing(symbols=self.symbols)

        logger.info("Full AMIE-APO pipeline completed successfully.")

if __name__ == "__main__":
    pipeline = DataPipeline(["SPY", "VIX"])
    pipeline.run_ingestion(period="1d", interval="1m")
