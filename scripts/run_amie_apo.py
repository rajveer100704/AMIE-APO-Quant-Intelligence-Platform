from src.data_ingestion.pipeline import DataPipeline
from src.utils.logger import logger

def main():
    logger.info("Starting Full AMIE-APO System Benchmark...")
    
    # Symbols: SPY (Equity), VIX (Volatility), ^TNX (Rates/Macro)
    symbols = ["SPY", "VIX", "^TNX"]
    pipeline = DataPipeline(symbols=symbols)
    
    # Run the full pipeline
    # 5 days of 1-minute data for a decent sample size
    pipeline.run_ingestion(period="5d", interval="1m")
    
    logger.info("Full System Benchmark Completed Successfully.")

if __name__ == "__main__":
    main()
