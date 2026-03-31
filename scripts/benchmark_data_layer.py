import time
import os
import pandas as pd
from src.data_ingestion.pipeline import DataPipeline
from src.utils.logger import logger

def benchmark():
    symbols = ["SPY", "VIX", "^TNX", "GLD", "QQQ"]
    pipeline = DataPipeline(symbols, "data/benchmark_processed")
    
    start_time = time.perf_counter()
    pipeline.run_ingestion(period="1mo", interval="1h")
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    logger.info(f"BENCHMARK: End-to-end ingestion for {len(symbols)} symbols took {total_time:.2f} seconds.")
    
    # Check storage size
    total_size = 0
    for root, dirs, files in os.walk("data/benchmark_processed"):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    
    logger.info(f"BENCHMARK: Total data stored: {total_size / 1024:.2f} KB")

    # Cleanup
    for file in os.listdir("data/benchmark_processed"):
        os.remove(os.path.join("data/benchmark_processed", file))
    os.rmdir("data/benchmark_processed")

if __name__ == "__main__":
    benchmark()
