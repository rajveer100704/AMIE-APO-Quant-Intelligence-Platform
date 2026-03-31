import sys
import os

sys.path.insert(0, '.')
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/regime', exist_ok=True)
os.makedirs('data/volatility', exist_ok=True)
os.makedirs('data/liquidity', exist_ok=True)
os.makedirs('data/portfolio', exist_ok=True)

from dotenv import load_dotenv
load_dotenv()

print("Starting data ingestion pipeline for SPY, AAPL...")
from src.data_ingestion.pipeline import DataPipeline

p = DataPipeline(
    ['SPY', 'AAPL'],
    data_dir='data/processed',
    regime_dir='data/regime',
    vol_dir='data/volatility',
    liq_dir='data/liquidity',
    port_dir='data/portfolio'
)

p.run_ingestion(
    period='5d',
    interval='1h',
    detect_regimes=True,
    run_stress_tests=True,
    run_liquidity_analysis=True,
    optimize_portfolio=False  # skip optimizer for now, test data endpoints first
)

print("--- PIPELINE DONE ---")

# Confirm files were created
for d in ['data/processed', 'data/regime', 'data/volatility', 'data/liquidity']:
    files = os.listdir(d) if os.path.exists(d) else []
    print(f"{d}: {files}")
