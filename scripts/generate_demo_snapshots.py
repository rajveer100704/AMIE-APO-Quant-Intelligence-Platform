import numpy as np
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.execution_snapshot import save_snapshot

def generate_demo():
    print("Generating Quant-Grade demo snapshots...")
    symbols = ["SPY", "VIX", "AAPL", "MSFT"]
    
    # Simulate a realistic PnL curve
    pnl_curve = np.cumsum(np.random.normal(0.001, 0.02, 20))
    
    for i in range(20):
        timestamp = int(time.time()) - (20 - i) * 3600
        weights = np.random.dirichlet(np.ones(len(symbols)))
        
        # New Standardized Schema
        result = {
            "snapshot_id": f"EX-{timestamp}",
            "timestamp": timestamp,
            "amis_score": 50 + np.random.uniform(5, 25),
            "regime": "Bullish" if i > 10 else "Neutral",
            
            "portfolio": {
                "weights": {s: float(w) for s, w in zip(symbols, weights)},
                "exposure": float(np.sum(np.abs(weights))),
                "drawdown": float(np.random.uniform(0.01, 0.03)),
                "pnl": float(pnl_curve[i] * 100) # In %
            },
            
            "execution": {
                "orders": [
                    {"symbol": s, "status": "DRY_RUN", "order_id": f"ORD-{i}-{s}"}
                    for s in symbols
                ],
                "success_rate": 1.0,
                "avg_latency_ms": 10 + np.random.uniform(2, 8) if i != 15 else 72.5, # Anomaly at index 15
                "slippage": 0.00012
            },
            
            "validation": {
                "status": "PASSED"
            }
        }
        
        path = save_snapshot(result)
        print(f"  - Snapshot {i+1} saved: {path}")

if __name__ == "__main__":
    generate_demo()
