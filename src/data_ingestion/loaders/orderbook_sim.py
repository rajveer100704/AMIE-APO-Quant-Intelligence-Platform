import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils.logger import logger, log_execution_time

class OrderBookSimulator:
    """Simulates high-fidelity L2 order book data (Bids/Asks with depth)."""

    def __init__(self, symbol="SPY", levels=10, tick_size=0.01):
        self.symbol = symbol
        self.levels = levels
        self.tick_size = tick_size
        logger.info(f"OrderBookSimulator initialized for {symbol} with {levels} levels and tick size {tick_size}")

    @log_execution_time
    def generate_snapshot(self, base_price, timestamp=None):
        """Generates a single snapshot of the order book around the base price."""
        if timestamp is None:
            timestamp = datetime.now()

        # Generate Bids and Asks prices
        mid_price = round(base_price / self.tick_size) * self.tick_size
        bids_prices = [mid_price - (i + 1) * self.tick_size for i in range(self.levels)]
        asks_prices = [mid_price + (i + 1) * self.tick_size for i in range(self.levels)]

        # Generate realistic volumes (Poisson/Pareto-like distribution)
        bids_volumes = np.random.poisson(1000, self.levels) # Basic volume simulation
        asks_volumes = np.random.poisson(1000, self.levels)

        # Create DataFrame
        snapshot = {
            "timestamp": [timestamp] * (2 * self.levels),
            "symbol": [self.symbol] * (2 * self.levels),
            "side": ["bid"] * self.levels + ["ask"] * self.levels,
            "price": bids_prices + asks_prices,
            "volume": list(bids_volumes) + list(asks_volumes),
            "level": list(range(1, self.levels + 1)) * 2
        }
        return pd.DataFrame(snapshot)

    @log_execution_time
    def generate_series(self, base_prices, start_time=None):
        """Generates a series of snapshots based on price history in wide format."""
        all_snapshots = []
        current_time = start_time or datetime.now()

        for price in base_prices:
            # Generate snapshot in long format
            df_long = self.generate_snapshot(price, current_time)
            
            # Pivot to wide format for each level
            # We want columns: bid_price_0, bid_size_0, ask_price_0, ask_size_0, ...
            wide_snap = {"timestamp": current_time}
            for side in ["bid", "ask"]:
                side_df = df_long[df_long["side"] == side].sort_values("level")
                for i in range(self.levels):
                    wide_snap[f"{side}_price_{i}"] = side_df.iloc[i]["price"]
                    wide_snap[f"{side}_size_{i}"] = side_df.iloc[i]["volume"]
            
            all_snapshots.append(wide_snap)
            current_time += timedelta(minutes=1)

        return pd.DataFrame(all_snapshots).set_index("timestamp")

if __name__ == "__main__":
    sim = OrderBookSimulator("SPY")
    base_prices = [450.0 + i * 0.1 for i in range(5)]
    data = sim.generate_series(base_prices)
    print(data.head(20))
