import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from src.regime_detection.hmm_model import MarketHMM
from src.utils.logger import logger, log_execution_time

class RegimeProcessor:
    """Orchestrates regime detection for high-frequency market data."""

    def __init__(self, data_dir="data/processed", output_dir="data/regime"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        logger.info(f"RegimeProcessor initialized. Reading from {data_dir}, writing to {output_dir}")

    @log_execution_time
    def engineer_features(self, df):
        """Creates features suitable for HMM to detect regimes."""
        if df.empty:
            return pd.DataFrame()

        features_df = pd.DataFrame(index=df.index)

        # 1. Log Returns (Volatility proxy)
        features_df["returns"] = df["Close"].pct_change().fillna(0)
        
        # 2. Daily Range (Volatility proxy)
        features_df["range"] = (df["High"] - df["Low"]) / df["Close"]
        
        # 3. Volatility (5-period rolling std of returns)
        features_df["volatility"] = features_df["returns"].rolling(window=5).std().fillna(0)

        # Handle NaNs
        features_df.replace([np.inf, -np.inf], 0, inplace=True)
        features_df.fillna(0, inplace=True)

        return features_df

    @log_execution_time
    def detect_regimes(self, symbol="SPY", n_states=3):
        """Loads data, trains HMM, and predicts regimes."""
        logger.info(f"Detecting regimes for {symbol}...")
        
        input_path = os.path.join(self.data_dir, f"{symbol}_ohlcv.parquet")
        if not os.path.exists(input_path):
            logger.error(f"Input data not found for {symbol} at {input_path}")
            return

        # Load data
        df = pd.read_parquet(input_path)
        features_df = self.engineer_features(df)
        
        # Use returns and volatility as features
        features = features_df[["returns", "volatility"]].values

        # Train HMM
        model = MarketHMM(n_components=n_states)
        model.fit(features)
        
        # Predict states
        states = model.predict(features)
        probas = model.predict_proba(features)

        # Map states (Ordered by returns: 0=Crisis, 1=Neutral, 2=Bullish)
        # We'll use the mean returns per state to order them
        state_means = [features_df.iloc[states == i]["returns"].mean() for i in range(n_states)]
        state_order = np.argsort(state_means) # 0=Lowest Mean, 2=Highest Mean
        
        # Create mapping: current_state -> ordered_state
        state_map = {old: new for new, old in enumerate(state_order)}
        ordered_states = np.array([state_map[s] for s in states])

        # Add to original DF
        df["regime"] = ordered_states
        for i in range(n_states):
            df[f"regime_prob_{i}"] = probas[:, state_order[i]]

        # Store transition matrix for reference
        trans_mat = model.get_transition_matrix()
        logger.info(f"Transition Matrix for {symbol}:\n{trans_mat}")

        # Store output
        output_path = os.path.join(self.output_dir, f"{symbol}_regimes.parquet")
        df.to_parquet(output_path)
        logger.info(f"Stored regimes for {symbol} at {output_path}")

        return df

if __name__ == "__main__":
    processor = RegimeProcessor()
    # Assume data from Phase 1 exists
    try:
        df = processor.detect_regimes("SPY")
        if df is not None:
             print(f"Detected regimes count:\n{df['regime'].value_counts()}")
    except Exception as e:
        logger.error(f"Error in testing RegimeProcessor: {e}")
