import pandas as pd
import numpy as np
from src.utils.logger import logger, log_execution_time

class DataCleaner:
    """Processes, cleans, and normalizes financial data."""

    def __init__(self, normalization="z-score"):
        self.normalization = normalization
        logger.info(f"DataCleaner initialized with normalization technique: {self.normalization}")

    @log_execution_time
    def clean(self, df):
        """Cleans the DataFrame by handling missing values and outliers."""
        if df.empty:
            logger.warning("Empty DataFrame passed to clean.")
            return df

        # Handle NaNs and Infs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Forward fill and then backfill for holes
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        return df

    @log_execution_time
    def normalize(self, df, columns=None):
        """Normalizes specified columns."""
        if df.empty or columns is None:
            return df

        for col in columns:
            if col in df.columns:
                if self.normalization == "z-score":
                    mean = df[col].mean()
                    std = df[col].std()
                    df[f"{col}_norm"] = (df[col] - mean) / std if std != 0 else 0
                elif self.normalization == "min-max":
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val) if max_val != min_val else 0

        return df

    @log_execution_time
    def detect_outliers(self, df, column, threshold=3):
        """Detects outliers using Z-score thresholding."""
        if df.empty or column not in df.columns:
            return df

        mean = df[column].mean()
        std = df[column].std()
        z_scores = (df[column] - mean) / std
        is_outlier = np.abs(z_scores) > threshold
        df[f"{column}_is_outlier"] = is_outlier
        logger.info(f"Detected {is_outlier.sum()} outliers in {column} with threshold {threshold}")

        return df

if __name__ == "__main__":
    test_df = pd.DataFrame({"price": [10.0, 11.0, 10.5, 10.2, 50.0]})
    cleaner = DataCleaner()
    test_df = cleaner.clean(test_df)
    test_df = cleaner.normalize(test_df, ["price"])
    test_df = cleaner.detect_outliers(test_df, "price")
    print(test_df)
