import numpy as np
import pandas as pd
from hmmlearn import hmm
from src.utils.logger import logger, log_execution_time

class MarketHMM:
    """Gaussian Hidden Markov Model for market regime detection."""

    def __init__(self, n_components=3, covariance_type="full", n_iter=1000):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components, 
            covariance_type=covariance_type, 
            n_iter=n_iter,
            random_state=42
        )
        self.is_fitted = False
        logger.info(f"MarketHMM initialized with {n_components} states.")

    @log_execution_time
    def fit(self, features):
        """Fits the HMM on the provided features (2D array-like)."""
        features_arr = np.asarray(features)
        logger.info(f"Fitting HMM on {len(features_arr)} samples...")
        
        # Calculate mean and standard deviation for robust internal scaling
        self.mean_ = np.mean(features_arr, axis=0)
        self.std_ = np.std(features_arr, axis=0)
        # Prevent division by zero for constant features
        self.std_ = np.where(self.std_ == 0.0, 1.0, self.std_)
        
        scaled_features = (features_arr - self.mean_) / self.std_
        
        try:
            self.model.fit(scaled_features)
            self.is_fitted = True
            logger.info("HMM fitting complete.")
            
            # Identify states (Sort states by mean of the first feature, typically returns)
            # High mean -> Bullish, Low mean -> Bearish/Crisis
            self.means = self.model.means_
            self.state_order = np.argsort(self.means[:, 0]) # Crude ordering: 0=Crisis, N=Bullish
            logger.info(f"HMM state means (feature 0): {self.means[:, 0]}")
        except Exception as e:
            logger.error(f"Error fitting HMM: {e}")
            raise


    @log_execution_time
    def predict(self, features):
        """Predicts the hidden states for the given features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        features_arr = np.asarray(features)
        scaled_features = (features_arr - self.mean_) / self.std_
        states = self.model.predict(scaled_features)
        return states

    @log_execution_time
    def predict_proba(self, features):
        """Predicts the posterior probability of each state."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        features_arr = np.asarray(features)
        scaled_features = (features_arr - self.mean_) / self.std_
        probas = self.model.predict_proba(scaled_features)
        return probas


    def get_transition_matrix(self):
        """Returns the state transition matrix."""
        if not self.is_fitted:
            return None
        return self.model.transmat_

if __name__ == "__main__":
    # Simple test with random data
    data = np.random.randn(1000, 2)
    model = MarketHMM(n_components=3)
    model.fit(data)
    states = model.predict(data)
    print(f"Detected states: {np.unique(states)}")
    print(f"Transition Matrix:\n{model.get_transition_matrix()}")
