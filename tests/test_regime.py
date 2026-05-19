import pytest
import pandas as pd
import numpy as np
import os
from src.regime_detection.hmm_model import MarketHMM
from src.regime_detection.ms_var_model import MarketMSVAR
from src.regime_detection.processor import RegimeProcessor

@pytest.fixture
def dummy_regime_data(tmp_path):
    """Create dummy data with 3 regimes for testing."""
    n = 300
    volatilities = [0.01, 0.05, 0.1]
    returns = []
    for vol in volatilities:
        returns.extend(np.random.normal(0, vol, n // 3))
    
    returns_arr = np.array(returns).reshape(-1, 1)
    df = pd.DataFrame({
        "Close": 100 * (1 + np.cumsum(returns)),
        "High": 101 * (1 + np.cumsum(returns)),
        "Low": 99 * (1 + np.cumsum(returns)),
        "Volume": 1000 * np.random.randn(n)
    })
    
    data_dir = tmp_path / "data_regime"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY_ohlcv.parquet")
    
    return returns_arr, df, str(data_dir)

@pytest.mark.unit
def test_hmm_fit_predict(dummy_regime_data):
    returns, df, data_dir = dummy_regime_data
    hmm_model = MarketHMM(n_components=3)
    hmm_model.fit(returns)
    states = hmm_model.predict(returns)
    assert len(states) == len(returns)
    # Transition matrix should be stochastic
    trans_mat = hmm_model.get_transition_matrix()
    assert np.allclose(trans_mat.sum(axis=1), 1.0)

@pytest.mark.unit
def test_msvar_fit_predict(dummy_regime_data):
    returns, df, data_dir = dummy_regime_data
    msvar_model = MarketMSVAR(k_regimes=2)
    
    # 1. Un-fitted state testing
    with pytest.raises(ValueError):
        msvar_model.predict_regimes()
    assert msvar_model.get_transition_matrix() is None
    
    # 2. Fit and predict
    msvar_model.fit(pd.Series(returns.flatten()))
    probas = msvar_model.predict_regimes()
    # MS-AR(1) loses the first observation due to lag
    assert len(probas) == len(returns) - 1
    
    # 3. Transition matrix
    trans_mat = msvar_model.get_transition_matrix()
    assert trans_mat is not None


@pytest.mark.integration
def test_processor_end_to_end(dummy_regime_data, tmp_path):
    returns, df, data_dir = dummy_regime_data
    output_dir = tmp_path / "output_regime"
    output_dir.mkdir()
    
    processor = RegimeProcessor(data_dir=data_dir, output_dir=str(output_dir))
    result_df = processor.detect_regimes("SPY")
    
    assert "regime" in result_df.columns
    assert "regime_prob_0" in result_df.columns
    assert len(result_df) == len(df)
    assert result_df["regime"].iloc[0] in [0, 1, 2] # Ensure we have valid regime labels
