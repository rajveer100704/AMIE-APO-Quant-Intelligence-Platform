import pytest
import pandas as pd
import numpy as np
import os
from src.optimizer.amis_fusion import AMISFusion

@pytest.fixture
def amis_setup(tmp_path):
    data_dirs = {
        "regime": str(tmp_path / "regime"),
        "volatility": str(tmp_path / "vol"),
        "liquidity": str(tmp_path / "liq")
    }
    for d in data_dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # Create valid dummy data for score calculation
    pd.DataFrame({"regime": [2]*100}).to_parquet(os.path.join(data_dirs["regime"], "AAPL_regime.parquet"))
    pd.DataFrame({"cvaR_99": [-0.01]*100}).to_parquet(os.path.join(data_dirs["volatility"], "AAPL_stress.parquet"))
    pd.DataFrame({"quoted_spread": [0.0001]*100}).to_parquet(os.path.join(data_dirs["liquidity"], "AAPL_liquidity.parquet"))
    
    return data_dirs

@pytest.mark.unit
def test_amis_score_calculation(amis_setup):
    fusion = AMISFusion(data_dirs=amis_setup)
    score = fusion.compute_amis("AAPL")
    assert 0 <= score <= 100
    # Bull regime, low vol, high liq should result in a high score
    assert score > 80

@pytest.mark.unit
def test_amis_missing_data(amis_setup):
    fusion = AMISFusion(data_dirs=amis_setup)
    # Symbol with no data
    score = fusion.compute_amis("NULL")
    # Should return a safe neutral/zero score rather than crashing
    assert score == 0 or score == 50 # Depends on implementation default
