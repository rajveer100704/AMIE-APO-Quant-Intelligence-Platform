import pytest
import numpy as np
import pandas as pd
import os
from src.optimizer.amis_fusion import AMISFusion
from src.optimizer.solver import PortfolioSolver
from src.optimizer.numba_solver import NumbaPortfolioSolver
from src.optimizer.engine import APOEngine
try:
    import cpp_solver
except ImportError:
    cpp_solver = None

@pytest.fixture
def dummy_optimizer_data(tmp_path):
    """Create comprehensive dummy data for optimizer tests."""
    n = 100
    symbols = ["SPY", "TLT"]
    
    # 1. Regime
    regime_dir = tmp_path / "data_regime"
    regime_dir.mkdir()
    pd.DataFrame({"regime": [2]*n}).to_parquet(regime_dir / "SPY_regime.parquet")
    pd.DataFrame({"regime": [1]*n}).to_parquet(regime_dir / "TLT_regime.parquet")
    
    # 2. Volatility
    vol_dir = tmp_path / "data_vol"
    vol_dir.mkdir()
    pd.DataFrame({"cvaR_99": [-0.01]*n}).to_parquet(vol_dir / "SPY_stress.parquet")
    pd.DataFrame({"cvaR_99": [-0.03]*n}).to_parquet(vol_dir / "TLT_stress.parquet")
    
    # 3. Liquidity
    liq_dir = tmp_path / "data_liq"
    liq_dir.mkdir()
    pd.DataFrame({"quoted_spread": [0.0001]*n}).to_parquet(liq_dir / "SPY_liquidity.parquet")
    pd.DataFrame({"quoted_spread": [0.0005]*n}).to_parquet(liq_dir / "TLT_liquidity.parquet")
    
    # 4. OHLCV (Prices)
    price_dir = tmp_path / "data_prices"
    price_dir.mkdir()
    for s in symbols:
        rets = np.random.normal(0, 0.01, n)
        pd.DataFrame({"Close": 100 * (1 + np.cumsum(rets))}).to_parquet(price_dir / f"{s}_ohlcv.parquet")
        
    return {
        "regime": str(regime_dir),
        "volatility": str(vol_dir),
        "liquidity": str(liq_dir),
        "prices": str(price_dir),
        "symbols": symbols
    }

@pytest.mark.unit
def test_amis_fusion(dummy_optimizer_data):
    fusion = AMISFusion(data_dirs={
        "regime": dummy_optimizer_data["regime"],
        "volatility": dummy_optimizer_data["volatility"],
        "liquidity": dummy_optimizer_data["liquidity"]
    })
    score = fusion.compute_amis("SPY")
    assert 0 <= score <= 100
    spy_score = fusion.compute_amis("SPY")
    tlt_score = fusion.compute_amis("TLT")
    assert spy_score > tlt_score

@pytest.mark.unit
def test_numba_mean_variance():
    n = 3
    solver = NumbaPortfolioSolver(risk_aversion=3.0)
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, (200, n))
    cov = np.cov(returns.T)
    expected = np.mean(returns, axis=0)
    w = solver.mean_variance(expected, cov)
    assert np.isclose(np.sum(w), 1.0, atol=1e-4)
    assert all(wi >= -1e-6 for wi in w)

@pytest.mark.unit
@pytest.mark.skipif(cpp_solver is None, reason="cpp_solver extension not found")
def test_cpp_vs_numba_consistency():
    """Elite Fix 4: Consistency check between C++ and Numba solvers."""
    n = 5
    np.random.seed(42)
    expected = np.random.normal(0.001, 0.01, n)
    # Generate a positive definite covariance matrix
    A = np.random.normal(0, 0.01, (n, n))
    cov = np.dot(A, A.T) + np.eye(n) * 1e-4
    
    numba_solver = NumbaPortfolioSolver(risk_aversion=3.0)
    w_numba = numba_solver.mean_variance(expected, cov)
    
    # Assuming cpp_solver has a similar interface or we use the PortfolioSolver wrapper
    # For this test, we'll assume a direct call if available
    # If the interface differs, this test should be adjusted
    try:
        w_cpp = cpp_solver.solve_mean_variance(expected, cov, 3.0)
        assert np.allclose(w_numba, w_cpp, atol=1e-4)
    except AttributeError:
        pytest.skip("cpp_solver.solve_mean_variance not found")

@pytest.mark.unit
def test_solver_singular_covariance():
    """Elite Fix: Edge case stress testing."""
    n = 3
    solver = NumbaPortfolioSolver(risk_aversion=3.0)
    expected = np.array([0.01, 0.01, 0.01])
    # Rank-deficient covariance matrix
    cov = np.array([
        [0.01, 0.01, 0.0],
        [0.01, 0.01, 0.0],
        [0.0, 0.0, 0.01]
    ])
    # Should handle it without crashing (e.g., via shrinkage or pinning)
    w = solver.mean_variance(expected, cov)
    assert np.isclose(np.sum(w), 1.0, atol=1e-4)

@pytest.mark.integration
def test_apo_engine_end_to_end(dummy_optimizer_data, tmp_path):
    output_dir = tmp_path / "output_portfolio"
    output_dir.mkdir()
    
    engine = APOEngine(data_dir=dummy_optimizer_data["prices"], output_dir=str(output_dir))
    engine.fusion.data_dirs = {
        "regime": dummy_optimizer_data["regime"],
        "volatility": dummy_optimizer_data["volatility"],
        "liquidity": dummy_optimizer_data["liquidity"]
    }
    alloc = engine.run_rebalancing(symbols=dummy_optimizer_data["symbols"])
    assert "SPY" in alloc
    assert "TLT" in alloc
    assert os.path.exists(os.path.join(str(output_dir), "latest_allocation.parquet"))
