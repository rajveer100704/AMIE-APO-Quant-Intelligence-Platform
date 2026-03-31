import pytest
import numpy as np
import os
import sys
from src.optimizer.numba_solver import NumbaPortfolioSolver, _mean_variance_objective
try:
    import cpp_solver
except ImportError:
    cpp_solver = None

from src.execution.execution_validator import validate_execution

def test_cpp_numba_consistency():
    """Ensure C++ solver and Numba solver produce identical objective values."""
    if cpp_solver is None:
        pytest.skip("cpp_solver not available")
        
    n = 5
    weights = np.random.dirichlet(np.ones(n))
    expected_returns = np.random.normal(0.01, 0.02, n)
    cov_matrix = np.eye(n) * 0.01 + np.random.normal(0, 0.001, (n, n))
    cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Symmetric
    risk_aversion = 3.0
    
    # Numba value
    numba_val = _mean_variance_objective(weights, expected_returns, cov_matrix, risk_aversion)
    
    # C++ value
    # C++ expects std::vector<std::vector<double>> for cov_matrix if passed as list of lists
    cpp_cov = [row.tolist() for row in cov_matrix]
    cpp_val = cpp_solver.mean_variance_objective(weights.tolist(), expected_returns.tolist(), cpp_cov, risk_aversion)
    
    assert np.allclose(numba_val, cpp_val, atol=1e-7), f"Inconsistency: Numba={numba_val}, CPP={cpp_val}"

def test_post_execution_validation_passed():
    """Verify validation passes for a correct execution."""
    weights = np.array([0.4, 0.6])
    execution_results = [
        {"symbol": "SPY", "status": "EXECUTED"},
        {"symbol": "VIX", "status": "EXECUTED"}
    ]
    status = validate_execution(weights, execution_results)
    assert status["status"] == "PASSED"
    assert status["weight_sum"] == 1.0

def test_post_execution_validation_failed_sum():
    """Verify validation fails if weights don't sum to 1."""
    weights = np.array([0.5, 0.6]) # Sums to 1.1
    execution_results = [{"symbol": "SPY", "status": "EXECUTED"}]
    status = validate_execution(weights, execution_results)
    assert status["status"] == "FAILED"
    assert "Weight sum mismatch" in status["errors"][0]

def test_post_execution_validation_failed_duplicates():
    """Verify validation fails on duplicate symbols."""
    weights = np.array([0.5, 0.5])
    execution_results = [
        {"symbol": "SPY", "status": "EXECUTED"},
        {"symbol": "SPY", "status": "EXECUTED"}
    ]
    status = validate_execution(weights, execution_results)
    assert status["status"] == "FAILED"
    assert "Duplicate symbol" in status["errors"][0]

def test_extreme_covariance_stability():
    """Test solver stability with near-singular covariance matrices."""
    solver = NumbaPortfolioSolver()
    n = 3
    expected_returns = np.array([0.01, 0.01, 0.01])
    # Near-singular matrix (highly correlated)
    cov_matrix = np.array([
        [1.0, 0.999, 0.999],
        [0.999, 1.0, 0.999],
        [0.999, 0.999, 1.0]
    ]) * 0.01
    
    weights = solver.mean_variance(expected_returns, cov_matrix)
    assert len(weights) == n
    assert np.allclose(np.sum(weights), 1.0, atol=1e-3)

def test_large_asset_size_performance():
    """Test solver with 100+ assets."""
    solver = NumbaPortfolioSolver()
    n = 100
    expected_returns = np.random.normal(0.001, 0.002, n)
    cov_matrix = np.eye(n) * 0.01
    
    import time
    start = time.perf_counter()
    weights = solver.mean_variance(expected_returns, cov_matrix)
    latency = (time.perf_counter() - start) * 1000
    
    assert len(weights) == n
    assert np.allclose(np.sum(weights), 1.0, atol=1e-3)
    assert latency < 500, f"Large asset solve too slow: {latency}ms"
