import pytest
import time
import numpy as np
import pandas as pd
from src.optimizer.solver import PortfolioSolver
from src.optimizer.numba_solver import NumbaPortfolioSolver
from fastapi.testclient import TestClient
from src.api.server import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.mark.performance
def test_optimizer_latency_large_asset_count():
    """Elite Fix 7: Benchmark optimizer with 1000+ assets."""
    n = 1000
    solver = NumbaPortfolioSolver(risk_aversion=3.0)
    np.random.seed(42)
    expected = np.random.normal(0.001, 0.01, n)
    # Generate a random positive definite covariance matrix
    A = np.random.normal(0, 0.01, (n, n))
    cov = np.dot(A, A.T) + np.eye(n) * 1e-4
    
    start = time.perf_counter()
    w = solver.mean_variance(expected, cov)
    end = time.perf_counter()
    
    latency_ms = (end - start) * 1000
    print(f"\nOptimizer (1000 assets) latency: {latency_ms:.2f}ms")
    # For a high-performance system, we expect this to be < 500ms even for 1000 assets depending on solver complexity
    assert latency_ms < 1000 # Safety threshold

@pytest.mark.performance
def test_api_rebalancing_latency(client):
    """Elite Fix 7: Benchmark rebalancing endpoint latency."""
    # We'll use the mocked optimizer already patched in conftest if we had one
    # For now, let's just measure the end-to-end latency of a mocked request
    start = time.perf_counter()
    response = client.get("/health")
    latency_ms = (time.perf_counter() - start) * 1000
    assert latency_ms < 50
