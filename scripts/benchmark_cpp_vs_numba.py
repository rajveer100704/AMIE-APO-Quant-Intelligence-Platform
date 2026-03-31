"""
Benchmark: C++ pybind11 vs Numba JIT vs Pure Python
Tests portfolio optimization kernel performance.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def benchmark_monte_carlo():
    """Benchmarks Monte Carlo across C++, Numba, and Python."""
    n_paths = 100000
    n_steps = 30
    
    print("=" * 60)
    print(f"MONTE CARLO BENCHMARK ({n_paths:,} paths, {n_steps} steps)")
    print("=" * 60)
    
    # C++ pybind11
    try:
        import cpp_solver
        start = time.perf_counter()
        result = cpp_solver.monte_carlo_stress(100, 0.05, 0.2, n_paths, n_steps)
        cpp_time = (time.perf_counter() - start) * 1000
        print(f"C++ pybind11:  {cpp_time:>8.1f} ms  |  VaR={result['var_99']:.4f}")
    except ImportError:
        cpp_time = None
        print("C++ pybind11:  NOT AVAILABLE")
    
    # Numba (Joblib parallel)
    from src.volatility.monte_carlo import MonteCarloSimulator
    sim = MonteCarloSimulator(n_paths=n_paths, n_steps=n_steps, n_jobs=-1)
    start = time.perf_counter()
    result = sim.run_simulation(100, 0.05, 0.2)
    numba_time = (time.perf_counter() - start) * 1000
    print(f"Python+Joblib: {numba_time:>8.1f} ms  |  VaR={result['vaR_99']:.4f}")
    
    # Python serial
    sim_serial = MonteCarloSimulator(n_paths=n_paths, n_steps=n_steps, n_jobs=1)
    start = time.perf_counter()
    result = sim_serial.run_simulation(100, 0.05, 0.2)
    serial_time = (time.perf_counter() - start) * 1000
    print(f"Python Serial: {serial_time:>8.1f} ms  |  VaR={result['vaR_99']:.4f}")
    
    print("-" * 60)
    if cpp_time:
        print(f"C++ Speedup vs Serial:  {serial_time/cpp_time:.1f}x")
        print(f"C++ Speedup vs Joblib:  {numba_time/cpp_time:.1f}x")
    print(f"Joblib Speedup vs Serial: {serial_time/numba_time:.1f}x")


def benchmark_objective():
    """Benchmarks portfolio objective function evaluation."""
    n_assets = 10
    n_iters = 10000
    
    weights = list(np.ones(n_assets) / n_assets)
    expected_returns = list(np.random.normal(0.01, 0.005, n_assets))
    cov_matrix = np.cov(np.random.randn(200, n_assets).T).tolist()
    
    print(f"\nOBJECTIVE FUNCTION BENCHMARK ({n_iters:,} evals, {n_assets} assets)")
    print("=" * 60)
    
    # C++
    try:
        import cpp_solver
        start = time.perf_counter()
        for _ in range(n_iters):
            cpp_solver.mean_variance_objective(weights, expected_returns, cov_matrix, 3.0)
        cpp_time = (time.perf_counter() - start) * 1000
        print(f"C++ pybind11:  {cpp_time:>8.1f} ms")
    except ImportError:
        cpp_time = None
        print("C++ pybind11:  NOT AVAILABLE")
    
    # Numba
    from src.optimizer.numba_solver import _mean_variance_objective
    w_np = np.array(weights)
    er_np = np.array(expected_returns)
    cov_np = np.array(cov_matrix)
    # Warmup
    _mean_variance_objective(w_np, er_np, cov_np, 3.0)
    start = time.perf_counter()
    for _ in range(n_iters):
        _mean_variance_objective(w_np, er_np, cov_np, 3.0)
    numba_time = (time.perf_counter() - start) * 1000
    print(f"Numba JIT:     {numba_time:>8.1f} ms")
    
    # Pure Python
    def pure_python_mv(w, er, cov, ra):
        pr = sum(w[i]*er[i] for i in range(len(w)))
        risk = sum(w[i]*cov[i][j]*w[j] for i in range(len(w)) for j in range(len(w)))
        return -(pr - 0.5*ra*risk)
    
    start = time.perf_counter()
    for _ in range(n_iters):
        pure_python_mv(weights, expected_returns, cov_matrix, 3.0)
    python_time = (time.perf_counter() - start) * 1000
    print(f"Pure Python:   {python_time:>8.1f} ms")
    
    print("-" * 60)
    if cpp_time:
        print(f"C++ Speedup vs Python:  {python_time/cpp_time:.1f}x")
        print(f"C++ Speedup vs Numba:   {numba_time/cpp_time:.1f}x" if numba_time else "")
    print(f"Numba Speedup vs Python: {python_time/numba_time:.1f}x")


if __name__ == "__main__":
    benchmark_monte_carlo()
    benchmark_objective()
    print("\n✅ Benchmark Complete")
