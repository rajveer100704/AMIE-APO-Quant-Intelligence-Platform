"""
AMIE-APO System Profiler (Phase 8)
Profiles CPU, memory, and execution time across all pipeline components.
"""
import os
import sys
import time
import cProfile
import pstats
import tracemalloc
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger


def profile_component(name, func, *args, **kwargs):
    """Profiles a single component for CPU time and memory."""
    # Memory tracking
    tracemalloc.start()
    
    # CPU profiling
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    
    result = func(*args, **kwargs)
    
    profiler.disable()
    elapsed = (time.perf_counter() - start) * 1000
    
    # Memory snapshot
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Extract top functions
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(5)
    top_funcs = stream.getvalue()
    
    report = {
        "component": name,
        "latency_ms": round(elapsed, 2),
        "memory_current_mb": round(current / 1024 / 1024, 2),
        "memory_peak_mb": round(peak / 1024 / 1024, 2),
        "top_functions": top_funcs
    }
    
    logger.info(f"[PROFILE] {name}: {elapsed:.1f}ms | Mem: {report['memory_peak_mb']}MB peak")
    return report, result


def run_full_profile():
    """Profiles every major component of the AMIE-APO system."""
    import numpy as np
    import pandas as pd
    
    reports = []
    
    # 1. Data Cleaning
    from src.data_ingestion.processors.cleaner import DataCleaner
    cleaner = DataCleaner()
    dummy_df = pd.DataFrame({
        "Close": np.random.randn(5000) * 100 + 500,
        "Volume": np.random.randint(1000, 100000, 5000).astype(float),
        "High": np.random.randn(5000) * 100 + 510,
        "Low": np.random.randn(5000) * 100 + 490,
    })
    r, _ = profile_component("DataCleaner", cleaner.clean, dummy_df)
    reports.append(r)
    
    # 2. Order Book Simulation
    from src.data_ingestion.loaders.orderbook_sim import OrderBookSimulator
    sim = OrderBookSimulator(levels=10)
    prices = np.linspace(450, 460, 1000)
    r, _ = profile_component("OrderBookSimulator", sim.generate_series, prices)
    reports.append(r)
    
    # 3. HMM Regime Detection
    from src.regime_detection.hmm_model import MarketHMM
    hmm = MarketHMM(n_components=3)
    returns = np.random.normal(0, 0.01, (500, 2))
    r, _ = profile_component("HMM_Fit", hmm.fit, returns)
    reports.append(r)
    r, _ = profile_component("HMM_Predict", hmm.predict, returns)
    reports.append(r)
    
    # 4. GARCH Volatility
    from src.volatility.garch_model import MarketVolatility
    vol = MarketVolatility()
    rets_1d = np.random.normal(0, 0.01, 500)
    r, _ = profile_component("GARCH_Fit", vol.fit, rets_1d)
    reports.append(r)
    
    # 5. Monte Carlo
    from src.volatility.monte_carlo import MonteCarloSimulator
    mc = MonteCarloSimulator(n_paths=5000, n_steps=30)
    r, _ = profile_component("MonteCarlo_5k", mc.run_simulation, 100, 0, 0.2)
    reports.append(r)
    
    # 6. Numba Solver
    from src.optimizer.numba_solver import NumbaPortfolioSolver
    solver = NumbaPortfolioSolver()
    cov = np.cov(np.random.randn(200, 3).T)
    exp_ret = np.array([0.01, 0.005, 0.002])
    r, _ = profile_component("Numba_MeanVar", solver.mean_variance, exp_ret, cov)
    reports.append(r)
    r, _ = profile_component("Numba_RiskParity", solver.risk_parity, cov)
    reports.append(r)
    
    # 7. AMIS Fusion
    from src.optimizer.amis_fusion import AMISFusion
    fusion = AMISFusion()
    r, _ = profile_component("AMIS_Fusion", fusion.compute_amis, "SPY")
    reports.append(r)
    
    # Summary
    print("\n" + "=" * 70)
    print("AMIE-APO PROFILING REPORT")
    print("=" * 70)
    print(f"{'Component':<25} {'Latency (ms)':>12} {'Peak Mem (MB)':>14}")
    print("-" * 55)
    for r in reports:
        print(f"{r['component']:<25} {r['latency_ms']:>12.1f} {r['memory_peak_mb']:>14.2f}")
    
    total_latency = sum(r["latency_ms"] for r in reports)
    print("-" * 55)
    print(f"{'TOTAL':<25} {total_latency:>12.1f}")
    print("=" * 70)
    
    # Identify bottleneck
    bottleneck = max(reports, key=lambda x: x["latency_ms"])
    print(f"\n⚠ BOTTLENECK: {bottleneck['component']} ({bottleneck['latency_ms']:.1f}ms)")
    
    return reports


if __name__ == "__main__":
    run_full_profile()
