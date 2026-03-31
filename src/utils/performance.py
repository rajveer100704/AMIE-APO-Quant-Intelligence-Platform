import time
import structlog
from typing import Dict, Any

logger = structlog.get_logger(__name__)

# Standard Latency Budgets (in MS)
LATENCY_BUDGETS = {
    "regime": 5.0,
    "volatility": 10.0,
    "liquidity": 5.0,
    "amis_fusion": 1.0,
    "optimizer": 10.0,
    "total_api": 50.0
}

class PerformanceMonitor:
    """
    Standardized performance monitor for tracking component latencies.
    """
    def __init__(self):
        self._traces = {}

    def start_trace(self, component: str):
        self._traces[component] = time.perf_counter()

    def end_trace(self, component: str) -> float:
        if component not in self._traces:
            return 0.0
        
        start_time = self._traces.pop(component)
        latency_ms = (time.perf_counter() - start_time) * 1000
        budget = LATENCY_BUDGETS.get(component, 100.0)
        
        status = "OK" if latency_ms <= budget else "VIOLATED"
        
        logger.info("performance_benchmark", 
                    component=component, 
                    latency_ms=round(latency_ms, 2), 
                    budget_ms=budget,
                    budget_status=status)
        
        return latency_ms

# Global monitor instance
perf_monitor = PerformanceMonitor()
