import pytest
import time
from src.utils.performance import PerformanceMonitor, perf_monitor

@pytest.mark.unit
def test_performance_monitor():
    monitor = PerformanceMonitor()
    
    # Test ending a non-existent trace returns 0.0
    assert monitor.end_trace("non-existent") == 0.0
    
    # Test valid trace
    monitor.start_trace("regime")
    time.sleep(0.01) # Sleep for ~10 ms
    latency = monitor.end_trace("regime")
    assert latency > 0.0
    
    # Test global instance
    perf_monitor.start_trace("amis_fusion")
    latency2 = perf_monitor.end_trace("amis_fusion")
    assert latency2 >= 0.0
