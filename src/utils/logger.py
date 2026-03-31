import sys
import time
import structlog
import logging
from functools import wraps

def setup_logger():
    """Configures structlog for high-performance machine-readable logging."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )
    # Standard logging fallback
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

def log_execution_time(func):
    """Decorator to log execution time using structlog."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = structlog.get_logger(func.__module__)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        logger.info("execution_trace", 
                    function=func.__name__, 
                    latency_ms=round(latency_ms, 2))
        return result
    return wrapper

# Initialize on import
setup_logger()
logger = structlog.get_logger()
