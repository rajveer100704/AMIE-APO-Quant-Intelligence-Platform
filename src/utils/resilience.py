import time
import structlog
from typing import Callable, Any, Dict, Optional

logger = structlog.get_logger(__name__)

class CircuitBreaker:
    """
    Electronic-style Circuit Breaker for protecting API calls.
    States: CLOSED, OPEN, HALF_OPEN
    """
    def __init__(self, name: str, threshold: int = 5, recovery_timeout: int = 30):
        self.name = name
        self.threshold = threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.state = "CLOSED"
        self.last_failure_time = 0

    def call(self, func: Callable, *args, **kwargs) -> Any:
        # Check if circuit is OPEN
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info("Circuit breaker HALF_OPEN", name=self.name)
                self.state = "HALF_OPEN"
            else:
                logger.warning("Circuit breaker OPEN", name=self.name)
                raise RuntimeError(f"Circuit Breaker {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            # Success: Reset on Half-Open
            if self.state == "HALF_OPEN":
                logger.info("Circuit breaker CLOSED", name=self.name)
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            logger.error("Circuit breaker call failed", 
                         name=self.name, 
                         failures=self.failures, 
                         error=str(e))
            
            if self.failures >= self.threshold:
                self.state = "OPEN"
                logger.critical("Circuit breaker TRIPPED (OPEN)", 
                                name=self.name, 
                                threshold=self.threshold)
            raise

class FallbackStrategy:
    """
    Defines graceful degradation logic for system components.
    """
    @staticmethod
    def regime_fallback():
        logger.warning("Resilience: Falling back to 'Neutral' regime")
        return {"current_regime": 1, "regime_label": "Neutral", "is_fallback": True}

    @staticmethod
    def volatility_fallback():
        logger.warning("Resilience: Falling back to rolling std volatility")
        return {"cond_vol": 0.02, "is_fallback": True} # Default 2% daily vol

    @staticmethod
    def liquidity_fallback():
        logger.warning("Resilience: Falling back to cached liquidity")
        return {"liquidity_score": 0.5, "is_fallback": True}

    @staticmethod
    def optimizer_fallback():
        logger.warning("Resilience: Falling back to Equal Weight allocation")
        return {"allocation": "EqualWeight", "is_fallback": True}

# Global instances for key components
regime_breaker = CircuitBreaker("regime_detection")
vol_breaker = CircuitBreaker("volatility_engine")
liq_breaker = CircuitBreaker("liquidity_engine")
optimizer_breaker = CircuitBreaker("portfolio_optimizer")
