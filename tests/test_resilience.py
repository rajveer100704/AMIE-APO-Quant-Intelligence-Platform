import pytest
import time
from src.utils.resilience import CircuitBreaker, FallbackStrategy

@pytest.mark.unit
def test_circuit_breaker_flow():
    cb = CircuitBreaker("test_breaker", threshold=2, recovery_timeout=1)
    assert cb.state == "CLOSED"

    # Define a helper function that fails
    def failing_func():
        raise ValueError("Failed")

    # Define a helper function that succeeds
    def succeeding_func():
        return "success"

    # First call: failure
    with pytest.raises(ValueError):
        cb.call(failing_func)
    assert cb.failures == 1
    assert cb.state == "CLOSED"

    # Second call: failure -> should trip to OPEN
    with pytest.raises(ValueError):
        cb.call(failing_func)
    assert cb.failures == 2
    assert cb.state == "OPEN"

    # Call when OPEN -> raises RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        cb.call(succeeding_func)
    assert "is OPEN" in str(exc_info.value)

    # Wait for recovery timeout to transition to HALF_OPEN
    time.sleep(1.1)

    # Calling succeeding_func in HALF_OPEN -> transitions to CLOSED
    res = cb.call(succeeding_func)
    assert res == "success"
    assert cb.state == "CLOSED"
    assert cb.failures == 0

@pytest.mark.unit
def test_fallback_strategies():
    assert FallbackStrategy.regime_fallback()["regime_label"] == "Neutral"
    assert FallbackStrategy.volatility_fallback()["cond_vol"] == 0.02
    assert FallbackStrategy.liquidity_fallback()["liquidity_score"] == 0.5
    assert FallbackStrategy.optimizer_fallback()["allocation"] == "EqualWeight"
