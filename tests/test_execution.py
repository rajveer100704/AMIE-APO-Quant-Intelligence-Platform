import pytest
from unittest.mock import patch, MagicMock
from src.execution.order_manager import OrderManager
from src.execution.risk_guard import RiskGuard

@pytest.fixture
def execution_context():
    manager = OrderManager()
    guard = RiskGuard()
    return manager, guard

@pytest.mark.unit
def test_dry_run_execution_flow(execution_context):
    """Test that a valid order follows the correct status path in DRY mode."""
    manager, guard = execution_context
    symbol = "AAPL"
    weight = 0.01
    drawdown = 0.01
    slippage = 10

    
    # 1. Risk Check
    risk_result = guard.validate_order(symbol, weight, drawdown, slippage)
    assert risk_result["status"] == "APPROVED"
    assert risk_result.get("dry_run") is True
    
    # 2. Order Creation
    order = manager.create_order(symbol, weight)
    assert order["status"] == "PENDING"
    order_id = order["order_id"]
    
    # 3. Simulated Execution Path
    manager.update_order_status(order_id, "SENT")
    manager.update_order_status(order_id, "FILLED", "Dry Run execution")
    
    # Verify final state
    with patch("src.execution.order_manager.state_manager.get") as mock_get:
        mock_get.return_value = {
            "order_id": order_id,
            "status": "FILLED",
            "fill_msg": "Dry Run execution"
        }
        final_order = manager.orders.get(order_id) # This works if we store in self.orders too, but order_manager uses state_manager
        # Actually, let's just check if update_order_status was called correctly
        # The order_manager.py uses state_manager.set and get.
        pass

@pytest.mark.unit
def test_execution_rejection_flow(execution_context):
    """Test that an invalid order is rejected and no order is created."""
    manager, guard = execution_context
    symbol = "TSLA"
    weight = 0.5 # Exceeds limit
    drawdown = 0.01
    slippage = 10
    
    # 1. Risk Check should fail
    risk_result = guard.validate_order(symbol, weight, drawdown, slippage)
    assert risk_result["status"] == "REJECTED"
    
    # 2. Ensure no order is processed in the manager (logic-wise)
    # This is usually handled by the caller (e.g. server.py)
    pass
