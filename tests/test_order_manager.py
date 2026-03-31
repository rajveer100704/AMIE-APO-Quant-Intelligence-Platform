import pytest
import time
from unittest.mock import patch, MagicMock
from src.execution.order_manager import OrderManager

@pytest.fixture
def mock_state_manager():
    with patch("src.execution.order_manager.state_manager") as mock:
        store = {}
        mock.get.side_effect = lambda k: store.get(k)
        mock.set.side_effect = lambda k, v, ttl=None: store.update({k: v})
        yield mock

@pytest.mark.unit
def test_order_creation_idempotency(mock_state_manager):
    manager = OrderManager()
    
    # 1. Create initial order
    order1 = manager.create_order("AAPL", 0.05)
    assert order1["status"] == "PENDING"
    assert "order_id" in order1
    
    # 2. Try to create same order immediately (should result in different ID due to timestamp)
    # But if we mock time.time() to be same, it should fail
    with patch("time.time", return_value=123456789.0):
        order2 = manager.create_order("AAPL", 0.05)
        # First call with this time
        assert order2["status"] == "PENDING"
        
        # Second call with same time -> collision
        order3 = manager.create_order("AAPL", 0.05)
        assert order3["status"] == "BLOCKED"
        assert order3["reason"] == "Idempotent collision"

@pytest.mark.unit
def test_order_status_progression(mock_state_manager):
    manager = OrderManager()
    order = manager.create_order("MSFT", 0.1)
    order_id = order["order_id"]
    
    # Update to SENT
    manager.update_order_status(order_id, "SENT")
    updated_order = mock_state_manager.get(f"order:{order_id}")
    assert updated_order["status"] == "SENT"
    
    # Update to FILLED
    manager.update_order_status(order_id, "FILLED", fill_msg="Partial Fill at 400.5")
    filled_order = mock_state_manager.get(f"order:{order_id}")
    assert filled_order["status"] == "FILLED"
    assert filled_order["fill_msg"] == "Partial Fill at 400.5"

@pytest.mark.unit
def test_reconcile_positions(mock_state_manager):
    manager = OrderManager()
    
    # Setup internal state
    mock_state_manager.set("portfolio:current_positions", {"AAPL": 10.0, "TSLA": 5.0})
    
    # Broker matches -> no change
    manager.reconcile_positions({"AAPL": 10.0, "TSLA": 5.0})
    
    # Broker mismatch -> update internal state
    manager.reconcile_positions({"AAPL": 10.0, "TSLA": 6.0}) # Extra TSLA
    new_state = mock_state_manager.get("portfolio:current_positions")
    assert new_state["TSLA"] == 6.0
