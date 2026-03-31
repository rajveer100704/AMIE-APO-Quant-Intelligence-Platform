import pytest
from unittest.mock import patch, MagicMock
from src.execution.alpaca_client import AlpacaClient

@pytest.fixture
def mock_alpaca_api():
    with patch("alpaca_trade_api.REST") as mock:
        yield mock

@pytest.mark.unit
def test_alpaca_client_initialization(mock_alpaca_api):
    client = AlpacaClient()
    assert client.api is not None

@pytest.mark.unit
def test_alpaca_client_get_account(mock_alpaca_api):
    # Setup mock response
    mock_instance = mock_alpaca_api.return_value
    mock_instance.get_account.return_value = MagicMock(_raw={"id": "uuid", "status": "ACTIVE"})
    
    client = AlpacaClient()
    account = client.get_account()
    assert account["status"] == "ACTIVE"
    assert account["id"] == "uuid"

@pytest.mark.unit
def test_alpaca_client_dry_run_safety():
    """Test that place_order is blocked in DRY mode even if API is initialized."""
    with patch.dict("os.environ", {"EXECUTION_MODE": "DRY"}):
        client = AlpacaClient()
        # Mocking the api just in case
        client.api = MagicMock()
        
        res = client.place_order("AAPL", 10)
        assert res["status"] == "DRY_RUN_BLOCKED"
        # Ensure the real submit_order was never called
        client.api.submit_order.assert_not_called()

@pytest.mark.unit
def test_alpaca_client_order_submission(mock_alpaca_api):
    """Test successful order submission in PAPER mode (mocked)."""
    with patch.dict("os.environ", {"EXECUTION_MODE": "PAPER"}):
        client = AlpacaClient()
        mock_instance = mock_alpaca_api.return_value
        mock_instance.submit_order.return_value = MagicMock(id="order_123", _raw={"id": "order_123", "status": "accepted"})
        
        res = client.place_order("TSLA", 5, side="buy")
        assert res["status"] == "accepted"
        assert res["id"] == "order_123"
        mock_instance.submit_order.assert_called_once()
