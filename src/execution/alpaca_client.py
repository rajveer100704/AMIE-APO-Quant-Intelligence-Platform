import os
import alpaca_trade_api as tradeapi
import structlog
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load env in case it's not already loaded
load_dotenv()

logger = structlog.get_logger(__name__)

class AlpacaClient:
    """
    Hardened Alpaca REST Client for AMIE-APO.
    Supports Paper and Live trading with built-in safety controls.
    """
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.execution_mode = os.getenv("EXECUTION_MODE", "DRY") # Must be 'LIVE' to send orders

        if not self.api_key or not self.secret_key:
            logger.error("Missing Alpaca API credentials in environment")
            self.api = None
        else:
            try:
                self.api = tradeapi.REST(
                    self.api_key,
                    self.secret_key,
                    self.base_url,
                    api_version='v2'
                )
                logger.info("Alpaca REST client initialized", 
                            base_url=self.base_url, 
                            mode=self.execution_mode)
            except Exception as e:
                logger.error("Failed to initialize Alpaca client", error=str(e))
                self.api = None

    def get_account(self) -> Optional[Dict[str, Any]]:
        """Returns the Alpaca account information."""
        if not self.api:
            return None
        try:
            account = self.api.get_account()
            return account._raw
        except Exception as e:
            logger.error("Failed to fetch Alpaca account", error=str(e))
            return None

    def get_positions(self) -> List[Dict[str, Any]]:
        """Returns currently held positions."""
        if not self.api:
            return []
        try:
            positions = self.api.list_positions()
            return [p._raw for p in positions]
        except Exception as e:
            logger.error("Failed to fetch Alpaca positions", error=str(e))
            return []

    def place_order(self, symbol: str, qty: int, side: str = "buy", 
                   type: str = "market", time_in_force: str = "gtc") -> Optional[Dict[str, Any]]:
        """
        Submits an order to Alpaca.
        STRICTLY BLOCKED if execution_mode is 'DRY'.
        """
        if self.execution_mode == "DRY":
            logger.warning("DRY RUN: Order placement blocked", symbol=symbol, qty=qty)
            return {"status": "DRY_RUN_BLOCKED", "symbol": symbol, "qty": qty}

        if not self.api:
            logger.error("Alpaca client not initialized, cannot place order")
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=type,
                time_in_force=time_in_force
            )
            logger.info("Order submitted to Alpaca", 
                        order_id=order.id, 
                        symbol=symbol, 
                        qty=qty, 
                        side=side)
            return order._raw
        except Exception as e:
            logger.error("Failed to submit Alpaca order", symbol=symbol, error=str(e))
            return None

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Fetches order status from Alpaca."""
        if not self.api:
            return None
        try:
            order = self.api.get_order(order_id)
            return order._raw
        except Exception as e:
            logger.error("Failed to fetch Alpaca order", order_id=order_id, error=str(e))
            return None

# Singleton instance for high-frequency access
alpaca_client = AlpacaClient()
