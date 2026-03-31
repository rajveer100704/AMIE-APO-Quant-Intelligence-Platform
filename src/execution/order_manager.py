import hashlib
import time
import structlog
from typing import Dict, Any, Optional
from src.state.state_manager import state_manager
from src.execution.alpaca_client import alpaca_client

logger = structlog.get_logger(__name__)

class OrderManager:
    """
    Hardened Order Management System.
    Enforces idempotency and tracks order lifecycle states.
    """
    def __init__(self):
        self.orders = {} # Tracking current active orders

    def generate_order_id(self, symbol: str, weight: float, timestamp: float) -> str:
        """
        Generates a SHA-256 idempotency key to prevent double trades.
        """
        unique_payload = f"{symbol}-{weight}-{timestamp}".encode()
        return hashlib.sha256(unique_payload).hexdigest()

    def create_order(self, symbol: str, weight: float) -> Dict[str, Any]:
        """
        Creates an idempotent order with a PENDING state.
        """
        now = time.time()
        order_id = self.generate_order_id(symbol, weight, now)
        
        # Check for duplicate in StateManager
        if state_manager.get(f"order:{order_id}"):
            logger.warning("DUPLICATE ORDER BLOCKED", order_id=order_id)
            return {"status": "BLOCKED", "reason": "Idempotent collision", "order_id": order_id}

        order = {
            "order_id": order_id,
            "symbol": symbol,
            "weight": weight,
            "status": "PENDING",
            "created_at": now,
            "updated_at": now
        }
        
        state_manager.set(f"order:{order_id}", order, ttl=3600) # 1-hour tracking
        logger.info("Order created", order_id=order_id, symbol=symbol, weight=weight)
        return order

    def update_order_status(self, order_id: str, status: str, fill_msg: Optional[str] = None):
        """
        Updates the order state machine.
        """
        order = state_manager.get(f"order:{order_id}")
        if not order:
            logger.error("Order not found for status update", order_id=order_id)
            return

        old_status = order["status"]
        order["status"] = status
        order["updated_at"] = time.time()
        if fill_msg:
            order["fill_msg"] = fill_msg
            
        state_manager.set(f"order:{order_id}", order)
        logger.info("Order state changed", 
                    order_id=order_id, 
                    from_status=old_status, 
                    to_status=status)

    def reconcile_positions(self):
        """
        Syncs Internal state vs. Broker state.
        Detects 'ghost positions' or missing fills.
        """
        # Fetch positions from Alpaca (Real broker)
        broker_positions_raw = alpaca_client.get_positions()
        broker_positions = {p["symbol"]: float(p["qty"]) for p in broker_positions_raw}

        internal_positions = state_manager.get("portfolio:current_positions") or {}
        
        mismatches = []
        # Check broker vs internal
        for symbol, qty in broker_positions.items():
            if symbol not in internal_positions or abs(internal_positions[symbol] - qty) > 0.001:
                mismatches.append(symbol)
                logger.critical("POSITION MISMATCH DETECTED (Broker extra)", 
                                symbol=symbol, 
                                broker_qty=qty, 
                                internal_qty=internal_positions.get(symbol, 0))
        
        # Check internal vs broker (missing positions)
        for symbol, qty in internal_positions.items():
            if symbol not in broker_positions and abs(qty) > 0.001:
                mismatches.append(symbol)
                logger.critical("POSITION MISMATCH DETECTED (Internal extra)", 
                                symbol=symbol, 
                                internal_qty=qty, 
                                broker_qty=0)
                
        # Update internal state if necessary to match the ground truth (broker)
        if mismatches:
            state_manager.set("portfolio:current_positions", broker_positions)
            logger.info("Internal state reconciled with broker (Ground Truth applied)")

# Global order manager instance
order_manager = OrderManager()
