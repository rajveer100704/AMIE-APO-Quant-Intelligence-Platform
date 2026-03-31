import os
import yaml
import structlog
from typing import Dict, Any, Optional

logger = structlog.get_logger(__name__)

class RiskGuard:
    """
    Mandatory pre-trade risk layer for AMIE-APO.
    Enforces configuration-driven safety limits and kill-switches.
    """
    def __init__(self, config_path: str = "src/config/risk_config.yaml"):
        self.config_path = config_path
        # Set execution_mode FIRST so _load_config logger can reference it safely
        self.execution_mode = os.getenv("EXECUTION_MODE", "DRY")
        self._load_config()
        # Allow config file to override if env var not explicitly set
        if not os.getenv("EXECUTION_MODE"):
            self.execution_mode = self.config.get("execution_policy", {}).get("default_mode", "DRY")

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                content = yaml.safe_load(f)
                self.config = content.get("risk_limits", {})
                self.policy = content.get("execution_policy", {})
            logger.info("Risk config loaded",
                        kill_switch=self.config.get("kill_switch_drawdown"),
                        mode=self.execution_mode)
        except Exception as e:
            logger.error("Failed to load risk config", error=str(e))
            # Safe defaults if config fails
            self.config = {"kill_switch_drawdown": 0.05, "max_position_per_asset": 0.1}
            self.policy = {"default_mode": "DRY"}

    def validate_order(self, symbol: str, target_weight: float, current_drawdown: float, expected_slippage_bps: int) -> Dict[str, Any]:
        """
        Validates an order against institutional risk limits.
        Returns: {"status": "APPROVED" | "REJECTED", "reason": str}
        """
        # 1. Kill Switch Check
        if current_drawdown > self.config.get("kill_switch_drawdown", 0.1):
            logger.critical("KILL SWITCH TRIPPED", drawdown=current_drawdown)
            return {"status": "REJECTED", "reason": "Kill Switch Active"}

        # 2. Position Limit Check
        if abs(target_weight) > self.config.get("max_position_per_asset", 0.2):
            logger.warning("REJECTED: Position limit violation", symbol=symbol, weight=target_weight)
            return {"status": "REJECTED", "reason": "Position Limit Violation"}

        # 3. Slippage Check
        if expected_slippage_bps > self.config.get("max_slippage_bps", 50):
            logger.warning("REJECTED: Slippage too high", symbol=symbol, slippage=expected_slippage_bps)
            return {"status": "REJECTED", "reason": "Slippage Violation"}

        # 4. Global Execution Mode Check
        if self.execution_mode == "DRY":
            logger.info("DRY RUN: Order validated but not sent", symbol=symbol, weight=target_weight)
            return {"status": "APPROVED", "reason": "Dry Run Validation (No Execution)", "dry_run": True}

        return {"status": "APPROVED", "reason": "Risk Validated", "dry_run": False}

# Global risk guard instance
risk_guard = RiskGuard()
