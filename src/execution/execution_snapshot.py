import os
import json
import time
import structlog
from typing import Any, Dict

logger = structlog.get_logger(__name__)

def save_snapshot(result: Dict[str, Any], base_dir: str = "logs"):
    """
    Saves a JSON snapshot of the execution for audit, debugging, and backtracking.
    Filename format: execution_YYYYMMDD_HHMMSS.json
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(base_dir, f"execution_{timestamp}.json")
        
        # Add metadata to snapshot
        from datetime import datetime
        snapshot = {
            "snapshot_id": f"EX-{timestamp}",
            "captured_at": datetime.utcnow().isoformat(),
            "data": result
        }
        
        with open(filename, "w") as f:
            json.dump(snapshot, f, indent=2)
            
        logger.info("execution_snapshot_saved", path=filename, snapshot_id=snapshot["snapshot_id"])
        return filename
    except Exception as e:
        logger.error("failed_to_save_snapshot", error=str(e))
        return None

def list_snapshots(base_dir: str = "logs"):
    """Lists all available execution snapshots."""
    if not os.path.exists(base_dir):
        return []
    return sorted([f for f in os.listdir(base_dir) if f.startswith("execution_") and f.endswith(".json")], reverse=True)
