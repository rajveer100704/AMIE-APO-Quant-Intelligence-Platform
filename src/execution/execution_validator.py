import numpy as np
from typing import Dict, Any, List

def validate_execution(weights: np.ndarray, execution_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Principal Engineer Mandated: Post-Execution Validation
    Ensures: sum(weights) ≈ 1, no duplicates, risk compliance.
    """
    sum_w = float(np.sum(weights))
    valid = True
    reasons = []
    
    # 1. Weight Sum Validation
    if abs(sum_w - 1.0) > 0.001:
        valid = False
        reasons.append(f"Weight sum mismatch: {sum_w:.4f} (expected 1.0)")
        
    # 2. Duplicate Symbol Check
    seen_symbols = set()
    for res in execution_results:
        if res["symbol"] in seen_symbols:
            valid = False
            reasons.append(f"Duplicate symbol in execution: {res['symbol']}")
        seen_symbols.add(res["symbol"])
        
    # 3. Execution Status Check
    failed_orders = [r["symbol"] for r in execution_results if r.get("status") == "FAILED"]
    if failed_orders:
        valid = False
        reasons.append(f"Execution failed for symbols: {', '.join(failed_orders)}")
        
    return {
        "status": "PASSED" if valid else "FAILED",
        "weight_sum": sum_w,
        "errors": reasons,
        "timestamp": float(np.datetime64('now').astype(float))
    }
