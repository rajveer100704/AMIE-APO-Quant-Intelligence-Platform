import os
import sys

os.chdir('c:/Users/BIT/AMIE_APO')
sys.path.insert(0, '.')

# Load env vars
from dotenv import load_dotenv
load_dotenv()

errors = []
passes = []

# Test 1: CacheLayer
try:
    from src.api.cache import CacheLayer
    c = CacheLayer()
    passes.append(f'CacheLayer OK, backend={c.backend}')
    print(f'[PASS] CacheLayer OK, backend={c.backend}')
except Exception as e:
    errors.append(f'CacheLayer: {e}')
    print(f'[FAIL] CacheLayer: {e}')

# Test 2: RiskGuard
try:
    from src.execution.risk_guard import RiskGuard
    rg = RiskGuard()
    passes.append(f'RiskGuard OK, mode={rg.execution_mode}')
    print(f'[PASS] RiskGuard OK, mode={rg.execution_mode}')
except Exception as e:
    errors.append(f'RiskGuard: {e}')
    print(f'[FAIL] RiskGuard: {e}')

# Test 3: AlpacaClient
try:
    from src.execution.alpaca_client import AlpacaClient
    ac = AlpacaClient()
    account = ac.get_account()
    status = account.get("status") if account else "NO ACCOUNT RETURNED"
    passes.append(f'AlpacaClient OK, account_status={status}')
    print(f'[PASS] AlpacaClient OK, account_status={status}')
except Exception as e:
    errors.append(f'AlpacaClient: {e}')
    print(f'[FAIL] AlpacaClient: {e}')

# Test 4: OrderManager
try:
    from src.execution.order_manager import OrderManager
    om = OrderManager()
    passes.append('OrderManager OK')
    print('[PASS] OrderManager OK')
except Exception as e:
    errors.append(f'OrderManager: {e}')
    print(f'[FAIL] OrderManager: {e}')

# Test 5: AMISFusion
try:
    from src.optimizer.amis_fusion import AMISFusion
    fusion = AMISFusion()
    passes.append('AMISFusion OK')
    print('[PASS] AMISFusion OK')
except Exception as e:
    errors.append(f'AMISFusion: {e}')
    print(f'[FAIL] AMISFusion: {e}')

# Test 6: NumbaPortfolioSolver
try:
    from src.optimizer.numba_solver import NumbaPortfolioSolver
    solver = NumbaPortfolioSolver()
    passes.append('NumbaPortfolioSolver OK')
    print('[PASS] NumbaPortfolioSolver OK')
except Exception as e:
    errors.append(f'NumbaPortfolioSolver: {e}')
    print(f'[FAIL] NumbaPortfolioSolver: {e}')

# Test 7: Server app import
try:
    from src.api.server import app
    passes.append('server.app import OK')
    print('[PASS] server.app import OK')
except Exception as e:
    errors.append(f'server.app: {e}')
    print(f'[FAIL] server.app: {e}')

print(f'\n=== DIAGNOSTIC SUMMARY ===')
print(f'PASSED: {len(passes)}')
print(f'FAILED: {len(errors)}')
for err in errors:
    print(f'  [ERROR] {err}')
