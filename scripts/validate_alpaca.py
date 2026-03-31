import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.alpaca_client import alpaca_client
from src.utils.logger import logger

def main():
    logger.info("Starting Alpaca Connectivity Validation")
    
    # 1. Check if client initialized
    if not alpaca_client.api:
        logger.error("Alpaca Client failed to initialize. Check your .env file.")
        return

    # 2. Fetch Account Info
    logger.info("Fetching account details...")
    account = alpaca_client.get_account()
    if account:
        logger.info("SUCCESS: Connected to Alpaca", 
                    status=account.get("status"),
                    currency=account.get("currency"),
                    buying_power=account.get("buying_power"),
                    account_number=account.get("account_number"))
    else:
        logger.error("FAILED: Could not retrieve account information.")
        return

    # 3. Fetch Positions
    logger.info("Fetching current positions...")
    positions = alpaca_client.get_positions()
    logger.info(f"Retrieve {len(positions)} active positions")
    for p in positions:
        logger.info(f"Position: {p['symbol']} | Qty: {p['qty']} | Market Value: {p['market_value']}")

    # 4. Dry Run Logic Test
    logger.info("Testing DRY RUN safety lock...")
    mode = os.getenv("EXECUTION_MODE", "DRY")
    order_res = alpaca_client.place_order("AAPL", 1)
    if mode == "DRY":
        if order_res and order_res.get("status") == "DRY_RUN_BLOCKED":
            logger.info("SUCCESS: Safety lock confirmed (DRY mode)")
        else:
            logger.error("FAILURE: Safety lock NOT triggered in DRY mode!")
    else:
        logger.warning(f"Note: Running in {mode} mode, order submission was attempted.")

    logger.info("Validation Complete.")

if __name__ == "__main__":
    load_dotenv()
    main()
