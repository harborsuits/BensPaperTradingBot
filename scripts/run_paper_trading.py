#!/usr/bin/env python3
"""
Simplified entry point for paper trading with BensBot.
This script initializes the trading bot in paper trading mode with minimal dependencies.
"""

import os
import sys
import logging
import time
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/paper_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("paper_trading")

# Load environment variables from .env file
load_dotenv()

def main():
    """Main entry point for paper trading"""
    logger.info("Starting BensBot in paper trading mode")
    
    try:
        # Import trading components here to catch any import errors
        from trading_bot.brokers.paper.adapter import PaperTradeAdapter 
        from trading_bot.core.main_orchestrator import MainOrchestrator  # Updated path
        from trading_bot.core.risk_manager import RiskManager
        from trading_bot.brokers.trade_executor import TradeExecutor
        from trading_bot.analytics.trade_logger import TradeLogger
        from trading_bot.persistence.connection_manager import ConnectionManager
        from trading_bot.persistence.recovery_manager import RecoveryManager
        
        # Initialize persistence layer if MongoDB is available
        try:
            conn_manager = ConnectionManager(
                mongo_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017/bensbot_trading"),
                redis_host=os.getenv("REDIS_HOST", "localhost"),
                redis_port=int(os.getenv("REDIS_PORT", "6379")),
                redis_db=int(os.getenv("REDIS_DB", "0")),
                database_name=os.getenv("MONGODB_DATABASE", "bensbot_trading")
            )
            recovery_manager = RecoveryManager(conn_manager)
            logger.info("Successfully connected to persistence layer")
            use_persistence = True
        except Exception as e:
            logger.warning(f"Could not initialize persistence layer: {e}")
            logger.warning("Running without persistence - state will not be saved")
            conn_manager = None
            recovery_manager = None
            use_persistence = False
        
        # Initialize paper trading adapter
        paper_adapter = PaperTradeAdapter(
            initial_balance=float(os.getenv("PAPER_INITIAL_BALANCE", "100000")),
            commission_rate=float(os.getenv("PAPER_COMMISSION_RATE", "0.0005")),
            slippage_model="proportional",
            slippage_factor=float(os.getenv("PAPER_SLIPPAGE_FACTOR", "0.0001")),
            data_source="yfinance"  # Use Yahoo Finance for market data
        )
        
        # Initialize trade logger
        trade_logger = TradeLogger(
            log_file="data/trades/paper_trades.csv",
            include_timestamps=True
        )
        
        # Initialize risk manager with conservative settings
        risk_manager = RiskManager(
            max_portfolio_risk=float(os.getenv("RISK_MAX_PORTFOLIO_RISK", "0.02")),
            max_position_risk=float(os.getenv("RISK_MAX_POSITION_RISK", "0.01")),
            max_leverage=float(os.getenv("RISK_MAX_LEVERAGE", "1.5")),
            circuit_breaker_enabled=True,
            max_drawdown_pct=float(os.getenv("RISK_MAX_DRAWDOWN_PCT", "5.0")),
            max_intraday_drawdown_pct=float(os.getenv("RISK_MAX_INTRADAY_DRAWDOWN", "3.0")),
            trade_logger=trade_logger
        )
        
        # Initialize trade executor
        trade_executor = TradeExecutor(
            broker=paper_adapter,
            risk_manager=risk_manager,
            max_retries=3,
            retry_delay=1.0,
            trade_logger=trade_logger
        )
        
        # Initialize main orchestrator
        orchestrator = MainOrchestrator(
            brokers={"paper": paper_adapter},
            trade_executor=trade_executor,
            risk_manager=risk_manager,
            recovery_manager=recovery_manager if use_persistence else None
        )
        
        # Recover state if persistence is enabled
        if use_persistence and recovery_manager:
            logger.info("Attempting to recover state from persistence layer")
            recovered = orchestrator.recover_state()
            if recovered:
                logger.info("Successfully recovered state")
            else:
                logger.info("No state to recover or recovery failed")
        
        # Start trading loop
        logger.info("Starting paper trading mode - press Ctrl+C to exit")
        
        # Main trading loop
        while True:
            # Sync state to persistence layer periodically
            if use_persistence:
                orchestrator.sync_state()
            
            # Update positions and account info
            orchestrator.update_all_positions()
            
            # Print status update
            positions = paper_adapter.get_positions()
            balance = paper_adapter.get_account_balance()
            logger.info(f"Account balance: ${balance:.2f}")
            logger.info(f"Open positions: {len(positions)}")
            for pos in positions:
                logger.info(f"  {pos['symbol']}: {pos['quantity']} shares at ${pos['avg_price']:.2f}")
            
            # Sleep for a bit
            time.sleep(60)  # Update every minute
            
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    except Exception as e:
        logger.exception(f"Unexpected error in paper trading: {e}")
    finally:
        logger.info("Paper trading mode stopped")
        
        # Close connections
        if 'conn_manager' in locals() and conn_manager:
            conn_manager.close_all_connections()

if __name__ == "__main__":
    # Make sure logs directory exists
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/trades", exist_ok=True)
    
    # Start the main function
    main()
