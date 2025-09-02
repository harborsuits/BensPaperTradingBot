#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import signal
import time
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the scheduler
from trading_bot.market_context.adaptive_context_scheduler import AdaptiveContextScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/adaptive_context.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Global scheduler instance for signal handling
scheduler = None

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {sig}, shutting down...")
    if scheduler:
        scheduler.stop()
    sys.exit(0)

def main():
    """Run the adaptive context scheduler"""
    global scheduler
    
    parser = argparse.ArgumentParser(description="Adaptive Trading Context Scheduler")
    parser.add_argument("--market-start", help="Market hours start time (24h format, e.g. '05:00')", default="05:00")
    parser.add_argument("--market-end", help="Market hours end time (24h format, e.g. '16:00')", default="16:00")
    parser.add_argument("--market-interval", help="Update interval during market hours (minutes)", type=int, default=15)
    parser.add_argument("--after-interval", help="Update interval after market hours (minutes)", type=int, default=60)
    parser.add_argument("--output-dir", help="Output directory for context files", default="data/market_context")
    parser.add_argument("--run-now", action="store_true", help="Run update immediately then start scheduler")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Configure the scheduler
    config = {
        "MARKETAUX_API_KEY": os.getenv("MARKETAUX_API_KEY", "7PgROm6BE4m6ejBW8unmZnnYS6kIygu5lwzpfd9K"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OUTPUT_DIR": os.getenv("CONTEXT_OUTPUT_DIR", args.output_dir),
        "MARKET_HOURS_START": args.market_start,
        "MARKET_HOURS_END": args.market_end,
        "MARKET_HOURS_INTERVAL": args.market_interval,
        "AFTER_HOURS_INTERVAL": args.after_interval,
        "STRATEGY_LIST": [
            "iron_condor", "gap_fill_daytrade", "theta_spread",
            "breakout_swing", "pullback_to_moving_average",
            "volatility_squeeze", "earnings_strangle"
        ]
    }
    
    # Create scheduler
    scheduler = AdaptiveContextScheduler(config)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run now if requested
    if args.run_now:
        logger.info("Running context generation immediately")
        scheduler.update_market_context()
    
    # Start scheduler
    scheduler.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        scheduler.stop()

if __name__ == "__main__":
    main() 