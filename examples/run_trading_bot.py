#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path

# Add the parent directory to sys.path to make the trading_bot package importable
sys.path.append(str(Path(__file__).parent.parent))

from trading_bot.trading_bot import TradingBot

def setup_config(config_path=None):
    """
    Set up a configuration file if one doesn't exist
    
    Args:
        config_path: Path to config file
        
    Returns:
        Path to config file
    """
    if config_path and os.path.exists(config_path):
        print(f"Using existing config file at {config_path}")
        return config_path
    
    # Default config path
    default_path = os.path.expanduser("~/.trading_bot/config.json")
    os.makedirs(os.path.dirname(default_path), exist_ok=True)
    
    if os.path.exists(default_path):
        print(f"Using existing config file at {default_path}")
        return default_path
    
    # Create a default config file
    default_config = {
        "tradier_api_key": os.environ.get("TRADIER_API_KEY", ""),
        "tradier_account_id": os.environ.get("TRADIER_ACCOUNT_ID", ""),
        "use_sandbox": True,
        "log_level": "INFO",
        "check_interval_seconds": 60,
        "max_position_pct": 0.05,
        "max_risk_pct": 0.01,
        "default_order_type": "market",
        "default_order_duration": "day",
        "context_refresh_minutes": 60,
        "watchlist": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
        "enabled_strategies": ["micro_momentum", "micro_breakout"],
    }
    
    # Write config to file
    with open(default_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Created default config file at {default_path}")
    print("Please edit this file to add your API keys and account settings.")
    return default_path

def main():
    parser = argparse.ArgumentParser(description='Run the trading bot')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--check-once', action='store_true', help='Check for signals once and exit')
    args = parser.parse_args()
    
    # Setup config
    config_path = setup_config(args.config)
    
    # Initialize the trading bot
    bot = TradingBot(config_path=config_path)
    
    # Print account summary
    account_summary = bot.get_account_summary()
    print("\nAccount Summary:")
    print(f"Account ID: {account_summary.get('account', {}).get('account_number', 'Unknown')}")
    print(f"Balance: ${account_summary.get('account', {}).get('equity', 0):.2f}")
    print(f"Market Context: {account_summary.get('bot_state', {}).get('market_context', {}).get('bias', 'unknown')}")
    print(f"Active Strategies: {', '.join(account_summary.get('bot_state', {}).get('active_strategies', []))}")
    print(f"Watchlist Size: {account_summary.get('bot_state', {}).get('watchlist_count', 0)}")
    
    if args.check_once:
        print("\nChecking for signals once...")
        bot.run_once()
        print("Done.")
    else:
        print("\nStarting trading bot...")
        try:
            bot.run()
        except KeyboardInterrupt:
            print("\nStopping trading bot...")
            bot.stop()
            print("Trading bot stopped.")

if __name__ == "__main__":
    main() 