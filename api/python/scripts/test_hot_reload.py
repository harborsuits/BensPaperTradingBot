#!/usr/bin/env python3
"""
Hot Reload Test Script

This script demonstrates the configuration hot-reload capability.
It watches the system configuration file and reloads when changes are detected.
"""

import os
import sys
import time
import logging
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.core.simple_config import load_config
from trading_bot.core.config_watcher import init_config_watcher, ConfigWatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hot_reload_test")


def config_reload_callback(new_config):
    """Called when configuration is reloaded"""
    logger.info("🔄 Configuration reloaded!")
    logger.info(f"📊 Watched symbols: {new_config.get('watched_symbols', [])}")
    logger.info(f"💰 Initial capital: {new_config.get('initial_capital', 0)}")
    logger.info(f"⚠️ Risk per trade: {new_config.get('risk_per_trade', 0)}")
    logger.info(f"🕒 Trading hours: {new_config.get('trading_hours', {})}")


def main():
    """Main entry point"""
    config_path = "config/system_config.json"
    
    # Create absolute path
    config_path = os.path.abspath(config_path)
    
    logger.info("🚀 Starting hot reload test")
    logger.info(f"👀 Watching configuration file: {config_path}")
    
    # Load initial configuration
    try:
        config = load_config(config_path)
        logger.info("✅ Initial configuration loaded successfully")
        logger.info(f"📊 Watched symbols: {config.get('watched_symbols', [])}")
    except Exception as e:
        logger.error(f"❌ Error loading initial configuration: {str(e)}")
        return 1
    
    # Initialize configuration watcher
    watcher = init_config_watcher(
        config_path=config_path,
        reload_callback=config_reload_callback,
        interval_seconds=10  # Check every 10 seconds for changes
    )
    
    # Start watching
    try:
        watcher.start()
        logger.info("🔍 Now watching for configuration changes")
        logger.info("ℹ️ Try modifying the config file to see hot reload in action")
        logger.info("ℹ️ Press Ctrl+C to exit")
        
        # Create a simple edit suggestion file to help users test
        suggestion_path = Path("config/suggested_edit.json")
        with open(suggestion_path, "w") as f:
            sample_config = config.copy()
            if "watched_symbols" in sample_config:
                # Add a new symbol if not already present
                if "TSLA" not in sample_config["watched_symbols"]:
                    sample_config["watched_symbols"].append("TSLA")
                # Remove a symbol if present
                elif "AAPL" in sample_config["watched_symbols"]:
                    sample_config["watched_symbols"].remove("AAPL")
            
            # Modify initial capital
            if "initial_capital" in sample_config:
                sample_config["initial_capital"] = int(sample_config["initial_capital"] * 1.5)
            
            json.dump(sample_config, f, indent=2)
        
        logger.info(f"📝 Created a suggested edit file at {suggestion_path}")
        logger.info("   You can copy its contents to system_config.json to test hot reload")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("👋 Exiting due to user interrupt")
    finally:
        # Stop the watcher
        watcher.stop()
        logger.info("🛑 Stopped watching for configuration changes")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
