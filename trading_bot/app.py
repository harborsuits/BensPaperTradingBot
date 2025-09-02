#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot - Main Application Entry Point

This module serves as the main entry point for the trading bot application,
initializing all necessary components and starting the main orchestrator.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from trading_bot.core.main_orchestrator import MainOrchestrator, setup_signal_handlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/trading_bot.log", mode='a')
    ]
)

logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist."""
    directories = ["logs", "data", "config"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Trading Bot Application")
    
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--mode",
        choices=["live", "paper", "backtest"],
        default="paper",
        help="Trading mode: live, paper, or backtest"
    )
    
    parser.add_argument(
        "--strategy",
        help="Only run specific strategy (by name)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the trading bot application."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set debug level if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Ensure required directories exist
        ensure_directories()
        
        # Log startup information
        logger.info("Starting Trading Bot")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Config: {args.config}")
        
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Initialize main orchestrator
        orchestrator = MainOrchestrator(str(config_path))
        
        # Set up signal handlers for graceful shutdown
        setup_signal_handlers(orchestrator)
        
        # Start orchestrator
        orchestrator.start()
        
        # If we get here, the orchestrator has stopped
        logger.info("Trading Bot exiting normally")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Trading Bot interrupted by user")
        return 0
        
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 