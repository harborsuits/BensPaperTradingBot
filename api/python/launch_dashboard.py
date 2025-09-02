#!/usr/bin/env python3
"""
Enhanced Trading Dashboard Launcher

This script provides an easy way to launch the enhanced trading dashboard
with configurable parameters.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

# Try to import dashboard
try:
    from trading_bot.enhanced_dashboard import EnhancedDashboard
except ImportError:
    print("Error importing dashboard. Make sure the trading_bot package is installed.")
    sys.exit(1)

def create_default_dirs():
    """Create default directories for dashboard data."""
    dirs = [
        os.path.expanduser("~/.trading_bot"),
        os.path.expanduser("~/.trading_bot/charts"),
        os.path.expanduser("~/.trading_bot/logs"),
        os.path.expanduser("~/.trading_bot/data")
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")

def copy_default_config(force: bool = False):
    """
    Copy default configuration file if it doesn't exist.
    
    Args:
        force: Whether to overwrite existing configuration
    """
    import shutil
    
    # Source config
    source_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "defaults",
        "dashboard_config.json"
    )
    
    # Destination config
    dest_config = os.path.expanduser("~/.trading_bot/dashboard_config.json")
    
    # Check if we should copy
    if force or not os.path.exists(dest_config):
        try:
            shutil.copy2(source_config, dest_config)
            print(f"Copied default configuration to {dest_config}")
        except Exception as e:
            print(f"Error copying default configuration: {e}")
    else:
        print(f"Configuration already exists at {dest_config}")

def configure_logging(log_level: str = "INFO"):
    """Configure logging for the dashboard."""
    log_dir = os.path.expanduser("~/.trading_bot/logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Map string level to actual level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    # Configure logging
    logging.basicConfig(
        level=level_map.get(log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "dashboard.log")),
            logging.StreamHandler()
        ]
    )

def main():
    """Parse command-line arguments and launch the dashboard."""
    parser = argparse.ArgumentParser(description="Launch the Enhanced Trading Dashboard")
    
    parser.add_argument(
        "--config",
        default=os.path.expanduser("~/.trading_bot/dashboard_config.json"),
        help="Path to dashboard configuration file"
    )
    
    parser.add_argument(
        "--api-url",
        help="Override API URL in configuration"
    )
    
    parser.add_argument(
        "--refresh",
        type=int,
        help="Override refresh interval in seconds"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level"
    )
    
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize dashboard directories and configuration"
    )
    
    parser.add_argument(
        "--reset-config",
        action="store_true",
        help="Reset configuration to defaults"
    )
    
    args = parser.parse_args()
    
    # Create default directories
    create_default_dirs()
    
    # Initialize or reset configuration if requested
    if args.init or args.reset_config:
        copy_default_config(force=args.reset_config)
        if not args.reset_config:
            print("Dashboard initialized. Run without --init to start the dashboard.")
            return 0
    
    # Configure logging
    configure_logging(args.log_level)
    
    # Create dashboard
    dashboard = EnhancedDashboard(config_path=args.config)
    
    # Override configuration with command-line arguments
    if args.api_url:
        dashboard.api_url = args.api_url
        dashboard.api_client = dashboard.api_client.__class__(base_url=args.api_url)
    
    if args.refresh:
        dashboard.refresh_interval = args.refresh
    
    # Run dashboard
    try:
        print(f"Starting Enhanced Trading Dashboard...")
        dashboard.run()
        return 0
    except KeyboardInterrupt:
        print("\nDashboard terminated by user.")
        return 0
    except Exception as e:
        print(f"\nError running dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 