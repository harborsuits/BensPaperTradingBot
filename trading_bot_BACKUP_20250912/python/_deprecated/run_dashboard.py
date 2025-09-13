#!/usr/bin/env python3
"""
Dashboard Launcher

This script launches the Streamlit-based trading dashboard.
"""

import os
import sys
import subprocess
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to launch the dashboard"""
    parser = argparse.ArgumentParser(description="Launch the trading dashboard")
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501,
        help="Port to run the dashboard on (default: 8501)"
    )
    
    parser.add_argument(
        "--alpaca-key",
        type=str,
        help="Alpaca API key (can also be set via ALPACA_API_KEY env var)"
    )
    
    parser.add_argument(
        "--alpaca-secret",
        type=str,
        help="Alpaca API secret (can also be set via ALPACA_API_SECRET env var)"
    )
    
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data instead of connecting to real data sources"
    )
    
    args = parser.parse_args()
    
    # Set environment variables if provided
    if args.alpaca_key:
        os.environ["ALPACA_API_KEY"] = args.alpaca_key
    
    if args.alpaca_secret:
        os.environ["ALPACA_API_SECRET"] = args.alpaca_secret
    
    # Determine dashboard path
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "visualization",
        "live_trading_dashboard.py"
    )
    
    if not os.path.exists(dashboard_path):
        logger.error(f"Dashboard file not found at: {dashboard_path}")
        sys.exit(1)
    
    # Construct command
    cmd = [
        "streamlit", "run", dashboard_path,
        "--server.port", str(args.port),
        "--server.address", "0.0.0.0"  # Allow external connections
    ]
    
    # Add additional Streamlit arguments
    cmd.extend([
        "--browser.serverAddress", "localhost",
        "--server.headless", "true",
        "--theme.base", "dark"
    ])
    
    try:
        logger.info(f"Launching dashboard on port {args.port}")
        if args.mock_data:
            logger.info("Using mock data for demonstration")
        
        # Run the Streamlit command
        process = subprocess.Popen(cmd)
        
        # Wait for the process to complete (this will block until the user stops the dashboard)
        process.wait()
    
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 