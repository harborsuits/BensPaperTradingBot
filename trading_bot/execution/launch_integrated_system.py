#!/usr/bin/env python
"""
Launch Integrated Trading System

This script provides a simple way to launch the integrated trading system
with various configurations and modes. It handles configuration setup,
initializes the system, and provides a dashboard URL for monitoring.
"""

import os
import sys
import argparse
import subprocess
import time
import webbrowser
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from trading_bot.execution.integrated_trading_runner import IntegratedTradingSystem
from trading_bot.alerts.telegram_alerts import send_system_alert

def setup_environment():
    """Setup environment for the trading system"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/paper_trading', exist_ok=True)
    os.makedirs('dashboard', exist_ok=True)
    
    # Create log file for current session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/trading_session_{timestamp}.log"
    
    # Return configuration
    return {
        'log_file': log_file,
        'timestamp': timestamp
    }

def start_dashboard(port=8501):
    """Start the Streamlit dashboard in a subprocess"""
    try:
        # Define dashboard path
        dashboard_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../dashboard/main.py')
        )
        
        # Check if dashboard exists
        if not os.path.exists(dashboard_path):
            print(f"Dashboard not found at {dashboard_path}")
            return None
        
        # Start dashboard in background
        cmd = [
            "streamlit", "run", dashboard_path,
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        dashboard_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for dashboard to start
        time.sleep(5)
        
        # Open browser
        dashboard_url = f"http://localhost:{port}"
        webbrowser.open(dashboard_url)
        
        print(f"Dashboard started at {dashboard_url}")
        return dashboard_process, dashboard_url
        
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return None, None

def main():
    """Main function to launch the integrated trading system"""
    parser = argparse.ArgumentParser(description='Launch Integrated Trading System')
    
    # Operation mode
    parser.add_argument(
        '--mode',
        choices=['live', 'simulation', 'backtest'],
        default='simulation',
        help='Operation mode'
    )
    
    # Trading parameters
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        default="SPY,QQQ,AAPL,MSFT,GOOGL",
        help='Symbols to trade (comma-separated)'
    )
    
    # Data source
    parser.add_argument(
        '--data-source',
        choices=['alpaca', 'tradier', 'finnhub'],
        default='alpaca',
        help='Real-time data source'
    )
    
    # Time parameters
    parser.add_argument(
        '--interval',
        type=float,
        default=60.0,
        help='Trading interval in seconds'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=8.0,
        help='Simulation duration in hours'
    )
    
    parser.add_argument(
        '--speed',
        type=float,
        default=60.0,
        help='Simulation speed multiplier'
    )
    
    # Dashboard
    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Start dashboard'
    )
    
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8501,
        help='Dashboard port'
    )
    
    # Telegram alerts
    parser.add_argument(
        '--alerts',
        action='store_true',
        help='Enable Telegram alerts'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup environment
    env_config = setup_environment()
    print(f"Trading session started at {env_config['timestamp']}")
    print(f"Logs will be written to {env_config['log_file']}")
    
    # Start dashboard if requested
    dashboard_process = None
    dashboard_url = None
    if args.dashboard:
        dashboard_process, dashboard_url = start_dashboard(args.dashboard_port)
    
    # Initialize trading system
    system = IntegratedTradingSystem()
    system.initialize(
        initial_cash=args.capital,
        data_source=args.data_source
    )
    
    # Set symbols to watch
    symbols = [s.strip() for s in args.symbols.split(',')]
    print(f"Watching symbols: {symbols}")
    
    # Add symbols to controller
    if system.controller:
        system.controller.set_watched_symbols(symbols)
    
    # Send startup alert if enabled
    if args.alerts:
        send_system_alert(
            component="Trading System",
            status="starting",
            message=f"Starting integrated trading system in {args.mode} mode",
            severity="info"
        )
    
    try:
        # Run based on mode
        if args.mode == 'live':
            print("Starting in LIVE mode")
            system.start(trading_interval=args.interval)
        
        elif args.mode == 'simulation':
            print("Starting in SIMULATION mode")
            system.run_overnight_simulation(
                hours=args.duration,
                interval_minutes=args.interval / 60,
                speed_multiplier=args.speed
            )
        
        elif args.mode == 'backtest':
            print("Starting in BACKTEST mode")
            # This would use the existing backtesting functionality
            from trading_bot.backtest.adaptive_backtest_runner import main as run_backtest
            run_backtest()
        
        else:
            print(f"Unknown mode: {args.mode}")
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error running trading system: {e}")
    finally:
        # Stop system
        system.stop()
        
        # Send shutdown alert if enabled
        if args.alerts:
            send_system_alert(
                component="Trading System",
                status="stopped",
                message="Integrated trading system shutdown complete",
                severity="info"
            )
        
        # Stop dashboard if started
        if dashboard_process:
            dashboard_process.terminate()
            print("Dashboard stopped")
        
        print("Trading session ended")

if __name__ == "__main__":
    main()
