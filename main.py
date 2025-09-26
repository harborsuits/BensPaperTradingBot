#!/usr/bin/env python3
"""
BenBot Main Entry Point
=======================

This is the single, consolidated entry point for the BenBot trading system.
It replaces the 15+ different entry points that were scattered throughout the codebase.

Usage:
    python main.py [command] [options]

Commands:
    server      Start the API server (default)
    bot         Start the trading bot
    backtest    Run backtesting
    dashboard   Start the dashboard (deprecated - use web UI)
    
Examples:
    python main.py                  # Start API server (default)
    python main.py server           # Start API server explicitly
    python main.py bot              # Start trading bot
    python main.py backtest SPY     # Run backtest on SPY
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def start_server(args):
    """Start the API server"""
    print("Starting BenBot API Server...")
    
    # Check which server to use based on environment or argument
    server_type = os.environ.get('BENBOT_SERVER_TYPE', 'live-api')
    
    if server_type == 'simple':
        print("Using simple_server.py...")
        from simple_server import app
        app.run(host='0.0.0.0', port=3000, debug=False)
    elif server_type == 'minimal':
        print("Using minimal_server.py...")
        from minimal_server import app
        app.run(host='0.0.0.0', port=3000, debug=False)
    elif server_type == 'live-api':
        print("Using live-api server (recommended)...")
        # The live-api server is Node.js based, so we need to launch it
        import subprocess
        subprocess.run(['node', 'live-api/server.js'], cwd=PROJECT_ROOT)
    else:
        print(f"Unknown server type: {server_type}")
        sys.exit(1)

def start_bot(args):
    """Start the trading bot"""
    print("Starting BenBot Trading Bot...")
    
    # Import and run the orchestrator
    try:
        from trading_bot.orchestrator import main as orchestrator_main
        orchestrator_main()
    except ImportError:
        print("Error: Could not import trading bot orchestrator")
        print("Make sure trading_bot/orchestrator.py exists")
        sys.exit(1)

def run_backtest(args):
    """Run backtesting"""
    print(f"Running backtest for {args.symbol}...")
    
    try:
        from trading_bot.backtest.run_backtest import main as backtest_main
        # Pass symbol to backtest
        sys.argv = ['run_backtest.py', '--symbol', args.symbol]
        if args.strategy:
            sys.argv.extend(['--strategy', args.strategy])
        backtest_main()
    except ImportError:
        print("Error: Could not import backtesting module")
        print("Make sure trading_bot/backtest/run_backtest.py exists")
        sys.exit(1)

def start_dashboard(args):
    """Start the dashboard (deprecated)"""
    print("Note: The Python dashboard is deprecated. Use the web UI instead.")
    print("To start the web UI, run: npm run dev in new-trading-dashboard/")
    print()
    print("Starting legacy Python dashboard anyway...")
    
    try:
        from trading_bot.dashboard.main import main as dashboard_main
        dashboard_main()
    except ImportError:
        print("Error: Could not import dashboard module")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='BenBot Trading System - Unified Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Server command (default)
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--type', choices=['simple', 'minimal', 'live-api'], 
                              default='live-api', help='Server type to use')
    
    # Bot command
    bot_parser = subparsers.add_parser('bot', help='Start trading bot')
    bot_parser.add_argument('--config', help='Config file path')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument('symbol', help='Symbol to backtest (e.g., SPY)')
    backtest_parser.add_argument('--strategy', help='Strategy to use')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Start dashboard (deprecated)')
    
    args = parser.parse_args()
    
    # Default to server if no command specified
    if not args.command:
        args.command = 'server'
    
    # Route to appropriate function
    commands = {
        'server': start_server,
        'bot': start_bot,
        'backtest': run_backtest,
        'dashboard': start_dashboard
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
