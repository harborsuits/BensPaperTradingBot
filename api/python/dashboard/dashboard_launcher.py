#!/usr/bin/env python3
"""
Enhanced Dashboard Launcher for BensBot Trading System

This script provides a reliable way to start and stop the trading dashboard
regardless of directory structure or path issues. It handles all dependencies
and ensures the correct Python environment is used.
"""
import os
import sys
import subprocess
import webbrowser
import time
import signal
import argparse
import pkg_resources

# Colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# Set the absolute paths to avoid any issues with special characters
DASHBOARD_DIR = "/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"
PROJECT_ROOT = "/Users/bendickinson/Desktop/Trading:BenBot"

def print_colored(message, color):
    """Print a colored message to the terminal."""
    print(f"{color}{message}{NC}")
    sys.stdout.flush()  # Ensure message is displayed immediately

def check_and_install_dependencies():
    """Check and install all required dependencies."""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "plotly",
        "yfinance",
        "matplotlib",
        "ta",
        "scikit-learn",
        "pytz",
        "ccxt",
        "requests",
        "websocket-client",  # For real-time data feeds
        "backtrader",        # Common trading framework
        "python-binance",    # For Binance API
        "alpaca-trade-api",  # For Alpaca API
        "polygon-api-client", # For Polygon.io data
        "pymongo",           # For database access
        "SQLAlchemy"         # For SQL database access
    ]
    
    # Get currently installed packages
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    
    # Check which packages need to be installed
    packages_to_install = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
    
    if packages_to_install:
        print_colored(f"Installing {len(packages_to_install)} missing packages...", YELLOW)
        
        try:
            # Use subprocess directly with the same Python interpreter
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--break-system-packages"
            ] + packages_to_install)
            
            print_colored("Successfully installed required packages.", GREEN)
        except subprocess.CalledProcessError as e:
            print_colored(f"Error installing packages: {str(e)}", RED)
            print_colored(f"Try installing manually: pip install --break-system-packages {' '.join(packages_to_install)}", YELLOW)
            return False
    else:
        print_colored("All required packages are already installed.", GREEN)
    
    return True

def is_streamlit_running():
    """Check if Streamlit is already running on port 8501."""
    try:
        # Try to connect to the port to see if it's open
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', 8501)) == 0
    except:
        return False

def find_streamlit_process():
    """Find the Streamlit process using port 8501."""
    try:
        # For macOS, use lsof to find process using port 8501
        result = subprocess.run(
            ["lsof", "-i", ":8501", "-t"],
            capture_output=True, text=True
        )
        
        if result.stdout.strip():
            pid = int(result.stdout.strip())
            return pid
        
        return None
    except:
        return None

def stop_dashboard():
    """Stop the running Streamlit dashboard."""
    pid = find_streamlit_process()
    
    if pid:
        print_colored(f"Stopping Streamlit dashboard (PID: {pid})...", YELLOW)
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            
            # Check if it's still running
            if not is_streamlit_running():
                print_colored("Dashboard stopped successfully.", GREEN)
                return True
            else:
                # Try a stronger signal
                print_colored("Dashboard did not stop gracefully. Force stopping...", YELLOW)
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
                if not is_streamlit_running():
                    print_colored("Dashboard force stopped successfully.", GREEN)
                    return True
                else:
                    print_colored("Failed to stop dashboard. Try stopping it manually.", RED)
                    return False
        except Exception as e:
            print_colored(f"Error stopping dashboard: {str(e)}", RED)
            return False
    else:
        print_colored("No running dashboard found.", YELLOW)
        return True  # Not running is considered a success

def start_dashboard():
    """Start the Streamlit dashboard."""
    # First, check if dashboard is already running
    if is_streamlit_running():
        print_colored("Dashboard is already running. Opening browser...", YELLOW)
        webbrowser.open("http://localhost:8501")
        
        # Ask if user wants to restart
        response = input("Do you want to restart it? (y/n): ").strip().lower()
        if response == 'y':
            if not stop_dashboard():
                print_colored("Failed to stop existing dashboard. Please try again.", RED)
                return False
            time.sleep(1)  # Give it a moment
        else:
            return True
    
    # Install dependencies if needed
    if not check_and_install_dependencies():
        return False
    
    # Add the project root to Python path
    os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ.get('PYTHONPATH', '')}"
    
    print_colored("Starting Streamlit dashboard...", GREEN)
    print_colored("The dashboard will be available at: http://localhost:8501", YELLOW)
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:8501")
    
    import threading
    threading.Thread(target=open_browser).start()
    
    # Change to the dashboard directory
    os.chdir(DASHBOARD_DIR)
    
    # Run streamlit as a subprocess and wait for it
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8501", 
        "--server.address=localhost"
    ])
    
    print_colored(f"Dashboard started with PID: {process.pid}", GREEN)
    
    # Wait for the process to complete (this blocks the script)
    process.wait()
    
    return True

def restart_dashboard():
    """Restart the dashboard."""
    print_colored("Restarting dashboard...", YELLOW)
    if stop_dashboard():
        time.sleep(1)  # Give it a moment to fully stop
        return start_dashboard()
    else:
        print_colored("Failed to stop dashboard for restart.", RED)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BensBot Trading Dashboard Controller')
    parser.add_argument('action', nargs='?', default='start', 
                      choices=['start', 'stop', 'restart'],
                      help='Action to perform: start, stop, or restart (default: start)')
    
    args = parser.parse_args()
    
    if args.action == 'stop':
        stop_dashboard()
    elif args.action == 'restart':
        restart_dashboard()
    else:  # start
        start_dashboard()
