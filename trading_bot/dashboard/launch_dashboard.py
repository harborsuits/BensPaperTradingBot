"""
Simple script to launch the trading dashboard directly.
This avoids issues with virtual environments and special characters in paths.
Supports start and stop functionality.
"""
import os
import sys
import subprocess
import webbrowser
import time
import argparse
import signal
import psutil

# Colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def print_colored(message, color):
    """Print a colored message to the terminal."""
    print(f"{color}{message}{NC}")

# Project paths
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(DASHBOARD_DIR))

print_colored(f"Dashboard directory: {DASHBOARD_DIR}", YELLOW)

# Check if streamlit is installed
try:
    import streamlit
    print_colored("Streamlit is already installed.", GREEN)
except ImportError:
    print_colored("Streamlit is not installed. Installing now...", YELLOW)
    try:
        # Try to install with --break-system-packages as suggested in the error message
        # Include all required dependencies for the trading bot
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--break-system-packages", "streamlit", "pandas", "numpy", "plotly", "requests",
            "yfinance", "matplotlib", "ta", "scikit-learn", "pytz", "ccxt", "psutil"
        ])
        print_colored("Successfully installed required packages.", GREEN)
    except subprocess.CalledProcessError:
        print_colored("Failed to install packages. Please install manually:", RED)
        print_colored("pip install --break-system-packages streamlit pandas numpy plotly requests yfinance matplotlib ta scikit-learn pytz ccxt psutil", YELLOW)
        sys.exit(1)

# Add the project root to Python path
sys.path.insert(0, PROJECT_ROOT)

# Start Streamlit
print_colored("Starting Streamlit dashboard...", GREEN)
print_colored("The dashboard will be available at: http://localhost:8501", YELLOW)
print_colored("Opening browser...", YELLOW)

# Open browser after a short delay
def open_browser():
    """Open the browser after a delay."""
    time.sleep(2)
    webbrowser.open("http://localhost:8501")

# Start browser in a separate thread
import threading
threading.Thread(target=open_browser).start()

def find_streamlit_process():
    """Find any running Streamlit processes."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('streamlit' in cmd for cmd in cmdline):
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def stop_dashboard():
    """Stop the running Streamlit dashboard."""
    proc = find_streamlit_process()
    if proc:
        print_colored(f"Stopping Streamlit dashboard (PID: {proc.pid})...", YELLOW)
        try:
            os.kill(proc.pid, signal.SIGTERM)
            print_colored("Dashboard stopped successfully.", GREEN)
            return True
        except Exception as e:
            print_colored(f"Error stopping dashboard: {str(e)}", RED)
            return False
    else:
        print_colored("No running dashboard found.", YELLOW)
        return False

def start_dashboard():
    """Start the Streamlit dashboard."""
    # Check if dashboard is already running
    proc = find_streamlit_process()
    if proc:
        print_colored(f"Dashboard is already running (PID: {proc.pid}).", YELLOW)
        print_colored("Visit http://localhost:8501 to view it.", GREEN)
        
        # Ask if user wants to restart
        response = input("Do you want to restart it? (y/n): ").strip().lower()
        if response == 'y':
            stop_dashboard()
        else:
            # Just open the browser
            webbrowser.open("http://localhost:8501")
            return

    # Start browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Run Streamlit
    os.chdir(DASHBOARD_DIR)
    os.system(f"{sys.executable} -m streamlit run app.py")

# Parse command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Control the Trading Dashboard')
    parser.add_argument('action', nargs='?', default='start', choices=['start', 'stop', 'restart'],
                      help='Action to perform: start, stop, or restart (default: start)')
    
    args = parser.parse_args()
    
    if args.action == 'stop':
        stop_dashboard()
    elif args.action == 'restart':
        stop_dashboard()
        time.sleep(1)
        start_dashboard()
    else:  # start
        start_dashboard()
