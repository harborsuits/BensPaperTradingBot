#!/usr/bin/env python3
"""
BenBot Trading System Launcher
Launches the modern React/Node.js trading dashboard
"""
import os
import sys
import subprocess
import webbrowser
import time
import signal
import argparse

# Colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

# Set the absolute paths
PROJECT_ROOT = "/Users/bendickinson/Desktop/benbot"
BACKEND_DIR = os.path.join(PROJECT_ROOT, "live-api")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "new-trading-dashboard")

def print_colored(message, color):
    """Print a colored message to the terminal."""
    print(f"{color}{message}{NC}")
    sys.stdout.flush()

def check_pm2():
    """Check if PM2 is installed."""
    try:
        subprocess.run(["pm2", "--version"], capture_output=True, check=True)
        return True
    except:
        print_colored("PM2 is not installed. Installing...", YELLOW)
        try:
            subprocess.run(["npm", "install", "-g", "pm2"], check=True)
            return True
        except:
            print_colored("Failed to install PM2. Please install manually: npm install -g pm2", RED)
            return False

def check_dependencies():
    """Check if npm dependencies are installed."""
    # Check backend dependencies
    backend_modules = os.path.join(BACKEND_DIR, "node_modules")
    if not os.path.exists(backend_modules):
        print_colored("Installing backend dependencies...", YELLOW)
        subprocess.run(["npm", "install"], cwd=BACKEND_DIR, check=True)
    
    # Check frontend dependencies
    frontend_modules = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(frontend_modules):
        print_colored("Installing frontend dependencies...", YELLOW)
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)
    
    print_colored("All dependencies installed.", GREEN)
    return True

def is_backend_running():
    """Check if backend is running."""
    try:
        result = subprocess.run(["pm2", "list"], capture_output=True, text=True)
        return "benbot-backend" in result.stdout and "online" in result.stdout
    except:
        return False

def is_frontend_running():
    """Check if frontend is running on port 3003."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', 3003)) == 0
    except:
        return False

def start_backend():
    """Start the backend API server."""
    if is_backend_running():
        print_colored("Backend is already running.", YELLOW)
        return True
    
    print_colored("Starting backend API server...", GREEN)
    os.chdir(PROJECT_ROOT)
    
    # Start backend with PM2
    subprocess.run(["pm2", "start", "ecosystem.config.js", "--only", "benbot-backend"], check=True)
    
    # Wait for backend to be ready
    time.sleep(3)
    
    if is_backend_running():
        print_colored("Backend started successfully on port 4000.", GREEN)
        return True
    else:
        print_colored("Failed to start backend.", RED)
        return False

def start_frontend():
    """Start the frontend development server."""
    if is_frontend_running():
        print_colored("Frontend is already running.", YELLOW)
        return True
    
    print_colored("Starting frontend development server...", GREEN)
    
    # Start frontend as a background process
    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for frontend to be ready
    for i in range(10):
        time.sleep(2)
        if is_frontend_running():
            print_colored("Frontend started successfully on port 3003.", GREEN)
            return True
    
    print_colored("Frontend is taking longer than expected to start...", YELLOW)
    return False

def stop_backend():
    """Stop the backend API server."""
    if not is_backend_running():
        print_colored("Backend is not running.", YELLOW)
        return True
    
    print_colored("Stopping backend...", YELLOW)
    subprocess.run(["pm2", "stop", "benbot-backend"], check=True)
    print_colored("Backend stopped.", GREEN)
    return True

def stop_frontend():
    """Stop the frontend development server."""
    if not is_frontend_running():
        print_colored("Frontend is not running.", YELLOW)
        return True
    
    print_colored("Stopping frontend...", YELLOW)
    
    # Find and kill the process using port 3003
    try:
        result = subprocess.run(["lsof", "-ti:3003"], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                os.kill(int(pid), signal.SIGTERM)
            time.sleep(2)
            print_colored("Frontend stopped.", GREEN)
            return True
    except Exception as e:
        print_colored(f"Error stopping frontend: {e}", RED)
        return False

def start_system():
    """Start the entire trading system."""
    print_colored("\n🚀 Starting BenBot Trading System...\n", GREEN)
    
    # Check PM2
    if not check_pm2():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Start backend
    if not start_backend():
        return False
    
    # Start frontend
    if not start_frontend():
        return False
    
    # Open browser
    print_colored("\n✅ BenBot Trading System is ready!", GREEN)
    print_colored("Opening dashboard in browser...", YELLOW)
    time.sleep(2)
    # Try macOS-specific open command first, then fallback to webbrowser
    try:
        subprocess.run(["open", "http://localhost:3003"], check=True)
    except:
        webbrowser.open("http://localhost:3003")
    
    print_colored("\n📊 Dashboard: http://localhost:3003", GREEN)
    print_colored("🔧 API: http://localhost:4000", GREEN)
    print_colored("\nPress Ctrl+C to stop the system.", YELLOW)
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print_colored("\n\nShutting down...", YELLOW)
        stop_system()

def stop_system():
    """Stop the entire trading system."""
    print_colored("\n🛑 Stopping BenBot Trading System...\n", YELLOW)
    
    stop_frontend()
    stop_backend()
    
    print_colored("\n✅ System stopped.", GREEN)

def restart_system():
    """Restart the entire trading system."""
    print_colored("\n🔄 Restarting BenBot Trading System...\n", YELLOW)
    
    stop_system()
    time.sleep(2)
    start_system()

def status():
    """Show system status."""
    print_colored("\n📊 BenBot Trading System Status\n", GREEN)
    
    # Backend status
    if is_backend_running():
        print_colored("✅ Backend API: Running on port 4000", GREEN)
        
        # Show PM2 status
        subprocess.run(["pm2", "status", "benbot-backend"])
    else:
        print_colored("❌ Backend API: Not running", RED)
    
    print()
    
    # Frontend status
    if is_frontend_running():
        print_colored("✅ Frontend: Running on port 3003", GREEN)
        print_colored("   Dashboard URL: http://localhost:3003", YELLOW)
    else:
        print_colored("❌ Frontend: Not running", RED)
    
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BenBot Trading System Launcher')
    parser.add_argument('action', nargs='?', default='start',
                      choices=['start', 'stop', 'restart', 'status'],
                      help='Action to perform (default: start)')
    
    args = parser.parse_args()
    
    if args.action == 'stop':
        stop_system()
    elif args.action == 'restart':
        restart_system()
    elif args.action == 'status':
        status()
    else:  # start
        start_system()
