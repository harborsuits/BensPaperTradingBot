#!/bin/bash

# Start Trading Bot System - Launch Script
# This script manages the startup of the entire trading system

echo "====================================="
echo "BensBot Trading System Startup Script"
echo "====================================="

# Set up environment
export PYTHONPATH=/Users/bendickinson/Desktop/Trading:BenBot
export TRADING_ENV="development"  # Use "production" for live trading

# Activate virtual environment if it exists
if [ -d "/Users/bendickinson/Desktop/trading_venv" ]; then
    echo "Activating Python virtual environment..."
    source /Users/bendickinson/Desktop/trading_venv/bin/activate
else
    echo "Warning: Virtual environment not found at /Users/bendickinson/Desktop/trading_venv"
    echo "Make sure you've set up the Python environment correctly."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if dashboard is already running (on port 3003)
if lsof -i :3003 > /dev/null; then
    echo "Dashboard already running on port 3003"
else
    # Start dashboard in the background
    echo "Starting React Dashboard..."
    cd /Users/bendickinson/Desktop/Trading:BenBot/new-trading-dashboard
    npm start &
    DASHBOARD_PID=$!
    echo "Dashboard started (PID: $DASHBOARD_PID)"
fi

# Check if backend API is already running (on port 8000)
if lsof -i :8000 > /dev/null; then
    echo "Backend API already running on port 8000"
else
    # Start the main orchestrator with news sentiment integration
    echo "Starting Trading Bot Orchestrator..."
    cd /Users/bendickinson/Desktop/Trading:BenBot
    python trading_bot/orchestrator.py &
    ORCHESTRATOR_PID=$!
    echo "Orchestrator started (PID: $ORCHESTRATOR_PID)"
    
    # Start backtester API (for news sentiment integration)
    echo "Starting Backtester API..."
    cd /Users/bendickinson/Desktop
    python backtester_api.py &
    BACKTESTER_PID=$!
    echo "Backtester API started (PID: $BACKTESTER_PID)"
    
    # Wait for API to come online
    echo "Waiting for API to start..."
    while ! curl -s http://localhost:8000/health > /dev/null; do
        sleep 1
    done
    echo "API is now available!"
fi

echo
echo "====================================="
echo "Trading System Started Successfully!"
echo "====================================="
echo
echo "Dashboard: http://localhost:3003"
echo "API:       http://localhost:8000"
echo
echo "Press Ctrl+C to shutdown all components"

# Wait for user to stop the script
trap "echo 'Shutting down...'; kill $ORCHESTRATOR_PID $DASHBOARD_PID $BACKTESTER_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
