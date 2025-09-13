#!/bin/bash

# Show a nice header
echo "╔═══════════════════════════════════════════════╗"
echo "║         BensBot Trading Dashboard             ║"
echo "╚═══════════════════════════════════════════════╝"

# Navigate to the dashboard directory (using quoted path to handle the colon)
cd "/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"

# Install any missing dependencies directly
echo "Installing critical dependencies if missing..."
pip3 install --break-system-packages websocket-client streamlit yfinance backtrader

# Run the enhanced launcher with start command
python3 dashboard_launcher.py start
