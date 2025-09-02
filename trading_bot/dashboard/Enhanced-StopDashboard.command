#!/bin/bash

# Show a nice header
echo "╔═══════════════════════════════════════════════╗"
echo "║   Stopping BensBot Trading Dashboard...       ║"
echo "╚═══════════════════════════════════════════════╝"

# Navigate to the dashboard directory (using quoted path to handle the colon)
cd "/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"

# Run the enhanced launcher with stop command
python3 dashboard_launcher.py stop
