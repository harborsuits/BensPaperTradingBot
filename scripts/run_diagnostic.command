#!/bin/bash

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Clear terminal
clear

# Print header
echo "============================================================"
echo "           TRADING DASHBOARD DIAGNOSTIC LAUNCHER            "
echo "============================================================"
echo "Running diagnostic tests to help troubleshoot the dashboard."
echo "Please wait..."
echo "============================================================"

# Run the diagnostic script
python dashboard_test.py 