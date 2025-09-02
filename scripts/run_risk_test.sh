#!/bin/bash
# Script to run the integrated risk management test

# Activate the virtual environment
echo "Activating virtual environment..."
source ~/bensbot_env/bin/activate

# Navigate to project directory
cd /Users/bendickinson/Desktop/Trading:BenBot

# Run the integrated risk management test
echo "Running integrated risk management test..."
python integrated_risk_management_test.py

# Keep terminal open
echo ""
echo "Test complete. Press any key to close."
read -n 1
