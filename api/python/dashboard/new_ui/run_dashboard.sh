#!/bin/bash
# Run the BensBot Trading Dashboard

# Set the Python path to include the project root
export PYTHONPATH="/Users/bendickinson/Desktop/Trading:BenBot:$PYTHONPATH"

# Run the dashboard
echo "Starting BensBot Trading Dashboard..."
cd /Users/bendickinson/Desktop/Trading:BenBot
streamlit run trading_bot/dashboard/new_ui/dashboard.py
