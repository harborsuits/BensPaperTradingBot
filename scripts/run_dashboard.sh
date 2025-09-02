#!/bin/bash
# Run the BensBot Enhanced Trading Dashboard with Risk Management

# Activate the virtual environment
echo "Activating virtual environment..."
source ~/bensbot_env/bin/activate

# Navigate to project directory
cd /Users/bendickinson/Desktop/Trading:BenBot

# Run the enhanced dashboard with risk management features
echo "Launching enhanced dashboard with risk management..."
streamlit run streamlit_dashboard_enhanced.py
