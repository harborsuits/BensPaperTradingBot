#!/bin/bash

# Set colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Display startup message
echo -e "${YELLOW}Starting BensBot Trading Dashboard...${NC}"

# Navigate to the project directory
cd "/Users/bendickinson/Desktop/Trading:BenBot"

# Activate the virtual environment
source ~/bensbot_env/bin/activate

# Make sure all dependencies are installed
pip install -q streamlit pymongo pandas yfinance plotly

# Set PYTHONPATH to include project root
export PYTHONPATH="${PYTHONPATH}:/Users/bendickinson/Desktop/Trading:BenBot"

# Go to dashboard directory
cd "/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"

# Launch the dashboard
echo -e "${GREEN}Launching dashboard...${NC}"
echo -e "${YELLOW}Opening web browser...${NC}"

# Open browser 
sleep 1
open "http://localhost:8501"

# Start Streamlit
streamlit run app.py
