#!/bin/bash

# Set colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Display startup message
echo -e "${YELLOW}Starting BensBot Professional Trading Dashboard...${NC}"

# Kill any existing Streamlit processes
echo -e "Stopping any existing dashboard instances..."
pkill -f streamlit 2>/dev/null

# Navigate to the project directory
cd "/Users/bendickinson/Desktop/Trading:BenBot"

# Activate the virtual environment
source ~/bensbot_env/bin/activate

# Make sure all dependencies are installed
echo -e "Installing dependencies..."
pip install -q streamlit pymongo pandas yfinance plotly

# Set PYTHONPATH to include project root
export PYTHONPATH="${PYTHONPATH}:/Users/bendickinson/Desktop/Trading:BenBot"

# Open browser first
echo -e "${YELLOW}Opening web browser...${NC}"
sleep 1
open "http://localhost:8501"

# Run the dashboard
echo -e "${GREEN}Launching professional dashboard...${NC}"
streamlit run bensbot_dashboard.py

# Exit message
echo -e "${YELLOW}Dashboard stopped.${NC}"
