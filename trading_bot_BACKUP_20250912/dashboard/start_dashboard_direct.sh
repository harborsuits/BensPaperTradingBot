#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading Dashboard...${NC}"

# Set the dashboard directory
DASHBOARD_DIR="/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"
cd "$DASHBOARD_DIR"

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo -e "${YELLOW}Streamlit not found. Installing required packages...${NC}"
    pip install -r requirements.txt
fi

# Get the parent directory (for imports)
PARENT_DIR="/Users/bendickinson/Desktop/Trading:BenBot"

# Start the dashboard directly with Streamlit (no Docker)
echo -e "${YELLOW}Starting Streamlit dashboard...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard when done.${NC}"

# Add the parent directory to Python path to enable imports
PYTHONPATH="$PARENT_DIR" streamlit run app.py
