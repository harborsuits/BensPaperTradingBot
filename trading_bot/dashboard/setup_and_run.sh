#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Dashboard directory
DASHBOARD_DIR="/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"
cd "$DASHBOARD_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Setting up virtual environment for the dashboard...${NC}"
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install Python 3 if not already installed.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Virtual environment created successfully.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements if needed
echo -e "${YELLOW}Installing required packages...${NC}"
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install required packages.${NC}"
    exit 1
fi

# Add parent directory to PYTHONPATH to enable imports
export PYTHONPATH="/Users/bendickinson/Desktop/Trading:BenBot:$PYTHONPATH"

# Run the dashboard
echo -e "${GREEN}Starting Streamlit dashboard...${NC}"
echo -e "${YELLOW}The dashboard will be available at: ${GREEN}http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the dashboard when done.${NC}"

streamlit run app.py
