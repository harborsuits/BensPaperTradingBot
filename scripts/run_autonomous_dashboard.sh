#!/bin/bash
# run_autonomous_dashboard.sh
# Launches the autonomous trading dashboard with BenBot AI Assistant

# Set terminal colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}Autonomous Trading Dashboard with BenBot AI${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating new virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please check your Python installation.${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/.requirements_installed" ]; then
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        touch venv/.requirements_installed
    else
        echo -e "${RED}Failed to install requirements.${NC}"
        echo -e "${YELLOW}Continuing anyway, but the app may not work properly.${NC}"
    fi
fi

# Make sure the trading_bot module is installed in development mode
if [ ! -f "venv/.trading_bot_installed" ]; then
    echo -e "${YELLOW}Installing trading_bot in development mode...${NC}"
    pip install -e .
    if [ $? -eq 0 ]; then
        touch venv/.trading_bot_installed
    else
        echo -e "${RED}Failed to install trading_bot module.${NC}"
        echo -e "${YELLOW}Continuing anyway, but the autonomous features may not work properly.${NC}"
    fi
fi

# Run the application
echo -e "${GREEN}Launching Autonomous Trading Dashboard...${NC}"
streamlit run app_new.py "$@"

# Deactivate virtual environment when done
deactivate 