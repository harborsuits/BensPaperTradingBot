#!/bin/bash
# run_with_chat.sh - Launch the autonomous trading dashboard with BenBot AI assistant
# This script focuses on fixing the AI assistant integration with the orchestrator

# Colors for nice output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}Autonomous Trading Dashboard with BenBot AI${NC}"
echo -e "${BLUE}=========================================${NC}"

# Make sure Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

# Check if we have a virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment.${NC}"
        exit 1
    fi
fi

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Verify dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
pip install -r requirements.txt > /dev/null

# Make sure the trading_bot module is installed in development mode
if ! python -c "import trading_bot" 2>/dev/null; then
    echo -e "${YELLOW}Installing trading_bot module in development mode...${NC}"
    pip install -e .
    if [ $? -ne 0 ]; then
        echo -e "${RED}Warning: Failed to install trading_bot module. AI assistant may not work.${NC}"
    fi
fi

# Check required directories
for dir in data results models; do
    if [ ! -d "$dir" ]; then
        echo -e "${YELLOW}Creating $dir directory...${NC}"
        mkdir -p "$dir"
    fi
done

# Debug the BenBot assistant with orchestrator integration
echo -e "${YELLOW}Testing BenBot assistant with orchestrator integration...${NC}"
python debug_benbot_orchestrator.py

# Prompt the user to continue
echo
echo -e "${BLUE}Would you like to start the dashboard? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    # Launch the application with streamlit
    echo -e "${GREEN}Launching dashboard with AI assistant...${NC}"
    # Enable verbose logging and set log level to debug
    export PYTHONUNBUFFERED=1
    export STREAMLIT_LOG_LEVEL=debug
    
    streamlit run app_new.py --server.enableCORS false
else
    echo -e "${YELLOW}Dashboard launch cancelled.${NC}"
fi

# Deactivate virtual environment when done
deactivate 