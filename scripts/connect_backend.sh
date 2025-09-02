#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  BensBot Backend Connection Setup    ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1. Start the API server
echo -e "${YELLOW}[1/3] Starting FastAPI Backend Server...${NC}"
echo -e "${YELLOW}This will enable real data connections for the dashboard${NC}"

# Create a new terminal window for the API server
if command -v osascript &> /dev/null; then
    # macOS
    osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && export PYTHONPATH=\$PYTHONPATH:$(pwd) && uvicorn trading_bot.api.main:app --reload --host 0.0.0.0 --port 8000\""
else
    # Fallback for other platforms
    echo -e "${RED}Could not open a new terminal window.${NC}"
    echo -e "${YELLOW}Please run this command in a new terminal:${NC}"
    echo "cd $(pwd) && export PYTHONPATH=\$PYTHONPATH:$(pwd) && uvicorn trading_bot.api.main:app --reload --host 0.0.0.0 --port 8000"
fi

# 2. Wait for API to start
echo -e "${YELLOW}[2/3] Waiting for API server to initialize (5 seconds)...${NC}"
sleep 5

# 3. Restart the dashboard to connect to real data
echo -e "${YELLOW}[3/3] Restarting dashboard to connect to real data...${NC}"

# Stop any existing dashboard
pkill -f "streamlit run trading_bot/dashboard/app.py" || true

# Start the dashboard with the API connection
if command -v osascript &> /dev/null; then
    # macOS
    osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && export PYTHONPATH=\$PYTHONPATH:$(pwd) && streamlit run trading_bot/dashboard/app.py\""
else
    # Fallback for other platforms
    echo -e "${RED}Could not open a new terminal window.${NC}"
    echo -e "${YELLOW}Please run this command in a new terminal:${NC}"
    echo "cd $(pwd) && export PYTHONPATH=\$PYTHONPATH:$(pwd) && streamlit run trading_bot/dashboard/app.py"
fi

echo -e "${GREEN}All components have been started!${NC}"
echo -e "${GREEN}Dashboard: http://localhost:8501${NC}"
echo -e "${GREEN}API Server: http://localhost:8000${NC}"
echo -e "${YELLOW}Note: The dashboard will automatically connect to real data once the API is available.${NC}"
echo -e "${BLUE}=======================================${NC}"
