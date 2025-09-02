#!/bin/bash
# ======================================================
# BensBot Trading System Launcher
# ======================================================
# This script launches both the API server and Streamlit dashboard
# with all necessary configuration in a single command.

# Set fixed port for consistency
API_PORT=9000
API_URL="http://localhost:$API_PORT"

# Colors for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}🚀 BensBot Trading System Launcher${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Get script directory for relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate the virtual environment
if [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${RED}Error: Virtual environment not found. Please run setup script first.${NC}"
    exit 1
fi

# Make sure dependencies are installed
echo -e "${GREEN}Checking dependencies...${NC}"
pip install -q uvicorn fastapi python-multipart email-validator pyjwt streamlit

# Create a function to clean up processes on exit
function cleanup() {
    echo -e "${YELLOW}Shutting down API server...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null
    fi
    # Kill any remaining uvicorn processes
    pkill -f "uvicorn trading_bot.api.app:app" 2>/dev/null
    echo -e "${GREEN}Cleanup complete. Goodbye!${NC}"
    exit 0
}

# Set up trap to catch Ctrl+C and exit signals
trap cleanup SIGINT SIGTERM EXIT

# Check if port is already in use and kill the process if needed
PORT_PID=$(lsof -ti:$API_PORT)
if [ ! -z "$PORT_PID" ]; then
    echo -e "${YELLOW}Port $API_PORT is already in use. Freeing up the port...${NC}"
    kill -9 $PORT_PID 2>/dev/null
fi

# Start the API server in the background
echo -e "${GREEN}Starting API server on port $API_PORT...${NC}"
PYTHONPATH="$SCRIPT_DIR" uvicorn trading_bot.api.app:app --host 0.0.0.0 --port $API_PORT --reload > api_server.log 2>&1 &
API_PID=$!

# Wait a moment for the API server to start
echo -e "${YELLOW}Waiting for API server to start...${NC}"
sleep 3

# Check if API server started successfully
if ! ps -p $API_PID > /dev/null; then
    echo -e "${RED}API server failed to start. See api_server.log for details.${NC}"
    cat api_server.log
    exit 1
fi

# Set environment variable for the dashboard
export BENSBOT_API_URL="$API_URL"
echo -e "${GREEN}API server running at $API_URL${NC}"

# Start the Streamlit dashboard
echo -e "${GREEN}Starting Streamlit dashboard...${NC}"
echo -e "${YELLOW}Dashboard will open in your browser automatically.${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop both the API server and dashboard.${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Run streamlit
PYTHONPATH="$SCRIPT_DIR" streamlit run streamlit_dashboard.py

# Note: The cleanup function will handle shutdown when script terminates
