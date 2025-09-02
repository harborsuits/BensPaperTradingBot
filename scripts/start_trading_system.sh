#!/bin/bash

# ====================================
# BenBot Trading System Startup Script
# ====================================

# Color definitions
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}======================================${NC}"
echo -e "${YELLOW}      BenBot Trading System          ${NC}"
echo -e "${YELLOW}======================================${NC}"

# Ensure the script runs from the project root
cd "$(dirname "$0")"
PROJECT_ROOT=$(pwd)

# Configure Python environment
echo -e "\n${BLUE}Setting up Python environment...${NC}"

# Check if trading_venv exists on the desktop (due to colon in path issue)
VENV_PATH="/Users/bendickinson/Desktop/trading_venv"
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}Virtual environment not found at $VENV_PATH${NC}"
    echo -e "${BLUE}Creating new virtual environment...${NC}"
    python3 -m venv $VENV_PATH
    
    echo -e "${BLUE}Installing dependencies...${NC}"
    source $VENV_PATH/bin/activate
    pip install fastapi uvicorn pandas pydantic websocket-client rich
else
    echo -e "${GREEN}Using existing virtual environment at $VENV_PATH${NC}"
    source $VENV_PATH/bin/activate
fi

# Set environment variables
export PYTHONPATH=$PROJECT_ROOT
export TRADING_ENV="development"
export TRADING_API_PORT=8000
export TRADING_API_HOST="0.0.0.0"

# Start the FastAPI server in the background
echo -e "\n${BLUE}Starting FastAPI server...${NC}"
echo -e "${YELLOW}API will be available at http://localhost:$TRADING_API_PORT${NC}"

# Navigate to the API directory
cd $PROJECT_ROOT/trading_bot/api

# Use nohup to keep the server running after the script ends
nohup python -m uvicorn app:app --reload --host $TRADING_API_HOST --port $TRADING_API_PORT > $PROJECT_ROOT/api_server.log 2>&1 &
API_PID=$!

# Check if the server started successfully
sleep 3
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}API server started successfully! (PID: $API_PID)${NC}"
else
    echo -e "${RED}Failed to start API server. Check logs at $PROJECT_ROOT/api_server.log${NC}"
    exit 1
fi

# Display instructions for React frontend
echo -e "\n${BLUE}============================================${NC}"
echo -e "${BLUE}    React Frontend Connection Information    ${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}To connect your React frontend to this backend:${NC}"
echo -e "1. Set your API base URL to: ${YELLOW}http://localhost:$TRADING_API_PORT/api${NC}"
echo -e "2. Set your WebSocket URL to: ${YELLOW}ws://localhost:$TRADING_API_PORT/ws${NC}"
echo -e "3. Available WebSocket channels: ${YELLOW}context, strategy, trading, portfolio, logging, evotester${NC}"

# Instructions for stopping the server
echo -e "\n${BLUE}To stop the API server:${NC}"
echo -e "Run: ${YELLOW}kill $API_PID${NC} or use the stop_trading_system.sh script"

# Create a stop script
cat > $PROJECT_ROOT/stop_trading_system.sh << EOL
#!/bin/bash
# Stop script for BenBot Trading System

# Find the API server process
API_PID=\$(ps aux | grep "uvicorn app:app" | grep -v grep | awk '{print \$2}')

if [ -n "\$API_PID" ]; then
    echo "Stopping API server (PID: \$API_PID)..."
    kill \$API_PID
    echo "API server stopped."
else
    echo "API server is not running."
fi
EOL

chmod +x $PROJECT_ROOT/stop_trading_system.sh

echo -e "\n${GREEN}Setup complete! The API server is now running.${NC}"
echo -e "${YELLOW}API documentation: ${NC}http://localhost:$TRADING_API_PORT/docs"
echo -e "${YELLOW}Server logs: ${NC}$PROJECT_ROOT/api_server.log"

# Keep the script running if desired (uncomment to enable)
# echo -e "\nPress Ctrl+C to exit this script (API server will continue running in background)"
# tail -f $PROJECT_ROOT/api_server.log
