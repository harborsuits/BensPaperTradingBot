#!/bin/bash
# Setup and run the trading bot API with AI integration

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up AI trading bot environment...${NC}"

# Create virtual environment if it doesn't exist in home directory (avoiding path issues)
if [ ! -d "$HOME/trading_env" ]; then
    echo -e "${BLUE}Creating virtual environment in $HOME/trading_env${NC}"
    python3 -m venv $HOME/trading_env
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Continuing without it.${NC}"
    fi
fi

# Try to activate virtual environment
if [ -d "$HOME/trading_env" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source $HOME/trading_env/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to activate virtual environment. Continuing without it.${NC}"
    else
        echo -e "${GREEN}Virtual environment activated successfully.${NC}"
    fi
fi

# Install required packages
echo -e "${BLUE}Installing required packages...${NC}"
pip3 install --user pyyaml openai anthropic fastapi uvicorn

# Set the correct Python path 
export PYTHONPATH="$PYTHONPATH:/Users/bendickinson/Desktop/Trading:BenBot"

# Start the trading bot API server
echo -e "${GREEN}Starting trading bot API with AI integration...${NC}"
echo -e "${BLUE}The dashboard will automatically connect to this API when available.${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server.${NC}"
echo -e "${GREEN}===============================================${NC}"

# Run the API server
cd /Users/bendickinson/Desktop/Trading:BenBot
python3 -m trading_bot.api.app
