#!/bin/bash
# Simple script to start the BenBot API with real AI integration
# Created by Cascade AI

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting BenBot API with OpenAI integration...${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ~/trading_env ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv ~/trading_env
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Continuing without it.${NC}"
    fi
fi

# Activate virtual environment
if [ -d ~/trading_env ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source ~/trading_env/bin/activate
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to activate virtual environment. Continuing without it.${NC}"
    else
        echo -e "${GREEN}Virtual environment activated.${NC}"
        
        # Install required packages
        echo -e "${BLUE}Installing required packages...${NC}"
        pip install openai==1.0.0 anthropic pyyaml fastapi uvicorn
    fi
fi

# Set Python path and start the API
echo -e "${GREEN}Starting BenBot API server...${NC}"
cd /Users/bendickinson/Desktop/Trading:BenBot

# Export Python path
export PYTHONPATH=/Users/bendickinson/Desktop/Trading:BenBot

# Start the API server
echo -e "${BLUE}API server starting at http://localhost:5000${NC}"
echo -e "${BLUE}Your dashboard will automatically connect to this backend${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server${NC}"
echo -e "${GREEN}============================================${NC}"
python3 -m trading_bot.api.app
