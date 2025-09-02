#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading Engine...${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:/Users/bendickinson/Desktop/Trading:BenBot

# Navigate to project root
cd /Users/bendickinson/Desktop/Trading:BenBot

# Check if config path is specified, otherwise use default
CONFIG_PATH=${1:-"./config/trading_config.json"}

echo -e "${YELLOW}Using configuration: ${CONFIG_PATH}${NC}"

# Start the trading engine
python -m trading_bot.main --config "${CONFIG_PATH}"

echo -e "${GREEN}Trading engine has completed its run${NC}"
