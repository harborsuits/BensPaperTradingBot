#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading API Server...${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:/Users/bendickinson/Desktop/Trading:BenBot

# Start the FastAPI server with uvicorn
cd /Users/bendickinson/Desktop/Trading:BenBot
echo -e "${YELLOW}Starting FastAPI server on port 8000...${NC}"
uvicorn trading_bot.api.main:app --reload --host 0.0.0.0 --port 8000

echo -e "${GREEN}API server is running at http://localhost:8000${NC}"
