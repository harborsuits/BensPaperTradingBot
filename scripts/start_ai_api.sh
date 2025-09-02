#!/bin/bash
# Start the BenBot API with AI integration
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Starting BenBot API with AI integration...{NC}"

# Paths
VENV_PATH="/Users/bendickinson/benbot_env"
PROJECT_PATH="/Users/bendickinson/Desktop/Trading:BenBot"

# Activate virtual environment
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
    echo -e "${GREEN}Virtual environment activated{NC}"
else
    echo -e "${RED}Virtual environment not found at ${VENV_PATH}{NC}"
    echo -e "${BLUE}Continuing without virtual environment{NC}"
fi

# Set Python path
export PYTHONPATH="${PROJECT_PATH}"

echo -e "${GREEN}Starting API server...{NC}"
echo -e "${BLUE}API will be available at: http://localhost:5000{NC}"
echo -e "${BLUE}Press Ctrl+C to stop{NC}"

cd "${PROJECT_PATH}"
python -m trading_bot.api.app
