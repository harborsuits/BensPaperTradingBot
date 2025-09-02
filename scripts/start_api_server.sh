#!/bin/bash

# Set project root directory for imports
export PYTHONPATH="$PYTHONPATH:/Users/bendickinson/Desktop/Trading:BenBot"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required dependencies
echo "Installing required dependencies..."
pip install fastapi uvicorn

# Start the API server 
echo "Starting API server..."
cd trading_bot
uvicorn api.app:app --host 0.0.0.0 --port 5000 --reload
