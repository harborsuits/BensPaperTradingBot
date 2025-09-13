#!/bin/bash

# Ensure the script runs from the correct directory
cd "$(dirname "$0")"

# Set the Python path to include the project root
export PYTHONPATH=/Users/bendickinson/Desktop/Trading:BenBot

# Start the FastAPI server with Uvicorn
echo "Starting Trading API server..."
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
