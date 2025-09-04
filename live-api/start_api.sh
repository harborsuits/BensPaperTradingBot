#!/bin/bash

# Start the FastAPI server
echo "Starting FastAPI server on port 8000..."
cd "$(dirname "$0")"
python -m uvicorn api.main:app --reload --port 8000


