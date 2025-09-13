#!/bin/bash
# Start script for the Trading Bot API with proper PYTHONPATH setup

# Get the project root directory (parent of parent directory of this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PROJECT_ROOT"

# Activate the virtual environment
source /Users/bendickinson/Desktop/trading_venv/bin/activate

# Print info
echo "Starting Trading Bot API..."
echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "API endpoint will be available at: http://127.0.0.1:8000/api"
echo "Using virtual environment: $(which python)"

# Start the FastAPI server
cd "$SCRIPT_DIR"
python -m uvicorn app_new:app --reload --host 0.0.0.0 --port 8000
