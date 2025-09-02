#!/bin/bash

# Kill any existing Streamlit processes
echo "Stopping any existing Streamlit processes..."
pkill -f streamlit || true

# Find and use the virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    VENV_PYTHON="$VENV_DIR/bin/python"
    echo "Using Python from virtual environment: $VENV_PYTHON"
else
    VENV_PYTHON="python"
    echo "No virtual environment found, using system Python"
fi

# Print Python version
echo "Using Python version:"
$VENV_PYTHON --version

# Set PYTHONPATH and run the minimal app
export PYTHONPATH="$(pwd)"
echo "Starting minimal Streamlit app..."
$VENV_PYTHON -m streamlit run minimal.py --server.port=8504 