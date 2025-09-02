#!/bin/bash
# Script to run a simple test Streamlit app with the virtual environment

# Current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found."
    echo "Please create it first with: python3 -m venv venv"
    exit 1
fi

# Kill any existing streamlit processes
echo "Stopping any existing Streamlit processes..."
pkill -f streamlit || true

# Clear Python cache files
echo "Clearing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Use the Python from the virtual environment
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"

# Print Python path to confirm we're using the virtual environment
echo "Using Python: $VENV_PYTHON"
echo "Python version: $($VENV_PYTHON --version)"

# Make sure packages are installed
echo "Installing required packages..."
$VENV_PIP install streamlit flask ta pandas numpy

# Set PYTHONPATH to include the current directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the test Streamlit app directly
echo "Starting test Streamlit app..."
$VENV_PYTHON -m streamlit run test_streamlit.py --server.port 8502 