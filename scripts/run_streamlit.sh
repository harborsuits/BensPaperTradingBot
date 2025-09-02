#!/bin/bash
# Script to run the Streamlit app with the virtual environment

# Current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found."
    echo "Please create it first with: python3 -m venv venv"
    echo "Then install dependencies with: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Clear Python cache files to ensure clean imports
echo "Clearing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Create requirements.txt
echo "# Trading Bot Requirements" > requirements.txt
echo "ta==0.11.0" >> requirements.txt
echo "flask==3.1.0" >> requirements.txt
echo "streamlit>=1.44.0" >> requirements.txt
echo "pandas>=2.0.0" >> requirements.txt
echo "numpy>=2.0.0" >> requirements.txt
echo "requests>=2.25.0" >> requirements.txt
echo "matplotlib>=3.5.0" >> requirements.txt
echo "seaborn>=0.11.0" >> requirements.txt
echo "scikit-learn>=1.0.0" >> requirements.txt

# Explicitly use the Python from the virtual environment
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
VENV_PIP="$SCRIPT_DIR/venv/bin/pip"

# Print Python path to confirm we're using the virtual environment
echo "Using Python: $VENV_PYTHON"
echo "Python version: $($VENV_PYTHON --version)"

# Install required packages directly with the venv pip
$VENV_PIP install -r requirements.txt

# Set PYTHONPATH to include the current directory
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run Streamlit directly with the venv Python
echo "Starting Streamlit with virtual environment Python..."
$VENV_PYTHON -m streamlit run app.py

# Get the python executable and streamlit path
PYTHON_PATH=$(which python)
STREAMLIT_PATH=$(python -c "import streamlit as st; import os; print(os.path.join(os.path.dirname(st.__file__), 'cli.py'))")

echo "Using Python: $PYTHON_PATH"
echo "Streamlit path: $STREAMLIT_PATH"

# Run the streamlit app using the full path
$PYTHON_PATH $STREAMLIT_PATH run simple_app.py --server.port 8501 