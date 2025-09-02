#!/bin/bash

# Navigate to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install required packages if not already installed
pip install streamlit pandas numpy ta flask

# Set Python path to include current directory
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR"

# Run the test Streamlit app
streamlit run test_streamlit.py

# Deactivate virtual environment when done
deactivate 