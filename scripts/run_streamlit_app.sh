#!/bin/bash
# Script to run the Streamlit app with the correct PYTHONPATH

# Activate the virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Get the current directory as the absolute path
CURRENT_DIR=$(pwd)
echo "Setting PYTHONPATH to: $CURRENT_DIR"

# Run the Streamlit app with the correct PYTHONPATH
PYTHONPATH=$CURRENT_DIR streamlit run app.py "$@"

# Print a message when the app exits
echo ""
echo "Streamlit app has exited. Press Enter to exit the script."
read 