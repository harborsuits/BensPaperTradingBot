#!/bin/bash
# Script to run the trading bot app using the virtual environment

# Print current directory for debugging
echo "Current directory: $(pwd)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found."
    echo "Please create it first with: python3 -m venv venv"
    echo "Then install dependencies with: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Create or update requirements.txt
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

# Activate the virtual environment
source venv/bin/activate

# Print Python path to confirm we're using the virtual environment
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# Verify required packages are installed
echo "Checking for required packages..."
pip list | grep ta
pip list | grep flask
pip list | grep streamlit

# Check if any required packages are missing
if ! pip list | grep -q ta || ! pip list | grep -q flask || ! pip list | grep -q streamlit; then
    echo "Installing missing packages..."
    pip install -r requirements.txt
fi

# Run the app using the virtual environment's Python
echo "Starting application with virtual environment Python..."
python -m streamlit run app.py

# Deactivate the virtual environment when done
deactivate

# Activate the virtual environment if it exists
if [ -d "trading_env" ]; then
    source trading_env/bin/activate
fi

# Install required packages if needed
pip install -q streamlit pandas numpy matplotlib yfinance

# Run the streamlit app
streamlit run app_simple.py 