# Trading Dashboard Setup Guide

This guide will help you set up and run the Trading Dashboard correctly.

## Installation

1. **Create and activate a virtual environment:**

```bash
# Create a virtual environment (if you haven't already)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

2. **Install dependencies:**

```bash
# Install required packages 
pip install -r requirements.txt
```

3. **Install additional packages if needed:**

```bash
# If you see specific import errors, try installing those packages directly
pip install plotly streamlit flask pandas numpy requests
```

## Running the Dashboard

### Option 1: Run Streamlit App (Recommended)

```bash
# Make sure you're in the project root directory
cd /path/to/Trading

# Set the Python path (important for importing modules)
export PYTHONPATH=/path/to/Trading

# Run the app directly
streamlit run app.py
```

### Option 2: Run Flask Dashboard Server

```bash
# Make sure you're in the project root directory
cd /path/to/Trading

# Set the Python path
export PYTHONPATH=/path/to/Trading

# Run the dashboard server
python trading_bot/dashboard/dashboard_server.py
```

## Troubleshooting Common Errors

### Error: No module named 'trading_bot'

This happens when Python can't find the trading_bot package. To fix:

```bash
# Set the Python path to include your project root
export PYTHONPATH=/path/to/Trading

# Then run your command
```

### Error: No module named 'plotly' (or other package)

```bash
# Install the missing package
pip install plotly

# Or install all required packages
pip install -r requirements.txt
```

### Error: Address already in use (Port 8080)

The port is being used by another application. To fix:

```bash
# Find the process using port 8080
lsof -i :8080
# or
netstat -anp | grep 8080

# Kill the process
kill -9 <PID>

# Or use a different port:
# Edit the dashboard_server.py or app.py file to use a different port (e.g., 8081)
```

### Error: command not found: python/pip

Your system might be using python3/pip3 instead:

```bash
# Try using python3 instead of python
python3 -m pip install -r requirements.txt

# Or make an alias
alias python=python3
alias pip=pip3
```

## Using the API Keys

The dashboard is configured with the following API keys:

- Alpaca API: For market data and portfolio tracking
- Finnhub API: For financial news and stock data
- MarketAux API: For alternative financial news
- NewsData.io API: For comprehensive news coverage
- GNews API: For general news articles

These APIs are used in a fallback sequence - if one API fails, the system tries the next one.

## Support

If you continue to have issues, please:

1. Check the logs for detailed error messages
2. Make sure all required packages are installed
3. Verify your Python path is set correctly
4. Ensure no other applications are using the required ports 