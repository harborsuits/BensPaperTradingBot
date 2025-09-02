#!/bin/bash
# Enhanced setup script for BensBot trading system

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         BensBot Trading System - Environment Setup         ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Create and activate virtual environment
echo -e "\n🔧 Setting up Python virtual environment..."

# Check for Python
PYTHON_CMD=$(which python3)
if [ -z "$PYTHON_CMD" ]; then
  echo "❌ Python 3 not found. Please install Python 3 and try again."
  exit 1
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
  echo "🧹 Removing existing virtual environment..."
  rm -rf .venv
fi

# Create fresh virtual environment
echo "🌱 Creating new virtual environment..."
$PYTHON_CMD -m venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Verify we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "❌ Failed to activate virtual environment. Please check your Python installation."
  exit 1
fi

# Upgrade pip
echo -e "\n🔄 Upgrading pip..."
python -m pip install --upgrade pip

# Install core dependencies from requirements.txt
echo -e "\n📦 Installing core requirements..."
python -m pip install -r requirements.txt

# Install additional required packages
echo -e "\n➕ Installing additional dependencies..."
python -m pip install yfinance flask flask-socketio flask-cors pytest black flake8

# Install TA-Lib if needed
echo -e "\n📊 Checking for TA-Lib..."
if ! python -m pip show ta-lib &>/dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🍎 macOS detected - you may need to install TA-Lib using Homebrew first:"
        echo "brew install ta-lib"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "🐧 Linux detected - you may need to install TA-Lib using your package manager:"
        echo "sudo apt-get install ta-lib"
    fi
    echo "Attempting to install TA-Lib Python wrapper..."
    python -m pip install ta-lib
fi

# Add the current directory to PYTHONPATH
echo -e "\n🐍 Setting PYTHONPATH environment variable..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "export PYTHONPATH=\$PYTHONPATH:$(pwd)" >> .venv/bin/activate

echo -e "\n✅ Environment setup complete!"
echo -e "\n╭────────────────────────────────────────╮"
echo "│  To activate the environment, run:       │"
echo "│  source .venv/bin/activate               │"
echo "│                                          │"
echo "│  The environment has been activated for  │"
echo "│  your current session.                   │"
echo "╰────────────────────────────────────────╯"
