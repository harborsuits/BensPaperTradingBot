#!/bin/bash
# Installation script for Trading Bot dependencies

echo "╔════════════════════════════════════════════════════════════╗"
echo "║             Trading Bot Dependency Installer                ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Detect Python version
PYTHON_CMD=$(which python3 || which python)
echo "Using Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Create virtual environment (optional)
if [ ! -d "venv" ]; then
    echo -e "\n📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "\n🔌 Activating virtual environment..."
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install core requirements
echo -e "\n📚 Installing core dependencies..."
pip install -r requirements.txt

# Install additional development packages
echo -e "\n🛠️ Installing development tools..."
pip install pytest black flake8

# Install Flask specifically (with extra emphasis)
echo -e "\n🌐 Making sure Flask is properly installed..."
pip install flask flask-socketio flask-cors

# Install TA-Lib if needed
echo -e "\n📊 Checking if TA-Lib is needed..."
if ! pip show ta-lib &>/dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS detected - you may need to install TA-Lib using Homebrew first:"
        echo "brew install ta-lib"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Linux detected - you may need to install TA-Lib using your package manager:"
        echo "sudo apt-get install ta-lib"
    fi
    echo "Attempting to install TA-Lib Python wrapper..."
    pip install ta-lib
fi

echo -e "\n✨ Installation complete! ✨"
echo "You can run your trading bot application now." 