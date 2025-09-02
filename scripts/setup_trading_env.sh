#!/bin/bash
# Fixed environment setup script that works with special directory names

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         BensBot Trading System - Environment Setup         ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Create a trading_env directory in the home directory (avoiding path issues)
ENV_DIR="$HOME/benbot_env"

echo -e "\n🔧 Setting up Python virtual environment at $ENV_DIR..."

# Check for Python
PYTHON_CMD=$(which python3)
if [ -z "$PYTHON_CMD" ]; then
  echo "❌ Python 3 not found. Please install Python 3 and try again."
  exit 1
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"

# Remove existing virtual environment if it exists
if [ -d "$ENV_DIR" ]; then
  echo "🧹 Removing existing virtual environment..."
  rm -rf "$ENV_DIR"
fi

# Create fresh virtual environment
echo "🌱 Creating new virtual environment..."
$PYTHON_CMD -m venv "$ENV_DIR"

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$ENV_DIR/bin/activate"

# Verify we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "❌ Failed to activate virtual environment. Please check your Python installation."
  exit 1
fi

# Upgrade pip
echo -e "\n🔄 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies from requirements.txt
echo -e "\n📦 Installing core requirements..."
pip install -r requirements.txt

# Install additional required packages
echo -e "\n➕ Installing additional dependencies..."
pip install yfinance flask flask-socketio flask-cors pytest black flake8

# Install TA-Lib if needed
echo -e "\n📊 Checking for TA-Lib..."
if ! pip show ta-lib &>/dev/null; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "🍎 macOS detected - you may need to install TA-Lib using Homebrew first:"
        echo "brew install ta-lib"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "🐧 Linux detected - you may need to install TA-Lib using your package manager:"
        echo "sudo apt-get install ta-lib"
    fi
    echo "Attempting to install TA-Lib Python wrapper..."
    pip install ta-lib
fi

# Create a run helper script that activates the environment
echo -e "\n🐍 Creating run helper script..."

cat > run_benbot.sh << EOL
#!/bin/bash
# Run script for BensBot with proper environment

# Activate the virtual environment
source "$ENV_DIR/bin/activate"

# Set Python path to include current directory
export PYTHONPATH=\$PYTHONPATH:$(pwd)

# Function to run a Python script
run_script() {
  echo "Running \$1..."
  python \$1 \$2 \$3 \$4 \$5
}

# Run requested script
if [ -z "\$1" ]; then
  echo "Please specify a script to run, e.g.:"
  echo "./run_benbot.sh app.py"
  echo "./run_benbot.sh test_autonomous_core.py"
  echo "./run_benbot.sh demo_autonomous_pipeline.py"
else
  run_script \$1 \$2 \$3 \$4 \$5
fi
EOL

chmod +x run_benbot.sh

echo -e "\n✅ Environment setup complete!"
echo -e "\n╭────────────────────────────────────────────────────╮"
echo "│  To run BensBot or tests, use the run script:       │"
echo "│  ./run_benbot.sh app.py                             │"
echo "│  ./run_benbot.sh test_autonomous_core.py            │"
echo "│                                                      │"
echo "│  This will ensure the Python path is set correctly.  │"
echo "╰────────────────────────────────────────────────────╯"
