#!/bin/bash
# Fixed environment setup script for BensBot trading system

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         BensBot Trading System - Environment Fix          ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Check if Python 3 is available
PYTHON_CMD=$(which python3)
if [ -z "$PYTHON_CMD" ]; then
  echo "❌ Python 3 not found. Please install Python 3 and try again."
  exit 1
fi

echo "✅ Found Python: $($PYTHON_CMD --version)"

# Create a temporary requirements file with everything we need
echo -e "\n📋 Creating comprehensive requirements file..."

cat > requirements_complete.txt << EOL
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
requests>=2.28.0
python-dotenv>=1.0.0
gym>=0.26.0
streamlit>=1.20.0
torch>=2.0.0
tenacity>=8.2.0
cachetools>=5.3.0
fastapi>=0.110
uvicorn[standard]>=0.28
yfinance>=0.2.33
flask>=2.0.0
flask-socketio>=5.0.0
flask-cors>=3.0.10
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0
EOL

# Install directly using pip with the user flag
echo -e "\n📦 Installing packages using pip with --user flag..."
$PYTHON_CMD -m pip install --user -r requirements_complete.txt

# Create a run helper script that sets PYTHONPATH properly
echo -e "\n🐍 Creating run helper script..."

cat > run_benbot.sh << EOL
#!/bin/bash
# Run script for BensBot with proper environment

# Set Python path to include current directory
export PYTHONPATH=\$PYTHONPATH:$(pwd)

# Function to run a Python script
run_script() {
  echo "Running \$1..."
  python3 \$1 \$2 \$3 \$4 \$5
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

# Add .env file if it doesn't exist
if [ ! -f ".env" ]; then
  echo -e "\n🔑 Creating .env file from example..."
  cp .env.example .env
fi

echo -e "\n✅ Environment setup complete!"
echo -e "\n╭────────────────────────────────────────────────────╮"
echo "│  To run BensBot or tests, use the run script:       │"
echo "│  ./run_benbot.sh app.py                             │"
echo "│  ./run_benbot.sh test_autonomous_core.py            │"
echo "│                                                      │"
echo "│  This will ensure the Python path is set correctly.  │"
echo "╰────────────────────────────────────────────────────╯"
