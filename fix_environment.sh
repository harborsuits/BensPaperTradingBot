#!/bin/bash
# Trading Bot Environment Fix Script

echo "🔧 Trading Bot Environment Fix"
echo "==============================="

# Check current Python
echo "Current Python setup:"
which python3
python3 --version

# Create clean virtual environment
echo ""
echo "Creating clean virtual environment..."
python3 -m venv trading_env

# Activate and install dependencies
echo ""
echo "Installing dependencies..."
source trading_env/bin/activate
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib

# Test the environment
echo ""
echo "Testing environment..."
python -c "
import sys
print('Python version:', sys.version)
import numpy as np
print('NumPy version:', np.__version__)
import pandas as pd
print('Pandas version:', pd.__version__)
import scipy
print('SciPy version:', scipy.__version__)
from sklearn import __version__ as sklearn_version
print('Scikit-learn version:', sklearn_version)
print('✅ All core dependencies working!')
"

echo ""
echo "Environment setup complete!"
echo "To use: source trading_env/bin/activate"
echo "To test: python backtests/quick_ma.py --mean-reversion"
