#!/bin/bash
# Complete Environment Setup and Strategy Testing

set -e  # Exit on any error

echo "🔧 Complete Trading Bot Environment Setup & Strategy Testing"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Step 1: Check current Python
echo ""
echo "1️⃣  Checking Current Python Environment:"
python3 --version || print_error "Python3 not found"
which python3 || print_error "Python3 not in PATH"

# Step 2: Create clean virtual environment
echo ""
echo "2️⃣  Creating Clean Virtual Environment:"
if [ -d "trading_env" ]; then
    rm -rf trading_env
    print_warning "Removed existing trading_env"
fi

python3 -m venv trading_env || {
    print_error "Failed to create virtual environment"
    exit 1
}
print_status "Created virtual environment"

# Step 3: Activate and install dependencies
echo ""
echo "3️⃣  Installing Dependencies:"
source trading_env/bin/activate || {
    print_error "Failed to activate virtual environment"
    exit 1
}
print_status "Activated virtual environment"

# Upgrade pip
python -m pip install --upgrade pip || {
    print_error "Failed to upgrade pip"
    exit 1
}
print_status "Upgraded pip"

# Install core dependencies with specific versions
pip install --no-cache-dir \
    "numpy==2.2.5" \
    "pandas==2.2.3" \
    "scipy==1.16.2" \
    "scikit-learn==1.7.2" || {
    print_error "Failed to install core dependencies"
    exit 1
}
print_status "Installed core dependencies"

# Step 4: Test basic imports
echo ""
echo "4️⃣  Testing Core Imports:"
python -c "
import sys
import importlib
import platform

print('Python version:', platform.python_version())

core_modules = ['numpy', 'pandas', 'scipy', 'sklearn']
for module in core_modules:
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✅ {module}: {version}')
    except ImportError as e:
        print(f'❌ {module}: Import failed - {e}')
        sys.exit(1)

print('🎉 All core imports successful!')
" || {
    print_error "Core import tests failed"
    exit 1
}

# Step 5: Test project imports
echo ""
echo "5️⃣  Testing Project Imports:"
export PYTHONPATH="$PWD:$PYTHONPATH"

python -c "
import sys
sys.path.insert(0, '.')

try:
    import trading_bot
    print('✅ trading_bot: imported successfully')
except ImportError as e:
    print(f'❌ trading_bot: Import failed - {e}')
    sys.exit(1)
" || {
    print_error "Project import tests failed"
    exit 1
}

# Step 6: Test simple strategy
echo ""
echo "6️⃣  Testing Simple Strategy:"
python test_strategy_pipeline.py || {
    print_error "Strategy testing failed"
    exit 1
}

# Step 7: Summary and next steps
echo ""
echo "🎯 TESTING COMPLETE!"
echo "==================="
print_status "Environment: Stable"
print_status "Core dependencies: Working"
print_status "Project imports: Working"
print_status "Basic strategy testing: Working"

echo ""
echo "📈 NEXT STEPS:"
echo "1. Review the strategy test results above"
echo "2. If mean reversion works, optimize parameters"
echo "3. Test more complex strategies (covered calls, etc.)"
echo "4. Compare against academic literature"
echo ""
echo "To continue working:"
echo "  source trading_env/bin/activate"
echo "  export PYTHONPATH=\"$PWD:$PYTHONPATH\""
echo "  python test_strategy_pipeline.py"
