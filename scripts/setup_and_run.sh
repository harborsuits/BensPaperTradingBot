#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  BensBot Trading System Setup Tool   ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Project root directory
ROOT_DIR="$(pwd)"
VENV_DIR="${ROOT_DIR}/venv"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "${VENV_DIR}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install venv package.${NC}"
        echo -e "${YELLOW}Try: python3 -m pip install --user virtualenv${NC}"
        exit 1
    fi
    echo -e "${GREEN}Virtual environment created at ${VENV_DIR}${NC}"
else
    echo -e "${GREEN}Using existing virtual environment at ${VENV_DIR}${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "${VENV_DIR}/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${YELLOW}Installing required packages...${NC}"
pip install streamlit fastapi uvicorn pandas numpy plotly matplotlib scikit-learn ta ipython requests python-dotenv pyyaml

# Install additional packages for trading
echo -e "${YELLOW}Installing trading-specific packages...${NC}"
pip install alpaca-trade-api yfinance ccxt python-binance

# Create data directories if they don't exist
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p "${ROOT_DIR}/data/performance"
mkdir -p "${ROOT_DIR}/logs"

# Update launcher scripts to use virtual environment
echo -e "${YELLOW}Updating launcher scripts...${NC}"

# Create enhanced API server launcher
echo -e "${YELLOW}Creating enhanced API server launcher...${NC}"
cat > "${ROOT_DIR}/start_api_server.sh" << 'EOF'
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading API Server...${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
source "$(pwd)/venv/bin/activate"

# Start the FastAPI server with uvicorn
echo -e "${YELLOW}Starting FastAPI server on port 8000...${NC}"
uvicorn trading_bot.api.main:app --reload --host 0.0.0.0 --port 8000

echo -e "${GREEN}API server is running at http://localhost:8000${NC}"
EOF
chmod +x "${ROOT_DIR}/start_api_server.sh"

# Create enhanced trading engine launcher
echo -e "${YELLOW}Creating enhanced trading engine launcher...${NC}"
cat > "${ROOT_DIR}/start_trading_engine.sh" << 'EOF'
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading Engine...${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
source "$(pwd)/venv/bin/activate"

# Check if config path is specified, otherwise use default
CONFIG_PATH=${1:-"./config/trading_config.json"}

echo -e "${YELLOW}Using configuration: ${CONFIG_PATH}${NC}"

# Start the trading engine
python -m trading_bot.main --config "${CONFIG_PATH}"

echo -e "${GREEN}Trading engine has completed its run${NC}"
EOF
chmod +x "${ROOT_DIR}/start_trading_engine.sh"

# Create enhanced dashboard launcher
echo -e "${YELLOW}Creating enhanced dashboard launcher...${NC}"
cat > "${ROOT_DIR}/start_dashboard.sh" << 'EOF'
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading Dashboard...${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
source "$(pwd)/venv/bin/activate"

# Start Streamlit dashboard
echo -e "${YELLOW}Starting Streamlit dashboard on port 8501...${NC}"
streamlit run trading_bot/dashboard/app.py --server.port 8501

echo -e "${GREEN}Dashboard is running at http://localhost:8501${NC}"
EOF
chmod +x "${ROOT_DIR}/start_dashboard.sh"

# Create enhanced autonomous system launcher
echo -e "${YELLOW}Creating enhanced autonomous system launcher...${NC}"
cat > "${ROOT_DIR}/start_autonomous_system.sh" << 'EOF'
#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  BensBot Trading System Launcher     ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate virtual environment
source "$(pwd)/venv/bin/activate"

# Start all components
echo -e "${YELLOW}Starting full autonomous trading system...${NC}"

# 1. Start the API server in the background
echo -e "${YELLOW}[1/3] Starting FastAPI Backend Server...${NC}"
terminal_cmd=""

# Try different terminal commands based on OS
if command -v gnome-terminal &> /dev/null; then
    # Linux with GNOME
    gnome-terminal --title="BensBot API Server" -- bash -c "cd $(pwd) && ./start_api_server.sh; read -p 'Press Enter to close...'" &
elif command -v osascript &> /dev/null; then
    # macOS
    osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && ./start_api_server.sh\"" &
elif command -v xterm &> /dev/null; then
    # Linux with X11
    xterm -title "BensBot API Server" -e "cd $(pwd) && ./start_api_server.sh; read -p 'Press Enter to close...'" &
else
    echo -e "${RED}Could not find a suitable terminal. Starting API server in the background.${NC}"
    nohup ./start_api_server.sh > ./logs/api_server.log 2>&1 &
fi

# Give API server time to start
echo -e "${YELLOW}Giving API server time to initialize (5 seconds)...${NC}"
sleep 5

# 2. Start the trading engine in another background terminal
echo -e "${YELLOW}[2/3] Starting Trading Engine...${NC}"
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal --title="BensBot Trading Engine" -- bash -c "cd $(pwd) && ./start_trading_engine.sh; read -p 'Press Enter to close...'" &
elif command -v osascript &> /dev/null; then
    osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && ./start_trading_engine.sh\"" &
elif command -v xterm &> /dev/null; then
    xterm -title "BensBot Trading Engine" -e "cd $(pwd) && ./start_trading_engine.sh; read -p 'Press Enter to close...'" &
else
    echo -e "${RED}Could not find a suitable terminal. Starting trading engine in the background.${NC}"
    nohup ./start_trading_engine.sh > ./logs/trading_engine.log 2>&1 &
fi

# Give trading engine time to start
echo -e "${YELLOW}Giving trading engine time to initialize (5 seconds)...${NC}"
sleep 5

# 3. Start the dashboard
echo -e "${YELLOW}[3/3] Starting Dashboard UI...${NC}"
if command -v gnome-terminal &> /dev/null; then
    gnome-terminal --title="BensBot Dashboard" -- bash -c "cd $(pwd) && ./start_dashboard.sh; read -p 'Press Enter to close...'" &
elif command -v osascript &> /dev/null; then
    osascript -e "tell application \"Terminal\" to do script \"cd $(pwd) && ./start_dashboard.sh\"" &
elif command -v xterm &> /dev/null; then
    xterm -title "BensBot Dashboard" -e "cd $(pwd) && ./start_dashboard.sh; read -p 'Press Enter to close...'" &
else
    echo -e "${RED}Could not find a suitable terminal. Starting dashboard in the background.${NC}"
    nohup ./start_dashboard.sh > ./logs/dashboard.log 2>&1 &
fi

echo -e "${GREEN}All components of the BensBot Trading System have been started!${NC}"
echo -e "${GREEN}Dashboard: http://localhost:8501${NC}"
echo -e "${GREEN}API Server: http://localhost:8000${NC}"
echo -e "${YELLOW}Check the logs directory for output from background processes${NC}"
EOF
chmod +x "${ROOT_DIR}/start_autonomous_system.sh"

echo -e "${GREEN}All launcher scripts updated successfully!${NC}"
echo -e "${BLUE}=======================================${NC}"
echo -e "${GREEN}Setup completed successfully! You can now run:${NC}"
echo -e "${YELLOW}./start_autonomous_system.sh${NC} - To start the full autonomous system"
echo -e "${YELLOW}./start_dashboard.sh${NC} - To start only the dashboard"
echo -e "${YELLOW}./start_api_server.sh${NC} - To start only the API server"
echo -e "${YELLOW}./start_trading_engine.sh${NC} - To start only the trading engine"
echo -e "${BLUE}=======================================${NC}"

# Offer to run the autonomous system
echo -e "${YELLOW}Would you like to start the full autonomous system now? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]; then
    ./start_autonomous_system.sh
else
    echo -e "${GREEN}Setup completed. Run ./start_autonomous_system.sh when you're ready.${NC}"
fi
