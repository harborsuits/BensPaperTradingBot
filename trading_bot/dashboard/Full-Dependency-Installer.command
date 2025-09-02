#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Show a nice header
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║      BensBot Trading Dashboard - Dependency Installer         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Navigate to the dashboard directory (using quoted path to handle the colon)
cd "/Users/bendickinson/Desktop/Trading:BenBot/trading_bot/dashboard"

echo -e "${YELLOW}Installing ALL possible trading bot dependencies...${NC}"

# Common trading packages
echo -e "${YELLOW}\n[1/5] Installing core data packages...${NC}"
pip3 install --break-system-packages pandas numpy matplotlib plotly

# Dashboard and UI packages
echo -e "${YELLOW}\n[2/5] Installing dashboard packages...${NC}"
pip3 install --break-system-packages streamlit

# Data provider packages
echo -e "${YELLOW}\n[3/5] Installing data provider packages...${NC}"
pip3 install --break-system-packages yfinance websocket-client backtrader ccxt ta scikit-learn

# Broker API packages
echo -e "${YELLOW}\n[4/5] Installing broker API packages...${NC}"
pip3 install --break-system-packages python-binance alpaca-trade-api polygon-api-client

# Utility packages
echo -e "${YELLOW}\n[5/5] Installing utility packages...${NC}"
pip3 install --break-system-packages requests psutil pytz pymongo SQLAlchemy frozendict peewee

echo -e "${GREEN}\nInstallation complete! All dependencies should now be installed.${NC}"
echo -e "${YELLOW}\nYou can now run the dashboard by double-clicking the 'Start Trading Dashboard (Enhanced).command' file on your desktop.${NC}"

# Wait for user to press a key before closing
echo ""
read -n 1 -s -r -p "Press any key to close this window..."
