#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}  BensBot Trading System Startup Tool  ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Set the Python path to include the project root
export PYTHONPATH=$PYTHONPATH:/Users/bendickinson/Desktop/Trading:BenBot

# 1. Start the API server in the background
echo -e "\n${YELLOW}[1/3] Starting FastAPI Backend Server...${NC}"
cd /Users/bendickinson/Desktop/Trading:BenBot
gnome-terminal --title="BensBot API Server" -- bash -c "cd /Users/bendickinson/Desktop/Trading:BenBot && ./trading_bot/api/start_api.sh; read -p 'Press Enter to close...'" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd /Users/bendickinson/Desktop/Trading:BenBot && ./trading_bot/api/start_api.sh"' || \
xterm -title "BensBot API Server" -e "cd /Users/bendickinson/Desktop/Trading:BenBot && ./trading_bot/api/start_api.sh; read -p 'Press Enter to close...'" || \
echo -e "${RED}Could not open a new terminal window. Please run ./trading_bot/api/start_api.sh manually in a new terminal.${NC}"

# Give API server time to start
echo -e "${YELLOW}Giving API server time to initialize (5 seconds)...${NC}"
sleep 5

# 2. Start the trading engine in another background terminal
echo -e "\n${YELLOW}[2/3] Starting Trading Engine...${NC}"
gnome-terminal --title="BensBot Trading Engine" -- bash -c "cd /Users/bendickinson/Desktop/Trading:BenBot && ./start_trading_engine.sh; read -p 'Press Enter to close...'" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd /Users/bendickinson/Desktop/Trading:BenBot && ./start_trading_engine.sh"' || \
xterm -title "BensBot Trading Engine" -e "cd /Users/bendickinson/Desktop/Trading:BenBot && ./start_trading_engine.sh; read -p 'Press Enter to close...'" || \
echo -e "${RED}Could not open a new terminal window. Please run ./start_trading_engine.sh manually in a new terminal.${NC}"

# Give trading engine time to start
echo -e "${YELLOW}Giving trading engine time to initialize (5 seconds)...${NC}"
sleep 5

# 3. Start React Dashboard UI
echo -e "\n${YELLOW}[3/3] Starting React Dashboard UI...${NC}"
gnome-terminal --title="React Dashboard" -- bash -c "cd /Users/bendickinson/Desktop/Trading:BenBot/new-trading-dashboard && bash setup_env.sh && npm install && npm start; read -p 'Press Enter to close...'" 2>/dev/null || \
osascript -e 'tell app "Terminal" to do script "cd /Users/bendickinson/Desktop/Trading:BenBot/new-trading-dashboard && bash setup_env.sh && npm install && npm start"' || \
xterm -title "React Dashboard" -e "cd /Users/bendickinson/Desktop/Trading:BenBot/new-trading-dashboard && bash setup_env.sh && npm install && npm start; read -p 'Press Enter to close...'" || \
echo -e "${RED}Could not open a new terminal window. Please run 'cd new-trading-dashboard && bash setup_env.sh && npm install && npm start' manually.${NC}"

echo -e "\n${GREEN}All components of the BensBot Trading System have been started!${NC}"
