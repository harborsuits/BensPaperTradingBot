#!/bin/bash
# Launch the trading dashboard with AI integration
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=================================={NC}"
echo -e "${GREEN}BenBot AI Dashboard Launcher{NC}"
echo -e "${BLUE}=================================={NC}"

# Start API server in a new terminal window
echo -e "${BLUE}Starting API server in new terminal...{NC}"
osascript -e 'tell application "Terminal" to do script "cd /Users/bendickinson/Desktop/Trading:BenBot && ./start_ai_api.sh"'

# Give API server time to start
echo -e "${YELLOW}Waiting for API server to start...{NC}"
sleep 3

# Open browser to dashboard
echo -e "${GREEN}Opening dashboard in browser...{NC}"
open "http://localhost:3000"

echo -e "${GREEN}Setup complete!{NC}"
echo -e "${BLUE}Your dashboard is available at: http://localhost:3000{NC}"
echo -e "${BLUE}The API server is running in a separate terminal window.{NC}"
echo -e "${BLUE}=================================={NC}"
