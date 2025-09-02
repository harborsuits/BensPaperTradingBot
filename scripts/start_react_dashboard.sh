#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================${NC}"
echo -e "${GREEN}Starting BensBot React Dashboard${NC}"
echo -e "${BLUE}=================================${NC}"

# Set project directory
PROJECT_DIR="/Users/bendickinson/Desktop/Trading:BenBot"
DASHBOARD_DIR="$PROJECT_DIR/new-trading-dashboard"

# Check if node_modules exists
if [ ! -d "$DASHBOARD_DIR/node_modules" ]; then
  echo -e "${YELLOW}Installing React dependencies (this may take a minute)...${NC}"
  cd "$DASHBOARD_DIR" && npm install
fi

# Start the React dashboard
echo -e "${GREEN}Starting React dashboard...${NC}"
echo -e "${BLUE}The dashboard will automatically try to connect to the API${NC}"
echo -e "${BLUE}If the API is not available, it will use enhanced simulation mode${NC}"
echo -e "${BLUE}You can access the dashboard at: http://localhost:3000${NC}"
echo -e "${BLUE}=================================${NC}"

# Set browser to none to avoid opening browser automatically
export BROWSER=none

# Start React development server
cd "$DASHBOARD_DIR" && npm start
