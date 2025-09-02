#!/bin/bash

# Set colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================================${NC}"
echo -e "${GREEN}    Opening BensBot Pro Portfolio Dashboard                ${NC}"
echo -e "${GREEN}===========================================================${NC}"

# Define path to the dashboard HTML file
DASHBOARD_FILE="/Users/bendickinson/Desktop/Trading:BenBot/portfolio_dashboard_fixed.html"

# Check if the file exists
if [ ! -f "$DASHBOARD_FILE" ]; then
    echo -e "${YELLOW}Dashboard file not found. Creating a new copy...${NC}"
    # If the script can't find the file, it will tell the user
    echo -e "${YELLOW}Please run the script again after creating the file.${NC}"
    exit 1
fi

echo -e "${GREEN}Opening dashboard in your default browser...${NC}"

# Open the dashboard file in the default browser
open "$DASHBOARD_FILE"

echo -e "${GREEN}Dashboard opened successfully!${NC}"
