#!/bin/bash

# Set colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Navigate to the project directory
cd "$(dirname "$0")/../BensBot-Pro-App"

# If the directory doesn't exist, give instructions
if [ ! -d "$(pwd)" ]; then
    echo -e "${RED}Error: BensBot-Pro-App directory not found at $(pwd)${NC}"
    echo -e "${YELLOW}Please make sure the BensBot-Pro-App directory exists in the Desktop folder${NC}"
    exit 1
fi

echo -e "${YELLOW}Setting up BensBot Pro Trading Dashboard...${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js to run this dashboard.${NC}"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
    
    # Add additional required dependencies
    echo -e "${YELLOW}Installing UI component dependencies...${NC}"
    npm install @radix-ui/react-tabs clsx tailwind-merge class-variance-authority
fi

# Create .env.local file for MongoDB connection if it doesn't exist
if [ ! -f ".env.local" ]; then
    echo -e "${YELLOW}Creating environment configuration...${NC}"
    echo "MONGODB_URI=mongodb://localhost:27017" > .env.local
fi

# Check if MongoDB is running
echo -e "${YELLOW}Checking MongoDB service...${NC}"
if ! pgrep -x "mongod" > /dev/null; then
    echo -e "${YELLOW}Starting MongoDB service...${NC}"
    brew services start mongodb-community
    sleep 2
fi

# Run the development server
echo -e "${GREEN}Starting BensBot Pro Trading Dashboard...${NC}"
echo -e "${YELLOW}The dashboard will be available at http://localhost:3000${NC}"

# Open browser (with a delay to ensure server has time to start)
(sleep 3 && open "http://localhost:3000") &

# Start the Next.js development server
npm run dev
