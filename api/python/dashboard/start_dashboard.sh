#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting BensBot Trading Dashboard...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker is not running. Starting Docker...${NC}"
    open -a Docker
    
    # Wait for Docker to start
    echo -e "${YELLOW}Waiting for Docker to start...${NC}"
    while ! docker info > /dev/null 2>&1; do
        sleep 1
    done
    echo -e "${GREEN}Docker is now running.${NC}"
fi

# Build and start the container
echo -e "${YELLOW}Building and starting the dashboard container...${NC}"
# Explicitly set PWD to handle special characters in path
export PWD="$(pwd)"
docker-compose up --build -d

# Wait for the container to be ready
echo -e "${YELLOW}Waiting for dashboard to be ready...${NC}"
sleep 3

# Open browser automatically
echo -e "${GREEN}Dashboard is ready! Opening in browser...${NC}"
open http://localhost:8501

# Show logs
echo -e "${YELLOW}Showing container logs (Ctrl+C to exit logs, dashboard will continue running)${NC}"
docker-compose logs -f
