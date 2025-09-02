#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}   BensBot Trading Environment Setup Script${NC}"
echo -e "${BLUE}===================================================${NC}"

# Project path definitions
PROJECT_ROOT="/Users/bendickinson/Desktop/Trading:BenBot"
VENV_PATH="/Users/bendickinson/Desktop/trading_venv"
API_DIR="$PROJECT_ROOT/trading_bot/api"
FRONTEND_DIR="$PROJECT_ROOT/new-trading-dashboard"

# ==========================================
# 1. Clean up any existing processes
# ==========================================
echo -e "\n${YELLOW}Step 1: Cleaning up existing processes...${NC}"

# Find and kill processes using ports 8000 (API) and 3003 (Frontend)
API_PIDS=$(lsof -ti:8000)
if [ ! -z "$API_PIDS" ]; then
  echo -e "Killing processes using port 8000: $API_PIDS"
  kill -9 $API_PIDS
fi

FRONTEND_PIDS=$(lsof -ti:3003)
if [ ! -z "$FRONTEND_PIDS" ]; then
  echo -e "Killing processes using port 3003: $FRONTEND_PIDS"
  kill -9 $FRONTEND_PIDS
fi

echo -e "${GREEN}Process cleanup complete${NC}"

# ==========================================
# 2. Set up virtual environment
# ==========================================
echo -e "\n${YELLOW}Step 2: Setting up virtual environment...${NC}"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
  echo -e "Creating virtual environment at $VENV_PATH..."
  python3 -m venv "$VENV_PATH"
else
  echo -e "Virtual environment already exists at $VENV_PATH"
fi

# Activate virtual environment
echo -e "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Install dependencies
echo -e "Installing dependencies..."
pip install -r "$PROJECT_ROOT/requirements.txt" 2>/dev/null || echo -e "${RED}No requirements.txt found${NC}"

# Install essential packages if they don't exist
pip install fastapi uvicorn pandas numpy pydantic websockets 

echo -e "${GREEN}Virtual environment setup complete${NC}"

# ==========================================
# 3. Fix import path issue (if needed)
# ==========================================
echo -e "\n${YELLOW}Step 3: Fixing import path issues...${NC}"

# Check for relative import issue in app.py
if grep -q "from trading_bot.api.websocket_manager import" "$API_DIR/app.py"; then
  echo -e "Found absolute import in app.py, creating backup and fixing..."
  cp "$API_DIR/app.py" "$API_DIR/app.py.backup"
  
  # Replace absolute import with relative import
  sed -i '' 's/from trading_bot.api.websocket_manager import/from .websocket_manager import/g' "$API_DIR/app.py"
  echo -e "${GREEN}Fixed import path issues in app.py${NC}"
else
  echo -e "No import path issues found or already fixed"
fi

# ==========================================
# 4. Start backend API server
# ==========================================
echo -e "\n${YELLOW}Step 4: Starting backend API server...${NC}"
echo -e "Setting PYTHONPATH to include project root..."
export PYTHONPATH="$PROJECT_ROOT"

echo -e "Starting FastAPI server in background..."
cd "$PROJECT_ROOT"
python -m uvicorn trading_bot.api.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/trading_api.log 2>&1 &
API_PID=$!

# Verify API has started
sleep 3
if ps -p $API_PID > /dev/null; then
  echo -e "${GREEN}API server started successfully (PID: $API_PID)${NC}"
  echo -e "API server logs can be found at /tmp/trading_api.log"
else
  echo -e "${RED}API server failed to start, trying alternative method...${NC}"
  cd "$API_DIR"
  python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 > /tmp/trading_api.log 2>&1 &
  API_PID=$!
  
  sleep 3
  if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}API server started with alternative method (PID: $API_PID)${NC}"
  else
    echo -e "${RED}ERROR: API server failed to start${NC}"
    echo -e "Check log file at /tmp/trading_api.log for details"
    echo -e "You may try running 'cd $API_DIR && python -m uvicorn app:app --reload' manually"
  fi
fi

# ==========================================
# 5. Start frontend React application (if available)
# ==========================================
echo -e "\n${YELLOW}Step 5: Starting frontend React application...${NC}"
if [ -d "$FRONTEND_DIR" ]; then
  echo -e "Found frontend directory at $FRONTEND_DIR"
  echo -e "Starting frontend application..."
  cd "$FRONTEND_DIR"
  
  # Install npm dependencies if needed
  if [ -f "package.json" ]; then
    echo -e "Installing npm dependencies if needed..."
    npm install --silent
    
    # Start development server
    echo -e "Starting React development server..."
    npm start > /tmp/trading_frontend.log 2>&1 &
    FRONTEND_PID=$!
    
    echo -e "${GREEN}Frontend started successfully${NC}"
    echo -e "Frontend logs can be found at /tmp/trading_frontend.log"
  else
    echo -e "${RED}No package.json found in frontend directory${NC}"
  fi
else
  echo -e "${RED}Frontend directory not found at $FRONTEND_DIR${NC}"
fi

# ==========================================
# 6. Display access information
# ==========================================
echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}   BensBot Trading Platform Started${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "Backend API:  ${BLUE}http://localhost:8000${NC}"
echo -e "Frontend UI:  ${BLUE}http://localhost:3003${NC}"
echo -e "API Docs:     ${BLUE}http://localhost:8000/docs${NC}"
echo -e "\nProcess IDs for manual termination:"
echo -e "API Server:   ${API_PID}"
if [ ! -z "$FRONTEND_PID" ]; then
  echo -e "Frontend:     ${FRONTEND_PID}"
fi
echo -e "\nTo stop all processes run: ${YELLOW}pkill -f 'uvicorn|npm start'${NC}"
echo -e "${GREEN}===================================================${NC}"
