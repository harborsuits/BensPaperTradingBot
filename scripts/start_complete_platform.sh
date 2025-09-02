#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}   BensBot Complete Trading Platform Startup${NC}"
echo -e "${BLUE}===================================================${NC}"

# Project path definitions
PROJECT_ROOT="/Users/bendickinson/Desktop/Trading:BenBot"
VENV_PATH="/Users/bendickinson/Desktop/trading_venv"
API_DIR="$PROJECT_ROOT/trading_bot/api"
BACKTESTER_API_PATH="/Users/bendickinson/Desktop/backtester_api.py"
if [ ! -f "$BACKTESTER_API_PATH" ]; then
    BACKTESTER_API_PATH="/Users/bendickinson/Desktop/Trading:BenBot/backtester_api.py"
fi

# ==========================================
# 1. Clean up any existing processes
# ==========================================
echo -e "\n${YELLOW}Step 1: Cleaning up existing processes...${NC}"

# Find and kill processes using ports 8000 (API), 5002 (Backtester), and 3003 (Frontend)
for PORT in 8000 5002 3003 5173; do
  PIDS=$(lsof -ti:$PORT)
  if [ ! -z "$PIDS" ]; then
    echo -e "Killing processes using port $PORT: $PIDS"
    kill -9 $PIDS
  fi
done

echo -e "${GREEN}Process cleanup complete${NC}"

# ==========================================
# 2. Set up virtual environment
# ==========================================
echo -e "\n${YELLOW}Step 2: Setting up virtual environment...${NC}"

# Activate virtual environment
echo -e "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo -e "${GREEN}Virtual environment activated${NC}"

# ==========================================
# 3. Start backend API server
# ==========================================
echo -e "\n${YELLOW}Step 3: Starting backend API server...${NC}"
echo -e "Setting PYTHONPATH to include project root..."
export PYTHONPATH="$PROJECT_ROOT"

echo -e "Starting FastAPI server on port 8000..."
cd "$PROJECT_ROOT"
python -m uvicorn trading_bot.api.main:app --reload --host 0.0.0.0 --port 8000 > /tmp/trading_api.log 2>&1 &
API_PID=$!

# Verify API has started
sleep 2
if ps -p $API_PID > /dev/null; then
  echo -e "${GREEN}API server started successfully on port 8000 (PID: $API_PID)${NC}"
else
  echo -e "${RED}API server failed to start${NC}"
  exit 1
fi

# ==========================================
# 4. Start backtester API
# ==========================================
echo -e "\n${YELLOW}Step 4: Starting backtester API server...${NC}"

echo -e "Starting backtester API on port 5002..."
cd $(dirname "$BACKTESTER_API_PATH")
python $(basename "$BACKTESTER_API_PATH") > /tmp/backtester_api.log 2>&1 &
BACKTESTER_PID=$!

# Verify backtester has started
sleep 2
if ps -p $BACKTESTER_PID > /dev/null; then
  echo -e "${GREEN}Backtester API started successfully on port 5002 (PID: $BACKTESTER_PID)${NC}"
else
  echo -e "${RED}Backtester API failed to start${NC}"
  # Continue even if backtester fails
fi

# ==========================================
# 5. Start React dashboard UI
# ==========================================
echo -e "\n${YELLOW}Step 5: Starting React dashboard UI...${NC}"

# Try starting from trading-dashboard first (port 3003)
if [ -d "$PROJECT_ROOT/trading-dashboard" ]; then
  echo -e "Starting trading-dashboard on port 3003..."
  cd "$PROJECT_ROOT/trading-dashboard"
  PORT=3003 npm start > /tmp/trading_dashboard.log 2>&1 &
  DASHBOARD_PID=$!
  sleep 4
  
  if curl -s http://localhost:3003 > /dev/null; then
    echo -e "${GREEN}Trading dashboard started successfully on port 3003 (PID: $DASHBOARD_PID)${NC}"
  else
    echo -e "${YELLOW}Trading dashboard on port 3003 failed to start, trying next option...${NC}"
    kill -9 $DASHBOARD_PID 2>/dev/null
    
    # Try react-dashboard next
    if [ -d "$PROJECT_ROOT/react-dashboard" ]; then
      echo -e "Starting react-dashboard..."
      cd "$PROJECT_ROOT/react-dashboard"
      npm start > /tmp/trading_dashboard.log 2>&1 &
      DASHBOARD_PID=$!
      sleep 4
      
      if curl -s http://localhost:3000 > /dev/null; then
        echo -e "${GREEN}React dashboard started successfully on port 3000 (PID: $DASHBOARD_PID)${NC}"
      else
        echo -e "${YELLOW}React dashboard failed to start, trying new-trading-dashboard...${NC}"
        kill -9 $DASHBOARD_PID 2>/dev/null
        
        # Try new-trading-dashboard with Vite
        if [ -d "$PROJECT_ROOT/new-trading-dashboard" ]; then
          echo -e "Starting new-trading-dashboard with Vite..."
          cd "$PROJECT_ROOT/new-trading-dashboard"
          npm run dev > /tmp/trading_dashboard.log 2>&1 &
          DASHBOARD_PID=$!
          sleep 4
          
          if curl -s http://localhost:5173 > /dev/null; then
            echo -e "${GREEN}New trading dashboard started successfully on port 5173 (PID: $DASHBOARD_PID)${NC}"
          else
            echo -e "${RED}All dashboard options failed to start${NC}"
          fi
        fi
      fi
    fi
  fi
else
  echo -e "${YELLOW}trading-dashboard not found, trying new-trading-dashboard...${NC}"
  
  # Try new-trading-dashboard with Vite
  if [ -d "$PROJECT_ROOT/new-trading-dashboard" ]; then
    echo -e "Starting new-trading-dashboard with Vite..."
    cd "$PROJECT_ROOT/new-trading-dashboard"
    npm run dev > /tmp/trading_dashboard.log 2>&1 &
    DASHBOARD_PID=$!
    sleep 4
    
    if curl -s http://localhost:5173 > /dev/null; then
      echo -e "${GREEN}New trading dashboard started successfully on port 5173 (PID: $DASHBOARD_PID)${NC}"
    else
      echo -e "${YELLOW}New trading dashboard failed to start, trying react-dashboard...${NC}"
      kill -9 $DASHBOARD_PID 2>/dev/null
      
      # Try react-dashboard as fallback
      if [ -d "$PROJECT_ROOT/react-dashboard" ]; then
        echo -e "Starting react-dashboard..."
        cd "$PROJECT_ROOT/react-dashboard"
        npm start > /tmp/trading_dashboard.log 2>&1 &
        DASHBOARD_PID=$!
        sleep 4
        
        if curl -s http://localhost:3000 > /dev/null; then
          echo -e "${GREEN}React dashboard started successfully on port 3000 (PID: $DASHBOARD_PID)${NC}"
        else
          echo -e "${RED}All dashboard options failed to start${NC}"
        fi
      fi
    fi
  else
    echo -e "${RED}No viable dashboard options found${NC}"
  fi
fi

# ==========================================
# 6. Display access information
# ==========================================
echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}   BensBot Trading Platform Started${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "Backend API:    ${BLUE}http://localhost:8000${NC}"
echo -e "API Docs:       ${BLUE}http://localhost:8000/docs${NC}"
echo -e "Backtester API: ${BLUE}http://localhost:5002${NC}"

# Determine which dashboard is running
if curl -s http://localhost:3003 > /dev/null; then
  echo -e "Trading UI:     ${BLUE}http://localhost:3003${NC}"
elif curl -s http://localhost:5173 > /dev/null; then
  echo -e "Trading UI:     ${BLUE}http://localhost:5173${NC}"
elif curl -s http://localhost:3000 > /dev/null; then
  echo -e "Trading UI:     ${BLUE}http://localhost:3000${NC}"
else
  echo -e "Trading UI:     ${RED}Not available${NC}"
fi

echo -e "\nProcess IDs for manual termination:"
echo -e "API Server:     ${API_PID}"
echo -e "Backtester API: ${BACKTESTER_PID}"
if [ ! -z "$DASHBOARD_PID" ]; then
  echo -e "Dashboard UI:   ${DASHBOARD_PID}"
fi

echo -e "\nLog Files:"
echo -e "API Server:     /tmp/trading_api.log"
echo -e "Backtester API: /tmp/backtester_api.log"
echo -e "Dashboard UI:   /tmp/trading_dashboard.log"

echo -e "\nTo stop all processes run: ${YELLOW}pkill -f 'uvicorn|npm|node'${NC}"
echo -e "${GREEN}===================================================${NC}"
