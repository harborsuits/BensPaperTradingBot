#!/bin/bash
# BenBot Production Launcher - Uses minimal_server.js API and new-trading-dashboard UI

# Ensure PATH has expected locations
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

LOG="$HOME/Desktop/BenBot_production.log"
API_DIR="$HOME/Desktop/benbot/live-api"
UI_DIR="$HOME/Desktop/benbot/new-trading-dashboard"
PYTHON_BRAIN_DIR="$HOME/Desktop/benbot/trading_bot"
API_PORT=4000
UI_PORT=3003
PYTHON_BRAIN_PORT=8001
NODE="/opt/homebrew/bin/node"
NPM="/opt/homebrew/bin/npm"
PYTHON="/opt/homebrew/bin/python3"
CURL="/usr/bin/curl"
LSOF="/usr/sbin/lsof"
OPEN="/usr/bin/open"

# Fallbacks if those binaries don't exist
[ -x "$NODE" ] || NODE="/usr/local/bin/node"
[ -x "$NODE" ] || NODE="/usr/bin/node"
[ -x "$NODE" ] || NODE="node"
[ -x "$NPM" ] || NPM="/usr/local/bin/npm"
[ -x "$NPM" ] || NPM="/usr/bin/npm"
[ -x "$NPM" ] || NPM="npm"
[ -x "$PYTHON" ] || PYTHON="/usr/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"

# Start clean log
{
  echo "==============================================="
  echo "🚀 BenBot Trading Dashboard Launch (Production)"
  date
  echo "==============================================="
} > "$LOG"

# Kill any existing processes
{
  echo "Cleaning up old processes..."
  pkill -f "node minimal_server.js" 2>/dev/null || true
  pkill -f "node server.js" 2>/dev/null || true
  pkill -f "python_brain_service.py" 2>/dev/null || true
  pkill -f "vite" 2>/dev/null || true
  pkill -f "npm run dev" 2>/dev/null || true
  $LSOF -ti:$API_PORT | xargs kill -9 2>/dev/null || true
  $LSOF -ti:$UI_PORT | xargs kill -9 2>/dev/null || true
  $LSOF -ti:$PYTHON_BRAIN_PORT | xargs kill -9 2>/dev/null || true
  sleep 1
} >> "$LOG" 2>&1

# Start Python Brain Service
{
  echo "Starting Python Brain Service..."
  cd "$PYTHON_BRAIN_DIR" || exit 1
  
  if [ -f python_brain_service.py ]; then
    nohup "$PYTHON" python_brain_service.py --host localhost --port $PYTHON_BRAIN_PORT >> "$LOG" 2>&1 &
    BRAIN_PID=$!
    echo $BRAIN_PID > "$HOME/Desktop/benbot_brain.pid"
    echo "Python Brain PID: $BRAIN_PID"
    
    # Wait for brain to start
    echo "Waiting for Python Brain to start..."
    for i in {1..10}; do
      if $CURL -fsS "http://localhost:$PYTHON_BRAIN_PORT/health" >/dev/null 2>&1; then
        echo "✓ Python Brain is UP on :$PYTHON_BRAIN_PORT"
        break
      fi
      sleep 1
    done
  else
    echo "Warning: Python brain service not found, continuing without it"
  fi
} >> "$LOG" 2>&1

# Start API
{
  echo "Starting Node.js API server..."
  cd "$API_DIR" || exit 1
  
  # Set environment variables for production mode
  export ALLOW_OFFHOURS=0
  export BREAKERS_ENABLED=1
  export AUTOLOOP_ENABLED=1
  export STRATEGIES_ENABLED=1
  
  nohup "$NODE" minimal_server.js >> "$LOG" 2>&1 &
  API_PID=$!
  echo $API_PID > "$HOME/Desktop/benbot_api.pid"
  echo "API PID: $API_PID"
} >> "$LOG" 2>&1

# Wait for API
{
  echo "Waiting for API to start..."
  for i in {1..15}; do
    echo "  Attempt $i/15..."
    if $CURL -fsS "http://localhost:$API_PORT/api/health" >/dev/null 2>&1; then
      echo "✓ API is UP on :$API_PORT"
      break
    fi
    sleep 1
    if [ "$i" -eq 15 ]; then
      echo "API failed to start"
      osascript -e 'display dialog "API failed to start. Check BenBot_production.log." with title "BenBot Launcher Error" buttons {"OK"} default button "OK"' || true
      exit 1
    fi
  done
} >> "$LOG" 2>&1

# Ensure UI env
{
  cd "$UI_DIR" || exit 1
  echo "Configuring UI environment..."
  # The vite.config.ts already has the proxy configured for port 4000
  # Just ensure MSW is disabled
  echo "VITE_USE_MSW=false" > .env.local
  echo "Created/updated .env.local"
} >> "$LOG" 2>&1

# Start UI
{
  echo "Starting UI..."
  cd "$UI_DIR" || exit 1
  nohup "$NPM" run dev -- --port "$UI_PORT" --strictPort >> "$LOG" 2>&1 &
  UI_PID=$!
  echo $UI_PID > "$HOME/Desktop/benbot_ui.pid"
  echo "UI PID: $UI_PID"
} >> "$LOG" 2>&1

# Wait for UI
{
  echo "Waiting for UI to start..."
  for i in {1..20}; do
    echo "  Attempt $i/20..."
    if $CURL -fsS "http://localhost:$UI_PORT/" >/dev/null 2>&1; then
      echo "✓ UI is UP on :$UI_PORT"
      break
    fi
    sleep 1
    if [ "$i" -eq 20 ]; then
      echo "UI failed to start"
      osascript -e 'display dialog "UI failed to start. Check BenBot_production.log." with title "BenBot Launcher Error" buttons {"OK"} default button "OK"' || true
      exit 1
    fi
  done
} >> "$LOG" 2>&1

# Open browser with cache busting
TIMESTAMP=$(date +%s)
BROWSER_URL="http://localhost:$UI_PORT?nocache=$TIMESTAMP"

{
  echo "Opening browser to $BROWSER_URL"
  $OPEN -a "Google Chrome" --args --disable-extensions "$BROWSER_URL" >/dev/null 2>&1 || $OPEN "$BROWSER_URL" >/dev/null 2>&1 || true
  osascript -e 'display notification "BenBot Trading Dashboard is running!" with title "BenBot Launcher"' || true
} >> "$LOG" 2>&1

echo "All done! Check $LOG for details."
echo "Services running:"
echo "  - Python Brain: http://localhost:$PYTHON_BRAIN_PORT"
echo "  - API Server: http://localhost:$API_PORT"
echo "  - UI Dashboard: http://localhost:$UI_PORT"
exit 0
