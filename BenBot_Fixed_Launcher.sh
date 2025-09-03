#!/bin/bash
# BenBot Launcher - Fixed version with port 3001 for API

# Ensure PATH has expected locations
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

LOG="$HOME/Desktop/BenBot_ui.log"
API_DIR="$HOME/Desktop/benbot"
UI_DIR="$HOME/Desktop/benbot/new-trading-dashboard"
API_PORT=3001  # Changed from 3000 to 3001
UI_PORT=3003
PYTHON="/opt/homebrew/bin/python3"
NPM="/opt/homebrew/bin/npm"
CURL="/usr/bin/curl"
LSOF="/usr/sbin/lsof"
OPEN="/usr/bin/open"

# Fallbacks if those binaries don't exist
[ -x "$PYTHON" ] || PYTHON="/usr/bin/python3"
[ -x "$PYTHON" ] || PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
[ -x "$PYTHON" ] || PYTHON="python3"  # Last resort
[ -x "$NPM" ] || NPM="/usr/local/bin/npm"
[ -x "$NPM" ] || NPM="/usr/bin/npm"
[ -x "$NPM" ] || NPM="npm"  # Last resort

# Start clean log
{
  echo "==============================================="
  echo "🚀 BenBot Trading Dashboard Launch (Fixed)"
  date
  echo "==============================================="
} > "$LOG"

# Kill any existing processes
{
  echo "Cleaning up old processes..."
  pkill -f "python basic_server.py" 2>/dev/null || true
  pkill -f "python3 basic_server.py" 2>/dev/null || true
  pkill -f "python simple_server.py" 2>/dev/null || true
  pkill -f "python3 simple_server.py" 2>/dev/null || true
  pkill -f "vite" 2>/dev/null || true
  pkill -f "npm run dev" 2>/dev/null || true
  $LSOF -ti:$API_PORT | xargs kill -9 2>/dev/null || true
  $LSOF -ti:$UI_PORT | xargs kill -9 2>/dev/null || true
  sleep 1
} >> "$LOG" 2>&1

# Start API
{
  echo "Starting API server with: $PYTHON"
  cd "$API_DIR" || exit 1
  
  # Try basic_server.py first (our new server on port 3001)
  if [ -f basic_server.py ]; then
    echo "Using basic_server.py"
    nohup "$PYTHON" basic_server.py >> "$LOG" 2>&1 &
  else
    # Fall back to simple_server.py
    echo "Using simple_server.py"
    nohup "$PYTHON" simple_server.py >> "$LOG" 2>&1 &
  fi
  
  API_PID=$!
  echo $API_PID > "$HOME/Desktop/benbot_api.pid"
  echo "API PID: $API_PID"
} >> "$LOG" 2>&1

# Wait for API
{
  echo "Waiting for API to start..."
  for i in {1..15}; do
    echo "  Attempt $i/15..."
    if $CURL -fsS "http://localhost:$API_PORT/api/v1/health" >/dev/null 2>&1; then
      echo "✓ API is UP on :$API_PORT"
      break
    elif $CURL -fsS "http://localhost:$API_PORT/" >/dev/null 2>&1; then
      echo "✓ API is UP on :$API_PORT (root endpoint)"
      break
    fi
    sleep 1
    if [ "$i" -eq 15 ]; then
      echo "API failed to start"
      osascript -e 'display dialog "API failed to start. Check BenBot_ui.log." with title "BenBot Launcher Error" buttons {"OK"} default button "OK"' || true
      exit 1
    fi
  done
} >> "$LOG" 2>&1

# Ensure UI env
{
  cd "$UI_DIR" || exit 1
  echo "Configuring UI environment..."
  echo "VITE_API_BASE=http://localhost:$API_PORT" > .env.local
  echo "VITE_API_URL=http://localhost:$API_PORT" >> .env.local
  echo "Created/updated .env.local"
} >> "$LOG" 2>&1

# Start UI
{
  echo "Starting UI with: $NPM"
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
      osascript -e 'display dialog "UI failed to start. Check BenBot_ui.log." with title "BenBot Launcher Error" buttons {"OK"} default button "OK"' || true
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
exit 0
