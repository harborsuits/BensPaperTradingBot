#!/bin/bash

# BenBot Minimal Launcher Script
# This script starts both the minimal API server and the UI server
# and ensures they are running correctly

echo "=== BenBot Minimal Launcher ==="
echo "Starting servers..."

# Kill any existing processes
echo "Stopping any existing servers..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:3003 | xargs kill -9 2>/dev/null || true
pkill -f "python minimal_server.py" 2>/dev/null || true
pkill -f "python3 minimal_server.py" 2>/dev/null || true
pkill -f "vite --port 3003" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
sleep 2

# Set up log files
API_LOG="$HOME/Desktop/BenBot_api.log"
UI_LOG="$HOME/Desktop/BenBot_ui.log"
echo "Logs will be written to:"
echo "  API: $API_LOG"
echo "  UI:  $UI_LOG"

# Start API server
echo "Starting minimal API server on port 3000..."
cd "$(dirname "$0")"
echo "$(date): Starting minimal API server" > "$API_LOG"
python3 minimal_server.py >> "$API_LOG" 2>&1 &
API_PID=$!
echo "API server PID: $API_PID"

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..10}; do
    if curl -s http://localhost:3000/api/v1/health > /dev/null; then
        echo "✓ API server is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "✗ API server failed to start. Check $API_LOG for details."
        exit 1
    fi
    echo "  Waiting... ($i/10)"
    sleep 1
done

# Start UI server with explicit port
echo "Starting UI server on port 3003..."
cd "$(dirname "$0")/new-trading-dashboard"
echo "$(date): Starting UI server" > "$UI_LOG"
npm run dev -- --port 3003 --strictPort >> "$UI_LOG" 2>&1 &
UI_PID=$!
echo "UI server PID: $UI_PID"

# Wait for UI to be ready
echo "Waiting for UI to start..."
for i in {1..15}; do
    if curl -s http://localhost:3003 > /dev/null; then
        echo "✓ UI server is ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "✗ UI server failed to start. Check $UI_LOG for details."
        exit 1
    fi
    echo "  Waiting... ($i/15)"
    sleep 1
done

echo ""
echo "=== Servers Started Successfully ==="
echo "API: http://localhost:3000 (PID: $API_PID)"
echo "UI:  http://localhost:3003 (PID: $UI_PID)"
echo ""
echo "Opening browser..."
open -a "Google Chrome" --args --disable-extensions --incognito "http://localhost:3003?nocache=$(date +%s)"

echo ""
echo "To stop servers, run: kill $API_PID $UI_PID"

# Save PIDs for later use
echo "$API_PID $UI_PID" > .benbot_pids
