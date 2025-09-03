#!/bin/bash

# BenBot Simple Launcher Script
# This script starts both the API server and the UI server

echo "=== BenBot Simple Launcher ==="
echo "Starting servers..."

# Kill any existing processes
echo "Stopping any existing servers..."
lsof -ti:3001 | xargs kill -9 2>/dev/null || true
lsof -ti:3003 | xargs kill -9 2>/dev/null || true
pkill -f "python basic_server.py" 2>/dev/null || true
pkill -f "python3 basic_server.py" 2>/dev/null || true
pkill -f "vite --port 3003" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
sleep 2

# Start API server
echo "Starting API server on port 3001..."
cd "$(dirname "$0")"
python3 basic_server.py &
API_PID=$!
echo "API server PID: $API_PID"

# Wait for API to be ready
echo "Waiting for API to start..."
for i in {1..10}; do
    if curl -s http://localhost:3001/api/v1/health > /dev/null; then
        echo "✓ API server is ready!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "✗ API server failed to start."
        exit 1
    fi
    echo "  Waiting... ($i/10)"
    sleep 1
done

# Choose which UI to start
echo "Choose which UI to start:"
echo "1) HTML Dashboard (simple)"
echo "2) React Dashboard (advanced)"
echo "3) Both"
read -p "Enter your choice (1-3): " choice

# Start HTML Dashboard
if [ "$choice" = "1" ] || [ "$choice" = "3" ]; then
    echo "Opening HTML Dashboard..."
    open benbot_dashboard.html
fi

# Start React UI
if [ "$choice" = "2" ] || [ "$choice" = "3" ]; then
    echo "Starting React UI on port 3003..."
    cd new-trading-dashboard
    npm run dev &
    UI_PID=$!
    echo "React UI PID: $UI_PID"
    
    # Wait for UI to be ready
    echo "Waiting for React UI to start..."
    for i in {1..15}; do
        if curl -s http://localhost:3003 > /dev/null; then
            echo "✓ React UI is ready!"
            break
        fi
        if [ $i -eq 15 ]; then
            echo "✗ React UI failed to start."
        fi
        echo "  Waiting... ($i/15)"
        sleep 1
    done
    
    # Open React UI in browser
    echo "Opening React UI in browser..."
    open -a "Google Chrome" --args --disable-extensions --incognito "http://localhost:3003?nocache=$(date +%s)"
fi

echo ""
echo "=== Servers Started Successfully ==="
echo "API: http://localhost:3001 (PID: $API_PID)"
if [ "$choice" = "2" ] || [ "$choice" = "3" ]; then
    echo "React UI: http://localhost:3003 (PID: $UI_PID)"
fi
echo ""
echo "To stop servers, run: kill $API_PID $UI_PID"
