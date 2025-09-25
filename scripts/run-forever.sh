#!/bin/bash
# Script to keep Benbot running indefinitely on Mac

echo "🤖 Benbot Forever Runner"
echo "========================"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create logs directory
mkdir -p "$BASE_DIR/logs"

# Function to check if process is running
is_running() {
    pgrep -f "$1" > /dev/null
}

# Function to start backend
start_backend() {
    echo "🚀 Starting backend..."
    cd "$BASE_DIR/live-api"
    
    # Export environment variables
    export TRADIER_API_KEY="${TRADIER_API_KEY:-KU2iUnOZIUFre0wypgyOn8TgmGxI}"
    export TRADIER_ACCOUNT_ID="${TRADIER_ACCOUNT_ID:-VA1201776}"
    export FINNHUB_API_KEY="${FINNHUB_API_KEY:-cqg6r9hr01qj0vhhf6fgcqg6r9hr01qj0vhhf6g0}"
    export MARKETAUX_API_TOKEN="${MARKETAUX_API_TOKEN:-sG6o8FgyTvJ8VxXFyBMOFWhJJ81QzB}"
    export PORT=4000
    export AUTOLOOP_ENABLED=1
    export STRATEGIES_ENABLED=1
    export AI_ORCHESTRATOR_ENABLED=1
    export AUTO_EVOLUTION_ENABLED=1
    export OPTIONS_ENABLED=1
    
    # Start with automatic restart on crash
    while true; do
        echo "[$(date)] Starting backend server..."
        node minimal_server.js >> "$BASE_DIR/logs/backend.log" 2>&1
        echo "[$(date)] Backend crashed, restarting in 5 seconds..."
        sleep 5
    done &
    
    BACKEND_PID=$!
    echo "✅ Backend started (PID: $BACKEND_PID)"
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting frontend..."
    cd "$BASE_DIR/new-trading-dashboard"
    
    # Start with automatic restart on crash
    while true; do
        echo "[$(date)] Starting frontend server..."
        npm run dev -- --port 3003 --strictPort >> "$BASE_DIR/logs/frontend.log" 2>&1
        echo "[$(date)] Frontend crashed, restarting in 5 seconds..."
        sleep 5
    done &
    
    FRONTEND_PID=$!
    echo "✅ Frontend started (PID: $FRONTEND_PID)"
}

# Function to monitor and restart if needed
monitor_services() {
    while true; do
        # Check backend
        if ! is_running "node minimal_server"; then
            echo "⚠️  Backend is down, restarting..."
            start_backend
        fi
        
        # Check frontend
        if ! is_running "vite.*3003"; then
            echo "⚠️  Frontend is down, restarting..."
            start_frontend
        fi
        
        # Sleep for 30 seconds before next check
        sleep 30
    done
}

# Trap to handle script termination
cleanup() {
    echo "🛑 Shutting down Benbot..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    pkill -f "node minimal_server" 2>/dev/null
    pkill -f "vite.*3003" 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM

# Main execution
echo "📋 Configuration:"
echo "  Backend: http://localhost:4000"
echo "  Frontend: http://localhost:3003"
echo "  Logs: $BASE_DIR/logs/"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Kill any existing processes
pkill -f "node minimal_server" 2>/dev/null
pkill -f "vite.*3003" 2>/dev/null
sleep 2

# Start services
start_backend
sleep 5
start_frontend
sleep 5

# Start monitoring
echo "👀 Monitoring services..."
monitor_services
