#!/bin/bash

# BenBot Complete Startup Script
# This ensures everything runs correctly

echo "🚀 Starting BenBot System..."

# Kill any existing servers
echo "Cleaning up old processes..."
lsof -ti:4000 | xargs kill -9 2>/dev/null
lsof -ti:3003 | xargs kill -9 2>/dev/null

# Start API server
echo "Starting API server..."
cd /Users/bendickinson/Desktop/benbot/live-api
./start-server.sh &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API server..."
sleep 5

# Activate strategies
echo "Activating strategies..."
curl -s -X POST http://localhost:4000/api/strategies/activate-all > /dev/null

# Start dashboard
echo "Starting dashboard..."
cd /Users/bendickinson/Desktop/benbot/new-trading-dashboard
npm run dev &
UI_PID=$!

sleep 3

echo "✅ BenBot is ready!"
echo "🌐 Dashboard: http://localhost:3003"
echo "🔧 API: http://localhost:4000"
echo ""
echo "To stop: kill $API_PID $UI_PID"
