#!/bin/bash

# BenBot Live API Server Startup Script

echo "Starting BenBot Live API Server..."

# First, verify all dashboard fixes are intact
echo "Verifying dashboard fixes..."
./verify-dashboard-fixes.sh
if [ $? -ne 0 ]; then
    echo "❌ Dashboard fixes have been reverted! Restoring from backup..."
    cp minimal_server.js.fixed-backup minimal_server.js
    echo "✅ Restored from backup"
fi

# Set environment variables
export STRATEGIES_ENABLED=1
export AI_ORCHESTRATOR_ENABLED=1
export OPTIONS_ENABLED=1
export FORCE_NO_MOCKS=false
export DISCONNECT_FEEDS=false
export PORT=4000

# Kill any existing process on port 4000
lsof -ti:4000 | xargs kill -9 2>/dev/null

# Start the server
node minimal_server.js
