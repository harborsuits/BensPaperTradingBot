#!/bin/bash

# AI Orchestrator Startup Script
# Starts the autonomous AI brain system for trading strategy management

echo "🤖 Starting AI Orchestrator System..."
echo "====================================="

# Set working directory
cd "$(dirname "$0")"

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if Python is available (for EvoTester integration)
if ! command -v python3 &> /dev/null; then
    echo "⚠️  Python3 not found. EvoTester integration will be simulated."
fi

# Start the live API server with AI orchestrator
echo "🚀 Starting Live API Server with AI Orchestrator..."
cd live-api

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start the server
echo "🔄 Starting server on port 4000..."
npm start &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Check if server is running
if curl -s http://localhost:4000/api/live/status > /dev/null; then
    echo "✅ Live API Server started successfully"
    echo "🔗 API available at: http://localhost:4000"
    echo ""
    echo "🤖 AI Orchestrator Status:"
    echo "   - Policy: Autonomous orchestration enabled"
    echo "   - Triggers: Capacity, Decay, Regime, Event, Drift, Novelty"
    echo "   - Cycle: Every 15 minutes during market hours"
    echo "   - Endpoints:"
    echo "     * GET  /api/ai/status      - AI orchestrator status"
    echo "     * GET  /api/ai/context     - Market context & roster"
    echo "     * GET  /api/ai/policy      - Current AI policy"
    echo "     * POST /api/ai/trigger-cycle - Manual cycle trigger"
    echo ""
    echo "🎛️  UI Dashboard:"
    echo "   - Main Dashboard: http://localhost:3003"
    echo "   - AI Status Card shows autonomous decisions"
    echo "   - Tournament ladder shows strategy progression"
    echo ""
    echo "📊 Key Features:"
    echo "   ✓ Autonomous strategy spawning based on market regime"
    echo "   ✓ Tournament progression R1→R2→R3→Live"
    echo "   ✓ Performance-based promotions/demotions"
    echo "   ✓ Dynamic gene bounds by market conditions"
    echo "   ✓ Capacity and budget management"
    echo "   ✓ Drift detection and reseeding"
    echo "   ✓ Event-driven strategy spawning"
    echo ""
    echo "⚡ System is now running autonomously!"
    echo "   Press Ctrl+C to stop all services"
else
    echo "❌ Failed to start Live API Server"
    exit 1
fi

# Wait for interrupt
trap 'echo ""; echo "🛑 Shutting down AI Orchestrator..."; kill $SERVER_PID 2>/dev/null; exit 0' INT

# Keep script running
wait $SERVER_PID
