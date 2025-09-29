#!/bin/bash
# BenBot Monday Trading Startup Script
# Run this before market open to start all systems

echo "🚀 Starting BenBot Trading System..."
echo "=================================="

# Kill any existing processes
echo "📍 Cleaning up old processes..."
pkill -f "python.*ai_scoring_service" 2>/dev/null
pkill -f "node.*server.js" 2>/dev/null
sleep 2

# Start Python AI Brain
echo "🧠 Starting AI Brain Service..."
cd /Users/bendickinson/Desktop/benbot
python ai_scoring_service.py > ai_scoring_monday.log 2>&1 &
AI_PID=$!
echo "   AI Brain started (PID: $AI_PID)"
sleep 3

# Check if AI Brain is responding
if curl -s http://localhost:8001/health | grep -q "ok"; then
    echo "   ✅ AI Brain responding"
else
    echo "   ⚠️  AI Brain not responding - check logs"
fi

# Start Backend with full configuration
echo "📊 Starting Trading Backend..."
cd /Users/bendickinson/Desktop/benbot/live-api
TRADIER_API_KEY="${TRADIER_API_KEY:-KU2iUnOZIUFre0wypgyOn8TgmGxI}" \
TRADIER_ACCOUNT_ID="${TRADIER_ACCOUNT_ID:-VA1201776}" \
PORT=4000 \
AUTOLOOP_ENABLED=1 \
STRATEGIES_ENABLED=1 \
AI_ORCHESTRATOR_ENABLED=1 \
AUTO_EVOLUTION_ENABLED=1 \
NODE_ENV=development \
node server.js > ../backend-monday.log 2>&1 &
BACKEND_PID=$!
echo "   Backend started (PID: $BACKEND_PID)"

# Wait for backend to initialize
echo "⏳ Waiting for services to initialize..."
sleep 15

# Check system health
echo "🔍 Checking system health..."
if curl -s http://localhost:4000/api/health | grep -q "true"; then
    echo "   ✅ Backend healthy"
    
    # Show AutoLoop status
    AUTOLOOP_STATUS=$(curl -s http://localhost:4000/api/autoloop/status | jq -r '.status')
    echo "   ✅ AutoLoop: $AUTOLOOP_STATUS"
    
    # Show strategy count
    STRATEGY_COUNT=$(curl -s http://localhost:4000/api/metrics | jq -r '.totalStrategies')
    echo "   ✅ Active Strategies: $STRATEGY_COUNT"
else
    echo "   ⚠️  Backend not healthy - check logs"
fi

# Start log monitoring
echo ""
echo "📋 Starting log monitor..."
echo "=================================="
tail -f /Users/bendickinson/Desktop/benbot/backend-monday.log | grep -E "Signal generated|Brain decision|Evolution|Trade executed|ERROR" &
TAIL_PID=$!

echo ""
echo "✅ BenBot Trading System Started!"
echo "=================================="
echo "PIDs: AI=$AI_PID, Backend=$BACKEND_PID, Logs=$TAIL_PID"
echo ""
echo "📍 Quick Commands:"
echo "   Check positions:  curl http://localhost:4000/api/paper/positions | jq ."
echo "   Force evolution:  curl -X POST http://localhost:4000/api/test/force-evolution"
echo "   Test signal:      curl -X POST http://localhost:4000/api/test/force-signal -d '{\"symbol\":\"AAPL\"}'"
echo "   Stop all:         pkill -f 'node.*server.js' && pkill -f 'python.*ai_scoring'"
echo ""
echo "🎯 Monitor the logs above for trading activity..."
