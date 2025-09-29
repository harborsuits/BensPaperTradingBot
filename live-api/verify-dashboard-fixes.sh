#!/bin/bash

# Verification script for dashboard fixes
# This ensures critical fixes haven't been reverted

echo "🔍 Verifying dashboard fixes in minimal_server.js..."

ERRORS=0

# Check 1: Strategy API fix
if grep -q "Object.values(allStrategies || {}).forEach" minimal_server.js; then
    echo "❌ ERROR: Strategy API fix has been reverted!"
    echo "   Found: Object.values(allStrategies || {}).forEach"
    echo "   Should be: (allStrategies || []).forEach"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ Strategy API fix is intact"
fi

# Check 2: Position transformation
if ! grep -q "quantity: pos.qty || 0" minimal_server.js; then
    echo "❌ ERROR: Position data transformation is missing!"
    echo "   Frontend expects 'quantity' field, not 'qty'"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ Position data transformation is intact"
fi

# Check 3: Mock quotes provider
if ! grep -q "const mockQuotes = symbols.map" minimal_server.js; then
    echo "❌ ERROR: Mock quotes provider is missing!"
    echo "   This provides quotes when no Tradier token is configured"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ Mock quotes provider is intact"
fi

# Check 4: Rate limiter fix
if grep -q "'/api/quotes': new TokenBucketLimiter(5, 10)" minimal_server.js; then
    echo "❌ ERROR: Rate limiter fix has been reverted!"
    echo "   Found: TokenBucketLimiter(5, 10)"
    echo "   Should be: TokenBucketLimiter(20, 50)"
    ERRORS=$((ERRORS + 1))
else
    echo "✅ Rate limiter fix is intact"
fi

# Check 5: Strategies enabled on startup
if ! grep -q "STRATEGIES_ENABLED === '1'" minimal_server.js; then
    echo "⚠️  WARNING: Strategies auto-enable check not found"
    echo "   Make sure to start with STRATEGIES_ENABLED=1"
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✅ All dashboard fixes are intact!"
    echo ""
    echo "To start the server properly:"
    echo "STRATEGIES_ENABLED=1 AI_ORCHESTRATOR_ENABLED=1 OPTIONS_ENABLED=1 node minimal_server.js"
else
    echo "❌ Found $ERRORS issues! Dashboard will not work correctly."
    echo ""
    echo "To restore from backup:"
    echo "cp minimal_server.js.fixed-backup minimal_server.js"
fi

exit $ERRORS
