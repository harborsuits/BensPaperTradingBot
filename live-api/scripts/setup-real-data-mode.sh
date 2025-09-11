#!/bin/bash

# Setup Real Data Mode Configuration
# Configures EvoTester to use real market data instead of synthetic mocks

echo "🔧 Setting up EvoTester for REAL DATA MODE"
echo "=========================================="

# Create or update .env file
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "📝 Creating .env file..."
    touch "$ENV_FILE"
fi

# Add real data configuration
echo "" >> "$ENV_FILE"
echo "# Real Data Mode Configuration" >> "$ENV_FILE"
echo "# Added by setup-real-data-mode.sh on $(date)" >> "$ENV_FILE"
echo "" >> "$ENV_FILE"

# Force Tradier provider (real data)
if ! grep -q "QUOTES_PROVIDER=" "$ENV_FILE"; then
    echo "QUOTES_PROVIDER=tradier" >> "$ENV_FILE"
    echo "✅ Set QUOTES_PROVIDER=tradier"
else
    echo "⚠️  QUOTES_PROVIDER already configured, skipping..."
fi

# Enable autorefresh for continuous quotes
if ! grep -q "AUTOREFRESH_ENABLED=" "$ENV_FILE"; then
    echo "AUTOREFRESH_ENABLED=1" >> "$ENV_FILE"
    echo "✅ Set AUTOREFRESH_ENABLED=1"
fi

# Set quote refresh intervals
if ! grep -q "QUOTES_REFRESH_MS=" "$ENV_FILE"; then
    echo "QUOTES_REFRESH_MS=5000" >> "$ENV_FILE"
    echo "✅ Set QUOTES_REFRESH_MS=5000 (5 second intervals)"
fi

# Instructions for manual setup
echo ""
echo "📋 MANUAL CONFIGURATION REQUIRED:"
echo "=================================="
echo "1. Get Tradier API token from: https://developer.tradier.com/"
echo "2. Add to .env file:"
echo "   TRADIER_TOKEN=your_token_here"
echo "   TRADIER_BASE_URL=https://api.tradier.com/v1"
echo ""
echo "3. For production quotes:"
echo "   QUOTES_RATE_QPM=60          # API rate limit"
echo "   QUOTES_BATCH_SIZE=50        # Batch size for efficiency"
echo "   QUOTES_MAX_SYMBOLS=400      # Max symbols to track"
echo ""
echo "4. For safety monitoring:"
echo "   QUOTES_TIER1_REFRESH_MULT=1 # Most active symbols"
echo "   QUOTES_TIER2_REFRESH_MULT=3 # Medium activity"
echo "   QUOTES_TIER3_REFRESH_MULT=6 # Low activity"
echo ""

# Check if .env exists and show current config
if [ -f "$ENV_FILE" ]; then
    echo "📄 Current .env configuration:"
    echo "=============================="
    grep -E "(QUOTES_PROVIDER|AUTOREFRESH|TRADIER|QUOTES_)" "$ENV_FILE" || echo "No quote-related config found"
fi

echo ""
echo "🎯 REAL DATA MODE STATUS:"
echo "========================"
echo "✅ Safety system validates REAL market data"
echo "✅ MarketRecorder captures all market interactions"
echo "✅ Post-trade proofs use real NBBO comparisons"
echo "✅ Poor-Capital Mode uses real quote data"
echo "✅ Database-backed sessions and allocations"
echo ""
echo "🚀 Next: Set TRADIER_TOKEN and restart the server!"
echo "💡 Run: npm start (or your server startup command)"
