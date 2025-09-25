#!/bin/bash
# Script to set ALL API keys for BenBot
# DO NOT COMMIT THIS FILE WITH KEYS!

echo "🔐 Setting up all API keys for BenBot..."

# Create .env file with all keys
cat > .env << 'EOF'
# BenBot Environment Variables
# DO NOT COMMIT THIS FILE!

# Tradier Paper Trading
TRADIER_TOKEN=KU2iUnOZIUFre0wypgyOn8TgmGxI
TRADIER_ACCOUNT_ID=VA1201776
TRADIER_BASE_URL=https://sandbox.tradier.com/v1
TRADIER_API_KEY=KU2iUnOZIUFre0wypgyOn8TgmGxI

# Alpaca Paper Trading (backup/alternative)
ALPACA_API_KEY=PKYBHCCT1DIMGZX6P64A
ALPACA_SECRET_KEY=ssidJ2cJU0EGBOhdHrXJd7HegoaPaAMQqs0AU2PO
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Market Data
POLYGON_API_KEY=pcIfIzF_AiAd2Ps0ifLTXRtuA2BbBVtS

# News APIs - Multiple sources for cross-validation
NEWSDATA_API_KEY=pub_81036c20e73907398317875951d4569722f2a
GNEWS_API_KEY=00c755186577632fbf651fc38e39858b
MEDIASTACK_API_KEY=3ff958493e0f1d8cf9af5e8425c8f5a3
CURRENTS_API_KEY=O5_JjrWdlLN2v93iuKbhEhA9OSIYfChf4Cx9XE9xXgW1oYTC
NYTIMES_API_KEY=NosApZGLGvPusEz30Fk4lQban19z9PTo

# Primary news source (since we don't have Marketaux)
PRIMARY_NEWS_SOURCE=newsdata
MARKETAUX_API_KEY=

# System Settings
FORCE_NO_MOCKS=true
PAPER_MOCK_MODE=false

# Evolution and Learning
ENABLE_AUTO_EVOLUTION=true
ENABLE_MACRO_AWARENESS=true

# Python Brain Path
PYTHON_BRAIN_PATH=/Users/bendickinson/Desktop/benbot/python-brain
EOF

echo "✅ .env file created with all keys"

# Export all variables for current session
export $(grep -v '^#' .env | xargs)

echo "✅ Environment variables exported"

# Add to .gitignore if not already there
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
    echo "set_all_keys.sh" >> .gitignore
    echo "✅ Added .env and this script to .gitignore"
fi

echo ""
echo "🚀 All API keys configured!"
echo ""
echo "Testing connections..."
echo "-------------------"

# Test Tradier
echo -n "Tradier API: "
curl -s -H "Authorization: Bearer $TRADIER_TOKEN" \
     -H "Accept: application/json" \
     "$TRADIER_BASE_URL/user/profile" | jq -r '.profile.id // "❌ Failed"' | head -c 20 && echo "... ✅"

# Test news sources
echo -n "Newsdata.io: "
curl -s "https://newsdata.io/api/1/news?apikey=$NEWSDATA_API_KEY&q=AAPL&language=en" | jq -r '.status // "❌ Failed"'

echo ""
echo "Next steps:"
echo "1. Restart backend: pm2 restart benbot-backend --update-env"
echo "2. Check health: curl http://localhost:4000/api/health"
echo "3. Test news: curl http://localhost:4000/api/context/news"
