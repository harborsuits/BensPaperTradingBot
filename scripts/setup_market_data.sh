#!/bin/bash

# Setup script for market data integration

# Create .env file for Alpaca API keys
cat > .env << EOL
ALPACA_KEY_ID=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_IS_PAPER=true
EOL

echo "Created .env file for Alpaca API keys"
echo "Please edit .env and add your Alpaca API keys"

# Create .env.local file for the frontend
mkdir -p new-trading-dashboard
cat > new-trading-dashboard/.env.local << EOL
VITE_API_URL=http://localhost:8000
VITE_USE_MOCK=false
EOL

echo "Created .env.local file for the frontend"

# Install required Python packages
pip install httpx aiolimiter cachetools

echo "Installed required Python packages"
echo ""
echo "Setup complete! Next steps:"
echo "1. Edit .env to add your Alpaca API keys"
echo "2. Start the backend: python -m uvicorn trading_bot.api.app:app --reload --port 8000"
echo "3. Start the frontend: cd new-trading-dashboard && npm run dev"
