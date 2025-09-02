#!/bin/bash
# Initialize testing environment for the trading bot

echo "Setting up testing environment..."

# Create required directories
mkdir -p ../logs
mkdir -p ../data
mkdir -p ../reports

# Check if MongoDB is installed and running
if command -v mongod &> /dev/null; then
    echo "MongoDB is installed"
    if pgrep -x mongod > /dev/null; then
        echo "MongoDB is running"
    else
        echo "MongoDB is not running, starting..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            brew services start mongodb-community || echo "Failed to start MongoDB, please start it manually"
        else
            # Linux
            sudo systemctl start mongod || echo "Failed to start MongoDB, please start it manually"
        fi
    fi
else
    echo "MongoDB is not installed, please install it for database testing"
fi

# Check if Redis is installed and running
if command -v redis-cli &> /dev/null; then
    echo "Redis is installed"
    if redis-cli ping &> /dev/null; then
        echo "Redis is running"
    else
        echo "Redis is not running, starting..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            brew services start redis || echo "Failed to start Redis, please start it manually"
        else
            # Linux
            sudo systemctl start redis || echo "Failed to start Redis, please start it manually"
        fi
    fi
else
    echo "Redis is not installed, please install it for caching testing"
fi

# Install Python dependencies
echo "Installing test dependencies..."
pip install -r requirements.txt

# Set up environment variables
echo "Setting up environment variables..."
cat << EOF > .env.test
API_BASE_URL=http://localhost:5000
API_TOKEN=6165f902-b7a3-408c-9512-4e554225d825
TEST_MODE=true
PAPER_TRADING=true
ALPACA_API_KEY=6165f902-b7a3-408c-9512-4e554225d825
ALPACA_SECRET=your_alpaca_secret_here
ALPACA_URL=https://paper-api.alpaca.markets
TRADIER_API_KEY=4wI27PVe8TTNSmgNp4SHLGXhOS41
MONGO_URI=mongodb://localhost:27017
MONGO_DB=trading_bot_test
REDIS_URL=redis://localhost:6379
EOF

echo "Environment setup complete!"
echo "To load test environment variables, run:"
echo "  source tests/.env.test" 