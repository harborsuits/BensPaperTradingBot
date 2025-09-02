#!/bin/bash
# Simple Coinbase API connectivity test using curl

echo "🔍 Testing Coinbase API connectivity with curl"
echo "---------------------------------------------"

# Test public endpoint (no auth required)
echo "Testing public endpoint..."
curl -s -X GET "https://api.coinbase.com/v2/currencies" | head -20

echo -e "\n\nTesting Coinbase Pro/Advanced public endpoint..."
curl -s -X GET "https://api.exchange.coinbase.com/products" | head -20

echo -e "\n\nNote: If you see valid JSON responses above, the public API endpoints are accessible."
echo "For authenticated endpoints, you'll need to use the read-only test script with your credentials."
echo "The script will use the credentials already stored in your configuration."
