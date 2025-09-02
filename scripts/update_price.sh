#!/bin/bash

# Script to update a stock price in the running dashboard
# Usage: ./update_price.sh SYMBOL PRICE
# Example: ./update_price.sh AAPL 175.50

if [ $# -ne 2 ]; then
    echo "Usage: $0 SYMBOL PRICE"
    echo "Example: $0 AAPL 175.50"
    exit 1
fi

SYMBOL=$1
PRICE=$2

echo "Updating $SYMBOL price to \$$PRICE..."

curl -X POST \
    -H "Content-Type: application/json" \
    -d "{\"symbol\":\"$SYMBOL\",\"price\":$PRICE}" \
    http://127.0.0.1:8080/api/update_position

echo ""
echo "Done! Check the dashboard to see the updated price." 