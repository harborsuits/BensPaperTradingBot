#!/bin/bash

# Script to add a new trade to the running dashboard
# Usage: ./add_trade.sh SYMBOL ACTION QUANTITY PRICE [STRATEGY]
# Example: ./add_trade.sh AAPL BUY 25 180.75 momentum

if [ $# -lt 4 ]; then
    echo "Usage: $0 SYMBOL ACTION QUANTITY PRICE [STRATEGY]"
    echo "Example: $0 AAPL BUY 25 180.75 momentum"
    exit 1
fi

SYMBOL=$1
ACTION=$2
QUANTITY=$3
PRICE=$4
STRATEGY=${5:-manual}  # Default to 'manual' if not provided

echo "Adding trade: $ACTION $QUANTITY shares of $SYMBOL @ \$$PRICE ($STRATEGY)..."

curl -X POST \
    -H "Content-Type: application/json" \
    -d "{\"symbol\":\"$SYMBOL\",\"action\":\"$ACTION\",\"quantity\":$QUANTITY,\"price\":$PRICE,\"strategy\":\"$STRATEGY\"}" \
    http://127.0.0.1:3000/api/add_trade

echo ""
echo "Done! Check the dashboard to see the new trade." 