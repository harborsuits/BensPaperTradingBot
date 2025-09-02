#!/bin/bash

STRATEGY=${1:-"sma_crossover"}
SYMBOL=${2:-"AAPL"}
START_DATE=${3:-"2022-01-01"}
END_DATE=${4:-"2023-01-01"}
CAPITAL=${5:-"10000"}

echo "Running backtest with the following parameters:"
echo "Strategy: $STRATEGY"
echo "Symbol: $SYMBOL"
echo "Date Range: $START_DATE to $END_DATE"
echo "Initial Capital: $CAPITAL"
echo ""

# Create a temporary JSON config for this run
cat > temp_config.json << EOF
{
  "strategy": "$STRATEGY",
  "symbol": "$SYMBOL",
  "start_date": "$START_DATE",
  "end_date": "$END_DATE",
  "initial_capital": $CAPITAL
}
EOF

# Run the backtest with this config
python run_backtest.py --config temp_config.json

# Clean up
rm temp_config.json

echo ""
echo "Backtest complete! Check the results directory for output." 