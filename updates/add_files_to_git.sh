#!/bin/bash

# Add the autonomous backtester files to git

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if files exist
if [ -f "trading_bot/backtesting/autonomous_backtester.py" ] && \
   [ -f "trading_bot/backtesting/data_integration.py" ] && \
   [ -f "trading_bot/backtesting/strategy_generator.py" ] && \
   [ -f "trading_bot/backtesting/ml_optimizer.py" ]; then
    
    echo "Adding autonomous backtester files to git..."
    
    # Add files to git
    git add trading_bot/backtesting/autonomous_backtester.py
    git add trading_bot/backtesting/data_integration.py
    git add trading_bot/backtesting/strategy_generator.py
    git add trading_bot/backtesting/ml_optimizer.py
    
    # Commit
    git commit -m "Add autonomous backtester with ML integration and news sentiment analysis"
    
    echo "Files added and committed successfully!"
else
    echo "Error: One or more autonomous backtester files not found."
    echo "Please make sure the following files exist:"
    echo "  - trading_bot/backtesting/autonomous_backtester.py"
    echo "  - trading_bot/backtesting/data_integration.py"
    echo "  - trading_bot/backtesting/strategy_generator.py"
    echo "  - trading_bot/backtesting/ml_optimizer.py"
    exit 1
fi 