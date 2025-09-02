#!/bin/bash
# BensBot Startup Script
# This script starts BensBot with proper setup and monitoring

# Set script to exit on errors
set -e

# Configuration
LOG_DIR="logs"
DATA_DIR="data"
CONFIG_DIR="config"
MARKET_REGIME_DIR="data/market_regime"
TRADING_MODE="paper"  # Options: paper, live

# Create required directories
mkdir -p "$LOG_DIR" "$DATA_DIR" "$CONFIG_DIR" "$MARKET_REGIME_DIR"

# Echo with timestamp
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if the market regime data is available
check_market_regime_data() {
  log "Checking market regime data..."
  if [ ! -f "$MARKET_REGIME_DIR/data_report.json" ]; then
    log "Market regime data not found. Collecting initial data..."
    python trading_bot/scripts/regime_data_collector.py \
      --symbols SPY,QQQ,AAPL,MSFT,GOOGL,AMZN \
      --timeframes 1d,1h \
      --output-dir "$MARKET_REGIME_DIR"
  else
    log "Market regime data found!"
  fi
}

# Check if parameters are optimized
check_parameters() {
  log "Checking strategy parameters..."
  PARAMS_DIR="$MARKET_REGIME_DIR/parameters"
  mkdir -p "$PARAMS_DIR"
  
  # Check if we have parameter files for key strategies
  for strategy in trend_following mean_reversion breakout; do
    if [ ! -f "$PARAMS_DIR/${strategy}_parameters.json" ]; then
      log "Optimizing parameters for $strategy strategy..."
      python trading_bot/scripts/regime_parameter_optimizer.py \
        --strategy "$strategy" \
        --data-dir "$MARKET_REGIME_DIR" \
        --output-dir "$PARAMS_DIR"
    else
      log "Parameters for $strategy strategy already optimized"
    fi
  done
}

# Start the trading bot
start_bot() {
  log "Starting BensBot in $TRADING_MODE mode..."
  
  # Set environment variables if necessary
  # export BENBOT_LOG_LEVEL=DEBUG
  
  # Start the bot
  if [ "$TRADING_MODE" = "live" ]; then
    python -m trading_bot.run_bot --live
  else
    python -m trading_bot.run_bot --paper
  fi
}

# Main execution
log "Starting BensBot preparation..."

# Step 1: Check market regime data
check_market_regime_data

# Step 2: Check strategy parameters
check_parameters

# Step 3: Start the trading bot
start_bot

log "BensBot startup script completed"
