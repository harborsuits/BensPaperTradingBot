#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingView Integration Example - Demonstrates how to use TradingView with the trading system.
"""

import os
import sys
import json
import logging
import asyncio
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import trading bot components
from trading_bot.data.repository import MarketDataRepository
from trading_bot.data.models import MarketData, DataSource, TimeFrame
from trading_bot.data.asset_indicators import AssetIndicatorSuite, AssetType, AssetIndicatorConfig
from trading_bot.strategy.strategy_rotator import StrategyRotator
from trading_bot.strategies.stocks.momentum.momentum_strategy import MomentumStrategy
from trading_bot.strategies.stocks.trend.trend_following_strategy import TrendFollowingStrategy
from trading_bot.strategies.stocks.mean_reversion.mean_reversion_strategy import MeanReversionStrategy
from trading_bot.integrations.tradingview_integration import TradingViewIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("TradingViewExample")

def signal_callback(combined_signal: float, signals: Dict[str, float]):
    """
    Callback function for strategy signals.
    
    Args:
        combined_signal: Combined signal from all strategies
        signals: Individual strategy signals
    """
    logger.info(f"Combined signal: {combined_signal:.4f}")
    for strategy, signal in signals.items():
        logger.info(f"  {strategy}: {signal:.4f}")

def simulate_tradingview_alert(webhook_port: int, data: Dict[str, Any]):
    """
    Simulate a TradingView alert by sending data to the webhook.
    
    Args:
        webhook_port: Port of the webhook server
        data: Alert data to send
    """
    import requests
    import json
    
    url = f"http://localhost:{webhook_port}/tradingview"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        logger.info(f"Sent simulated alert: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Error response: {response.text}")
    except Exception as e:
        logger.error(f"Error sending simulated alert: {str(e)}")

def main():
    """Main function."""
    logger.info("Starting TradingView integration example")
    
    # 1. Create MarketDataRepository
    data_repository = MarketDataRepository()
    logger.info("Created MarketDataRepository")
    
    # 2. Create AssetIndicatorSuite for different asset types
    # Crypto indicator suite
    crypto_config = AssetIndicatorConfig(
        asset_type=AssetType.CRYPTO,
        enable_adaptive_parameters=True,
        tradingview_priority=True
    )
    crypto_indicators = AssetIndicatorSuite(crypto_config)
    
    # Equity indicator suite
    equity_config = AssetIndicatorConfig(
        asset_type=AssetType.EQUITY,
        is_24h_market=False,
        enable_adaptive_parameters=True,
        tradingview_priority=True
    )
    equity_indicators = AssetIndicatorSuite(equity_config)
    
    logger.info("Created AssetIndicatorSuites for crypto and equities")
    
    # 3. Create strategies for the StrategyRotator
    strategies = [
        MomentumStrategy("MomentumStrategy", {"fast_period": 5, "slow_period": 20}),
        TrendFollowingStrategy("TrendFollowingStrategy", {"short_ma_period": 10, "long_ma_period": 30}),
        MeanReversionStrategy("MeanReversionStrategy", {"period": 20, "std_dev_factor": 2.0})
    ]
    
    # 4. Create StrategyRotator
    strategy_rotator = StrategyRotator(strategies=strategies)
    
    # Register signal callback
    strategy_rotator.register_signal_handler(signal_callback)
    
    logger.info("Created StrategyRotator with strategies")
    
    # 5. Create TradingView integration
    webhook_port = 5000
    tradingview_integration = TradingViewIntegration(
        data_repository=data_repository,
        strategy_rotator=strategy_rotator,
        indicator_suite=crypto_indicators,  # Using crypto as default
        port=webhook_port
    )
    
    logger.info("Created TradingView integration")
    
    # 6. Start the integration
    tradingview_integration.start()
    logger.info(f"Started TradingView webhook on port {webhook_port}")
    
    # 7. Simulate TradingView alerts for different asset types
    # Wait for webhook server to start
    time.sleep(2)
    
    logger.info("Simulating TradingView alerts...")
    
    # Simulate crypto alert
    btc_alert = {
        "symbol": "BTCUSDT",
        "asset_type": "crypto",
        "timestamp": datetime.now().isoformat(),
        "open": 40000,
        "high": 41000,
        "low": 39500,
        "close": 40500,
        "volume": 1500,
        "indicators": {
            "rsi": 65.5,
            "macd": 250.5,
            "macd_signal": 200.2,
            "macd_hist": 50.3,
            "bb_upper": 42000,
            "bb_middle": 40000,
            "bb_lower": 38000
        }
    }
    simulate_tradingview_alert(webhook_port, btc_alert)
    time.sleep(1)
    
    # Simulate equity alert
    spy_alert = {
        "symbol": "SPY",
        "asset_type": "stock",
        "timestamp": datetime.now().isoformat(),
        "open": 450.25,
        "high": 453.50,
        "low": 449.75,
        "close": 452.80,
        "volume": 75000000,
        "indicators": {
            "rsi": 58.2,
            "macd": 1.25,
            "macd_signal": 0.85,
            "macd_hist": 0.40,
            "bb_upper": 455.00,
            "bb_middle": 450.00,
            "bb_lower": 445.00,
            "vwap": 451.50
        }
    }
    simulate_tradingview_alert(webhook_port, spy_alert)
    time.sleep(1)
    
    # Simulate forex alert
    forex_alert = {
        "symbol": "EURUSD",
        "asset_type": "forex",
        "timestamp": datetime.now().isoformat(),
        "open": 1.0950,
        "high": 1.0975,
        "low": 1.0940,
        "close": 1.0965,
        "volume": 125000,
        "indicators": {
            "rsi": 48.5,
            "macd": -0.0015,
            "macd_signal": -0.0010,
            "macd_hist": -0.0005,
            "atr": 0.0045
        }
    }
    simulate_tradingview_alert(webhook_port, forex_alert)
    
    # 8. Retrieve and display the processed data
    time.sleep(2)  # Wait for processing
    
    logger.info("Retrieving processed data...")
    
    # Check if we have data for BTC
    btc_data = data_repository.get_latest("BTCUSDT", lookback_periods=1)
    if btc_data:
        df_btc = data_repository.to_dataframe(btc_data)
        logger.info(f"BTC data: {df_btc.shape[0]} rows, {df_btc.shape[1]} columns")
        logger.info(f"BTC columns: {list(df_btc.columns)}")
    
    # Check if we have data for SPY
    spy_data = data_repository.get_latest("SPY", lookback_periods=1)
    if spy_data:
        df_spy = data_repository.to_dataframe(spy_data)
        logger.info(f"SPY data: {df_spy.shape[0]} rows, {df_spy.shape[1]} columns")
        logger.info(f"SPY columns: {list(df_spy.columns)}")
    
    # 9. Get latest indicators from the repository
    btc_indicators = data_repository.get_latest_indicators("BTCUSDT")
    if btc_indicators:
        logger.info(f"BTC indicators: {btc_indicators}")
    
    spy_indicators = data_repository.get_latest_indicators("SPY")
    if spy_indicators:
        logger.info(f"SPY indicators: {spy_indicators}")
    
    logger.info("Example complete. Press CTRL+C to exit.")
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        tradingview_integration.stop()

if __name__ == "__main__":
    main() 