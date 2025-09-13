#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingView Integration - Integrates TradingView alerts with the trading system.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable

from trading_bot.integrations.tradingview_webhook import TradingViewWebhook
from trading_bot.data.repository import MarketDataRepository
from trading_bot.data.asset_indicators import AssetIndicatorSuite, AssetType
from trading_bot.strategy.strategy_rotator import StrategyRotator

# Setup logging
logger = logging.getLogger("TradingViewIntegration")

class TradingViewIntegration:
    """
    Integration component that connects TradingView alerts
    with the trading system components.
    """
    
    def __init__(self,
                data_repository: Optional[MarketDataRepository] = None,
                strategy_rotator: Optional[StrategyRotator] = None,
                indicator_suite: Optional[AssetIndicatorSuite] = None,
                port: int = 5000,
                auth_token: Optional[str] = None,
                config_path: Optional[str] = None):
        """
        Initialize the TradingView integration.
        
        Args:
            data_repository: MarketDataRepository instance
            strategy_rotator: StrategyRotator instance
            indicator_suite: AssetIndicatorSuite instance
            port: Port to run the webhook server on
            auth_token: Authentication token for webhook security
            config_path: Path to configuration file
        """
        self.data_repository = data_repository
        self.strategy_rotator = strategy_rotator
        self.indicator_suite = indicator_suite
        self.webhook = None
        self.webhook_port = port
        self.auth_token = auth_token
        
        # Asset type mapping
        self.asset_type_mapping = {
            'crypto': AssetType.CRYPTO,
            'stock': AssetType.EQUITY,
            'forex': AssetType.FOREX,
            'futures': AssetType.FUTURES,
            'index': AssetType.INDEX,
            'commodity': AssetType.COMMODITY
        }
        
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        # Initialize the webhook with callback
        self._initialize_webhook()
        
        logger.info("TradingView integration initialized")
    
    def _initialize_webhook(self):
        """Initialize the TradingView webhook."""
        self.webhook = TradingViewWebhook(
            port=self.webhook_port,
            data_repository=self.data_repository,
            strategy_rotator=self.strategy_rotator,
            callback=self.process_tradingview_alert,
            auth_token=self.auth_token
        )
    
    def process_tradingview_alert(self, data: Dict[str, Any]):
        """
        Process a TradingView alert.
        
        This is the main callback that handles alerts from TradingView.
        
        Args:
            data: Alert data from TradingView
        """
        try:
            logger.info(f"Processing TradingView alert for {data.get('symbol')}")
            
            # Extract key fields
            symbol = data.get('symbol')
            
            if not symbol:
                logger.warning("Alert missing symbol, cannot process")
                return
            
            # Determine asset type
            asset_type_str = data.get('asset_type', 'unknown').lower()
            asset_type = self.asset_type_mapping.get(asset_type_str, AssetType.UNKNOWN)
            
            # Extract indicator data if available
            indicators = data.get('indicators', {})
            
            # Update the asset indicator suite if available
            if self.indicator_suite:
                # Update TradingView indicators
                if indicators:
                    self.indicator_suite.update_tradingview_indicators(symbol, indicators)
                    logger.info(f"Updated TradingView indicators for {symbol}")
            
            # Additional processing can be added here
            # For example, custom logic for specific alert types
            if 'alert_type' in data:
                alert_type = data.get('alert_type')
                if alert_type == 'price_target':
                    self._handle_price_target_alert(data)
                elif alert_type == 'breakout':
                    self._handle_breakout_alert(data)
            
            logger.info(f"Successfully processed alert for {symbol}")
            
        except Exception as e:
            logger.error(f"Error processing TradingView alert: {str(e)}")
    
    def _handle_price_target_alert(self, data: Dict[str, Any]):
        """Handle price target alert."""
        symbol = data.get('symbol')
        price = data.get('price')
        target = data.get('target_price')
        
        logger.info(f"Price target alert for {symbol}: current {price}, target {target}")
        
        # Add custom handling logic here
    
    def _handle_breakout_alert(self, data: Dict[str, Any]):
        """Handle breakout alert."""
        symbol = data.get('symbol')
        price = data.get('price')
        level = data.get('breakout_level')
        direction = data.get('direction', 'unknown')
        
        logger.info(f"Breakout alert for {symbol}: {direction} breakout at {price}, level {level}")
        
        # Add custom handling logic here
    
    def start(self):
        """Start the TradingView integration."""
        if self.webhook:
            self.webhook.start()
            logger.info("TradingView integration started")
        else:
            logger.error("Webhook not initialized, cannot start")
    
    def stop(self):
        """Stop the TradingView integration."""
        if self.webhook:
            self.webhook.stop()
            logger.info("TradingView integration stopped")
        else:
            logger.warning("Webhook not initialized, nothing to stop")


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create instances of needed components
    data_repo = MarketDataRepository(use_mock=True)
    
    # Create the TradingView integration
    integration = TradingViewIntegration(
        data_repository=data_repo,
        port=5000
    )
    
    # Start the integration
    integration.start()
    
    print("TradingView integration is running. Press CTRL+C to stop.")
    
    try:
        import time
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        integration.stop() 