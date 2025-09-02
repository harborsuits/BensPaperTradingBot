#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingView Webhook - Receives and processes real-time alerts and indicator data from TradingView.
"""

import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from flask import Flask, request, jsonify

# Setup logging
logger = logging.getLogger("TradingViewWebhook")

class TradingViewWebhook:
    """
    Webhook server that receives and processes TradingView alerts.
    Integrates with the MarketDataRepository to update indicator data.
    """
    
    def __init__(self, 
                 port: int = 5000, 
                 data_repository = None,
                 strategy_rotator = None,
                 callback: Optional[Callable] = None,
                 auth_token: Optional[str] = None):
        """
        Initialize the TradingView webhook.
        
        Args:
            port: Port to run the webhook server on
            data_repository: MarketDataRepository instance for storing data
            strategy_rotator: StrategyRotator instance for direct signal updates
            callback: Optional callback function to process alerts
            auth_token: Optional authentication token for webhook security
        """
        self.port = port
        self.data_repository = data_repository
        self.strategy_rotator = strategy_rotator
        self.callback = callback
        self.auth_token = auth_token
        self.app = Flask(__name__)
        self.server_thread = None
        self.running = False
        
        # Register routes
        self.setup_routes()
        
        logger.info(f"TradingView webhook initialized on port {port}")
    
    def setup_routes(self):
        """Set up Flask routes for the webhook."""
        @self.app.route('/tradingview', methods=['POST'])
        def tradingview_webhook():
            """Endpoint to receive TradingView alerts."""
            # Verify auth token if configured
            if self.auth_token:
                auth_header = request.headers.get('Authorization')
                if not auth_header or auth_header != f"Bearer {self.auth_token}":
                    logger.warning("Unauthorized request to webhook")
                    return jsonify({"error": "Unauthorized"}), 401
            
            # Parse JSON data
            try:
                data = request.json
                
                # Validate payload
                if not data or 'symbol' not in data:
                    logger.warning("Invalid data received: missing required fields")
                    return jsonify({"error": "Invalid data format"}), 400
                
                # Process the data
                self.process_alert(data)
                
                return jsonify({"status": "success"}), 200
                
            except Exception as e:
                logger.error(f"Error processing webhook data: {str(e)}")
                return jsonify({"error": f"Server error: {str(e)}"}), 500
    
    def process_alert(self, data: Dict[str, Any]):
        """
        Process a TradingView alert.
        
        Args:
            data: Alert data from TradingView
        """
        try:
            logger.info(f"Received alert for {data.get('symbol')}")
            
            # Extract key fields
            symbol = data.get('symbol')
            asset_type = data.get('asset_type', 'unknown')
            timestamp = data.get('timestamp', datetime.now().isoformat())
            indicators = data.get('indicators', {})
            
            # Log the received data
            logger.debug(f"Alert data: {json.dumps(data)}")
            
            # Update MarketDataRepository if available
            if self.data_repository:
                self.update_data_repository(data)
            
            # Update StrategyRotator if available
            if self.strategy_rotator:
                self.update_strategy_rotator(data)
            
            # Call custom callback if provided
            if self.callback:
                self.callback(data)
                
        except Exception as e:
            logger.error(f"Error in process_alert: {str(e)}")
    
    def update_data_repository(self, data: Dict[str, Any]):
        """
        Update the MarketDataRepository with the received data.
        
        Args:
            data: Alert data from TradingView
        """
        try:
            symbol = data.get('symbol')
            
            # Check if OHLCV data is included
            ohlcv_data = {}
            for field in ['open', 'high', 'low', 'close', 'volume']:
                if field in data:
                    ohlcv_data[field] = data[field]
            
            # If we have all OHLCV data, update market data
            if len(ohlcv_data) == 5:
                timestamp = data.get('timestamp')
                if timestamp:
                    try:
                        timestamp_dt = datetime.fromisoformat(timestamp)
                    except:
                        timestamp_dt = datetime.now()
                else:
                    timestamp_dt = datetime.now()
                
                # Create market data object with the OHLCV data
                from trading_bot.data.models import MarketData, DataSource
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=timestamp_dt,
                    open=float(ohlcv_data['open']),
                    high=float(ohlcv_data['high']),
                    low=float(ohlcv_data['low']),
                    close=float(ohlcv_data['close']),
                    volume=float(ohlcv_data['volume']),
                    source=DataSource.TRADINGVIEW if hasattr(DataSource, 'TRADINGVIEW') else DataSource.UNKNOWN
                )
                
                # Add indicators to additional_data if present
                if 'indicators' in data:
                    market_data.additional_data['indicators'] = data['indicators']
                
                # Update the repository
                self.data_repository.add_market_data([market_data])
                logger.info(f"Updated market data for {symbol}")
            
            # If we only have indicator data, update the indicators
            elif 'indicators' in data:
                self.data_repository.update_indicators(symbol, data['indicators'])
                logger.info(f"Updated indicators for {symbol}")
                
        except Exception as e:
            logger.error(f"Error updating data repository: {str(e)}")
    
    def update_strategy_rotator(self, data: Dict[str, Any]):
        """
        Update the StrategyRotator with the received data.
        
        Args:
            data: Alert data from TradingView
        """
        try:
            # Convert the alert data to the format expected by the strategy rotator
            market_data = {
                'symbol': data.get('symbol'),
                'timestamp': data.get('timestamp', datetime.now().isoformat())
            }
            
            # Add price if available
            if 'close' in data:
                market_data['price'] = data['close']
            
            # Add indicators if available
            if 'indicators' in data:
                market_data.update(data['indicators'])
            
            # If the strategy rotator has a method to process TradingView data, call it
            if hasattr(self.strategy_rotator, 'process_tradingview_data'):
                self.strategy_rotator.process_tradingview_data(market_data)
                logger.info(f"Updated strategy rotator with data for {data.get('symbol')}")
                
        except Exception as e:
            logger.error(f"Error updating strategy rotator: {str(e)}")
    
    def start(self):
        """Start the webhook server in a separate thread."""
        if self.running:
            logger.warning("Webhook server is already running")
            return
        
        def run_server():
            self.app.run(host='0.0.0.0', port=self.port, debug=False)
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        self.running = True
        
        logger.info(f"Webhook server started on port {self.port}")
    
    def stop(self):
        """Stop the webhook server."""
        if not self.running:
            logger.warning("Webhook server is not running")
            return
        
        # There's no clean way to stop a Flask server in a thread
        # Typically this would be handled by a more robust server setup
        # For now, just mark as not running
        self.running = False
        logger.info("Webhook server stop requested (will terminate when thread completes)")


# Example usage when run directly
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create webhook with no dependencies for testing
    webhook = TradingViewWebhook(port=5000)
    
    # Start the webhook
    webhook.start()
    
    print("TradingView webhook server is running. Press CTRL+C to stop.")
    
    try:
        import time
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        webhook.stop() 