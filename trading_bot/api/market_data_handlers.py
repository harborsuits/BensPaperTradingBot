"""
WebSocket handlers for market data broadcasting.
These handlers manage real-time updates for market data, price updates,
risk metrics, and market signals.
"""

import asyncio
import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from pydantic import BaseModel

from trading_bot.api.websocket_handlers import ConnectionManager, WebSocketMessage

# Import trading components
from trading_bot.components.market_analyzer import MarketAnalyzer
from trading_bot.components.risk_manager import RiskManager
from trading_bot.components.trading_engine import TradingEngine

logger = logging.getLogger(__name__)

class MarketDataHandler:
    """Handler for market data WebSocket messages and updates"""
    
    def __init__(self, connection_manager: ConnectionManager, trading_engine=None, market_analyzer=None, risk_manager=None):
        self.connection_manager = connection_manager
        self.price_update_task = None
        self.risk_metrics_task = None
        self.market_signals_task = None
        
        # Store references to trading components
        self.trading_engine = trading_engine
        self.market_analyzer = market_analyzer
        self.risk_manager = risk_manager
        
        # Initialize with default values in case real data is not available
        self.latest_prices = {}
        self.risk_metrics = {
            'paper': [
                {'name': 'Portfolio VaR', 'value': 2.35, 'threshold': 5, 'maxThreshold': 8},
                {'name': 'Concentration', 'value': 28.4, 'threshold': 40, 'maxThreshold': 60},
                {'name': 'Correlation', 'value': 0.42, 'threshold': 0.6, 'maxThreshold': 0.8},
                {'name': 'Market Exposure', 'value': 65.8, 'threshold': 80, 'maxThreshold': 90},
            ],
            'live': [
                {'name': 'Portfolio VaR', 'value': 1.85, 'threshold': 5, 'maxThreshold': 8},
                {'name': 'Concentration', 'value': 22.7, 'threshold': 40, 'maxThreshold': 60},
                {'name': 'Correlation', 'value': 0.38, 'threshold': 0.6, 'maxThreshold': 0.8},
                {'name': 'Market Exposure', 'value': 48.2, 'threshold': 80, 'maxThreshold': 90},
            ]
        }
        
        # Default market signals for common tickers as fallback
        self.market_signals = {
            'AAPL': {'symbol': 'AAPL', 'sentiment': 'bullish', 'volume_trend': 'increasing', 
                    'momentum': 0.72, 'volatility': 0.38, 'market_regime': 'uptrend'},
            'MSFT': {'symbol': 'MSFT', 'sentiment': 'bullish', 'volume_trend': 'stable', 
                    'momentum': 0.68, 'volatility': 0.32, 'market_regime': 'uptrend'},
            'GOOG': {'symbol': 'GOOG', 'sentiment': 'neutral', 'volume_trend': 'decreasing', 
                    'momentum': 0.15, 'volatility': 0.45, 'market_regime': 'consolidation'},
            'AMZN': {'symbol': 'AMZN', 'sentiment': 'bullish', 'volume_trend': 'increasing', 
                    'momentum': 0.56, 'volatility': 0.41, 'market_regime': 'uptrend'},
            'TSLA': {'symbol': 'TSLA', 'sentiment': 'bearish', 'volume_trend': 'increasing', 
                    'momentum': -0.38, 'volatility': 0.82, 'market_regime': 'downtrend'},
            'META': {'symbol': 'META', 'sentiment': 'neutral', 'volume_trend': 'stable', 
                    'momentum': 0.12, 'volatility': 0.48, 'market_regime': 'consolidation'},
            'NVDA': {'symbol': 'NVDA', 'sentiment': 'bullish', 'volume_trend': 'increasing', 
                    'momentum': 0.78, 'volatility': 0.56, 'market_regime': 'uptrend'},
            'AMD': {'symbol': 'AMD', 'sentiment': 'bullish', 'volume_trend': 'increasing', 
                    'momentum': 0.65, 'volatility': 0.52, 'market_regime': 'uptrend'},
            'INTC': {'symbol': 'INTC', 'sentiment': 'bearish', 'volume_trend': 'decreasing', 
                    'momentum': -0.42, 'volatility': 0.38, 'market_regime': 'downtrend'},
            'IBM': {'symbol': 'IBM', 'sentiment': 'neutral', 'volume_trend': 'stable', 
                    'momentum': 0.05, 'volatility': 0.28, 'market_regime': 'consolidation'}
        }
        
        # Initialize prices (either real or mock)
        self._initialize_prices()

    def _initialize_prices(self):
        """Initialize prices, using real data if available, otherwise falling back to mock data"""
        try:
            # Try to get real prices from trading engine
            if self.trading_engine and hasattr(self.trading_engine, 'get_current_prices'):
                symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'IBM']
                real_prices = self.trading_engine.get_current_prices(symbols)
                
                if real_prices and isinstance(real_prices, dict) and len(real_prices) > 0:
                    logger.info(f"Initialized market data handler with real prices for {len(real_prices)} symbols")
                    self.latest_prices = real_prices
                    return
                else:
                    logger.warning("Failed to get real prices from trading engine, falling back to mock data")
            else:
                logger.warning("Trading engine not available or doesn't support price retrieval, using mock data")
        except Exception as e:
            logger.error(f"Error initializing real prices: {str(e)}. Using mock data instead.")
        
        # Fallback to mock prices if real data is not available
        self.latest_prices = {
            'AAPL': 175.50,
            'MSFT': 350.25,
            'GOOG': 138.90,
            'AMZN': 178.20,
            'TSLA': 175.35,
            'META': 470.80,
            'NVDA': 890.40,
            'AMD': 145.65,
            'INTC': 31.20,
            'IBM': 168.75
        }
        logger.info("Initialized market data handler with mock prices")
    
    async def start_price_updates(self):
        """Start the background task for periodic price updates"""
        if self.price_update_task is None or self.price_update_task.done():
            self.price_update_task = asyncio.create_task(self._send_periodic_price_updates())
            logger.info("Started price update background task")
    
    async def start_risk_metrics_updates(self):
        """Start the background task for periodic risk metrics updates"""
        if self.risk_metrics_task is None or self.risk_metrics_task.done():
            self.risk_metrics_task = asyncio.create_task(self._send_periodic_risk_metrics())
            logger.info("Started risk metrics update background task")
    
    async def start_market_signals_updates(self):
        """Start the background task for periodic market signals updates"""
        if self.market_signals_task is None or self.market_signals_task.done():
            self.market_signals_task = asyncio.create_task(self._send_periodic_market_signals())
            logger.info("Started market signals update background task")
    
    async def stop_all_updates(self):
        """Stop all background tasks"""
        tasks = [
            (self.price_update_task, "price update"),
            (self.risk_metrics_task, "risk metrics"), 
            (self.market_signals_task, "market signals")
        ]
        
        for task, name in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                logger.info(f"Stopped {name} background task")
    
    async def _update_prices(self):
        """Update prices using real data if available, otherwise generate mock changes"""
        try:
            # Try to get real prices from trading engine
            if self.trading_engine and hasattr(self.trading_engine, 'get_current_prices'):
                symbols = list(self.latest_prices.keys())
                real_prices = self.trading_engine.get_current_prices(symbols)
                
                if real_prices and isinstance(real_prices, dict) and len(real_prices) > 0:
                    # Update with real prices
                    for symbol, price in real_prices.items():
                        if symbol in self.latest_prices:
                            old_price = self.latest_prices[symbol]
                            self.latest_prices[symbol] = price
                            
                            # Calculate percentage change for logging
                            if old_price > 0:
                                change_pct = (price - old_price) / old_price
                                if abs(change_pct) > 0.005:  # Log significant changes
                                    logger.info(f"Real price update: {symbol} ${price:.2f} (change: {change_pct*100:.2f}%)")
                    
                    logger.debug(f"Updated prices with real data for {len(real_prices)} symbols")
                    return
        except Exception as e:
            logger.warning(f"Error updating real prices: {str(e)}. Falling back to mock updates.")
        
        # Fallback to mock price updates
        for symbol in self.latest_prices.keys():
            # Generate small random price changes (between -1% and 1%)
            change_pct = random.uniform(-0.01, 0.01)
            current_price = self.latest_prices[symbol]
            new_price = current_price * (1 + change_pct)
            
            # Update the price
            self.latest_prices[symbol] = round(new_price, 2)
            
            # Log every 10th update or large price changes
            if random.random() < 0.1 or abs(change_pct) > 0.005:
                logger.debug(f"Mock update: {symbol} ${new_price:.2f} (change: {change_pct*100:.2f}%)")
        
        logger.debug(f"Updated mock prices for {len(self.latest_prices)} symbols")

    async def _update_mock_risk_metrics(self):
        """Update mock risk metrics with small random changes"""
        for account in self.risk_metrics:
            for metric in self.risk_metrics[account]:
                # Generate small random changes
                change = random.uniform(-2, 2)
                
                # Update the metric values within bounds
                current = metric['value']
                if metric['name'] == 'Portfolio VaR':
                    # VaR should be between 0.5 and 10
                    metric['value'] = max(0.5, min(10, current + change * 0.1))
                elif metric['name'] == 'Concentration':
                    # Concentration should be between 5 and 100
                    metric['value'] = max(5, min(100, current + change))
                elif metric['name'] == 'Correlation':
                    # Correlation should be between 0 and 1
                    metric['value'] = max(0, min(1, current + change * 0.05))
                elif metric['name'] == 'Market Exposure':
                    # Market exposure should be between 0 and 100
                    metric['value'] = max(0, min(100, current + change))
                
                # Round to one decimal place
                metric['value'] = round(metric['value'], 1)
    
    async def _update_mock_market_signals(self):
        """Update mock market signals with random changes"""
        for symbol in self.market_signals:
            signal = self.market_signals[symbol]
            
            # Randomly update momentum (10% chance)
            if random.random() < 0.1:
                momentum_change = random.uniform(-0.15, 0.15)
                new_momentum = max(-1, min(1, signal['momentum'] + momentum_change))
                signal['momentum'] = round(new_momentum, 2)
                
                # Update sentiment based on momentum
                if new_momentum > 0.3:
                    signal['sentiment'] = 'bullish'
                elif new_momentum < -0.3:
                    signal['sentiment'] = 'bearish'
                else:
                    signal['sentiment'] = 'neutral'
            
            # Randomly update volatility (5% chance)
            if random.random() < 0.05:
                volatility_change = random.uniform(-0.1, 0.1)
                signal['volatility'] = round(max(0.1, min(1, signal['volatility'] + volatility_change)), 2)
            
            # Randomly update volume trend (15% chance)
            if random.random() < 0.15:
                trends = ['increasing', 'decreasing', 'stable']
                signal['volume_trend'] = random.choice(trends)
            
            # Randomly update market regime (1% chance)
            if random.random() < 0.01:
                regimes = ['uptrend', 'downtrend', 'consolidation', 'breakout', 'reversal']
                signal['market_regime'] = random.choice(regimes)
    
    # Fallback to mock prices if real data is not available
    self.latest_prices = {
        'AAPL': 175.50,
        'MSFT': 350.25,
        'GOOG': 138.90,
        'AMZN': 178.20,
        'TSLA': 175.35,
        'META': 470.80,
        'NVDA': 890.40,
        'AMD': 145.65,
        'INTC': 31.20,
        'IBM': 168.75
    }
    logger.info("Initialized market data handler with mock prices")

async def start_price_updates(self):
    """Start the background task for periodic price updates"""
    if self.price_update_task is None or self.price_update_task.done():
        self.price_update_task = asyncio.create_task(self._send_periodic_price_updates())
        logger.info("Started price update background task")

async def start_risk_metrics_updates(self):
    """Start the background task for periodic risk metrics updates"""
    if self.risk_metrics_task is None or self.risk_metrics_task.done():
        self.risk_metrics_task = asyncio.create_task(self._send_periodic_risk_metrics())
        logger.info("Started risk metrics update background task")

async def start_market_signals_updates(self):
    """Start the background task for periodic market signals updates"""
    if self.market_signals_task is None or self.market_signals_task.done():
        self.market_signals_task = asyncio.create_task(self._send_periodic_market_signals())
        logger.info("Started market signals update background task")

async def stop_all_updates(self):
    """Stop all background tasks"""
    tasks = [
        (self.price_update_task, "price update"),
        (self.risk_metrics_task, "risk metrics"), 
        (self.market_signals_task, "market signals")
    ]
    
    for task, name in tasks:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped {name} background task")

async def _update_prices(self):
    """Update prices using real data if available, otherwise generate mock changes"""
    try:
        # Try to get real prices from trading engine
        if self.trading_engine and hasattr(self.trading_engine, 'get_current_prices'):
            symbols = list(self.latest_prices.keys())
            real_prices = self.trading_engine.get_current_prices(symbols)
                await self._update_mock_risk_metrics()
                
                # Send risk metrics for paper account
                await self.connection_manager.broadcast(
                    WebSocketMessage(
                        type="risk_metrics_update",
                        channel="risk_metrics",
                        data={
                            'account': 'paper',
                            'metrics': self.risk_metrics['paper']
                        },
                        timestamp=datetime.now().isoformat()
                    )
                )
                
                # Send risk metrics for live account
                await self.connection_manager.broadcast(
                    WebSocketMessage(
                        type="risk_metrics_update",
                        channel="risk_metrics",
                        data={
                            'account': 'live',
                            'metrics': self.risk_metrics['live']
                        },
                        timestamp=datetime.now().isoformat()
                    )
                )
                
                # Wait for the next update interval (5 seconds for risk metrics)
                await asyncio.sleep(5)
        
        except asyncio.CancelledError:
            logger.info("Risk metrics update task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in risk metrics update task: {e}")
    
    async def _send_periodic_market_signals(self):
        """Send periodic market signals updates to all connected clients"""
        try:
            while True:
                # Update mock market signals
                await self._update_mock_market_signals()
                
                # Broadcast market signals to all clients subscribed to market_context channel
                await self.connection_manager.broadcast(
                    WebSocketMessage(
                        type="market_signals_update",
                        channel="market_context",
                        data={
                            'signals': list(self.market_signals.values())
                        },
                        timestamp=datetime.now().isoformat()
                    )
                )
                
                # Wait for the next update interval (10 seconds for market signals)
                await asyncio.sleep(10)
        
        except asyncio.CancelledError:
            logger.info("Market signals update task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in market signals update task: {e}")
    
    async def get_latest_prices(self):
        """Get the latest prices for all symbols"""
        price_updates = []
        for symbol, price in self.latest_prices.items():
            price_updates.append({
                'symbol': symbol,
                'price': price,
                'change_pct': random.uniform(-0.5, 0.5)  # Mock change percentage
            })
        
        return price_updates
    
    async def handle_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming WebSocket messages related to market data"""
        if message.get("type") == "market_data_request":
            request_type = message.get("request")
            
            if request_type == "latest_prices":
                # Return the latest prices
                price_updates = await self.get_latest_prices()
                return {
                    "type": "price_updates",
                    "data": price_updates,
                    "timestamp": datetime.now().isoformat()
                }
            
            elif request_type == "risk_metrics":
                # Return risk metrics for the specified account
                account = message.get("account", "paper")
                return {
                    "type": "risk_metrics_update",
                    "data": {
                        'account': account,
                        'metrics': self.risk_metrics.get(account, [])
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            elif request_type == "market_signals":
                # Return market signals
                return {
                    "type": "market_signals_update",
                    "data": {
                        'signals': list(self.market_signals.values())
                    },
                    "timestamp": datetime.now().isoformat()
                }
        
        return None
