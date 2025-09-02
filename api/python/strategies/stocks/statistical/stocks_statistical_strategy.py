#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks Statistical Strategy

This strategy implements statistical trading for stocks.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from trading_bot.strategies.stocks.base.stocks_base_strategy import StocksBaseStrategy
from trading_bot.strategies.factory.strategy_registry import register_strategy
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.strategies.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

@register_strategy({
    'asset_class': 'stocks',
    'strategy_type': 'statistical',
    'compatible_market_regimes': ['trending', 'all_weather'],
    'timeframe': 'daily',
    'regime_compatibility_scores': {
        'trending': 0.70,       # Good compatibility with trending markets
        'ranging': 0.60,        # Moderate compatibility with ranging markets
        'volatile': 0.50,       # Moderate compatibility with volatile markets
        'low_volatility': 0.60, # Moderate compatibility with low volatility markets
        'all_weather': 0.65     # Good overall compatibility
    }
})
class StocksStatisticalStrategy(StocksBaseStrategy):
    """
    Stocks Statistical Strategy
    
    This strategy implements statistical trading for stocks, using:
    - Specialized indicators for stocks
    - Statistical-based approach to market analysis
    - Risk management tailored to stocks markets
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Strategy parameters
        'lookback_period': 20,
        'entry_threshold': 0.5,
        'exit_threshold': -0.2,
        
        # Risk parameters
        'max_risk_per_trade_percent': 0.01  # 1% risk per trade
    }
    
    def __init__(self, name: str = "StocksStatisticalStrategy", 
                parameters: Optional[Dict[str, Any]] = None, 
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize StocksStatisticalStrategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMS)
            metadata: Strategy metadata
        """
        # Merge default parameters with provided parameters
        merged_params = self.DEFAULT_PARAMS.copy()
        if parameters:
            merged_params.update(parameters)
        
        super().__init__(name, merged_params, metadata)
        
        # Strategy-specific state variables
        self.signals = {}  # Last signals by symbol
        
        logger.info(f"StocksStatisticalStrategy initialized")
    
    def register_events(self, event_bus: EventBus) -> None:
        """
        Register strategy events with the event bus.
        
        Args:
            event_bus: Event bus to register with
        """
        super().register_events(event_bus)
        
        # Register for additional events specific to this strategy
        # event_bus.register(EventType.CUSTOM_EVENT, self._on_custom_event)
        
        logger.info(f"Strategy registered for events")
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed events.
        
        This is when we'll generate new trading signals based on the 
        completed candle data.
        """
        data = event.data
        if not data or 'symbol' not in data or 'timeframe' not in data:
            return
        
        # Check if this is our target timeframe
        if data['timeframe'] != self.parameters.get('timeframe', 'daily'):
            return
        
        symbol = data['symbol']
        logger.debug(f"Timeframe completed for {symbol}")
        
        # Get the current market data
        universe = {}
        if self.event_bus:
            # Request market data from the system
            market_data_event = Event(
                event_type=EventType.MARKET_DATA_REQUEST,
                data={
                    'symbols': [symbol],
                    'timeframe': data['timeframe']
                }
            )
            response = self.event_bus.request(market_data_event)
            if response and 'data' in response:
                universe = response['data']
        
        # If we have market data, generate signals
        if universe:
            signals = self.generate_signals(universe)
            
            # Publish signals
            if signals and self.event_bus:
                for sym, signal in signals.items():
                    signal_event = Event(
                        event_type=EventType.SIGNAL_GENERATED,
                        data={
                            'signal': signal.to_dict()
                        }
                    )
                    self.event_bus.publish(signal_event)
                    logger.info(f"Published signal for {sym}: {signal.signal_type}")
    
    def calculate_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol for the data
            
        Returns:
            Dictionary of calculated indicators
        """
        result = {}
        
        # Input validation
        if len(data) < self.parameters['lookback_period']:
            logger.warning(f"Insufficient data for {symbol}: {len(data)} bars")
            return result
        
        try:
            # Calculate indicators - actual implementation depends on strategy type
            # This is just a placeholder
            
            # Example indicators that might be used:
            result['sma_20'] = data['close'].rolling(window=20).mean().iloc[-1]
            result['sma_50'] = data['close'].rolling(window=50).mean().iloc[-1]
            result['rsi'] = self._calculate_rsi(data['close'], period=14).iloc[-1]
            result['volatility'] = data['close'].pct_change().std() * np.sqrt(252)
            
            # Strategy-specific indicators would be added here
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI technical indicator."""
        delta = prices.diff()
        
        gain = delta.copy()
        gain[gain < 0] = 0
        
        loss = delta.copy()
        loss[loss > 0] = 0
        loss = -loss
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for the universe of symbols.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Process each symbol
        for symbol, data in universe.items():
            if len(data) < self.parameters['lookback_period']:
                logger.debug(f"Skipping {symbol}: insufficient data")
                continue
            
            # Calculate indicators
            indicators = self.calculate_indicators(data, symbol)
            if not indicators:
                continue
            
            # Generate signal based on indicators
            # This is a placeholder - actual logic would depend on strategy type
            signal = None
            
            # Example signal generation logic:
            if indicators.get('sma_20', 0) > indicators.get('sma_50', 0) and indicators.get('rsi', 50) < 70:
                # Bullish conditions
                
                # Create a long signal
                price = data['close'].iloc[-1]
                stop_loss = price * 0.95  # 5% stop loss
                take_profit = price * 1.10  # 10% take profit
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    confidence=0.7,  # Confidence level
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'strategy': self.name,
                        'indicators': indicators
                    }
                )
                
            elif indicators.get('sma_20', 0) < indicators.get('sma_50', 0) and indicators.get('rsi', 50) > 30:
                # Bearish conditions
                
                # Create a short signal
                price = data['close'].iloc[-1]
                stop_loss = price * 1.05  # 5% stop loss
                take_profit = price * 0.90  # 10% take profit
                
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    confidence=0.7,  # Confidence level
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    metadata={
                        'strategy': self.name,
                        'indicators': indicators
                    }
                )
            
            # Add signal to results if generated
            if signal:
                signals[symbol] = signal
                self.signals[symbol] = signal  # Store the last signal
        
        return signals
