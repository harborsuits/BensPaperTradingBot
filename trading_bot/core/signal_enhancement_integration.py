#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Enhancement Integration

This module provides examples and utilities for integrating signal quality
enhancements with existing trading strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd

from trading_bot.core.signal_quality_enhancer import SignalQualityEnhancer
from trading_bot.strategies.factory.strategy_template import Signal, SignalType
from trading_bot.core.market_data_manager import MarketDataManager
from trading_bot.core.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class SignalEnhancementIntegrator:
    """
    Signal Enhancement Integrator
    
    This class helps integrate signal quality enhancements with existing strategies:
    - Initializes and configures the SignalQualityEnhancer
    - Provides methods to enhance signals from any strategy
    - Handles multi-timeframe data collection
    - Integrates with market context and news data
    """
    
    def __init__(self, 
               market_data_manager: MarketDataManager,
               event_bus: EventBus,
               parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Signal Enhancement Integrator.
        
        Args:
            market_data_manager: Market data manager for additional data access
            event_bus: Event bus for publishing enhancement events
            parameters: Parameters to pass to SignalQualityEnhancer
        """
        self.market_data_manager = market_data_manager
        self.event_bus = event_bus
        self.enhancer = SignalQualityEnhancer(parameters)
        
        # Register for events
        self._register_events()
        
        logger.info("Signal Enhancement Integrator initialized")
    
    def _register_events(self):
        """Register for events of interest."""
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal_generated)
        
    def _on_signal_generated(self, event: Event):
        """
        Handle signal generated events by enhancing the signal.
        
        Args:
            event: Signal generated event
        """
        if 'signal' not in event.data:
            return
            
        signal = event.data['signal']
        
        # Enhance the signal
        enhanced_signal = self.enhance_signal(signal)
        
        # Publish enhanced signal event
        self.event_bus.publish(Event(
            event_type=EventType.SIGNAL_ENHANCED,
            data={
                'original_signal': signal,
                'enhanced_signal': enhanced_signal,
                'valid': self.enhancer.is_valid_signal(enhanced_signal)
            }
        ))
    
    def enhance_signal(self, signal: Signal) -> Signal:
        """
        Enhance a signal with quality metadata.
        
        Args:
            signal: Signal to enhance
            
        Returns:
            Enhanced signal
        """
        # Get multi-timeframe data for the symbol
        multi_tf_data = self._get_multi_timeframe_data(signal.symbol)
        
        # Get market context data
        market_data = self._get_market_context_data(signal)
        
        # Get news data
        news_data = self._get_news_data(signal.symbol)
        
        # Enhance the signal
        enhanced_signal = self.enhancer.enhance_signal(
            signal=signal,
            data=multi_tf_data,
            market_data=market_data,
            news_data=news_data
        )
        
        return enhanced_signal
    
    def _get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple timeframes for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        # Get base timeframe from market data manager
        base_tf_data = self.market_data_manager.get_data(symbol)
        if base_tf_data is None or len(base_tf_data) == 0:
            logger.warning(f"No data available for {symbol}")
            return {}
            
        result = {1: base_tf_data}  # Base timeframe (normalized to 1)
        
        # Get higher timeframes if available
        for tf_multiple in [2, 4, 8]:
            tf_data = self.market_data_manager.get_higher_timeframe_data(
                symbol=symbol,
                timeframe_multiple=tf_multiple
            )
            
            if tf_data is not None and len(tf_data) > 0:
                result[tf_multiple] = tf_data
                
        return result
    
    def _get_market_context_data(self, signal: Signal) -> Dict[str, Any]:
        """
        Get market context data for enhancing signals.
        
        Args:
            signal: Signal to get context for
            
        Returns:
            Market context data
        """
        result = {}
        
        # Add market breadth data for stocks
        if signal.asset_class == 'stock':
            # Get market breadth data
            try:
                market_breadth = self.market_data_manager.get_market_breadth()
                if market_breadth:
                    result['market_breadth'] = market_breadth
            except Exception as e:
                logger.error(f"Error getting market breadth data: {e}")
            
            # Get sector/industry data
            try:
                symbol_info = self.market_data_manager.get_symbol_info(signal.symbol)
                if symbol_info:
                    result[signal.symbol] = symbol_info
            except Exception as e:
                logger.error(f"Error getting symbol info for {signal.symbol}: {e}")
        
        # Add spread data for forex
        elif signal.asset_class == 'forex':
            try:
                forex_data = self.market_data_manager.get_forex_market_data(signal.symbol)
                if forex_data:
                    result[signal.symbol] = forex_data
            except Exception as e:
                logger.error(f"Error getting forex market data for {signal.symbol}: {e}")
                
        return result
    
    def _get_news_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get news data for a symbol.
        
        Args:
            symbol: Symbol to get news for
            
        Returns:
            News data dictionary
        """
        result = {}
        
        try:
            # Get recent news for the symbol
            news = self.market_data_manager.get_recent_news(symbol)
            if news:
                result[symbol] = {'recent': news}
        except Exception as e:
            logger.error(f"Error getting news data for {symbol}: {e}")
            
        return result

def enhance_strategy_signals(strategy_instance, enhancer: SignalQualityEnhancer):
    """
    Helper function to wrap a strategy's generate_signals method with enhancement.
    
    Args:
        strategy_instance: Strategy instance to enhance
        enhancer: SignalQualityEnhancer instance
        
    Returns:
        None - modifies strategy instance in-place
    """
    # Store original method
    original_generate_signals = strategy_instance.generate_signals
    
    # Create enhanced version
    def enhanced_generate_signals(universe: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        # Call original method
        signals = original_generate_signals(universe)
        
        # Enhance each signal
        enhanced_signals = {}
        for symbol, signal in signals.items():
            # Create multi-timeframe data structure
            multi_tf_data = {1: universe[symbol]}  # Base timeframe
            
            # Enhance the signal
            enhanced_signal = enhancer.enhance_signal(
                signal=signal,
                data=multi_tf_data
            )
            
            # Only include valid signals
            if enhancer.is_valid_signal(enhanced_signal):
                enhanced_signals[symbol] = enhanced_signal
            else:
                logger.info(f"Signal for {symbol} filtered out by quality enhancer")
                
        return enhanced_signals
    
    # Replace method
    strategy_instance.generate_signals = enhanced_generate_signals
    
    logger.info(f"Enhanced signals for strategy: {strategy_instance.__class__.__name__}")
    
def get_default_enhancer_params(asset_class: str = None) -> Dict[str, Any]:
    """
    Get default parameters for SignalQualityEnhancer based on asset class.
    
    Args:
        asset_class: Asset class (stock, forex, crypto, options)
        
    Returns:
        Dictionary of parameters
    """
    # Base parameters
    params = {
        'volume_spike_threshold': 1.5,
        'volume_lookback_period': 20,
        'require_timeframe_confirmation': True,
        'timeframe_confirmation_levels': [2, 4],
        'confirmation_threshold': 0.7,
    }
    
    # Asset-specific adjustments
    if asset_class == 'stock':
        params.update({
            'require_market_breadth_check': True,
            'market_breadth_threshold': 0.6,
            'volume_spike_threshold': 1.8,  # Stocks need stronger volume confirmation
        })
    elif asset_class == 'forex':
        params.update({
            'require_market_breadth_check': False,
            'require_timeframe_confirmation': True,
            'confirmation_threshold': 0.8,  # Higher threshold for forex
            'volume_spike_threshold': 1.3,  # Lower volume threshold for forex
        })
    elif asset_class == 'crypto':
        params.update({
            'require_market_breadth_check': False,
            'volume_spike_threshold': 2.0,  # Crypto needs stronger volume confirmation
        })
    elif asset_class == 'options':
        params.update({
            'require_market_breadth_check': True,
            'market_breadth_threshold': 0.7,  # Higher threshold for options
            'require_timeframe_confirmation': False,  # Options often don't have multi-timeframe data
        })
        
    return params
