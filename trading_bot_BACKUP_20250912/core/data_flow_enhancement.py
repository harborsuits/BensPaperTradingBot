#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Flow Enhancement

This module provides utilities to enhance the data flow between trading system components,
ensuring all strategies receive necessary context data for proper decision making.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd

from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.core.signal_quality_enhancer import SignalQualityEnhancer
from trading_bot.strategies.factory.strategy_template import Signal

logger = logging.getLogger(__name__)

class DataFlowEnhancer:
    """
    Data Flow Enhancer
    
    This class improves communication and data flow between system components:
    1. Ensures strategies have complete market context before generating signals
    2. Enriches signals with metadata needed for proper scoring
    3. Coordinates multi-asset data sharing for cross-asset insights
    4. Manages data timing to prevent stale or incomplete data usage
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize Data Flow Enhancer.
        
        Args:
            event_bus: System event bus
        """
        self.event_bus = event_bus
        self.signal_enhancer = SignalQualityEnhancer()
        
        # Cache for context data
        self._market_context_cache = {}
        self._multi_timeframe_data = {}
        self._news_cache = {}
        self._breadth_cache = {}
        
        # Data ready tracking
        self._ready_status = {
            'market_data': False,
            'context_analysis': False,
            'news': False,
            'vix': False,
            'fundamentals': False
        }
        
        # Register for events
        self._register_events()
        
        logger.info("Data Flow Enhancer initialized")
        
    def _register_events(self):
        """Register for events of interest."""
        # Data ingestion events
        self.event_bus.subscribe(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        self.event_bus.subscribe(EventType.NEWS_DATA_UPDATED, self._on_news_updated)
        self.event_bus.subscribe(EventType.VIX_DATA_UPDATED, self._on_vix_updated)
        self.event_bus.subscribe(EventType.FUNDAMENTAL_DATA_UPDATED, self._on_fundamental_updated)
        
        # Analysis events
        self.event_bus.subscribe(EventType.CONTEXT_ANALYSIS_COMPLETED, self._on_context_analysis)
        
        # Strategy selection events
        self.event_bus.subscribe(EventType.STRATEGY_SELECTION_REQUESTED, self._on_strategy_selection_requested)
        
        # Signal events
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal_generated)
    
    def _on_market_data_updated(self, event: Event):
        """
        Handle market data updated events.
        
        Args:
            event: Market data event
        """
        # Update market data cache
        if 'data' in event.data:
            symbol = event.data.get('symbol')
            timeframe = event.data.get('timeframe', 'default')
            
            if symbol:
                # Store multi-timeframe data
                if symbol not in self._multi_timeframe_data:
                    self._multi_timeframe_data[symbol] = {}
                    
                self._multi_timeframe_data[symbol][timeframe] = event.data['data']
        
        # Mark market data as ready
        self._ready_status['market_data'] = True
        
        # Check if we can proceed with strategy selection
        self._check_strategy_selection_readiness()
    
    def _on_news_updated(self, event: Event):
        """
        Handle news data updated events.
        
        Args:
            event: News data event
        """
        # Update news cache
        if 'data' in event.data:
            for symbol, news in event.data['data'].items():
                self._news_cache[symbol] = news
        
        # Mark news as ready
        self._ready_status['news'] = True
        
        # Check if we can proceed with strategy selection
        self._check_strategy_selection_readiness()
    
    def _on_vix_updated(self, event: Event):
        """
        Handle VIX data updated events.
        
        Args:
            event: VIX data event
        """
        # Update VIX in market context
        if 'value' in event.data:
            self._market_context_cache['vix'] = event.data['value']
        
        # Mark VIX as ready
        self._ready_status['vix'] = True
        
        # Check if we can proceed with strategy selection
        self._check_strategy_selection_readiness()
    
    def _on_fundamental_updated(self, event: Event):
        """
        Handle fundamental data updated events.
        
        Args:
            event: Fundamental data event
        """
        # Update fundamental data cache
        if 'data' in event.data:
            self._market_context_cache['fundamentals'] = event.data['data']
        
        # Mark fundamentals as ready
        self._ready_status['fundamentals'] = True
        
        # Check if we can proceed with strategy selection
        self._check_strategy_selection_readiness()
    
    def _on_context_analysis(self, event: Event):
        """
        Handle context analysis completed events.
        
        Args:
            event: Context analysis event
        """
        # Update market context with analysis results
        if 'context' in event.data:
            self._market_context_cache.update(event.data['context'])
        
        # Mark context analysis as ready
        self._ready_status['context_analysis'] = True
        
        # Check if we can proceed with strategy selection
        self._check_strategy_selection_readiness()
    
    def _check_strategy_selection_readiness(self):
        """Check if we have all required data for strategy selection."""
        # Define required data types for different phases
        strategy_selection_requirements = ['market_data', 'context_analysis', 'vix']
        
        # Check if all required data is ready for strategy selection
        if all(self._ready_status[req] for req in strategy_selection_requirements):
            # Publish event to trigger strategy selection
            self.event_bus.publish(Event(
                event_type=EventType.ALL_DATA_READY_FOR_STRATEGY_SELECTION,
                data={
                    'market_context': self._market_context_cache
                }
            ))
    
    def _on_strategy_selection_requested(self, event: Event):
        """
        Handle strategy selection requested events.
        
        Args:
            event: Strategy selection request event
        """
        # Ensure we have all required data
        if not all(self._ready_status[req] for req in ['market_data', 'context_analysis', 'vix']):
            logger.warning("Strategy selection requested but required data is not ready")
            return
        
        # Publish event with all context data needed for selection
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_SELECTION_CONTEXT,
            data={
                'market_context': self._market_context_cache,
                'vix': self._market_context_cache.get('vix'),
                'market_regime': self._market_context_cache.get('market_regime', 'unknown'),
                'news_sentiment': self._market_context_cache.get('news_sentiment'),
                'asset_class_data': {
                    'stocks': self._get_asset_class_data('stock'),
                    'forex': self._get_asset_class_data('forex'),
                    'crypto': self._get_asset_class_data('crypto'),
                    'options': self._get_asset_class_data('options')
                }
            }
        ))
    
    def _get_asset_class_data(self, asset_class: str) -> Dict[str, Any]:
        """
        Get aggregated data for a specific asset class.
        
        Args:
            asset_class: Asset class to get data for
            
        Returns:
            Asset class specific data
        """
        result = {
            'strength': 0.5,  # Default neutral strength
            'volatility': 'normal',
            'opportunity_score': 0.5,
            'liquidity': 'normal'
        }
        
        # Extract from market context if available
        if 'asset_classes' in self._market_context_cache:
            asset_data = self._market_context_cache['asset_classes'].get(asset_class, {})
            result.update(asset_data)
            
        return result
    
    def _on_signal_generated(self, event: Event):
        """
        Handle signal generated events.
        
        Args:
            event: Signal generated event
        """
        if 'signal' not in event.data:
            return
            
        signal = event.data['signal']
        symbol = signal.symbol
        
        # Get multi-timeframe data for the symbol
        multi_tf_data = self._multi_timeframe_data.get(symbol, {})
        
        # Convert to format expected by enhancer
        enhancer_data = {}
        for tf, data in multi_tf_data.items():
            # Convert timeframe to numeric if it's a string
            if isinstance(tf, str):
                # Parse common timeframe strings
                if tf == '1m':
                    tf_num = 1
                elif tf == '5m':
                    tf_num = 5
                elif tf == '15m':
                    tf_num = 15
                elif tf == '1h':
                    tf_num = 60
                elif tf == '4h':
                    tf_num = 240
                elif tf == '1d':
                    tf_num = 1440
                else:
                    tf_num = 1  # Default
                enhancer_data[tf_num] = data
            else:
                enhancer_data[tf] = data
        
        # Get market context for the symbol
        market_data = {
            symbol: self._market_context_cache.get(symbol, {})
        }
        
        # Add market breadth if available
        if 'market_breadth' in self._market_context_cache:
            market_data['market_breadth'] = self._market_context_cache['market_breadth']
            
        # Get news data for the symbol
        news_data = {
            symbol: {'recent': self._news_cache.get(symbol, [])}
        }
        
        # Enhance the signal
        enhanced_signal = self.signal_enhancer.enhance_signal(
            signal=signal,
            data=enhancer_data,
            market_data=market_data,
            news_data=news_data
        )
        
        # Check if signal is valid
        is_valid = self.signal_enhancer.is_valid_signal(enhanced_signal)
        
        # Publish enhanced signal event
        self.event_bus.publish(Event(
            event_type=EventType.SIGNAL_ENHANCED,
            data={
                'original_signal': signal,
                'enhanced_signal': enhanced_signal,
                'valid': is_valid,
                'symbol': symbol,
                'asset_class': signal.asset_class,
                'strategy_name': signal.strategy_name
            }
        ))

# Add custom event types
EventType.ALL_DATA_READY_FOR_STRATEGY_SELECTION = "ALL_DATA_READY_FOR_STRATEGY_SELECTION"
EventType.STRATEGY_SELECTION_CONTEXT = "STRATEGY_SELECTION_CONTEXT"
EventType.SIGNAL_ENHANCED = "SIGNAL_ENHANCED"

class SignalEnhancementWrapper:
    """
    Signal Enhancement Wrapper
    
    This class provides a decorator-style wrapper for strategy signal generation
    to automatically enhance signals with quality metadata.
    """
    
    @staticmethod
    def enhance_strategy(strategy_instance):
        """
        Decorator method to enhance a strategy's signal generation.
        
        Args:
            strategy_instance: Strategy instance to enhance
            
        Returns:
            Enhanced strategy instance
        """
        # Store original method
        original_generate_signals = strategy_instance.generate_signals
        
        # Create enhanced version
        def enhanced_generate_signals(universe):
            # Call original method
            signals = original_generate_signals(universe)
            
            # Create enhancer with appropriate parameters
            asset_class = getattr(strategy_instance, 'asset_class', None)
            enhancer = SignalQualityEnhancer()
            
            # Enhance each signal
            enhanced_signals = {}
            for symbol, signal in signals.items():
                # Skip if symbol not in universe
                if symbol not in universe:
                    continue
                    
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
        
        return strategy_instance
