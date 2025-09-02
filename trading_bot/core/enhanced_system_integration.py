#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced System Integration

This module demonstrates how to integrate the signal quality and data flow
enhancements with the existing trading system components.
"""

import logging
from typing import Dict, List, Optional, Any

from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.core.signal_quality_enhancer import SignalQualityEnhancer
from trading_bot.core.data_flow_enhancement import DataFlowEnhancer, SignalEnhancementWrapper
from trading_bot.strategies.factory.strategy_factory import StrategyFactory
from trading_bot.core.market_data_manager import MarketDataManager
from trading_bot.core.trade_scorer import TradeScorer

logger = logging.getLogger(__name__)

class EnhancedTradingSystem:
    """
    Enhanced Trading System
    
    This class demonstrates how to integrate all the signal quality and
    data flow enhancements with your existing trading system.
    """
    
    def __init__(self):
        """Initialize Enhanced Trading System."""
        # Initialize core components
        self.event_bus = EventBus()
        self.market_data_manager = MarketDataManager(self.event_bus)
        self.strategy_factory = StrategyFactory(self.event_bus)
        self.trade_scorer = TradeScorer(self.event_bus)
        
        # Initialize enhancement components
        self.data_flow_enhancer = DataFlowEnhancer(self.event_bus)
        
        # Register for events
        self._register_events()
        
        logger.info("Enhanced Trading System initialized")
    
    def _register_events(self):
        """Register for events of interest."""
        # Register for signal enhanced events
        self.event_bus.subscribe(EventType.SIGNAL_ENHANCED, self._on_signal_enhanced)
        
        # Register for strategy selection events
        self.event_bus.subscribe(EventType.ALL_DATA_READY_FOR_STRATEGY_SELECTION, 
                               self._on_data_ready_for_strategy_selection)
    
    def _on_data_ready_for_strategy_selection(self, event: Event):
        """
        Handle data ready for strategy selection events.
        
        Args:
            event: Data ready event
        """
        logger.info("All data is ready for strategy selection")
        
        # Extract market context
        market_context = event.data.get('market_context', {})
        
        # Determine market regime
        market_regime = market_context.get('market_regime', 'unknown')
        vix_value = market_context.get('vix', 0)
        news_sentiment = market_context.get('news_sentiment', 'neutral')
        
        logger.info(f"Market regime: {market_regime}, VIX: {vix_value}, Sentiment: {news_sentiment}")
        
        # Select strategies based on market context
        selected_strategies = self.strategy_factory.select_strategies(
            market_regime=market_regime,
            vix_value=vix_value,
            news_sentiment=news_sentiment
        )
        
        # Enhance all selected strategies with signal quality improvements
        enhanced_strategies = {}
        for asset_class, strategies in selected_strategies.items():
            enhanced_strategies[asset_class] = [
                SignalEnhancementWrapper.enhance_strategy(strategy)
                for strategy in strategies
            ]
        
        # Publish strategy selection completed event
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_SELECTION_COMPLETED,
            data={
                'selected_strategies': enhanced_strategies,
                'market_context': market_context,
                'selection_timestamp': self.market_data_manager.current_timestamp,
            }
        ))
    
    def _on_signal_enhanced(self, event: Event):
        """
        Handle signal enhanced events.
        
        Args:
            event: Signal enhanced event
        """
        # Extract enhanced signal and validity
        enhanced_signal = event.data.get('enhanced_signal')
        is_valid = event.data.get('valid', False)
        
        if not enhanced_signal:
            return
            
        if not is_valid:
            logger.info(f"Signal for {enhanced_signal.symbol} filtered out by quality enhancer")
            return
            
        # Log enhancement details
        metadata = enhanced_signal.metadata or {}
        logger.info(f"Enhanced signal for {enhanced_signal.symbol}: "
                   f"Quality score: {metadata.get('quality_score', 0):.2f}, "
                   f"MTF confirmed: {metadata.get('timeframe_confirmed', False)}, "
                   f"Volume confirmed: {metadata.get('volume_confirmed', False)}")
        
        # Score the enhanced signal
        self.trade_scorer.score_signal(enhanced_signal)
    
    def start(self):
        """Start the enhanced trading system."""
        logger.info("Starting Enhanced Trading System")
        
        # Start market data manager
        self.market_data_manager.start()
        
        logger.info("Enhanced Trading System started")
    
    def stop(self):
        """Stop the enhanced trading system."""
        logger.info("Stopping Enhanced Trading System")
        
        # Stop market data manager
        self.market_data_manager.stop()
        
        logger.info("Enhanced Trading System stopped")


def apply_enhancements_to_existing_system(existing_system):
    """
    Apply enhancements to an existing trading system.
    
    Args:
        existing_system: Existing trading system instance
        
    Returns:
        Enhanced system
    """
    # Add data flow enhancer
    existing_system.data_flow_enhancer = DataFlowEnhancer(existing_system.event_bus)
    
    # Enhance all strategies in the system
    for strategy in existing_system.strategies:
        SignalEnhancementWrapper.enhance_strategy(strategy)
    
    # Register for enhanced signal events
    existing_system.event_bus.subscribe(
        EventType.SIGNAL_ENHANCED, 
        lambda event: handle_enhanced_signal(existing_system, event)
    )
    
    logger.info("Applied enhancements to existing trading system")
    return existing_system


def handle_enhanced_signal(system, event):
    """
    Handle enhanced signal events for an existing system.
    
    Args:
        system: Trading system
        event: Signal enhanced event
    """
    # Extract enhanced signal and validity
    enhanced_signal = event.data.get('enhanced_signal')
    is_valid = event.data.get('valid', False)
    
    if not enhanced_signal or not is_valid:
        return
        
    # Process the valid enhanced signal
    system.process_signal(enhanced_signal)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start enhanced trading system
    trading_system = EnhancedTradingSystem()
    trading_system.start()
    
    # Keep running until interrupted
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        trading_system.stop()
