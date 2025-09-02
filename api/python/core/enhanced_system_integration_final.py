#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced System Integration - Final Implementation

This file demonstrates how all enhanced components work together in the event-driven
trading system architecture, including:
1. Signal Quality Enhancement
2. Cross-Asset Opportunity Ranking
3. Capital Allocation Optimization
4. Cross-Asset Risk Management
5. Strategy Performance Feedback Loop

The integration provides a seamless flow of events and data between components,
ensuring signals are properly enhanced, opportunities are ranked across asset classes,
capital is optimally allocated, and risk is managed across the entire portfolio.
"""

import logging
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Import enhanced system components
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.core.signal_quality_enhancer import SignalQualityEnhancer
from trading_bot.core.data_flow_enhancement import DataFlowEnhancer
from trading_bot.core.cross_asset_opportunity_ranker import CrossAssetOpportunityRanker
from trading_bot.core.capital_allocation_optimizer import CapitalAllocationOptimizer
from trading_bot.core.cross_asset_risk_manager import CrossAssetRiskManager
from trading_bot.core.strategy_performance_feedback import StrategyPerformanceFeedback

# Import strategy components
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.models.signal import Signal
from trading_bot.models.market_data import MarketData
from trading_bot.models.market_context import MarketContext

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedTradingSystem:
    """
    Enhanced Trading System Integration
    
    This class orchestrates the operation of all enhanced components in the 
    event-driven trading system architecture. It manages the workflow from 
    market data ingestion to trade execution, ensuring all components work
    together seamlessly.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the enhanced trading system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        
        # Initialize event bus
        self.event_bus = EventBus()
        
        # Initialize enhanced components
        self.signal_enhancer = SignalQualityEnhancer(self.event_bus)
        self.data_flow_enhancer = DataFlowEnhancer(self.event_bus)
        self.opportunity_ranker = CrossAssetOpportunityRanker(self.event_bus)
        self.allocation_optimizer = CapitalAllocationOptimizer(self.event_bus)
        self.risk_manager = CrossAssetRiskManager(self.event_bus)
        self.strategy_performance = StrategyPerformanceFeedback(self.event_bus)
        
        # Initialize strategy factory
        self.strategy_factory = StrategyFactory(self.event_bus)
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Initialize system state
        self.market_context = None
        self.market_regime = "unknown"
        self.current_opportunities = []
        self.current_allocations = {}
        
        logger.info("Enhanced Trading System initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "account_size": 100000.0,
            "max_allocation_percentage": 0.8,
            "data_sources": {
                "market_data": True,
                "news_data": True,
                "fundamental_data": True,
                "alternative_data": False
            },
            "risk_parameters": {
                "max_position_size": 0.1,
                "max_asset_class_exposure": {
                    "stocks": 0.5,
                    "forex": 0.4,
                    "crypto": 0.3,
                    "options": 0.2
                },
                "correlation_threshold": 0.7,
                "vix_scaling": True
            },
            "performance_parameters": {
                "min_trades_for_significance": 10,
                "adaptation_speed": 0.2,
                "min_strategy_weight": 0.05,
                "max_strategy_weight": 0.3
            }
        }
        
        # If config file provided, load and merge with defaults
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge configurations
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def _subscribe_to_events(self):
        """Subscribe to relevant system events."""
        # Market data and context events
        self.event_bus.subscribe(EventType.MARKET_DATA_UPDATED, self._on_market_data_updated)
        self.event_bus.subscribe(EventType.MARKET_CONTEXT_UPDATED, self._on_market_context_updated)
        
        # Signal and opportunity events
        self.event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal_generated)
        self.event_bus.subscribe(EventType.SIGNALS_ENHANCED, self._on_signals_enhanced)
        self.event_bus.subscribe(EventType.OPPORTUNITIES_RANKED, self._on_opportunities_ranked)
        
        # Capital and risk events
        self.event_bus.subscribe(EventType.CAPITAL_ALLOCATIONS_UPDATED, self._on_allocations_updated)
        self.event_bus.subscribe(EventType.RISK_METRICS_UPDATED, self._on_risk_metrics_updated)
        
        # Trade events
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
        self.event_bus.subscribe(EventType.TRADE_CLOSED, self._on_trade_closed)
        
        # Strategy performance events
        self.event_bus.subscribe(EventType.STRATEGY_PERFORMANCE_WEIGHTS, self._on_strategy_weights_updated)
        
        logger.info("Subscribed to system events")
    
    def _on_market_data_updated(self, event: Event):
        """
        Handle market data updated events.
        
        This is the entry point for the data flow pipeline.
        """
        logger.info("Received market data update")
        
        # Extract market data from event
        market_data = event.data.get('market_data', {})
        
        # Check for required data sources
        if not self.data_flow_enhancer.validate_market_data(market_data):
            logger.warning("Market data is incomplete or invalid")
            return
        
        # Trigger market context update
        self.event_bus.publish(Event(
            event_type=EventType.UPDATE_MARKET_CONTEXT,
            data={'market_data': market_data}
        ))
    
    def _on_market_context_updated(self, event: Event):
        """Handle market context updated events."""
        logger.info("Market context updated")
        
        # Update local context
        self.market_context = event.data.get('market_context')
        
        # Get market regime
        self.market_regime = self.market_context.regime
        
        # Log current regime
        logger.info(f"Current market regime: {self.market_regime}")
        
        # Trigger strategy selection based on new context
        self._select_strategies_for_context()
    
    def _select_strategies_for_context(self):
        """Select and activate strategies based on current market context."""
        if not self.market_context:
            logger.warning("Cannot select strategies: Market context not available")
            return
        
        # Get strategy weights for current regime
        strategy_weights = self.strategy_performance.get_strategy_weights_for_regime(self.market_regime)
        
        # Activate strategies based on weights
        for strategy_name, weight in strategy_weights.items():
            # Only activate strategies with significant weight
            if weight >= self.config['performance_parameters']['min_strategy_weight']:
                self.strategy_factory.activate_strategy(strategy_name)
                logger.info(f"Activated strategy: {strategy_name} with weight {weight:.2f}")
        
        # Publish strategy activation event
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGIES_ACTIVATED,
            data={
                'active_strategies': list(strategy_weights.keys()),
                'weights': strategy_weights,
                'market_regime': self.market_regime
            }
        ))
        
        # Trigger signal generation
        self._request_signal_generation()
    
    def _request_signal_generation(self):
        """Request signal generation from active strategies."""
        logger.info("Requesting signal generation from active strategies")
        
        # Publish signal generation request event
        self.event_bus.publish(Event(
            event_type=EventType.REQUEST_SIGNAL_GENERATION,
            data={
                'market_context': self.market_context,
                'timestamp': datetime.now().isoformat()
            }
        ))
    
    def _on_signal_generated(self, event: Event):
        """Handle signal generated events."""
        logger.info("Signal generated event received")
        
        # Extract signals from event
        signals = event.data.get('signals', [])
        strategy_name = event.data.get('strategy_name', 'unknown')
        
        if not signals:
            logger.info(f"No signals generated by strategy: {strategy_name}")
            return
        
        # Log generated signals
        logger.info(f"Received {len(signals)} signals from strategy: {strategy_name}")
        
        # Send signals to signal enhancer
        self.event_bus.publish(Event(
            event_type=EventType.ENHANCE_SIGNALS,
            data={
                'signals': signals,
                'strategy_name': strategy_name,
                'market_context': self.market_context
            }
        ))
    
    def _on_signals_enhanced(self, event: Event):
        """Handle signals enhanced events."""
        logger.info("Signals enhanced event received")
        
        # Extract enhanced signals from event
        enhanced_signals = event.data.get('enhanced_signals', [])
        strategy_name = event.data.get('strategy_name', 'unknown')
        
        if not enhanced_signals:
            logger.info(f"No enhanced signals from strategy: {strategy_name}")
            return
        
        # Log enhanced signals
        logger.info(f"Received {len(enhanced_signals)} enhanced signals from strategy: {strategy_name}")
        
        # Send enhanced signals to opportunity ranker
        self.event_bus.publish(Event(
            event_type=EventType.RANK_OPPORTUNITIES,
            data={
                'signals': enhanced_signals,
                'strategy_name': strategy_name,
                'market_context': self.market_context
            }
        ))
    
    def _on_opportunities_ranked(self, event: Event):
        """Handle opportunities ranked events."""
        logger.info("Opportunities ranked event received")
        
        # Extract opportunities from event
        opportunities = event.data.get('opportunities', {})
        top_opportunities = opportunities.get('top_opportunities', [])
        
        if not top_opportunities:
            logger.info("No viable opportunities after ranking")
            return
        
        # Update current opportunities
        self.current_opportunities = top_opportunities
        
        # Log top opportunities
        logger.info(f"Top {len(top_opportunities)} opportunities received after ranking")
        
        # Send opportunities to capital allocation optimizer
        self.event_bus.publish(Event(
            event_type=EventType.OPTIMIZE_CAPITAL_ALLOCATION,
            data={
                'opportunities': top_opportunities,
                'account_size': self.config['account_size'],
                'max_allocation_percentage': self.config['max_allocation_percentage'],
                'market_context': self.market_context
            }
        ))
    
    def _on_allocations_updated(self, event: Event):
        """Handle capital allocations updated events."""
        logger.info("Capital allocations updated event received")
        
        # Extract allocations from event
        allocations = event.data.get('allocations', {})
        
        if not allocations:
            logger.info("No capital allocations received")
            return
        
        # Update current allocations
        self.current_allocations = allocations
        
        # Log allocation summary
        asset_class_percentages = allocations.get('asset_class_percentages', {})
        total_allocated = allocations.get('total_allocated_percent', 0)
        
        logger.info(f"Capital allocation: Total allocated: {total_allocated*100:.1f}%")
        for asset_class, percentage in asset_class_percentages.items():
            logger.info(f"  {asset_class.upper()}: {percentage*100:.1f}%")
        
        # Send allocations to risk manager for final verification
        self.event_bus.publish(Event(
            event_type=EventType.VERIFY_RISK_PROFILE,
            data={
                'allocations': allocations,
                'opportunities': self.current_opportunities,
                'market_context': self.market_context
            }
        ))
    
    def _on_risk_metrics_updated(self, event: Event):
        """Handle risk metrics updated events."""
        logger.info("Risk metrics updated event received")
        
        # Extract risk metrics and warnings from event
        risk_metrics = event.data.get('risk_metrics', {})
        warnings = event.data.get('warnings', [])
        adjusted_allocations = event.data.get('adjusted_allocations')
        
        # If allocations were adjusted, update current allocations
        if adjusted_allocations:
            logger.info("Allocations adjusted by risk manager")
            self.current_allocations = adjusted_allocations
        
        # Log risk warnings
        if warnings:
            logger.warning(f"Risk warnings received: {len(warnings)}")
            for warning in warnings:
                warning_type = warning.get('type', 'unknown')
                severity = warning.get('severity', 'medium')
                logger.warning(f"  {severity.upper()} risk warning: {warning_type}")
        
        # If no blocking warnings, proceed to trade execution
        if not any(warning.get('severity') == 'high' for warning in warnings):
            logger.info("No blocking risk warnings, proceeding to trade execution")
            self._execute_trades()
    
    def _execute_trades(self):
        """Execute trades based on optimized allocations."""
        logger.info("Executing trades based on optimized allocations")
        
        # Extract individual position allocations
        positions = self.current_allocations.get('positions', [])
        
        if not positions:
            logger.info("No positions to execute")
            return
        
        # Execute each position
        for position in positions:
            symbol = position.get('symbol')
            asset_class = position.get('asset_class')
            direction = position.get('direction')
            allocated_amount = position.get('allocated_amount')
            opportunity_score = position.get('opportunity_score')
            strategy_name = position.get('strategy_name')
            
            logger.info(f"Executing trade: {symbol} ({asset_class}) - {direction} - ${allocated_amount:.2f}")
            
            # Publish trade execution event
            self.event_bus.publish(Event(
                event_type=EventType.EXECUTE_TRADE,
                data={
                    'symbol': symbol,
                    'asset_class': asset_class,
                    'direction': direction,
                    'amount': allocated_amount,
                    'opportunity_score': opportunity_score,
                    'strategy_name': strategy_name,
                    'market_regime': self.market_regime,
                    'timestamp': datetime.now().isoformat()
                }
            ))
    
    def _on_trade_executed(self, event: Event):
        """Handle trade executed events."""
        logger.info("Trade executed event received")
        
        # Extract trade information for strategy performance tracking
        trade_info = event.data.copy()
        
        # Record trade for strategy performance tracking
        self.strategy_performance.record_trade_opened(
            strategy_name=trade_info.get('strategy_name', 'unknown'),
            asset_class=trade_info.get('asset_class', 'unknown'),
            market_regime=self.market_regime,
            trade_id=trade_info.get('trade_id', ''),
            timestamp=trade_info.get('timestamp', datetime.now().isoformat()),
            metadata=trade_info
        )
    
    def _on_trade_closed(self, event: Event):
        """Handle trade closed events."""
        logger.info("Trade closed event received")
        
        # Extract trade information
        trade_info = event.data.copy()
        
        # Record trade outcome for strategy performance tracking
        self.strategy_performance.record_trade_closed(
            trade_id=trade_info.get('trade_id', ''),
            profit_loss=trade_info.get('profit_loss', 0),
            outcome='win' if trade_info.get('profit_loss', 0) > 0 else 'loss',
            timestamp=trade_info.get('timestamp', datetime.now().isoformat()),
            metadata=trade_info
        )
    
    def _on_strategy_weights_updated(self, event: Event):
        """Handle strategy performance weights updated events."""
        logger.info("Strategy performance weights updated event received")
        
        # Extract weights information
        weights = event.data.get('strategy_weights', {})
        regime = event.data.get('market_regime', 'unknown')
        
        # Log updated weights
        logger.info(f"Updated strategy weights for regime: {regime}")
        for strategy, weight in weights.items():
            if weight > 0:
                logger.info(f"  {strategy}: {weight:.2f}")
    
    def start(self):
        """Start the enhanced trading system."""
        logger.info("Starting enhanced trading system")
        
        # Initialize all components
        self.data_flow_enhancer.initialize()
        self.signal_enhancer.initialize()
        self.opportunity_ranker.initialize()
        self.allocation_optimizer.initialize()
        self.risk_manager.initialize()
        self.strategy_performance.initialize()
        self.strategy_factory.initialize()
        
        # Load strategies
        self.strategy_factory.load_strategies()
        
        # Start market data ingestion (in a real system, this would connect to data feeds)
        logger.info("System running - awaiting market data")
    
    def stop(self):
        """Stop the enhanced trading system."""
        logger.info("Stopping enhanced trading system")
        
        # Clean up resources and save state if needed
        self.strategy_performance.save_performance_data()
        
        logger.info("Enhanced trading system stopped")


if __name__ == "__main__":
    # Example usage
    system = EnhancedTradingSystem()
    system.start()
    
    # In a real implementation, we would keep the system running here
    # For example, in a web application or as a service
    
    # For demonstration purposes, we'll just print that it's running
    print("Enhanced trading system is running.")
    print("Press Ctrl+C to stop.")
    
    try:
        # Keep the system running
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system.stop()
        print("System stopped.")
