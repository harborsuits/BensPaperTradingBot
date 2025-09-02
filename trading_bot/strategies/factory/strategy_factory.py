#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Factory Module

This module provides a factory class for creating and managing trading strategy instances.
It builds on the successful ForexTrendFollowingStrategy pattern and integrates with the
strategy registry system.
"""

import logging
import importlib
import os
from typing import Dict, List, Any, Optional, Type, Union, Tuple
import pandas as pd

from .strategy_registry import (
    StrategyRegistry, 
    AssetClass, 
    StrategyType, 
    MarketRegime, 
    TimeFrame
)

# Assume base strategy imports exist
# Fix import path to match the actual location
try:
    from trading_bot.strategies.forex.base.forex_base_strategy import ForexBaseStrategy
except ImportError:
    # Fallback for backward compatibility or provide a mock
    class ForexBaseStrategy:
        """Placeholder for ForexBaseStrategy if not available"""
        def __init__(self, *args, **kwargs):
            pass
from trading_bot.strategies.base.stock_base import StockBaseStrategy
from trading_bot.strategies.base.crypto_base import CryptoBaseStrategy
from trading_bot.strategies.base.options_base import OptionsBaseStrategy

# Import the event system
from trading_bot.core.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class StrategyFactory:
    """
    Factory class for creating and managing trading strategy instances.
    
    This class is responsible for:
    1. Creating strategy instances
    2. Managing strategy lifecycle
    3. Selecting appropriate strategies based on market conditions
    4. Strategy rotation and retirement
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the strategy factory.
        
        Args:
            event_bus: Optional event bus for strategy events
        """
        self.event_bus = event_bus
        self.active_strategies = {}
        self._discover_strategies()
        
        # Track strategy performance for rotation
        self.strategy_performance = {}
        
        logger.info(f"StrategyFactory initialized with {len(StrategyRegistry.get_all_strategy_names())} strategies")
    
    def _discover_strategies(self):
        """
        Discover and register all available strategies.
        
        This method scans the strategies directory for strategy modules and registers
        them with the StrategyRegistry.
        """
        # This is a simplified version - in production, you'd scan directories
        # and dynamically import modules to discover strategies
        
        # The actual strategies should be registered in their respective __init__.py files
        # using the @register_strategy decorator
        
        logger.info("Strategy discovery complete")
    
    def create_strategy(self, 
                       strategy_name: str, 
                       parameters: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create a strategy instance.
        
        Args:
            strategy_name: Name of the strategy class
            parameters: Optional parameters to override defaults
            metadata: Optional metadata to override defaults
            
        Returns:
            Strategy instance
        """
        strategy_class = StrategyRegistry.get_strategy_class(strategy_name)
        
        if not strategy_class:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        # Create the strategy instance
        strategy = strategy_class(
            name=strategy_name, 
            parameters=parameters or {}, 
            metadata=metadata or {}
        )
        
        # Register with event bus if available
        if self.event_bus and hasattr(strategy, 'register_events'):
            strategy.register_events(self.event_bus)
        
        return strategy
    
    def select_strategies_for_market_regime(self, 
                                           regime: MarketRegime,
                                           asset_class: Optional[AssetClass] = None,
                                           timeframe: Optional[TimeFrame] = None,
                                           max_strategies: int = 3) -> List[str]:
        """
        Select appropriate strategies for the current market regime.
        
        Args:
            regime: Current market regime
            asset_class: Optional asset class filter
            timeframe: Optional timeframe filter
            max_strategies: Maximum number of strategies to select
            
        Returns:
            List of strategy names suitable for the market regime
        """
        # Find strategies compatible with the market regime
        candidates = StrategyRegistry.find_strategies(
            asset_class=asset_class,
            market_regime=regime,
            timeframe=timeframe
        )
        
        # If we have performance data, prioritize by performance
        if self.strategy_performance:
            candidates = sorted(
                candidates,
                key=lambda s: self.strategy_performance.get(s, {}).get('score', 0),
                reverse=True
            )
        
        return candidates[:max_strategies]
    
    def create_optimal_strategy_ensemble(self, 
                                        market_data: Dict[str, pd.DataFrame],
                                        regime: MarketRegime,
                                        asset_class: AssetClass,
                                        timeframe: TimeFrame) -> List[Any]:
        """
        Create an optimal ensemble of strategies for current market conditions.
        
        Args:
            market_data: Dictionary of market data by symbol
            regime: Current market regime
            asset_class: Asset class to trade
            timeframe: Trading timeframe
            
        Returns:
            List of instantiated strategy objects
        """
        # Select strategy names
        strategy_names = self.select_strategies_for_market_regime(
            regime=regime,
            asset_class=asset_class,
            timeframe=timeframe
        )
        
        # Create strategy instances
        strategies = []
        for name in strategy_names:
            try:
                # Get optimal parameters for this regime if available
                metadata = StrategyRegistry.get_strategy_metadata(name)
                optimal_params = metadata.get('optimal_parameters', {}).get(regime.value, {})
                
                # Create strategy with optimal parameters
                strategy = self.create_strategy(name, parameters=optimal_params)
                strategies.append(strategy)
                
                logger.info(f"Added {name} to strategy ensemble for {regime.value} regime")
            except Exception as e:
                logger.error(f"Failed to create strategy {name}: {str(e)}")
        
        return strategies
    
    def update_strategy_performance(self, 
                                   strategy_name: str, 
                                   performance_metrics: Dict[str, Any]):
        """
        Update performance tracking for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            performance_metrics: Dictionary of performance metrics
        """
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {}
        
        # Update metrics
        self.strategy_performance[strategy_name].update(performance_metrics)
        
        # Calculate overall score (this is a simple example)
        metrics = self.strategy_performance[strategy_name]
        
        # Sample score calculation based on Sharpe ratio and win rate
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)
        drawdown = metrics.get('max_drawdown', 0) or 1  # Avoid division by zero
        
        # Simple score formula, can be much more sophisticated
        score = (sharpe * 0.4) + (win_rate * 0.4) + ((1 / abs(drawdown)) * 0.2)
        self.strategy_performance[strategy_name]['score'] = score
        
        # Publish performance update event if available
        if self.event_bus:
            event = Event(
                event_type=EventType.STRATEGY_PERFORMANCE_UPDATED,
                data={
                    'strategy_name': strategy_name,
                    'metrics': metrics
                }
            )
            self.event_bus.publish(event)
    
    def retire_underperforming_strategies(self, 
                                         min_score_threshold: float = 0.5,
                                         lookback_days: int = 30):
        """
        Identify and retire underperforming strategies.
        
        Args:
            min_score_threshold: Minimum acceptable score
            lookback_days: Performance evaluation period in days
        """
        for strategy_name, metrics in self.strategy_performance.items():
            # Check if we have recent data
            last_updated = metrics.get('last_updated')
            if not last_updated:
                continue
                
            # Check score against threshold
            score = metrics.get('score', 0)
            if score < min_score_threshold:
                logger.info(f"Strategy {strategy_name} underperforming, marking for retirement")
                
                # Mark as retired in performance tracking
                metrics['retired'] = True
                metrics['retirement_reason'] = f"Score below threshold: {score} < {min_score_threshold}"
                
                # Publish retirement event if available
                if self.event_bus:
                    event = Event(
                        event_type=EventType.STRATEGY_RETIRED,
                        data={
                            'strategy_name': strategy_name,
                            'reason': metrics['retirement_reason'],
                            'metrics': metrics
                        }
                    )
                    self.event_bus.publish(event)
    
    def promote_strategies(self, min_promotion_score: float = 0.8):
        """
        Identify and promote high-performing strategies.
        
        Args:
            min_promotion_score: Minimum score for promotion
        """
        for strategy_name, metrics in self.strategy_performance.items():
            # Skip already promoted or retired strategies
            if metrics.get('promoted', False) or metrics.get('retired', False):
                continue
                
            # Check score against threshold
            score = metrics.get('score', 0)
            if score >= min_promotion_score:
                logger.info(f"Strategy {strategy_name} performing well, promoting")
                
                # Mark as promoted in performance tracking
                metrics['promoted'] = True
                
                # Publish promotion event if available
                if self.event_bus:
                    event = Event(
                        event_type=EventType.STRATEGY_PROMOTED,
                        data={
                            'strategy_name': strategy_name,
                            'metrics': metrics
                        }
                    )
                    self.event_bus.publish(event)
    
    def get_strategy_compatibility_score(self, 
                                       strategy_name: str, 
                                       regime: MarketRegime) -> float:
        """
        Calculate compatibility score between a strategy and market regime.
        
        Args:
            strategy_name: Name of the strategy
            regime: Market regime
            
        Returns:
            Compatibility score (0-1)
        """
        metadata = StrategyRegistry.get_strategy_metadata(strategy_name)
        
        # Get compatibility scores from metadata
        regime_scores = metadata.get('regime_compatibility_scores', {})
        
        # Default compatibility if not specified
        default_compatibility = 0.5
        
        # Return compatibility score for the regime
        return regime_scores.get(regime.value, default_compatibility)
