#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sector Rotation Strategy Factory

This module provides the factory class for creating and configuring
SectorRotationStrategy instances with various presets and customization options.

The factory supports creating strategies based on different rotation models
(economic cycle, momentum, relative strength) with appropriate configuration
parameters.
"""

import logging
from typing import Dict, Any, Optional, List, Union

from trading_bot.strategies_new.stocks.sector.sector_rotation_strategy import SectorRotationStrategy
from trading_bot.core.events.event_bus import EventBus

logger = logging.getLogger(__name__)


class SectorStrategyFactory:
    """
    Factory class for creating and configuring SectorRotationStrategy instances.
    """

    @classmethod
    def create_economic_cycle_strategy(cls, 
                                      event_bus: Optional[EventBus] = None,
                                      rotation_frequency: str = 'monthly',
                                      gradual_rotation: bool = True,
                                      defensive_threshold: float = 0.7,
                                      cycle_prediction_window: int = 6,
                                      **kwargs) -> SectorRotationStrategy:
        """
        Create a sector rotation strategy based on economic cycle forecasting.
        
        Args:
            event_bus: Optional event bus for strategy events
            rotation_frequency: How often to rotate ('weekly', 'monthly', 'quarterly')
            gradual_rotation: Whether to gradually rotate or all at once
            defensive_threshold: Threshold for switching to defensive positioning
            cycle_prediction_window: Months ahead to predict economic cycle
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured SectorRotationStrategy instance
        """
        parameters = {
            'rotation_model': 'economic_cycle',
            'rotation_frequency': rotation_frequency,
            'rebalance_threshold': 0.1,  # 10% allocation change triggers rebalance
            'gradual_rotation': gradual_rotation,
            'rotation_steps': 3 if gradual_rotation else 1,
            'defensive_threshold': defensive_threshold,
            'cycle_prediction_window': cycle_prediction_window,
            'trade_sector_etfs': True,
            'use_leading_indicators': True,
            'use_yield_curve': True,
            'use_pmi_data': True,
            'use_employment_data': True,
            'use_inflation_data': True,
            'cycle_forecast_confidence_threshold': 0.65,
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating economic cycle sector rotation strategy with {rotation_frequency} rotation")
        
        return SectorRotationStrategy(
            parameters=parameters,
            event_bus=event_bus
        )

    @classmethod
    def create_momentum_strategy(cls,
                               event_bus: Optional[EventBus] = None,
                               lookback_period: int = 3,
                               rotation_frequency: str = 'monthly',
                               top_n_sectors: int = 3,
                               momentum_smoothing: int = 5,
                               use_volume: bool = True,
                               **kwargs) -> SectorRotationStrategy:
        """
        Create a sector rotation strategy based on momentum factors.
        
        Args:
            event_bus: Optional event bus for strategy events
            lookback_period: Months of lookback for momentum calculation
            rotation_frequency: How often to rotate ('weekly', 'monthly', 'quarterly')
            top_n_sectors: Number of top sectors to include
            momentum_smoothing: Period for smoothing momentum signals
            use_volume: Whether to incorporate volume in momentum calculation
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured SectorRotationStrategy instance
        """
        parameters = {
            'rotation_model': 'momentum',
            'rotation_frequency': rotation_frequency,
            'rebalance_threshold': 0.15,  # 15% allocation change triggers rebalance
            'gradual_rotation': True,
            'rotation_steps': 2,
            'defensive_threshold': 0.6,
            'lookback_period': lookback_period,
            'momentum_smoothing': momentum_smoothing,
            'top_n_sectors': top_n_sectors,
            'equal_weight': False,  # Weight by momentum strength
            'trade_sector_etfs': True,
            'use_volume': use_volume,
            'volatility_adjusted': True,
            'use_turnover_filter': True,
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating momentum sector rotation strategy with {lookback_period}-month lookback")
        
        return SectorRotationStrategy(
            parameters=parameters,
            event_bus=event_bus
        )

    @classmethod
    def create_relative_strength_strategy(cls,
                                        event_bus: Optional[EventBus] = None,
                                        base_index: str = 'SPY',
                                        lookback_period: int = 3,
                                        rotation_frequency: str = 'monthly',
                                        min_sectors: int = 2,
                                        max_sectors: int = 5,
                                        rs_smoothing: int = 10,
                                        **kwargs) -> SectorRotationStrategy:
        """
        Create a sector rotation strategy based on relative strength to a benchmark.
        
        Args:
            event_bus: Optional event bus for strategy events
            base_index: Ticker of the benchmark index
            lookback_period: Months of lookback for RS calculation
            rotation_frequency: How often to rotate ('weekly', 'monthly', 'quarterly')
            min_sectors: Minimum number of sectors to include
            max_sectors: Maximum number of sectors to include
            rs_smoothing: Period for smoothing RS metrics
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured SectorRotationStrategy instance
        """
        parameters = {
            'rotation_model': 'relative_strength',
            'rotation_frequency': rotation_frequency,
            'rebalance_threshold': 0.12,  # 12% allocation change triggers rebalance
            'gradual_rotation': True,
            'rotation_steps': 2,
            'defensive_threshold': 0.65,
            'lookback_period': lookback_period,
            'rs_smoothing': rs_smoothing,
            'base_index': base_index,
            'min_sectors': min_sectors,
            'max_sectors': max_sectors,
            'equal_weight': False,  # Weight by RS ratio
            'trade_sector_etfs': True,
            'use_rs_ratio': True,
            'use_rs_momentum': True,
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating relative strength sector rotation strategy vs {base_index}")
        
        return SectorRotationStrategy(
            parameters=parameters,
            event_bus=event_bus
        )

    @classmethod
    def create_adaptive_strategy(cls,
                               event_bus: Optional[EventBus] = None,
                               max_sectors: int = 5,
                               rotation_frequency: str = 'monthly',
                               volatility_window: int = 20,
                               market_regime_window: int = 60,
                               **kwargs) -> SectorRotationStrategy:
        """
        Create an adaptive sector rotation strategy that combines multiple
        models and adapts to market regimes.
        
        Args:
            event_bus: Optional event bus for strategy events
            max_sectors: Maximum number of sectors to include
            rotation_frequency: How often to rotate ('weekly', 'monthly', 'quarterly')
            volatility_window: Window for volatility calculation
            market_regime_window: Window for market regime detection
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured SectorRotationStrategy instance
        """
        parameters = {
            'rotation_model': 'adaptive',
            'rotation_frequency': rotation_frequency,
            'rebalance_threshold': 0.1,  # 10% allocation change triggers rebalance
            'gradual_rotation': True,
            'rotation_steps': 3,
            'defensive_threshold': 0.6,
            'max_sectors': max_sectors,
            'lookback_period': 3,  # 3 months lookback for momentum/RS
            'use_economic_cycle': True,
            'use_momentum': True,
            'use_relative_strength': True,
            'market_regime_window': market_regime_window,
            'volatility_window': volatility_window,
            'combine_method': 'weighted_average',  # How to combine rankings
            'weight_economic': 0.4,
            'weight_momentum': 0.3,
            'weight_rs': 0.3,
            'trade_sector_etfs': True,
            'adapt_to_volatility': True,
            'adapt_to_correlations': True,
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info(f"Creating adaptive sector rotation strategy with {rotation_frequency} rotation")
        
        return SectorRotationStrategy(
            parameters=parameters,
            event_bus=event_bus
        )

    @classmethod
    def create_conservative_strategy(cls, 
                                   event_bus: Optional[EventBus] = None,
                                   **kwargs) -> SectorRotationStrategy:
        """
        Create a conservative sector rotation strategy with lower turnover
        and higher defensive positioning threshold.
        
        Args:
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured SectorRotationStrategy instance
        """
        # Start with an economic cycle strategy as base
        parameters = {
            'rotation_model': 'economic_cycle',
            'rotation_frequency': 'quarterly',  # Less frequent rotation
            'rebalance_threshold': 0.2,  # Higher threshold to reduce turnover
            'gradual_rotation': True,
            'rotation_steps': 4,  # More gradual rotation
            'defensive_threshold': 0.5,  # More sensitive to defensive positioning
            'max_sectors': 3,  # Fewer sectors for concentration
            'use_stop_loss': True,
            'stop_loss_pct': 0.05,  # 5% stop loss
            'trade_sector_etfs': True,
            'economic_cycle_emphasis': 'late_expansion',  # Focus on more stable late cycle sectors
            'use_volatility_filter': True,  # Filter high volatility sectors
            'max_sector_volatility': 0.2,  # 20% max volatility
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info("Creating conservative sector rotation strategy")
        
        return SectorRotationStrategy(
            parameters=parameters,
            event_bus=event_bus
        )

    @classmethod
    def create_aggressive_strategy(cls,
                                 event_bus: Optional[EventBus] = None,
                                 **kwargs) -> SectorRotationStrategy:
        """
        Create an aggressive sector rotation strategy with higher turnover
        and focus on momentum.
        
        Args:
            event_bus: Optional event bus for strategy events
            **kwargs: Additional parameters to override defaults
        
        Returns:
            Configured SectorRotationStrategy instance
        """
        # Start with a momentum strategy as base
        parameters = {
            'rotation_model': 'momentum',
            'rotation_frequency': 'biweekly',  # More frequent rotation
            'rebalance_threshold': 0.08,  # Lower threshold for more activity
            'gradual_rotation': False,  # Immediate rotation
            'defensive_threshold': 0.8,  # Less sensitive to defensive positioning
            'lookback_period': 1,  # Shorter lookback (1 month)
            'top_n_sectors': 2,  # Focus on top performers only
            'equal_weight': False,
            'trade_sector_etfs': True,
            'use_leveraged_etfs': True,  # Use leveraged sector ETFs
            'leverage_factor': 2.0,  # 2x leverage
            'use_volatility_filter': False,  # Don't filter high volatility
        }
        
        # Override defaults with any provided kwargs
        parameters.update(kwargs)
        
        logger.info("Creating aggressive sector rotation strategy")
        
        return SectorRotationStrategy(
            parameters=parameters,
            event_bus=event_bus
        )

    @classmethod
    def create_strategy_from_config(cls,
                                  config: Dict[str, Any],
                                  event_bus: Optional[EventBus] = None) -> SectorRotationStrategy:
        """
        Create a sector rotation strategy from a configuration dictionary.
        
        Args:
            config: Strategy configuration dictionary
            event_bus: Optional event bus for strategy events
        
        Returns:
            Configured SectorRotationStrategy instance
        """
        # Determine which factory method to use based on config
        strategy_type = config.get('strategy_type', 'economic_cycle')
        
        # Extract common kwargs
        kwargs = {k: v for k, v in config.items() if k != 'strategy_type'}
        
        # Create the appropriate strategy type
        if strategy_type == 'momentum':
            return cls.create_momentum_strategy(event_bus=event_bus, **kwargs)
        elif strategy_type == 'relative_strength':
            return cls.create_relative_strength_strategy(event_bus=event_bus, **kwargs)
        elif strategy_type == 'adaptive':
            return cls.create_adaptive_strategy(event_bus=event_bus, **kwargs)
        elif strategy_type == 'conservative':
            return cls.create_conservative_strategy(event_bus=event_bus, **kwargs)
        elif strategy_type == 'aggressive':
            return cls.create_aggressive_strategy(event_bus=event_bus, **kwargs)
        else:
            # Default to economic cycle
            return cls.create_economic_cycle_strategy(event_bus=event_bus, **kwargs)
