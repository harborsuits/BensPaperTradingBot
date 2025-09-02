"""
Modular Strategy Integration

Integration module that connects the modular strategy system with the existing trading bot framework.
Provides adapters, factories, and configuration utilities for seamless integration.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import pandas as pd

from trading_bot.event_system.event_handler import EventHandlerManager
from trading_bot.strategies.base_strategy import Strategy, SignalType
from trading_bot.strategies.modular_strategy import ModularStrategy
from trading_bot.strategies.modular_strategy_system import (
    MarketCondition, ActivationCondition, ComponentType
)
from trading_bot.strategies.components.component_registry import get_component_registry
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.risk.margin_manager import MarginManager

logger = logging.getLogger(__name__)

class ModularStrategyFactory:
    """
    Factory for creating modular strategy instances from configuration.
    
    Acts as an adapter between the existing strategy factory and the modular strategy system.
    """
    
    @staticmethod
    def create_from_config(config: Dict[str, Any], 
                          event_handler_manager: Optional[EventHandlerManager] = None,
                          risk_manager: Optional[RiskManager] = None,
                          margin_manager: Optional[MarginManager] = None) -> ModularStrategy:
        """
        Create a modular strategy from configuration.
        
        Args:
            config: Strategy configuration
            event_handler_manager: Event handler manager instance
            risk_manager: Risk manager instance
            margin_manager: Margin manager instance
            
        Returns:
            ModularStrategy instance
        """
        # Get component registry
        registry = get_component_registry()
        
        # Extract strategy configuration
        strategy_id = config.get('strategy_id', 'modular_strategy')
        strategy_name = config.get('strategy_name', 'Modular Strategy')
        strategy_description = config.get('description', 'A modular trading strategy')
        
        # Create modular strategy
        strategy = ModularStrategy(
            strategy_id=strategy_id,
            name=strategy_name,
            description=strategy_description,
            event_handler_manager=event_handler_manager,
            risk_manager=risk_manager,
            margin_manager=margin_manager
        )
        
        # Set up activation conditions
        activation_config = config.get('activation_conditions', [])
        for act_cfg in activation_config:
            condition_type = act_cfg.get('type')
            if not condition_type:
                continue
                
            params = act_cfg.get('parameters', {})
            strategy.add_activation_condition(condition_type, **params)
        
        # Add components
        components_config = config.get('components', {})
        
        # Signal generators
        for sg_config in components_config.get('signal_generators', []):
            class_name = sg_config.get('class')
            if not class_name:
                continue
                
            params = sg_config.get('parameters', {})
            signal_generator = registry.create_component(
                ComponentType.SIGNAL_GENERATOR, class_name, params
            )
            strategy.add_signal_generator(signal_generator)
        
        # Filters
        for filter_config in components_config.get('filters', []):
            class_name = filter_config.get('class')
            if not class_name:
                continue
                
            params = filter_config.get('parameters', {})
            filter_component = registry.create_component(
                ComponentType.FILTER, class_name, params
            )
            strategy.add_filter(filter_component)
        
        # Position sizers
        for ps_config in components_config.get('position_sizers', []):
            class_name = ps_config.get('class')
            if not class_name:
                continue
                
            params = ps_config.get('parameters', {})
            position_sizer = registry.create_component(
                ComponentType.POSITION_SIZER, class_name, params
            )
            strategy.add_position_sizer(position_sizer)
        
        # Exit managers
        for exit_config in components_config.get('exit_managers', []):
            class_name = exit_config.get('class')
            if not class_name:
                continue
                
            params = exit_config.get('parameters', {})
            exit_manager = registry.create_component(
                ComponentType.EXIT_MANAGER, class_name, params
            )
            strategy.add_exit_manager(exit_manager)
        
        # Set up market conditions
        market_conditions_config = config.get('market_conditions', {})
        for symbol, conditions in market_conditions_config.items():
            for condition in conditions:
                condition_enum = MarketCondition[condition]
                strategy.set_market_condition(symbol, condition_enum)
        
        # Additional config
        log_signals = config.get('log_signals', True)
        strategy.set_log_signals(log_signals)
        
        performance_tracking = config.get('performance_tracking', True)
        strategy.set_performance_tracking(performance_tracking)
        
        return strategy
    
    @staticmethod
    def create_default_strategy(strategy_id: str = 'default_modular_strategy',
                              event_handler_manager: Optional[EventHandlerManager] = None,
                              risk_manager: Optional[RiskManager] = None,
                              margin_manager: Optional[MarginManager] = None) -> ModularStrategy:
        """
        Create a default modular strategy with reasonable components.
        
        Args:
            strategy_id: Strategy ID
            event_handler_manager: Event handler manager instance
            risk_manager: Risk manager instance
            margin_manager: Margin manager instance
            
        Returns:
            ModularStrategy instance
        """
        # Get component registry
        registry = get_component_registry()
        
        # Create modular strategy
        strategy = ModularStrategy(
            strategy_id=strategy_id,
            name='Default Modular Strategy',
            description='A default modular strategy with reasonable components',
            event_handler_manager=event_handler_manager,
            risk_manager=risk_manager,
            margin_manager=margin_manager
        )
        
        # Add a time-based activation condition (market hours)
        strategy.add_activation_condition(
            'time',
            time_zone='America/New_York',
            start_time='09:30',
            end_time='16:00',
            days_of_week=[0, 1, 2, 3, 4]  # Monday to Friday
        )
        
        # Add signal generators
        macd_generator = registry.create_component(
            ComponentType.SIGNAL_GENERATOR,
            'MacdSignalGenerator',
            {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        )
        
        rsi_generator = registry.create_component(
            ComponentType.SIGNAL_GENERATOR,
            'RSISignalGenerator',
            {'period': 14, 'overbought': 70, 'oversold': 30}
        )
        
        # Composite signal generator
        composite_generator = registry.create_component(
            ComponentType.SIGNAL_GENERATOR,
            'CompositeSignalGenerator',
            {'threshold': 0.6}
        )
        
        # Add generators to composite
        composite_generator.add_generator(macd_generator, 0.6)
        composite_generator.add_generator(rsi_generator, 0.4)
        
        # Add composite generator to strategy
        strategy.add_signal_generator(composite_generator)
        
        # Add filters
        volume_filter = registry.create_component(
            ComponentType.FILTER,
            'VolumeFilter',
            {'min_volume_percentile': 40.0, 'lookback_period': 20}
        )
        
        trend_filter = registry.create_component(
            ComponentType.FILTER,
            'TrendFilter',
            {'trend_period': 50, 'uptrend_threshold': 0.0, 'downtrend_threshold': 0.0}
        )
        
        # Add filters to strategy
        strategy.add_filter(volume_filter)
        strategy.add_filter(trend_filter)
        
        # Add position sizer
        position_sizer = registry.create_component(
            ComponentType.POSITION_SIZER,
            'FixedRiskPositionSizer',
            {'risk_per_trade': 1.0, 'max_position_size': 5.0}
        )
        
        strategy.add_position_sizer(position_sizer)
        
        # Add exit manager
        trailing_stop = registry.create_component(
            ComponentType.EXIT_MANAGER,
            'TrailingStopExitManager',
            {
                'atr_period': 14,
                'initial_stop_multiplier': 3.0,
                'trailing_stop_multiplier': 2.0
            }
        )
        
        time_exit = registry.create_component(
            ComponentType.EXIT_MANAGER,
            'TimeBasedExitManager',
            {'max_days_in_trade': 10, 'market_close_exit': True}
        )
        
        # Composite exit manager
        composite_exit = registry.create_component(
            ComponentType.EXIT_MANAGER,
            'CompositeExitManager',
            {'require_confirmation': False}
        )
        
        # Add exit managers to composite
        composite_exit.add_exit_manager(trailing_stop)
        composite_exit.add_exit_manager(time_exit)
        
        # Add composite exit manager to strategy
        strategy.add_exit_manager(composite_exit)
        
        return strategy

class StrategyAdapter:
    """
    Adapter for converting between modular strategies and traditional strategies.
    
    Allows modular strategies to be used in the existing framework and vice versa.
    """
    
    @staticmethod
    def adapt_modular_to_traditional(modular_strategy: ModularStrategy) -> Strategy:
        """
        Adapt a modular strategy to a traditional strategy interface.
        
        Args:
            modular_strategy: ModularStrategy instance
            
        Returns:
            Strategy instance
        """
        # Create wrapper class
        class ModularStrategyWrapper(Strategy):
            def __init__(self, modular_strategy):
                super().__init__(strategy_id=modular_strategy.strategy_id)
                self.modular_strategy = modular_strategy
                self.name = modular_strategy.name
                self.description = modular_strategy.description
            
            def generate_signal(self, symbol, data, context=None):
                # Process signals through modular strategy
                signals = self.modular_strategy.generate_signals(
                    {symbol: data}, context or {}
                )
                
                # Return signal for this symbol
                return signals.get(symbol, SignalType.FLAT)
            
            def calculate_position_size(self, symbol, signal, data, account_value, context=None):
                # Process position sizing through modular strategy
                signals = {symbol: signal}
                position_sizes = self.modular_strategy.calculate_position_sizes(
                    signals, {symbol: data}, context or {'account_value': account_value}
                )
                
                # Return position size for this symbol
                return position_sizes.get(symbol, 0.0)
            
            def should_exit_position(self, symbol, position, data, context=None):
                # Process exits through modular strategy
                positions = {symbol: position}
                exits = self.modular_strategy.calculate_exits(
                    positions, {symbol: data}, context or {}
                )
                
                # Return exit decision for this symbol
                return exits.get(symbol, False)
            
            def is_active(self, context=None):
                return self.modular_strategy.is_active(context or {})
        
        # Create and return wrapper instance
        return ModularStrategyWrapper(modular_strategy)
    
    @staticmethod
    def adapt_traditional_to_modular(strategy: Strategy,
                                   event_handler_manager: Optional[EventHandlerManager] = None,
                                   risk_manager: Optional[RiskManager] = None,
                                   margin_manager: Optional[MarginManager] = None) -> ModularStrategy:
        """
        Adapt a traditional strategy to a modular strategy interface.
        
        Args:
            strategy: Strategy instance
            event_handler_manager: Event handler manager instance
            risk_manager: Risk manager instance
            margin_manager: Margin manager instance
            
        Returns:
            ModularStrategy instance
        """
        # Create a wrapper modular strategy
        modular_strategy = ModularStrategy(
            strategy_id=strategy.strategy_id,
            name=getattr(strategy, 'name', strategy.strategy_id),
            description=getattr(strategy, 'description', 'Adapted traditional strategy'),
            event_handler_manager=event_handler_manager,
            risk_manager=risk_manager,
            margin_manager=margin_manager
        )
        
        # Create a custom signal generator to wrap the traditional strategy
        class TraditionalStrategySignalGenerator:
            def __init__(self, strategy):
                self.strategy = strategy
                self.component_id = f"traditional_{strategy.strategy_id}"
                self.component_type = ComponentType.SIGNAL_GENERATOR
                self.parameters = {}
                self.description = f"Traditional {strategy.strategy_id}"
            
            def generate_signals(self, data, context):
                signals = {}
                
                for symbol, df in data.items():
                    signal = self.strategy.generate_signal(symbol, df, context)
                    signals[symbol] = signal
                
                return signals
        
        # Create a custom position sizer to wrap the traditional strategy
        class TraditionalStrategyPositionSizer:
            def __init__(self, strategy):
                self.strategy = strategy
                self.component_id = f"traditional_sizer_{strategy.strategy_id}"
                self.component_type = ComponentType.POSITION_SIZER
                self.parameters = {}
                self.description = f"Traditional {strategy.strategy_id} Sizer"
            
            def calculate_position_sizes(self, signals, data, context):
                position_sizes = {}
                account_value = context.get('account_value', 0)
                
                for symbol, signal in signals.items():
                    if symbol not in data:
                        position_sizes[symbol] = 0.0
                        continue
                        
                    df = data[symbol]
                    
                    # Call traditional strategy for position sizing
                    size = self.strategy.calculate_position_size(
                        symbol, signal, df, account_value, context
                    )
                    
                    position_sizes[symbol] = size
                
                return position_sizes
        
        # Create a custom exit manager to wrap the traditional strategy
        class TraditionalStrategyExitManager:
            def __init__(self, strategy):
                self.strategy = strategy
                self.component_id = f"traditional_exit_{strategy.strategy_id}"
                self.component_type = ComponentType.EXIT_MANAGER
                self.parameters = {}
                self.description = f"Traditional {strategy.strategy_id} Exit"
            
            def calculate_exits(self, positions, data, context):
                exits = {}
                
                for symbol, position in positions.items():
                    if symbol not in data:
                        exits[symbol] = False
                        continue
                        
                    df = data[symbol]
                    
                    # Call traditional strategy for exit decision
                    exit_decision = self.strategy.should_exit_position(
                        symbol, position, df, context
                    )
                    
                    exits[symbol] = exit_decision
                
                return exits
        
        # Create component instances
        signal_generator = TraditionalStrategySignalGenerator(strategy)
        position_sizer = TraditionalStrategyPositionSizer(strategy)
        exit_manager = TraditionalStrategyExitManager(strategy)
        
        # Add components to modular strategy
        modular_strategy.add_signal_generator(signal_generator)
        modular_strategy.add_position_sizer(position_sizer)
        modular_strategy.add_exit_manager(exit_manager)
        
        # Add activation condition based on traditional strategy's is_active method
        class TraditionalStrategyActivationCondition(ActivationCondition):
            def __init__(self, strategy):
                self.strategy = strategy
            
            def is_active(self, context):
                return self.strategy.is_active(context)
        
        modular_strategy.activation_conditions.append(
            TraditionalStrategyActivationCondition(strategy)
        )
        
        return modular_strategy

class StrategyConfigGenerator:
    """
    Utility for generating and managing modular strategy configurations.
    
    Provides methods for creating, saving, and loading strategy configurations.
    """
    
    @staticmethod
    def generate_config_from_strategy(strategy: ModularStrategy) -> Dict[str, Any]:
        """
        Generate a configuration dictionary from a modular strategy.
        
        Args:
            strategy: ModularStrategy instance
            
        Returns:
            Configuration dictionary
        """
        registry = get_component_registry()
        
        config = {
            'strategy_id': strategy.strategy_id,
            'strategy_name': strategy.name,
            'description': strategy.description,
            'components': {
                'signal_generators': [],
                'filters': [],
                'position_sizers': [],
                'exit_managers': []
            },
            'activation_conditions': [],
            'market_conditions': {},
            'log_signals': strategy.log_signals,
            'performance_tracking': strategy.performance_tracking
        }
        
        # Serialize signal generators
        for sg in strategy.signal_generators:
            if hasattr(sg, 'component_id'):
                serialized = registry.serialize_component(sg)
                config['components']['signal_generators'].append({
                    'class': serialized['class_name'],
                    'parameters': serialized['parameters']
                })
        
        # Serialize filters
        for filter_comp in strategy.filters:
            if hasattr(filter_comp, 'component_id'):
                serialized = registry.serialize_component(filter_comp)
                config['components']['filters'].append({
                    'class': serialized['class_name'],
                    'parameters': serialized['parameters']
                })
        
        # Serialize position sizers
        for ps in strategy.position_sizers:
            if hasattr(ps, 'component_id'):
                serialized = registry.serialize_component(ps)
                config['components']['position_sizers'].append({
                    'class': serialized['class_name'],
                    'parameters': serialized['parameters']
                })
        
        # Serialize exit managers
        for em in strategy.exit_managers:
            if hasattr(em, 'component_id'):
                serialized = registry.serialize_component(em)
                config['components']['exit_managers'].append({
                    'class': serialized['class_name'],
                    'parameters': serialized['parameters']
                })
        
        # Serialize activation conditions
        for condition in strategy.activation_conditions:
            if hasattr(condition, 'type'):
                config['activation_conditions'].append({
                    'type': condition.type,
                    'parameters': condition.parameters
                })
        
        # Serialize market conditions
        for symbol, conditions in strategy.market_conditions.items():
            config['market_conditions'][symbol] = [c.name for c in conditions]
        
        return config
    
    @staticmethod
    def save_strategy_config(strategy: ModularStrategy, file_path: str) -> bool:
        """
        Save a strategy configuration to a JSON file.
        
        Args:
            strategy: ModularStrategy instance
            file_path: Path to save the configuration
            
        Returns:
            True if the configuration was saved successfully, False otherwise
        """
        try:
            config = StrategyConfigGenerator.generate_config_from_strategy(strategy)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Strategy configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving strategy configuration: {e}")
            return False
    
    @staticmethod
    def load_strategy_config(file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a strategy configuration from a JSON file.
        
        Args:
            file_path: Path to load the configuration from
            
        Returns:
            Configuration dictionary or None if loading failed
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Strategy configuration file {file_path} does not exist")
                return None
            
            # Read from file
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Strategy configuration loaded from {file_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading strategy configuration: {e}")
            return None
