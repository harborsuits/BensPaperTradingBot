"""
Modular Strategy Implementation

Main component of the modular strategy system that provides:
- Composition of strategy components
- Conditional activation based on market conditions
- Integration with the existing strategy framework
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Type, Callable
import logging
import time
from datetime import datetime
import uuid
import copy
import json

from trading_bot.strategies.base_strategy import Strategy, SignalType, Position
from trading_bot.strategies.modular_strategy_system import (
    StrategyComponent, ComponentType, MarketCondition, ActivationRule,
    SignalGeneratorComponent, FilterComponent, PositionSizerComponent, ExitManagerComponent
)
from trading_bot.event_system import Event, EventBus

logger = logging.getLogger(__name__)

class ComponentRegistry:
    """Registry for strategy components"""
    
    def __init__(self):
        """Initialize component registry"""
        self.components: Dict[str, StrategyComponent] = {}
        self.component_types: Dict[ComponentType, List[str]] = {ct: [] for ct in ComponentType}
    
    def register_component(self, component: StrategyComponent) -> None:
        """
        Register a strategy component
        
        Args:
            component: Component to register
        """
        if component.component_id in self.components:
            logger.warning(f"Component with ID {component.component_id} already registered. Replacing.")
        
        self.components[component.component_id] = component
        self.component_types[component.component_type].append(component.component_id)
        logger.debug(f"Registered component: {component.component_id} ({component.component_type.name})")
    
    def get_component(self, component_id: str) -> Optional[StrategyComponent]:
        """
        Get a component by ID
        
        Args:
            component_id: Component ID
            
        Returns:
            Component or None if not found
        """
        return self.components.get(component_id)
    
    def get_components_by_type(self, component_type: ComponentType) -> List[StrategyComponent]:
        """
        Get all components of a specific type
        
        Args:
            component_type: Component type
            
        Returns:
            List of components
        """
        return [self.components[cid] for cid in self.component_types[component_type]]
    
    def remove_component(self, component_id: str) -> bool:
        """
        Remove a component from the registry
        
        Args:
            component_id: Component ID
            
        Returns:
            True if removed, False if not found
        """
        if component_id not in self.components:
            return False
        
        component = self.components[component_id]
        del self.components[component_id]
        self.component_types[component.component_type].remove(component_id)
        
        logger.debug(f"Removed component: {component_id}")
        return True
    
    def clear(self) -> None:
        """Clear all registered components"""
        self.components.clear()
        self.component_types = {ct: [] for ct in ComponentType}
        logger.debug("Cleared component registry")

class ActivationCondition:
    """Represents a condition for strategy activation"""
    
    def __init__(self, 
                rule_type: ActivationRule, 
                parameters: Dict[str, Any], 
                description: str = ""):
        """
        Initialize activation condition
        
        Args:
            rule_type: Type of activation rule
            parameters: Parameters for the rule
            description: Human-readable description
        """
        self.rule_type = rule_type
        self.parameters = parameters
        self.description = description
        self.last_evaluation = False
        self.last_evaluation_time = None
    
    def evaluate(self, market_data: Dict[str, pd.DataFrame], context: Dict[str, Any]) -> bool:
        """
        Evaluate the condition
        
        Args:
            market_data: Market data dictionary
            context: Evaluation context
            
        Returns:
            True if condition is met, False otherwise
        """
        self.last_evaluation_time = datetime.now()
        
        # ALWAYS activation rule is always True
        if self.rule_type == ActivationRule.ALWAYS:
            self.last_evaluation = True
            return True
        
        # TIME_BASED activation
        elif self.rule_type == ActivationRule.TIME_BASED:
            current_time = context.get('current_time', datetime.now())
            
            # Check day of week
            if 'days_of_week' in self.parameters:
                allowed_days = self.parameters['days_of_week']
                if current_time.weekday() not in allowed_days:
                    self.last_evaluation = False
                    return False
            
            # Check time range
            if 'start_time' in self.parameters and 'end_time' in self.parameters:
                start_time = self.parameters['start_time']  # Format: "HH:MM"
                end_time = self.parameters['end_time']      # Format: "HH:MM"
                
                start_hour, start_min = map(int, start_time.split(':'))
                end_hour, end_min = map(int, end_time.split(':'))
                
                current_hour, current_min = current_time.hour, current_time.minute
                
                # Convert to minutes for comparison
                start_minutes = start_hour * 60 + start_min
                end_minutes = end_hour * 60 + end_min
                current_minutes = current_hour * 60 + current_min
                
                if not (start_minutes <= current_minutes <= end_minutes):
                    self.last_evaluation = False
                    return False
            
            self.last_evaluation = True
            return True
            
        # CONDITION_BASED activation
        elif self.rule_type == ActivationRule.CONDITION_BASED:
            required_conditions = self.parameters.get('conditions', [])
            detected_conditions = context.get('market_conditions', [])
            
            # Check if all required conditions are met
            for condition in required_conditions:
                if condition not in detected_conditions:
                    self.last_evaluation = False
                    return False
            
            self.last_evaluation = True
            return True
            
        # INDICATOR_BASED activation
        elif self.rule_type == ActivationRule.INDICATOR_BASED:
            indicators = self.parameters.get('indicators', {})
            symbol = self.parameters.get('symbol', list(market_data.keys())[0] if market_data else None)
            
            if not symbol or symbol not in market_data:
                self.last_evaluation = False
                return False
            
            # Check all indicator conditions
            for indicator_name, condition in indicators.items():
                if indicator_name not in market_data[symbol]:
                    self.last_evaluation = False
                    return False
                
                indicator_value = market_data[symbol][indicator_name].iloc[-1]
                threshold = condition.get('threshold')
                
                if 'operator' in condition:
                    operator = condition['operator']
                    
                    if operator == '>' and not (indicator_value > threshold):
                        self.last_evaluation = False
                        return False
                    elif operator == '>=' and not (indicator_value >= threshold):
                        self.last_evaluation = False
                        return False
                    elif operator == '<' and not (indicator_value < threshold):
                        self.last_evaluation = False
                        return False
                    elif operator == '<=' and not (indicator_value <= threshold):
                        self.last_evaluation = False
                        return False
                    elif operator == '==' and not (indicator_value == threshold):
                        self.last_evaluation = False
                        return False
                    elif operator == '!=' and not (indicator_value != threshold):
                        self.last_evaluation = False
                        return False
            
            self.last_evaluation = True
            return True
            
        # PERFORMANCE_BASED activation
        elif self.rule_type == ActivationRule.PERFORMANCE_BASED:
            # Get performance metrics from context
            performance_metrics = context.get('performance_metrics', {})
            
            # Check performance thresholds
            thresholds = self.parameters.get('thresholds', {})
            for metric_name, condition in thresholds.items():
                if metric_name not in performance_metrics:
                    # Skip if metric not available
                    continue
                
                metric_value = performance_metrics[metric_name]
                threshold = condition.get('threshold', 0)
                operator = condition.get('operator', '>')
                
                if operator == '>' and not (metric_value > threshold):
                    self.last_evaluation = False
                    return False
                elif operator == '>=' and not (metric_value >= threshold):
                    self.last_evaluation = False
                    return False
                elif operator == '<' and not (metric_value < threshold):
                    self.last_evaluation = False
                    return False
                elif operator == '<=' and not (metric_value <= threshold):
                    self.last_evaluation = False
                    return False
            
            self.last_evaluation = True
            return True
            
        # HYBRID activation (combination of other conditions)
        elif self.rule_type == ActivationRule.HYBRID:
            subconditions = self.parameters.get('subconditions', [])
            operator = self.parameters.get('operator', 'AND')
            
            if not subconditions:
                self.last_evaluation = False
                return False
            
            # Evaluate all subconditions
            results = []
            for subcond in subconditions:
                # Recursive evaluation of subconditions
                subcond_obj = ActivationCondition(
                    subcond['rule_type'], 
                    subcond['parameters'], 
                    subcond.get('description', '')
                )
                results.append(subcond_obj.evaluate(market_data, context))
            
            # Combine results based on operator
            if operator == 'AND':
                self.last_evaluation = all(results)
            elif operator == 'OR':
                self.last_evaluation = any(results)
            else:
                logger.warning(f"Unknown operator in hybrid condition: {operator}")
                self.last_evaluation = False
            
            return self.last_evaluation
            
        # Unknown rule type
        else:
            logger.warning(f"Unknown activation rule type: {self.rule_type}")
            self.last_evaluation = False
            return False
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation
        
        Returns:
            Dictionary representation
        """
        return {
            'rule_type': self.rule_type.name,
            'parameters': self.parameters,
            'description': self.description,
            'last_evaluation': self.last_evaluation,
            'last_evaluation_time': self.last_evaluation_time.isoformat() if self.last_evaluation_time else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActivationCondition':
        """
        Create from dictionary representation
        
        Args:
            data: Dictionary representation
            
        Returns:
            ActivationCondition instance
        """
        rule_type = ActivationRule[data['rule_type']]
        parameters = data['parameters']
        description = data.get('description', '')
        
        condition = cls(rule_type, parameters, description)
        condition.last_evaluation = data.get('last_evaluation', False)
        
        if data.get('last_evaluation_time'):
            condition.last_evaluation_time = datetime.fromisoformat(data['last_evaluation_time'])
        
        return condition

class ModularStrategy(Strategy):
    """
    Modular, composable trading strategy that implements the EA31337-Libre approach.
    
    This strategy consists of multiple components that can be composed and configured
    to create sophisticated trading logic with conditional activation.
    """
    
    def __init__(self, 
                name: str, 
                symbols: List[str], 
                parameters: Optional[Dict[str, Any]] = None, 
                min_history_bars: int = 20):
        """
        Initialize modular strategy
        
        Args:
            name: Strategy name
            symbols: List of symbols to trade
            parameters: Strategy parameters
            min_history_bars: Minimum history bars required
        """
        super().__init__(name, symbols, parameters, min_history_bars)
        
        # Component registry
        self.registry = ComponentRegistry()
        
        # Signal chain (ordered list of signal generators and filters)
        self.signal_chain: List[str] = []
        
        # Position sizer components
        self.position_sizers: List[str] = []
        
        # Exit manager components
        self.exit_managers: List[str] = []
        
        # Activation conditions
        self.activation_conditions: List[ActivationCondition] = []
        self.is_active = True
        
        # Market condition detector components
        self.condition_detectors: List[str] = []
        self.detected_conditions: Dict[str, List[MarketCondition]] = {symbol: [] for symbol in symbols}
        
        # Processing context
        self.context: Dict[str, Any] = {
            'strategy_name': name,
            'symbols': symbols,
            'parameters': parameters or {},
            'current_time': datetime.now(),
            'market_conditions': {},
            'performance_metrics': {},
            'signals': {},
            'position_sizes': {},
            'exit_parameters': {}
        }
        
        # Configuration
        self.config = {
            'log_component_timing': True,
            'skip_disabled_components': True,
            'fail_on_component_error': False,
            'log_market_conditions': True,
            'parallel_component_execution': False  # For future implementation
        }
        
        # Event bus reference
        self.event_bus = None
    
    def add_component(self, component: StrategyComponent) -> None:
        """
        Add a component to the strategy
        
        Args:
            component: Component to add
        """
        # Set parent strategy reference
        component.set_parent_strategy(self)
        
        # Register component
        self.registry.register_component(component)
        
        # Add to appropriate list based on type
        if component.component_type == ComponentType.SIGNAL_GENERATOR:
            self.signal_chain.append(component.component_id)
        elif component.component_type == ComponentType.FILTER:
            self.signal_chain.append(component.component_id)
        elif component.component_type == ComponentType.POSITION_SIZER:
            self.position_sizers.append(component.component_id)
        elif component.component_type == ComponentType.EXIT_MANAGER:
            self.exit_managers.append(component.component_id)
        elif component.component_type == ComponentType.CONDITION_DETECTOR:
            self.condition_detectors.append(component.component_id)
        
        logger.info(f"Added component {component.component_id} to strategy {self.name}")
    
    def add_activation_condition(self, condition: ActivationCondition) -> None:
        """
        Add an activation condition
        
        Args:
            condition: Activation condition
        """
        self.activation_conditions.append(condition)
        logger.info(f"Added activation condition ({condition.rule_type.name}) to strategy {self.name}")
    
    def set_event_bus(self, event_bus: Any) -> None:
        """
        Set event bus for event publishing
        
        Args:
            event_bus: Event bus instance
        """
        self.event_bus = event_bus
    
    def is_strategy_active(self, market_data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> bool:
        """
        Check if strategy is active based on activation conditions
        
        Args:
            market_data: Market data dictionary
            current_time: Current timestamp
            
        Returns:
            True if strategy should be active, False otherwise
        """
        # Update context
        self.context['current_time'] = current_time
        self.context['performance_metrics'] = self.get_performance_metrics()
        
        # Detect market conditions first
        self._detect_market_conditions(market_data, current_time)
        
        # If no activation conditions are defined, strategy is always active
        if not self.activation_conditions:
            return True
        
        # Evaluate all activation conditions
        for condition in self.activation_conditions:
            if condition.evaluate(market_data, self.context):
                return True
        
        # No conditions met
        return False
    
    def _detect_market_conditions(self, market_data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> None:
        """
        Detect market conditions using condition detector components
        
        Args:
            market_data: Market data dictionary
            current_time: Current timestamp
        """
        # Clear previously detected conditions
        self.detected_conditions = {symbol: [] for symbol in self.symbols}
        
        # Process each condition detector
        for detector_id in self.condition_detectors:
            detector = self.registry.get_component(detector_id)
            if not detector or not detector.is_enabled():
                continue
            
            try:
                # Process detector
                detector_results = detector.process(market_data, self.context)
                
                # Update detected conditions
                for symbol, conditions in detector_results.items():
                    if symbol in self.detected_conditions:
                        self.detected_conditions[symbol].extend(conditions)
            except Exception as e:
                logger.error(f"Error in condition detector {detector_id}: {e}")
                if self.config['fail_on_component_error']:
                    raise
        
        # Update context with detected conditions
        self.context['market_conditions'] = self.detected_conditions
        
        # Log detected conditions if enabled
        if self.config['log_market_conditions']:
            for symbol, conditions in self.detected_conditions.items():
                if conditions:
                    condition_names = [c.name for c in conditions]
                    logger.debug(f"Detected conditions for {symbol}: {condition_names}")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, SignalType]:
        """
        Generate trading signals using component chain
        
        Args:
            data: Market data dictionary
            current_time: Current timestamp
            
        Returns:
            Dictionary of symbol -> signal type
        """
        # Check if strategy is active
        if not self.is_strategy_active(data, current_time):
            # Strategy is inactive, return flat signals for all symbols
            logger.debug(f"Strategy {self.name} is inactive at {current_time}")
            return {symbol: SignalType.FLAT for symbol in self.symbols}
        
        # Update context
        self.context['current_time'] = current_time
        
        # Initialize signals
        signals = {}
        
        # Process signal chain (generators and filters)
        for component_id in self.signal_chain:
            component = self.registry.get_component(component_id)
            if not component or not component.is_enabled():
                continue
            
            start_time = time.time()
            try:
                if component.component_type == ComponentType.SIGNAL_GENERATOR:
                    # Generate new signals
                    generator_signals = component.process(data, self.context)
                    
                    # Merge with existing signals, new signals take precedence
                    signals.update(generator_signals)
                    
                elif component.component_type == ComponentType.FILTER:
                    # Filter existing signals
                    signals = component.process((signals, data), self.context)
            except Exception as e:
                logger.error(f"Error in component {component_id}: {e}")
                if self.config['fail_on_component_error']:
                    raise
            
            if self.config['log_component_timing']:
                elapsed = time.time() - start_time
                logger.debug(f"Component {component_id} processed in {elapsed:.4f}s")
        
        # Update context with generated signals
        self.context['signals'] = signals
        
        # Publish signals event if event bus is available
        if self.event_bus:
            self._publish_signals_event(signals, current_time)
        
        return signals
    
    def calculate_position_size(self, symbol: str, signal: SignalType, price: float, 
                               volatility: float, account_size: float) -> float:
        """
        Calculate position size using position sizer components
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility
            account_size: Account size
            
        Returns:
            Position size (in units or contracts)
        """
        # Update context
        self.context['account_size'] = account_size
        self.context['volatility'] = {symbol: volatility}
        
        # Default position size (fallback)
        position_size = 0.0
        
        # Data for position sizer components
        sizer_data = {symbol: (signal, price)}
        
        # Process position sizers
        for sizer_id in self.position_sizers:
            sizer = self.registry.get_component(sizer_id)
            if not sizer or not sizer.is_enabled():
                continue
            
            try:
                # Process sizer
                sizes = sizer.process(sizer_data, self.context)
                
                # Get position size for this symbol
                if symbol in sizes:
                    position_size = sizes[symbol]
                    break  # Use first valid position size
            except Exception as e:
                logger.error(f"Error in position sizer {sizer_id}: {e}")
                if self.config['fail_on_component_error']:
                    raise
        
        # If no position sizers provided a size, use parent class implementation
        if position_size <= 0:
            position_size = super().calculate_position_size(
                symbol, signal, price, volatility, account_size
            )
        
        # Update context
        self.context['position_sizes'] = {symbol: position_size}
        
        return position_size
    
    def calculate_stop_loss(self, symbol: str, signal: SignalType, price: float, volatility: float) -> Optional[float]:
        """
        Calculate stop loss price using exit manager components
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility
            
        Returns:
            Stop loss price or None
        """
        # Exit managers are handled in calculate_exit_parameters
        # This is maintained for compatibility with base Strategy class
        
        # For now, just call parent implementation
        return super().calculate_stop_loss(symbol, signal, price, volatility)
    
    def calculate_take_profit(self, symbol: str, signal: SignalType, price: float, volatility: float) -> Optional[float]:
        """
        Calculate take profit price using exit manager components
        
        Args:
            symbol: Symbol to trade
            signal: Signal type
            price: Current price
            volatility: Symbol volatility
            
        Returns:
            Take profit price or None
        """
        # Exit managers are handled in calculate_exit_parameters
        # This is maintained for compatibility with base Strategy class
        
        # For now, just call parent implementation
        return super().calculate_take_profit(symbol, signal, price, volatility)
    
    def calculate_exit_parameters(self, symbol: str, position: Position, 
                                 data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate complete exit parameters using exit manager components
        
        Args:
            symbol: Symbol
            position: Current position
            data: Market data
            
        Returns:
            Dictionary with exit parameters
        """
        # Default exit parameters
        exit_params = {
            'stop_loss': position.stop_loss,
            'take_profit': position.take_profit,
            'trailing_stop': position.trailing_stop
        }
        
        # Data for exit manager components
        exit_data = {symbol: position}
        
        # Process exit managers
        for manager_id in self.exit_managers:
            manager = self.registry.get_component(manager_id)
            if not manager or not manager.is_enabled():
                continue
            
            try:
                # Process exit manager
                params = manager.process(exit_data, {**self.context, 'market_data': data})
                
                # Get parameters for this symbol
                if symbol in params:
                    exit_params.update(params[symbol])
                    break  # Use first valid exit parameters
            except Exception as e:
                logger.error(f"Error in exit manager {manager_id}: {e}")
                if self.config['fail_on_component_error']:
                    raise
        
        # Update context
        self.context['exit_parameters'] = exit_params
        
        return exit_params
    
    def update(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp, 
              account_size: float) -> Dict[str, Any]:
        """
        Update strategy with new data and generate orders
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current timestamp
            account_size: Current account size
            
        Returns:
            Dictionary with update results including any new orders
        """
        # First, check if strategy is active
        if not self.is_strategy_active(data, current_time):
            logger.debug(f"Strategy {self.name} is inactive, skipping update")
            return {
                "orders": [],
                "positions": list(self.positions.values()),
                "equity": account_size
            }
        
        # Then, use parent class implementation
        return super().update(data, current_time, account_size)
    
    def _publish_signals_event(self, signals: Dict[str, SignalType], timestamp: pd.Timestamp) -> None:
        """
        Publish signals as an event
        
        Args:
            signals: Generated signals
            timestamp: Signal timestamp
        """
        if not self.event_bus:
            return
        
        # Convert signals to serializable format
        serialized_signals = {
            symbol: signal.value for symbol, signal in signals.items()
        }
        
        # Create event
        event = Event(
            topic="trading_signals",
            data={
                'strategy': self.name,
                'signals': serialized_signals,
                'timestamp': timestamp.isoformat()
            },
            metadata={
                'type': 'SIGNAL',
                'strategy': self.name,
                'timestamp': timestamp.isoformat()
            }
        )
        
        # Publish event
        self.event_bus.publish(event)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert strategy to dictionary representation
        
        Returns:
            Dictionary representation
        """
        # Convert components
        components = {c_id: c.to_dict() for c_id, c in self.registry.components.items()}
        
        # Convert activation conditions
        conditions = [c.to_dict() for c in self.activation_conditions]
        
        return {
            'name': self.name,
            'symbols': self.symbols,
            'parameters': self.parameters,
            'min_history_bars': self.min_history_bars,
            'components': components,
            'signal_chain': self.signal_chain,
            'position_sizers': self.position_sizers,
            'exit_managers': self.exit_managers,
            'condition_detectors': self.condition_detectors,
            'activation_conditions': conditions,
            'config': self.config
        }
    
    def save_to_file(self, filepath: str) -> None:
        """
        Save strategy configuration to file
        
        Args:
            filepath: File path
        """
        # Convert to dictionary
        data = self.to_dict()
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Strategy saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ModularStrategy':
        """
        Load strategy from file
        
        Args:
            filepath: File path
            
        Returns:
            ModularStrategy instance
        """
        # Load from file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create strategy
        strategy = cls(
            name=data['name'],
            symbols=data['symbols'],
            parameters=data['parameters'],
            min_history_bars=data['min_history_bars']
        )
        
        # Set configuration
        strategy.config = data['config']
        
        # TODO: Load components and activation conditions
        # This requires a component factory to instantiate
        # components from their serialized representation
        
        logger.info(f"Strategy loaded from {filepath}")
        return strategy
