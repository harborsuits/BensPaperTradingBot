#!/usr/bin/env python3
"""
Strategy Management System Interfaces

This module defines the core interfaces and abstract base classes for the strategy management system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """Base interface for all trading strategies"""
    
    @abstractmethod
    def get_id(self) -> str:
        """Get unique identifier for the strategy"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get human-readable name for the strategy"""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Get strategy type (e.g., trend_following, mean_reversion, etc.)"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of the strategy"""
        pass
    
    @abstractmethod
    def evaluate(self, market_context: 'MarketContext') -> Dict[str, Any]:
        """
        Evaluate the strategy against current market conditions
        Returns dict with evaluation results
        """
        pass
    
    @abstractmethod
    def generate_signals(self, market_context: 'MarketContext') -> List[Dict[str, Any]]:
        """
        Generate trading signals based on strategy logic and current market conditions
        Returns list of signal dictionaries
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the strategy"""
        pass
    
    @abstractmethod
    def update_performance(self, metrics: Dict[str, Any]) -> None:
        """Update performance metrics for the strategy"""
        pass
    
    @abstractmethod
    def get_portfolio_allocation(self) -> float:
        """Get current portfolio allocation percentage (0.0 to 1.0)"""
        pass
    
    @abstractmethod
    def set_portfolio_allocation(self, allocation: float) -> None:
        """Set portfolio allocation percentage (0.0 to 1.0)"""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current strategy parameters"""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set strategy parameters"""
        pass
    
    @abstractmethod
    def on_trade_executed(self, trade_data: Dict[str, Any]) -> None:
        """Handle trade execution notification"""
        pass
    
    @abstractmethod
    def on_market_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle market event notification"""
        pass

class MarketRegimeClassifier(ABC):
    """Interface for market regime classification systems"""
    
    @abstractmethod
    def classify_regime(self, market_data: Dict[str, Any]) -> str:
        """Classify the current market regime based on provided market data"""
        pass
        
    @abstractmethod
    def get_regime_confidence(self) -> float:
        """Get confidence level for the current regime classification"""
        pass
        
    @abstractmethod
    def get_regime_description(self, regime: Optional[str] = None) -> str:
        """Get the description for a specific regime or the current regime"""
        pass

class MarketContext(ABC):
    """
    Interface for market context that stores market state data
    """
    
    @abstractmethod
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the context by key"""
        pass
    
    @abstractmethod
    def set_value(self, key: str, value: Any) -> None:
        """Set a value in the context by key"""
        pass
    
    @abstractmethod
    def get_all_values(self) -> Dict[str, Any]:
        """Get all key-value pairs in the context"""
        pass
    
    @abstractmethod
    def get_last_updated(self, key: str) -> Optional[float]:
        """Get the timestamp when a value was last updated"""
        pass
    
    @abstractmethod
    def get_regime(self) -> str:
        """Get the current market regime"""
        pass
    
    @abstractmethod
    def get_market_state(self) -> Dict[str, Any]:
        """Get comprehensive market state data"""
        pass
    
    @abstractmethod
    def export_data(self) -> Dict[str, Any]:
        """Export all context data for serialization"""
        pass
    
    @abstractmethod
    def import_data(self, data: Dict[str, Any]) -> bool:
        """Import data from a serialized format"""
        pass
    
    @abstractmethod
    def save_to_file(self, filepath: str) -> bool:
        """Save context to a file"""
        pass
    
    @abstractmethod
    def load_from_file(self, filepath: str) -> bool:
        """Load context from a file"""
        pass

class CoreContext(ABC):
    """Central context manager interface"""
    
    @abstractmethod
    def register_strategy(self, strategy: Strategy) -> bool:
        """Register a strategy with the context"""
        pass
    
    @abstractmethod
    def unregister_strategy(self, strategy_id: str) -> bool:
        """Unregister a strategy from the context"""
        pass
    
    @abstractmethod
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy by ID"""
        pass
    
    @abstractmethod
    def get_strategies(self, strategy_type: Optional[str] = None) -> List[Strategy]:
        """Get all registered strategies, optionally filtered by type"""
        pass
    
    @abstractmethod
    def get_enabled_strategies(self) -> List[Strategy]:
        """Get all enabled strategies"""
        pass
    
    @abstractmethod
    def enable_strategy(self, strategy_id: str) -> bool:
        """Enable a strategy"""
        pass
    
    @abstractmethod
    def disable_strategy(self, strategy_id: str) -> bool:
        """Disable a strategy"""
        pass
    
    @abstractmethod
    def is_strategy_enabled(self, strategy_id: str) -> bool:
        """Check if a strategy is enabled"""
        pass
    
    @abstractmethod
    def get_market_context(self) -> MarketContext:
        """Get the market context"""
        pass
    
    @abstractmethod
    def update_market_context(self, updates: Dict[str, Any]) -> bool:
        """Update market context with multiple values"""
        pass
    
    @abstractmethod
    def subscribe_to_event(self, event_type: str, callback: callable) -> bool:
        """Subscribe to an event"""
        pass
    
    @abstractmethod
    def unsubscribe_from_event(self, event_type: str, callback: callable) -> bool:
        """Unsubscribe from an event"""
        pass
    
    @abstractmethod
    def save_state(self, directory: Optional[str] = None) -> bool:
        """Save current state to disk"""
        pass
    
    @abstractmethod
    def load_state(self, directory: Optional[str] = None) -> bool:
        """Load state from disk"""
        pass

class StrategyPrioritizer(ABC):
    """Interface for strategy prioritization system"""
    
    @abstractmethod
    def prioritize_strategies(self, strategies: List[Strategy], market_context: MarketContext) -> List[Tuple[Strategy, float]]:
        """
        Prioritize strategies based on market context and performance
        Returns list of (strategy, score) tuples sorted by score (descending)
        """
        pass
    
    @abstractmethod
    def update_strategy_performance(self, strategy_id: str, performance_metrics: Dict[str, Any]) -> None:
        """Update performance data for a strategy"""
        pass
    
    @abstractmethod
    def get_allocation_recommendation(self, strategy_id: str) -> float:
        """Get recommended allocation for a strategy (0.0 to 1.0)"""
        pass

@dataclass
class EventPayload:
    """Data class for event payloads"""
    event_type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None

class EventListener:
    """Base class for event listeners"""
    
    def __init__(self, event_types: List[str] = None):
        self.event_types = event_types or []
    
    def handle_event(self, event: EventPayload) -> None:
        """Handle an event"""
        handler_method = getattr(self, f"on_{event.event_type}", None)
        if handler_method and callable(handler_method):
            try:
                handler_method(event.data)
            except Exception as e:
                logger.error(f"Error handling event {event.event_type}: {str(e)}")
        else:
            logger.warning(f"No handler for event {event.event_type}")

class CoreContext:
    """
    Core context for the strategy management system
    
    Maintains strategy registry, market context, and event dispatching
    """
    
    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self._market_context = MarketContext()
        self._event_listeners: Dict[str, List[EventListener]] = {}
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self._global_constraints: Dict[str, Any] = {
            "max_allocation_per_strategy": 80.0,
            "min_allocation_per_strategy": 0.0,
            "total_allocation_limit": 100.0
        }
    
    def add_strategy(self, strategy_id: str, strategy: Strategy) -> None:
        """Add a strategy to the registry"""
        self._strategies[strategy_id] = strategy
        self._performance_history[strategy_id] = []
        logger.info(f"Added strategy: {strategy_id}")
    
    def remove_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from the registry"""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            del self._performance_history[strategy_id]
            logger.info(f"Removed strategy: {strategy_id}")
        else:
            logger.warning(f"Strategy not found: {strategy_id}")
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """Get a strategy by ID"""
        return self._strategies.get(strategy_id)
    
    def get_all_strategies(self) -> Dict[str, Strategy]:
        """Get all registered strategies"""
        return self._strategies.copy()
    
    def get_strategy_ids(self) -> List[str]:
        """Get all strategy IDs"""
        return list(self._strategies.keys())
    
    def add_listener(self, event_type: str, listener: EventListener) -> None:
        """Add an event listener"""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        
        if listener not in self._event_listeners[event_type]:
            self._event_listeners[event_type].append(listener)
            logger.debug(f"Added listener for event: {event_type}")
    
    def remove_listener(self, event_type: str, listener: EventListener) -> None:
        """Remove an event listener"""
        if event_type in self._event_listeners and listener in self._event_listeners[event_type]:
            self._event_listeners[event_type].remove(listener)
            logger.debug(f"Removed listener for event: {event_type}")
    
    def update_market_context(self, context_data: Dict[str, Any]) -> None:
        """
        Update market context with new data
        
        Args:
            context_data: Dictionary with context attributes to update
        """
        old_regime = getattr(self._market_context, "regime", "unknown")
        self._market_context.update(context_data)
        new_regime = getattr(self._market_context, "regime", "unknown")
        
        # Dispatch market context updated event
        self._notify_listeners("market_context_updated", self._market_context)
        
        # Dispatch regime change event if applicable
        if old_regime != new_regime:
            self._notify_listeners("regime_changed", {
                "old_regime": old_regime,
                "new_regime": new_regime,
                "market_context": self._market_context
            })
            logger.info(f"Market regime changed: {old_regime} -> {new_regime}")
    
    def get_market_context(self) -> MarketContext:
        """Get current market context"""
        return self._market_context
    
    def update_strategy_performance(self, strategy_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update performance metrics for a strategy
        
        Args:
            strategy_id: ID of the strategy
            metrics: Performance metrics
        """
        if strategy_id in self._strategies:
            # Add timestamp if not present
            if "timestamp" not in metrics:
                metrics["timestamp"] = datetime.now().isoformat()
            
            # Store in performance history
            self._performance_history[strategy_id].append(metrics)
            
            # Limit history size
            max_history = 1000
            if len(self._performance_history[strategy_id]) > max_history:
                self._performance_history[strategy_id] = self._performance_history[strategy_id][-max_history:]
            
            # Notify listeners
            self._notify_listeners("strategy_performance_updated", {
                "strategy_id": strategy_id,
                "metrics": metrics
            })
            
            logger.debug(f"Updated performance for strategy {strategy_id}")
        else:
            logger.warning(f"Cannot update performance for unknown strategy: {strategy_id}")
    
    def get_performance_history(self, strategy_id: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get performance history for strategies
        
        Args:
            strategy_id: Optional ID to get history for a specific strategy
            
        Returns:
            Dictionary mapping strategy IDs to lists of performance metrics
        """
        if strategy_id:
            if strategy_id in self._performance_history:
                return {strategy_id: self._performance_history[strategy_id]}
            else:
                logger.warning(f"No performance history for strategy: {strategy_id}")
                return {}
        
        return self._performance_history.copy()
    
    def set_global_constraint(self, constraint_name: str, value: Any) -> None:
        """Set a global constraint value"""
        self._global_constraints[constraint_name] = value
        logger.info(f"Set global constraint {constraint_name} = {value}")
    
    def get_global_constraints(self) -> Dict[str, Any]:
        """Get all global constraints"""
        return self._global_constraints.copy()
    
    def _notify_listeners(self, event_type: str, data: Any) -> None:
        """
        Notify listeners of an event
        
        Args:
            event_type: Type of event
            data: Event data
        """
        event = EventPayload(
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            source="core_context"
        )
        
        # Notify specific listeners
        if event_type in self._event_listeners:
            for listener in self._event_listeners[event_type]:
                try:
                    listener.handle_event(event)
                except Exception as e:
                    logger.error(f"Error in listener for {event_type}: {str(e)}")
        
        # Notify wildcard listeners
        if "*" in self._event_listeners:
            for listener in self._event_listeners["*"]:
                try:
                    listener.handle_event(event)
                except Exception as e:
                    logger.error(f"Error in wildcard listener for {event_type}: {str(e)}")

class TradingStrategy(ABC):
    """
    Interface for trading strategies
    """
    
    @abstractmethod
    def is_suitable(self, market_context: 'MarketContext') -> Tuple[bool, float]:
        """
        Determine if strategy is suitable for current market conditions
        Returns (is_suitable, confidence)
        """
        pass
    
    @abstractmethod
    def generate_signals(self, market_data: Dict[str, Any], market_context: 'MarketContext') -> List[Dict[str, Any]]:
        """
        Generate trading signals based on market data and context
        """
        pass
    
    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get strategy performance metrics
        """
        pass
    
    @abstractmethod
    def update(self, market_context: 'MarketContext', feedback: Dict[str, Any]) -> bool:
        """
        Update strategy based on feedback
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Get strategy description
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get strategy parameters
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Set strategy parameters
        """
        pass

class StrategyRotator(ABC):
    """
    Interface for dynamic strategy rotator
    """
    
    @abstractmethod
    def register_strategy(self, strategy: TradingStrategy, preferred_regimes: List[str]) -> bool:
        """
        Register a trading strategy with preferred market regimes
        """
        pass
    
    @abstractmethod
    def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister a trading strategy
        """
        pass
    
    @abstractmethod
    def select_strategy(self, market_context: 'MarketContext') -> Optional[TradingStrategy]:
        """
        Select the most suitable strategy for current market conditions
        """
        pass
    
    @abstractmethod
    def get_all_strategies(self) -> List[Tuple[TradingStrategy, List[str], float]]:
        """
        Get all registered strategies with their preferred regimes and scores
        """
        pass
    
    @abstractmethod
    def update_strategy_scores(self, market_context: 'MarketContext', 
                               performance_data: Dict[str, Dict[str, float]]) -> None:
        """
        Update strategy suitability scores based on performance data
        """
        pass

class StrategySelector(ABC):
    """Interface for strategy selection systems"""
    
    @abstractmethod
    def select_strategy(self, market_context: Dict[str, Any]) -> str:
        """Select the most appropriate strategy based on market context"""
        pass
        
    @abstractmethod
    def register_strategy(self, strategy_id: str, 
                         performance_profile: Dict[str, Any]) -> bool:
        """Register a new strategy with its performance profile"""
        pass
        
    @abstractmethod
    def get_strategy_confidence(self) -> float:
        """Get confidence level for the current strategy selection"""
        pass
        
    @abstractmethod
    def update_strategy_performance(self, strategy_id: str,
                                  performance_metrics: Dict[str, float]) -> None:
        """Update performance metrics for a specific strategy"""
        pass

class ContinuousLearningSystem(ABC):
    """Interface for continuous learning systems"""
    
    @abstractmethod
    def process_market_feedback(self, market_data: Dict[str, Any],
                               strategy_id: str,
                               performance_metrics: Dict[str, float]) -> None:
        """Process market feedback for learning"""
        pass
        
    @abstractmethod
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get the current learning progress metrics"""
        pass
        
    @abstractmethod
    def apply_learning(self) -> bool:
        """Apply learned adjustments to the system"""
        pass

class MarketContextProvider(ABC):
    """Interface for market context providers"""
    
    @abstractmethod
    def get_market_context(self) -> Dict[str, Any]:
        """Get the current market context data"""
        pass
        
    @abstractmethod
    def get_historical_context(self, timeframe: str) -> List[Dict[str, Any]]:
        """Get historical market context data"""
        pass
        
    @abstractmethod
    def register_data_source(self, source_id: str, 
                           data_fetcher: Callable[[], Dict[str, Any]]) -> bool:
        """Register a new data source for market context"""
        pass 