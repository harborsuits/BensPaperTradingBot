import logging
import os
import json
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime

from trading_bot.strategy_management.interfaces import CoreContext, Strategy, MarketContext
from trading_bot.strategy_management.market_context import TradingMarketContext

logger = logging.getLogger(__name__)

class TradingCoreContext(CoreContext):
    """
    Central context manager for the trading system that handles:
    - Strategy registration and management
    - Market context updates
    - Event broadcasting
    - Configuration management
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        market_context: Optional[MarketContext] = None
    ):
        """
        Initialize the core context
        
        Args:
            config_path: Path to configuration file
            market_context: Market context instance (optional)
        """
        # Initialize strategies dictionary
        self._strategies: Dict[str, Strategy] = {}
        
        # Initialize enabled strategies set
        self._enabled_strategies: Set[str] = set()
        
        # Initialize event subscribers
        self._event_subscribers: Dict[str, List[callable]] = {}
        
        # Initialize market context
        self._market_context = market_context or TradingMarketContext()
        
        # Configuration
        self.config_path = config_path
        self.config = {}
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._set_default_config()
            
        logger.info(f"Initialized TradingCoreContext with {len(self.config)} configuration options")
    
    def _set_default_config(self):
        """Set default configuration"""
        self.config = {
            "max_strategies": 10,
            "strategy_evaluation_interval_seconds": 300,  # 5 minutes
            "event_processing_batch_size": 100,
            "default_strategy_enabled": True,
            "persistence": {
                "enabled": True,
                "directory": "data/strategies",
                "backup_interval_hours": 24
            },
            "logging": {
                "strategy_changes": True,
                "market_context_updates": True,
                "allocation_changes": True
            }
        }
    
    def _load_config(self, config_path: str):
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self._set_default_config()
    
    def _save_config(self):
        """Save configuration to file"""
        if not self.config_path:
            logger.warning("No configuration path specified, cannot save config")
            return
            
        try:
            directory = os.path.dirname(self.config_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def register_strategy(self, strategy: Strategy) -> bool:
        """
        Register a strategy with the context
        
        Args:
            strategy: Strategy instance
            
        Returns:
            True if registered successfully
        """
        strategy_id = strategy.get_id()
        
        # Check if already registered
        if strategy_id in self._strategies:
            logger.warning(f"Strategy {strategy_id} already registered")
            return False
            
        # Check maximum strategies
        max_strategies = self.config.get("max_strategies", 10)
        if len(self._strategies) >= max_strategies:
            logger.error(f"Cannot register strategy {strategy_id}: maximum number of strategies reached")
            return False
            
        # Register strategy
        self._strategies[strategy_id] = strategy
        
        # Enable by default if configured
        if self.config.get("default_strategy_enabled", True):
            self.enable_strategy(strategy_id)
            
        # Log registration
        strategy_name = strategy.get_name()
        strategy_type = strategy.get_type()
        logger.info(f"Registered strategy: {strategy_id} ({strategy_name}, type: {strategy_type})")
        
        # Broadcast event
        self._broadcast_event("strategy_registered", {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def unregister_strategy(self, strategy_id: str) -> bool:
        """
        Unregister a strategy from the context
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if unregistered successfully
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Cannot unregister strategy {strategy_id}: not registered")
            return False
            
        # Get strategy info for event
        strategy = self._strategies[strategy_id]
        strategy_name = strategy.get_name()
        strategy_type = strategy.get_type()
        
        # Remove from enabled strategies
        if strategy_id in self._enabled_strategies:
            self._enabled_strategies.remove(strategy_id)
            
        # Remove from strategies dictionary
        del self._strategies[strategy_id]
        
        # Log unregistration
        logger.info(f"Unregistered strategy: {strategy_id} ({strategy_name}, type: {strategy_type})")
        
        # Broadcast event
        self._broadcast_event("strategy_unregistered", {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """
        Get a strategy by ID
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Strategy instance or None if not found
        """
        return self._strategies.get(strategy_id)
    
    def get_strategy_ids(self) -> List[str]:
        """
        Get all registered strategy IDs
        
        Returns:
            List of strategy IDs
        """
        return list(self._strategies.keys())
    
    def get_strategies(self, strategy_type: Optional[str] = None) -> List[Strategy]:
        """
        Get all registered strategies, optionally filtered by type
        
        Args:
            strategy_type: Strategy type to filter by
            
        Returns:
            List of strategy instances
        """
        if strategy_type:
            return [
                strategy for strategy in self._strategies.values()
                if strategy.get_type() == strategy_type
            ]
        else:
            return list(self._strategies.values())
    
    def get_enabled_strategies(self) -> List[Strategy]:
        """
        Get all enabled strategies
        
        Returns:
            List of enabled strategy instances
        """
        return [
            self._strategies[strategy_id]
            for strategy_id in self._enabled_strategies
            if strategy_id in self._strategies
        ]
    
    def enable_strategy(self, strategy_id: str) -> bool:
        """
        Enable a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if enabled successfully
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Cannot enable strategy {strategy_id}: not registered")
            return False
            
        if strategy_id in self._enabled_strategies:
            # Already enabled
            return True
            
        # Enable strategy
        self._enabled_strategies.add(strategy_id)
        
        # Get strategy info for event
        strategy = self._strategies[strategy_id]
        strategy_name = strategy.get_name()
        
        # Log enabling
        logger.info(f"Enabled strategy: {strategy_id} ({strategy_name})")
        
        # Broadcast event
        self._broadcast_event("strategy_enabled", {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def disable_strategy(self, strategy_id: str) -> bool:
        """
        Disable a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if disabled successfully
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Cannot disable strategy {strategy_id}: not registered")
            return False
            
        if strategy_id not in self._enabled_strategies:
            # Already disabled
            return True
            
        # Disable strategy
        self._enabled_strategies.remove(strategy_id)
        
        # Get strategy info for event
        strategy = self._strategies[strategy_id]
        strategy_name = strategy.get_name()
        
        # Log disabling
        logger.info(f"Disabled strategy: {strategy_id} ({strategy_name})")
        
        # Broadcast event
        self._broadcast_event("strategy_disabled", {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def is_strategy_enabled(self, strategy_id: str) -> bool:
        """
        Check if a strategy is enabled
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if the strategy is enabled
        """
        return strategy_id in self._enabled_strategies
    
    def get_market_context(self) -> MarketContext:
        """
        Get the market context
        
        Returns:
            Market context instance
        """
        return self._market_context
    
    def update_market_context(self, updates: Dict[str, Any]) -> bool:
        """
        Update market context with multiple values
        
        Args:
            updates: Dictionary of updates
            
        Returns:
            True if updated successfully
        """
        # Apply updates
        for key, value in updates.items():
            self._market_context.set_value(key, value)
            
        # Broadcast update event with timestamp
        update_event = {
            "updates": updates,
            "timestamp": datetime.now().isoformat()
        }
        self._broadcast_event("market_context_updated", update_event)
        
        # Log if configured
        if self.config.get("logging", {}).get("market_context_updates", True):
            logger.debug(f"Updated market context: {len(updates)} values")
            
        return True
    
    def subscribe_to_event(self, event_type: str, callback: callable) -> bool:
        """
        Subscribe to an event
        
        Args:
            event_type: Event type to subscribe to
            callback: Callback function
            
        Returns:
            True if subscribed successfully
        """
        if event_type not in self._event_subscribers:
            self._event_subscribers[event_type] = []
            
        if callback in self._event_subscribers[event_type]:
            logger.warning(f"Callback already subscribed to event {event_type}")
            return False
            
        self._event_subscribers[event_type].append(callback)
        logger.debug(f"Subscribed to event: {event_type}")
        
        return True
    
    def unsubscribe_from_event(self, event_type: str, callback: callable) -> bool:
        """
        Unsubscribe from an event
        
        Args:
            event_type: Event type to unsubscribe from
            callback: Callback function
            
        Returns:
            True if unsubscribed successfully
        """
        if event_type not in self._event_subscribers:
            return False
            
        if callback not in self._event_subscribers[event_type]:
            return False
            
        self._event_subscribers[event_type].remove(callback)
        logger.debug(f"Unsubscribed from event: {event_type}")
        
        return True
    
    def _broadcast_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Broadcast an event to all subscribers
        
        Args:
            event_type: Event type
            event_data: Event data
        """
        if event_type not in self._event_subscribers:
            return
            
        subscribers = self._event_subscribers[event_type]
        for callback in subscribers:
            try:
                callback(event_type, event_data)
            except Exception as e:
                logger.error(f"Error in event subscriber callback for {event_type}: {str(e)}")
                
    def save_state(self, directory: Optional[str] = None) -> bool:
        """
        Save the current state
        
        Args:
            directory: Directory to save state to
            
        Returns:
            True if saved successfully
        """
        if not directory:
            directory = self.config.get("persistence", {}).get("directory", "data/strategies")
            
        if not directory:
            logger.warning("No directory specified for saving state")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save strategies state
            strategies_data = {}
            for strategy_id, strategy in self._strategies.items():
                try:
                    strategy_data = {
                        "id": strategy_id,
                        "name": strategy.get_name(),
                        "type": strategy.get_type(),
                        "enabled": strategy_id in self._enabled_strategies,
                        "performance": strategy.get_performance_metrics(),
                        "allocation": strategy.get_portfolio_allocation(),
                        "parameters": strategy.get_parameters()
                    }
                    strategies_data[strategy_id] = strategy_data
                except Exception as e:
                    logger.error(f"Error saving strategy {strategy_id} state: {str(e)}")
                    
            # Save strategies file
            strategies_file = os.path.join(directory, "strategies.json")
            with open(strategies_file, 'w') as f:
                json.dump(strategies_data, f, indent=2)
                
            # Save market context
            market_context_file = os.path.join(directory, "market_context.json")
            market_context_data = self._market_context.export_data()
            with open(market_context_file, 'w') as f:
                json.dump(market_context_data, f, indent=2)
                
            # Save configuration
            config_file = os.path.join(directory, "core_config.json")
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            logger.info(f"Saved state to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            return False
    
    def load_state(self, directory: Optional[str] = None) -> bool:
        """
        Load state from disk
        
        Args:
            directory: Directory to load state from
            
        Returns:
            True if loaded successfully
        """
        if not directory:
            directory = self.config.get("persistence", {}).get("directory", "data/strategies")
            
        if not directory or not os.path.exists(directory):
            logger.warning(f"Directory {directory} does not exist, cannot load state")
            return False
            
        try:
            # Load configuration
            config_file = os.path.join(directory, "core_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
                
            # Load market context
            market_context_file = os.path.join(directory, "market_context.json")
            if os.path.exists(market_context_file):
                with open(market_context_file, 'r') as f:
                    market_context_data = json.load(f)
                self._market_context.import_data(market_context_data)
                logger.info(f"Loaded market context from {market_context_file}")
                
            # Note: Strategies are not loaded here because they need to be registered
            # by the application. The performance and other data can be applied later.
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            return False
    
    def apply_strategy_state(self, strategy_id: str, state_data: Dict[str, Any]) -> bool:
        """
        Apply saved state to a strategy
        
        Args:
            strategy_id: Strategy ID
            state_data: Strategy state data
            
        Returns:
            True if applied successfully
        """
        if strategy_id not in self._strategies:
            logger.warning(f"Cannot apply state to strategy {strategy_id}: not registered")
            return False
            
        strategy = self._strategies[strategy_id]
        
        try:
            # Apply performance metrics
            if "performance" in state_data:
                strategy.update_performance(state_data["performance"])
                
            # Apply allocation
            if "allocation" in state_data:
                strategy.set_portfolio_allocation(state_data["allocation"])
                
            # Apply parameters
            if "parameters" in state_data:
                strategy.set_parameters(state_data["parameters"])
                
            # Apply enabled state
            if state_data.get("enabled", False):
                self.enable_strategy(strategy_id)
            else:
                self.disable_strategy(strategy_id)
                
            logger.info(f"Applied saved state to strategy {strategy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying state to strategy {strategy_id}: {str(e)}")
            return False 