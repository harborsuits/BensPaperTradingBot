"""
Regime Strategy Adapter

This module provides an adapter that applies regime-optimized parameters
to trading strategies based on current market conditions.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Union
import copy
from datetime import datetime

# Import system components
from trading_bot.analytics.market_regime.detector import MarketRegimeType
from trading_bot.analytics.market_regime.integration import MarketRegimeManager
from trading_bot.core.event_bus import EventBus, Event

logger = logging.getLogger(__name__)

class RegimeStrategyAdapter:
    """
    Adapts trading strategies to current market regimes by dynamically
    adjusting their parameters based on the detected market conditions.
    
    This class serves as a bridge between the market regime system and
    the strategy implementations, enabling automatic parameter optimization
    without modifying the strategy code.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        regime_manager: MarketRegimeManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the regime strategy adapter.
        
        Args:
            event_bus: System event bus
            regime_manager: Market regime manager
            config: Configuration parameters
        """
        self.event_bus = event_bus
        self.regime_manager = regime_manager
        self.config = config or {}
        
        # Store adapted strategy instances
        self.adapted_strategies: Dict[str, Dict[str, Any]] = {}
        
        # Store original parameters for each strategy
        self.original_parameters: Dict[str, Dict[str, Any]] = {}
        
        # Store current regime parameters for each strategy
        self.current_parameters: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Adaptation tracking
        self.last_adaptation: Dict[str, Dict[str, datetime]] = {}
        
        # Register for events
        self._register_event_handlers()
        
        logger.info("Regime Strategy Adapter initialized")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant system events."""
        try:
            # Register for market regime change events
            self.event_bus.register("market_regime_change", self._handle_regime_change)
            
            # Register for strategy registration events
            self.event_bus.register("strategy_registered", self._handle_strategy_registered)
            
            logger.info("Registered regime strategy adapter event handlers")
            
        except Exception as e:
            logger.error(f"Error registering event handlers: {str(e)}")
    
    def register_strategy(
        self, strategy_id: str, strategy_instance: Any, parameters: Dict[str, Any]
    ) -> None:
        """
        Register a strategy for regime adaptation.
        
        Args:
            strategy_id: Strategy identifier
            strategy_instance: Strategy object instance
            parameters: Current strategy parameters
        """
        try:
            self.adapted_strategies[strategy_id] = {
                "instance": strategy_instance,
                "symbol": getattr(strategy_instance, "symbol", None),
                "timeframe": getattr(strategy_instance, "timeframe", None),
            }
            
            # Store original parameters
            self.original_parameters[strategy_id] = copy.deepcopy(parameters)
            
            # Initialize current parameters
            if strategy_id not in self.current_parameters:
                self.current_parameters[strategy_id] = {}
            
            # Initialize last adaptation
            if strategy_id not in self.last_adaptation:
                self.last_adaptation[strategy_id] = {}
            
            logger.info(f"Registered strategy {strategy_id} for regime adaptation")
            
            # Apply initial parameters based on current regime
            self._adapt_strategy_to_current_regime(strategy_id)
            
        except Exception as e:
            logger.error(f"Error registering strategy: {str(e)}")
    
    def _handle_strategy_registered(self, event: Event) -> None:
        """
        Handle strategy registration event.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            strategy_id = data.get("strategy_id")
            strategy_instance = data.get("strategy_instance")
            parameters = data.get("parameters", {})
            
            if not strategy_id or not strategy_instance:
                return
            
            self.register_strategy(strategy_id, strategy_instance, parameters)
            
        except Exception as e:
            logger.error(f"Error handling strategy registration: {str(e)}")
    
    def _handle_regime_change(self, event: Event) -> None:
        """
        Handle market regime change event.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            symbol = data.get("symbol")
            timeframe = data.get("timeframe")
            new_regime = data.get("new_regime")
            
            if not symbol or not timeframe or not new_regime:
                return
            
            # Find strategies that match this symbol and timeframe
            for strategy_id, strategy_data in self.adapted_strategies.items():
                if (strategy_data.get("symbol") == symbol and 
                    strategy_data.get("timeframe") == timeframe):
                    
                    # Adapt strategy to new regime
                    self._adapt_strategy(strategy_id, symbol, timeframe, new_regime)
            
        except Exception as e:
            logger.error(f"Error handling regime change: {str(e)}")
    
    def _adapt_strategy_to_current_regime(self, strategy_id: str) -> bool:
        """
        Adapt a strategy to its current market regime.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            bool: Success status
        """
        try:
            strategy_data = self.adapted_strategies.get(strategy_id)
            if not strategy_data:
                return False
            
            symbol = strategy_data.get("symbol")
            timeframe = strategy_data.get("timeframe")
            
            if not symbol or not timeframe:
                return False
            
            # Get current regime
            current_regimes = self.regime_manager.detector.get_current_regimes(symbol)
            
            if timeframe not in current_regimes:
                # Try to use primary timeframe if specified one isn't available
                primary_timeframe = self.regime_manager.config.get("primary_timeframe", "1d")
                if primary_timeframe in current_regimes:
                    timeframe = primary_timeframe
                else:
                    return False
            
            regime_info = current_regimes[timeframe]
            regime_type = regime_info.get("regime")
            
            if not regime_type:
                return False
            
            # Adapt to the regime
            return self._adapt_strategy(strategy_id, symbol, timeframe, regime_type)
            
        except Exception as e:
            logger.error(f"Error adapting strategy to current regime: {str(e)}")
            return False
    
    def _adapt_strategy(
        self, strategy_id: str, symbol: str, timeframe: str, regime_type: Union[MarketRegimeType, str]
    ) -> bool:
        """
        Adapt a strategy to a specific market regime.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Symbol
            timeframe: Timeframe
            regime_type: Market regime type
            
        Returns:
            bool: Success status
        """
        try:
            # Convert string to enum if needed
            if isinstance(regime_type, str):
                try:
                    regime_type = MarketRegimeType(regime_type)
                except ValueError:
                    logger.warning(f"Invalid regime type: {regime_type}")
                    return False
            
            # Get optimized parameters
            optimized_params = self.regime_manager.get_parameter_set(
                strategy_id, symbol, timeframe
            )
            
            if not optimized_params:
                logger.debug(f"No optimized parameters for {strategy_id} in {regime_type}")
                return False
            
            # Get strategy instance
            strategy_data = self.adapted_strategies.get(strategy_id)
            if not strategy_data:
                return False
            
            strategy_instance = strategy_data.get("instance")
            if not strategy_instance:
                return False
            
            # Store current parameters
            self.current_parameters[strategy_id][symbol] = {
                "regime": regime_type,
                "timeframe": timeframe,
                "parameters": optimized_params
            }
            
            # Update last adaptation time
            if symbol not in self.last_adaptation[strategy_id]:
                self.last_adaptation[strategy_id][symbol] = {}
            
            self.last_adaptation[strategy_id][symbol] = datetime.now()
            
            # Apply parameters to strategy
            self._apply_parameters(strategy_instance, optimized_params)
            
            logger.info(f"Adapted strategy {strategy_id} for {symbol} to {regime_type} regime")
            
            # Emit event
            self.event_bus.emit("strategy_adapted", {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": regime_type.value,
                "parameters": optimized_params,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error adapting strategy: {str(e)}")
            return False
    
    def _apply_parameters(self, strategy_instance: Any, parameters: Dict[str, Any]) -> None:
        """
        Apply parameters to a strategy instance.
        
        Args:
            strategy_instance: Strategy object instance
            parameters: Parameters to apply
        """
        try:
            # Apply parameters to strategy instance
            # First try to use a dedicated method if it exists
            if hasattr(strategy_instance, "update_parameters"):
                strategy_instance.update_parameters(parameters)
            else:
                # Fall back to directly setting attributes
                for param_name, param_value in parameters.items():
                    if hasattr(strategy_instance, param_name):
                        setattr(strategy_instance, param_name, param_value)
            
        except Exception as e:
            logger.error(f"Error applying parameters: {str(e)}")
    
    def reset_strategy_parameters(self, strategy_id: str) -> bool:
        """
        Reset a strategy to its original parameters.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            bool: Success status
        """
        try:
            if strategy_id not in self.adapted_strategies or strategy_id not in self.original_parameters:
                return False
            
            strategy_data = self.adapted_strategies[strategy_id]
            strategy_instance = strategy_data.get("instance")
            
            if not strategy_instance:
                return False
            
            # Reset to original parameters
            self._apply_parameters(strategy_instance, self.original_parameters[strategy_id])
            
            # Clear current parameters
            if strategy_id in self.current_parameters:
                self.current_parameters[strategy_id] = {}
            
            logger.info(f"Reset strategy {strategy_id} to original parameters")
            
            # Emit event
            self.event_bus.emit("strategy_reset", {
                "strategy_id": strategy_id,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting strategy parameters: {str(e)}")
            return False
    
    def get_strategy_adaptation_info(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get information about a strategy's adaptation.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict with adaptation information
        """
        try:
            if strategy_id not in self.adapted_strategies:
                return {"error": "Strategy not found"}
            
            strategy_data = self.adapted_strategies[strategy_id]
            
            # Get symbols
            symbol = strategy_data.get("symbol")
            
            # Get adaptation info
            result = {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "timeframe": strategy_data.get("timeframe"),
                "original_parameters": self.original_parameters.get(strategy_id, {}),
                "current_adaptation": {}
            }
            
            # Add current adaptation info
            if strategy_id in self.current_parameters and symbol in self.current_parameters[strategy_id]:
                adaptation = self.current_parameters[strategy_id][symbol]
                
                result["current_adaptation"] = {
                    "regime": adaptation["regime"].value,
                    "timeframe": adaptation["timeframe"],
                    "parameters": adaptation["parameters"],
                    "adapted_at": self.last_adaptation[strategy_id][symbol].isoformat()
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting strategy adaptation info: {str(e)}")
            return {"error": str(e)}
    
    def get_all_adaptations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all strategy adaptations.
        
        Returns:
            Dict mapping strategy IDs to adaptation information
        """
        result = {}
        
        for strategy_id in self.adapted_strategies:
            result[strategy_id] = self.get_strategy_adaptation_info(strategy_id)
        
        return result

# Helper function to create and initialize the adapter
def create_regime_strategy_adapter(
    event_bus: EventBus,
    regime_manager: MarketRegimeManager,
    config: Optional[Dict[str, Any]] = None
) -> RegimeStrategyAdapter:
    """
    Create and initialize a regime strategy adapter.
    
    Args:
        event_bus: System event bus
        regime_manager: Market regime manager
        config: Optional configuration
        
    Returns:
        Initialized RegimeStrategyAdapter
    """
    return RegimeStrategyAdapter(event_bus, regime_manager, config)
