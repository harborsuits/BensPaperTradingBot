"""
Hybrid Strategy Combiner

Combines multiple strategy signals into a unified hybrid strategy with configurable weights
and signal processing methods.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import numpy as np
import pandas as pd
import importlib
import json
import os
import logging
from datetime import datetime
import uuid

from evotrader.core.strategy import Strategy

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridStrategySignalProcessor:
    """Signal processor for hybrid strategies."""
    
    @staticmethod
    def weighted_vote(signals: List[Dict[str, Any]], weights: List[float]) -> str:
        """
        Process signals using a weighted voting mechanism.
        
        Args:
            signals: List of signal dictionaries 
            weights: List of weights for each strategy
            
        Returns:
            Combined signal direction
        """
        if not signals or not weights or len(signals) != len(weights):
            return "none"
        
        # Initialize vote counters
        votes = {"buy": 0, "sell": 0, "none": 0}
        
        # Process each signal with its weight
        for signal, weight in zip(signals, weights):
            direction = signal.get("signal", "none")
            votes[direction] += weight
        
        # Find the signal with the highest weighted vote
        max_vote = max(votes.items(), key=lambda x: x[1])
        
        # If there's a tie with "none", default to none
        if max_vote[1] == votes["none"] and max_vote[0] != "none":
            return "none"
        
        return max_vote[0]
    
    @staticmethod
    def consensus(signals: List[Dict[str, Any]], threshold: float = 0.65) -> str:
        """
        Process signals using a consensus mechanism.
        
        Args:
            signals: List of signal dictionaries
            threshold: Required threshold of agreement (0.0-1.0)
            
        Returns:
            Combined signal direction
        """
        if not signals:
            return "none"
        
        # Count signals
        signal_counts = {"buy": 0, "sell": 0, "none": 0}
        for signal in signals:
            direction = signal.get("signal", "none")
            signal_counts[direction] += 1
        
        # Calculate proportions
        total = len(signals)
        buy_proportion = signal_counts["buy"] / total
        sell_proportion = signal_counts["sell"] / total
        
        # Check for consensus
        if buy_proportion >= threshold:
            return "buy"
        elif sell_proportion >= threshold:
            return "sell"
        else:
            return "none"
    
    @staticmethod
    def majority_confidence(signals: List[Dict[str, Any]]) -> str:
        """
        Process signals based on the highest total confidence.
        
        Args:
            signals: List of signal dictionaries
            
        Returns:
            Combined signal direction
        """
        if not signals:
            return "none"
        
        # Sum confidence by direction
        confidence_sum = {"buy": 0.0, "sell": 0.0, "none": 0.0}
        
        for signal in signals:
            direction = signal.get("signal", "none")
            confidence = signal.get("confidence", 0.0)
            confidence_sum[direction] += confidence
        
        # Find direction with highest confidence
        max_confidence = max(confidence_sum.items(), key=lambda x: x[1])
        
        # If max confidence is 0, return none
        if max_confidence[1] == 0:
            return "none"
        
        return max_confidence[0]
    
    @staticmethod
    def weighted_confidence(signals: List[Dict[str, Any]], weights: List[float]) -> str:
        """
        Process signals based on weighted confidence.
        
        Args:
            signals: List of signal dictionaries
            weights: List of weights for each strategy
            
        Returns:
            Combined signal direction
        """
        if not signals or not weights or len(signals) != len(weights):
            return "none"
        
        # Sum weighted confidence by direction
        confidence_sum = {"buy": 0.0, "sell": 0.0, "none": 0.0}
        
        for signal, weight in zip(signals, weights):
            direction = signal.get("signal", "none")
            confidence = signal.get("confidence", 0.0)
            confidence_sum[direction] += confidence * weight
        
        # Find direction with highest weighted confidence
        max_confidence = max(confidence_sum.items(), key=lambda x: x[1])
        
        # If max confidence is 0, return none
        if max_confidence[1] == 0:
            return "none"
        
        return max_confidence[0]
    
    @staticmethod
    def priority_cascade(signals: List[Dict[str, Any]], 
                        priorities: List[int]) -> Dict[str, Any]:
        """
        Process signals using a priority cascade mechanism. Returns the signal
        with the highest priority that is not "none".
        
        Args:
            signals: List of signal dictionaries
            priorities: List of priority values (lower number = higher priority)
            
        Returns:
            Complete signal dictionary, not just direction
        """
        if not signals or not priorities or len(signals) != len(priorities):
            return {"signal": "none", "confidence": 0}
        
        # Sort signals by priority
        sorted_signals = sorted(zip(signals, priorities), key=lambda x: x[1])
        
        # Return first non-none signal
        for signal, _ in sorted_signals:
            if signal.get("signal", "none") != "none":
                return signal
        
        # If all signals are none, return the highest priority signal
        return sorted_signals[0][0] if sorted_signals else {"signal": "none", "confidence": 0}


class HybridStrategy(Strategy):
    """
    Combines multiple strategies into a unified hybrid strategy.
    
    Features:
    - Configurable weights for each component strategy
    - Multiple signal combination methods
    - Customizable risk management rules
    - Support for strategy specialization by market condition
    """
    
    # Signal combination method mapping
    SIGNAL_METHODS = {
        "weighted_vote": HybridStrategySignalProcessor.weighted_vote,
        "consensus": HybridStrategySignalProcessor.consensus,
        "majority_confidence": HybridStrategySignalProcessor.majority_confidence,
        "weighted_confidence": HybridStrategySignalProcessor.weighted_confidence,
        "priority_cascade": HybridStrategySignalProcessor.priority_cascade
    }
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid strategy.
        
        Args:
            parameters: Strategy parameters including component strategies
        """
        # Set up a unique strategy ID
        self.strategy_id = f"HybridStrategy_{str(uuid.uuid4())[:8]}"
        
        # Default parameters
        default_params = {
            "component_strategies": [],  # List of strategy paths
            "weights": [],               # List of weights for each strategy
            "signal_method": "weighted_vote",  # Method to combine signals
            "consensus_threshold": 0.65,  # Threshold for consensus method
            "position_sizing": "average",  # How to determine position size
            "max_position_size": 1.0,     # Maximum position size
            "risk_management": {
                "use_stop_loss": True,
                "stop_loss_method": "average",  # average, tightest, or custom
                "stop_loss_pct": 0.05,        # Only used if method is custom
                "use_take_profit": True,
                "take_profit_method": "average",  # average, tightest, or custom
                "take_profit_pct": 0.1        # Only used if method is custom
            }
        }
        
        # Override defaults with provided parameters
        self.parameters = default_params.copy()
        if parameters:
            self.parameters.update(parameters)
        
        # Initialize component strategies
        self.strategies = []
        self.load_component_strategies()
        
        # Initialize metadata
        self.metadata = {
            "strategy_type": "HybridStrategy",
            "component_count": len(self.strategies),
            "component_types": [s.__class__.__name__ for s in self.strategies],
            "description": "Combined strategy using multiple signal sources",
            "version": "1.0.0"
        }
        
        # Initialize state
        self.position = 0
        self.entry_price = 0
        
        logger.info(f"Initialized HybridStrategy with {len(self.strategies)} components")
    
    def load_component_strategies(self):
        """Load all component strategies specified in parameters."""
        self.strategies = []
        
        # Get strategy paths and ensure weights list matches
        strategy_paths = self.parameters.get("component_strategies", [])
        weights = self.parameters.get("weights", [])
        
        # If weights not provided, use equal weights
        if not weights and strategy_paths:
            weights = [1.0 / len(strategy_paths)] * len(strategy_paths)
        
        # Ensure weights match number of strategies
        if len(weights) != len(strategy_paths):
            logger.warning(f"Weight count {len(weights)} doesn't match strategy count {len(strategy_paths)}. Using equal weights.")
            weights = [1.0 / len(strategy_paths)] * len(strategy_paths)
        
        # Load each strategy
        for path in strategy_paths:
            try:
                strategy = self._load_strategy(path)
                self.strategies.append(strategy)
            except Exception as e:
                logger.error(f"Error loading strategy {path}: {e}")
        
        # Update weights if needed
        if len(self.strategies) != len(weights):
            self.parameters["weights"] = [1.0 / len(self.strategies)] * len(self.strategies)
    
    def _load_strategy(self, strategy_path: str) -> Strategy:
        """
        Load a strategy from a module path.
        
        Args:
            strategy_path: Path to strategy class (module.ClassName)
            
        Returns:
            Initialized strategy
        """
        try:
            module_path, class_name = strategy_path.rsplit(".", 1)
            
            # Add current directory to path if needed
            import sys
            if os.getcwd() not in sys.path:
                sys.path.insert(0, os.getcwd())
            
            module = importlib.import_module(module_path)
            strategy_class = getattr(module, class_name)
            strategy = strategy_class()
            
            logger.info(f"Loaded component strategy: {class_name}")
            return strategy
        
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load strategy {strategy_path}: {e}")
            raise
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate combined trading signal from all component strategies.
        
        Args:
            market_data: Market data DataFrame
            
        Returns:
            Combined signal dictionary
        """
        if market_data is None or len(market_data) < 2:
            return {"signal": "none", "confidence": 0, "reason": "insufficient_data"}
        
        # Collect signals from all component strategies
        signals = []
        for strategy in self.strategies:
            try:
                # Get signal using appropriate method
                if hasattr(strategy, "calculate_signal"):
                    signal = strategy.calculate_signal(market_data)
                elif hasattr(strategy, "generate_signals"):
                    signal = strategy.generate_signals(market_data)
                else:
                    logger.warning(f"Strategy {strategy.__class__.__name__} has no signal method")
                    signal = {"signal": "none", "confidence": 0}
                
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error getting signal from {strategy.__class__.__name__}: {e}")
                signals.append({"signal": "none", "confidence": 0, "error": str(e)})
        
        # Return none if no signals collected
        if not signals:
            return {"signal": "none", "confidence": 0, "reason": "no_component_signals"}
        
        # Get weights and combination method
        weights = self.parameters.get("weights", [1.0 / len(signals)] * len(signals))
        method_name = self.parameters.get("signal_method", "weighted_vote")
        
        # Process signals based on selected method
        if method_name in self.SIGNAL_METHODS:
            method = self.SIGNAL_METHODS[method_name]
            
            # Priority cascade returns a full signal, others just return direction
            if method_name == "priority_cascade":
                priorities = list(range(len(signals)))  # Default priorities
                combined_signal = method(signals, priorities)
            elif method_name == "consensus":
                threshold = self.parameters.get("consensus_threshold", 0.65)
                combined_direction = method(signals, threshold)
                combined_signal = {"signal": combined_direction}
            else:
                combined_direction = method(signals, weights)
                combined_signal = {"signal": combined_direction}
        else:
            logger.warning(f"Unknown signal method: {method_name}, using weighted_vote")
            combined_direction = HybridStrategySignalProcessor.weighted_vote(signals, weights)
            combined_signal = {"signal": combined_direction}
        
        # Calculate confidence
        if "confidence" not in combined_signal:
            # Average confidence of components with matching signal direction
            matching_signals = [s for s in signals if s.get("signal") == combined_signal["signal"]]
            if matching_signals:
                avg_confidence = sum(s.get("confidence", 0) for s in matching_signals) / len(matching_signals)
                combined_signal["confidence"] = avg_confidence
            else:
                combined_signal["confidence"] = 0
        
        # Calculate position size
        sizing_method = self.parameters.get("position_sizing", "average")
        max_size = self.parameters.get("max_position_size", 1.0)
        
        if sizing_method == "average":
            # Average of component position sizes
            sizes = [s.get("position_size", 0) for s in signals if s.get("signal") == combined_signal["signal"]]
            if sizes:
                position_size = min(sum(sizes) / len(sizes), max_size)
            else:
                position_size = 0
        elif sizing_method == "weighted":
            # Weighted average of component position sizes
            total_weight = sum(w for w, s in zip(weights, signals) if s.get("signal") == combined_signal["signal"])
            if total_weight > 0:
                position_size = min(
                    sum(w * s.get("position_size", 0) for w, s in zip(weights, signals) 
                        if s.get("signal") == combined_signal["signal"]) / total_weight,
                    max_size
                )
            else:
                position_size = 0
        elif sizing_method == "confidence":
            # Scale by combined confidence
            position_size = min(max_size * combined_signal["confidence"], max_size)
        else:
            position_size = min(max_size * 0.5, max_size)  # Default to half max size
        
        combined_signal["position_size"] = position_size
        
        # Determine risk management parameters
        if self.parameters["risk_management"]["use_stop_loss"]:
            stop_loss_method = self.parameters["risk_management"]["stop_loss_method"]
            
            if stop_loss_method == "custom":
                stop_loss_pct = self.parameters["risk_management"]["stop_loss_pct"]
            elif stop_loss_method == "tightest":
                # Use the tightest stop from component strategies
                stop_losses = []
                for signal in signals:
                    risk_mgmt = signal.get("risk_management", {})
                    if risk_mgmt and "stop_loss" in risk_mgmt:
                        stop_losses.append(risk_mgmt["stop_loss"])
                
                if stop_losses and combined_signal["signal"] != "none":
                    if combined_signal["signal"] == "buy":
                        stop_loss_price = max(stop_losses)  # Highest stop for long positions
                    else:
                        stop_loss_price = min(stop_losses)  # Lowest stop for short positions
                    
                    current_price = market_data["close"].iloc[-1]
                    stop_loss_pct = abs(stop_loss_price - current_price) / current_price
                else:
                    stop_loss_pct = self.parameters["risk_management"]["stop_loss_pct"]
            elif stop_loss_method == "average":
                # Use average of component strategy stops
                stop_losses = []
                for signal in signals:
                    risk_mgmt = signal.get("risk_management", {})
                    if risk_mgmt and "stop_loss" in risk_mgmt:
                        stop_losses.append(risk_mgmt["stop_loss"])
                
                if stop_losses:
                    avg_stop = sum(stop_losses) / len(stop_losses)
                    current_price = market_data["close"].iloc[-1]
                    stop_loss_pct = abs(avg_stop - current_price) / current_price
                else:
                    stop_loss_pct = self.parameters["risk_management"]["stop_loss_pct"]
            else:
                stop_loss_pct = self.parameters["risk_management"]["stop_loss_pct"]
        else:
            stop_loss_pct = 0
        
        # Similar logic for take profit
        if self.parameters["risk_management"]["use_take_profit"]:
            take_profit_method = self.parameters["risk_management"]["take_profit_method"]
            
            if take_profit_method == "custom":
                take_profit_pct = self.parameters["risk_management"]["take_profit_pct"]
            elif take_profit_method == "tightest":
                # Use the tightest take profit from component strategies
                take_profits = []
                for signal in signals:
                    risk_mgmt = signal.get("risk_management", {})
                    if risk_mgmt and "take_profit" in risk_mgmt:
                        take_profits.append(risk_mgmt["take_profit"])
                
                if take_profits and combined_signal["signal"] != "none":
                    if combined_signal["signal"] == "buy":
                        take_profit_price = min(take_profits)  # Lowest take profit for long positions
                    else:
                        take_profit_price = max(take_profits)  # Highest take profit for short positions
                    
                    current_price = market_data["close"].iloc[-1]
                    take_profit_pct = abs(take_profit_price - current_price) / current_price
                else:
                    take_profit_pct = self.parameters["risk_management"]["take_profit_pct"]
            elif take_profit_method == "average":
                # Use average of component strategy take profits
                take_profits = []
                for signal in signals:
                    risk_mgmt = signal.get("risk_management", {})
                    if risk_mgmt and "take_profit" in risk_mgmt:
                        take_profits.append(risk_mgmt["take_profit"])
                
                if take_profits:
                    avg_take_profit = sum(take_profits) / len(take_profits)
                    current_price = market_data["close"].iloc[-1]
                    take_profit_pct = abs(avg_take_profit - current_price) / current_price
                else:
                    take_profit_pct = self.parameters["risk_management"]["take_profit_pct"]
            else:
                take_profit_pct = self.parameters["risk_management"]["take_profit_pct"]
        else:
            take_profit_pct = 0
        
        # Calculate stop and target prices
        current_price = market_data["close"].iloc[-1]
        
        if combined_signal["signal"] == "buy":
            stop_price = current_price * (1 - stop_loss_pct) if stop_loss_pct > 0 else 0
            target_price = current_price * (1 + take_profit_pct) if take_profit_pct > 0 else 0
        elif combined_signal["signal"] == "sell":
            stop_price = current_price * (1 + stop_loss_pct) if stop_loss_pct > 0 else 0
            target_price = current_price * (1 - take_profit_pct) if take_profit_pct > 0 else 0
        else:
            stop_price = 0
            target_price = 0
        
        # Add risk management to combined signal
        combined_signal["risk_management"] = {
            "stop_loss": float(stop_price),
            "take_profit": float(target_price)
        }
        
        # Add additional information
        combined_signal["reason"] = f"hybrid_signal_{method_name}"
        combined_signal["component_signals"] = [
            {"signal": s.get("signal", "none"), 
             "confidence": s.get("confidence", 0)} 
            for s in signals
        ]
        
        return combined_signal
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals for the strategy adapter interface.
        
        Args:
            market_data: Market data with OHLCV columns
            
        Returns:
            Signal dictionary
        """
        return self.calculate_signal(market_data)
    
    def update_position(self, position: float, entry_price: float):
        """
        Update the strategy's position information.
        
        Args:
            position: Current position size
            entry_price: Position entry price
        """
        self.position = position
        self.entry_price = entry_price
        
        # Update component strategies if they support it
        for strategy in self.strategies:
            if hasattr(strategy, "update_position"):
                strategy.update_position(position, entry_price)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all parameters including component strategy parameters."""
        params = self.parameters.copy()
        
        # Add component parameters
        component_params = []
        for i, strategy in enumerate(self.strategies):
            if hasattr(strategy, "get_parameters"):
                try:
                    strategy_params = strategy.get_parameters()
                    component_params.append({
                        "index": i,
                        "type": strategy.__class__.__name__,
                        "weight": self.parameters["weights"][i] if i < len(self.parameters["weights"]) else 0,
                        "parameters": strategy_params
                    })
                except Exception as e:
                    logger.error(f"Error getting parameters from component {i}: {e}")
        
        params["component_parameters"] = component_params
        return params
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Update parameters. Will reload component strategies if needed."""
        old_components = self.parameters.get("component_strategies", [])
        
        # Update parameters
        self.parameters.update(parameters)
        
        # Check if components changed
        new_components = self.parameters.get("component_strategies", [])
        if set(old_components) != set(new_components):
            self.load_component_strategies()


def create_hybrid_strategy(config_path: Optional[str] = None,
                         config_dict: Optional[Dict[str, Any]] = None) -> HybridStrategy:
    """
    Create a hybrid strategy from configuration.
    
    Args:
        config_path: Path to config JSON file
        config_dict: Dict with configuration (takes precedence over file)
        
    Returns:
        Initialized hybrid strategy
    """
    # Load configuration
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Override with provided dict
    if config_dict:
        config.update(config_dict)
    
    return HybridStrategy(config)


def create_optimized_hybrid() -> HybridStrategy:
    """
    Create a pre-configured optimized hybrid strategy based on evolutionary results.
    
    Returns:
        Initialized hybrid strategy
    """
    # Define component strategies with appropriate weights based on evolutionary test results
    config = {
        "component_strategies": [
            "evotrader.strategies.optimized_bollinger_strategy.OptimizedBollingerBandsStrategy",
            "evotrader.core.strategies.rsi_strategy.RSIStrategy",
            "evotrader.core.strategies.moving_average_strategy.MovingAverageCrossoverStrategy"
        ],
        "weights": [0.6, 0.25, 0.15],  # Higher weight to the evolved Bollinger strategy
        "signal_method": "weighted_confidence",
        "position_sizing": "confidence",
        "max_position_size": 0.75,  # Conservative position sizing
        "risk_management": {
            "use_stop_loss": True,
            "stop_loss_method": "tightest",  # Use tightest stop loss for safety
            "stop_loss_pct": 0.05,
            "use_take_profit": True,
            "take_profit_method": "average",
            "take_profit_pct": 0.1
        }
    }
    
    return HybridStrategy(config)


if __name__ == "__main__":
    # Simple test code
    hybrid = create_optimized_hybrid()
    print(f"Created hybrid strategy with {len(hybrid.strategies)} components")
    
    for i, strategy in enumerate(hybrid.strategies):
        print(f"Component {i}: {strategy.__class__.__name__} (Weight: {hybrid.parameters['weights'][i]})")
    
    print(f"Signal method: {hybrid.parameters['signal_method']}")
    print(f"Position sizing: {hybrid.parameters['position_sizing']}")
    
    # Save configuration
    os.makedirs("config", exist_ok=True)
    with open("config/optimized_hybrid.json", "w") as f:
        json.dump(hybrid.get_parameters(), f, indent=2)
    
    print("Saved configuration to config/optimized_hybrid.json")
