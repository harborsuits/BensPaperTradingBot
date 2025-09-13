#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Validator

This module provides utility functions to validate strategy implementations
for consistency and best practices. It helps ensure that strategies follow
the required patterns for the event-driven architecture and contain proper
error handling and risk management.
"""

import logging
import inspect
import importlib
import os
import sys
from typing import List, Dict, Any, Optional, Set, Tuple
import ast
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class RequiredMethod:
    """Class representing a required method in a strategy."""
    
    def __init__(self, name, required_args=None, required_error_handling=False):
        self.name = name
        self.required_args = required_args or []
        self.required_error_handling = required_error_handling

class StrategyValidator:
    """
    Validates strategies for consistency and best practices.
    """
    
    # Core methods that should be present in all strategies
    CORE_METHODS = [
        RequiredMethod("__init__", ["session", "data_pipeline", "parameters"]),
        RequiredMethod("calculate_indicators", ["data"]),
        RequiredMethod("generate_signals", ["data", "indicators"]),
        RequiredMethod("_check_exit_conditions", ["position", "data", "indicators"]),
        RequiredMethod("_execute_signals", required_error_handling=True),
        RequiredMethod("register_events"),
        RequiredMethod("on_event", ["event"], required_error_handling=True)
    ]
    
    # Methods that should have error handling
    METHODS_REQUIRING_ERROR_HANDLING = [
        "on_event",
        "_execute_signals", 
        "filter_option_chains",
        "construct_iron_condor",
        "construct_butterfly",
        "construct_straddle",
        "construct_strangle",
        "construct_calendar_spread"
    ]
    
    # Required event subscriptions
    REQUIRED_EVENT_SUBSCRIPTIONS = [
        "EventType.VOLATILITY_SPIKE",
        "EventType.MARKET_REGIME_CHANGE"
    ]
    
    @staticmethod
    def validate_strategy_class(strategy_class) -> Dict[str, Any]:
        """
        Validate a strategy class for consistency with best practices.
        
        Args:
            strategy_class: The strategy class to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "class_name": strategy_class.__name__,
            "is_valid": True,
            "missing_methods": [],
            "methods_missing_error_handling": [],
            "missing_event_subscriptions": [],
            "missing_validation_checks": [],
            "passed_checks": []
        }
        
        # Check for required methods
        for required_method in StrategyValidator.CORE_METHODS:
            method_name = required_method.name
            if not hasattr(strategy_class, method_name):
                results["missing_methods"].append(method_name)
                results["is_valid"] = False
            else:
                results["passed_checks"].append(f"Has required method: {method_name}")
                
                # Check method signature if args specified
                if required_method.required_args:
                    method = getattr(strategy_class, method_name)
                    signature = inspect.signature(method)
                    
                    for arg in required_method.required_args:
                        if arg not in signature.parameters:
                            results["missing_methods"].append(f"{method_name} missing arg: {arg}")
                            results["is_valid"] = False
        
        # Check for error handling in critical methods
        source_code = inspect.getsource(strategy_class)
        for method_name in StrategyValidator.METHODS_REQUIRING_ERROR_HANDLING:
            if hasattr(strategy_class, method_name):
                method = getattr(strategy_class, method_name)
                method_source = inspect.getsource(method)
                
                if "try:" not in method_source or "except" not in method_source:
                    results["methods_missing_error_handling"].append(method_name)
                    results["is_valid"] = False
                else:
                    results["passed_checks"].append(f"Has error handling in: {method_name}")
        
        # Check for event subscriptions
        if hasattr(strategy_class, "register_events"):
            method_source = inspect.getsource(strategy_class.register_events)
            for event_subscription in StrategyValidator.REQUIRED_EVENT_SUBSCRIPTIONS:
                if event_subscription not in method_source:
                    results["missing_event_subscriptions"].append(event_subscription)
                    results["is_valid"] = False
                else:
                    results["passed_checks"].append(f"Subscribes to: {event_subscription}")
        
        # Check for validation in position construction
        validation_patterns = [
            "risk_reward_ratio",
            "credit_to_loss_ratio",
            "return None",
            "if position is None"
        ]
        
        construction_methods = [
            method for method in dir(strategy_class) 
            if method.startswith("construct_") and callable(getattr(strategy_class, method))
        ]
        
        for method_name in construction_methods:
            method = getattr(strategy_class, method_name)
            method_source = inspect.getsource(method)
            
            for pattern in validation_patterns:
                if pattern not in method_source:
                    results["missing_validation_checks"].append(f"{method_name} missing: {pattern}")
                    results["is_valid"] = False
                else:
                    results["passed_checks"].append(f"Has validation in {method_name}: {pattern}")
        
        return results
    
    @staticmethod
    def validate_all_strategies(strategies_dir: str) -> Dict[str, Dict[str, Any]]:
        """
        Validate all strategies in the given directory.
        
        Args:
            strategies_dir: Directory containing strategy modules
            
        Returns:
            Dictionary mapping strategy names to validation results
        """
        results = {}
        
        # Get the absolute path
        abs_path = os.path.abspath(strategies_dir)
        
        # Add to path temporarily
        sys.path.insert(0, abs_path)
        
        # Get all Python files in the directory and subdirectories
        for dirpath, dirnames, filenames in os.walk(abs_path):
            for filename in [f for f in filenames if f.endswith('.py') and not f.startswith('__')]:
                module_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(module_path, abs_path)
                
                # Convert path to module name
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                
                try:
                    # Import the module
                    module = importlib.import_module(module_name)
                    
                    # Find strategy classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            "Strategy" in name and 
                            not name.startswith("__") and
                            not name.startswith("Abstract") and
                            not name.startswith("Base")):
                            
                            # Validate the strategy
                            results[name] = StrategyValidator.validate_strategy_class(obj)
                            
                except Exception as e:
                    logger.error(f"Error validating module {module_name}: {str(e)}")
        
        # Remove the added path
        sys.path.remove(abs_path)
        
        return results

    @staticmethod
    def generate_fix_template(validation_results: Dict[str, Any]) -> str:
        """
        Generate a template with fixes for a strategy based on validation results.
        
        Args:
            validation_results: Results from validate_strategy_class
            
        Returns:
            String with code snippets to fix issues
        """
        template = f"""
# Fix Template for {validation_results['class_name']}
# ================================================
# This template contains code snippets to fix issues in the strategy.
# Copy and paste the relevant sections into your strategy file.

"""
        
        # Add missing methods
        if validation_results["missing_methods"]:
            template += "\n# Missing Methods\n# --------------\n"
            
            for method in validation_results["missing_methods"]:
                if method == "__init__":
                    template += """
    def __init__(self, session, data_pipeline, parameters=None):
        \"\"\"
        Initialize the strategy.
        
        Args:
            session: Trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        \"\"\"
        # Initialize base class
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Add your default parameters here
        }
        
        # Apply defaults for any missing parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
"""
                
                elif method == "calculate_indicators":
                    template += """
    def calculate_indicators(self, data):
        \"\"\"
        Calculate technical indicators for the strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        \"\"\"
        # Start with parent class indicators
        indicators = super().calculate_indicators(data)
        
        if data.empty or len(data) < 20:
            return indicators
            
        try:
            # Add strategy-specific indicator calculations here
            pass
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            
        return indicators
"""
                
                elif method == "generate_signals":
                    template += """
    def generate_signals(self, data, indicators):
        \"\"\"
        Generate trading signals based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        \"\"\"
        # Start with empty signals dictionary
        signals = {
            "entry": False,
            "exit_positions": [],
            "adjust_positions": False,
            "positions_to_adjust": []
        }
        
        try:
            # Generate strategy-specific signals here
            pass
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            
        return signals
"""
                
                elif method == "_check_exit_conditions":
                    template += """
    def _check_exit_conditions(self, position, data, indicators):
        \"\"\"
        Check exit conditions for an open position.
        
        Args:
            position: The position to check
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Boolean indicating whether to exit
        \"\"\"
        try:
            # Add exit condition checks here
            pass
            
            # Check profit target
            if 'target_profit_pct' in self.parameters:
                target_profit = self.parameters['target_profit_pct'] / 100
                current_profit = position.get_current_profit_pct() / 100
                
                if current_profit >= target_profit:
                    logger.info(f"Exit signal: reached profit target {target_profit:.2%}")
                    return True
                    
            # Check stop loss
            if 'stop_loss_pct' in self.parameters:
                stop_loss = self.parameters['stop_loss_pct'] / 100
                current_loss = -position.get_current_profit_pct() / 100
                
                if current_loss >= stop_loss:
                    logger.info(f"Exit signal: reached stop loss {stop_loss:.2%}")
                    return True
                    
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
            
        return False
"""
                
                elif method == "_execute_signals":
                    template += """
    def _execute_signals(self):
        \"\"\"
        Execute strategy-specific trading signals.
        \"\"\"
        try:
            # Use parent class for basic execution
            super()._execute_signals()
            
            # Add strategy-specific execution logic here
            
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
"""
                
                elif method == "register_events":
                    template += """
    def register_events(self):
        \"\"\"Register for events relevant to this strategy.\"\"\"
        # Register for common event types from the parent class
        super().register_events()
        
        # Add strategy-specific event subscriptions
        EventBus.subscribe(EventType.VOLATILITY_SPIKE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
"""
                
                elif method == "on_event":
                    template += """
    def on_event(self, event):
        \"\"\"
        Process incoming events for the strategy.
        
        Args:
            event: The event to process
        \"\"\"
        try:
            # Let parent class handle common events first
            super().on_event(event)
            
            # Handle strategy-specific events
            if event.type == EventType.VOLATILITY_SPIKE:
                spike_pct = event.data.get('percentage', 0)
                
                if spike_pct > 15:
                    logger.info(f"Volatility spike of {spike_pct}% detected, adjusting parameters")
                    # Add parameter adjustments here
                    
            elif event.type == EventType.MARKET_REGIME_CHANGE:
                new_regime = event.data.get('new_regime')
                logger.info(f"Market regime changed to {new_regime}, adjusting strategy")
                # Add regime-specific adjustments here
                
        except Exception as e:
            logger.error(f"Error processing event: {str(e)}")
"""
        
        # Add error handling sections
        if validation_results["methods_missing_error_handling"]:
            template += "\n# Missing Error Handling\n# -------------------\n"
            template += "# Add try/except blocks to these methods:\n\n"
            
            for method in validation_results["methods_missing_error_handling"]:
                template += f"""
# For the {method} method, add error handling:
try:
    # Existing code goes here
    pass
except Exception as e:
    logger.error(f"Error in {method}: {{str(e)}}")
    # Add appropriate fallback/recovery code
"""
        
        # Add event subscription sections
        if validation_results["missing_event_subscriptions"]:
            template += "\n# Missing Event Subscriptions\n# -------------------------\n"
            template += "# Add these event subscriptions to the register_events method:\n\n"
            
            for event in validation_results["missing_event_subscriptions"]:
                template += f"EventBus.subscribe({event}, self.on_event)\n"
        
        # Add validation check sections
        if validation_results["missing_validation_checks"]:
            template += "\n# Missing Validation Checks\n# -----------------------\n"
            template += "# Add these validation checks to position construction methods:\n\n"
            
            template += """
# Risk/reward validation pattern:
profit_potential = net_credit  # or calculate appropriately for your strategy
loss_potential = max_loss - net_credit
risk_reward_ratio = loss_potential / profit_potential if profit_potential > 0 else float('inf')

# Check if risk/reward ratio is acceptable
max_risk_reward = self.parameters.get('max_risk_reward_ratio', 3.0)
if risk_reward_ratio > max_risk_reward:
    logger.info(f"Position rejected: risk/reward ratio {risk_reward_ratio:.2f} exceeds maximum {max_risk_reward:.2f}")
    return None
"""
        
        return template
