#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Parser

This module provides utilities for parsing and validating
configuration files for trading strategies.
"""

import os
import yaml
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        
    Returns:
        Dictionary containing configuration
    """
    try:
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return {}
        
        # Determine file type by extension
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() in ['.yaml', '.yml']:
            # Load YAML file
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        elif ext.lower() == '.json':
            # Load JSON file
            with open(config_path, 'r') as file:
                config = json.load(file)
        else:
            logger.error(f"Unsupported configuration file format: {ext}")
            return {}
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration file: {e}")
        return {}

def save_config_file(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save a configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(config_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Determine file type by extension
        _, ext = os.path.splitext(config_path)
        
        if ext.lower() in ['.yaml', '.yml']:
            # Save as YAML
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
        elif ext.lower() == '.json':
            # Save as JSON
            with open(config_path, 'w') as file:
                json.dump(config, file, indent=2)
        else:
            logger.error(f"Unsupported configuration file format: {ext}")
            return False
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration file: {e}")
        return False

def flatten_config(config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested configuration dictionary.
    
    Args:
        config: Nested configuration dictionary
        separator: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    def _flatten(d, parent_key=''):
        items = []
        for key, value in d.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(_flatten(value, new_key).items())
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    return _flatten(config)

def unflatten_config(config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Unflatten a flat configuration dictionary.
    
    Args:
        config: Flat configuration dictionary
        separator: Separator used for nested keys
        
    Returns:
        Nested dictionary
    """
    result = {}
    
    for key, value in config.items():
        parts = key.split(separator)
        
        # Traverse the nested dictionary
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        
        # Set the value
        d[parts[-1]] = value
    
    return result

def get_calendar_spread_config(config_path: str) -> Dict[str, Any]:
    """
    Load and process calendar spread configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Processed configuration dictionary
    """
    try:
        # Load raw configuration
        config = load_config_file(config_path)
        
        if not config:
            logger.error("Failed to load calendar spread configuration")
            return {}
        
        # Validate and provide defaults for missing values
        validated_config = validate_calendar_spread_config(config)
        
        # Flatten for easy parameter access
        flat_config = flatten_config(validated_config)
        
        return {
            'raw_config': config,
            'validated_config': validated_config,
            'flat_config': flat_config
        }
    
    except Exception as e:
        logger.error(f"Error processing calendar spread configuration: {e}")
        return {}

def validate_calendar_spread_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate calendar spread configuration and provide defaults for missing values.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    # Define default configuration
    default_config = {
        "name": "Calendar Spread Strategy",
        
        # 1. Strategy Philosophy parameters
        "strategy_philosophy": {
            "use_same_strike": True,
            "net_theta_decay_target": 0.01
        },
        
        # 2. Underlying & Option Universe & Timeframe parameters
        "universe": {
            "underlying_universe": ["SPY", "QQQ", "AAPL"],
            "short_leg_min_dte": 7,
            "short_leg_max_dte": 21,
            "long_leg_min_dte": 45,
            "long_leg_max_dte": 90,
            "roll_short_leg_dte": 7,
            "roll_long_leg_dte": 30
        },
        
        # 3. Selection Criteria for Underlying parameters
        "selection_criteria": {
            "min_iv_rank": 30,
            "max_iv_rank": 60,
            "min_underlying_adv": 500000,
            "min_option_open_interest": 1000,
            "max_bid_ask_spread_pct": 0.15
        },
        
        # 4. Spread Construction parameters
        "spread_construction": {
            "strike_selection": "ATM",
            "strike_bias": 0,
            "max_net_debit_pct": 1.0,
            "leg_ratio": 1,
            "option_type": "call"
        },
        
        # 5. Expiration & Roll Timing parameters
        "expiration_and_roll": {
            "roll_trigger_dte": 7,
            "early_roll_volatility_change_pct": 20
        },
        
        # 6. Entry Execution parameters
        "entry_execution": {
            "use_combo_orders": True,
            "max_slippage_pct": 5.0
        },
        
        # 7. Exit & Adjustment Rules parameters
        "exit_rules": {
            "profit_target_pct": 50,
            "stop_loss_multiplier": 1.0,
            "adjustment_threshold_pct": 10
        },
        
        # 8. Position Sizing & Risk Controls parameters
        "risk_controls": {
            "position_size_pct": 1.0,
            "max_concurrent_spreads": 5,
            "max_margin_usage_pct": 10.0,
            "max_sector_concentration": 1
        },
        
        # 9. Backtesting & Performance Metrics parameters
        "backtesting": {
            "historical_window_days": 1095,
            "performance_metrics": [
                "theta_capture_ratio",
                "win_rate",
                "avg_profit_per_trade",
                "max_drawdown",
                "roll_cost_impact",
                "net_roi_per_cycle"
            ]
        },
        
        # 10. Continuous Optimization parameters
        "optimization": {
            "optimization_frequency_days": 30,
            "iv_rank_adaptation": True,
            "strike_bias_optimization": True,
            "use_ml_model": False
        }
    }
    
    # Start with default configuration
    validated_config = default_config.copy()
    
    # Update with provided configuration (recursive update)
    def update_recursive(d, u):
        for key, value in u.items():
            if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                d[key] = update_recursive(d[key], value)
            else:
                d[key] = value
        return d
    
    validated_config = update_recursive(validated_config, config)
    
    # Additional validations and type conversions
    
    # Strategy philosophy validations
    if not isinstance(validated_config["strategy_philosophy"]["net_theta_decay_target"], (int, float)):
        logger.warning("net_theta_decay_target should be a number, using default 0.01")
        validated_config["strategy_philosophy"]["net_theta_decay_target"] = 0.01
    
    # Universe validations
    if not isinstance(validated_config["universe"]["underlying_universe"], list):
        logger.warning("underlying_universe should be a list, using default")
        validated_config["universe"]["underlying_universe"] = default_config["universe"]["underlying_universe"]
    
    for dte_param in ["short_leg_min_dte", "short_leg_max_dte", "long_leg_min_dte", "long_leg_max_dte", 
                      "roll_short_leg_dte", "roll_long_leg_dte"]:
        if not isinstance(validated_config["universe"][dte_param], int) or validated_config["universe"][dte_param] < 0:
            logger.warning(f"{dte_param} should be a positive integer, using default")
            validated_config["universe"][dte_param] = default_config["universe"][dte_param]
    
    # Selection criteria validations
    for pct_param in ["min_iv_rank", "max_iv_rank"]:
        if not isinstance(validated_config["selection_criteria"][pct_param], (int, float)):
            logger.warning(f"{pct_param} should be a number, using default")
            validated_config["selection_criteria"][pct_param] = default_config["selection_criteria"][pct_param]
        else:
            # Ensure IV rank is between 0 and 100
            validated_config["selection_criteria"][pct_param] = max(0, min(100, validated_config["selection_criteria"][pct_param]))
    
    # Spread construction validations
    if validated_config["spread_construction"]["strike_selection"] not in ["ATM", "ITM", "OTM", "delta"]:
        logger.warning("strike_selection should be one of: ATM, ITM, OTM, delta. Using default ATM")
        validated_config["spread_construction"]["strike_selection"] = "ATM"
    
    if validated_config["spread_construction"]["strike_bias"] not in [-1, 0, 1]:
        logger.warning("strike_bias should be -1, 0, or 1. Using default 0")
        validated_config["spread_construction"]["strike_bias"] = 0
    
    if validated_config["spread_construction"]["option_type"] not in ["call", "put"]:
        logger.warning("option_type should be 'call' or 'put'. Using default 'call'")
        validated_config["spread_construction"]["option_type"] = "call"
    
    # Position sizing validations
    if not isinstance(validated_config["risk_controls"]["position_size_pct"], (int, float)) or validated_config["risk_controls"]["position_size_pct"] <= 0:
        logger.warning("position_size_pct should be a positive number, using default 1.0")
        validated_config["risk_controls"]["position_size_pct"] = 1.0
    
    if not isinstance(validated_config["risk_controls"]["max_concurrent_spreads"], int) or validated_config["risk_controls"]["max_concurrent_spreads"] <= 0:
        logger.warning("max_concurrent_spreads should be a positive integer, using default 5")
        validated_config["risk_controls"]["max_concurrent_spreads"] = 5
    
    # Add timestamp for when configuration was validated
    validated_config["_metadata"] = {
        "validated_at": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    return validated_config

def config_to_strategy_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert configuration to strategy parameters format.
    
    Args:
        config: Configuration dictionary (validated)
        
    Returns:
        Dictionary in strategy parameters format
    """
    # Use flattened configuration for easy access
    flat_config = flatten_config(config)
    
    # Map configuration keys to strategy parameter keys
    parameter_mapping = {
        # Strategy Philosophy
        "strategy_philosophy.use_same_strike": "use_same_strike",
        "strategy_philosophy.net_theta_decay_target": "net_theta_decay_target",
        
        # Universe & Timeframe
        "universe.underlying_universe": "underlying_universe",
        "universe.short_leg_min_dte": "short_leg_min_dte",
        "universe.short_leg_max_dte": "short_leg_max_dte",
        "universe.long_leg_min_dte": "long_leg_min_dte",
        "universe.long_leg_max_dte": "long_leg_max_dte",
        "universe.roll_short_leg_dte": "roll_short_leg_dte",
        "universe.roll_long_leg_dte": "roll_long_leg_dte",
        
        # Selection Criteria
        "selection_criteria.min_iv_rank": "min_iv_rank",
        "selection_criteria.max_iv_rank": "max_iv_rank",
        "selection_criteria.min_underlying_adv": "min_underlying_adv",
        "selection_criteria.min_option_open_interest": "min_option_open_interest",
        "selection_criteria.max_bid_ask_spread_pct": "max_bid_ask_spread_pct",
        
        # Spread Construction
        "spread_construction.strike_selection": "strike_selection",
        "spread_construction.strike_bias": "strike_bias",
        "spread_construction.max_net_debit_pct": "max_net_debit_pct",
        "spread_construction.leg_ratio": "leg_ratio",
        "spread_construction.option_type": "option_type",
        
        # Expiration & Roll
        "expiration_and_roll.roll_trigger_dte": "roll_trigger_dte",
        "expiration_and_roll.early_roll_volatility_change_pct": "early_roll_volatility_change_pct",
        
        # Entry Execution
        "entry_execution.use_combo_orders": "use_combo_orders",
        "entry_execution.max_slippage_pct": "max_slippage_pct",
        
        # Exit Rules
        "exit_rules.profit_target_pct": "profit_target_pct",
        "exit_rules.stop_loss_multiplier": "stop_loss_multiplier",
        "exit_rules.adjustment_threshold_pct": "adjustment_threshold_pct",
        
        # Risk Controls
        "risk_controls.position_size_pct": "position_size_pct",
        "risk_controls.max_concurrent_spreads": "max_concurrent_spreads",
        "risk_controls.max_margin_usage_pct": "max_margin_usage_pct",
        "risk_controls.max_sector_concentration": "max_sector_concentration"
    }
    
    # Build parameters dictionary
    parameters = {}
    
    for config_key, param_key in parameter_mapping.items():
        if config_key in flat_config:
            parameters[param_key] = flat_config[config_key]
    
    # Add strategy name
    if "name" in config:
        parameters["name"] = config["name"]
    
    # Add default risk-free rate if not present
    if "risk_free_rate" not in parameters:
        parameters["risk_free_rate"] = 0.04  # 4% default
    
    # Add default direction
    if "direction" not in parameters:
        parameters["direction"] = "neutral"
    
    return parameters 