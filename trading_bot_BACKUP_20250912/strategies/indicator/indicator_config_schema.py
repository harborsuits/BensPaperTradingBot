#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicator Strategy Configuration Schema

This module defines the JSON schema for indicator strategy configurations
and provides validation utilities.
"""

import json
import logging
import jsonschema
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# JSON Schema for indicator strategy configurations
INDICATOR_STRATEGY_SCHEMA = {
    "type": "object",
    "required": ["name", "indicators"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Unique name for the strategy"
        },
        "strategy_type": {
            "type": "string",
            "description": "Type of strategy (default: 'indicator')",
            "default": "indicator"
        },
        "symbol": {
            "type": "string",
            "description": "Trading symbol"
        },
        "timeframe": {
            "type": "string",
            "description": "Timeframe (e.g., '1m', '5m', '1h', '1d')",
            "default": "1h"
        },
        "indicators": {
            "type": "object",
            "description": "Technical indicators with parameters",
            "additionalProperties": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Indicator type (e.g., 'SMA', 'RSI')"
                    }
                },
                "additionalProperties": true
            }
        },
        "entry_rules": {
            "type": "array",
            "description": "Rules for entering a position",
            "items": {
                "type": "object",
                "required": ["condition"],
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "Rule condition expression"
                    },
                    "entry_signal": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Signal to generate when rule is met"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the rule"
                    }
                }
            }
        },
        "exit_rules": {
            "type": "array",
            "description": "Rules for exiting a position",
            "items": {
                "type": "object",
                "required": ["condition"],
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "Rule condition expression"
                    },
                    "exit_signal": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Signal to generate when rule is met (optional)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Human-readable description of the rule"
                    }
                }
            }
        },
        "position_sizing": {
            "type": "object",
            "description": "Position sizing configuration",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["fixed", "percent", "risk_based"],
                    "description": "Position sizing method"
                },
                "value": {
                    "type": "number",
                    "description": "Position size value (units, percent, or risk factor)"
                }
            }
        },
        "risk_management": {
            "type": "object",
            "description": "Risk management parameters",
            "properties": {
                "stop_loss_pct": {
                    "type": "number",
                    "description": "Stop loss percentage"
                },
                "take_profit_pct": {
                    "type": "number",
                    "description": "Take profit percentage"
                },
                "trailing_stop_pct": {
                    "type": "number",
                    "description": "Trailing stop percentage"
                },
                "max_drawdown_pct": {
                    "type": "number",
                    "description": "Maximum allowed drawdown percentage"
                }
            }
        }
    }
}

def validate_strategy_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a strategy configuration against the schema.
    
    Args:
        config: Strategy configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    validator = jsonschema.Draft7Validator(INDICATOR_STRATEGY_SCHEMA)
    errors = list(validator.iter_errors(config))
    
    return [error.message for error in errors]

def load_and_validate_config(config_file: str) -> Optional[Dict[str, Any]]:
    """
    Load a configuration file and validate it.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary if valid, None otherwise
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        errors = validate_strategy_config(config)
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return None
        
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration file {config_file}: {str(e)}")
        return None

def create_example_config() -> Dict[str, Any]:
    """
    Create an example strategy configuration.
    
    Returns:
        Example configuration dictionary
    """
    return {
        "name": "RSI_Reversal",
        "strategy_type": "indicator",
        "symbol": "AAPL",
        "timeframe": "1h",
        "indicators": {
            "rsi": {
                "type": "RSI",
                "timeperiod": 14
            },
            "sma50": {
                "type": "SMA",
                "timeperiod": 50
            },
            "sma200": {
                "type": "SMA",
                "timeperiod": 200
            },
            "bb": {
                "type": "BBANDS",
                "timeperiod": 20,
                "nbdevup": 2,
                "nbdevdn": 2
            }
        },
        "entry_rules": [
            {
                "condition": "rsi < 30",
                "entry_signal": "buy",
                "description": "RSI oversold condition"
            },
            {
                "condition": "rsi > 70",
                "entry_signal": "sell",
                "description": "RSI overbought condition"
            },
            {
                "condition": "close < bb_lower",
                "entry_signal": "buy",
                "description": "Price below lower Bollinger Band"
            }
        ],
        "exit_rules": [
            {
                "condition": "rsi > 50",
                "description": "Exit long when RSI returns to neutral"
            },
            {
                "condition": "rsi < 50",
                "description": "Exit short when RSI returns to neutral"
            }
        ],
        "position_sizing": {
            "type": "percent",
            "value": 5.0
        },
        "risk_management": {
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "trailing_stop_pct": 1.0
        }
    }
