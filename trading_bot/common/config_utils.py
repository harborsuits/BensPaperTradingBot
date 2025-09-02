#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration and state management utilities for the trading bot.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

def setup_directories(base_dir: str = "data") -> Dict[str, Path]:
    """
    Set up the directory structure for the trading bot.
    
    Args:
        base_dir: Base directory for all data
        
    Returns:
        Dictionary of directory paths
    """
    base_path = Path(base_dir)
    directories = {
        'base': base_path,
        'config': base_path / 'config',
        'state': base_path / 'state',
        'logs': base_path / 'logs',
        'data': base_path / 'market_data',
        'backtest': base_path / 'backtest',
        'models': base_path / 'models'
    }
    
    # Create directories if they don't exist
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return directories

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise

def save_state(state: Dict[str, Any], state_path: str) -> None:
    """
    Save state to a JSON file.
    
    Args:
        state: State dictionary to save
        state_path: Path to save the state file
    """
    try:
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved state to {state_path}")
    except Exception as e:
        logger.error(f"Error saving state to {state_path}: {e}")
        raise

def load_state(state_path: str) -> Optional[Dict[str, Any]]:
    """
    Load state from a JSON file.
    
    Args:
        state_path: Path to the state file
        
    Returns:
        Dictionary containing state, or None if file doesn't exist
    """
    try:
        if not os.path.exists(state_path):
            logger.warning(f"State file {state_path} does not exist")
            return None
            
        with open(state_path, 'r') as f:
            state = json.load(f)
        logger.info(f"Loaded state from {state_path}")
        return state
    except Exception as e:
        logger.error(f"Error loading state from {state_path}: {e}")
        raise 