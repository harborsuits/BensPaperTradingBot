#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Compatibility Layer

This module provides compatibility functions for accessing configuration
during the migration to the new organization structure.
"""

import logging
import warnings
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a configuration value with support for both old and new systems.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    # First try the new unified config
    try:
        from trading_bot.config.unified_config import get
        return get(key, default)
    except ImportError:
        pass
    
    # Fall back to old config manager
    try:
        from trading_bot.config_manager import ConfigManager
        config = ConfigManager.instance()
        return config.get(key, default)
    except (ImportError, AttributeError):
        # Try other config patterns that might exist
        try:
            from trading_bot.config import config
            if hasattr(config, 'get'):
                return config.get(key, default)
            elif isinstance(config, dict):
                return config.get(key, default)
            
            # Direct attribute access as last resort
            if '.' in key:
                parts = key.split('.')
                current = config
                for part in parts:
                    if not hasattr(current, part):
                        return default
                    current = getattr(current, part)
                return current
            
            return getattr(config, key, default)
        except (ImportError, AttributeError):
            logger.warning(f"Config key '{key}' not found in any configuration system")
            return default

def get_config_section(section: str) -> Dict[str, Any]:
    """
    Get a configuration section with support for both old and new systems.
    
    Args:
        section: Section name
        
    Returns:
        Configuration section as dictionary
    """
    # First try the new unified config
    try:
        from trading_bot.config.unified_config import get_config
        config = get_config()
        section_obj = config.get_section(section)
        if section_obj:
            return section_obj.as_dict()
    except (ImportError, AttributeError):
        pass
    
    # Fall back to old config manager
    try:
        from trading_bot.config_manager import ConfigManager
        config = ConfigManager.instance()
        return config.get_section(section) or {}
    except (ImportError, AttributeError):
        # Try other config patterns
        try:
            from trading_bot.config import config
            if hasattr(config, section):
                section_value = getattr(config, section)
                if isinstance(section_value, dict):
                    return section_value
                
            # Try dictionary-like access
            if isinstance(config, dict) and section in config:
                section_value = config[section]
                if isinstance(section_value, dict):
                    return section_value
        except (ImportError, AttributeError):
            pass
    
    logger.warning(f"Config section '{section}' not found in any configuration system")
    return {}

def set_config_value(key: str, value: Any, persist: bool = False) -> None:
    """
    Set a configuration value with support for both old and new systems.
    
    Args:
        key: Configuration key
        value: Value to set
        persist: Whether to persist the change
    """
    # First try the new unified config
    try:
        from trading_bot.config.unified_config import get_config
        config = get_config()
        config.set(key, value, persist)
        
        # Successfully set in new system, add warning about old system
        warnings.warn(
            f"Configuration '{key}' set in the new system only. "
            "Old configuration system not updated.",
            DeprecationWarning, stacklevel=2
        )
        return
    except ImportError:
        pass
    
    # Fall back to old config manager
    try:
        from trading_bot.config_manager import ConfigManager
        config = ConfigManager.instance()
        config.set(key, value)
        
        if persist:
            config.save()
        return
    except (ImportError, AttributeError):
        # Try other config patterns
        try:
            from trading_bot.config import config
            if hasattr(config, 'set'):
                config.set(key, value)
                return
            
            # Direct attribute setting for simple cases
            if '.' not in key:
                setattr(config, key, value)
                return
        except (ImportError, AttributeError):
            pass
    
    logger.error(f"Failed to set config key '{key}' in any configuration system")
