#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migration helpers for transitioning to the new organization structure.

This package provides tools to help maintain backward compatibility 
during the reorganization process.
"""

from .strategy_compat import get_strategy, create_strategy
from .config_compat import get_config_value, get_config_section
from .processor_compat import get_data_processor

__all__ = [
    'get_strategy',
    'create_strategy',
    'get_config_value',
    'get_config_section',
    'get_data_processor'
]
