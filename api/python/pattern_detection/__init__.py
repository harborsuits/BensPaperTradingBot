"""
Pattern Detection Module

This module provides comprehensive chart pattern detection capabilities with
implementations of classic technical analysis patterns across multiple timeframes.
"""

from .pattern_engine import PatternDetectionEngine
from .pattern_definitions import (
    ChartPattern, 
    DoubleBottom, 
    DoubleTop, 
    BullFlag, 
    BearFlag, 
    HeadAndShoulders,
    get_all_patterns
)

__all__ = [
    'PatternDetectionEngine',
    'ChartPattern',
    'DoubleBottom',
    'DoubleTop',
    'BullFlag',
    'BearFlag',
    'HeadAndShoulders',
    'get_all_patterns'
] 