#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Strategies Package

This package contains all forex trading strategies, organized by category.
"""

# Import subpackages
from . import trend
from . import range
from . import breakout
from . import carry

# Re-export all strategies for easy importing
from .trend import *
from .range import *
from .breakout import *
from .carry import *

# List of all submodules for strategy discovery
__all__ = (
    trend.__all__ +
    range.__all__ +
    breakout.__all__ +
    carry.__all__
)
