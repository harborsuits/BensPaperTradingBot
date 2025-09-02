#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Structure Module

This module defines the structure and classification of trading patterns
used for pattern recognition and matching.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Types of patterns that can be recognized"""
    PRICE_ACTION = "price_action"
    CANDLESTICK = "candlestick"
    CHART_FORMATION = "chart_formation"
    INDICATOR_SIGNAL = "indicator_signal"
    MULTI_TIMEFRAME = "multi_timeframe"
    VOLATILITY_BASED = "volatility_based"
    VOLUME_BASED = "volume_based"
    CUSTOM = "custom"

class MarketContext(Enum):
    """Market context classifications"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CHOPPY = "choppy"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"

class Pattern:
    """Base class for all trading patterns"""
    
    def __init__(self, name: str, pattern_type: PatternType, description: str = ""):
        """
        Initialize a pattern definition.
        
        Args:
            name: Unique name for the pattern
            pattern_type: Type classification of the pattern
            description: Human-readable description of the pattern
        """
        self.name = name
        self.pattern_type = pattern_type
        self.description = description
        self.created_at = datetime.now()
        self.modified_at = self.created_at
        self.occurrences = 0
        self.success_rate = 0.0
        self.successes = 0
        self.failures = 0
        self.avg_profit_pips = 0.0
        self.avg_loss_pips = 0.0
        self.best_context = {}
        self.parameters = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for storage"""
        return {
            "name": self.name,
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "occurrences": self.occurrences,
            "success_rate": self.success_rate,
            "successes": self.successes,
            "failures": self.failures,
            "avg_profit_pips": self.avg_profit_pips,
            "avg_loss_pips": self.avg_loss_pips,
            "best_context": self.best_context,
            "parameters": self.parameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Create pattern from dictionary"""
        pattern = cls(
            name=data["name"],
            pattern_type=PatternType(data["pattern_type"]),
            description=data.get("description", "")
        )
        pattern.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        pattern.modified_at = datetime.fromisoformat(data.get("modified_at", datetime.now().isoformat()))
        pattern.occurrences = data.get("occurrences", 0)
        pattern.success_rate = data.get("success_rate", 0.0)
        pattern.successes = data.get("successes", 0)
        pattern.failures = data.get("failures", 0)
        pattern.avg_profit_pips = data.get("avg_profit_pips", 0.0)
        pattern.avg_loss_pips = data.get("avg_loss_pips", 0.0)
        pattern.best_context = data.get("best_context", {})
        pattern.parameters = data.get("parameters", {})
        return pattern
    
    def update_performance(self, success: bool, profit_pips: float, context: MarketContext):
        """
        Update pattern performance metrics.
        
        Args:
            success: Whether the pattern-based trade was successful
            profit_pips: Profit/loss in pips (positive for profit, negative for loss)
            context: Market context when the pattern was observed
        """
        self.occurrences += 1
        context_value = context.value if isinstance(context, MarketContext) else context
        
        if success:
            self.successes += 1
            if profit_pips > 0:
                self.avg_profit_pips = ((self.avg_profit_pips * (self.successes - 1)) + profit_pips) / self.successes
        else:
            self.failures += 1
            if profit_pips < 0:
                self.avg_loss_pips = ((self.avg_loss_pips * (self.failures - 1)) + profit_pips) / self.failures
        
        if self.occurrences > 0:
            self.success_rate = self.successes / self.occurrences
        
        # Track pattern performance by context
        if context_value not in self.best_context:
            self.best_context[context_value] = {
                "occurrences": 0, 
                "success_rate": 0.0,
                "successes": 0,
                "failures": 0
            }
        
        self.best_context[context_value]["occurrences"] += 1
        if success:
            self.best_context[context_value]["successes"] += 1
        else:
            self.best_context[context_value]["failures"] += 1
            
        context_occurrences = self.best_context[context_value]["occurrences"]
        context_successes = self.best_context[context_value]["successes"]
        
        if context_occurrences > 0:
            self.best_context[context_value]["success_rate"] = context_successes / context_occurrences
        
        self.modified_at = datetime.now()

class PriceActionPattern(Pattern):
    """Price action pattern definitions"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, PatternType.PRICE_ACTION, description)
        
        # Price action specific parameters
        self.parameters = {
            "bar_count": 3,          # Number of bars to consider
            "min_body_size": 0.5,    # Minimum candle body size (% of range)
            "max_wick_ratio": 0.3,   # Maximum wick to body ratio
            "confirmation_bars": 1    # Number of confirmation bars required
        }

class CandlestickPattern(Pattern):
    """Candlestick pattern definitions"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, PatternType.CANDLESTICK, description)
        
        # Candlestick specific parameters
        self.parameters = {
            "bar_count": 1,           # Number of bars in the pattern
            "confirmation_bars": 1,    # Number of confirmation bars
            "min_size_atr": 0.5,       # Minimum size as multiple of ATR
            "nearby_support": False,   # Whether pattern is near support/resistance
        }

class ChartFormationPattern(Pattern):
    """Chart formation pattern definitions"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, PatternType.CHART_FORMATION, description)
        
        # Chart formation specific parameters
        self.parameters = {
            "min_points": 3,          # Minimum points required to form pattern
            "max_deviation": 0.1,     # Maximum deviation from ideal pattern
            "min_duration_bars": 10,  # Minimum bars to form the pattern
            "volume_confirmation": False,  # Requires volume confirmation
        }

class IndicatorSignalPattern(Pattern):
    """Indicator-based signal pattern definitions"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, PatternType.INDICATOR_SIGNAL, description)
        
        # Indicator signal specific parameters
        self.parameters = {
            "indicators": [],          # List of indicators used
            "crossover_type": None,    # Type of crossover if applicable
            "oversold_threshold": 30,  # Oversold threshold for oscillators
            "overbought_threshold": 70,# Overbought threshold for oscillators
            "divergence_required": False, # Whether divergence confirmation is needed
        }

class MultiTimeframePattern(Pattern):
    """Multi-timeframe pattern definitions"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, PatternType.MULTI_TIMEFRAME, description)
        
        # Multi-timeframe specific parameters
        self.parameters = {
            "timeframes": [],          # List of timeframes to consider
            "alignment_required": True, # Whether all timeframes must align
            "primary_timeframe": None, # Primary timeframe for signal
            "confirmation_timeframes": [], # Timeframes needed for confirmation
        }

class VolatilityPattern(Pattern):
    """Volatility-based pattern definitions"""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, PatternType.VOLATILITY_BASED, description)
        
        # Volatility specific parameters
        self.parameters = {
            "atr_threshold": 1.5,      # ATR threshold multiplier
            "bollinger_width": 2.0,    # Bollinger Band width standard deviations
            "contraction_period": 10,  # Bars to measure volatility contraction
            "expansion_threshold": 1.5, # Threshold for volatility expansion
        }

class PatternRegistry:
    """Registry of all known trading patterns"""
    
    def __init__(self):
        """Initialize the pattern registry"""
        self.patterns = {}
        self.pattern_by_context = {context.value: [] for context in MarketContext}
        self._initialize_common_patterns()
    
    def _initialize_common_patterns(self):
        """Initialize common built-in patterns"""
        # Price Action Patterns
        self.register_pattern(PriceActionPattern(
            "pin_bar", 
            "A pin bar with a long wick showing rejection in one direction"
        ))
        
        self.register_pattern(PriceActionPattern(
            "inside_bar",
            "A bar fully contained within the high-low range of the previous bar"
        ))
        
        self.register_pattern(PriceActionPattern(
            "outside_bar",
            "A bar that fully contains the previous bar's high-low range"
        ))
        
        # Candlestick Patterns
        self.register_pattern(CandlestickPattern(
            "doji",
            "A candle with very small body showing indecision"
        ))
        
        self.register_pattern(CandlestickPattern(
            "engulfing",
            "A candle that completely engulfs the previous candle"
        ))
        
        self.register_pattern(CandlestickPattern(
            "evening_star",
            "Three-candle bearish reversal pattern at the top of an uptrend"
        ))
        
        self.register_pattern(CandlestickPattern(
            "morning_star",
            "Three-candle bullish reversal pattern at the bottom of a downtrend"
        ))
        
        # Chart Formations
        self.register_pattern(ChartFormationPattern(
            "double_top",
            "Price forms two distinct peaks at approximately the same level"
        ))
        
        self.register_pattern(ChartFormationPattern(
            "double_bottom",
            "Price forms two distinct bottoms at approximately the same level"
        ))
        
        self.register_pattern(ChartFormationPattern(
            "head_and_shoulders",
            "Three-peak pattern with middle peak higher than the other two"
        ))
        
        # Indicator Signals
        self.register_pattern(IndicatorSignalPattern(
            "macd_crossover",
            "MACD line crosses above or below the signal line"
        ))
        
        self.register_pattern(IndicatorSignalPattern(
            "rsi_oversold_bounce",
            "RSI moves from oversold back above 30"
        ))
        
        self.register_pattern(IndicatorSignalPattern(
            "rsi_overbought_drop",
            "RSI moves from overbought back below 70"
        ))
        
        # Volatility Patterns
        self.register_pattern(VolatilityPattern(
            "bollinger_squeeze",
            "Bollinger bands contract showing decreased volatility before expansion"
        ))
        
        self.register_pattern(VolatilityPattern(
            "volatility_breakout",
            "Price breaks out after a period of low volatility"
        ))
    
    def register_pattern(self, pattern: Pattern) -> bool:
        """
        Register a new pattern with the registry.
        
        Args:
            pattern: The pattern object to register
            
        Returns:
            True if registration was successful, False if pattern already exists
        """
        if pattern.name in self.patterns:
            logger.warning(f"Pattern {pattern.name} already exists in registry")
            return False
        
        self.patterns[pattern.name] = pattern
        return True
    
    def get_pattern(self, name: str) -> Optional[Pattern]:
        """
        Get a pattern by name.
        
        Args:
            name: The name of the pattern to retrieve
            
        Returns:
            The pattern object if found, None otherwise
        """
        return self.patterns.get(name)
    
    def update_pattern_context_performance(self, pattern_name: str, context: Union[str, MarketContext]):
        """
        Update the pattern-context mapping based on performance.
        
        Args:
            pattern_name: The name of the pattern to update
            context: The market context to associate with
        """
        pattern = self.get_pattern(pattern_name)
        if not pattern:
            return
        
        context_value = context.value if isinstance(context, MarketContext) else context
        
        # If this pattern is not yet in this context's list, add it
        if pattern_name not in self.pattern_by_context[context_value]:
            self.pattern_by_context[context_value].append(pattern_name)
            
        # Sort patterns in each context by success rate
        for ctx, patterns in self.pattern_by_context.items():
            sorted_patterns = []
            for pat_name in patterns:
                pat = self.get_pattern(pat_name)
                if pat and ctx in pat.best_context:
                    ctx_success_rate = pat.best_context[ctx]["success_rate"]
                    sorted_patterns.append((pat_name, ctx_success_rate))
            
            # Sort by success rate in descending order
            sorted_patterns.sort(key=lambda x: x[1], reverse=True)
            self.pattern_by_context[ctx] = [p[0] for p in sorted_patterns]
    
    def get_best_patterns_for_context(self, context: Union[str, MarketContext], min_success_rate: float = 0.5, 
                                      min_occurrences: int = 5) -> List[Pattern]:
        """
        Get the best performing patterns for a given market context.
        
        Args:
            context: The market context to get patterns for
            min_success_rate: Minimum success rate required
            min_occurrences: Minimum number of occurrences required
            
        Returns:
            List of pattern objects that perform well in the given context
        """
        context_value = context.value if isinstance(context, MarketContext) else context
        
        if context_value not in self.pattern_by_context:
            return []
        
        best_patterns = []
        for pattern_name in self.pattern_by_context[context_value]:
            pattern = self.get_pattern(pattern_name)
            if not pattern:
                continue
                
            if context_value in pattern.best_context:
                ctx_data = pattern.best_context[context_value]
                if (ctx_data["occurrences"] >= min_occurrences and 
                    ctx_data["success_rate"] >= min_success_rate):
                    best_patterns.append(pattern)
        
        return best_patterns
    
    def save_to_file(self, filename: str):
        """Save pattern registry to file"""
        data = {
            "patterns": {name: pattern.to_dict() for name, pattern in self.patterns.items()},
            "pattern_by_context": self.pattern_by_context
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'PatternRegistry':
        """Load pattern registry from file"""
        import json
        import os
        
        registry = cls()
        
        if not os.path.exists(filename):
            logger.warning(f"Pattern registry file {filename} not found, using default patterns")
            return registry
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Clear existing patterns
            registry.patterns = {}
            registry.pattern_by_context = {context.value: [] for context in MarketContext}
            
            # Load patterns
            for name, pattern_data in data.get("patterns", {}).items():
                pattern_type = pattern_data.get("pattern_type")
                
                if pattern_type == PatternType.PRICE_ACTION.value:
                    pattern = PriceActionPattern.from_dict(pattern_data)
                elif pattern_type == PatternType.CANDLESTICK.value:
                    pattern = CandlestickPattern.from_dict(pattern_data)
                elif pattern_type == PatternType.CHART_FORMATION.value:
                    pattern = ChartFormationPattern.from_dict(pattern_data)
                elif pattern_type == PatternType.INDICATOR_SIGNAL.value:
                    pattern = IndicatorSignalPattern.from_dict(pattern_data)
                elif pattern_type == PatternType.MULTI_TIMEFRAME.value:
                    pattern = MultiTimeframePattern.from_dict(pattern_data)
                elif pattern_type == PatternType.VOLATILITY_BASED.value:
                    pattern = VolatilityPattern.from_dict(pattern_data)
                else:
                    pattern = Pattern.from_dict(pattern_data)
                
                registry.patterns[name] = pattern
            
            # Load context mappings
            registry.pattern_by_context = data.get("pattern_by_context", {})
            
            logger.info(f"Loaded {len(registry.patterns)} patterns from {filename}")
            return registry
            
        except Exception as e:
            logger.error(f"Error loading pattern registry from {filename}: {e}")
            return registry
