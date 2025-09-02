#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Analyzer Module

This module provides the main interface for pattern detection, scoring, and
integration with the trading strategy.
"""

import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# Import our pattern structure classes
from trading_bot.analysis.pattern_structure import (
    Pattern, PatternType, MarketContext, PatternRegistry
)

# Import pattern detectors
from trading_bot.analysis.pattern_recognition import PATTERN_DETECTORS
from trading_bot.analysis.pattern_recognition_ext import register_extended_detectors

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    """
    Main class for analyzing market patterns and providing trading signals
    based on pattern recognition.
    """
    
    def __init__(self, registry_file="data/patterns/pattern_registry.json"):
        """
        Initialize the pattern analyzer.
        
        Args:
            registry_file: Path to pattern registry file
        """
        # Initialize pattern registry
        os.makedirs(os.path.dirname(registry_file), exist_ok=True)
        self.registry = PatternRegistry.load_from_file(registry_file)
        self.registry_file = registry_file
        
        # Register all pattern detectors
        self.detectors = register_extended_detectors()
        
        # Track detected patterns
        self.recent_patterns = {}
        self.pattern_history = {}
    
    def detect_patterns(self, symbol: str, data: pd.DataFrame, 
                       market_context: Union[str, MarketContext] = None,
                       min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect all relevant patterns in the market data.
        
        Args:
            symbol: Trading symbol
            data: Market data in DataFrame format
            market_context: Current market context/regime
            min_confidence: Minimum confidence threshold for pattern detection
            
        Returns:
            List of detected patterns with match details
        """
        # Convert context to string if it's an enum
        context_value = market_context.value if hasattr(market_context, 'value') else market_context
        
        detected_patterns = []
        
        # If we have context, prioritize patterns that work well in this context
        if context_value:
            # Get patterns that have historically performed well in this context
            best_patterns = self.registry.get_best_patterns_for_context(
                context_value, min_success_rate=0.5, min_occurrences=3
            )
            
            # Also check some common patterns regardless of context
            all_patterns = best_patterns + [
                self.registry.get_pattern(name) for name in [
                    "pin_bar", "engulfing", "macd_crossover", "rsi_oversold_bounce", 
                    "rsi_overbought_drop"
                ] if self.registry.get_pattern(name) is not None
            ]
            
            # Remove duplicates
            seen = set()
            unique_patterns = []
            for p in all_patterns:
                if p.name not in seen:
                    seen.add(p.name)
                    unique_patterns.append(p)
                    
            all_patterns = unique_patterns
        else:
            # Without context, check all patterns
            all_patterns = [p for name, p in self.registry.patterns.items()]
        
        # Detect each pattern
        for pattern in all_patterns:
            detector = self.detectors.get(pattern.pattern_type.value)
            if not detector:
                continue
                
            try:
                result = detector.detect(data, pattern)
                
                # If pattern is detected with sufficient confidence, add to results
                if result["match"] and result["confidence"] >= min_confidence:
                    result["pattern_name"] = pattern.name
                    result["pattern_type"] = pattern.pattern_type.value
                    result["timestamp"] = data.index[-1]
                    detected_patterns.append(result)
            except Exception as e:
                logger.error(f"Error detecting pattern {pattern.name}: {str(e)}")
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Store recent patterns for this symbol
        self.recent_patterns[symbol] = detected_patterns
        
        # Add to pattern history
        if symbol not in self.pattern_history:
            self.pattern_history[symbol] = []
        
        for pattern in detected_patterns:
            self.pattern_history[symbol].append({
                "timestamp": pattern["timestamp"],
                "pattern_name": pattern["pattern_name"],
                "confidence": pattern["confidence"],
                "direction": pattern["direction"]
            })
            
            # Keep history manageable
            if len(self.pattern_history[symbol]) > 100:
                self.pattern_history[symbol] = self.pattern_history[symbol][-100:]
        
        return detected_patterns
    
    def get_trade_signal(self, symbol: str, data: pd.DataFrame, 
                        market_context: Union[str, MarketContext],
                        min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Get the strongest trade signal based on pattern detection.
        
        Args:
            symbol: Trading symbol
            data: Market data in DataFrame format
            market_context: Current market context/regime
            min_confidence: Minimum confidence threshold for signaling a trade
            
        Returns:
            Dictionary with trade signal information
        """
        # Detect patterns
        patterns = self.detect_patterns(symbol, data, market_context, min_confidence)
        
        if not patterns:
            return {
                "signal": "none",
                "confidence": 0.0,
                "pattern": None,
                "direction": None,
                "metadata": {}
            }
        
        # Get highest confidence pattern
        best_pattern = patterns[0]
        
        # Build trade signal
        signal = {
            "signal": "buy" if best_pattern["direction"] == "buy" else 
                     "sell" if best_pattern["direction"] == "sell" else "none",
            "confidence": best_pattern["confidence"],
            "pattern": best_pattern["pattern_name"],
            "pattern_type": best_pattern["pattern_type"],
            "direction": best_pattern["direction"],
            "metadata": {
                "location": best_pattern["location"],
                "pattern_data": best_pattern["metadata"]
            }
        }
        
        return signal
    
    def update_pattern_performance(self, pattern_name: str, success: bool, 
                                  profit_pips: float, market_context: Union[str, MarketContext]):
        """
        Update pattern performance metrics after a trade.
        
        Args:
            pattern_name: Name of the pattern used for the trade
            success: Whether the trade was successful
            profit_pips: Profit/loss in pips
            market_context: Market context when pattern was used
        """
        pattern = self.registry.get_pattern(pattern_name)
        if not pattern:
            logger.warning(f"Cannot update unknown pattern: {pattern_name}")
            return
        
        # Update pattern performance
        pattern.update_performance(success, profit_pips, market_context)
        
        # Update pattern-context mapping
        self.registry.update_pattern_context_performance(pattern_name, market_context)
        
        # Save registry to file
        self.registry.save_to_file(self.registry_file)
        
        logger.info(f"Updated {pattern_name} performance: success={success}, profit={profit_pips}, context={market_context}")
    
    def get_best_patterns_for_regime(self, regime: str, min_success_rate: float = 0.6,
                                  min_occurrences: int = 5) -> List[Dict[str, Any]]:
        """
        Get the best performing patterns for a specific market regime.
        
        Args:
            regime: Market regime to get patterns for
            min_success_rate: Minimum success rate
            min_occurrences: Minimum number of occurrences
            
        Returns:
            List of patterns with performance metrics
        """
        patterns = self.registry.get_best_patterns_for_context(
            regime, min_success_rate, min_occurrences
        )
        
        return [
            {
                "name": p.name,
                "type": p.pattern_type.value,
                "success_rate": p.best_context.get(regime, {}).get("success_rate", 0),
                "occurrences": p.best_context.get(regime, {}).get("occurrences", 0),
                "avg_profit": p.avg_profit_pips,
                "description": p.description
            }
            for p in patterns
        ]
    
    def add_custom_pattern(self, name: str, pattern_type: PatternType, 
                          description: str, parameters: Dict[str, Any]) -> bool:
        """
        Add a new custom pattern to the registry.
        
        Args:
            name: Unique pattern name
            pattern_type: Type of pattern
            description: Description of the pattern
            parameters: Pattern-specific parameters
            
        Returns:
            True if pattern was added successfully
        """
        # Create appropriate pattern class based on type
        if pattern_type == PatternType.PRICE_ACTION:
            from trading_bot.analysis.pattern_structure import PriceActionPattern
            pattern = PriceActionPattern(name, description)
        elif pattern_type == PatternType.CANDLESTICK:
            from trading_bot.analysis.pattern_structure import CandlestickPattern
            pattern = CandlestickPattern(name, description)
        elif pattern_type == PatternType.CHART_FORMATION:
            from trading_bot.analysis.pattern_structure import ChartFormationPattern
            pattern = ChartFormationPattern(name, description)
        elif pattern_type == PatternType.INDICATOR_SIGNAL:
            from trading_bot.analysis.pattern_structure import IndicatorSignalPattern
            pattern = IndicatorSignalPattern(name, description)
        elif pattern_type == PatternType.VOLATILITY_BASED:
            from trading_bot.analysis.pattern_structure import VolatilityPattern
            pattern = VolatilityPattern(name, description)
        else:
            from trading_bot.analysis.pattern_structure import Pattern
            pattern = Pattern(name, pattern_type, description)
        
        # Update parameters
        pattern.parameters.update(parameters)
        
        # Register pattern
        success = self.registry.register_pattern(pattern)
        
        # Save registry
        if success:
            self.registry.save_to_file(self.registry_file)
            logger.info(f"Added new pattern: {name} ({pattern_type.value})")
        
        return success
    
    def generate_pattern_report(self) -> Dict[str, Any]:
        """Generate a report of pattern performance across different regimes"""
        regimes = [context.value for context in MarketContext]
        
        report = {
            "patterns_by_regime": {},
            "overall_performance": {},
            "total_patterns": len(self.registry.patterns),
            "regimes_analyzed": regimes
        }
        
        # Get best patterns for each regime
        for regime in regimes:
            report["patterns_by_regime"][regime] = self.get_best_patterns_for_regime(
                regime, min_success_rate=0.5, min_occurrences=3
            )
        
        # Get overall performance stats for each pattern
        for name, pattern in self.registry.patterns.items():
            report["overall_performance"][name] = {
                "success_rate": pattern.success_rate,
                "occurrences": pattern.occurrences,
                "successes": pattern.successes,
                "failures": pattern.failures,
                "avg_profit_pips": pattern.avg_profit_pips,
                "avg_loss_pips": pattern.avg_loss_pips,
                "best_regimes": sorted(
                    [(regime, data["success_rate"]) 
                     for regime, data in pattern.best_context.items()
                     if data["occurrences"] >= 3],
                    key=lambda x: x[1],
                    reverse=True
                )
            }
        
        return report
