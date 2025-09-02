#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Integration Module

This module integrates pattern recognition with the trading strategy
and contextual awareness system.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union

# Import our components
from trading_bot.analysis.pattern_analyzer import PatternAnalyzer
from trading_bot.analysis.pattern_structure import MarketContext

logger = logging.getLogger(__name__)

class PatternIntegrationManager:
    """
    Integrates pattern recognition with trading strategies by providing
    pattern-aware entry and exit signals.
    """
    
    def __init__(self, event_bus=None, pattern_registry_file="data/patterns/pattern_registry.json"):
        """
        Initialize the pattern integration manager.
        
        Args:
            event_bus: Event bus for event-driven communication
            pattern_registry_file: File to store pattern registry
        """
        self.event_bus = event_bus
        self.pattern_analyzer = PatternAnalyzer(registry_file=pattern_registry_file)
        self.current_patterns = {}
        self.active_pattern_trades = {}
    
    def map_regime_to_context(self, regime: str) -> MarketContext:
        """
        Map a market regime to a pattern context.
        
        Args:
            regime: Market regime from the regime detector
            
        Returns:
            Corresponding MarketContext enum value
        """
        # Map regime names to MarketContext values
        regime_map = {
            "trending_up": MarketContext.TRENDING_UP,
            "trending_down": MarketContext.TRENDING_DOWN,
            "ranging": MarketContext.RANGING,
            "breakout": MarketContext.BREAKOUT,
            "reversal": MarketContext.REVERSAL,
            "choppy": MarketContext.CHOPPY,
            "volatile": MarketContext.VOLATILE,
            "low_volatility": MarketContext.LOW_VOLATILITY
        }
        
        return regime_map.get(regime.lower(), MarketContext.UNKNOWN)
    
    def analyze_patterns(self, symbol: str, data: pd.DataFrame, market_regime: str) -> Dict[str, Any]:
        """
        Analyze current market data for patterns and return analysis results.
        
        Args:
            symbol: The trading symbol
            data: Market data as DataFrame
            market_regime: Current market regime
            
        Returns:
            Dictionary with pattern analysis results
        """
        # Map regime to pattern context
        context = self.map_regime_to_context(market_regime)
        
        # Detect patterns
        patterns = self.pattern_analyzer.detect_patterns(
            symbol=symbol,
            data=data,
            market_context=context,
            min_confidence=0.5  # Lower threshold for detection
        )
        
        # Store current patterns
        self.current_patterns[symbol] = patterns
        
        # Build result dictionary
        result = {
            "has_patterns": len(patterns) > 0,
            "pattern_count": len(patterns),
            "patterns": patterns,
            "best_pattern": patterns[0] if patterns else None,
            "regime": market_regime,
            "context": context.value
        }
        
        # Calculate combined signal strength and direction
        if patterns:
            buy_signals = [p for p in patterns if p["direction"] == "buy"]
            sell_signals = [p for p in patterns if p["direction"] == "sell"]
            
            buy_strength = sum(p["confidence"] for p in buy_signals)
            sell_strength = sum(p["confidence"] for p in sell_signals)
            
            result["buy_strength"] = buy_strength
            result["sell_strength"] = sell_strength
            result["signal_bias"] = "buy" if buy_strength > sell_strength else "sell" if sell_strength > buy_strength else "neutral"
            result["signal_strength"] = max(buy_strength, sell_strength)
        else:
            result["buy_strength"] = 0.0
            result["sell_strength"] = 0.0
            result["signal_bias"] = "neutral"
            result["signal_strength"] = 0.0
        
        return result
    
    def get_trade_signal(self, symbol: str, data: pd.DataFrame, market_regime: str, 
                        min_confidence: float = 0.7) -> Dict[str, Any]:
        """
        Get a trade signal based on pattern recognition.
        
        Args:
            symbol: Trading symbol
            data: Market data
            market_regime: Current market regime
            min_confidence: Minimum confidence for trade signals
            
        Returns:
            Dictionary with trade signal information
        """
        # Map regime to pattern context
        context = self.map_regime_to_context(market_regime)
        
        # Get trade signal
        signal = self.pattern_analyzer.get_trade_signal(
            symbol=symbol,
            data=data,
            market_context=context,
            min_confidence=min_confidence
        )
        
        # Enhanced signal with additional information
        enhanced_signal = signal.copy()
        enhanced_signal["market_regime"] = market_regime
        
        # Add regime compatibility
        if signal["pattern"] and signal["direction"] != "none":
            # Get the match reliability for this pattern in the current regime
            best_patterns = self.pattern_analyzer.get_best_patterns_for_regime(
                regime=context.value, min_success_rate=0.0, min_occurrences=0
            )
            
            regime_compatibility = 0.5  # Default moderate compatibility
            for p in best_patterns:
                if p["name"] == signal["pattern"]:
                    regime_compatibility = p["success_rate"]
                    break
            
            enhanced_signal["regime_compatibility"] = regime_compatibility
            
            # Adjust confidence based on regime compatibility
            enhanced_signal["adjusted_confidence"] = signal["confidence"] * (0.5 + (regime_compatibility / 2))
            
            # Track active pattern trade
            if signal["signal"] != "none":
                self.active_pattern_trades[symbol] = {
                    "pattern": signal["pattern"],
                    "direction": signal["direction"],
                    "entry_time": data.index[-1],
                    "regime": market_regime,
                    "confidence": enhanced_signal["adjusted_confidence"]
                }
        else:
            enhanced_signal["regime_compatibility"] = 0.0
            enhanced_signal["adjusted_confidence"] = 0.0
        
        return enhanced_signal
    
    def update_trade_outcome(self, symbol: str, success: bool, profit_pips: float, 
                            pattern_name: str = None, market_regime: str = None):
        """
        Update pattern performance based on trade outcome.
        
        Args:
            symbol: Trading symbol
            success: Whether the trade was successful
            profit_pips: Profit/loss in pips
            pattern_name: Name of the pattern used (optional)
            market_regime: Market regime during the trade (optional)
        """
        # If pattern name and regime not specified, try to get from active trades
        if pattern_name is None or market_regime is None:
            if symbol in self.active_pattern_trades:
                trade_info = self.active_pattern_trades[symbol]
                pattern_name = pattern_name or trade_info.get("pattern")
                market_regime = market_regime or trade_info.get("regime")
                
                # Clear active trade
                del self.active_pattern_trades[symbol]
        
        # If we have a pattern name and regime, update performance
        if pattern_name and market_regime:
            context = self.map_regime_to_context(market_regime)
            
            self.pattern_analyzer.update_pattern_performance(
                pattern_name=pattern_name,
                success=success,
                profit_pips=profit_pips,
                market_context=context
            )
            
            logger.info(f"Updated pattern {pattern_name} performance in {market_regime}: {'success' if success else 'failure'}, {profit_pips} pips")
        else:
            logger.warning(f"Could not update pattern performance for {symbol}: missing pattern or regime information")
    
    def get_pattern_enhanced_params(self, symbol: str, base_params: Dict[str, Any], 
                                   market_regime: str) -> Dict[str, Any]:
        """
        Enhance strategy parameters based on pattern recognition.
        
        Args:
            symbol: Trading symbol
            base_params: Base strategy parameters
            market_regime: Current market regime
            
        Returns:
            Enhanced parameters incorporating pattern insights
        """
        enhanced_params = base_params.copy()
        
        # Get current patterns
        patterns = self.current_patterns.get(symbol, [])
        
        if not patterns:
            return enhanced_params
            
        # Get best pattern
        best_pattern = patterns[0]
        
        # Enhance TP/SL based on pattern type and direction
        if best_pattern["pattern_type"] == "price_action" or best_pattern["pattern_type"] == "candlestick":
            # For reversal patterns, we often want tighter stops
            if best_pattern["pattern_name"] in ["pin_bar", "engulfing", "evening_star", "morning_star"]:
                enhanced_params["tp_sl_ratio"] = max(1.5, base_params.get("tp_sl_ratio", 2.0) * 0.8)
                enhanced_params["use_tight_stop"] = True
                
        elif best_pattern["pattern_type"] == "chart_formation":
            # For chart formations, we can often use larger profit targets
            if best_pattern["pattern_name"] in ["double_top", "double_bottom", "head_and_shoulders"]:
                enhanced_params["tp_sl_ratio"] = min(3.0, base_params.get("tp_sl_ratio", 2.0) * 1.3)
                
                # Adjust trailing stop settings based on pattern
                enhanced_params["use_trailing_stop"] = True
                enhanced_params["trailing_activation"] = 0.6
                enhanced_params["trailing_distance"] = 0.5
        
        # Volatility-based adjustments
        if best_pattern["pattern_type"] == "volatility_based":
            if best_pattern["pattern_name"] == "bollinger_squeeze":
                # For volatility expansion patterns, wider stops may be needed
                enhanced_params["stop_loss_multiplier"] = 1.25
                enhanced_params["tp_sl_ratio"] = 2.5
                
            elif best_pattern["pattern_name"] == "volatility_breakout":
                # For breakout patterns, trailing stops are valuable
                enhanced_params["use_trailing_stop"] = True
                enhanced_params["trailing_activation"] = 0.4
                enhanced_params["trailing_distance"] = 0.4
        
        # Add pattern information to parameters
        enhanced_params["pattern_name"] = best_pattern["pattern_name"]
        enhanced_params["pattern_confidence"] = best_pattern["confidence"]
        enhanced_params["pattern_direction"] = best_pattern["direction"]
        
        # Ensure parameters stay within reasonable bounds
        enhanced_params["tp_sl_ratio"] = max(1.0, min(5.0, enhanced_params.get("tp_sl_ratio", 2.0)))
        
        return enhanced_params
    
    def get_pattern_report(self) -> Dict[str, Any]:
        """Get a comprehensive report on pattern performance"""
        return self.pattern_analyzer.generate_pattern_report()
    
    def get_best_patterns_for_regime(self, regime: str) -> List[Dict[str, Any]]:
        """Get best patterns for a specific regime"""
        return self.pattern_analyzer.get_best_patterns_for_regime(
            regime=regime,
            min_success_rate=0.0,  # Include all patterns with some data
            min_occurrences=0      # Include all patterns with any occurrences
        )
