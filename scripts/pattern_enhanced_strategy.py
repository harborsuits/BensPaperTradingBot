#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern-Enhanced Contextual Strategy

This module extends the enhanced contextual strategy with pattern recognition
capabilities for more precise entry and exit signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
import os
from datetime import datetime

# Import the simplified pattern recognition system
from standalone_pattern_test import (
    PatternType, MarketContext, 
    PriceActionDetector, CandlestickDetector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PatternEnhancedStrategy")

class PatternRecognizer:
    """
    Pattern recognition system that works with the contextual strategy.
    """
    
    def __init__(self):
        """Initialize the pattern recognizer"""
        self.price_action_detector = PriceActionDetector()
        self.candlestick_detector = CandlestickDetector()
        
        # Pattern performance stats
        self.pattern_stats = {
            "by_pattern": {},
            "by_regime": {}
        }
    
    def detect_patterns(self, data: pd.DataFrame, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """
        Detect patterns in the provided market data.
        
        Args:
            data: Market data as DataFrame
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of detected patterns
        """
        # Patterns to check
        patterns_to_check = [
            {"name": "pin_bar", "type": PatternType.PRICE_ACTION},
            {"name": "engulfing", "type": PatternType.CANDLESTICK},
            {"name": "doji", "type": PatternType.CANDLESTICK},
            {"name": "inside_bar", "type": PatternType.PRICE_ACTION},
            {"name": "morning_star", "type": PatternType.CANDLESTICK}
        ]
        
        # Results storage
        detected_patterns = []
        
        # Check each pattern
        for pattern in patterns_to_check:
            result = None
            
            if pattern["type"] == PatternType.PRICE_ACTION:
                result = self.price_action_detector.detect_pattern(data, pattern["name"])
            elif pattern["type"] == PatternType.CANDLESTICK:
                result = self.candlestick_detector.detect_pattern(data, pattern["name"])
            
            if result and result["match"] and result["confidence"] >= min_confidence:
                # Add timestamp
                result["timestamp"] = data.index[-1]
                detected_patterns.append(result)
        
        # Sort by confidence
        detected_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        
        return detected_patterns
    
    def update_pattern_performance(self, pattern_name: str, 
                                  success: bool, profit_pips: float, 
                                  regime: str) -> None:
        """
        Update pattern performance metrics.
        
        Args:
            pattern_name: Name of the pattern
            success: Whether the trade was successful 
            profit_pips: Profit/loss in pips
            regime: Market regime during the trade
        """
        # Update by pattern
        if pattern_name not in self.pattern_stats["by_pattern"]:
            self.pattern_stats["by_pattern"][pattern_name] = {
                "successes": 0,
                "failures": 0,
                "total_pips": 0,
                "win_rate": 0.0
            }
        
        stats = self.pattern_stats["by_pattern"][pattern_name]
        
        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        
        stats["total_pips"] += profit_pips
        stats["win_rate"] = stats["successes"] / (stats["successes"] + stats["failures"])
        
        # Update by regime
        if regime not in self.pattern_stats["by_regime"]:
            self.pattern_stats["by_regime"][regime] = {}
        
        if pattern_name not in self.pattern_stats["by_regime"][regime]:
            self.pattern_stats["by_regime"][regime][pattern_name] = {
                "successes": 0,
                "failures": 0,
                "total_pips": 0,
                "win_rate": 0.0
            }
        
        regime_stats = self.pattern_stats["by_regime"][regime][pattern_name]
        
        if success:
            regime_stats["successes"] += 1
        else:
            regime_stats["failures"] += 1
        
        regime_stats["total_pips"] += profit_pips
        regime_stats["win_rate"] = regime_stats["successes"] / (regime_stats["successes"] + regime_stats["failures"])
    
    def get_best_patterns_for_regime(self, regime: str, 
                                    min_occurrences: int = 3,
                                    min_win_rate: float = 0.5) -> List[str]:
        """
        Get the best performing patterns for a specific market regime.
        
        Args:
            regime: Market regime
            min_occurrences: Minimum number of occurrences required
            min_win_rate: Minimum win rate required
            
        Returns:
            List of pattern names sorted by win rate
        """
        if regime not in self.pattern_stats["by_regime"]:
            return []
        
        best_patterns = []
        
        for pattern_name, stats in self.pattern_stats["by_regime"][regime].items():
            total_trades = stats["successes"] + stats["failures"]
            
            if total_trades >= min_occurrences and stats["win_rate"] >= min_win_rate:
                best_patterns.append((pattern_name, stats["win_rate"]))
        
        # Sort by win rate
        best_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in best_patterns]

class PatternEnhancedStrategy:
    """
    Enhanced contextual trading strategy with pattern recognition.
    """
    
    def __init__(self, event_bus=None, initial_balance=10000.0):
        """
        Initialize the pattern-enhanced strategy.
        
        Args:
            event_bus: Event bus for event-driven communication
            initial_balance: Initial account balance
        """
        self.event_bus = event_bus
        self.balance = initial_balance
        
        # Pattern recognition
        self.pattern_recognizer = PatternRecognizer()
        
        # Current context tracking
        self.current_context = {
            "market_regime": "unknown",
            "volatility_state": "medium",
            "bars_since_regime_change": 0
        }
        
        # Last trade tracking
        self.last_trade_bar = {}
        self.active_trades = {}
    
    def update_context(self, symbol: str, market_regime: str, 
                      volatility_state: str = "medium") -> None:
        """
        Update the current market context.
        
        Args:
            symbol: Trading symbol
            market_regime: Current market regime
            volatility_state: Current volatility state
        """
        prev_regime = self.current_context.get("market_regime", "unknown")
        
        # Update context
        self.current_context["market_regime"] = market_regime
        self.current_context["volatility_state"] = volatility_state
        
        # Reset bars since regime change if regime changed
        if prev_regime != market_regime:
            self.current_context["bars_since_regime_change"] = 0
            logger.info(f"Market regime changed from {prev_regime} to {market_regime}")
        else:
            self.current_context["bars_since_regime_change"] += 1
    
    def update_balance(self, new_balance: float) -> None:
        """Update the account balance"""
        self.balance = new_balance
    
    def select_strategy(self, symbol: str, data: pd.DataFrame, recent_trades=None) -> Dict[str, Any]:
        """
        Select the best strategy based on market conditions and patterns.
        
        Args:
            symbol: Trading symbol
            data: Market data
            recent_trades: Recent trades for this symbol, if available
            
        Returns:
            Dictionary with strategy details and signal
        """
        market_regime = self.current_context.get("market_regime", "unknown")
        volatility_state = self.current_context.get("volatility_state", "medium")
        bars_since_change = self.current_context.get("bars_since_regime_change", 0)
        current_bar = data.iloc[-1]
        prev_bar = data.iloc[-2] if len(data) > 1 else None
        
        # Default strategy with no signal
        strategy = {
            "id": "contextual_pattern",
            "name": "Contextual Pattern Strategy",
            "signal": "none",
            "direction": None,
            "tp_sl_ratio": 2.0,
            "use_trailing_stop": False,
            "skip_trading": False,
            "entry_conditions": {}
        }
        
        # 1. Check for regime change buffer - enhanced version
        # Longer buffer for more volatile regimes
        if market_regime == "breakout":
            regime_buffer = 5  # Wait longer after breakout regime
        else:
            regime_buffer = 3  # Normal buffer
            
        if bars_since_change < regime_buffer:
            strategy["skip_trading"] = True
            strategy["entry_conditions"]["regime_buffer"] = f"Need {regime_buffer} bars after regime change, currently at {bars_since_change}"
            return strategy
        
        # Skip trading if we recently made a trade on this symbol
        if symbol in self.last_trade_bar and bars_since_change - self.last_trade_bar[symbol] < 5:
            strategy["skip_trading"] = True
            strategy["entry_conditions"]["recent_trade"] = "Too soon after last trade"
            return strategy
        
        # 2. ENHANCEMENT: Additional Confirmation Filters
        # Add trend confirmation - only for trending regimes
        trend_strength = abs(data['EMA12'].iloc[-1] - data['EMA26'].iloc[-1]) / data['ATR'].iloc[-1]
        if trend_strength < 0.3 and (market_regime == "trending_up" or market_regime == "trending_down"):  # Lowered from 0.5
            strategy["skip_trading"] = True
            strategy["entry_conditions"]["weak_trend"] = f"Trend strength too low: {trend_strength:.2f}"
            return strategy
        
        # Add volatility filter - less restrictive
        if data['ATR'].iloc[-1] < data['ATR'].rolling(window=20).mean().iloc[-1] * 0.6:  # Lowered from 0.7
            strategy["skip_trading"] = True
            strategy["entry_conditions"]["low_volatility"] = "Volatility too low for pattern trading"
            return strategy
        
        # 3. ENHANCEMENT: Stricter Pattern Selection
        # Use moderate confidence threshold - balancing quality and quantity
        min_confidence = 0.70  # Adjusted from 0.80 (too strict) and 0.65 (too loose)
        patterns = self.pattern_recognizer.detect_patterns(data, min_confidence)
        
        # If no patterns detected, skip trading
        if not patterns:
            strategy["skip_trading"] = True
            strategy["entry_conditions"]["no_patterns"] = "No patterns detected with required confidence"
            return strategy
        
        # Get best pattern
        best_pattern = patterns[0]
        pattern_name = best_pattern["pattern_name"]
        pattern_confidence = best_pattern["confidence"]
        
        # 4. ENHANCEMENT: Add more pattern qualification criteria - balanced approach
        if pattern_name == "pin_bar":
            # Require moderate price action confirmation
            if not (current_bar['Close'] > current_bar['Open'] * 1.0005 or  # Reduced from 1.001
                    current_bar['Open'] > current_bar['Close'] * 1.0005):  # Reduced from 1.001
                strategy["skip_trading"] = True
                strategy["entry_conditions"]["weak_pin_bar"] = "Pin bar price action not strong enough"
                return strategy
        
        # 5. ENHANCEMENT: Regime-Specific Pattern Filtering - expanded pattern selection
        pattern_regime_map = {
            "trending_up": ["engulfing", "inside_bar", "morning_star", "pin_bar"],  # Added pin_bar
            "trending_down": ["engulfing", "inside_bar", "evening_star", "pin_bar"],  # Added pin_bar
            "ranging": ["doji", "inside_bar", "pin_bar"],  # Added pin_bar
            "breakout": ["engulfing", "pin_bar", "inside_bar"]  # Added inside_bar
        }
        
        if market_regime in pattern_regime_map and pattern_name not in pattern_regime_map[market_regime]:
            strategy["skip_trading"] = True
            strategy["entry_conditions"]["pattern_regime_mismatch"] = f"{pattern_name} not optimal for {market_regime} regime"
            return strategy
        
        # Get best patterns for current regime from historical performance
        best_regime_patterns = self.pattern_recognizer.get_best_patterns_for_regime(
            market_regime
        )
        
        # Check if pattern is good for current regime based on past performance
        pattern_regime_multiplier = 1.0
        if best_regime_patterns and pattern_name in best_regime_patterns:
            pattern_regime_multiplier = 1.2  # Boost confidence if pattern works well in this regime
        
        # Adjust confidence based on regime compatibility
        adjusted_confidence = pattern_confidence * pattern_regime_multiplier
        
        # Only generate signal if confidence is high enough
        if adjusted_confidence < 0.75:  # Increased from 0.7
            strategy["skip_trading"] = True
            strategy["entry_conditions"]["low_confidence"] = f"Pattern confidence too low: {adjusted_confidence:.2f}"
            return strategy
        
        # 6. ENHANCEMENT: Adaptive TP/SL Ratio and Trailing Stop
        # Set strategy parameters based on regime and pattern
        if market_regime == "trending_up" or market_regime == "trending_down":
            strategy["tp_sl_ratio"] = 2.5  # Adjusted from 3.0
            strategy["use_trailing_stop"] = True
            strategy["trailing_activation"] = 0.5  # Activate at 50% of target
            strategy["trailing_distance"] = 0.3  # Trail at 30% of ATR
        elif market_regime == "ranging":
            strategy["tp_sl_ratio"] = 1.2  # Adjusted from 1.5
            strategy["use_trailing_stop"] = False
        elif market_regime == "breakout":
            strategy["tp_sl_ratio"] = 2.0  # Adjusted from 2.5
            strategy["use_trailing_stop"] = True
            strategy["trailing_activation"] = 0.4
            strategy["trailing_distance"] = 0.4
        
        # Further adjust based on volatility
        if volatility_state == "high":
            strategy["tp_sl_ratio"] *= 0.8  # Reduce reward:risk in high volatility
        elif volatility_state == "low":
            strategy["tp_sl_ratio"] *= 1.2  # Increase reward:risk in low volatility
        
        # 7. ENHANCEMENT: Improved Capital Preservation - adjust for consecutive losses and drawdown
        # Find consecutive losses if recent trades are provided
        if recent_trades is not None:
            consecutive_losses = len([t for t in recent_trades 
                                    if t.get("profit_pips", 0) < 0 and t.get("symbol") == symbol])
            if consecutive_losses > 2:
                # Reduce position size after 3 consecutive losses
                strategy["position_size_modifier"] = 0.5
                strategy["entry_conditions"]["consecutive_losses"] = f"{consecutive_losses} consecutive losses"
        
        # Set the trade signal based on the pattern
        if best_pattern["direction"]:
            strategy["signal"] = best_pattern["direction"]
            strategy["direction"] = best_pattern["direction"]
            
            # Add pattern details to entry conditions
            strategy["entry_conditions"]["pattern"] = pattern_name
            strategy["entry_conditions"]["confidence"] = f"{adjusted_confidence:.2f}"
            strategy["entry_conditions"]["regime"] = market_regime
            
            # Add current ATR for advanced stop loss/take profit calculation
            strategy["current_atr"] = data['ATR'].iloc[-1] if 'ATR' in data.columns else None
        
        return strategy
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss_pips: float, account_balance: float) -> Dict[str, Any]:
        """
        Calculate position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pips: Stop loss in pips
            account_balance: Current account balance
            
        Returns:
            Dictionary with position sizing details
        """
        # Default pip value (for forex)
        pip_value = 0.0001
        
        # Adjust for JPY pairs
        if symbol.endswith('JPY'):
            pip_value = 0.01
        
        # Risk percentage based on account balance
        # Progressive risk scaling - more aggressive with lower balances
        if account_balance < 1000:
            risk_pct = 0.03  # 3% risk for very small accounts
        elif account_balance < 5000:
            risk_pct = 0.025  # 2.5% risk for small accounts
        elif account_balance < 10000:
            risk_pct = 0.02   # 2% risk for medium accounts
        elif account_balance < 25000:
            risk_pct = 0.015  # 1.5% risk for larger accounts
        else:
            risk_pct = 0.01   # 1% risk for accounts over PDT threshold
        
        # Adjust risk based on market regime
        market_regime = self.current_context.get("market_regime", "unknown")
        volatility = self.current_context.get("volatility_state", "medium")
        
        # Regime-specific risk adjustments
        if market_regime == "ranging":
            risk_pct *= 0.8  # Reduce risk in ranging markets
        elif market_regime == "breakout":
            risk_pct *= 1.1  # Slightly increase risk in breakouts
        
        # Volatility-specific risk adjustments
        if volatility == "high":
            risk_pct *= 0.8  # Reduce risk in high volatility
        elif volatility == "low":
            risk_pct *= 1.2  # Increase risk in low volatility
        
        # Calculate risk amount
        risk_amount = account_balance * risk_pct
        
        # Calculate pip value in account currency (simplified)
        pip_value_account = pip_value * 10000
        
        # Calculate position size (lots)
        position_size = risk_amount / (stop_loss_pips * pip_value_account)
        
        # Adjust for mini lots
        position_size = round(position_size, 2)
        
        return {
            "position_size": position_size, 
            "risk_amount": risk_amount,
            "risk_percentage": risk_pct * 100,
            "stop_loss_pips": stop_loss_pips
        }
    
    def update_trade_outcome(self, symbol: str, entry_time: datetime, 
                            exit_time: datetime, direction: str, 
                            pnl: float, pips: float, exit_reason: str,
                            pattern_name: str, market_regime: str) -> None:
        """
        Update trade outcome for learning.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            direction: Trade direction ('buy' or 'sell')
            pnl: Profit and loss amount
            pips: Profit and loss in pips
            exit_reason: Reason for exit
            pattern_name: Pattern used for entry
            market_regime: Market regime during trade
        """
        # Calculate if trade was successful
        success = pnl > 0
        
        # Update pattern performance
        self.pattern_recognizer.update_pattern_performance(
            pattern_name=pattern_name,
            success=success,
            profit_pips=pips,
            regime=market_regime
        )
        
        logger.info(f"Trade outcome: {symbol} {direction} {success} {pips} pips ({exit_reason})")
        
        # Remove from active trades if present
        if symbol in self.active_trades:
            del self.active_trades[symbol]
    
    def start_trade(self, symbol: str, entry_time: datetime, entry_price: float,
                   direction: str, pattern_name: str, market_regime: str) -> None:
        """
        Record a new trade start.
        
        Args:
            symbol: Trading symbol
            entry_time: Entry timestamp
            entry_price: Entry price
            direction: Trade direction
            pattern_name: Pattern used for entry
            market_regime: Market regime at entry
        """
        self.active_trades[symbol] = {
            "entry_time": entry_time,
            "entry_price": entry_price,
            "direction": direction,
            "pattern": pattern_name,
            "regime": market_regime
        }
        
        logger.info(f"New trade: {symbol} {direction} at {entry_price} based on {pattern_name} pattern in {market_regime} regime")

# Function to calculate technical indicators
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Calculate Moving Averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    
    # Calculate ATR
    high_low = data['High'] - data['Low']
    high_close = abs(data['High'] - data['Close'].shift())
    low_close = abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    data['BB_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + 2 * data['BB_std']
    data['BB_lower'] = data['BB_middle'] - 2 * data['BB_std']
    
    # Calculate MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    return data

# Detect market regime
def detect_market_regime(data: pd.DataFrame) -> str:
    """
    Detect the current market regime.
    
    Args:
        data: Market data with indicators
        
    Returns:
        Market regime as string
    """
    if len(data) < 20:
        return "unknown"
    
    # Get the most recent values
    current = data.iloc[-1]
    
    # Trend indicators
    trend_up = (
        current['SMA20'] > current['SMA50'] and
        current['Close'] > current['SMA20']
    )
    
    trend_down = (
        current['SMA20'] < current['SMA50'] and
        current['Close'] < current['SMA20']
    )
    
    # Range indicators 
    bb_width = (current['BB_upper'] - current['BB_lower']) / current['BB_middle']
    is_narrow_range = bb_width < 0.03
    
    # RSI in mid-range is a ranging sign
    is_rsi_mid = 40 < current['RSI'] < 60
    
    # Breakout indicators
    recent_range = data['High'].rolling(window=20).max() - data['Low'].rolling(window=20).min()
    avg_range = recent_range.mean()
    
    # Recent volatility expansion
    recent_atr = data['ATR'].iloc[-5:].mean()
    older_atr = data['ATR'].iloc[-20:-5].mean()
    volatility_expansion = recent_atr > older_atr * 1.5
    
    # Determine regime
    if trend_up:
        if volatility_expansion:
            return "breakout"
        else:
            return "trending_up"
    elif trend_down:
        if volatility_expansion:
            return "breakout"
        else:
            return "trending_down"
    elif is_narrow_range and is_rsi_mid:
        return "ranging"
    elif volatility_expansion:
        return "breakout"
    else:
        return "ranging"  # Default to ranging if no clear regime

# Detect volatility state
def detect_volatility(data: pd.DataFrame) -> str:
    """
    Detect the current volatility state.
    
    Args:
        data: Market data with indicators
        
    Returns:
        Volatility state as string
    """
    if len(data) < 20:
        return "medium"
    
    # Compare recent ATR with historical ATR
    recent_atr = data['ATR'].iloc[-5:].mean()
    historical_atr = data['ATR'].iloc[-100:].mean()
    
    # Calculate volatility ratio
    volatility_ratio = recent_atr / historical_atr if historical_atr > 0 else 1.0
    
    if volatility_ratio > 1.5:
        return "high"
    elif volatility_ratio < 0.7:
        return "low"
    else:
        return "medium"
