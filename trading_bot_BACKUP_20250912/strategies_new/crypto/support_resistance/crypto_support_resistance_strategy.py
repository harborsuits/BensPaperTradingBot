#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Support & Resistance Strategy Module

This module implements a comprehensive support and resistance trading strategy
for cryptocurrencies. It identifies key price levels using multiple techniques
and generates trading signals based on price interactions with these levels.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
import math

# Import base strategy
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="CryptoSupportResistanceStrategy",
    market_type="crypto",
    description="A strategy that identifies support and resistance levels using multiple techniques and trades based on price interactions with these levels",
    timeframes=["1h", "4h", "1d", "1w"],
    parameters={
        # Core parameters
        "sr_detection_methods": {
            "type": "list",
            "default": ["pivot_points", "fractals", "horizontal_levels", "moving_averages", "fibonacci", "volume_profile"],
            "description": "Methods to use for support/resistance detection"
        },
        "timeframes_to_analyze": {
            "type": "list",
            "default": ["1h", "4h", "1d"],
            "description": "Timeframes to use for multi-timeframe support/resistance analysis"
        },
        "lookback_periods": {
            "type": "int",
            "default": 100,
            "description": "Number of periods to look back for identifying support/resistance levels"
        },
        "min_touches": {
            "type": "int",
            "default": 2,
            "description": "Minimum number of touches required to confirm a support/resistance level"
        },
        "level_significance_threshold": {
            "type": "float",
            "default": 0.05,
            "description": "Threshold (as percentage) for combining nearby S/R levels"
        },
        "level_expiration_periods": {
            "type": "int",
            "default": 50,
            "description": "Number of periods after which a level expires if not touched"
        },
        "dynamic_levels": {
            "type": "bool",
            "default": True,
            "description": "Whether to use dynamic support/resistance levels (e.g., moving averages)"
        },
        
        # Pivot point parameters
        "pivot_type": {
            "type": "str",
            "default": "standard",
            "enum": ["standard", "fibonacci", "woodie", "camarilla", "demark"],
            "description": "Type of pivot point calculation to use"
        },
        "use_higher_timeframe_pivots": {
            "type": "bool",
            "default": True,
            "description": "Whether to include pivots from higher timeframes"
        },
        
        # Fractal parameters
        "fractal_periods": {
            "type": "int",
            "default": 5,
            "description": "Number of periods for Williams fractal detection"
        },
        "filter_weak_fractals": {
            "type": "bool",
            "default": True,
            "description": "Whether to filter out weak fractals"
        },
        
        # Fibonacci parameters
        "fibonacci_levels": {
            "type": "list", 
            "default": [0.236, 0.382, 0.5, 0.618, 0.786],
            "description": "Fibonacci retracement levels to use"
        },
        "use_fib_extensions": {
            "type": "bool",
            "default": True,
            "description": "Whether to include Fibonacci extensions" 
        },
        "fib_extension_levels": {
            "type": "list",
            "default": [1.272, 1.414, 1.618, 2.0, 2.618],
            "description": "Fibonacci extension levels to use"
        },
        
        # Volume profile parameters
        "vp_num_bins": {
            "type": "int",
            "default": 50,
            "description": "Number of bins for volume profile analysis" 
        },
        "vp_threshold_pct": {
            "type": "float",
            "default": 0.7,
            "description": "Threshold percentage for identifying high volume nodes"
        },
        
        # Moving average parameters
        "ma_types": {
            "type": "list",
            "default": ["sma", "ema"],
            "description": "Types of moving averages to use as dynamic S/R levels"
        },
        "ma_periods": {
            "type": "list",
            "default": [20, 50, 100, 200],
            "description": "Moving average periods to use as dynamic S/R levels"
        },
        
        # Trading parameters
        "trade_on_level_break": {
            "type": "bool",
            "default": True,
            "description": "Whether to trade on breakouts/breakdowns of levels"
        },
        "trade_on_level_bounce": {
            "type": "bool",
            "default": True,
            "description": "Whether to trade on bounces from levels"
        },
        "breakout_confirmation_periods": {
            "type": "int",
            "default": 3,
            "description": "Number of periods to confirm a breakout before trading"
        },
        "bounce_distance_pct": {
            "type": "float",
            "default": 0.01,
            "description": "Maximum distance (as percentage) from level to consider a bounce"
        },
        "min_risk_reward_ratio": {
            "type": "float",
            "default": 1.5,
            "description": "Minimum risk-to-reward ratio for taking a trade"
        },
        "stop_distance_atr_multiple": {
            "type": "float",
            "default": 1.0,
            "description": "ATR multiple for stop placement"
        },
        "profit_target_method": {
            "type": "str",
            "default": "next_level",
            "enum": ["next_level", "fixed_atr", "risk_multiple", "fibonacci"],
            "description": "Method for determining profit targets"
        },
        "profit_risk_multiple": {
            "type": "float",
            "default": 2.0,
            "description": "Risk multiple for profit targets (if using risk_multiple method)"
        },
        "profit_atr_multiple": {
            "type": "float",
            "default": 3.0, 
            "description": "ATR multiple for profit targets (if using fixed_atr method)"
        },
        "max_simultaneous_trades": {
            "type": "int",
            "default": 2,
            "description": "Maximum number of simultaneous trades"
        },
        "enable_partial_take_profit": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable partial profit taking"
        },
        "partial_tp_levels": {
            "type": "list",
            "default": [0.5, 0.75, 1.0],
            "description": "Levels for partial profit taking (as multiples of initial risk)"
        },
        "enable_trailing_stop": {
            "type": "bool",
            "default": True,
            "description": "Whether to enable trailing stops"
        },
        
        # Consolidation zone parameters
        "min_consolidation_bars": {
            "type": "int",
            "default": 10,
            "description": "Minimum number of bars required to identify a consolidation zone"
        },
        "max_consolidation_range_atr": {
            "type": "float",
            "default": 2.0,
            "description": "Maximum range (in ATR) for identifying a consolidation zone"
        },
        
        # Position sizing parameters
        "default_size": {
            "type": "float",
            "default": 0.02,
            "description": "Default position size as percentage of account"
        },
        "min_size": {
            "type": "float",
            "default": 0.01,
            "description": "Minimum position size as percentage of account"
        },
        "max_size": {
            "type": "float",
            "default": 0.05,
            "description": "Maximum position size as percentage of account"
        },
        "target_volatility": {
            "type": "float",
            "default": 2.0,
            "description": "Target volatility level for position sizing adjustments"
        },
        
        # Primary analysis timeframe
        "primary_timeframe": {
            "type": "str",
            "default": "1h",
            "description": "Primary timeframe for support/resistance analysis"
        },
        "trailing_stop_activation_pct": {
            "type": "float",
            "default": 1.0,
            "description": "Profit percentage required to activate trailing stop"
        },
        "trailing_stop_distance_atr": {
            "type": "float",
            "default": 2.0,
            "description": "ATR multiple for trailing stop distance"
        },
        "level_strength_weightings": {
            "type": "dict",
            "default": {
                "pivot_points": 1.0,
                "fractals": 0.7,
                "horizontal_levels": 1.0,
                "moving_averages": 0.8,
                "fibonacci": 0.9,
                "volume_profile": 1.0
            },
            "description": "Weightings for different types of S/R levels"
        },
        "confluence_bonus": {
            "type": "float",
            "default": 0.5,
            "description": "Bonus multiplier for confluence between multiple S/R levels"
        }
    },
    asset_classes=["crypto"],
    timeframes=["5m", "15m", "30m", "1h", "4h", "1d"]
)
class CryptoSupportResistanceStrategy(CryptoBaseStrategy):
    """
    A cryptocurrency support and resistance strategy that identifies key price levels
    using multiple techniques and generates trading signals based on price interactions
    with these levels.
    
    The strategy uses a combination of pivot points, fractals, historical price levels,
    moving averages, Fibonacci retracements, and volume profile to identify significant
    support and resistance zones.
    """
    
    def __init__(self, session: CryptoSession, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Support & Resistance strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize strategy-specific state variables
        self.support_levels = []  # List of {price, strength, type, created_date, touches, last_touch_date}
        self.resistance_levels = []  # List of {price, strength, type, created_date, touches, last_touch_date}
        self.dynamic_support_levels = []  # List of {price, name, period, value, last_updated}
        self.dynamic_resistance_levels = []  # List of {price, name, period, value, last_updated}
        self.level_touches = {}  # Dict of {level_id: [list of touch dates]}
        self.broken_levels = []  # List of recently broken levels
        self.price_swings = []  # List of significant price swings for Fibonacci analysis
        
        # Track recent highs and lows for fractal detection
        self.recent_highs = []
        self.recent_lows = []
        
        # Pivot point calculations
        self.current_pivots = {}  # Current pivot points for each timeframe
        
        # Volume profile data
        self.volume_profile = None
        self.high_volume_nodes = []
        
        # Track current ATR and market data for multiple timeframes
        self.current_atr = None
        self.multi_tf_data = {}  # Market data for each analyzed timeframe
        
        # Current trading signals and active positions
        self.current_signals = []  # List of active signals
        self.confirmed_breakouts = {}  # Dict of {level_id: confirmation_count}
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_pct = 0.0
        
        # Register additional event handlers
        self.register_event_handler("market_data_updated", self._on_market_data_updated)
        self.register_event_handler("timeframe_completed", self._on_timeframe_completed)
        
        logger.info(f"CryptoSupportResistanceStrategy initialized for {self.session.symbol} "
                   f"with methods: {', '.join(self.parameters['sr_detection_methods'])}")
        
        # Initial level detection (will be called again with each update)
        if not self.market_data.empty:
            self.detect_support_resistance_levels(self.market_data, self.parameters["primary_timeframe"])
    
    def detect_support_resistance_levels(self, data: pd.DataFrame, timeframe: str) -> None:
        """
        Main method to detect support and resistance levels using multiple techniques.
        
        Args:
            data: Market data for the symbol
            timeframe: Current timeframe being analyzed
        """
        if data.empty or len(data) < 20:
            logger.warning(f"Not enough data to detect support/resistance levels for {self.symbol}")
            return
        
        # Reset levels for this detection cycle
        self._reset_levels()
        
        # Get current price
        current_price = data["close"].iloc[-1]
        
        # Update ATR for use in detection methods
        self.current_atr = self._calculate_atr(data, 14).iloc[-1] if len(data) > 14 else 0
        
        # Execute all detection methods
        self._detect_pivot_points(data)
        self._detect_fractals(data)
        self._detect_horizontal_levels(data)
        self._add_psychological_levels(current_price)
        self._detect_moving_average_levels(data)
        self._detect_fibonacci_levels(data)
        self._detect_volume_profile_levels(data)
        self._detect_consolidation_zones(data)
        
        # Merge close levels and remove duplicates
        self._merge_close_levels(current_price)
        
        # Sort and rank levels by strength and proximity
        self._sort_and_rank_levels(current_price)
        
        # If this is the primary analysis timeframe, publish an event with the levels
        if timeframe == self.parameters["primary_timeframe"]:
            self._publish_levels_event()
            
    def _reset_levels(self) -> None:
        """
        Reset and initialize the support and resistance level collections.
        Called at the beginning of each detection cycle.
        """
        # Reset static levels (persist between updates, but reset for clarity)
        self.support_levels = []
        self.resistance_levels = []
        
        # Reset dynamic levels (recalculated with each update)
        self.dynamic_support_levels = []
        self.dynamic_resistance_levels = []
        
        # Reset zone collections
        self.consolidation_zones = []
        self.high_volume_nodes = []
    
    def _merge_close_levels(self, current_price: float) -> None:
        """
        Merge levels that are very close to each other to avoid excessive levels.
        Combines strength and keeps the most informative subtype.
        
        Args:
            current_price: Current market price
        """
        # Separate by support/resistance type
        for level_type in ["support", "resistance"]:
            levels = self.support_levels if level_type == "support" else self.resistance_levels
            
            if not levels:
                continue
                
            # Sort by price
            levels.sort(key=lambda x: x["price"])
            
            # Look for close levels and merge them
            merged_levels = []
            skip_indices = set()
            
            for i in range(len(levels)):
                if i in skip_indices:
                    continue
                    
                current_level = levels[i]
                
                # Find levels close to this one
                close_levels = []
                
                for j in range(i + 1, len(levels)):
                    if j in skip_indices:
                        continue
                        
                    next_level = levels[j]
                    price_diff = abs(next_level["price"] - current_level["price"])
                    
                    # If levels are close (within 0.3% of current price or 0.3 ATR)
                    if (price_diff / current_price < 0.003 or 
                        (self.current_atr > 0 and price_diff < 0.3 * self.current_atr)):
                        close_levels.append(j)
                        skip_indices.add(j)
                
                # If we found close levels, merge them
                if close_levels:
                    # Gather all levels to merge
                    all_levels = [current_level] + [levels[j] for j in close_levels]
                    
                    # Calculate combined values
                    total_strength = sum(lvl["strength"] for lvl in all_levels)
                    weighted_price = sum(lvl["price"] * lvl["strength"] for lvl in all_levels) / total_strength
                    
                    # Find the level with maximum strength to use its type/subtype
                    max_strength_idx = max(range(len(all_levels)), key=lambda k: all_levels[k]["strength"])
                    strongest_level = all_levels[max_strength_idx]
                    
                    # Create merged level
                    merged_level = {
                        "price": weighted_price,
                        "level_type": strongest_level["level_type"],
                        "subtype": strongest_level["subtype"],
                        "strength": min(3.0, total_strength),  # Cap strength
                        "confluence_count": len(all_levels),
                        "confluent_types": list(set(lvl["level_type"] for lvl in all_levels))
                    }
                    
                    merged_levels.append(merged_level)
                else:
                    # No close levels, just add the current one
                    merged_levels.append(current_level)
            
            # Replace the original levels with merged ones
            if level_type == "support":
                self.support_levels = merged_levels
            else:
                self.resistance_levels = merged_levels
    
    def _sort_and_rank_levels(self, current_price: float) -> None:
        """
        Sort levels by price and recompute levels' proximity to current price.
        
        Args:
            current_price: Current market price
        """
        # Process support levels (higher levels are more important)
        self.support_levels.sort(key=lambda x: x["price"], reverse=True)
        
        # Process resistance levels (lower levels are more important)
        self.resistance_levels.sort(key=lambda x: x["price"])
        
        # Calculate proximity factor and add rank
        for idx, level in enumerate(self.support_levels):
            level["proximity"] = 1.0 - (current_price - level["price"]) / current_price
            level["rank"] = idx + 1
        
        for idx, level in enumerate(self.resistance_levels):
            level["proximity"] = 1.0 - (level["price"] - current_price) / current_price
            level["rank"] = idx + 1
    
    def _publish_levels_event(self) -> None:
        """
        Publish support and resistance levels as an event for other strategies and visualizations.
        """
        event_data = {
            "symbol": self.symbol,
            "timestamp": datetime.now(),
            "support_levels": self.support_levels[:10],  # Top 10 levels
            "resistance_levels": self.resistance_levels[:10],  # Top 10 levels
            "consolidation_zones": self.consolidation_zones,
            "high_volume_nodes": self.high_volume_nodes,
            "source_strategy": self.__class__.__name__
        }
        
        self.event_bus.publish(
            "SUPPORT_RESISTANCE_LEVELS",
            event_data
        )
        
        logger.info(f"Published support/resistance levels for {self.symbol}: "
                  f"{len(self.support_levels)} support, {len(self.resistance_levels)} resistance")
                  
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators used by the strategy.
        
        Args:
            data: Market data DataFrame
        
        Returns:
            Dictionary of indicator values
        """
        if data.empty or len(data) < 20:
            return {}
        
        # Calculate basic indicators for confirmation and volatility assessment
        indicators = {}
        
        # Current price and volatility
        indicators["current_price"] = data["close"].iloc[-1]
        indicators["atr"] = self._calculate_atr(data, 14).iloc[-1] if len(data) > 14 else 0
        self.current_atr = indicators["atr"]
        
        # Moving averages for trend confirmation
        for period in self.parameters["ma_periods"]:
            if len(data) >= period:
                indicators[f"sma_{period}"] = data["close"].rolling(window=period).mean().iloc[-1]
                indicators[f"ema_{period}"] = data["close"].ewm(span=period, adjust=False).mean().iloc[-1]
        
        # RSI for general momentum indication
        if len(data) >= 14:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators["rsi"] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Detect current market structure
        price_action = data["close"].iloc[-20:]
        indicators["trend_direction"] = "up" if price_action.diff().mean() > 0 else "down"
        
        # Calculate price volatility
        if len(data) >= 20:
            indicators["volatility"] = data["close"].pct_change().rolling(20).std().iloc[-1] * 100
        
        # Detect basic chart patterns near current price
        indicators["near_resistance"] = False
        indicators["near_support"] = False
        indicators["in_consolidation"] = False
        
        # Detect support and resistance for primary timeframe
        timeframe = self.parameters["primary_timeframe"]
        self.detect_support_resistance_levels(data, timeframe)
        
        # Extract nearest levels for the indicators dictionary
        current_price = indicators["current_price"]
        proximity_threshold = 0.02  # 2% from current price
        
        # Find nearest support level
        if self.support_levels:
            nearest_support = self.support_levels[0]["price"]
            support_distance = (current_price - nearest_support) / current_price
            indicators["nearest_support"] = nearest_support
            indicators["support_distance"] = support_distance
            indicators["near_support"] = support_distance < proximity_threshold
        
        # Find nearest resistance level
        if self.resistance_levels:
            nearest_resistance = self.resistance_levels[0]["price"]
            resistance_distance = (nearest_resistance - current_price) / current_price
            indicators["nearest_resistance"] = nearest_resistance
            indicators["resistance_distance"] = resistance_distance
            indicators["near_resistance"] = resistance_distance < proximity_threshold
        
        # Check for consolidation (presence of multiple significant consolidation zones)
        indicators["in_consolidation"] = len(self.consolidation_zones) > 0
        
        # Calculate support/resistance ratio (higher means strong support, lower means strong resistance)
        total_support_strength = sum(level["strength"] for level in self.support_levels[:5]) if self.support_levels else 0
        total_resistance_strength = sum(level["strength"] for level in self.resistance_levels[:5]) if self.resistance_levels else 0
        
        if total_support_strength + total_resistance_strength > 0:
            indicators["sr_ratio"] = total_support_strength / (total_support_strength + total_resistance_strength)
        else:
            indicators["sr_ratio"] = 0.5  # Neutral if no levels
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on support and resistance levels.
        
        Args:
            data: Market data DataFrame
            indicators: Dictionary of indicator values
        
        Returns:
            Dictionary of trading signals
        """
        if data.empty or not indicators:
            return {"signal": "neutral", "confidence": 0, "triggers": []}
        
        # Initialize signal components
        signals = {
            "signal": "neutral",
            "confidence": 0,
            "direction": "neutral",
            "triggers": [],
            "stops": [],
            "targets": []
        }
        
        current_price = indicators["current_price"]
        
        # Check if we're near any significant support or resistance levels
        near_support = indicators.get("near_support", False)
        near_resistance = indicators.get("near_resistance", False)
        
        # Get trend direction for confirmation
        trend_direction = indicators.get("trend_direction", "neutral")
        
        # Get support/resistance ratio (higher means stronger support)
        sr_ratio = indicators.get("sr_ratio", 0.5)
        
        # Check for bounces from support/resistance
        bounce_from_support = False
        bounce_from_resistance = False
        
        # Look for price action evidence of bounces
        if len(data) >= 5:
            recent_lows = data["low"].iloc[-5:]
            recent_highs = data["high"].iloc[-5:]
            recent_close = data["close"].iloc[-1]
            previous_close = data["close"].iloc[-2]
            
            # Indicators of a bounce from support
            if near_support and recent_close > previous_close and recent_lows.min() > 0:
                # Price is near support and showing upward momentum
                nearest_support = indicators.get("nearest_support", 0)
                if nearest_support > 0:
                    support_penetration = min(recent_lows) / nearest_support
                    if 0.98 <= support_penetration <= 1.02:  # Touched support but didn't break it significantly
                        bounce_from_support = True
                        signals["triggers"].append("bounce_from_support")
            
            # Indicators of a bounce from resistance
            if near_resistance and recent_close < previous_close and recent_highs.max() > 0:
                # Price is near resistance and showing downward momentum
                nearest_resistance = indicators.get("nearest_resistance", float('inf'))
                if nearest_resistance < float('inf'):
                    resistance_penetration = max(recent_highs) / nearest_resistance
                    if 0.98 <= resistance_penetration <= 1.02:  # Touched resistance but didn't break it significantly
                        bounce_from_resistance = True
                        signals["triggers"].append("bounce_from_resistance")
        
        # Check for breakouts
        breakout_up = False
        breakout_down = False
        
        if len(data) >= 10:
            # Detect bullish breakout: price clearly above recent resistance with volume
            if near_resistance and current_price > indicators.get("nearest_resistance", float('inf')):
                # Check for strong momentum and volume
                recent_closes = data["close"].iloc[-5:]
                if all(recent_closes.diff().iloc[1:] > 0):  # Consecutive higher closes
                    # Confirm with volume if available
                    if "volume" in data.columns and not data["volume"].empty:
                        recent_volume = data["volume"].iloc[-5:]
                        avg_volume = data["volume"].iloc[-20:].mean()
                        if recent_volume.mean() > avg_volume * 1.2:  # 20% above average volume
                            breakout_up = True
                            signals["triggers"].append("breakout_up_with_volume")
                    else:
                        # No volume data, rely on price action only
                        breakout_up = True
                        signals["triggers"].append("breakout_up")
            
            # Detect bearish breakdown: price clearly below recent support with volume
            if near_support and current_price < indicators.get("nearest_support", 0):
                # Check for strong momentum and volume
                recent_closes = data["close"].iloc[-5:]
                if all(recent_closes.diff().iloc[1:] < 0):  # Consecutive lower closes
                    # Confirm with volume if available
                    if "volume" in data.columns and not data["volume"].empty:
                        recent_volume = data["volume"].iloc[-5:]
                        avg_volume = data["volume"].iloc[-20:].mean()
                        if recent_volume.mean() > avg_volume * 1.2:  # 20% above average volume
                            breakout_down = True
                            signals["triggers"].append("breakout_down_with_volume")
                    else:
                        # No volume data, rely on price action only
                        breakout_down = True
                        signals["triggers"].append("breakout_down")
        
        # Check for bull/bear market structure
        bullish_structure = False
        bearish_structure = False
        
        if "trend_direction" in indicators and "sr_ratio" in indicators:
            # Bullish structure: uptrend with strong support
            if indicators["trend_direction"] == "up" and indicators["sr_ratio"] > 0.6:
                bullish_structure = True
                signals["triggers"].append("bullish_structure")
            
            # Bearish structure: downtrend with strong resistance
            if indicators["trend_direction"] == "down" and indicators["sr_ratio"] < 0.4:
                bearish_structure = True
                signals["triggers"].append("bearish_structure")
        
        # Determine primary signal
        if bounce_from_support or breakout_up or bullish_structure:
            signals["direction"] = "long"
            
            # Prioritize signals by conviction
            if breakout_up:
                signals["signal"] = "strong_buy"
                signals["confidence"] = 0.8
            elif bounce_from_support and bullish_structure:
                signals["signal"] = "buy"
                signals["confidence"] = 0.7
            elif bounce_from_support:
                signals["signal"] = "weak_buy"
                signals["confidence"] = 0.6
            elif bullish_structure:
                signals["signal"] = "weak_buy"
                signals["confidence"] = 0.5
        
        elif bounce_from_resistance or breakout_down or bearish_structure:
            signals["direction"] = "short"
            
            # Prioritize signals by conviction
            if breakout_down:
                signals["signal"] = "strong_sell"
                signals["confidence"] = 0.8
            elif bounce_from_resistance and bearish_structure:
                signals["signal"] = "sell"
                signals["confidence"] = 0.7
            elif bounce_from_resistance:
                signals["signal"] = "weak_sell"
                signals["confidence"] = 0.6
            elif bearish_structure:
                signals["signal"] = "weak_sell"
                signals["confidence"] = 0.5
        
        # Set stop loss levels based on support/resistance
        if signals["direction"] == "long":
            if self.support_levels:
                # Use nearest strong support as stop level
                for level in self.support_levels:
                    if level["price"] < current_price and level["strength"] > 1.0:
                        stop_price = level["price"] * 0.99  # Just below support
                        signals["stops"] = [{"price": stop_price, "type": "fixed"}]
                        break
            
            # If no suitable support found, use a percentage stop
            if not signals["stops"] and "atr" in indicators and indicators["atr"] > 0:
                stop_price = current_price - (2 * indicators["atr"])
                signals["stops"] = [{"price": stop_price, "type": "atr_based"}]
            
            # Set take profit at a resistance level
            if self.resistance_levels:
                for level in self.resistance_levels:
                    if level["price"] > current_price:
                        take_profit = level["price"] * 0.99  # Just below resistance
                        signals["targets"] = [{"price": take_profit, "type": "resistance"}]
                        break
        
        elif signals["direction"] == "short":
            if self.resistance_levels:
                # Use nearest strong resistance as stop level
                for level in self.resistance_levels:
                    if level["price"] > current_price and level["strength"] > 1.0:
                        stop_price = level["price"] * 1.01  # Just above resistance
                        signals["stops"] = [{"price": stop_price, "type": "fixed"}]
                        break
            
            # If no suitable resistance found, use a percentage stop
            if not signals["stops"] and "atr" in indicators and indicators["atr"] > 0:
                stop_price = current_price + (2 * indicators["atr"])
                signals["stops"] = [{"price": stop_price, "type": "atr_based"}]
            
            # Set take profit at a support level
            if self.support_levels:
                for level in self.support_levels:
                    if level["price"] < current_price:
                        take_profit = level["price"] * 1.01  # Just above support
                        signals["targets"] = [{"price": take_profit, "type": "support"}]
                        break
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on ATR and level strength.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Market data DataFrame
            indicators: Dictionary of indicator values
        
        Returns:
            Position size as a decimal (0.0-1.0) representing account % to risk
        """
        # Default to base sizing if we don't have required data
        if data.empty or not indicators or direction == "neutral":
            return self.parameters["default_size"]
        
        # Base position size from parameters
        base_size = self.parameters["default_size"]
        min_size = self.parameters["min_size"]
        max_size = self.parameters["max_size"]
        
        # Adjust based on signal confidence
        confidence_modifier = 1.0
        
        # Get confidence from signals if available
        signals = self.generate_signals(data, indicators)
        if signals and "confidence" in signals:
            signal_confidence = signals["confidence"]
            
            # Scale position linearly with confidence
            confidence_modifier = signal_confidence / 0.5  # Normalize to 1.0 at 50% confidence
        
        # Adjust based on volatility
        volatility_modifier = 1.0
        if "volatility" in indicators:
            volatility = indicators["volatility"]
            
            # Inverse relationship with volatility - higher volatility, smaller position
            if volatility > 0:
                volatility_ratio = self.parameters["target_volatility"] / volatility
                volatility_modifier = min(1.5, max(0.5, volatility_ratio))
        
        # Adjust based on proximity to support/resistance
        proximity_modifier = 1.0
        
        if direction == "long" and "resistance_distance" in indicators:
            # Reduce size as we get closer to resistance
            resistance_distance = indicators["resistance_distance"]
            if resistance_distance < 0.1:  # Within 10% of resistance
                proximity_modifier = max(0.7, resistance_distance * 10)  # Scale down to 70% at minimum
        
        elif direction == "short" and "support_distance" in indicators:
            # Reduce size as we get closer to support
            support_distance = indicators["support_distance"]
            if support_distance < 0.1:  # Within 10% of support
                proximity_modifier = max(0.7, support_distance * 10)  # Scale down to 70% at minimum
        
        # Calculate final position size with all modifiers
        position_size = base_size * confidence_modifier * volatility_modifier * proximity_modifier
        
        # Ensure within min/max bounds
        position_size = max(min_size, min(max_size, position_size))
        
        return position_size
    
    def regime_compatibility(self, market_regime: Dict[str, Any]) -> float:
        """
        Calculate compatibility score for the current market regime.
        
        Args:
            market_regime: Dictionary of market regime information
        
        Returns:
            Compatibility score (0.0-1.0)
        """
        # Default compatibility if no regime info
        if not market_regime:
            return 0.6  # Moderate default compatibility
        
        regime_type = market_regime.get("regime_type", "unknown")
        volatility = market_regime.get("volatility", "normal")
        trend_strength = market_regime.get("trend_strength", 0.5)
        
        # Base compatibility
        compatibility = 0.5
        
        # Support & resistance strategy works well in range-bound markets
        if regime_type == "range_bound" or regime_type == "consolidation":
            compatibility += 0.3
        
        # Works moderately well in trending markets for breakouts
        elif regime_type == "trending":
            compatibility += 0.1
            
            # Better in strong trends for breakout trades
            if trend_strength > 0.7:
                compatibility += 0.1
        
        # Works moderately in volatile markets (support/resistance becomes more important)
        if volatility == "high":
            compatibility += 0.1
        elif volatility == "low":
            compatibility += 0.1  # Also good in low volatility for range plays
        
        # Add additional compatibility for specific conditions
        if "market_structure" in market_regime:
            market_structure = market_regime["market_structure"]
            
            # Very compatible with established support/resistance zones
            if "clear_sr_zones" in market_structure:
                compatibility += 0.2
            
            # Less compatible with chaotic price action
            if "choppy" in market_structure or "erratic" in market_structure:
                compatibility -= 0.2
        
        # Ensure result is between 0 and 1
        return max(0.1, min(1.0, compatibility))
        
        # Consolidate nearby levels and remove expired levels
        self._consolidate_levels()
        self._remove_expired_levels(data.index[-1])
        
        # Sort levels by price
        self.support_levels.sort(key=lambda x: x["price"])
        self.resistance_levels.sort(key=lambda x: x["price"])
        
        # Log significant levels
        self._log_significant_levels()
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range for the given data.
        
        Args:
            data: Market data DataFrame
            period: ATR period
            
        Returns:
            Current ATR value
        """
        if len(data) < period + 1:
            return None
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        
        return atr
    
    def _detect_pivot_points(self, data: pd.DataFrame) -> None:
        """
        Detect pivot points using the selected pivot point type.
        
        Args:
            data: Market data DataFrame
        """
        if data.empty or len(data) < 2:
            return
        
        pivot_type = self.parameters["pivot_type"]
        current_date = data.index[-1]
        
        # Get the latest completed period (typically the previous day/week)
        # For simplicity, we'll use the second-to-last bar in the data
        if len(data) < 2:
            return
            
        prev_high = data["high"].iloc[-2]
        prev_low = data["low"].iloc[-2]
        prev_close = data["close"].iloc[-2]
        prev_open = data["open"].iloc[-2]
        
        # Calculate pivots based on the selected type
        if pivot_type == "standard":
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = pivot + 2 * (prev_high - prev_low)
            s3 = pivot - 2 * (prev_high - prev_low)
            
        elif pivot_type == "fibonacci":
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = pivot + 0.382 * (prev_high - prev_low)
            s1 = pivot - 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            r3 = pivot + 1.0 * (prev_high - prev_low)
            s3 = pivot - 1.0 * (prev_high - prev_low)
            
        elif pivot_type == "woodie":
            pivot = (prev_high + prev_low + 2 * prev_close) / 4
            r1 = (2 * pivot) - prev_low
            s1 = (2 * pivot) - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            
        elif pivot_type == "camarilla":
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
            s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
            r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
            s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
            r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
            s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
            
        elif pivot_type == "demark":
            # DeMark pivots depend on the relationship between open and close
            if prev_close < prev_open:
                pivot = prev_high + (2 * prev_low) + prev_close
            elif prev_close > prev_open:
                pivot = (2 * prev_high) + prev_low + prev_close
            else:
                pivot = prev_high + prev_low + (2 * prev_close)
                
            pivot = pivot / 4
            r1 = pivot * 2 - prev_low
            s1 = pivot * 2 - prev_high
            # DeMark typically only uses one level of S/R
            r2 = r1 + (pivot - s1)
            s2 = s1 - (r1 - pivot)
            r3 = r2 + (pivot - s1)
            s3 = s2 - (r1 - pivot)
        
        # Store pivot points
        pivot_levels = {
            "pp": pivot,
            "r1": r1, "r2": r2, "r3": r3,
            "s1": s1, "s2": s2, "s3": s3,
        }
        
        self.current_pivots = pivot_levels
        
        # Add support levels
        for i, level_name in enumerate(["s1", "s2", "s3"]):
            if level_name in pivot_levels:
                level_price = pivot_levels[level_name]
                self._add_level(
                    price=level_price,
                    level_type="pivot_support",
                    subtype=f"{pivot_type}_{level_name}",
                    strength=1.0 - (i * 0.2),  # Higher levels (s1) are stronger than lower ones (s3)
                    is_support=True
                )
        
        # Add resistance levels
        for i, level_name in enumerate(["r1", "r2", "r3"]):
            if level_name in pivot_levels:
                level_price = pivot_levels[level_name]
                self._add_level(
                    price=level_price,
                    level_type="pivot_resistance",
                    subtype=f"{pivot_type}_{level_name}",
                    strength=1.0 - (i * 0.2),  # Lower levels (r1) are stronger than higher ones (r3)
                    is_support=False
                )
        
        # Add pivot point as both support and resistance (it often acts as both)
        self._add_level(
            price=pivot_levels["pp"],
            level_type="pivot_point",
            subtype=f"{pivot_type}_pp",
            strength=1.2,  # Pivot points are often significant
            is_support=True
        )
        
        self._add_level(
            price=pivot_levels["pp"],
            level_type="pivot_point",
            subtype=f"{pivot_type}_pp",
            strength=1.2,
            is_support=False
        )
        
        logger.debug(f"Calculated {pivot_type} pivot points: PP: {pivot:.2f}, "
                    f"R1: {r1:.2f}, R2: {r2:.2f}, R3: {r3:.2f}, "
                    f"S1: {s1:.2f}, S2: {s2:.2f}, S3: {s3:.2f}")
    
    def _add_level(self, price: float, level_type: str, subtype: str, strength: float, is_support: bool) -> None:
        """
        Add a support or resistance level to the appropriate collection.
        
        Args:
            price: Price level
            level_type: Type of level (e.g., 'pivot_point', 'fractal', 'horizontal')
            subtype: Subtype with additional details (e.g., 'standard_r1')
            strength: Level strength (higher is stronger)
            is_support: Whether this is a support (True) or resistance (False) level
        """
        # Apply strength weighting based on level type
        if level_type in self.parameters["level_strength_weightings"]:
            strength *= self.parameters["level_strength_weightings"][level_type]
        
        # Generate a unique ID for this level
        level_id = f"{level_type}_{subtype}_{price:.2f}"
        
        # Create level object
        level = {
            "id": level_id,
            "price": price,
            "type": level_type,
            "subtype": subtype,
            "strength": strength,
            "created_date": datetime.now(),
            "touches": 0,
            "last_touch_date": None,
        }
        
        # Add to appropriate collection
        current_price = self.market_data["close"].iloc[-1] if not self.market_data.empty else 0
        
        if is_support:
            # Only add as support if below current price
            if price < current_price or level_type.startswith("dynamic"):
                # Check if this level already exists (same price within threshold)
                for existing_level in self.support_levels:
                    if abs(existing_level["price"] - price) / price < 0.005:  # Within 0.5%
                        # Update strength if new level is stronger
                        if strength > existing_level["strength"]:
                            existing_level["strength"] = strength
                            existing_level["type"] = level_type  # Use the stronger level type
                            existing_level["subtype"] = subtype
                        return
                
                # If we get here, it's a new level
                self.support_levels.append(level)
        else:
            # Only add as resistance if above current price
            if price > current_price or level_type.startswith("dynamic"):
                # Check if this level already exists
                for existing_level in self.resistance_levels:
                    if abs(existing_level["price"] - price) / price < 0.005:  # Within 0.5%
                        # Update strength if new level is stronger
                        if strength > existing_level["strength"]:
                            existing_level["strength"] = strength
                            existing_level["type"] = level_type
                            existing_level["subtype"] = subtype
                        return
                
                # If we get here, it's a new level
                self.resistance_levels.append(level)
    
    def _detect_fractals(self, data: pd.DataFrame) -> None:
        """
        Detect Williams fractals in the price data and use them as support/resistance levels.
        
        Args:
            data: Market data DataFrame
        """
        if data.empty or len(data) < self.parameters["fractal_periods"] * 2 + 1:
            return
        
        fractal_periods = self.parameters["fractal_periods"]
        filter_weak = self.parameters["filter_weak_fractals"]
        
        # We need a window of 2*n+1 candles for an n-period fractal
        window_size = 2 * fractal_periods + 1
        
        # Look for fractals in recent data (last 100 periods)
        lookback = min(len(data), 100)
        analysis_data = data.iloc[-lookback:]
        
        for i in range(fractal_periods, len(analysis_data) - fractal_periods):
            # Check for a bullish fractal (support - low point surrounded by higher lows)
            is_bullish_fractal = True
            current_low = analysis_data["low"].iloc[i]
            
            # Check if center point is lower than surrounding points
            for j in range(1, fractal_periods + 1):
                if (analysis_data["low"].iloc[i-j] <= current_low or 
                    analysis_data["low"].iloc[i+j] <= current_low):
                    is_bullish_fractal = False
                    break
            
            if is_bullish_fractal:
                # Calculate fractal strength based on neighboring price action
                strength = 1.0
                
                # Stronger if it's a significant low relative to neighbors
                left_min = analysis_data["low"].iloc[i-fractal_periods:i].min()
                right_min = analysis_data["low"].iloc[i+1:i+fractal_periods+1].min()
                depth = min(left_min - current_low, right_min - current_low)
                
                # Skip very weak fractals if filtering is enabled
                if filter_weak and depth < self.current_atr * 0.3:
                    continue
                    
                # Adjust strength based on depth relative to ATR
                if self.current_atr:
                    strength += min(3.0, depth / self.current_atr)
                
                # Stronger if it aligns with a previous fractal
                for existing_low in self.recent_lows:
                    if abs(existing_low - current_low) / current_low < 0.01:  # Within 1%
                        strength += 0.5
                        break
                
                # Add to support levels
                self._add_level(
                    price=current_low,
                    level_type="fractal",
                    subtype="bullish",
                    strength=strength,
                    is_support=True
                )
                
                # Add to recent lows for future reference
                self.recent_lows.append(current_low)
                if len(self.recent_lows) > 10:  # Keep only the most recent ones
                    self.recent_lows.pop(0)
            
            # Check for a bearish fractal (resistance - high point surrounded by lower highs)
            is_bearish_fractal = True
            current_high = analysis_data["high"].iloc[i]
            
            # Check if center point is higher than surrounding points
            for j in range(1, fractal_periods + 1):
                if (analysis_data["high"].iloc[i-j] >= current_high or 
                    analysis_data["high"].iloc[i+j] >= current_high):
                    is_bearish_fractal = False
                    break
            
            if is_bearish_fractal:
                # Calculate fractal strength
                strength = 1.0
                
                # Stronger if it's a significant high relative to neighbors
                left_max = analysis_data["high"].iloc[i-fractal_periods:i].max()
                right_max = analysis_data["high"].iloc[i+1:i+fractal_periods+1].max()
                height = min(current_high - left_max, current_high - right_max)
                
                # Skip very weak fractals if filtering is enabled
                if filter_weak and height < self.current_atr * 0.3:
                    continue
                    
                # Adjust strength based on height relative to ATR
                if self.current_atr:
                    strength += min(3.0, height / self.current_atr)
                
                # Stronger if it aligns with a previous fractal
                for existing_high in self.recent_highs:
                    if abs(existing_high - current_high) / current_high < 0.01:  # Within 1%
                        strength += 0.5
                        break
                
                # Add to resistance levels
                self._add_level(
                    price=current_high,
                    level_type="fractal",
                    subtype="bearish",
                    strength=strength,
                    is_support=False
                )
                
                # Add to recent highs for future reference
                self.recent_highs.append(current_high)
                if len(self.recent_highs) > 10:  # Keep only the most recent ones
                    self.recent_highs.pop(0)
        
        logger.debug(f"Detected {len(self.recent_lows)} bullish and {len(self.recent_highs)} bearish fractals")
    
    def _detect_horizontal_levels(self, data: pd.DataFrame) -> None:
        """
        Detect significant horizontal support and resistance levels based on historical price action.
        
        This method identifies swing highs and lows and areas where price has reacted multiple times.
        
        Args:
            data: Market data DataFrame
        """
        if data.empty or len(data) < 20:  # Need sufficient history
            return
        
        lookback = min(len(data), self.parameters["lookback_periods"])
        analysis_data = data.iloc[-lookback:].copy()
        
        # Window size for detecting swings
        window = 5
        min_touches = self.parameters["min_touches"]
        price_threshold = self.parameters["level_significance_threshold"]
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(analysis_data) - window):
            # Check for swing high
            is_swing_high = True
            current_high = analysis_data["high"].iloc[i]
            
            for j in range(1, window + 1):
                if (analysis_data["high"].iloc[i-j] > current_high or 
                    analysis_data["high"].iloc[i+j] > current_high):
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    "price": current_high,
                    "index": i,
                    "date": analysis_data.index[i]
                })
            
            # Check for swing low
            is_swing_low = True
            current_low = analysis_data["low"].iloc[i]
            
            for j in range(1, window + 1):
                if (analysis_data["low"].iloc[i-j] < current_low or 
                    analysis_data["low"].iloc[i+j] < current_low):
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    "price": current_low,
                    "index": i,
                    "date": analysis_data.index[i]
                })
        
        # Group similar price levels and count touches
        current_price = analysis_data["close"].iloc[-1]
        
        # Helper function to cluster price levels
        def cluster_levels(levels, is_support):
            if not levels:
                return
                
            # Sort by price
            sorted_levels = sorted(levels, key=lambda x: x["price"])
            
            # Cluster similar price levels
            clusters = []
            current_cluster = [sorted_levels[0]]
            
            for i in range(1, len(sorted_levels)):
                prev_price = sorted_levels[i-1]["price"]
                curr_price = sorted_levels[i]["price"]
                
                # If prices are within threshold, add to current cluster
                if abs(curr_price - prev_price) / prev_price <= price_threshold:
                    current_cluster.append(sorted_levels[i])
                else:
                    # Start a new cluster
                    if len(current_cluster) >= min_touches:
                        clusters.append(current_cluster)
                    current_cluster = [sorted_levels[i]]
            
            # Add the last cluster if it meets the min touches requirement
            if len(current_cluster) >= min_touches:
                clusters.append(current_cluster)
            
            # Process each cluster to create a level
            for cluster in clusters:
                # Calculate average price of the cluster
                avg_price = sum(point["price"] for point in cluster) / len(cluster)
                
                # Calculate level strength based on number of touches and recency
                strength = min(3.0, len(cluster) / min_touches)  # Base strength from number of touches
                
                # Add recency factor - more recent touches increase strength
                recent_touches = sum(1 for point in cluster if 
                                   (analysis_data.index[-1] - point["date"]).days < 30)
                strength += recent_touches * 0.2
                
                # Create level subtype based on number of touches
                subtype = f"touch_{len(cluster)}"
                
                # Add the level
                if (is_support and avg_price < current_price) or \
                   (not is_support and avg_price > current_price) or \
                   abs(avg_price - current_price) / current_price < 0.01:  # Very close to current price
                    self._add_level(
                        price=avg_price,
                        level_type="horizontal",
                        subtype=subtype,
                        strength=strength,
                        is_support=is_support
                    )
        
        # Process swing highs and lows
        cluster_levels(swing_highs, is_support=False)
        cluster_levels(swing_lows, is_support=True)
        
        # Find areas of high price rejection from candlestick shadows
        shadow_analysis = analysis_data.copy()
        
        # Calculate upper and lower shadows
        shadow_analysis["upper_shadow"] = shadow_analysis["high"] - \
                                    shadow_analysis[["open", "close"]].max(axis=1)
        shadow_analysis["lower_shadow"] = shadow_analysis[["open", "close"]].min(axis=1) - \
                                     shadow_analysis["low"]
        
        # Identify significant shadows (longer than average)
        avg_upper_shadow = shadow_analysis["upper_shadow"].mean()
        avg_lower_shadow = shadow_analysis["lower_shadow"].mean()
        
        significant_upper_shadows = shadow_analysis[shadow_analysis["upper_shadow"] > 2 * avg_upper_shadow]
        significant_lower_shadows = shadow_analysis[shadow_analysis["lower_shadow"] > 2 * avg_lower_shadow]
        
        # Convert shadows to potential levels
        upper_shadow_points = [{
            "price": row["high"],
            "index": idx,
            "date": shadow_analysis.index[idx]
        } for idx, row in enumerate(significant_upper_shadows.itertuples())]
        
        lower_shadow_points = [{
            "price": row["low"],
            "index": idx,
            "date": shadow_analysis.index[idx]
        } for idx, row in enumerate(significant_lower_shadows.itertuples())]
        
        # Add shadow-based levels (these are often good rejection areas)
        cluster_levels(upper_shadow_points, is_support=False)
        cluster_levels(lower_shadow_points, is_support=True)
        
        # Also look for areas of price congestion (many closes in similar area)
        close_points = [{
            "price": price,
            "index": i,
            "date": analysis_data.index[i]
        } for i, price in enumerate(analysis_data["close"])]
        
        # These can be both support or resistance depending on current price
        cluster_levels(close_points, is_support=True)
        cluster_levels(close_points, is_support=False)
        
        # Round numbers often act as support/resistance in crypto
        self._add_psychological_levels(current_price)
    
    def _add_psychological_levels(self, current_price: float) -> None:
        """
        Add psychological price levels (round numbers) that often act as support/resistance.
        
        Args:
            current_price: Current market price
        """
        # Determine suitable price magnitude based on current price
        if current_price <= 0:
            return
            
        # Create different magnitudes of round numbers
        magnitudes = []
        
        # For very high-priced assets (like BTC)
        if current_price > 10000:
            magnitudes.extend([1000, 5000, 10000])
        
        # For mid-priced assets
        if 100 < current_price <= 10000:
            magnitudes.extend([100, 500, 1000])
        
        # For lower-priced assets
        if 1 < current_price <= 100:
            magnitudes.extend([1, 5, 10, 25, 50])
        
        # For very low-priced assets
        if 0.01 < current_price <= 1:
            magnitudes.extend([0.01, 0.05, 0.1, 0.25, 0.5])
        
        # For micro-priced assets
        if current_price <= 0.01:
            magnitudes.extend([0.001, 0.005, 0.01])
        
        # Find nearest psychological levels
        max_levels = 6  # Limit to avoid too many levels
        added_levels = 0
        
        for magnitude in magnitudes:
            # Find the nearest round numbers above and below current price
            lower_level = math.floor(current_price / magnitude) * magnitude
            upper_level = math.ceil(current_price / magnitude) * magnitude
            
            # Skip if it's exactly the current price
            if abs(lower_level - current_price) / current_price < 0.0001:
                lower_level -= magnitude
            
            if abs(upper_level - current_price) / current_price < 0.0001:
                upper_level += magnitude
            
            # Calculate additional levels on each side
            lower_levels = [lower_level - (i * magnitude) for i in range(0, 2)]
            upper_levels = [upper_level + (i * magnitude) for i in range(0, 2)]
            
            # Sort by distance from current price
            all_levels = [(price, abs(price - current_price)) for price in lower_levels + upper_levels]
            all_levels.sort(key=lambda x: x[1])
            
            # Add the closest levels first
            for price, _ in all_levels:
                # Skip if too close to current price
                if abs(price - current_price) / current_price < 0.005:
                    continue
                    
                # Add as support if below current price
                if price < current_price:
                    # Strength decays with distance from current price
                    distance_factor = max(0.5, 1.0 - (abs(price - current_price) / current_price) * 10)
                    # Higher magnitudes (rounder numbers) are stronger
                    magnitude_factor = min(1.5, max(0.7, math.log10(magnitude * 100) * 0.3))
                    
                    strength = 0.8 * distance_factor * magnitude_factor
                    
                    self._add_level(
                        price=price,
                        level_type="psychological",
                        subtype=f"round_{magnitude}",
                        strength=strength,
                        is_support=True
                    )
                    added_levels += 1
                
                # Add as resistance if above current price
                else:
                    # Strength calculation same as above
                    distance_factor = max(0.5, 1.0 - (abs(price - current_price) / current_price) * 10)
                    magnitude_factor = min(1.5, max(0.7, math.log10(magnitude * 100) * 0.3))
                    
                    strength = 0.8 * distance_factor * magnitude_factor
                    
                    self._add_level(
                        price=price,
                        level_type="psychological",
                        subtype=f"round_{magnitude}",
                        strength=strength,
                        is_support=False
                    )
                    added_levels += 1
                
                # Limit the number of psychological levels
                if added_levels >= max_levels:
                    break
            
            if added_levels >= max_levels:
                break
        
        logger.debug(f"Added {added_levels} psychological price levels around {current_price:.2f}")
    
    def _detect_moving_average_levels(self, data: pd.DataFrame) -> None:
        """
        Detect dynamic support and resistance levels based on moving averages.
        
        Args:
            data: Market data DataFrame
        """
        if data.empty or len(data) < 20:  # Need sufficient history
            return
        
        ma_types = self.parameters["ma_types"]
        ma_periods = self.parameters["ma_periods"]
        current_price = data["close"].iloc[-1]
        
        for ma_type in ma_types:
            for period in ma_periods:
                # Skip if we don't have enough data for this period
                if len(data) < period:
                    continue
                
                # Calculate the moving average
                if ma_type == "sma":
                    ma_value = data["close"].rolling(window=period).mean().iloc[-1]
                    ma_name = f"SMA{period}"
                elif ma_type == "ema":
                    ma_value = data["close"].ewm(span=period, adjust=False).mean().iloc[-1]
                    ma_name = f"EMA{period}"
                elif ma_type == "wma":
                    # Weighted moving average - newer values have higher weights
                    weights = np.arange(1, period + 1)
                    ma_value = np.average(
                        data["close"].iloc[-period:].values,
                        weights=weights
                    )
                    ma_name = f"WMA{period}"
                else:
                    continue
                
                # Skip if MA value is invalid
                if pd.isna(ma_value) or ma_value <= 0:
                    continue
                
                # Determine if this MA is support or resistance
                is_support = ma_value < current_price
                level_type = "dynamic_support" if is_support else "dynamic_resistance"
                
                # Calculate strength based on period length and MA type
                # Longer periods generally create stronger levels
                period_factor = min(1.5, max(0.5, period / 50))
                
                # EMA and WMA tend to be more responsive to recent price action
                type_factor = 1.0
                if ma_type == "ema":
                    type_factor = 1.1
                elif ma_type == "wma":
                    type_factor = 1.2
                
                # Proximity factor - MAs very close to current price are usually stronger
                proximity = abs(ma_value - current_price) / current_price
                proximity_factor = max(0.8, 1.2 - proximity * 10)  # Higher if closer to price
                
                strength = 0.7 * period_factor * type_factor * proximity_factor
                
                # Add to dynamic levels collection
                dynamic_level = {
                    "price": ma_value,
                    "name": ma_name,
                    "period": period,
                    "type": ma_type,
                    "strength": strength,
                    "last_updated": datetime.now()
                }
                
                if is_support:
                    self.dynamic_support_levels.append(dynamic_level)
                else:
                    self.dynamic_resistance_levels.append(dynamic_level)
                
                # Also add to regular levels collection for signal generation
                self._add_level(
                    price=ma_value,
                    level_type=level_type,
                    subtype=ma_name,
                    strength=strength,
                    is_support=is_support
                )
        
        # Sort dynamic levels by price
        self.dynamic_support_levels.sort(key=lambda x: x["price"], reverse=True)  # Higher support levels first
        self.dynamic_resistance_levels.sort(key=lambda x: x["price"])  # Lower resistance levels first
        
        logger.debug(f"Added {len(self.dynamic_support_levels)} dynamic support and "
                    f"{len(self.dynamic_resistance_levels)} dynamic resistance levels")
    
    def _detect_fibonacci_levels(self, data: pd.DataFrame) -> None:
        """
        Detect support and resistance levels based on Fibonacci retracements and extensions.
        
        Args:
            data: Market data DataFrame
        """
        if data.empty or len(data) < 20:  # Need sufficient history
            return
        
        # Find significant swing highs and lows for Fibonacci analysis
        # We'll use the most recent significant swings
        lookback = min(len(data), self.parameters["lookback_periods"])
        analysis_data = data.iloc[-lookback:].copy()
        
        # Parameters
        fib_levels = self.parameters["fibonacci_levels"]
        use_extensions = self.parameters["use_fib_extensions"]
        extension_levels = self.parameters["fib_extension_levels"] if use_extensions else []
        
        # Find significant swings using a zigzag approach
        current_price = analysis_data["close"].iloc[-1]
        threshold = self.current_atr * 3  # Significant move threshold (3 ATR)
        
        # Identify the most recent swing points (high and low)
        # For uptrend: find the most recent significant low, then the high before it
        # For downtrend: find the most recent significant high, then the low before it
        
        # Find the current trend based on last 20 periods
        short_trend = analysis_data["close"].iloc[-20:].diff().mean() > 0
        
        swing_high = None
        swing_low = None
        swing_high_idx = None
        swing_low_idx = None
        
        # Start from the most recent data and go backward
        for i in range(len(analysis_data) - 2, 5, -1):
            # Skip already identified points
            if swing_high is not None and swing_low is not None:
                break
                
            current_high = analysis_data["high"].iloc[i]
            current_low = analysis_data["low"].iloc[i]
            
            # Looking for a swing high
            if swing_high is None:
                # Check if it's a local maximum (higher than neighbors)
                if (analysis_data["high"].iloc[i] > analysis_data["high"].iloc[i+1] and 
                    analysis_data["high"].iloc[i] > analysis_data["high"].iloc[i-1] and
                    analysis_data["high"].iloc[i] > analysis_data["high"].iloc[i+2] and 
                    analysis_data["high"].iloc[i] > analysis_data["high"].iloc[i-2]):
                    
                    # Check if the move is significant
                    subsequent_low = analysis_data["low"].iloc[i+1:].min()
                    if current_high - subsequent_low > threshold:
                        swing_high = current_high
                        swing_high_idx = i
            
            # Looking for a swing low
            if swing_low is None:
                # Check if it's a local minimum (lower than neighbors)
                if (analysis_data["low"].iloc[i] < analysis_data["low"].iloc[i+1] and 
                    analysis_data["low"].iloc[i] < analysis_data["low"].iloc[i-1] and
                    analysis_data["low"].iloc[i] < analysis_data["low"].iloc[i+2] and 
                    analysis_data["low"].iloc[i] < analysis_data["low"].iloc[i-2]):
                    
                    # Check if the move is significant
                    subsequent_high = analysis_data["high"].iloc[i+1:].max()
                    if subsequent_high - current_low > threshold:
                        swing_low = current_low
                        swing_low_idx = i
        
        # If both swing points are not found, try a more aggressive approach
        if swing_high is None or swing_low is None:
            # Just use the highest high and lowest low from the lookback period
            swing_high = analysis_data["high"].max()
            swing_high_idx = analysis_data["high"].idxmax()
            swing_low = analysis_data["low"].min()
            swing_low_idx = analysis_data["low"].idxmin()
        
        # Calculate Fibonacci levels based on the swings
        # Direction depends on which swing is more recent
        if swing_high is not None and swing_low is not None:
            direction = "up" if swing_high_idx > swing_low_idx else "down"
            
            if direction == "up":
                # Uptrend: retracements from low to high
                range_size = swing_high - swing_low
                
                # Retracement levels (potential support in an uptrend)
                for level in fib_levels:
                    # Calculate the retracement price level
                    price_level = swing_high - (range_size * level)
                    
                    # Skip if the level is above the current price (not support anymore)
                    if price_level >= current_price:
                        continue
                    
                    # Add as support level
                    strength = 1.0
                    
                    # The 0.618 level is traditionally considered stronger
                    if abs(level - 0.618) < 0.01:
                        strength = 1.3
                    
                    # The 0.5 level is also widely watched
                    elif abs(level - 0.5) < 0.01:
                        strength = 1.2
                    
                    self._add_level(
                        price=price_level,
                        level_type="fibonacci",
                        subtype=f"retracement_{level}",
                        strength=strength,
                        is_support=True
                    )
                
                # Extension levels (potential resistance in an uptrend)
                if use_extensions:
                    for level in extension_levels:
                        # Calculate the extension price level
                        price_level = swing_high + (range_size * (level - 1))
                        
                        # Skip if the level is below the current price
                        if price_level <= current_price:
                            continue
                        
                        # Add as resistance level
                        strength = 0.9  # Generally weaker than retracements
                        
                        # The 1.618 extension is widely watched
                        if abs(level - 1.618) < 0.01:
                            strength = 1.1
                        
                        self._add_level(
                            price=price_level,
                            level_type="fibonacci",
                            subtype=f"extension_{level}",
                            strength=strength,
                            is_support=False
                        )
            
            else:  # direction == "down"
                # Downtrend: retracements from high to low
                range_size = swing_high - swing_low
                
                # Retracement levels (potential resistance in a downtrend)
                for level in fib_levels:
                    # Calculate the retracement price level
                    price_level = swing_low + (range_size * level)
                    
                    # Skip if the level is below the current price (not resistance anymore)
                    if price_level <= current_price:
                        continue
                    
                    # Add as resistance level
                    strength = 1.0
                    
                    # The 0.618 level is traditionally considered stronger
                    if abs(level - 0.618) < 0.01:
                        strength = 1.3
                    
                    # The 0.5 level is also widely watched
                    elif abs(level - 0.5) < 0.01:
                        strength = 1.2
                    
                    self._add_level(
                        price=price_level,
                        level_type="fibonacci",
                        subtype=f"retracement_{level}",
                        strength=strength,
                        is_support=False
                    )
                
                # Extension levels (potential support in a downtrend)
                if use_extensions:
                    for level in extension_levels:
                        # Calculate the extension price level
                        price_level = swing_low - (range_size * (level - 1))
                        
                        # Skip if the level is above the current price
                        if price_level >= current_price:
                            continue
                        
                        # Add as support level
                        strength = 0.9  # Generally weaker than retracements
                        
                        # The 1.618 extension is widely watched
                        if abs(level - 1.618) < 0.01:
                            strength = 1.1
                        
                        self._add_level(
                            price=price_level,
                            level_type="fibonacci",
                            subtype=f"extension_{level}",
                            strength=strength,
                            is_support=True
                        )
                        
        logger.debug(f"Calculated Fibonacci levels based on swing high {swing_high:.2f} and swing low {swing_low:.2f}")
    
    def _detect_volume_profile_levels(self, data: pd.DataFrame) -> None:
        """
        Detect support and resistance levels based on volume profile analysis.
        
        Volume profile identifies price ranges where the most trading activity has occurred,
        which often act as support and resistance.
        
        Args:
            data: Market data DataFrame
        """
        if data.empty or len(data) < 20:  # Need sufficient history
            return
        
        lookback = min(len(data), self.parameters["lookback_periods"])
        analysis_data = data.iloc[-lookback:].copy()
        
        # Parameters
        num_bins = self.parameters["vp_num_bins"]
        volume_threshold = self.parameters["vp_threshold_pct"]
        current_price = analysis_data["close"].iloc[-1]
        
        # Define price range for the volume profile
        price_min = analysis_data["low"].min()
        price_max = analysis_data["high"].max()
        price_range = price_max - price_min
        
        # Create price bins
        bin_size = price_range / num_bins
        bins = [price_min + (i * bin_size) for i in range(num_bins + 1)]
        
        # Initialize the volume profile
        volume_profile = np.zeros(num_bins)
        
        # Calculate volume profile
        for i in range(len(analysis_data)):
            # For each candle, distribute its volume across the price range it covers
            candle_high = analysis_data["high"].iloc[i]
            candle_low = analysis_data["low"].iloc[i]
            candle_volume = analysis_data["volume"].iloc[i]
            
            if pd.isna(candle_volume) or candle_volume <= 0:
                continue
            
            # Find which bins this candle spans
            low_bin = max(0, int((candle_low - price_min) / bin_size))
            high_bin = min(num_bins - 1, int((candle_high - price_min) / bin_size))
            
            # Distribute volume across the bins proportionally
            # For simplicity, distribute equally
            bins_covered = high_bin - low_bin + 1
            if bins_covered > 0:
                volume_per_bin = candle_volume / bins_covered
                for bin_idx in range(low_bin, high_bin + 1):
                    volume_profile[bin_idx] += volume_per_bin
        
        # Calculate Volume Points of Control (VPOC) and high volume nodes (HVN)
        max_volume = volume_profile.max()
        significant_volume_threshold = max_volume * volume_threshold
        
        # Store high volume nodes for reference
        self.high_volume_nodes = []
        
        # Process volume profile to identify support/resistance levels
        for i in range(num_bins):
            if volume_profile[i] >= significant_volume_threshold:
                # This is a high volume node
                bin_price_low = price_min + (i * bin_size)
                bin_price_high = bin_price_low + bin_size
                bin_price_mid = (bin_price_low + bin_price_high) / 2
                
                # Calculate node strength based on volume relative to max
                volume_ratio = volume_profile[i] / max_volume
                node_strength = 0.8 + (volume_ratio * 0.7)  # 0.8 to 1.5 range
                
                # Is this node the VPOC (highest volume)?               
                is_vpoc = (volume_profile[i] == max_volume)
                
                # Add extra strength if this is the VPOC
                if is_vpoc:
                    node_strength += 0.3
                    subtype = "vpoc"
                else:
                    subtype = "hvn"
                
                # Store the high volume node
                self.high_volume_nodes.append({
                    "price_low": bin_price_low,
                    "price_high": bin_price_high,
                    "price_mid": bin_price_mid,
                    "volume": volume_profile[i],
                    "is_vpoc": is_vpoc,
                    "strength": node_strength
                })
                
                # Add as support if below current price
                if bin_price_mid < current_price:
                    self._add_level(
                        price=bin_price_mid,
                        level_type="volume_profile",
                        subtype=subtype,
                        strength=node_strength,
                        is_support=True
                    )
                
                # Add as resistance if above current price
                elif bin_price_mid > current_price:
                    self._add_level(
                        price=bin_price_mid,
                        level_type="volume_profile",
                        subtype=subtype,
                        strength=node_strength,
                        is_support=False
                    )
                
                # If very close to current price, add as both
                elif abs(bin_price_mid - current_price) / current_price < 0.01:
                    # For prices very close to current, add as both support and resistance
                    # as they can act in either direction
                    self._add_level(
                        price=bin_price_mid,
                        level_type="volume_profile",
                        subtype=subtype,
                        strength=node_strength,
                        is_support=True
                    )
                    
                    self._add_level(
                        price=bin_price_mid,
                        level_type="volume_profile",
                        subtype=subtype,
                        strength=node_strength,
                        is_support=False
                    )
        
        # Also detect low volume nodes (LVN) as potential breakout areas
        # These are often areas of little interest that price moves through quickly
        # For simplicity, we'll skip this part but it could be added later
        
        logger.debug(f"Detected {len(self.high_volume_nodes)} high volume nodes from volume profile")
    
    def _detect_consolidation_zones(self, data: pd.DataFrame) -> None:
        """
        Detect consolidation (congestion) zones where price has traded sideways in a range.
        These areas often act as support/resistance when price returns to them.
        
        Args:
            data: Market data DataFrame
        """
        if data.empty or len(data) < 30:  # Need sufficient history
            return
            
        # Parameters
        min_zone_length = self.parameters["min_consolidation_bars"]
        max_zone_range_atr = self.parameters["max_consolidation_range_atr"]
        current_price = data["close"].iloc[-1]
        
        # Create a clean copy of recent data for analysis
        lookback = min(len(data), self.parameters["lookback_periods"] * 2)  # Use more data for this analysis
        analysis_data = data.iloc[-lookback:].copy()
        
        # Initialize consolidation zones list
        self.consolidation_zones = []
        
        # Calculate the average true range for normalizing price ranges
        # We've already calculated this in initialize(), but let's ensure we're using recent data
        atr_period = 14
        if len(analysis_data) > atr_period:
            current_atr = self._calculate_atr(analysis_data, atr_period).iloc[-1]
        else:
            current_atr = self.current_atr
        
        # Window-based detection (looking for periods of low volatility and tight ranges)
        # Note: We could do more sophisticated clustering, but this simple approach works well
        
        # Normalize price movements by ATR
        analysis_data["norm_range"] = (analysis_data["high"] - analysis_data["low"]) / current_atr
        analysis_data["norm_move"] = abs(analysis_data["close"] - analysis_data["close"].shift(1)) / current_atr
        
        # Find continuous periods of low volatility
        zone_start = None
        zone_bars = 0
        zone_highs = []
        zone_lows = []
        
        for i in range(1, len(analysis_data)):
            # If this candle has a small range (< 0.8 ATR) and small price move from previous
            if (analysis_data["norm_range"].iloc[i] < 0.8 and 
                analysis_data["norm_move"].iloc[i] < 0.5):
                
                # Start or continue a zone
                if zone_start is None:
                    zone_start = i
                
                zone_bars += 1
                zone_highs.append(analysis_data["high"].iloc[i])
                zone_lows.append(analysis_data["low"].iloc[i])
                
                # If we hit a big enough window, check zone quality
                if zone_bars >= min_zone_length:
                    zone_high = max(zone_highs)
                    zone_low = min(zone_lows)
                    zone_range = zone_high - zone_low
                    
                    # Normalize zone range by ATR
                    normalized_range = zone_range / current_atr
                    
                    # If range is tight enough, we found a consolidation zone
                    if normalized_range <= max_zone_range_atr:
                        zone_mid = (zone_high + zone_low) / 2
                        
                        # Strength increases with more bars and tighter range
                        bar_factor = min(1.5, max(0.8, zone_bars / 20))
                        range_factor = 1.6 - normalized_range / max_zone_range_atr
                        
                        strength = 1.0 * bar_factor * range_factor
                        
                        # Store the zone
                        zone_idx = {
                            "start": zone_start,
                            "end": i,
                            "high": zone_high,
                            "low": zone_low,
                            "mid": zone_mid,
                            "bars": zone_bars,
                            "strength": strength
                        }
                        
                        # Add to our list for reference
                        self.consolidation_zones.append(zone_idx)
                        
                        # Add as support/resistance based on current price
                        if zone_mid < current_price:
                            # Add both upper and lower boundaries, with upper being stronger (last resistance)
                            self._add_level(
                                price=zone_high,
                                level_type="consolidation",
                                subtype=f"upper_boundary_{zone_bars}bars",
                                strength=strength * 1.1,  # Upper boundary is a bit stronger
                                is_support=True
                            )
                            
                            self._add_level(
                                price=zone_low,
                                level_type="consolidation",
                                subtype=f"lower_boundary_{zone_bars}bars",
                                strength=strength * 0.9,  # Lower boundary is a bit weaker
                                is_support=True
                            )
                        
                        elif zone_mid > current_price:
                            # Add both boundaries, with lower being stronger (last support)
                            self._add_level(
                                price=zone_high,
                                level_type="consolidation",
                                subtype=f"upper_boundary_{zone_bars}bars",
                                strength=strength * 0.9,  # Upper boundary is a bit weaker
                                is_support=False
                            )
                            
                            self._add_level(
                                price=zone_low,
                                level_type="consolidation",
                                subtype=f"lower_boundary_{zone_bars}bars",
                                strength=strength * 1.1,  # Lower boundary is a bit stronger
                                is_support=False
                            )
                            
                        # Also add the zone mid point with lower strength
                        self._add_level(
                            price=zone_mid,
                            level_type="consolidation",
                            subtype=f"zone_mid_{zone_bars}bars",
                            strength=strength * 0.7,
                            is_support=(zone_mid < current_price)
                        )
            else:
                # Reset the zone if volatility increases
                zone_start = None
                zone_bars = 0
                zone_highs = []
                zone_lows = []
        
        logger.debug(f"Detected {len(self.consolidation_zones)} consolidation zones")
