#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covered Call Strategy Module

This module implements a covered call options strategy which involves holding a long position
in an asset and selling call options on that same asset to generate additional income.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

# Import base strategy
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsBaseStrategy, OptionsSession
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="CoveredCallStrategy",
    market_type="options",
    description="A strategy that combines a long stock position with short call options to generate income while allowing for limited upside potential",
    timeframes=["1d", "1w"],
    parameters={
        # Stock selection parameters
        "stock_selection_criteria": {
            "type": "list",
            "default": ["fundamentals", "technical", "volatility"],
            "description": "Criteria for selecting underlying stocks"
        },
        "min_stock_price": {
            "type": "float",
            "default": 20.0,
            "description": "Minimum stock price for covered call candidates"
        },
        "min_avg_daily_volume": {
            "type": "int",
            "default": 500000,
            "description": "Minimum average daily trading volume"
        },
        "min_market_cap": {
            "type": "float",
            "default": 1000000000.0,  # $1B
            "description": "Minimum market capitalization"
        },
        
        # Option selection parameters
        "days_to_expiration_min": {
            "type": "int",
            "default": 21,
            "description": "Minimum days to expiration"
        },
        "days_to_expiration_max": {
            "type": "int",
            "default": 45,
            "description": "Maximum days to expiration"
        },
        "target_delta": {
            "type": "float",
            "default": 0.30,
            "description": "Target delta for selling calls"
        },
        "delta_range": {
            "type": "float",
            "default": 0.10,
            "description": "Acceptable range around target delta"
        },
        "min_premium_percentage": {
            "type": "float",
            "default": 0.01,  # 1% of stock price
            "description": "Minimum premium as percentage of stock price"
        },
        
        # Exit and adjustment parameters
        "call_profit_target_pct": {
            "type": "float",
            "default": 0.50,  # 50% of max premium
            "description": "Profit target as percentage of collected premium"
        },
        "close_calls_dte": {
            "type": "int",
            "default": 5,
            "description": "Days to expiration at which to consider closing calls"
        },
        "roll_when_itm": {
            "type": "bool",
            "default": True,
            "description": "Whether to roll calls when they go in-the-money"
        },
        "roll_dte_threshold": {
            "type": "int",
            "default": 10,
            "description": "Roll calls this many days before expiration if ITM"
        },
        
        # Stock exit parameters
        "stock_stop_loss_pct": {
            "type": "float",
            "default": 0.12,  # 12%
            "description": "Stock position stop loss percentage"
        },
        "use_trailing_stop": {
            "type": "bool",
            "default": True,
            "description": "Use trailing stop for stock position"
        },
        "trailing_stop_pct": {
            "type": "float",
            "default": 0.10,  # 10%
            "description": "Trailing stop percentage"
        },
        
        # Risk management
        "max_position_size_pct": {
            "type": "float",
            "default": 0.05,  # 5% of portfolio
            "description": "Maximum position size as percentage of portfolio"
        },
        "max_sector_exposure_pct": {
            "type": "float",
            "default": 0.20,  # 20% of portfolio
            "description": "Maximum sector exposure as percentage of portfolio"
        }
    }
)
class CoveredCallStrategy(OptionsBaseStrategy, AccountAwareMixin):
    """
    A covered call strategy that combines long stock positions with short call options 
    to generate income while allowing for limited upside potential.
    
    This strategy:
    1. Selects stocks based on fundamental and technical criteria
    2. Buys shares of the selected stock
    3. Sells call options against the owned shares to collect premium
    4. Manages the position with adjustments based on market movements
    5. Aims to generate income through premium collection and potential stock appreciation
    """
    
    def __init__(self, session: OptionsSession, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Covered Call strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        # Initialize account awareness functionality
        AccountAwareMixin.__init__(self)
        
        # Initialize strategy-specific state variables
        self.stock_positions = {}  # Track stock positions
        self.call_positions = {}  # Track short call positions
        self.paired_positions = {}  # Track stock and option pairs
        self.highest_stock_prices = {}  # For trailing stops
        
        logger.info(f"Covered Call Strategy initialized with target delta {self.parameters['target_delta']}")
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for stock selection and option evaluation.
        
        Args:
            data: Market data DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty:
            return {}
        
        indicators = {}
        
        # Calculate moving averages
        indicators["sma_50"] = data["close"].rolling(window=50).mean()
        indicators["sma_200"] = data["close"].rolling(window=200).mean()
        
        # Calculate ATR for volatility assessment
        high_low = data["high"] - data["low"]
        high_close = (data["high"] - data["close"].shift()).abs()
        low_close = (data["low"] - data["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators["atr_14"] = tr.rolling(window=14).mean()
        
        # Calculate implied volatility rank (would use option data in full implementation)
        # Placeholder for demonstration
        indicators["iv_rank"] = 0.50
        
        # Calculate stock trend
        if len(data) >= 50:
            indicators["trend"] = "uptrend" if indicators["sma_50"].iloc[-1] > indicators["sma_200"].iloc[-1] else "downtrend"
        else:
            indicators["trend"] = "unknown"
            
        # Volume analysis
        indicators["volume_sma_50"] = data["volume"].rolling(window=50).mean()
        indicators["relative_volume"] = data["volume"] / indicators["volume_sma_50"]
        
        return indicators
    
    def evaluate_stock_candidate(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Evaluate a stock as a covered call candidate.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Score representing stock suitability (0.0-1.0)
        """
        if data.empty or not indicators:
            return 0.0
        
        score_components = []
        
        # Trend score - prefer stocks in gentle uptrends
        if "trend" in indicators:
            trend_score = 0.8 if indicators["trend"] == "uptrend" else 0.4
            score_components.append(trend_score)
        
        # Volatility score - prefer moderate volatility for premium
        if "atr_14" in indicators and len(data) > 0:
            atr_pct = indicators["atr_14"].iloc[-1] / data["close"].iloc[-1]
            # Ideal ATR% is around 1.5-3.5%
            if 0.015 <= atr_pct <= 0.035:
                vol_score = 0.9
            elif 0.01 <= atr_pct < 0.015 or 0.035 < atr_pct <= 0.05:
                vol_score = 0.7
            elif atr_pct > 0.05:
                vol_score = 0.3  # Too volatile
            else:
                vol_score = 0.4  # Too stable
            score_components.append(vol_score)
        
        # Price level score - avoid very low priced stocks
        if len(data) > 0:
            price = data["close"].iloc[-1]
            if price >= self.parameters["min_stock_price"]:
                price_score = min(1.0, price / 100)  # Higher prices get better scores up to $100
            else:
                price_score = 0.0  # Below minimum price
            score_components.append(price_score)
        
        # Volume score - prefer liquid stocks
        if "relative_volume" in indicators:
            volume_score = min(1.0, indicators["relative_volume"].iloc[-1])
            score_components.append(volume_score)
        
        # IV Rank score - prefer stocks with relatively high IV
        if "iv_rank" in indicators:
            iv_score = min(1.0, indicators["iv_rank"] * 1.5)  # Higher IV gets better scores
            score_components.append(iv_score)
        
        # Calculate final score
        return sum(score_components) / len(score_components) if score_components else 0.0
    
    def select_option_contract(self, data: pd.DataFrame, option_chain: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the appropriate call option to sell.
        
        Args:
            data: Stock data DataFrame
            option_chain: Option chain data
            
        Returns:
            Selected option contract or None
        """
        # This is a simplified implementation
        # A real implementation would analyze the full options chain
        
        if not option_chain or "calls" not in option_chain:
            return None
            
        stock_price = data["close"].iloc[-1] if not data.empty else 0
        if stock_price <= 0:
            return None
            
        # Filter by days to expiration
        min_dte = self.parameters["days_to_expiration_min"]
        max_dte = self.parameters["days_to_expiration_max"]
        
        valid_calls = []
        for call in option_chain["calls"]:
            if "expiration_date" not in call or "delta" not in call or "bid" not in call:
                continue
                
            # Calculate days to expiration
            exp_date = datetime.strptime(call["expiration_date"], "%Y-%m-%d")
            today = datetime.now()
            dte = (exp_date - today).days
            
            # Check if within our DTE range
            if min_dte <= dte <= max_dte:
                # Check if delta is within our target range
                target_delta = self.parameters["target_delta"]
                delta_range = self.parameters["delta_range"]
                
                if abs(call["delta"] - target_delta) <= delta_range:
                    # Check if premium meets minimum requirement
                    premium = call["bid"]
                    min_premium = stock_price * self.parameters["min_premium_percentage"]
                    
                    if premium >= min_premium:
                        valid_calls.append(call)
        
        if not valid_calls:
            return None
            
        # Find the call closest to our target delta
        target_delta = self.parameters["target_delta"]
        best_call = min(valid_calls, key=lambda x: abs(x["delta"] - target_delta))
        
        return best_call

    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the covered call strategy.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        if data.empty or not indicators:
            return {}
        
        signals = {
            "stock_entry": False,
            "stock_exit": False,
            "sell_call": False,
            "buy_call": False,
            "roll_call": False,
            "signal_strength": 0.0,
            "entry_type": None,
            "exit_type": None
        }
        
        symbol = self.session.symbol
        current_price = data["close"].iloc[-1]
        
        # 1. Check if we already have a stock position
        stock_position_exists = symbol in self.stock_positions
        
        # 2. Check if we already have a short call position
        call_position_exists = symbol in self.call_positions
        
        # Entry logic for stock
        if not stock_position_exists:
            # Evaluate stock as a covered call candidate
            stock_score = self.evaluate_stock_candidate(data, indicators)
            
            if stock_score >= 0.7:  # Stock meets our criteria
                signals["stock_entry"] = True
                signals["entry_type"] = "covered_call_stock_entry"
                signals["signal_strength"] = stock_score
                logger.info(f"Stock entry signal for {symbol} with score {stock_score:.2f}")
        
        # Entry logic for short call (if we have stock but no call)
        elif stock_position_exists and not call_position_exists:
            signals["sell_call"] = True
            signals["entry_type"] = "covered_call_initial_sell"
            signals["signal_strength"] = 0.8
            logger.info(f"Sell call signal for {symbol} against existing stock position")
        
        # Exit logic for stock position (stop loss or trailing stop)
        if stock_position_exists:
            # Get entry price
            entry_price = self.stock_positions.get(symbol, {}).get("entry_price", current_price)
            
            # Update highest price seen (for trailing stop)
            if symbol not in self.highest_stock_prices:
                self.highest_stock_prices[symbol] = current_price
            else:
                self.highest_stock_prices[symbol] = max(self.highest_stock_prices[symbol], current_price)
            
            # Check stop loss
            stop_loss_triggered = current_price <= entry_price * (1 - self.parameters["stock_stop_loss_pct"])
            
            # Check trailing stop
            trailing_stop_triggered = False
            if self.parameters["use_trailing_stop"]:
                highest_price = self.highest_stock_prices[symbol]
                trail_stop_level = highest_price * (1 - self.parameters["trailing_stop_pct"])
                trailing_stop_triggered = current_price <= trail_stop_level
            
            if stop_loss_triggered or trailing_stop_triggered:
                signals["stock_exit"] = True
                signals["exit_type"] = "stop_loss" if stop_loss_triggered else "trailing_stop"
                signals["signal_strength"] = 0.9
                
                # Also close the call if we have one
                if call_position_exists:
                    signals["buy_call"] = True
                
                logger.info(f"Stock exit signal for {symbol} due to {signals['exit_type']}")
        
        # Exit/roll logic for call position
        if call_position_exists:
            call_data = self.call_positions.get(symbol, {})
            
            # Check days to expiration
            if "expiration_date" in call_data:
                exp_date = datetime.strptime(call_data["expiration_date"], "%Y-%m-%d")
                today = datetime.now()
                dte = (exp_date - today).days
                
                # Close calls near expiration
                if dte <= self.parameters["close_calls_dte"]:
                    signals["buy_call"] = True
                    signals["exit_type"] = "call_near_expiration"
                    signals["signal_strength"] = 0.8
                    logger.info(f"Buy call signal for {symbol} due to approaching expiration ({dte} DTE)")
                
                # Roll calls that are in the money
                elif (self.parameters["roll_when_itm"] and 
                      dte <= self.parameters["roll_dte_threshold"] and
                      "strike" in call_data and call_data["strike"] < current_price):
                    signals["roll_call"] = True
                    signals["exit_type"] = "roll_itm_call"
                    signals["signal_strength"] = 0.75
                    logger.info(f"Roll call signal for {symbol} due to ITM status with {dte} DTE")
            
            # Check profit target
            if "entry_price" in call_data and "current_price" in call_data:
                entry_price = call_data["entry_price"]
                current_price = call_data["current_price"]
                profit_pct = (entry_price - current_price) / entry_price  # For short calls
                
                if profit_pct >= self.parameters["call_profit_target_pct"]:
                    signals["buy_call"] = True
                    signals["exit_type"] = "call_profit_target"
                    signals["signal_strength"] = 0.85
                    logger.info(f"Buy call signal for {symbol} due to profit target reached ({profit_pct:.1%})")
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size for the stock component of the covered call.
        
        Args:
            direction: Trade direction ('long' for stock, 'short' for call options)
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size as a decimal (0.0-1.0) representing portfolio percentage
        """
        # For covered calls, position sizing is primarily about the stock position
        # The call position size will be determined by the stock position (1 call per 100 shares)
        
        if direction != "long" or data.empty:
            
        # Calculate position size based on strategy logic
        original_size = 0.0
        
        # Apply account-aware constraints
        is_day_trade = hasattr(self, 'is_day_trade') and self.is_day_trade
        max_size, _ = self.calculate_max_position_size(price, is_day_trade=is_day_trade)
        
        return min(original_size, max_size)  # Use lower of the two
        
        # Start with maximum position size
        position_size = self.parameters["max_position_size_pct"]
        
        # Adjust based on volatility
        if "atr_14" in indicators and len(data) > 0:
            atr_pct = indicators["atr_14"].iloc[-1] / data["close"].iloc[-1]
            
            # Reduce position size for higher volatility
            if atr_pct > 0.03:
                volatility_factor = 0.03 / atr_pct
                position_size *= max(0.5, min(1.0, volatility_factor))
        
        # Adjust based on trend strength
        if "trend" in indicators:
            if indicators["trend"] == "downtrend":
                position_size *= 0.7  # Reduce position in downtrends
        
        # Adjust based on sector exposure
        symbol = self.session.symbol
        sector = self._get_symbol_sector(symbol)
        if sector:
            current_sector_exposure = self._get_sector_exposure(sector)
            remaining_sector_capacity = max(0, self.parameters["max_sector_exposure_pct"] - current_sector_exposure)
            position_size = min(position_size, remaining_sector_capacity)
        
        logger.info(f"Calculated stock position size for {symbol}: {position_size:.2%}")
        return position_size
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get the sector for a symbol (would use external data in production)."""
        return "technology"  # Placeholder
    
    def _get_sector_exposure(self, sector: str) -> float:
        """Get current exposure to a sector (would track actual positions in production)."""
        return 0.05  # Placeholder
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the covered call strategy is with the current market regime.
        
        Covered calls work best in flat to slowly rising markets with moderate volatility.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "flat": 0.95,                  # Excellent in flat markets
            "slow_bull": 0.90,             # Excellent in slow bull markets
            "moderate_bull": 0.75,         # Good in moderate bull markets
            "strong_bull": 0.40,           # Poor in strong bull markets (caps upside)
            "slow_bear": 0.60,             # Above average in slow bear markets
            "moderate_bear": 0.30,         # Poor in moderate bear markets
            "strong_bear": 0.10,           # Very poor in strong bear markets
            "high_volatility": 0.30,       # Poor in high volatility environments
            "moderate_volatility": 0.85,   # Very good in moderate volatility
            "low_volatility": 0.70,        # Good in low volatility (but lower premiums)
            "sector_rotation": 0.60,       # Above average during sector rotations
            "earnings_season": 0.50,       # Average during earnings seasons
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.60)
