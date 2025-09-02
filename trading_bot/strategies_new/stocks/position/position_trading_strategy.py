#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Trading Strategy Module

This module implements a position trading strategy for stocks. Position trading is a longer-term
strategy that aims to capture significant price moves over weeks, months, or even years.
The strategy focuses on identifying long-term trends and fundamental shifts in company 
or sector performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta

# Import base strategy and registry
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

@register_strategy(
    name="PositionTradingStrategy",
    market_type="stocks",
    description="A long-term trading strategy that aims to capture significant price moves over weeks, months, or years by focusing on fundamental trends and major technical shifts",
    timeframes=["1d", "1w", "1M"],
    parameters={
        # Core parameters
        "trend_identification": {
            "type": "str",
            "default": "moving_average",
            "enum": ["moving_average", "trend_line", "regression", "fundamentals"],
            "description": "Method used to identify long-term trends"
        },
        "long_ma_period": {
            "type": "int",
            "default": 200,
            "description": "Long moving average period for trend determination"
        },
        "medium_ma_period": {
            "type": "int",
            "default": 50,
            "description": "Medium moving average period for trend determination"
        },
        "min_holding_period": {
            "type": "int", 
            "default": 20,
            "description": "Minimum holding period in days"
        },
        "max_holding_period": {
            "type": "int",
            "default": 252,
            "description": "Maximum holding period in days (252 = ~1 year of trading days)"
        },
        "use_fundamentals": {
            "type": "bool",
            "default": True,
            "description": "Whether to incorporate fundamental data in analysis"
        },
        "fundamental_metrics": {
            "type": "list",
            "default": ["pe_ratio", "eps_growth", "revenue_growth", "debt_to_equity", "return_on_equity"],
            "description": "Fundamental metrics to consider if use_fundamentals is True"
        },
        
        # Entry parameters
        "entry_confirmation_indicators": {
            "type": "list",
            "default": ["macd", "volume", "rsi"],
            "description": "Indicators used to confirm entry signals"
        },
        "entry_filters": {
            "type": "list",
            "default": ["volatility", "liquidity", "sector_performance"],
            "description": "Filters applied to potential entry setups"
        },
        "required_volume_multiple": {
            "type": "float",
            "default": 1.5,
            "description": "Required volume as a multiple of the average volume"
        },
        
        # Exit parameters
        "trailing_stop_type": {
            "type": "str",
            "default": "percentage",
            "enum": ["percentage", "atr", "moving_average", "parabolic_sar"],
            "description": "Type of trailing stop to use for long-term positions"
        },
        "trailing_stop_percentage": {
            "type": "float",
            "default": 0.15,
            "description": "Trailing stop percentage if using percentage method"
        },
        "trailing_stop_atr_multiple": {
            "type": "float",
            "default": 3.0,
            "description": "ATR multiple for trailing stop if using ATR method"
        },
        "take_profit_targets": {
            "type": "list",
            "default": [1.5, 2.0, 3.0],
            "description": "Take profit targets as R multiples (risk multiples)"
        },
        "partial_exit_points": {
            "type": "list",
            "default": [0.33, 0.33, 0.34],
            "description": "Percentage of position to exit at each take profit target"
        },
        
        # Risk management
        "max_risk_per_trade": {
            "type": "float",
            "default": 0.01,
            "description": "Maximum risk per trade as a fraction of account"
        },
        "max_correlated_positions": {
            "type": "int",
            "default": 3,
            "description": "Maximum number of correlated positions (same sector/industry)"
        },
        "sector_exposure_limit": {
            "type": "float",
            "default": 0.25,
            "description": "Maximum exposure to a single sector"
        }
    }
)
class PositionTradingStrategy(StocksBaseStrategy):
    """
    A position trading strategy that focuses on capturing major price moves over 
    longer time horizons (weeks to years) based on fundamental and technical factors.
    
    This strategy:
    1. Identifies long-term trends using moving averages and/or fundamental analysis
    2. Enters positions with the major trend and holds through minor countertrend moves
    3. Uses wider stops and longer holding periods than swing or day trading
    4. Incorporates fundamental metrics to validate technical signals
    5. Manages risk through position sizing, correlation control, and trailing exits
    """
    
    def __init__(self, session: StocksSession, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Position Trading strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize strategy-specific state variables
        self.positions_data = {}  # Track positions and their metadata
        self.sector_exposure = {}  # Track exposure by sector
        self.industry_exposure = {}  # Track exposure by industry
        self.entry_timestamps = {}  # Track entry times for holding period enforcement
        self.highest_prices = {}  # Track highest prices seen for trailing stops
        self.partial_exits_taken = {}  # Track which partial exits have been taken
        
        # Initialize fundamental data cache
        self.fundamental_data = {}
        
        # Precompute some derived parameters
        self.min_holding_period_delta = timedelta(days=self.parameters["min_holding_period"])
        
        logger.info(f"Position Trading Strategy initialized with {self.parameters['trend_identification']} "
                   f"trend identification and {self.parameters['min_holding_period']} day minimum holding period")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the position trading strategy.
        
        Args:
            data: Market data DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty or len(data) < self.parameters["long_ma_period"]:
            return {}
        
        indicators = {}
        
        # Calculate moving averages for trend determination
        indicators["long_ma"] = data["close"].rolling(window=self.parameters["long_ma_period"]).mean()
        indicators["medium_ma"] = data["close"].rolling(window=self.parameters["medium_ma_period"]).mean()
        
        # Calculate MACD if used in entry confirmation
        if "macd" in self.parameters["entry_confirmation_indicators"]:
            ema_fast = data["close"].ewm(span=12, adjust=False).mean()
            ema_slow = data["close"].ewm(span=26, adjust=False).mean()
            indicators["macd"] = ema_fast - ema_slow
            indicators["macd_signal"] = indicators["macd"].ewm(span=9, adjust=False).mean()
            indicators["macd_histogram"] = indicators["macd"] - indicators["macd_signal"]
        
        # Calculate RSI if used in entry confirmation
        if "rsi" in self.parameters["entry_confirmation_indicators"]:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators["rsi"] = 100 - (100 / (1 + rs))
        
        # Calculate ATR if used for stops
        if self.parameters["trailing_stop_type"] == "atr":
            high_low = data["high"] - data["low"]
            high_close = (data["high"] - data["close"].shift()).abs()
            low_close = (data["low"] - data["close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators["atr"] = true_range.rolling(window=14).mean()
        
        # Calculate Parabolic SAR if used for stops
        if self.parameters["trailing_stop_type"] == "parabolic_sar":
            # Simplified implementation - would use a library in production
            indicators["parabolic_sar"] = data["low"].rolling(window=10).min()  # Placeholder
        
        # Trend strength
        if len(data) >= 50:
            # Directional Movement Index for trend strength
            plus_dm = data["high"].diff()
            minus_dm = data["low"].diff().multiply(-1)
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            tr = high_low.combine(high_close, max).combine(low_close, max)
            atr_14 = tr.rolling(window=14).mean()
            
            plus_di = 100 * plus_dm.rolling(window=14).mean() / atr_14
            minus_di = 100 * minus_dm.rolling(window=14).mean() / atr_14
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            indicators["adx"] = dx.rolling(window=14).mean()  # ADX for trend strength
        
        # Volume analysis
        indicators["volume_sma"] = data["volume"].rolling(window=20).mean()
        indicators["volume_ratio"] = data["volume"] / indicators["volume_sma"]
        
        # Linear regression slope for trend direction and strength
        if len(data) >= 50:
            y = data["close"].values[-50:]
            x = np.arange(len(y))
            slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
            indicators["regression_slope"] = slope[0]
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the position trading strategy.
        
        Args:
            data: Market data DataFrame with OHLCV data
            indicators: Dictionary of pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        if data.empty or not indicators or len(data) < self.parameters["long_ma_period"]:
            return {}
        
        signals = {
            "long_entry": False,
            "long_exit": False,
            "short_entry": False,
            "short_exit": False,
            "signal_strength": 0.0,
            "entry_type": None,
            "exit_type": None
        }
        
        symbol = self.session.symbol
        current_price = data["close"].iloc[-1]
        current_date = data.index[-1]
        
        # Check existing position
        position_exists = self._position_exists(symbol)
        
        # Only look for entries if we don't have a position
        if not position_exists:
            # Long-term trend based on moving averages
            trend_bullish = indicators["medium_ma"].iloc[-1] > indicators["long_ma"].iloc[-1]
            
            # Entry conditions
            entry_confirmed = False
            
            # Strategy-specific entry logic based on trend identification method
            if self.parameters["trend_identification"] == "moving_average":
                # Check for price above both MAs and medium MA above long MA (golden cross)
                ma_setup = (current_price > indicators["medium_ma"].iloc[-1] > indicators["long_ma"].iloc[-1])
                
                # Check for recent golden cross
                golden_cross = False
                if len(indicators["medium_ma"]) > 5 and len(indicators["long_ma"]) > 5:
                    was_below = indicators["medium_ma"].iloc[-5] < indicators["long_ma"].iloc[-5]
                    is_above = indicators["medium_ma"].iloc[-1] > indicators["long_ma"].iloc[-1]
                    golden_cross = was_below and is_above
                
                # Either established uptrend or recent cross
                entry_setup = ma_setup or golden_cross
                
            elif self.parameters["trend_identification"] == "trend_line":
                # For trend line we would need more complex logic
                # Simplified version using slope of regression line
                if "regression_slope" in indicators:
                    entry_setup = indicators["regression_slope"] > 0.1  # Positive slope, would need calibration
                else:
                    entry_setup = False
                    
            elif self.parameters["trend_identification"] == "regression":
                # Linear regression based trend
                if "regression_slope" in indicators:
                    entry_setup = indicators["regression_slope"] > 0.1  # Positive slope, would need calibration
                else:
                    entry_setup = False
                
            elif self.parameters["trend_identification"] == "fundamentals":
                # This would require fetching fundamental data
                # Simplified version assuming fundamental data is positive
                entry_setup = True  # Placeholder for actual fundamental analysis
            
            # Confirmation indicators
            confirmations = []
            
            if "macd" in self.parameters["entry_confirmation_indicators"]:
                if "macd" in indicators and "macd_signal" in indicators:
                    macd_bullish = indicators["macd"].iloc[-1] > indicators["macd_signal"].iloc[-1]
                    macd_rising = indicators["macd"].iloc[-1] > indicators["macd"].iloc[-2] if len(indicators["macd"]) > 1 else False
                    confirmations.append(macd_bullish and macd_rising)
            
            if "rsi" in self.parameters["entry_confirmation_indicators"]:
                if "rsi" in indicators:
                    rsi_bullish = indicators["rsi"].iloc[-1] > 50 and indicators["rsi"].iloc[-1] < 70
                    confirmations.append(rsi_bullish)
            
            if "volume" in self.parameters["entry_confirmation_indicators"]:
                if "volume_ratio" in indicators:
                    volume_confirmed = indicators["volume_ratio"].iloc[-1] > self.parameters["required_volume_multiple"]
                    confirmations.append(volume_confirmed)
            
            # Need at least 2/3 of confirmations to be true
            entry_confirmed = len(confirmations) > 0 and sum(confirmations) / len(confirmations) >= 0.66
            
            # Combine setup and confirmation for entry signal
            if entry_setup and entry_confirmed:
                signals["long_entry"] = True
                signals["entry_type"] = "position_trend_following"
                signals["signal_strength"] = 0.8 if trend_bullish else 0.6  # Stronger if long-term trend aligns
                
                logger.info(f"Position entry signal for {symbol} at {current_price}")
        
        # Exit logic for existing positions
        elif position_exists:
            position = self._get_position(symbol)
            
            if symbol in self.entry_timestamps:
                entry_date = self.entry_timestamps[symbol]
                min_hold_reached = current_date - entry_date >= self.min_holding_period_delta
            else:
                min_hold_reached = True  # If we don't know the entry date, assume min hold is reached
            
            # Initialize highest price if not already tracking
            if symbol not in self.highest_prices:
                self.highest_prices[symbol] = current_price
            else:
                # Update highest price seen
                self.highest_prices[symbol] = max(self.highest_prices[symbol], current_price)
            
            # Trailing stop logic
            exit_triggered = False
            exit_reason = None
            
            if self.parameters["trailing_stop_type"] == "percentage":
                stop_level = self.highest_prices[symbol] * (1 - self.parameters["trailing_stop_percentage"])
                if current_price < stop_level and min_hold_reached:
                    exit_triggered = True
                    exit_reason = "trailing_stop_percentage"
            
            elif self.parameters["trailing_stop_type"] == "atr" and "atr" in indicators:
                stop_distance = indicators["atr"].iloc[-1] * self.parameters["trailing_stop_atr_multiple"]
                stop_level = self.highest_prices[symbol] - stop_distance
                if current_price < stop_level and min_hold_reached:
                    exit_triggered = True
                    exit_reason = "trailing_stop_atr"
            
            elif self.parameters["trailing_stop_type"] == "moving_average":
                # Use medium MA as trailing stop
                if current_price < indicators["medium_ma"].iloc[-1] and min_hold_reached:
                    exit_triggered = True
                    exit_reason = "trailing_stop_ma"
            
            elif self.parameters["trailing_stop_type"] == "parabolic_sar" and "parabolic_sar" in indicators:
                if current_price < indicators["parabolic_sar"].iloc[-1] and min_hold_reached:
                    exit_triggered = True
                    exit_reason = "trailing_stop_parabolic"
            
            # Check for trend reversal
            trend_reversed = False
            if "adx" in indicators:
                adx_weakening = indicators["adx"].iloc[-1] < indicators["adx"].iloc[-2] if len(indicators["adx"]) > 1 else False
                ma_bearish = indicators["medium_ma"].iloc[-1] < indicators["long_ma"].iloc[-1]
                if adx_weakening and ma_bearish and min_hold_reached:
                    trend_reversed = True
            
            # Check if max holding period exceeded
            max_hold_exceeded = False
            if symbol in self.entry_timestamps:
                max_hold_period = timedelta(days=self.parameters["max_holding_period"])
                max_hold_exceeded = current_date - self.entry_timestamps[symbol] >= max_hold_period
            
            # Generate exit signal if any exit condition is met
            if exit_triggered or trend_reversed or max_hold_exceeded:
                signals["long_exit"] = True
                signals["exit_type"] = exit_reason if exit_triggered else "trend_reversal" if trend_reversed else "max_hold_period"
                signals["signal_strength"] = 0.9 if exit_triggered else 0.7 if trend_reversed else 0.8
                
                logger.info(f"Position exit signal for {symbol} at {current_price} due to {signals['exit_type']}")
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Market data DataFrame with OHLCV data
            indicators: Dictionary of pre-calculated indicators
            
        Returns:
            Position size as a decimal representing account percentage to allocate
        """
        # Base position size on account risk
        risk_per_trade = self.parameters["max_risk_per_trade"]
        
        # For position trading, we typically use wider stops
        # Calculate stop distance based on ATR or fixed percentage
        if "atr" in indicators and len(indicators["atr"]) > 0:
            # ATR-based stop (more adaptive to volatility)
            stop_distance_pct = (indicators["atr"].iloc[-1] * 2) / data["close"].iloc[-1]
        else:
            # Fixed percentage stop (fallback)
            stop_distance_pct = 0.05  # 5% default stop
        
        # Calculate position size based on risk
        position_size = risk_per_trade / stop_distance_pct
        
        # Factor in sector exposure limits
        symbol = self.session.symbol
        sector = self._get_symbol_sector(symbol)
        
        if sector:
            current_sector_exposure = self.sector_exposure.get(sector, 0.0)
            remaining_sector_capacity = max(0, self.parameters["sector_exposure_limit"] - current_sector_exposure)
            position_size = min(position_size, remaining_sector_capacity)
        
        # Cap at max correlated positions
        # This would need more sophisticated logic in production
        
        logger.info(f"Calculated position size: {position_size:.2%} for {symbol} with " 
                   f"stop distance: {stop_distance_pct:.2%}")
        
        return position_size
    
    def on_position_opened(self, position: Position, event: Event) -> None:
        """
        Handle position opened event.
        
        Args:
            position: New position object
            event: Position opened event
        """
        super().on_position_opened(position, event)
        
        symbol = position.symbol
        
        # Track entry timestamp
        self.entry_timestamps[symbol] = datetime.now()
        
        # Initialize highest price for trailing stop
        self.highest_prices[symbol] = position.entry_price
        
        # Initialize partial exits tracking
        self.partial_exits_taken[symbol] = []
        
        # Update sector exposure tracking
        sector = self._get_symbol_sector(symbol)
        if sector:
            current_exposure = self.sector_exposure.get(sector, 0.0)
            self.sector_exposure[sector] = current_exposure + position.size
        
        logger.info(f"Position opened tracking initialized for {symbol}")
    
    def on_position_closed(self, position: Position, event: Event) -> None:
        """
        Handle position closed event.
        
        Args:
            position: Closed position object
            event: Position closed event
        """
        super().on_position_closed(position, event)
        
        symbol = position.symbol
        
        # Clean up tracking data
        if symbol in self.entry_timestamps:
            del self.entry_timestamps[symbol]
        
        if symbol in self.highest_prices:
            del self.highest_prices[symbol]
        
        if symbol in self.partial_exits_taken:
            del self.partial_exits_taken[symbol]
        
        # Update sector exposure tracking
        sector = self._get_symbol_sector(symbol)
        if sector and symbol in self.sector_exposure:
            self.sector_exposure[sector] = max(0, self.sector_exposure.get(sector, 0.0) - position.size)
        
        logger.info(f"Position closed tracking cleaned up for {symbol}")
    
    def on_bar_closed(self, event: Event) -> None:
        """
        Handle bar closed event to check for partial take-profit exits.
        
        Args:
            event: Bar closed event
        """
        super().on_bar_closed(event)
        
        # Check for partial take-profit opportunities
        # This is a simplified implementation
        for symbol, position in self._get_active_positions().items():
            if symbol not in self.partial_exits_taken:
                continue
                
            current_price = event.data.get("close", 0)
            if current_price <= 0:
                continue
                
            # Calculate current R multiple (return in terms of initial risk)
            initial_risk = position.entry_price * 0.05  # Simplified, would use actual stop distance
            current_return = current_price - position.entry_price
            r_multiple = current_return / initial_risk if initial_risk > 0 else 0
            
            # Check if we've hit any take-profit targets
            for i, target in enumerate(self.parameters["take_profit_targets"]):
                if r_multiple >= target and i not in self.partial_exits_taken[symbol]:
                    # Take partial profit
                    exit_portion = self.parameters["partial_exit_points"][i]
                    
                    # Record the partial exit
                    self.partial_exits_taken[symbol].append(i)
                    
                    # In a real implementation, we would send a partial close order here
                    logger.info(f"Take profit {i+1} triggered for {symbol} at {current_price} "
                               f"({r_multiple:.1f}R, {exit_portion:.1%} of position)")
    
    def _position_exists(self, symbol: str) -> bool:
        """Check if a position exists for the symbol."""
        # In a real implementation, this would check with the position manager
        return symbol in self.positions_data
    
    def _get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position data for a symbol."""
        return self.positions_data.get(symbol)
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get the sector for a symbol."""
        # In a real implementation, this would lookup the sector from a data source
        return "technology"  # Placeholder
    
    def _get_active_positions(self) -> Dict[str, Any]:
        """Get all active positions."""
        return self.positions_data
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the position trading strategy is with the current market regime.
        
        Position trading works best in strong trending markets and is less effective
        in highly volatile or ranging markets.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            "strong_bull_trend": 0.95,    # Excellent in strong bull trends
            "weak_bull_trend": 0.85,      # Very good in weak bull trends
            "strong_bear_trend": 0.30,    # Poor in strong bear trends (better to be short)
            "weak_bear_trend": 0.40,      # Below average in weak bear trends
            "bull_volatile": 0.60,        # Moderate in bull but volatile markets
            "bear_volatile": 0.20,        # Very poor in bear volatile markets
            "ranging_narrow": 0.35,       # Poor in narrow ranges
            "ranging_wide": 0.50,         # Average in wide ranges
            "high_volatility": 0.30,      # Poor in high volatility environments
            "low_volatility": 0.70,       # Good in low volatility environments
            "transitioning": 0.50,        # Average during market transitions
            "sector_rotation": 0.65,      # Above average during sector rotations (selective entries)
            "earnings_season": 0.55,      # Average during earnings seasons (event risk)
            "macro_news_driven": 0.40,    # Below average in news-driven markets
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.60)
