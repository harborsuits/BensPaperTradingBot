#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gap Trading Strategy Module

This module implements a strategy to capture sharp moves that occur when markets
"gap" open above or below key reference levels, then trades the initial momentum
or the reversal based on pre-market setups and intraday confirmation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, time

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Setup logging
logger = logging.getLogger(__name__)

class GapTradingStrategy(StrategyOptimizable):
    """
    Gap Trading Strategy designed to capture price moves at market open.
    
    This strategy identifies and trades significant price gaps at market open,
    either by riding the momentum in gap direction or fading the gap with a reversal
    play. It relies on pre-market setups and disciplined execution to exploit
    overnight information.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Gap Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # Highly liquid stocks and ETFs with narrow bid-ask spreads
            "trading_start_morning": "09:30",  # Trading window start (ET)
            "trading_end_morning": "10:30",    # Trading window end (ET)
            "trading_start_afternoon": "13:30", # Afternoon window start (ET)
            "trading_end_afternoon": "14:30",   # Afternoon window end (ET)
            "use_afternoon_window": False,      # Whether to trade in the afternoon window
            
            # Core Indicators
            "gap_threshold_percent": 1.0,       # Minimum gap size (%)
            "premarket_volume_multiplier": 1.5, # Pre-market volume vs 20-day average
            "premarket_avg_days": 20,           # Days to average pre-market volume
            "atr_period": 14,                   # ATR period
            "market_breadth_threshold": 0.6,    # Market breadth confirmation (0-1)
            
            # Entry Criteria - Momentum Play
            "momentum_enabled": True,           # Enable momentum gap play
            "momentum_min_slope": 0.0001,       # Minimum VWAP slope for trend confirmation
            
            # Entry Criteria - Fade Play
            "fade_enabled": True,               # Enable fade gap play
            "fade_vol_multiplier": 2.0,         # Volume spike threshold for fade confirmation
            
            # Exit Criteria
            "momentum_profit_target_atr": 1.0,  # Profit target for momentum plays (ATR multiple)
            "fade_profit_target_atr": 0.8,      # Profit target for fade plays (ATR multiple)
            "momentum_stop_loss_atr": 0.5,      # Stop loss for momentum plays (ATR multiple)
            "fade_stop_loss_atr": 1.0,          # Stop loss for fade plays (ATR multiple)
            "trailing_stop_activation_atr": 0.75, # When to activate trailing stop (ATR multiple)
            "trailing_stop_atr": 0.5,           # Trailing stop distance (ATR multiple)
            
            # Position Sizing & Risk Controls
            "momentum_risk_percent": 0.0075,    # 0.75% of equity for momentum plays
            "fade_risk_percent": 0.005,         # 0.5% of equity for fade plays
            "max_concurrent_positions": 2,      # Maximum concurrent gap positions
            "daily_loss_limit_percent": 0.015,  # Daily loss limit (1.5% of equity)
            
            # Order Execution
            "slippage_buffer_percent": 0.0002,  # 0.02% buffer for slippage
            "max_entry_delay_bars": 2,          # Cancel entry if not filled within N bars
            
            # Operational Rules
            "skip_earnings_gaps": True,         # Skip gaps caused by earnings
            "max_market_gap_percent": 2.0,      # Skip if overall market gap exceeds this %
            "catalyst_instruments": []          # List of instruments to trade despite news
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Gap Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "gap_threshold_percent": [0.5, 0.75, 1.0, 1.5, 2.0],
            "premarket_volume_multiplier": [1.0, 1.5, 2.0, 2.5],
            "premarket_avg_days": [10, 20, 30],
            "atr_period": [10, 14, 20],
            "market_breadth_threshold": [0.5, 0.6, 0.7],
            "momentum_min_slope": [0.00005, 0.0001, 0.0002],
            "fade_vol_multiplier": [1.5, 2.0, 2.5, 3.0],
            "momentum_profit_target_atr": [0.8, 1.0, 1.2, 1.5],
            "fade_profit_target_atr": [0.6, 0.8, 1.0],
            "momentum_stop_loss_atr": [0.3, 0.5, 0.7],
            "fade_stop_loss_atr": [0.8, 1.0, 1.2],
            "trailing_stop_activation_atr": [0.5, 0.75, 1.0],
            "trailing_stop_atr": [0.3, 0.5, 0.7],
            "momentum_risk_percent": [0.005, 0.0075, 0.01],
            "fade_risk_percent": [0.0025, 0.005, 0.0075],
            "max_concurrent_positions": [1, 2, 3],
            "daily_loss_limit_percent": [0.01, 0.015, 0.02],
            "skip_earnings_gaps": [True, False],
            "max_market_gap_percent": [1.5, 2.0, 2.5, 3.0]
        }
    
    def calculate_gap_percent(self, current_open: float, previous_close: float) -> float:
        """
        Calculate gap size as a percentage.
        
        Args:
            current_open: Current day's opening price
            previous_close: Previous day's closing price
            
        Returns:
            Gap percentage
        """
        return (current_open - previous_close) / previous_close * 100
    
    def is_gap_large_enough(self, gap_percent: float) -> bool:
        """
        Check if gap size meets the threshold.
        
        Args:
            gap_percent: Gap size as percentage
            
        Returns:
            Boolean indicating if gap is large enough
        """
        threshold = self.parameters.get("gap_threshold_percent", 1.0)
        return abs(gap_percent) >= threshold
    
    def is_premarket_volume_sufficient(self, 
                                      premarket_volume: float, 
                                      avg_premarket_volume: float) -> bool:
        """
        Check if pre-market volume is sufficient.
        
        Args:
            premarket_volume: Current pre-market volume
            avg_premarket_volume: Average pre-market volume for the lookback period
            
        Returns:
            Boolean indicating if pre-market volume is sufficient
        """
        multiplier = self.parameters.get("premarket_volume_multiplier", 1.5)
        return premarket_volume >= (multiplier * avg_premarket_volume)
    
    def calculate_vwap(self, ohlcv_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        typical_price = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
        volume = ohlcv_df['volume']
        
        # Calculate VWAP
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    def calculate_vwap_slope(self, vwap: pd.Series, window: int = 3) -> pd.Series:
        """
        Calculate the slope of VWAP.
        
        Args:
            vwap: Series with VWAP values
            window: Window size for slope calculation
            
        Returns:
            Series with VWAP slope values
        """
        return vwap.diff(window) / window
    
    def calculate_atr(self, ohlcv_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        high = ohlcv_df['high']
        low = ohlcv_df['low']
        close = ohlcv_df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def is_market_breadth_confirming(self, market_breadth_data: pd.DataFrame, gap_direction: int) -> bool:
        """
        Check if market breadth confirms the gap direction.
        
        Args:
            market_breadth_data: DataFrame with market breadth data (e.g., advance-decline)
            gap_direction: Direction of the gap (1 for up, -1 for down)
            
        Returns:
            Boolean indicating if market breadth confirms the gap
        """
        threshold = self.parameters.get("market_breadth_threshold", 0.6)
        
        # TODO: Implement market breadth confirmation based on advance-decline or sector strength
        # This is a placeholder implementation
        if 'advance_decline_ratio' in market_breadth_data.columns:
            ad_ratio = market_breadth_data['advance_decline_ratio'].iloc[-1]
            if gap_direction > 0:  # Up gap
                return ad_ratio >= threshold
            else:  # Down gap
                return (1 - ad_ratio) >= threshold
        
        return True  # Default to true if no market breadth data available
    
    def is_within_trading_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is within the trading hours.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            Boolean indicating if timestamp is within trading hours
        """
        # Parse trading window times
        morning_start = self.parameters.get("trading_start_morning", "09:30")
        morning_end = self.parameters.get("trading_end_morning", "10:30")
        
        # Convert to datetime.time objects
        morning_start_time = time(*map(int, morning_start.split(":")))
        morning_end_time = time(*map(int, morning_end.split(":")))
        
        # Check if timestamp is within morning window
        timestamp_time = timestamp.time()
        in_morning_session = morning_start_time <= timestamp_time <= morning_end_time
        
        # If afternoon window is enabled, check that too
        if self.parameters.get("use_afternoon_window", False):
            afternoon_start = self.parameters.get("trading_start_afternoon", "13:30")
            afternoon_end = self.parameters.get("trading_end_afternoon", "14:30")
            
            afternoon_start_time = time(*map(int, afternoon_start.split(":")))
            afternoon_end_time = time(*map(int, afternoon_end.split(":")))
            
            in_afternoon_session = afternoon_start_time <= timestamp_time <= afternoon_end_time
            return in_morning_session or in_afternoon_session
        
        return in_morning_session
    
    def is_reversal_candle(self, candle: pd.Series, gap_direction: int) -> bool:
        """
        Check if the candle shows reversal characteristics.
        
        Args:
            candle: Series with candle data (open, high, low, close)
            gap_direction: Direction of the gap (1 for up, -1 for down)
            
        Returns:
            Boolean indicating if it's a reversal candle
        """
        open_price = candle['open']
        high_price = candle['high']
        low_price = candle['low']
        close_price = candle['close']
        
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range <= 0:
            return False
        
        # For up gaps, look for shooting star (small body at bottom, long upper wick)
        if gap_direction > 0:
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            # Shooting star: upper wick at least 2x body, small or no lower wick
            return (upper_wick >= 2 * body_size) and (lower_wick < body_size) and (close_price < open_price)
        
        # For down gaps, look for hammer (small body at top, long lower wick)
        else:
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            # Hammer: lower wick at least 2x body, small or no upper wick
            return (lower_wick >= 2 * body_size) and (upper_wick < body_size) and (close_price > open_price)
    
    def is_first_bar_confirming(self, first_bar: pd.Series, gap_direction: int) -> bool:
        """
        Check if the first 5-min bar confirms gap direction (for momentum play).
        
        Args:
            first_bar: Series with first bar data
            gap_direction: Direction of the gap (1 for up, -1 for down)
            
        Returns:
            Boolean indicating if first bar confirms direction
        """
        if gap_direction > 0:  # Up gap
            return first_bar['close'] > first_bar['open']
        else:  # Down gap
            return first_bar['close'] < first_bar['open']
    
    def is_volume_spike(self, current_volume: float, avg_volume: float) -> bool:
        """
        Check if current volume is a spike compared to average.
        
        Args:
            current_volume: Current volume
            avg_volume: Average volume
            
        Returns:
            Boolean indicating if volume is a spike
        """
        multiplier = self.parameters.get("fade_vol_multiplier", 2.0)
        return current_volume >= (multiplier * avg_volume)
    
    def should_skip_due_to_market_gap(self, market_index_gap_percent: float) -> bool:
        """
        Check if strategy should skip due to large market index gap.
        
        Args:
            market_index_gap_percent: Gap percentage of market index
            
        Returns:
            Boolean indicating if should skip
        """
        max_market_gap = self.parameters.get("max_market_gap_percent", 2.0)
        return abs(market_index_gap_percent) > max_market_gap
    
    def should_skip_due_to_earnings(self, symbol: str, has_earnings: bool) -> bool:
        """
        Check if strategy should skip due to earnings announcement.
        
        Args:
            symbol: Instrument symbol
            has_earnings: Whether there was an earnings announcement
            
        Returns:
            Boolean indicating if should skip
        """
        if not self.parameters.get("skip_earnings_gaps", True):
            return False
            
        # Don't skip if symbol is in catalyst instruments list
        catalyst_instruments = self.parameters.get("catalyst_instruments", [])
        if symbol in catalyst_instruments:
            return False
            
        return has_earnings
    
    def calculate_position_size(self, equity: float, stop_distance: float, is_momentum: bool) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            equity: Total equity
            stop_distance: Stop loss distance in currency units
            is_momentum: Whether this is a momentum play
            
        Returns:
            Position size
        """
        if is_momentum:
            risk_percent = self.parameters.get("momentum_risk_percent", 0.0075)
        else:
            risk_percent = self.parameters.get("fade_risk_percent", 0.005)
            
        risk_amount = equity * risk_percent
        
        # Position size formula: size = (equity × risk%) / stop_distance
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        return position_size
    
    def generate_signals(
        self, 
        data: Dict[str, Dict[str, pd.DataFrame]], 
        equity: float,
        market_breadth_data: Optional[pd.DataFrame] = None,
        market_index_data: Optional[pd.DataFrame] = None,
        earnings_data: Optional[Dict[str, bool]] = None,
        current_positions_count: int = 0
    ) -> Dict[str, Signal]:
        """
        Generate gap trading signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            equity: Current equity value
            market_breadth_data: Optional DataFrame with market breadth data
            market_index_data: Optional DataFrame with market index data
            earnings_data: Optional Dictionary mapping symbols to earnings flag
            current_positions_count: Number of currently open positions
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Check if max concurrent positions reached
        max_positions = self.parameters.get("max_concurrent_positions", 2)
        if current_positions_count >= max_positions:
            return {}
            
        # Check if market gap is too large
        should_skip_market = False
        if market_index_data is not None and 'daily' in market_index_data:
            market_daily = market_index_data['daily']
            if len(market_daily) >= 2:
                market_prev_close = market_daily.iloc[-2]['close']
                market_open = market_daily.iloc[-1]['open']
                market_gap_percent = self.calculate_gap_percent(market_open, market_prev_close)
                should_skip_market = self.should_skip_due_to_market_gap(market_gap_percent)
        
        if should_skip_market:
            logger.info("Skipping gap trades due to large market gap")
            return {}
        
        # Generate signals
        signals = {}
        
        for symbol, timeframe_data in data.items():
            try:
                # Need both daily and 5-minute data
                if 'daily' not in timeframe_data or '5min' not in timeframe_data:
                    continue
                
                daily_data = timeframe_data['daily']
                intraday_data = timeframe_data['5min']
                
                # Need at least 2 days of daily data and some intraday data
                if len(daily_data) < 2 or len(intraday_data) < 1:
                    continue
                
                # Get previous close and current open
                prev_day = daily_data.iloc[-2]
                current_day = daily_data.iloc[-1]
                
                prev_close = prev_day['close']
                current_open = current_day['open']
                
                # Calculate gap percent
                gap_percent = self.calculate_gap_percent(current_open, prev_close)
                
                # Check if gap is large enough
                if not self.is_gap_large_enough(gap_percent):
                    continue
                
                # Determine gap direction
                gap_direction = 1 if gap_percent > 0 else -1
                
                # Get current timestamp from the latest intraday bar
                latest_timestamp = intraday_data.index[-1]
                
                # Skip if not within trading hours
                if not self.is_within_trading_hours(latest_timestamp):
                    continue
                
                # Skip if earnings announcement and not a catalyst instrument
                if earnings_data and symbol in earnings_data:
                    has_earnings = earnings_data[symbol]
                    if self.should_skip_due_to_earnings(symbol, has_earnings):
                        continue
                
                # Get pre-market volume (if available)
                premarket_volume_sufficient = True
                if 'premarket_volume' in current_day and 'avg_premarket_volume' in current_day:
                    premarket_volume = current_day['premarket_volume']
                    avg_premarket_volume = current_day['avg_premarket_volume']
                    premarket_volume_sufficient = self.is_premarket_volume_sufficient(
                        premarket_volume, avg_premarket_volume
                    )
                
                if not premarket_volume_sufficient:
                    continue
                
                # Check market breadth confirmation
                market_breadth_confirming = True
                if market_breadth_data is not None:
                    market_breadth_confirming = self.is_market_breadth_confirming(
                        market_breadth_data, gap_direction
                    )
                
                if not market_breadth_confirming:
                    continue
                
                # Calculate intraday indicators
                vwap = self.calculate_vwap(intraday_data)
                vwap_slope = self.calculate_vwap_slope(vwap)
                atr = self.calculate_atr(intraday_data, self.parameters.get("atr_period", 14))
                
                # Get latest intraday bar and values
                latest_bar = intraday_data.iloc[-1]
                latest_price = latest_bar['close']
                latest_vwap = vwap.iloc[-1]
                latest_vwap_slope = vwap_slope.iloc[-1]
                latest_atr = atr.iloc[-1]
                
                # Get first bar of the day (assuming sorted chronologically)
                # In a real implementation, this would need to identify the first bar after market open
                first_bar_idx = 0
                for i, idx in enumerate(intraday_data.index):
                    if idx.time() >= time(9, 30):  # 9:30 AM
                        first_bar_idx = i
                        break
                
                if first_bar_idx >= len(intraday_data):
                    continue
                
                first_bar = intraday_data.iloc[first_bar_idx]
                
                # Check 5-min volume
                recent_volume_avg = intraday_data['volume'].rolling(window=5).mean().iloc[-1]
                volume_spike = self.is_volume_spike(latest_bar['volume'], recent_volume_avg)
                
                # Signal variables
                signal_type = None
                is_momentum = False
                confidence = 0.0
                
                # ------------- MOMENTUM PLAY (RIDE THE GAP) -------------
                if self.parameters.get("momentum_enabled", True):
                    first_bar_confirms = self.is_first_bar_confirming(first_bar, gap_direction)
                    vwap_trend_confirms = (
                        (gap_direction > 0 and latest_vwap_slope > self.parameters.get("momentum_min_slope", 0.0001)) or
                        (gap_direction < 0 and latest_vwap_slope < -self.parameters.get("momentum_min_slope", 0.0001))
                    )
                    
                    if first_bar_confirms and vwap_trend_confirms:
                        signal_type = SignalType.BUY if gap_direction > 0 else SignalType.SELL
                        is_momentum = True
                        
                        # Calculate confidence based on multiple factors
                        gap_strength = min(0.3, abs(gap_percent) / 5.0)  # 0.3 max for a 5% gap
                        trend_strength = min(0.2, abs(latest_vwap_slope) * 1000)  # Scale slope
                        volume_strength = 0.1 if premarket_volume_sufficient else 0.0
                        first_bar_strength = 0.2 if first_bar_confirms else 0.0
                        
                        confidence = min(0.9, 0.3 + gap_strength + trend_strength + volume_strength + first_bar_strength)
                
                # ------------- FADE PLAY (REVERSAL) -------------
                elif self.parameters.get("fade_enabled", True) and not is_momentum:
                    is_reversal = self.is_reversal_candle(first_bar, gap_direction)
                    price_near_vwap = abs(latest_price - latest_vwap) / latest_price < 0.002  # Within 0.2%
                    
                    if is_reversal and price_near_vwap and volume_spike:
                        signal_type = SignalType.SELL if gap_direction > 0 else SignalType.BUY
                        is_momentum = False
                        
                        # Calculate confidence based on multiple factors
                        gap_strength = min(0.2, abs(gap_percent) / 5.0)  # 0.2 max for a 5% gap
                        reversal_strength = 0.3 if is_reversal else 0.0
                        vwap_touch_strength = 0.2 if price_near_vwap else 0.0
                        volume_strength = 0.2 if volume_spike else 0.0
                        
                        confidence = min(0.9, 0.2 + gap_strength + reversal_strength + vwap_touch_strength + volume_strength)
                
                # If we have a signal, calculate stops and targets
                if signal_type:
                    # Calculate stop loss and take profit based on ATR
                    if is_momentum:
                        profit_target_multiple = self.parameters.get("momentum_profit_target_atr", 1.0)
                        stop_loss_multiple = self.parameters.get("momentum_stop_loss_atr", 0.5)
                    else:
                        profit_target_multiple = self.parameters.get("fade_profit_target_atr", 0.8)
                        stop_loss_multiple = self.parameters.get("fade_stop_loss_atr", 1.0)
                    
                    # Apply slippage buffer
                    slippage_buffer = self.parameters.get("slippage_buffer_percent", 0.0002)
                    
                    if signal_type == SignalType.BUY:
                        stop_loss = latest_price * (1 - stop_loss_multiple * latest_atr / latest_price - slippage_buffer)
                        take_profit = latest_price * (1 + profit_target_multiple * latest_atr / latest_price + slippage_buffer)
                    else:  # SELL
                        stop_loss = latest_price * (1 + stop_loss_multiple * latest_atr / latest_price + slippage_buffer)
                        take_profit = latest_price * (1 - profit_target_multiple * latest_atr / latest_price - slippage_buffer)
                    
                    # Calculate position size
                    stop_distance = abs(latest_price - stop_loss)
                    position_size = self.calculate_position_size(equity, stop_distance, is_momentum)
                    
                    # Create signal
                    trailing_stop_activation = self.parameters.get("trailing_stop_activation_atr", 0.75) * latest_atr
                    trailing_stop_value = self.parameters.get("trailing_stop_atr", 0.5) * latest_atr
                    
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=latest_price,
                        timestamp=latest_timestamp,
                        confidence=confidence,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        size=position_size,
                        metadata={
                            "strategy_type": "gap_trading",
                            "play_type": "momentum" if is_momentum else "fade",
                            "gap_percent": gap_percent,
                            "atr": latest_atr,
                            "vwap": latest_vwap,
                            "trailing_stop": True,
                            "trailing_stop_activation": trailing_stop_activation,
                            "trailing_stop_value": trailing_stop_value,
                            "max_entry_delay_bars": self.parameters.get("max_entry_delay_bars", 2)
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating gap trading signal for {symbol}: {e}")
        
        return signals
    
    def is_daily_loss_limit_reached(self, daily_pnl: float, equity: float) -> bool:
        """
        Check if the daily loss limit has been reached.
        
        Args:
            daily_pnl: Current day's P&L
            equity: Total equity
            
        Returns:
            Boolean indicating if daily loss limit is reached
        """
        daily_loss_limit = self.parameters.get("daily_loss_limit_percent", 0.015)
        return daily_pnl <= -(equity * daily_loss_limit)
    
    def should_cancel_entry_order(self, order_age_bars: int) -> bool:
        """
        Check if a pending entry order should be canceled due to age.
        
        Args:
            order_age_bars: Age of the order in bars
            
        Returns:
            Boolean indicating if order should be canceled
        """
        max_delay = self.parameters.get("max_entry_delay_bars", 2)
        return order_age_bars >= max_delay 