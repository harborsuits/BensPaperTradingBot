#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Range Trading Strategy Module

This module implements a strategy to capture price oscillations between established 
support and resistance levels by buying low and selling high (or vice-versa),
emphasizing consistency and small, repeatable profits.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, time, timedelta

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

class RangeTradingStrategy(StrategyOptimizable):
    """
    Range Trading Strategy designed to capture price oscillations.
    
    This strategy identifies and trades established price ranges by buying at support
    and selling at resistance, using oscillator indicators to confirm entries and exits.
    It operates in both intraday and multi-day timeframes with appropriate validations
    to ensure the market is rangebound.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Range Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # Liquid stocks, ETFs, futures or FX pairs that exhibit clear ranges
            "timeframe_mode": "intraday",  # 'intraday' or 'multi_day'
            "signal_timeframe_intraday": "15min",  # or '1h'
            "signal_timeframe_multi_day": "1d",
            "context_timeframe_intraday": "4h",
            "context_timeframe_multi_day": "1w",
            "max_positions": 3,  # Maximum concurrent range positions
            
            # Core Indicators
            "donchian_period": 20,  # Period for Donchian channels
            "rsi_period": 14,  # Period for RSI calculation
            "stoch_k_period": 14,  # %K period for Stochastic Oscillator
            "stoch_d_period": 3,  # %D period for Stochastic Oscillator
            "stoch_slowing": 3,  # Slowing period for Stochastic Oscillator
            "volume_ma_period": 20,  # Period for volume moving average
            "atr_period": 14,  # Period for ATR calculation
            
            # Entry Criteria
            "rsi_oversold": 35,  # RSI oversold threshold for long entries
            "rsi_overbought": 65,  # RSI overbought threshold for short entries
            "stoch_oversold": 20,  # Stochastic oversold threshold for long entries
            "stoch_overbought": 80,  # Stochastic overbought threshold for short entries
            "volume_threshold": 1.2,  # Volume vs MA threshold for confirmation
            
            # Exit Criteria
            "profit_target_atr_multiple": 1.0,  # Profit target as ATR multiple
            "stop_loss_atr_multiple": 0.5,  # Stop loss as ATR multiple
            "use_midpoint_exit": True,  # Whether to use channel midpoint as first target
            "scale_out_enabled": False,  # Whether to scale out of positions
            "max_bars_in_trade_intraday": 4,  # Maximum bars to hold intraday trade
            "max_bars_in_trade_multi_day": 10,  # Maximum bars to hold multi-day trade
            
            # Position Sizing & Risk Controls
            "risk_percent_intraday": 0.005,  # 0.5% for intraday ranges
            "risk_percent_multi_day": 0.01,  # 1% for multi-day ranges
            "max_exposure_percent": 0.15,  # 15% max exposure
            "max_consecutive_losses": 2,  # Halt after consecutive losses
            "intraday_pause_hours": 1,  # Hours to pause after consecutive losses (intraday)
            "multi_day_pause_days": 3,  # Days to pause after consecutive losses (multi-day)
            
            # Order Execution
            "use_limit_orders": True,  # Use limit orders for entry
            "limit_order_max_bars": 2,  # Maximum bars to wait for limit order fill
            "slippage_buffer_percent": 0.0001,  # 0.01% buffer for slippage
            "max_spread_percent": 0.0005,  # Maximum acceptable spread (0.05%)
            
            # Operational Rules
            "max_atr_percent": 0.02,  # Maximum ATR as percent of price (avoid trending)
            "news_filter_enabled": True,  # Avoid entries near news events
            "reset_donchian_at_open": True,  # Reset Donchian channels at market open
            "max_sector_positions": 2,  # Maximum positions per sector
            
            # Continuous Optimization
            "volatility_adjustment_threshold": 0.2,  # Threshold for channel adjustment
            "use_ml_overlay": False  # Whether to use ML for filtering setups
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize state variables
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.current_exposure = 0.0
        self.sector_positions = {}  # Track positions per sector
        self.range_entries = {}  # Track entries on range boundaries
        
        logger.info(f"Initialized Range Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "donchian_period": [10, 15, 20, 25, 30],
            "rsi_period": [7, 10, 14, 21],
            "rsi_oversold": [25, 30, 35, 40],
            "rsi_overbought": [60, 65, 70, 75],
            "stoch_k_period": [9, 14, 21],
            "stoch_oversold": [10, 15, 20, 25],
            "stoch_overbought": [75, 80, 85, 90],
            "volume_threshold": [1.0, 1.2, 1.5, 2.0],
            "profit_target_atr_multiple": [0.8, 1.0, 1.2, 1.5],
            "stop_loss_atr_multiple": [0.3, 0.5, 0.7, 1.0],
            "use_midpoint_exit": [True, False],
            "scale_out_enabled": [True, False],
            "max_bars_in_trade_intraday": [3, 4, 5, 6],
            "max_bars_in_trade_multi_day": [5, 7, 10, 14],
            "risk_percent_intraday": [0.003, 0.005, 0.007],
            "risk_percent_multi_day": [0.007, 0.01, 0.015],
            "use_limit_orders": [True, False],
            "max_atr_percent": [0.01, 0.015, 0.02, 0.03]
        }
    
    # --------------------------------------------------------
    # 3. Core Indicators
    # --------------------------------------------------------
    
    def calculate_donchian_channels(
        self, high: pd.Series, low: pd.Series, period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channels to define range boundaries.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            period: Lookback period
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # TODO: compute Donchian channels
        upper_band = high.rolling(window=period).max()
        lower_band = low.rolling(window=period).min()
        middle_band = (upper_band + lower_band) / 2
        
        return upper_band, middle_band, lower_band
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        # TODO: compute RSI
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses over the period
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_stochastic(
        self, high: pd.Series, low: pd.Series, close: pd.Series, 
        k_period: int = 14, d_period: int = 3, slowing: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            k_period: Period for %K line
            d_period: Period for %D line
            slowing: Slowing period
            
        Returns:
            Tuple of (%K, %D)
        """
        # TODO: compute stochastic oscillator
        # Calculate %K
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        # Handle division by zero
        range_diff = highest_high - lowest_low
        range_diff = range_diff.replace(0, np.nan)
        
        fast_k = 100 * ((close - lowest_low) / range_diff)
        
        # Apply slowing if specified
        if slowing > 1:
            slow_k = fast_k.rolling(window=slowing).mean()
        else:
            slow_k = fast_k
            
        # Calculate %D (moving average of %K)
        slow_d = slow_k.rolling(window=d_period).mean()
        
        return slow_k, slow_d
    
    def calculate_vwap(self, ohlcv_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        # TODO: compute VWAP
        typical_price = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
        volume = ohlcv_df['volume']
        
        # Calculate VWAP
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        return vwap
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for volatility measurement.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ATR period
            
        Returns:
            Series with ATR values
        """
        # TODO: compute ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_volume_surge(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate volume surge relative to average volume.
        
        Args:
            volume: Series of volume data
            period: Period for volume average
            
        Returns:
            Series with volume surge ratio
        """
        # TODO: compute volume surge ratio
        volume_ma = volume.rolling(window=period).mean()
        volume_ratio = volume / volume_ma
        
        return volume_ratio
    
    # --------------------------------------------------------
    # 4. Entry Criteria
    # --------------------------------------------------------
    
    def is_at_support(
        self, close: float, lower_band: float, atr: float
    ) -> bool:
        """
        Check if price is at support level (lower Donchian band).
        
        Args:
            close: Current close price
            lower_band: Lower Donchian band value
            atr: Current ATR value
            
        Returns:
            Boolean indicating if price is at support
        """
        # TODO: check support level
        # Allow for a small buffer (5% of ATR) below the band
        buffer = 0.05 * atr
        return close <= lower_band + buffer
    
    def is_at_resistance(
        self, close: float, upper_band: float, atr: float
    ) -> bool:
        """
        Check if price is at resistance level (upper Donchian band).
        
        Args:
            close: Current close price
            upper_band: Upper Donchian band value
            atr: Current ATR value
            
        Returns:
            Boolean indicating if price is at resistance
        """
        # TODO: check resistance level
        # Allow for a small buffer (5% of ATR) above the band
        buffer = 0.05 * atr
        return close >= upper_band - buffer
    
    def is_oversold(
        self, rsi: float, stoch_k: float, stoch_d: float
    ) -> bool:
        """
        Check if oscillators indicate oversold conditions.
        
        Args:
            rsi: Current RSI value
            stoch_k: Current Stochastic %K value
            stoch_d: Current Stochastic %D value
            
        Returns:
            Boolean indicating if market is oversold
        """
        # TODO: check oversold oscillators
        rsi_threshold = self.parameters.get("rsi_oversold", 35)
        stoch_threshold = self.parameters.get("stoch_oversold", 20)
        
        rsi_oversold = rsi < rsi_threshold
        
        # Check for stochastic crossover from oversold
        stoch_oversold = stoch_k < stoch_threshold and stoch_k > stoch_d
        
        return rsi_oversold or stoch_oversold
    
    def is_overbought(
        self, rsi: float, stoch_k: float, stoch_d: float
    ) -> bool:
        """
        Check if oscillators indicate overbought conditions.
        
        Args:
            rsi: Current RSI value
            stoch_k: Current Stochastic %K value
            stoch_d: Current Stochastic %D value
            
        Returns:
            Boolean indicating if market is overbought
        """
        # TODO: check overbought oscillators
        rsi_threshold = self.parameters.get("rsi_overbought", 65)
        stoch_threshold = self.parameters.get("stoch_overbought", 80)
        
        rsi_overbought = rsi > rsi_threshold
        
        # Check for stochastic crossover from overbought
        stoch_overbought = stoch_k > stoch_threshold and stoch_k < stoch_d
        
        return rsi_overbought or stoch_overbought
    
    def has_volume_confirmation(self, volume_ratio: float) -> bool:
        """
        Check if there is sufficient volume to confirm entry.
        
        Args:
            volume_ratio: Current volume ratio vs MA
            
        Returns:
            Boolean indicating if volume confirms
        """
        # TODO: check volume confirmation
        volume_threshold = self.parameters.get("volume_threshold", 1.2)
        return volume_ratio >= volume_threshold
    
    def is_range_context(
        self, close: float, htf_upper_band: float, htf_lower_band: float
    ) -> bool:
        """
        Check if higher timeframe context confirms range environment.
        
        Args:
            close: Current close price
            htf_upper_band: Higher timeframe upper Donchian band
            htf_lower_band: Higher timeframe lower Donchian band
            
        Returns:
            Boolean indicating if higher timeframe context confirms range
        """
        # TODO: check range context
        # Price should be within higher timeframe bands
        return htf_lower_band < close < htf_upper_band
    
    # --------------------------------------------------------
    # 5. Exit Criteria
    # --------------------------------------------------------
    
    def calculate_profit_target(
        self, entry_price: float, middle_band: float, opposite_band: float, 
        atr: float, signal_type: SignalType
    ) -> Union[float, Tuple[float, float]]:
        """
        Calculate profit target(s) for the trade.
        
        Args:
            entry_price: Entry price
            middle_band: Middle Donchian band value
            opposite_band: Opposite Donchian band value (upper for longs, lower for shorts)
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Either a single target price or tuple of (first_target, second_target)
        """
        # TODO: compute profit target
        atr_target_multiple = self.parameters.get("profit_target_atr_multiple", 1.0)
        use_midpoint = self.parameters.get("use_midpoint_exit", True)
        scale_out = self.parameters.get("scale_out_enabled", False)
        
        # Calculate ATR-based target
        atr_target = entry_price + (atr * atr_target_multiple) if signal_type == SignalType.BUY else entry_price - (atr * atr_target_multiple)
        
        # Use Donchian channel targets if available
        if use_midpoint and scale_out:
            # Return both midpoint and opposite band as targets
            if signal_type == SignalType.BUY:
                return middle_band, opposite_band
            else:
                return middle_band, opposite_band
        elif use_midpoint:
            # Return midpoint as target
            return middle_band
        else:
            # Return opposite band or ATR-based target, whichever is closer
            if signal_type == SignalType.BUY:
                return min(opposite_band, atr_target)
            else:
                return max(opposite_band, atr_target)
    
    def calculate_stop_loss(
        self, entry_price: float, entry_band: float, atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate stop loss level.
        
        Args:
            entry_price: Entry price
            entry_band: Donchian band at entry (lower for longs, upper for shorts)
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Stop loss price
        """
        # TODO: compute stop loss
        stop_multiple = self.parameters.get("stop_loss_atr_multiple", 0.5)
        stop_distance = atr * stop_multiple
        
        # Place stop beyond the band
        if signal_type == SignalType.BUY:
            return min(entry_price - stop_distance, entry_band - (0.1 * atr))
        else:
            return max(entry_price + stop_distance, entry_band + (0.1 * atr))
    
    def should_exit_by_time(
        self, bars_in_trade: int, is_intraday: bool
    ) -> bool:
        """
        Check if position should be exited based on time criteria.
        
        Args:
            bars_in_trade: Number of bars since entry
            is_intraday: Whether this is an intraday trade
            
        Returns:
            Boolean indicating if time-based exit should be triggered
        """
        # TODO: evaluate time-based exit
        max_bars = self.parameters.get(
            "max_bars_in_trade_intraday" if is_intraday else "max_bars_in_trade_multi_day",
            4 if is_intraday else 10
        )
        
        return bars_in_trade >= max_bars
    
    # --------------------------------------------------------
    # 6. Position Sizing & Risk Controls
    # --------------------------------------------------------
    
    def calculate_position_size(
        self, equity: float, entry_price: float, stop_price: float, is_intraday: bool
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            equity: Total equity
            entry_price: Entry price
            stop_price: Stop loss price
            is_intraday: Whether this is an intraday trade
            
        Returns:
            Position size
        """
        # TODO: compute position size
        if is_intraday:
            risk_percent = self.parameters.get("risk_percent_intraday", 0.005)
        else:
            risk_percent = self.parameters.get("risk_percent_multi_day", 0.01)
            
        risk_amount = equity * risk_percent
        stop_distance = abs(entry_price - stop_price)
        
        # Position size formula: size = (equity × risk%) / stop_distance
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        return position_size
    
    def can_add_new_position(
        self, equity: float, current_exposure: float
    ) -> bool:
        """
        Check if a new position can be added based on exposure limits.
        
        Args:
            equity: Total equity
            current_exposure: Current exposure amount
            
        Returns:
            Boolean indicating if a new position can be added
        """
        # TODO: apply exposure limit check
        max_exposure = self.parameters.get("max_exposure_percent", 0.15) * equity
        
        return current_exposure < max_exposure
    
    def can_trade_after_losses(
        self, current_time: datetime, is_intraday: bool
    ) -> bool:
        """
        Check if trading is allowed after consecutive losses.
        
        Args:
            current_time: Current timestamp
            is_intraday: Whether this is an intraday strategy
            
        Returns:
            Boolean indicating if trading is allowed
        """
        # TODO: apply consecutive loss check
        max_losses = self.parameters.get("max_consecutive_losses", 2)
        
        if self.consecutive_losses >= max_losses and self.last_loss_time is not None:
            if is_intraday:
                pause_hours = self.parameters.get("intraday_pause_hours", 1)
                return (current_time - self.last_loss_time) >= timedelta(hours=pause_hours)
            else:
                pause_days = self.parameters.get("multi_day_pause_days", 3)
                return (current_time - self.last_loss_time) >= timedelta(days=pause_days)
        
        return True
    
    def can_add_sector_position(self, sector: str, current_sector_positions: Dict[str, int]) -> bool:
        """
        Check if a new position can be added to a sector.
        
        Args:
            sector: Sector of the new position
            current_sector_positions: Dictionary mapping sectors to current position counts
            
        Returns:
            Boolean indicating if a new sector position can be added
        """
        # TODO: apply sector position limit check
        max_sector_positions = self.parameters.get("max_sector_positions", 2)
        current_count = current_sector_positions.get(sector, 0)
        
        return current_count < max_sector_positions
    
    # --------------------------------------------------------
    # 7 & 8. Order Execution & Operational Rules
    # --------------------------------------------------------
    
    def is_spread_acceptable(self, bid: float, ask: float) -> bool:
        """
        Check if the bid-ask spread is acceptable for entry.
        
        Args:
            bid: Current bid price
            ask: Current ask price
            
        Returns:
            Boolean indicating if spread is acceptable
        """
        # TODO: check spread
        max_spread = self.parameters.get("max_spread_percent", 0.0005)
        mid_price = (bid + ask) / 2
        spread_percent = (ask - bid) / mid_price
        
        return spread_percent <= max_spread
    
    def can_reenter_range_boundary(
        self, symbol: str, boundary: str, current_time: datetime
    ) -> bool:
        """
        Check if a range boundary can be re-entered.
        
        Args:
            symbol: Symbol to check
            boundary: Range boundary ('upper' or 'lower')
            current_time: Current timestamp
            
        Returns:
            Boolean indicating if range boundary can be re-entered
        """
        # TODO: apply range re-entry check
        if symbol in self.range_entries and boundary in self.range_entries[symbol]:
            last_entry = self.range_entries[symbol][boundary]
            
            # Don't re-enter unless range has reset (at least 1 day for multi-day)
            return (current_time - last_entry).days >= 1
            
        return True
    
    def is_volatility_suitable(self, close: float, atr: float) -> bool:
        """
        Check if volatility is suitable for range trading.
        
        Args:
            close: Current close price
            atr: Current ATR value
            
        Returns:
            Boolean indicating if volatility is suitable
        """
        # TODO: check volatility suitability
        max_atr_percent = self.parameters.get("max_atr_percent", 0.02)
        atr_percent = atr / close
        
        # ATR should be below threshold (avoid trending instruments)
        return atr_percent <= max_atr_percent
    
    def should_avoid_due_to_news(
        self, symbol: str, current_time: datetime, news_events: Dict[str, List[datetime]]
    ) -> bool:
        """
        Check if trading should be avoided due to upcoming news.
        
        Args:
            symbol: Symbol to check
            current_time: Current timestamp
            news_events: Dictionary mapping symbols to lists of news event timestamps
            
        Returns:
            Boolean indicating if trading should be avoided
        """
        # TODO: apply news filter
        if not self.parameters.get("news_filter_enabled", True):
            return False
            
        if symbol in news_events:
            for event_time in news_events[symbol]:
                # Avoid trading if news is within 24 hours
                time_diff = abs((event_time - current_time).total_seconds() / 3600)
                if time_diff <= 24:
                    return True
                    
        return False
    
    # --------------------------------------------------------
    # 10. Continuous Optimization
    # --------------------------------------------------------
    
    def should_adjust_donchian_length(
        self, current_atr: float, median_atr: float
    ) -> bool:
        """
        Check if Donchian channel length should be adjusted based on volatility.
        
        Args:
            current_atr: Current ATR value
            median_atr: Median ATR over long term
            
        Returns:
            Boolean indicating if channel length should be adjusted
        """
        # TODO: implement volatility-adaptive channels
        adjustment_threshold = self.parameters.get("volatility_adjustment_threshold", 0.2)
        deviation = abs(current_atr / median_atr - 1)
        
        return deviation > adjustment_threshold
    
    def get_adjusted_donchian_length(
        self, current_atr: float, median_atr: float, current_length: int
    ) -> int:
        """
        Get adjusted Donchian channel length based on volatility.
        
        Args:
            current_atr: Current ATR value
            median_atr: Median ATR over long term
            current_length: Current Donchian length
            
        Returns:
            Adjusted Donchian length
        """
        # TODO: implement donchian length adjustment
        # If current ATR is higher, use shorter period
        if current_atr > median_atr * 1.2:
            return max(10, current_length - 5)
        # If current ATR is lower, use longer period
        elif current_atr < median_atr * 0.8:
            return min(30, current_length + 5)
            
        return current_length  # No change needed
    
    # --------------------------------------------------------
    # Main Signal Generation
    # --------------------------------------------------------
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, Any]:
        """
        Calculate all indicators needed for range trading signals.
        
        Args:
            data: Dictionary mapping timeframes to DataFrames with OHLCV data
            symbol: Symbol to calculate indicators for
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Get parameters
        donchian_period = self.parameters.get("donchian_period", 20)
        rsi_period = self.parameters.get("rsi_period", 14)
        stoch_k_period = self.parameters.get("stoch_k_period", 14)
        stoch_d_period = self.parameters.get("stoch_d_period", 3)
        stoch_slowing = self.parameters.get("stoch_slowing", 3)
        volume_ma_period = self.parameters.get("volume_ma_period", 20)
        atr_period = self.parameters.get("atr_period", 14)
        
        # Get the appropriate signal and context timeframes
        is_intraday = self.parameters.get("timeframe_mode", "intraday") == "intraday"
        signal_tf = self.parameters.get(
            "signal_timeframe_intraday" if is_intraday else "signal_timeframe_multi_day", 
            "15min" if is_intraday else "1d"
        )
        context_tf = self.parameters.get(
            "context_timeframe_intraday" if is_intraday else "context_timeframe_multi_day",
            "4h" if is_intraday else "1w"
        )
        
        # Check if we have data for both timeframes
        if signal_tf not in data or context_tf not in data:
            return {}
        
        signal_data = data[signal_tf]
        context_data = data[context_tf]
        
        # Calculate indicators on signal timeframe
        upper_band, middle_band, lower_band = self.calculate_donchian_channels(
            signal_data["high"], signal_data["low"], donchian_period
        )
        indicators["donchian_upper"] = upper_band
        indicators["donchian_middle"] = middle_band
        indicators["donchian_lower"] = lower_band
        
        indicators["rsi"] = self.calculate_rsi(signal_data["close"], rsi_period)
        
        stoch_k, stoch_d = self.calculate_stochastic(
            signal_data["high"], signal_data["low"], signal_data["close"],
            stoch_k_period, stoch_d_period, stoch_slowing
        )
        indicators["stoch_k"] = stoch_k
        indicators["stoch_d"] = stoch_d
        
        if "volume" in signal_data.columns:
            indicators["volume_ratio"] = self.calculate_volume_surge(signal_data["volume"], volume_ma_period)
        
        indicators["atr"] = self.calculate_atr(
            signal_data["high"], signal_data["low"], signal_data["close"], atr_period
        )
        
        # Calculate context indicators (higher timeframe)
        htf_upper, htf_middle, htf_lower = self.calculate_donchian_channels(
            context_data["high"], context_data["low"], donchian_period
        )
        indicators["htf_donchian_upper"] = htf_upper
        indicators["htf_donchian_middle"] = htf_middle
        indicators["htf_donchian_lower"] = htf_lower
        
        # Calculate VWAP if intraday
        if is_intraday and "volume" in signal_data.columns:
            indicators["vwap"] = self.calculate_vwap(signal_data)
        
        # Check for adaptive channel length
        if "atr" in indicators and len(indicators["atr"]) > 100:
            median_atr = indicators["atr"].rolling(window=100).median().iloc[-1]
            indicators["median_atr"] = median_atr
        
        return indicators
    
    def generate_signals(
        self, 
        data: Dict[str, Dict[str, pd.DataFrame]], 
        equity: float,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        news_events: Optional[Dict[str, List[datetime]]] = None,
        symbol_sectors: Optional[Dict[str, str]] = None,
        order_book: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Signal]:
        """
        Generate range trading signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            equity: Current equity value
            market_data: Optional market-wide data
            news_events: Optional Dictionary mapping symbols to lists of news event timestamps
            symbol_sectors: Optional Dictionary mapping symbols to sector classifications
            order_book: Optional order book data for spread calculation
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Check if we can add new positions
        if not self.can_add_new_position(equity, self.current_exposure):
            logger.info("Skipping range signals due to exposure restrictions")
            return {}
        
        # Get current time
        current_time = datetime.now()
        
        # Check if trading is allowed after consecutive losses
        is_intraday = self.parameters.get("timeframe_mode", "intraday") == "intraday"
        if not self.can_trade_after_losses(current_time, is_intraday):
            logger.info("Skipping range signals due to consecutive loss pause")
            return {}
        
        # Generate signals
        signals = {}
        
        # Get signal timeframe
        signal_tf = self.parameters.get(
            "signal_timeframe_intraday" if is_intraday else "signal_timeframe_multi_day", 
            "15min" if is_intraday else "1d"
        )
        
        for symbol, timeframe_data in data.items():
            try:
                # Check news filter
                if news_events and self.should_avoid_due_to_news(symbol, current_time, news_events):
                    continue
                
                # Check sector position limit
                if symbol_sectors and symbol in symbol_sectors:
                    sector = symbol_sectors[symbol]
                    if not self.can_add_sector_position(sector, self.sector_positions):
                        continue
                
                # Calculate indicators
                indicators = self.calculate_indicators(timeframe_data, symbol)
                if not indicators:
                    continue
                
                # Get latest data
                signal_data = timeframe_data[signal_tf]
                if len(signal_data) < 2:
                    continue
                
                latest_bar = signal_data.iloc[-1]
                latest_close = latest_bar["close"]
                latest_timestamp = signal_data.index[-1] if isinstance(signal_data.index[-1], datetime) else current_time
                
                # Check spread if order book is available
                if order_book and symbol in order_book:
                    bid = order_book[symbol].get("bid", latest_close * 0.9999)
                    ask = order_book[symbol].get("ask", latest_close * 1.0001)
                    if not self.is_spread_acceptable(bid, ask):
                        continue
                
                # Get latest indicator values
                latest_upper = indicators["donchian_upper"].iloc[-1]
                latest_middle = indicators["donchian_middle"].iloc[-1]
                latest_lower = indicators["donchian_lower"].iloc[-1]
                
                latest_rsi = indicators["rsi"].iloc[-1]
                latest_stoch_k = indicators["stoch_k"].iloc[-1]
                latest_stoch_d = indicators["stoch_d"].iloc[-1]
                
                latest_atr = indicators["atr"].iloc[-1]
                
                # Check if volatility is suitable for range trading
                if not self.is_volatility_suitable(latest_close, latest_atr):
                    continue
                
                # Get higher timeframe context
                htf_upper = indicators["htf_donchian_upper"].iloc[-1]
                htf_lower = indicators["htf_donchian_lower"].iloc[-1]
                
                # Check volume confirmation if available
                volume_confirmed = True
                if "volume_ratio" in indicators:
                    latest_volume_ratio = indicators["volume_ratio"].iloc[-1]
                    volume_confirmed = self.has_volume_confirmation(latest_volume_ratio)
                
                # Signal variables
                signal_type = None
                boundary = None
                confidence = 0.0
                
                # Check long setup (at support)
                if (self.is_at_support(latest_close, latest_lower, latest_atr) and
                    self.is_oversold(latest_rsi, latest_stoch_k, latest_stoch_d) and
                    volume_confirmed and
                    self.is_range_context(latest_close, htf_upper, htf_lower)):
                    
                    # Check if we can re-enter this boundary
                    boundary = "lower"
                    if self.can_reenter_range_boundary(symbol, boundary, latest_timestamp):
                        signal_type = SignalType.BUY
                        
                        # Calculate confidence based on indicator strengths
                        price_proximity = 1.0 - abs(latest_close - latest_lower) / latest_atr
                        rsi_strength = (35 - latest_rsi) / 35 if latest_rsi < 35 else 0
                        stoch_strength = (20 - latest_stoch_k) / 20 if latest_stoch_k < 20 else 0
                        
                        # Higher confidence near the channel boundary with strong oscillator readings
                        confidence = min(0.95, 0.5 + (0.3 * price_proximity) + (0.1 * rsi_strength) + (0.1 * stoch_strength))
                
                # Check short setup (at resistance)
                elif (self.is_at_resistance(latest_close, latest_upper, latest_atr) and
                      self.is_overbought(latest_rsi, latest_stoch_k, latest_stoch_d) and
                      volume_confirmed and
                      self.is_range_context(latest_close, htf_upper, htf_lower)):
                    
                    # Check if we can re-enter this boundary
                    boundary = "upper"
                    if self.can_reenter_range_boundary(symbol, boundary, latest_timestamp):
                        signal_type = SignalType.SELL
                        
                        # Calculate confidence based on indicator strengths
                        price_proximity = 1.0 - abs(latest_close - latest_upper) / latest_atr
                        rsi_strength = (latest_rsi - 65) / 35 if latest_rsi > 65 else 0
                        stoch_strength = (latest_stoch_k - 80) / 20 if latest_stoch_k > 80 else 0
                        
                        # Higher confidence near the channel boundary with strong oscillator readings
                        confidence = min(0.95, 0.5 + (0.3 * price_proximity) + (0.1 * rsi_strength) + (0.1 * stoch_strength))
                
                # Generate signal if we have a valid signal type
                if signal_type:
                    # Calculate stop loss
                    entry_band = latest_lower if signal_type == SignalType.BUY else latest_upper
                    stop_price = self.calculate_stop_loss(latest_close, entry_band, latest_atr, signal_type)
                    
                    # Calculate profit target
                    opposite_band = latest_upper if signal_type == SignalType.BUY else latest_lower
                    target_price = self.calculate_profit_target(
                        latest_close, latest_middle, opposite_band, latest_atr, signal_type
                    )
                    
                    # Handle tuple return for scaled exits
                    if isinstance(target_price, tuple):
                        take_profit = target_price[0]  # Use first target for signal
                    else:
                        take_profit = target_price
                    
                    # Calculate position size
                    size = self.calculate_position_size(equity, latest_close, stop_price, is_intraday)
                    
                    # Record range entry
                    if symbol not in self.range_entries:
                        self.range_entries[symbol] = {}
                    self.range_entries[symbol][boundary] = latest_timestamp
                    
                    # Create signal
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=latest_close,
                        timestamp=latest_timestamp,
                        confidence=confidence,
                        stop_loss=stop_price,
                        take_profit=take_profit,
                        size=size,
                        metadata={
                            "strategy_type": "range_trading",
                            "timeframe_mode": "intraday" if is_intraday else "multi_day",
                            "signal_timeframe": signal_tf,
                            "donchian_upper": latest_upper,
                            "donchian_middle": latest_middle,
                            "donchian_lower": latest_lower,
                            "rsi": latest_rsi,
                            "stoch_k": latest_stoch_k,
                            "stoch_d": latest_stoch_d,
                            "atr": latest_atr,
                            "range_boundary": boundary,
                            "use_limit_order": self.parameters.get("use_limit_orders", True),
                            "scale_out": self.parameters.get("scale_out_enabled", False),
                            "scale_targets": target_price if isinstance(target_price, tuple) else None
                        }
                    )
                    
                    # Update sector positions if sector information is available
                    if symbol_sectors and symbol in symbol_sectors:
                        sector = symbol_sectors[symbol]
                        self.sector_positions[sector] = self.sector_positions.get(sector, 0) + 1
                
            except Exception as e:
                logger.error(f"Error generating range signal for {symbol}: {e}")
        
        return signals
    
    def update_state_after_exit(
        self, symbol: str, exit_price: float, entry_price: float, 
        signal_type: SignalType, sector: Optional[str] = None
    ) -> None:
        """
        Update strategy state after position exit.
        
        Args:
            symbol: Symbol of the exited position
            exit_price: Exit price
            entry_price: Entry price
            signal_type: Type of signal (BUY or SELL)
            sector: Optional sector of the symbol
        """
        # Check if exit was a loss
        is_loss = (signal_type == SignalType.BUY and exit_price < entry_price) or (signal_type == SignalType.SELL and exit_price > entry_price)
        
        if is_loss:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
        else:
            self.consecutive_losses = 0
        
        # Update sector positions
        if sector and sector in self.sector_positions and self.sector_positions[sector] > 0:
            self.sector_positions[sector] -= 1
    
    def reset_session_state(self) -> None:
        """
        Reset strategy state at the start of a new trading session.
        """
        # Reset Donchian channels at market open if enabled
        if self.parameters.get("reset_donchian_at_open", True) and self.parameters.get("timeframe_mode", "intraday") == "intraday":
            # Clear range entries to allow fresh setups
            self.range_entries = {}
            
        # Reset consecutive losses for intraday mode
        if self.parameters.get("timeframe_mode", "intraday") == "intraday":
            self.consecutive_losses = 0 