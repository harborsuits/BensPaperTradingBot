#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum Trading Strategy Module

This module implements a strategy to capture strong directional moves when price
accelerates beyond normal volatility, entering early in trend extensions and
exiting before exhaustion.
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

class MomentumTradingStrategy(StrategyOptimizable):
    """
    Momentum Trading Strategy designed to ride strong directional thrusts.
    
    This strategy identifies and trades strong momentum moves by focusing on
    high-conviction price accelerations confirmed by multiple indicators.
    It operates in both intraday and swing timeframes, aligning entries with
    the higher timeframe trend direction.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Momentum Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # Highly liquid large-caps, sector ETFs, momentum leaders
            "timeframe_mode": "intraday",  # 'intraday' or 'swing'
            "signal_timeframe_intraday": "15min",  # or '1h'
            "signal_timeframe_swing": "1d",
            "trend_timeframe_intraday": "4h",
            "trend_timeframe_swing": "1w",
            "max_positions_per_timeframe": 3,
            
            # Core Indicators
            "roc_period": 12,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "volume_ma_period": 20,
            "ema_trend_period": 50,
            
            # Entry Criteria
            "roc_threshold_percent": 2.0,  # Minimum ROC for entry
            "volume_multiplier": 1.7,  # Volume vs 20-day average
            "adx_threshold": 25,  # Minimum ADX for entry
            "price_breakout_bars": 3,  # Bars for new high/low breakout
            
            # Exit Criteria
            "profit_target_roc_multiplier": 1.5,  # Target as multiple of entry ROC
            "fixed_rr_ratio": 2.0,  # Fixed risk-reward ratio
            "stop_loss_atr_multiple": 1.0,  # Stop loss as ATR multiple
            "trailing_stop_activation_atr": 1.0,  # When to activate trailing stop
            "trailing_stop_atr_multiple": 0.75,  # Trailing stop as ATR multiple
            "max_bars_without_progress": 3,  # Exit if no new highs/lows
            
            # Position Sizing & Risk Controls
            "risk_percent_intraday": 0.005,  # 0.5% for intraday
            "risk_percent_swing": 0.01,  # 1% for swing
            "max_exposure_percent": 0.15,  # 15% max exposure
            "max_consecutive_losses": 2,  # Halt after 2 consecutive losses
            
            # Order Execution
            "entry_order_type": "market",  # 'market' or 'limit'
            "slippage_percent": 0.0002,  # 0.02% slippage allowance
            "reentry_bar_delay": 3,  # Bars to wait for re-entry after stop-out
            
            # Operational Rules
            "news_filter_minutes": 5,  # Avoid entries near news (minutes)
            "max_positions_per_sector": 2,  # Maximum positions per sector
            "vix_min": 12,  # Minimum VIX for momentum trading
            "vix_max": 30,  # Maximum VIX for momentum trading
            
            # Continuous Optimization
            "atr_deviation_threshold": 0.2,  # ATR deviation from median for timeframe shift
            "use_ml_overlay": False  # Whether to use ML overlay for entry refinement
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize state variables
        self.consecutive_losses = 0
        self.stopped_symbols = {}  # Track stopped symbols and their last stop bar
        self.current_exposure = 0.0
        self.sector_positions = {}  # Track positions per sector
        
        logger.info(f"Initialized Momentum Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "roc_period": [8, 10, 12, 14],
            "roc_threshold_percent": [1.0, 1.5, 2.0, 2.5, 3.0],
            "macd_fast": [8, 10, 12, 14],
            "macd_slow": [20, 24, 26, 30],
            "macd_signal": [7, 9, 11],
            "adx_period": [10, 14, 20],
            "adx_threshold": [20, 25, 30, 35],
            "volume_ma_period": [10, 15, 20, 25],
            "volume_multiplier": [1.5, 1.7, 2.0, 2.5],
            "ema_trend_period": [30, 40, 50, 60],
            "profit_target_roc_multiplier": [1.0, 1.5, 2.0, 2.5],
            "fixed_rr_ratio": [1.5, 2.0, 2.5, 3.0],
            "stop_loss_atr_multiple": [0.75, 1.0, 1.25, 1.5],
            "trailing_stop_activation_atr": [0.75, 1.0, 1.25],
            "trailing_stop_atr_multiple": [0.5, 0.75, 1.0],
            "max_bars_without_progress": [2, 3, 4, 5],
            "risk_percent_intraday": [0.003, 0.005, 0.007],
            "risk_percent_swing": [0.007, 0.01, 0.015],
            "max_exposure_percent": [0.1, 0.15, 0.2, 0.25],
            "max_consecutive_losses": [1, 2, 3],
            "reentry_bar_delay": [2, 3, 4, 5],
            "max_positions_per_sector": [1, 2, 3],
            "vix_min": [10, 12, 15],
            "vix_max": [25, 30, 35]
        }
    
    # --------------------------------------------------------
    # 3. Core Indicators
    # --------------------------------------------------------
    
    def calculate_roc(self, prices: pd.Series, period: int = 12) -> pd.Series:
        """
        Calculate Rate of Change (ROC) to measure price acceleration.
        
        Args:
            prices: Series of price data
            period: ROC period
            
        Returns:
            Series with ROC values as percentages
        """
        # TODO: compute ROC
        return (prices / prices.shift(period) - 1) * 100
    
    def calculate_macd(
        self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator for momentum confirmation.
        
        Args:
            prices: Series of price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # TODO: compute MACD components
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) to measure trend strength.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        # TODO: compute ADX
        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff(-1).abs()
        
        # Adjust +DM and -DM
        plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
        minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
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
    
    def calculate_ema(self, prices: pd.Series, period: int = 50) -> pd.Series:
        """
        Calculate Exponential Moving Average for trend filter.
        
        Args:
            prices: Series of price data
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        # TODO: compute EMA
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for stop placement and volatility estimation.
        
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
    
    # --------------------------------------------------------
    # 4. Entry Criteria
    # --------------------------------------------------------
    
    def is_aligned_with_trend(
        self, signal_close: float, trend_ema: float, signal_type: SignalType
    ) -> bool:
        """
        Check if the signal is aligned with the higher timeframe trend.
        
        Args:
            signal_close: Current close price on signal timeframe
            trend_ema: EMA value from the higher trend timeframe
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if signal aligns with trend
        """
        # TODO: apply trend alignment filter
        if signal_type == SignalType.BUY:
            return signal_close > trend_ema
        else:
            return signal_close < trend_ema
    
    def has_momentum_burst(
        self, roc: float, macd_line: float, macd_hist: float, signal_type: SignalType
    ) -> bool:
        """
        Check if there is a momentum burst based on ROC and MACD.
        
        Args:
            roc: Current ROC value
            macd_line: Current MACD line value
            macd_hist: Current MACD histogram value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if there is a momentum burst
        """
        # TODO: apply momentum burst filter
        roc_threshold = self.parameters.get("roc_threshold_percent", 2.0)
        
        if signal_type == SignalType.BUY:
            # For long, need positive ROC above threshold OR MACD crossover above zero
            return (roc >= roc_threshold) or (macd_line > 0 and macd_hist > 0)
        else:
            # For short, need negative ROC below negative threshold OR MACD crossover below zero
            return (roc <= -roc_threshold) or (macd_line < 0 and macd_hist < 0)
    
    def has_volume_confirmation(self, volume_ratio: float) -> bool:
        """
        Check if there is sufficient volume to confirm the signal.
        
        Args:
            volume_ratio: Current volume ratio vs MA
            
        Returns:
            Boolean indicating if volume confirms
        """
        # TODO: apply volume confirmation filter
        volume_threshold = self.parameters.get("volume_multiplier", 1.7)
        return volume_ratio >= volume_threshold
    
    def is_adx_strong_enough(self, adx: float) -> bool:
        """
        Check if ADX is strong enough to confirm a trend.
        
        Args:
            adx: Current ADX value
            
        Returns:
            Boolean indicating if ADX confirms trend strength
        """
        # TODO: apply ADX filter
        adx_threshold = self.parameters.get("adx_threshold", 25)
        return adx >= adx_threshold
    
    def is_price_breakout(
        self, prices: pd.Series, signal_type: SignalType, bars: int = 3
    ) -> bool:
        """
        Check if price is making a new high/low over the specified lookback.
        
        Args:
            prices: Series of price data
            signal_type: Type of signal (BUY or SELL)
            bars: Number of bars to check for breakout
            
        Returns:
            Boolean indicating if there's a breakout
        """
        # TODO: apply price breakout filter
        if signal_type == SignalType.BUY:
            # For long, check if current price is higher than previous 'bars' highs
            return prices.iloc[-1] > prices.iloc[-bars-1:-1].max()
        else:
            # For short, check if current price is lower than previous 'bars' lows
            return prices.iloc[-1] < prices.iloc[-bars-1:-1].min()
    
    # --------------------------------------------------------
    # 5. Exit Criteria
    # --------------------------------------------------------
    
    def calculate_profit_target(
        self, entry_price: float, roc_value: float, atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate profit target based on ROC value and/or fixed R:R.
        
        Args:
            entry_price: Entry price
            roc_value: ROC value at entry
            atr: ATR value at entry
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Profit target price
        """
        # TODO: compute profit target
        roc_multiplier = self.parameters.get("profit_target_roc_multiplier", 1.5)
        fixed_rr = self.parameters.get("fixed_rr_ratio", 2.0)
        stop_loss_atr = self.parameters.get("stop_loss_atr_multiple", 1.0)
        
        # Target based on ROC
        roc_based_target = entry_price * (1 + roc_value * roc_multiplier / 100) if signal_type == SignalType.BUY else entry_price * (1 - roc_value * roc_multiplier / 100)
        
        # Target based on fixed R:R
        stop_distance = atr * stop_loss_atr
        rr_based_target = entry_price + (fixed_rr * stop_distance) if signal_type == SignalType.BUY else entry_price - (fixed_rr * stop_distance)
        
        # Use the more conservative target
        if signal_type == SignalType.BUY:
            return min(roc_based_target, rr_based_target)
        else:
            return max(roc_based_target, rr_based_target)
    
    def calculate_stop_loss(
        self, entry_price: float, atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate initial stop loss based on ATR.
        
        Args:
            entry_price: Entry price
            atr: ATR value at entry
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Stop loss price
        """
        # TODO: compute stop loss
        stop_multiple = self.parameters.get("stop_loss_atr_multiple", 1.0)
        stop_distance = atr * stop_multiple
        
        if signal_type == SignalType.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_trailing_stop(
        self, current_price: float, highest_price: float, lowest_price: float, 
        atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate trailing stop once position has moved in favor.
        
        Args:
            current_price: Current price
            highest_price: Highest price since entry
            lowest_price: Lowest price since entry
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Trailing stop price
        """
        # TODO: compute trailing stop
        trailing_multiple = self.parameters.get("trailing_stop_atr_multiple", 0.75)
        trail_distance = atr * trailing_multiple
        
        if signal_type == SignalType.BUY:
            return highest_price - trail_distance
        else:
            return lowest_price + trail_distance
    
    def should_exit_by_time(
        self, entry_time: datetime, current_time: datetime, 
        is_intraday: bool, market_close_time: Optional[time] = None
    ) -> bool:
        """
        Check if position should be exited based on time criteria.
        
        Args:
            entry_time: Entry timestamp
            current_time: Current timestamp
            is_intraday: Whether this is an intraday position
            market_close_time: Market close time for intraday exits
            
        Returns:
            Boolean indicating if time-based exit should be triggered
        """
        # TODO: evaluate time-based exit
        if is_intraday and market_close_time:
            # Exit intraday positions before market close
            return current_time.time() >= market_close_time
        else:
            # For swing positions, the time-based exit logic will be handled by the 
            # "no new highs/lows" logic in should_exit_by_progress
            return False
    
    def should_exit_by_progress(
        self, prices: pd.Series, entry_price: float, signal_type: SignalType
    ) -> bool:
        """
        Check if position should be exited due to lack of progress.
        
        Args:
            prices: Series of recent price data
            entry_price: Entry price
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if progress-based exit should be triggered
        """
        # TODO: evaluate progress-based exit
        max_bars = self.parameters.get("max_bars_without_progress", 3)
        
        if len(prices) < max_bars + 1:
            return False
            
        if signal_type == SignalType.BUY:
            # For longs, check if no new highs have been made in the last max_bars
            return prices.iloc[-max_bars:].max() <= prices.iloc[-max_bars-1]
        else:
            # For shorts, check if no new lows have been made in the last max_bars
            return prices.iloc[-max_bars:].min() >= prices.iloc[-max_bars-1]
    
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
            is_intraday: Whether this is an intraday position
            
        Returns:
            Position size
        """
        # TODO: compute position size
        if is_intraday:
            risk_percent = self.parameters.get("risk_percent_intraday", 0.005)
        else:
            risk_percent = self.parameters.get("risk_percent_swing", 0.01)
            
        risk_amount = equity * risk_percent
        stop_distance = abs(entry_price - stop_price)
        
        # Position size formula: size = (equity × risk%) / stop_distance
        position_size = risk_amount / stop_distance if stop_distance > 0 else 0
        
        return position_size
    
    def can_add_new_position(
        self, equity: float, current_exposure: float, consecutive_losses: int
    ) -> bool:
        """
        Check if a new position can be added based on exposure and loss limits.
        
        Args:
            equity: Total equity
            current_exposure: Current exposure amount
            consecutive_losses: Number of consecutive losses
            
        Returns:
            Boolean indicating if a new position can be added
        """
        # TODO: apply risk control checks
        max_exposure = self.parameters.get("max_exposure_percent", 0.15) * equity
        max_losses = self.parameters.get("max_consecutive_losses", 2)
        
        return (current_exposure < max_exposure) and (consecutive_losses < max_losses)
    
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
        max_sector_positions = self.parameters.get("max_positions_per_sector", 2)
        current_count = current_sector_positions.get(sector, 0)
        
        return current_count < max_sector_positions
    
    # --------------------------------------------------------
    # 7 & 8. Order Execution & Operational Rules
    # --------------------------------------------------------
    
    def can_trade_in_current_vix_regime(self, vix_value: float) -> bool:
        """
        Check if momentum trading is appropriate in the current VIX regime.
        
        Args:
            vix_value: Current VIX index value
            
        Returns:
            Boolean indicating if trading is appropriate
        """
        # TODO: apply VIX regime filter
        vix_min = self.parameters.get("vix_min", 12)
        vix_max = self.parameters.get("vix_max", 30)
        
        return vix_min <= vix_value <= vix_max
    
    def should_avoid_due_to_news(
        self, current_time: datetime, news_events: List[datetime], symbol: str
    ) -> bool:
        """
        Check if trading should be avoided due to upcoming or recent news.
        
        Args:
            current_time: Current timestamp
            news_events: List of news event timestamps
            symbol: Symbol to check
            
        Returns:
            Boolean indicating if trading should be avoided
        """
        # TODO: apply news filter
        buffer_minutes = self.parameters.get("news_filter_minutes", 5)
        
        for event_time in news_events:
            time_diff = abs((current_time - event_time).total_seconds() / 60)
            if time_diff <= buffer_minutes:
                return True
                
        return False
    
    def can_reenter_symbol(self, symbol: str, current_bar: int) -> bool:
        """
        Check if a symbol can be re-entered after a stop-out.
        
        Args:
            symbol: Symbol to check
            current_bar: Current bar index
            
        Returns:
            Boolean indicating if symbol can be re-entered
        """
        # TODO: apply re-entry filter
        reentry_delay = self.parameters.get("reentry_bar_delay", 3)
        
        if symbol in self.stopped_symbols:
            last_stop_bar = self.stopped_symbols[symbol]
            return (current_bar - last_stop_bar) >= reentry_delay
            
        return True
    
    # --------------------------------------------------------
    # 10. Continuous Optimization
    # --------------------------------------------------------
    
    def should_adapt_timeframe(self, current_atr: float, median_atr: float) -> bool:
        """
        Check if signal timeframe should be adapted based on ATR deviation.
        
        Args:
            current_atr: Current ATR value
            median_atr: Median ATR over long term
            
        Returns:
            Boolean indicating if timeframe should be adapted
        """
        # TODO: implement adaptive timeframe logic
        atr_deviation_threshold = self.parameters.get("atr_deviation_threshold", 0.2)
        deviation = abs(current_atr / median_atr - 1)
        
        return deviation > atr_deviation_threshold
    
    def get_adapted_timeframe(self, current_timeframe: str, current_atr: float, median_atr: float) -> str:
        """
        Get adapted timeframe based on volatility conditions.
        
        Args:
            current_timeframe: Current signal timeframe
            current_atr: Current ATR value
            median_atr: Median ATR over long term
            
        Returns:
            Adapted timeframe
        """
        # TODO: implement timeframe adaptation
        if current_timeframe == "15min" and current_atr > median_atr * 1.2:
            return "5min"  # Shift to faster timeframe for higher volatility
        elif current_timeframe == "15min" and current_atr < median_atr * 0.8:
            return "30min"  # Shift to slower timeframe for lower volatility
        elif current_timeframe == "1h" and current_atr > median_atr * 1.2:
            return "15min"
        elif current_timeframe == "1h" and current_atr < median_atr * 0.8:
            return "4h"
        elif current_timeframe == "1d" and current_atr > median_atr * 1.2:
            return "4h"
        elif current_timeframe == "1d" and current_atr < median_atr * 0.8:
            return "1w"
            
        return current_timeframe  # No change needed
    
    # --------------------------------------------------------
    # Main Signal Generation
    # --------------------------------------------------------
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, pd.Series]:
        """
        Calculate all indicators needed for momentum signals.
        
        Args:
            data: Dictionary mapping timeframes to DataFrames with OHLCV data
            symbol: Symbol to calculate indicators for
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Get parameters
        roc_period = self.parameters.get("roc_period", 12)
        macd_fast = self.parameters.get("macd_fast", 12)
        macd_slow = self.parameters.get("macd_slow", 26)
        macd_signal = self.parameters.get("macd_signal", 9)
        adx_period = self.parameters.get("adx_period", 14)
        volume_ma_period = self.parameters.get("volume_ma_period", 20)
        ema_trend_period = self.parameters.get("ema_trend_period", 50)
        
        # Get the appropriate signal and trend timeframes
        is_intraday = self.parameters.get("timeframe_mode", "intraday") == "intraday"
        signal_tf = self.parameters.get(
            "signal_timeframe_intraday" if is_intraday else "signal_timeframe_swing", 
            "15min" if is_intraday else "1d"
        )
        trend_tf = self.parameters.get(
            "trend_timeframe_intraday" if is_intraday else "trend_timeframe_swing",
            "4h" if is_intraday else "1w"
        )
        
        # Check if we have data for both timeframes
        if signal_tf not in data or trend_tf not in data:
            return {}
        
        signal_data = data[signal_tf]
        trend_data = data[trend_tf]
        
        # Calculate indicators on signal timeframe
        indicators["roc"] = self.calculate_roc(signal_data["close"], roc_period)
        
        macd_line, macd_signal_line, macd_hist = self.calculate_macd(
            signal_data["close"], macd_fast, macd_slow, macd_signal
        )
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = macd_signal_line
        indicators["macd_hist"] = macd_hist
        
        indicators["adx"] = self.calculate_adx(
            signal_data["high"], signal_data["low"], signal_data["close"], adx_period
        )
        
        indicators["volume_ratio"] = self.calculate_volume_surge(signal_data["volume"], volume_ma_period)
        
        indicators["atr"] = self.calculate_atr(
            signal_data["high"], signal_data["low"], signal_data["close"]
        )
        
        # Calculate trend filter on trend timeframe
        indicators["trend_ema"] = self.calculate_ema(trend_data["close"], ema_trend_period)
        
        # Adapted timeframe calculation
        if len(indicators["atr"]) > 100:  # Need enough data to calculate median
            indicators["median_atr"] = indicators["atr"].rolling(window=100).median().iloc[-1]
        
        return indicators
    
    def generate_signals(
        self, 
        data: Dict[str, Dict[str, pd.DataFrame]], 
        equity: float,
        vix_data: Optional[pd.DataFrame] = None,
        news_events: Optional[Dict[str, List[datetime]]] = None,
        symbol_sectors: Optional[Dict[str, str]] = None,
        current_bar_idx: Optional[int] = None
    ) -> Dict[str, Signal]:
        """
        Generate momentum trading signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            equity: Current equity value
            vix_data: Optional DataFrame with VIX data
            news_events: Optional Dictionary mapping symbols to lists of news event timestamps
            symbol_sectors: Optional Dictionary mapping symbols to sector classifications
            current_bar_idx: Optional current bar index for re-entry check
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Check if VIX regime is appropriate for momentum trading
        if vix_data is not None and not self.can_trade_in_current_vix_regime(vix_data.iloc[-1]["close"]):
            logger.info("Skipping momentum signals due to inappropriate VIX regime")
            return {}
        
        # Check if risk controls allow new positions
        if not self.can_add_new_position(equity, self.current_exposure, self.consecutive_losses):
            logger.info("Skipping momentum signals due to risk control restrictions")
            return {}
        
        # Generate signals
        signals = {}
        
        # Get mode (intraday or swing)
        is_intraday = self.parameters.get("timeframe_mode", "intraday") == "intraday"
        signal_tf = self.parameters.get(
            "signal_timeframe_intraday" if is_intraday else "signal_timeframe_swing", 
            "15min" if is_intraday else "1d"
        )
        
        for symbol, timeframe_data in data.items():
            try:
                # Check if we can re-enter this symbol
                if current_bar_idx is not None and not self.can_reenter_symbol(symbol, current_bar_idx):
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
                
                latest_close = signal_data["close"].iloc[-1]
                latest_timestamp = signal_data.index[-1] if isinstance(signal_data.index[-1], datetime) else datetime.now()
                
                # Check if we should avoid trading due to news
                if news_events and symbol in news_events and self.should_avoid_due_to_news(
                    latest_timestamp, news_events[symbol], symbol
                ):
                    continue
                
                # Get latest indicator values
                roc_value = indicators["roc"].iloc[-1]
                macd_line_value = indicators["macd_line"].iloc[-1]
                macd_hist_value = indicators["macd_hist"].iloc[-1]
                adx_value = indicators["adx"].iloc[-1]
                volume_ratio_value = indicators["volume_ratio"].iloc[-1]
                latest_atr = indicators["atr"].iloc[-1]
                
                # Interpolate trend EMA to the signal timeframe
                trend_ema_value = indicators["trend_ema"].reindex(signal_data.index, method="ffill").iloc[-1]
                
                # Check long signal conditions
                long_conditions = (
                    self.is_aligned_with_trend(latest_close, trend_ema_value, SignalType.BUY) and
                    self.has_momentum_burst(roc_value, macd_line_value, macd_hist_value, SignalType.BUY) and
                    self.has_volume_confirmation(volume_ratio_value) and
                    self.is_adx_strong_enough(adx_value) and
                    self.is_price_breakout(signal_data["close"], SignalType.BUY, self.parameters.get("price_breakout_bars", 3))
                )
                
                # Check short signal conditions
                short_conditions = (
                    self.is_aligned_with_trend(latest_close, trend_ema_value, SignalType.SELL) and
                    self.has_momentum_burst(roc_value, macd_line_value, macd_hist_value, SignalType.SELL) and
                    self.has_volume_confirmation(volume_ratio_value) and
                    self.is_adx_strong_enough(adx_value) and
                    self.is_price_breakout(signal_data["close"], SignalType.SELL, self.parameters.get("price_breakout_bars", 3))
                )
                
                # Generate signal based on conditions
                signal_type = None
                if long_conditions:
                    signal_type = SignalType.BUY
                elif short_conditions:
                    signal_type = SignalType.SELL
                
                if signal_type:
                    # Calculate stop loss
                    stop_price = self.calculate_stop_loss(latest_close, latest_atr, signal_type)
                    
                    # Calculate profit target
                    target_price = self.calculate_profit_target(latest_close, abs(roc_value), latest_atr, signal_type)
                    
                    # Calculate position size
                    size = self.calculate_position_size(equity, latest_close, stop_price, is_intraday)
                    
                    # Calculate confidence based on indicator strength
                    roc_strength = min(0.3, abs(roc_value) / 10)  # Scale ROC value
                    adx_strength = min(0.2, adx_value / 100)  # Scale ADX
                    volume_strength = min(0.2, volume_ratio_value / 5)  # Scale volume ratio
                    macd_strength = min(0.2, abs(macd_line_value) * 10)  # Scale MACD
                    
                    confidence = min(0.95, 0.2 + roc_strength + adx_strength + volume_strength + macd_strength)
                    
                    # Create signal
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=latest_close,
                        timestamp=latest_timestamp,
                        confidence=confidence,
                        stop_loss=stop_price,
                        take_profit=target_price,
                        size=size,
                        metadata={
                            "strategy_type": "momentum_trading",
                            "timeframe_mode": "intraday" if is_intraday else "swing",
                            "signal_timeframe": signal_tf,
                            "roc_value": roc_value,
                            "adx_value": adx_value,
                            "volume_ratio": volume_ratio_value,
                            "atr": latest_atr,
                            "trailing_stop": True,
                            "trailing_stop_activation_atr": self.parameters.get("trailing_stop_activation_atr", 1.0) * latest_atr,
                            "trailing_stop_atr_multiple": self.parameters.get("trailing_stop_atr_multiple", 0.75)
                        }
                    )
                    
                    # Update sector positions if sector information is available
                    if symbol_sectors and symbol in symbol_sectors:
                        sector = symbol_sectors[symbol]
                        self.sector_positions[sector] = self.sector_positions.get(sector, 0) + 1
                
            except Exception as e:
                logger.error(f"Error generating momentum signal for {symbol}: {e}")
        
        return signals
    
    def update_state_after_exit(
        self, symbol: str, exit_price: float, entry_price: float, 
        signal_type: SignalType, sector: Optional[str] = None,
        current_bar_idx: Optional[int] = None
    ) -> None:
        """
        Update strategy state after position exit.
        
        Args:
            symbol: Symbol of the exited position
            exit_price: Exit price
            entry_price: Entry price
            signal_type: Type of signal (BUY or SELL)
            sector: Optional sector of the symbol
            current_bar_idx: Optional current bar index
        """
        # Check if exit was a loss
        is_loss = (signal_type == SignalType.BUY and exit_price < entry_price) or (signal_type == SignalType.SELL and exit_price > entry_price)
        
        if is_loss:
            self.consecutive_losses += 1
            
            # Add to stopped symbols if stopped out
            if current_bar_idx is not None:
                self.stopped_symbols[symbol] = current_bar_idx
        else:
            self.consecutive_losses = 0
        
        # Update sector positions
        if sector and sector in self.sector_positions and self.sector_positions[sector] > 0:
            self.sector_positions[sector] -= 1
    
    def reset_daily_state(self) -> None:
        """
        Reset strategy state at the start of a new trading day.
        """
        # For intraday mode, reset consecutive losses each day
        if self.parameters.get("timeframe_mode", "intraday") == "intraday":
            self.consecutive_losses = 0
        
        # Clear stale stopped symbols (older than 1 day)
        self.stopped_symbols = {} 