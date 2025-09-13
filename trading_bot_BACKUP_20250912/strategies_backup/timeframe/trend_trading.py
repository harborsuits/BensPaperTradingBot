#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trend Trading Strategy Module

This module implements a strategy to ride sustained directional moves by identifying
clear, multi-timeframe trends and entering with momentum confirmation, aiming for
large reward targets relative to risk.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

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

class TrendTradingStrategy(StrategyOptimizable):
    """
    Trend Trading Strategy designed to capture sustained directional moves.
    
    This strategy identifies clear trends across multiple timeframes and enters
    in the direction of the trend when momentum confirms. It aims for larger profit
    targets relative to risk and holds positions through short-term pullbacks.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Trend Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # Highly liquid large-caps, sector ETFs, futures, currency pairs
            "max_positions": 5,  # Maximum concurrent trend positions
            "min_holding_weeks": 2,  # Minimum holding period in weeks
            "max_holding_weeks": 12,  # Maximum holding period in weeks
            
            # Core Indicators
            "weekly_ema_period": 20,  # Weekly EMA period for primary trend
            "daily_sma_fast_period": 50,  # Daily SMA fast period
            "daily_sma_slow_period": 200,  # Daily SMA slow period
            "adx_period": 14,  # ADX period
            "macd_fast": 12,  # MACD fast period
            "macd_slow": 26,  # MACD slow period
            "macd_signal": 9,  # MACD signal period
            "atr_period": 14,  # ATR period
            
            # Entry Criteria
            "adx_threshold": 25,  # Minimum ADX for trend strength
            "pullback_atr_multiple": 1.0,  # ATR multiple for pullback filter
            
            # Exit Criteria
            "stop_loss_atr_multiple": 1.0,  # Stop loss as ATR multiple
            "profit_target_atr_multiple": 3.0,  # Profit target as ATR multiple
            "trailing_stop_activation_atr": 2.0,  # ATR multiple to activate trailing stop
            "trailing_stop_atr_multiple": 1.0,  # Trailing stop as ATR multiple
            
            # Position Sizing & Risk Controls
            "risk_percent": 0.01,  # 1% risk per trade
            "max_exposure_percent": 0.40,  # Maximum 40% exposure
            "max_consecutive_losses": 2,  # Maximum consecutive losses before pause
            "loss_pause_weeks": 2,  # Weeks to pause after consecutive losses
            
            # Order Execution
            "slippage_buffer_percent": 0.0002,  # 0.02% buffer for slippage
            
            # Operational Rules
            "news_filter_days": 3,  # Days to avoid entries before/after news
            "max_sector_positions": 2,  # Maximum positions per sector
            
            # Adaptive Indicators (for Continuous Optimization)
            "use_keltner_channels": False,  # Whether to use Keltner channels as adaptive filter
            "use_donchian_bands": False,  # Whether to use Donchian bands as adaptive filter
            "keltner_period": 20,  # Keltner channel period
            "keltner_atr_multiple": 2.0,  # Keltner channel ATR multiple
            "donchian_period": 20,  # Donchian band period
            "use_ml_overlay": False  # Whether to use ML overlay for entry refinement
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
        self.position_entry_times = {}  # Track entry times for time-based exits
        
        logger.info(f"Initialized Trend Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "weekly_ema_period": [10, 15, 20, 25, 30],
            "daily_sma_fast_period": [20, 35, 50, 65],
            "daily_sma_slow_period": [150, 175, 200, 225],
            "adx_period": [10, 14, 20],
            "adx_threshold": [20, 25, 30, 35],
            "macd_fast": [8, 10, 12, 16],
            "macd_slow": [20, 24, 26, 30],
            "macd_signal": [7, 9, 11],
            "atr_period": [10, 14, 21],
            "pullback_atr_multiple": [0.5, 1.0, 1.5, 2.0],
            "stop_loss_atr_multiple": [0.75, 1.0, 1.25, 1.5],
            "profit_target_atr_multiple": [2.0, 2.5, 3.0, 4.0],
            "trailing_stop_activation_atr": [1.5, 2.0, 2.5],
            "trailing_stop_atr_multiple": [0.75, 1.0, 1.25],
            "risk_percent": [0.005, 0.01, 0.015, 0.02],
            "max_exposure_percent": [0.3, 0.4, 0.5],
            "min_holding_weeks": [1, 2, 3],
            "max_holding_weeks": [8, 10, 12, 16]
        }
    
    # --------------------------------------------------------
    # 3. Core Indicators
    # --------------------------------------------------------
    
    def calculate_ema(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Series of price data
            period: EMA period
            
        Returns:
            Series with EMA values
        """
        # TODO: compute weekly EMA
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_sma(self, prices: pd.Series, period: int = 50) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Series of price data
            period: SMA period
            
        Returns:
            Series with SMA values
        """
        # TODO: compute daily SMA
        return prices.rolling(window=period).mean()
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) for trend strength.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            close: Series of close prices
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        # TODO: compute ADX
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate directional movement
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
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for stop loss/target calculation.
        
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
    
    def calculate_keltner_channels(
        self, close: pd.Series, high: pd.Series, low: pd.Series, 
        period: int = 20, atr_multiple: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels for adaptive filtering.
        
        Args:
            close: Series of close prices
            high: Series of high prices
            low: Series of low prices
            period: Keltner channel period
            atr_multiple: ATR multiple for channel width
            
        Returns:
            Tuple of (middle_line, upper_line, lower_line)
        """
        # TODO: compute Keltner channels for adaptive filtering
        middle_line = close.rolling(window=period).mean()  # SMA of closing prices
        atr = self.calculate_atr(high, low, close, period)
        
        upper_line = middle_line + (atr * atr_multiple)
        lower_line = middle_line - (atr * atr_multiple)
        
        return middle_line, upper_line, lower_line
    
    def calculate_donchian_bands(
        self, high: pd.Series, low: pd.Series, period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Bands for adaptive filtering.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            period: Donchian band period
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band)
        """
        # TODO: compute Donchian bands for adaptive filtering
        upper_band = high.rolling(window=period).max()
        lower_band = low.rolling(window=period).min()
        middle_band = (upper_band + lower_band) / 2
        
        return middle_band, upper_band, lower_band
    
    # --------------------------------------------------------
    # 4. Entry Criteria
    # --------------------------------------------------------
    
    def is_macro_trend_aligned(
        self, weekly_close: float, weekly_ema: float, signal_type: SignalType
    ) -> bool:
        """
        Check if macro trend is aligned using weekly EMA.
        
        Args:
            weekly_close: Current weekly close price
            weekly_ema: Current weekly EMA value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if macro trend is aligned
        """
        # TODO: check macro trend alignment
        if signal_type == SignalType.BUY:
            return weekly_close > weekly_ema
        else:
            return weekly_close < weekly_ema
    
    def is_intermediate_trend_aligned(
        self, daily_close: float, daily_sma_fast: float, daily_sma_slow: float, 
        signal_type: SignalType
    ) -> bool:
        """
        Check if intermediate trend is aligned using daily SMAs.
        
        Args:
            daily_close: Current daily close price
            daily_sma_fast: Current daily fast SMA value
            daily_sma_slow: Current daily slow SMA value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if intermediate trend is aligned
        """
        # TODO: check intermediate trend alignment
        if signal_type == SignalType.BUY:
            return daily_sma_fast > daily_sma_slow and daily_close > daily_sma_fast
        else:
            return daily_sma_fast < daily_sma_slow and daily_close < daily_sma_fast
    
    def has_momentum_crossover(
        self, macd_line: float, signal_line: float, prev_macd_line: float, 
        prev_signal_line: float, signal_type: SignalType
    ) -> bool:
        """
        Check if there is a momentum crossover in MACD.
        
        Args:
            macd_line: Current MACD line value
            signal_line: Current MACD signal line value
            prev_macd_line: Previous MACD line value
            prev_signal_line: Previous MACD signal line value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if momentum crossover occurred
        """
        # TODO: check MACD momentum crossover
        if signal_type == SignalType.BUY:
            # Current MACD line is above signal line and wasn't above in previous bar
            return (macd_line > signal_line) and (prev_macd_line <= prev_signal_line)
        else:
            # Current MACD line is below signal line and wasn't below in previous bar
            return (macd_line < signal_line) and (prev_macd_line >= prev_signal_line)
    
    def is_trend_strength_sufficient(self, adx: float) -> bool:
        """
        Check if trend strength is sufficient using ADX.
        
        Args:
            adx: Current ADX value
            
        Returns:
            Boolean indicating if trend strength is sufficient
        """
        # TODO: apply ADX filter
        adx_threshold = self.parameters.get("adx_threshold", 25)
        return adx >= adx_threshold
    
    def is_pullback_in_range(
        self, close: float, sma_fast: float, atr: float, signal_type: SignalType
    ) -> bool:
        """
        Check if price has pulled back to a reasonable entry point.
        
        Args:
            close: Current close price
            sma_fast: Current fast SMA value
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if price pullback is in acceptable range
        """
        # TODO: check pullback filter
        pullback_atr = self.parameters.get("pullback_atr_multiple", 1.0) * atr
        
        if signal_type == SignalType.BUY:
            # For long entries, price should be within pullback_atr of SMA50
            return abs(close - sma_fast) <= pullback_atr
        else:
            # For short entries, price should be within pullback_atr of SMA50
            return abs(close - sma_fast) <= pullback_atr
    
    # --------------------------------------------------------
    # 5. Exit Criteria
    # --------------------------------------------------------
    
    def calculate_stop_loss(
        self, entry_price: float, atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate initial stop loss level.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Stop loss price
        """
        # TODO: compute initial stop loss
        stop_multiple = self.parameters.get("stop_loss_atr_multiple", 1.0)
        stop_distance = atr * stop_multiple
        
        if signal_type == SignalType.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_profit_target(
        self, entry_price: float, atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate profit target level.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Profit target price
        """
        # TODO: compute profit target
        target_multiple = self.parameters.get("profit_target_atr_multiple", 3.0)
        target_distance = atr * target_multiple
        
        if signal_type == SignalType.BUY:
            return entry_price + target_distance
        else:
            return entry_price - target_distance
    
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
        trailing_multiple = self.parameters.get("trailing_stop_atr_multiple", 1.0)
        trail_distance = atr * trailing_multiple
        
        if signal_type == SignalType.BUY:
            return highest_price - trail_distance
        else:
            return lowest_price + trail_distance
    
    def should_activate_trailing_stop(
        self, entry_price: float, current_price: float, atr: float, signal_type: SignalType
    ) -> bool:
        """
        Check if trailing stop should be activated.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if trailing stop should be activated
        """
        # TODO: check trailing stop activation
        activation_multiple = self.parameters.get("trailing_stop_activation_atr", 2.0)
        activation_distance = atr * activation_multiple
        
        if signal_type == SignalType.BUY:
            return current_price >= (entry_price + activation_distance)
        else:
            return current_price <= (entry_price - activation_distance)
    
    def should_exit_by_time(
        self, entry_time: datetime, current_time: datetime
    ) -> bool:
        """
        Check if position should be exited based on time criteria.
        
        Args:
            entry_time: Entry timestamp
            current_time: Current timestamp
            
        Returns:
            Boolean indicating if time-based exit should be triggered
        """
        # TODO: evaluate time-based exit
        max_holding_weeks = self.parameters.get("max_holding_weeks", 12)
        
        # Calculate time difference in weeks
        time_diff_weeks = (current_time - entry_time).days / 7
        
        return time_diff_weeks >= max_holding_weeks
    
    # --------------------------------------------------------
    # 6. Position Sizing & Risk Controls
    # --------------------------------------------------------
    
    def calculate_position_size(
        self, equity: float, entry_price: float, stop_price: float
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            equity: Total equity
            entry_price: Entry price
            stop_price: Stop loss price
            
        Returns:
            Position size
        """
        # TODO: compute position size
        risk_percent = self.parameters.get("risk_percent", 0.01)
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
        max_exposure = self.parameters.get("max_exposure_percent", 0.40) * equity
        max_positions = self.parameters.get("max_positions", 5)
        
        return (current_exposure < max_exposure) and (len(self.position_entry_times) < max_positions)
    
    def can_trade_after_losses(
        self, current_time: datetime
    ) -> bool:
        """
        Check if trading is allowed after consecutive losses.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Boolean indicating if trading is allowed
        """
        # TODO: apply consecutive loss check
        max_losses = self.parameters.get("max_consecutive_losses", 2)
        
        if self.consecutive_losses >= max_losses and self.last_loss_time is not None:
            pause_weeks = self.parameters.get("loss_pause_weeks", 2)
            return (current_time - self.last_loss_time) >= timedelta(weeks=pause_weeks)
        
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
    
    def should_avoid_due_to_news(
        self, symbol: str, current_time: datetime, news_events: Dict[str, List[datetime]]
    ) -> bool:
        """
        Check if trading should be avoided due to upcoming or recent news.
        
        Args:
            symbol: Symbol to check
            current_time: Current timestamp
            news_events: Dictionary mapping symbols to lists of news event timestamps
            
        Returns:
            Boolean indicating if trading should be avoided
        """
        # TODO: apply news filter
        if symbol in news_events:
            news_filter_days = self.parameters.get("news_filter_days", 3)
            
            for event_time in news_events[symbol]:
                # Check if event is within the filter period (before or after)
                time_diff = abs((current_time - event_time).total_seconds() / (24 * 3600))
                if time_diff <= news_filter_days:
                    return True
                    
        return False
    
    def is_weekly_recalibration_time(self, current_time: datetime) -> bool:
        """
        Check if it's time for weekly recalibration.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Boolean indicating if it's recalibration time
        """
        # TODO: check recalibration window
        # Friday is weekday 4 (0-based, with Monday as 0)
        return current_time.weekday() == 4
    
    # --------------------------------------------------------
    # 10. Continuous Optimization
    # --------------------------------------------------------
    
    def should_use_adaptive_indicators(
        self, current_volatility: float, historical_volatility: float
    ) -> bool:
        """
        Check if adaptive indicators should be used based on volatility.
        
        Args:
            current_volatility: Current market volatility
            historical_volatility: Historical market volatility
            
        Returns:
            Boolean indicating if adaptive indicators should be used
        """
        # TODO: implement adaptive indicator selection
        # Use adaptive indicators if current volatility deviates significantly from historical
        return abs(current_volatility / historical_volatility - 1) > 0.3
    
    def get_adaptive_indicator_type(
        self, price_series: pd.Series, volatility: float
    ) -> str:
        """
        Determine which adaptive indicator to use based on price behavior.
        
        Args:
            price_series: Recent price data
            volatility: Current volatility
            
        Returns:
            String indicating which adaptive indicator to use
        """
        # TODO: implement adaptive indicator type selection
        # Analyze price behavior to determine best adaptive indicator
        price_range = (price_series.max() - price_series.min()) / price_series.mean()
        
        if price_range > 0.1:  # Wide range, more suitable for Donchian bands
            return "donchian"
        else:  # Narrower range, more suitable for Keltner channels
            return "keltner"
    
    # --------------------------------------------------------
    # Main Signal Generation
    # --------------------------------------------------------
    
    def calculate_indicators(
        self, daily_data: pd.DataFrame, weekly_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate all indicators needed for trend trading signals.
        
        Args:
            daily_data: DataFrame with daily OHLCV data
            weekly_data: DataFrame with weekly OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Get parameters
        weekly_ema_period = self.parameters.get("weekly_ema_period", 20)
        daily_sma_fast_period = self.parameters.get("daily_sma_fast_period", 50)
        daily_sma_slow_period = self.parameters.get("daily_sma_slow_period", 200)
        adx_period = self.parameters.get("adx_period", 14)
        macd_fast = self.parameters.get("macd_fast", 12)
        macd_slow = self.parameters.get("macd_slow", 26)
        macd_signal = self.parameters.get("macd_signal", 9)
        atr_period = self.parameters.get("atr_period", 14)
        
        # Calculate weekly indicators
        indicators["weekly_ema"] = self.calculate_ema(weekly_data["close"], weekly_ema_period)
        
        # Calculate daily indicators
        indicators["daily_sma_fast"] = self.calculate_sma(daily_data["close"], daily_sma_fast_period)
        indicators["daily_sma_slow"] = self.calculate_sma(daily_data["close"], daily_sma_slow_period)
        
        indicators["adx"] = self.calculate_adx(
            daily_data["high"], daily_data["low"], daily_data["close"], adx_period
        )
        
        macd_line, signal_line, histogram = self.calculate_macd(
            daily_data["close"], macd_fast, macd_slow, macd_signal
        )
        indicators["macd_line"] = macd_line
        indicators["macd_signal"] = signal_line
        indicators["macd_histogram"] = histogram
        
        indicators["atr"] = self.calculate_atr(
            daily_data["high"], daily_data["low"], daily_data["close"], atr_period
        )
        
        # Calculate adaptive indicators if enabled
        if self.parameters.get("use_keltner_channels", False):
            keltner_period = self.parameters.get("keltner_period", 20)
            keltner_atr_multiple = self.parameters.get("keltner_atr_multiple", 2.0)
            
            keltner_middle, keltner_upper, keltner_lower = self.calculate_keltner_channels(
                daily_data["close"], daily_data["high"], daily_data["low"],
                keltner_period, keltner_atr_multiple
            )
            
            indicators["keltner_middle"] = keltner_middle
            indicators["keltner_upper"] = keltner_upper
            indicators["keltner_lower"] = keltner_lower
        
        if self.parameters.get("use_donchian_bands", False):
            donchian_period = self.parameters.get("donchian_period", 20)
            
            donchian_middle, donchian_upper, donchian_lower = self.calculate_donchian_bands(
                daily_data["high"], daily_data["low"], donchian_period
            )
            
            indicators["donchian_middle"] = donchian_middle
            indicators["donchian_upper"] = donchian_upper
            indicators["donchian_lower"] = donchian_lower
        
        return indicators
    
    def generate_signals(
        self, 
        data: Dict[str, Dict[str, pd.DataFrame]], 
        equity: float,
        news_events: Optional[Dict[str, List[datetime]]] = None,
        symbol_sectors: Optional[Dict[str, str]] = None,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Signal]:
        """
        Generate trend trading signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            equity: Current equity value
            news_events: Optional Dictionary mapping symbols to lists of news event timestamps
            symbol_sectors: Optional Dictionary mapping symbols to sector classifications
            current_time: Optional current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Set current time if not provided
        if current_time is None:
            current_time = datetime.now()
            
        # Check if we can add new positions
        if not self.can_add_new_position(equity, self.current_exposure):
            logger.info("Skipping trend signals due to exposure restrictions")
            return {}
        
        # Check if trading is allowed after consecutive losses
        if not self.can_trade_after_losses(current_time):
            logger.info("Skipping trend signals due to consecutive loss pause")
            return {}
        
        # Generate signals
        signals = {}
        
        for symbol, timeframe_data in data.items():
            try:
                # Check if we have both daily and weekly data
                if "1d" not in timeframe_data or "1w" not in timeframe_data:
                    continue
                
                # Make sure we have enough data
                daily_data = timeframe_data["1d"]
                weekly_data = timeframe_data["1w"]
                
                if len(daily_data) < 200 or len(weekly_data) < 20:  # Need enough history
                    continue
                
                # Check news filter
                if news_events and self.should_avoid_due_to_news(symbol, current_time, news_events):
                    continue
                
                # Check sector position limit
                if symbol_sectors and symbol in symbol_sectors:
                    sector = symbol_sectors[symbol]
                    if not self.can_add_sector_position(sector, self.sector_positions):
                        continue
                
                # Calculate indicators
                indicators = self.calculate_indicators(daily_data, weekly_data)
                
                # Get latest values
                latest_daily = daily_data.iloc[-1]
                latest_weekly = weekly_data.iloc[-1]
                daily_close = latest_daily["close"]
                weekly_close = latest_weekly["close"]
                
                # Get latest indicator values
                weekly_ema = indicators["weekly_ema"].iloc[-1]
                daily_sma_fast = indicators["daily_sma_fast"].iloc[-1]
                daily_sma_slow = indicators["daily_sma_slow"].iloc[-1]
                
                adx = indicators["adx"].iloc[-1]
                
                macd_line = indicators["macd_line"].iloc[-1]
                macd_signal = indicators["macd_signal"].iloc[-1]
                prev_macd_line = indicators["macd_line"].iloc[-2] if len(indicators["macd_line"]) > 1 else 0
                prev_macd_signal = indicators["macd_signal"].iloc[-2] if len(indicators["macd_signal"]) > 1 else 0
                
                atr = indicators["atr"].iloc[-1]
                
                # Check for long entry signal
                long_conditions = (
                    self.is_macro_trend_aligned(weekly_close, weekly_ema, SignalType.BUY) and
                    self.is_intermediate_trend_aligned(daily_close, daily_sma_fast, daily_sma_slow, SignalType.BUY) and
                    self.has_momentum_crossover(macd_line, macd_signal, prev_macd_line, prev_macd_signal, SignalType.BUY) and
                    self.is_trend_strength_sufficient(adx) and
                    self.is_pullback_in_range(daily_close, daily_sma_fast, atr, SignalType.BUY)
                )
                
                # Check for short entry signal
                short_conditions = (
                    self.is_macro_trend_aligned(weekly_close, weekly_ema, SignalType.SELL) and
                    self.is_intermediate_trend_aligned(daily_close, daily_sma_fast, daily_sma_slow, SignalType.SELL) and
                    self.has_momentum_crossover(macd_line, macd_signal, prev_macd_line, prev_macd_signal, SignalType.SELL) and
                    self.is_trend_strength_sufficient(adx) and
                    self.is_pullback_in_range(daily_close, daily_sma_fast, atr, SignalType.SELL)
                )
                
                # Determine signal type
                signal_type = None
                if long_conditions:
                    signal_type = SignalType.BUY
                elif short_conditions:
                    signal_type = SignalType.SELL
                
                # Generate signal if conditions are met
                if signal_type:
                    # Calculate stop loss
                    stop_price = self.calculate_stop_loss(daily_close, atr, signal_type)
                    
                    # Calculate profit target
                    target_price = self.calculate_profit_target(daily_close, atr, signal_type)
                    
                    # Calculate position size
                    size = self.calculate_position_size(equity, daily_close, stop_price)
                    
                    # Calculate confidence based on indicator strengths
                    adx_strength = min(0.3, adx / 100)  # Scale ADX
                    trend_strength = min(0.3, abs(daily_close - daily_sma_fast) / daily_close * 20)  # Scale trend strength
                    macd_strength = min(0.2, abs(macd_line - macd_signal) * 10)  # Scale MACD
                    
                    confidence = min(0.95, 0.2 + adx_strength + trend_strength + macd_strength)
                    
                    # Record entry time
                    self.position_entry_times[symbol] = current_time
                    
                    # Create signal
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=daily_close,
                        timestamp=current_time,
                        confidence=confidence,
                        stop_loss=stop_price,
                        take_profit=target_price,
                        size=size,
                        metadata={
                            "strategy_type": "trend_trading",
                            "weekly_ema": weekly_ema,
                            "daily_sma_fast": daily_sma_fast,
                            "daily_sma_slow": daily_sma_slow,
                            "adx": adx,
                            "macd_line": macd_line,
                            "macd_signal": macd_signal,
                            "atr": atr,
                            "entry_time": current_time.isoformat(),
                            "trailing_stop": True,
                            "trailing_stop_activated": False,
                            "trailing_stop_activation_atr": self.parameters.get("trailing_stop_activation_atr", 2.0) * atr,
                            "trailing_stop_atr_multiple": self.parameters.get("trailing_stop_atr_multiple", 1.0)
                        }
                    )
                    
                    # Update sector positions if sector information is available
                    if symbol_sectors and symbol in symbol_sectors:
                        sector = symbol_sectors[symbol]
                        self.sector_positions[sector] = self.sector_positions.get(sector, 0) + 1
            
            except Exception as e:
                logger.error(f"Error generating trend signal for {symbol}: {e}")
        
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
        
        # Remove entry time tracking
        if symbol in self.position_entry_times:
            del self.position_entry_times[symbol]
        
        # Update sector positions
        if sector and sector in self.sector_positions and self.sector_positions[sector] > 0:
            self.sector_positions[sector] -= 1
    
    def recalibrate_indicators(self) -> None:
        """
        Recalibrate indicators after weekly close.
        """
        # This method would be called after the weekly close to reset/recalibrate indicators
        # In a real implementation, this might involve adjusting parameters based on recent market conditions
        
        # For now, just log that recalibration was performed
        logger.info("Recalibrated trend trading indicators after weekly close") 