#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breakout Trading Strategy Module

This module implements a strategy to exploit volatility expansions when price breaks key
structural levels or consolidation boundaries, entering early on confirmed breakouts and
riding directional thrusts.
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

class BreakoutTradingStrategy(StrategyOptimizable):
    """
    Breakout Trading Strategy designed to exploit volatility expansions.
    
    This strategy identifies and trades price breakouts from key structural levels
    or consolidation boundaries, using volume and volatility confirmation to filter
    out false breaks and ride directional thrusts.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Breakout Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # Liquid large-caps, ETFs, futures, FX pairs with clear range behavior
            "timeframe_mode": "intraday",  # 'intraday' or 'swing'
            "signal_timeframe_intraday": "15min",  # or '1h'
            "signal_timeframe_swing": "1d",
            "context_timeframe_intraday": "4h",
            "context_timeframe_swing": "1w",
            "max_positions": 5,  # Maximum concurrent breakout positions
            
            # Core Indicators
            "donchian_period": 20,  # Period for Donchian channels
            "bollinger_period": 20,  # Period for Bollinger Bands
            "bollinger_std_dev": 2.0,  # Standard deviations for Bollinger Bands
            "volume_ma_period": 20,  # Period for volume moving average
            "atr_period": 14,  # Period for ATR calculation
            "adx_period": 14,  # Period for ADX calculation
            
            # Entry Criteria
            "bandwidth_threshold": 0.1,  # Maximum Bollinger bandwidth for volatility squeeze
            "volume_multiplier": 1.5,  # Minimum volume vs MA for confirmation
            "context_ema_period": 50,  # Higher timeframe EMA period
            "adx_threshold": 20,  # Minimum ADX for trend strength
            
            # Exit Criteria
            "profit_target_atr_intraday": 1.5,  # Profit target as ATR multiple (intraday)
            "profit_target_atr_swing": 2.0,  # Profit target as ATR multiple (swing)
            "stop_loss_atr_multiple": 1.0,  # Stop loss as ATR multiple
            "trailing_stop_atr_multiple": 0.75,  # Trailing stop as ATR multiple
            "trailing_stop_activation_atr": 1.0,  # ATR multiple to activate trailing stop
            "max_bars_in_trade_intraday": 16,  # Approximately 4 hours in 15-min bars
            "max_bars_in_trade_swing": 6,  # Maximum days to hold swing breakout
            "close_before_session_end_mins": 15,  # Minutes before session end to close intraday positions
            
            # Position Sizing & Risk Controls
            "risk_percent_intraday": 0.005,  # 0.5% risk per intraday trade
            "risk_percent_swing": 0.01,  # 1% risk per swing trade
            "max_exposure_percent": 0.20,  # Maximum 20% exposure
            "daily_drawdown_limit_percent": 0.01,  # Halt new trades if daily P&L <= -1%
            
            # Order Execution
            "entry_strategy": "limit",  # 'limit' or 'market'
            "entry_acceleration_threshold": 0.002,  # 0.2% acceleration beyond level for market entry
            "slippage_buffer_percent": 0.0001,  # 0.01% buffer for slippage
            "max_wait_bars": 2,  # Maximum bars to wait for order fill
            
            # Operational Rules
            "news_filter_minutes": 5,  # Minutes to avoid entries near major news
            "max_sector_positions": 2,  # Maximum positions per sector
            "min_consolidation_bars": 10,  # Minimum bars of consolidation before breakout
            "reset_at_session_start": True,  # Whether to recalculate parameters at session start
            
            # Continuous Optimization
            "adaptive_adx_threshold": True,  # Whether to adjust ADX threshold based on volatility
            "use_ml_overlay": False  # Whether to use ML for filtering setups
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize state variables
        self.daily_pnl = 0.0
        self.current_exposure = 0.0
        self.sector_positions = {}  # Track positions per sector
        self.position_entry_times = {}  # Track entry times
        self.position_entry_bars = {}  # Track bars since entry
        self.last_session_reset = None  # Track last session reset
        
        logger.info(f"Initialized Breakout Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "donchian_period": [10, 15, 20, 25, 30],
            "bollinger_period": [15, 20, 25],
            "bollinger_std_dev": [1.5, 2.0, 2.5],
            "bandwidth_threshold": [0.05, 0.1, 0.15, 0.2],
            "volume_multiplier": [1.2, 1.5, 1.8, 2.0],
            "adx_threshold": [15, 20, 25, 30],
            "context_ema_period": [20, 35, 50, 75],
            "profit_target_atr_intraday": [1.0, 1.5, 2.0],
            "profit_target_atr_swing": [1.5, 2.0, 2.5, 3.0],
            "stop_loss_atr_multiple": [0.75, 1.0, 1.25],
            "trailing_stop_atr_multiple": [0.5, 0.75, 1.0],
            "trailing_stop_activation_atr": [0.75, 1.0, 1.5],
            "min_consolidation_bars": [5, 10, 15, 20]
        }
    
    # --------------------------------------------------------
    # 3. Core Indicators
    # --------------------------------------------------------
    
    def calculate_donchian_channels(
        self, high: pd.Series, low: pd.Series, period: int = 20
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channels for tracking breakout levels.
        
        Args:
            high: Series of high prices
            low: Series of low prices
            period: Lookback period
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        # TODO: calculate Donchian channels
        upper_band = high.rolling(window=period).max()
        lower_band = low.rolling(window=period).min()
        middle_band = (upper_band + lower_band) / 2
        
        return upper_band, middle_band, lower_band
    
    def calculate_bollinger_bands(
        self, close: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands for volatility squeeze detection.
        
        Args:
            close: Series of close prices
            period: Bollinger period
            std_dev: Standard deviation multiplier
            
        Returns:
            Tuple of (middle_band, upper_band, lower_band, bandwidth)
        """
        # TODO: calculate Bollinger Bands and bandwidth
        middle_band = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()
        
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)
        
        # Calculate bandwidth (normalized by middle band)
        bandwidth = (upper_band - lower_band) / middle_band
        
        return middle_band, upper_band, lower_band, bandwidth
    
    def calculate_volume_ma(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Volume Moving Average for breakout confirmation.
        
        Args:
            volume: Series of volume data
            period: MA period
            
        Returns:
            Series with volume MA values
        """
        # TODO: calculate volume moving average
        return volume.rolling(window=period).mean()
    
    def calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
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
    
    def calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index for trend strength.
        
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
    
    # --------------------------------------------------------
    # 4. Entry Criteria
    # --------------------------------------------------------
    
    def is_level_breakout(
        self, 
        close: float, 
        donchian_upper: float, 
        donchian_lower: float,
        bollinger_upper: float,
        bollinger_lower: float,
        signal_type: SignalType
    ) -> bool:
        """
        Check if price is breaking out of key levels.
        
        Args:
            close: Current close price
            donchian_upper: Upper Donchian channel
            donchian_lower: Lower Donchian channel
            bollinger_upper: Upper Bollinger Band
            bollinger_lower: Lower Bollinger Band
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if price is breaking out
        """
        # TODO: detect level breakout
        if signal_type == SignalType.BUY:
            # Long breakout - price closing above upper Donchian or Bollinger band
            return close > donchian_upper or close > bollinger_upper
        else:
            # Short breakout - price closing below lower Donchian or Bollinger band
            return close < donchian_lower or close < bollinger_lower
    
    def is_volatility_squeezed(self, bandwidth: float) -> bool:
        """
        Check if volatility is squeezed (low Bollinger bandwidth).
        
        Args:
            bandwidth: Current Bollinger bandwidth
            
        Returns:
            Boolean indicating if volatility is squeezed
        """
        # TODO: detect volatility squeeze
        bandwidth_threshold = self.parameters.get("bandwidth_threshold", 0.1)
        return bandwidth < bandwidth_threshold
    
    def has_volume_confirmation(self, volume: float, volume_ma: float) -> bool:
        """
        Check if volume confirms the breakout.
        
        Args:
            volume: Current volume
            volume_ma: Volume moving average
            
        Returns:
            Boolean indicating if volume confirms breakout
        """
        # TODO: check volume confirmation
        volume_multiplier = self.parameters.get("volume_multiplier", 1.5)
        return volume >= (volume_ma * volume_multiplier)
    
    def is_higher_timeframe_aligned(
        self, htf_close: float, htf_ema: float, signal_type: SignalType
    ) -> bool:
        """
        Check if higher timeframe confirms directional bias.
        
        Args:
            htf_close: Higher timeframe close price
            htf_ema: Higher timeframe EMA value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Boolean indicating if higher timeframe confirms
        """
        # TODO: check higher timeframe alignment
        if signal_type == SignalType.BUY:
            return htf_close > htf_ema
        else:
            return htf_close < htf_ema
    
    def is_adx_sufficient(self, adx: float) -> bool:
        """
        Check if ADX indicates sufficient trend strength.
        
        Args:
            adx: Current ADX value
            
        Returns:
            Boolean indicating if ADX is sufficient
        """
        # TODO: enforce ADX filter
        adx_threshold = self.parameters.get("adx_threshold", 20)
        
        # Adjust threshold based on volatility if adaptive settings enabled
        if self.parameters.get("adaptive_adx_threshold", True):
            # Lower threshold in low volatility environments
            # This would be implemented with a more sophisticated approach in practice
            pass
            
        return adx >= adx_threshold
    
    def is_consolidation_sufficient(self, price_range: pd.Series) -> bool:
        """
        Check if price has consolidated enough before breakout.
        
        Args:
            price_range: Series of (high-low)/close values
            
        Returns:
            Boolean indicating if consolidation is sufficient
        """
        # TODO: check consolidation requirement
        min_bars = self.parameters.get("min_consolidation_bars", 10)
        
        # Check if the recent bars show tight consolidation
        # A simple approach is to check if price range is below a threshold
        recent_ranges = price_range.tail(min_bars)
        avg_range = recent_ranges.mean()
        
        # Consider consolidated if average range is small (customize threshold as needed)
        return len(recent_ranges) >= min_bars and avg_range < 0.01  # 1% average range
    
    # --------------------------------------------------------
    # 5. Exit Criteria
    # --------------------------------------------------------
    
    def calculate_profit_target(
        self, entry_price: float, atr: float, signal_type: SignalType, is_intraday: bool
    ) -> float:
        """
        Calculate profit target level based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            is_intraday: Whether this is an intraday or swing trade
            
        Returns:
            Profit target price
        """
        # TODO: compute profit target
        # Use different ATR multiples for intraday vs swing
        if is_intraday:
            target_multiple = self.parameters.get("profit_target_atr_intraday", 1.5)
        else:
            target_multiple = self.parameters.get("profit_target_atr_swing", 2.0)
            
        target_distance = atr * target_multiple
        
        if signal_type == SignalType.BUY:
            return entry_price + target_distance
        else:
            return entry_price - target_distance
    
    def calculate_stop_loss(
        self, entry_price: float, atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate stop loss level based on ATR.
        
        Args:
            entry_price: Entry price
            atr: Current ATR value
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
        self, current_price: float, high_since_entry: float, low_since_entry: float, 
        atr: float, signal_type: SignalType
    ) -> float:
        """
        Calculate trailing stop level once in profit.
        
        Args:
            current_price: Current price
            high_since_entry: Highest price since entry
            low_since_entry: Lowest price since entry
            atr: Current ATR value
            signal_type: Type of signal (BUY or SELL)
            
        Returns:
            Trailing stop price
        """
        # TODO: compute trailing stop
        trailing_multiple = self.parameters.get("trailing_stop_atr_multiple", 0.75)
        trail_distance = atr * trailing_multiple
        
        if signal_type == SignalType.BUY:
            return high_since_entry - trail_distance
        else:
            return low_since_entry + trail_distance
    
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
        activation_multiple = self.parameters.get("trailing_stop_activation_atr", 1.0)
        activation_distance = atr * activation_multiple
        
        if signal_type == SignalType.BUY:
            return current_price >= (entry_price + activation_distance)
        else:
            return current_price <= (entry_price - activation_distance)
    
    def should_exit_by_time(
        self, bars_in_trade: int, is_intraday: bool, current_time: datetime = None
    ) -> bool:
        """
        Check if position should be exited based on time criteria.
        
        Args:
            bars_in_trade: Number of bars since entry
            is_intraday: Whether this is an intraday trade
            current_time: Optional current timestamp
            
        Returns:
            Boolean indicating if time-based exit should be triggered
        """
        # TODO: evaluate time-based exit
        if is_intraday:
            # Check if approaching session end
            if current_time:
                # Assuming US market hours for example
                market_close = datetime(
                    current_time.year, current_time.month, current_time.day, 16, 0, 0
                )
                mins_before_close = self.parameters.get("close_before_session_end_mins", 15)
                
                # Exit if within X minutes of market close
                if (market_close - current_time).total_seconds() <= (mins_before_close * 60):
                    return True
            
            # Also check max bars
            max_bars = self.parameters.get("max_bars_in_trade_intraday", 16)
            return bars_in_trade >= max_bars
        else:
            # For swing trades, exit if no new highs/lows in consecutive bars
            # This is a placeholder - would need more data to implement properly
            max_bars = self.parameters.get("max_bars_in_trade_swing", 6)
            return bars_in_trade >= max_bars
    
    def should_exit_on_consolidation(
        self, price_series: pd.Series, highs: pd.Series, lows: pd.Series, 
        signal_type: SignalType, bars_in_trade: int
    ) -> bool:
        """
        Check if swing position should be exited due to new consolidation.
        
        Args:
            price_series: Recent price data
            highs: Recent high prices
            lows: Recent low prices
            signal_type: Type of signal (BUY or SELL)
            bars_in_trade: Number of bars since entry
            
        Returns:
            Boolean indicating if consolidation exit should trigger
        """
        # TODO: check for consolidation exit
        # For swing trades, check if no new highs/lows made in last 2 bars
        if bars_in_trade < 2:
            return False
            
        if signal_type == SignalType.BUY:
            # Check if we've made new highs in last 2 bars
            recent_high = highs.iloc[-2:].max()
            prev_high = highs.iloc[-4:-2].max()
            return recent_high <= prev_high
        else:
            # Check if we've made new lows in last 2 bars
            recent_low = lows.iloc[-2:].min()
            prev_low = lows.iloc[-4:-2].min()
            return recent_low >= prev_low
    
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
        # Use different risk percentages for intraday vs swing
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
        max_exposure = self.parameters.get("max_exposure_percent", 0.20) * equity
        max_positions = self.parameters.get("max_positions", 5)
        
        return (current_exposure < max_exposure) and (len(self.position_entry_times) < max_positions)
    
    def should_halt_trading(self, daily_pnl: float, equity: float) -> bool:
        """
        Check if trading should be halted due to daily drawdown.
        
        Args:
            daily_pnl: Current daily P&L
            equity: Total equity
            
        Returns:
            Boolean indicating if trading should be halted
        """
        # TODO: apply daily drawdown limit
        drawdown_limit = -self.parameters.get("daily_drawdown_limit_percent", 0.01) * equity
        return daily_pnl <= drawdown_limit
    
    def can_add_sector_position(self, sector: str, current_sector_positions: Dict[str, int]) -> bool:
        """
        Check if a new position can be added to a sector.
        
        Args:
            sector: Sector of the new position
            current_sector_positions: Dictionary mapping sectors to current position counts
            
        Returns:
            Boolean indicating if a new sector position can be added
        """
        # TODO: apply sector position limit
        max_sector_positions = self.parameters.get("max_sector_positions", 2)
        current_count = current_sector_positions.get(sector, 0)
        
        return current_count < max_sector_positions
    
    # --------------------------------------------------------
    # 7 & 8. Order Execution & Operational Rules
    # --------------------------------------------------------
    
    def should_use_market_entry(
        self, current_price: float, breakout_level: float
    ) -> bool:
        """
        Check if market order should be used instead of limit order.
        
        Args:
            current_price: Current price
            breakout_level: Level that was broken out
            
        Returns:
            Boolean indicating if market order should be used
        """
        # TODO: determine order type
        if self.parameters.get("entry_strategy", "limit") == "market":
            return True
            
        # Check if price has accelerated beyond the level
        acceleration_threshold = self.parameters.get("entry_acceleration_threshold", 0.002)
        return abs(current_price - breakout_level) / breakout_level > acceleration_threshold
    
    def should_avoid_news(
        self, current_time: datetime, news_times: List[datetime]
    ) -> bool:
        """
        Check if entry should be avoided due to upcoming news.
        
        Args:
            current_time: Current timestamp
            news_times: List of news event timestamps
            
        Returns:
            Boolean indicating if entry should be avoided
        """
        # TODO: apply news filter
        news_filter_minutes = self.parameters.get("news_filter_minutes", 5)
        
        for news_time in news_times:
            # Check if within buffer of news event
            time_diff_minutes = abs((current_time - news_time).total_seconds() / 60)
            if time_diff_minutes <= news_filter_minutes:
                return True
                
        return False
    
    def should_reset_session(
        self, current_time: datetime, last_reset_time: Optional[datetime]
    ) -> bool:
        """
        Check if parameters should be reset for new session.
        
        Args:
            current_time: Current timestamp
            last_reset_time: Last reset timestamp
            
        Returns:
            Boolean indicating if session reset should occur
        """
        # TODO: check session reset
        if not self.parameters.get("reset_at_session_start", True):
            return False
            
        if last_reset_time is None:
            return True
            
        # Reset if we're in a new trading day
        return (
            current_time.date() > last_reset_time.date() and
            current_time.hour >= 9 and current_time.minute >= 30  # Assuming 9:30 AM market open
        )
    
    # --------------------------------------------------------
    # Main Signal Generation
    # --------------------------------------------------------
    
    def calculate_indicators(
        self, 
        signal_data: pd.DataFrame, 
        context_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate all indicators needed for breakout trading signals.
        
        Args:
            signal_data: DataFrame with signal timeframe OHLCV data
            context_data: DataFrame with context timeframe OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Get parameters
        donchian_period = self.parameters.get("donchian_period", 20)
        bollinger_period = self.parameters.get("bollinger_period", 20)
        bollinger_std_dev = self.parameters.get("bollinger_std_dev", 2.0)
        volume_ma_period = self.parameters.get("volume_ma_period", 20)
        atr_period = self.parameters.get("atr_period", 14)
        adx_period = self.parameters.get("adx_period", 14)
        context_ema_period = self.parameters.get("context_ema_period", 50)
        
        # Calculate signal timeframe indicators
        upper_donchian, middle_donchian, lower_donchian = self.calculate_donchian_channels(
            signal_data["high"], signal_data["low"], donchian_period
        )
        indicators["donchian_upper"] = upper_donchian
        indicators["donchian_middle"] = middle_donchian
        indicators["donchian_lower"] = lower_donchian
        
        bb_mid, bb_upper, bb_lower, bb_bandwidth = self.calculate_bollinger_bands(
            signal_data["close"], bollinger_period, bollinger_std_dev
        )
        indicators["bb_middle"] = bb_mid
        indicators["bb_upper"] = bb_upper
        indicators["bb_lower"] = bb_lower
        indicators["bb_bandwidth"] = bb_bandwidth
        
        if "volume" in signal_data.columns:
            volume_ma = self.calculate_volume_ma(signal_data["volume"], volume_ma_period)
            indicators["volume_ma"] = volume_ma
        
        indicators["atr"] = self.calculate_atr(
            signal_data["high"], signal_data["low"], signal_data["close"], atr_period
        )
        
        indicators["adx"] = self.calculate_adx(
            signal_data["high"], signal_data["low"], signal_data["close"], adx_period
        )
        
        # Calculate price range for consolidation detection
        indicators["price_range"] = (signal_data["high"] - signal_data["low"]) / signal_data["close"]
        
        # Calculate context timeframe indicators
        context_ema = self.calculate_ema(context_data["close"], context_ema_period)
        indicators["context_ema"] = context_ema
        
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
        Generate breakout trading signals.
        
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
            
        # Check if session reset is needed
        if self.should_reset_session(current_time, self.last_session_reset):
            logger.info("Resetting session parameters for breakout trading")
            self.last_session_reset = current_time
            self.daily_pnl = 0.0  # Reset daily P&L
        
        # Check if we can add new positions
        if not self.can_add_new_position(equity, self.current_exposure):
            logger.info("Skipping breakout signals due to exposure restrictions")
            return {}
        
        # Check if trading should be halted due to daily drawdown
        if self.should_halt_trading(self.daily_pnl, equity):
            logger.info("Skipping breakout signals due to daily drawdown limit")
            return {}
        
        # Generate signals
        signals = {}
        
        # Determine if we're in intraday or swing mode
        is_intraday = self.parameters.get("timeframe_mode", "intraday") == "intraday"
        
        # Get the appropriate signal and context timeframes
        signal_tf = self.parameters.get(
            "signal_timeframe_intraday" if is_intraday else "signal_timeframe_swing", 
            "15min" if is_intraday else "1d"
        )
        context_tf = self.parameters.get(
            "context_timeframe_intraday" if is_intraday else "context_timeframe_swing",
            "4h" if is_intraday else "1w"
        )
        
        for symbol, timeframe_data in data.items():
            try:
                # Check if we have data for both required timeframes
                if signal_tf not in timeframe_data or context_tf not in timeframe_data:
                    continue
                    
                signal_data = timeframe_data[signal_tf]
                context_data = timeframe_data[context_tf]
                
                # Need enough bars for indicators
                min_bars = max(
                    self.parameters.get("donchian_period", 20),
                    self.parameters.get("bollinger_period", 20),
                    self.parameters.get("adx_period", 14)
                ) + 5  # Add buffer
                
                if len(signal_data) < min_bars or len(context_data) < 10:
                    continue
                
                # Check news filter
                if news_events and symbol in news_events:
                    if self.should_avoid_news(current_time, news_events[symbol]):
                        continue
                
                # Check sector position limit
                if symbol_sectors and symbol in symbol_sectors:
                    sector = symbol_sectors[symbol]
                    if not self.can_add_sector_position(sector, self.sector_positions):
                        continue
                
                # Calculate indicators
                indicators = self.calculate_indicators(signal_data, context_data)
                
                # Get latest data
                latest_signal = signal_data.iloc[-1]
                latest_context = context_data.iloc[-1]
                
                # Get latest indicator values
                latest_donchian_upper = indicators["donchian_upper"].iloc[-1]
                latest_donchian_lower = indicators["donchian_lower"].iloc[-1]
                latest_bb_upper = indicators["bb_upper"].iloc[-1]
                latest_bb_lower = indicators["bb_lower"].iloc[-1]
                latest_bb_bandwidth = indicators["bb_bandwidth"].iloc[-1]
                latest_adx = indicators["adx"].iloc[-1]
                latest_context_ema = indicators["context_ema"].iloc[-1]
                
                # Check volume if available
                has_volume_data = "volume" in signal_data.columns and "volume_ma" in indicators
                if has_volume_data:
                    latest_volume = latest_signal["volume"]
                    latest_volume_ma = indicators["volume_ma"].iloc[-1]
                    volume_confirmed = self.has_volume_confirmation(latest_volume, latest_volume_ma)
                else:
                    # If volume data isn't available, don't use it as a filter
                    volume_confirmed = True
                
                # Check consolidation
                price_range = indicators["price_range"]
                consolidation_confirmed = self.is_consolidation_sufficient(price_range)
                
                # Check for long breakout signal
                long_breakout = (
                    self.is_level_breakout(
                        latest_signal["close"], 
                        latest_donchian_upper,
                        latest_donchian_lower,
                        latest_bb_upper,
                        latest_bb_lower,
                        SignalType.BUY
                    ) and
                    self.is_volatility_squeezed(latest_bb_bandwidth) and
                    volume_confirmed and
                    self.is_higher_timeframe_aligned(latest_context["close"], latest_context_ema, SignalType.BUY) and
                    self.is_adx_sufficient(latest_adx) and
                    consolidation_confirmed
                )
                
                # Check for short breakout signal
                short_breakout = (
                    self.is_level_breakout(
                        latest_signal["close"], 
                        latest_donchian_upper,
                        latest_donchian_lower,
                        latest_bb_upper,
                        latest_bb_lower,
                        SignalType.SELL
                    ) and
                    self.is_volatility_squeezed(latest_bb_bandwidth) and
                    volume_confirmed and
                    self.is_higher_timeframe_aligned(latest_context["close"], latest_context_ema, SignalType.SELL) and
                    self.is_adx_sufficient(latest_adx) and
                    consolidation_confirmed
                )
                
                # Process breakout signal
                signal_type = None
                breakout_level = None
                
                if long_breakout:
                    signal_type = SignalType.BUY
                    # Determine which level was broken
                    if latest_signal["close"] > latest_donchian_upper:
                        breakout_level = latest_donchian_upper
                    else:
                        breakout_level = latest_bb_upper
                elif short_breakout:
                    signal_type = SignalType.SELL
                    # Determine which level was broken
                    if latest_signal["close"] < latest_donchian_lower:
                        breakout_level = latest_donchian_lower
                    else:
                        breakout_level = latest_bb_lower
                
                # Generate signal if we have a valid signal type
                if signal_type:
                    # Calculate ATR for stop/target
                    latest_atr = indicators["atr"].iloc[-1]
                    
                    # Use breakout level as entry if using limit orders
                    entry_price = breakout_level if not self.should_use_market_entry(
                        latest_signal["close"], breakout_level
                    ) else latest_signal["close"]
                    
                    # Add slippage buffer for realistic execution
                    slippage_buffer = self.parameters.get("slippage_buffer_percent", 0.0001)
                    if signal_type == SignalType.BUY:
                        entry_price = entry_price * (1 + slippage_buffer)
                    else:
                        entry_price = entry_price * (1 - slippage_buffer)
                    
                    # Calculate stop loss
                    stop_price = self.calculate_stop_loss(entry_price, latest_atr, signal_type)
                    
                    # Calculate profit target
                    target_price = self.calculate_profit_target(entry_price, latest_atr, signal_type, is_intraday)
                    
                    # Calculate position size
                    size = self.calculate_position_size(equity, entry_price, stop_price, is_intraday)
                    
                    # Calculate confidence
                    # Base confidence on ADX strength, volatility squeeze, and volume confirmation
                    adx_factor = min(0.3, latest_adx / 100)
                    squeeze_factor = min(0.3, (0.2 - latest_bb_bandwidth) / 0.2) if latest_bb_bandwidth < 0.2 else 0
                    volume_factor = min(0.2, (latest_volume / latest_volume_ma - 1) * 0.5) if has_volume_data else 0
                    
                    confidence = min(0.9, 0.3 + adx_factor + squeeze_factor + volume_factor)
                    
                    # Record entry time
                    self.position_entry_times[symbol] = current_time
                    self.position_entry_bars[symbol] = 0
                    
                    # Create signal
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=entry_price,
                        timestamp=current_time,
                        confidence=confidence,
                        stop_loss=stop_price,
                        take_profit=target_price,
                        size=size,
                        metadata={
                            "strategy_type": "breakout_trading",
                            "timeframe_mode": "intraday" if is_intraday else "swing",
                            "breakout_level": breakout_level,
                            "donchian_upper": latest_donchian_upper,
                            "donchian_lower": latest_donchian_lower,
                            "bb_upper": latest_bb_upper,
                            "bb_lower": latest_bb_lower,
                            "bb_bandwidth": latest_bb_bandwidth,
                            "adx": latest_adx,
                            "atr": latest_atr,
                            "entry_time": current_time.isoformat(),
                            "trailing_stop": True,
                            "trailing_stop_activated": False,
                            "trailing_stop_activation_atr": self.parameters.get("trailing_stop_activation_atr", 1.0) * latest_atr,
                            "trailing_stop_atr_multiple": self.parameters.get("trailing_stop_atr_multiple", 0.75),
                            "use_market_order": self.should_use_market_entry(latest_signal["close"], breakout_level),
                            "max_wait_bars": self.parameters.get("max_wait_bars", 2)
                        }
                    )
                    
                    # Update sector positions if sector information is available
                    if symbol_sectors and symbol in symbol_sectors:
                        sector = symbol_sectors[symbol]
                        self.sector_positions[sector] = self.sector_positions.get(sector, 0) + 1
            
            except Exception as e:
                logger.error(f"Error generating breakout signal for {symbol}: {e}")
        
        return signals
    
    def update_state_after_exit(
        self, symbol: str, exit_price: float, entry_price: float, 
        signal_type: SignalType, pnl: float = None, sector: Optional[str] = None
    ) -> None:
        """
        Update strategy state after position exit.
        
        Args:
            symbol: Symbol of the exited position
            exit_price: Exit price
            entry_price: Entry price
            signal_type: Type of signal (BUY or SELL)
            pnl: Optional P&L from the trade
            sector: Optional sector of the symbol
        """
        # Update daily P&L if provided
        if pnl is not None:
            self.daily_pnl += pnl
        
        # Remove entry time and bar tracking
        if symbol in self.position_entry_times:
            del self.position_entry_times[symbol]
        
        if symbol in self.position_entry_bars:
            del self.position_entry_bars[symbol]
        
        # Update sector positions
        if sector and sector in self.sector_positions and self.sector_positions[sector] > 0:
            self.sector_positions[sector] -= 1
    
    def update_position_bars(self) -> None:
        """
        Update bar count for all active positions.
        """
        for symbol in self.position_entry_bars:
            self.position_entry_bars[symbol] += 1 