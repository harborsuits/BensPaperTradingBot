#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalping Strategy Module

This module implements a scalping strategy designed to exploit very short-lived
price imbalances and micro-trends to capture small, consistent gains with tight 
risk control, focusing on high-volume, high-liquidity instruments.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
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

class ScalpingStrategy(StrategyOptimizable):
    """
    Scalping Strategy designed to exploit very short-lived price imbalances.
    
    This strategy relies on speed, precision, and tight risk control to capture
    small but consistent gains on ultra-short timeframes, focusing on high-liquidity
    periods in highly liquid markets.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Scalping strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # Ultra-liquid large-caps, major forex pairs, high-volume ETFs
            "trading_start_morning": "09:30",  # Start of morning session (ET)
            "trading_end_morning": "11:00",    # End of morning session (ET)
            "trading_start_afternoon": "15:00", # Start of afternoon session (ET)
            "trading_end_afternoon": "16:00",   # End of afternoon session (ET)
            
            # Core Indicators
            "ema_periods": [5, 8, 13],         # EMA ribbon periods
            "vwap_sigma": 0.5,                 # VWAP bands width (standard deviations)
            "volume_ma_period": 5,             # Volume moving average period
            "volume_multiplier": 1.8,          # Volume spike threshold
            "stoch_k_period": 5,               # Stochastic K period
            "stoch_d_period": 3,               # Stochastic D period
            "stoch_slowing": 3,                # Stochastic slowing
            "stoch_lower_bound": 20,           # Stochastic lower bound
            "stoch_upper_bound": 80,           # Stochastic upper bound
            "rsi_period": 6,                   # RSI period (alternative to stochastic)
            "use_stochastic": True,            # Use stochastic instead of RSI
            
            # Order Flow (if available)
            "order_flow_imbalance_threshold": 0.6,  # 60/40 imbalance ratio
            "use_order_flow": False,           # Whether to use order flow data
            
            # Entry Criteria
            "min_ribbon_slope": 0.0001,        # Minimum slope of EMA ribbon for trend confirmation
            
            # Exit Criteria
            "profit_target_percent": 0.1,      # Fixed profit target (%)
            "stop_loss_percent": 0.05,         # Initial stop loss (%)
            "max_trade_duration_seconds": 120, # Time cutoff (2 minutes)
            "consecutive_adverse_bars": 2,     # Adaptive exit on consecutive adverse bars
            
            # Position Sizing & Risk Controls
            "risk_per_trade": 0.001,           # 0.1% of equity
            "max_concurrent_trades": 2,        # Maximum concurrent trades
            "daily_loss_limit_percent": 0.5,   # Daily drawdown stop (% of equity)
            
            # Order Execution
            "use_limit_orders": True,          # Use limit orders
            "use_market_orders_fallback": True, # Fall back to market orders if needed
            "max_spread_percent": 0.02,        # Maximum allowable spread (%)
            "max_latency_ms": 200,             # Maximum acceptable latency (ms)
            
            # Operational Rules
            "news_blackout_minutes": 5,        # No scalps X min before/after news
            "max_spread_multiplier": 2.0,      # Pause if spread > X times average
            "session_reset_time": "09:30",     # Time to reset session calculations
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Scalping strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "ema_periods": [[4, 7, 11], [5, 8, 13], [6, 9, 15]],
            "vwap_sigma": [0.3, 0.5, 0.7],
            "volume_ma_period": [3, 5, 7],
            "volume_multiplier": [1.5, 1.8, 2.0, 2.2],
            "stoch_k_period": [3, 5, 7],
            "stoch_d_period": [2, 3, 4],
            "stoch_slowing": [2, 3, 4],
            "stoch_lower_bound": [15, 20, 25],
            "stoch_upper_bound": [75, 80, 85],
            "rsi_period": [4, 6, 8],
            "use_stochastic": [True, False],
            "order_flow_imbalance_threshold": [0.55, 0.6, 0.65, 0.7],
            "min_ribbon_slope": [0.00005, 0.0001, 0.0002],
            "profit_target_percent": [0.05, 0.07, 0.1, 0.12],
            "stop_loss_percent": [0.025, 0.05, 0.075],
            "max_trade_duration_seconds": [60, 90, 120, 180],
            "consecutive_adverse_bars": [1, 2, 3],
            "risk_per_trade": [0.0005, 0.001, 0.002],
            "max_concurrent_trades": [1, 2, 3],
            "daily_loss_limit_percent": [0.3, 0.5, 0.7],
            "max_spread_percent": [0.01, 0.02, 0.03],
            "news_blackout_minutes": [3, 5, 7, 10]
        }
    
    def _calculate_ema_ribbon(self, prices: pd.Series, periods: List[int]) -> Dict[int, pd.Series]:
        """
        Calculate EMA ribbon for multiple periods.
        
        Args:
            prices: Series of price data
            periods: List of EMA periods
            
        Returns:
            Dictionary mapping periods to EMA Series
        """
        ema_dict = {}
        for period in periods:
            ema_dict[period] = prices.ewm(span=period, adjust=False).mean()
        return ema_dict
    
    def _calculate_vwap_bands(self, ohlcv_df: pd.DataFrame, sigma: float = 0.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate VWAP and its bands.
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            sigma: Standard deviation multiplier for bands
            
        Returns:
            Tuple of (vwap, upper_band, lower_band)
        """
        typical_price = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
        volume = ohlcv_df['volume']
        
        # Calculate VWAP
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Calculate standard deviation of price
        sq_diff = ((typical_price - vwap) ** 2) * volume
        variance = sq_diff.cumsum() / volume.cumsum()
        std_dev = np.sqrt(variance)
        
        # Calculate bands
        upper_band = vwap + (sigma * std_dev)
        lower_band = vwap - (sigma * std_dev)
        
        return vwap, upper_band, lower_band
    
    def _calculate_stochastic(self, ohlcv_df: pd.DataFrame, k_period: int = 5, 
                             d_period: int = 3, slowing: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            k_period: %K period
            d_period: %D period
            slowing: Slowing period
            
        Returns:
            Tuple of (%K, %D)
        """
        # Calculate %K
        low_min = ohlcv_df['low'].rolling(window=k_period).min()
        high_max = ohlcv_df['high'].rolling(window=k_period).max()
        
        # Prevent division by zero
        denominator = high_max - low_min
        denominator = denominator.replace(0, np.nan)
        
        # Raw %K calculation
        k_fast = 100 * ((ohlcv_df['close'] - low_min) / denominator)
        
        # Apply slowing to get final %K
        k = k_fast.rolling(window=slowing).mean()
        
        # Calculate %D (SMA of %K)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 6) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of price data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        # Calculate price changes
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
    
    def _calculate_ribbon_slope(self, ema_dict: Dict[int, pd.Series], window: int = 3) -> Dict[int, pd.Series]:
        """
        Calculate the slope of each EMA in the ribbon.
        
        Args:
            ema_dict: Dictionary of EMAs
            window: Window size for slope calculation
            
        Returns:
            Dictionary of EMA slopes
        """
        slope_dict = {}
        for period, ema_series in ema_dict.items():
            # Calculate simple slope as current value minus previous value
            slope_dict[period] = ema_series.diff(window) / window
        
        return slope_dict
    
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
        morning_end = self.parameters.get("trading_end_morning", "11:00")
        afternoon_start = self.parameters.get("trading_start_afternoon", "15:00")
        afternoon_end = self.parameters.get("trading_end_afternoon", "16:00")
        
        # Convert to datetime.time objects
        morning_start_time = time(*map(int, morning_start.split(":")))
        morning_end_time = time(*map(int, morning_end.split(":")))
        afternoon_start_time = time(*map(int, afternoon_start.split(":")))
        afternoon_end_time = time(*map(int, afternoon_end.split(":")))
        
        # Check if timestamp is within either trading window
        timestamp_time = timestamp.time()
        
        in_morning_session = morning_start_time <= timestamp_time <= morning_end_time
        in_afternoon_session = afternoon_start_time <= timestamp_time <= afternoon_end_time
        
        return in_morning_session or in_afternoon_session
    
    def calculate_indicators(self, data: Dict[str, Dict[str, pd.DataFrame]], 
                            order_flow_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate scalping indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            order_flow_data: Optional dictionary with order flow data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        ema_periods = self.parameters.get("ema_periods", [5, 8, 13])
        vwap_sigma = self.parameters.get("vwap_sigma", 0.5)
        volume_ma_period = self.parameters.get("volume_ma_period", 5)
        stoch_k_period = self.parameters.get("stoch_k_period", 5)
        stoch_d_period = self.parameters.get("stoch_d_period", 3)
        stoch_slowing = self.parameters.get("stoch_slowing", 3)
        rsi_period = self.parameters.get("rsi_period", 6)
        use_stochastic = self.parameters.get("use_stochastic", True)
        
        for symbol, timeframe_data in data.items():
            try:
                # Ensure we have 1-min and 5-min data
                if "1min" not in timeframe_data or "5min" not in timeframe_data:
                    logger.warning(f"Missing required timeframe data for {symbol}")
                    continue
                
                # Get the dataframes
                df_1min = timeframe_data["1min"]
                df_5min = timeframe_data["5min"]
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df_1min.columns for col in required_columns) or \
                   not all(col in df_5min.columns for col in required_columns):
                    logger.warning(f"Required OHLCV columns not found for {symbol}")
                    continue
                
                # Calculate EMA ribbon (5, 8, 13 on 1-min)
                ema_ribbon = self._calculate_ema_ribbon(df_1min['close'], ema_periods)
                
                # Calculate EMA ribbon slope
                ema_ribbon_slope = self._calculate_ribbon_slope(ema_ribbon)
                
                # Calculate VWAP and bands (±0.5 σ on 1-min)
                vwap, upper_band, lower_band = self._calculate_vwap_bands(df_1min, sigma=vwap_sigma)
                
                # Calculate Volume MA (5-min)
                volume_ma = df_1min['volume'].rolling(window=volume_ma_period).mean()
                
                # Calculate oscillator (Stochastic or RSI)
                stoch_k, stoch_d = None, None
                rsi = None
                
                if use_stochastic:
                    stoch_k, stoch_d = self._calculate_stochastic(
                        df_1min, k_period=stoch_k_period, d_period=stoch_d_period, slowing=stoch_slowing
                    )
                else:
                    rsi = self._calculate_rsi(df_1min['close'], period=rsi_period)
                
                # Calculate spread if available
                spread = None
                if 'ask' in df_1min.columns and 'bid' in df_1min.columns:
                    spread = (df_1min['ask'] - df_1min['bid']) / ((df_1min['ask'] + df_1min['bid']) / 2)
                
                # Store indicators
                indicators[symbol] = {
                    "ema_ribbon": {period: pd.DataFrame({f"ema_{period}": ema}) for period, ema in ema_ribbon.items()},
                    "ema_ribbon_slope": {period: pd.DataFrame({f"ema_slope_{period}": slope}) for period, slope in ema_ribbon_slope.items()},
                    "vwap": pd.DataFrame({"vwap": vwap}),
                    "vwap_upper": pd.DataFrame({"vwap_upper": upper_band}),
                    "vwap_lower": pd.DataFrame({"vwap_lower": lower_band}),
                    "volume_ma": pd.DataFrame({"volume_ma": volume_ma}),
                }
                
                # Add oscillator
                if use_stochastic and stoch_k is not None and stoch_d is not None:
                    indicators[symbol]["stoch_k"] = pd.DataFrame({"stoch_k": stoch_k})
                    indicators[symbol]["stoch_d"] = pd.DataFrame({"stoch_d": stoch_d})
                elif rsi is not None:
                    indicators[symbol]["rsi"] = pd.DataFrame({"rsi": rsi})
                
                # Add spread if available
                if spread is not None:
                    indicators[symbol]["spread"] = pd.DataFrame({"spread": spread})
                
                # Add order flow data if available
                if self.parameters.get("use_order_flow", False) and order_flow_data and symbol in order_flow_data:
                    of_data = order_flow_data[symbol]
                    if 'bid_volume' in of_data.columns and 'ask_volume' in of_data.columns:
                        total_volume = of_data['bid_volume'] + of_data['ask_volume']
                        bid_ratio = of_data['bid_volume'] / total_volume
                        ask_ratio = of_data['ask_volume'] / total_volume
                        
                        indicators[symbol]["bid_ratio"] = pd.DataFrame({"bid_ratio": bid_ratio})
                        indicators[symbol]["ask_ratio"] = pd.DataFrame({"ask_ratio": ask_ratio})
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def calculate_position_size(self, equity: float, stop_distance_percent: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            equity: Total equity
            stop_distance_percent: Stop distance as percentage
            
        Returns:
            Position size
        """
        risk_per_trade = self.parameters.get("risk_per_trade", 0.001)
        risk_amount = equity * risk_per_trade
        
        # Position size formula: size = (equity × 0.001) / stop_distance
        position_size = risk_amount / stop_distance_percent if stop_distance_percent > 0 else 0
        
        return position_size
    
    def is_news_blackout_period(self, timestamp: datetime, news_times: List[datetime]) -> bool:
        """
        Check if timestamp is within the news blackout period.
        
        Args:
            timestamp: Current timestamp
            news_times: List of scheduled news release times
            
        Returns:
            Boolean indicating if in news blackout period
        """
        news_blackout_minutes = self.parameters.get("news_blackout_minutes", 5)
        
        for news_time in news_times:
            blackout_start = news_time - pd.Timedelta(minutes=news_blackout_minutes)
            blackout_end = news_time + pd.Timedelta(minutes=news_blackout_minutes)
            
            if blackout_start <= timestamp <= blackout_end:
                return True
                
        return False
    
    def is_spread_too_wide(self, current_spread: float, average_spread: float) -> bool:
        """
        Check if the current spread is too wide compared to the average.
        
        Args:
            current_spread: Current spread
            average_spread: Average spread
            
        Returns:
            Boolean indicating if spread is too wide
        """
        max_spread_multiplier = self.parameters.get("max_spread_multiplier", 2.0)
        max_spread_percent = self.parameters.get("max_spread_percent", 0.02)
        
        # Check if spread exceeds the maximum percentage
        if current_spread > max_spread_percent:
            return True
            
        # Check if spread exceeds the multiplier of the average
        if average_spread > 0 and current_spread > (average_spread * max_spread_multiplier):
            return True
            
        return False
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]], equity: float,
                         order_flow_data: Optional[Dict[str, pd.DataFrame]] = None,
                         news_times: Optional[List[datetime]] = None,
                         current_trades_count: int = 0) -> Dict[str, Signal]:
        """
        Generate scalping signals based on short-term price action.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            equity: Current equity value
            order_flow_data: Optional dictionary with order flow data
            news_times: Optional list of scheduled news release times
            current_trades_count: Number of currently open trades
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Check if max concurrent trades reached
        max_concurrent_trades = self.parameters.get("max_concurrent_trades", 2)
        if current_trades_count >= max_concurrent_trades:
            return {}
            
        # Calculate indicators
        indicators = self.calculate_indicators(data, order_flow_data)
        
        # Get parameters
        ema_periods = self.parameters.get("ema_periods", [5, 8, 13])
        min_ribbon_slope = self.parameters.get("min_ribbon_slope", 0.0001)
        volume_multiplier = self.parameters.get("volume_multiplier", 1.8)
        stoch_lower_bound = self.parameters.get("stoch_lower_bound", 20)
        stoch_upper_bound = self.parameters.get("stoch_upper_bound", 80)
        use_stochastic = self.parameters.get("use_stochastic", True)
        order_flow_imbalance_threshold = self.parameters.get("order_flow_imbalance_threshold", 0.6)
        use_order_flow = self.parameters.get("use_order_flow", False)
        profit_target_percent = self.parameters.get("profit_target_percent", 0.1) / 100  # Convert to decimal
        stop_loss_percent = self.parameters.get("stop_loss_percent", 0.05) / 100  # Convert to decimal
        
        # Generate signals
        signals = {}
        
        for symbol, timeframe_data in data.items():
            try:
                # Skip if we don't have indicators for this symbol
                if symbol not in indicators:
                    continue
                
                # Get the dataframes
                df_1min = timeframe_data["1min"]
                
                # Get the latest data
                latest_1min = df_1min.iloc[-1]
                latest_timestamp = latest_1min.name if isinstance(latest_1min.name, datetime) else datetime.now()
                
                # Skip if not within trading hours
                if not self.is_within_trading_hours(latest_timestamp):
                    continue
                
                # Skip if in news blackout period
                if news_times and self.is_news_blackout_period(latest_timestamp, news_times):
                    continue
                
                # Get latest price
                latest_price = latest_1min['close']
                latest_volume = latest_1min['volume']
                
                # Get latest indicator values
                symbol_indicators = indicators[symbol]
                
                # Get latest EMA ribbon values
                latest_ema_values = {period: symbol_indicators["ema_ribbon"][period].iloc[-1][f"ema_{period}"] 
                                     for period in ema_periods}
                
                # Get latest EMA ribbon slope values
                latest_slope_values = {period: symbol_indicators["ema_ribbon_slope"][period].iloc[-1][f"ema_slope_{period}"] 
                                       for period in ema_periods}
                
                # Get latest VWAP and bands
                latest_vwap = symbol_indicators["vwap"].iloc[-1]["vwap"]
                latest_upper_band = symbol_indicators["vwap_upper"].iloc[-1]["vwap_upper"]
                latest_lower_band = symbol_indicators["vwap_lower"].iloc[-1]["vwap_lower"]
                
                # Get latest volume MA
                latest_volume_ma = symbol_indicators["volume_ma"].iloc[-1]["volume_ma"]
                
                # Get latest oscillator values
                latest_stoch_k = None
                latest_stoch_d = None
                latest_rsi = None
                
                if use_stochastic:
                    if "stoch_k" in symbol_indicators and "stoch_d" in symbol_indicators:
                        latest_stoch_k = symbol_indicators["stoch_k"].iloc[-1]["stoch_k"]
                        latest_stoch_d = symbol_indicators["stoch_d"].iloc[-1]["stoch_d"]
                else:
                    if "rsi" in symbol_indicators:
                        latest_rsi = symbol_indicators["rsi"].iloc[-1]["rsi"]
                
                # Get latest spread
                latest_spread = None
                average_spread = None
                
                if "spread" in symbol_indicators:
                    latest_spread = symbol_indicators["spread"].iloc[-1]["spread"]
                    average_spread = symbol_indicators["spread"].iloc[-10:]["spread"].mean() if len(symbol_indicators["spread"]) >= 10 else None
                    
                    # Skip if spread is too wide
                    if average_spread is not None and self.is_spread_too_wide(latest_spread, average_spread):
                        continue
                
                # Get order flow data if available
                latest_bid_ratio = None
                latest_ask_ratio = None
                
                if use_order_flow and "bid_ratio" in symbol_indicators and "ask_ratio" in symbol_indicators:
                    latest_bid_ratio = symbol_indicators["bid_ratio"].iloc[-1]["bid_ratio"]
                    latest_ask_ratio = symbol_indicators["ask_ratio"].iloc[-1]["ask_ratio"]
                
                # Check signal conditions
                
                # 1. Micro-trend alignment
                # For long: Price above all EMAs, and EMAs in ascending order (shortest on top)
                ema_sorted = sorted(ema_periods)  # Sort from shortest to longest
                long_ema_alignment = all(latest_price > latest_ema_values[period] for period in ema_periods) and \
                                     all(latest_ema_values[ema_sorted[i]] > latest_ema_values[ema_sorted[i+1]] 
                                         for i in range(len(ema_sorted)-1))
                
                # For short: Price below all EMAs, and EMAs in descending order (shortest on bottom)
                short_ema_alignment = all(latest_price < latest_ema_values[period] for period in ema_periods) and \
                                      all(latest_ema_values[ema_sorted[i]] < latest_ema_values[ema_sorted[i+1]] 
                                          for i in range(len(ema_sorted)-1))
                
                # Check EMA ribbon slope (all slopes should be in the same direction and above threshold)
                long_slope_alignment = all(latest_slope_values[period] > min_ribbon_slope for period in ema_periods)
                short_slope_alignment = all(latest_slope_values[period] < -min_ribbon_slope for period in ema_periods)
                
                # 2. VWAP re-test
                # For long: Price touches or briefly crosses lower VWAP band
                long_vwap_retest = latest_price <= latest_lower_band and latest_1min['low'] <= latest_lower_band
                
                # For short: Price touches or briefly crosses upper VWAP band
                short_vwap_retest = latest_price >= latest_upper_band and latest_1min['high'] >= latest_upper_band
                
                # 3. Order flow confirmation
                order_flow_long_confirm = True
                order_flow_short_confirm = True
                
                if use_order_flow and latest_bid_ratio is not None and latest_ask_ratio is not None:
                    order_flow_long_confirm = latest_bid_ratio >= order_flow_imbalance_threshold
                    order_flow_short_confirm = latest_ask_ratio >= order_flow_imbalance_threshold
                
                # 4. Volume surge
                volume_surge = latest_volume >= (volume_multiplier * latest_volume_ma)
                
                # 5. Oscillator filter
                oscillator_long_ok = True
                oscillator_short_ok = True
                
                if use_stochastic and latest_stoch_k is not None:
                    # Ensure stochastic is in neutral zone (not overbought/oversold)
                    oscillator_long_ok = latest_stoch_k > stoch_lower_bound and latest_stoch_k < stoch_upper_bound
                    oscillator_short_ok = latest_stoch_k > stoch_lower_bound and latest_stoch_k < stoch_upper_bound
                elif latest_rsi is not None:
                    # Similar check for RSI
                    oscillator_long_ok = latest_rsi > 30 and latest_rsi < 70
                    oscillator_short_ok = latest_rsi > 30 and latest_rsi < 70
                
                # Generate signal based on conditions
                signal_type = None
                
                # Long signal
                if (long_ema_alignment and long_slope_alignment and  # Micro-trend alignment
                    long_vwap_retest and  # VWAP re-test
                    order_flow_long_confirm and  # Order flow
                    volume_surge and  # Volume surge
                    oscillator_long_ok):  # Oscillator filter
                    
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on signal strength
                    ribbon_strength = sum(latest_slope_values[period] for period in ema_periods) / len(ema_periods) * 1000
                    volume_strength = latest_volume / (latest_volume_ma * volume_multiplier) - 1
                    price_distance = (latest_price - latest_lower_band) / latest_price
                    
                    # Higher confidence with ideal conditions
                    confidence = min(0.9, 0.6 + min(0.1, ribbon_strength) + min(0.1, volume_strength) + 
                                    max(0, 0.1 - price_distance))
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(equity, stop_loss_percent)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price * (1 - stop_loss_percent)
                    take_profit = latest_price * (1 + profit_target_percent)
                
                # Short signal
                elif (short_ema_alignment and short_slope_alignment and  # Micro-trend alignment
                      short_vwap_retest and  # VWAP re-test
                      order_flow_short_confirm and  # Order flow
                      volume_surge and  # Volume surge
                      oscillator_short_ok):  # Oscillator filter
                    
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence based on signal strength
                    ribbon_strength = abs(sum(latest_slope_values[period] for period in ema_periods) / len(ema_periods)) * 1000
                    volume_strength = latest_volume / (latest_volume_ma * volume_multiplier) - 1
                    price_distance = (latest_upper_band - latest_price) / latest_price
                    
                    # Higher confidence with ideal conditions
                    confidence = min(0.9, 0.6 + min(0.1, ribbon_strength) + min(0.1, volume_strength) + 
                                    max(0, 0.1 - price_distance))
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(equity, stop_loss_percent)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price * (1 + stop_loss_percent)
                    take_profit = latest_price * (1 - profit_target_percent)
                
                # Create signal if we have a valid signal type
                if signal_type:
                    max_trade_duration_seconds = self.parameters.get("max_trade_duration_seconds", 120)
                    consecutive_adverse_bars = self.parameters.get("consecutive_adverse_bars", 2)
                    
                    # Metadata for broker
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
                            "strategy_type": "scalping",
                            "max_trade_duration_seconds": max_trade_duration_seconds,
                            "consecutive_adverse_bars": consecutive_adverse_bars,
                            "ema_values": {f"ema_{period}": value for period, value in latest_ema_values.items()},
                            "vwap": latest_vwap,
                            "vwap_band": latest_upper_band if signal_type == SignalType.SELL else latest_lower_band,
                            "volume_surge_ratio": latest_volume / latest_volume_ma if latest_volume_ma > 0 else 1.0,
                            "profit_target_percent": profit_target_percent * 100,  # Convert back to percent for readability
                            "stop_loss_percent": stop_loss_percent * 100,
                            "order_flow_signal": latest_bid_ratio if signal_type == SignalType.BUY else latest_ask_ratio,
                            "spread": latest_spread
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
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
        daily_loss_limit_percent = self.parameters.get("daily_loss_limit_percent", 0.5) / 100  # Convert to decimal
        return daily_pnl <= -(equity * daily_loss_limit_percent)
    
    def should_exit_trade(self, entry_price: float, current_price: float, signal_type: SignalType,
                         entry_time: datetime, current_time: datetime, 
                         consecutive_adverse_bars: int) -> Tuple[bool, str]:
        """
        Check if a trade should be exited based on time or price action.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            signal_type: Original signal type (BUY/SELL)
            entry_time: Entry time
            current_time: Current time
            consecutive_adverse_bars: Number of consecutive adverse bars
            
        Returns:
            Tuple of (should_exit, reason)
        """
        max_trade_duration_seconds = self.parameters.get("max_trade_duration_seconds", 120)
        max_consecutive_adverse = self.parameters.get("consecutive_adverse_bars", 2)
        
        # Check time cutoff
        elapsed_seconds = (current_time - entry_time).total_seconds()
        if elapsed_seconds >= max_trade_duration_seconds:
            return True, "time_cutoff"
        
        # Check consecutive adverse bars
        if consecutive_adverse_bars >= max_consecutive_adverse:
            return True, "adverse_bars"
        
        return False, "" 