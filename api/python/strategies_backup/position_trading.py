#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Trading Strategy Module

This module implements a position trading strategy designed to capture 
medium-to-long-term directional moves by riding multi-day trends, 
using volatility and momentum filters to avoid chop.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
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

class PositionTradingStrategy(StrategyOptimizable):
    """
    Position Trading Strategy designed to capture medium-to-long-term directional moves.
    
    This strategy focuses on riding multi-day trends using weekly and daily timeframes,
    with volatility and momentum filters to avoid chop. It emphasizes high conviction
    setups with wider stops and higher reward targets.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Position Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # Liquid large-caps, ETFs, select futures
            "max_positions": 5,  # Maximum number of concurrent positions
            "min_holding_weeks": 1,  # Minimum holding period in weeks
            "max_holding_weeks": 8,  # Maximum holding period in weeks
            
            # Core Indicators
            "weekly_ema_period": 20,  # Weekly EMA period for macro trend
            "daily_sma_period": 50,  # Daily SMA period for intermediate trend
            "rsi_period": 14,  # RSI period
            "rsi_lower_bound": 40,  # RSI lower bound for entry
            "rsi_upper_bound": 70,  # RSI upper bound for entry
            "adx_period": 14,  # ADX period
            "adx_threshold": 25,  # ADX threshold for trend strength
            "atr_period": 14,  # ATR period
            
            # Entry Criteria
            "pullback_atr_multiple": 1.0,  # ATR multiple for pullback calculation
            "breakout_atr_multiple": 0.5,  # ATR multiple for breakout calculation
            
            # Exit Criteria
            "initial_stop_atr": 1.5,  # Initial stop loss in ATR units
            "profit_target_atr": 3.0,  # Profit target in ATR units
            "trailing_stop_activation_atr": 2.0,  # Activate trailing stop after 2 ATR gain
            "trailing_stop_atr": 1.0,  # Trail at 1 ATR
            
            # Position Sizing & Risk Controls
            "risk_per_trade": 0.01,  # 1% of equity
            "max_exposure": 0.3,  # Max 30% of equity simultaneously
            "max_consecutive_stops": 2,  # Maximum consecutive stop-loss hits
            "cooling_period_weeks": 1,  # Halt new entries for this period after hitting max stops
            
            # Order Execution
            "use_limit_orders": True,  # Use limit orders at signal close
            "use_scaling": False,  # Optional scaling feature
            "scaling_atr_multiple": 1.0,  # ATR multiple for scaling in
            
            # Operational Rules
            "earnings_blackout_days": 3,  # No new positions days before earnings
            "news_filter_days": 1,  # Avoid opening within this many days of major news
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Position Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "weekly_ema_period": [15, 20, 25, 30],
            "daily_sma_period": [40, 50, 60, 70],
            "rsi_period": [10, 14, 21],
            "rsi_lower_bound": [35, 40, 45],
            "rsi_upper_bound": [65, 70, 75],
            "adx_period": [10, 14, 21],
            "adx_threshold": [20, 25, 30],
            "atr_period": [10, 14, 21],
            "pullback_atr_multiple": [0.8, 1.0, 1.2],
            "breakout_atr_multiple": [0.3, 0.5, 0.7],
            "initial_stop_atr": [1.2, 1.5, 1.8, 2.0],
            "profit_target_atr": [2.5, 3.0, 3.5, 4.0],
            "trailing_stop_activation_atr": [1.5, 2.0, 2.5],
            "trailing_stop_atr": [0.75, 1.0, 1.25],
            "risk_per_trade": [0.005, 0.01, 0.015],
            "max_exposure": [0.25, 0.3, 0.35, 0.4],
            "max_consecutive_stops": [1, 2, 3],
            "cooling_period_weeks": [1, 2, 3]
        }
    
    def _calculate_atr(self, ohlcv_df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
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
    
    def _calculate_adx(self, ohlcv_df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        high = ohlcv_df['high']
        low = ohlcv_df['low']
        close = ohlcv_df['close']
        
        # Calculate +DM, -DM
        high_change = high.diff()
        low_change = low.diff()
        
        plus_dm = np.where(
            (high_change > 0) & (high_change > low_change.abs()),
            high_change,
            0
        )
        minus_dm = np.where(
            (low_change < 0) & (low_change.abs() > high_change),
            low_change.abs(),
            0
        )
        
        # Calculate TR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate smoothed TR and DM values
        tr_smoothed = tr.rolling(window=period).mean()
        plus_dm_smoothed = pd.Series(plus_dm).rolling(window=period).mean()
        minus_dm_smoothed = pd.Series(minus_dm).rolling(window=period).mean()
        
        # Calculate +DI, -DI
        plus_di = 100 * plus_dm_smoothed / tr_smoothed
        minus_di = 100 * minus_dm_smoothed / tr_smoothed
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_indicators(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate position trading indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        weekly_ema_period = self.parameters.get("weekly_ema_period", 20)
        daily_sma_period = self.parameters.get("daily_sma_period", 50)
        rsi_period = self.parameters.get("rsi_period", 14)
        adx_period = self.parameters.get("adx_period", 14)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, timeframe_data in data.items():
            try:
                # Ensure we have both daily and weekly data
                if "daily" not in timeframe_data or "weekly" not in timeframe_data:
                    logger.warning(f"Missing required timeframe data for {symbol}")
                    continue
                
                # Get the dataframes
                df_daily = timeframe_data["daily"]
                df_weekly = timeframe_data["weekly"]
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close']
                if not all(col in df_daily.columns for col in required_columns) or \
                   not all(col in df_weekly.columns for col in required_columns):
                    logger.warning(f"Required price columns not found for {symbol}")
                    continue
                
                # Calculate Weekly EMA(20)
                weekly_ema = df_weekly['close'].ewm(span=weekly_ema_period, adjust=False).mean()
                
                # Calculate Daily SMA(50)
                daily_sma = df_daily['close'].rolling(window=daily_sma_period).mean()
                
                # Calculate RSI(14) on daily
                daily_rsi = self._calculate_rsi(df_daily['close'], period=rsi_period)
                
                # Calculate ADX(14) on daily
                daily_adx = self._calculate_adx(df_daily, period=adx_period)
                
                # Calculate ATR(14) on daily
                daily_atr = self._calculate_atr(df_daily, period=atr_period)
                
                # Store indicators
                indicators[symbol] = {
                    "weekly_ema": pd.DataFrame({"ema": weekly_ema}),
                    "daily_sma": pd.DataFrame({"sma": daily_sma}),
                    "daily_rsi": pd.DataFrame({"rsi": daily_rsi}),
                    "daily_adx": pd.DataFrame({"adx": daily_adx}),
                    "daily_atr": pd.DataFrame({"atr": daily_atr})
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def calculate_position_size(self, equity: float, atr: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            equity: Total equity
            atr: Current ATR value
            
        Returns:
            Position size
        """
        risk_per_trade = self.parameters.get("risk_per_trade", 0.01)
        initial_stop_atr = self.parameters.get("initial_stop_atr", 1.5)
        risk_amount = equity * risk_per_trade
        
        # Position size formula: size = (equity × 0.01) / (1.5 × ATR)
        position_size = risk_amount / (initial_stop_atr * atr)
        
        return position_size
    
    def is_within_earnings_blackout(self, timestamp: datetime, earnings_dates: Dict[str, datetime]) -> bool:
        """
        Check if the timestamp is within the earnings blackout period.
        
        Args:
            timestamp: Current timestamp
            earnings_dates: Dictionary mapping symbols to their upcoming earnings dates
            
        Returns:
            Boolean indicating if within earnings blackout period
        """
        if not earnings_dates:
            return False
            
        earnings_blackout_days = self.parameters.get("earnings_blackout_days", 3)
        
        for earnings_date in earnings_dates.values():
            if earnings_date - timedelta(days=earnings_blackout_days) <= timestamp <= earnings_date:
                return True
                
        return False
    
    def is_within_news_filter(self, timestamp: datetime, news_dates: List[datetime]) -> bool:
        """
        Check if the timestamp is within the news filter period.
        
        Args:
            timestamp: Current timestamp
            news_dates: List of important news release timestamps
            
        Returns:
            Boolean indicating if within news filter period
        """
        if not news_dates:
            return False
            
        news_filter_days = self.parameters.get("news_filter_days", 1)
        
        for news_date in news_dates:
            # Check if timestamp is within news_filter_days of any news date
            if abs((timestamp - news_date).days) <= news_filter_days:
                return True
                
        return False
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]], equity: float, 
                         earnings_dates: Optional[Dict[str, Dict[str, datetime]]] = None,
                         news_dates: Optional[List[datetime]] = None) -> Dict[str, Signal]:
        """
        Generate position trading signals based on trend alignment.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            equity: Current equity value
            earnings_dates: Dictionary mapping symbols to their upcoming earnings dates
            news_dates: List of important news release timestamps
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Get parameters
        rsi_lower_bound = self.parameters.get("rsi_lower_bound", 40)
        rsi_upper_bound = self.parameters.get("rsi_upper_bound", 70)
        adx_threshold = self.parameters.get("adx_threshold", 25)
        pullback_atr_multiple = self.parameters.get("pullback_atr_multiple", 1.0)
        breakout_atr_multiple = self.parameters.get("breakout_atr_multiple", 0.5)
        initial_stop_atr = self.parameters.get("initial_stop_atr", 1.5)
        profit_target_atr = self.parameters.get("profit_target_atr", 3.0)
        
        # Generate signals
        signals = {}
        
        for symbol, timeframe_data in data.items():
            try:
                # Skip if we don't have indicators for this symbol
                if symbol not in indicators:
                    continue
                
                # Get the dataframes
                df_daily = timeframe_data["daily"]
                df_weekly = timeframe_data["weekly"]
                
                # Get the latest data
                latest_daily = df_daily.iloc[-1]
                latest_weekly = df_weekly.iloc[-1]
                latest_timestamp = latest_daily.name if isinstance(latest_daily.name, datetime) else datetime.now()
                
                # Skip if within earnings blackout or news filter period
                symbol_earnings = earnings_dates.get(symbol, {}) if earnings_dates else {}
                if (self.is_within_earnings_blackout(latest_timestamp, symbol_earnings) or 
                    self.is_within_news_filter(latest_timestamp, news_dates or [])):
                    continue
                
                # Get latest indicator values
                latest_weekly_ema = indicators[symbol]["weekly_ema"].iloc[-1]["ema"]
                latest_daily_sma = indicators[symbol]["daily_sma"].iloc[-1]["sma"]
                latest_daily_rsi = indicators[symbol]["daily_rsi"].iloc[-1]["rsi"]
                latest_daily_adx = indicators[symbol]["daily_adx"].iloc[-1]["adx"]
                latest_daily_atr = indicators[symbol]["daily_atr"].iloc[-1]["atr"]
                
                # Get previous values to check for bounces and breakouts
                prev_daily = df_daily.iloc[-2] if len(df_daily) > 1 else None
                
                # Latest price
                latest_price = latest_daily['close']
                
                # Check signal conditions
                
                # 1. Macro bias: weekly close > EMA20_Weekly for long, < EMA20_Weekly for short
                long_macro_bias = latest_weekly['close'] > latest_weekly_ema
                short_macro_bias = latest_weekly['close'] < latest_weekly_ema
                
                # 2. Intermediate trend: daily close > SMA50_Daily for long, < SMA50_Daily for short
                long_intermediate_trend = latest_daily['close'] > latest_daily_sma
                short_intermediate_trend = latest_daily['close'] < latest_daily_sma
                
                # 3. Momentum filter: RSI14 ∈ [40, 70]
                momentum_filter = rsi_lower_bound <= latest_daily_rsi <= rsi_upper_bound
                
                # 4. Volatility confirmation: ADX14_Daily > 25
                adx_filter = latest_daily_adx > adx_threshold
                
                # 5a. Pullback version: price dips to ≤ 1 × ATR below SMA50_Daily and bounces
                pullback_level = latest_daily_sma - (pullback_atr_multiple * latest_daily_atr)
                long_pullback = (prev_daily and prev_daily['low'] <= pullback_level and 
                                 latest_daily['close'] > prev_daily['close'])
                
                # 5b. Breakout version: daily close ≥ SMA50_Daily + 0.5 × ATR
                breakout_level = latest_daily_sma + (breakout_atr_multiple * latest_daily_atr)
                long_breakout = latest_daily['close'] >= breakout_level
                
                # 5c. Short pullback: price rises to ≥ 1 × ATR above SMA50_Daily and drops
                short_pullback_level = latest_daily_sma + (pullback_atr_multiple * latest_daily_atr)
                short_pullback = (prev_daily and prev_daily['high'] >= short_pullback_level and 
                                  latest_daily['close'] < prev_daily['close'])
                
                # 5d. Short breakout: daily close ≤ SMA50_Daily - 0.5 × ATR
                short_breakout_level = latest_daily_sma - (breakout_atr_multiple * latest_daily_atr)
                short_breakout = latest_daily['close'] <= short_breakout_level
                
                # Generate signal based on conditions
                signal_type = None
                entry_style = None
                
                # Long signal
                if (long_macro_bias and 
                    long_intermediate_trend and 
                    momentum_filter and 
                    adx_filter and 
                    (long_pullback or long_breakout)):
                    
                    signal_type = SignalType.BUY
                    entry_style = "pullback" if long_pullback else "breakout"
                    
                    # Calculate confidence based on signal strength
                    macro_strength = (latest_weekly['close'] / latest_weekly_ema - 1) * 5
                    trend_strength = (latest_daily['close'] / latest_daily_sma - 1) * 5
                    adx_strength = (latest_daily_adx - adx_threshold) / 25
                    
                    # Higher confidence for pullbacks at support
                    pattern_confidence = 0.3 if long_pullback else 0.25
                    
                    # Combine factors for confidence score (0.6-0.9)
                    confidence = min(0.9, 0.6 + macro_strength * 0.05 + trend_strength * 0.05 + 
                                    adx_strength * 0.1 + pattern_confidence)
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(equity, latest_daily_atr)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price - (initial_stop_atr * latest_daily_atr)
                    take_profit = latest_price + (profit_target_atr * latest_daily_atr)
                
                # Short signal
                elif (short_macro_bias and 
                      short_intermediate_trend and 
                      momentum_filter and 
                      adx_filter and 
                      (short_pullback or short_breakout)):
                    
                    signal_type = SignalType.SELL
                    entry_style = "pullback" if short_pullback else "breakout"
                    
                    # Calculate confidence based on signal strength
                    macro_strength = (1 - latest_weekly['close'] / latest_weekly_ema) * 5
                    trend_strength = (1 - latest_daily['close'] / latest_daily_sma) * 5
                    adx_strength = (latest_daily_adx - adx_threshold) / 25
                    
                    # Higher confidence for pullbacks at resistance
                    pattern_confidence = 0.3 if short_pullback else 0.25
                    
                    # Combine factors for confidence score (0.6-0.9)
                    confidence = min(0.9, 0.6 + macro_strength * 0.05 + trend_strength * 0.05 + 
                                    adx_strength * 0.1 + pattern_confidence)
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(equity, latest_daily_atr)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price + (initial_stop_atr * latest_daily_atr)
                    take_profit = latest_price - (profit_target_atr * latest_daily_atr)
                
                # Create signal if we have a valid signal type
                if signal_type:
                    trailing_stop_activation = self.parameters.get("trailing_stop_activation_atr", 2.0)
                    trailing_stop_value = self.parameters.get("trailing_stop_atr", 1.0)
                    max_holding_weeks = self.parameters.get("max_holding_weeks", 8)
                    
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
                            "weekly_ema": latest_weekly_ema,
                            "daily_sma": latest_daily_sma,
                            "daily_rsi": latest_daily_rsi,
                            "daily_adx": latest_daily_adx,
                            "daily_atr": latest_daily_atr,
                            "entry_style": entry_style,
                            "trailing_stop": True,
                            "trailing_stop_activation": trailing_stop_activation * latest_daily_atr,
                            "trailing_stop_value": trailing_stop_value * latest_daily_atr,
                            "strategy_type": "position_trading",
                            "max_holding_weeks": max_holding_weeks
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def is_position_count_exceeded(self, current_positions: int) -> bool:
        """
        Check if the maximum number of positions is exceeded.
        
        Args:
            current_positions: Number of currently open positions
            
        Returns:
            Boolean indicating if position count is exceeded
        """
        max_positions = self.parameters.get("max_positions", 5)
        return current_positions >= max_positions
    
    def is_exposure_limit_exceeded(self, current_exposure: float) -> bool:
        """
        Check if the maximum exposure limit is exceeded.
        
        Args:
            current_exposure: Current exposure as a fraction of equity
            
        Returns:
            Boolean indicating if exposure limit is exceeded
        """
        max_exposure = self.parameters.get("max_exposure", 0.3)
        return current_exposure >= max_exposure
    
    def is_in_cooling_period(self, consecutive_stops: int, last_stop_date: Optional[datetime] = None) -> bool:
        """
        Check if in cooling period after consecutive stop losses.
        
        Args:
            consecutive_stops: Number of consecutive stop losses
            last_stop_date: Date of the last stop loss
            
        Returns:
            Boolean indicating if in cooling period
        """
        max_consecutive_stops = self.parameters.get("max_consecutive_stops", 2)
        cooling_period_weeks = self.parameters.get("cooling_period_weeks", 1)
        
        if consecutive_stops < max_consecutive_stops:
            return False
            
        if last_stop_date is None:
            return False
            
        cooling_end_date = last_stop_date + timedelta(weeks=cooling_period_weeks)
        return datetime.now() < cooling_end_date 