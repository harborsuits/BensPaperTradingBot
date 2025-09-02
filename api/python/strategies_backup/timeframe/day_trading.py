#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trading Strategy Module

This module implements a day trading strategy focusing on intraday momentum 
breakouts within established micro-trends, using volume and volatility filters 
to avoid false moves.
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

class DayTradingStrategy(StrategyOptimizable):
    """
    Day Trading Strategy designed to capitalize on intraday momentum breakouts.
    
    This strategy identifies breakouts that occur within established micro-trends,
    using volume and volatility filters to avoid false moves. It implements strict 
    risk controls and time constraints to keep drawdowns in check.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Day Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the blueprint
        default_params = {
            # Market Universe & Timeframe
            "symbols": [],  # List of highly liquid large-caps or ETFs
            "signal_timeframe": 5,  # 5-minute bars
            "trend_timeframe": 15,  # 15-minute bars
            "trading_start_time": "10:00",  # Trading window start (ET)
            "trading_end_time": "15:00",  # Trading window end (ET)
            
            # Core Indicators
            "ema_period": 50,  # 15-min EMA period for trend direction
            "volume_ma_period": 20,  # 5-min Volume MA period
            "atr_period": 14,  # 5-min ATR period
            "volume_multiplier": 1.5,  # Volume spike threshold
            
            # Entry Criteria
            "vwap_pullback_threshold": 0.2,  # Price within ±0.2% of VWAP
            "breakout_threshold": 0.15,  # Close ≥ VWAP + 0.15% for long
            
            # Exit Criteria
            "initial_stop_atr": 1.0,  # Initial stop loss in ATR units
            "profit_target_atr": 2.0,  # Profit target in ATR units
            "trailing_stop_activation_atr": 1.0,  # Move to breakeven after 1 ATR profit
            "trailing_stop_atr": 0.5,  # Trail at 0.5 ATR once activated
            "time_based_stop": True,  # Enable time-based stop
            "unwinding_start_time": "15:00",  # Start unwinding 15 min before close
            
            # Position Sizing & Risk Controls
            "risk_per_trade": 0.005,  # 0.5% of total equity
            "max_exposure": 0.1,  # Max 10% of equity simultaneously
            "daily_loss_limit": 0.015,  # Stop trading after 1.5% daily loss
            "max_trades_per_day": 5,  # Maximum trades per day
            
            # Order Execution
            "use_limit_orders": True,  # Prefer limit orders
            "slippage_buffer": 0.0002,  # 0.02% for slippage
            
            # Operational Rules
            "avoid_news_minutes": 10,  # Avoid trades within 10 minutes of news
            "reset_time": "09:30",  # Reset daily counters at market open
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Day Trading strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "ema_period": [40, 50, 60],
            "volume_ma_period": [15, 20, 25],
            "atr_period": [10, 14, 21],
            "volume_multiplier": [1.3, 1.5, 1.7, 2.0],
            "vwap_pullback_threshold": [0.15, 0.2, 0.25, 0.3],
            "breakout_threshold": [0.1, 0.15, 0.2, 0.25],
            "initial_stop_atr": [0.8, 1.0, 1.2],
            "profit_target_atr": [1.5, 2.0, 2.5, 3.0],
            "trailing_stop_activation_atr": [0.8, 1.0, 1.2],
            "trailing_stop_atr": [0.3, 0.5, 0.7],
            "risk_per_trade": [0.003, 0.005, 0.007],
            "max_exposure": [0.08, 0.1, 0.12, 0.15],
            "daily_loss_limit": [0.01, 0.015, 0.02]
        }
    
    def _calculate_vwap(self, ohlcv_df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        if not all(col in ohlcv_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("OHLCV dataframe must contain 'open', 'high', 'low', 'close', and 'volume' columns")
        
        # Calculate typical price
        typical_price = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
        
        # Calculate VWAP
        vwap = (typical_price * ohlcv_df['volume']).cumsum() / ohlcv_df['volume'].cumsum()
        
        return vwap
    
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
    
    def is_within_trading_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is within trading hours.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            Boolean indicating if timestamp is within trading hours
        """
        # Parse trading window times
        start_time_str = self.parameters.get("trading_start_time", "10:00")
        end_time_str = self.parameters.get("trading_end_time", "15:00")
        
        start_hour, start_minute = map(int, start_time_str.split(":"))
        end_hour, end_minute = map(int, end_time_str.split(":"))
        
        start_time = time(start_hour, start_minute)
        end_time = time(end_hour, end_minute)
        
        # Check if timestamp is within trading window
        timestamp_time = timestamp.time()
        return start_time <= timestamp_time <= end_time
    
    def calculate_indicators(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate day trading indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        # Get parameters
        ema_period = self.parameters.get("ema_period", 50)
        volume_ma_period = self.parameters.get("volume_ma_period", 20)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, timeframe_data in data.items():
            try:
                # Ensure we have both 5-min and 15-min data
                if "5min" not in timeframe_data or "15min" not in timeframe_data:
                    logger.warning(f"Missing required timeframe data for {symbol}")
                    continue
                
                # Get the dataframes
                df_5min = timeframe_data["5min"]
                df_15min = timeframe_data["15min"]
                
                # Ensure required columns exist
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df_5min.columns for col in required_columns) or \
                   not all(col in df_15min.columns for col in required_columns):
                    logger.warning(f"Required price columns not found for {symbol}")
                    continue
                
                # Calculate 15-min EMA(50)
                ema_15min = df_15min['close'].ewm(span=ema_period, adjust=False).mean()
                
                # Calculate 5-min VWAP
                vwap_5min = self._calculate_vwap(df_5min)
                
                # Calculate 5-min Volume MA(20)
                volume_ma_5min = df_5min['volume'].rolling(window=volume_ma_period).mean()
                
                # Calculate 5-min ATR(14)
                atr_5min = self._calculate_atr(df_5min, period=atr_period)
                
                # Store indicators
                indicators[symbol] = {
                    "ema_15min": pd.DataFrame({"ema": ema_15min}),
                    "vwap_5min": pd.DataFrame({"vwap": vwap_5min}),
                    "volume_ma_5min": pd.DataFrame({"volume_ma": volume_ma_5min}),
                    "atr_5min": pd.DataFrame({"atr": atr_5min})
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
        risk_per_trade = self.parameters.get("risk_per_trade", 0.005)
        risk_amount = equity * risk_per_trade
        
        # Position size formula: size = (equity × 0.005) / ATR
        position_size = risk_amount / atr
        
        return position_size
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]], equity: float) -> Dict[str, Signal]:
        """
        Generate day trading signals based on momentum breakouts.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data for different timeframes
            equity: Current equity value
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Get parameters
        volume_multiplier = self.parameters.get("volume_multiplier", 1.5)
        vwap_pullback_threshold = self.parameters.get("vwap_pullback_threshold", 0.2) / 100  # Convert to decimal
        breakout_threshold = self.parameters.get("breakout_threshold", 0.15) / 100  # Convert to decimal
        
        # Initial stop and profit target in ATR units
        initial_stop_atr = self.parameters.get("initial_stop_atr", 1.0)
        profit_target_atr = self.parameters.get("profit_target_atr", 2.0)
        
        # Generate signals
        signals = {}
        
        for symbol, timeframe_data in data.items():
            try:
                # Skip if we don't have indicators for this symbol
                if symbol not in indicators:
                    continue
                
                # Get the dataframes
                df_5min = timeframe_data["5min"]
                df_15min = timeframe_data["15min"]
                
                # Get the latest data
                latest_5min = df_5min.iloc[-1]
                latest_15min = df_15min.iloc[-1]
                latest_timestamp = latest_5min.name if isinstance(latest_5min.name, datetime) else datetime.now()
                
                # Skip if not within trading hours
                if not self.is_within_trading_hours(latest_timestamp):
                    continue
                
                # Get latest indicator values
                latest_ema = indicators[symbol]["ema_15min"].iloc[-1]["ema"]
                latest_vwap = indicators[symbol]["vwap_5min"].iloc[-1]["vwap"]
                latest_volume_ma = indicators[symbol]["volume_ma_5min"].iloc[-1]["volume_ma"]
                latest_atr = indicators[symbol]["atr_5min"].iloc[-1]["atr"]
                
                # Latest price
                latest_price = latest_5min['close']
                latest_volume = latest_5min['volume']
                
                # Trend filter: 15-min close > EMA50 for long, < EMA50 for short
                uptrend = latest_15min['close'] > latest_ema
                downtrend = latest_15min['close'] < latest_ema
                
                # Pullback zone: Price within ±0.2% of VWAP
                in_pullback_zone = abs(latest_price / latest_vwap - 1) <= vwap_pullback_threshold
                
                # Volume confirmation: Bar volume ≥ 1.5 × VolumeMA20
                volume_confirmed = latest_volume >= (volume_multiplier * latest_volume_ma)
                
                # Generate signal based on conditions
                signal_type = None
                
                # Long signal
                if (uptrend and  # Trend filter
                    in_pullback_zone and  # Pullback zone
                    latest_price >= latest_vwap * (1 + breakout_threshold) and  # Breakout candle
                    volume_confirmed):  # Volume confirmation
                    
                    signal_type = SignalType.BUY
                    
                    # Calculate confidence based on signal strength
                    trend_strength = (latest_15min['close'] / latest_ema - 1) * 10
                    breakout_strength = (latest_price / latest_vwap - 1) / breakout_threshold
                    volume_strength = latest_volume / (volume_multiplier * latest_volume_ma) - 1
                    
                    # Combine factors for confidence score (0.5-0.9)
                    confidence = min(0.9, 0.5 + trend_strength * 0.1 + breakout_strength * 0.2 + volume_strength * 0.1)
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(equity, latest_atr)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price - (initial_stop_atr * latest_atr)
                    take_profit = latest_price + (profit_target_atr * latest_atr)
                
                # Short signal
                elif (downtrend and  # Trend filter
                      in_pullback_zone and  # Pullback zone
                      latest_price <= latest_vwap * (1 - breakout_threshold) and  # Breakout candle
                      volume_confirmed):  # Volume confirmation
                    
                    signal_type = SignalType.SELL
                    
                    # Calculate confidence based on signal strength
                    trend_strength = (1 - latest_15min['close'] / latest_ema) * 10
                    breakout_strength = (1 - latest_price / latest_vwap) / breakout_threshold
                    volume_strength = latest_volume / (volume_multiplier * latest_volume_ma) - 1
                    
                    # Combine factors for confidence score (0.5-0.9)
                    confidence = min(0.9, 0.5 + trend_strength * 0.1 + breakout_strength * 0.2 + volume_strength * 0.1)
                    
                    # Calculate position size
                    position_size = self.calculate_position_size(equity, latest_atr)
                    
                    # Calculate stop loss and take profit
                    stop_loss = latest_price + (initial_stop_atr * latest_atr)
                    take_profit = latest_price - (profit_target_atr * latest_atr)
                
                # Create signal if we have a valid signal type
                if signal_type:
                    trailing_stop_activation = self.parameters.get("trailing_stop_activation_atr", 1.0)
                    trailing_stop_value = self.parameters.get("trailing_stop_atr", 0.5)
                    
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
                            "ema": latest_ema,
                            "vwap": latest_vwap,
                            "atr": latest_atr,
                            "trailing_stop": True,
                            "trailing_stop_activation": trailing_stop_activation * latest_atr,
                            "trailing_stop_value": trailing_stop_value * latest_atr,
                            "strategy_type": "day_trading",
                            "time_based_stop": self.parameters.get("time_based_stop", True),
                            "unwinding_time": self.parameters.get("unwinding_start_time", "15:00")
                        }
                    )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def is_daily_limit_reached(self, daily_pnl: float, trades_today: int, equity: float) -> bool:
        """
        Check if daily trading limits have been reached.
        
        Args:
            daily_pnl: Current day's P&L
            trades_today: Number of trades executed today
            equity: Total equity
            
        Returns:
            Boolean indicating if daily limit is reached
        """
        daily_loss_limit = self.parameters.get("daily_loss_limit", 0.015)
        max_trades_per_day = self.parameters.get("max_trades_per_day", 5)
        
        # Check if daily loss limit is hit
        loss_limit_hit = daily_pnl <= -(equity * daily_loss_limit)
        
        # Check if max trades limit is hit
        trades_limit_hit = trades_today >= max_trades_per_day
        
        return loss_limit_hit or trades_limit_hit
    
    def should_avoid_news(self, timestamp: datetime, news_times: List[datetime]) -> bool:
        """
        Check if we should avoid trading due to proximity to news events.
        
        Args:
            timestamp: Current timestamp
            news_times: List of news release timestamps
            
        Returns:
            Boolean indicating if trading should be avoided
        """
        avoid_news_minutes = self.parameters.get("avoid_news_minutes", 10)
        
        for news_time in news_times:
            time_diff = abs((timestamp - news_time).total_seconds() / 60)
            if time_diff < avoid_news_minutes:
                return True
        
        return False 