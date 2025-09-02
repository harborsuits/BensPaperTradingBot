#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VWAP Pullback Day Trading Strategy Module

This module implements a day trading strategy focused on pullbacks to VWAP
with strict timing, trend filtering, and risk management rules.
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

class VWAPPullbackStrategy(StrategyOptimizable):
    """
    VWAP Pullback Day Trading Strategy.
    
    This strategy focuses on intraday pullbacks to VWAP with trend filtering,
    strict time-based rules, and risk management parameters. It uses a combination
    of 5-minute and 15-minute timeframes for signal generation.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize VWAP Pullback Day Trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            "ema_period": 50,                 # EMA period for trend filter (15-min)
            "vwap_deviation_entry": 0.002,    # Max deviation from VWAP for entry setup (± 0.2%)
            "vwap_breakout_threshold": 0.0015, # Min breakout beyond VWAP (0.15%)
            "volume_threshold": 1.5,          # Volume multiple vs 20-period MA
            "volume_ma_period": 20,           # Period for volume moving average
            "atr_period": 14,                 # ATR period for volatility measurement
            "stop_loss_atr_multiple": 1.0,    # ATR multiple for stop loss
            "take_profit_atr_multiple": 2.0,  # ATR multiple for take profit (1:2 risk/reward)
            "trailing_stop_activation": 1.0,  # ATR multiple to activate trailing stop
            "trailing_stop_distance": 0.5,    # ATR multiple for trailing stop distance
            "max_trades_per_day": 5,          # Maximum trades per day
            "risk_per_trade": 0.005,          # Risk per trade (0.5% of equity)
            "max_exposure": 0.10,             # Maximum total exposure (10% of equity)
            "daily_drawdown_limit": 0.015,    # Daily drawdown limit (-1.5% of equity)
            "trading_start_time": "10:00",    # Trading session start (ET)
            "trading_end_time": "15:00",      # Trading session end (ET)
            "exit_buffer_minutes": 15,        # Exit before session close buffer
            "entry_buffer_minutes": 10        # Avoid trading after market open buffer
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Set appropriate timeframes for this strategy
        if metadata is None or not hasattr(metadata, 'timeframes'):
            self.timeframes = [
                TimeFrame.MINUTE_5,
                TimeFrame.MINUTE_15
            ]
        
        # Initialize daily trading state
        self.reset_daily_state()
        
        logger.info(f"Initialized VWAP Pullback Day Trading strategy: {name}")
    
    def reset_daily_state(self):
        """Reset daily trading state variables."""
        self.trades_today = 0
        self.daily_pnl = 0.0
        self.current_exposure = 0.0
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "ema_period": [40, 50, 60],
            "vwap_deviation_entry": [0.001, 0.002, 0.003],
            "vwap_breakout_threshold": [0.001, 0.0015, 0.002],
            "volume_threshold": [1.3, 1.5, 1.8],
            "volume_ma_period": [15, 20, 25],
            "atr_period": [10, 14, 20],
            "stop_loss_atr_multiple": [0.8, 1.0, 1.2],
            "take_profit_atr_multiple": [1.6, 2.0, 2.4],
            "trailing_stop_activation": [0.8, 1.0, 1.2],
            "trailing_stop_distance": [0.3, 0.5, 0.7],
            "max_trades_per_day": [3, 5, 7],
            "risk_per_trade": [0.003, 0.005, 0.007],
            "max_exposure": [0.08, 0.10, 0.15],
            "daily_drawdown_limit": [0.01, 0.015, 0.02],
            "exit_buffer_minutes": [10, 15, 20]
        }
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with VWAP values
        """
        if 'volume' not in df.columns:
            logger.warning("Volume data not available for VWAP calculation")
            return pd.Series(index=df.index)
        
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Reset cumulative values at the start of each trading day
        day_index = df.index.date
        day_change = day_index != day_index.shift(1)
        
        # Calculate VWAP components
        df_vwap = pd.DataFrame(index=df.index)
        df_vwap['typical_price'] = typical_price
        df_vwap['volume'] = df['volume']
        df_vwap['day_change'] = day_change
        
        # Calculate cumulative values for each day
        df_vwap['cumulative_tpv'] = df_vwap.apply(
            lambda x: x['typical_price'] * x['volume'] if x['day_change'] 
            else x['typical_price'] * x['volume'] + df_vwap['cumulative_tpv'].shift(1),
            axis=1
        )
        
        df_vwap['cumulative_volume'] = df_vwap.apply(
            lambda x: x['volume'] if x['day_change'] 
            else x['volume'] + df_vwap['cumulative_volume'].shift(1),
            axis=1
        )
        
        # Calculate VWAP
        vwap = df_vwap['cumulative_tpv'] / df_vwap['cumulative_volume']
        
        return vwap
    
    def _is_within_trading_hours(self, timestamp: datetime) -> bool:
        """
        Check if the timestamp is within allowed trading hours.
        
        Args:
            timestamp: Datetime to check
            
        Returns:
            Boolean indicating if time is within trading hours
        """
        if not timestamp:
            return False
        
        # Parse trading hours
        start_str = self.parameters.get("trading_start_time", "10:00")
        end_str = self.parameters.get("trading_end_time", "15:00")
        entry_buffer = self.parameters.get("entry_buffer_minutes", 10)
        exit_buffer = self.parameters.get("exit_buffer_minutes", 15)
        
        # Convert to datetime.time objects
        start_parts = [int(x) for x in start_str.split(":")]
        end_parts = [int(x) for x in end_str.split(":")]
        
        trading_start = time(start_parts[0], start_parts[1])
        trading_end = time(end_parts[0], end_parts[1])
        
        # Calculate market open with buffer
        market_open = time(9, 30)  # Standard market open (ET)
        market_open_buffer = time(9, 30 + entry_buffer)
        
        # Calculate market close with buffer
        market_close = time(16, 0)  # Standard market close (ET)
        exit_hour, exit_minute = divmod(60 * market_close.hour + market_close.minute - exit_buffer, 60)
        market_close_buffer = time(int(exit_hour), int(exit_minute))
        
        # Check current time
        current_time = timestamp.time()
        
        # Ensure we're after market open buffer and before market close buffer
        if current_time < market_open_buffer:
            return False
            
        if current_time > market_close_buffer:
            return False
        
        # Check if we're within trading window
        return trading_start <= current_time <= trading_end
    
    def _check_daily_limits(self, equity: float) -> bool:
        """
        Check if we've hit any daily trading limits.
        
        Args:
            equity: Current account equity
            
        Returns:
            Boolean indicating if we can take more trades
        """
        max_trades = self.parameters.get("max_trades_per_day", 5)
        max_exposure = self.parameters.get("max_exposure", 0.10)
        drawdown_limit = self.parameters.get("daily_drawdown_limit", 0.015)
        
        # Check trade count
        if self.trades_today >= max_trades:
            logger.info(f"Maximum daily trades reached: {self.trades_today}/{max_trades}")
            return False
        
        # Check exposure
        if self.current_exposure >= max_exposure * equity:
            logger.info(f"Maximum exposure reached: {self.current_exposure:.2f}/{max_exposure * equity:.2f}")
            return False
        
        # Check drawdown
        if self.daily_pnl <= -drawdown_limit * equity:
            logger.info(f"Daily drawdown limit hit: {self.daily_pnl:.2f}/{-drawdown_limit * equity:.2f}")
            return False
        
        return True
    
    def calculate_indicators(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Calculate indicators for all timeframes for all symbols.
        
        Args:
            data: Dictionary mapping symbols to dictionaries of timeframes to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol and timeframe
        """
        indicators = {}
        
        # Get parameters
        ema_period = self.parameters.get("ema_period", 50)
        volume_ma_period = self.parameters.get("volume_ma_period", 20)
        atr_period = self.parameters.get("atr_period", 14)
        
        for symbol, timeframes in data.items():
            indicators[symbol] = {}
            
            # Process 5-minute timeframe
            if TimeFrame.MINUTE_5 in timeframes:
                df_5min = timeframes[TimeFrame.MINUTE_5]
                
                # Ensure required columns exist
                if not all(col in df_5min.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    logger.warning(f"Required columns not found for {symbol} on 5-minute timeframe")
                    continue
                
                try:
                    # Calculate VWAP
                    vwap = self._calculate_vwap(df_5min)
                    
                    # Calculate Volume MA
                    volume_ma = df_5min['volume'].rolling(window=volume_ma_period).mean()
                    
                    # Calculate ATR for volatility assessment
                    high_low = df_5min['high'] - df_5min['low']
                    high_close_prev = np.abs(df_5min['high'] - df_5min['close'].shift(1))
                    low_close_prev = np.abs(df_5min['low'] - df_5min['close'].shift(1))
                    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                    atr = tr.rolling(window=atr_period).mean()
                    
                    # Calculate relative volume
                    relative_volume = df_5min['volume'] / volume_ma
                    
                    # Calculate distance from VWAP
                    vwap_distance = (df_5min['close'] - vwap) / vwap
                    
                    # Store 5-min indicators
                    indicators[symbol][TimeFrame.MINUTE_5] = {
                        "vwap": pd.DataFrame({"vwap": vwap}),
                        "volume_ma": pd.DataFrame({"volume_ma": volume_ma}),
                        "atr": pd.DataFrame({"atr": atr}),
                        "relative_volume": pd.DataFrame({"relative_volume": relative_volume}),
                        "vwap_distance": pd.DataFrame({"vwap_distance": vwap_distance})
                    }
                
                except Exception as e:
                    logger.error(f"Error calculating 5-min indicators for {symbol}: {e}")
            
            # Process 15-minute timeframe
            if TimeFrame.MINUTE_15 in timeframes:
                df_15min = timeframes[TimeFrame.MINUTE_15]
                
                # Ensure required columns exist
                if 'close' not in df_15min.columns:
                    logger.warning(f"Required columns not found for {symbol} on 15-minute timeframe")
                    continue
                
                try:
                    # Calculate EMA-50 for trend filter
                    ema_50 = df_15min['close'].ewm(span=ema_period, adjust=False).mean()
                    
                    # Store 15-min indicators
                    indicators[symbol][TimeFrame.MINUTE_15] = {
                        "ema_50": pd.DataFrame({"ema_50": ema_50})
                    }
                
                except Exception as e:
                    logger.error(f"Error calculating 15-min indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, Dict[str, pd.DataFrame]], indicators: Optional[Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = None, equity: float = 100000.0) -> Dict[str, Signal]:
        """
        Generate VWAP pullback day trading signals.
        
        Args:
            data: Dictionary mapping symbols to dictionaries of timeframes to DataFrames with OHLCV data
            indicators: Pre-calculated indicators (optional, will be computed if not provided)
            equity: Account equity for position sizing
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Check if we can make more trades today
        if not self._check_daily_limits(equity):
            return {}
        
        # Calculate indicators if not provided
        if indicators is None:
            indicators = self.calculate_indicators(data)
        
        # Get parameters
        vwap_deviation = self.parameters.get("vwap_deviation_entry", 0.002)
        vwap_breakout = self.parameters.get("vwap_breakout_threshold", 0.0015)
        volume_threshold = self.parameters.get("volume_threshold", 1.5)
        stop_loss_atr_multiple = self.parameters.get("stop_loss_atr_multiple", 1.0)
        take_profit_atr_multiple = self.parameters.get("take_profit_atr_multiple", 2.0)
        risk_per_trade = self.parameters.get("risk_per_trade", 0.005)
        
        # Generate signals
        signals = {}
        
        for symbol in data.keys():
            try:
                # Skip if we don't have all required timeframes
                if TimeFrame.MINUTE_5 not in data[symbol] or TimeFrame.MINUTE_15 not in data[symbol]:
                    continue
                
                # Skip if we don't have indicators for all timeframes
                if TimeFrame.MINUTE_5 not in indicators[symbol] or TimeFrame.MINUTE_15 not in indicators[symbol]:
                    continue
                
                # Get latest data
                latest_5min = data[symbol][TimeFrame.MINUTE_5].iloc[-1]
                latest_15min = data[symbol][TimeFrame.MINUTE_15].iloc[-1]
                
                # Get latest timestamp and check trading hours
                latest_timestamp = latest_5min.name if isinstance(latest_5min.name, datetime) else datetime.now()
                if not self._is_within_trading_hours(latest_timestamp):
                    logger.debug(f"Outside trading hours for {symbol}: {latest_timestamp}")
                    continue
                
                # Get latest indicator values for 5-min timeframe
                latest_vwap = indicators[symbol][TimeFrame.MINUTE_5]["vwap"].iloc[-1]["vwap"]
                latest_volume_ma = indicators[symbol][TimeFrame.MINUTE_5]["volume_ma"].iloc[-1]["volume_ma"]
                latest_atr = indicators[symbol][TimeFrame.MINUTE_5]["atr"].iloc[-1]["atr"]
                latest_rel_volume = indicators[symbol][TimeFrame.MINUTE_5]["relative_volume"].iloc[-1]["relative_volume"]
                latest_vwap_distance = indicators[symbol][TimeFrame.MINUTE_5]["vwap_distance"].iloc[-1]["vwap_distance"]
                
                # Get latest indicator values for 15-min timeframe (trend filter)
                latest_ema_50 = indicators[symbol][TimeFrame.MINUTE_15]["ema_50"].iloc[-1]["ema_50"]
                
                # Check if we have valid indicator data
                if np.isnan(latest_vwap) or np.isnan(latest_ema_50) or np.isnan(latest_atr) or np.isnan(latest_rel_volume):
                    continue
                
                # Determine trend direction based on 15-min EMA-50
                trend_up = latest_15min['close'] > latest_ema_50
                
                # Check for VWAP pullback setup
                vwap_pullback = abs(latest_vwap_distance) <= vwap_deviation
                
                # Check for previous bar's pullback to VWAP if we have enough history
                prev_vwap_pullback = False
                if len(indicators[symbol][TimeFrame.MINUTE_5]["vwap_distance"]) > 1:
                    prev_vwap_distance = indicators[symbol][TimeFrame.MINUTE_5]["vwap_distance"].iloc[-2]["vwap_distance"]
                    prev_vwap_pullback = abs(prev_vwap_distance) <= vwap_deviation
                
                # Check for momentum confirmation beyond VWAP on current bar
                if trend_up:
                    momentum_breakout = latest_vwap_distance >= vwap_breakout
                else:
                    momentum_breakout = latest_vwap_distance <= -vwap_breakout
                
                # Check volume confirmation
                volume_confirmation = latest_rel_volume >= volume_threshold
                
                # Generate signal based on strategy conditions
                signal_type = None
                
                # LONG signal (pullback to VWAP on previous bar, breakout on current bar)
                if (trend_up and prev_vwap_pullback and momentum_breakout and volume_confirmation):
                    signal_type = SignalType.BUY
                
                # SHORT signal (pullback to VWAP on previous bar, breakdown on current bar)
                elif (not trend_up and prev_vwap_pullback and momentum_breakout and volume_confirmation):
                    signal_type = SignalType.SELL
                
                # If valid signal is found, create Signal object
                if signal_type:
                    # Calculate position size based on ATR and risk per trade
                    stop_loss = latest_5min['close'] - (latest_atr * stop_loss_atr_multiple) if signal_type == SignalType.BUY else \
                                latest_5min['close'] + (latest_atr * stop_loss_atr_multiple)
                    
                    take_profit = latest_5min['close'] + (latest_atr * take_profit_atr_multiple) if signal_type == SignalType.BUY else \
                                 latest_5min['close'] - (latest_atr * take_profit_atr_multiple)
                    
                    # Calculate position size (risk-based)
                    risk_amount = equity * risk_per_trade
                    stop_distance = abs(latest_5min['close'] - stop_loss)
                    position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                    
                    # Calculate confidence level based on multiple factors
                    # 1. Trend strength
                    if signal_type == SignalType.BUY:
                        trend_strength = (latest_15min['close'] / latest_ema_50 - 1) * 20
                    else:
                        trend_strength = (1 - latest_15min['close'] / latest_ema_50) * 20
                    trend_conf = min(0.3, max(0.1, trend_strength))
                    
                    # 2. Volume strength
                    volume_conf = min(0.3, (latest_rel_volume - volume_threshold) * 0.2 + 0.1)
                    
                    # 3. VWAP breakout strength
                    breakout_strength = abs(latest_vwap_distance) / vwap_breakout
                    breakout_conf = min(0.3, max(0.1, breakout_strength * 0.2))
                    
                    # Calculate overall confidence
                    confidence = min(0.9, trend_conf + volume_conf + breakout_conf)
                    
                    # Create signal
                    signals[symbol] = Signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        price=latest_5min['close'],
                        timestamp=latest_timestamp,
                        confidence=confidence,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        quantity=position_size,
                        timeframe=TimeFrame.MINUTE_5,
                        metadata={
                            "vwap": latest_vwap,
                            "ema_50_15min": latest_ema_50,
                            "atr": latest_atr,
                            "relative_volume": latest_rel_volume,
                            "vwap_distance": latest_vwap_distance,
                            "trailing_stop_activation": latest_atr * self.parameters.get("trailing_stop_activation", 1.0),
                            "trailing_stop_distance": latest_atr * self.parameters.get("trailing_stop_distance", 0.5),
                            "risk_per_trade": risk_per_trade,
                            "strategy_type": "vwap_pullback"
                        }
                    )
                    
                    # Update daily state
                    self.trades_today += 1
                    
                    # Log signal details
                    logger.info(f"Generated VWAP pullback {signal_type} signal for {symbol} at {latest_timestamp}")
                    logger.info(f"Entry: {latest_5min['close']:.2f}, Stop: {stop_loss:.2f}, Target: {take_profit:.2f}")
                    logger.info(f"Position size: {position_size:.2f}, Daily trades: {self.trades_today}/{self.parameters.get('max_trades_per_day', 5)}")
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals 