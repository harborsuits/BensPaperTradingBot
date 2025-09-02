#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day Trading Strategy

A comprehensive day trading strategy for intraday profit capturing.
This strategy focuses on short-term price movements within a single trading day,
with all positions closed before market close.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.options.base.strategy_adjustments import StrategyAdjustments

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="DayTradingStrategy",
    market_type="stocks",
    description="Intraday trading strategy for capturing short-term price movements",
    timeframes=["1m", "5m", "15m", "30m"],
    parameters={
        "session_start_time": {"description": "Trading session start time", "type": "time"},
        "session_end_time": {"description": "Trading session end time", "type": "time"},
        "max_trades_per_day": {"description": "Maximum number of trades per day", "type": "integer"}
    }
)
class DayTradingStrategy(StocksBaseStrategy, StrategyAdjustments):
    """
    Day Trading Strategy
    
    This strategy is designed for intraday trading to capture price movements
    within a single trading day. All positions are closed before market close.
    
    Features:
    - Multiple intraday entry techniques (breakouts, reversals, momentum)
    - Tight risk management with predefined stop losses
    - Profit taking based on intraday support/resistance
    - Time-based position management (reducing size as day progresses)
    - Multiple timeframe analysis for confirmation
    """
    
    def __init__(self, session, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Day Trading strategy.
        
        Args:
            session: Trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize base class
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Strategy identification
            'strategy_name': 'Day Trading',
            'strategy_id': 'day_trading',
            
            # Trading session parameters
            'session_start_time': time(9, 30),  # 9:30 AM
            'session_end_time': time(15, 45),   # 3:45 PM (to ensure closing before 4 PM)
            'max_trades_per_day': 5,
            'min_time_between_trades': 15,      # Minutes
            
            # Entry parameters
            'entry_strategies': ['breakout', 'reversal', 'momentum', 'scalp'],
            'min_volume': 100000,
            'min_price': 5.0,
            'max_price': 500.0,
            'min_atr_percent': 1.0,            # Minimum volatility for day trades
            'min_relative_volume': 1.5,        # Minimum volume relative to avg
            
            # Technical parameters
            'vwap_deviation_entry': 0.5,        # % deviation from VWAP for entry
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'ema_fast_period': 9,
            'ema_slow_period': 20,
            
            # Risk management
            'max_risk_per_trade_pct': 0.5,      # % of portfolio to risk per trade
            'stop_loss_atr_multiple': 1.5,      # Stop loss as multiple of ATR
            'profit_target_atr_multiple': 2.0,  # Profit target as multiple of ATR
            'trailing_stop_activation_pct': 1.0, # % gain before trailing stop activates
            'trailing_stop_atr_multiple': 1.0,   # Trailing stop as multiple of ATR
            
            # Exit parameters
            'time_based_exits': True,           # Whether to use time-based exits
            'reduce_position_after_time': time(14, 30),  # Start reducing after 2:30 PM
            'max_trade_duration': 120,          # Maximum time in minutes for a trade
        }
        
        # Apply defaults for any missing parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
                
        # Initialize strategy state
        self.trades_today = 0
        self.last_trade_time = None
        self.market_open = False
        self.market_open_time = None
        self.market_close_warning = False
        self.pdt_rule_warning_shown = False
        
        # Add PDT rule parameter
        if 'pdt_rule_min_equity' not in self.parameters:
            self.parameters['pdt_rule_min_equity'] = 25000  # $25K minimum for pattern day trading
        
        if 'enforce_pdt_rule' not in self.parameters:
            self.parameters['enforce_pdt_rule'] = True  # Default to enforce PDT rule
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate strategy parameters."""
        try:
            # Check time parameters
            if self.parameters['session_start_time'] >= self.parameters['session_end_time']:
                raise ValueError("Session start time must be before session end time")
                
            # Check trading parameters
            if self.parameters['max_trades_per_day'] < 1:
                raise ValueError("Maximum trades per day must be at least 1")
                
            if self.parameters['min_time_between_trades'] < 1:
                raise ValueError("Minimum time between trades must be at least 1 minute")
                
            # Check risk parameters
            if self.parameters['max_risk_per_trade_pct'] <= 0 or self.parameters['max_risk_per_trade_pct'] > 5:
                raise ValueError("Maximum risk per trade must be between 0 and 5 percent")
                
            if self.parameters['stop_loss_atr_multiple'] <= 0:
                raise ValueError("Stop loss ATR multiple must be positive")
                
            if self.parameters['profit_target_atr_multiple'] <= 0:
                raise ValueError("Profit target ATR multiple must be positive")
                
            # Technical parameter validation
            if self.parameters['ema_fast_period'] >= self.parameters['ema_slow_period']:
                raise ValueError("Fast EMA period must be less than slow EMA period")
                
        except Exception as e:
            logger.error(f"Parameter validation error: {str(e)}")
            raise
            
    def reset_daily(self):
        """Reset daily trading statistics."""
        try:
            self.trades_today = 0
            self.last_trade_time = None
            self.market_open = False
            self.market_open_time = None
            self.market_close_warning = False
            
            logger.info("Day trading strategy daily statistics reset")
            
        except Exception as e:
            logger.error(f"Error resetting daily statistics: {str(e)}")
    
    def check_pdt_rule_compliance(self) -> bool:
        """
        Check if account complies with Pattern Day Trader (PDT) rule.
        
        Returns:
            Boolean indicating whether account complies with PDT rule
        """
        try:
            # Check if PDT rule enforcement is enabled
            if not self.parameters['enforce_pdt_rule']:
                return True
                
            # Get account equity
            if hasattr(self.session, 'get_account_equity'):
                equity = self.session.get_account_equity()
                
                # Check if equity meets minimum requirement
                if equity < self.parameters['pdt_rule_min_equity']:
                    if not self.pdt_rule_warning_shown:
                        logger.warning(f"Account equity (${equity}) is below PDT rule minimum (${self.parameters['pdt_rule_min_equity']})")
                        logger.warning("Day trading activity restricted due to Pattern Day Trader (PDT) rule")
                        logger.warning("To day trade, maintain minimum equity of $25,000 or disable PDT rule enforcement")
                        self.pdt_rule_warning_shown = True
                    return False
                    
                return True
            else:
                # If we can't check equity, assume non-compliance for safety
                logger.warning("Unable to verify account equity for PDT rule compliance")
                return False
                
        except Exception as e:
            logger.error(f"Error checking PDT rule compliance: {str(e)}")
            # Default to not allowing day trading if we encounter an error
            return False
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Day Trading strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        # Start with parent class indicators
        indicators = super().calculate_indicators(data)
        
        if data.empty or len(data) < self.parameters['ema_slow_period']:
            return indicators
        
        try:
            # Calculate VWAP
            if 'volume' in data.columns:
                data['pv'] = data['close'] * data['volume']
                data['cumulative_pv'] = data['pv'].cumsum()
                data['cumulative_volume'] = data['volume'].cumsum()
                indicators['vwap'] = data['cumulative_pv'] / data['cumulative_volume']
            
            # Calculate EMAs
            fast_period = self.parameters['ema_fast_period']
            slow_period = self.parameters['ema_slow_period']
            
            indicators['ema_fast'] = data['close'].ewm(span=fast_period, adjust=False).mean()
            indicators['ema_slow'] = data['close'].ewm(span=slow_period, adjust=False).mean()
            
            # EMA crossover signal
            indicators['ema_crossover'] = np.where(
                indicators['ema_fast'] > indicators['ema_slow'], 
                1, 
                np.where(indicators['ema_fast'] < indicators['ema_slow'], -1, 0)
            )
            
            # Calculate ATR for volatility
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr_14'] = true_range.rolling(window=14).mean()
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            indicators['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Volume analysis
            if 'volume' in data.columns:
                indicators['volume_sma_20'] = data['volume'].rolling(window=20).mean()
                indicators['relative_volume'] = data['volume'] / indicators['volume_sma_20']
            
            # Breakout detection
            high_20 = data['high'].rolling(window=20).max()
            low_20 = data['low'].rolling(window=20).min()
            
            indicators['distance_to_high'] = (high_20 - data['close']) / data['close'] * 100
            indicators['distance_to_low'] = (data['close'] - low_20) / data['close'] * 100
            
            # Intraday momentum
            indicators['intraday_return'] = data['close'] / data['open'] - 1
            
            # Price to VWAP relationship
            if 'vwap' in indicators:
                indicators['price_to_vwap'] = (data['close'] - indicators['vwap']) / indicators['vwap'] * 100
            
            # Rate of change
            indicators['roc_5'] = data['close'].pct_change(periods=5) * 100
            
            # Identify intraday patterns
            patterns = {}
            
            # Breakout pattern
            if len(data) > 30:
                latest_close = data['close'].iloc[-1]
                latest_high = data['high'].iloc[-1]
                prev_high = data['high'].iloc[-2:-20].max()
                
                if latest_high > prev_high:
                    patterns['breakout'] = True
                    patterns['breakout_strength'] = (latest_high - prev_high) / prev_high * 100
                else:
                    patterns['breakout'] = False
            
            # Reversal pattern
            if len(data) > 20:
                latest_close = data['close'].iloc[-1]
                latest_open = data['open'].iloc[-1]
                latest_high = data['high'].iloc[-1]
                latest_low = data['low'].iloc[-1]
                prev_close = data['close'].iloc[-2]
                
                # Bullish reversal: previous down, current up, closed above midpoint
                bullish_reversal = (
                    data['close'].iloc[-3:-1].pct_change().mean() < -0.005 and  # Previous downtrend
                    latest_close > latest_open and  # Current bullish candle
                    latest_close > (latest_high + latest_low) / 2  # Closed above midpoint
                )
                
                # Bearish reversal: previous up, current down, closed below midpoint
                bearish_reversal = (
                    data['close'].iloc[-3:-1].pct_change().mean() > 0.005 and  # Previous uptrend
                    latest_close < latest_open and  # Current bearish candle
                    latest_close < (latest_high + latest_low) / 2  # Closed below midpoint
                )
                
                patterns['bullish_reversal'] = bullish_reversal
                patterns['bearish_reversal'] = bearish_reversal
            
            indicators['patterns'] = patterns
            
        except Exception as e:
            logger.error(f"Error calculating day trading indicators: {str(e)}")
        
        return indicators
