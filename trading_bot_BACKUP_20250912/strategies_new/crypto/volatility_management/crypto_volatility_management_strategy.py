#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Volatility Management Strategy

This strategy focuses on managing positions based on market volatility conditions,
dynamically adjusting position sizes, entry/exit points, and risk parameters.
It can operate in three distinct modes:
1. Volatility breakout trading
2. Volatility-based position sizing
3. Volatility regime-adaptive trading

The strategy monitors multiple volatility indicators across timeframes to
determine optimal trade parameters and risk exposure in different market conditions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
import math
from enum import Enum

# Import base strategy
from trading_bot.strategies_new.crypto.base.crypto_base_strategy import CryptoBaseStrategy, CryptoSession
from trading_bot.strategies_new.factory.strategy_factory import register_strategy
from trading_bot.event_system.event import Event
from trading_bot.position_management.position import Position

# Configure logger
logger = logging.getLogger(__name__)

# Define volatility regime enum
class VolatilityRegime(Enum):
    VERY_LOW = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    VERY_HIGH = 4
    EXTREME = 5

@register_strategy(
    name="CryptoVolatilityManagementStrategy",
    category="crypto",
    description="A strategy that adapts trading parameters based on market volatility conditions and can trade volatility breakouts",
    parameters={
        # Strategy mode
        "strategy_mode": {
            "type": "str",
            "default": "adaptive",
            "enum": ["breakout", "position_sizing", "adaptive"],
            "description": "Trading mode: breakout (trade volatility breakouts), position_sizing (dynamic sizing based on volatility), adaptive (adjust all parameters)"
        },
        "primary_timeframe": {
            "type": "str",
            "default": "1h",
            "description": "Primary timeframe for analysis"
        },
        "volatility_timeframes": {
            "type": "list",
            "default": ["5m", "15m", "1h", "4h", "1d"],
            "description": "Timeframes to monitor for volatility analysis"
        },
        
        # Volatility calculation parameters
        "volatility_windows": {
            "type": "list",
            "default": [14, 30, 60, 90],
            "description": "Lookback periods for volatility calculations"
        },
        "preferred_volatility_metric": {
            "type": "str",
            "default": "atr_pct",
            "enum": ["atr_pct", "std_dev", "parkinson", "garman_klass", "range_pct"],
            "description": "Preferred volatility measurement method"
        },
        "atr_period": {
            "type": "int",
            "default": 14,
            "description": "Period for ATR calculation"
        },
        "bollinger_period": {
            "type": "int",
            "default": 20,
            "description": "Period for Bollinger Bands calculation"
        },
        "bollinger_std": {
            "type": "float",
            "default": 2.0,
            "description": "Standard deviation multiplier for Bollinger Bands"
        },
        
        # Volatility regime thresholds
        "vol_regime_thresholds": {
            "type": "dict",
            "default": {
                "very_low": 0.5,    # 50% of average volatility
                "low": 0.75,        # 75% of average volatility
                "normal": 1.0,      # average volatility
                "high": 1.5,        # 150% of average volatility
                "very_high": 2.5,   # 250% of average volatility
                "extreme": 4.0      # 400% of average volatility
            },
            "description": "Thresholds for defining volatility regimes as multiples of average volatility"
        },
        "vol_lookback_days": {
            "type": "int",
            "default": 90,
            "description": "Days to look back for average volatility calculation"
        },
        
        # Volatility breakout parameters
        "breakout_trigger_std": {
            "type": "float",
            "default": 2.5,
            "description": "Standard deviation threshold to trigger a breakout trade"
        },
        "breakout_confirmation_periods": {
            "type": "int",
            "default": 3,
            "description": "Number of periods to confirm a volatility breakout"
        },
        "breakout_reset_periods": {
            "type": "int",
            "default": 24,
            "description": "Number of periods after which a breakout signal resets"
        },
        "min_consolidation_periods": {
            "type": "int",
            "default": 12,
            "description": "Minimum number of periods of low volatility before considering a breakout valid"
        },
        
        # Position sizing parameters
        "base_position_size": {
            "type": "float",
            "default": 0.02,
            "description": "Base position size as percentage of account (normal volatility)"
        },
        "min_position_size": {
            "type": "float",
            "default": 0.005,
            "description": "Minimum position size as percentage of account"
        },
        "max_position_size": {
            "type": "float",
            "default": 0.05,
            "description": "Maximum position size as percentage of account"
        },
        "volatility_size_adjustment": {
            "type": "bool",
            "default": True,
            "description": "Whether to adjust position size based on volatility"
        },
        "volatility_multiplier_map": {
            "type": "dict",
            "default": {
                "very_low": 1.5,    # Larger positions in very low volatility
                "low": 1.2,         # Slightly larger positions in low volatility
                "normal": 1.0,      # Normal position size in normal volatility
                "high": 0.7,        # Smaller positions in high volatility
                "very_high": 0.4,   # Much smaller positions in very high volatility
                "extreme": 0.2      # Tiny positions in extreme volatility
            },
            "description": "Position size multipliers for different volatility regimes"
        },
        
        # Stop loss & take profit parameters
        "base_stop_atr_multiplier": {
            "type": "float",
            "default": 1.5,
            "description": "Base ATR multiplier for stop loss placement"
        },
        "base_target_atr_multiplier": {
            "type": "float",
            "default": 3.0,
            "description": "Base ATR multiplier for take profit placement"
        },
        "dynamic_stop_loss": {
            "type": "bool",
            "default": True,
            "description": "Whether to adjust stop loss based on volatility regime"
        },
        "stop_loss_volatility_map": {
            "type": "dict",
            "default": {
                "very_low": 2.0,    # Wider stops in very low volatility
                "low": 1.7,         # Slightly wider stops in low volatility
                "normal": 1.5,      # Normal stop distance in normal volatility
                "high": 1.2,        # Tighter stops in high volatility
                "very_high": 1.0,   # Much tighter stops in very high volatility
                "extreme": 0.8      # Even tighter stops in extreme volatility
            },
            "description": "Stop loss ATR multipliers for different volatility regimes"
        },
        
        # Entry/exit parameters
        "adjust_entry_points": {
            "type": "bool",
            "default": True,
            "description": "Whether to adjust entry points based on volatility"
        },
        "adjust_exit_points": {
            "type": "bool",
            "default": True,
            "description": "Whether to adjust exit points based on volatility"
        },
        
        # Trend confirmation parameters
        "trend_confirmation_needed": {
            "type": "bool",
            "default": True,
            "description": "Whether to require trend confirmation for trades"
        },
        "trend_indicators": {
            "type": "list",
            "default": ["ema", "macd", "adx"],
            "description": "Indicators to use for trend confirmation"
        },
        
        # Risk management parameters
        "max_open_trades": {
            "type": "dict",
            "default": {
                "very_low": 5,      # More trades in very low volatility
                "low": 4,           # Several trades in low volatility
                "normal": 3,        # Normal number of trades in normal volatility
                "high": 2,          # Fewer trades in high volatility
                "very_high": 1,     # Just one trade in very high volatility
                "extreme": 0        # No new trades in extreme volatility
            },
            "description": "Maximum number of open trades per volatility regime"
        },
        "reduce_exposure_high_vol": {
            "type": "bool",
            "default": True,
            "description": "Whether to automatically reduce exposure in high volatility environments"
        }
    }
)
class CryptoVolatilityManagementStrategy(CryptoBaseStrategy):
    """
    A strategy that adapts position sizing, entry/exit criteria, and risk parameters
    based on market volatility conditions. Can also trade volatility breakouts directly.
    """
    
    def __init__(self, session: CryptoSession, parameters: Dict[str, Any] = None):
        """
        Initialize the Crypto Volatility Management Strategy.
        
        Args:
            session: The trading session
            parameters: Strategy parameters
        """
        super().__init__(session, parameters)
        
        # Initialize volatility tracking
        self.volatility_history = {}  # {timeframe: DataFrame with volatility metrics}
        self.volatility_metrics = {}  # {timeframe: {metric: current_value}}
        self.volatility_regime = VolatilityRegime.NORMAL  # Current volatility regime
        self.regime_history = []  # List of (timestamp, regime) tuples
        self.average_volatility = None  # Average volatility over lookback period
        
        # Breakout tracking
        self.consolidation_start = None  # When current consolidation began
        self.last_breakout_time = None  # When the last breakout occurred
        self.breakout_triggered = False  # Whether a breakout trade is currently active
        self.breakout_entry_price = None  # Entry price of breakout trade
        self.breakout_direction = None  # Direction of breakout trade
        
        # Trade parameters
        self.current_position_size_multiplier = 1.0  # Current position size adjustment
        self.current_stop_multiplier = self.parameters["base_stop_atr_multiplier"]
        self.current_target_multiplier = self.parameters["base_target_atr_multiplier"]
        self.max_trades_allowed = self.parameters["max_open_trades"]["normal"]
        
        # Initialize historical data and indicators
        self.historical_data = {}  # {timeframe: DataFrame}
        
        # Register event handlers
        self._register_events()
        
        logger.info(f"Initialized Crypto Volatility Management Strategy in {self.parameters['strategy_mode']} mode")
    
    def _register_events(self) -> None:
        """
        Register for relevant events for the strategy.
        """
        # Register for market data events
        self.event_bus.subscribe("MARKET_DATA_UPDATE", self._on_market_data_update)
        
        # Register for timeframe events for all monitored timeframes
        volatility_timeframes = self.parameters["volatility_timeframes"]
        for timeframe in volatility_timeframes:
            self.event_bus.subscribe(f"TIMEFRAME_{timeframe}", self._on_timeframe_event)
        
        # Register for position/order events
        self.event_bus.subscribe("POSITION_UPDATE", self._on_position_update)
        self.event_bus.subscribe("ORDER_UPDATE", self._on_order_update)
        
        # Register for market regime events
        self.event_bus.subscribe("MARKET_REGIME_UPDATE", self._on_market_regime_update)
    
    def _on_market_data_update(self, event: Event) -> None:
        """
        Handle market data updates.
        
        Args:
            event: Market data update event
        """
        # Extract relevant data
        if not event.data:
            return
            
        symbol = event.data.get("symbol")
        if not symbol or symbol != self.symbol:
            return
            
        # Update market data
        self.market_data = event.data.get("data")
        
        # If we're in breakout mode, continuously check for breakouts
        if self.parameters["strategy_mode"] == "breakout" and not self.breakout_triggered:
            self._check_volatility_breakout()
    
    def _on_timeframe_event(self, event: Event) -> None:
        """
        Handle timeframe events for periodic operations.
        
        Args:
            event: Timeframe event
        """
        if not event.data:
            return
            
        timeframe = event.data.get("timeframe")
        if not timeframe or timeframe not in self.parameters["volatility_timeframes"]:
            return
            
        # Update historical data for this timeframe
        self._update_historical_data(timeframe)
        
        # Calculate volatility metrics for this timeframe
        self._calculate_volatility_metrics(timeframe)
        
        # Determine current volatility regime (using primary timeframe)
        if timeframe == self.parameters["primary_timeframe"]:
            self._determine_volatility_regime()
            self._adjust_parameters_for_regime()
            
            # Log regime changes
            logger.info(f"Current volatility regime: {self.volatility_regime.name}, "
                       f"Position size multiplier: {self.current_position_size_multiplier:.2f}, "
                       f"Stop multiplier: {self.current_stop_multiplier:.2f}")
            
            # Publish volatility update event
            self._publish_volatility_event()
            
            # Check for volatility breakout signals
            if self.parameters["strategy_mode"] == "breakout":
                self._check_volatility_breakout()
    
    def _on_position_update(self, event: Event) -> None:
        """
        Handle position update events.
        
        Args:
            event: Position update event
        """
        # Check if we need to adjust stop loss based on new volatility
        if self.parameters["dynamic_stop_loss"] and event.data.get("symbol") == self.symbol:
            position = self.session.get_position(self.symbol)
            if position and position.is_open:
                self._adjust_stop_loss_for_position(position)
    
    def _on_order_update(self, event: Event) -> None:
        """
        Handle order update events.
        
        Args:
            event: Order update event
        """
        if not event.data:
            return
            
        # Reset breakout flags when breakout trades are closed
        if self.parameters["strategy_mode"] == "breakout" and self.breakout_triggered:
            order = event.data
            if order.get("symbol") == self.symbol and order.get("status") == "FILLED":
                if order.get("reduce_only", False):  # This is a closing order
                    self._reset_breakout_state()
    
    def _on_market_regime_update(self, event: Event) -> None:
        """
        Handle market regime updates.
        
        Args:
            event: Market regime update event
        """
        # Use external market regime information to further adjust our parameters
        if not event.data:
            return
            
        # Extract regime information
        regime = event.data.get("regime_type")
        volatility = event.data.get("volatility")
        
        if volatility in ["high", "very_high", "extreme"] and self.parameters["reduce_exposure_high_vol"]:
            # Consider reducing exposure in highly volatile markets
            position = self.session.get_position(self.symbol)
            if position and position.is_open and position.quantity > 0:
                self._consider_reducing_exposure(volatility)
    
    def _update_historical_data(self, timeframe: str) -> None:
        """
        Update historical data for the specified timeframe.
        
        Args:
            timeframe: Timeframe to update data for
        """
        # Get historical data for this timeframe
        lookback = max(self.parameters["volatility_windows"]) * 3  # Ensure enough data for all calculations
        data = self.session.get_historical_data(self.symbol, timeframe, lookback)
        
        if data is None or data.empty:
            logger.warning(f"No historical data available for {self.symbol} on {timeframe} timeframe")
            return
            
        # Store in historical data dict
        self.historical_data[timeframe] = data
    
    def _calculate_volatility_metrics(self, timeframe: str) -> None:
        """
        Calculate volatility metrics for the specified timeframe.
        
        Args:
            timeframe: Timeframe to calculate metrics for
        """
        if timeframe not in self.historical_data or self.historical_data[timeframe].empty:
            return
            
        data = self.historical_data[timeframe]
        
        # If this is a new timeframe, initialize volatility metrics dict
        if timeframe not in self.volatility_metrics:
            self.volatility_metrics[timeframe] = {}
            
        # Calculate each volatility metric based on configuration
        metrics = self.volatility_metrics[timeframe]
        
        # ATR (Absolute and Percentage)
        atr_period = self.parameters["atr_period"]
        if len(data) >= atr_period:
            metrics["atr"] = self._calculate_atr(data, atr_period).iloc[-1]
            metrics["atr_pct"] = metrics["atr"] / data["close"].iloc[-1] * 100  # As percentage of price
        
        # Standard deviation of returns
        for window in self.parameters["volatility_windows"]:
            if len(data) >= window:
                returns = data["close"].pct_change().dropna()
                metrics[f"std_dev_{window}"] = returns.iloc[-window:].std() * 100  # As percentage
                
                # Annualized volatility (standard deviation * sqrt(periods per year))
                annualizer = self._get_annualizer(timeframe)
                metrics[f"annualized_vol_{window}"] = metrics[f"std_dev_{window}"] * np.sqrt(annualizer)
        
        # Bollinger Bands Width (as volatility indicator)
        bb_period = self.parameters["bollinger_period"]
        bb_std = self.parameters["bollinger_std"]
        if len(data) >= bb_period:
            # Calculate Bollinger Bands
            middle = data["close"].rolling(window=bb_period).mean()
            stddev = data["close"].rolling(window=bb_period).std()
            upper = middle + (stddev * bb_std)
            lower = middle - (stddev * bb_std)
            
            # Calculate width normalized by price
            bb_width = (upper - lower) / middle
            metrics["bb_width"] = bb_width.iloc[-1] * 100  # As percentage
        
        # Range-based volatility (high-low range as % of close)
        metrics["range_pct"] = (data["high"].iloc[-1] - data["low"].iloc[-1]) / data["close"].iloc[-1] * 100
        
        # Parkinson volatility (uses high/low range, more accurate than close-to-close)
        if len(data) >= 30:  # Need reasonable history
            hl_square = np.log(data["high"] / data["low"]) ** 2
            metrics["parkinson"] = np.sqrt(hl_square.iloc[-30:].mean() / (4 * np.log(2))) * 100
        
        # Store historical volatility for analysis
        if timeframe not in self.volatility_history:
            self.volatility_history[timeframe] = pd.DataFrame(index=data.index)
            
        # Add current values to history
        for metric, value in metrics.items():
            self.volatility_history[timeframe].loc[data.index[-1], metric] = value
    
    def _determine_volatility_regime(self) -> None:
        """
        Determine the current volatility regime based on historical averages.
        """
        primary_timeframe = self.parameters["primary_timeframe"]
        if primary_timeframe not in self.volatility_metrics:
            # Default to NORMAL if we don't have data yet
            self.volatility_regime = VolatilityRegime.NORMAL
            return
            
        # Use preferred volatility metric
        preferred_metric = self.parameters["preferred_volatility_metric"]
        if preferred_metric not in self.volatility_metrics[primary_timeframe]:
            # Fall back to ATR percentage if preferred metric is not available
            preferred_metric = "atr_pct"
            if preferred_metric not in self.volatility_metrics[primary_timeframe]:
                # Default to NORMAL if we still don't have data
                self.volatility_regime = VolatilityRegime.NORMAL
                return
        
        # Get current volatility value
        current_volatility = self.volatility_metrics[primary_timeframe][preferred_metric]
        
        # Calculate average volatility if we don't have it yet
        if self.average_volatility is None:
            self._calculate_average_volatility(preferred_metric)
            
        if self.average_volatility is None or self.average_volatility <= 0:
            # Default to NORMAL if we still don't have a valid average
            self.volatility_regime = VolatilityRegime.NORMAL
            return
        
        # Calculate relative volatility (current compared to average)
        relative_volatility = current_volatility / self.average_volatility
        
        # Determine regime based on thresholds
        thresholds = self.parameters["vol_regime_thresholds"]
        
        if relative_volatility <= thresholds["very_low"]:
            new_regime = VolatilityRegime.VERY_LOW
        elif relative_volatility <= thresholds["low"]:
            new_regime = VolatilityRegime.LOW
        elif relative_volatility <= thresholds["normal"]:
            new_regime = VolatilityRegime.NORMAL
        elif relative_volatility <= thresholds["high"]:
            new_regime = VolatilityRegime.HIGH
        elif relative_volatility <= thresholds["very_high"]:
            new_regime = VolatilityRegime.VERY_HIGH
        else:
            new_regime = VolatilityRegime.EXTREME
        
        # Log regime changes
        if new_regime != self.volatility_regime:
            logger.info(f"Volatility regime changed from {self.volatility_regime.name} to {new_regime.name}. "
                       f"Relative volatility: {relative_volatility:.2f}")
            
            # Record regime change in history
            self.regime_history.append((datetime.now(), new_regime))
            
            # Update current regime
            self.volatility_regime = new_regime
    
    def _calculate_average_volatility(self, metric: str) -> None:
        """
        Calculate the average volatility over the lookback period.
        
        Args:
            metric: The volatility metric to use
        """
        primary_timeframe = self.parameters["primary_timeframe"]
        lookback_days = self.parameters["vol_lookback_days"]
        
        # Get historical data for lookback period
        if primary_timeframe not in self.historical_data:
            return
            
        # Must have enough data
        if len(self.volatility_history.get(primary_timeframe, pd.DataFrame())) < 10:
            return
            
        # Get volatility history for the selected metric
        vol_history = self.volatility_history[primary_timeframe]
        if metric not in vol_history.columns:
            return
            
        # Calculate average over the lookback period
        lookback_periods = self._convert_days_to_periods(lookback_days, primary_timeframe)
        lookback_periods = min(lookback_periods, len(vol_history))  # Ensure we don't exceed available data
        
        # Calculate the average, excluding outliers
        volatility_values = vol_history[metric].iloc[-lookback_periods:].values
        if len(volatility_values) > 0:
            # Filter out extreme outliers (outside 3 standard deviations)
            mean = np.mean(volatility_values)
            std = np.std(volatility_values)
            filtered_values = volatility_values[(volatility_values > mean - 3 * std) & 
                                             (volatility_values < mean + 3 * std)]
            
            if len(filtered_values) > 0:
                self.average_volatility = np.mean(filtered_values)
                logger.debug(f"Calculated average {metric} volatility: {self.average_volatility:.4f} "
                           f"from {len(filtered_values)} samples over {lookback_days} days")
    
    def _adjust_parameters_for_regime(self) -> None:
        """
        Adjust strategy parameters based on the current volatility regime.
        """
        # Get current regime
        regime = self.volatility_regime.name.lower()
        
        # Position sizing adjustment
        if self.parameters["volatility_size_adjustment"]:
            multiplier_map = self.parameters["volatility_multiplier_map"]
            if regime in multiplier_map:
                self.current_position_size_multiplier = multiplier_map[regime]
        
        # Stop loss adjustment
        if self.parameters["dynamic_stop_loss"]:
            stop_map = self.parameters["stop_loss_volatility_map"]
            if regime in stop_map:
                self.current_stop_multiplier = stop_map[regime]
        
        # Max trades allowed
        max_trades_map = self.parameters["max_open_trades"]
        if regime in max_trades_map:
            self.max_trades_allowed = max_trades_map[regime]
    
    def _publish_volatility_event(self) -> None:
        """
        Publish a volatility update event for other components.
        """
        primary_timeframe = self.parameters["primary_timeframe"]
        if primary_timeframe not in self.volatility_metrics:
            return
            
        # Prepare event data
        metrics = self.volatility_metrics[primary_timeframe].copy()
        
        event_data = {
            "symbol": self.symbol,
            "timestamp": datetime.now(),
            "regime": self.volatility_regime.name,
            "metrics": metrics,
            "average_volatility": self.average_volatility,
            "position_size_multiplier": self.current_position_size_multiplier,
            "stop_multiplier": self.current_stop_multiplier,
            "max_trades_allowed": self.max_trades_allowed
        }
        
        # Publish event
        self.event_bus.publish("VOLATILITY_UPDATE", event_data)
        
    def _convert_days_to_periods(self, days: int, timeframe: str) -> int:
        """
        Convert a number of days to the equivalent number of periods for a given timeframe.
        
        Args:
            days: Number of days
            timeframe: Timeframe string (e.g., '1h', '15m', '1d')
            
        Returns:
            Number of periods
        """
        # Parse timeframe into value and unit
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            periods_per_day = 24 * 60 / minutes
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            periods_per_day = 24 / hours
        elif timeframe.endswith('d'):
            days_tf = int(timeframe[:-1])
            periods_per_day = 1 / days_tf
        else:
            # Default to daily timeframe
            periods_per_day = 1
            
        return int(days * periods_per_day)
    
    def _get_annualizer(self, timeframe: str) -> float:
        """
        Get the annualizer factor for a given timeframe.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '15m', '1d')
            
        Returns:
            Annualizer factor
        """
        # Trading days per year (approximate)
        trading_days = 365  # For crypto (24/7 markets)
        
        # Parse timeframe into value and unit
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            return trading_days * 24 * 60 / minutes
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            return trading_days * 24 / hours
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            return trading_days / days
        elif timeframe.endswith('w'):
            weeks = int(timeframe[:-1])
            return 52 / weeks
        else:
            # Default to daily
            return trading_days
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: Price data DataFrame with OHLC columns
            period: Lookback period for ATR calculation
            
        Returns:
            Pandas Series with ATR values
        """
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)
        
        # True range calculation
        tr1 = high - low  # Current high - current low
        tr2 = abs(high - close)  # Current high - previous close
        tr3 = abs(low - close)  # Current low - previous close
        
        # True range is the maximum of these three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Average true range
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _check_volatility_breakout(self) -> None:
        """
        Check for volatility breakout signals in breakout mode.
        """
        # Only relevant in breakout mode
        if self.parameters["strategy_mode"] != "breakout":
            return
            
        # Need primary timeframe data
        primary_timeframe = self.parameters["primary_timeframe"]
        if primary_timeframe not in self.historical_data or self.historical_data[primary_timeframe].empty:
            return
        
        # If a breakout is already active, don't look for new ones
        if self.breakout_triggered:
            return
            
        data = self.historical_data[primary_timeframe]
        
        # Check if we have the necessary volatility metrics
        if primary_timeframe not in self.volatility_metrics:
            return
            
        metrics = self.volatility_metrics[primary_timeframe]
        preferred_metric = self.parameters["preferred_volatility_metric"]
        
        if preferred_metric not in metrics:
            # Fall back to Bollinger Band width if available
            if "bb_width" in metrics:
                preferred_metric = "bb_width"
            else:
                # No suitable metric available
                return
        
        current_volatility = metrics[preferred_metric]
        
        # Get historical volatility for this metric
        if primary_timeframe not in self.volatility_history:
            return
            
        vol_history = self.volatility_history[primary_timeframe]
        if preferred_metric not in vol_history.columns:
            return
        
        # Calculate volatility Z-score (how many standard deviations from mean)
        lookback = min(self.parameters["vol_lookback_days"], len(vol_history))
        historical_vol = vol_history[preferred_metric].iloc[-lookback:].dropna()
        
        if len(historical_vol) < 10:  # Need enough history
            return
            
        vol_mean = historical_vol.mean()
        vol_std = historical_vol.std()
        
        if vol_std == 0:  # Avoid division by zero
            return
            
        # Calculate Z-score
        z_score = (current_volatility - vol_mean) / vol_std
        
        # Check if we're in a low volatility consolidation phase or not
        min_consolidation_periods = self.parameters["min_consolidation_periods"]
        recent_vol = vol_history[preferred_metric].iloc[-min_consolidation_periods:]
        consolidation_phase = (recent_vol < vol_mean).mean() >= 0.6  # At least 60% of recent periods are below avg volatility
        
        # Check if we have a volatility breakout
        breakout_trigger = self.parameters["breakout_trigger_std"]
        
        if z_score > breakout_trigger and consolidation_phase:
            # We have a potential breakout
            logger.info(f"Detected volatility breakout on {self.symbol}. Z-score: {z_score:.2f}, ")
            
            # Determine direction based on price movement
            # In volatility breakout, we look at the recent price action to determine direction
            price_change = (data["close"].iloc[-1] / data["close"].iloc[-5] - 1) * 100
            
            if price_change > 0:
                direction = "long"
                logger.info(f"Volatility breakout: LONG direction (price change: {price_change:.2f}%)")
            else:
                direction = "short"
                logger.info(f"Volatility breakout: SHORT direction (price change: {price_change:.2f}%)")
            
            # Set breakout state
            self.breakout_triggered = True
            self.breakout_direction = direction
            self.breakout_entry_price = data["close"].iloc[-1]
            self.last_breakout_time = datetime.now()
            
            # Generate a signal
            self._generate_breakout_signal(direction, z_score, data)
    
    def _generate_breakout_signal(self, direction: str, z_score: float, data: pd.DataFrame) -> None:
        """
        Generate a trading signal based on a volatility breakout.
        
        Args:
            direction: Trade direction ('long' or 'short')
            z_score: Volatility Z-score at breakout
            data: Price data DataFrame
        """
        # Get current price
        current_price = data["close"].iloc[-1]
        
        # Calculate position size based on volatility
        position_size = self.calculate_position_size(direction, data, {})
        
        # Calculate stop loss and take profit
        atr = self._calculate_atr(data, self.parameters["atr_period"]).iloc[-1]
        
        # Base stop distance on ATR (wider for higher z-scores)
        # The more extreme the breakout, the wider the stop to allow for volatility
        stop_distance = atr * max(self.current_stop_multiplier, z_score / 2)
        
        if direction == "long":
            stop_price = current_price - stop_distance
            take_profit = current_price + (stop_distance * self.parameters["base_target_atr_multiplier"])
        else:  # short
            stop_price = current_price + stop_distance
            take_profit = current_price - (stop_distance * self.parameters["base_target_atr_multiplier"])
        
        # Create signal event
        signal = {
            "timestamp": datetime.now(),
            "symbol": self.symbol,
            "strategy": self.__class__.__name__,
            "signal_type": "VOLATILITY_BREAKOUT",
            "direction": direction,
            "confidence": min(0.8, 0.5 + (z_score / 10)),  # Higher confidence for stronger breakouts
            "price": current_price,
            "stop_loss": stop_price,
            "take_profit": take_profit,
            "metadata": {
                "volatility_z_score": z_score,
                "breakout_type": "volatility",
                "regime": self.volatility_regime.name
            },
            "position_size": position_size
        }
        
        # Publish signal event
        self.event_bus.publish("SIGNAL", signal)
        
        # Attempt to open a position
        if self.session.can_open_position(self.symbol, direction):
            self.session.open_position(
                symbol=self.symbol,
                direction=direction,
                quantity=position_size,
                stop_loss=stop_price,
                take_profit=take_profit,
                metadata={
                    "strategy": self.__class__.__name__,
                    "signal_type": "VOLATILITY_BREAKOUT",
                    "volatility_z_score": z_score,
                    "regime": self.volatility_regime.name
                }
            )
    
    def _reset_breakout_state(self) -> None:
        """
        Reset the breakout state after a trade is completed or after reset periods.
        """
        logger.debug("Resetting volatility breakout state")
        self.breakout_triggered = False
        self.breakout_direction = None
        self.breakout_entry_price = None
    
    def _adjust_stop_loss_for_position(self, position: Position) -> None:
        """
        Adjust stop loss for an existing position based on current volatility.
        
        Args:
            position: The position to adjust the stop loss for
        """
        if not position or not position.is_open:
            return
            
        # Check if we're due for an adjustment
        primary_timeframe = self.parameters["primary_timeframe"]
        if primary_timeframe not in self.historical_data or self.historical_data[primary_timeframe].empty:
            return
            
        data = self.historical_data[primary_timeframe]
        
        # Calculate new stop based on current volatility (ATR)
        atr = self._calculate_atr(data, self.parameters["atr_period"]).iloc[-1]
        direction = position.direction
        current_price = data["close"].iloc[-1]
        
        # Calculate stop distance
        stop_distance = atr * self.current_stop_multiplier
        
        # Calculate new stop price
        if direction == "long":
            new_stop = current_price - stop_distance
            # Only update if new stop is higher (tightening) and not too close
            if position.stop_loss and new_stop > position.stop_loss and new_stop < current_price * 0.98:
                logger.info(f"Adjusting stop loss for {self.symbol} from {position.stop_loss} to {new_stop}")
                position.update_stop_loss(new_stop)
        else:  # short
            new_stop = current_price + stop_distance
            # Only update if new stop is lower (tightening) and not too close
            if position.stop_loss and new_stop < position.stop_loss and new_stop > current_price * 1.02:
                logger.info(f"Adjusting stop loss for {self.symbol} from {position.stop_loss} to {new_stop}")
                position.update_stop_loss(new_stop)
    
    def _consider_reducing_exposure(self, volatility_level: str) -> None:
        """
        Consider reducing position size when volatility increases dramatically.
        
        Args:
            volatility_level: Current volatility level ('high', 'very_high', 'extreme')
        """
        if not self.parameters["reduce_exposure_high_vol"]:
            return
            
        position = self.session.get_position(self.symbol)
        if not position or not position.is_open:
            return
            
        # Determine reduction factor based on volatility level
        reduction_factors = {
            "high": 0.25,        # Reduce by 25%
            "very_high": 0.5,   # Reduce by 50%
            "extreme": 0.75     # Reduce by 75%
        }
        
        reduction_factor = reduction_factors.get(volatility_level, 0.25)
        
        # Calculate quantity to reduce
        reduce_qty = position.quantity * reduction_factor
        if reduce_qty <= 0:
            return
            
        # Log planned reduction
        logger.info(f"High volatility ({volatility_level}) detected - reducing {self.symbol} ")
        logger.info(f"position by {reduction_factor * 100:.0f}% ({reduce_qty} units)")
        
        # Reduce position
        self.session.reduce_position(symbol=self.symbol, quantity=reduce_qty)
        
        # Publish risk adjustment event
        event_data = {
            "timestamp": datetime.now(),
            "symbol": self.symbol,
            "strategy": self.__class__.__name__,
            "action": "REDUCE_EXPOSURE",
            "reason": f"HIGH_VOLATILITY_{volatility_level.upper()}",
            "reduction_factor": reduction_factor,
            "quantity_reduced": reduce_qty
        }
        
        self.event_bus.publish("RISK_ADJUSTMENT", event_data)
    
    # Abstract method implementations required by CryptoBaseStrategy
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate indicators for trading strategy.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        if data.empty:
            return {}
            
        # Dictionary to store calculated indicators
        indicators = {}
        
        # Calculate ATR
        atr_period = self.parameters["atr_period"]
        if len(data) >= atr_period:
            indicators["atr"] = self._calculate_atr(data, atr_period)
            indicators["atr_pct"] = indicators["atr"] / data["close"] * 100
        
        # Calculate volatility metrics
        # Standard deviation of returns over multiple windows
        for window in self.parameters["volatility_windows"]:
            if len(data) >= window:
                returns = data["close"].pct_change().dropna()
                indicators[f"std_dev_{window}"] = returns.rolling(window=window).std() * 100
        
        # Bollinger Bands and width
        bb_period = self.parameters["bollinger_period"]
        bb_std = self.parameters["bollinger_std"]
        if len(data) >= bb_period:
            indicators["bb_middle"] = data["close"].rolling(window=bb_period).mean()
            std_dev = data["close"].rolling(window=bb_period).std()
            indicators["bb_upper"] = indicators["bb_middle"] + (std_dev * bb_std)
            indicators["bb_lower"] = indicators["bb_middle"] - (std_dev * bb_std)
            indicators["bb_width"] = (indicators["bb_upper"] - indicators["bb_lower"]) / indicators["bb_middle"] * 100
        
        # Additional trend indicators for confirmation if needed
        if self.parameters["trend_confirmation_needed"]:
            # Add EMA for trend direction
            if "ema" in self.parameters["trend_indicators"]:
                indicators["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
                indicators["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()
                indicators["trend_direction"] = np.where(
                    indicators["ema_20"] > indicators["ema_50"], 1, 
                    np.where(indicators["ema_20"] < indicators["ema_50"], -1, 0)
                )
            
            # Add MACD for trend confirmation
            if "macd" in self.parameters["trend_indicators"]:
                ema12 = data["close"].ewm(span=12, adjust=False).mean()
                ema26 = data["close"].ewm(span=26, adjust=False).mean()
                indicators["macd"] = ema12 - ema26
                indicators["macd_signal"] = indicators["macd"].ewm(span=9, adjust=False).mean()
                indicators["macd_hist"] = indicators["macd"] - indicators["macd_signal"]
            
            # Add ADX for trend strength
            if "adx" in self.parameters["trend_indicators"]:
                # Calculate directional movement
                high_diff = data["high"] - data["high"].shift(1)
                low_diff = data["low"].shift(1) - data["low"]
                
                # Positive/Negative Directional Movement (DM)
                plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
                minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
                
                # Get ATR
                tr = indicators.get("atr", self._calculate_atr(data, 14))
                
                # Calculate Directional Indicators
                # First smooth the DM values
                plus_di = pd.Series(plus_dm).rolling(window=14).mean() / tr * 100
                minus_di = pd.Series(minus_dm).rolling(window=14).mean() / tr * 100
                
                # Calculate Directional Index (DX)
                dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
                
                # Average Directional Index (ADX)
                indicators["adx"] = dx.rolling(window=14).mean()
                indicators["plus_di"] = plus_di
                indicators["minus_di"] = minus_di
        
        # Volatility Z-score (for breakout detection)
        for metric in ["atr_pct", "bb_width"]:
            if metric in indicators and len(indicators[metric].dropna()) > 30:
                rolling_mean = indicators[metric].rolling(window=30).mean()
                rolling_std = indicators[metric].rolling(window=30).std()
                indicators[f"{metric}_z_score"] = (indicators[metric] - rolling_mean) / rolling_std
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on indicators and volatility regime.
        
        Args:
            data: Price data DataFrame
            indicators: Calculated indicators
            
        Returns:
            Dictionary of signals
        """
        if data.empty or not indicators:
            return {}
        
        signals = {}
        
        # Get current volatility regime
        # In a real implementation, this would come from _determine_volatility_regime
        # but for simplicity we'll calculate it here for the current data point
        preferred_metric = self.parameters["preferred_volatility_metric"]
        if f"{preferred_metric}_z_score" in indicators:
            current_z_score = indicators[f"{preferred_metric}_z_score"].iloc[-1]
            
            # For breakout mode, check for volatility breakout
            if self.parameters["strategy_mode"] == "breakout":
                breakout_trigger = self.parameters["breakout_trigger_std"]
                # Look for low volatility followed by spike
                # To simplify, we'll check if recent volatility was low then spiked
                recent_vol_z = indicators[f"{preferred_metric}_z_score"].iloc[-5:-1]  # Last few bars
                was_low_vol = (recent_vol_z < 0).mean() > 0.5  # Mostly below average
                
                if was_low_vol and current_z_score > breakout_trigger:
                    # Breakout detected
                    signals["volatility_breakout"] = True
                    signals["breakout_z_score"] = current_z_score
                    
                    # Determine direction based on recent price action
                    price_change = (data["close"].iloc[-1] / data["close"].iloc[-3] - 1) * 100
                    signals["breakout_direction"] = "long" if price_change > 0 else "short"
                    signals["confidence"] = min(0.8, 0.5 + (current_z_score / 10))
                else:
                    signals["volatility_breakout"] = False
        
        # For adaptive mode, determine if we should be trading based on volatility
        if self.parameters["strategy_mode"] == "adaptive":
            # Get current ATR
            if "atr_pct" in indicators:
                current_atr_pct = indicators["atr_pct"].iloc[-1]
                
                # Adjust signal confidence and position sizing based on volatility
                if self.volatility_regime in [VolatilityRegime.VERY_LOW, VolatilityRegime.LOW]:
                    signals["should_trade"] = True
                    signals["size_multiplier"] = self.current_position_size_multiplier
                    signals["base_confidence"] = 0.7  # Higher confidence in lower volatility
                elif self.volatility_regime == VolatilityRegime.NORMAL:
                    signals["should_trade"] = True
                    signals["size_multiplier"] = 1.0
                    signals["base_confidence"] = 0.6
                elif self.volatility_regime == VolatilityRegime.HIGH:
                    signals["should_trade"] = True
                    signals["size_multiplier"] = 0.7
                    signals["base_confidence"] = 0.5  # Lower confidence in higher volatility
                elif self.volatility_regime == VolatilityRegime.VERY_HIGH:
                    signals["should_trade"] = False  # Consider not trading in very high volatility
                    signals["size_multiplier"] = 0.4
                    signals["base_confidence"] = 0.3
                else:  # EXTREME
                    signals["should_trade"] = False  # Don't trade in extreme volatility
                    signals["size_multiplier"] = 0.2
                    signals["base_confidence"] = 0.2
        
        # Add trend confirmation if required
        if self.parameters["trend_confirmation_needed"]:
            if "trend_direction" in indicators:
                signals["trend_direction"] = indicators["trend_direction"].iloc[-1]
            
            if "adx" in indicators:
                signals["trend_strength"] = indicators["adx"].iloc[-1]
                # Strong trend: ADX > 25
                signals["strong_trend"] = signals.get("trend_strength", 0) > 25
            
            if "macd_hist" in indicators:
                signals["macd_direction"] = 1 if indicators["macd_hist"].iloc[-1] > 0 else -1
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on volatility and risk parameters.
        
        Args:
            direction: Trade direction ('long' or 'short')
            data: Price data DataFrame
            indicators: Calculated indicators
            
        Returns:
            Position size as a float
        """
        # Get account balance
        account_balance = self.session.get_account_balance()
        if account_balance <= 0:
            return 0.0
        
        # Base position size from parameters
        base_size = self.parameters["base_position_size"]
        
        # Apply volatility-based adjustment
        adjusted_size = base_size * self.current_position_size_multiplier
        
        # Ensure within min/max bounds
        min_size = self.parameters["min_position_size"]
        max_size = self.parameters["max_position_size"]
        adjusted_size = max(min_size, min(adjusted_size, max_size))
        
        # Calculate position size in account currency
        position_value = account_balance * adjusted_size
        
        # Convert to quantity based on current price
        current_price = data["close"].iloc[-1] if not data.empty else 0
        if current_price <= 0:
            return 0.0
            
        # Calculate quantity
        quantity = position_value / current_price
        
        # If we have ATR, we can adjust to ensure stop loss is within risk tolerance
        if "atr" in indicators and not data.empty:
            atr = indicators["atr"].iloc[-1]
            stop_distance = atr * self.current_stop_multiplier
            
            # Calculate risk per trade as percentage of account
            risk_pct = base_size * 0.5  # Risk half of the position size percentage
            
            # Maximum quantity based on risk management
            risk_amount = account_balance * risk_pct
            price_risk = stop_distance  # Risk in price terms
            
            if price_risk > 0:
                risk_based_qty = risk_amount / price_risk
                # Use the smaller of our calculations to ensure we don't exceed risk parameters
                quantity = min(quantity, risk_based_qty)
        
        return quantity
    
    def regime_compatibility(self, regime_data: Dict[str, Any]) -> float:
        """
        Calculate compatibility score for a given market regime.
        
        Args:
            regime_data: Market regime data
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        # Default score
        score = 0.5
        
        # Extract regime characteristics
        regime_type = regime_data.get("regime_type", "unknown")
        volatility = regime_data.get("volatility", "normal")
        trend_strength = regime_data.get("trend_strength", 0.5)
        
        # Adjust score based on strategy mode
        if self.parameters["strategy_mode"] == "breakout":
            # Breakout strategy works best in transitions from low to high volatility
            if volatility in ["low", "very_low"] and regime_type in ["ranging", "consolidation"]:
                score = 0.8  # Highly compatible - waiting for breakout
            elif volatility in ["high", "very_high"] and regime_type == "trending":
                score = 0.6  # Moderately compatible - may catch breakouts
            else:
                score = 0.3  # Less compatible
                
        elif self.parameters["strategy_mode"] == "position_sizing":
            # Position sizing strategy works in all regimes as it adapts
            score = 0.7  # Generally compatible
            
            # Especially good in higher trend strength environments
            if trend_strength > 0.7:
                score += 0.2
                
        elif self.parameters["strategy_mode"] == "adaptive":
            # Adaptive strategy works in all regimes
            score = 0.8  # Highly compatible overall
            
            # But performs best in environments that aren't extreme
            if volatility == "extreme":
                score -= 0.3
        
        # Adjust based on key regime characteristics
        # Volatility-based strategies generally don't do well in extremely stable or chaotic markets
        if volatility == "extreme":
            score -= 0.2
        elif volatility in ["normal", "high"]:
            score += 0.1
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
