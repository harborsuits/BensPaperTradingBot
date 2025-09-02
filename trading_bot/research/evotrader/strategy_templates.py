"""
Strategy Templates for BensBot-EvoTrader Integration

This module provides template strategies for forex and crypto trading that
can be used as starting points for evolutionary optimization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger("benbot.research.evotrader.templates")

class BaseStrategy:
    """Base class for all strategy templates."""
    
    def __init__(self, parameters=None):
        """
        Initialize strategy with parameters.
        
        Args:
            parameters: Optional dictionary of strategy parameters
        """
        self.name = "BaseStrategy"
        self.description = "Base strategy template"
        
        # Default parameters
        self.parameters = {
            "fast_period": 10,
            "slow_period": 30,
            "signal_threshold": 0.0,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "risk_per_trade_pct": 1.0,
            "use_trailing_stop": False,
            "trailing_stop_pct": 1.0
        }
        
        # Update with provided parameters
        if parameters:
            self.parameters.update(parameters)
        
        # Parameter definitions with constraints for evolution
        self.parameter_definitions = {
            "fast_period": {
                "default": 10,
                "min": 2,
                "max": 50,
                "type": "int",
                "mutable": True
            },
            "slow_period": {
                "default": 30,
                "min": 10,
                "max": 200,
                "type": "int",
                "mutable": True
            },
            "signal_threshold": {
                "default": 0.0,
                "min": -1.0,
                "max": 1.0,
                "type": "float",
                "mutable": True
            },
            "stop_loss_pct": {
                "default": 2.0,
                "min": 0.5,
                "max": 10.0,
                "type": "float",
                "mutable": True
            },
            "take_profit_pct": {
                "default": 4.0,
                "min": 1.0,
                "max": 20.0,
                "type": "float",
                "mutable": True
            },
            "risk_per_trade_pct": {
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "type": "float",
                "mutable": True
            },
            "use_trailing_stop": {
                "default": False,
                "type": "bool",
                "mutable": True
            },
            "trailing_stop_pct": {
                "default": 1.0,
                "min": 0.1,
                "max": 5.0,
                "type": "float",
                "mutable": True
            }
        }
    
    def calculate_indicators(self, data):
        """
        Calculate technical indicators for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        # Default implementation (Moving Average Crossover)
        df = data.copy()
        
        # Extract parameters
        fast_period = self.parameters["fast_period"]
        slow_period = self.parameters["slow_period"]
        
        # Calculate indicators
        df["fast_ma"] = df["close"].rolling(window=fast_period).mean()
        df["slow_ma"] = df["close"].rolling(window=slow_period).mean()
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        
        return df
    
    def generate_signals(self, data):
        """
        Generate trading signals.
        
        Args:
            data: DataFrame with indicators
            
        Returns:
            DataFrame with signals added
        """
        # Default implementation (Moving Average Crossover signals)
        df = data.copy()
        
        # Extract parameters
        signal_threshold = self.parameters["signal_threshold"]
        
        # Generate signals
        df["signal"] = 0  # 0 = no signal, 1 = buy, -1 = sell
        
        # Buy signal: fast MA crosses above slow MA
        df.loc[df["ma_diff"] > signal_threshold, "signal"] = 1
        
        # Sell signal: fast MA crosses below slow MA
        df.loc[df["ma_diff"] < -signal_threshold, "signal"] = -1
        
        return df
    
    def get_position_size(self, price, balance):
        """
        Calculate position size based on risk parameters.
        
        Args:
            price: Current asset price
            balance: Account balance
            
        Returns:
            Position size
        """
        risk_pct = self.parameters["risk_per_trade_pct"] / 100
        stop_loss_pct = self.parameters["stop_loss_pct"] / 100
        
        # Calculate position size based on risk per trade
        risk_amount = balance * risk_pct
        stop_loss_amount = price * stop_loss_pct
        
        # Position size is risk amount divided by stop loss amount
        if stop_loss_amount > 0:
            return risk_amount / stop_loss_amount
        else:
            return 0
    
    def __str__(self):
        """String representation of the strategy."""
        return f"{self.name} - {self.description}"


class ForexTrendStrategy(BaseStrategy):
    """Forex trend-following strategy template."""
    
    def __init__(self, parameters=None):
        """Initialize forex trend strategy."""
        super().__init__(parameters)
        
        self.name = "ForexTrendStrategy"
        self.description = "Forex trend-following strategy using moving averages and RSI"
        
        # Override default parameters
        default_params = {
            "fast_period": 8,
            "slow_period": 21,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
            "signal_threshold": 0.0001,
            "stop_loss_pct": 1.0,
            "take_profit_pct": 2.0,
            "risk_per_trade_pct": 1.0,
            "use_trailing_stop": True,
            "trailing_stop_pct": 0.5
        }
        
        # Update parameters
        self.parameters.update(default_params)
        if parameters:
            self.parameters.update(parameters)
        
        # Add forex-specific parameter definitions
        forex_param_defs = {
            "rsi_period": {
                "default": 14,
                "min": 2,
                "max": 30,
                "type": "int",
                "mutable": True
            },
            "rsi_overbought": {
                "default": 70,
                "min": 60,
                "max": 90,
                "type": "int",
                "mutable": True
            },
            "rsi_oversold": {
                "default": 30,
                "min": 10,
                "max": 40,
                "type": "int",
                "mutable": True
            },
            "atr_period": {
                "default": 14,
                "min": 5,
                "max": 30,
                "type": "int",
                "mutable": True
            },
            "atr_stop_multiplier": {
                "default": 2.0,
                "min": 1.0,
                "max": 5.0,
                "type": "float",
                "mutable": True
            }
        }
        
        # Update parameter definitions
        self.parameter_definitions.update(forex_param_defs)
    
    def calculate_indicators(self, data):
        """
        Calculate forex-specific indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        df = super().calculate_indicators(data)
        
        # Extract parameters
        rsi_period = self.parameters["rsi_period"]
        atr_period = self.parameters["atr_period"]
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        
        if loss.any():
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
        else:
            df["rsi"] = 100
        
        # Calculate ATR (Average True Range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_period).mean()
        
        return df
    
    def generate_signals(self, data):
        """
        Generate forex-specific trading signals.
        
        Args:
            data: DataFrame with indicators
            
        Returns:
            DataFrame with signals added
        """
        df = data.copy()
        
        # Extract parameters
        rsi_overbought = self.parameters["rsi_overbought"]
        rsi_oversold = self.parameters["rsi_oversold"]
        signal_threshold = self.parameters["signal_threshold"]
        
        # Initialize signal column
        df["signal"] = 0
        
        # Trend condition: MA crossover
        trend_up = df["fast_ma"] > df["slow_ma"]
        trend_down = df["fast_ma"] < df["slow_ma"]
        
        # RSI conditions
        rsi_buy = df["rsi"] < rsi_oversold
        rsi_sell = df["rsi"] > rsi_overbought
        
        # Generate signals with confirmation
        # Buy: Uptrend and RSI oversold
        df.loc[(trend_up) & (rsi_buy) & (df["ma_diff"] > signal_threshold), "signal"] = 1
        
        # Sell: Downtrend and RSI overbought
        df.loc[(trend_down) & (rsi_sell) & (df["ma_diff"] < -signal_threshold), "signal"] = -1
        
        return df
    
    def calculate_stop_loss(self, data, entry_price, direction):
        """
        Calculate stop loss based on ATR.
        
        Args:
            data: DataFrame with indicators
            entry_price: Entry price
            direction: Trade direction (1 for long, -1 for short)
            
        Returns:
            Stop loss price
        """
        # Get latest ATR value
        atr = data["atr"].iloc[-1]
        
        # Extract parameters
        atr_multiplier = self.parameters["atr_stop_multiplier"]
        
        # Calculate stop loss
        if direction > 0:  # Long position
            return entry_price - (atr * atr_multiplier)
        else:  # Short position
            return entry_price + (atr * atr_multiplier)


class CryptoVolatilityStrategy(BaseStrategy):
    """Crypto volatility-based strategy template."""
    
    def __init__(self, parameters=None):
        """Initialize crypto volatility strategy."""
        super().__init__(parameters)
        
        self.name = "CryptoVolatilityStrategy"
        self.description = "Crypto volatility-based strategy using Bollinger Bands and MACD"
        
        # Override default parameters
        default_params = {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
            "volume_ma_period": 20,
            "min_volume_factor": 1.5,
            "stop_loss_pct": 3.0,
            "take_profit_pct": 6.0,
            "risk_per_trade_pct": 2.0,
            "use_trailing_stop": True,
            "trailing_stop_pct": 1.5
        }
        
        # Update parameters
        self.parameters.update(default_params)
        if parameters:
            self.parameters.update(parameters)
        
        # Add crypto-specific parameter definitions
        crypto_param_defs = {
            "bb_period": {
                "default": 20,
                "min": 5,
                "max": 50,
                "type": "int",
                "mutable": True
            },
            "bb_std_dev": {
                "default": 2.0,
                "min": 1.0,
                "max": 4.0,
                "type": "float",
                "mutable": True
            },
            "macd_fast_period": {
                "default": 12,
                "min": 5,
                "max": 30,
                "type": "int",
                "mutable": True
            },
            "macd_slow_period": {
                "default": 26,
                "min": 15,
                "max": 50,
                "type": "int",
                "mutable": True
            },
            "macd_signal_period": {
                "default": 9,
                "min": 3,
                "max": 20,
                "type": "int",
                "mutable": True
            },
            "volume_ma_period": {
                "default": 20,
                "min": 5,
                "max": 50,
                "type": "int",
                "mutable": True
            },
            "min_volume_factor": {
                "default": 1.5,
                "min": 1.0,
                "max": 3.0,
                "type": "float",
                "mutable": True
            }
        }
        
        # Update parameter definitions
        self.parameter_definitions.update(crypto_param_defs)
    
    def calculate_indicators(self, data):
        """
        Calculate crypto-specific indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        df = data.copy()
        
        # Extract parameters
        bb_period = self.parameters["bb_period"]
        bb_std_dev = self.parameters["bb_std_dev"]
        macd_fast = self.parameters["macd_fast_period"]
        macd_slow = self.parameters["macd_slow_period"]
        macd_signal = self.parameters["macd_signal_period"]
        vol_period = self.parameters["volume_ma_period"]
        
        # Calculate Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=bb_period).mean()
        df["bb_std"] = df["close"].rolling(window=bb_period).std()
        df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * bb_std_dev)
        df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * bb_std_dev)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        
        # Calculate MACD
        df["macd_fast_ema"] = df["close"].ewm(span=macd_fast, adjust=False).mean()
        df["macd_slow_ema"] = df["close"].ewm(span=macd_slow, adjust=False).mean()
        df["macd_line"] = df["macd_fast_ema"] - df["macd_slow_ema"]
        df["macd_signal"] = df["macd_line"].ewm(span=macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd_line"] - df["macd_signal"]
        
        # Calculate volume indicators
        df["volume_ma"] = df["volume"].rolling(window=vol_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]
        
        return df
    
    def generate_signals(self, data):
        """
        Generate crypto-specific trading signals.
        
        Args:
            data: DataFrame with indicators
            
        Returns:
            DataFrame with signals added
        """
        df = data.copy()
        
        # Extract parameters
        min_volume_factor = self.parameters["min_volume_factor"]
        
        # Initialize signal column
        df["signal"] = 0
        
        # Volume condition
        high_volume = df["volume_ratio"] >= min_volume_factor
        
        # MACD conditions
        macd_buy = (df["macd_histogram"] > 0) & (df["macd_histogram"].shift(1) <= 0)
        macd_sell = (df["macd_histogram"] < 0) & (df["macd_histogram"].shift(1) >= 0)
        
        # Bollinger Band conditions
        bb_buy = df["close"] <= df["bb_lower"]
        bb_sell = df["close"] >= df["bb_upper"]
        
        # Generate signals
        # Buy: Price at lower band + MACD crossover + High volume
        df.loc[(bb_buy) & (macd_buy) & (high_volume), "signal"] = 1
        
        # Sell: Price at upper band + MACD crossover + High volume
        df.loc[(bb_sell) & (macd_sell) & (high_volume), "signal"] = -1
        
        return df


def create_strategy_template(asset_class="forex", strategy_type="default", parameters=None):
    """
    Create a strategy template for the specified asset class.
    
    Args:
        asset_class: "forex" or "crypto"
        strategy_type: Strategy type within the asset class
        parameters: Optional parameters dictionary
        
    Returns:
        Strategy instance
    """
    if asset_class.lower() == "forex":
        if strategy_type == "trend":
            return ForexTrendStrategy(parameters)
        else:
            # Default forex strategy
            return ForexTrendStrategy(parameters)
            
    elif asset_class.lower() == "crypto":
        if strategy_type == "volatility":
            return CryptoVolatilityStrategy(parameters)
        else:
            # Default crypto strategy
            return CryptoVolatilityStrategy(parameters)
    
    else:
        # Default to base strategy
        return BaseStrategy(parameters)
