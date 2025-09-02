#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indicator Strategy Base Class

This module implements a configurable technical indicator-based strategy that can
be instantiated from a configuration file.
"""

import logging
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import re

from trading_bot.strategy.base.strategy import Strategy
from trading_bot.core.constants import EventType, SignalDirection

logger = logging.getLogger(__name__)

# Dictionary of supported indicators and their calculation functions
INDICATOR_FUNCTIONS = {
    # Trend indicators
    "SMA": talib.SMA,
    "EMA": talib.EMA,
    "WMA": talib.WMA,
    "DEMA": talib.DEMA,
    "TEMA": talib.TEMA,
    "TRIMA": talib.TRIMA,
    "KAMA": talib.KAMA,
    "MACD": lambda close, **kwargs: talib.MACD(close, 
                                        fastperiod=kwargs.get('fastperiod', 12),
                                        slowperiod=kwargs.get('slowperiod', 26),
                                        signalperiod=kwargs.get('signalperiod', 9)),
    "ADX": talib.ADX,
    "ADXR": talib.ADXR,
    "AROON": talib.AROON,
    "AROONOSC": talib.AROONOSC,
    "BOP": talib.BOP,
    "CCI": talib.CCI,
    "CMO": talib.CMO,
    "DX": talib.DX,
    "MINUS_DI": talib.MINUS_DI,
    "PLUS_DI": talib.PLUS_DI,
    "MINUS_DM": talib.MINUS_DM,
    "PLUS_DM": talib.PLUS_DM,
    "MOM": talib.MOM,
    "ROC": talib.ROC,
    "ROCP": talib.ROCP,
    "ROCR": talib.ROCR,
    "ROCR100": talib.ROCR100,
    "RSI": talib.RSI,
    "STOCH": lambda high, low, close, **kwargs: talib.STOCH(high, low, close,
                                            fastk_period=kwargs.get('fastk_period', 5),
                                            slowk_period=kwargs.get('slowk_period', 3),
                                            slowk_matype=kwargs.get('slowk_matype', 0),
                                            slowd_period=kwargs.get('slowd_period', 3),
                                            slowd_matype=kwargs.get('slowd_matype', 0)),
    "STOCHF": lambda high, low, close, **kwargs: talib.STOCHF(high, low, close,
                                                fastk_period=kwargs.get('fastk_period', 5),
                                                fastd_period=kwargs.get('fastd_period', 3),
                                                fastd_matype=kwargs.get('fastd_matype', 0)),
    "STOCHRSI": lambda close, **kwargs: talib.STOCHRSI(close,
                                            timeperiod=kwargs.get('timeperiod', 14),
                                            fastk_period=kwargs.get('fastk_period', 5),
                                            fastd_period=kwargs.get('fastd_period', 3),
                                            fastd_matype=kwargs.get('fastd_matype', 0)),
    "TRIX": talib.TRIX,
    "ULTOSC": lambda high, low, close, **kwargs: talib.ULTOSC(high, low, close,
                                                timeperiod1=kwargs.get('timeperiod1', 7),
                                                timeperiod2=kwargs.get('timeperiod2', 14),
                                                timeperiod3=kwargs.get('timeperiod3', 28)),
    "WILLR": talib.WILLR,
    
    # Volatility indicators
    "ATR": talib.ATR,
    "NATR": talib.NATR,
    "TRANGE": talib.TRANGE,
    
    # Volume indicators
    "AD": talib.AD,
    "ADOSC": lambda high, low, close, volume, **kwargs: talib.ADOSC(high, low, close, volume,
                                                        fastperiod=kwargs.get('fastperiod', 3),
                                                        slowperiod=kwargs.get('slowperiod', 10)),
    "OBV": talib.OBV,
    
    # Price channel indicators
    "BBANDS": lambda close, **kwargs: talib.BBANDS(close,
                                        timeperiod=kwargs.get('timeperiod', 20),
                                        nbdevup=kwargs.get('nbdevup', 2),
                                        nbdevdn=kwargs.get('nbdevdn', 2),
                                        matype=kwargs.get('matype', 0)),
}

# Define comparison operators
OPERATORS = {
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "crosses_above": lambda a_series, b_series: (a_series.shift(1) <= b_series.shift(1)) & (a_series > b_series),
    "crosses_below": lambda a_series, b_series: (a_series.shift(1) >= b_series.shift(1)) & (a_series < b_series),
    "increasing": lambda series, periods: all(series.diff(1).iloc[-periods:] > 0) if isinstance(series, pd.Series) else False,
    "decreasing": lambda series, periods: all(series.diff(1).iloc[-periods:] < 0) if isinstance(series, pd.Series) else False
}

class IndicatorStrategy(Strategy):
    """
    Configurable strategy based on technical indicators.
    
    This strategy can be dynamically configured with different technical indicators
    and rules to generate trading signals.
    """
    
    def __init__(self, 
                 name: str,
                 config: Dict[str, Any],
                 symbol: str = None):
        """
        Initialize an indicator-based strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration containing indicators and rules
            symbol: Optional symbol to trade (can set later)
        """
        super().__init__(name, config)
        
        # Extract configuration parameters
        self.symbol = symbol or config.get('symbol')
        self.timeframe = config.get('timeframe', '1h')
        self.indicators = config.get('indicators', {})
        self.entry_rules = config.get('entry_rules', [])
        self.exit_rules = config.get('exit_rules', [])
        self.position_sizing = config.get('position_sizing', {'type': 'fixed', 'value': 1.0})
        
        # Strategy state
        self.calculated_indicators = {}
        self.last_update_time = None
        self.current_position = 0
        self.in_position = False
        self.entry_price = None
        self.trade_start_time = None
        
        # Validation
        self._validate_config()
        
        logger.info(f"Initialized indicator strategy: {name} for symbol: {self.symbol}")
    
    def _validate_config(self) -> None:
        """Validate the strategy configuration."""
        # Validate indicators
        for indicator_name, params in self.indicators.items():
            indicator_type = params.get('type')
            if indicator_type not in INDICATOR_FUNCTIONS:
                logger.warning(f"Unsupported indicator type: {indicator_type} in {indicator_name}")
        
        # Validate entry rules
        for rule in self.entry_rules:
            if not self._is_valid_rule(rule):
                logger.warning(f"Invalid entry rule: {rule}")
        
        # Validate exit rules
        for rule in self.exit_rules:
            if not self._is_valid_rule(rule):
                logger.warning(f"Invalid exit rule: {rule}")
    
    def _is_valid_rule(self, rule: Dict[str, Any]) -> bool:
        """Check if a rule is valid."""
        if 'condition' not in rule or not isinstance(rule['condition'], str):
            return False
        
        # Validate entry_signal and exit_signal values
        if 'entry_signal' in rule:
            signal = rule['entry_signal']
            if signal not in [SignalDirection.BUY, SignalDirection.SELL]:
                return False
                
        return True
    
    def _calculate_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Calculate indicators based on configuration.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        if ohlcv_data is None or len(ohlcv_data) < 5:
            logger.warning(f"Insufficient data for calculating indicators for {self.name}")
            return {}
        
        result = {}
        
        # Extract price data
        try:
            open_prices = ohlcv_data['open']
            high_prices = ohlcv_data['high']
            low_prices = ohlcv_data['low']
            close_prices = ohlcv_data['close']
            volume = ohlcv_data.get('volume', pd.Series([0] * len(close_prices)))
        except KeyError as e:
            logger.error(f"Missing required price data: {e}")
            return {}
        
        # Calculate each configured indicator
        for indicator_name, params in self.indicators.items():
            try:
                indicator_type = params.get('type')
                if indicator_type not in INDICATOR_FUNCTIONS:
                    logger.warning(f"Skipping unsupported indicator: {indicator_type}")
                    continue
                
                # Get indicator function
                indicator_func = INDICATOR_FUNCTIONS[indicator_type]
                
                # Extract parameters
                indicator_params = {k: v for k, v in params.items() if k != 'type'}
                
                # Handle different input requirements
                if indicator_type in ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA', 'RSI', 'ROC', 'MOM']:
                    # Most require just close prices and parameters
                    indicator_value = indicator_func(close_prices, **indicator_params)
                elif indicator_type in ['BBANDS', 'MACD', 'STOCHRSI']:
                    # Some need just close prices but return multiple values
                    indicator_value = indicator_func(close_prices, **indicator_params)
                elif indicator_type in ['STOCH', 'STOCHF', 'WILLR']:
                    # Some need high, low, close
                    indicator_value = indicator_func(high_prices, low_prices, close_prices, **indicator_params)
                elif indicator_type in ['ATR', 'NATR', 'ADX', 'ADXR', 'CCI']:
                    # Some need high, low, close
                    indicator_value = indicator_func(high_prices, low_prices, close_prices, **indicator_params)
                elif indicator_type in ['OBV', 'AD', 'ADOSC']:
                    # Some need volume
                    indicator_value = indicator_func(high_prices, low_prices, close_prices, volume, **indicator_params)
                else:
                    # Default to close prices
                    indicator_value = indicator_func(close_prices, **indicator_params)
                
                # Some indicators return multiple series
                if isinstance(indicator_value, tuple):
                    # Handle multi-value indicators (like MACD, BBANDS, STOCH)
                    if indicator_type == 'MACD':
                        result[f"{indicator_name}_macd"] = indicator_value[0]
                        result[f"{indicator_name}_signal"] = indicator_value[1]
                        result[f"{indicator_name}_hist"] = indicator_value[2]
                    elif indicator_type == 'BBANDS':
                        result[f"{indicator_name}_upper"] = indicator_value[0]
                        result[f"{indicator_name}_middle"] = indicator_value[1]
                        result[f"{indicator_name}_lower"] = indicator_value[2]
                    elif indicator_type in ['STOCH', 'STOCHF']:
                        result[f"{indicator_name}_k"] = indicator_value[0]
                        result[f"{indicator_name}_d"] = indicator_value[1]
                    elif indicator_type == 'AROON':
                        result[f"{indicator_name}_down"] = indicator_value[0]
                        result[f"{indicator_name}_up"] = indicator_value[1]
                    else:
                        # Generic handling for other multi-value indicators
                        for i, val in enumerate(indicator_value):
                            result[f"{indicator_name}_{i}"] = val
                else:
                    # Single value indicator
                    result[indicator_name] = indicator_value
                
            except Exception as e:
                logger.error(f"Error calculating indicator {indicator_name}: {str(e)}")
        
        # Add original price data to results
        result['open'] = open_prices
        result['high'] = high_prices
        result['low'] = low_prices
        result['close'] = close_prices
        result['volume'] = volume
        
        return result
    
    def _evaluate_condition(self, condition: str, indicators: Dict[str, Union[pd.Series, pd.DataFrame]]) -> bool:
        """
        Evaluate a condition string against calculated indicators.
        
        Args:
            condition: String condition (e.g., "RSI < 30")
            indicators: Dictionary of calculated indicator values
            
        Returns:
            Boolean result of condition evaluation
        """
        # Safety checks
        if not condition or not indicators:
            return False
        
        try:
            # Parse the comparison operator
            operator_pattern = '|'.join(map(re.escape, OPERATORS.keys()))
            match = re.search(f"(.*?)\\s*({operator_pattern})\\s*(.*)", condition)
            
            if not match:
                logger.error(f"Invalid condition format: {condition}")
                return False
            
            left_operand_str, operator_str, right_operand_str = match.groups()
            
            # Get the operator function
            operator_func = OPERATORS.get(operator_str)
            if not operator_func:
                logger.error(f"Unsupported operator: {operator_str}")
                return False
            
            # Special handling for crosses_above, crosses_below operators
            if operator_str in ["crosses_above", "crosses_below"]:
                # These require series objects
                left_operand = self._get_operand(left_operand_str.strip(), indicators)
                right_operand = self._get_operand(right_operand_str.strip(), indicators)
                
                if not isinstance(left_operand, pd.Series) or not isinstance(right_operand, pd.Series):
                    logger.error(f"Crosses operators require series operands: {condition}")
                    return False
                
                result = operator_func(left_operand, right_operand)
                return result.iloc[-1] if isinstance(result, pd.Series) else result
            
            # Special handling for increasing, decreasing operators
            elif operator_str in ["increasing", "decreasing"]:
                series = self._get_operand(left_operand_str.strip(), indicators)
                periods = int(right_operand_str.strip())
                
                if not isinstance(series, pd.Series) or len(series) < periods:
                    return False
                
                return operator_func(series, periods)
            
            # Standard comparison operators
            else:
                left_operand = self._get_operand(left_operand_str.strip(), indicators)
                right_operand = self._get_operand(right_operand_str.strip(), indicators)
                
                # Get the latest values for series objects
                if isinstance(left_operand, pd.Series):
                    left_operand = left_operand.iloc[-1]
                if isinstance(right_operand, pd.Series):
                    right_operand = right_operand.iloc[-1]
                
                return operator_func(left_operand, right_operand)
                
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {str(e)}")
            return False
    
    def _get_operand(self, operand_str: str, indicators: Dict[str, Any]) -> Any:
        """
        Get the value of an operand from indicators or convert to numeric if possible.
        
        Args:
            operand_str: String representation of operand
            indicators: Dictionary of indicator values
            
        Returns:
            Operand value (numeric or series)
        """
        # Check if it's a numeric value
        try:
            return float(operand_str)
        except ValueError:
            pass
        
        # Check if it's an indicator name
        if operand_str in indicators:
            return indicators[operand_str]
        
        # Check if it's an indicator field (e.g., macd_hist)
        for name in indicators:
            if operand_str.startswith(f"{name}_"):
                return indicators[operand_str]
        
        # Not found
        logger.error(f"Unknown operand: {operand_str}")
        return 0
    
    def _evaluate_rules(self, rules: List[Dict[str, Any]], indicators: Dict[str, Any]) -> Optional[SignalDirection]:
        """
        Evaluate a list of rules against indicators.
        
        Args:
            rules: List of rule dictionaries
            indicators: Dictionary of indicator values
            
        Returns:
            Signal direction if any rule is satisfied, None otherwise
        """
        for rule in rules:
            condition = rule.get('condition')
            if not condition:
                continue
                
            if self._evaluate_condition(condition, indicators):
                # Return the specified signal direction for this rule
                if 'entry_signal' in rule:
                    return rule['entry_signal']
                elif 'exit_signal' in rule:
                    return rule['exit_signal']
                else:
                    return SignalDirection.BUY  # Default to buy if not specified
        
        return None
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal based on indicator rules.
        
        Args:
            market_data: Dictionary with market data, must include 'ohlcv' key with DataFrame
            
        Returns:
            Signal value between -1.0 (strong sell) and 1.0 (strong buy)
        """
        # Extract OHLCV data
        ohlcv_data = market_data.get('ohlcv')
        if ohlcv_data is None:
            logger.warning(f"No OHLCV data provided for {self.name}")
            return 0.0
        
        # Calculate indicators
        self.calculated_indicators = self._calculate_indicators(ohlcv_data)
        if not self.calculated_indicators:
            logger.warning(f"Failed to calculate indicators for {self.name}")
            return 0.0
        
        # Track last update time
        self.last_update_time = datetime.now()
        
        # Evaluate rules based on position status
        if not self.in_position:
            # Evaluate entry rules
            entry_signal = self._evaluate_rules(self.entry_rules, self.calculated_indicators)
            
            if entry_signal == SignalDirection.BUY:
                self.in_position = True
                self.current_position = 1
                self.entry_price = ohlcv_data['close'].iloc[-1]
                self.trade_start_time = self.last_update_time
                signal_value = 1.0  # Strong buy
            elif entry_signal == SignalDirection.SELL:
                self.in_position = True
                self.current_position = -1
                self.entry_price = ohlcv_data['close'].iloc[-1]
                self.trade_start_time = self.last_update_time
                signal_value = -1.0  # Strong sell
            else:
                signal_value = 0.0  # No signal
        else:
            # Evaluate exit rules
            exit_signal = self._evaluate_rules(self.exit_rules, self.calculated_indicators)
            
            if exit_signal is not None:
                # Exit the current position
                self.in_position = False
                
                # Generate exit signal opposite to current position
                if self.current_position > 0:
                    signal_value = -0.5  # Moderate sell to exit long
                else:
                    signal_value = 0.5  # Moderate buy to exit short
                
                self.current_position = 0
                self.entry_price = None
                self.trade_start_time = None
            else:
                # Hold current position
                signal_value = 0.0
        
        return signal_value
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get strategy state for persistence.
        
        Returns:
            Dictionary with strategy state
        """
        state = super().to_dict()
        state.update({
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_position': self.current_position,
            'in_position': self.in_position,
            'entry_price': self.entry_price,
            'trade_start_time': self.trade_start_time.isoformat() if self.trade_start_time else None
        })
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore strategy state from persistence.
        
        Args:
            state: Dictionary with strategy state
        """
        self.symbol = state.get('symbol', self.symbol)
        self.timeframe = state.get('timeframe', self.timeframe)
        self.current_position = state.get('current_position', 0)
        self.in_position = state.get('in_position', False)
        self.entry_price = state.get('entry_price')
        
        trade_start_time = state.get('trade_start_time')
        if trade_start_time:
            self.trade_start_time = datetime.fromisoformat(trade_start_time)
        else:
            self.trade_start_time = None
        
        logger.info(f"Restored state for {self.name}, position: {self.current_position}")
    
    def restart(self) -> bool:
        """
        Restart the strategy (for recovery).
        
        Returns:
            True if restart was successful
        """
        try:
            logger.info(f"Restarting strategy: {self.name}")
            return True
        except Exception as e:
            logger.error(f"Error restarting strategy {self.name}: {str(e)}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the strategy.
        
        Returns:
            Dictionary with health status information
        """
        now = datetime.now()
        last_update_age = (now - self.last_update_time).total_seconds() if self.last_update_time else float('inf')
        
        return {
            "status": "healthy" if last_update_age < 300 else "warning",
            "last_update_age_seconds": last_update_age,
            "in_position": self.in_position,
            "current_position": self.current_position,
            "indicators_calculated": len(self.calculated_indicators) > 0
        }
