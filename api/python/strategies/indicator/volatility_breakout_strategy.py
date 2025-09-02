#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Volatility Breakout Strategy

A specialized indicator strategy that trades breakouts based on volatility.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from trading_bot.strategies.indicator.indicator_strategy import IndicatorStrategy
from trading_bot.core.constants import SignalDirection

logger = logging.getLogger(__name__)

class VolatilityBreakoutStrategy(IndicatorStrategy):
    """
    Strategy that trades breakouts based on volatility measures.
    
    This strategy extends the base IndicatorStrategy to add specialized
    volatility-based calculations.
    """
    
    def __init__(self, name: str, config: Dict[str, Any], symbol: str = None):
        """
        Initialize volatility breakout strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
            symbol: Optional symbol to trade
        """
        # Add required indicators if not in config
        if 'indicators' not in config:
            config['indicators'] = {}
            
        # Ensure we have ATR for volatility measurement
        if not any(ind.get('type') == 'ATR' for ind in config['indicators'].values()):
            config['indicators']['atr'] = {
                'type': 'ATR',
                'timeperiod': config.get('atr_period', 14)
            }
        
        # Ensure we have Bollinger Bands
        if not any(ind.get('type') == 'BBANDS' for ind in config['indicators'].values()):
            config['indicators']['bb'] = {
                'type': 'BBANDS',
                'timeperiod': config.get('bb_period', 20),
                'nbdevup': config.get('bb_dev', 2),
                'nbdevdn': config.get('bb_dev', 2)
            }
        
        # Extract specific configuration parameters
        self.volatility_factor = config.get('volatility_factor', 1.0)
        self.lookback_period = config.get('lookback_period', 20)
        self.entry_threshold = config.get('entry_threshold', 1.5)
        self.exit_threshold = config.get('exit_threshold', 0.5)
        
        # Call parent initializer
        super().__init__(name, config, symbol)
        
        logger.info(f"Initialized VolatilityBreakoutStrategy: {name}")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate trading signal based on volatility breakout.
        
        Args:
            market_data: Market data dictionary with OHLCV data
            
        Returns:
            Signal value between -1.0 and 1.0
        """
        # Extract OHLCV data
        ohlcv_data = market_data.get('ohlcv')
        if ohlcv_data is None or len(ohlcv_data) < self.lookback_period:
            logger.warning(f"Insufficient data for volatility calculation: {self.name}")
            return 0.0
        
        # Calculate indicators
        self.calculated_indicators = self._calculate_indicators(ohlcv_data)
        if not self.calculated_indicators:
            logger.warning(f"Failed to calculate indicators for {self.name}")
            return 0.0
        
        # Calculate volatility-based levels
        try:
            # Get ATR and close price
            atr = next((ind for name, ind in self.calculated_indicators.items() if name.endswith('atr')), None)
            close_prices = self.calculated_indicators.get('close')
            
            if atr is None or close_prices is None:
                logger.warning(f"Missing required indicators for {self.name}")
                return 0.0
            
            # Current values
            current_close = close_prices.iloc[-1]
            current_atr = atr.iloc[-1]
            
            # Calculate high/low of lookback period
            high_prices = self.calculated_indicators.get('high')
            low_prices = self.calculated_indicators.get('low')
            
            lookback_high = high_prices.iloc[-self.lookback_period:].max()
            lookback_low = low_prices.iloc[-self.lookback_period:].min()
            
            # Calculate breakout levels
            upper_level = lookback_high + (current_atr * self.volatility_factor)
            lower_level = lookback_low - (current_atr * self.volatility_factor)
            
            # Add these to our indicators for rule evaluation
            self.calculated_indicators['upper_level'] = pd.Series([upper_level] * len(close_prices), index=close_prices.index)
            self.calculated_indicators['lower_level'] = pd.Series([lower_level] * len(close_prices), index=close_prices.index)
            
            # Track last update time
            self.last_update_time = datetime.now()
            
            # Check for breakouts
            signal_value = 0.0
            
            # Not in position - check for entry
            if not self.in_position:
                # Bullish breakout
                if current_close > lookback_high + (current_atr * self.entry_threshold):
                    self.in_position = True
                    self.current_position = 1
                    self.entry_price = current_close
                    self.trade_start_time = self.last_update_time
                    signal_value = 1.0  # Strong buy
                
                # Bearish breakout
                elif current_close < lookback_low - (current_atr * self.entry_threshold):
                    self.in_position = True
                    self.current_position = -1
                    self.entry_price = current_close
                    self.trade_start_time = self.last_update_time
                    signal_value = -1.0  # Strong sell
            
            # In position - check for exit
            else:
                # Exit long position
                if self.current_position > 0 and current_close < (lookback_high - (current_atr * self.exit_threshold)):
                    self.in_position = False
                    signal_value = -0.5  # Exit long
                    self.current_position = 0
                    self.entry_price = None
                    self.trade_start_time = None
                
                # Exit short position
                elif self.current_position < 0 and current_close > (lookback_low + (current_atr * self.exit_threshold)):
                    self.in_position = False
                    signal_value = 0.5  # Exit short
                    self.current_position = 0
                    self.entry_price = None
                    self.trade_start_time = None
            
            # Also check standard indicator rules
            if signal_value == 0.0:
                # Use parent class rule evaluation if no signal from volatility breakout
                if not self.in_position:
                    entry_signal = self._evaluate_rules(self.entry_rules, self.calculated_indicators)
                    if entry_signal == SignalDirection.BUY:
                        self.in_position = True
                        self.current_position = 1
                        self.entry_price = current_close
                        self.trade_start_time = self.last_update_time
                        signal_value = 1.0
                    elif entry_signal == SignalDirection.SELL:
                        self.in_position = True
                        self.current_position = -1
                        self.entry_price = current_close
                        self.trade_start_time = self.last_update_time
                        signal_value = -1.0
                else:
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
            
            return signal_value
        
        except Exception as e:
            logger.error(f"Error generating signal for {self.name}: {str(e)}")
            return 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get strategy state for persistence.
        
        Returns:
            Dictionary with strategy state
        """
        state = super().get_state()
        state.update({
            'volatility_factor': self.volatility_factor,
            'lookback_period': self.lookback_period,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold
        })
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """
        Restore strategy state from persistence.
        
        Args:
            state: Dictionary with strategy state
        """
        super().restore_state(state)
        
        # Restore specialized state
        self.volatility_factor = state.get('volatility_factor', self.volatility_factor)
        self.lookback_period = state.get('lookback_period', self.lookback_period)
        self.entry_threshold = state.get('entry_threshold', self.entry_threshold)
        self.exit_threshold = state.get('exit_threshold', self.exit_threshold)
