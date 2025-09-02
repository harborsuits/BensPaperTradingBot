#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Strategy Template for BensBot Trading System

This template defines the standard structure that all forex strategies should follow
to ensure proper integration with the backtester, strategy finder, and live trading.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from trading_bot.strategy_templates.strategy_template import StrategyTemplate, register_strategy_with_registry
from trading_bot.market.market_data import MarketData
from trading_bot.market.universe import Universe

logger = logging.getLogger(__name__)

class ForexStrategyTemplate(StrategyTemplate):
    """
    Base template for all forex trading strategies that ensures consistent
    interface for the backtester, strategy finder, and live trading systems.
    
    All forex strategies should inherit from this template and implement the 
    required methods with the same signatures.
    """
    
    # Default parameters for forex strategies
    DEFAULT_PARAMS = {
        'strategy_name': 'forex_template_strategy',
        'strategy_version': '1.0.0',
        'asset_class': 'forex',
        'strategy_type': 'all_weather',
        'timeframe': 'daily',
        'market_regime': 'all_weather',
        
        # Forex-specific parameters
        'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF'],
        'min_atr': 0.001,              # Minimum ATR for volatility
        'min_daily_volume': 10000,     # Minimum daily volume in lots
        
        # Technical parameters
        'sma_fast': 10,                # Fast SMA period
        'sma_slow': 50,                # Slow SMA period
        'rsi_period': 14,              # RSI period
        'rsi_overbought': 70,          # RSI overbought level
        'rsi_oversold': 30,            # RSI oversold level
        
        # Risk management parameters
        'max_risk_per_trade_pct': 0.01,  # Maximum risk per trade (1% of account)
        'max_open_positions': 5,       # Maximum number of open positions
        'position_sizing_method': 'risk_based',  # 'fixed', 'risk_based', 'volatility_based'
        'stop_loss_atr_multiplier': 2.0,  # ATR multiplier for stop loss calculation
        'take_profit_atr_multiplier': 3.0,  # ATR multiplier for take profit calculation
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None):
        """
        Initialize the forex strategy with parameters.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        super().__init__(strategy_id, name, parameters)
        
        # Forex-specific tracking
        self.pair_data = {}            # Store pair-specific data
        self.indicators = {}           # Technical indicators by pair
        self.atr_values = {}           # ATR values by pair
        self.swap_rates = {}           # Swap/rollover rates by pair
        
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable forex pairs for this strategy.
        
        Args:
            market_data: Market data to use for filtering
            
        Returns:
            Universe object containing filtered forex pairs
        """
        # Default universe definition for forex strategies
        universe = Universe()
        
        # Get configured pairs from parameters
        pairs = self.parameters.get('pairs', [])
        
        # Filter pairs based on data availability and other criteria
        filtered_pairs = []
        for pair in pairs:
            # Check if data is available
            if market_data.has_symbol(pair):
                # Additional filters can be applied here
                filtered_pairs.append(pair)
        
        universe.add_symbols(filtered_pairs)
        self.logger.info(f"Forex universe defined with {len(filtered_pairs)} pairs")
        return universe
    
    def generate_signals(self, market_data: Union[MarketData, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate forex trading signals based on market data.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            List of signal dictionaries with standard format
        """
        # This method should be implemented by specific forex strategies
        # Default implementation returns empty list
        return []
    
    def _calculate_technical_indicators(self, pair: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for a forex pair.
        
        Args:
            pair: Forex pair
            data: OHLCV data for the pair
            
        Returns:
            Dictionary of calculated indicators
        """
        # Default indicator calculation
        indicators = {}
        
        # Extract parameters
        sma_fast_period = self.parameters.get('sma_fast', 10)
        sma_slow_period = self.parameters.get('sma_slow', 50)
        rsi_period = self.parameters.get('rsi_period', 14)
        
        # Calculate SMAs
        if len(data) >= sma_slow_period:
            data['sma_fast'] = data['close'].rolling(window=sma_fast_period).mean()
            data['sma_slow'] = data['close'].rolling(window=sma_slow_period).mean()
            
            indicators['sma_fast'] = data['sma_fast'].iloc[-1]
            indicators['sma_slow'] = data['sma_slow'].iloc[-1]
            indicators['sma_cross'] = (data['sma_fast'].iloc[-2] < data['sma_slow'].iloc[-2] and 
                                      data['sma_fast'].iloc[-1] >= data['sma_slow'].iloc[-1])
            indicators['sma_cross_down'] = (data['sma_fast'].iloc[-2] > data['sma_slow'].iloc[-2] and 
                                          data['sma_fast'].iloc[-1] <= data['sma_slow'].iloc[-1])
        
        # Calculate RSI
        if len(data) >= rsi_period:
            delta = data['close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            indicators['rsi'] = data['rsi'].iloc[-1]
            indicators['rsi_overbought'] = indicators['rsi'] > self.parameters.get('rsi_overbought', 70)
            indicators['rsi_oversold'] = indicators['rsi'] < self.parameters.get('rsi_oversold', 30)
        
        # Calculate ATR
        if len(data) >= 14:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift()).abs()
            low_close = (data['low'] - data['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data['atr'] = true_range.rolling(14).mean()
            
            indicators['atr'] = data['atr'].iloc[-1]
            self.atr_values[pair] = indicators['atr']
        
        return indicators
    
    def position_sizing(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size for a forex signal.
        
        Args:
            signal: Trading signal dictionary
            account_info: Account information including equity, margin, etc.
            
        Returns:
            Position size in lots
        """
        # Default forex position sizing
        account_balance = account_info.get('equity', 0)
        if account_balance <= 0:
            return 0
            
        # Extract parameters
        max_risk_pct = self.parameters.get('max_risk_per_trade_pct', 0.01)
        sizing_method = self.parameters.get('position_sizing_method', 'risk_based')
        
        # Get pair and current ATR
        pair = signal.get('symbol', '')
        atr = self.atr_values.get(pair, 0)
        if atr <= 0:
            atr = 0.001  # Default if ATR not available
        
        # Extract signal information
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        
        # Different position sizing methods
        if sizing_method == 'fixed':
            # Fixed position size (0.1 lot)
            return 0.1
            
        elif sizing_method == 'risk_based' and stop_loss > 0:
            # Risk-based position sizing
            max_risk_amount = account_balance * max_risk_pct
            pip_value = self._calculate_pip_value(pair)
            
            # Calculate risk per pip
            pips_at_risk = abs(entry_price - stop_loss) / 0.0001  # For 4-decimal pairs
            if 'JPY' in pair:
                pips_at_risk = abs(entry_price - stop_loss) / 0.01  # For JPY pairs
                
            # Calculate position size in lots
            if pips_at_risk > 0 and pip_value > 0:
                lots = max_risk_amount / (pips_at_risk * pip_value)
                # Round to 2 decimal places (0.01 lot precision)
                return round(lots, 2)
                
        elif sizing_method == 'volatility_based' and atr > 0:
            # Volatility-based position sizing
            max_risk_amount = account_balance * max_risk_pct
            pip_value = self._calculate_pip_value(pair)
            
            # Use ATR for stop loss calculation
            atr_multiplier = self.parameters.get('stop_loss_atr_multiplier', 2.0)
            atr_pips = atr / 0.0001  # For 4-decimal pairs
            if 'JPY' in pair:
                atr_pips = atr / 0.01  # For JPY pairs
                
            pips_at_risk = atr_pips * atr_multiplier
            
            # Calculate position size in lots
            if pips_at_risk > 0 and pip_value > 0:
                lots = max_risk_amount / (pips_at_risk * pip_value)
                # Round to 2 decimal places (0.01 lot precision)
                return round(lots, 2)
        
        # Default if no valid calculation
        return 0.1
    
    def _calculate_pip_value(self, pair: str) -> float:
        """
        Calculate the value of one pip for a forex pair.
        
        Args:
            pair: Forex pair
            
        Returns:
            Value of one pip in account currency
        """
        # Simplified pip value calculation
        # In a real implementation, this would consider account currency and exchange rates
        base_pip_value = 10  # Approximate value for 1 standard lot (100k units)
        
        # Adjust for JPY pairs
        if 'JPY' in pair:
            base_pip_value = 8.5  # Approximate for JPY pairs
            
        return base_pip_value
    
    def calculate_risk_metrics(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk metrics for a forex trade.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Dictionary of risk metrics
        """
        # Forex-specific risk metrics
        pair = signal.get('symbol', '')
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        take_profit = signal.get('take_profit', 0)
        
        # Calculate pips at risk and potential gain
        pips_at_risk = 0
        pips_potential_gain = 0
        pip_multiplier = 10000  # For 4-decimal pairs
        
        if 'JPY' in pair:
            pip_multiplier = 100  # For JPY pairs
            
        if stop_loss > 0:
            pips_at_risk = abs(entry_price - stop_loss) * pip_multiplier
            
        if take_profit > 0:
            pips_potential_gain = abs(take_profit - entry_price) * pip_multiplier
            
        # Calculate risk-reward ratio
        risk_reward = 0
        if pips_at_risk > 0:
            risk_reward = pips_potential_gain / pips_at_risk
            
        return {
            'pips_at_risk': pips_at_risk,
            'pips_potential_gain': pips_potential_gain,
            'risk_reward_ratio': risk_reward,
            'swap_long': self.swap_rates.get(pair, {}).get('long', 0),
            'swap_short': self.swap_rates.get(pair, {}).get('short', 0),
        }
    
    def create_forex_signal(self, pair: str, action: str, reason: str, 
                          entry_price: float, stop_loss: float, take_profit: float, 
                          strength: float = 1.0) -> Dict[str, Any]:
        """
        Create a standardized forex signal dictionary.
        
        Args:
            pair: Forex pair
            action: Signal action ('BUY', 'SELL')
            reason: Reason for the signal
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            strength: Signal strength from 0.0 to 1.0
            
        Returns:
            Standardized forex signal dictionary
        """
        # Create base signal
        signal = self.create_signal(
            symbol=pair,
            action=action,
            reason=reason,
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add forex-specific fields
        pip_multiplier = 10000  # For 4-decimal pairs
        if 'JPY' in pair:
            pip_multiplier = 100  # For JPY pairs
            
        pips_at_risk = abs(entry_price - stop_loss) * pip_multiplier
        pips_potential_gain = abs(take_profit - entry_price) * pip_multiplier
        
        signal.update({
            'pips_at_risk': pips_at_risk,
            'pips_potential_gain': pips_potential_gain,
            'swap_long': self.swap_rates.get(pair, {}).get('long', 0),
            'swap_short': self.swap_rates.get(pair, {}).get('short', 0),
        })
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(signal)
        signal.update(risk_metrics)
        
        return signal
