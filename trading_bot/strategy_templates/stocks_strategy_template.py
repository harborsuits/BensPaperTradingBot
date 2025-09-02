#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stocks Strategy Template for BensBot Trading System

This template defines the standard structure that all stock strategies should follow
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

class StocksStrategyTemplate(StrategyTemplate):
    """
    Base template for all stock trading strategies that ensures consistent
    interface for the backtester, strategy finder, and live trading systems.
    
    All stock strategies should inherit from this template and implement the 
    required methods with the same signatures.
    """
    
    # Default parameters for stock strategies
    DEFAULT_PARAMS = {
        'strategy_name': 'stocks_template_strategy',
        'strategy_version': '1.0.0',
        'asset_class': 'stocks',
        'strategy_type': 'all_weather',
        'timeframe': 'daily',
        'market_regime': 'all_weather',
        
        # Stock universe parameters
        'min_price': 5.0,               # Minimum stock price
        'max_price': 1000.0,            # Maximum stock price
        'min_volume': 500000,           # Minimum average daily volume
        'min_market_cap': 500000000,    # Minimum market cap ($500M)
        'exclude_sectors': [],          # Sectors to exclude
        'include_sectors': [],          # Sectors to include (empty = all)
        
        # Technical parameters
        'sma_fast': 20,                 # Fast SMA period
        'sma_slow': 50,                 # Slow SMA period
        'rsi_period': 14,               # RSI period
        'rsi_overbought': 70,           # RSI overbought level
        'rsi_oversold': 30,             # RSI oversold level
        'macd_fast': 12,                # MACD fast period
        'macd_slow': 26,                # MACD slow period
        'macd_signal': 9,               # MACD signal period
        
        # Risk management parameters
        'position_size_pct': 0.05,      # Position size as % of portfolio
        'max_positions': 20,            # Maximum number of positions
        'stop_loss_pct': 0.05,          # Stop loss percentage
        'trailing_stop_pct': 0.10,      # Trailing stop percentage
        'take_profit_pct': 0.15,        # Take profit percentage
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None):
        """
        Initialize the stock strategy with parameters.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        super().__init__(strategy_id, name, parameters)
        
        # Stocks-specific tracking
        self.stock_data = {}            # Store stock-specific data
        self.indicators = {}            # Technical indicators by symbol
        self.sector_data = {}           # Sector performance data
        self.earnings_calendar = {}     # Upcoming earnings releases
        
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable stocks for this strategy.
        
        Args:
            market_data: Market data to use for filtering
            
        Returns:
            Universe object containing filtered stock symbols
        """
        # Default universe definition for stock strategies
        universe = Universe()
        
        # Extract filtering parameters
        min_price = self.parameters.get('min_price', 5.0)
        max_price = self.parameters.get('max_price', 1000.0)
        min_volume = self.parameters.get('min_volume', 500000)
        min_market_cap = self.parameters.get('min_market_cap', 500000000)
        exclude_sectors = self.parameters.get('exclude_sectors', [])
        include_sectors = self.parameters.get('include_sectors', [])
        
        # Get all available symbols
        all_symbols = market_data.get_all_symbols()
        filtered_symbols = []
        
        for symbol in all_symbols:
            # Get basic price and volume data
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                continue
                
            price = quote.get('price', 0)
            volume = quote.get('volume', 0)
            
            # Get fundamental data
            fundamentals = market_data.get_fundamentals(symbol)
            if not fundamentals:
                continue
                
            market_cap = fundamentals.get('market_cap', 0)
            sector = fundamentals.get('sector', '')
            
            # Apply price filter
            if not (min_price <= price <= max_price):
                continue
                
            # Apply volume filter
            if volume < min_volume:
                continue
                
            # Apply market cap filter
            if market_cap < min_market_cap:
                continue
                
            # Apply sector filters
            if exclude_sectors and sector in exclude_sectors:
                continue
                
            if include_sectors and sector not in include_sectors:
                continue
                
            # Symbol passes all filters
            filtered_symbols.append(symbol)
        
        universe.add_symbols(filtered_symbols)
        self.logger.info(f"Stock universe defined with {len(filtered_symbols)} symbols")
        return universe
    
    def generate_signals(self, market_data: Union[MarketData, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate stock trading signals based on market data.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            List of signal dictionaries with standard format
        """
        # This method should be implemented by specific stock strategies
        # Default implementation returns empty list
        return []
    
    def _calculate_technical_indicators(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for a stock.
        
        Args:
            symbol: Stock symbol
            data: OHLCV data for the stock
            
        Returns:
            Dictionary of calculated indicators
        """
        # Default indicator calculation
        indicators = {}
        
        # Extract parameters
        sma_fast_period = self.parameters.get('sma_fast', 20)
        sma_slow_period = self.parameters.get('sma_slow', 50)
        rsi_period = self.parameters.get('rsi_period', 14)
        macd_fast = self.parameters.get('macd_fast', 12)
        macd_slow = self.parameters.get('macd_slow', 26)
        macd_signal = self.parameters.get('macd_signal', 9)
        
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
        
        # Calculate MACD
        if len(data) >= macd_slow + macd_signal:
            data['ema_fast'] = data['close'].ewm(span=macd_fast, adjust=False).mean()
            data['ema_slow'] = data['close'].ewm(span=macd_slow, adjust=False).mean()
            data['macd'] = data['ema_fast'] - data['ema_slow']
            data['macd_signal'] = data['macd'].ewm(span=macd_signal, adjust=False).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            indicators['macd'] = data['macd'].iloc[-1]
            indicators['macd_signal'] = data['macd_signal'].iloc[-1]
            indicators['macd_hist'] = data['macd_hist'].iloc[-1]
            indicators['macd_cross'] = (data['macd'].iloc[-2] < data['macd_signal'].iloc[-2] and 
                                      data['macd'].iloc[-1] >= data['macd_signal'].iloc[-1])
            indicators['macd_cross_down'] = (data['macd'].iloc[-2] > data['macd_signal'].iloc[-2] and 
                                          data['macd'].iloc[-1] <= data['macd_signal'].iloc[-1])
        
        # Calculate Bollinger Bands
        if len(data) >= 20:
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['stddev'] = data['close'].rolling(window=20).std()
            data['upper_band'] = data['sma_20'] + (data['stddev'] * 2)
            data['lower_band'] = data['sma_20'] - (data['stddev'] * 2)
            
            indicators['bb_upper'] = data['upper_band'].iloc[-1]
            indicators['bb_middle'] = data['sma_20'].iloc[-1]
            indicators['bb_lower'] = data['lower_band'].iloc[-1]
            indicators['bb_width'] = (data['upper_band'].iloc[-1] - data['lower_band'].iloc[-1]) / data['sma_20'].iloc[-1]
        
        # Store indicators for this symbol
        self.indicators[symbol] = indicators
        
        return indicators
    
    def position_sizing(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """
        Calculate position size for a stock signal.
        
        Args:
            signal: Trading signal dictionary
            account_info: Account information including equity, margin, etc.
            
        Returns:
            Number of shares to trade
        """
        # Default stock position sizing
        account_value = account_info.get('equity', 0)
        if account_value <= 0:
            return 0
            
        # Extract parameters
        position_size_pct = self.parameters.get('position_size_pct', 0.05)
        max_positions = self.parameters.get('max_positions', 20)
        
        # Calculate maximum amount to allocate to this position
        max_amount = account_value * position_size_pct
        
        # Get entry price
        entry_price = signal.get('entry_price', 0)
        if entry_price <= 0:
            return 0
            
        # Calculate number of shares
        shares = int(max_amount / entry_price)
        
        return shares
    
    def calculate_risk_metrics(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate risk metrics for a stock trade.
        
        Args:
            signal: Trading signal dictionary
            
        Returns:
            Dictionary of risk metrics
        """
        # Stock-specific risk metrics
        action = signal.get('action', '')
        entry_price = signal.get('entry_price', 0)
        stop_loss = signal.get('stop_loss', 0)
        take_profit = signal.get('take_profit', 0)
        
        if entry_price <= 0:
            return {'risk_reward_ratio': 0}
            
        # Calculate risk and reward
        risk = 0
        reward = 0
        
        if action == 'BUY':
            if stop_loss > 0:
                risk = (entry_price - stop_loss) / entry_price
            if take_profit > 0:
                reward = (take_profit - entry_price) / entry_price
        elif action == 'SELL':
            if stop_loss > 0:
                risk = (stop_loss - entry_price) / entry_price
            if take_profit > 0:
                reward = (entry_price - take_profit) / entry_price
                
        # Calculate risk-reward ratio
        risk_reward = 0
        if risk > 0:
            risk_reward = reward / risk
            
        return {
            'risk_pct': risk * 100,
            'reward_pct': reward * 100,
            'risk_reward_ratio': risk_reward,
        }
    
    def create_stock_signal(self, symbol: str, action: str, reason: str, 
                          entry_price: float, stop_loss: float, take_profit: float, 
                          strength: float = 1.0) -> Dict[str, Any]:
        """
        Create a standardized stock signal dictionary.
        
        Args:
            symbol: Stock symbol
            action: Signal action ('BUY', 'SELL')
            reason: Reason for the signal
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            strength: Signal strength from 0.0 to 1.0
            
        Returns:
            Standardized stock signal dictionary
        """
        # Create base signal
        signal = self.create_signal(
            symbol=symbol,
            action=action,
            reason=reason,
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Calculate risk percentage
        risk_pct = 0
        if action == 'BUY' and stop_loss > 0 and entry_price > 0:
            risk_pct = (entry_price - stop_loss) / entry_price * 100
        elif action == 'SELL' and stop_loss > 0 and entry_price > 0:
            risk_pct = (stop_loss - entry_price) / entry_price * 100
            
        # Calculate reward percentage
        reward_pct = 0
        if action == 'BUY' and take_profit > 0 and entry_price > 0:
            reward_pct = (take_profit - entry_price) / entry_price * 100
        elif action == 'SELL' and take_profit > 0 and entry_price > 0:
            reward_pct = (entry_price - take_profit) / entry_price * 100
            
        # Add stock-specific fields
        signal.update({
            'risk_pct': risk_pct,
            'reward_pct': reward_pct,
        })
        
        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics(signal)
        signal.update(risk_metrics)
        
        return signal
