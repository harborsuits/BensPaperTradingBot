#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Base Strategy Module

This module provides the base class for forex trading strategies, with
forex-specific functionality built in.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, time
from enum import Enum

from trading_bot.strategies.strategy_template import StrategyOptimizable, Signal, SignalType, TimeFrame, MarketRegime

logger = logging.getLogger(__name__)

class ForexSession(Enum):
    """Enum for forex trading sessions."""
    SYDNEY = "sydney"
    TOKYO = "tokyo"
    LONDON = "london"
    NEWYORK = "newyork"
    OVERLAP_TOKYO_LONDON = "tokyo_london_overlap"
    OVERLAP_LONDON_NEWYORK = "london_newyork_overlap"

class ForexBaseStrategy(StrategyOptimizable):
    """
    Base class for forex trading strategies.
    
    This class extends the StrategyOptimizable to add forex-specific
    functionality including:
    - Trading session awareness
    - Currency pair strength analysis
    - Economic calendar integration
    - Interest rate differential analysis
    - Forex-specific indicators
    - Pip-based risk management
    """
    
    # Default parameters specific to forex trading
    DEFAULT_FOREX_PARAMS = {
        # Trading session parameters
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],  # Sessions to trade
        'trade_session_overlaps': True,    # Whether to focus on session overlaps
        
        # Pair selection parameters
        'major_pairs_only': True,          # Focus on major pairs only
        'exotic_pairs': False,             # Whether to include exotic pairs
        'cross_pairs': True,               # Whether to include cross pairs
        
        # Technical parameters
        'pip_value': 0.0001,               # Default pip value (0.01 for JPY pairs)
        'min_daily_range_pips': 50,        # Minimum average daily range in pips
        'atr_period': 14,                  # ATR period
        'use_atr_for_exits': True,         # Whether to use ATR for stop loss
        
        # Economic parameters
        'use_economic_calendar': False,    # Whether to use economic calendar
        'avoid_high_impact_news': True,    # Whether to avoid trading during high impact news
        'interest_rate_filter': False,     # Whether to use interest rate differentials
        
        # Risk parameters
        'max_risk_per_trade_percent': 0.01, # Maximum risk per trade (1%)
        'max_risk_per_day_percent': 0.03,   # Maximum risk per day (3%)
        'max_positions_per_day': 5,         # Maximum positions per day
        'max_correlation_threshold': 0.7,   # Maximum allowed pair correlation
        
        # Carry trade parameters
        'enable_carry_trades': False,       # Whether to consider carry trades
        'min_rate_differential': 2.0,       # Minimum interest rate differential (%)
    }
    
    # Major currency pairs
    MAJOR_PAIRS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD']
    
    # Minor (cross) pairs
    CROSS_PAIRS = [
        'EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
        'GBP/JPY', 'GBP/CHF', 'GBP/AUD', 'GBP/CAD', 'GBP/NZD',
        'AUD/JPY', 'AUD/CHF', 'AUD/CAD', 'AUD/NZD',
        'CAD/JPY', 'CAD/CHF', 'NZD/JPY', 'NZD/CHF'
    ]
    
    # Forex session hours (UTC)
    SESSION_HOURS = {
        ForexSession.SYDNEY: (time(20, 0), time(5, 0)),     # 20:00-05:00 UTC
        ForexSession.TOKYO: (time(23, 0), time(8, 0)),      # 23:00-08:00 UTC
        ForexSession.LONDON: (time(7, 0), time(16, 0)),     # 07:00-16:00 UTC
        ForexSession.NEWYORK: (time(12, 0), time(21, 0)),   # 12:00-21:00 UTC
        ForexSession.OVERLAP_TOKYO_LONDON: (time(7, 0), time(8, 0)),  # 07:00-08:00 UTC
        ForexSession.OVERLAP_LONDON_NEWYORK: (time(12, 0), time(16, 0))  # 12:00-16:00 UTC
    }
    
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a forex trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_FOREX_PARAMS)
            metadata: Strategy metadata
        """
        # Start with default forex parameters
        forex_params = self.DEFAULT_FOREX_PARAMS.copy()
        
        # Override with provided parameters
        if parameters:
            forex_params.update(parameters)
        
        # Initialize the parent class
        super().__init__(name=name, parameters=forex_params, metadata=metadata)
        
        # Forex-specific member variables
        self.currency_strength = {}  # Track relative strength of individual currencies
        self.pair_correlations = {}  # Track correlations between pairs
        self.interest_rates = {}     # Track interest rates for carry analysis
        
        logger.info(f"Initialized forex strategy: {name}")
    
    def filter_universe(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter the universe based on forex-specific criteria.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with forex data
            
        Returns:
            Filtered universe
        """
        filtered_universe = {}
        
        # Filter pairs based on preferences
        for symbol, data in universe.items():
            # Skip if no data
            if data.empty:
                continue
            
            # Apply pair type filters
            is_major = symbol in self.MAJOR_PAIRS
            is_cross = symbol in self.CROSS_PAIRS
            is_exotic = not (is_major or is_cross)
            
            if self.parameters['major_pairs_only'] and not is_major:
                continue
                
            if not self.parameters['cross_pairs'] and is_cross:
                continue
                
            if not self.parameters['exotic_pairs'] and is_exotic:
                continue
            
            # Filter by average daily range
            if 'high' in data.columns and 'low' in data.columns:
                # Calculate average daily range in pips
                pip_multiplier = 10000  # Standard pip multiplier (100 for JPY pairs)
                if 'JPY' in symbol:
                    pip_multiplier = 100
                    
                # Daily range in pips
                daily_range = (data['high'] - data['low']) * pip_multiplier
                avg_daily_range = daily_range.mean()
                
                if avg_daily_range < self.parameters['min_daily_range_pips']:
                    continue
            
            # Filter by interest rate differential if enabled
            if self.parameters['interest_rate_filter'] and symbol in self.interest_rates:
                base_ccy, quote_ccy = symbol.split('/')
                if base_ccy in self.interest_rates and quote_ccy in self.interest_rates:
                    rate_diff = self.interest_rates[base_ccy] - self.interest_rates[quote_ccy]
                    
                    # For carry trades, we want positive rate differential
                    if self.parameters['enable_carry_trades'] and rate_diff < self.parameters['min_rate_differential']:
                        continue
            
            # Symbol passed all filters
            filtered_universe[symbol] = data
        
        logger.info(f"Filtered forex universe from {len(universe)} to {len(filtered_universe)} symbols")
        return filtered_universe
    
    def calculate_forex_indicators(self, data: pd.DataFrame, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Calculate forex-specific technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            symbol: Forex pair symbol
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate Moving Averages
        for period in [20, 50, 200]:
            ma_key = f'ma_{period}'
            indicators[ma_key] = pd.DataFrame({
                ma_key: data['close'].rolling(window=period).mean()
            })
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = pd.DataFrame({'rsi': rsi})
        
        # Calculate MACD
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        indicators['macd'] = pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_hist': macd_hist
        })
        
        # Calculate ATR for forex volatility and stop-loss setting
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift(1))
        low_close_prev = np.abs(data['low'] - data['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = tr.rolling(window=self.parameters['atr_period']).mean()
        
        indicators['atr'] = pd.DataFrame({'atr': atr})
        
        # Calculate pip-based ATR
        pip_multiplier = 10000  # Standard pip multiplier
        if 'JPY' in symbol:
            pip_multiplier = 100
            
        atr_pips = atr * pip_multiplier
        indicators['atr_pips'] = pd.DataFrame({'atr_pips': atr_pips})
        
        # Calculate Bollinger Bands
        ma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        
        indicators['bbands'] = pd.DataFrame({
            'middle_band': ma20,
            'upper_band': upper_band,
            'lower_band': lower_band
        })
        
        # Forex-specific: Add stochastic oscillator (popular in forex)
        high_14 = data['high'].rolling(window=14).max()
        low_14 = data['low'].rolling(window=14).min()
        
        # Calculate %K (fast stochastic)
        stoch_k = 100 * ((data['close'] - low_14) / (high_14 - low_14))
        
        # Calculate %D (slow stochastic)
        stoch_d = stoch_k.rolling(window=3).mean()
        
        indicators['stochastic'] = pd.DataFrame({
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        })
        
        # Add Ichimoku Cloud (very popular in forex)
        high_9 = data['high'].rolling(window=9).max()
        low_9 = data['low'].rolling(window=9).min()
        high_26 = data['high'].rolling(window=26).max()
        low_26 = data['low'].rolling(window=26).min()
        high_52 = data['high'].rolling(window=52).max()
        low_52 = data['low'].rolling(window=52).min()
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price, shifted backwards 26 periods
        chikou_span = data['close'].shift(-26)
        
        indicators['ichimoku'] = pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        })
        
        return indicators
    
    def calculate_currency_strength(self, pairs_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate relative strength of individual currencies.
        
        Args:
            pairs_data: Dictionary mapping pair symbols to DataFrames with historical data
            
        Returns:
            Dictionary mapping currencies to strength scores
        """
        # Initialize currency strength dictionary
        all_currencies = set()
        
        # Extract all currencies from pairs
        for pair in pairs_data.keys():
            if '/' in pair:
                base, quote = pair.split('/')
                all_currencies.add(base)
                all_currencies.add(quote)
        
        # Initialize strength with zeros
        currency_strength = {ccy: 0.0 for ccy in all_currencies}
        pair_count = {ccy: 0 for ccy in all_currencies}
        
        # Calculate percentage changes for each pair
        for pair, data in pairs_data.items():
            if data.empty or '/' not in pair or len(data) < 2:
                continue
                
            base, quote = pair.split('/')
            
            # Calculate 1-day percent change
            latest_close = data['close'].iloc[-1]
            prev_close = data['close'].iloc[-2]
            percent_change = (latest_close / prev_close - 1) * 100
            
            # Update base currency strength
            currency_strength[base] += percent_change
            pair_count[base] += 1
            
            # Update quote currency strength (inversely related)
            currency_strength[quote] -= percent_change
            pair_count[quote] += 1
        
        # Average strength by pair count
        for ccy in currency_strength:
            if pair_count[ccy] > 0:
                currency_strength[ccy] /= pair_count[ccy]
        
        return currency_strength
    
    def is_current_session_active(self, target_sessions: Optional[List[ForexSession]] = None) -> bool:
        """
        Check if the current time is within the specified forex sessions.
        
        Args:
            target_sessions: List of sessions to check (defaults to strategy parameters)
            
        Returns:
            True if current time is within any of the target sessions
        """
        if target_sessions is None:
            target_sessions = self.parameters.get('trading_sessions', [])
            
        current_time = datetime.utcnow().time()
        
        for session in target_sessions:
            start_time, end_time = self.SESSION_HOURS[session]
            
            # Handle sessions that span midnight
            if start_time > end_time:
                if current_time >= start_time or current_time <= end_time:
                    return True
            else:
                if start_time <= current_time <= end_time:
                    return True
        
        return False
    
    def calculate_position_size_pips(self, symbol: str, entry_price: float, 
                                  stop_loss_pips: float, risk_amount: float) -> float:
        """
        Calculate position size for forex based on risk and stop-loss in pips.
        
        Args:
            symbol: Forex pair symbol
            entry_price: Entry price
            stop_loss_pips: Stop loss distance in pips
            risk_amount: Amount to risk in account currency
            
        Returns:
            Position size in standard lots (100,000 units)
        """
        # Determine pip value based on the pair
        is_jpy_pair = 'JPY' in symbol
        
        # Convert pip values to price values
        pip_value = 0.01 if is_jpy_pair else 0.0001
        pip_decimal_places = 2 if is_jpy_pair else 4
        
        # Convert stop loss from pips to price value
        stop_loss_price = stop_loss_pips * pip_value
        
        # For a standard lot (100,000 units):
        # Value per pip in USD = 10 USD for pairs with USD as quote currency
        # For other pairs, need to convert
        
        # Simplified calculation (assumes USD account or proper conversion):
        standard_lot_pip_value = 10.0  # USD per pip for standard lot (simplified)
        
        # Calculate position size in standard lots
        if stop_loss_price > 0:
            lot_size = risk_amount / (stop_loss_pips * standard_lot_pip_value)
        else:
            lot_size = 0.0
        
        # Round to standard lot increments (0.01 lot = 1,000 units)
        return round(lot_size, 2)
    
    def adjust_for_trading_session(self, signals: Dict[str, Signal]) -> Dict[str, Signal]:
        """
        Adjust signals based on current trading session.
        
        Args:
            signals: Dictionary of generated signals
            
        Returns:
            Adjusted signals
        """
        adjusted_signals = signals.copy()
        
        # Check if current session is in the preferred sessions
        is_preferred_session = self.is_current_session_active()
        
        # Check if current session is an overlap period (generally more volatile)
        is_overlap_session = self.is_current_session_active([
            ForexSession.OVERLAP_TOKYO_LONDON,
            ForexSession.OVERLAP_LONDON_NEWYORK
        ])
        
        for symbol, signal in adjusted_signals.items():
            # Adjust confidence based on session
            if not is_preferred_session:
                # Reduce confidence if not in preferred session
                signal.confidence *= 0.7
            elif is_overlap_session and self.parameters['trade_session_overlaps']:
                # Increase confidence during overlap sessions if we prefer those
                signal.confidence = min(1.0, signal.confidence * 1.3)
            
            # Specific adjustments based on currency and session
            base, quote = symbol.split('/') if '/' in symbol else (symbol, '')
            
            # USD pairs during NY session
            if (base == 'USD' or quote == 'USD') and self.is_current_session_active([ForexSession.NEWYORK]):
                signal.confidence = min(1.0, signal.confidence * 1.2)
                
            # EUR/USD during London/NY overlap
            if symbol == 'EUR/USD' and self.is_current_session_active([ForexSession.OVERLAP_LONDON_NEWYORK]):
                signal.confidence = min(1.0, signal.confidence * 1.25)
                
            # JPY pairs during Tokyo session
            if (base == 'JPY' or quote == 'JPY') and self.is_current_session_active([ForexSession.TOKYO]):
                signal.confidence = min(1.0, signal.confidence * 1.2)
                
            # AUD/NZD pairs during Sydney session
            if (base in ['AUD', 'NZD'] or quote in ['AUD', 'NZD']) and self.is_current_session_active([ForexSession.SYDNEY]):
                signal.confidence = min(1.0, signal.confidence * 1.15)
        
        return adjusted_signals
    
    def analyze_interest_rate_differentials(self, symbols: List[str]) -> Dict[str, float]:
        """
        Analyze interest rate differentials between currency pairs for potential carry trades.
        
        Args:
            symbols: List of forex pair symbols
            
        Returns:
            Dictionary mapping symbols to interest rate differentials
        """
        if not self.interest_rates or not self.parameters['interest_rate_filter']:
            return {}
            
        differentials = {}
        
        for symbol in symbols:
            if '/' not in symbol:
                continue
                
            base, quote = symbol.split('/')
            
            if base in self.interest_rates and quote in self.interest_rates:
                rate_diff = self.interest_rates[base] - self.interest_rates[quote]
                differentials[symbol] = rate_diff
        
        return differentials
    
    def is_high_impact_news_time(self, symbol: str, hours_before: int = 1, 
                              hours_after: int = 1) -> bool:
        """
        Check if there's high impact news scheduled for the given symbol.
        
        Args:
            symbol: Forex pair symbol
            hours_before: Hours before news to avoid trading
            hours_after: Hours after news to avoid trading
            
        Returns:
            True if there's high impact news in the specified window
        """
        # This is a placeholder method - in a real implementation, you would
        # integrate with an economic calendar API
        
        # For now, always return False
        return False 