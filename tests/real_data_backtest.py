#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Backtesting with Yahoo Finance Data

This script uses Yahoo Finance to download real historical market data
and test our contextual trading system with improved features.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from typing import Dict, List, Any, Tuple
import json
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RealDataBacktest")

# Constants for the backtesting system
class EventType:
    """Constants for event types"""
    MARKET_REGIME_CHANGE = "market_regime_change"
    VOLATILITY_UPDATE = "volatility_update"
    CORRELATION_UPDATE = "correlation_update"
    TRADE_EXECUTED = "trade_executed"
    TRADE_CLOSED = "trade_closed"

class Event:
    """Simple event class for the system"""
    def __init__(self, event_type, data):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()

class EventBus:
    """Event bus for communication between components"""
    def __init__(self):
        self.subscribers = {}
        self.event_history = []
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        
    def publish(self, event):
        """Publish an event to subscribers"""
        logger.debug(f"Publishing event: {event.event_type}")
        self.event_history.append(event)
        
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                callback(event)

class YahooFinanceDataProvider:
    """Data provider that fetches and processes real market data from Yahoo Finance"""
    
    def __init__(self, event_bus=None, data_dir="data/market_data"):
        """
        Initialize the Yahoo Finance data provider.
        
        Args:
            event_bus: Event bus for publishing market data events
            data_dir: Directory for storing downloaded data
        """
        self.event_bus = event_bus or EventBus()
        self.data_dir = data_dir
        self.market_data = {}
        self.volatility_states = {}
        self.correlations = {}
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Try to import yfinance, use mock data if not available
        try:
            import yfinance as yf
            self.yf = yf
            self.has_yfinance = True
            logger.info("Using yfinance library for market data")
        except ImportError:
            self.has_yfinance = False
            logger.warning("yfinance library not available. Will use cached data if available.")
    
    def download_data(self, symbols, start_date, end_date, interval='1d', force_download=False):
        """
        Download historical data for specified symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for data download (YYYY-MM-DD)
            end_date: End date for data download (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
            force_download: Whether to force download even if cache exists
            
        Returns:
            Dictionary of symbol -> DataFrame with market data
        """
        for symbol in symbols:
            cache_path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
            
            if os.path.exists(cache_path) and not force_download:
                # Load from cache
                logger.info(f"Loading cached data for {symbol}")
                df = pd.read_csv(cache_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                self.market_data[symbol] = df
            else:
                # Download data
                logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
                
                if not self.has_yfinance:
                    logger.error(f"yfinance is not available. Cannot download data for {symbol}.")
                    continue
                
                try:
                    # Format ticker symbol for yfinance
                    yf_symbol = symbol
                    
                    # Special handling for forex pairs
                    if len(symbol) == 6 and not symbol.endswith('=X'):
                        # Looks like a forex pair (e.g. EURUSD), so format for yfinance
                        yf_symbol = f"{symbol}=X"
                        logger.info(f"Forex pair detected, using yfinance symbol: {yf_symbol}")
                    
                    data = self.yf.download(
                        yf_symbol,
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=True,
                        progress=False
                    )
                    
                    if not data.empty:
                        # Add derived metrics
                        data['Returns'] = data['Close'].pct_change()
                        
                        # Save to cache
                        data.to_csv(cache_path)
                        
                        # Store in memory
                        self.market_data[symbol] = data
                        logger.info(f"Downloaded {len(data)} rows for {symbol}")
                    else:
                        logger.warning(f"No data found for {symbol}")
                except Exception as e:
                    logger.error(f"Error downloading data for {symbol}: {str(e)}")
        
        return self.market_data
    
    def load_cached_data(self, symbols, interval='1d'):
        """
        Load cached data for specified symbols.
        
        Args:
            symbols: List of ticker symbols
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            Dictionary of symbol -> DataFrame with market data
        """
        for symbol in symbols:
            cache_path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
            
            if os.path.exists(cache_path):
                logger.info(f"Loading cached data for {symbol}")
                try:
                    # Check first line to determine file format
                    with open(cache_path, 'r') as f:
                        first_line = f.readline().strip()
                    
                    if first_line.startswith('Price,'):
                        # This is the yfinance format with multi-level headers
                        logger.info(f"Detected yfinance multi-level format for {symbol}")
                        
                        # Custom parsing for yfinance format
                        # Skip the first 3 rows (Price, Ticker, Date)
                        df = pd.read_csv(cache_path, skiprows=3)
                        
                        # First column is the date
                        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
                        df['Date'] = pd.to_datetime(df['Date'])
                        df.set_index('Date', inplace=True)
                        
                        # Rename remaining columns to standard OHLC
                        column_map = {}
                        if len(df.columns) >= 5:  # We expect at least 5 columns
                            column_map = {
                                df.columns[0]: 'Close',
                                df.columns[1]: 'High',
                                df.columns[2]: 'Low',
                                df.columns[3]: 'Open',
                                df.columns[4]: 'Volume'
                            }
                            df.rename(columns=column_map, inplace=True)
                    else:
                        # Standard CSV format
                        df = pd.read_csv(cache_path)
                        
                        # Find the date column
                        date_col = None
                        for col in df.columns:
                            if col.lower() in ['date', 'datetime', 'time']:
                                date_col = col
                                break
                        
                        if date_col:
                            df[date_col] = pd.to_datetime(df[date_col])
                            df.set_index(date_col, inplace=True)
                        else:
                            # Assume the first column is the date
                            logger.warning(f"No date column found for {symbol}, using first column as index")
                            first_col = df.columns[0]
                            df[first_col] = pd.to_datetime(df[first_col])
                            df.set_index(first_col, inplace=True)
                    
                    # Make sure we have all required OHLC columns
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    for col_name in required_cols:
                        if col_name not in df.columns:
                            # Try to find a matching column case-insensitive
                            for existing_col in df.columns:
                                if existing_col.lower() == col_name.lower():
                                    df[col_name] = df[existing_col]
                                    break
                            else:
                                # If we still don't have the column, create it with Close values
                                if 'Close' in df.columns:
                                    df[col_name] = df['Close']
                                else:
                                    # If no Close column, use the first numeric column
                                    for col in df.columns:
                                        if pd.api.types.is_numeric_dtype(df[col]):
                                            df[col_name] = df[col]
                                            break
                    
                    # Add a Returns column if not present
                    if 'Returns' not in df.columns and 'Close' in df.columns:
                        df['Returns'] = df['Close'].pct_change()
                    
                    # Store the loaded data
                    self.market_data[symbol] = df
                    logger.info(f"Loaded {len(df)} rows for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {str(e)}")
            else:
                logger.warning(f"No cached data found for {symbol}")
        
        return self.market_data
    
    def calculate_volatility(self, symbols=None, window=20):
        """
        Calculate and classify volatility for specified symbols.
        
        Args:
            symbols: List of symbols to calculate volatility for
            window: Window size for volatility calculation
            
        Returns:
            Dictionary of symbol -> volatility state
        """
        symbols = symbols or list(self.market_data.keys())
        volatility_states = {}
        
        for symbol in symbols:
            if symbol not in self.market_data:
                logger.warning(f"No data found for {symbol} when calculating volatility")
                continue
            
            df = self.market_data[symbol]
            
            # Calculate historical volatility (standard deviation of returns)
            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change()
            
            # Calculate rolling volatility
            df['Volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
            
            # Classify volatility state
            volatility_values = df['Volatility'].dropna()
            if len(volatility_values) < window:
                logger.warning(f"Not enough data to calculate volatility for {symbol}")
                continue
            
            # Get current volatility and classify
            current_vol = volatility_values.iloc[-1]
            
            # Calculate percentiles for classification
            vol_20 = np.percentile(volatility_values, 20)
            vol_80 = np.percentile(volatility_values, 80)
            
            if current_vol < vol_20:
                state = 'low'
            elif current_vol > vol_80:
                state = 'high'
            else:
                state = 'medium'
                
            volatility_states[symbol] = state
            
            # Publish volatility update event
            if self.event_bus:
                self.event_bus.publish(Event(
                    EventType.VOLATILITY_UPDATE,
                    {
                        'symbol': symbol,
                        'volatility_state': state,
                        'volatility_value': current_vol,
                        'previous_state': self.volatility_states.get(symbol, 'unknown')
                    }
                ))
            
            logger.info(f"Volatility for {symbol}: {state} ({current_vol:.4f})")
        
        self.volatility_states = volatility_states
        return volatility_states
    
    def calculate_correlations(self, symbols=None, window=60):
        """
        Calculate correlation matrix for specified symbols.
        
        Args:
            symbols: List of symbols to include in correlation matrix
            window: Window size for correlation calculation
            
        Returns:
            Correlation matrix as DataFrame
        """
        symbols = symbols or list(self.market_data.keys())
        
        # Collect returns for each symbol
        returns_dict = {}
        for symbol in symbols:
            if symbol not in self.market_data:
                logger.warning(f"No data found for {symbol} when calculating correlations")
                continue
            
            df = self.market_data[symbol]
            
            if 'Returns' not in df.columns:
                df['Returns'] = df['Close'].pct_change()
            
            returns_dict[symbol] = df['Returns']
        
        # Create a DataFrame with all returns
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate rolling correlation
        correlations = returns_df.rolling(window=window).corr()
        
        # Get the most recent correlation matrix
        latest_date = returns_df.index[-1]
        latest_corr = correlations.xs(latest_date, level=0, drop_level=True)
        
        self.correlations = latest_corr
        
        # Publish correlation update event
        if self.event_bus:
            self.event_bus.publish(Event(
                EventType.CORRELATION_UPDATE,
                {
                    'correlation_matrix': latest_corr.to_dict(),
                    'timestamp': latest_date.strftime('%Y-%m-%d %H:%M:%S')
                }
            ))
        
        return latest_corr


class MarketRegimeDetector:
    """Detects market regimes based on technical indicators"""
    
    def __init__(self, event_bus=None):
        self.event_bus = event_bus or EventBus()
        self.current_regimes = {}
        
        # Thresholds for regime detection
        self.volatility_threshold_high = 0.015  # 1.5% daily volatility is high
        self.volatility_threshold_low = 0.005   # 0.5% daily volatility is low
        self.trend_strength_threshold = 0.6     # ADX above 20 indicates trend
        
    def detect_regime(self, symbol, data, lookback=20):
        """
        Detect the current market regime for a symbol.
        
        Regimes:
        - trending_up: Price in uptrend with consistent momentum
        - trending_down: Price in downtrend with consistent momentum
        - ranging: Price moving sideways in a range
        - breakout: Price breaking out of a range with increased volatility
        - unknown: Not enough data or unclear regime
        """
        if len(data) < lookback*2:
            return {'regime': 'unknown', 'confidence': 0.0}
            
        # Calculate indicators
        # 1. Trend indicators (EMA, ADX)
        data['EMA20'] = data['Close'].ewm(span=20).mean()
        data['EMA50'] = data['Close'].ewm(span=50).mean()
        
        # Calculate directional movement for ADX
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift(1)),
                abs(data['Low'] - data['Close'].shift(1))
            )
        )
        data['ATR14'] = data['TR'].rolling(window=14).mean()
        
        # 2. Volatility indicators (Bollinger Bands width)
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['StdDev'] = data['Close'].rolling(window=20).std()
        data['BBUpper'] = data['SMA20'] + 2 * data['StdDev']
        data['BBLower'] = data['SMA20'] - 2 * data['StdDev']
        data['BBWidth'] = (data['BBUpper'] - data['BBLower']) / data['SMA20']
        
        # 3. Oscillator (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Get recent data for analysis
        recent = data.iloc[-lookback:].copy()
        
        # Calculate regime features
        is_uptrend = recent['Close'].iloc[-1] > recent['EMA20'].iloc[-1] > recent['EMA50'].iloc[-1]
        is_downtrend = recent['Close'].iloc[-1] < recent['EMA20'].iloc[-1] < recent['EMA50'].iloc[-1]
        
        bb_width = recent['BBWidth'].iloc[-1]
        bb_width_avg = recent['BBWidth'].mean()
        
        is_tight_range = bb_width < bb_width_avg * 0.8
        is_expanding_range = bb_width > bb_width_avg * 1.2
        
        rsi = recent['RSI'].iloc[-1]
        is_overbought = rsi > 70
        is_oversold = rsi < 30
        
        # Determine regime and confidence
        regime = 'unknown'
        confidence = 0.5
        
        if is_uptrend:
            if is_expanding_range and recent['ATR14'].pct_change(5).iloc[-1] > 0.1:
                regime = 'breakout'
                confidence = 0.65
            else:
                regime = 'trending_up'
                confidence = 0.7
        elif is_downtrend:
            regime = 'trending_down'
            confidence = 0.7
        elif is_tight_range:
            regime = 'ranging'
            confidence = 0.6
            
        # Modify confidence based on RSI extremes
        if regime == 'trending_up' and is_overbought:
            confidence *= 0.9  # Reduce confidence if overbought
        elif regime == 'trending_down' and is_oversold:
            confidence *= 0.9  # Reduce confidence if oversold
            
        # Return regime information
        regime_info = {
            'regime': regime,
            'confidence': confidence,
            'details': {
                'uptrend': is_uptrend,
                'downtrend': is_downtrend,
                'bb_width': bb_width,
                'rsi': rsi
            }
        }
        
        # Check if regime changed
        if symbol in self.current_regimes and self.current_regimes[symbol] != regime:
            if self.event_bus:
                self.event_bus.publish(Event(
                    EventType.MARKET_REGIME_CHANGE,
                    {
                        'symbol': symbol,
                        'regime': regime,
                        'previous_regime': self.current_regimes[symbol],
                        'confidence': confidence
                    }
                ))
            
        # Update current regime
        self.current_regimes[symbol] = regime
        
        return regime_info
