#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Data Provider

This module provides a simplified data provider that works with CSV files
instead of requiring external dependencies like yfinance and matplotlib.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import csv
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleDataProvider")

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

class SimpleDataProvider:
    """Data provider that works with CSV files or generates synthetic data"""
    
    def __init__(self, event_bus=None, data_dir="data/market_data"):
        """
        Initialize the simple data provider.
        
        Args:
            event_bus: Event bus for publishing market data events
            data_dir: Directory for storing data
        """
        self.event_bus = event_bus or EventBus()
        self.data_dir = data_dir
        self.market_data = {}
        self.volatility_states = {}
        self.correlations = {}
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def load_data(self, symbols, interval='1d'):
        """
        Load data for specified symbols from CSV files.
        
        Args:
            symbols: List of ticker symbols
            interval: Data interval ('1d', '1h', etc.)
            
        Returns:
            Dictionary of symbol -> DataFrame with market data
        """
        for symbol in symbols:
            cache_path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
            
            if os.path.exists(cache_path):
                logger.info(f"Loading data for {symbol} from {cache_path}")
                try:
                    df = pd.read_csv(cache_path)
                    
                    # Convert date column to datetime
                    date_col = None
                    for col in df.columns:
                        if 'date' in col.lower() or 'time' in col.lower():
                            date_col = col
                            break
                    
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col])
                        df.set_index(date_col, inplace=True)
                    
                    # Standardize column names
                    col_map = {}
                    for col in df.columns:
                        if 'open' in col.lower():
                            col_map[col] = 'Open'
                        elif 'high' in col.lower():
                            col_map[col] = 'High'
                        elif 'low' in col.lower():
                            col_map[col] = 'Low'
                        elif 'close' in col.lower():
                            col_map[col] = 'Close'
                        elif 'volume' in col.lower():
                            col_map[col] = 'Volume'
                    
                    if col_map:
                        df.rename(columns=col_map, inplace=True)
                    
                    # Make sure we have all required columns
                    required_cols = ['Open', 'High', 'Low', 'Close']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                        if 'Close' in df.columns:
                            # Fill missing columns with Close price if available
                            for col in missing_cols:
                                df[col] = df['Close']
                    
                    # Add derived columns
                    if 'Returns' not in df.columns:
                        df['Returns'] = df['Close'].pct_change()
                    
                    self.market_data[symbol] = df
                    logger.info(f"Loaded {len(df)} rows for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {str(e)}")
            else:
                logger.warning(f"No data file found for {symbol} at {cache_path}")
                
                # Generate synthetic data if file doesn't exist
                logger.info(f"Generating synthetic data for {symbol}")
                self._generate_synthetic_data(symbol, interval)
        
        return self.market_data
    
    def _generate_synthetic_data(self, symbol, interval='1d'):
        """Generate synthetic price data for backtesting."""
        logger.info(f"Generating synthetic data for {symbol}")
        
        # Number of days to generate
        days = 500
        
        # Create a date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize price at 100
        price = 100.0
        
        # Create data lists
        data = []
        
        # Random parameters for this symbol
        volatility = np.random.uniform(0.005, 0.02)  # Daily volatility
        drift = np.random.uniform(0.0001, 0.0005)    # Daily drift
        
        # Generate price series
        for i, date in enumerate(dates):
            # Random daily return
            daily_return = np.random.normal(drift, volatility)
            
            # Update price
            price *= (1 + daily_return)
            
            # Calculate high, low, open prices
            daily_range = price * np.random.uniform(0.002, 0.015)
            high = price + daily_range / 2
            low = price - daily_range / 2
            open_price = low + np.random.uniform(0, daily_range)
            
            # Volume (not really used but include for completeness)
            volume = np.random.randint(1000, 10000)
            
            # Add to data list
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': price,
                'Volume': volume
            })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # Add derived columns
        df['Returns'] = df['Close'].pct_change()
        
        # Save to CSV
        cache_path = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
        df.to_csv(cache_path)
        
        # Store in memory
        self.market_data[symbol] = df
        
        logger.info(f"Generated and saved {len(df)} rows of data for {symbol}")
        return df
    
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
        
        # Calculate correlation
        correlations = returns_df.corr()
        
        self.correlations = correlations
        
        # Publish correlation update event
        if self.event_bus:
            self.event_bus.publish(Event(
                EventType.CORRELATION_UPDATE,
                {
                    'correlation_matrix': correlations.to_dict(),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            ))
        
        return correlations
