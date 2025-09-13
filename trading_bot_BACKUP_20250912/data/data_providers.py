"""
Data Providers Module

This module provides data provider interfaces for accessing market data 
from various sources.
"""

import logging
import pandas as pd
import os
from datetime import datetime, timedelta
import random
import numpy as np

logger = logging.getLogger(__name__)

class BaseDataProvider:
    """Base class for all data providers"""
    
    def __init__(self, config=None):
        """
        Initialize the data provider
        
        Args:
            config: Optional configuration dictionary or path to config file
        """
        self.config = config or {}
        logger.info(f"Initialized {self.__class__.__name__}")
    
    def get_historical_data(self, symbols, start_date, end_date):
        """
        Get historical price data for symbols
        
        Args:
            symbols: List of symbols to get data for
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        raise NotImplementedError("Subclasses must implement get_historical_data")
    
    def get_current_price(self, symbols):
        """
        Get current price for symbols
        
        Args:
            symbols: List of symbols to get prices for
            
        Returns:
            Dictionary mapping symbols to current prices
        """
        raise NotImplementedError("Subclasses must implement get_current_price")


class MockDataProvider(BaseDataProvider):
    """Mock data provider that generates random data"""
    
    def get_historical_data(self, symbols, start_date, end_date):
        """
        Generate mock historical data
        
        Args:
            symbols: List of symbols to get data for
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        result = {}
        
        # Convert strings to datetime objects
        start = datetime.strptime(start_date.split()[0], "%Y-%m-%d") 
        end = datetime.strptime(end_date.split()[0], "%Y-%m-%d")
        
        # Generate date range
        dates = pd.date_range(start=start, end=end)
        
        for symbol in symbols:
            # Start with a random price
            base_price = random.uniform(50, 500)
            
            # Generate random price data
            prices = [base_price]
            for i in range(1, len(dates)):
                # Random price change with slight upward bias
                change = np.random.normal(0.0002, 0.015)
                prices.append(prices[-1] * (1 + change))
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + random.uniform(0, 0.02)) for p in prices],
                'low': [p * (1 - random.uniform(0, 0.02)) for p in prices],
                'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
                'volume': [int(random.uniform(100000, 10000000)) for _ in range(len(dates))]
            }, index=dates)
            
            result[symbol] = df
            
        logger.info(f"Generated mock data for {len(symbols)} symbols from {start_date} to {end_date}")
        return result
    
    def get_current_price(self, symbols):
        """
        Generate mock current prices
        
        Args:
            symbols: List of symbols to get prices for
            
        Returns:
            Dictionary mapping symbols to current prices
        """
        result = {}
        for symbol in symbols:
            result[symbol] = random.uniform(50, 500)
        
        logger.info(f"Generated mock current prices for {len(symbols)} symbols")
        return result


class AlpacaDataProvider(BaseDataProvider):
    """Simple implementation of Alpaca data provider"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = self.config.get('api_key', 'demo')
        logger.info("Initialized AlpacaDataProvider")
    
    def get_historical_data(self, symbols, start_date, end_date):
        """
        Get mock data since we can't actually connect to Alpaca 
        without API credentials
        """
        # Create mock data provider and delegate
        mock = MockDataProvider(self.config)
        return mock.get_historical_data(symbols, start_date, end_date)
    
    def get_current_price(self, symbols):
        """
        Get mock prices since we can't actually connect to Alpaca
        without API credentials
        """
        # Create mock data provider and delegate
        mock = MockDataProvider(self.config)
        return mock.get_current_price(symbols)


def create_data_provider(provider_type, config_path=None):
    """
    Create a data provider based on type
    
    Args:
        provider_type: Type of provider ('mock', 'alpaca', etc.)
        config_path: Optional path to configuration file
    
    Returns:
        Data provider instance
    """
    # Load config if provided
    config = {}
    if config_path and os.path.exists(config_path):
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
    
    # Create provider based on type
    if provider_type.lower() == 'mock':
        return MockDataProvider(config)
    elif provider_type.lower() == 'alpaca':
        return AlpacaDataProvider(config)
    else:
        logger.warning(f"Unknown provider type: {provider_type}. Using mock provider.")
        return MockDataProvider(config) 