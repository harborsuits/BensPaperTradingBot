#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Manager

This module provides a central manager for all data operations,
coordinating the data providers and storage.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.data.yahoo_finance_provider import YahooFinanceProvider
from trading_bot.data.data_storage import DataStorage
from trading_bot.data.real_time_provider import RealTimeProvider

logger = logging.getLogger(__name__)

class DataManager:
    """
    Data manager that coordinates data providers and storage.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.market_data_providers = {}
        self.real_time_providers = {}
        self.data_storage = None
        self.cache_enabled = config.get("enable_cache", True)
        self.cache_expiry_minutes = config.get("cache_expiry_minutes", 30)
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize components
        self._initialize_components()
        
        logger.info("Data manager initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all data components based on configuration."""
        try:
            # Initialize data storage
            storage_config = self.config.get("data_storage", {})
            base_dir = storage_config.get("base_dir", "data")
            self.data_storage = DataStorage(base_dir=base_dir)
            
            # Register data storage in service registry
            ServiceRegistry.register("data_storage", self.data_storage, DataStorage)
            
            # Initialize market data providers
            providers_config = self.config.get("data_providers", [])
            for provider_config in providers_config:
                provider_name = provider_config.get("name", "")
                if not provider_name:
                    continue
                
                provider_type = provider_config.get("type", "")
                
                if provider_name == "yahoo_finance" or provider_type == "python_library":
                    self._initialize_yahoo_finance(provider_config)
                    
            # Initialize real-time providers
            realtime_config = self.config.get("realtime_providers", [])
            for provider_config in realtime_config:
                provider_name = provider_config.get("provider", "")
                if not provider_name:
                    continue
                
                if provider_name == "finnhub":
                    self._initialize_finnhub(provider_config)
            
            # Register data manager in service registry
            ServiceRegistry.register("data_manager", self, DataManager)
            
        except Exception as e:
            logger.error(f"Error initializing data components: {e}")
            raise
    
    def _initialize_yahoo_finance(self, config: Dict[str, Any]) -> None:
        """
        Initialize Yahoo Finance data provider.
        
        Args:
            config: Provider configuration
        """
        try:
            provider = YahooFinanceProvider(config)
            provider_name = config.get("name", "yahoo_finance")
            
            self.market_data_providers[provider_name] = provider
            
            # Register in service registry
            ServiceRegistry.register(f"data_provider.{provider_name}", provider, YahooFinanceProvider)
            
            logger.info(f"Initialized Yahoo Finance data provider as '{provider_name}'")
            
        except Exception as e:
            logger.error(f"Error initializing Yahoo Finance data provider: {e}")
    
    def _initialize_finnhub(self, config: Dict[str, Any]) -> None:
        """
        Initialize Finnhub real-time data provider.
        
        Args:
            config: Provider configuration
        """
        try:
            provider = RealTimeProvider(config)
            provider_name = config.get("name", "finnhub")
            
            self.real_time_providers[provider_name] = provider
            
            # Register in service registry
            ServiceRegistry.register(f"realtime_provider.{provider_name}", provider, RealTimeProvider)
            
            logger.info(f"Initialized Finnhub real-time provider as '{provider_name}'")
            
        except Exception as e:
            logger.error(f"Error initializing Finnhub real-time provider: {e}")
    
    def get_market_data(self, symbols: Union[str, List[str]], 
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      provider_name: Optional[str] = None,
                      use_cache: Optional[bool] = None) -> Dict[str, pd.DataFrame]:
        """
        Get market data for symbols.
        
        Args:
            symbols: Symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            provider_name: Specific provider to use
            use_cache: Whether to use cache
            
        Returns:
            Dictionary mapping symbols to DataFrames with market data
        """
        # Normalize symbols to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # Determine whether to use cache
        if use_cache is None:
            use_cache = self.cache_enabled
        
        # Initialize result dict
        result = {}
        
        # Symbols to fetch from providers
        symbols_to_fetch = set(symbols)
        
        # Try to get data from cache first
        if use_cache:
            for symbol in symbols:
                cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                
                if self._is_cache_valid(cache_key):
                    result[symbol] = self.cache[cache_key]
                    symbols_to_fetch.remove(symbol)
                    logger.debug(f"Using cached data for {symbol}")
        
        # If all symbols were found in cache, return result
        if not symbols_to_fetch:
            return result
        
        # Try to get data from storage for remaining symbols
        for symbol in list(symbols_to_fetch):
            data = self.data_storage.load_market_data(symbol, "ohlcv")
            
            if data is not None:
                # Filter data by date range
                if 'date' in data.columns:
                    data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
                
                if not data.empty:
                    result[symbol] = data
                    symbols_to_fetch.remove(symbol)
                    
                    # Add to cache
                    if use_cache:
                        cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                        self.cache[cache_key] = data
                        self.cache_timestamps[cache_key] = datetime.now()
                    
                    logger.debug(f"Using stored data for {symbol}")
        
        # If all symbols were found in storage, return result
        if not symbols_to_fetch:
            return result
        
        # Fetch remaining symbols from providers
        if not self.market_data_providers:
            logger.warning("No market data providers available")
            return result
        
        # Determine which provider to use
        if provider_name and provider_name in self.market_data_providers:
            providers_to_try = [provider_name]
        else:
            # Try all providers in order of priority
            providers_to_try = list(self.market_data_providers.keys())
        
        # Try each provider until data is found
        for provider_name in providers_to_try:
            provider = self.market_data_providers[provider_name]
            
            try:
                provider_data = provider.get_market_data(
                    list(symbols_to_fetch),
                    start_date=start_date,
                    end_date=end_date
                )
                
                for symbol, data in provider_data.items():
                    if data is not None and not data.empty:
                        result[symbol] = data
                        symbols_to_fetch.remove(symbol)
                        
                        # Save to storage
                        self.data_storage.save_market_data(symbol, data, "ohlcv")
                        
                        # Add to cache
                        if use_cache:
                            cache_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                            self.cache[cache_key] = data
                            self.cache_timestamps[cache_key] = datetime.now()
                
                # If all symbols were found, break
                if not symbols_to_fetch:
                    break
                
            except Exception as e:
                logger.error(f"Error fetching data from provider '{provider_name}': {e}")
        
        return result
    
    def get_option_chain(self, symbol: str, expiration_date: Optional[str] = None,
                       provider_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get option chain data for a symbol.
        
        Args:
            symbol: Symbol to get options for
            expiration_date: Specific expiration date
            provider_name: Specific provider to use
            
        Returns:
            Option chain data
        """
        # Try to get data from storage first
        if expiration_date:
            data = self.data_storage.load_option_data(symbol, expiration_date)
            
            if data is not None:
                logger.debug(f"Using stored option data for {symbol} expiration {expiration_date}")
                return data
        
        # Fetch from providers if not found in storage
        if not self.market_data_providers:
            logger.warning("No market data providers available")
            return {}
        
        # Determine which provider to use
        if provider_name and provider_name in self.market_data_providers:
            providers_to_try = [provider_name]
        else:
            # Try all providers in order of priority
            providers_to_try = list(self.market_data_providers.keys())
        
        # Try each provider until data is found
        for provider_name in providers_to_try:
            provider = self.market_data_providers[provider_name]
            
            try:
                data = provider.get_option_chain(symbol, expiration_date)
                
                if data and (
                    ('chains' in data and data['chains']) or 
                    ('calls' in data and not data['calls'].empty) or
                    ('puts' in data and not data['puts'].empty)
                ):
                    # Save to storage if specific expiration
                    if expiration_date:
                        self.data_storage.save_option_data(symbol, expiration_date, data)
                    
                    return data
                
            except Exception as e:
                logger.error(f"Error fetching option data from provider '{provider_name}': {e}")
        
        return {}
    
    def start_realtime_stream(self, provider_name: Optional[str] = None) -> bool:
        """
        Start real-time data streaming.
        
        Args:
            provider_name: Specific provider to start
            
        Returns:
            True if started successfully, False otherwise
        """
        if not self.real_time_providers:
            logger.warning("No real-time providers available")
            return False
        
        success = True
        
        # Determine which providers to start
        if provider_name:
            providers_to_start = {provider_name: self.real_time_providers.get(provider_name)}
        else:
            providers_to_start = self.real_time_providers
        
        # Start each provider
        for name, provider in providers_to_start.items():
            if provider is None:
                logger.warning(f"Real-time provider '{name}' not found")
                success = False
                continue
            
            try:
                provider_success = provider.start()
                if not provider_success:
                    logger.error(f"Failed to start real-time provider '{name}'")
                    success = False
                
            except Exception as e:
                logger.error(f"Error starting real-time provider '{name}': {e}")
                success = False
        
        return success
    
    def stop_realtime_stream(self, provider_name: Optional[str] = None) -> None:
        """
        Stop real-time data streaming.
        
        Args:
            provider_name: Specific provider to stop
        """
        if not self.real_time_providers:
            return
        
        # Determine which providers to stop
        if provider_name:
            providers_to_stop = {provider_name: self.real_time_providers.get(provider_name)}
        else:
            providers_to_stop = self.real_time_providers
        
        # Stop each provider
        for name, provider in providers_to_stop.items():
            if provider is None:
                continue
            
            try:
                provider.stop()
            except Exception as e:
                logger.error(f"Error stopping real-time provider '{name}': {e}")
    
    def subscribe_to_realtime(self, symbol: str, callback, provider_name: Optional[str] = None) -> bool:
        """
        Subscribe to real-time data for a symbol.
        
        Args:
            symbol: Symbol to subscribe to
            callback: Callback function for data updates
            provider_name: Specific provider to use
            
        Returns:
            True if subscribed successfully, False otherwise
        """
        if not self.real_time_providers:
            logger.warning("No real-time providers available")
            return False
        
        # Determine which provider to use
        if provider_name and provider_name in self.real_time_providers:
            provider = self.real_time_providers[provider_name]
            
            if provider is None:
                logger.warning(f"Real-time provider '{provider_name}' not found")
                return False
            
            # Subscribe to symbol
            return provider.subscribe(symbol, callback)
            
        else:
            # Try all providers, use first successful one
            for name, provider in self.real_time_providers.items():
                try:
                    success = provider.subscribe(symbol, callback)
                    if success:
                        logger.info(f"Subscribed to {symbol} using provider '{name}'")
                        return True
                
                except Exception as e:
                    logger.error(f"Error subscribing to {symbol} with provider '{name}': {e}")
            
            logger.warning(f"Failed to subscribe to {symbol} with any provider")
            return False
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        if cache_key not in self.cache or cache_key not in self.cache_timestamps:
            return False
        
        # Check if cache has expired
        cache_age = datetime.now() - self.cache_timestamps[cache_key]
        return cache_age.total_seconds() < (self.cache_expiry_minutes * 60)
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Data manager cache cleared")
        
        # Clear provider caches as well
        for provider_name, provider in self.market_data_providers.items():
            if hasattr(provider, 'clear_cache'):
                try:
                    provider.clear_cache()
                except Exception as e:
                    logger.error(f"Error clearing cache for provider '{provider_name}': {e}")
    
    def shutdown(self) -> None:
        """Shutdown all data components."""
        # Stop real-time providers
        self.stop_realtime_stream()
        
        # Clear cache
        self.clear_cache()
        
        logger.info("Data manager shutdown complete") 