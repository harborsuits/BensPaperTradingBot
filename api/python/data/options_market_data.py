#!/usr/bin/env python3
"""
Options Market Data

This module provides functionality for retrieving options market data from multiple sources.
It handles data source prioritization, caching, and error recovery.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DataSourcePriority(Enum):
    """Priority levels for data sources."""
    PRIMARY = auto()
    SECONDARY = auto()
    TERTIARY = auto()
    FALLBACK = auto()

@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    priority: DataSourcePriority
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    active: bool = True
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize the config dict if None."""
        if self.config is None:
            self.config = {}

class OptionsMarketData:
    """
    Provides unified access to options market data from multiple sources.
    
    Features:
    - Multi-source data retrieval with fallback mechanisms
    - Local caching of options chains and historical data
    - Automatic data quality assessment and gap filling
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        cache_dir: str = "data/options_cache",
        enable_local_cache: bool = True,
        cache_ttl: int = 3600  # Default cache TTL in seconds
    ):
        """
        Initialize the options market data provider.
        
        Args:
            config_path: Path to configuration file (optional)
            cache_dir: Directory for caching options data
            enable_local_cache: Whether to use local caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.data_sources: Dict[str, DataSource] = {}
        self.cache_dir = cache_dir
        self.enable_local_cache = enable_local_cache
        self.cache_ttl = cache_ttl
        
        # Create cache directory if it doesn't exist
        if self.enable_local_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract options data source configuration
            if 'options_data_sources' in config:
                for source_config in config['options_data_sources']:
                    name = source_config.get('name')
                    if not name:
                        logger.warning("Skipping data source without name in config")
                        continue
                    
                    # Get priority from config or default to FALLBACK
                    priority_str = source_config.get('priority', 'FALLBACK')
                    priority = getattr(DataSourcePriority, priority_str.upper(), 
                                      DataSourcePriority.FALLBACK)
                    
                    # Create data source
                    self.data_sources[name] = DataSource(
                        name=name,
                        priority=priority,
                        api_key=source_config.get('api_key'),
                        base_url=source_config.get('base_url'),
                        active=source_config.get('active', True),
                        config=source_config.get('config', {})
                    )
                    
                    logger.info(f"Loaded data source: {name} with priority {priority}")
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading options data configuration: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading options data configuration: {str(e)}")
    
    def register_data_source(
        self, 
        name: str, 
        priority: DataSourcePriority,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new data source.
        
        Args:
            name: Name of the data source
            priority: Priority level
            api_key: API key for the data source (if needed)
            base_url: Base URL for API calls (if needed)
            config: Additional configuration parameters
        """
        self.data_sources[name] = DataSource(
            name=name,
            priority=priority,
            api_key=api_key,
            base_url=base_url,
            active=True,
            config=config or {}
        )
        logger.info(f"Registered data source: {name} with priority {priority}")
    
    def register_mock_data_source(
        self,
        source_name: str = "mock",
        priority: DataSourcePriority = DataSourcePriority.FALLBACK
    ) -> None:
        """
        Register a mock data source for testing or demo purposes.
        
        Args:
            source_name: Name for the mock data source
            priority: Priority level
        """
        self.register_data_source(
            name=source_name,
            priority=priority,
            config={"is_mock": True}
        )
        logger.info(f"Registered mock data source: {source_name}")
    
    def deactivate_data_source(self, name: str) -> bool:
        """
        Deactivate a data source.
        
        Args:
            name: Name of the data source to deactivate
            
        Returns:
            True if deactivated, False if not found
        """
        if name in self.data_sources:
            self.data_sources[name].active = False
            logger.info(f"Deactivated data source: {name}")
            return True
        return False
    
    def get_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        strikes: Optional[List[float]] = None,
        option_type: Optional[str] = None,  # 'call', 'put', or None for both
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get options chain data for a symbol.
        
        Args:
            symbol: Ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD format)
            strikes: List of specific strike prices to retrieve
            option_type: Filter by option type ('call', 'put', or None for both)
            force_refresh: Force refresh from source instead of using cache
            
        Returns:
            Options chain data dictionary
        """
        # Check cache first if enabled and not forcing refresh
        if self.enable_local_cache and not force_refresh:
            cached_data = self._get_from_cache(
                symbol=symbol,
                data_type="options_chain",
                expiration=expiration_date
            )
            if cached_data:
                logger.debug(f"Using cached options chain for {symbol}")
                return cached_data
        
        # Order data sources by priority
        ordered_sources = self._get_ordered_data_sources()
        
        # Try each data source until we get data
        errors = []
        for source in ordered_sources:
            try:
                # Get data from source based on source type
                data = self._fetch_options_chain_from_source(
                    source=source,
                    symbol=symbol,
                    expiration_date=expiration_date,
                    strikes=strikes,
                    option_type=option_type
                )
                
                if data:
                    # Cache the data if caching is enabled
                    if self.enable_local_cache:
                        self._save_to_cache(
                            data=data,
                            symbol=symbol,
                            data_type="options_chain",
                            expiration=expiration_date
                        )
                    return data
            
            except Exception as e:
                error_msg = f"Error getting options chain from {source.name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        # If we've tried all sources and still don't have data, use mock data
        logger.warning(f"Failed to get options chain for {symbol} from all sources")
        mock_data = self._generate_mock_options_chain(
            symbol=symbol,
            expiration_date=expiration_date,
            strikes=strikes,
            option_type=option_type
        )
        
        # Return mock data with errors
        mock_data["_errors"] = errors
        mock_data["_is_mock"] = True
        
        return mock_data
    
    def get_iv_surface(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get implied volatility surface data.
        
        Args:
            symbol: Ticker symbol
            force_refresh: Force refresh from source instead of using cache
            
        Returns:
            IV surface data dictionary
        """
        # Check cache first if enabled and not forcing refresh
        if self.enable_local_cache and not force_refresh:
            cached_data = self._get_from_cache(
                symbol=symbol,
                data_type="iv_surface"
            )
            if cached_data:
                logger.debug(f"Using cached IV surface for {symbol}")
                return cached_data
        
        # Order data sources by priority
        ordered_sources = self._get_ordered_data_sources()
        
        # Try each data source until we get data
        errors = []
        for source in ordered_sources:
            try:
                # Get data from source based on source type
                data = self._fetch_iv_surface_from_source(
                    source=source,
                    symbol=symbol
                )
                
                if data:
                    # Cache the data if caching is enabled
                    if self.enable_local_cache:
                        self._save_to_cache(
                            data=data,
                            symbol=symbol,
                            data_type="iv_surface"
                        )
                    return data
            
            except Exception as e:
                error_msg = f"Error getting IV surface from {source.name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        # If we've tried all sources and still don't have data, use mock data
        logger.warning(f"Failed to get IV surface for {symbol} from all sources")
        mock_data = self._generate_mock_iv_surface(symbol=symbol)
        
        # Return mock data with errors
        mock_data["_errors"] = errors
        mock_data["_is_mock"] = True
        
        return mock_data
    
    def get_iv_history(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: int = 30,  # Default to 30 days if no dates provided
        fill_gaps: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical implied volatility data.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            period: Number of days to retrieve if no dates provided
            fill_gaps: Fill gaps in the data using interpolation
            force_refresh: Force refresh from source instead of using cache
            
        Returns:
            DataFrame with historical IV data
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            start = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=period)
            start_date = start.strftime('%Y-%m-%d')
        
        # Check cache first if enabled and not forcing refresh
        if self.enable_local_cache and not force_refresh:
            cached_data = self._get_from_cache(
                symbol=symbol,
                data_type="iv_history",
                start_date=start_date,
                end_date=end_date
            )
            if isinstance(cached_data, dict) and "data" in cached_data:
                try:
                    df = pd.DataFrame(cached_data["data"])
                    logger.debug(f"Using cached IV history for {symbol}")
                    return df
                except Exception as e:
                    logger.warning(f"Error parsing cached IV history: {str(e)}")
        
        # Order data sources by priority
        ordered_sources = self._get_ordered_data_sources()
        
        # Try each data source until we get data
        errors = []
        for source in ordered_sources:
            try:
                # Get data from source based on source type
                df = self._fetch_iv_history_from_source(
                    source=source,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and not df.empty:
                    # Fill gaps if requested
                    if fill_gaps:
                        df = self._fill_iv_history_gaps(df, start_date, end_date)
                    
                    # Cache the data if caching is enabled
                    if self.enable_local_cache:
                        cache_data = {
                            "data": df.to_dict(orient="records"),
                            "metadata": {
                                "symbol": symbol,
                                "start_date": start_date,
                                "end_date": end_date,
                                "cached_at": datetime.now().isoformat()
                            }
                        }
                        self._save_to_cache(
                            data=cache_data,
                            symbol=symbol,
                            data_type="iv_history",
                            start_date=start_date,
                            end_date=end_date
                        )
                    return df
            
            except Exception as e:
                error_msg = f"Error getting IV history from {source.name}: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
        
        # If we've tried all sources and still don't have data, use mock data
        logger.warning(f"Failed to get IV history for {symbol} from all sources")
        mock_df = self._generate_mock_iv_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add a column to indicate this is mock data
        mock_df['is_mock'] = True
        
        return mock_df
    
    def _get_ordered_data_sources(self) -> List[DataSource]:
        """
        Get data sources ordered by priority.
        
        Returns:
            List of active data sources ordered by priority
        """
        active_sources = [s for s in self.data_sources.values() if s.active]
        
        # Map enum values to numeric priorities for sorting
        priority_map = {
            DataSourcePriority.PRIMARY: 1,
            DataSourcePriority.SECONDARY: 2,
            DataSourcePriority.TERTIARY: 3,
            DataSourcePriority.FALLBACK: 4
        }
        
        # Sort by priority
        return sorted(active_sources, key=lambda s: priority_map[s.priority])
    
    def _fetch_options_chain_from_source(
        self,
        source: DataSource,
        symbol: str,
        expiration_date: Optional[str] = None,
        strikes: Optional[List[float]] = None,
        option_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch options chain data from a specific source.
        
        Args:
            source: Data source to use
            symbol: Ticker symbol
            expiration_date: Optional specific expiration date
            strikes: Optional list of specific strikes
            option_type: Optional option type filter
            
        Returns:
            Options chain data dictionary
        """
        # Handle mock data source
        if source.config.get("is_mock", False):
            return self._generate_mock_options_chain(
                symbol=symbol,
                expiration_date=expiration_date,
                strikes=strikes,
                option_type=option_type
            )
        
        # In a real implementation, we would have source-specific fetching logic here
        # For example:
        if source.name == "tradier":
            return self._fetch_from_tradier(
                source=source,
                symbol=symbol,
                expiration_date=expiration_date,
                strikes=strikes,
                option_type=option_type
            )
        elif source.name == "polygon":
            return self._fetch_from_polygon(
                source=source,
                symbol=symbol,
                expiration_date=expiration_date
            )
        elif source.name == "yahoo":
            return self._fetch_from_yahoo(
                symbol=symbol,
                expiration_date=expiration_date
            )
        
        # Default - unsupported source
        raise ValueError(f"Unsupported data source: {source.name}")
    
    def _fetch_iv_surface_from_source(
        self,
        source: DataSource,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Fetch IV surface data from a specific source.
        
        Args:
            source: Data source to use
            symbol: Ticker symbol
            
        Returns:
            IV surface data dictionary
        """
        # Handle mock data source
        if source.config.get("is_mock", False):
            return self._generate_mock_iv_surface(symbol=symbol)
        
        # In a real implementation, we would have source-specific fetching logic here
        # For example:
        if source.name == "tradier":
            # Get options chain for multiple expirations
            # Calculate IV surface from the options data
            pass
        elif source.name == "ivolatility":
            # Fetch IV surface data from IVolatility API
            pass
        
        # Default - unsupported source
        raise ValueError(f"Unsupported data source for IV surface: {source.name}")
    
    def _fetch_iv_history_from_source(
        self,
        source: DataSource,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical IV data from a specific source.
        
        Args:
            source: Data source to use
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical IV data
        """
        # Handle mock data source
        if source.config.get("is_mock", False):
            return self._generate_mock_iv_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
        
        # In a real implementation, we would have source-specific fetching logic here
        # For example:
        if source.name == "tradier":
            # Fetch historical IV data from Tradier
            pass
        elif source.name == "ivolatility":
            # Fetch historical IV data from IVolatility API
            pass
        
        # Default - unsupported source
        raise ValueError(f"Unsupported data source for IV history: {source.name}")
    
    def _fetch_from_tradier(
        self,
        source: DataSource,
        symbol: str,
        expiration_date: Optional[str] = None,
        strikes: Optional[List[float]] = None,
        option_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch options data from Tradier API.
        
        Args:
            source: Tradier data source
            symbol: Ticker symbol
            expiration_date: Optional specific expiration date
            strikes: Optional list of specific strikes
            option_type: Optional option type filter
            
        Returns:
            Options chain data dictionary
        """
        # This would be implemented with actual API calls to Tradier
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("Tradier API integration not implemented")
    
    def _fetch_from_polygon(
        self,
        source: DataSource,
        symbol: str,
        expiration_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch options data from Polygon API.
        
        Args:
            source: Polygon data source
            symbol: Ticker symbol
            expiration_date: Optional specific expiration date
            
        Returns:
            Options chain data dictionary
        """
        # This would be implemented with actual API calls to Polygon
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("Polygon API integration not implemented")
    
    def _fetch_from_yahoo(
        self,
        symbol: str,
        expiration_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch options data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol
            expiration_date: Optional specific expiration date
            
        Returns:
            Options chain data dictionary
        """
        # This would be implemented using a library like yfinance
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("Yahoo Finance integration not implemented")
    
    def _get_from_cache(
        self,
        symbol: str,
        data_type: str,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get data from local cache.
        
        Args:
            symbol: Ticker symbol
            data_type: Type of data to retrieve
            **kwargs: Additional parameters for cache file identification
            
        Returns:
            Cached data or None if not found/expired
        """
        if not self.enable_local_cache:
            return None
        
        # Generate cache file path
        cache_file = self._get_cache_file_path(symbol, data_type, **kwargs)
        
        try:
            # Check if file exists and is not expired
            if not os.path.exists(cache_file):
                return None
            
            # Check file modification time for expiration
            mod_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - mod_time > self.cache_ttl:
                logger.debug(f"Cache expired for {symbol} {data_type}")
                return None
            
            # Read and parse the cache file
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
            return None
    
    def _save_to_cache(
        self,
        data: Dict[str, Any],
        symbol: str,
        data_type: str,
        **kwargs
    ) -> bool:
        """
        Save data to local cache.
        
        Args:
            data: Data to cache
            symbol: Ticker symbol
            data_type: Type of data
            **kwargs: Additional parameters for cache file identification
            
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enable_local_cache:
            return False
        
        # Generate cache file path
        cache_file = self._get_cache_file_path(symbol, data_type, **kwargs)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Write data to cache file
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
            return False
    
    def _get_cache_file_path(
        self,
        symbol: str,
        data_type: str,
        **kwargs
    ) -> str:
        """
        Generate cache file path.
        
        Args:
            symbol: Ticker symbol
            data_type: Type of data
            **kwargs: Additional parameters for file name
            
        Returns:
            Cache file path
        """
        # Create a standardized file name based on parameters
        filename_parts = [symbol.upper(), data_type]
        
        # Add any additional parameters to the file name
        for key, value in kwargs.items():
            if value:
                # Clean up the value for filename use
                clean_value = str(value).replace('/', '-').replace(':', '-')
                filename_parts.append(f"{key}_{clean_value}")
        
        # Join parts with underscores and add .json extension
        filename = "_".join(filename_parts) + ".json"
        
        # Return full cache file path
        return os.path.join(self.cache_dir, filename)
    
    def _fill_iv_history_gaps(
        self,
        df: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fill gaps in historical IV data.
        
        Args:
            df: DataFrame with historical IV data
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with gaps filled
        """
        if df.empty:
            return df
        
        try:
            # Ensure the date column is properly formatted
            if 'date' not in df.columns:
                raise ValueError("DataFrame must have a 'date' column")
            
            # Convert dates to datetime if they're not already
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Create a complete date range
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            full_date_range = pd.date_range(start=start, end=end)
            
            # Create a new DataFrame with the complete date range
            full_df = pd.DataFrame({'date': full_date_range})
            
            # Merge with the original data
            merged_df = pd.merge(full_df, df, on='date', how='left')
            
            # For any missing IV values, interpolate
            if 'iv' in merged_df.columns:
                merged_df['iv'] = merged_df['iv'].interpolate(method='linear')
            
            # Convert date back to string format if it was originally
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                merged_df['date'] = merged_df['date'].dt.strftime('%Y-%m-%d')
            
            return merged_df
            
        except Exception as e:
            logger.warning(f"Error filling IV history gaps: {str(e)}")
            return df
    
    def _generate_mock_options_chain(
        self,
        symbol: str,
        expiration_date: Optional[str] = None,
        strikes: Optional[List[float]] = None,
        option_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate mock options chain data for testing.
        
        Args:
            symbol: Ticker symbol
            expiration_date: Optional specific expiration date
            strikes: Optional list of specific strikes
            option_type: Optional option type filter
            
        Returns:
            Mock options chain data
        """
        current_date = datetime.now()
        
        # Generate expiration dates if not provided
        if not expiration_date:
            expirations = [
                (current_date + timedelta(days=7*i)).strftime('%Y-%m-%d')
                for i in range(1, 9)  # 8 weeks out
            ]
        else:
            expirations = [expiration_date]
        
        # Mock stock price (random value between 10 and 1000)
        stock_price = np.random.uniform(10, 1000)
        
        # Generate strikes around the stock price if not provided
        if not strikes:
            strike_percentage_range = np.arange(-0.30, 0.31, 0.05)
            strikes = [round(stock_price * (1 + pct), 2) for pct in strike_percentage_range]
        
        # Determine which option types to include
        include_calls = option_type is None or option_type.lower() == 'call'
        include_puts = option_type is None or option_type.lower() == 'put'
        
        # Create options data structure
        options_data = {
            "options": {
                "summary": {
                    "symbol": symbol,
                    "underlying_price": stock_price,
                    "updated_at": current_date.isoformat(),
                    "is_mock": True
                },
                "expirations": {
                    "date": expirations
                },
                "strikes": strikes,
                "option": []
            }
        }
        
        # Generate option contracts
        for exp_date in expirations:
            days_to_expiry = (datetime.strptime(exp_date, '%Y-%m-%d') - current_date).days
            
            for strike in strikes:
                # Calculate mock IV (higher for further OTM options)
                moneyness = abs(stock_price - strike) / stock_price
                iv_base = 0.20 + (moneyness * 0.50)  # Base IV increases with distance from ATM
                iv_term = 1.0 - (0.5 * days_to_expiry / 365)  # IV term structure factor
                iv = iv_base * iv_term
                
                # Add call option
                if include_calls:
                    call_option = self._generate_mock_option_contract(
                        symbol=symbol,
                        strike=strike,
                        expiration=exp_date,
                        option_type="call",
                        stock_price=stock_price,
                        iv=iv,
                        days_to_expiry=days_to_expiry
                    )
                    options_data["options"]["option"].append(call_option)
                
                # Add put option
                if include_puts:
                    put_option = self._generate_mock_option_contract(
                        symbol=symbol,
                        strike=strike,
                        expiration=exp_date,
                        option_type="put",
                        stock_price=stock_price,
                        iv=iv,
                        days_to_expiry=days_to_expiry
                    )
                    options_data["options"]["option"].append(put_option)
        
        return options_data
    
    def _generate_mock_option_contract(
        self,
        symbol: str,
        strike: float,
        expiration: str,
        option_type: str,
        stock_price: float,
        iv: float,
        days_to_expiry: int
    ) -> Dict[str, Any]:
        """
        Generate a mock option contract.
        
        Args:
            symbol: Underlying symbol
            strike: Strike price
            expiration: Expiration date
            option_type: Option type ('call' or 'put')
            stock_price: Current stock price
            iv: Implied volatility
            days_to_expiry: Days to expiration
            
        Returns:
            Mock option contract data
        """
        # Calculate option price and greeks
        is_call = option_type.lower() == 'call'
        time_to_expiry = days_to_expiry / 365.0
        
        # Simple Black-Scholes approximation
        d1 = (np.log(stock_price / strike) + (0.02 + 0.5 * iv**2) * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
        d2 = d1 - iv * np.sqrt(time_to_expiry)
        
        if is_call:
            # Call option
            delta = self._norm_cdf(d1)
            option_price = stock_price * delta - strike * np.exp(-0.02 * time_to_expiry) * self._norm_cdf(d2)
        else:
            # Put option
            delta = self._norm_cdf(d1) - 1
            option_price = strike * np.exp(-0.02 * time_to_expiry) * self._norm_cdf(-d2) - stock_price * self._norm_cdf(-d1)
        
        # Ensure the price is positive
        option_price = max(0.01, option_price)
        
        # Calculate other greeks
        gamma = self._norm_pdf(d1) / (stock_price * iv * np.sqrt(time_to_expiry))
        vega = stock_price * self._norm_pdf(d1) * np.sqrt(time_to_expiry) / 100  # Scaled for percentage point
        theta = -(stock_price * iv * self._norm_pdf(d1)) / (2 * np.sqrt(time_to_expiry) * 365)
        if not is_call:
            theta -= 0.02 * strike * np.exp(-0.02 * time_to_expiry) * self._norm_cdf(-d2) / 365
        else:
            theta -= 0.02 * strike * np.exp(-0.02 * time_to_expiry) * self._norm_cdf(d2) / 365
        
        # Create option symbol
        option_symbol = f"{symbol}{expiration.replace('-', '')}{'C' if is_call else 'P'}{int(strike * 1000):08d}"
        
        # Create the contract dictionary
        contract = {
            "symbol": option_symbol,
            "underlying": symbol,
            "strike": round(strike, 2),
            "expiration": expiration,
            "type": option_type.lower(),
            "last": round(option_price, 2),
            "bid": round(max(0.01, option_price - 0.05), 2),
            "ask": round(option_price + 0.05, 2),
            "volume": int(np.random.randint(1, 1000)),
            "open_interest": int(np.random.randint(10, 5000)),
            "greeks": {
                "delta": round(delta, 4),
                "gamma": round(gamma, 4),
                "theta": round(theta, 4),
                "vega": round(vega, 4),
                "rho": round(np.random.uniform(-0.1, 0.1), 4),
                "iv": round(iv * 100, 2)  # IV as percentage
            }
        }
        
        return contract
    
    def _generate_mock_iv_surface(self, symbol: str) -> Dict[str, Any]:
        """
        Generate mock IV surface data for testing.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Mock IV surface data
        """
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Generate mock dates (expiration tenors)
        tenors = [30, 60, 90, 180, 270, 365]  # Days
        expirations = [
            (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
            for days in tenors
        ]
        
        # Generate moneyness levels
        moneyness_levels = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        
        # Mock ATM volatility and skew parameters
        atm_vol = np.random.uniform(0.15, 0.35)
        skew_steepness = np.random.uniform(0.5, 2.0)
        vol_term_structure_slope = np.random.uniform(-0.05, 0.1)
        
        # Create the surface data structure
        surface_data = {
            "iv_surface": {
                "symbol": symbol,
                "generated": current_date,
                "is_mock": True,
                "expirations": expirations,
                "moneyness": moneyness_levels,
                "data": []
            }
        }
        
        # Generate IV values for each expiration and moneyness level
        for i, expiration in enumerate(expirations):
            for moneyness in moneyness_levels:
                # Calculate days to expiry
                days_to_expiry = tenors[i]
                
                # Calculate term structure effect
                term_effect = 1.0 + vol_term_structure_slope * (days_to_expiry / 365)
                
                # Calculate vol based on moneyness (skew)
                if moneyness < 0:
                    # OTM puts typically have higher IV (volatility skew)
                    vol = atm_vol * (1 + skew_steepness * abs(moneyness)) * term_effect
                else:
                    # OTM calls typically have lower or flat IV
                    vol = atm_vol * (1 + 0.3 * moneyness) * term_effect
                
                # Add some noise
                vol += np.random.normal(0, 0.01)
                
                # Ensure vol is positive
                vol = max(0.05, vol)
                
                # Add to surface data
                point = {
                    "expiration": expiration,
                    "days_to_expiry": days_to_expiry,
                    "moneyness": moneyness,
                    "iv": round(vol * 100, 2)  # IV as percentage
                }
                
                surface_data["iv_surface"]["data"].append(point)
        
        return surface_data
    
    def _generate_mock_iv_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Generate mock historical IV data for testing.
        
        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with mock historical IV data
        """
        # Convert date strings to datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Create date range
        date_range = pd.date_range(start=start, end=end)
        
        # Generate mock IV values with some trend and randomness
        base_iv = np.random.uniform(15, 35)  # Base IV level (percentage)
        
        # Create trend (random walk with drift)
        n_days = len(date_range)
        drift = np.random.uniform(-0.1, 0.1)
        random_walk = np.cumsum(np.random.normal(drift, 0.5, n_days))
        
        # Scale the random walk
        scaled_walk = random_walk * 2.0
        
        # Add seasonality (higher in expiration weeks)
        week_of_month = np.array([(d.day - 1) // 7 for d in date_range])
        expiration_effect = np.where(week_of_month == 2, 2.0, 0)  # Third week effect
        
        # Combine base, trend, and seasonality
        iv_values = base_iv + scaled_walk + expiration_effect
        
        # Ensure all values are positive
        iv_values = np.maximum(5, iv_values)
        
        # Calculate IV percentile (relative to this historical period)
        iv_min = np.min(iv_values)
        iv_max = np.max(iv_values)
        iv_range = iv_max - iv_min
        if iv_range > 0:
            iv_percentiles = 100 * (iv_values - iv_min) / iv_range
        else:
            iv_percentiles = np.ones_like(iv_values) * 50
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': date_range.strftime('%Y-%m-%d'),
            'iv': iv_values,
            'iv_percentile': iv_percentiles
        })
        
        return df
    
    def _norm_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))
    
    def _norm_pdf(self, x: float) -> float:
        """Standard normal probability density function."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi) 