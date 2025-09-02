#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataPipeline - Integrates data sources, processors, and feature extractors.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from trading_bot.data.sources.base_source import DataSource
from trading_bot.data.processors.base_processor import DataProcessor
from trading_bot.data.features.base_feature import FeatureExtractor

logger = logging.getLogger("DataPipeline")

class DataPipeline:
    """
    DataPipeline integrates data sources, processors, and feature extractors
    to create a complete data flow from raw data to processed features.
    """
    
    def __init__(
        self, 
        name: str,
        data_source: Optional[DataSource] = None,
        data_processor: Optional[DataProcessor] = None,
        feature_extractors: Optional[List[FeatureExtractor]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data pipeline.
        
        Args:
            name: Name of the pipeline
            data_source: Data source component
            data_processor: Data processor component
            feature_extractors: List of feature extractor components
            config: Pipeline configuration
        """
        self.name = name
        self.data_source = data_source
        self.data_processor = data_processor
        self.feature_extractors = feature_extractors or []
        self.config = config or {}
        
        self.cache = {
            "raw_data": None,
            "processed_data": None,
            "features": None,
            "last_update": None
        }
        
        # Set up any additional attributes from config
        self.init_from_config()
        
        logger.info(f"Initialized {self.name} data pipeline")
    
    def init_from_config(self) -> None:
        """Initialize additional attributes from configuration."""
        # Extract cache configuration
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 600)  # 10 minutes by default
    
    def set_data_source(self, data_source: DataSource) -> None:
        """
        Set the data source for the pipeline.
        
        Args:
            data_source: DataSource instance
        """
        self.data_source = data_source
        logger.info(f"Set data source: {data_source.name}")
        
        # Clear cache since data source changed
        self._clear_cache()
    
    def set_data_processor(self, data_processor: DataProcessor) -> None:
        """
        Set the data processor for the pipeline.
        
        Args:
            data_processor: DataProcessor instance
        """
        self.data_processor = data_processor
        logger.info(f"Set data processor: {data_processor.name}")
        
        # Clear processed data and features cache
        self.cache["processed_data"] = None
        self.cache["features"] = None
    
    def add_feature_extractor(self, feature_extractor: FeatureExtractor) -> None:
        """
        Add a feature extractor to the pipeline.
        
        Args:
            feature_extractor: FeatureExtractor instance
        """
        self.feature_extractors.append(feature_extractor)
        logger.info(f"Added feature extractor: {feature_extractor.name}")
        
        # Clear features cache
        self.cache["features"] = None
    
    def remove_feature_extractor(self, extractor_name: str) -> bool:
        """
        Remove a feature extractor from the pipeline.
        
        Args:
            extractor_name: Name of the feature extractor to remove
            
        Returns:
            bool: True if extractor was removed, False otherwise
        """
        for i, extractor in enumerate(self.feature_extractors):
            if extractor.name == extractor_name:
                del self.feature_extractors[i]
                logger.info(f"Removed feature extractor: {extractor_name}")
                
                # Clear features cache
                self.cache["features"] = None
                
                return True
                
        logger.warning(f"Feature extractor '{extractor_name}' not found")
        return False
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """
        Check if cache for a specific key is valid.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            bool: True if cache is valid, False otherwise
        """
        if not self.enable_caching:
            return False
            
        if (self.cache[cache_key] is None or 
            self.cache["last_update"] is None):
            return False
            
        # Check if cache has expired
        elapsed = (datetime.now() - self.cache["last_update"]).total_seconds()
        return elapsed < self.cache_ttl
    
    def _clear_cache(self) -> None:
        """Clear all cache entries."""
        self.cache = {
            "raw_data": None,
            "processed_data": None,
            "features": None,
            "last_update": None
        }
        logger.debug("Cleared pipeline cache")
    
    def get_raw_data(
        self, 
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: Optional[str] = None,
        force_refresh: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get raw data from the data source.
        
        Args:
            symbol: Asset symbol
            start_date: Start date (required if timeframe not provided)
            end_date: End date
            timeframe: Timeframe string (e.g., "1d", "1w", "1m", "1y")
            force_refresh: Whether to force refresh the cache
            **kwargs: Additional arguments for the data source
            
        Returns:
            DataFrame with raw data
        """
        if self.data_source is None:
            logger.error("No data source configured for pipeline")
            return pd.DataFrame()
            
        # Check if cache is valid and not forced to refresh
        if not force_refresh and self._is_cache_valid("raw_data"):
            logger.debug("Using cached raw data")
            return self.cache["raw_data"]
            
        # Connect to data source if not connected
        if not self.data_source.is_connected:
            success = self.data_source.connect()
            if not success:
                logger.error(f"Failed to connect to data source: {self.data_source.name}")
                return pd.DataFrame()
        
        try:
            # Get data based on provided parameters
            if timeframe is not None:
                # Get data for timeframe
                market_data_list = self.data_source.get_data_for_timeframe(
                    symbol=symbol,
                    timeframe=timeframe,
                    end_date=end_date,
                    **kwargs
                )
            elif start_date is not None:
                # Get historical data
                market_data_list = self.data_source.get_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
            else:
                logger.error("Either timeframe or start_date must be provided")
                return pd.DataFrame()
                
            # Convert to DataFrame
            if self.data_processor:
                df = self.data_processor.convert_to_dataframe(market_data_list)
            else:
                # Basic conversion if no processor
                data_dict = {
                    "symbol": [],
                    "timestamp": [],
                    "price": [],
                    "volume": []
                }
                for md in market_data_list:
                    data_dict["symbol"].append(md.symbol)
                    data_dict["timestamp"].append(md.timestamp)
                    data_dict["price"].append(md.price)
                    data_dict["volume"].append(md.volume)
                
                df = pd.DataFrame(data_dict)
                
                # Set timestamp as index
                if not df.empty:
                    df.set_index("timestamp", inplace=True)
            
            # Update cache
            self.cache["raw_data"] = df
            self.cache["last_update"] = datetime.now()
            
            logger.info(f"Retrieved {len(df)} rows of raw data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting raw data: {str(e)}")
            return pd.DataFrame()
    
    def get_processed_data(
        self, 
        symbol: str = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: Optional[str] = None,
        force_refresh: bool = False,
        raw_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get processed data.
        
        Args:
            symbol: Asset symbol (not required if raw_data provided)
            start_date: Start date (required if timeframe not provided)
            end_date: End date
            timeframe: Timeframe string (e.g., "1d", "1w", "1m", "1y")
            force_refresh: Whether to force refresh the cache
            raw_data: Raw data DataFrame (if already available)
            **kwargs: Additional arguments for the data source
            
        Returns:
            DataFrame with processed data
        """
        if self.data_processor is None:
            logger.error("No data processor configured for pipeline")
            return pd.DataFrame()
            
        # Check if cache is valid and not forced to refresh
        if not force_refresh and self._is_cache_valid("processed_data"):
            logger.debug("Using cached processed data")
            return self.cache["processed_data"]
        
        # Get raw data if not provided
        df = raw_data
        if df is None or df.empty:
            df = self.get_raw_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                force_refresh=force_refresh,
                **kwargs
            )
            
        if df.empty:
            logger.warning("No raw data available for processing")
            return pd.DataFrame()
        
        try:
            # Process the data
            processed_df = self.data_processor.process(df)
            
            # Update cache
            self.cache["processed_data"] = processed_df
            if self.cache["last_update"] is None:
                self.cache["last_update"] = datetime.now()
            
            logger.info(f"Processed {len(processed_df)} rows of data")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return pd.DataFrame()
    
    def get_features(
        self, 
        symbol: str = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: Optional[str] = None,
        force_refresh: bool = False,
        processed_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get features extracted from processed data.
        
        Args:
            symbol: Asset symbol (not required if processed_data provided)
            start_date: Start date (required if timeframe not provided)
            end_date: End date
            timeframe: Timeframe string (e.g., "1d", "1w", "1m", "1y")
            force_refresh: Whether to force refresh the cache
            processed_data: Processed data DataFrame (if already available)
            **kwargs: Additional arguments for the data source
            
        Returns:
            DataFrame with extracted features
        """
        if not self.feature_extractors:
            logger.error("No feature extractors configured for pipeline")
            return pd.DataFrame()
            
        # Check if cache is valid and not forced to refresh
        if not force_refresh and self._is_cache_valid("features"):
            logger.debug("Using cached features")
            return self.cache["features"]
        
        # Get processed data if not provided
        df = processed_data
        if df is None or df.empty:
            df = self.get_processed_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                force_refresh=force_refresh,
                **kwargs
            )
            
        if df.empty:
            logger.warning("No processed data available for feature extraction")
            return pd.DataFrame()
        
        try:
            # Apply each feature extractor in sequence
            features_df = df.copy()
            for extractor in self.feature_extractors:
                features_df = extractor.extract_features(features_df)
                logger.debug(f"Applied feature extractor: {extractor.name}")
            
            # Update cache
            self.cache["features"] = features_df
            if self.cache["last_update"] is None:
                self.cache["last_update"] = datetime.now()
            
            logger.info(f"Extracted features for {len(features_df)} rows of data")
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return df  # Return processed data if feature extraction fails
    
    def __str__(self) -> str:
        """String representation of the data pipeline."""
        source_name = self.data_source.name if self.data_source else "None"
        processor_name = self.data_processor.name if self.data_processor else "None"
        extractor_names = ", ".join([e.name for e in self.feature_extractors]) if self.feature_extractors else "None"
        
        return (f"{self.name} DataPipeline (Source: {source_name}, "
                f"Processor: {processor_name}, Extractors: {extractor_names})") 