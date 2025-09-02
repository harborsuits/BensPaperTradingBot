#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Pipeline - Unified entry point for data processing with clean separation of
data cleaning and quality assurance components.

This module provides a configurable pipeline that integrates:
1. Data normalization and cleaning (via DataCleaningProcessor)
2. Data quality assurance (via DataQualityProcessor)
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from trading_bot.core.event_system import EventBus
from trading_bot.data.processors.data_cleaning_processor_refactored import DataCleaningProcessor
from trading_bot.data.quality.integration import DataQualityProcessor, setup_data_quality_system
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger("DataPipeline")

class DataPipeline:
    """
    Unified data processing pipeline with separation of concerns.
    
    This pipeline:
    1. Applies data cleaning operations (normalization, standardization)
    2. Performs quality assurance (validation, repair, monitoring)
    3. Integrates with the event system for monitoring and alerting
    4. Records quality metrics to the persistence layer
    """
    
    def __init__(self,
                 name: str = "DataPipeline",
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None,
                 persistence: Optional[PersistenceManager] = None):
        """
        Initialize the DataPipeline.
        
        Args:
            name: Pipeline name
            config: Configuration dictionary
            event_bus: EventBus for publishing events
            persistence: PersistenceManager for storing metrics
        """
        self.name = name
        self.config = config or {}
        self.event_bus = event_bus
        self.persistence = persistence
        
        # Create the pipeline components with clear separation of responsibilities
        self.cleaning_processor = DataCleaningProcessor(
            name="DataCleaningProcessor",
            config=self._extract_config_section('cleaning')
        )
        
        # Quality processor is responsible for validation and repair
        self.quality_processor = setup_data_quality_system(
            event_bus=event_bus,
            persistence=persistence,
            config=self._extract_config_section('quality')
        )
        
        # Configure pipeline behavior
        self.skip_cleaning = self.config.get('skip_cleaning', False)
        self.skip_quality = self.config.get('skip_quality', False)
        self.raise_on_critical = self.config.get('raise_on_critical_issues', False)
        
        logger.info(f"Data pipeline initialized ({name})")
        logger.info(f"Pipeline configuration: cleaning={not self.skip_cleaning}, " +
                    f"quality={not self.skip_quality}, raise_on_critical={self.raise_on_critical}")
    
    def _extract_config_section(self, section: str) -> Dict[str, Any]:
        """Extract a section from the config dictionary."""
        if not self.config:
            return {}
        
        return self.config.get(section, {})
    
    def process(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
               symbol: Optional[str] = None,
               source: Optional[str] = None) -> Tuple[Union[pd.DataFrame, Dict[str, pd.DataFrame]], Dict[str, Any]]:
        """
        Process data through the pipeline.
        
        Args:
            data: Input DataFrame or dictionary of DataFrames
            symbol: Optional symbol identifier (required if data is a DataFrame)
            source: Optional source identifier (required if data is a DataFrame)
            
        Returns:
            Tuple of (processed_data, processing_metadata)
        """
        start_time = datetime.now()
        processing_metadata = {
            "pipeline": self.name,
            "started_at": start_time,
            "symbol": symbol,
            "source": source,
            "steps_executed": [],
            "issues_detected": 0,
            "issues_fixed": 0,
            "quality_score": 100.0,
        }
        
        try:
            # Process a single DataFrame
            if isinstance(data, pd.DataFrame):
                result = self._process_single_dataframe(data, symbol, source, processing_metadata)
                
            # Process multiple DataFrames
            else:
                result = self._process_multiple_dataframes(data, processing_metadata)
            
            # Complete metadata
            processing_metadata["completed_at"] = datetime.now()
            processing_metadata["processing_time_ms"] = (
                processing_metadata["completed_at"] - start_time
            ).total_seconds() * 1000
            
            return result, processing_metadata
            
        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            processing_metadata["error"] = str(e)
            processing_metadata["completed_at"] = datetime.now()
            processing_metadata["processing_time_ms"] = (
                processing_metadata["completed_at"] - start_time
            ).total_seconds() * 1000
            
            # Re-raise if configured to do so
            if self.raise_on_critical:
                raise
            
            # Return original data if processing failed
            return data, processing_metadata
    
    def _process_single_dataframe(self, df: pd.DataFrame, 
                                 symbol: str, 
                                 source: str,
                                 metadata: Dict[str, Any]) -> pd.DataFrame:
        """Process a single DataFrame through the pipeline."""
        result = df.copy()
        
        # 1. Data Cleaning (normalization and standardization)
        if not self.skip_cleaning:
            try:
                cleaning_start = datetime.now()
                result = self.cleaning_processor.process(result)
                
                metadata["steps_executed"].append("cleaning")
                metadata["cleaning_time_ms"] = (
                    datetime.now() - cleaning_start
                ).total_seconds() * 1000
                
                # Get cleaning stats
                metadata["cleaning_stats"] = self.cleaning_processor.get_cleaning_stats()
                
                logger.debug(f"Cleaning complete for {symbol} from {source}")
            except Exception as e:
                logger.error(f"Error during cleaning for {symbol}: {str(e)}")
                metadata["cleaning_error"] = str(e)
        
        # 2. Data Quality (validation and repair)
        if not self.skip_quality:
            try:
                quality_start = datetime.now()
                result = self.quality_processor.process(result, symbol, source)
                
                metadata["steps_executed"].append("quality")
                metadata["quality_time_ms"] = (
                    datetime.now() - quality_start
                ).total_seconds() * 1000
                
                # Get quality stats
                quality_summary = self.quality_processor.get_quality_summary()
                metadata["quality_stats"] = quality_summary
                
                # Update metadata with quality information
                metadata["issues_detected"] = quality_summary.get("total_issues_detected", 0)
                metadata["issues_fixed"] = quality_summary.get("total_issues_fixed", 0)
                metadata["quality_score"] = quality_summary.get("average_quality_score", 0)
                
                logger.debug(f"Quality checks complete for {symbol} from {source}")
            except Exception as e:
                logger.error(f"Error during quality checks for {symbol}: {str(e)}")
                metadata["quality_error"] = str(e)
        
        # Check for critical quality issues
        if metadata.get("quality_score", 100) < 50 and self.raise_on_critical:
            logger.critical(f"Critical quality issues detected for {symbol} from {source}")
            raise ValueError(f"Data quality below acceptable threshold: {metadata.get('quality_score', 0)}")
        
        return result
    
    def _process_multiple_dataframes(self, data_dict: Dict[str, pd.DataFrame],
                                     metadata: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Process multiple DataFrames through the pipeline."""
        results = {}
        metadata["symbols"] = list(data_dict.keys())
        metadata["per_symbol"] = {}
        
        for symbol, df in data_dict.items():
            # Create per-symbol metadata
            symbol_metadata = {
                "rows": len(df),
                "started_at": datetime.now(),
            }
            
            # Process this dataframe
            source = metadata.get("source", "unknown")  # Use global source if available
            results[symbol] = self._process_single_dataframe(
                df, symbol, source, symbol_metadata
            )
            
            # Record completion time
            symbol_metadata["completed_at"] = datetime.now()
            symbol_metadata["processing_time_ms"] = (
                symbol_metadata["completed_at"] - symbol_metadata["started_at"]
            ).total_seconds() * 1000
            
            # Store in global metadata
            metadata["per_symbol"][symbol] = symbol_metadata
        
        # Aggregate metrics across symbols
        if metadata["per_symbol"]:
            metadata["issues_detected"] = sum(
                m.get("issues_detected", 0) for m in metadata["per_symbol"].values()
            )
            metadata["issues_fixed"] = sum(
                m.get("issues_fixed", 0) for m in metadata["per_symbol"].values()
            )
            
            quality_scores = [
                m.get("quality_score", 0) for m in metadata["per_symbol"].values()
                if "quality_score" in m
            ]
            metadata["quality_score"] = (
                sum(quality_scores) / max(1, len(quality_scores))
                if quality_scores else 100.0
            )
        
        return results
    
    def standardize_timestamps(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Standardize timestamps in a DataFrame to ensure consistent formatting and timezone.
        
        This method performs the following operations:
        1. Converts string timestamps to datetime objects if needed
        2. Ensures UTC timezone consistency
        3. Sorts data by timestamp
        4. Handles duplicate timestamps
        5. Fixes timezone-naive timestamps
        
        Args:
            df: Input DataFrame with timestamp column
            timestamp_column: Name of the timestamp column (default: 'timestamp')
            
        Returns:
            DataFrame with standardized timestamps
        """
        if timestamp_column not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_column}' not found in DataFrame")
            return df
            
        result = df.copy()
        
        # 1. Convert string timestamps to datetime objects if needed
        if pd.api.types.is_string_dtype(result[timestamp_column]):
            result[timestamp_column] = pd.to_datetime(result[timestamp_column], errors='coerce')
            
        # 2. Ensure timezone consistency (convert to UTC)
        if pd.api.types.is_datetime64_dtype(result[timestamp_column]):
            # Check if timestamps are timezone-naive
            if result[timestamp_column].dt.tz is None:
                # Assume UTC for timezone-naive timestamps
                result[timestamp_column] = result[timestamp_column].dt.tz_localize('UTC')
            else:
                # Convert to UTC if not already
                result[timestamp_column] = result[timestamp_column].dt.tz_convert('UTC')
                
        # 3. Sort by timestamp
        result = result.sort_values(by=timestamp_column)
        
        # 4. Handle duplicate timestamps if needed
        if result.duplicated(subset=[timestamp_column]).any():
            # Log the issue
            duplicate_count = result.duplicated(subset=[timestamp_column]).sum()
            logger.warning(f"Found {duplicate_count} duplicate timestamps")
            
            # Strategy: Keep first occurrence of each timestamp
            result = result.drop_duplicates(subset=[timestamp_column], keep='first')
            
        # 5. Check for and fix NaT values
        nat_count = result[timestamp_column].isna().sum()
        if nat_count > 0:
            logger.warning(f"Found {nat_count} NaT/null timestamps, dropping rows")
            result = result.dropna(subset=[timestamp_column])
        
        return result
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and metrics."""
        stats = {
            "name": self.name,
            "cleaning_enabled": not self.skip_cleaning,
            "quality_enabled": not self.skip_quality,
            "cleaning_stats": (
                self.cleaning_processor.get_cleaning_stats() 
                if not self.skip_cleaning else {}
            ),
            "quality_stats": (
                self.quality_processor.get_quality_summary()
                if not self.skip_quality else {}
            )
        }
        
        # Calculate aggregate quality score
        if not self.skip_quality:
            stats["aggregate_quality_score"] = stats["quality_stats"].get("average_quality_score", 0)
        
        return stats
    
    def __str__(self) -> str:
        """String representation of the DataPipeline."""
        components = []
        if not self.skip_cleaning:
            components.append("Cleaning")
        if not self.skip_quality:
            components.append("Quality")
        
        return f"DataPipeline({' -> '.join(components)})"


def create_data_pipeline(config: Optional[Dict[str, Any]] = None,
                         event_bus: Optional[EventBus] = None,
                         persistence: Optional[PersistenceManager] = None) -> DataPipeline:
    """
    Create a configured data pipeline.
    
    Args:
        config: Configuration dictionary
        event_bus: EventBus for events
        persistence: PersistenceManager for metrics
        
    Returns:
        Configured DataPipeline
    """
    pipeline = DataPipeline(
        config=config,
        event_bus=event_bus,
        persistence=persistence
    )
    
    logger.info(f"Created data pipeline: {str(pipeline)}")
    
    return pipeline
