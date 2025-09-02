#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Integration - Connect the data quality framework with the trading platform's 
event system and data pipeline.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta

from trading_bot.data.quality.data_quality_manager import DataQualityManager
from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.data.processors.base_processor import DataProcessor
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger("DataQualityIntegration")

class DataQualityProcessor(DataProcessor):
    """
    Data processor that integrates the DataQualityManager into the data processing pipeline.
    
    This processor:
    1. Performs data quality checks on incoming data
    2. Applies necessary cleaning and fixes
    3. Publishes quality events to the event bus
    4. Records quality metrics to the persistence layer
    """
    
    def __init__(self, 
                 name: str = "DataQualityProcessor",
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None,
                 persistence: Optional[PersistenceManager] = None):
        """
        Initialize DataQualityProcessor.
        
        Args:
            name: Processor name
            config: Configuration dictionary
            event_bus: EventBus for publishing quality events
            persistence: PersistenceManager for storing quality metrics
        """
        super().__init__(name, config)
        
        # Create quality manager
        self.quality_manager = DataQualityManager(config=config, event_bus=event_bus)
        
        # Store references to event system and persistence
        self.event_bus = event_bus
        self.persistence = persistence
        
        # Set default auto_repair from config
        self.auto_repair = self.config.get('auto_repair', True)
        
        # Tracking metrics
        self.total_processed = 0
        self.total_issues_detected = 0
        self.total_issues_fixed = 0
        
        # Last reports by symbol/source
        self.last_reports = {}
        
        logger.info(f"DataQualityProcessor initialized (auto_repair={self.auto_repair})")
    
    def process(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
               symbol: Optional[str] = None, 
               source: Optional[str] = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Process the input data through quality checks.
        
        Args:
            data: Input DataFrame or dictionary of DataFrames
            symbol: Optional symbol identifier (required if data is a DataFrame)
            source: Optional source identifier (required if data is a DataFrame)
            
        Returns:
            Cleaned/verified data
        """
        if isinstance(data, pd.DataFrame):
            # Single DataFrame processing
            if not symbol or not source:
                raise ValueError("Symbol and source must be provided when processing a single DataFrame")
                
            return self._process_single_dataframe(data, symbol, source)
        else:
            # Multi-DataFrame processing
            return self._process_multiple_dataframes(data)
    
    def _process_single_dataframe(self, 
                                 df: pd.DataFrame, 
                                 symbol: str, 
                                 source: str) -> pd.DataFrame:
        """Process a single DataFrame through quality checks."""
        try:
            # Run quality checks
            start_time = datetime.now()
            cleaned_df, quality_report = self.quality_manager.check_data_quality(
                df, symbol, source, auto_repair=self.auto_repair
            )
            
            # Update metrics
            self.total_processed += 1
            self.total_issues_detected += len(quality_report.get("issues", []))
            self.total_issues_fixed += len(quality_report.get("fixed_issues", []))
            
            # Store the report
            self.last_reports[f"{symbol}_{source}"] = quality_report
            
            # Store quality report in persistence if available
            if self.persistence:
                try:
                    # Store quality metrics
                    self._store_quality_metrics(symbol, source, quality_report)
                except Exception as e:
                    logger.error(f"Failed to store quality metrics: {str(e)}")
            
            # Publish event if critical issues found
            if quality_report.get("status") == "critical" and self.event_bus:
                self._publish_critical_quality_event(symbol, source, quality_report)
            
            process_time = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Quality processing for {symbol} from {source} completed in {process_time:.3f}s")
            
            return cleaned_df
        except Exception as e:
            logger.error(f"Error in quality processing for {symbol} from {source}: {str(e)}")
            # Return original data on error
            return df
    
    def _process_multiple_dataframes(self, 
                                    data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process multiple DataFrames through quality checks."""
        result = {}
        
        for key, df in data_dict.items():
            # Extract symbol and source from key if possible
            parts = key.split('_', 1)
            if len(parts) == 2:
                symbol, source = parts
            else:
                # Use key as both symbol and source
                symbol = key
                source = "unknown"
            
            result[key] = self._process_single_dataframe(df, symbol, source)
            
        return result
    
    def _store_quality_metrics(self, symbol: str, source: str, quality_report: Dict[str, Any]) -> None:
        """Store quality metrics in the persistence layer."""
        if not self.persistence:
            return
            
        # Create a document with important metrics
        doc = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "source": source,
            "quality_score": quality_report.get("quality_score", 0),
            "row_count": quality_report.get("row_count", 0),
            "issue_count": len(quality_report.get("issues", [])),
            "warning_count": len(quality_report.get("warnings", [])),
            "fixed_count": len(quality_report.get("fixed_issues", [])),
            "status": quality_report.get("status", "unknown")
        }
        
        # Add detailed issue types if present
        issue_types = {}
        for issue in quality_report.get("issues", []):
            issue_type = issue.get("type", "unknown")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
            
        if issue_types:
            doc["issue_types"] = issue_types
            
        # Store in database
        self.persistence.insert_document("data_quality_metrics", doc)
        
        # If there are critical issues, also store detailed report
        if quality_report.get("status") == "critical":
            # Create an ID that prevents duplicates
            report_id = f"{symbol}_{source}_{datetime.now().strftime('%Y%m%d')}"
            
            detailed_doc = {
                "_id": report_id,
                "timestamp": datetime.now(),
                "symbol": symbol,
                "source": source,
                "quality_score": quality_report.get("quality_score", 0),
                "status": quality_report.get("status", "unknown"),
                "issues": quality_report.get("issues", []),
                "fixes": quality_report.get("fixed_issues", [])
            }
            
            # Store detailed report
            self.persistence.insert_document("data_quality_issues", detailed_doc, upsert=True)
    
    def _publish_critical_quality_event(self, symbol: str, source: str, quality_report: Dict[str, Any]) -> None:
        """Publish a critical quality event to the event bus."""
        if not self.event_bus:
            return
            
        # Create event data
        event_data = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "source": source,
            "quality_score": quality_report.get("quality_score", 0),
            "status": quality_report.get("status", "unknown"),
            "issue_count": len(quality_report.get("issues", [])),
            "issue_summary": [issue.get("message", "") for issue in quality_report.get("issues", [])[:3]]
        }
        
        # Create and publish event
        event = Event(
            event_type=EventType.DATA_QUALITY_CRITICAL,
            data=event_data
        )
        
        self.event_bus.publish(event)
        logger.info(f"Published critical quality event for {symbol} from {source}")
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get a summary of data quality metrics."""
        return {
            "total_processed": self.total_processed,
            "total_issues_detected": self.total_issues_detected,
            "total_issues_fixed": self.total_issues_fixed,
            "fix_rate": self.total_issues_fixed / max(1, self.total_issues_detected),
            "quality_scores": {
                key: report.get("quality_score", 0) 
                for key, report in self.last_reports.items()
            },
            "average_quality_score": sum(report.get("quality_score", 0) for report in self.last_reports.values()) / 
                                   max(1, len(self.last_reports))
        }
    
    def generate_quality_report(self, output_format: str = "json") -> Any:
        """Generate a comprehensive quality report using the quality manager."""
        return self.quality_manager.generate_quality_report(output_format=output_format)


# Event handler function
def handle_data_quality_event(event: Event) -> None:
    """
    Handle data quality events for system monitoring and alerting.
    
    This function should be registered as a handler for DATA_QUALITY_* events
    in the application.
    """
    if event.event_type == EventType.DATA_QUALITY_CRITICAL:
        # Critical quality issues detected - should trigger alerts
        data = event.data
        symbol = data.get("symbol", "unknown")
        source = data.get("source", "unknown")
        quality_score = data.get("quality_score", 0)
        
        logger.critical(f"CRITICAL DATA QUALITY ISSUE: {symbol} from {source} - Score: {quality_score}")
        for issue in data.get("issue_summary", []):
            logger.critical(f"  - {issue}")
        
        # This could trigger additional alerts through other channels (SMS, email, etc.)
        
    elif event.event_type == EventType.DATA_QUALITY_WARNING:
        # Warning quality issues detected
        data = event.data
        symbol = data.get("symbol", "unknown")
        source = data.get("source", "unknown")
        quality_score = data.get("quality_score", 0)
        
        logger.warning(f"DATA QUALITY WARNING: {symbol} from {source} - Score: {quality_score}")
        for issue in data.get("issue_summary", []):
            logger.warning(f"  - {issue}")


# Convenience function to set up the data quality system
def setup_data_quality_system(event_bus: EventBus, 
                             persistence: Optional[PersistenceManager] = None, 
                             config: Optional[Dict[str, Any]] = None) -> DataQualityProcessor:
    """
    Set up the data quality system with event handlers and processor.
    
    Args:
        event_bus: The system's event bus
        persistence: Optional persistence manager
        config: Optional configuration dictionary
        
    Returns:
        Configured DataQualityProcessor
    """
    # Create processor
    processor = DataQualityProcessor(
        config=config,
        event_bus=event_bus,
        persistence=persistence
    )
    
    # Register event handlers
    event_bus.register(EventType.DATA_QUALITY_CRITICAL, handle_data_quality_event)
    event_bus.register(EventType.DATA_QUALITY_WARNING, handle_data_quality_event)
    
    logger.info("Data quality system initialized and registered with event bus")
    
    return processor
