#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Connector - Connect the data quality metrics to the monitoring dashboard.

This module bridges the data pipeline quality metrics with the trading system's
monitoring dashboard, providing real-time visibility into data quality.
"""

import logging
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time
import threading

from trading_bot.core.event_system import EventBus, Event, EventType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.data.persistence import PersistenceManager

logger = logging.getLogger("DashboardConnector")

class DataQualityDashboardConnector:
    """
    Connector between data quality metrics and system dashboard.
    
    This connector:
    1. Subscribes to data quality events
    2. Collects and aggregates quality metrics
    3. Formats data for dashboard display
    4. Pushes updates to the dashboard
    5. Provides historical quality metrics
    """
    
    def __init__(self, 
                event_bus: EventBus,
                pipeline: DataPipeline,
                persistence: Optional[PersistenceManager] = None,
                update_interval: int = 60):  # Seconds
        """
        Initialize the dashboard connector.
        
        Args:
            event_bus: EventBus for subscribing to events
            pipeline: DataPipeline for accessing quality metrics
            persistence: PersistenceManager for storing historical metrics
            update_interval: Interval (seconds) for pushing updates to dashboard
        """
        self.event_bus = event_bus
        self.pipeline = pipeline
        self.persistence = persistence
        self.update_interval = update_interval
        
        # Quality metrics storage
        self.current_metrics = {
            'overall_quality': 100.0,
            'symbols': {},
            'issues_by_type': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Historical metrics
        self.historical_metrics = []
        self.max_history_entries = 1000  # About 16 hours at 1-minute intervals
        
        # Register for quality events
        self._register_event_handlers()
        
        # Flag for background thread
        self.running = False
        self.update_thread = None
        
        logger.info("DataQualityDashboardConnector initialized")
    
    def _register_event_handlers(self) -> None:
        """Register for quality-related events."""
        self.event_bus.register(EventType.DATA_QUALITY_WARNING, self._handle_quality_event)
        self.event_bus.register(EventType.DATA_QUALITY_CRITICAL, self._handle_quality_event)
        self.event_bus.register(EventType.DATA_PROCESSED, self._handle_data_processed)
        
        logger.info("Registered for quality events")
    
    def _handle_quality_event(self, event: Event) -> None:
        """
        Handle data quality events.
        
        Args:
            event: Quality event with data quality information
        """
        event_data = event.data
        if not event_data:
            return
        
        symbol = event_data.get('symbol', 'unknown')
        source = event_data.get('source', 'unknown')
        quality_score = event_data.get('quality_score', 0)
        issue_count = event_data.get('issue_count', 0)
        issues = event_data.get('issue_summary', [])
        
        # Update the metrics
        self._update_symbol_metrics(symbol, quality_score, issue_count, issues, event.event_type)
        
        # If critical, make sure it's flagged accordingly
        if event.event_type == EventType.DATA_QUALITY_CRITICAL:
            logger.warning(f"Critical quality issue for {symbol}: score={quality_score}")
        
        # Push update to dashboard if this is a significant change
        if issue_count > 0 or quality_score < 80:
            self._push_dashboard_update()
    
    def _handle_data_processed(self, event: Event) -> None:
        """
        Handle data processed events.
        
        Args:
            event: Data processed event with metadata
        """
        event_data = event.data
        if not event_data or 'metadata' not in event_data:
            return
        
        metadata = event_data['metadata']
        
        # Check if quality data is available
        if 'quality_score' in metadata:
            symbol = metadata.get('symbol', 'unknown')
            quality_score = metadata.get('quality_score', 100.0)
            issues_detected = metadata.get('issues_detected', 0)
            issues_fixed = metadata.get('issues_fixed', 0)
            
            # Update metrics
            self._update_symbol_metrics(
                symbol, quality_score, issues_detected, 
                [], EventType.DATA_QUALITY_WARNING
            )
    
    def _update_symbol_metrics(self, symbol: str, quality_score: float, 
                             issue_count: int, issues: List[str],
                             event_type: EventType) -> None:
        """
        Update metrics for a specific symbol.
        
        Args:
            symbol: Symbol identifier
            quality_score: Quality score (0-100)
            issue_count: Number of issues detected
            issues: List of issue descriptions
            event_type: Type of the triggering event
        """
        timestamp = datetime.now()
        
        # Create or update symbol entry
        if symbol not in self.current_metrics['symbols']:
            self.current_metrics['symbols'][symbol] = {
                'quality_score': quality_score,
                'issue_count': issue_count,
                'last_issues': issues[:5],  # Limit to 5 issues
                'last_updated': timestamp.isoformat(),
                'critical': event_type == EventType.DATA_QUALITY_CRITICAL
            }
        else:
            # Update existing metrics
            self.current_metrics['symbols'][symbol].update({
                'quality_score': quality_score,
                'issue_count': issue_count,
                'last_updated': timestamp.isoformat()
            })
            
            # Only update issues if there are new ones
            if issues:
                self.current_metrics['symbols'][symbol]['last_issues'] = issues[:5]
            
            # Mark as critical if applicable
            if event_type == EventType.DATA_QUALITY_CRITICAL:
                self.current_metrics['symbols'][symbol]['critical'] = True
        
        # Update overall metrics
        all_scores = [s.get('quality_score', 0) for s in self.current_metrics['symbols'].values()]
        self.current_metrics['overall_quality'] = sum(all_scores) / max(1, len(all_scores))
        self.current_metrics['last_updated'] = timestamp.isoformat()
        
        # Update issue types
        for issue in issues:
            issue_type = self._categorize_issue(issue)
            if issue_type not in self.current_metrics['issues_by_type']:
                self.current_metrics['issues_by_type'][issue_type] = 0
            self.current_metrics['issues_by_type'][issue_type] += 1
        
        # Save this data point to history
        self._add_historical_datapoint(timestamp, symbol, quality_score, issue_count)
    
    def _categorize_issue(self, issue_description: str) -> str:
        """
        Categorize an issue based on its description.
        
        Args:
            issue_description: Description of the issue
            
        Returns:
            Category name
        """
        issue_lower = issue_description.lower()
        
        # Define categories and keywords
        categories = {
            'missing_data': ['missing', 'null', 'nan', 'empty'],
            'outliers': ['outlier', 'extreme', 'anomaly', 'spike'],
            'ohlc_integrity': ['ohlc', 'high < low', 'integrity'],
            'duplicates': ['duplicate', 'repeated'],
            'gaps': ['gap', 'missing bar', 'time gap'],
            'stale_data': ['stale', 'unchanged', 'frozen'],
            'timestamp': ['timestamp', 'datetime', 'time order']
        }
        
        # Find matching category
        for category, keywords in categories.items():
            if any(keyword in issue_lower for keyword in keywords):
                return category
        
        # Default category
        return 'other'
    
    def _add_historical_datapoint(self, timestamp: datetime, symbol: str, 
                               quality_score: float, issue_count: int) -> None:
        """
        Add a data point to historical metrics.
        
        Args:
            timestamp: Time of the data point
            symbol: Symbol identifier
            quality_score: Quality score (0-100)
            issue_count: Number of issues detected
        """
        # Add to in-memory history
        self.historical_metrics.append({
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'quality_score': quality_score,
            'issue_count': issue_count
        })
        
        # Trim history if needed
        if len(self.historical_metrics) > self.max_history_entries:
            self.historical_metrics = self.historical_metrics[-self.max_history_entries:]
        
        # Store in persistence if available
        if self.persistence:
            try:
                self.persistence.store_object(
                    f"quality_metrics_{timestamp.strftime('%Y%m%d')}",
                    {
                        'timestamp': timestamp.isoformat(),
                        'symbol': symbol,
                        'quality_score': quality_score,
                        'issue_count': issue_count
                    }
                )
            except Exception as e:
                logger.error(f"Error storing quality metrics: {str(e)}")
    
    def _push_dashboard_update(self) -> None:
        """Push updated metrics to the dashboard."""
        try:
            # Create dashboard event
            dashboard_event = Event(
                event_type=EventType.DASHBOARD_UPDATE,
                data={
                    'type': 'data_quality',
                    'metrics': self.current_metrics
                }
            )
            
            # Publish the event
            self.event_bus.publish(dashboard_event)
            logger.debug("Published dashboard update")
        except Exception as e:
            logger.error(f"Error pushing dashboard update: {str(e)}")
    
    def start_automatic_updates(self) -> None:
        """Start automatic dashboard updates."""
        if self.running:
            logger.warning("Automatic updates already running")
            return
        
        self.running = True
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        logger.info(f"Started automatic dashboard updates (interval: {self.update_interval}s)")
    
    def stop_automatic_updates(self) -> None:
        """Stop automatic dashboard updates."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
            self.update_thread = None
        
        logger.info("Stopped automatic dashboard updates")
    
    def _update_loop(self) -> None:
        """Background thread for periodic dashboard updates."""
        while self.running:
            try:
                # Get latest pipeline stats
                pipeline_stats = self.pipeline.get_pipeline_stats()
                
                # Update metrics from pipeline stats
                if 'quality_stats' in pipeline_stats:
                    quality_stats = pipeline_stats['quality_stats']
                    if 'average_quality_score' in quality_stats:
                        self.current_metrics['overall_quality'] = quality_stats['average_quality_score']
                
                # Push update to dashboard
                self._push_dashboard_update()
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {str(e)}")
            
            # Sleep for the update interval
            time.sleep(self.update_interval)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current quality metrics.
        
        Returns:
            Dictionary of current quality metrics
        """
        return self.current_metrics.copy()
    
    def get_historical_metrics(self, 
                             symbol: Optional[str] = None, 
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get historical quality metrics.
        
        Args:
            symbol: Optional symbol filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of historical metrics
        """
        # Default time range (last 24 hours)
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        # Filter by time first (most efficient)
        filtered = []
        for entry in self.historical_metrics:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if start_time <= entry_time <= end_time:
                # Filter by symbol if specified
                if not symbol or entry['symbol'] == symbol:
                    filtered.append(entry)
        
        return filtered
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """
        Get a summary of data quality metrics.
        
        Returns:
            Dictionary with quality summary information
        """
        # Get overall metrics
        overall_quality = self.current_metrics['overall_quality']
        symbols_count = len(self.current_metrics['symbols'])
        
        # Calculate metrics
        critical_symbols = sum(1 for s in self.current_metrics['symbols'].values() 
                              if s.get('critical', False))
        
        # Find lowest quality symbols
        sorted_symbols = sorted(
            self.current_metrics['symbols'].items(),
            key=lambda x: x[1].get('quality_score', 100)
        )
        lowest_quality = {
            s[0]: s[1] for s in sorted_symbols[:3]  # Top 3 worst
        } if sorted_symbols else {}
        
        # Compile summary
        return {
            'overall_quality': overall_quality,
            'symbols_count': symbols_count,
            'critical_count': critical_symbols,
            'lowest_quality_symbols': lowest_quality,
            'recent_history': self.historical_metrics[-10:],  # Last 10 entries
            'last_updated': self.current_metrics['last_updated']
        }

def connect_to_dashboard(event_bus: EventBus, 
                        pipeline: DataPipeline,
                        persistence: Optional[PersistenceManager] = None,
                        auto_start: bool = True) -> DataQualityDashboardConnector:
    """
    Connect the data quality metrics to the dashboard.
    
    Args:
        event_bus: EventBus for events
        pipeline: DataPipeline for quality metrics
        persistence: Optional persistence manager for historical data
        auto_start: Whether to automatically start dashboard updates
        
    Returns:
        Configured DataQualityDashboardConnector
    """
    connector = DataQualityDashboardConnector(
        event_bus=event_bus,
        pipeline=pipeline,
        persistence=persistence
    )
    
    if auto_start:
        connector.start_automatic_updates()
    
    logger.info("Quality metrics connected to dashboard")
    
    return connector
