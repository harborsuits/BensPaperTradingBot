#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataQualityMetrics - Track and report data quality metrics across the trading system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger("DataQualityMetrics")

class DataQualityMetrics:
    """
    Track and manage data quality metrics across the trading system.
    
    This class provides:
    1. Aggregation of quality metrics across symbols and sources
    2. Tracking of data quality over time
    3. Statistical analysis of data quality trends
    4. Identification of systematic quality issues
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize the DataQualityMetrics.
        
        Args:
            history_size: Number of recent quality events to keep in memory
        """
        self.history_size = history_size
        
        # Overall metrics
        self.total_checks = 0
        self.total_issues = 0
        self.total_warnings = 0
        self.total_fixed = 0
        
        # Track specific issue types
        self.issue_counts = {
            "duplicate_data": 0,
            "missing_values": 0,
            "price_outliers": 0,
            "volume_outliers": 0,
            "timestamp_irregularities": 0,
            "ohlc_integrity": 0,
            "stale_data": 0,
            "data_gaps": 0,
            "other": 0
        }
        
        # Track metrics by source and symbol
        self.source_metrics = {}
        self.symbol_metrics = {}
        
        # Keep track of recent issues
        self.recent_issues = deque(maxlen=history_size)
        
        # Track rates
        self.duplicate_rate = 0.0
        self.missing_data_rate = 0.0
        self.outlier_rate = 0.0
        self.overall_quality_score = 100.0
        
        # Initialization timestamp
        self.start_time = datetime.now()
    
    def update_metrics(self, symbol: str, source: str, quality_report: Dict[str, Any]) -> None:
        """
        Update quality metrics based on a quality report.
        
        Args:
            symbol: Symbol being checked
            source: Data source
            quality_report: Quality check report
        """
        self.total_checks += 1
        
        # Update issue counts
        issues = quality_report.get("issues", [])
        warnings = quality_report.get("warnings", [])
        fixed = quality_report.get("fixed_issues", [])
        
        self.total_issues += len(issues)
        self.total_warnings += len(warnings)
        self.total_fixed += len(fixed)
        
        # Track overall quality score (weighted average)
        quality_score = quality_report.get("quality_score", 100.0)
        row_count = quality_report.get("row_count", 0)
        
        weight = row_count / max(1, (self.total_checks * 100))  # Approximate weight
        self.overall_quality_score = (self.overall_quality_score * (1 - weight)) + (quality_score * weight)
        
        # Update issue type counts
        for issue in issues:
            issue_type = issue.get("type", "other")
            if issue_type in self.issue_counts:
                self.issue_counts[issue_type] += 1
            else:
                self.issue_counts["other"] += 1
                
        # Add issues to recent issues list
        for issue in issues:
            self.recent_issues.append({
                "timestamp": quality_report.get("timestamp", datetime.now().isoformat()),
                "symbol": symbol,
                "source": source,
                "type": issue.get("type", "unknown"),
                "message": issue.get("message", "Unknown issue")
            })
            
        # Update source metrics
        if source not in self.source_metrics:
            self.source_metrics[source] = {
                "total_checks": 0,
                "total_issues": 0,
                "total_fixed": 0,
                "quality_scores": [],
                "symbols_checked": set()
            }
            
        self.source_metrics[source]["total_checks"] += 1
        self.source_metrics[source]["total_issues"] += len(issues)
        self.source_metrics[source]["total_fixed"] += len(fixed)
        self.source_metrics[source]["quality_scores"].append(quality_score)
        self.source_metrics[source]["symbols_checked"].add(symbol)
        
        # Trim quality scores history if needed
        if len(self.source_metrics[source]["quality_scores"]) > self.history_size:
            self.source_metrics[source]["quality_scores"] = self.source_metrics[source]["quality_scores"][-self.history_size:]
            
        # Update symbol metrics
        if symbol not in self.symbol_metrics:
            self.symbol_metrics[symbol] = {
                "total_checks": 0,
                "total_issues": 0,
                "total_fixed": 0,
                "quality_scores": [],
                "sources_checked": set()
            }
            
        self.symbol_metrics[symbol]["total_checks"] += 1
        self.symbol_metrics[symbol]["total_issues"] += len(issues)
        self.symbol_metrics[symbol]["total_fixed"] += len(fixed)
        self.symbol_metrics[symbol]["quality_scores"].append(quality_score)
        self.symbol_metrics[symbol]["sources_checked"].add(source)
        
        # Trim quality scores history if needed
        if len(self.symbol_metrics[symbol]["quality_scores"]) > self.history_size:
            self.symbol_metrics[symbol]["quality_scores"] = self.symbol_metrics[symbol]["quality_scores"][-self.history_size:]
            
        # Update rate calculations
        if "duplicate_count" in quality_report:
            self.duplicate_rate = (self.duplicate_rate * (self.total_checks - 1) + 
                                  quality_report["duplicate_count"] / max(1, row_count)) / self.total_checks
                                  
        if "missing_count" in quality_report:
            self.missing_data_rate = (self.missing_data_rate * (self.total_checks - 1) + 
                                     quality_report["missing_count"] / max(1, row_count)) / self.total_checks
                                     
        if "outlier_count" in quality_report:
            self.outlier_rate = (self.outlier_rate * (self.total_checks - 1) + 
                                quality_report["outlier_count"] / max(1, row_count)) / self.total_checks
    
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Get overall system-wide quality metrics.
        
        Returns:
            Dictionary of overall metrics
        """
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600.0  # hours
        
        return {
            "total_checks": self.total_checks,
            "total_issues": self.total_issues,
            "total_warnings": self.total_warnings,
            "total_fixed": self.total_fixed,
            "issue_counts": self.issue_counts.copy(),
            "overall_quality_score": self.overall_quality_score,
            "duplicate_rate": self.duplicate_rate,
            "missing_data_rate": self.missing_data_rate,
            "outlier_rate": self.outlier_rate,
            "issues_per_hour": self.total_issues / max(1.0, runtime),
            "sources_tracked": len(self.source_metrics),
            "symbols_tracked": len(self.symbol_metrics),
            "run_time_hours": runtime
        }
    
    def get_source_metrics(self, source: str = None) -> Dict[str, Any]:
        """
        Get metrics for data sources.
        
        Args:
            source: Optional specific source to get metrics for
            
        Returns:
            Dictionary of metrics by source
        """
        if source:
            if source not in self.source_metrics:
                return {"source": source, "total_checks": 0}
                
            metrics = self.source_metrics[source]
            scores = metrics["quality_scores"]
            
            return {
                "source": source,
                "total_checks": metrics["total_checks"],
                "total_issues": metrics["total_issues"],
                "total_fixed": metrics["total_fixed"],
                "unique_symbols": len(metrics["symbols_checked"]),
                "avg_quality_score": sum(scores) / len(scores) if scores else None,
                "recent_quality_score": scores[-1] if scores else None,
                "quality_trend": "stable" if len(scores) < 2 else 
                                 "improving" if scores[-1] > scores[0] else 
                                 "declining" if scores[-1] < scores[0] else "stable",
                "symbols_checked": list(metrics["symbols_checked"])
            }
        else:
            # Return metrics for all sources
            result = {}
            for src, metrics in self.source_metrics.items():
                scores = metrics["quality_scores"]
                result[src] = {
                    "total_checks": metrics["total_checks"],
                    "total_issues": metrics["total_issues"],
                    "total_fixed": metrics["total_fixed"],
                    "unique_symbols": len(metrics["symbols_checked"]),
                    "avg_quality_score": sum(scores) / len(scores) if scores else None,
                    "recent_quality_score": scores[-1] if scores else None,
                    "quality_trend": "stable" if len(scores) < 2 else 
                                     "improving" if scores[-1] > scores[0] else 
                                     "declining" if scores[-1] < scores[0] else "stable"
                }
            return result
    
    def get_symbol_metrics(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get metrics for symbols.
        
        Args:
            symbol: Optional specific symbol to get metrics for
            
        Returns:
            Dictionary of metrics by symbol
        """
        if symbol:
            if symbol not in self.symbol_metrics:
                return {"symbol": symbol, "total_checks": 0}
                
            metrics = self.symbol_metrics[symbol]
            scores = metrics["quality_scores"]
            
            return {
                "symbol": symbol,
                "total_checks": metrics["total_checks"],
                "total_issues": metrics["total_issues"],
                "total_fixed": metrics["total_fixed"],
                "unique_sources": len(metrics["sources_checked"]),
                "avg_quality_score": sum(scores) / len(scores) if scores else None,
                "recent_quality_score": scores[-1] if scores else None,
                "quality_trend": "stable" if len(scores) < 2 else 
                                 "improving" if scores[-1] > scores[0] else 
                                 "declining" if scores[-1] < scores[0] else "stable",
                "sources_checked": list(metrics["sources_checked"])
            }
        else:
            # Return metrics for all symbols
            result = {}
            for sym, metrics in self.symbol_metrics.items():
                scores = metrics["quality_scores"]
                result[sym] = {
                    "total_checks": metrics["total_checks"],
                    "total_issues": metrics["total_issues"],
                    "total_fixed": metrics["total_fixed"],
                    "unique_sources": len(metrics["sources_checked"]),
                    "avg_quality_score": sum(scores) / len(scores) if scores else None,
                    "recent_quality_score": scores[-1] if scores else None,
                    "quality_trend": "stable" if len(scores) < 2 else 
                                     "improving" if scores[-1] > scores[0] else 
                                     "declining" if scores[-1] < scores[0] else "stable"
                }
            return result
    
    def get_recent_issues(self, limit: int = 10, issue_type: str = None, 
                         symbol: str = None, source: str = None) -> List[Dict[str, Any]]:
        """
        Get most recent data quality issues.
        
        Args:
            limit: Maximum number of issues to return
            issue_type: Optional filter by issue type
            symbol: Optional filter by symbol
            source: Optional filter by source
            
        Returns:
            List of recent issues matching filters
        """
        filtered_issues = list(self.recent_issues)
        
        # Apply filters
        if issue_type:
            filtered_issues = [i for i in filtered_issues if i.get("type") == issue_type]
            
        if symbol:
            filtered_issues = [i for i in filtered_issues if i.get("symbol") == symbol]
            
        if source:
            filtered_issues = [i for i in filtered_issues if i.get("source") == source]
            
        # Return most recent first, up to limit
        return sorted(filtered_issues, 
                     key=lambda x: x.get("timestamp", ""), 
                     reverse=True)[:limit]
    
    def get_issue_frequency(self, window_hours: float = 24.0) -> Dict[str, int]:
        """
        Get frequency of different issue types within a time window.
        
        Args:
            window_hours: Time window in hours
            
        Returns:
            Dictionary mapping issue types to their frequency
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        cutoff_str = cutoff.isoformat()
        
        # Filter issues by timestamp
        recent = [i for i in self.recent_issues if i.get("timestamp", "") >= cutoff_str]
        
        # Count by type
        frequency = {}
        for issue in recent:
            issue_type = issue.get("type", "unknown")
            frequency[issue_type] = frequency.get(issue_type, 0) + 1
            
        return frequency
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self.__init__(history_size=self.history_size)
        logger.info("Data quality metrics have been reset")
