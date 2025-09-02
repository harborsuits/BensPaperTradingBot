#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataQualityManager - Comprehensive system for ensuring data quality throughout the trading platform.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
import time
import json
import os
from pathlib import Path

from trading_bot.data.quality.quality_metrics import DataQualityMetrics
from trading_bot.data.quality.quality_checks import DataQualityCheck
from trading_bot.core.event_system import EventBus, Event, EventType

logger = logging.getLogger("DataQualityManager")

class DataQualityManager:
    """
    Comprehensive data quality management system for the trading platform.
    
    This class provides:
    1. Real-time data quality monitoring
    2. Automated data cleaning and repair
    3. Quality metrics tracking and reporting
    4. Integration with the event system for alerts
    5. Historical data quality analysis
    6. Detection and handling of systematic data issues
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 event_bus: Optional[EventBus] = None):
        """
        Initialize the DataQualityManager.
        
        Args:
            config: Configuration dictionary
            event_bus: EventBus for publishing quality events
        """
        self.config = config or {}
        self.event_bus = event_bus
        
        # Initialize metrics
        self.metrics = DataQualityMetrics()
        
        # Data quality thresholds
        self.duplicate_threshold = self.config.get('duplicate_threshold', 0.01)  # 1% duplicates tolerated
        self.missing_data_threshold = self.config.get('missing_data_threshold', 0.05)  # 5% missing data tolerated
        self.outlier_threshold = self.config.get('outlier_threshold', 0.02)  # 2% outliers tolerated
        self.critical_quality_threshold = self.config.get('critical_quality_threshold', 60)  # Min acceptable quality score
        
        # Data source quality tracking
        self.source_quality_scores = {}
        self.symbol_quality_scores = {}
        
        # Data quality history
        self.quality_history = {}
        self.quality_history_size = self.config.get('quality_history_size', 100)
        
        # Path for quality reports
        self.reports_path = self.config.get('reports_path', 'data/quality_reports')
        os.makedirs(self.reports_path, exist_ok=True)
        
        # Flag for whether to auto-repair data
        self.auto_repair = self.config.get('auto_repair', True)
        
        # Register checks (order matters for efficient processing)
        self.quality_checks = [
            DataQualityCheck.check_duplicate_data,
            DataQualityCheck.check_missing_values,
            DataQualityCheck.check_price_outliers,
            DataQualityCheck.check_volume_outliers,
            DataQualityCheck.check_timestamp_irregularities,
            DataQualityCheck.check_ohlc_integrity,
            DataQualityCheck.check_stale_data,
            DataQualityCheck.check_data_gaps
        ]
        
        # Last check timestamp
        self.last_check_time = {}
        
        logger.info("DataQualityManager initialized")
        
    def check_data_quality(self, 
                          data: pd.DataFrame, 
                          symbol: str, 
                          source: str,
                          auto_repair: bool = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform comprehensive data quality checks on the provided DataFrame.
        
        Args:
            data: Input market data DataFrame
            symbol: The symbol/ticker this data represents
            source: Data source identifier
            auto_repair: Whether to automatically repair detected issues (overrides global setting)
            
        Returns:
            Tuple of (processed_data, quality_report)
        """
        if data is None or data.empty:
            logger.warning(f"Empty data received for {symbol} from {source}")
            return data, {"quality_score": 0, "status": "error", "message": "Empty data received"}
            
        # Track check time
        start_time = time.time()
        check_key = f"{symbol}_{source}"
        self.last_check_time[check_key] = datetime.now()
        
        # Initialize quality report
        quality_report = {
            "symbol": symbol,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "row_count": len(data),
            "issues": [],
            "warnings": [],
            "fixed_issues": [],
            "quality_score": 100.0,  # Start with perfect score and deduct
            "fields_analyzed": list(data.columns),
            "status": "ok"
        }
        
        # Create a working copy
        result = data.copy()
        repair_enabled = self.auto_repair if auto_repair is None else auto_repair
        
        # Run all quality checks
        for check_func in self.quality_checks:
            try:
                result, check_results = check_func(result, symbol, source)
                
                # Apply quality deductions
                quality_report["quality_score"] -= check_results.get("quality_deduction", 0)
                
                # Add any detected issues to the report
                if check_results.get("issues"):
                    quality_report["issues"].extend(check_results["issues"])
                
                # Add any warnings to the report
                if check_results.get("warnings"):
                    quality_report["warnings"].extend(check_results["warnings"])
                    
                # If auto-repair is enabled and repairs were made
                if repair_enabled and check_results.get("fixes"):
                    quality_report["fixed_issues"].extend(check_results["fixes"])
            except Exception as e:
                logger.error(f"Error in data quality check {check_func.__name__}: {str(e)}")
                quality_report["issues"].append({
                    "type": "check_error",
                    "check": check_func.__name__,
                    "message": str(e)
                })
                quality_report["quality_score"] -= 5.0  # Penalize for check errors
        
        # Ensure quality score is in valid range
        quality_report["quality_score"] = max(0.0, min(100.0, quality_report["quality_score"]))
        
        # Set status based on quality score
        if quality_report["quality_score"] < self.critical_quality_threshold:
            quality_report["status"] = "critical"
        elif quality_report["quality_score"] < 80:
            quality_report["status"] = "warning"
            
        # Update quality metrics tracking
        self._update_quality_metrics(symbol, source, quality_report)
        
        # Check if we should publish a quality event
        if quality_report["status"] != "ok" and self.event_bus:
            self._publish_quality_event(symbol, source, quality_report)
            
        # Log processing time for performance monitoring
        processing_time = time.time() - start_time
        quality_report["processing_time"] = processing_time
        logger.debug(f"Data quality check for {symbol} from {source} completed in {processing_time:.3f}s")
        
        return result, quality_report
    
    def _update_quality_metrics(self, symbol: str, source: str, quality_report: Dict[str, Any]) -> None:
        """Update internal quality metrics tracking."""
        # Update source quality scores
        if source not in self.source_quality_scores:
            self.source_quality_scores[source] = []
            
        self.source_quality_scores[source].append(quality_report["quality_score"])
        if len(self.source_quality_scores[source]) > self.quality_history_size:
            self.source_quality_scores[source].pop(0)
            
        # Update symbol quality scores
        if symbol not in self.symbol_quality_scores:
            self.symbol_quality_scores[symbol] = []
            
        self.symbol_quality_scores[symbol].append(quality_report["quality_score"])
        if len(self.symbol_quality_scores[symbol]) > self.quality_history_size:
            self.symbol_quality_scores[symbol].pop(0)
            
        # Update quality history
        key = f"{symbol}_{source}"
        if key not in self.quality_history:
            self.quality_history[key] = []
            
        self.quality_history[key].append({
            "timestamp": quality_report["timestamp"],
            "quality_score": quality_report["quality_score"],
            "issue_count": len(quality_report["issues"]),
            "fixed_count": len(quality_report["fixed_issues"])
        })
        
        if len(self.quality_history[key]) > self.quality_history_size:
            self.quality_history[key].pop(0)
            
        # Update metrics in the metrics object
        self.metrics.update_metrics(symbol, source, quality_report)
    
    def _publish_quality_event(self, symbol: str, source: str, quality_report: Dict[str, Any]) -> None:
        """Publish a data quality event to the event bus."""
        if not self.event_bus:
            return
            
        event_type = (EventType.DATA_QUALITY_CRITICAL if quality_report["status"] == "critical" 
                      else EventType.DATA_QUALITY_WARNING)
            
        event_data = {
            "symbol": symbol,
            "source": source,
            "quality_score": quality_report["quality_score"],
            "issue_count": len(quality_report["issues"]),
            "fixed_count": len(quality_report["fixed_issues"]),
            "status": quality_report["status"],
            "timestamp": quality_report["timestamp"]
        }
        
        # Add first 3 issues as summary (avoid making event too large)
        if quality_report["issues"]:
            event_data["issue_summary"] = [i["message"] for i in quality_report["issues"][:3]]
            
        event = Event(
            event_type=event_type,
            data=event_data
        )
        
        self.event_bus.publish(event)
        logger.info(f"Published {event_type} event for {symbol} from {source}")
    
    def get_data_source_quality(self, source: str = None) -> Dict[str, Any]:
        """
        Get quality metrics for data sources.
        
        Args:
            source: Optional specific source to get metrics for
            
        Returns:
            Dictionary of quality metrics by source
        """
        if source:
            if source not in self.source_quality_scores:
                return {"source": source, "quality_score": None, "data_points": 0}
                
            scores = self.source_quality_scores[source]
            return {
                "source": source,
                "quality_score": sum(scores) / len(scores) if scores else None,
                "data_points": len(scores),
                "trend": "stable" if len(scores) < 2 else 
                         "improving" if scores[-1] > scores[0] else 
                         "declining" if scores[-1] < scores[0] else "stable"
            }
        else:
            # Return all sources
            result = {}
            for src in self.source_quality_scores:
                scores = self.source_quality_scores[src]
                result[src] = {
                    "quality_score": sum(scores) / len(scores) if scores else None,
                    "data_points": len(scores),
                    "trend": "stable" if len(scores) < 2 else 
                             "improving" if scores[-1] > scores[0] else 
                             "declining" if scores[-1] < scores[0] else "stable"
                }
            return result
    
    def get_symbol_quality(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get quality metrics for symbols.
        
        Args:
            symbol: Optional specific symbol to get metrics for
            
        Returns:
            Dictionary of quality metrics by symbol
        """
        if symbol:
            if symbol not in self.symbol_quality_scores:
                return {"symbol": symbol, "quality_score": None, "data_points": 0}
                
            scores = self.symbol_quality_scores[symbol]
            return {
                "symbol": symbol,
                "quality_score": sum(scores) / len(scores) if scores else None,
                "data_points": len(scores),
                "trend": "stable" if len(scores) < 2 else 
                         "improving" if scores[-1] > scores[0] else 
                         "declining" if scores[-1] < scores[0] else "stable"
            }
        else:
            # Return all symbols
            result = {}
            for sym in self.symbol_quality_scores:
                scores = self.symbol_quality_scores[sym]
                result[sym] = {
                    "quality_score": sum(scores) / len(scores) if scores else None,
                    "data_points": len(scores),
                    "trend": "stable" if len(scores) < 2 else 
                             "improving" if scores[-1] > scores[0] else 
                             "declining" if scores[-1] < scores[0] else "stable"
                }
            return result
    
    def generate_quality_report(self, output_format: str = "json") -> Union[Dict[str, Any], str]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            output_format: Format for the report ('json' or 'html')
            
        Returns:
            Report in the requested format
        """
        # Collect overall metrics
        report = {
            "timestamp": datetime.now().isoformat(),
            "sources": self.get_data_source_quality(),
            "symbols": self.get_symbol_quality(),
            "overall_metrics": self.metrics.get_overall_metrics(),
            "most_recent_issues": self.metrics.get_recent_issues(limit=10)
        }
        
        # Add system recommendations
        report["recommendations"] = self._generate_recommendations()
        
        # Save report to file
        filename = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filepath = os.path.join(self.reports_path, f"{filename}.json")
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Saved data quality report to {filepath}")
        
        # Return in requested format
        if output_format == "html":
            return self._convert_report_to_html(report)
        else:
            return report
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate system recommendations based on quality metrics."""
        recommendations = []
        
        # Check source quality
        for source, metrics in self.get_data_source_quality().items():
            if metrics.get("quality_score", 100) < 70:
                recommendations.append({
                    "type": "source_quality",
                    "source": source,
                    "severity": "high" if metrics.get("quality_score", 0) < 50 else "medium",
                    "message": f"Data source {source} has low quality score ({metrics.get('quality_score', 0):.1f}). Consider alternative sources."
                })
            
            if metrics.get("trend") == "declining":
                recommendations.append({
                    "type": "source_trend",
                    "source": source,
                    "severity": "medium",
                    "message": f"Data quality from {source} is declining. Monitor for continued degradation."
                })
                
        # Check symbol quality
        for symbol, metrics in self.get_symbol_quality().items():
            if metrics.get("quality_score", 100) < 70:
                recommendations.append({
                    "type": "symbol_quality",
                    "symbol": symbol,
                    "severity": "high" if metrics.get("quality_score", 0) < 50 else "medium",
                    "message": f"Symbol {symbol} has low quality score ({metrics.get('quality_score', 0):.1f}). Verify data sources or consider excluding from analysis."
                })
        
        # Add overall system recommendations
        overall = self.metrics.get_overall_metrics()
        
        if overall.get("duplicate_rate", 0) > self.duplicate_threshold:
            recommendations.append({
                "type": "system",
                "severity": "high",
                "message": f"System-wide duplicate data rate ({overall.get('duplicate_rate', 0)*100:.1f}%) exceeds threshold. Check data collection and storage systems."
            })
            
        if overall.get("missing_data_rate", 0) > self.missing_data_threshold:
            recommendations.append({
                "type": "system",
                "severity": "high",
                "message": f"System-wide missing data rate ({overall.get('missing_data_rate', 0)*100:.1f}%) exceeds threshold. Review data collection process and provider reliability."
            })
            
        return recommendations
    
    def _convert_report_to_html(self, report: Dict[str, Any]) -> str:
        """Convert JSON report to HTML format."""
        # Simple HTML conversion - could be enhanced with templates
        html = f"""
        <html>
        <head>
            <title>Data Quality Report - {report['timestamp']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333366; }}
                .metric {{ margin-bottom: 10px; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .critical {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Data Quality Report</h1>
            <p>Generated: {report['timestamp']}</p>
            
            <h2>Overall Metrics</h2>
        """
        
        # Add overall metrics
        for key, value in report['overall_metrics'].items():
            if isinstance(value, float):
                html += f"<div class='metric'><b>{key}:</b> {value:.2f}</div>"
            else:
                html += f"<div class='metric'><b>{key}:</b> {value}</div>"
        
        # Add recommendations
        html += "<h2>Recommendations</h2>"
        if report['recommendations']:
            html += "<ul>"
            for rec in report['recommendations']:
                css_class = "critical" if rec['severity'] == "high" else "warning"
                html += f"<li class='{css_class}'>{rec['message']}</li>"
            html += "</ul>"
        else:
            html += "<p>No recommendations at this time.</p>"
        
        # Add data sources table
        html += "<h2>Data Sources</h2>"
        html += """
        <table>
            <tr>
                <th>Source</th>
                <th>Quality Score</th>
                <th>Data Points</th>
                <th>Trend</th>
            </tr>
        """
        
        for source, metrics in report['sources'].items():
            quality_score = metrics.get('quality_score')
            if quality_score is not None:
                css_class = "good" if quality_score > 80 else "warning" if quality_score > 60 else "critical"
                score_display = f"<span class='{css_class}'>{quality_score:.1f}</span>"
            else:
                score_display = "N/A"
                
            trend = metrics.get('trend', 'stable')
            trend_display = "↗️" if trend == "improving" else "↘️" if trend == "declining" else "→"
                
            html += f"""
            <tr>
                <td>{source}</td>
                <td>{score_display}</td>
                <td>{metrics.get('data_points', 0)}</td>
                <td>{trend_display} {trend}</td>
            </tr>
            """
            
        html += "</table>"
        
        # Add recent issues
        html += "<h2>Recent Issues</h2>"
        if report['most_recent_issues']:
            html += "<ul>"
            for issue in report['most_recent_issues']:
                html += f"<li><b>{issue['timestamp']}:</b> {issue['symbol']} from {issue['source']} - {issue['message']}</li>"
            html += "</ul>"
        else:
            html += "<p>No recent issues detected.</p>"
            
        html += """
        </body>
        </html>
        """
        
        # Save HTML report
        filename = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        filepath = os.path.join(self.reports_path, f"{filename}.html")
        
        with open(filepath, 'w') as f:
            f.write(html)
            
        logger.info(f"Saved HTML data quality report to {filepath}")
        
        return html
