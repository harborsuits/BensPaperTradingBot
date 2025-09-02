#!/usr/bin/env python3
"""
Feature Flag Metrics Integration

This module provides metrics tracking and analysis for feature flags, 
allowing assessment of feature performance impact and usage patterns.
"""

import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
from collections import defaultdict

# Setup logging
logger = logging.getLogger(__name__)

class FeatureFlagMetrics:
    """
    Tracks and analyzes metrics related to feature flag usage and performance impact.
    
    This class provides mechanisms for:
    - Recording feature flag evaluations and their contexts
    - Associating business and technical metrics with feature flags
    - Analyzing performance differences between flag states
    - Generating reports on feature flag impact
    """
    
    def __init__(self, storage_path: str = "data/feature_flags/metrics"):
        """
        Initialize the metrics tracking system.
        
        Args:
            storage_path: Directory to store metrics data
        """
        self.storage_path = storage_path
        self._ensure_storage_path()
        
        # In-memory data structures for metrics
        self.evaluations = defaultdict(int)  # Count of flag evaluations
        self.evaluation_contexts = defaultdict(list)  # Contexts of evaluations
        self.performance_samples = defaultdict(list)  # Performance metrics
        self.user_segments = defaultdict(lambda: defaultdict(int))  # User segment data
        
        # Track active experiments
        self.experiments = {}
        
        # Load existing data if available
        self._load_existing_data()
        
        logger.info(f"Feature flag metrics tracking initialized with storage at {storage_path}")
    
    def _ensure_storage_path(self):
        """Create storage directories if they don't exist"""
        os.makedirs(f"{self.storage_path}/daily", exist_ok=True)
        os.makedirs(f"{self.storage_path}/experiments", exist_ok=True)
        os.makedirs(f"{self.storage_path}/reports", exist_ok=True)
    
    def _load_existing_data(self):
        """Load existing metrics data if available"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            daily_file = f"{self.storage_path}/daily/{today}.json"
            
            if os.path.exists(daily_file):
                with open(daily_file, 'r') as f:
                    data = json.load(f)
                    
                # Restore in-memory structures
                self.evaluations = defaultdict(int, data.get('evaluations', {}))
                
                # Convert stored contexts back to defaultdict structure
                stored_contexts = data.get('evaluation_contexts', {})
                for flag_name, contexts in stored_contexts.items():
                    self.evaluation_contexts[flag_name] = contexts
                
                # Restore user segments
                segments = data.get('user_segments', {})
                for flag_name, segment_data in segments.items():
                    self.user_segments[flag_name] = defaultdict(int, segment_data)
                
                logger.info(f"Loaded existing metrics data from {daily_file}")
        except Exception as e:
            logger.warning(f"Failed to load existing metrics data: {str(e)}")
    
    def record_evaluation(self, flag_name: str, value: bool, context: Optional[Dict[str, Any]] = None):
        """
        Record a feature flag evaluation.
        
        Args:
            flag_name: Name of the feature flag
            value: Resulting value of the evaluation (True/False)
            context: Optional context data about the evaluation (user, environment, etc.)
        """
        # Increment evaluation counter
        key = f"{flag_name}:{str(value).lower()}"
        self.evaluations[key] += 1
        
        # Record context if provided
        if context:
            # Add timestamp to context
            eval_context = context.copy()
            eval_context['timestamp'] = datetime.now().isoformat()
            eval_context['value'] = value
            
            # Limit stored contexts to prevent memory issues
            if len(self.evaluation_contexts[flag_name]) >= 1000:
                self.evaluation_contexts[flag_name] = self.evaluation_contexts[flag_name][-999:]
            
            self.evaluation_contexts[flag_name].append(eval_context)
            
            # Track user segments if user_id is in context
            if 'user_id' in context:
                user_id = context['user_id']
                segment_key = 'unknown'
                
                # Determine user segment from context
                if 'user_segment' in context:
                    segment_key = context['user_segment']
                elif 'user_role' in context:
                    segment_key = context['user_role']
                    
                self.user_segments[flag_name][segment_key] += 1
    
    def associate_metrics(self, flag_name: str, metrics: Dict[str, Any], value: bool, 
                         context: Optional[Dict[str, Any]] = None):
        """
        Associate business or technical metrics with a feature flag state.
        
        Args:
            flag_name: Name of the feature flag
            metrics: Dictionary of metrics to associate
            value: State of the flag (True/False)
            context: Optional context data
        """
        sample = {
            'timestamp': datetime.now().isoformat(),
            'flag_value': value,
            'metrics': metrics
        }
        
        if context:
            sample['context'] = context
            
        self.performance_samples[flag_name].append(sample)
        
        # Persist metrics immediately for important performance data
        if len(self.performance_samples[flag_name]) % 50 == 0:
            self._persist_metrics()
    
    def _persist_metrics(self):
        """Persist current metrics to storage"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            daily_file = f"{self.storage_path}/daily/{today}.json"
            
            # Prepare data for serialization
            data = {
                'evaluations': dict(self.evaluations),
                'evaluation_contexts': dict(self.evaluation_contexts),
                'user_segments': {k: dict(v) for k, v in self.user_segments.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            # Store performance samples separately as they can grow large
            performance_file = f"{self.storage_path}/daily/{today}_performance.json"
            performance_data = {k: v for k, v in self.performance_samples.items()}
            
            # Write files
            with open(daily_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            with open(performance_file, 'w') as f:
                json.dump(performance_data, f, indent=2)
                
            logger.debug(f"Persisted metrics data to {daily_file} and {performance_file}")
        except Exception as e:
            logger.error(f"Failed to persist metrics: {str(e)}")
    
    def analyze_flag_impact(self, flag_name: str, 
                           metric_keys: List[str], 
                           time_period: timedelta = timedelta(days=7)
                          ) -> Dict[str, Any]:
        """
        Analyze the impact of a feature flag on specific metrics.
        
        Args:
            flag_name: Name of the feature flag to analyze
            metric_keys: List of metric keys to analyze
            time_period: Time period to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Filter samples by time period
        cutoff = datetime.now() - time_period
        filtered_samples = []
        
        for sample in self.performance_samples[flag_name]:
            sample_time = datetime.fromisoformat(sample['timestamp'])
            if sample_time >= cutoff:
                filtered_samples.append(sample)
        
        if len(filtered_samples) < 10:
            return {
                'status': 'insufficient_data',
                'message': f'Insufficient data for analysis. Only {len(filtered_samples)} samples available.'
            }
        
        # Split samples by flag value
        true_samples = [s for s in filtered_samples if s['flag_value']]
        false_samples = [s for s in filtered_samples if not s['flag_value']]
        
        if len(true_samples) < 5 or len(false_samples) < 5:
            return {
                'status': 'insufficient_variation',
                'message': f'Insufficient variation in flag values. {len(true_samples)} true samples and {len(false_samples)} false samples.'
            }
        
        # Analyze each requested metric
        results = {
            'status': 'success',
            'sample_count': len(filtered_samples),
            'true_sample_count': len(true_samples),
            'false_sample_count': len(false_samples),
            'metrics': {}
        }
        
        for metric_key in metric_keys:
            # Extract metric values
            true_values = [s['metrics'].get(metric_key) for s in true_samples if metric_key in s['metrics']]
            false_values = [s['metrics'].get(metric_key) for s in false_samples if metric_key in s['metrics']]
            
            # Filter out None values
            true_values = [v for v in true_values if v is not None]
            false_values = [v for v in false_values if v is not None]
            
            if not true_values or not false_values:
                results['metrics'][metric_key] = {
                    'status': 'missing_data',
                    'message': f'Missing data for metric {metric_key}'
                }
                continue
            
            # Calculate statistics
            try:
                true_mean = np.mean(true_values)
                false_mean = np.mean(false_values)
                true_std = np.std(true_values)
                false_std = np.std(false_values)
                
                # Calculate percent difference
                if false_mean != 0:
                    percent_diff = ((true_mean - false_mean) / abs(false_mean)) * 100
                else:
                    percent_diff = np.nan
                
                # Simple statistical significance check
                import scipy.stats as stats
                t_stat, p_value = stats.ttest_ind(true_values, false_values, equal_var=False)
                significant = p_value < 0.05
                
                results['metrics'][metric_key] = {
                    'true_mean': true_mean,
                    'false_mean': false_mean,
                    'true_std': true_std,
                    'false_std': false_std,
                    'percent_difference': percent_diff,
                    'p_value': p_value,
                    'statistically_significant': significant
                }
            except Exception as e:
                results['metrics'][metric_key] = {
                    'status': 'analysis_error',
                    'message': f'Error analyzing metric {metric_key}: {str(e)}'
                }
        
        return results
    
    def get_usage_statistics(self, flag_name: Optional[str] = None, 
                            period: timedelta = timedelta(days=7)
                           ) -> Dict[str, Any]:
        """
        Get usage statistics for feature flags.
        
        Args:
            flag_name: Optional specific flag to get statistics for
            period: Time period to analyze
            
        Returns:
            Dictionary with usage statistics
        """
        # If specific flag requested, filter for just that flag
        flag_names = [flag_name] if flag_name else self._get_all_flag_names()
        
        results = {}
        for name in flag_names:
            true_key = f"{name}:true"
            false_key = f"{name}:false"
            
            true_count = self.evaluations.get(true_key, 0)
            false_count = self.evaluations.get(false_key, 0)
            total = true_count + false_count
            
            if total == 0:
                continue
                
            results[name] = {
                'total_evaluations': total,
                'true_count': true_count,
                'false_count': false_count,
                'true_percentage': (true_count / total) * 100 if total > 0 else 0,
                'segments': dict(self.user_segments.get(name, {}))
            }
            
        return results
    
    def _get_all_flag_names(self) -> List[str]:
        """Extract all flag names from the evaluations dict"""
        flag_names = set()
        for key in self.evaluations.keys():
            parts = key.split(':')
            if len(parts) == 2:
                flag_names.add(parts[0])
        return list(flag_names)
    
    def generate_report(self, output_format: str = 'json') -> Union[Dict[str, Any], str]:
        """
        Generate a comprehensive report on feature flag usage and impact.
        
        Args:
            output_format: Format of the report ('json' or 'html')
            
        Returns:
            Report in the requested format
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'flags': self.get_usage_statistics(),
            'experiments': self.experiments
        }
        
        # Add impact analysis for each flag with sufficient data
        for flag_name in self._get_all_flag_names():
            if flag_name in self.performance_samples and len(self.performance_samples[flag_name]) > 20:
                # Determine which metrics to analyze based on available data
                available_metrics = set()
                for sample in self.performance_samples[flag_name][:100]:  # Sample from first 100
                    available_metrics.update(sample.get('metrics', {}).keys())
                
                if available_metrics:
                    report.setdefault('impact_analysis', {})[flag_name] = self.analyze_flag_impact(
                        flag_name, list(available_metrics)
                    )
        
        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{self.storage_path}/reports/report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        if output_format == 'html':
            return self._generate_html_report(report)
        
        return report
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """
        Generate an HTML report from the report data.
        
        Args:
            report_data: Report data dictionary
            
        Returns:
            HTML string
        """
        # Simple HTML report template
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '  <title>Feature Flag Metrics Report</title>',
            '  <style>',
            '    body { font-family: Arial, sans-serif; margin: 20px; }',
            '    .card { border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 4px; }',
            '    .positive { color: green; }',
            '    .negative { color: red; }',
            '    .neutral { color: orange; }',
            '    table { border-collapse: collapse; width: 100%; }',
            '    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            '    th { background-color: #f2f2f2; }',
            '  </style>',
            '</head>',
            '<body>',
            f'  <h1>Feature Flag Metrics Report</h1>',
            f'  <p>Generated at: {report_data["generated_at"]}</p>',
            '  <h2>Flag Usage Summary</h2>'
        ]
        
        # Add flag usage tables
        if report_data.get('flags'):
            html_parts.append('  <table>')
            html_parts.append('    <tr><th>Flag Name</th><th>Total Evaluations</th><th>True %</th><th>False %</th></tr>')
            
            for flag_name, stats in report_data['flags'].items():
                true_pct = f"{stats['true_percentage']:.1f}%"
                false_pct = f"{100 - stats['true_percentage']:.1f}%"
                
                html_parts.append('    <tr>')
                html_parts.append(f'      <td>{flag_name}</td>')
                html_parts.append(f'      <td>{stats["total_evaluations"]}</td>')
                html_parts.append(f'      <td>{true_pct}</td>')
                html_parts.append(f'      <td>{false_pct}</td>')
                html_parts.append('    </tr>')
                
            html_parts.append('  </table>')
        else:
            html_parts.append('  <p>No flag usage data available.</p>')
            
        # Add impact analysis section
        if 'impact_analysis' in report_data:
            html_parts.append('  <h2>Impact Analysis</h2>')
            
            for flag_name, analysis in report_data['impact_analysis'].items():
                html_parts.append(f'  <div class="card">')
                html_parts.append(f'    <h3>Impact of {flag_name}</h3>')
                
                if analysis['status'] != 'success':
                    html_parts.append(f'    <p>{analysis["message"]}</p>')
                    html_parts.append('  </div>')
                    continue
                
                html_parts.append(f'    <p>Based on {analysis["sample_count"]} samples ({analysis["true_sample_count"]} true, {analysis["false_sample_count"]} false)</p>')
                
                # Add metrics table
                html_parts.append('    <table>')
                html_parts.append('      <tr><th>Metric</th><th>Control (false)</th><th>Treatment (true)</th><th>Diff %</th><th>Significant?</th></tr>')
                
                for metric_name, metric_data in analysis['metrics'].items():
                    if 'status' in metric_data and metric_data['status'] != 'success':
                        row = f'      <tr><td>{metric_name}</td><td colspan="4">{metric_data["message"]}</td></tr>'
                        html_parts.append(row)
                        continue
                    
                    diff_pct = metric_data.get('percent_difference', float('nan'))
                    diff_class = 'neutral'
                    
                    if not np.isnan(diff_pct):
                        diff_class = 'positive' if diff_pct > 0 else 'negative'
                        diff_formatted = f"{diff_pct:.2f}%"
                    else:
                        diff_formatted = "N/A"
                        
                    significant = "Yes" if metric_data.get('statistically_significant', False) else "No"
                    
                    row = [
                        f'      <tr>',
                        f'        <td>{metric_name}</td>',
                        f'        <td>{metric_data["false_mean"]:.4f}</td>',
                        f'        <td>{metric_data["true_mean"]:.4f}</td>',
                        f'        <td class="{diff_class}">{diff_formatted}</td>',
                        f'        <td>{significant}</td>',
                        f'      </tr>'
                    ]
                    html_parts.extend(row)
                
                html_parts.append('    </table>')
                html_parts.append('  </div>')
        
        # Add experiments section if available
        if report_data.get('experiments'):
            html_parts.append('  <h2>Experiments</h2>')
            
            for exp_name, exp_data in report_data['experiments'].items():
                html_parts.append(f'  <div class="card">')
                html_parts.append(f'    <h3>Experiment: {exp_name}</h3>')
                html_parts.append(f'    <p>Status: {exp_data.get("status", "Unknown")}</p>')
                html_parts.append(f'    <p>Start Date: {exp_data.get("start_date", "Unknown")}</p>')
                
                if 'results' in exp_data:
                    html_parts.append('    <h4>Results</h4>')
                    html_parts.append('    <table>')
                    html_parts.append('      <tr><th>Metric</th><th>Control</th><th>Treatment</th><th>Lift</th><th>Significant</th></tr>')
                    
                    for metric, result in exp_data['results'].items():
                        html_parts.append('      <tr>')
                        html_parts.append(f'        <td>{metric}</td>')
                        html_parts.append(f'        <td>{result.get("control", "N/A")}</td>')
                        html_parts.append(f'        <td>{result.get("treatment", "N/A")}</td>')
                        
                        lift = result.get('lift')
                        lift_class = 'neutral'
                        if lift is not None:
                            lift_class = 'positive' if lift > 0 else 'negative'
                            lift_str = f"{lift:.2f}%"
                        else:
                            lift_str = "N/A"
                            
                        html_parts.append(f'        <td class="{lift_class}">{lift_str}</td>')
                        html_parts.append(f'        <td>{result.get("significant", "No")}</td>')
                        html_parts.append('      </tr>')
                        
                    html_parts.append('    </table>')
                
                html_parts.append('  </div>')
        
        # Close HTML
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        # Save HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = f"{self.storage_path}/reports/report_{timestamp}.html"
        
        with open(html_path, 'w') as f:
            f.write('\n'.join(html_parts))
        
        logger.info(f"Generated HTML report at {html_path}")
        return '\n'.join(html_parts) 