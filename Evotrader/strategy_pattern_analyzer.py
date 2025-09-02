#!/usr/bin/env python3
"""
Strategy Pattern Analyzer

Analyzes strategy characteristics to identify successful patterns across market regimes.
"""

import os
import json
import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import Counter, defaultdict
from itertools import combinations
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('strategy_pattern_analyzer')

class StrategyPatternAnalyzer:
    """
    Analyzes strategies to extract patterns from successful and unsuccessful strategies.
    """
    
    def __init__(self, output_dir="./reports/pattern_analysis"):
        """
        Initialize the strategy pattern analyzer.
        
        Args:
            output_dir: Directory for reports and visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_strategy_pool(self, 
                             strategies: List[Dict[str, Any]], 
                             performance_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Analyze a pool of strategies to identify common patterns.
        
        Args:
            strategies: List of strategies with performance data
            performance_threshold: Threshold for considering a strategy successful
            
        Returns:
            Dictionary with pattern information
        """
        if not strategies:
            logger.warning("No strategies provided for analysis")
            return {}
        
        # Separate successful from unsuccessful strategies
        successful_strategies = []
        unsuccessful_strategies = []
        
        for strategy in strategies:
            # Check confidence score or other success metric
            confidence_score = self._get_confidence_score(strategy)
            status = strategy.get('status', '').lower()
            
            if confidence_score >= performance_threshold and status in ['promoted', 'live']:
                successful_strategies.append(strategy)
            elif confidence_score < 0.5 or status in ['demoted', 'rejected', 'terminated']:
                unsuccessful_strategies.append(strategy)
        
        logger.info(f"Analyzing {len(successful_strategies)} successful and {len(unsuccessful_strategies)} unsuccessful strategies")
        
        # Extract patterns
        indicator_patterns = self.extract_indicator_patterns(successful_strategies, unsuccessful_strategies)
        parameter_patterns = self.extract_parameter_patterns(successful_strategies)
        regime_sensitivity = self.extract_regime_sensitivity(successful_strategies)
        
        # Create pattern report
        analysis_results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'successful_count': len(successful_strategies),
            'unsuccessful_count': len(unsuccessful_strategies),
            'total_analyzed': len(strategies),
            'performance_threshold': performance_threshold,
            'indicator_patterns': indicator_patterns,
            'parameter_patterns': parameter_patterns,
            'regime_sensitivity': regime_sensitivity
        }
        
        # Generate report files
        self.generate_pattern_report(analysis_results)
        
        return analysis_results
    
    def extract_indicator_patterns(self, 
                                  successful_strategies: List[Dict[str, Any]], 
                                  unsuccessful_strategies: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract common indicator combinations from successful strategies.
        
        Args:
            successful_strategies: List of successful strategies
            unsuccessful_strategies: Optional list of unsuccessful strategies
            
        Returns:
            Dictionary with indicator pattern information
        """
        if not successful_strategies:
            return {}
        
        # Count individual indicators
        indicator_counts = Counter()
        
        # Count indicator combinations (pairs)
        combination_counts = Counter()
        
        # Strategy type to indicator mapping
        strategy_type_indicators = defaultdict(list)
        
        # Process successful strategies
        for strategy in successful_strategies:
            strategy_type = strategy.get('strategy_type', strategy.get('type', 'unknown'))
            indicators = self._get_indicators(strategy)
            
            # Skip if no indicators
            if not indicators:
                continue
            
            # Count individual indicators
            indicator_counts.update(indicators)
            
            # Count combinations (pairs of indicators)
            for combo in combinations(sorted(indicators), 2):
                combination_counts[combo] += 1
            
            # Add to strategy type mapping
            strategy_type_indicators[strategy_type].extend(indicators)
        
        # Calculate effectiveness against unsuccessful strategies (if available)
        relative_effectiveness = {}
        if unsuccessful_strategies:
            unsuccessful_indicator_counts = Counter()
            
            for strategy in unsuccessful_strategies:
                indicators = self._get_indicators(strategy)
                unsuccessful_indicator_counts.update(indicators)
            
            # Calculate relative effectiveness
            total_successful = len(successful_strategies)
            total_unsuccessful = len(unsuccessful_strategies)
            
            if total_successful > 0 and total_unsuccessful > 0:
                for indicator, count in indicator_counts.items():
                    # Success rate
                    success_rate = count / total_successful
                    
                    # Failure rate
                    failure_rate = unsuccessful_indicator_counts.get(indicator, 0) / total_unsuccessful
                    
                    # Relative effectiveness (higher is better)
                    if failure_rate > 0:
                        effectiveness = success_rate / failure_rate
                    else:
                        effectiveness = success_rate * 2  # Arbitrary boost for indicators with no failures
                    
                    relative_effectiveness[indicator] = {
                        'success_rate': success_rate,
                        'failure_rate': failure_rate,
                        'effectiveness': effectiveness
                    }
        
        # Popular indicators by strategy type
        popular_by_type = {}
        for strategy_type, indicators in strategy_type_indicators.items():
            counter = Counter(indicators)
            popular_by_type[strategy_type] = {
                'indicators': dict(counter.most_common(5)),
                'total_strategies': len(set(s.get('strategy_id') for s in successful_strategies 
                                          if s.get('strategy_type', s.get('type')) == strategy_type))
            }
        
        return {
            'indicator_frequency': dict(indicator_counts.most_common()),
            'indicator_combinations': {str(k): v for k, v in combination_counts.most_common(10)},
            'relative_effectiveness': relative_effectiveness,
            'popular_by_strategy_type': popular_by_type
        }
    
    def extract_parameter_patterns(self, successful_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract parameter ranges that consistently work well.
        
        Args:
            successful_strategies: List of successful strategies
            
        Returns:
            Dictionary with parameter pattern information
        """
        if not successful_strategies:
            return {}
        
        # Group by strategy type
        strategies_by_type = defaultdict(list)
        
        for strategy in successful_strategies:
            strategy_type = strategy.get('strategy_type', strategy.get('type', 'unknown'))
            strategies_by_type[strategy_type].append(strategy)
        
        # Analyze parameters for each strategy type
        parameter_patterns = {}
        
        for strategy_type, strategies in strategies_by_type.items():
            if len(strategies) < 2:
                continue  # Need at least 2 strategies to find patterns
            
            # Collect parameter values
            parameter_values = defaultdict(list)
            
            for strategy in strategies:
                parameters = self._get_parameters(strategy)
                
                if not parameters:
                    continue
                
                for param_name, param_value in parameters.items():
                    if isinstance(param_value, (int, float)):
                        parameter_values[param_name].append(param_value)
            
            # Calculate statistics for each parameter
            param_stats = {}
            
            for param_name, values in parameter_values.items():
                if len(values) < 2:
                    continue
                
                param_stats[param_name] = {
                    'min': float(min(values)),
                    'max': float(max(values)),
                    'mean': float(np.mean(values)),
                    'median': float(np.median(values)),
                    'std': float(np.std(values)),
                    'count': len(values),
                    'clustered_values': self._cluster_parameter_values(values)
                }
            
            parameter_patterns[strategy_type] = param_stats
        
        return parameter_patterns
    
    def extract_regime_sensitivity(self, successful_strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze which strategies work best in which market regimes.
        
        Args:
            successful_strategies: List of successful strategies
            
        Returns:
            Dictionary with regime sensitivity information
        """
        if not successful_strategies:
            return {}
        
        # Count strategies by market regime and strategy type
        regime_counts = defaultdict(lambda: defaultdict(int))
        regime_performance = defaultdict(lambda: defaultdict(list))
        
        for strategy in successful_strategies:
            strategy_type = strategy.get('strategy_type', strategy.get('type', 'unknown'))
            market_regime = strategy.get('market_regime', 'unknown')
            
            if market_regime == 'unknown':
                # Try to get from market conditions
                market_conditions = strategy.get('market_conditions', {})
                market_regime = market_conditions.get('regime', 'unknown')
            
            if market_regime == 'unknown':
                continue
            
            # Count by type and regime
            regime_counts[market_regime][strategy_type] += 1
            
            # Track performance metrics
            confidence = self._get_confidence_score(strategy)
            performance = strategy.get('consistency', strategy.get('consistency_metrics', {}))
            
            if confidence > 0 and performance:
                # Track by strategy type within regime
                regime_performance[market_regime][strategy_type].append({
                    'confidence': confidence,
                    'return_delta': performance.get('return_delta', 0),
                    'sharpe_delta': performance.get('sharpe_delta', 0),
                    'drawdown_delta': performance.get('drawdown_delta', 0)
                })
        
        # Calculate average performance by regime and strategy type
        regime_avg_performance = {}
        
        for regime, type_performances in regime_performance.items():
            regime_avg_performance[regime] = {}
            
            for strategy_type, performances in type_performances.items():
                if not performances:
                    continue
                
                avg_confidence = np.mean([p['confidence'] for p in performances])
                avg_return_delta = np.mean([p['return_delta'] for p in performances])
                avg_sharpe_delta = np.mean([p['sharpe_delta'] for p in performances])
                avg_drawdown_delta = np.mean([p['drawdown_delta'] for p in performances])
                
                regime_avg_performance[regime][strategy_type] = {
                    'avg_confidence': float(avg_confidence),
                    'avg_return_delta': float(avg_return_delta),
                    'avg_sharpe_delta': float(avg_sharpe_delta),
                    'avg_drawdown_delta': float(avg_drawdown_delta),
                    'count': len(performances)
                }
        
        return {
            'regime_strategy_counts': {k: dict(v) for k, v in regime_counts.items()},
            'regime_avg_performance': regime_avg_performance
        }
    
    def generate_pattern_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive report of identified patterns.
        
        Args:
            analysis_results: Results from pattern analysis
            
        Returns:
            Path to the generated report
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"pattern_analysis_{timestamp}.json")
        
        # Save analysis results
        with open(report_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Generate markdown report
        markdown_file = os.path.join(self.output_dir, f"pattern_analysis_{timestamp}.md")
        
        with open(markdown_file, 'w') as f:
            f.write("# Strategy Pattern Analysis Report\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"- Analyzed {analysis_results['total_analyzed']} strategies\n")
            f.write(f"- Successful strategies: {analysis_results['successful_count']}\n")
            f.write(f"- Unsuccessful strategies: {analysis_results['unsuccessful_count']}\n")
            f.write(f"- Performance threshold: {analysis_results['performance_threshold']}\n\n")
            
            # Indicator patterns
            f.write("## Indicator Patterns\n\n")
            
            if analysis_results.get('indicator_patterns', {}).get('indicator_frequency'):
                f.write("### Top Indicators\n\n")
                f.write("| Indicator | Frequency |\n")
                f.write("|-----------|----------|\n")
                
                for indicator, count in list(analysis_results['indicator_patterns']['indicator_frequency'].items())[:10]:
                    f.write(f"| {indicator} | {count} |\n")
                
                f.write("\n")
            
            if analysis_results.get('indicator_patterns', {}).get('indicator_combinations'):
                f.write("### Top Indicator Combinations\n\n")
                f.write("| Combination | Frequency |\n")
                f.write("|------------|----------|\n")
                
                for combo, count in analysis_results['indicator_patterns']['indicator_combinations'].items():
                    f.write(f"| {combo} | {count} |\n")
                
                f.write("\n")
            
            # Parameter patterns
            f.write("## Parameter Patterns\n\n")
            
            param_patterns = analysis_results.get('parameter_patterns', {})
            
            for strategy_type, params in param_patterns.items():
                f.write(f"### {strategy_type}\n\n")
                f.write("| Parameter | Min | Max | Mean | Median | Std Dev | Count |\n")
                f.write("|-----------|-----|-----|------|--------|---------|-------|\n")
                
                for param_name, stats in params.items():
                    f.write(f"| {param_name} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['std']:.2f} | {stats['count']} |\n")
                
                f.write("\n")
            
            # Regime sensitivity
            f.write("## Regime Sensitivity\n\n")
            
            regime_counts = analysis_results.get('regime_sensitivity', {}).get('regime_strategy_counts', {})
            
            if regime_counts:
                f.write("### Strategy Type Distribution by Market Regime\n\n")
                f.write("| Regime | Strategy Types |\n")
                f.write("|--------|---------------|\n")
                
                for regime, type_counts in regime_counts.items():
                    types_str = ", ".join([f"{t}: {c}" for t, c in type_counts.items()])
                    f.write(f"| {regime} | {types_str} |\n")
                
                f.write("\n")
            
            # Generate performance heatmap
            self._generate_performance_heatmap(analysis_results)
            
            # Add path to visualization
            f.write("## Visualizations\n\n")
            f.write(f"![Strategy Performance Heatmap](pattern_performance_heatmap_{timestamp}.png)\n\n")
        
        logger.info(f"Generated pattern analysis report at {markdown_file}")
        return markdown_file
    
    def _generate_performance_heatmap(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a heatmap of strategy performance by market regime.
        
        Args:
            analysis_results: Results from pattern analysis
            
        Returns:
            Path to the generated visualization
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        image_file = os.path.join(self.output_dir, f"pattern_performance_heatmap_{timestamp}.png")
        
        try:
            # Extract performance data
            regime_perf = analysis_results.get('regime_sensitivity', {}).get('regime_avg_performance', {})
            
            if not regime_perf:
                logger.warning("No regime performance data available for visualization")
                return ""
            
            # Collect strategy types and regimes
            strategy_types = set()
            regimes = set()
            
            for regime, type_performances in regime_perf.items():
                regimes.add(regime)
                for strategy_type in type_performances.keys():
                    strategy_types.add(strategy_type)
            
            strategy_types = sorted(strategy_types)
            regimes = sorted(regimes)
            
            # Create performance matrix
            performance_matrix = np.zeros((len(strategy_types), len(regimes)))
            
            for i, strategy_type in enumerate(strategy_types):
                for j, regime in enumerate(regimes):
                    if regime in regime_perf and strategy_type in regime_perf[regime]:
                        performance_matrix[i, j] = regime_perf[regime][strategy_type]['avg_confidence']
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(performance_matrix, cmap='viridis', aspect='auto')
            
            # Add labels
            plt.xticks(np.arange(len(regimes)), regimes, rotation=45, ha='right')
            plt.yticks(np.arange(len(strategy_types)), strategy_types)
            
            plt.colorbar(label='Confidence Score')
            plt.title('Strategy Type Performance by Market Regime')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(image_file)
            plt.close()
            
            logger.info(f"Generated performance heatmap at {image_file}")
            return image_file
        except Exception as e:
            logger.error(f"Error generating performance heatmap: {e}")
            return ""
    
    def _get_confidence_score(self, strategy: Dict[str, Any]) -> float:
        """Extract confidence score from strategy."""
        # Try different possible paths to confidence score
        consistency = strategy.get('consistency', strategy.get('consistency_metrics', {}))
        
        if isinstance(consistency, dict) and 'confidence_score' in consistency:
            return consistency['confidence_score']
        
        # Try direct path
        if 'confidence_score' in strategy:
            return strategy['confidence_score']
        
        # Default
        return 0.0
    
    def _get_indicators(self, strategy: Dict[str, Any]) -> List[str]:
        """Extract indicators from strategy."""
        # Try different paths to indicators
        indicators = strategy.get('indicators', [])
        
        # Check if indicators is a JSON string
        if isinstance(indicators, str):
            try:
                indicators = json.loads(indicators)
            except:
                pass
        
        # Try getting from parameters
        if not indicators:
            parameters = self._get_parameters(strategy)
            if 'indicators' in parameters and isinstance(parameters['indicators'], list):
                indicators = parameters['indicators']
        
        return indicators if isinstance(indicators, list) else []
    
    def _get_parameters(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters from strategy."""
        # Try different paths to parameters
        parameters = strategy.get('parameters', {})
        
        # Check if parameters is a JSON string
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except:
                parameters = {}
        
        return parameters if isinstance(parameters, dict) else {}
    
    def _cluster_parameter_values(self, values: List[float], min_cluster_size: int = 2) -> List[Dict[str, Any]]:
        """
        Find clusters of parameter values.
        
        Args:
            values: List of parameter values
            min_cluster_size: Minimum number of values to form a cluster
            
        Returns:
            List of cluster information
        """
        if len(values) < min_cluster_size:
            return []
        
        # Simple clustering - group values that are close to each other
        sorted_values = sorted(values)
        clusters = []
        current_cluster = [sorted_values[0]]
        
        for i in range(1, len(sorted_values)):
            # If value is close to last value in cluster, add to cluster
            if sorted_values[i] - current_cluster[-1] < 0.05 * abs(current_cluster[-1]):
                current_cluster.append(sorted_values[i])
            else:
                # Start a new cluster if current one is big enough
                if len(current_cluster) >= min_cluster_size:
                    clusters.append({
                        'center': float(np.mean(current_cluster)),
                        'range': [float(min(current_cluster)), float(max(current_cluster))],
                        'count': len(current_cluster)
                    })
                
                current_cluster = [sorted_values[i]]
        
        # Add the last cluster if it's big enough
        if len(current_cluster) >= min_cluster_size:
            clusters.append({
                'center': float(np.mean(current_cluster)),
                'range': [float(min(current_cluster)), float(max(current_cluster))],
                'count': len(current_cluster)
            })
        
        return clusters


if __name__ == "__main__":
    # Test with some sample data
    analyzer = StrategyPatternAnalyzer()
    
    # Create some sample strategies
    sample_strategies = [
        {
            'strategy_id': 'strategy_001',
            'strategy_type': 'trend_following',
            'indicators': ['macd', 'rsi', 'bollinger_bands'],
            'parameters': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26, 'bollinger_period': 20},
            'market_regime': 'bullish',
            'consistency': {'confidence_score': 0.85},
            'status': 'promoted'
        },
        {
            'strategy_id': 'strategy_002',
            'strategy_type': 'mean_reversion',
            'indicators': ['rsi', 'stochastic', 'bollinger_bands'],
            'parameters': {'rsi_period': 7, 'stoch_k': 14, 'stoch_d': 3, 'bollinger_period': 20},
            'market_regime': 'choppy',
            'consistency': {'confidence_score': 0.92},
            'status': 'live'
        },
        {
            'strategy_id': 'strategy_003',
            'strategy_type': 'trend_following',
            'indicators': ['ema', 'adx', 'macd'],
            'parameters': {'ema_period': 50, 'adx_period': 14, 'macd_fast': 8, 'macd_slow': 21},
            'market_regime': 'bullish',
            'consistency': {'confidence_score': 0.78},
            'status': 'promoted'
        },
        {
            'strategy_id': 'strategy_004',
            'strategy_type': 'breakout',
            'indicators': ['bollinger_bands', 'volume', 'atr'],
            'parameters': {'bollinger_period': 20, 'atr_period': 14},
            'market_regime': 'volatile_bullish',
            'consistency': {'confidence_score': 0.45},
            'status': 'rejected'
        }
    ]
    
    # Run analysis
    results = analyzer.analyze_strategy_pool(sample_strategies)
    print(f"Analysis complete. Report generated at: {analyzer.output_dir}")
