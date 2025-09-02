#!/usr/bin/env python3
"""
BensBot Dashboard Integration

Integrates meta-learning visualizations into the main BensBot dashboard.
"""

import os
import sys
import json
import logging
import datetime
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import components
from meta_learning_db import MetaLearningDB
from market_regime_detector import MarketRegimeDetector
from benbot_api import BenBotAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'benbot_dashboard_integration.log')),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger('benbot_dashboard_integration')

class BenBotDashboardIntegration:
    """
    Integrates meta-learning visualizations with BensBot dashboard.
    
    This class:
    1. Generates visualizations based on meta-learning data
    2. Delivers them to BensBot's dashboard API
    3. Syncs real-time market regime data with BensBot
    4. Creates comparative charts between meta-learning guided and traditional evolution
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dashboard integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Setup paths
        self.meta_db_path = self.config.get('meta_db_path', os.path.join(project_root, 'meta_learning', 'meta_db.sqlite'))
        
        # Initialize components
        try:
            self.meta_db = MetaLearningDB(db_path=self.meta_db_path)
            self.regime_detector = MarketRegimeDetector()
            
            # BensBot API for dashboard integration
            self.benbot_api = BenBotAPI(
                api_endpoint=self.config.get('benbot_api_endpoint', 'http://localhost:8080/benbot/api'),
                api_key=self.config.get('benbot_api_key', ''),
                test_mode=self.config.get('test_mode', False)
            )
            
            logger.info("BensBot dashboard integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard integration: {e}")
            raise
        
        # Initialize cache for visualizations
        self.viz_cache = {}
    
    def generate_meta_learning_widgets(self) -> Dict[str, Any]:
        """
        Generate widgets for the BensBot dashboard.
        
        Returns:
            Dictionary of visualization widgets
        """
        try:
            logger.info("Generating meta-learning dashboard widgets")
            
            widgets = {
                'current_regime': self._generate_regime_widget(),
                'strategy_performance': self._generate_strategy_performance_widget(),
                'meta_learning_impact': self._generate_meta_learning_impact_widget(),
                'indicator_effectiveness': self._generate_indicator_effectiveness_widget()
            }
            
            logger.info(f"Generated {len(widgets)} dashboard widgets")
            return widgets
        except Exception as e:
            logger.error(f"Failed to generate dashboard widgets: {e}")
            return {}
    
    def update_benbot_dashboard(self) -> bool:
        """
        Update the BensBot dashboard with meta-learning visualizations.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Updating BensBot dashboard with meta-learning insights")
            
            # Generate widgets
            widgets = self.generate_meta_learning_widgets()
            
            if not widgets:
                logger.warning("No widgets generated")
                return False
            
            # Update dashboard through BensBot API
            for widget_name, widget_data in widgets.items():
                success = self.benbot_api.update_dashboard_widget(
                    widget_name=f"meta_learning_{widget_name}",
                    widget_data=widget_data,
                    section="Meta-Learning Insights"
                )
                
                if not success:
                    logger.warning(f"Failed to update dashboard widget: {widget_name}")
            
            logger.info("Updated BensBot dashboard with meta-learning insights")
            return True
        except Exception as e:
            logger.error(f"Failed to update BensBot dashboard: {e}")
            return False
    
    def sync_market_regime_data(self, price_data: pd.DataFrame = None) -> bool:
        """
        Sync current market regime data with BensBot.
        
        Args:
            price_data: Recent price data (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Syncing market regime data with BensBot")
            
            # Get price data if not provided
            if price_data is None:
                symbol = self.config.get('default_symbol', 'EURUSD')
                timeframe = self.config.get('default_timeframe', '1h')
                
                price_data = self.benbot_api.get_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=100
                )
                
                if price_data is None or len(price_data) < 30:
                    logger.warning("Insufficient price data for regime detection")
                    return False
                
                # Convert to DataFrame if needed
                if not isinstance(price_data, pd.DataFrame):
                    price_data = pd.DataFrame(price_data)
            
            # Detect current regime
            regime_info = self.regime_detector.detect_regime(price_data)
            current_regime = regime_info['regime']
            confidence = regime_info['confidence']
            
            # Get regime history
            regime_history = self.regime_detector.regime_history(price_data, lookback_periods=20)
            
            # Format for BensBot
            regime_data = {
                'current_regime': current_regime,
                'confidence': confidence,
                'metrics': regime_info.get('metrics', {}),
                'history': [
                    {
                        'date': entry['date'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(entry['date'], 'strftime') else entry['date'],
                        'regime': entry['regime'],
                        'confidence': entry['confidence']
                    }
                    for entry in regime_history
                ]
            }
            
            # Send to BensBot
            success = self.benbot_api.update_market_condition(regime_data)
            
            logger.info(f"Synced market regime data with BensBot: {current_regime} (confidence: {confidence:.2f})")
            return success
        except Exception as e:
            logger.error(f"Failed to sync market regime data: {e}")
            return False
    
    def create_comparative_visualization(self, 
                                        with_meta_learning: List[Dict[str, Any]],
                                        without_meta_learning: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create comparative visualization between strategies with and without meta-learning guidance.
        
        Args:
            with_meta_learning: Strategy results with meta-learning
            without_meta_learning: Strategy results without meta-learning
            
        Returns:
            Visualization data
        """
        try:
            logger.info("Creating comparative visualization")
            
            # Extract metrics for comparison
            metrics = ['win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown', 'consistency_score']
            
            with_meta = self._extract_metrics(with_meta_learning, metrics)
            without_meta = self._extract_metrics(without_meta_learning, metrics)
            
            # Calculate improvements
            improvements = {}
            
            for metric in metrics:
                if metric in with_meta and metric in without_meta and without_meta[metric] != 0:
                    improvements[metric] = (with_meta[metric] - without_meta[metric]) / without_meta[metric] * 100
                else:
                    improvements[metric] = 0
            
            # Format for visualization
            comparison_data = {
                'metrics': {
                    'with_meta_learning': with_meta,
                    'without_meta_learning': without_meta,
                    'improvement_percent': improvements
                },
                'chart_data': {
                    'metrics': list(with_meta.keys()),
                    'with_meta': list(with_meta.values()),
                    'without_meta': list(without_meta.values())
                },
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            logger.info("Created comparative visualization")
            return comparison_data
        except Exception as e:
            logger.error(f"Failed to create comparative visualization: {e}")
            return {}
    
    def _generate_regime_widget(self) -> Dict[str, Any]:
        """Generate current regime widget."""
        try:
            # Get recent regimes from meta-learning database
            recent_regimes = self.meta_db.get_recent_regimes(limit=30)
            
            if not recent_regimes:
                return {
                    'title': 'Current Market Regime',
                    'type': 'info',
                    'content': 'No regime data available'
                }
            
            # Get latest regime
            latest_regime = recent_regimes[0]
            regime = latest_regime['regime']
            confidence = latest_regime['confidence']
            
            # Determine regime characteristics
            regime_characteristics = {
                'bullish': {
                    'color': '#4CAF50',
                    'description': 'Strong uptrend with low volatility',
                    'strategy_types': ['Trend Following', 'Breakout']
                },
                'bearish': {
                    'color': '#F44336',
                    'description': 'Strong downtrend with low volatility',
                    'strategy_types': ['Trend Following', 'Breakout']
                },
                'volatile_bullish': {
                    'color': '#8BC34A',
                    'description': 'Uptrend with high volatility',
                    'strategy_types': ['Breakout', 'Multi Timeframe']
                },
                'volatile_bearish': {
                    'color': '#FF5722',
                    'description': 'Downtrend with high volatility',
                    'strategy_types': ['Breakout', 'Multi Timeframe']
                },
                'ranging': {
                    'color': '#2196F3',
                    'description': 'Sideways movement with low volatility',
                    'strategy_types': ['Mean Reversion', 'Pattern Recognition']
                },
                'choppy': {
                    'color': '#9C27B0',
                    'description': 'Sideways movement with high volatility',
                    'strategy_types': ['Mean Reversion', 'Pattern Recognition']
                }
            }
            
            characteristics = regime_characteristics.get(regime, {
                'color': '#9E9E9E',
                'description': 'Unknown market regime',
                'strategy_types': []
            })
            
            # Format regime history for chart
            history_data = []
            regime_counts = {}
            
            for entry in recent_regimes:
                r = entry['regime']
                history_data.append({
                    'date': entry['timestamp'],
                    'regime': r,
                    'confidence': entry['confidence']
                })
                
                # Count regimes
                regime_counts[r] = regime_counts.get(r, 0) + 1
            
            # Return widget data
            return {
                'title': 'Current Market Regime',
                'type': 'regime_status',
                'content': {
                    'current_regime': regime,
                    'confidence': confidence,
                    'color': characteristics['color'],
                    'description': characteristics['description'],
                    'recommended_strategies': characteristics['strategy_types'],
                    'history': history_data,
                    'distribution': [
                        {'regime': r, 'count': c} for r, c in regime_counts.items()
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate regime widget: {e}")
            return {
                'title': 'Current Market Regime',
                'type': 'error',
                'content': f'Error: {str(e)}'
            }
    
    def _generate_strategy_performance_widget(self) -> Dict[str, Any]:
        """Generate strategy performance widget."""
        try:
            # Get strategy type insights
            strategy_insights = self.meta_db.get_strategy_type_insights()
            
            if not strategy_insights:
                return {
                    'title': 'Strategy Performance by Regime',
                    'type': 'info',
                    'content': 'No strategy performance data available'
                }
            
            # Get regime insights
            regime_insights = self.meta_db.get_all_regime_insights()
            
            # Prepare data for heatmap
            strategy_types = []
            regimes = []
            performance_data = []
            
            for regime, insights in regime_insights.items():
                if 'strategy_type_performance' not in insights:
                    continue
                
                regimes.append(regime)
                
                for strategy_type, performance in insights['strategy_type_performance'].items():
                    if strategy_type not in strategy_types:
                        strategy_types.append(strategy_type)
                    
                    performance_data.append({
                        'regime': regime,
                        'strategy_type': strategy_type,
                        'win_rate': performance.get('win_rate', 0),
                        'profit_factor': performance.get('profit_factor', 0),
                        'avg_return': performance.get('avg_return', 0),
                        'confidence': performance.get('confidence', 0)
                    })
            
            # Return widget data
            return {
                'title': 'Strategy Performance by Regime',
                'type': 'heatmap',
                'content': {
                    'strategy_types': strategy_types,
                    'regimes': regimes,
                    'performance_data': performance_data
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate strategy performance widget: {e}")
            return {
                'title': 'Strategy Performance by Regime',
                'type': 'error',
                'content': f'Error: {str(e)}'
            }
    
    def _generate_meta_learning_impact_widget(self) -> Dict[str, Any]:
        """Generate meta-learning impact widget."""
        try:
            # Get evolution metrics
            evolution_metrics = self.meta_db.get_evolution_metrics()
            
            if not evolution_metrics:
                return {
                    'title': 'Meta-Learning Impact',
                    'type': 'info',
                    'content': 'No evolution metrics available'
                }
            
            # Extract with/without meta-learning metrics
            with_meta = []
            without_meta = []
            generations = []
            
            for metric in evolution_metrics:
                if metric.get('meta_learning_enabled', False):
                    with_meta.append({
                        'generation': metric.get('generation', 0),
                        'max_fitness': metric.get('max_fitness', 0),
                        'avg_fitness': metric.get('avg_fitness', 0),
                        'min_fitness': metric.get('min_fitness', 0)
                    })
                else:
                    without_meta.append({
                        'generation': metric.get('generation', 0),
                        'max_fitness': metric.get('max_fitness', 0),
                        'avg_fitness': metric.get('avg_fitness', 0),
                        'min_fitness': metric.get('min_fitness', 0)
                    })
                
                gen = metric.get('generation', 0)
                if gen not in generations:
                    generations.append(gen)
            
            # Calculate improvement percentages
            improvements = {}
            
            if with_meta and without_meta:
                with_meta_latest = max(with_meta, key=lambda x: x['generation'])
                without_meta_latest = max(without_meta, key=lambda x: x['generation'])
                
                for key in ['max_fitness', 'avg_fitness']:
                    if without_meta_latest[key] > 0:
                        improvements[key] = (with_meta_latest[key] - without_meta_latest[key]) / without_meta_latest[key] * 100
                    else:
                        improvements[key] = 0
            
            # Return widget data
            return {
                'title': 'Meta-Learning Impact',
                'type': 'evolution_comparison',
                'content': {
                    'with_meta_learning': with_meta,
                    'without_meta_learning': without_meta,
                    'generations': sorted(generations),
                    'improvements': improvements
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate meta-learning impact widget: {e}")
            return {
                'title': 'Meta-Learning Impact',
                'type': 'error',
                'content': f'Error: {str(e)}'
            }
    
    def _generate_indicator_effectiveness_widget(self) -> Dict[str, Any]:
        """Generate indicator effectiveness widget."""
        try:
            # Get indicator insights
            indicator_insights = self.meta_db.get_indicator_insights()
            
            if not indicator_insights:
                return {
                    'title': 'Indicator Effectiveness',
                    'type': 'info',
                    'content': 'No indicator effectiveness data available'
                }
            
            # Extract indicator effectiveness data
            indicators = []
            effectiveness = []
            win_rates = []
            
            for indicator, metrics in indicator_insights.items():
                indicators.append(indicator)
                effectiveness.append(metrics.get('effectiveness', 0))
                win_rates.append(metrics.get('win_rate', 0))
            
            # Return widget data
            return {
                'title': 'Indicator Effectiveness',
                'type': 'indicator_chart',
                'content': {
                    'indicators': indicators,
                    'effectiveness': effectiveness,
                    'win_rates': win_rates
                }
            }
        except Exception as e:
            logger.error(f"Failed to generate indicator effectiveness widget: {e}")
            return {
                'title': 'Indicator Effectiveness',
                'type': 'error',
                'content': f'Error: {str(e)}'
            }
    
    def _extract_metrics(self, strategies: List[Dict[str, Any]], metrics: List[str]) -> Dict[str, float]:
        """Extract average metrics from strategies."""
        result = {}
        
        for metric in metrics:
            values = []
            
            for strategy in strategies:
                value = None
                
                # Try different paths for metrics
                if metric in strategy:
                    value = strategy[metric]
                elif 'metrics' in strategy and metric in strategy['metrics']:
                    value = strategy['metrics'][metric]
                elif 'live_metrics' in strategy and metric in strategy['live_metrics']:
                    value = strategy['live_metrics'][metric]
                elif 'backtest_metrics' in strategy and metric in strategy['backtest_metrics']:
                    value = strategy['backtest_metrics'][metric]
                elif 'consistency' in strategy and metric in strategy['consistency']:
                    value = strategy['consistency'][metric]
                
                if value is not None and isinstance(value, (int, float)):
                    values.append(value)
            
            if values:
                result[metric] = sum(values) / len(values)
            else:
                result[metric] = 0
        
        return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BensBot Dashboard Integration")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--action", type=str, default="update", 
                        choices=["update", "sync_regime"],
                        help="Action to perform")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                if args.config.endswith('.json'):
                    config = json.load(f)
                elif args.config.endswith(('.yaml', '.yml')):
                    import yaml
                    config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
    
    # Create integration
    integration = BenBotDashboardIntegration(config)
    
    # Perform action
    if args.action == "update":
        integration.update_benbot_dashboard()
    elif args.action == "sync_regime":
        integration.sync_market_regime_data()
