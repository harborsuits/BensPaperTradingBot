#!/usr/bin/env python3
"""
Meta-Learning Insights Publisher

Shares EvoTrader's meta-learning insights with BensBot without modifying either system.
Provides strategy selection guidance based on regime, historical performance, and pattern analysis.
"""

import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import EvoTrader components
from meta_learning_db import MetaLearningDB
from market_regime_detector import MarketRegimeDetector
from strategy_pattern_analyzer import StrategyPatternAnalyzer
from benbot_api import BenBotAPI
from benbot.evotrader_bridge.regime_sync import MarketRegimeSync

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('meta_insights_publisher')

class MetaInsightsPublisher:
    """
    Publishes EvoTrader's meta-learning insights to BensBot.
    
    This component:
    1. Extracts key insights from EvoTrader's meta-learning database
    2. Formats them for BensBot's consumption
    3. Publishes them via BensBot's API
    4. Provides strategy selection guidance based on current regime
    """
    
    def __init__(self, 
                meta_db_path: str = None,
                benbot_api_url: str = None,
                update_interval: int = 86400,  # Default 24 hours
                mock_mode: bool = False):
        """
        Initialize meta-insights publisher.
        
        Args:
            meta_db_path: Path to meta-learning database
            benbot_api_url: URL for BensBot API
            update_interval: Update interval in seconds
            mock_mode: Use mock data for testing
        """
        logger.info("Initializing meta-insights publisher")
        
        # Default paths
        if not meta_db_path:
            meta_db_path = os.path.join(project_root, 'meta_learning', 'meta_db.sqlite')
        
        # Initialize components
        self.meta_db = MetaLearningDB(db_path=meta_db_path)
        self.pattern_analyzer = StrategyPatternAnalyzer(meta_db=self.meta_db)
        self.regime_sync = MarketRegimeSync(
            benbot_api_url=benbot_api_url,
            mock_mode=mock_mode
        )
        self.benbot_api = BenBotAPI(base_url=benbot_api_url, mock=mock_mode)
        
        # Configuration
        self.update_interval = update_interval
        self.mock_mode = mock_mode
        
        # State
        self.last_update_time = None
        self.running = False
        self.insights_thread = None
        
        logger.info(f"Meta-insights publisher initialized with update interval: {update_interval}s")
    
    def extract_insights(self, current_regime: str = None) -> Dict[str, Any]:
        """
        Extract meta-learning insights.
        
        Args:
            current_regime: Current market regime
            
        Returns:
            Meta-learning insights
        """
        try:
            logger.info(f"Extracting meta-learning insights for regime: {current_regime}")
            
            # Default insights structure
            insights = {
                'timestamp': datetime.now().isoformat(),
                'regime': current_regime,
                'strategy_recommendations': [],
                'parameter_insights': {},
                'indicator_effectiveness': {},
                'regime_transition_probabilities': {},
                'strategy_combinations': [],
                'long_term_trends': {}
            }
            
            # Get current regime if not provided
            if not current_regime:
                regime_info = self.regime_sync.detect_and_push_regime()
                current_regime = regime_info.get('regime', 'unknown')
                insights['regime'] = current_regime
            
            # Top strategy types for current regime
            regime_insights = self.meta_db.get_regime_insights(current_regime)
            
            if regime_insights and 'strategy_type_performance' in regime_insights:
                strategy_performance = regime_insights['strategy_type_performance']
                
                # Sort by performance
                sorted_strategies = sorted(
                    strategy_performance.items(),
                    key=lambda x: x[1].get('mean_performance', 0),
                    reverse=True
                )
                
                # Add top strategies
                for strategy_type, performance in sorted_strategies[:5]:
                    insights['strategy_recommendations'].append({
                        'strategy_type': strategy_type,
                        'confidence': min(1.0, performance.get('mean_performance', 0) / 100),
                        'avg_sharpe': performance.get('mean_sharpe', 0),
                        'avg_trades_per_day': performance.get('trades_per_day', 0),
                        'sample_size': performance.get('count', 0)
                    })
            
            # Parameter insights for current regime
            if regime_insights and 'parameter_clusters' in regime_insights:
                for param, clusters in regime_insights['parameter_clusters'].items():
                    # Find best cluster
                    best_cluster = max(clusters, key=lambda c: c.get('performance', 0)) if clusters else None
                    if best_cluster:
                        insights['parameter_insights'][param] = {
                            'optimal_range': [best_cluster.get('min'), best_cluster.get('max')],
                            'central_value': best_cluster.get('center'),
                            'confidence': best_cluster.get('confidence', 0)
                        }
            
            # Indicator effectiveness
            indicator_insights = self.pattern_analyzer.get_indicator_effectiveness(regime=current_regime)
            
            if indicator_insights:
                # Format for easier consumption
                for indicator, data in indicator_insights.items():
                    insights['indicator_effectiveness'][indicator] = {
                        'effectiveness_score': data.get('effectiveness', 0),
                        'confidence': data.get('confidence', 0),
                        'signal_quality': data.get('signal_quality', 0),
                        'false_positive_rate': data.get('false_positive_rate', 0)
                    }
            
            # Regime transition probabilities
            regime_transitions = self.meta_db.get_regime_transitions()
            
            if regime_transitions:
                # Initialize with current regime
                transitions = {}
                
                # Extract transitions from current regime
                for from_regime, to_regimes in regime_transitions.items():
                    if from_regime == current_regime:
                        transitions = to_regimes
                        break
                
                # Format for easier consumption
                for to_regime, probability in transitions.items():
                    insights['regime_transition_probabilities'][to_regime] = probability
            
            # Strategy combinations (ensemble insights)
            ensemble_insights = self.pattern_analyzer.get_strategy_combinations(regime=current_regime)
            
            if ensemble_insights:
                for combo in ensemble_insights:
                    insights['strategy_combinations'].append({
                        'strategies': combo.get('strategies', []),
                        'synergy_score': combo.get('synergy_score', 0),
                        'correlation': combo.get('correlation', 0),
                        'combined_sharpe': combo.get('combined_sharpe', 0)
                    })
            
            # Long-term trends in performance
            long_term_trends = self.meta_db.get_performance_trends()
            
            if long_term_trends:
                insights['long_term_trends'] = long_term_trends
            
            logger.info(f"Successfully extracted meta-learning insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error extracting meta-learning insights: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'regime': current_regime or 'unknown',
                'error': str(e)
            }
    
    def publish_insights(self, insights: Dict[str, Any]) -> bool:
        """
        Publish insights to BensBot.
        
        Args:
            insights: Meta-learning insights
            
        Returns:
            Success status
        """
        try:
            logger.info("Publishing meta-learning insights to BensBot")
            
            # Publish to BensBot
            result = self.benbot_api.update_meta_learning_insights(insights)
            
            if result:
                logger.info("Successfully published insights to BensBot")
                return True
            else:
                logger.warning("Failed to publish insights to BensBot")
                return False
            
        except Exception as e:
            logger.error(f"Error publishing meta-learning insights: {e}")
            return False
    
    def get_strategy_recommendations(self, 
                                   regime: str = None, 
                                   count: int = 3) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations for current regime.
        
        Args:
            regime: Target market regime
            count: Number of recommendations to return
            
        Returns:
            List of strategy recommendations
        """
        try:
            logger.info(f"Getting strategy recommendations for regime: {regime}")
            
            # Get current regime if not provided
            if not regime:
                regime_info = self.regime_sync.detect_and_push_regime()
                regime = regime_info.get('regime', 'unknown')
            
            # Get regime insights
            regime_insights = self.meta_db.get_regime_insights(regime)
            
            recommendations = []
            
            if regime_insights and 'strategy_type_performance' in regime_insights:
                strategy_performance = regime_insights['strategy_type_performance']
                
                # Sort by performance
                sorted_strategies = sorted(
                    strategy_performance.items(),
                    key=lambda x: x[1].get('mean_performance', 0),
                    reverse=True
                )
                
                # Add top strategies
                for strategy_type, performance in sorted_strategies[:count]:
                    recommendations.append({
                        'strategy_type': strategy_type,
                        'confidence': min(1.0, performance.get('mean_performance', 0) / 100),
                        'avg_sharpe': performance.get('mean_sharpe', 0),
                        'avg_trades_per_day': performance.get('trades_per_day', 0),
                        'sample_size': performance.get('count', 0),
                        'regime': regime
                    })
            
            # If not enough recommendations, add defaults based on regime
            if len(recommendations) < count:
                # Default recommendations by regime
                default_recs = {
                    'bullish': ['trend_following', 'momentum', 'breakout'],
                    'bearish': ['trend_following', 'reversal', 'hedged_momentum'],
                    'ranging': ['mean_reversion', 'range_trading', 'swing_trading'],
                    'volatile': ['breakout', 'adaptive', 'paired_trading'],
                    'choppy': ['mean_reversion', 'channel_trading', 'volatility_trading'],
                    'unknown': ['adaptive', 'balanced', 'defensive']
                }
                
                # Find closest regime match
                matched_regime = 'unknown'
                for r in default_recs.keys():
                    if r in regime:
                        matched_regime = r
                        break
                
                # Add default recommendations
                existing_types = [r['strategy_type'] for r in recommendations]
                for strategy_type in default_recs.get(matched_regime, default_recs['unknown']):
                    if strategy_type not in existing_types and len(recommendations) < count:
                        recommendations.append({
                            'strategy_type': strategy_type,
                            'confidence': 0.5,  # Medium confidence
                            'avg_sharpe': None,
                            'avg_trades_per_day': None,
                            'sample_size': 0,
                            'regime': regime,
                            'note': 'Default recommendation based on regime'
                        })
            
            logger.info(f"Generated {len(recommendations)} strategy recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            return []
    
    def extract_and_publish(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Extract insights and publish them to BensBot.
        
        Args:
            force_update: Force update regardless of interval
            
        Returns:
            Published insights
        """
        try:
            # Check if update is needed
            current_time = datetime.now()
            
            if not force_update and self.last_update_time:
                elapsed = (current_time - self.last_update_time).total_seconds()
                if elapsed < self.update_interval:
                    logger.debug(f"Skipping update, {elapsed}s elapsed since last update (interval: {self.update_interval}s)")
                    return {}
            
            # Get current regime
            regime_info = self.regime_sync.detect_and_push_regime()
            current_regime = regime_info.get('regime', 'unknown')
            
            # Extract insights
            insights = self.extract_insights(current_regime=current_regime)
            
            # Publish insights
            if insights:
                success = self.publish_insights(insights)
                
                if success:
                    self.last_update_time = current_time
                    return insights
            
            return {}
            
        except Exception as e:
            logger.error(f"Error in extract_and_publish: {e}")
            return {}
    
    def start_insights_thread(self):
        """Start background insights publishing thread."""
        if self.running:
            logger.warning("Insights publisher already running, not starting new thread")
            return
        
        self.running = True
        self.insights_thread = threading.Thread(target=self._insights_loop)
        self.insights_thread.daemon = True
        self.insights_thread.start()
        
        logger.info("Started insights publishing thread")
    
    def stop_insights_thread(self):
        """Stop background insights publishing thread."""
        self.running = False
        
        if self.insights_thread:
            # Wait for thread to finish (with timeout)
            self.insights_thread.join(timeout=5)
            self.insights_thread = None
        
        logger.info("Stopped insights publishing thread")
    
    def _insights_loop(self):
        """Background insights publishing loop."""
        logger.info("Insights publishing loop started")
        
        while self.running:
            try:
                # Extract and publish insights
                self.extract_and_publish()
                
                # Sleep for interval
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in insights publishing loop: {e}")
                
                # Sleep for a shorter interval on error
                time.sleep(min(3600, self.update_interval / 4))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Learning Insights Publisher")
    parser.add_argument("--meta-db", type=str, help="Path to meta-learning database")
    parser.add_argument("--benbot-api", type=str, help="BensBot API URL")
    parser.add_argument("--interval", type=int, default=86400, help="Update interval in seconds")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for testing")
    parser.add_argument("--daemon", action="store_true", help="Run as a background daemon")
    parser.add_argument("--recommendations", action="store_true", help="Print strategy recommendations")
    
    args = parser.parse_args()
    
    # Create publisher
    publisher = MetaInsightsPublisher(
        meta_db_path=args.meta_db,
        benbot_api_url=args.benbot_api,
        update_interval=args.interval,
        mock_mode=args.mock
    )
    
    if args.daemon:
        # Run as background thread
        publisher.start_insights_thread()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            publisher.stop_insights_thread()
            print("Stopped insights publisher")
    elif args.recommendations:
        # Print strategy recommendations
        recommendations = publisher.get_strategy_recommendations()
        
        print("Strategy Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['strategy_type'].upper()}")
            print(f"   Confidence: {rec['confidence']:.2f}")
            
            if rec['avg_sharpe'] is not None:
                print(f"   Avg. Sharpe: {rec['avg_sharpe']:.2f}")
            
            if rec['avg_trades_per_day'] is not None:
                print(f"   Avg. Trades/Day: {rec['avg_trades_per_day']:.1f}")
            
            print(f"   Regime: {rec['regime']}")
            
            if 'note' in rec:
                print(f"   Note: {rec['note']}")
    else:
        # Run single extract and publish
        insights = publisher.extract_and_publish(force_update=True)
        
        if insights:
            print(f"Published insights for regime: {insights.get('regime', 'unknown')}")
            
            if 'strategy_recommendations' in insights:
                print("\nTop Strategy Recommendations:")
                for i, rec in enumerate(insights['strategy_recommendations'], 1):
                    print(f"{i}. {rec['strategy_type']} (Confidence: {rec['confidence']:.2f})")
            
            if 'parameter_insights' in insights:
                print("\nKey Parameter Insights:")
                for param, insight in list(insights['parameter_insights'].items())[:3]:
                    print(f"{param}: Optimal range {insight['optimal_range']}, center {insight['central_value']}")
        else:
            print("Failed to publish insights")
