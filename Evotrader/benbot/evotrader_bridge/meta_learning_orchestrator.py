#!/usr/bin/env python3
"""
Meta-Learning Orchestrator

Provides end-to-end integration between BensBot and EvoTrader's meta-learning system.
Coordinates data flow, meta-learning processes, and strategy deployment.
"""

import os
import sys
import json
import yaml
import logging
import argparse
import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import EvoTrader components
from meta_learning_db import MetaLearningDB
from market_regime_detector import MarketRegimeDetector
from strategy_pattern_analyzer import StrategyPatternAnalyzer
from meta_learning_integration import MetaLearningIntegrator
from prop_strategy_registry import PropStrategyRegistry
from live_score_updater import LiveScoreUpdater
from benbot_api import BenBotAPI
from evolution_feedback_integration import EvolutionFeedbackIntegrator

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'meta_learning_orchestrator.log')),
        logging.StreamHandler()
    ]
)

# Create logger
logger = logging.getLogger('meta_learning_orchestrator')

class MetaLearningOrchestrator:
    """
    Orchestrates the entire meta-learning workflow between BensBot and EvoTrader.
    
    Responsibilities:
    1. Coordinate data flow between systems
    2. Trigger meta-learning processes at appropriate times
    3. Apply meta-learning insights to evolution and strategy management
    4. Handle deployment of strategies based on meta-learning recommendations
    5. Provide monitoring and reporting of the meta-learning system
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the meta-learning orchestrator.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup paths
        self.meta_db_path = self.config.get('meta_db_path', os.path.join(project_root, 'meta_learning', 'meta_db.sqlite'))
        self.registry_path = self.config.get('registry_path', os.path.join(project_root, 'forex_prop_strategies.db'))
        
        # Initialize components
        try:
            # Core components
            self.meta_db = MetaLearningDB(db_path=self.meta_db_path)
            self.registry = PropStrategyRegistry(db_path=self.registry_path)
            self.regime_detector = MarketRegimeDetector()
            self.pattern_analyzer = StrategyPatternAnalyzer()
            self.meta_integrator = MetaLearningIntegrator(
                config_path=self.config.get('evolution_config_path'),
                meta_db_path=self.meta_db_path
            )
            
            # BensBot integration
            self.benbot_api = BenBotAPI(
                api_endpoint=self.config.get('benbot_api_endpoint', 'http://localhost:8080/benbot/api'),
                api_key=self.config.get('benbot_api_key', ''),
                test_mode=self.config.get('test_mode', False)
            )
            
            # Live feedback and evolution integration
            self.live_score_updater = LiveScoreUpdater(
                registry=self.registry,
                benbot_api=self.benbot_api,
                meta_db=self.meta_db
            )
            
            self.evolution_integrator = EvolutionFeedbackIntegrator(
                meta_db=self.meta_db,
                config_path=self.config.get('evolution_config_path')
            )
            
            logger.info("Meta-learning orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize meta-learning orchestrator: {e}")
            raise
    
    def run_complete_cycle(self):
        """
        Run a complete meta-learning cycle.
        
        1. Update live scores from BensBot
        2. Detect current market regime
        3. Analyze strategy patterns
        4. Update meta-learning database
        5. Generate optimized evolution configuration
        6. Deploy recommendations to BensBot
        """
        try:
            logger.info("Starting complete meta-learning cycle")
            
            # 1. Update live scores from BensBot
            self._update_live_scores()
            
            # 2. Detect current market regime
            current_regime = self._detect_market_regime()
            
            # 3. Analyze strategy patterns
            self._analyze_strategy_patterns()
            
            # 4. Update meta-learning insights
            self._update_meta_learning()
            
            # 5. Generate evolution configuration
            self._generate_evolution_config(current_regime)
            
            # 6. Deploy recommendations to BensBot
            self._deploy_recommendations()
            
            logger.info("Completed meta-learning cycle successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to complete meta-learning cycle: {e}")
            return False
    
    def run_targeted_analysis(self, analysis_type: str, **kwargs):
        """
        Run a specific analysis or update task.
        
        Args:
            analysis_type: Type of analysis to run ('regime', 'patterns', 'live_scores', etc.)
            **kwargs: Additional arguments for the analysis
        """
        try:
            logger.info(f"Running targeted analysis: {analysis_type}")
            
            if analysis_type == 'regime':
                return self._detect_market_regime()
            elif analysis_type == 'patterns':
                return self._analyze_strategy_patterns(**kwargs)
            elif analysis_type == 'live_scores':
                return self._update_live_scores(**kwargs)
            elif analysis_type == 'meta_learning':
                return self._update_meta_learning(**kwargs)
            elif analysis_type == 'evolution_config':
                regime = kwargs.get('regime') or self._detect_market_regime()
                return self._generate_evolution_config(regime, **kwargs)
            elif analysis_type == 'recommendations':
                return self._deploy_recommendations(**kwargs)
            else:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to run targeted analysis {analysis_type}: {e}")
            return None
    
    def _update_live_scores(self, strategy_ids: List[str] = None, **kwargs):
        """Update live scores from BensBot."""
        try:
            logger.info("Updating live scores from BensBot")
            
            # Get active strategies if not specified
            if not strategy_ids:
                strategies = self.registry.get_active_strategies()
                strategy_ids = [s['strategy_id'] for s in strategies]
            
            # Update scores
            updated_count = self.live_score_updater.update_scores(strategy_ids)
            
            # Generate recommendations
            recommendations = self.live_score_updater.generate_recommendations()
            
            logger.info(f"Updated live scores for {updated_count} strategies")
            return recommendations
        except Exception as e:
            logger.error(f"Failed to update live scores: {e}")
            return []
    
    def _detect_market_regime(self, **kwargs):
        """Detect current market regime."""
        try:
            logger.info("Detecting current market regime")
            
            # Get recent price data from BensBot
            symbol = kwargs.get('symbol', self.config.get('default_symbol', 'EURUSD'))
            timeframe = kwargs.get('timeframe', self.config.get('default_timeframe', '1h'))
            
            # Fetch data
            price_data = self.benbot_api.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=100
            )
            
            if price_data is None or len(price_data) < 30:
                logger.warning("Insufficient price data for regime detection")
                return 'unknown'
            
            # Convert to DataFrame if needed
            if not isinstance(price_data, pd.DataFrame):
                price_data = pd.DataFrame(price_data)
            
            # Detect regime
            regime_info = self.regime_detector.detect_regime(price_data)
            current_regime = regime_info['regime']
            confidence = regime_info['confidence']
            
            logger.info(f"Detected market regime: {current_regime} (confidence: {confidence:.2f})")
            
            # Update meta-learning database with regime info
            self.meta_db.log_market_regime(current_regime, confidence, price_data.iloc[-1].name)
            
            return current_regime
        except Exception as e:
            logger.error(f"Failed to detect market regime: {e}")
            return 'unknown'
    
    def _analyze_strategy_patterns(self, **kwargs):
        """Analyze strategy patterns."""
        try:
            logger.info("Analyzing strategy patterns")
            
            # Get strategies from registry
            promoted = self.registry.get_promotion_candidates(min_confidence=0.7)
            active = self.registry.get_active_strategies()
            demoted = self.registry.get_demotion_candidates(max_confidence=0.4)
            
            strategy_pool = promoted + active + demoted
            
            # Analyze patterns
            pattern_results = self.pattern_analyzer.analyze_strategy_pool(
                strategy_pool,
                performance_threshold=kwargs.get('performance_threshold', 0.7)
            )
            
            # Log insights to meta-learning database
            if 'indicator_patterns' in pattern_results:
                for indicator, stats in pattern_results['indicator_patterns'].get('relative_effectiveness', {}).items():
                    self.meta_db.update_indicator_effectiveness(
                        indicator,
                        effectiveness=stats.get('effectiveness', 1.0),
                        win_rate=stats.get('success_rate', 0.5)
                    )
            
            # Log parameter insights
            if 'parameter_patterns' in pattern_results:
                for strategy_type, params in pattern_results['parameter_patterns'].items():
                    for param_name, stats in params.items():
                        clusters = stats.get('clustered_values', [])
                        
                        for cluster in clusters:
                            self.meta_db.log_parameter_cluster(
                                strategy_type,
                                param_name,
                                center=cluster.get('center', 0),
                                min_value=cluster.get('range', [0, 0])[0],
                                max_value=cluster.get('range', [0, 0])[1],
                                count=cluster.get('count', 0)
                            )
            
            logger.info(f"Analyzed patterns for {len(strategy_pool)} strategies")
            return pattern_results
        except Exception as e:
            logger.error(f"Failed to analyze strategy patterns: {e}")
            return {}
    
    def _update_meta_learning(self, **kwargs):
        """Update meta-learning database with latest insights."""
        try:
            logger.info("Updating meta-learning database")
            
            # Get recent strategy results from registry
            strategies = self.registry.get_all_strategies(
                limit=kwargs.get('limit', 100),
                include_metrics=True
            )
            
            # Log strategy results to meta-learning database
            count = 0
            for strategy in strategies:
                # Only log strategies with both backtest and live metrics
                if 'backtest_metrics' in strategy and 'live_metrics' in strategy:
                    regime = strategy.get('market_regime', 'unknown')
                    
                    # Log to meta-learning database
                    self.meta_db.log_strategy_result(
                        strategy_id=strategy['strategy_id'],
                        strategy_type=strategy.get('strategy_type', 'unknown'),
                        parameters=strategy.get('parameters', {}),
                        indicators=strategy.get('indicators', []),
                        backtest_metrics=strategy.get('backtest_metrics', {}),
                        live_metrics=strategy.get('live_metrics', {}),
                        market_regime=regime,
                        status=strategy.get('status', 'unknown')
                    )
                    count += 1
            
            # Update aggregated meta-learning tables
            self.meta_db.update_aggregated_metrics()
            
            logger.info(f"Updated meta-learning database with {count} strategy results")
            return count
        except Exception as e:
            logger.error(f"Failed to update meta-learning database: {e}")
            return 0
    
    def _generate_evolution_config(self, current_regime: str, **kwargs):
        """Generate optimized evolution configuration based on meta-learning insights."""
        try:
            logger.info(f"Generating evolution configuration for regime: {current_regime}")
            
            # Get price data
            symbol = kwargs.get('symbol', self.config.get('default_symbol', 'EURUSD'))
            timeframe = kwargs.get('timeframe', self.config.get('default_timeframe', '1h'))
            
            price_data = self.benbot_api.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=100
            )
            
            if price_data is None or len(price_data) < 30:
                logger.warning("Insufficient price data for evolution configuration")
                # Fall back to default config
                return self.meta_integrator._get_default_config()
            
            # Convert to DataFrame if needed
            if not isinstance(price_data, pd.DataFrame):
                price_data = pd.DataFrame(price_data)
            
            # Generate evolution configuration
            config = self.meta_integrator.generate_evolution_config(
                price_data=price_data,
                registry=self.registry,
                strategy_types=kwargs.get('strategy_types'),
                min_confidence=kwargs.get('min_confidence', 0.6),
                apply_biasing=kwargs.get('apply_biasing', True)
            )
            
            # Save configuration
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            config_dir = os.path.join(project_root, 'configs', 'evolution')
            os.makedirs(config_dir, exist_ok=True)
            
            config_path = os.path.join(config_dir, f"evolution_config_{current_regime}_{timestamp}.yaml")
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            logger.info(f"Generated evolution configuration: {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to generate evolution configuration: {e}")
            # Fall back to default config
            return self.meta_integrator._get_default_config()
    
    def _deploy_recommendations(self, **kwargs):
        """Deploy recommendations to BensBot."""
        try:
            logger.info("Deploying recommendations to BensBot")
            
            # Get promotion and demotion candidates
            promotion_candidates = kwargs.get('promotion_candidates') or self.registry.get_promotion_candidates(
                min_confidence=kwargs.get('min_confidence', 0.7),
                min_trades=kwargs.get('min_trades', 20)
            )
            
            demotion_candidates = kwargs.get('demotion_candidates') or self.registry.get_demotion_candidates(
                max_confidence=kwargs.get('max_confidence', 0.4),
                min_trades=kwargs.get('min_trades', 10)
            )
            
            # Deploy to BensBot
            promoted = []
            demoted = []
            
            # Promote strategies
            for strategy in promotion_candidates:
                strategy_id = strategy['strategy_id']
                
                # Update status in BensBot
                success = self.benbot_api.update_strategy_status(
                    strategy_id=strategy_id,
                    status='active',
                    notes=f"Promoted based on meta-learning confidence score: {strategy.get('confidence_score', 0):.2f}"
                )
                
                if success:
                    # Update status in registry
                    self.registry.update_strategy_status(strategy_id, 'promoted')
                    promoted.append(strategy_id)
            
            # Demote strategies
            for strategy in demotion_candidates:
                strategy_id = strategy['strategy_id']
                
                # Update status in BensBot
                success = self.benbot_api.update_strategy_status(
                    strategy_id=strategy_id,
                    status='inactive',
                    notes=f"Demoted based on meta-learning confidence score: {strategy.get('confidence_score', 0):.2f}"
                )
                
                if success:
                    # Update status in registry
                    self.registry.update_strategy_status(strategy_id, 'demoted')
                    demoted.append(strategy_id)
            
            logger.info(f"Deployed recommendations: {len(promoted)} promoted, {len(demoted)} demoted")
            
            return {
                'promoted': promoted,
                'demoted': demoted
            }
        except Exception as e:
            logger.error(f"Failed to deploy recommendations: {e}")
            return {
                'promoted': [],
                'demoted': []
            }
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            'meta_db_path': os.path.join(project_root, 'meta_learning', 'meta_db.sqlite'),
            'registry_path': os.path.join(project_root, 'forex_prop_strategies.db'),
            'evolution_config_path': os.path.join(project_root, 'configs', 'evolution_base_config.yaml'),
            'benbot_api_endpoint': 'http://localhost:8080/benbot/api',
            'benbot_api_key': '',
            'test_mode': False,
            'default_symbol': 'EURUSD',
            'default_timeframe': '1h',
            'schedule': {
                'full_cycle': '0 0 * * *',  # Daily at midnight
                'live_scores': '0 */4 * * *',  # Every 4 hours
                'regime_detection': '0 */1 * * *',  # Every hour
                'meta_learning_update': '0 */8 * * *'  # Every 8 hours
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        config = yaml.safe_load(f)
                    elif config_path.endswith('.json'):
                        config = json.load(f)
                    else:
                        logger.warning(f"Unknown config format: {config_path}")
                        config = {}
                
                # Merge with default config
                for key, value in config.items():
                    default_config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
        
        return default_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-Learning Orchestrator")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--action", type=str, default="full_cycle", 
                        choices=["full_cycle", "live_scores", "regime", "patterns", "meta_learning", 
                                "evolution_config", "recommendations"],
                        help="Action to perform")
    parser.add_argument("--symbol", type=str, help="Symbol to analyze")
    parser.add_argument("--timeframe", type=str, help="Timeframe to analyze")
    
    args = parser.parse_args()
    
    # Create orchestrator
    try:
        orchestrator = MetaLearningOrchestrator(config_path=args.config)
        
        # Run requested action
        if args.action == "full_cycle":
            orchestrator.run_complete_cycle()
        else:
            kwargs = {}
            if args.symbol:
                kwargs['symbol'] = args.symbol
            if args.timeframe:
                kwargs['timeframe'] = args.timeframe
                
            orchestrator.run_targeted_analysis(args.action, **kwargs)
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        sys.exit(1)
