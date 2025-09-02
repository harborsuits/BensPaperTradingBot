#!/usr/bin/env python3
"""
Meta-Learning Test Suite

Comprehensive testing suite for the meta-learning system components.
"""

import os
import sys
import json
import unittest
import sqlite3
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import components to test
from meta_learning_db import MetaLearningDB
from market_regime_detector import MarketRegimeDetector
from strategy_pattern_analyzer import StrategyPatternAnalyzer
from meta_learning_integration import MetaLearningIntegrator
from benbot.evotrader_bridge.meta_learning_orchestrator import MetaLearningOrchestrator


class TestMetaLearningDB(unittest.TestCase):
    """Test the meta-learning database."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite').name
        self.meta_db = MetaLearningDB(db_path=self.temp_db)
    
    def tearDown(self):
        """Clean up test environment."""
        # Close database connection
        if hasattr(self.meta_db, 'conn') and self.meta_db.conn:
            self.meta_db.conn.close()
        
        # Remove temp database
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
    
    def test_database_initialization(self):
        """Test database initialization."""
        # Check if tables exist
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Check required tables
        required_tables = [
            'strategy_results', 
            'market_regimes', 
            'indicator_performance',
            'parameter_clusters',
            'strategy_type_performance',
            'evolution_metrics'
        ]
        
        for table in required_tables:
            self.assertIn(table, tables, f"Table {table} should exist in the database")
        
        conn.close()
    
    def test_log_strategy_result(self):
        """Test logging strategy results."""
        # Test data
        strategy_id = "test_strategy_001"
        strategy_type = "trend_following"
        parameters = {"rsi_period": 14, "ema_period": 20}
        indicators = ["rsi", "ema"]
        backtest_metrics = {"win_rate": 0.65, "profit_factor": 1.8}
        live_metrics = {"win_rate": 0.62, "profit_factor": 1.7}
        market_regime = "bullish"
        
        # Log strategy result
        result = self.meta_db.log_strategy_result(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            parameters=parameters,
            indicators=indicators,
            backtest_metrics=backtest_metrics,
            live_metrics=live_metrics,
            market_regime=market_regime
        )
        
        # Check result
        self.assertTrue(result, "Logging strategy result should succeed")
        
        # Verify it was logged
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT strategy_id FROM strategy_results WHERE strategy_id = ?", (strategy_id,))
        self.assertIsNotNone(cursor.fetchone(), "Strategy result should be in the database")
        
        conn.close()
    
    def test_update_aggregated_metrics(self):
        """Test updating aggregated metrics."""
        # Log some strategy results first
        strategy_types = ["trend_following", "mean_reversion"]
        regimes = ["bullish", "bearish"]
        
        for i in range(10):
            strategy_id = f"test_strategy_{i:03d}"
            strategy_type = strategy_types[i % len(strategy_types)]
            regime = regimes[i % len(regimes)]
            
            self.meta_db.log_strategy_result(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                parameters={"param1": i, "param2": i*2},
                indicators=["rsi", "ema"] if i % 2 == 0 else ["macd", "bollinger"],
                backtest_metrics={"win_rate": 0.6 + i/100, "profit_factor": 1.5 + i/10},
                live_metrics={"win_rate": 0.55 + i/100, "profit_factor": 1.4 + i/10},
                market_regime=regime
            )
        
        # Update aggregated metrics
        result = self.meta_db.update_aggregated_metrics()
        
        # Check result
        self.assertTrue(result, "Updating aggregated metrics should succeed")
        
        # Verify strategy type performance was updated
        conn = sqlite3.connect(self.temp_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM strategy_type_performance")
        count = cursor.fetchone()[0]
        self.assertGreater(count, 0, "Strategy type performance table should have entries")
        
        # Verify indicator performance was updated
        cursor.execute("SELECT COUNT(*) FROM indicator_performance")
        count = cursor.fetchone()[0]
        self.assertGreater(count, 0, "Indicator performance table should have entries")
        
        conn.close()
    
    def test_get_regime_insights(self):
        """Test getting regime insights."""
        # Log some strategy results first
        regime = "bullish"
        
        for i in range(5):
            strategy_id = f"test_strategy_{i:03d}"
            
            self.meta_db.log_strategy_result(
                strategy_id=strategy_id,
                strategy_type="trend_following",
                parameters={"param1": i, "param2": i*2},
                indicators=["rsi", "ema"],
                backtest_metrics={"win_rate": 0.6 + i/100, "profit_factor": 1.5 + i/10},
                live_metrics={"win_rate": 0.55 + i/100, "profit_factor": 1.4 + i/10},
                market_regime=regime
            )
        
        # Update aggregated metrics
        self.meta_db.update_aggregated_metrics()
        
        # Get regime insights
        insights = self.meta_db.get_regime_insights(regime)
        
        # Check result
        self.assertIsInstance(insights, dict, "Regime insights should be a dictionary")
        self.assertIn("strategy_type_performance", insights, "Insights should include strategy type performance")
        self.assertIn("indicator_performance", insights, "Insights should include indicator performance")


class TestMarketRegimeDetector(unittest.TestCase):
    """Test the market regime detector."""
    
    def setUp(self):
        """Set up test environment."""
        self.detector = MarketRegimeDetector()
        
        # Create sample price data
        dates = pd.date_range('2025-01-01', periods=100)
        
        # Bullish trend
        self.bullish_data = pd.DataFrame({
            'open': np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
            'high': np.linspace(102, 153, 100) + np.random.normal(0, 2, 100),
            'low': np.linspace(98, 147, 100) + np.random.normal(0, 2, 100),
            'close': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # Bearish trend
        self.bearish_data = pd.DataFrame({
            'open': np.linspace(150, 100, 100) + np.random.normal(0, 2, 100),
            'high': np.linspace(153, 102, 100) + np.random.normal(0, 2, 100),
            'low': np.linspace(147, 98, 100) + np.random.normal(0, 2, 100),
            'close': np.linspace(150, 100, 100) + np.random.normal(0, 1, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # Ranging market
        self.ranging_data = pd.DataFrame({
            'open': np.ones(100) * 125 + np.random.normal(0, 3, 100),
            'high': np.ones(100) * 128 + np.random.normal(0, 3, 100),
            'low': np.ones(100) * 122 + np.random.normal(0, 3, 100),
            'close': np.ones(100) * 125 + np.random.normal(0, 2, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
    
    def test_detect_bullish_regime(self):
        """Test detecting bullish regime."""
        result = self.detector.detect_regime(self.bullish_data)
        
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn("regime", result, "Result should include regime")
        self.assertIn("confidence", result, "Result should include confidence")
        
        # Check if detected as bullish or volatile_bullish
        self.assertIn(result.get("regime"), [self.detector.REGIME_BULLISH, self.detector.REGIME_VOLATILE_BULLISH], 
                     "Should detect bullish or volatile bullish regime")
    
    def test_detect_bearish_regime(self):
        """Test detecting bearish regime."""
        result = self.detector.detect_regime(self.bearish_data)
        
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn("regime", result, "Result should include regime")
        self.assertIn("confidence", result, "Result should include confidence")
        
        # Check if detected as bearish or volatile_bearish
        self.assertIn(result.get("regime"), [self.detector.REGIME_BEARISH, self.detector.REGIME_VOLATILE_BEARISH], 
                     "Should detect bearish or volatile bearish regime")
    
    def test_detect_ranging_regime(self):
        """Test detecting ranging regime."""
        result = self.detector.detect_regime(self.ranging_data)
        
        self.assertIsInstance(result, dict, "Result should be a dictionary")
        self.assertIn("regime", result, "Result should include regime")
        self.assertIn("confidence", result, "Result should include confidence")
        
        # Check if detected as ranging or choppy
        self.assertIn(result.get("regime"), [self.detector.REGIME_RANGING, self.detector.REGIME_CHOPPY], 
                     "Should detect ranging or choppy regime")
    
    def test_regime_history(self):
        """Test getting regime history."""
        history = self.detector.regime_history(self.bullish_data, lookback_periods=5)
        
        self.assertIsInstance(history, list, "History should be a list")
        self.assertEqual(len(history), 5, "History should have 5 entries")
        
        for entry in history:
            self.assertIn("regime", entry, "Entry should include regime")
            self.assertIn("confidence", entry, "Entry should include confidence")
            self.assertIn("date", entry, "Entry should include date")


class TestStrategyPatternAnalyzer(unittest.TestCase):
    """Test the strategy pattern analyzer."""
    
    def setUp(self):
        """Set up test environment."""
        self.analyzer = StrategyPatternAnalyzer()
        
        # Create test strategies
        self.test_strategies = [
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
    
    def test_analyze_strategy_pool(self):
        """Test analyzing strategy pool."""
        results = self.analyzer.analyze_strategy_pool(self.test_strategies, performance_threshold=0.7)
        
        self.assertIsInstance(results, dict, "Results should be a dictionary")
        self.assertIn("indicator_patterns", results, "Results should include indicator patterns")
        self.assertIn("parameter_patterns", results, "Results should include parameter patterns")
        self.assertIn("regime_sensitivity", results, "Results should include regime sensitivity")
    
    def test_extract_indicator_patterns(self):
        """Test extracting indicator patterns."""
        patterns = self.analyzer.extract_indicator_patterns(self.test_strategies[:3], [self.test_strategies[3]])
        
        self.assertIsInstance(patterns, dict, "Patterns should be a dictionary")
        self.assertIn("indicator_frequency", patterns, "Patterns should include indicator frequency")
        self.assertIn("indicator_combinations", patterns, "Patterns should include indicator combinations")
        self.assertIn("relative_effectiveness", patterns, "Patterns should include relative effectiveness")
    
    def test_extract_parameter_patterns(self):
        """Test extracting parameter patterns."""
        patterns = self.analyzer.extract_parameter_patterns(self.test_strategies[:3])
        
        self.assertIsInstance(patterns, dict, "Patterns should be a dictionary")
        self.assertIn("trend_following", patterns, "Patterns should include trend following strategy type")
        
        # Check parameter stats
        trend_following = patterns.get("trend_following", {})
        self.assertIn("macd_fast", trend_following, "Should include macd_fast parameter")
        
        # Check stats structure
        macd_stats = trend_following.get("macd_fast", {})
        self.assertIn("min", macd_stats, "Stats should include min")
        self.assertIn("max", macd_stats, "Stats should include max")
        self.assertIn("mean", macd_stats, "Stats should include mean")
    
    def test_extract_regime_sensitivity(self):
        """Test extracting regime sensitivity."""
        sensitivity = self.analyzer.extract_regime_sensitivity(self.test_strategies)
        
        self.assertIsInstance(sensitivity, dict, "Sensitivity should be a dictionary")
        self.assertIn("regime_strategy_counts", sensitivity, "Sensitivity should include regime strategy counts")
        self.assertIn("regime_avg_performance", sensitivity, "Sensitivity should include regime average performance")


class TestMetaLearningIntegration(unittest.TestCase):
    """Test the meta-learning integration."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp files
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.sqlite').name
        self.temp_config = tempfile.NamedTemporaryFile(suffix='.yaml').name
        
        # Create sample config
        with open(self.temp_config, 'w') as f:
            f.write("""
population:
  size: 100
  initial_diversity: 0.8
  strategy_type_weights:
    trend_following: 1.0
    mean_reversion: 1.0

evolution:
  generations: 20
  selection_percent: 0.3
  mutation_rate: 0.2

indicators:
  macd: {weight: 1.0}
  rsi: {weight: 1.0}
  bollinger_bands: {weight: 1.0}

parameters:
  rsi_period: {min: 7, max: 21, default: 14}
  macd_fast: {min: 8, max: 16, default: 12}
  macd_slow: {min: 18, max: 30, default: 26}
            """)
        
        # Initialize integrator
        self.integrator = MetaLearningIntegrator(
            config_path=self.temp_config,
            meta_db_path=self.temp_db
        )
        
        # Create sample price data
        dates = pd.date_range('2025-01-01', periods=100)
        self.price_data = pd.DataFrame({
            'open': np.linspace(100, 150, 100) + np.random.normal(0, 2, 100),
            'high': np.linspace(102, 153, 100) + np.random.normal(0, 2, 100),
            'low': np.linspace(98, 147, 100) + np.random.normal(0, 2, 100),
            'close': np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temp files
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)
        
        if os.path.exists(self.temp_config):
            os.remove(self.temp_config)
    
    def test_generate_evolution_config(self):
        """Test generating evolution config."""
        config = self.integrator.generate_evolution_config(
            price_data=self.price_data,
            apply_biasing=False  # Disable biasing for this test
        )
        
        self.assertIsInstance(config, dict, "Config should be a dictionary")
        self.assertIn("population", config, "Config should include population")
        self.assertIn("evolution", config, "Config should include evolution")
        self.assertIn("indicators", config, "Config should include indicators")
        self.assertIn("parameters", config, "Config should include parameters")
    
    def test_apply_biases_to_config(self):
        """Test applying biases to config."""
        # Create a test bias config
        bias_config = {
            'strategy_type_weights': {
                'trend_following': 2.0,
                'mean_reversion': 0.5
            },
            'indicator_weights': {
                'rsi': 1.5,
                'macd': 0.8
            },
            'parameter_biases': {
                'rsi_period': {
                    'center': 12,
                    'range': [10, 14]
                }
            },
            'regime': 'bullish'
        }
        
        # Get base config
        base_config = self.integrator._get_default_config()
        
        # Apply biases
        biased_config = self.integrator._apply_biases_to_config(base_config, bias_config)
        
        self.assertIsInstance(biased_config, dict, "Biased config should be a dictionary")
        
        # Check if biases were applied
        self.assertEqual(biased_config.get('market_regime'), 'bullish',
                         "Market regime should be set to 'bullish'")
        
        # Check strategy weights
        population = biased_config.get('population', {})
        weights = population.get('strategy_type_weights', {})
        self.assertEqual(weights.get('trend_following'), 2.0,
                         "trend_following weight should be 2.0")
        
        # Check indicator weights
        indicators = biased_config.get('indicators', {})
        self.assertEqual(indicators.get('rsi', {}).get('weight'), 1.5,
                         "rsi weight should be 1.5")


class TestMetaLearningOrchestrator(unittest.TestCase):
    """Test the meta-learning orchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temp directory for tests
        self.test_dir = tempfile.mkdtemp()
        
        # Create necessary subdirectories
        os.makedirs(os.path.join(self.test_dir, 'meta_learning'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'configs'), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, 'logs'), exist_ok=True)
        
        # Create config file
        self.config_path = os.path.join(self.test_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            f.write("""
meta_db_path: {}/meta_learning/meta_db.sqlite
registry_path: {}/test_registry.db
test_mode: true
default_symbol: EURUSD
default_timeframe: 1h
            """.format(self.test_dir, self.test_dir))
        
        # Create empty registry database
        self.registry_path = os.path.join(self.test_dir, 'test_registry.db')
        conn = sqlite3.connect(self.registry_path)
        cursor = conn.cursor()
        
        # Create basic tables for testing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                strategy_id TEXT PRIMARY KEY,
                strategy_type TEXT,
                status TEXT,
                created_at TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS live_metrics (
                strategy_id TEXT PRIMARY KEY,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                updated_at TEXT
            )
        """)
        
        # Insert some test data
        cursor.execute("""
            INSERT INTO strategies (strategy_id, strategy_type, status, created_at)
            VALUES 
                ('test_strategy_001', 'trend_following', 'active', '2025-01-01'),
                ('test_strategy_002', 'mean_reversion', 'active', '2025-01-02')
        """)
        
        cursor.execute("""
            INSERT INTO live_metrics (strategy_id, win_rate, profit_factor, sharpe_ratio, updated_at)
            VALUES 
                ('test_strategy_001', 0.65, 1.8, 1.2, '2025-01-10'),
                ('test_strategy_002', 0.58, 1.5, 0.9, '2025-01-10')
        """)
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temp directory and contents
        import shutil
        shutil.rmtree(self.test_dir)
    
    @patch('benbot_api.BenBotAPI')
    @patch('prop_strategy_registry.PropStrategyRegistry')
    @patch('market_regime_detector.MarketRegimeDetector')
    @patch('strategy_pattern_analyzer.StrategyPatternAnalyzer')
    @patch('meta_learning_db.MetaLearningDB')
    @patch('meta_learning_integration.MetaLearningIntegrator')
    def test_orchestrator_initialization(self, mock_integrator, mock_meta_db, mock_analyzer, 
                                         mock_detector, mock_registry, mock_benbot):
        """Test orchestrator initialization."""
        # Configure mocks
        mock_registry.return_value = MagicMock()
        mock_benbot.return_value = MagicMock()
        mock_detector.return_value = MagicMock()
        mock_analyzer.return_value = MagicMock()
        mock_meta_db.return_value = MagicMock()
        mock_integrator.return_value = MagicMock()
        
        # Initialize orchestrator
        orchestrator = MetaLearningOrchestrator(config_path=self.config_path)
        
        # Check if components were initialized
        self.assertIsNotNone(orchestrator.meta_db, "Meta DB should be initialized")
        self.assertIsNotNone(orchestrator.registry, "Registry should be initialized")
        self.assertIsNotNone(orchestrator.regime_detector, "Regime detector should be initialized")
        self.assertIsNotNone(orchestrator.pattern_analyzer, "Pattern analyzer should be initialized")
        self.assertIsNotNone(orchestrator.meta_integrator, "Meta integrator should be initialized")
        self.assertIsNotNone(orchestrator.benbot_api, "BenBot API should be initialized")
    
    @patch('benbot_api.BenBotAPI')
    @patch('prop_strategy_registry.PropStrategyRegistry')
    @patch('market_regime_detector.MarketRegimeDetector')
    @patch('strategy_pattern_analyzer.StrategyPatternAnalyzer')
    @patch('meta_learning_db.MetaLearningDB')
    @patch('meta_learning_integration.MetaLearningIntegrator')
    def test_run_targeted_analysis(self, mock_integrator, mock_meta_db, mock_analyzer, 
                                   mock_detector, mock_registry, mock_benbot):
        """Test running targeted analysis."""
        # Configure mocks
        mock_registry.return_value = MagicMock()
        mock_benbot.return_value = MagicMock()
        mock_detector.return_value = MagicMock()
        mock_analyzer.return_value = MagicMock()
        mock_meta_db.return_value = MagicMock()
        mock_integrator.return_value = MagicMock()
        
        # Configure detector mock to return a regime
        mock_detector.return_value.detect_regime.return_value = {
            'regime': 'bullish',
            'confidence': 0.8
        }
        
        # Initialize orchestrator
        orchestrator = MetaLearningOrchestrator(config_path=self.config_path)
        
        # Run targeted analysis
        result = orchestrator.run_targeted_analysis('regime')
        
        # Check if detector was called
        mock_detector.return_value.detect_regime.assert_called_once()
        self.assertEqual(result, 'bullish', "Result should be 'bullish'")


if __name__ == '__main__':
    unittest.main()
