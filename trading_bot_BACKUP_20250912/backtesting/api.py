#!/usr/bin/env python3
"""
API endpoints for the Autonomous ML Backtesting system.

This module connects the backtesting system to the frontend, allowing
users to interact with the ML-powered backtesting features.
"""

import logging
import os
from typing import Dict, Any, Optional, Union

# Import typed settings
from trading_bot.config.typed_settings import APISettings, TradingBotSettings, BacktestSettings, load_config

# Try to import Flask, but handle the case where it's not installed
try:
    from flask import jsonify, request, Blueprint
    FLASK_AVAILABLE = True
except ImportError:
    print("Flask is not installed. ML Backtesting API features will be disabled.")
    FLASK_AVAILABLE = False
    # Create dummy objects to prevent import errors
    class DummyRequest:
        def __init__(self):
            self.json = {}
            self.args = {}
    
    class DummyBlueprint:
        def __init__(self, name, import_name):
            self.name = name
            self.import_name = import_name
        
        def route(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    
    request = DummyRequest()
    Blueprint = lambda name, import_name: DummyBlueprint(name, import_name)
    
    def jsonify(data, *args, **kwargs):
        return data

from trading_bot.backtesting.data_integration import DataIntegrationLayer, SentimentAnalyzer
from trading_bot.backtesting.strategy_generator import StrategyGenerator, MLStrategyModel, StrategyTemplateLibrary, RiskManager
from trading_bot.backtesting.autonomous_backtester import AutonomousBacktester, BacktestResultAnalyzer
from trading_bot.backtesting.ml_optimizer import MLStrategyOptimizer

logger = logging.getLogger(__name__)

# Load typed settings if available
api_settings = None
backtest_settings = None
try:
    config = load_config()
    api_settings = config.api
    backtest_settings = config.backtest
    logger.info("Loaded API and backtest settings from typed config")
except Exception as e:
    logger.warning(f"Could not load typed settings: {str(e)}. Using defaults.")
    api_settings = APISettings()
    backtest_settings = BacktestSettings()

# Global instances of the backtesting components
data_layer = None
strategy_generator = None
backtester = None
ml_optimizer = None

def initialize_ml_backtesting(news_fetcher):
    """
    Initialize the ML backtesting system components
    
    Args:
        news_fetcher: Existing NewsFetcher instance
    """
    global data_layer, strategy_generator, backtester, ml_optimizer
    
    logger.info("Initializing ML backtesting system")
    
    # Initialize components
    data_layer = DataIntegrationLayer(news_fetcher=news_fetcher)
    
    ml_model = MLStrategyModel()
    strategy_templates = StrategyTemplateLibrary()
    risk_manager = RiskManager()
    
    strategy_generator = StrategyGenerator(
        ml_model=ml_model,
        strategy_templates=strategy_templates,
        risk_manager=risk_manager
    )
    
    result_analyzer = BacktestResultAnalyzer()
    
    backtester = AutonomousBacktester(
        data_layer=data_layer,
        strategy_generator=strategy_generator,
        result_analyzer=result_analyzer
    )
    
    ml_optimizer = MLStrategyOptimizer()
    
    logger.info("ML backtesting system initialized")

def register_ml_backtest_endpoints(app):
    """
    Register API endpoints for ML backtesting
    
    Args:
        app: Flask application
    """
    # Configure app with typed settings if Flask is available
    if FLASK_AVAILABLE and hasattr(app, 'config'):
        if api_settings:
            app.config['RATE_LIMIT_REQUESTS'] = api_settings.rate_limit_requests
            app.config['RATE_LIMIT_PERIOD_SECONDS'] = api_settings.rate_limit_period_seconds
            app.config['REQUIRE_AUTH'] = api_settings.require_authentication
        
        if backtest_settings:
            app.config['DEFAULT_SYMBOLS'] = backtest_settings.default_symbols
            app.config['DEFAULT_START_DATE'] = backtest_settings.default_start_date
            app.config['DEFAULT_END_DATE'] = backtest_settings.default_end_date
            app.config['INITIAL_CAPITAL'] = backtest_settings.initial_capital
    @app.route('/api/autonomous-backtest', methods=['POST'])
    def run_autonomous_backtest():
        """Run a full autonomous backtesting cycle"""
        if not data_layer or not backtester:
            return jsonify({
                'success': False,
                'error': 'ML backtesting system not initialized'
            }), 500
        
        try:
            params = request.json or {}
            
            # Extract parameters with fallbacks to typed settings
            tickers = params.get('tickers')
            if params.get('ticker'):
                # Support single ticker parameter
                tickers = [params.get('ticker')]
            elif not tickers and backtest_settings and backtest_settings.default_symbols:
                # Fall back to default symbols from typed settings
                tickers = backtest_settings.default_symbols
                
            timeframes = params.get('timeframes')
            if params.get('timeframe') and params.get('timeframe') != 'all':
                # Support single timeframe parameter
                timeframes = [params.get('timeframe')]
                
            sectors = params.get('sectors')
            if params.get('sector') and params.get('sector') != 'all':
                # Support single sector parameter
                sectors = [params.get('sector')]
                
            logger.info(f"Running autonomous backtest with tickers: {tickers}, timeframes: {timeframes}, sectors: {sectors}")
            
            # Run the full autonomous cycle
            results = backtester.run_full_autonomous_cycle(
                tickers=tickers,
                timeframes=timeframes,
                sectors=sectors
            )
            
            # Learn from the results if enabled
            if params.get('enable_learning', True):
                learning_metrics = ml_optimizer.learn_from_results(results)
                # Add learning metrics to results
                results['learning_metrics'] = learning_metrics
                
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            logger.error(f"Error running autonomous backtest: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/api/ml-strategy-suggestions', methods=['GET'])
    def get_ml_strategy_suggestions():
        """Get ML-suggested strategies for current market conditions"""
        if not data_layer or not strategy_generator:
            return jsonify({
                'success': False,
                'error': 'ML backtesting system not initialized'
            }), 500
            
        try:
            # Get parameters
            ticker = request.args.get('ticker', 'SPY')
            
            # Get market data for analysis
            market_data = data_layer.get_comprehensive_data(ticker=ticker)
            
            # Generate strategies
            strategies = strategy_generator.generate_strategies(
                market_data,
                num_strategies=int(request.args.get('count', 3))
            )
            
            # Return suggested strategies with reasoning
            return jsonify({
                'success': True,
                'suggestions': strategies,
                'market_data_summary': {
                    'ticker': ticker,
                    'timestamp': market_data.get('timestamp').isoformat(),
                    'sentiment': market_data.get('sentiment', {}).get('overall_sentiment', 0),
                    'indicator_count': len(market_data.get('indicators', {}))
                }
            })
            
        except Exception as e:
            logger.error(f"Error getting ML strategy suggestions: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/api/ml-improve-strategy', methods=['POST'])
    def improve_strategy():
        """Suggest improvements for an existing strategy"""
        if not ml_optimizer:
            return jsonify({
                'success': False,
                'error': 'ML backtesting system not initialized'
            }), 500
            
        try:
            data = request.json
            
            if not data or 'strategy' not in data or 'backtest_result' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Missing required fields'
                }), 400
                
            # Get strategy and backtest result
            strategy = data.get('strategy', {})
            backtest_result = data.get('backtest_result', {})
            
            # Get improvement suggestions
            improvements = ml_optimizer.suggest_strategy_improvements(strategy, backtest_result)
            
            return jsonify({
                'success': True,
                'improvements': improvements
            })
            
        except Exception as e:
            logger.error(f"Error improving strategy: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    @app.route('/api/ml-model-insights', methods=['GET'])
    def get_ml_model_insights():
        """Get insights from the ML model"""
        if not ml_optimizer:
            return jsonify({
                'success': False,
                'error': 'ML backtesting system not initialized'
            }), 500
            
        try:
            # Return ML model insights
            return jsonify({
                'success': True,
                'model_version': ml_optimizer.model_version,
                'last_trained': ml_optimizer.last_trained.isoformat() if ml_optimizer.last_trained else None,
                'feature_importance': ml_optimizer.feature_importance,
                'strategy_performance': ml_optimizer.strategy_performance
            })
            
        except Exception as e:
            logger.error(f"Error getting ML model insights: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
            
    logger.info("ML backtesting API endpoints registered") 