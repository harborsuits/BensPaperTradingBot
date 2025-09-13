"""
Autonomous ML Backtesting System

This package provides a complete backtesting system that uses machine learning
to autonomously generate, test, and improve trading strategies based on
market data, news sentiment, and technical analysis.

Main Components:
- DataIntegrationLayer: Integrates news, market data, and technical indicators
- StrategyGenerator: Generates trading strategies based on ML analysis
- AutonomousBacktester: Runs backtests for ML-generated strategies
- MLStrategyOptimizer: Learns from backtest results to improve strategies
"""

from trading_bot.backtesting.data_integration import DataIntegrationLayer, SentimentAnalyzer
from trading_bot.backtesting.strategy_generator import (
    StrategyGenerator, MLStrategyModel, StrategyTemplateLibrary, RiskManager
)
from trading_bot.backtesting.autonomous_backtester import AutonomousBacktester, BacktestResultAnalyzer
from trading_bot.backtesting.ml_optimizer import MLStrategyOptimizer
from trading_bot.backtesting.api import initialize_ml_backtesting, register_ml_backtest_endpoints

__all__ = [
    'DataIntegrationLayer',
    'SentimentAnalyzer',
    'StrategyGenerator',
    'MLStrategyModel',
    'StrategyTemplateLibrary',
    'RiskManager',
    'AutonomousBacktester',
    'BacktestResultAnalyzer',
    'MLStrategyOptimizer',
    'initialize_ml_backtesting',
    'register_ml_backtest_endpoints'
] 