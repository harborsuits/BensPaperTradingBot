"""
Machine learning components for the trading bot.

This module provides state-of-the-art machine learning models and utilities
for quantitative trading, focusing on ensemble methods, deep learning 
architectures, and model explainability.
"""

from trading_bot.ml.base_model import BaseMLModel
from trading_bot.ml.ensemble_model import EnsembleModel, EnsembleMethod
from trading_bot.ml.lstm_model import LSTMModel
from trading_bot.ml.model_factory import MLModelFactory

__all__ = [
    'BaseMLModel',
    'EnsembleModel',
    'EnsembleMethod',
    'LSTMModel',
    'MLModelFactory'
] 