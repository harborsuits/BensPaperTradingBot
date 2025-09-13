#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strategy Templates for Evolution

This module defines templates for evolving different types of trading strategies.
Each template specifies parameter ranges and constraints for the evolution process.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Indicator-based strategy templates
INDICATOR_STRATEGY_TEMPLATES = [
    {
        'template_id': 'ma_crossover',
        'base_name': 'MA_Crossover',
        'strategy_type': 'indicator',
        'description': 'Moving average crossover strategy with customizable fast and slow periods',
        'parameter_ranges': {
            'fast_period': [5, 50],
            'slow_period': [20, 200],
            'signal_filter': ['none', 'volume', 'volatility'],
            'trend_filter_enabled': {'type': 'boolean'},
            'exit_strategy': ['fixed', 'trailing', 'opposite', 'time'],
            'risk_factor': [0.5, 2.0]
        },
        'fixed_parameters': {
            'strategy_class': 'MAStrategy'
        }
    },
    {
        'template_id': 'bollinger_bands',
        'base_name': 'Bollinger_Strategy',
        'strategy_type': 'indicator',
        'description': 'Bollinger Bands mean reversion or breakout strategy',
        'parameter_ranges': {
            'bb_period': [10, 50],
            'bb_std_dev': [1.5, 3.0],
            'entry_type': ['mean_reversion', 'breakout'],
            'confirmation_indicator': ['none', 'rsi', 'stochastic', 'macd'],
            'confirmation_period': [5, 30],
            'position_sizing_method': ['fixed', 'volatility', 'atr']
        },
        'fixed_parameters': {
            'strategy_class': 'BollingerStrategy'
        }
    },
    {
        'template_id': 'rsi_strategy',
        'base_name': 'RSI_Strategy',
        'strategy_type': 'indicator',
        'description': 'RSI overbought/oversold strategy with dynamic thresholds',
        'parameter_ranges': {
            'rsi_period': [5, 30],
            'overbought_threshold': [65, 85],
            'oversold_threshold': [15, 35],
            'smoothing_period': [1, 10],
            'use_confirmation': {'type': 'boolean'},
            'confirmation_type': ['price_action', 'volume', 'trend'],
            'exit_method': ['threshold', 'time', 'trailing', 'fixed']
        },
        'fixed_parameters': {
            'strategy_class': 'RSIStrategy'
        }
    },
    {
        'template_id': 'macd_strategy',
        'base_name': 'MACD_Strategy',
        'strategy_type': 'indicator',
        'description': 'MACD crossover and divergence strategy',
        'parameter_ranges': {
            'fast_period': [8, 24],
            'slow_period': [16, 52],
            'signal_period': [5, 18],
            'trigger_type': ['crossover', 'zero_line', 'divergence'],
            'filter_trades': {'type': 'boolean'},
            'minimum_histogram': [0.0, 0.01],
            'trade_direction': ['long_only', 'short_only', 'both']
        },
        'fixed_parameters': {
            'strategy_class': 'MACDStrategy'
        }
    }
]

# Pattern recognition strategy templates
PATTERN_STRATEGY_TEMPLATES = [
    {
        'template_id': 'candlestick_pattern',
        'base_name': 'Candle_Pattern',
        'strategy_type': 'pattern',
        'description': 'Candlestick pattern recognition with confirmation filters',
        'parameter_ranges': {
            'patterns': [
                {'type': 'weighted_choice', 'choices': [
                    'hammer', 'inverted_hammer', 'engulfing', 'doji',
                    'morning_star', 'evening_star', 'harami', 'three_outside'
                ]}
            ],
            'pattern_strength': [0.5, 1.5],
            'confirmation_bars': [1, 5],
            'volume_confirmation': {'type': 'boolean'},
            'trend_filter': ['none', 'sma', 'ema', 'auto'],
            'trend_period': [20, 200]
        },
        'fixed_parameters': {
            'strategy_class': 'CandlestickPatternStrategy'
        }
    },
    {
        'template_id': 'chart_pattern',
        'base_name': 'Chart_Pattern',
        'strategy_type': 'pattern',
        'description': 'Chart pattern recognition for triangles, flags, and more',
        'parameter_ranges': {
            'pattern_types': [
                {'type': 'weighted_choice', 'choices': [
                    'triangle', 'flag', 'wedge', 'head_shoulders',
                    'double_top', 'double_bottom', 'rectangle'
                ]}
            ],
            'min_pattern_bars': [5, 50],
            'max_pattern_bars': [20, 100],
            'confirmation_percentage': [0.5, 5.0],
            'invalidation_percentage': [0.5, 5.0],
            'target_ratio': [1.0, 3.0]
        },
        'fixed_parameters': {
            'strategy_class': 'ChartPatternStrategy'
        }
    },
    {
        'template_id': 'breakout_strategy',
        'base_name': 'Breakout_Strategy',
        'strategy_type': 'pattern',
        'description': 'Support/resistance breakout strategy',
        'parameter_ranges': {
            'period': [10, 100],
            'lookback_bars': [5, 50],
            'breakout_sensitivity': [0.5, 3.0],
            'confirmation_bars': [1, 5],
            'volume_confirmation': {'type': 'boolean'},
            'breakout_filter': ['none', 'atr', 'pivot', 'volatility'],
            'fake_breakout_filter': {'type': 'boolean'}
        },
        'fixed_parameters': {
            'strategy_class': 'BreakoutStrategy'
        }
    }
]

# Machine learning strategy templates
ML_STRATEGY_TEMPLATES = [
    {
        'template_id': 'boosting_model',
        'base_name': 'Boosting_Model',
        'strategy_type': 'ml',
        'description': 'Gradient boosting model for price direction prediction',
        'parameter_ranges': {
            'estimators': [50, 500],
            'max_depth': [3, 10],
            'learning_rate': [0.01, 0.3],
            'min_child_weight': [1, 10],
            'gamma': [0.0, 0.5],
            'colsample_bytree': [0.3, 1.0],
            'subsample': [0.5, 1.0],
            'prediction_threshold': [0.55, 0.8],
            'confirmation_filter': ['none', 'technical', 'volatility', 'regime']
        },
        'fixed_parameters': {
            'strategy_class': 'GradientBoostingStrategy',
            'model_type': 'xgboost'
        }
    },
    {
        'template_id': 'ensemble_strategy',
        'base_name': 'Ensemble_Strategy',
        'strategy_type': 'ml',
        'description': 'Ensemble of multiple models with voting mechanism',
        'parameter_ranges': {
            'base_models': [2, 5],
            'voting_method': ['majority', 'weighted', 'confidence'],
            'confidence_threshold': [0.6, 0.9],
            'feature_selection': ['auto', 'pca', 'recursive', 'importance'],
            'validation_method': ['walk_forward', 'cross_validation', 'split'],
            'adaptation_speed': [0.05, 0.5]
        },
        'fixed_parameters': {
            'strategy_class': 'EnsembleStrategy'
        }
    }
]

# Combined templates
STRATEGY_TEMPLATES = INDICATOR_STRATEGY_TEMPLATES + PATTERN_STRATEGY_TEMPLATES + ML_STRATEGY_TEMPLATES

def get_template_by_id(template_id: str) -> Dict[str, Any]:
    """Get a strategy template by ID."""
    for template in STRATEGY_TEMPLATES:
        if template['template_id'] == template_id:
            return template
    return None

def get_templates_by_type(strategy_type: str) -> List[Dict[str, Any]]:
    """Get all templates of a specific strategy type."""
    return [t for t in STRATEGY_TEMPLATES if t['strategy_type'] == strategy_type]

def get_all_templates() -> List[Dict[str, Any]]:
    """Get all available templates."""
    return STRATEGY_TEMPLATES
