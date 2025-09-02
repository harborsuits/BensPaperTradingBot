# Machine Learning Components for Trading Bot

This module provides state-of-the-art machine learning models and utilities for quantitative trading, focusing on ensemble methods, deep learning architectures, and model explainability.

## Overview

The ML module includes:

1. **Base ML Model** - Abstract base class for all ML models
2. **Ensemble Model** - Combines multiple models using various ensemble methods
3. **LSTM Model** - Deep learning model for time series prediction
4. **ML Feature Extractor** - Generates comprehensive features for ML models
5. **Model Factory** - Creates and configures ML models and strategies
6. **ML Strategy** - Strategy implementation that uses ML models

## Key Features

- **Ensemble Learning & Model Stacking**: Combine multiple models to reduce variance and improve prediction accuracy
- **Deep Learning Approaches**: LSTM-based models for capturing temporal patterns in market data
- **Comprehensive Feature Engineering**: Generate a rich set of technical indicators and statistical features
- **Model Explainability**: Understand why models make specific predictions
- **Regime-Based Adaptation**: Adjust strategy weights based on detected market regimes

## How to Use

### Basic Usage

```python
from trading_bot.ml.model_factory import MLModelFactory
from trading_bot.data.features.ml_features import MLFeatureExtractor
from trading_bot.strategy.strategy_rotator import StrategyRotator

# Create a model factory
factory = MLModelFactory(data_dir="./data")

# Create a feature extractor
feature_extractor = factory.create_feature_extractor()

# Create an LSTM model
lstm_model = factory.create_lstm_model(
    name="price_predictor",
    sequence_length=10,
    lstm_units=[64, 32],
    target_type="return"
)

# Create an ML strategy
ml_strategy = factory.create_ml_strategy(
    name="LSTMStrategy",
    model=lstm_model,
    feature_extractor=feature_extractor
)

# Add to a strategy rotator
rotator = StrategyRotator()
rotator.add_strategy(ml_strategy)
```

### Creating an Ensemble Strategy

```python
# Create base models
lstm_model = factory.create_lstm_model(name="lstm_model", target_type="direction")
lstm_model2 = factory.create_lstm_model(name="lstm_model2", target_type="return")

# Create ensemble strategy (all in one step)
ensemble_strategy = factory.create_ml_ensemble_strategy(
    name="EnsembleStrategy",
    models=[lstm_model, lstm_model2],
    feature_extractor=feature_extractor,
    ensemble_method="weighted"
)
```

### Setting Up Regime-Based Rotation

```python
# Configure regime-based rotation
factory.setup_regime_based_rotation(
    rotator=rotator,
    bull_strategies=["EnsembleStrategy", "TrendFollowing"],
    bear_strategies=["DefensiveStrategy", "MeanReversion"],
    sideways_strategies=["MeanReversion", "RangeTrading"],
    volatile_strategies=["Ensemble", "MultiTimeframe"]
)
```

## Component Details

### BaseMLModel

Abstract base class that defines the interface for all ML models:

- `preprocess()`: Preprocess data before training or prediction
- `train()`: Train the model on the given data
- `predict()`: Generate predictions from the model
- `get_feature_importance()`: Get feature importance from the model
- `explain_prediction()`: Explain a single prediction using feature importance
- `save()` / `load()`: Save and load models from disk

### EnsembleModel

Combines multiple base models using various ensemble methods:

- Simple averaging
- Weighted averaging (based on validation performance)
- Stacking (meta-model)
- Majority voting (for classification)
- Rank-based averaging

### LSTMModel

Deep learning model for time series prediction:

- Configurable LSTM architecture
- Multiple prediction types (price, return, direction)
- Sequence preprocessing
- Feature scaling
- Supports different prediction horizons

### MLFeatureExtractor

Generates a comprehensive set of features for ML models:

- Price-based features (returns, momentum, etc.)
- Moving average features
- Volatility features
- RSI and other oscillators
- Bollinger Bands features
- Volume features
- Trend indicators
- Pattern recognition features

### MLStrategy

Trading strategy that uses ML models for signal generation:

- Converts model predictions to trading signals
- Provides confidence scores for predictions
- Offers explanations for generated signals

## Example

See `examples/ml_strategy_example.py` for a complete example of creating, training, and using ML-based trading strategies with the strategy rotator. 