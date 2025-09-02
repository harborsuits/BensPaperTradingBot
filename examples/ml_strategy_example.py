#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML Strategy Example - Demonstrates how to create and use ML-based trading strategies.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_bot.data.sources.alpha_vantage import AlphaVantageDataSource
from trading_bot.data.sources.yahoo import YahooFinanceDataSource
from trading_bot.data.features.ml_features import MLFeatureExtractor
from trading_bot.ml.model_factory import MLModelFactory
from trading_bot.strategy.strategy_rotator import StrategyRotator, MomentumStrategy, TrendFollowingStrategy, MeanReversionStrategy
from trading_bot.common.market_types import MarketRegime


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fetch_historical_data(symbol='SPY', api_key=None, days=365):
    """Fetch historical data for a symbol."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Try Yahoo Finance first (no API key needed)
    try:
        data_source = YahooFinanceDataSource()
        data = data_source.get_data(symbol, start_date, end_date)
        if data:
            print(f"Fetched {len(data)} datapoints from Yahoo Finance")
            return data_source.to_dataframe(data)
    except Exception as e:
        print(f"Error fetching data from Yahoo Finance: {str(e)}")
    
    # Fall back to Alpha Vantage if API key provided
    if api_key:
        try:
            data_source = AlphaVantageDataSource(api_key=api_key)
            data = data_source.get_data(symbol, start_date, end_date)
            if data:
                print(f"Fetched {len(data)} datapoints from Alpha Vantage")
                return data_source.to_dataframe(data)
        except Exception as e:
            print(f"Error fetching data from Alpha Vantage: {str(e)}")
    
    # Create dummy data if both data sources fail
    print("Using dummy data")
    return create_dummy_data(days)

def create_dummy_data(days=365):
    """Create dummy market data for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate initial price and add random walk
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.0005, 0.01, len(dates))
    price = base_price * (1 + returns).cumprod()
    
    # Generate OHLCV data
    volatility = 0.01 * price
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.normal(0, 0.003, len(dates))),
        'high': price + volatility * np.random.uniform(0.5, 1.5, len(dates)),
        'low': price - volatility * np.random.uniform(0.5, 1.5, len(dates)),
        'close': price,
        'volume': np.random.lognormal(15, 1, len(dates)),
        'symbol': 'DUMMY'
    })
    
    # Ensure high >= open, close and low <= open, close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    df.set_index('timestamp', inplace=True)
    return df

def create_strategies():
    """Create and configure ML and traditional strategies."""
    # Create a model factory
    factory = MLModelFactory(data_dir="./data")
    
    # Create a feature extractor
    feature_extractor = factory.create_feature_extractor(
        include_returns=True,
        include_bbands=True,
        include_volume_features=True,
        include_trend_features=True,
        normalize_features=True
    )
    
    # Create an LSTM model for price direction prediction
    lstm_model = factory.create_lstm_model(
        name="lstm_direction",
        sequence_length=10,
        lstm_units=[64, 32],
        target_type="direction",
        target_horizon=5,
        learning_rate=0.001,
        dropout_rate=0.3
    )
    
    # Create another LSTM model with different parameters
    lstm_model2 = factory.create_lstm_model(
        name="lstm_return",
        sequence_length=20,
        lstm_units=[128, 64],
        target_type="return",
        target_horizon=1,
        learning_rate=0.0005,
        dropout_rate=0.2
    )
    
    # Create an ensemble model
    ensemble_model = factory.create_ensemble_model(
        name="ensemble_model",
        base_models=[lstm_model, lstm_model2],
        ensemble_method="weighted"
    )
    
    # Create ML strategies
    ml_strategy1 = factory.create_ml_strategy(
        name="LSTM_Direction",
        model=lstm_model,
        feature_extractor=feature_extractor,
        signal_threshold=0.1,
        signal_scaling=1.5
    )
    
    ml_strategy2 = factory.create_ml_strategy(
        name="LSTM_Return",
        model=lstm_model2,
        feature_extractor=feature_extractor,
        signal_threshold=0.05,
        signal_scaling=2.0
    )
    
    ensemble_strategy = factory.create_ml_strategy(
        name="Ensemble",
        model=ensemble_model,
        feature_extractor=feature_extractor,
        signal_threshold=0.0,
        signal_scaling=1.0
    )
    
    # Create a multi-timeframe ensemble strategy
    mtf_strategy = factory.create_multi_timeframe_ensemble(
        name="MultiTimeframe",
        base_model_type="lstm",
        timeframes=[1, 5, 10],
        feature_extractor=feature_extractor,
        signal_threshold=0.05,
        signal_scaling=1.2
    )
    
    # Create traditional strategies for comparison
    momentum_strategy = MomentumStrategy(
        name="Momentum",
        config={"fast_period": 5, "slow_period": 20}
    )
    
    trend_strategy = TrendFollowingStrategy(
        name="TrendFollow",
        config={"short_ma_period": 10, "long_ma_period": 30}
    )
    
    mean_reversion_strategy = MeanReversionStrategy(
        name="MeanReversion",
        config={"period": 20, "std_dev_factor": 2.0}
    )
    
    return {
        "ml_strategies": [ml_strategy1, ml_strategy2, ensemble_strategy, mtf_strategy],
        "traditional_strategies": [momentum_strategy, trend_strategy, mean_reversion_strategy],
        "feature_extractor": feature_extractor,
        "factory": factory
    }

def setup_strategy_rotator(ml_strategies, traditional_strategies, factory):
    """Set up a strategy rotator with ML and traditional strategies."""
    # Create the rotator
    rotator = StrategyRotator(regime_adaptation=True)
    
    # Add traditional strategies
    for strategy in traditional_strategies:
        rotator.add_strategy(strategy)
    
    # Add ML strategies using the factory
    factory.add_ml_strategies_to_rotator(
        rotator=rotator,
        strategies=ml_strategies,
        initial_weights={
            "Ensemble": 0.3,
            "MultiTimeframe": 0.2,
            "LSTM_Direction": 0.1,
            "LSTM_Return": 0.1,
            "Momentum": 0.1,
            "TrendFollow": 0.1,
            "MeanReversion": 0.1
        }
    )
    
    # Set up regime-based rotation
    factory.setup_regime_based_rotation(
        rotator=rotator,
        bull_strategies=["Ensemble", "LSTM_Direction", "Momentum", "TrendFollow"],
        bear_strategies=["Ensemble", "MultiTimeframe", "TrendFollow"],
        sideways_strategies=["MeanReversion", "LSTM_Return"],
        volatile_strategies=["Ensemble", "MultiTimeframe"]
    )
    
    return rotator

def train_models(strategies, data):
    """Train the ML models in the strategies using the provided data."""
    print("\nTraining ML models...")
    for strategy in strategies:
        if hasattr(strategy, 'model') and hasattr(strategy.model, 'train'):
            print(f"Training model for strategy: {strategy.name}")
            
            # Prepare features
            if hasattr(strategy, 'feature_extractor') and strategy.feature_extractor:
                features_df = strategy.feature_extractor.extract_features(data)
            else:
                features_df = data
                
            # For this example, we'll use price changes as target
            if hasattr(strategy.model, 'target_type'):
                if strategy.model.target_type == "direction":
                    # Direction of price movement
                    target = (data['close'].pct_change() > 0).astype(float)
                    target_df = pd.DataFrame(target)
                elif strategy.model.target_type == "return":
                    # Price return
                    target = data['close'].pct_change()
                    target_df = pd.DataFrame(target)
                else:
                    # Price level
                    target_df = pd.DataFrame(data['close'])
            else:
                # Default to price return
                target = data['close'].pct_change()
                target_df = pd.DataFrame(target)
                
            # Train the model
            try:
                strategy.model.train(features_df, target_df)
                print(f"Successfully trained model for strategy: {strategy.name}")
            except Exception as e:
                print(f"Error training model for strategy {strategy.name}: {str(e)}")
    
    print("Training complete.")

def test_strategies(rotator, data):
    """Test strategies by generating signals from market data."""
    print("\nTesting strategies...")
    
    signals = {}
    for strategy in rotator.strategies:
        # Prepare market data in the format expected by strategies
        market_data = {
            "prices": data['close'].values,
            "open": data['open'].values if 'open' in data.columns else None,
            "high": data['high'].values if 'high' in data.columns else None,
            "low": data['low'].values if 'low' in data.columns else None,
            "volume": data['volume'].values if 'volume' in data.columns else None,
            "timestamps": data.index.values
        }
        
        # Generate signal
        signal = strategy.generate_signal(market_data)
        signals[strategy.name] = signal
        print(f"Strategy: {strategy.name}, Signal: {signal:.4f}")
    
    # Generate combined signal
    combined_signal = rotator.get_combined_signal()
    print(f"Combined signal: {combined_signal:.4f}")
    
    return signals, combined_signal

def simulate_regime_changes(rotator):
    """Simulate different market regimes and see how strategy weights change."""
    print("\nSimulating market regime changes...")
    
    # Get initial weights
    initial_weights = rotator.get_strategy_weights()
    print("Initial weights:")
    for name, weight in initial_weights.items():
        print(f"  {name}: {weight:.2f}")
    
    # Test different regimes
    regimes = [
        MarketRegime.BULL,
        MarketRegime.BEAR,
        MarketRegime.SIDEWAYS,
        MarketRegime.HIGH_VOL,
        MarketRegime.CRISIS
    ]
    
    for regime in regimes:
        # Update market regime
        rotator.update_market_regime(regime, confidence=1.0)
        
        # Get weights after regime change
        weights = rotator.get_strategy_weights()
        print(f"\nWeights after regime change to {regime.name}:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.2f}")

def main():
    """Main function to run the example."""
    print("ML Strategy Example - Demonstrates how to create and use ML-based trading strategies.")
    
    # Set your Alpha Vantage API key if you have one
    api_key = None  # Replace with your key if you have one
    
    # Fetch historical data
    print("\nFetching historical data...")
    data = fetch_historical_data(symbol='SPY', api_key=api_key, days=365)
    print(f"Data shape: {data.shape}")
    
    # Create strategies
    print("\nCreating strategies...")
    strategies = create_strategies()
    
    # Set up strategy rotator
    print("\nSetting up strategy rotator...")
    rotator = setup_strategy_rotator(
        strategies["ml_strategies"],
        strategies["traditional_strategies"],
        strategies["factory"]
    )
    
    # Train ML models
    train_models(strategies["ml_strategies"], data)
    
    # Test strategies
    signals, combined_signal = test_strategies(rotator, data)
    
    # Simulate regime changes
    simulate_regime_changes(rotator)
    
    print("\nExample completed.")

if __name__ == "__main__":
    main() 