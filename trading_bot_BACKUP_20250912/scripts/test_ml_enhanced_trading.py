#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Script for ML-Enhanced Trading System

This script demonstrates how to initialize, train, and use the ML-enhanced
trading system with sample market data.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, project_root)

from trading_bot.ml.enhanced_features import EnhancedFeatureEngineering
from trading_bot.ml.signal_model import SignalModel, create_signal_model
from trading_bot.ml.market_regime_autoencoder import MarketRegimeAutoencoder
from trading_bot.rl.position_sizer_env import PositionSizerEnv
from trading_bot.rl.position_sizer_agent import PositionSizerAgent
from trading_bot.ml.ml_enhanced_trading import MLEnhancedTradingSystem, create_ml_enhanced_trading_system
from trading_bot.strategies.ml_enhanced_strategy import MLEnhancedStrategy

logger = logging.getLogger(__name__)

def fetch_sample_data(symbols=None, period='3y', interval='1d'):
    """
    Fetch sample data for testing
    
    Args:
        symbols: List of symbols to fetch
        period: Data period (default: 3y)
        interval: Data interval (default: 1d)
        
    Returns:
        Dictionary of DataFrames by symbol
    """
    if symbols is None:
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
    
    logger.info(f"Fetching data for {len(symbols)} symbols: {', '.join(symbols)}")
    
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                continue
                
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only required columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Add to data dict
            data[symbol] = df
            logger.info(f"Fetched {len(df)} bars for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return data

def train_feature_engineering(market_data):
    """
    Test the enhanced feature engineering module
    
    Args:
        market_data: Dictionary of market data by symbol
        
    Returns:
        Trained EnhancedFeatureEngineering instance and processed data
    """
    logger.info("Testing Enhanced Feature Engineering")
    
    # Initialize feature engineering
    feature_engine = EnhancedFeatureEngineering()
    
    # Process each symbol
    processed_data = {}
    for symbol, data in market_data.items():
        try:
            # Generate features
            features_df = feature_engine.generate_features(data)
            
            # Store processed data
            processed_data[symbol] = features_df
            
            logger.info(f"Generated {len(features_df.columns)} features for {symbol}")
            logger.info(f"Top features: {list(features_df.columns)[:5]}")
            
        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {e}")
    
    return feature_engine, processed_data

def train_signal_model(market_data, processed_data):
    """
    Train and test the signal model
    
    Args:
        market_data: Dictionary of market data by symbol
        processed_data: Dictionary of processed data by symbol
        
    Returns:
        Trained SignalModel instance
    """
    logger.info("Testing Signal Model")
    
    # Initialize signal model
    signal_model = create_signal_model()
    
    # Train on each symbol
    for symbol, data in processed_data.items():
        try:
            # Train model
            for horizon in [1, 3, 5]:
                logger.info(f"Training signal model for {symbol} with horizon {horizon}")
                
                # Train the model
                train_result = signal_model.train(
                    data, 
                    prediction_horizon=horizon,
                    test_size=0.3,
                    save_model=True
                )
                
                # Log results
                logger.info(f"Training results for {symbol} horizon {horizon}:")
                logger.info(f"  Accuracy: {train_result.get('accuracy', 0):.4f}")
                logger.info(f"  Precision: {train_result.get('precision', 0):.4f}")
                logger.info(f"  Recall: {train_result.get('recall', 0):.4f}")
                logger.info(f"  F1 Score: {train_result.get('f1', 0):.4f}")
                
                # Test prediction
                predictions = signal_model.predict(data, horizon)
                
                if predictions:
                    logger.info(f"Generated predictions for {symbol} horizon {horizon}")
                    
        except Exception as e:
            logger.error(f"Error training signal model for {symbol}: {e}")
    
    # Test ensemble signal generation
    try:
        symbol = list(processed_data.keys())[0]
        data = processed_data[symbol]
        
        # Generate ensemble signal
        weights = {1: 0.4, 3: 0.4, 5: 0.2}
        ensemble = signal_model.generate_ensemble_signal(data, weights)
        
        if not ensemble.empty:
            logger.info(f"Generated ensemble signal for {symbol}")
            logger.info(f"Signal stats: Bullish: {(ensemble['ensemble_signal'] == 'buy').mean():.2%}, "
                      f"Bearish: {(ensemble['ensemble_signal'] == 'sell').mean():.2%}, "
                      f"Neutral: {(ensemble['ensemble_signal'] == 'neutral').mean():.2%}")
    except Exception as e:
        logger.error(f"Error generating ensemble signal: {e}")
    
    return signal_model

def train_regime_detector(market_data):
    """
    Train and test the market regime autoencoder
    
    Args:
        market_data: Dictionary of market data by symbol
        
    Returns:
        Trained MarketRegimeAutoencoder instance
    """
    logger.info("Testing Market Regime Autoencoder")
    
    # Initialize regime detector
    regime_detector = MarketRegimeAutoencoder()
    
    # Use SPY for training if available
    train_symbol = 'SPY' if 'SPY' in market_data else list(market_data.keys())[0]
    train_data = market_data[train_symbol]
    
    try:
        # Train model
        train_result = regime_detector.train(train_data)
        
        # Log results
        logger.info(f"Trained regime detector on {train_symbol}")
        logger.info(f"Training loss: {train_result.get('loss', 0):.6f}")
        
        # Test regime detection
        for symbol, data in market_data.items():
            # Detect regime
            regimes = regime_detector.detect_market_regime(data)
            
            if not regimes.empty:
                logger.info(f"Detected regimes for {symbol}")
                regime_counts = regimes['regime'].value_counts()
                logger.info(f"Regime distribution: {regime_counts.to_dict()}")
                
    except Exception as e:
        logger.error(f"Error training regime detector: {e}")
    
    return regime_detector

def train_position_sizer(market_data, signal_model, regime_detector):
    """
    Train and test the position sizer
    
    Args:
        market_data: Dictionary of market data by symbol
        signal_model: Trained SignalModel instance
        regime_detector: Trained MarketRegimeAutoencoder instance
        
    Returns:
        Trained PositionSizerAgent instance
    """
    logger.info("Testing Position Sizer Agent")
    
    # Create environment config
    env_config = {
        "max_position_size": 0.25,
        "trading_cost": 0.001,
        "max_steps": 252,  # Roughly one trading year
        "reward_scaling": 10.0,
        "starting_balance": 100000
    }
    
    # Initialize environment
    env = PositionSizerEnv(env_config)
    
    # Initialize agent
    position_sizer = PositionSizerAgent(env)
    
    try:
        # Prepare training data
        symbol = 'SPY' if 'SPY' in market_data else list(market_data.keys())[0]
        data = market_data[symbol]
        
        # Generate features and signals
        features_df = EnhancedFeatureEngineering().generate_features(data)
        signals = signal_model.generate_ensemble_signal(
            features_df, 
            {1: 0.4, 3: 0.4, 5: 0.2}
        )
        regimes = regime_detector.detect_market_regime(data)
        
        # Combine data for environment
        combined_data = data.copy()
        
        # Add signal and confidence
        if not signals.empty:
            combined_data['signal'] = signals['ensemble_signal']
            combined_data['confidence'] = signals['weighted_probability']
        else:
            # Create default signals if none available
            combined_data['signal'] = 'neutral'
            combined_data['confidence'] = 0.5
        
        # Add regime
        if not regimes.empty:
            combined_data['regime'] = regimes['regime']
        else:
            # Default regime if none available
            combined_data['regime'] = 0
        
        # Set environment data
        env.set_data(combined_data)
        
        # Train agent (short training for test)
        logger.info("Training position sizer agent")
        position_sizer.train(total_timesteps=10000, eval_freq=2000)
        
        # Test agent
        test_obs = env.reset()
        
        # Run a few steps
        for i in range(20):
            action, _states = position_sizer.model.predict(test_obs)
            obs, reward, done, info = env.step(action)
            logger.info(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}")
            
            if done:
                break
            
            test_obs = obs
        
        # Test position sizing function
        account_state = {"equity": 100000, "balance": 100000}
        sizing = position_sizer.get_position_size(
            signal_confidence=0.75,
            market_regime=0,
            account_state=account_state
        )
        
        logger.info(f"Position sizing test: {sizing}")
        
    except Exception as e:
        logger.error(f"Error training position sizer: {e}")
    
    return position_sizer

def test_ml_system(market_data):
    """
    Test the complete ML-enhanced trading system
    
    Args:
        market_data: Dictionary of market data by symbol
    """
    logger.info("Testing Complete ML-Enhanced Trading System")
    
    # Initialize the ML system
    ml_system = create_ml_enhanced_trading_system()
    
    try:
        # Update the system with market data
        update_result = ml_system.update(market_data)
        
        logger.info(f"ML system updated with {update_result['num_symbols']} symbols")
        
        # Get signals for each symbol
        for symbol in market_data.keys():
            signal = ml_system.get_trade_signal(symbol)
            regime = ml_system.get_market_regime(symbol)
            
            logger.info(f"Signal for {symbol}: {signal['signal']}, "
                      f"Confidence: {signal['confidence']:.4f}, "
                      f"Regime: {regime['regime_name']}")
            
            # Get position size
            account_state = {"equity": 100000, "balance": 100000}
            position = ml_system.get_position_size(symbol, account_state)
            
            logger.info(f"Position for {symbol}: Size={position['position_size_pct']:.2%}, "
                      f"Value=${position['position_size_currency']:.2f}")
        
        # Test portfolio optimization
        signals = {symbol: ml_system.get_trade_signal(symbol) for symbol in market_data.keys()}
        portfolio = ml_system.optimize_portfolio(signals, {"equity": 100000})
        
        total_allocation = sum(p["position_size_pct"] for p in portfolio.values())
        logger.info(f"Portfolio optimization: {len(portfolio)} positions, "
                  f"Total allocation: {total_allocation:.2%}")
        
        # Test state saving and loading
        state_file = ml_system.save_state()
        load_result = ml_system.load_state(state_file)
        
        logger.info(f"State save/load test: {load_result}")
        
    except Exception as e:
        logger.error(f"Error testing ML system: {e}")

def test_ml_strategy(market_data):
    """
    Test the ML-enhanced strategy
    
    Args:
        market_data: Dictionary of market data by symbol
    """
    logger.info("Testing ML-Enhanced Strategy")
    
    # Strategy config
    config = {
        "symbols": list(market_data.keys()),
        "timeframe": "1d",
        "min_confidence": 0.6,
        "execute_trades": False  # Disable actual trade execution for test
    }
    
    # Initialize strategy
    strategy = MLEnhancedStrategy(config)
    
    try:
        # Initialize the strategy
        strategy.initialize()
        
        # Start strategy
        strategy.on_start()
        
        # Feed market data to strategy
        for symbol, data in market_data.items():
            for i in range(min(50, len(data))):  # Process last 50 bars
                index = -50 + i
                
                # Create bar data
                bar = {
                    'symbol': symbol,
                    'timestamp': data.index[index],
                    'open': data['open'].iloc[index],
                    'high': data['high'].iloc[index],
                    'low': data['low'].iloc[index],
                    'close': data['close'].iloc[index],
                    'volume': data['volume'].iloc[index]
                }
                
                # Send bar to strategy
                strategy.on_bar(bar)
        
        # Log results
        logger.info(f"Strategy signals: {len(strategy.current_signals)}")
        logger.info(f"Strategy positions: {len(strategy.current_positions)}")
        
        for symbol, position in strategy.current_positions.items():
            logger.info(f"Position for {symbol}: "
                      f"Signal={position['signal']}, "
                      f"Size={position['position_size']:.2%}, "
                      f"Value=${position['position_value']:.2f}")
        
        # Stop strategy
        strategy.on_stop()
        
    except Exception as e:
        logger.error(f"Error testing ML strategy: {e}")

def plot_sample_predictions(market_data, ml_system):
    """
    Plot sample predictions and signals
    
    Args:
        market_data: Dictionary of market data by symbol
        ml_system: Trained ML system
    """
    # Select a symbol to plot
    symbol = 'SPY' if 'SPY' in market_data else list(market_data.keys())[0]
    data = market_data[symbol]
    
    try:
        # Get signals
        signal = ml_system.signal_model.generate_ensemble_signal(
            data, 
            ml_system.config["signal_model_config"]["ensemble_weights"]
        )
        
        if signal.empty:
            logger.warning(f"No signals available for {symbol}")
            return
        
        # Get regimes
        regimes = ml_system.regime_detector.detect_market_regime(data)
        
        if regimes.empty:
            logger.warning(f"No regimes available for {symbol}")
            return
        
        # Create a new figure
        plt.figure(figsize=(12, 8))
        
        # Plot price
        plt.subplot(3, 1, 1)
        plt.plot(data.index, data['close'], label='Close Price')
        plt.title(f'{symbol} Price and Signals')
        plt.ylabel('Price')
        plt.grid(True)
        plt.legend()
        
        # Plot signals
        plt.subplot(3, 1, 2)
        signal_values = signal['weighted_probability']
        
        # Convert signals to numeric values for plotting
        numeric_signals = signal['ensemble_signal'].map({
            'buy': 1, 
            'neutral': 0, 
            'sell': -1
        })
        
        plt.plot(signal.index, signal_values, 'b-', label='Signal Confidence')
        plt.scatter(signal.index, numeric_signals, c=numeric_signals, 
                   cmap='RdYlGn', label='Trading Signals')
        plt.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
        plt.ylabel('Signal')
        plt.grid(True)
        plt.legend()
        
        # Plot regimes
        plt.subplot(3, 1, 3)
        plt.plot(regimes.index, regimes['regime'], 'r-', label='Market Regime')
        plt.ylabel('Regime')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/{symbol}_ml_signals.png')
        
        logger.info(f"Created plot for {symbol} at plots/{symbol}_ml_signals.png")
        
    except Exception as e:
        logger.error(f"Error creating plots: {e}")

def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test ML-Enhanced Trading System')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN'],
                        help='Symbols to test')
    parser.add_argument('--period', default='3y', help='Data period')
    parser.add_argument('--interval', default='1d', help='Data interval')
    parser.add_argument('--skip-training', action='store_true', help='Skip training steps')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting ML-Enhanced Trading System Test")
    
    # Fetch sample data
    market_data = fetch_sample_data(args.symbols, args.period, args.interval)
    
    if not market_data:
        logger.error("No market data available. Exiting.")
        return
    
    # Choose training path
    if args.skip_training:
        # Create ML system without training
        logger.info("Skipping training steps and testing complete system")
        ml_system = create_ml_enhanced_trading_system()
        
        # Test complete system
        test_ml_system(market_data)
        
    else:
        # Train and test each component
        feature_engine, processed_data = train_feature_engineering(market_data)
        signal_model = train_signal_model(market_data, processed_data)
        regime_detector = train_regime_detector(market_data)
        position_sizer = train_position_sizer(market_data, signal_model, regime_detector)
        
        # Test complete system
        ml_system = create_ml_enhanced_trading_system()
        test_ml_system(market_data)
    
        # Test strategy
        test_ml_strategy(market_data)
    
        # Create plots
        plot_sample_predictions(market_data, ml_system)
    
    logger.info("ML-Enhanced Trading System Test Complete")

if __name__ == "__main__":
    main()
