#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Optimization Example using Reinforcement Learning

This script demonstrates how to use the RL components of the trading bot
to train and evaluate a portfolio optimization agent.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

from trading_bot.rl.trading_env import TradingEnv
from trading_bot.rl.agent_trainer import AgentTrainer
from trading_bot.utils.feature_engineering import FeatureEngineering

def load_market_data(data_dir: str) -> dict:
    """
    Load market data from CSV files in the specified directory.
    
    Args:
        data_dir: Directory containing CSV files with market data
        
    Returns:
        Dictionary mapping asset symbols to DataFrames
    """
    market_data = {}
    
    # Get all CSV files in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            # Extract symbol from filename (e.g., "BTC_USD.csv" -> "BTC_USD")
            symbol = os.path.splitext(filename)[0]
            
            # Load CSV file
            file_path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                df.set_index('date', inplace=True)
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"Warning: {symbol} is missing required columns: {missing_cols}")
                    continue
                
                market_data[symbol] = df
                print(f"Loaded {symbol} data with {len(df)} rows")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return market_data

def generate_features(market_data: dict, config: dict) -> dict:
    """
    Generate features for all assets using the FeatureEngineering class.
    
    Args:
        market_data: Dictionary mapping asset symbols to DataFrames
        config: Configuration parameters for feature engineering
        
    Returns:
        Dictionary mapping asset symbols to DataFrames with features
    """
    feature_eng = FeatureEngineering(config)
    feature_data = {}
    
    for symbol, df in market_data.items():
        print(f"Generating features for {symbol}...")
        try:
            # Generate features
            features_df = feature_eng.generate_features(df)
            
            # Add target variables
            features_with_labels = feature_eng.add_return_labels(
                features_df,
                future_windows=config.get('target', {}).get('horizons', [1, 5, 10]),
                thresholds=config.get('target', {}).get('thresholds', [0.0, 0.005, 0.01])
            )
            
            # Store in dictionary
            feature_data[symbol] = features_with_labels
            print(f"  Generated {len(features_with_labels.columns)} features for {symbol}")
        except Exception as e:
            print(f"  Error generating features for {symbol}: {str(e)}")
    
    return feature_data

def create_training_environment(
    feature_data: dict,
    feature_list: list,
    config: dict,
    train_start: str,
    train_end: str
) -> TradingEnv:
    """
    Create a training environment with the specified data.
    
    Args:
        feature_data: Dictionary mapping asset symbols to DataFrames with features
        feature_list: List of feature column names to use
        config: Configuration parameters for the environment
        train_start: Start date for training data (YYYY-MM-DD)
        train_end: End date for training data (YYYY-MM-DD)
        
    Returns:
        TradingEnv instance
    """
    # Filter data to training period
    train_data = {}
    for symbol, df in feature_data.items():
        # Filter by date range
        mask = (df.index >= train_start) & (df.index <= train_end)
        filtered_df = df.loc[mask].copy()
        
        if not filtered_df.empty:
            train_data[symbol] = filtered_df
    
    # Create environment
    env = TradingEnv(
        df_dict=train_data,
        features_list=feature_list,
        initial_balance=config.get('initial_balance', 100000.0),
        trading_cost=config.get('trading_cost', 0.001),
        slippage=config.get('slippage', 0.0005),
        window_size=config.get('window_size', 50),
        reward_type=config.get('reward_type', 'sharpe'),
        reward_scale=config.get('reward_scale', 1.0),
        allow_short=config.get('allow_short', False)
    )
    
    return env

def create_test_environment(
    feature_data: dict,
    feature_list: list,
    config: dict,
    test_start: str,
    test_end: str
) -> TradingEnv:
    """
    Create a test environment with the specified data.
    
    Args:
        feature_data: Dictionary mapping asset symbols to DataFrames with features
        feature_list: List of feature column names to use
        config: Configuration parameters for the environment
        test_start: Start date for test data (YYYY-MM-DD)
        test_end: End date for test data (YYYY-MM-DD)
        
    Returns:
        TradingEnv instance
    """
    # Filter data to test period
    test_data = {}
    for symbol, df in feature_data.items():
        # Filter by date range
        mask = (df.index >= test_start) & (df.index <= test_end)
        filtered_df = df.loc[mask].copy()
        
        if not filtered_df.empty:
            test_data[symbol] = filtered_df
    
    # Create environment with same parameters as training
    env = TradingEnv(
        df_dict=test_data,
        features_list=feature_list,
        initial_balance=config.get('initial_balance', 100000.0),
        trading_cost=config.get('trading_cost', 0.001),
        slippage=config.get('slippage', 0.0005),
        window_size=config.get('window_size', 50),
        reward_type=config.get('reward_type', 'sharpe'),
        reward_scale=config.get('reward_scale', 1.0),
        allow_short=config.get('allow_short', False)
    )
    
    return env

def visualize_results(eval_metrics: dict, portfolio_history: pd.DataFrame, output_dir: str):
    """
    Visualize the evaluation results.
    
    Args:
        eval_metrics: Dictionary with evaluation metrics
        portfolio_history: DataFrame with portfolio history
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, 'eval_metrics.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=2)
    
    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_history['portfolio_value'])
    plt.title(f'Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'portfolio_value.png'))
    plt.close()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_history['cumulative_returns'] * 100)
    plt.title(f'Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'cumulative_returns.png'))
    plt.close()
    
    # Plot drawdown
    plt.figure(figsize=(12, 6))
    plt.fill_between(portfolio_history.index, 
                    portfolio_history['drawdown'] * 100, 
                    0, 
                    color='red', 
                    alpha=0.3)
    plt.plot(portfolio_history['drawdown'] * 100, color='red')
    plt.title('Portfolio Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'drawdown.png'))
    plt.close()
    
    # Save portfolio history
    portfolio_history.to_csv(os.path.join(output_dir, 'portfolio_history.csv'))
    
    # Create summary report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Portfolio Optimization Results\n")
        f.write("============================\n\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"  Final Portfolio Value: ${eval_metrics.get('final_value', 0):,.2f}\n")
        f.write(f"  Total Return: {eval_metrics.get('total_return', 0)*100:.2f}%\n")
        f.write(f"  Sharpe Ratio: {eval_metrics.get('sharpe_ratio', 0):.4f}\n")
        f.write(f"  Sortino Ratio: {eval_metrics.get('sortino_ratio', 0):.4f}\n")
        f.write(f"  Maximum Drawdown: {eval_metrics.get('max_drawdown', 0)*100:.2f}%\n")
        f.write(f"  Volatility: {eval_metrics.get('volatility', 0)*100:.2f}%\n")
        f.write(f"  Total Trades: {eval_metrics.get('total_trades', 0)}\n")
        f.write(f"  Trading Costs: ${eval_metrics.get('total_costs', 0):,.2f}\n")
    
    print(f"Results saved to {output_dir}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Portfolio Optimization with RL')
    
    parser.add_argument('--config', '-c', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--data_dir', '-d', type=str, required=True,
                        help='Directory containing market data CSV files')
    parser.add_argument('--output_dir', '-o', type=str, default='./rl_results',
                        help='Directory to save results')
    parser.add_argument('--train', action='store_true',
                        help='Train a new model')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate a model')
    parser.add_argument('--model_path', '-m', type=str,
                        help='Path to saved model for evaluation')
    parser.add_argument('--train_start', type=str, default='2018-01-01',
                        help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--train_end', type=str, default='2021-12-31',
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--test_start', type=str, default='2022-01-01',
                        help='Start date for test data (YYYY-MM-DD)')
    parser.add_argument('--test_end', type=str, default='2022-12-31',
                        help='End date for test data (YYYY-MM-DD)')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load market data
    market_data = load_market_data(args.data_dir)
    if not market_data:
        print("No market data found. Exiting.")
        return
    
    # Generate features
    feature_data = generate_features(market_data, config)
    if not feature_data:
        print("Failed to generate features. Exiting.")
        return
    
    # Determine feature list (use the first asset's features as reference)
    first_symbol = next(iter(feature_data))
    all_features = feature_data[first_symbol].columns.tolist()
    
    # Filter to keep only technical features (exclude OHLCV and target columns)
    exclude_patterns = ['open', 'high', 'low', 'close', 'volume', 'label_', 'future_']
    feature_list = [col for col in all_features if not any(pattern in col.lower() for pattern in exclude_patterns)]
    
    # Print feature list
    print(f"Using {len(feature_list)} features for training")
    
    # Save feature list
    with open(os.path.join(args.output_dir, 'feature_list.json'), 'w') as f:
        json.dump(feature_list, f, indent=2)
    
    # Get RL environment parameters from config
    rl_config = config.get('rl', {})
    
    # Create environments
    if args.train or not args.model_path:
        # Create training environment
        train_env = create_training_environment(
            feature_data=feature_data,
            feature_list=feature_list,
            config=rl_config,
            train_start=args.train_start,
            train_end=args.train_end
        )
        
        # Create trainer
        agent_type = rl_config.get('agent_type', 'ppo')
        model_params = rl_config.get('model_params', {})
        
        trainer = AgentTrainer(
            train_env=train_env,
            agent_type=agent_type,
            model_params=model_params,
            output_dir=args.output_dir
        )
        
        # Train model if requested
        if args.train:
            print(f"Training {agent_type.upper()} agent...")
            training_timesteps = rl_config.get('training_timesteps', 100000)
            eval_freq = rl_config.get('eval_freq', 10000)
            save_freq = rl_config.get('save_freq', 50000)
            
            training_metrics = trainer.train(
                total_timesteps=training_timesteps,
                eval_freq=eval_freq,
                save_freq=save_freq
            )
            
            # Save training metrics
            with open(os.path.join(args.output_dir, 'training_metrics.json'), 'w') as f:
                json.dump(training_metrics, f, indent=2)
            
            print(f"Training completed. Model saved to {args.output_dir}")
    
    # Evaluate model if requested
    if args.eval or (args.train and not args.model_path):
        # Create test environment
        test_env = create_test_environment(
            feature_data=feature_data,
            feature_list=feature_list,
            config=rl_config,
            test_start=args.test_start,
            test_end=args.test_end
        )
        
        # If we're only evaluating, create a trainer with the test environment
        if not args.train and args.model_path:
            trainer = AgentTrainer(
                train_env=test_env,
                agent_type=rl_config.get('agent_type', 'ppo'),
                output_dir=args.output_dir
            )
            # Load the model
            trainer.load(args.model_path)
        
        # Evaluate model
        print(f"Evaluating model on test data ({args.test_start} to {args.test_end})...")
        eval_metrics = trainer.evaluate(num_episodes=1)
        
        # Get portfolio history
        portfolio_history = test_env.get_portfolio_history()
        
        # Visualize results
        visualize_results(
            eval_metrics=test_env.get_episode_stats(),
            portfolio_history=portfolio_history,
            output_dir=os.path.join(args.output_dir, 'evaluation')
        )
        
        print("Evaluation completed.")

if __name__ == "__main__":
    main() 