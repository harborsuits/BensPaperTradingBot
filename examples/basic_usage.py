#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic usage example for the Trading Bot.

This example demonstrates how to use the trading bot components programmatically
rather than through the command-line interface.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Import trading bot components
from trading_bot.utils.feature_engineering import FeatureEngineering
from trading_bot.models.model_trainer import ModelTrainer
from trading_bot.analysis.trade_analyzer import TradeAnalyzer
from trading_bot.visualization.model_dashboard import ModelDashboard

# Create output directory if it doesn't exist
os.makedirs('./output', exist_ok=True)

def generate_sample_data(n_samples=1000):
    """Generate sample OHLCV data for demonstration."""
    
    # Start with a random walk for close prices
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.normal(0.0005, 0.01, n_samples))
    
    # Generate other OHLCV data
    high = close * (1 + np.random.uniform(0, 0.015, n_samples))
    low = close * (1 - np.random.uniform(0, 0.015, n_samples))
    open_price = low + np.random.uniform(0, 1, n_samples) * (high - low)
    volume = np.random.uniform(1000000, 5000000, n_samples)
    
    # Create DataFrame with date index
    date_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=date_index)
    
    return df

def main():
    """Example workflow with the trading bot."""
    print("Generating sample market data...")
    data = generate_sample_data()
    
    # Save sample data
    os.makedirs('./data', exist_ok=True)
    data.to_csv('./data/sample_data.csv')
    
    # 1. Initialize components
    print("\nInitializing components...")
    
    # Feature engineering parameters
    fe_params = {
        'include_technicals': True,
        'ta_feature_sets': ['momentum', 'trend', 'volatility'],
        'lookback_periods': [5, 10, 20, 50],
        'include_lags': True,
        'lags': [1, 3, 5],
        'normalize_features': True,
        'detect_market_regime': True,
        'feature_selection': 'importance',
        'max_features': 50,
        'output_dir': './output/features'
    }
    
    # Model trainer parameters
    mt_params = {
        'model_algorithm': 'random_forest',
        'n_estimators': 100,
        'max_depth': 8,
        'cv_splits': 3,
        'calculate_shap': True,
        'class_weight': 'balanced',
        'min_regime_samples': 30,
        'random_seed': 42,
        'output_dir': './output/models',
        'model_dir': './output/models'
    }
    
    # Trade analyzer parameters
    ta_params = {
        'log_dir': './output/logs/trades',
        'log_save_frequency': 10
    }
    
    # Visualization parameters
    vd_params = {
        'output_dir': './output/visualizations'
    }
    
    # Initialize components
    feature_engineering = FeatureEngineering(fe_params)
    model_trainer = ModelTrainer(mt_params)
    trade_analyzer = TradeAnalyzer(ta_params)
    model_dashboard = ModelDashboard(vd_params)
    
    # Connect visualization dashboard to other components
    model_dashboard.connect_components(
        trade_analyzer=trade_analyzer,
        model_trainer=model_trainer,
        feature_engineering=feature_engineering
    )
    
    # 2. Generate features
    print("Generating features...")
    features_df = feature_engineering.generate_features(data)
    
    # 3. Add return labels
    print("Adding return labels...")
    labeled_df = feature_engineering.add_return_labels(
        df=features_df,
        future_windows=[5],
        thresholds=[0.0, 0.01, 0.02]
    )
    
    # Choose target column for binary classification
    target_column = 'label_5d_1pct'  # 1% threshold, 5-day horizon
    
    # 4. Prepare ML dataset
    print("Preparing ML dataset...")
    X, y, metadata = feature_engineering.to_ml_dataset(labeled_df, target_column)
    
    # Split into train/test
    train_size = int(len(X) * 0.8)
    X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
    X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]
    
    # 5. Train model
    print("Training model...")
    model = model_trainer.train_model(
        X=X_train,
        y=y_train,
        model_type='classification',
        model_name='sample_model'
    )
    
    # Perform cross-validation
    print("Performing cross-validation...")
    cv_results = model_trainer.time_series_cv(
        X=X_train,
        y=y_train,
        model_type='classification',
        model_name='sample_model'
    )
    
    # Train regime-specific models if regime data is available
    if 'market_regime' in labeled_df.columns:
        print("Training regime-specific models...")
        regime_models = model_trainer.train_regime_specific_models(
            X=X_train.copy(),
            y=y_train.copy(),
            regime_column='market_regime',
            model_type='classification',
            base_model_name='sample_model'
        )
    
    # 6. Evaluate model
    print("Evaluating model...")
    evaluation = model_trainer.evaluate_model(
        X=X_test,
        y=y_test,
        model_name='sample_model'
    )
    
    # Print evaluation results
    print("\nEvaluation results:")
    for metric, value in evaluation.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
    
    # 7. Get feature importance
    top_features = model_trainer.get_top_features(
        model_name='sample_model',
        top_n=10
    )
    
    print("\nTop 10 features:")
    for feature, importance in top_features.items():
        print(f"  {feature}: {importance:.4f}")
    
    # 8. Run a simple backtest on test data
    print("\nRunning backtest on test data...")
    
    # Perform predictions for each test sample
    for i in range(len(X_test)):
        # Get features for current period
        current_features = X_test.iloc[i:i+1]
        current_timestamp = X_test.index[i]
        
        # Get regime if available
        regime = None
        if 'market_regime' in labeled_df.columns:
            regime_idx = train_size + i
            if regime_idx < len(labeled_df):
                regime = labeled_df.iloc[regime_idx]['market_regime']
        
        # Get prediction and explanation
        try:
            # For classification with probabilities
            pred_proba = model_trainer.predict_proba(
                current_features, 
                model_name='sample_model', 
                regime=regime
            )
            prediction = model_trainer.predict(
                current_features, 
                model_name='sample_model', 
                regime=regime
            )[0]
            confidence = np.max(pred_proba[0])
            
            # Get feature explanation
            explanations = model_trainer.get_feature_explanation(
                current_features, 
                model_name='sample_model', 
                regime=regime
            )
            top_features = explanations[0]['top_features'] if explanations else {}
            
            # Log prediction
            prediction_entry = trade_analyzer.log_prediction(
                timestamp=current_timestamp,
                features=current_features,
                prediction=prediction,
                confidence=confidence,
                top_features=top_features,
                regime=regime if regime else 'unknown',
                model_name='sample_model',
                metadata={'price': data.iloc[train_size + i]['close']}
            )
            
            # Log outcome (actual)
            if i < len(X_test) - 5:  # If we have future data
                future_idx = train_size + i + 5  # 5-day horizon
                if future_idx < len(data):
                    future_return = data.iloc[future_idx]['close'] / data.iloc[train_size + i]['close'] - 1
                    actual_outcome = 1 if future_return > 0.01 else (-1 if future_return < -0.01 else 0)
                    pnl = future_return if prediction == np.sign(future_return) else -future_return
                    
                    trade_analyzer.log_trade_outcome(
                        prediction_id=current_timestamp,
                        actual_outcome=actual_outcome,
                        pnl=pnl,
                        trade_metadata={'future_price': data.iloc[future_idx]['close']}
                    )
                
        except Exception as e:
            print(f"Error in prediction at {current_timestamp}: {str(e)}")
    
    # 9. Create performance dashboard
    print("\nCreating performance dashboard...")
    dashboard_paths = model_dashboard.create_dashboard(interactive=True)
    
    # Print dashboard location
    if 'index' in dashboard_paths:
        print(f"Dashboard created at: {dashboard_paths['index']}")
    
    # 10. Performance analysis
    print("\nAnalyzing performance...")
    
    # Overall performance
    performance = trade_analyzer.analyze_model_performance()
    
    print("\nBacktest Performance:")
    print(f"Total trades: {performance['total_trades']}")
    print(f"Accuracy: {performance['accuracy']:.4f}")
    print(f"Win rate: {performance['win_rate']:.4f}")
    print(f"Profit factor: {performance['profit_factor']:.4f}")
    print(f"Total P&L: {performance['total_pnl']:.4f}")
    
    # 11. Plot a sample trade explanation
    if len(trade_analyzer.trades_history) > 0:
        print("\nPlotting a sample trade explanation...")
        fig = model_dashboard.plot_trade_explanations(n_trades=1)
        fig.savefig('./output/sample_trade_explanation.png')
        print("Sample trade explanation saved to ./output/sample_trade_explanation.png")
    
    print("\nExample completed successfully.")

if __name__ == "__main__":
    main() 