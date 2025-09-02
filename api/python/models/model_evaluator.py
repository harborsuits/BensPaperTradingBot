#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Evaluator Module for Trading Strategies

This module provides robust model evaluation capabilities with time-series awareness,
realistic trade simulation, and comprehensive performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import json
import os
from datetime import datetime
import uuid

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)

class WalkForwardEvaluator:
    """
    Time-series aware evaluator for trading models that prevents look-ahead bias
    and provides realistic performance metrics including slippage and fees.
    """
    
    def __init__(self, 
                 model_trainer,
                 feature_engineer=None,
                 fee_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 params: Dict[str, Any] = None):
        """
        Initialize the walk-forward evaluator.
        
        Args:
            model_trainer: Model trainer instance with train/predict methods
            feature_engineer: Optional feature engineering instance
            fee_rate: Trading fee as a percentage (e.g., 0.001 = 0.1%)
            slippage_rate: Estimated slippage as a percentage
            params: Additional configuration parameters
        """
        self.model_trainer = model_trainer
        self.feature_engineer = feature_engineer
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.params = params or {}
        
        # Results storage
        self.evaluation_results = {}
        self.fold_results = []
        self.trade_history = []
        self.model_ids = []
        
        # Configuration
        self.eval_id = f"eval_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        
    def walk_forward_split(self, 
                          df: pd.DataFrame, 
                          train_size: float = 0.6, 
                          test_size: float = 0.2, 
                          step_size: float = 0.1,
                          min_train_samples: int = 100) -> List[Tuple[slice, slice]]:
        """
        Generate walk-forward train/test splits that respect time order.
        
        Args:
            df: DataFrame with time-series data
            train_size: Proportion of data for training window
            test_size: Proportion of data for testing window
            step_size: Proportion to slide forward between splits
            min_train_samples: Minimum required training samples
            
        Returns:
            List of (train_idx, test_idx) pairs as slices
        """
        n = len(df)
        splits = []

        train_len = max(int(n * train_size), min_train_samples)
        test_len = int(n * test_size)
        step_len = max(int(n * step_size), 1)

        for start in range(0, n - train_len - test_len + 1, step_len):
            train_idx = slice(start, start + train_len)
            test_idx = slice(start + train_len, start + train_len + test_len)
            splits.append((train_idx, test_idx))

        return splits
    
    def evaluate(self, 
                df: pd.DataFrame, 
                target_col: str,
                feature_cols: List[str] = None,
                regime_col: str = None,
                model_type: str = 'classification',
                train_size: float = 0.6,
                test_size: float = 0.2,
                step_size: float = 0.1,
                model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run walk-forward evaluation on the dataset.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            feature_cols: List of feature column names (if None, use all except target)
            regime_col: Optional column indicating market regime 
            model_type: 'classification' or 'regression'
            train_size: Proportion of data for training
            test_size: Proportion of data for testing
            step_size: Step size between folds
            model_params: Parameters to pass to model trainer
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex for time-series evaluation")
            
        # Prepare data
        df = self._prepare_data(df, target_col, feature_cols, regime_col)
        
        # Generate time-based splits
        splits = self.walk_forward_split(df, train_size, test_size, step_size)
        
        # Store evaluation configuration
        self.evaluation_config = {
            'eval_id': self.eval_id,
            'model_type': model_type,
            'train_size': train_size,
            'test_size': test_size,
            'step_size': step_size,
            'fee_rate': self.fee_rate,
            'slippage_rate': self.slippage_rate,
            'n_folds': len(splits),
            'feature_count': len(df.columns) - (1 + (1 if regime_col else 0)),
            'start_date': df.index[0].strftime('%Y-%m-%d'),
            'end_date': df.index[-1].strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"Starting walk-forward evaluation with {len(splits)} folds")
        
        # Run evaluation for each fold
        self.fold_results = []
        for i, (train_idx, test_idx) in enumerate(splits):
            fold_result = self._evaluate_fold(
                df, 
                train_idx, 
                test_idx, 
                target_col, 
                regime_col,
                model_type,
                model_params,
                fold_id=i
            )
            self.fold_results.append(fold_result)
            
            # Print progress
            print(f"Fold {i+1}/{len(splits)} completed: "
                  f"Sharpe={fold_result['metrics']['sharpe_ratio']:.2f}, "
                  f"PnL={fold_result['metrics']['total_return']:.2%}, "
                  f"Trades={fold_result['metrics']['trade_count']}")
                  
        # Combine results from all folds
        final_results = self._combine_fold_results()
        
        # Store complete evaluation
        self.evaluation_results = {
            'config': self.evaluation_config,
            'overall_metrics': final_results,
            'fold_results': self.fold_results,
            'model_ids': self.model_ids
        }
        
        return final_results
    
    def _prepare_data(self, 
                     df: pd.DataFrame, 
                     target_col: str,
                     feature_cols: List[str] = None,
                     regime_col: str = None) -> pd.DataFrame:
        """
        Prepare data for evaluation, ensuring no lookahead bias.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: Feature column names
            regime_col: Regime column name
            
        Returns:
            Processed DataFrame
        """
        df = df.copy()
        
        # Ensure the target exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
            
        # If feature columns not specified, use all except target
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col != target_col]
            if regime_col:
                feature_cols = [col for col in feature_cols if col != regime_col]
                
        # Add regime col to features if it exists
        used_cols = feature_cols.copy()
        if regime_col and regime_col in df.columns:
            used_cols.append(regime_col)
        
        # Add target column
        used_cols.append(target_col)
        
        # Only keep necessary columns
        result_df = df[used_cols].copy()
        
        # Ensure no NaN values
        result_df = result_df.dropna()
        
        return result_df
        
    def _evaluate_fold(self, 
                      df: pd.DataFrame,
                      train_idx: slice,
                      test_idx: slice,
                      target_col: str,
                      regime_col: str = None,
                      model_type: str = 'classification',
                      model_params: Dict[str, Any] = None,
                      fold_id: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single walk-forward fold.
        
        Args:
            df: DataFrame with features and target
            train_idx: Training data slice
            test_idx: Testing data slice
            target_col: Target column name
            regime_col: Regime column name (if any)
            model_type: 'classification' or 'regression'
            model_params: Model trainer parameters
            fold_id: Fold identifier
            
        Returns:
            Dictionary with fold evaluation results
        """
        # Split data
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        # Prepare feature matrix and target
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Generate any additional features if feature engineer exists
        if self.feature_engineer:
            try:
                # Generate features, being careful not to leak test data into training
                X_train = self.feature_engineer.generate_features(X_train)
                X_test = self.feature_engineer.generate_features(X_test)
            except Exception as e:
                print(f"Error in feature generation: {str(e)}")
        
        # Train model for this fold
        if model_type == 'classification':
            # For classification, we need to handle different regimes if specified
            if regime_col and regime_col in X_train.columns:
                # Train regime-specific models
                model_id = f"fold_{fold_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                self.model_trainer.build_regime_ensemble(
                    X_train, y_train, regime_col, 
                    model_type='classification',
                    base_name=f"regime_{model_id}",
                    include_meta=True
                )
                self.model_ids.append(model_id)
                
                # Make predictions using ensemble
                y_pred = self.model_trainer.ensemble_predict(
                    X_test, 
                    regime_column=regime_col,
                    base_name=f"regime_{model_id}",
                    model_type='classification'
                )
                
                # Get probabilities if available for more realistic trade simulation
                try:
                    y_proba = self.model_trainer.ensemble_predict_proba(
                        X_test, 
                        regime_column=regime_col,
                        base_name=f"regime_{model_id}"
                    )[:, 1]  # Probability of positive class
                except:
                    y_proba = None
            else:
                # Train a single model
                model_id = f"fold_{fold_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                model = self.model_trainer.train_model(
                    X_train, y_train, 
                    model_type='classification',
                    model_name=model_id
                )
                self.model_ids.append(model_id)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Get probabilities if available
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
                except:
                    y_proba = None
        else:
            # For regression, similar approach but different evaluation methods
            if regime_col and regime_col in X_train.columns:
                # Train regime-specific models
                model_id = f"fold_{fold_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                self.model_trainer.build_regime_ensemble(
                    X_train, y_train, regime_col, 
                    model_type='regression',
                    base_name=f"regime_{model_id}",
                    include_meta=True
                )
                self.model_ids.append(model_id)
                
                # Make predictions using ensemble
                y_pred = self.model_trainer.ensemble_predict(
                    X_test, 
                    regime_column=regime_col,
                    base_name=f"regime_{model_id}",
                    model_type='regression'
                )
                y_proba = None
            else:
                # Train a single model
                model_id = f"fold_{fold_id}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                model = self.model_trainer.train_model(
                    X_train, y_train, 
                    model_type='regression',
                    model_name=model_id
                )
                self.model_ids.append(model_id)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_proba = None
        
        # Calculate ML metrics
        ml_metrics = self._calculate_ml_metrics(y_test, y_pred, model_type)
        
        # Simulate trades and calculate performance metrics
        close_prices = test_df['close'] if 'close' in test_df.columns else None
        trade_metrics, trades = self._simulate_trades(
            y_test, y_pred, y_proba, close_prices, model_type
        )
        
        # Store trades from this fold
        for trade in trades:
            trade['fold_id'] = fold_id
            trade['model_id'] = model_id
            self.trade_history.append(trade)
        
        # Combine all metrics
        metrics = {**ml_metrics, **trade_metrics}
        
        # Create fold result
        fold_result = {
            'fold_id': fold_id,
            'model_id': model_id,
            'train_start': train_df.index[0],
            'train_end': train_df.index[-1],
            'test_start': test_df.index[0],
            'test_end': test_df.index[-1],
            'metrics': metrics,
            'trade_count': len(trades)
        }
        
        return fold_result
    
    def _calculate_ml_metrics(self, 
                            y_true: pd.Series, 
                            y_pred: np.ndarray,
                            model_type: str) -> Dict[str, float]:
        """
        Calculate machine learning performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_type: 'classification' or 'regression'
            
        Returns:
            Dictionary of ML metrics
        """
        if model_type == 'classification':
            # Classification metrics
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
        else:
            # Regression metrics
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
        return metrics
    
    def _simulate_trades(self, 
                       y_true: pd.Series, 
                       y_pred: np.ndarray,
                       y_proba: np.ndarray = None,
                       close_prices: pd.Series = None,
                       model_type: str = 'classification') -> Tuple[Dict[str, float], List[Dict]]:
        """
        Simulate trades with realistic execution including fees and slippage.
        
        Args:
            y_true: True target values (actual returns for regression)
            y_pred: Predicted values (signals for classification, predicted returns for regression)
            y_proba: Prediction probabilities (for confidence filtering)
            close_prices: Close prices for the test period
            model_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (metrics dictionary, list of trade records)
        """
        # For trading simulation we need prices
        if close_prices is None:
            # Fall back to basic metrics if no prices provided
            return self._calculate_basic_trade_metrics(y_true, y_pred, model_type), []
            
        # Convert everything to numpy for speed
        y_true = y_true.values if isinstance(y_true, pd.Series) else y_true
        close = close_prices.values if isinstance(close_prices, pd.Series) else close_prices
        dates = close_prices.index if isinstance(close_prices, pd.Series) else None
        
        # Parameters for trade simulation
        confidence_threshold = self.params.get('confidence_threshold', 0.55)
        position_size = self.params.get('position_size', 1.0)
        allow_short = self.params.get('allow_short', True)
        
        # Lists to store trade information
        pnl_list = []
        trade_records = []
        positions = np.zeros(len(y_true))
        entry_prices = np.zeros(len(y_true))
        
        # Simulate trades based on model type
        if model_type == 'classification':
            # For classification, simulate entry/exit based on predicted class
            for i in range(len(y_pred)):
                # Use probability threshold if available
                if y_proba is not None:
                    # Only take trades with sufficient confidence
                    if y_pred[i] == 1 and y_proba[i] >= confidence_threshold:
                        signal = 1  # Long
                    elif y_pred[i] == 0 and y_proba[i] <= (1 - confidence_threshold):
                        signal = -1 if allow_short else 0  # Short or flat
                    else:
                        signal = 0  # No trade (insufficient confidence)
                else:
                    # Use predictions directly if no probabilities
                    if y_pred[i] == 1:
                        signal = 1  # Long
                    elif y_pred[i] == 0 and allow_short:
                        signal = -1  # Short
                    else:
                        signal = 0  # No trade
                
                # Process the signal
                if signal != 0 and positions[i-1] != signal:
                    # Calculate entry with slippage and fees
                    entry_price = close[i] * (1 + signal * self.slippage_rate)
                    entry_price *= (1 + self.fee_rate)  # Add transaction fee
                    
                    # Record entry
                    positions[i] = signal
                    entry_prices[i] = entry_price
                    
                    # Create trade record
                    trade = {
                        'entry_date': dates[i] if dates is not None else i,
                        'entry_price': entry_price,
                        'position': signal,
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': None,
                        'pnl_pct': None,
                        'bars_held': None
                    }
                    trade_records.append(trade)
                elif signal == 0 and positions[i-1] != 0:
                    # Exit position
                    exit_price = close[i] * (1 - positions[i-1] * self.slippage_rate)
                    exit_price *= (1 - self.fee_rate)  # Subtract transaction fee
                    
                    # Calculate PnL
                    for trade in reversed(trade_records):
                        if trade['exit_date'] is None:
                            entry = trade['entry_price']
                            pnl = (exit_price - entry) * trade['position'] * position_size
                            pnl_pct = pnl / entry
                            
                            # Update trade record
                            trade['exit_date'] = dates[i] if dates is not None else i
                            trade['exit_price'] = exit_price
                            trade['pnl'] = pnl
                            trade['pnl_pct'] = pnl_pct
                            trade['bars_held'] = i - (trade['entry_date'] if isinstance(trade['entry_date'], int) else 0)
                            
                            # Add to PnL list
                            pnl_list.append(pnl_pct)
                            break
                    
                    # Reset position
                    positions[i] = 0
                    entry_prices[i] = 0
                else:
                    # Maintain previous position
                    positions[i] = positions[i-1]
                    entry_prices[i] = entry_prices[i-1]
        else:
            # For regression, use predicted return as signal strength
            for i in range(len(y_pred)):
                # Skip if not enough predicted return to overcome friction
                min_return = self.fee_rate + self.slippage_rate
                
                if y_pred[i] > min_return:
                    # Long position
                    signal = 1
                elif y_pred[i] < -min_return and allow_short:
                    # Short position
                    signal = -1
                else:
                    # No trade
                    signal = 0
                
                # Apply the same logic as classification for entries and exits
                if signal != 0 and positions[i-1] != signal:
                    # Calculate entry with slippage and fees
                    entry_price = close[i] * (1 + signal * self.slippage_rate)
                    entry_price *= (1 + self.fee_rate)  # Add transaction fee
                    
                    # Record entry
                    positions[i] = signal
                    entry_prices[i] = entry_price
                    
                    # Create trade record
                    trade = {
                        'entry_date': dates[i] if dates is not None else i,
                        'entry_price': entry_price,
                        'position': signal,
                        'predicted_return': y_pred[i],
                        'exit_date': None,
                        'exit_price': None,
                        'pnl': None,
                        'pnl_pct': None,
                        'bars_held': None
                    }
                    trade_records.append(trade)
                elif signal == 0 and positions[i-1] != 0:
                    # Exit position
                    exit_price = close[i] * (1 - positions[i-1] * self.slippage_rate)
                    exit_price *= (1 - self.fee_rate)  # Subtract transaction fee
                    
                    # Calculate PnL
                    for trade in reversed(trade_records):
                        if trade['exit_date'] is None:
                            entry = trade['entry_price']
                            pnl = (exit_price - entry) * trade['position'] * position_size
                            pnl_pct = pnl / entry
                            
                            # Update trade record
                            trade['exit_date'] = dates[i] if dates is not None else i
                            trade['exit_price'] = exit_price
                            trade['pnl'] = pnl
                            trade['pnl_pct'] = pnl_pct
                            trade['bars_held'] = i - (trade['entry_date'] if isinstance(trade['entry_date'], int) else 0)
                            
                            # Add to PnL list
                            pnl_list.append(pnl_pct)
                            break
                    
                    # Reset position
                    positions[i] = 0
                    entry_prices[i] = 0
                else:
                    # Maintain previous position
                    positions[i] = positions[i-1]
                    entry_prices[i] = entry_prices[i-1]
        
        # Close any open positions at the end
        for i in range(len(trade_records)):
            if trade_records[i]['exit_date'] is None:
                # Use last price for exit
                exit_price = close[-1] * (1 - trade_records[i]['position'] * self.slippage_rate)
                exit_price *= (1 - self.fee_rate)  # Subtract transaction fee
                
                entry = trade_records[i]['entry_price']
                pnl = (exit_price - entry) * trade_records[i]['position'] * position_size
                pnl_pct = pnl / entry
                
                # Update trade record
                trade_records[i]['exit_date'] = dates[-1] if dates is not None else len(close)-1
                trade_records[i]['exit_price'] = exit_price
                trade_records[i]['pnl'] = pnl
                trade_records[i]['pnl_pct'] = pnl_pct
                trade_records[i]['bars_held'] = len(close) - (trade_records[i]['entry_date'] if isinstance(trade_records[i]['entry_date'], int) else 0)
                
                # Add to PnL list
                pnl_list.append(pnl_pct)
        
        # Calculate trading metrics
        metrics = self._calculate_trading_metrics(pnl_list)
        
        return metrics, trade_records
    
    def _calculate_basic_trade_metrics(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     model_type: str) -> Dict[str, float]:
        """
        Calculate basic trade metrics when price data isn't available.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with basic metrics
        """
        # For classification, estimate win rate and accuracy
        if model_type == 'classification':
            correct_signals = (y_pred == y_true).mean()
            return {
                'estimated_win_rate': correct_signals,
                'sharpe_ratio': 0,  # Can't calculate without returns
                'total_return': 0,
                'max_drawdown': 0,
                'trade_count': sum(y_pred == 1)
            }
        else:
            # For regression, use correlation and direction accuracy
            direction_accuracy = (np.sign(y_pred) == np.sign(y_true)).mean()
            return {
                'return_correlation': np.corrcoef(y_true, y_pred)[0, 1],
                'direction_accuracy': direction_accuracy,
                'sharpe_ratio': 0,  # Can't calculate without trade simulation
                'total_return': 0,
                'max_drawdown': 0,
                'trade_count': sum(np.abs(y_pred) > 0)
            }
    
    def _calculate_trading_metrics(self, pnl_list: List[float]) -> Dict[str, float]:
        """
        Calculate trading performance metrics from the PnL list.
        
        Args:
            pnl_list: List of percentage returns from trades
            
        Returns:
            Dictionary with trading metrics
        """
        if not pnl_list:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'trade_count': 0
            }
            
        # Convert to numpy array for calculations
        pnl_array = np.array(pnl_list)
        
        # Calculate metrics
        total_return = np.sum(pnl_array)
        win_rate = np.mean(pnl_array > 0) if len(pnl_array) > 0 else 0
        
        # Separate wins and losses
        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array < 0]
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        # Calculate profit factor (sum of gains / sum of losses)
        sum_wins = np.sum(wins) if len(wins) > 0 else 0
        sum_losses = abs(np.sum(losses)) if len(losses) > 0 else 1  # Avoid division by zero
        profit_factor = sum_wins / sum_losses if sum_losses != 0 else 0
        
        # Calculate Sharpe ratio (annualized)
        sharpe = 0
        if len(pnl_array) > 1 and np.std(pnl_array) > 0:
            # Assuming 252 trading days per year
            trading_periods_per_year = self.params.get('trading_periods_per_year', 252)
            sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(trading_periods_per_year)
        
        # Calculate drawdown
        cum_returns = np.cumsum(pnl_array)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = peak - cum_returns
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trade_count': len(pnl_array)
        }
    
    def _combine_fold_results(self) -> Dict[str, float]:
        """
        Combine results from all folds into an overall evaluation.
        
        Returns:
            Dictionary with aggregated metrics
        """
        if not self.fold_results:
            return {}
            
        # Extract metrics from all folds
        all_metrics = [fold['metrics'] for fold in self.fold_results]
        
        # Calculate aggregate metrics
        result = {}
        
        # Trading metrics - focus on these for final evaluation
        result['total_return'] = sum(m.get('total_return', 0) for m in all_metrics)
        result['avg_sharpe_ratio'] = np.mean([m.get('sharpe_ratio', 0) for m in all_metrics])
        result['avg_max_drawdown'] = np.mean([m.get('max_drawdown', 0) for m in all_metrics])
        result['avg_win_rate'] = np.mean([m.get('win_rate', 0) for m in all_metrics])
        
        total_trades = sum(m.get('trade_count', 0) for m in all_metrics)
        result['total_trades'] = total_trades
        
        # ML metrics - these are secondary for trading
        if 'accuracy' in all_metrics[0]:
            # Classification metrics
            result['avg_accuracy'] = np.mean([m.get('accuracy', 0) for m in all_metrics])
            result['avg_f1'] = np.mean([m.get('f1', 0) for m in all_metrics])
        else:
            # Regression metrics
            result['avg_r2'] = np.mean([m.get('r2', 0) for m in all_metrics])
            result['avg_mse'] = np.mean([m.get('mse', 0) for m in all_metrics])
            
        return result
    
    def save_results(self, filepath: str = None) -> str:
        """
        Save evaluation results to a file.
        
        Args:
            filepath: Path where to save the results
            
        Returns:
            Path where results were saved
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results to save. Run evaluate() first.")
            
        if filepath is None:
            # Create default filepath
            output_dir = self.params.get('output_dir', './results')
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"walkforward_eval_{self.eval_id}.json")
            
        # Convert datetime objects to strings
        serializable_results = self._make_json_serializable(self.evaluation_results)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        return filepath
    
    def _make_json_serializable(self, data: Any) -> Any:
        """
        Make data JSON serializable by converting datetime objects to ISO format.
        
        Args:
            data: Any data structure to convert
            
        Returns:
            JSON serializable data
        """
        if isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif isinstance(data, (pd.Timestamp, datetime)):
            return data.isoformat()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return self._make_json_serializable(data.tolist())
        else:
            return data
            
    def plot_results(self, plot_type: str = 'equity_curve', save_path: str = None):
        """
        Plot evaluation results.
        
        Args:
            plot_type: Type of plot ('equity_curve', 'drawdown', 'trade_dist', 'fold_comparison')
            save_path: Path to save the plot (if None, just display it)
            
        Returns:
            matplotlib Figure object
        """
        if not self.fold_results:
            raise ValueError("No evaluation results to plot. Run evaluate() first.")
            
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'equity_curve':
            # Plot equity curve across all folds
            plt.title('Simulated Equity Curve', fontsize=14)
            
            # Combine all trades in chronological order
            all_trades = sorted(self.trade_history, key=lambda x: x['entry_date'])
            
            # Calculate cumulative returns
            returns = [t['pnl_pct'] for t in all_trades if t['pnl_pct'] is not None]
            cumulative = np.cumprod(1 + np.array(returns)) - 1
            
            plt.plot(cumulative, linewidth=2)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
            plt.ylabel('Cumulative Return (%)', fontsize=12)
            plt.xlabel('Trade Number', fontsize=12)
            plt.grid(True, alpha=0.3)
            
        elif plot_type == 'drawdown':
            # Plot drawdown chart
            plt.title('Drawdown Analysis', fontsize=14)
            
            # Get returns from all trades
            returns = [t['pnl_pct'] for t in self.trade_history if t['pnl_pct'] is not None]
            
            # Calculate equity curve and drawdown
            equity = np.cumprod(1 + np.array(returns))
            previous_peaks = np.maximum.accumulate(equity)
            drawdowns = (equity - previous_peaks) / previous_peaks
            
            plt.plot(drawdowns, linewidth=2, color='r')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.ylabel('Drawdown (%)', fontsize=12)
            plt.xlabel('Trade Number', fontsize=12)
            plt.grid(True, alpha=0.3)
            
        elif plot_type == 'trade_dist':
            # Plot distribution of trade returns
            plt.title('Trade Return Distribution', fontsize=14)
            
            returns = [t['pnl_pct'] for t in self.trade_history if t['pnl_pct'] is not None]
            
            plt.hist(returns, bins=50, alpha=0.7, color='blue')
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=np.mean(returns), color='g', linestyle='-', alpha=0.8, 
                        label=f'Mean: {np.mean(returns):.2%}')
            
            plt.xlabel('Trade Return (%)', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif plot_type == 'fold_comparison':
            # Compare performance across folds
            plt.title('Performance by Fold', fontsize=14)
            
            fold_ids = [r['fold_id'] for r in self.fold_results]
            returns = [r['metrics'].get('total_return', 0) for r in self.fold_results]
            sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in self.fold_results]
            
            x = np.arange(len(fold_ids))
            width = 0.35
            
            plt.bar(x - width/2, returns, width, label='Return')
            plt.bar(x + width/2, sharpes, width, label='Sharpe')
            
            plt.xlabel('Fold', fontsize=12)
            plt.xticks(x)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()  # Return the figure
        
    def get_best_model_id(self) -> str:
        """
        Get the ID of the best performing model across all folds.
        
        Returns:
            Best model ID
        """
        if not self.fold_results:
            raise ValueError("No evaluation results. Run evaluate() first.")
            
        # Choose best model based on Sharpe ratio
        best_fold = max(self.fold_results, 
                        key=lambda x: x['metrics'].get('sharpe_ratio', 0))
        
        return best_fold['model_id'] 