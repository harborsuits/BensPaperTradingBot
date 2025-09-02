#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated backtester for machine learning-based trading strategies.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from trading_bot.backtesting.backtester import Backtester
from trading_bot.backtesting.data_manager import BacktestDataManager
from trading_bot.backtesting.learner import BacktestLearner

logger = logging.getLogger(__name__)

class AutomatedBacktester:
    """
    Automated backtester for machine learning-based trading strategies.
    
    This class combines the functionality of BacktestDataManager, BacktestLearner,
    and Backtester to create a complete machine learning workflow for trading strategies.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        data_dir: str = "data",
        models_dir: str = "models",
        results_dir: str = "results"
    ):
        """
        Initialize the AutomatedBacktester.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data
            start_date: Start date for backtesting
            end_date: End date for backtesting
            data_dir: Directory for data storage
            models_dir: Directory for model storage
            results_dir: Directory for results storage
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        for directory in [data_dir, models_dir, results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Initialize components
        self.data_manager = BacktestDataManager(data_dir=data_dir)
        self.learner = BacktestLearner(models_dir=models_dir)
        self.backtester = Backtester()
        
        # Initialize tracking dictionaries
        self.experiments = {}
        self.experiment_results = {}
        
        # Load experiments if they exist
        experiments_path = os.path.join(results_dir, "experiments.json")
        if os.path.exists(experiments_path):
            with open(experiments_path, 'r') as f:
                self.experiments = json.load(f)
    
    def prepare_data(
        self,
        data_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for backtesting.
        
        Args:
            data_config: Configuration for data preparation
                - indicators: List of technical indicators to add
                - target_type: Type of target variable (e.g., 'direction', 'return', 'classification')
                - target_lookahead: Number of periods to look ahead for target
                - target_threshold: Threshold for classification targets
                - split_method: Method for splitting data ('time', 'random', 'walk_forward')
                - test_size: Size of the test set
                - val_size: Size of the validation set
                - scaling_method: Method for scaling features
                - custom_features: Configuration for custom features
        
        Returns:
            Dictionary with prepared data
        """
        # Load data
        df = self.data_manager.load_market_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Process data according to configuration
        data_result = self.data_manager.prepare_data(
            df=df,
            indicators=data_config.get('indicators', []),
            target_type=data_config.get('target_type', 'direction'),
            target_lookahead=data_config.get('target_lookahead', 1),
            target_threshold=data_config.get('target_threshold', 0.0),
            split_method=data_config.get('split_method', 'time'),
            test_size=data_config.get('test_size', 0.2),
            val_size=data_config.get('val_size', 0.1),
            scaling_method=data_config.get('scaling_method', 'standard'),
            custom_features=data_config.get('custom_features', {})
        )
        
        return data_result
    
    def train_model(
        self,
        data_result: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a machine learning model.
        
        Args:
            data_result: Result from prepare_data
            model_config: Configuration for model training
                - model_type: Type of model to train
                - model_params: Parameters for the model
                - model_name: Name for the model
                - hyperparameter_tuning: Configuration for hyperparameter tuning
        
        Returns:
            Dictionary with trained model and metadata
        """
        # Extract data from data_result
        X_train = data_result['X_train']
        y_train = data_result['y_train']
        X_val = data_result.get('X_val')
        y_val = data_result.get('y_val')
        feature_names = data_result['feature_names']
        
        # Train model
        model_result = self.learner.train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_type=model_config.get('model_type', 'random_forest_classifier'),
            model_params=model_config.get('model_params', {}),
            model_name=model_config.get('model_name'),
            feature_names=feature_names,
            hyperparameter_tuning=model_config.get('hyperparameter_tuning')
        )
        
        return model_result
    
    def backtest_model(
        self,
        data_result: Dict[str, Any],
        model_result: Dict[str, Any],
        backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Backtest a trained model.
        
        Args:
            data_result: Result from prepare_data
            model_result: Result from train_model
            backtest_config: Configuration for backtesting
                - initial_capital: Initial capital for backtesting
                - commission: Commission rate
                - slippage: Slippage amount
                - prediction_threshold: Threshold for binary classification
                - entry_rules: Additional entry rules
                - exit_rules: Additional exit rules
                - position_sizing: Position sizing rules
        
        Returns:
            Dictionary with backtest results
        """
        # Extract model and data
        model = model_result['model']
        model_metadata = model_result['metadata']
        is_classifier = model_metadata['is_classifier']
        
        # Get test data
        X_test = data_result['X_test']
        y_test = data_result['y_test']
        test_indices = data_result['test_indices']
        
        # Get original data
        df = data_result['original_data']
        
        # Generate predictions
        prediction_threshold = backtest_config.get('prediction_threshold', 0.5)
        predictions = self.learner.generate_predictions(
            model=model,
            X=X_test,
            is_classifier=is_classifier,
            prediction_threshold=prediction_threshold
        )
        
        # Add predictions to test data
        test_df = df.iloc[test_indices].copy()
        test_df['prediction'] = predictions
        
        # Create strategy from predictions
        entry_rules = backtest_config.get('entry_rules', {})
        exit_rules = backtest_config.get('exit_rules', {})
        position_sizing = backtest_config.get('position_sizing', {"type": "fixed", "value": 1.0})
        
        # Set up basic rules based on model type
        if is_classifier:
            # For classifier, use class labels
            if 'prediction' not in entry_rules:
                entry_rules['prediction'] = 1  # Long position if prediction is 1
            
            if 'prediction' not in exit_rules:
                exit_rules['prediction'] = 0  # Exit if prediction is 0
        else:
            # For regressor, use prediction value
            if 'prediction' not in entry_rules:
                entry_rules['prediction_value'] = "> 0"  # Long position if prediction > 0
            
            if 'prediction' not in exit_rules:
                exit_rules['prediction_value'] = "< 0"  # Exit if prediction < 0
        
        # Create strategy configuration
        strategy_config = {
            "name": f"ML_{model_metadata['model_name']}",
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "position_sizing": position_sizing
        }
        
        # Run backtest
        backtest_result = self.backtester.run_backtest(
            df=test_df,
            strategy_config=strategy_config,
            initial_capital=backtest_config.get('initial_capital', 10000),
            commission=backtest_config.get('commission', 0.001),
            slippage=backtest_config.get('slippage', 0.0)
        )
        
        # Evaluate model on test data
        model_metrics = self.learner.evaluate_model(
            model=model,
            X=X_test,
            y=y_test,
            is_classifier=is_classifier,
            prediction_threshold=prediction_threshold
        )
        
        # Calculate feature importance
        feature_importance = self.learner.feature_importance(
            model=model,
            feature_names=data_result['feature_names']
        )
        
        # Combine results
        result = {
            "backtest_result": backtest_result,
            "model_metrics": model_metrics,
            "feature_importance": feature_importance,
            "predictions": predictions.tolist(),
            "true_values": y_test.tolist(),
            "test_data": test_df.to_dict(orient='records')
        }
        
        return result
    
    def run_experiment(
        self,
        experiment_name: str,
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        backtest_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a complete experiment.
        
        Args:
            experiment_name: Name for the experiment
            data_config: Configuration for data preparation
            model_config: Configuration for model training
            backtest_config: Configuration for backtesting
        
        Returns:
            Dictionary with experiment results
        """
        # Create a timestamp for the experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up the model name if not provided
        if 'model_name' not in model_config:
            model_config['model_name'] = f"{experiment_name}_{timestamp}"
        
        # Step 1: Prepare data
        logger.info(f"Preparing data for experiment '{experiment_name}'")
        data_result = self.prepare_data(data_config)
        
        # Step 2: Train model
        logger.info(f"Training model for experiment '{experiment_name}'")
        model_result = self.train_model(data_result, model_config)
        
        # Step 3: Backtest model
        logger.info(f"Backtesting model for experiment '{experiment_name}'")
        backtest_result = self.backtest_model(data_result, model_result, backtest_config)
        
        # Combine all results
        experiment_result = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "data_config": data_config,
            "model_config": model_config,
            "backtest_config": backtest_config,
            "data_stats": {
                "train_samples": data_result['X_train'].shape[0],
                "val_samples": data_result.get('X_val', np.array([])).shape[0] if data_result.get('X_val') is not None else 0,
                "test_samples": data_result['X_test'].shape[0],
                "feature_count": data_result['X_train'].shape[1],
                "feature_names": data_result['feature_names']
            },
            "model_info": {
                "model_name": model_result['metadata']['model_name'],
                "model_type": model_result['metadata']['model_type'],
                "is_classifier": model_result['metadata']['is_classifier'],
                "train_metrics": model_result['metadata']['train_metrics'],
                "val_metrics": model_result['metadata']['val_metrics']
            },
            "backtest_metrics": backtest_result['backtest_result']['metrics'],
            "model_metrics": backtest_result['model_metrics'],
            "feature_importance": backtest_result['feature_importance']
        }
        
        # Save experiment result
        self._save_experiment(experiment_name, experiment_result)
        
        return experiment_result
    
    def _save_experiment(self, experiment_name: str, experiment_result: Dict[str, Any]) -> None:
        """
        Save experiment results.
        
        Args:
            experiment_name: Name of the experiment
            experiment_result: Result of the experiment
        """
        # Add to experiments dictionary
        self.experiments[experiment_name] = {
            "timestamp": experiment_result["timestamp"],
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model_name": experiment_result["model_info"]["model_name"],
            "model_type": experiment_result["model_info"]["model_type"],
            "backtest_profit": experiment_result["backtest_metrics"]["total_profit"],
            "backtest_sharpe": experiment_result["backtest_metrics"]["sharpe_ratio"]
        }
        
        # Save full result
        result_file = os.path.join(
            self.results_dir,
            f"{experiment_name}_{experiment_result['timestamp']}.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump(experiment_result, f, indent=2, default=str)
        
        # Save updated experiments list
        experiments_file = os.path.join(self.results_dir, "experiments.json")
        with open(experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)
    
    def get_experiment_list(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a list of all experiments.
        
        Returns:
            Dictionary with experiment information
        """
        return self.experiments
    
    def load_experiment(self, experiment_name: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Load experiment results.
        
        Args:
            experiment_name: Name of the experiment
            timestamp: Timestamp of the experiment (if multiple runs exist)
            
        Returns:
            Dictionary with experiment results
        """
        # Check if experiment exists
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        # If timestamp not provided, use the latest
        if timestamp is None:
            timestamp = self.experiments[experiment_name]["timestamp"]
        
        # Load experiment file
        result_file = os.path.join(self.results_dir, f"{experiment_name}_{timestamp}.json")
        
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"Experiment file not found: {result_file}")
        
        with open(result_file, 'r') as f:
            experiment_result = json.load(f)
        
        return experiment_result
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_names: List of experiment names to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        # Load each experiment
        for name in experiment_names:
            if name in self.experiments:
                experiment = self.load_experiment(name)
                
                # Extract key metrics
                comparison[name] = {
                    "timestamp": experiment["timestamp"],
                    "model_type": experiment["model_info"]["model_type"],
                    "backtest_metrics": experiment["backtest_metrics"],
                    "model_metrics": experiment["model_metrics"],
                    "data_config": {
                        "indicators": experiment["data_config"].get("indicators", []),
                        "target_type": experiment["data_config"].get("target_type", "direction"),
                        "split_method": experiment["data_config"].get("split_method", "time")
                    }
                }
        
        return comparison
    
    def optimize_hyperparameters(
        self,
        experiment_name: str,
        data_config: Dict[str, Any],
        model_type: str,
        param_grid: Dict[str, List[Any]],
        method: str = "grid",
        cv: int = 5,
        scoring: Optional[str] = None,
        n_iter: int = 10
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            experiment_name: Name for the experiment
            data_config: Configuration for data preparation
            model_type: Type of model to optimize
            param_grid: Grid of hyperparameters to search
            method: Tuning method ('grid' or 'random')
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_iter: Number of iterations for random search
            
        Returns:
            Dictionary with optimization results
        """
        # Create a timestamp for the experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Prepare data
        logger.info(f"Preparing data for hyperparameter optimization '{experiment_name}'")
        data_result = self.prepare_data(data_config)
        
        # Step 2: Optimize hyperparameters
        logger.info(f"Optimizing hyperparameters for model type '{model_type}'")
        
        # Extract data from data_result
        X_train = data_result['X_train']
        y_train = data_result['y_train']
        X_val = data_result.get('X_val')
        y_val = data_result.get('y_val')
        feature_names = data_result['feature_names']
        
        # Optimize hyperparameters
        optimization_result = self.learner.optimize_model_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_type=model_type,
            param_grid=param_grid,
            method=method,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            feature_names=feature_names
        )
        
        # Extract best parameters
        best_params = optimization_result['metadata'].get('tuning_results', {}).get('best_params', {})
        
        # Step 3: Test optimized model
        logger.info(f"Testing optimized model for experiment '{experiment_name}'")
        
        # Create backtest config
        backtest_config = {
            "initial_capital": 10000,
            "commission": 0.001,
            "slippage": 0.0,
            "prediction_threshold": 0.5
        }
        
        # Run backtest with optimized model
        backtest_result = self.backtest_model(data_result, optimization_result, backtest_config)
        
        # Combine results
        result = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "model_type": model_type,
            "optimization_method": method,
            "param_grid": param_grid,
            "best_params": best_params,
            "model_name": optimization_result['metadata']['model_name'],
            "train_metrics": optimization_result['metadata']['train_metrics'],
            "val_metrics": optimization_result['metadata']['val_metrics'],
            "test_metrics": backtest_result['model_metrics'],
            "backtest_metrics": backtest_result['backtest_result']['metrics'],
            "feature_importance": backtest_result['feature_importance']
        }
        
        # Save optimization result
        result_file = os.path.join(
            self.results_dir,
            f"{experiment_name}_optimization_{timestamp}.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result
    
    def feature_selection(
        self,
        experiment_name: str,
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        selection_method: str = "importance",
        threshold: float = 0.01,
        num_features: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform feature selection.
        
        Args:
            experiment_name: Name for the experiment
            data_config: Configuration for data preparation
            model_config: Configuration for model training
            selection_method: Method for feature selection ('importance', 'recursive')
            threshold: Importance threshold for selecting features
            num_features: Number of top features to select
            
        Returns:
            Dictionary with feature selection results
        """
        # Create a timestamp for the experiment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Prepare data
        logger.info(f"Preparing data for feature selection '{experiment_name}'")
        data_result = self.prepare_data(data_config)
        
        # Step 2: Train model
        logger.info(f"Training model for feature selection")
        model_result = self.train_model(data_result, model_config)
        
        # Step 3: Get feature importance
        model = model_result['model']
        feature_names = data_result['feature_names']
        
        # Calculate feature importance
        importance_dict = self.learner.feature_importance(model, feature_names)
        
        # Select features based on method
        selected_features = []
        
        if selection_method == "importance":
            # Select based on importance threshold
            for feature, importance in importance_dict.items():
                if importance >= threshold:
                    selected_features.append(feature)
            
            # If num_features is specified, limit to that number
            if num_features is not None and len(selected_features) > num_features:
                selected_features = list(importance_dict.keys())[:num_features]
        
        elif selection_method == "recursive":
            # Implementation for recursive feature elimination would go here
            # For now, just use top features
            if num_features is not None:
                selected_features = list(importance_dict.keys())[:num_features]
            else:
                # Use threshold
                for feature, importance in importance_dict.items():
                    if importance >= threshold:
                        selected_features.append(feature)
        
        # Step 4: Create new data config with selected features
        new_data_config = data_config.copy()
        
        # Filter indicators to include only selected features
        all_indicators = data_config.get('indicators', [])
        selected_indicators = []
        
        for indicator in all_indicators:
            # Check if any of the generated feature names from this indicator are in selected_features
            indicator_name = indicator.get('name', '')
            for feature in selected_features:
                if indicator_name in feature:
                    selected_indicators.append(indicator)
                    break
        
        new_data_config['indicators'] = selected_indicators
        
        # Step 5: Train model with selected features
        logger.info(f"Training model with selected features")
        data_result_selected = self.prepare_data(new_data_config)
        model_result_selected = self.train_model(data_result_selected, model_config)
        
        # Step 6: Test model with selected features
        logger.info(f"Testing model with selected features")
        backtest_config = {
            "initial_capital": 10000,
            "commission": 0.001,
            "slippage": 0.0,
            "prediction_threshold": 0.5
        }
        
        backtest_result = self.backtest_model(data_result_selected, model_result_selected, backtest_config)
        
        # Combine results
        result = {
            "experiment_name": experiment_name,
            "timestamp": timestamp,
            "selection_method": selection_method,
            "threshold": threshold,
            "num_features": num_features,
            "original_feature_count": len(feature_names),
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "feature_importance": importance_dict,
            "original_model": {
                "model_name": model_result['metadata']['model_name'],
                "train_metrics": model_result['metadata']['train_metrics'],
                "val_metrics": model_result['metadata']['val_metrics']
            },
            "selected_model": {
                "model_name": model_result_selected['metadata']['model_name'],
                "train_metrics": model_result_selected['metadata']['train_metrics'],
                "val_metrics": model_result_selected['metadata']['val_metrics']
            },
            "backtest_metrics": backtest_result['backtest_result']['metrics'],
            "model_metrics": backtest_result['model_metrics']
        }
        
        # Save feature selection result
        result_file = os.path.join(
            self.results_dir,
            f"{experiment_name}_feature_selection_{timestamp}.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        return result 