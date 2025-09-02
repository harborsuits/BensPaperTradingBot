#!/usr/bin/env python3
"""
Machine Learning Strategy Optimizer for advanced strategy parameter optimization.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Callable, Optional, Union
import logging
import warnings
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize, forest_minimize, dummy_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Import local modules
from trading_bot.optimization.strategy_optimizer import StrategyOptimizer

class MLStrategyOptimizer(StrategyOptimizer):
    """Machine Learning Strategy Optimizer extends StrategyOptimizer with ML capabilities."""
    
    def __init__(self, 
                 strategy_class: Any,
                 param_grid: Dict[str, List[Any]],
                 scoring_function: Union[str, Callable] = 'sharpe_ratio',
                 test_period: Tuple[str, str] = None,
                 validation_period: Tuple[str, str] = None,
                 initial_capital: float = 10000.0,
                 n_jobs: int = 1,
                 verbose: bool = False,
                 ml_model: str = 'random_forest',
                 bayesian_optimization: bool = True,
                 n_bayesian_iterations: int = 50,
                 n_initial_points: int = 10):
        """
        Initialize the ML strategy optimizer.
        
        Args:
            strategy_class: The strategy class to optimize
            param_grid: Dictionary of parameters to optimize with lists of values to try
            scoring_function: Metric to evaluate performance
            test_period: (start_date, end_date) for the backtest
            validation_period: Optional (start_date, end_date) for validation
            initial_capital: Starting capital for backtests
            n_jobs: Number of parallel jobs to run
            verbose: Whether to print progress information
            ml_model: ML model to use for surrogate modeling ('random_forest', 'svr', 'gpr')
            bayesian_optimization: Whether to use Bayesian optimization
            n_bayesian_iterations: Number of iterations for Bayesian optimization
            n_initial_points: Number of random initial points for Bayesian optimization
        """
        super().__init__(
            strategy_class=strategy_class,
            param_grid=param_grid,
            scoring_function=scoring_function,
            test_period=test_period,
            validation_period=validation_period,
            initial_capital=initial_capital,
            n_jobs=n_jobs,
            verbose=verbose
        )
        
        # ML specific settings
        self.ml_model = ml_model
        self.bayesian_optimization = bayesian_optimization
        self.n_bayesian_iterations = n_bayesian_iterations
        self.n_initial_points = n_initial_points
        
        # ML related storage
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.param_features = None
        self.feature_names = None
        self.performance_data = None
        
        # Suppress sklearn warnings if not verbose
        if not verbose:
            warnings.filterwarnings("ignore", category=UserWarning)
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML model from optimization results.
        
        Returns:
            Tuple of (X, y) for training ML model
        """
        if not self.results:
            self.logger.warning("No results available for ML model training")
            return np.array([]), np.array([])
        
        # Extract data from results
        features = []
        labels = []
        param_names = list(self.param_grid.keys())
        self.feature_names = param_names.copy()
        
        for result in self.results:
            if "error" in result:
                continue
                
            # Extract parameter values as features
            feature_vector = []
            for param in param_names:
                value = result["params"].get(param)
                # Convert non-numeric values to numeric if possible
                if not isinstance(value, (int, float)):
                    if isinstance(value, bool):
                        value = 1 if value else 0
                    else:
                        # For categorical features, we'll need to handle them specially
                        continue
                feature_vector.append(value)
            
            features.append(feature_vector)
            labels.append(result["score"])
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Filter out any NaN values
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Store for later use
        self.param_features = X
        self.performance_data = y
        
        return X, y
    
    def _train_ml_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Train an ML model on the optimization results data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Trained ML model
        """
        if X.shape[0] < 5:
            self.logger.warning("Not enough data points to train ML model")
            return None
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Select model based on user preference
        if self.ml_model == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                n_jobs=self.n_jobs if self.n_jobs > 0 else None
            )
        elif self.ml_model == 'svr':
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif self.ml_model == 'gpr':
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
        else:
            self.logger.warning(f"Unknown ML model '{self.ml_model}', using random forest")
            model = RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs if self.n_jobs > 0 else None)
        
        # Train model
        model.fit(X_scaled, y)
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            self.logger.info(f"Feature importance: {self.feature_importance}")
        
        return model
    
    def _predict_performance(self, params: Dict[str, Any]) -> float:
        """
        Predict performance for a parameter set using the trained ML model.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Predicted performance score
        """
        if self.model is None or self.scaler is None:
            return float('-inf')
        
        # Extract features from params
        feature_vector = []
        for param in self.feature_names:
            value = params.get(param)
            # Convert non-numeric values
            if not isinstance(value, (int, float)):
                if isinstance(value, bool):
                    value = 1 if value else 0
                else:
                    # For categorical features, handle specially
                    value = 0  # Default value for now
            feature_vector.append(value)
        
        # Convert to numpy array and scale
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        # Predict performance
        return self.model.predict(X_scaled)[0]
    
    def _prepare_search_space(self) -> List[Any]:
        """
        Prepare search space for Bayesian optimization.
        
        Returns:
            List of search dimensions for scikit-optimize
        """
        dimensions = []
        
        for param_name, param_values in self.param_grid.items():
            # Check if parameter values are numeric
            numeric_values = [v for v in param_values if isinstance(v, (int, float))]
            
            if len(numeric_values) == len(param_values):
                # All values are numeric
                if all(isinstance(v, int) for v in param_values):
                    # Integer parameter
                    dimensions.append(Integer(min(param_values), max(param_values), name=param_name))
                else:
                    # Real parameter
                    dimensions.append(Real(min(param_values), max(param_values), name=param_name))
            else:
                # Categorical parameter
                dimensions.append(Categorical(param_values, name=param_name))
        
        return dimensions
    
    def _bayesian_objective(self, params: List[Any]) -> float:
        """
        Objective function for Bayesian optimization.
        
        Args:
            params: Parameter values as a list
            
        Returns:
            Negative score (for minimization)
        """
        # Convert parameters list to dictionary
        param_dict = dict(zip(self.feature_names, params))
        
        # First check if we have a trained model to predict the score
        if self.model is not None and len(self.results) > self.n_initial_points:
            predicted_score = self._predict_performance(param_dict)
            
            # If prediction is poor, evaluate with actual backtest
            if predicted_score < float('-inf') + 1:
                result = self._evaluate_params(param_dict)
                return -result['score']  # Negative for minimization
            
            # Use model prediction with exploration factor
            return -predicted_score  # Negative for minimization
        
        # Otherwise evaluate with actual backtest
        result = self._evaluate_params(param_dict)
        return -result['score']  # Negative for minimization
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization using machine learning and/or Bayesian optimization.
        
        Returns:
            Dictionary containing optimization results
        """
        if self.bayesian_optimization:
            return self._run_bayesian_optimization()
        else:
            # Run grid search first to get initial data
            self.logger.info("Running initial grid search for ML modeling")
            initial_results = super().optimize()
            
            # Train ML model on grid search results
            X, y = self._prepare_training_data()
            if X.shape[0] > 0:
                self.model = self._train_ml_model(X, y)
                
                # Use ML to find promising parameters
                self.logger.info("Using ML model to find optimal parameters")
                best_params = self._find_optimal_parameters_with_ml()
                
                # Evaluate best parameters
                if best_params:
                    self.logger.info(f"Evaluating ML-suggested parameters: {best_params}")
                    result = self._evaluate_params(best_params)
                    
                    # Update best params if better
                    if result['score'] > self.best_score:
                        self.best_params = best_params
                        self.best_score = result['score']
                        self.best_performance = result['performance']
                        self.logger.info(f"Found better parameters with ML: score={self.best_score}")
            
            return self._create_optimization_report()
    
    def _run_bayesian_optimization(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find optimal parameters.
        
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Running Bayesian optimization")
        
        # Prepare search space
        dimensions = self._prepare_search_space()
        self.feature_names = [d.name for d in dimensions]
        
        # Initialize results list
        self.results = []
        
        # Define objective function with named arguments
        @use_named_args(dimensions)
        def objective(**params):
            result = self._evaluate_params(params)
            self.results.append(result)
            return -result['score']  # Negative for minimization
        
        # Run optimization
        if len(dimensions) == 0:
            self.logger.warning("No valid dimensions for Bayesian optimization")
            return self._create_optimization_report()
        
        # Choose optimization method
        if self.ml_model == 'random_forest':
            res = forest_minimize(
                objective,
                dimensions,
                n_calls=self.n_bayesian_iterations,
                n_initial_points=self.n_initial_points,
                n_jobs=self.n_jobs if self.n_jobs > 0 else 1,
                verbose=self.verbose
            )
        elif self.ml_model in ['gpr', 'svr']:
            res = gp_minimize(
                objective,
                dimensions,
                n_calls=self.n_bayesian_iterations,
                n_initial_points=self.n_initial_points,
                n_jobs=self.n_jobs if self.n_jobs > 0 else 1,
                verbose=self.verbose
            )
        else:
            res = dummy_minimize(
                objective,
                dimensions,
                n_calls=self.n_bayesian_iterations,
                verbose=self.verbose
            )
        
        # Extract best parameters
        best_params_dict = {}
        for dim, value in zip(dimensions, res.x):
            best_params_dict[dim.name] = value
        
        self.best_params = best_params_dict
        self.best_score = -res.fun  # Convert back to maximization
        
        # Find best performance from results
        for result in self.results:
            if result.get('score', float('-inf')) == self.best_score:
                self.best_performance = result.get('performance', {})
                break
        
        # Train ML model on collected data for future use
        X, y = self._prepare_training_data()
        if X.shape[0] > 0:
            self.model = self._train_ml_model(X, y)
        
        return self._create_optimization_report()
    
    def _find_optimal_parameters_with_ml(self) -> Dict[str, Any]:
        """
        Use ML model to find optimal parameters.
        
        Returns:
            Dictionary of optimal parameters
        """
        if self.model is None:
            return None
        
        # Generate new parameter combinations using grid expansion
        expanded_grid = self._expand_parameter_grid()
        
        # Predict performance for each combination
        best_params = None
        best_predicted_score = float('-inf')
        
        for params in expanded_grid:
            predicted_score = self._predict_performance(params)
            
            if predicted_score > best_predicted_score:
                best_predicted_score = predicted_score
                best_params = params
        
        return best_params
    
    def _expand_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Expand parameter grid with more fine-grained values based on ML insights.
        
        Returns:
            List of parameter dictionaries with expanded search space
        """
        expanded_grid = []
        
        # For each parameter, expand the search space
        expanded_param_grid = {}
        
        for param_name, param_values in self.param_grid.items():
            # Only expand numeric parameters
            numeric_values = [v for v in param_values if isinstance(v, (int, float))]
            
            if len(numeric_values) == len(param_values):
                # All values are numeric, expand the range
                min_val = min(param_values)
                max_val = max(param_values)
                range_val = max_val - min_val
                
                # Check if parameter is integer or float
                if all(isinstance(v, int) for v in param_values):
                    # Integer parameter
                    expanded_values = list(range(
                        max(int(min_val - range_val * 0.1), 1),  # Don't go below 1 for integers
                        int(max_val + range_val * 0.1) + 1,
                        max(1, int(range_val / 10))  # Step size, minimum 1
                    ))
                else:
                    # Float parameter
                    step = range_val / 20
                    expanded_values = [
                        min_val - range_val * 0.1 + i * step 
                        for i in range(int(range_val * 1.2 / step) + 1)
                    ]
                
                expanded_param_grid[param_name] = expanded_values
            else:
                # Keep categorical parameters as is
                expanded_param_grid[param_name] = param_values
        
        # Generate combinations from expanded grid
        # Limit to a reasonable number to avoid explosion
        max_combinations = 1000
        
        # Generate combinations based on importance
        if self.feature_importance is not None:
            # Sort parameters by importance
            sorted_params = sorted(
                expanded_param_grid.keys(),
                key=lambda p: self.feature_importance.get(p, 0),
                reverse=True
            )
            
            # Take more values for important parameters, fewer for less important
            for i, param in enumerate(sorted_params):
                importance = self.feature_importance.get(param, 0)
                if importance < 0.05:  # Low importance parameter
                    # Take fewer values
                    values = expanded_param_grid[param]
                    if len(values) > 3:
                        step = max(1, len(values) // 3)
                        expanded_param_grid[param] = values[::step]
        
        # Generate combinations (limited)
        keys = expanded_param_grid.keys()
        values = expanded_param_grid.values()
        combinations = list(itertools.product(*values))
        
        # Limit combinations if too many
        if len(combinations) > max_combinations:
            import random
            random.seed(42)  # For reproducibility
            combinations = random.sample(combinations, max_combinations)
        
        # Convert to list of dictionaries
        for combo in combinations:
            expanded_grid.append(dict(zip(keys, combo)))
        
        self.logger.info(f"Expanded grid search space to {len(expanded_grid)} combinations")
        
        return expanded_grid
    
    def save_ml_model(self, filepath: str) -> None:
        """
        Save the trained ML model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            self.logger.warning("No ML model to save")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model, scaler, and metadata
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'ml_model_type': self.ml_model
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"ML model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving ML model: {str(e)}")
    
    def load_ml_model(self, filepath: str) -> bool:
        """
        Load a trained ML model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # Load model data
            model_data = joblib.load(filepath)
            
            # Extract components
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data['feature_importance']
            self.ml_model = model_data.get('ml_model_type', 'unknown')
            
            self.logger.info(f"ML model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading ML model: {str(e)}")
            return False
    
    def cross_validate_ml_model(self, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on the ML model.
        
        Args:
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation metrics
        """
        if self.param_features is None or self.performance_data is None:
            self.logger.warning("No data available for cross-validation")
            return {}
        
        if len(self.param_features) < cv:
            self.logger.warning(f"Not enough data points for {cv}-fold cross-validation")
            return {}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.param_features)
        
        # Initialize model for cross-validation
        if self.ml_model == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs if self.n_jobs > 0 else None)
        elif self.ml_model == 'svr':
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif self.ml_model == 'gpr':
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        else:
            model = RandomForestRegressor(n_estimators=100, n_jobs=self.n_jobs if self.n_jobs > 0 else None)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_scaled, self.performance_data, cv=cv, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(model, X_scaled, self.performance_data, cv=cv, scoring='r2')
        
        # Calculate metrics
        mse = -cv_scores.mean()
        rmse = np.sqrt(mse)
        r2 = r2_scores.mean()
        
        self.logger.info(f"Cross-validation results: RMSE={rmse:.4f}, R²={r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_scores': cv_scores.tolist(),
            'r2_scores': r2_scores.tolist()
        }
    
    def plot_ml_predictions(self, save_path: Optional[str] = None) -> None:
        """
        Plot actual vs. predicted performance from ML model.
        
        Args:
            save_path: Optional path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.model is None or self.param_features is None or self.performance_data is None:
                self.logger.warning("No ML model or data available for plotting")
                return
            
            # Scale features
            X_scaled = self.scaler.transform(self.param_features)
            
            # Get predictions
            y_pred = self.model.predict(X_scaled)
            
            # Plot actual vs. predicted
            plt.figure(figsize=(10, 6))
            plt.scatter(self.performance_data, y_pred, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(min(self.performance_data), min(y_pred))
            max_val = max(max(self.performance_data), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title(f'Actual vs. Predicted Performance ({self.ml_model.upper()})')
            plt.xlabel('Actual Performance')
            plt.ylabel('Predicted Performance')
            
            # Add R² value
            from sklearn.metrics import r2_score
            r2 = r2_score(self.performance_data, y_pred)
            plt.text(
                0.05, 0.95, f'R² = {r2:.4f}', 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            if save_path:
                plt.savefig(save_path)
                self.logger.info(f"Plot saved to {save_path}")
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            self.logger.error("Matplotlib and/or seaborn are required for plotting")
        except Exception as e:
            self.logger.error(f"Error plotting ML predictions: {str(e)}")

if __name__ == "__main__":
    # Example usage
    from trading_bot.strategy.moving_average_strategy import MovingAverageStrategy
    
    # Define parameter grid
    param_grid = {
        'short_window': [5, 10, 20, 30, 40],
        'long_window': [50, 100, 150, 200],
        'symbol': ['SPY']
    }
    
    # Create optimizer
    optimizer = MLStrategyOptimizer(
        strategy_class=MovingAverageStrategy,
        param_grid=param_grid,
        scoring_function='sharpe_ratio',
        test_period=('2020-01-01', '2022-01-01'),
        validation_period=('2022-01-01', '2022-12-31'),
        initial_capital=10000.0,
        n_jobs=-1,
        verbose=True,
        ml_model='random_forest',
        bayesian_optimization=True,
        n_bayesian_iterations=30,
        n_initial_points=10
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results and model
    optimizer.save_results('ml_optimization_results.json')
    optimizer.save_ml_model('ml_strategy_model.joblib')
    
    # Plot results
    optimizer.plot_optimization_results('ml_optimization_plot.png')
    optimizer.plot_ml_predictions('ml_predictions_plot.png') 