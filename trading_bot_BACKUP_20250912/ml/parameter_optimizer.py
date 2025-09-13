"""
Parameter Optimizer Module

This module provides tools for optimizing trading strategy parameters based on
market conditions and historical performance data. It uses Bayesian optimization
to efficiently search the parameter space and machine learning to predict
optimal parameters for current market conditions.
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)

class ParameterOptimizer:
    """
    Parameter optimizer for trading strategies.
    
    This class implements tools for:
    1. Optimizing strategy parameters based on market conditions
    2. Using Bayesian optimization for efficient parameter search
    3. Training a model to predict optimal parameters for current conditions
    
    The optimizer can adapt parameters based on changing market regimes
    identified by the MarketConditionClassifier.
    """
    
    def __init__(
        self,
        strategy_name: str,
        parameter_space: Dict[str, Any],
        market_condition_classifier = None,
        objective_function: Optional[Callable] = None,
        model_dir: str = "models",
        max_trials: int = 100,
        n_startup_trials: int = 10
    ):
        """
        Initialize the parameter optimizer.
        
        Args:
            strategy_name: Name of the trading strategy
            parameter_space: Dictionary defining parameter names and their ranges
            market_condition_classifier: Optional MarketConditionClassifier instance
            objective_function: Function to evaluate parameter sets
            model_dir: Directory to store trained models
            max_trials: Maximum number of optimization trials
            n_startup_trials: Number of random trials before optimization
        """
        self.strategy_name = strategy_name
        self.parameter_space = parameter_space
        self.market_condition_classifier = market_condition_classifier
        self.objective_function = objective_function
        self.model_dir = model_dir
        self.max_trials = max_trials
        self.n_startup_trials = n_startup_trials
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize optimization components
        self.model = None
        self.scaler = None
        self.best_parameters = {}
        self.best_parameters_by_condition = {}
        self.optimization_history = []
        self.studies = {}
        
        # Load model if available
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load trained model from disk.
        """
        model_path = self._get_model_path()
        metadata_path = self._get_metadata_path()
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            try:
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.best_parameters = metadata.get('best_parameters', {})
                self.best_parameters_by_condition = metadata.get('best_parameters_by_condition', {})
                
                # Load model
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.optimization_history = model_data.get('optimization_history', [])
                
                logger.info(f"Loaded parameter optimizer for {self.strategy_name}")
            except Exception as e:
                logger.error(f"Error loading model for {self.strategy_name}: {str(e)}")
    
    def _save_model(self) -> None:
        """
        Save trained model to disk.
        """
        model_path = self._get_model_path()
        metadata_path = self._get_metadata_path()
        
        try:
            # Save model data
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'optimization_history': self.optimization_history,
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metadata
            metadata = {
                'strategy_name': self.strategy_name,
                'parameter_space': self.parameter_space,
                'best_parameters': self.best_parameters,
                'best_parameters_by_condition': self.best_parameters_by_condition,
                'created_at': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved parameter optimizer for {self.strategy_name}")
        except Exception as e:
            logger.error(f"Error saving model for {self.strategy_name}: {str(e)}")
    
    def _get_model_path(self) -> str:
        """
        Get the path to the model file.
        
        Returns:
            Path to the model file
        """
        model_name = f"{self.strategy_name}_optimizer"
        return os.path.join(self.model_dir, f"{model_name}.pkl")
    
    def _get_metadata_path(self) -> str:
        """
        Get the path to the model metadata file.
        
        Returns:
            Path to the model metadata file
        """
        model_name = f"{self.strategy_name}_optimizer"
        return os.path.join(self.model_dir, f"{model_name}_metadata.json")
    
    def _create_trial_params(self, trial) -> Dict[str, Any]:
        """
        Create parameters for a trial using optuna.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary with parameter values for the trial
        """
        params = {}
        
        for param_name, param_spec in self.parameter_space.items():
            param_type = param_spec.get('type', 'float')
            
            if param_type == 'float':
                low = param_spec.get('low', 0.0)
                high = param_spec.get('high', 1.0)
                log = param_spec.get('log', False)
                
                if log:
                    params[param_name] = trial.suggest_loguniform(param_name, low, high)
                else:
                    params[param_name] = trial.suggest_float(param_name, low, high)
                    
            elif param_type == 'int':
                low = param_spec.get('low', 0)
                high = param_spec.get('high', 10)
                log = param_spec.get('log', False)
                
                if log:
                    params[param_name] = trial.suggest_int(param_name, low, high, log=True)
                else:
                    params[param_name] = trial.suggest_int(param_name, low, high)
                    
            elif param_type == 'categorical':
                choices = param_spec.get('choices', [])
                params[param_name] = trial.suggest_categorical(param_name, choices)
                
            else:
                logger.warning(f"Unknown parameter type: {param_type}")
        
        return params
    
    def _evaluate_parameters(
        self,
        params: Dict[str, Any],
        market_data: pd.DataFrame,
        market_condition: str = None
    ) -> float:
        """
        Evaluate a set of parameters.
        
        Args:
            params: Dictionary with parameter values
            market_data: DataFrame with market data for testing
            market_condition: Optional market condition label
            
        Returns:
            Objective function value (higher is better)
        """
        if self.objective_function is None:
            raise ValueError("Objective function not defined")
        
        try:
            # Call user-defined objective function
            result = self.objective_function(params, market_data, market_condition)
            
            # Track result in history
            history_entry = {
                'params': params.copy(),
                'objective_value': result,
                'timestamp': datetime.now().isoformat(),
                'market_condition': market_condition
            }
            
            self.optimization_history.append(history_entry)
            
            # Update best parameters overall
            if not self.best_parameters or result > self.best_parameters.get('objective_value', float('-inf')):
                self.best_parameters = {
                    'params': params.copy(),
                    'objective_value': result,
                    'timestamp': datetime.now().isoformat(),
                    'market_condition': market_condition
                }
            
            # Update best parameters by condition
            if market_condition:
                if market_condition not in self.best_parameters_by_condition or \
                   result > self.best_parameters_by_condition[market_condition].get('objective_value', float('-inf')):
                    self.best_parameters_by_condition[market_condition] = {
                        'params': params.copy(),
                        'objective_value': result,
                        'timestamp': datetime.now().isoformat()
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {str(e)}")
            return float('-inf')
    
    def optimize(
        self,
        market_data: pd.DataFrame,
        market_condition: str = None,
        max_trials: int = None,
        n_startup_trials: int = None,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Optimize parameters using Bayesian optimization.
        
        Args:
            market_data: DataFrame with market data for testing
            market_condition: Optional market condition label
            max_trials: Maximum number of trials (overrides instance value)
            n_startup_trials: Number of random trials (overrides instance value)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with optimization results
        """
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Please install with 'pip install optuna'.")
            raise
        
        # Use provided values or instance defaults
        max_trials = max_trials or self.max_trials
        n_startup_trials = n_startup_trials or self.n_startup_trials
        
        # Create unique study name for this market condition
        study_name = f"{self.strategy_name}_{market_condition or 'default'}"
        
        # Define objective function for optuna
        def objective(trial):
            params = self._create_trial_params(trial)
            return self._evaluate_parameters(params, market_data, market_condition)
        
        # Create and run optimization study
        try:
            # Create study with TPE sampler
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials,
                seed=seed
            )
            
            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=sampler
            )
            
            # Store study for later use
            self.studies[study_name] = study
            
            # Run optimization
            study.optimize(objective, n_trials=max_trials)
            
            # Extract best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Save model
            self._save_model()
            
            # Return results
            return {
                "status": "success",
                "best_params": best_params,
                "best_value": best_value,
                "study_name": study_name,
                "trials_count": len(study.trials),
                "market_condition": market_condition
            }
            
        except Exception as e:
            logger.error(f"Error during parameter optimization: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def optimize_by_market_condition(
        self,
        market_data: pd.DataFrame,
        max_trials_per_condition: int = None,
        conditions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize parameters for different market conditions.
        
        Args:
            market_data: DataFrame with market data for testing
            max_trials_per_condition: Maximum trials for each condition
            conditions: List of market conditions to optimize for
                
        Returns:
            Dictionary with optimization results for each condition
        """
        if self.market_condition_classifier is None:
            return {
                "status": "error",
                "error": "Market condition classifier not provided"
            }
        
        results = {}
        
        try:
            # Get market conditions if not provided
            if not conditions:
                conditions = self.market_condition_classifier.market_conditions
            
            if not conditions:
                logger.warning("No market conditions defined in classifier")
                return {
                    "status": "error",
                    "error": "No market conditions defined"
                }
            
            # Split data by market condition
            for condition in conditions:
                logger.info(f"Optimizing parameters for {condition} market condition")
                
                # Filter data for this market condition
                condition_mask = market_data['market_condition'] == condition
                condition_data = market_data[condition_mask]
                
                if len(condition_data) < 50:  # Minimum data requirement
                    logger.warning(f"Not enough data for {condition} market condition")
                    continue
                
                # Optimize parameters for this condition
                condition_result = self.optimize(
                    condition_data,
                    market_condition=condition,
                    max_trials=max_trials_per_condition
                )
                
                results[condition] = condition_result
            
            # Save model with all optimized parameters
            self._save_model()
            
            return {
                "status": "success",
                "condition_results": results
            }
            
        except Exception as e:
            logger.error(f"Error during multi-condition optimization: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def train_parameter_predictor(
        self,
        market_features: pd.DataFrame,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train a model to predict optimal parameters based on market features.
        
        Args:
            market_features: DataFrame with market features and conditions
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training results
        """
        try:
            # Check if we have enough optimization history
            if len(self.optimization_history) < 50:
                return {
                    "status": "error",
                    "error": "Not enough optimization history for training"
                }
                
            # Prepare training data
            X = []
            y = []
            
            for entry in self.optimization_history:
                if 'market_condition' not in entry or not entry['market_condition']:
                    continue
                
                condition = entry['market_condition']
                params = entry['params']
                objective_value = entry['objective_value']
                
                # Find market features for this condition
                condition_mask = market_features['market_condition'] == condition
                
                if not condition_mask.any():
                    continue
                
                # Get market features for this condition
                features = market_features[condition_mask].iloc[0]
                
                # Extract relevant features
                feature_cols = [col for col in features.index if col != 'market_condition']
                feature_vector = features[feature_cols].values
                
                # Add to training data with parameter values and objective
                X.append(feature_vector)
                
                # Flatten parameters into a list for prediction
                param_vector = [params[key] for key in sorted(params.keys())]
                y.append(param_vector)
            
            if not X or not y:
                return {
                    "status": "error",
                    "error": "Could not create training data"
                }
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self._train_parameter_model(X_train_scaled, X_test_scaled, y_train, y_test)
            
            # Save model
            self._save_model()
            
            return {
                "status": "success",
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "parameter_count": y.shape[1],
                "feature_count": X.shape[1]
            }
            
        except Exception as e:
            logger.error(f"Error training parameter predictor: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _train_parameter_model(
        self,
        X_train, X_test, y_train, y_test
    ) -> None:
        """
        Train a model to predict parameters.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets (parameter vectors)
            y_test: Test targets (parameter vectors)
        """
        # Try to use a multi-output regressor for parameter prediction
        try:
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.ensemble import GradientBoostingRegressor
            
            base_model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            
            self.model = MultiOutputRegressor(base_model)
            self.model.fit(X_train, y_train)
            
            # Evaluate on test set
            score = self.model.score(X_test, y_test)
            logger.info(f"Parameter predictor R² score: {score:.4f}")
            
        except Exception as e:
            logger.error(f"Error training parameter predictor model: {str(e)}")
            raise
    
    def predict_parameters(
        self,
        market_features: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predict optimal parameters for current market features.
        
        Args:
            market_features: DataFrame with current market features
            
        Returns:
            Dictionary with predicted parameters
        """
        if self.model is None:
            return {
                "status": "error",
                "error": "Parameter predictor model not trained"
            }
        
        try:
            # Extract relevant features
            feature_cols = [col for col in market_features.columns if col != 'market_condition']
            X = market_features[feature_cols].values.reshape(1, -1)
            
            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Predict parameters
            predicted_params = self.model.predict(X_scaled)[0]
            
            # Create parameter dictionary
            param_dict = {}
            param_names = sorted(self.parameter_space.keys())
            
            for i, name in enumerate(param_names):
                if i < len(predicted_params):
                    # Apply constraints based on parameter type
                    param_spec = self.parameter_space[name]
                    param_type = param_spec.get('type', 'float')
                    
                    if param_type == 'int':
                        param_dict[name] = int(round(predicted_params[i]))
                    elif param_type == 'categorical':
                        # Find closest categorical value
                        choices = param_spec.get('choices', [])
                        if choices:
                            # For categorical parameters, find the closest choice
                            # based on the predicted value normalized to [0, 1]
                            choices_range = len(choices) - 1
                            choice_idx = min(
                                int(round(predicted_params[i] * choices_range / 1.0)),
                                choices_range
                            )
                            param_dict[name] = choices[choice_idx]
                    else:
                        # Float type - apply bounds
                        low = param_spec.get('low', 0.0)
                        high = param_spec.get('high', 1.0)
                        param_dict[name] = max(low, min(high, predicted_params[i]))
            
            return {
                "status": "success",
                "predicted_parameters": param_dict
            }
            
        except Exception as e:
            logger.error(f"Error predicting parameters: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_best_parameters(
        self,
        market_condition: str = None
    ) -> Dict[str, Any]:
        """
        Get best parameters (overall or for a specific market condition).
        
        Args:
            market_condition: Optional market condition to get parameters for
            
        Returns:
            Dictionary with best parameters
        """
        if market_condition and market_condition in self.best_parameters_by_condition:
            return {
                "status": "success",
                "parameters": self.best_parameters_by_condition[market_condition]['params'],
                "objective_value": self.best_parameters_by_condition[market_condition]['objective_value'],
                "market_condition": market_condition
            }
        elif self.best_parameters:
            return {
                "status": "success",
                "parameters": self.best_parameters['params'],
                "objective_value": self.best_parameters['objective_value'],
                "market_condition": self.best_parameters.get('market_condition')
            }
        else:
            return {
                "status": "error",
                "error": "No optimized parameters available"
            }
    
    def get_optimization_history(
        self,
        market_condition: str = None,
        top_n: int = None,
        sort_by: str = 'objective_value'
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get optimization history (overall or for a specific market condition).
        
        Args:
            market_condition: Optional market condition to filter history
            top_n: Optional limit on number of results
            sort_by: Field to sort results by
            
        Returns:
            Dictionary with optimization history
        """
        if not self.optimization_history:
            return {
                "status": "error",
                "error": "No optimization history available"
            }
        
        # Filter by market condition if provided
        if market_condition:
            history = [entry for entry in self.optimization_history 
                      if entry.get('market_condition') == market_condition]
        else:
            history = self.optimization_history
        
        # Sort by specified field
        if sort_by in history[0]:
            history = sorted(history, key=lambda x: x[sort_by], reverse=True)
        
        # Limit to top N results if specified
        if top_n is not None and top_n > 0:
            history = history[:top_n]
        
        return {
            "status": "success",
            "history": history,
            "market_condition": market_condition
        }
    
    def get_parameter_importance(
        self,
        top_n: int = None
    ) -> Dict[str, Any]:
        """
        Get parameter importance based on optimization history.
        
        Args:
            top_n: Number of top parameters to return
            
        Returns:
            Dictionary with parameter importance
        """
        if not self.optimization_history or len(self.optimization_history) < 10:
            return {
                "status": "error",
                "error": "Not enough optimization history for importance analysis"
            }
        
        try:
            import optuna
            
            # Get all parameter names from history
            param_names = set()
            for entry in self.optimization_history:
                param_names.update(entry['params'].keys())
            
            param_names = sorted(list(param_names))
            
            # Analyze importance for each study
            importance_by_study = {}
            
            for study_name, study in self.studies.items():
                if len(study.trials) < 10:
                    continue
                
                # Calculate parameter importance
                try:
                    importance = optuna.importance.FanovaImportanceEvaluator().evaluate(study)
                    importance_by_study[study_name] = importance
                except Exception as e:
                    logger.warning(f"Could not calculate importance for {study_name}: {str(e)}")
            
            if not importance_by_study:
                return {
                    "status": "error",
                    "error": "Could not calculate parameter importance"
                }
            
            # Average importance across studies
            avg_importance = {}
            for param in param_names:
                values = [
                    imp.get(param, 0.0) for imp in importance_by_study.values()
                    if param in imp
                ]
                
                if values:
                    avg_importance[param] = sum(values) / len(values)
            
            # Sort by importance
            sorted_importance = dict(sorted(
                avg_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Return top N parameters if specified
            if top_n is not None and top_n > 0:
                sorted_importance = dict(list(sorted_importance.items())[:top_n])
            
            return {
                "status": "success",
                "importance": sorted_importance
            }
            
        except ImportError:
            logger.error("Optuna not installed. Cannot calculate parameter importance.")
            return {
                "status": "error",
                "error": "Optuna not installed"
            }
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def update_parameter_space(
        self,
        new_parameter_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update the parameter space for optimization.
        
        Args:
            new_parameter_space: Dictionary with new parameter specifications
            
        Returns:
            Status of the update operation
        """
        try:
            # Validate new parameter space
            for param_name, param_spec in new_parameter_space.items():
                if not isinstance(param_spec, dict):
                    return {
                        "status": "error",
                        "error": f"Invalid parameter specification for {param_name}"
                    }
                
                param_type = param_spec.get('type', 'float')
                
                if param_type not in ['float', 'int', 'categorical']:
                    return {
                        "status": "error",
                        "error": f"Invalid parameter type for {param_name}: {param_type}"
                    }
                
                if param_type == 'categorical' and 'choices' not in param_spec:
                    return {
                        "status": "error",
                        "error": f"Categorical parameter {param_name} missing 'choices'"
                    }
            
            # Update parameter space
            self.parameter_space = new_parameter_space
            
            # Save changes
            self._save_model()
            
            return {
                "status": "success",
                "parameter_space": self.parameter_space
            }
            
        except Exception as e:
            logger.error(f"Error updating parameter space: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            } 