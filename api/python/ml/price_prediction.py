"""
Price Prediction Module

This module provides machine learning tools for predicting short-term price movements
using technical indicators, market data, and option analytics.
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)

class PricePredictionModel:
    """
    Machine learning model for predicting short-term price movements.
    
    This class implements tools for:
    1. Creating technical indicator features
    2. Training models to predict price movements
    3. Making predictions with confidence scores
    
    The model can be configured to predict various time horizons and
    can be trained for different symbols and market conditions.
    """
    
    def __init__(
        self,
        symbol: str = None,
        prediction_horizon: int = 5,
        model_dir: str = "models",
        model_type: str = "lightgbm",
        use_options_data: bool = False,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the price prediction model.
        
        Args:
            symbol: Trading symbol to predict
            prediction_horizon: Number of periods to predict ahead
            model_dir: Directory to store trained models
            model_type: Type of ML model to use (lightgbm, xgboost, random_forest)
            use_options_data: Whether to incorporate options data as features
            config: Additional configuration parameters
        """
        self.symbol = symbol
        self.prediction_horizon = prediction_horizon
        self.model_dir = model_dir
        self.model_type = model_type
        self.use_options_data = use_options_data
        self.config = config or {}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.feature_columns = []
        self.feature_importance = {}
        self.scaler = None
        self.label_encoder = None
        
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
                
                self.feature_columns = metadata.get('feature_columns', [])
                
                # Load model
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.label_encoder = model_data.get('label_encoder')
                self.feature_importance = model_data.get('feature_importance', {})
                
                logger.info(f"Loaded price prediction model for {self.symbol}")
            except Exception as e:
                logger.error(f"Error loading model for {self.symbol}: {str(e)}")
    
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
                'label_encoder': self.label_encoder,
                'feature_importance': self.feature_importance
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metadata
            metadata = {
                'symbol': self.symbol,
                'prediction_horizon': self.prediction_horizon,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'use_options_data': self.use_options_data,
                'created_at': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved price prediction model for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving model for {self.symbol}: {str(e)}")
    
    def _get_model_path(self) -> str:
        """
        Get the path to the model file.
        
        Returns:
            Path to the model file
        """
        model_name = f"{self.symbol}_price_prediction_h{self.prediction_horizon}"
        return os.path.join(self.model_dir, f"{model_name}.pkl")
    
    def _get_metadata_path(self) -> str:
        """
        Get the path to the model metadata file.
        
        Returns:
            Path to the model metadata file
        """
        model_name = f"{self.symbol}_price_prediction_h{self.prediction_horizon}"
        return os.path.join(self.model_dir, f"{model_name}_metadata.json")
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features for prediction.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicator features
        """
        df = data.copy()
        
        # Technical indicators
        
        # Ensure necessary columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            missing_cols_str = ', '.join(missing_cols)
            logger.error(f"Required columns missing: {missing_cols_str}")
            raise ValueError(f"Required columns missing: {missing_cols_str}")
        
        # Moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_to_ma_{window}'] = df['close'] / df[f'ma_{window}'] - 1
        
        # Exponential moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
            df[f'close_to_ema_{window}'] = df['close'] / df[f'ema_{window}'] - 1
        
        # Volatility
        for window in [5, 10, 20, 50]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window=window).std()
        
        # Price momentum
        for window in [1, 3, 5, 10, 20]:
            df[f'momentum_{window}'] = df['close'].pct_change(periods=window)
        
        # Volume indicators
        df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        
        # RSI
        for window in [6, 14, 26]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for window in [10, 20]:
            df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
            df[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * df[f'bb_std_{window}']
            df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * df[f'bb_std_{window}']
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
            df[f'bb_percent_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # Stochastic Oscillator
        for window in [7, 14]:
            df[f'stoch_{window}_k'] = 100 * ((df['close'] - df['low'].rolling(window).min()) / 
                                          (df['high'].rolling(window).max() - df['low'].rolling(window).min()))
            df[f'stoch_{window}_d'] = df[f'stoch_{window}_k'].rolling(3).mean()
        
        # Average True Range (ATR)
        for window in [7, 14]:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{window}'] = tr.rolling(window).mean()
            df[f'atr_ratio_{window}'] = df[f'atr_{window}'] / df['close']
        
        # Price channels
        for window in [10, 20, 50]:
            df[f'high_{window}'] = df['high'].rolling(window).max()
            df[f'low_{window}'] = df['low'].rolling(window).min()
            df[f'high_ratio_{window}'] = df['close'] / df[f'high_{window}'] - 1
            df[f'low_ratio_{window}'] = df['close'] / df[f'low_{window}'] - 1
        
        # Add options data if available and enabled
        if self.use_options_data and 'options_data' in self.config:
            options_data = self.config['options_data']
            if isinstance(options_data, pd.DataFrame) and not options_data.empty:
                # Implement options-based features here
                # This depends on the structure of your options data
                pass
        
        # Create target labels for training (future price movement)
        df['future_close'] = df['close'].shift(-self.prediction_horizon)
        df['future_return'] = df['future_close'] / df['close'] - 1
        
        # Create classification labels
        threshold = self.config.get('movement_threshold', 0.01)
        df['target'] = 0  # Neutral
        df.loc[df['future_return'] > threshold, 'target'] = 1  # Up
        df.loc[df['future_return'] < -threshold, 'target'] = -1  # Down
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        classification: bool = True,
        evaluation_metric: str = None,
        advanced_features: bool = True
    ) -> Dict[str, Any]:
        """
        Train the price prediction model.
        
        Args:
            data: DataFrame with OHLCV data
            test_size: Proportion of data to use for testing
            classification: Whether to train a classification model
            evaluation_metric: Metric to use for evaluation
            advanced_features: Whether to use advanced features
            
        Returns:
            Dictionary with training results
        """
        try:
            # Create features
            df = self._create_features(data)
            
            # Update symbol if not set
            if self.symbol is None and 'symbol' in df.columns:
                self.symbol = df['symbol'].iloc[0]
            
            # Define feature columns (exclude target and other non-feature columns)
            exclude_cols = ['date', 'timestamp', 'datetime', 'symbol', 'future_close', 
                          'future_return', 'target', 'open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # If not using advanced features, reduce feature set
            if not advanced_features:
                basic_features = [col for col in feature_cols if any(
                    pattern in col for pattern in [
                        'ma_', 'ema_', 'momentum_', 'rsi_', 'macd', 'volume_ratio'
                    ]
                )]
                feature_cols = basic_features
            
            self.feature_columns = feature_cols
            
            # Prepare data
            X = df[feature_cols]
            
            if classification:
                # Classification task
                y = df['target']
                class_weights = {}
                
                # Handle class imbalance
                class_counts = y.value_counts()
                for cls in class_counts.index:
                    class_weights[cls] = 1.0 / class_counts[cls]
                    
                # Normalize class weights
                total_weight = sum(class_weights.values())
                for cls in class_weights:
                    class_weights[cls] = class_weights[cls] / total_weight * len(class_weights)
            else:
                # Regression task
                y = df['future_return']
            
            # Scale features if needed
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Train model based on model type
            if self.model_type.lower() == 'lightgbm':
                self._train_lightgbm(
                    X_train, X_test, y_train, y_test, 
                    classification, class_weights if classification else None,
                    evaluation_metric
                )
            elif self.model_type.lower() == 'xgboost':
                self._train_xgboost(
                    X_train, X_test, y_train, y_test, 
                    classification, class_weights if classification else None,
                    evaluation_metric
                )
            else:
                # Default to random forest
                self._train_random_forest(
                    X_train, X_test, y_train, y_test, 
                    classification, class_weights if classification else None
                )
            
            # Save model
            self._save_model()
            
            # Return training results
            return {
                "status": "success",
                "model_type": self.model_type,
                "feature_count": len(feature_cols),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_importance": self.feature_importance
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _train_lightgbm(
        self,
        X_train, X_test, y_train, y_test,
        classification: bool = True,
        class_weights: Dict[int, float] = None,
        evaluation_metric: str = None
    ) -> None:
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            classification: Whether this is a classification task
            class_weights: Class weights for handling imbalance
            evaluation_metric: Metric to use for evaluation
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. Please install with 'pip install lightgbm'.")
            raise
        
        # Prepare LightGBM dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        
        # Set parameters
        params = {
            'boosting_type': 'gbdt',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if classification:
            params['objective'] = 'multiclass' if len(set(y_train)) > 2 else 'binary'
            params['num_class'] = len(set(y_train)) if len(set(y_train)) > 2 else 1
            params['metric'] = evaluation_metric or 'multi_logloss' if len(set(y_train)) > 2 else 'binary_logloss'
            
            if class_weights:
                # Convert class weights to sample weights
                sample_weights = np.ones(len(y_train))
                for cls, weight in class_weights.items():
                    sample_weights[y_train == cls] = weight
                
                train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        else:
            params['objective'] = 'regression'
            params['metric'] = evaluation_metric or 'rmse'
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[test_data],
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = dict(zip(self.feature_columns, importance))
    
    def _train_xgboost(
        self,
        X_train, X_test, y_train, y_test,
        classification: bool = True,
        class_weights: Dict[int, float] = None,
        evaluation_metric: str = None
    ) -> None:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            classification: Whether this is a classification task
            class_weights: Class weights for handling imbalance
            evaluation_metric: Metric to use for evaluation
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Please install with 'pip install xgboost'.")
            raise
        
        # Set parameters
        params = {
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if classification:
            if len(set(y_train)) > 2:
                params['objective'] = 'multi:softprob'
                params['num_class'] = len(set(y_train))
                params['eval_metric'] = evaluation_metric or 'mlogloss'
            else:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = evaluation_metric or 'logloss'
            
            if class_weights:
                # Convert class weights to sample weights
                sample_weights = np.ones(len(y_train))
                for cls, weight in class_weights.items():
                    sample_weights[y_train == cls] = weight
            else:
                sample_weights = None
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = evaluation_metric or 'rmse'
            sample_weights = None
        
        # Train model
        self.model = xgb.XGBClassifier(**params) if classification else xgb.XGBRegressor(**params)
        
        # Fit model
        eval_set = [(X_test, y_test)]
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=eval_set,
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_columns, importance))
    
    def _train_random_forest(
        self,
        X_train, X_test, y_train, y_test,
        classification: bool = True,
        class_weights: Dict[int, float] = None
    ) -> None:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            classification: Whether this is a classification task
            class_weights: Class weights for handling imbalance
        """
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Set parameters
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if classification:
            if class_weights:
                params['class_weight'] = class_weights
            
            self.model = RandomForestClassifier(**params)
        else:
            self.model = RandomForestRegressor(**params)
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(self.feature_columns, importance))
    
    def predict(
        self,
        data: pd.DataFrame,
        include_probabilities: bool = True,
        include_features: bool = False
    ) -> Dict[str, Any]:
        """
        Make price movement predictions.
        
        Args:
            data: DataFrame with OHLCV data
            include_probabilities: Whether to include class probabilities
            include_features: Whether to include calculated features
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            return {
                "status": "error",
                "error": "Model not trained or loaded"
            }
        
        try:
            # Create features
            df = self._create_features(data)
            
            # Ensure all feature columns are available
            missing_cols = [col for col in self.feature_columns if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
                
                # Fill missing columns with zeros
                for col in missing_cols:
                    df[col] = 0
            
            # Extract features
            X = df[self.feature_columns]
            
            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                # Classification model with probabilities
                class_probabilities = self.model.predict_proba(X_scaled)
                predictions = self.model.predict(X_scaled)
                
                # Format results
                results = []
                for i, (pred, probs) in enumerate(zip(predictions, class_probabilities)):
                    entry = {
                        "prediction": int(pred),
                        "direction": "up" if pred == 1 else ("down" if pred == -1 else "neutral"),
                        "confidence": float(np.max(probs))
                    }
                    
                    if include_probabilities:
                        # Get class probabilities
                        prob_dict = {}
                        for j, prob in enumerate(probs):
                            if hasattr(self.model, 'classes_'):
                                prob_dict[int(self.model.classes_[j])] = float(prob)
                            else:
                                prob_dict[j] = float(prob)
                        
                        entry["probabilities"] = prob_dict
                    
                    if include_features:
                        entry["features"] = {col: float(X.iloc[i][col]) for col in self.feature_columns}
                    
                    # Add timestamp if available
                    if 'date' in df.columns:
                        entry["timestamp"] = str(df['date'].iloc[i])
                    elif 'timestamp' in df.columns:
                        entry["timestamp"] = str(df['timestamp'].iloc[i])
                    
                    results.append(entry)
                
                return {
                    "status": "success",
                    "predictions": results
                }
            
            else:
                # Regression model
                predictions = self.model.predict(X_scaled)
                
                # Format results
                results = []
                for i, pred in enumerate(predictions):
                    entry = {
                        "predicted_return": float(pred),
                        "direction": "up" if pred > 0 else ("down" if pred < 0 else "neutral"),
                        "magnitude": abs(float(pred))
                    }
                    
                    if include_features:
                        entry["features"] = {col: float(X.iloc[i][col]) for col in self.feature_columns}
                    
                    # Add timestamp if available
                    if 'date' in df.columns:
                        entry["timestamp"] = str(df['date'].iloc[i])
                    elif 'timestamp' in df.columns:
                        entry["timestamp"] = str(df['timestamp'].iloc[i])
                    
                    results.append(entry)
                
                return {
                    "status": "success",
                    "predictions": results
                }
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def evaluate(
        self,
        data: pd.DataFrame,
        include_report: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            data: DataFrame with OHLCV data
            include_report: Whether to include detailed classification report
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            return {
                "status": "error",
                "error": "Model not trained or loaded"
            }
        
        try:
            # Create features
            df = self._create_features(data)
            
            # Extract features and target
            X = df[self.feature_columns]
            
            # Determine if it's a classification task
            is_classification = hasattr(self.model, 'predict_proba')
            
            if is_classification:
                y = df['target']
            else:
                y = df['future_return']
            
            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Make predictions
            y_pred = self.model.predict(X_scaled)
            
            # Calculate metrics
            if is_classification:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                # Classification metrics
                accuracy = accuracy_score(y, y_pred)
                
                # Calculate precision, recall and F1 with different averaging methods
                metrics = {}
                
                for avg in ['micro', 'macro', 'weighted']:
                    metrics[f'precision_{avg}'] = precision_score(y, y_pred, average=avg, zero_division=0)
                    metrics[f'recall_{avg}'] = recall_score(y, y_pred, average=avg, zero_division=0)
                    metrics[f'f1_{avg}'] = f1_score(y, y_pred, average=avg, zero_division=0)
                
                # Confusion matrix
                cm = confusion_matrix(y, y_pred)
                cm_dict = {}
                
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        cm_dict[f"{i}_{j}"] = int(cm[i, j])
                
                results = {
                    "status": "success",
                    "accuracy": float(accuracy),
                    **{k: float(v) for k, v in metrics.items()},
                    "confusion_matrix": cm_dict
                }
                
                if include_report:
                    from sklearn.metrics import classification_report
                    report = classification_report(y, y_pred, output_dict=True)
                    results["classification_report"] = report
                
                # Calculate profitability metrics
                results["profitability"] = self._calculate_profitability(df, y_pred)
                
                return results
            
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Regression metrics
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                return {
                    "status": "success",
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2)
                }
        
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_profitability(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate profitability metrics for classification predictions.
        
        Args:
            data: DataFrame with features and actual returns
            predictions: Model predictions
            
        Returns:
            Dictionary with profitability metrics
        """
        df = data.copy()
        df['prediction'] = predictions
        
        # Calculate returns based on predictions
        df['traded_return'] = 0.0
        
        # Long positions (predicted positive movement)
        df.loc[df['prediction'] == 1, 'traded_return'] = df.loc[df['prediction'] == 1, 'future_return']
        
        # Short positions (predicted negative movement)
        df.loc[df['prediction'] == -1, 'traded_return'] = -df.loc[df['prediction'] == -1, 'future_return']
        
        # Skip neutral positions
        
        # Calculate cumulative returns
        df['cumulative_traded_return'] = (1 + df['traded_return']).cumprod() - 1
        df['cumulative_market_return'] = (1 + df['future_return']).cumprod() - 1
        
        # Calculate metrics
        total_trades = len(df[df['prediction'] != 0])
        profitable_trades = len(df[(df['prediction'] != 0) & (df['traded_return'] > 0)])
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Final returns
        final_traded_return = df['cumulative_traded_return'].iloc[-1] if len(df) > 0 else 0
        final_market_return = df['cumulative_market_return'].iloc[-1] if len(df) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + df['traded_return']).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        return {
            "win_rate": float(win_rate),
            "total_return": float(final_traded_return),
            "market_return": float(final_market_return),
            "outperformance": float(final_traded_return - final_market_return),
            "max_drawdown": float(max_drawdown),
            "total_trades": int(total_trades),
            "profitable_trades": int(profitable_trades)
        }
    
    def get_feature_importance(
        self,
        top_n: int = None
    ) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance
        """
        if not self.feature_importance:
            return {}
        
        # Sort features by importance
        sorted_features = dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Return top N features if specified
        if top_n is not None and top_n > 0:
            return dict(list(sorted_features.items())[:top_n])
        
        return sorted_features 