#!/usr/bin/env python3
"""
Machine Learning Prediction Module for Broker Performance

Uses historical performance data to predict potential broker issues
before they occur, providing early warning and allowing for
proactive intervention.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Import for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from trading_bot.brokers.intelligence.historical_tracker import BrokerPerformanceTracker


logger = logging.getLogger(__name__)


class BrokerPerformancePredictor:
    """
    Predictive module for broker performance
    
    Uses machine learning to predict:
    1. Anomaly detection - Identify unusual performance patterns
    2. Failure prediction - Predict potential failures within timeframe
    3. Performance forecasting - Project future performance metrics
    """
    
    def __init__(
        self,
        performance_tracker: 'BrokerPerformanceTracker',
        model_dir: str = 'data/broker_ml_models',
        min_training_days: int = 7,
        prediction_window: int = 24  # Hours to predict ahead
    ):
        """
        Initialize the performance predictor
        
        Args:
            performance_tracker: BrokerPerformanceTracker for historical data
            model_dir: Directory to store trained models
            min_training_days: Minimum days of data required for training
            prediction_window: Hours to predict ahead
        """
        self.tracker = performance_tracker
        self.model_dir = model_dir
        self.min_training_days = min_training_days
        self.prediction_window = prediction_window
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Models by broker ID
        self.anomaly_models = {}
        self.failure_models = {}
        self.forecast_models = {}
        
        # Feature engineering settings
        self.n_lags = 5  # Number of lag features
        self.rolling_windows = [3, 6, 12, 24]  # Rolling windows in hours
        
        logger.info(f"BrokerPerformancePredictor initialized with {min_training_days} day minimum training period")
    
    def _extract_features(
        self,
        df: pd.DataFrame,
        lag_features: bool = True,
        rolling_features: bool = True,
        diff_features: bool = True
    ) -> pd.DataFrame:
        """
        Extract features for machine learning models
        
        Args:
            df: DataFrame with raw metrics
            lag_features: Whether to include lagged values
            rolling_features: Whether to include rolling statistics
            diff_features: Whether to include differenced values
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return pd.DataFrame()
        
        # Make a copy to avoid modifying original
        features_df = df.copy()
        
        # Get base metrics (exclude scores, broker_id, etc.)
        base_metrics = []
        for col in df.columns:
            if col.startswith('latency_') or col.startswith('reliability_') or \
               col.startswith('execution_quality_') or col.startswith('cost_'):
                base_metrics.append(col)
        
        if not base_metrics:
            logger.warning("No base metrics found for feature engineering")
            return features_df
        
        # Add lag features
        if lag_features:
            for metric in base_metrics:
                for lag in range(1, self.n_lags + 1):
                    features_df[f"{metric}_lag{lag}"] = features_df[metric].shift(lag)
        
        # Add rolling statistics
        if rolling_features:
            for metric in base_metrics:
                for window in self.rolling_windows:
                    # Rolling mean
                    features_df[f"{metric}_roll_mean{window}"] = features_df[metric].rolling(window=window).mean()
                    
                    # Rolling std
                    features_df[f"{metric}_roll_std{window}"] = features_df[metric].rolling(window=window).std()
                    
                    # Rolling min/max
                    features_df[f"{metric}_roll_min{window}"] = features_df[metric].rolling(window=window).min()
                    features_df[f"{metric}_roll_max{window}"] = features_df[metric].rolling(window=window).max()
        
        # Add differenced features
        if diff_features:
            for metric in base_metrics:
                # First difference
                features_df[f"{metric}_diff1"] = features_df[metric].diff()
                
                # Percent change
                features_df[f"{metric}_pct"] = features_df[metric].pct_change()
        
        # Add time-based features
        features_df['hour'] = features_df.index.hour
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Fill NAs (resulting from lags, rolling windows, etc.)
        features_df = features_df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return features_df
    
    def _get_training_data(
        self,
        broker_id: str,
        lookback_days: int = 30,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get training data with feature engineering
        
        Args:
            broker_id: Broker ID
            lookback_days: Days of historical data to use
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            DataFrame with features for training
        """
        # Get analyzer from tracker
        analyzer = self.tracker.get_analyzer()
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Get raw data
        df = analyzer.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty:
            logger.warning(f"No historical data found for broker {broker_id}")
            return pd.DataFrame()
        
        # Check if we have enough data
        if (end_time - df.index[0]).days < self.min_training_days:
            logger.warning(f"Insufficient data for broker {broker_id}. Need at least {self.min_training_days} days.")
            return pd.DataFrame()
        
        # Extract features
        features_df = self._extract_features(df)
        
        return features_df
    
    def build_anomaly_detection_model(
        self,
        broker_id: str,
        lookback_days: int = 30,
        contamination: float = 0.05,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> bool:
        """
        Build an anomaly detection model for a broker
        
        Args:
            broker_id: Broker ID
            lookback_days: Days of historical data to use
            contamination: Expected proportion of anomalies
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            bool: True if model was built successfully
        """
        logger.info(f"Building anomaly detection model for broker {broker_id}")
        
        # Get training data
        features_df = self._get_training_data(
            broker_id=broker_id,
            lookback_days=lookback_days,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if features_df.empty:
            return False
        
        # Filter out non-feature columns
        feature_cols = [col for col in features_df.columns if col not in ['broker_id', 'asset_class', 'operation_type']]
        
        # Create model pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('isolation_forest', IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                bootstrap=True
            ))
        ])
        
        # Fit model
        try:
            model.fit(features_df[feature_cols])
            
            # Save model
            self.anomaly_models[broker_id] = {
                'model': model,
                'feature_columns': feature_cols,
                'last_updated': datetime.now(),
                'lookback_days': lookback_days,
                'contamination': contamination,
                'asset_class': asset_class,
                'operation_type': operation_type
            }
            
            logger.info(f"Anomaly detection model for broker {broker_id} built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building anomaly detection model: {str(e)}")
            return False
    
    def detect_anomalies(
        self,
        broker_id: str,
        recent_hours: int = 24,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> Tuple[pd.DataFrame, float]:
        """
        Detect anomalies in recent broker performance
        
        Args:
            broker_id: Broker ID
            recent_hours: Hours of recent data to analyze
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            Tuple of (DataFrame with anomaly flags, anomaly score)
        """
        # Check if we have a model
        if broker_id not in self.anomaly_models:
            logger.info(f"No anomaly model for broker {broker_id}, building one")
            if not self.build_anomaly_detection_model(broker_id):
                return pd.DataFrame(), 0.0
        
        # Get analyzer from tracker
        analyzer = self.tracker.get_analyzer()
        
        # Get recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=recent_hours)
        
        df = analyzer.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty:
            logger.warning(f"No recent data found for broker {broker_id}")
            return pd.DataFrame(), 0.0
        
        # Extract features
        features_df = self._extract_features(df)
        
        # Get model information
        model_info = self.anomaly_models[broker_id]
        model = model_info['model']
        feature_cols = model_info['feature_columns']
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Detect anomalies
        try:
            # Predict (-1 for anomalies, 1 for normal)
            predictions = model.predict(features_df[feature_cols])
            
            # Calculate anomaly scores
            scores = model.decision_function(features_df[feature_cols])
            
            # Add to DataFrame
            features_df['anomaly'] = predictions
            features_df['anomaly_score'] = scores
            
            # Calculate overall anomaly percentage
            anomaly_pct = (predictions == -1).mean()
            
            logger.info(f"Detected {(predictions == -1).sum()} anomalies in {len(predictions)} records for broker {broker_id}")
            
            return features_df, float(anomaly_pct)
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return pd.DataFrame(), 0.0
    
    def _create_failure_labels(
        self,
        df: pd.DataFrame,
        failure_definition: Dict[str, Any]
    ) -> pd.Series:
        """
        Create failure labels for supervised learning
        
        Args:
            df: DataFrame with metrics
            failure_definition: Definition of what constitutes a failure
                Examples:
                {'metric': 'reliability_errors', 'threshold': 5, 'op': 'gt'}
                {'metric': 'score_overall', 'threshold': 70, 'op': 'lt'}
                
        Returns:
            Series with binary labels (1 for failure)
        """
        if df.empty:
            return pd.Series()
        
        metric = failure_definition.get('metric')
        threshold = failure_definition.get('threshold')
        op = failure_definition.get('op', 'gt')  # Default to 'greater than'
        
        if metric not in df.columns:
            logger.warning(f"Metric {metric} not found in data")
            return pd.Series(0, index=df.index)
        
        if op == 'gt':
            # Greater than threshold
            labels = (df[metric] > threshold).astype(int)
        elif op == 'lt':
            # Less than threshold
            labels = (df[metric] < threshold).astype(int)
        elif op == 'eq':
            # Equal to threshold
            labels = (df[metric] == threshold).astype(int)
        else:
            logger.warning(f"Unknown operation {op}")
            return pd.Series(0, index=df.index)
        
        # Shift labels forward to predict future failures
        prediction_periods = self.prediction_window
        labels_shifted = labels.shift(-prediction_periods).fillna(0).astype(int)
        
        return labels_shifted
    
    def build_failure_prediction_model(
        self,
        broker_id: str,
        failure_definition: Dict[str, Any],
        lookback_days: int = 30,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> bool:
        """
        Build a failure prediction model for a broker
        
        Args:
            broker_id: Broker ID
            failure_definition: Definition of what constitutes a failure
            lookback_days: Days of historical data to use
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            bool: True if model was built successfully
        """
        logger.info(f"Building failure prediction model for broker {broker_id}")
        
        # Get training data
        features_df = self._get_training_data(
            broker_id=broker_id,
            lookback_days=lookback_days,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if features_df.empty:
            return False
        
        # Create failure labels
        y = self._create_failure_labels(features_df, failure_definition)
        
        # Check if we have any failures in the dataset
        if y.sum() == 0:
            logger.warning(f"No failure events found for broker {broker_id} with definition {failure_definition}")
            return False
        
        # Filter out non-feature columns and the target metric
        feature_cols = [
            col for col in features_df.columns 
            if col not in ['broker_id', 'asset_class', 'operation_type', failure_definition.get('metric')]
        ]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df[feature_cols], 
            y, 
            test_size=0.2, 
            random_state=42,
            stratify=y  # Ensure balanced split given likely class imbalance
        )
        
        # Handle class imbalance
        # Determine class weights based on imbalance
        class_counts = y_train.value_counts()
        total = class_counts.sum()
        class_weights = {0: total / (class_counts[0] * 2), 1: total / (class_counts[1] * 2)}
        
        # Create model pipeline
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('random_forest', RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weights,
                random_state=42
            ))
        ])
        
        # Fit model
        try:
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            # Get predictions on test set
            y_pred = model.predict(X_test)
            
            # Calculate additional metrics
            from sklearn.metrics import classification_report, confusion_matrix, f1_score
            
            f1 = f1_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Save model
            self.failure_models[broker_id] = {
                'model': model,
                'feature_columns': feature_cols,
                'last_updated': datetime.now(),
                'lookback_days': lookback_days,
                'failure_definition': failure_definition,
                'asset_class': asset_class,
                'operation_type': operation_type,
                'metrics': {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'f1_score': f1,
                    'classification_report': report,
                    'confusion_matrix': cm.tolist()
                }
            }
            
            logger.info(f"Failure prediction model for broker {broker_id} built successfully")
            logger.info(f"Model metrics: Accuracy={test_accuracy:.3f}, F1={f1:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error building failure prediction model: {str(e)}")
            return False
    
    def predict_failure_probability(
        self,
        broker_id: str,
        recent_hours: int = 24,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> Tuple[pd.DataFrame, float]:
        """
        Predict probability of broker failure within prediction window
        
        Args:
            broker_id: Broker ID
            recent_hours: Hours of recent data to analyze
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            Tuple of (DataFrame with prediction, overall failure probability)
        """
        # Check if we have a model
        if broker_id not in self.failure_models:
            logger.warning(f"No failure prediction model for broker {broker_id}")
            return pd.DataFrame(), 0.0
        
        # Get analyzer from tracker
        analyzer = self.tracker.get_analyzer()
        
        # Get recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=recent_hours)
        
        df = analyzer.store.get_as_dataframe(
            broker_id=broker_id,
            start_time=start_time,
            end_time=end_time,
            asset_class=asset_class,
            operation_type=operation_type
        )
        
        if df.empty:
            logger.warning(f"No recent data found for broker {broker_id}")
            return pd.DataFrame(), 0.0
        
        # Extract features
        features_df = self._extract_features(df)
        
        # Get model information
        model_info = self.failure_models[broker_id]
        model = model_info['model']
        feature_cols = model_info['feature_columns']
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Make prediction
        try:
            # Predict class (0 or 1)
            predictions = model.predict(features_df[feature_cols])
            
            # Get probability of failure (class 1)
            probabilities = model.predict_proba(features_df[feature_cols])[:, 1]
            
            # Add to DataFrame
            features_df['failure_predicted'] = predictions
            features_df['failure_probability'] = probabilities
            
            # Calculate overall failure probability
            # Use the highest probability in the most recent data
            recent_df = features_df.iloc[-min(6, len(features_df)):]  # Last 6 records
            overall_probability = recent_df['failure_probability'].max()
            
            logger.info(f"Failure probability for broker {broker_id}: {overall_probability:.3f}")
            
            return features_df, float(overall_probability)
            
        except Exception as e:
            logger.error(f"Error predicting failure: {str(e)}")
            return pd.DataFrame(), 0.0
    
    def get_prediction_summary(
        self,
        broker_id: str,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary of predictions for a broker
        
        Args:
            broker_id: Broker ID
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            Dict with prediction summary
        """
        summary = {
            'broker_id': broker_id,
            'timestamp': datetime.now().isoformat(),
            'predictions': {},
            'models': {}
        }
        
        # Add anomaly detection results
        if broker_id in self.anomaly_models:
            anomalies_df, anomaly_pct = self.detect_anomalies(
                broker_id=broker_id,
                asset_class=asset_class,
                operation_type=operation_type
            )
            
            if not anomalies_df.empty:
                summary['predictions']['anomalies'] = {
                    'percentage': anomaly_pct,
                    'recent_anomalies': int((anomalies_df['anomaly'] == -1).sum()),
                    'total_records': len(anomalies_df),
                    'severity': 'high' if anomaly_pct > 0.2 else 'medium' if anomaly_pct > 0.05 else 'low'
                }
            
            # Add model information
            model_info = self.anomaly_models[broker_id]
            summary['models']['anomaly_detection'] = {
                'last_updated': model_info['last_updated'].isoformat(),
                'lookback_days': model_info['lookback_days']
            }
        
        # Add failure prediction results
        if broker_id in self.failure_models:
            failure_df, failure_prob = self.predict_failure_probability(
                broker_id=broker_id,
                asset_class=asset_class,
                operation_type=operation_type
            )
            
            if not failure_df.empty:
                summary['predictions']['failure'] = {
                    'probability': failure_prob,
                    'timeframe_hours': self.prediction_window,
                    'severity': 'high' if failure_prob > 0.7 else 'medium' if failure_prob > 0.3 else 'low'
                }
            
            # Add model information
            model_info = self.failure_models[broker_id]
            summary['models']['failure_prediction'] = {
                'last_updated': model_info['last_updated'].isoformat(),
                'lookback_days': model_info['lookback_days'],
                'failure_definition': model_info['failure_definition'],
                'accuracy': model_info['metrics']['test_accuracy']
            }
        
        # Add overall risk assessment
        if 'anomalies' in summary['predictions'] or 'failure' in summary['predictions']:
            # Determine overall risk level
            risk_level = 'low'
            
            if 'anomalies' in summary['predictions'] and summary['predictions']['anomalies']['severity'] == 'high':
                risk_level = 'high'
            elif 'failure' in summary['predictions'] and summary['predictions']['failure']['severity'] == 'high':
                risk_level = 'high'
            elif ('anomalies' in summary['predictions'] and summary['predictions']['anomalies']['severity'] == 'medium') or \
                 ('failure' in summary['predictions'] and summary['predictions']['failure']['severity'] == 'medium'):
                risk_level = 'medium'
            
            summary['overall_risk'] = {
                'level': risk_level,
                'action_recommended': risk_level == 'high'
            }
        
        return summary
    
    def generate_prediction_report(
        self,
        broker_ids: Optional[List[str]] = None,
        asset_class: Optional[str] = None,
        operation_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive prediction report for multiple brokers
        
        Args:
            broker_ids: List of broker IDs (if None, uses all with models)
            asset_class: Optional asset class filter
            operation_type: Optional operation type filter
            
        Returns:
            Dict with prediction report
        """
        # If no broker IDs provided, use all brokers with models
        if broker_ids is None:
            broker_ids = list(set(list(self.anomaly_models.keys()) + list(self.failure_models.keys())))
        
        if not broker_ids:
            logger.warning("No brokers to generate prediction report for")
            return {}
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'brokers': {},
            'high_risk_brokers': [],
            'medium_risk_brokers': [],
            'recommendations': []
        }
        
        for broker_id in broker_ids:
            # Get prediction summary
            summary = self.get_prediction_summary(
                broker_id=broker_id,
                asset_class=asset_class,
                operation_type=operation_type
            )
            
            # Add to report
            report['brokers'][broker_id] = summary
            
            # Check risk level
            if 'overall_risk' in summary:
                if summary['overall_risk']['level'] == 'high':
                    report['high_risk_brokers'].append(broker_id)
                elif summary['overall_risk']['level'] == 'medium':
                    report['medium_risk_brokers'].append(broker_id)
        
        # Generate recommendations
        if report['high_risk_brokers']:
            report['recommendations'].append({
                'type': 'failover',
                'priority': 'high',
                'description': f"Consider failover from high-risk brokers: {', '.join(report['high_risk_brokers'])}",
                'affected_brokers': report['high_risk_brokers']
            })
        
        if report['medium_risk_brokers']:
            report['recommendations'].append({
                'type': 'monitoring',
                'priority': 'medium',
                'description': f"Increase monitoring for medium-risk brokers: {', '.join(report['medium_risk_brokers'])}",
                'affected_brokers': report['medium_risk_brokers']
            })
        
        return report
