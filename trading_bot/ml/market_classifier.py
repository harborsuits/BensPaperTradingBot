"""
Market Condition Classifier Module

This module provides a classification system for identifying market regimes
(trending, ranging, volatile, etc.) based on various technical indicators.
It helps adjust trading strategies based on current market conditions.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum, auto
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Enum for different market conditions/regimes"""
    STRONG_UPTREND = auto()
    WEAK_UPTREND = auto()
    SIDEWAYS = auto()
    WEAK_DOWNTREND = auto()
    STRONG_DOWNTREND = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()

class MarketConditionClassifier:
    """
    Classifier for identifying current market conditions/regimes.
    
    Uses technical indicators to classify the market into different regimes,
    which can be used to adjust trading strategies accordingly.
    """
    
    def __init__(
        self,
        lookback_window: int = 20,
        ma_short: int = 10,
        ma_medium: int = 50,
        ma_long: int = 200,
        volatility_window: int = 20
    ) -> None:
        """
        Initialize the market condition classifier.
        
        Args:
            lookback_window: Window size for feature calculations
            ma_short: Short-term moving average period
            ma_medium: Medium-term moving average period
            ma_long: Long-term moving average period
            volatility_window: Window for volatility calculations
        """
        self.lookback_window = lookback_window
        self.ma_short = ma_short
        self.ma_medium = ma_medium
        self.ma_long = ma_long
        self.volatility_window = volatility_window
        
        # Initialize classifier
        self.classifier = None
        self.feature_names = []
        self.last_train_time = None
        self.condition_probabilities = {}
        
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for market condition classification.
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with features for classification
        """
        if data is None or data.empty:
            logger.error("Cannot create features: data is empty")
            return pd.DataFrame()
            
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            # Make sure columns are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Calculate moving averages
            df['ma_short'] = df['close'].rolling(self.ma_short).mean()
            df['ma_medium'] = df['close'].rolling(self.ma_medium).mean()
            df['ma_long'] = df['close'].rolling(self.ma_long).mean()
            
            # Calculate moving average slopes (momentum)
            df['ma_short_slope'] = df['ma_short'].diff(5) / 5
            df['ma_medium_slope'] = df['ma_medium'].diff(10) / 10
            df['ma_long_slope'] = df['ma_long'].diff(20) / 20
            
            # Calculate MA crossovers
            df['ma_short_over_medium'] = (df['ma_short'] > df['ma_medium']).astype(int)
            df['ma_medium_over_long'] = (df['ma_medium'] > df['ma_long']).astype(int)
            
            # Calculate price relative to moving averages
            df['price_vs_ma_short'] = df['close'] / df['ma_short'] - 1
            df['price_vs_ma_medium'] = df['close'] / df['ma_medium'] - 1
            df['price_vs_ma_long'] = df['close'] / df['ma_long'] - 1
            
            # Calculate volatility
            df['volatility'] = df['close'].rolling(self.volatility_window).std() / df['close']
            df['volatility_change'] = df['volatility'].pct_change(5)
            
            # Calculate average true range (ATR)
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(self.ma_short).mean()
            df['volume_change'] = df['volume'] / df['volume_ma'] - 1
            
            # Trend strength (ADX-like)
            df['up_move'] = df['high'].diff()
            df['down_move'] = df['low'].diff().mul(-1)
            
            df['plus_dm'] = ((df['up_move'] > df['down_move']) & (df['up_move'] > 0)).astype(int) * df['up_move']
            df['minus_dm'] = ((df['down_move'] > df['up_move']) & (df['down_move'] > 0)).astype(int) * df['down_move']
            
            df['plus_di'] = 100 * df['plus_dm'].rolling(14).mean() / df['atr']
            df['minus_di'] = 100 * df['minus_dm'].rolling(14).mean() / df['atr']
            
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
            df['adx'] = df['dx'].rolling(14).mean()
            
            # Medium-term trend
            df['returns_5d'] = df['close'].pct_change(5)
            df['returns_10d'] = df['close'].pct_change(10)
            df['returns_20d'] = df['close'].pct_change(20)
            
            # Remove NaN values
            df = df.dropna()
            
            # Feature selection (drop raw price and volume data)
            features = df.drop(['open', 'high', 'low', 'close', 'volume', 'tr1', 'tr2', 'tr3'], axis=1)
            
            # Drop date columns if they exist
            if 'date' in features.columns:
                features = features.drop(['date'], axis=1)
            if 'datetime' in features.columns:
                features = features.drop(['datetime'], axis=1)
            
            return features
        
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def _add_labels(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Add market condition labels based on rules.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with added market condition labels
        """
        if features is None or features.empty:
            return pd.DataFrame()
            
        try:
            df = features.copy()
            
            # Create labels based on rules
            conditions = []
            
            # Strong uptrend
            strong_uptrend = (
                (df['ma_short_over_medium'] == 1) & 
                (df['ma_medium_over_long'] == 1) &
                (df['ma_short_slope'] > 0) & 
                (df['ma_medium_slope'] > 0) &
                (df['adx'] > 25)
            )
            conditions.append((strong_uptrend, MarketCondition.STRONG_UPTREND.name))
            
            # Weak uptrend
            weak_uptrend = (
                (df['ma_short_over_medium'] == 1) & 
                ~(df['ma_medium_over_long'] == 1) |
                (df['ma_short_slope'] > 0) & 
                (df['ma_medium_slope'] <= 0) &
                (df['adx'] <= 25)
            )
            conditions.append((weak_uptrend, MarketCondition.WEAK_UPTREND.name))
            
            # Sideways
            sideways = (
                (abs(df['ma_short_slope']) < 0.001) & 
                (abs(df['ma_medium_slope']) < 0.0005) &
                (df['adx'] < 20) &
                (df['volatility'] < df['volatility'].quantile(0.5))
            )
            conditions.append((sideways, MarketCondition.SIDEWAYS.name))
            
            # Strong downtrend
            strong_downtrend = (
                (df['ma_short_over_medium'] == 0) & 
                (df['ma_medium_over_long'] == 0) &
                (df['ma_short_slope'] < 0) & 
                (df['ma_medium_slope'] < 0) &
                (df['adx'] > 25)
            )
            conditions.append((strong_downtrend, MarketCondition.STRONG_DOWNTREND.name))
            
            # Weak downtrend
            weak_downtrend = (
                (df['ma_short_over_medium'] == 0) & 
                ~(df['ma_medium_over_long'] == 0) |
                (df['ma_short_slope'] < 0) & 
                (df['ma_medium_slope'] >= 0) &
                (df['adx'] <= 25)
            )
            conditions.append((weak_downtrend, MarketCondition.WEAK_DOWNTREND.name))
            
            # High volatility
            high_volatility = (
                (df['volatility'] > df['volatility'].quantile(0.75)) &
                (df['atr_pct'] > df['atr_pct'].quantile(0.75))
            )
            conditions.append((high_volatility, MarketCondition.HIGH_VOLATILITY.name))
            
            # Low volatility
            low_volatility = (
                (df['volatility'] < df['volatility'].quantile(0.25)) &
                (df['atr_pct'] < df['atr_pct'].quantile(0.25))
            )
            conditions.append((low_volatility, MarketCondition.LOW_VOLATILITY.name))
            
            # Apply conditions in order of precedence
            df['market_condition'] = MarketCondition.SIDEWAYS.name  # Default
            for condition, value in conditions:
                df.loc[condition, 'market_condition'] = value
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding labels: {e}")
            return pd.DataFrame()
    
    def train(self, data: pd.DataFrame, use_provided_labels: bool = False,
              label_column: str = None) -> bool:
        """
        Train the market condition classifier.
        
        Args:
            data: DataFrame with OHLCV price data
            use_provided_labels: Whether to use provided labels in data
            label_column: Column name for provided labels
            
        Returns:
            True if training successful, False otherwise
        """
        if data is None or data.empty:
            logger.error("Cannot train classifier: data is empty")
            return False
            
        try:
            # Create features
            features_df = self._create_features(data)
            if features_df.empty:
                logger.error("Failed to create features")
                return False
                
            # Add labels if not using provided ones
            if not use_provided_labels:
                features_df = self._add_labels(features_df)
            elif label_column and label_column in data.columns:
                # Use provided labels
                features_df['market_condition'] = data[label_column]
            else:
                logger.error(f"Label column {label_column} not found in data")
                return False
                
            # Split features and target
            X = features_df.drop(['market_condition'], axis=1)
            y = features_df['market_condition']
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create pipeline with scaling and classifier
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ))
            ])
            
            # Train classifier
            pipeline.fit(X_train, y_train)
            
            # Evaluate classifier
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Classifier accuracy: {accuracy:.4f}")
            logger.info("\nClassification Report:\n" + 
                        classification_report(y_test, y_pred))
            
            # Store classifier
            self.classifier = pipeline
            self.last_train_time = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training classifier: {e}")
            return False
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict market condition from price data.
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            Dictionary with predicted market condition and probabilities
        """
        if self.classifier is None:
            logger.error("Classifier not trained")
            return None
            
        if data is None or data.empty:
            logger.error("Cannot predict: data is empty")
            return None
            
        try:
            # Create features
            features_df = self._create_features(data)
            if features_df.empty:
                logger.error("Failed to create features")
                return None
                
            # Get the last row for prediction
            last_row = features_df.iloc[-1:].copy()
            
            # Check if all features are available
            missing_features = [f for f in self.feature_names if f not in last_row.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Handle missing features - only keep features in both
                common_features = [f for f in self.feature_names if f in last_row.columns]
                if not common_features:
                    logger.error("No common features for prediction")
                    return None
                X = last_row[common_features]
            else:
                # Use all trained features
                X = last_row[self.feature_names]
                
            # Make prediction
            condition = self.classifier.predict(X)[0]
            
            # Get probabilities
            probabilities = self.classifier.predict_proba(X)[0]
            
            # Map class indices to class names
            class_names = self.classifier.classes_
            
            # Create dictionary of probabilities
            prob_dict = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
            
            # Store condition probabilities for later use
            self.condition_probabilities = prob_dict
            
            return {
                "market_condition": condition,
                "probabilities": prob_dict,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting market condition: {e}")
            return None
    
    def get_historical_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get historical market conditions for a dataset.
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with original data and market conditions
        """
        if self.classifier is None:
            logger.error("Classifier not trained")
            return data.copy()
            
        try:
            # Create features
            features_df = self._create_features(data)
            if features_df.empty:
                logger.error("Failed to create features")
                return data.copy()
                
            # Check if all features are available
            missing_features = [f for f in self.feature_names if f not in features_df.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Handle missing features - only keep features in both
                common_features = [f for f in self.feature_names if f in features_df.columns]
                if not common_features:
                    logger.error("No common features for prediction")
                    return data.copy()
                X = features_df[common_features]
            else:
                # Use all trained features
                X = features_df[self.feature_names]
                
            # Predict conditions
            conditions = self.classifier.predict(X)
            
            # Add conditions to original data
            result = data.copy()
            
            # Align indices - conditions may have fewer rows due to rolling calculations
            conditions_index = features_df.index
            result = result.loc[conditions_index]
            
            # Add conditions
            result['market_condition'] = conditions
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting historical conditions: {e}")
            return data.copy()
    
    def plot_market_conditions(self, data: pd.DataFrame, title: str = "Market Conditions") -> None:
        """
        Plot historical market conditions with price.
        
        Args:
            data: DataFrame with price data and market conditions
            title: Plot title
        """
        if 'market_condition' not in data.columns:
            logger.error("No market conditions in data. Run get_historical_conditions first.")
            return
            
        try:
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price
            ax1.plot(data.index, data['close'], label='Price')
            
            # Highlight different market conditions
            conditions = data['market_condition'].unique()
            colors = {
                MarketCondition.STRONG_UPTREND.name: 'green',
                MarketCondition.WEAK_UPTREND.name: 'lightgreen',
                MarketCondition.SIDEWAYS.name: 'gray',
                MarketCondition.WEAK_DOWNTREND.name: 'pink',
                MarketCondition.STRONG_DOWNTREND.name: 'red',
                MarketCondition.HIGH_VOLATILITY.name: 'orange',
                MarketCondition.LOW_VOLATILITY.name: 'lightblue'
            }
            
            # Plot market conditions as a categorical heatmap
            cmap = plt.cm.get_cmap('viridis', len(conditions))
            condition_map = {cond: i for i, cond in enumerate(conditions)}
            condition_numeric = data['market_condition'].map(condition_map)
            
            # Second subplot for market conditions
            im = ax2.scatter(data.index, [0] * len(data), c=condition_numeric, 
                          cmap=cmap, marker='s', s=100)
            
            # Set up colorbar
            cbar = plt.colorbar(im, ax=ax2, orientation='horizontal')
            cbar.set_ticks(range(len(conditions)))
            cbar.set_ticklabels(conditions)
            
            # Format
            ax1.set_title(title)
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            ax2.set_yticks([])
            ax2.set_xlabel('Date')
            ax2.set_title('Market Condition')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting market conditions: {e}")
    
    def save_model(self, filepath: str) -> bool:
        """
        Save classifier to file.
        
        Args:
            filepath: Path to save classifier
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self.classifier is None:
            logger.error("No trained classifier to save")
            return False
            
        try:
            state = {
                'classifier': self.classifier,
                'feature_names': self.feature_names,
                'last_train_time': self.last_train_time,
                'lookback_window': self.lookback_window,
                'ma_short': self.ma_short,
                'ma_medium': self.ma_medium,
                'ma_long': self.ma_long,
                'volatility_window': self.volatility_window
            }
            joblib.dump(state, filepath)
            logger.info(f"Classifier saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving classifier: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load classifier from file.
        
        Args:
            filepath: Path to load classifier from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            state = joblib.load(filepath)
            self.classifier = state.get('classifier')
            self.feature_names = state.get('feature_names', [])
            self.last_train_time = state.get('last_train_time')
            self.lookback_window = state.get('lookback_window', self.lookback_window)
            self.ma_short = state.get('ma_short', self.ma_short)
            self.ma_medium = state.get('ma_medium', self.ma_medium)
            self.ma_long = state.get('ma_long', self.ma_long)
            self.volatility_window = state.get('volatility_window', self.volatility_window)
            logger.info(f"Classifier loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
            return False 