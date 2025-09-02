#!/usr/bin/env python3
"""
Advanced Market Regime Detector Module

This module provides sophisticated market regime detection using statistical measures
and machine learning techniques to identify different market states.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import joblib
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedMarketRegimeDetector:
    """
    Advanced market regime detection using statistical measures and machine learning.
    """
    def __init__(self, config=None):
        """
        Initialize the market regime detector.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Regime definitions
        self.regimes = self.config.get("regimes", [
            "bullish",             # Strong uptrend, low-medium volatility
            "bearish",             # Strong downtrend, medium-high volatility
            "volatile_bullish",    # Uptrend with high volatility
            "volatile_bearish",    # Downtrend with high volatility
            "sideways",            # Range-bound with medium volatility
            "low_volatility",      # Range-bound with low volatility
            "high_volatility",     # Range-bound with high volatility
            "transition"           # Regime change/transition period
        ])
        
        # Feature windows
        self.windows = self.config.get("windows", {
            "short": 20,   # 1 month
            "medium": 60,  # 3 months
            "long": 120    # 6 months
        })
        
        # Model settings
        self.model_type = self.config.get("model_type", "random_forest")
        self.use_unsupervised = self.config.get("use_unsupervised", True)
        self.model_path = self.config.get("model_path", "models/market_regime_model.joblib")
        
        # Initialize models
        self.supervised_model = None
        self.kmeans_model = None
        self.pca_model = None
        self.scaler = StandardScaler()
        
        # Market data
        self.market_data = None
        
        # Historical regimes
        self.historical_regimes = None
        
        # Cluster to regime mapping
        self.cluster_to_regime = {}
    
    def compute_features(self, prices):
        """
        Compute features for regime detection.
        
        Args:
            prices: DataFrame with OHLCV data
            
        Returns:
            DataFrame: Features for regime detection
        """
        # Basic feature computation
        features = pd.DataFrame(index=prices.index)
        
        # Ensure we have all required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in prices.columns]
        
        if missing_cols:
            self.logger.warning(f"Missing columns: {missing_cols}. Some features may not be computed.")
        
        # 1. Return-based features
        for window in [5, 10, self.windows['short'], self.windows['medium'], self.windows['long']]:
            # Returns
            features[f'return_{window}d'] = prices['close'].pct_change(window)
            
            # Volatility (standard deviation of returns)
            features[f'volatility_{window}d'] = prices['close'].pct_change().rolling(window).std() * np.sqrt(252)
            
            # Drawdown
            rolling_max = prices['close'].rolling(window).max()
            drawdown = (prices['close'] / rolling_max - 1) * 100
            features[f'drawdown_{window}d'] = drawdown
        
        # 2. Trend strength features
        for window in [self.windows['short'], self.windows['medium']]:
            # Trend strength using linear regression R²
            features[f'trend_strength_{window}d'] = features.index.map(
                lambda date: self._compute_trend_strength(prices['close'], date, window)
            )
            
            # Directional strength
            if 'high' in prices.columns and 'low' in prices.columns:
                # Calculate ADX-like measure
                tr = self._calculate_true_range(prices)
                dm_plus = self._calculate_directional_movement(prices, positive=True)
                dm_minus = self._calculate_directional_movement(prices, positive=False)
                
                # Smoothed values
                tr_smooth = tr.rolling(window=14).mean()
                dm_plus_smooth = dm_plus.rolling(window=14).mean()
                dm_minus_smooth = dm_minus.rolling(window=14).mean()
                
                # Directional indicators
                di_plus = 100 * dm_plus_smooth / tr_smooth
                di_minus = 100 * dm_minus_smooth / tr_smooth
                
                # ADX
                dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                features[f'adx_{window}d'] = dx.rolling(window=window).mean()
        
        # Moving average features
        for window in [20, 50, 200]:
            if window <= len(prices):
                features[f'ma_{window}'] = prices['close'].rolling(window=window).mean()
                features[f'ma_dist_{window}'] = (prices['close'] / features[f'ma_{window}'] - 1) * 100
        
        # MACD
        if len(prices) >= 26:
            ema12 = prices['close'].ewm(span=12, adjust=False).mean()
            ema26 = prices['close'].ewm(span=26, adjust=False).mean()
            features['macd'] = ema12 - ema26
            features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
            features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # 3. Volatility features
        # Bollinger Band Width
        if len(prices) >= 20:
            ma20 = prices['close'].rolling(window=20).mean()
            std20 = prices['close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            features['bb_width'] = (upper_band - lower_band) / ma20
        
        # ATR
        if 'high' in prices.columns and 'low' in prices.columns:
            tr = self._calculate_true_range(prices)
            features['atr'] = tr.rolling(window=14).mean()
            features['atr_percent'] = features['atr'] / prices['close'] * 100
        
        # 4. Momentum indicators
        # RSI
        delta = prices['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum
        for window in [10, 20, 50]:
            features[f'momentum_{window}d'] = prices['close'].pct_change(window)
        
        # 5. Volume features
        if 'volume' in prices.columns:
            features['volume_change'] = prices['volume'].pct_change()
            features['volume_ma_ratio'] = prices['volume'] / prices['volume'].rolling(20).mean()
            
            # On-balance volume
            obv = pd.Series(0, index=prices.index)
            for i in range(1, len(prices)):
                if prices['close'].iloc[i] > prices['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + prices['volume'].iloc[i]
                elif prices['close'].iloc[i] < prices['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - prices['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            features['obv'] = obv
            features['obv_ma_ratio'] = obv / obv.rolling(20).mean()
        
        # 6. Market breadth indicators (if provided in data)
        breadth_indicators = ['advance_decline_ratio', 'percent_above_ma50', 'percent_above_ma200']
        for indicator in breadth_indicators:
            if indicator in prices.columns:
                features[indicator] = prices[indicator]
        
        # 7. Correlation and dispersion metrics
        if 'sp500' in prices.columns:
            # Correlation with market index
            for window in [20, 60]:
                features[f'market_corr_{window}d'] = features.index.map(
                    lambda date: self._compute_correlation(prices['close'], prices['sp500'], date, window)
                )
        
        # Golden/Death cross indicator (50 vs 200 day MA)
        if 'ma_50' in features.columns and 'ma_200' in features.columns:
            features['golden_cross'] = (features['ma_50'] > features['ma_200']).astype(int)
        
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_true_range(self, prices):
        """Calculate True Range."""
        high = prices['high']
        low = prices['low']
        close = prices['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def _calculate_directional_movement(self, prices, positive=True):
        """Calculate Directional Movement."""
        if positive:
            dm = prices['high'].diff()
            dm = np.where((dm > 0) & (prices['high'] - prices['high'].shift(1) > prices['low'].shift(1) - prices['low']), 
                          dm, 0)
        else:
            dm = prices['low'].shift(1) - prices['low']
            dm = np.where((dm > 0) & (prices['low'].shift(1) - prices['low'] > prices['high'] - prices['high'].shift(1)), 
                          dm, 0)
        
        return pd.Series(dm, index=prices.index)
    
    def _compute_trend_strength(self, price_series, date, window):
        """Compute trend strength using linear regression."""
        if date not in price_series.index:
            return 0
        
        idx = price_series.index.get_loc(date)
        if idx < window:
            return 0
        
        # Get window of data
        windowed_data = price_series.iloc[idx-window+1:idx+1]
        
        if len(windowed_data) < window / 2:
            return 0
        
        # Log prices for better linear relationship
        log_prices = np.log(windowed_data.values)
        x = np.arange(len(log_prices))
        
        try:
            # Linear regression
            slope, _, r_value, _, _ = stats.linregress(x, log_prices)
            
            # R² as trend strength measure
            trend_strength = r_value ** 2
            
            # Adjust for direction
            if slope < 0:
                trend_strength *= -1
                
            return trend_strength
        except:
            return 0
    
    def _compute_correlation(self, series1, series2, date, window):
        """Compute correlation between two series."""
        if date not in series1.index or date not in series2.index:
            return 0
        
        idx = series1.index.get_loc(date)
        if idx < window:
            return 0
        
        # Get window of data
        s1 = series1.iloc[idx-window+1:idx+1]
        s2 = series2.iloc[idx-window+1:idx+1]
        
        if len(s1) < window / 2:
            return 0
        
        try:
            # Compute correlation
            return s1.corr(s2)
        except:
            return 0
    
    def preprocess_features(self, features, fit=False):
        """Scale features for model input."""
        if fit:
            return self.scaler.fit_transform(features)
        return self.scaler.transform(features)
    
    def train_unsupervised(self, features, n_clusters=8):
        """
        Train unsupervised clustering model for regime detection.
        
        Args:
            features: DataFrame of features
            n_clusters: Number of clusters (regimes)
            
        Returns:
            dict: Training results
        """
        # Preprocess features
        X = self.preprocess_features(features, fit=True)
        
        # Apply PCA for dimensionality reduction
        self.pca_model = PCA(n_components=min(10, X.shape[1]))
        X_pca = self.pca_model.fit_transform(X)
        
        # Train KMeans model
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans_model.fit(X_pca)
        
        # Get cluster labels
        labels = self.kmeans_model.labels_
        
        # Map clusters to interpretable regimes
        cluster_features = pd.DataFrame(X, index=features.index)
        cluster_features['cluster'] = labels
        
        # Create regime profiles
        regime_profiles = {}
        for i in range(n_clusters):
            cluster_data = cluster_features[cluster_features['cluster'] == i]
            
            if len(cluster_data) == 0:
                continue
                
            # Get original features for this cluster
            original_features = features.loc[cluster_data.index]
            
            # Create profile based on available features
            profile = {
                "size": len(cluster_data),
                "percent": len(cluster_data) / len(features) * 100
            }
            
            # Add key metrics if available
            for metric in ['return_20d', 'volatility_20d', 'trend_strength_60d', 'rsi']:
                if metric in original_features.columns:
                    profile[f"avg_{metric}"] = original_features[metric].mean()
            
            # Add ADX if available
            if 'adx_60d' in original_features.columns:
                profile["avg_adx"] = original_features['adx_60d'].mean()
            
            regime_profiles[i] = profile
        
        # Map clusters to named regimes
        self.cluster_to_regime = self._map_clusters_to_regimes(regime_profiles)
        
        # Map regime labels to dates
        regime_labels = pd.Series(
            [self.cluster_to_regime.get(label, "unknown") for label in labels],
            index=features.index
        )
        
        return {
            "profiles": regime_profiles,
            "labels": regime_labels,
            "mapping": self.cluster_to_regime
        }
    
    def _map_clusters_to_regimes(self, profiles):
        """Map cluster numbers to interpretable regime names."""
        mapping = {}
        
        for cluster, profile in profiles.items():
            # Extract key characteristics (use defaults if not available)
            return_20d = profile.get("avg_return_20d", 0) * 100 if "avg_return_20d" in profile else 0
            volatility = profile.get("avg_volatility_20d", 0.15) if "avg_volatility_20d" in profile else 0.15
            trend_strength = profile.get("avg_trend_strength_60d", 0) if "avg_trend_strength_60d" in profile else 0
            adx = profile.get("avg_adx", 15) if "avg_adx" in profile else 15
            rsi = profile.get("avg_rsi", 50) if "avg_rsi" in profile else 50
            
            # Define thresholds
            high_volatility = volatility > 0.2  # 20% annualized
            strong_trend = abs(trend_strength) > 0.7 or adx > 25
            bullish = return_20d > 0 and rsi > 50
            bearish = return_20d < 0 and rsi < 50
            
            # Classify
            if strong_trend and bullish:
                if high_volatility:
                    mapping[cluster] = "volatile_bullish"
                else:
                    mapping[cluster] = "bullish"
            elif strong_trend and bearish:
                if high_volatility:
                    mapping[cluster] = "volatile_bearish"
                else:
                    mapping[cluster] = "bearish"
            elif high_volatility:
                mapping[cluster] = "high_volatility"
            elif volatility < 0.1 and abs(return_20d) < 2:  # 10% annualized volatility
                mapping[cluster] = "low_volatility"
            else:
                mapping[cluster] = "sideways"
        
        return mapping
    
    def train_supervised(self, features, labels):
        """
        Train supervised model for regime detection.
        
        Args:
            features: DataFrame of features
            labels: Series of regime labels
            
        Returns:
            dict: Training results with model performance
        """
        # Preprocess features
        X = self.preprocess_features(features, fit=True)
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define model
        if self.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
            
            # Parameter grid for optimization
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            
            # Use grid search for hyperparameter optimization
            grid_search = GridSearchCV(
                model, param_grid, cv=5, 
                scoring='f1_weighted', n_jobs=-1
            )
            
            # Train model
            grid_search.fit(X_train, y_train)
            
            # Get best model
            self.supervised_model = grid_search.best_estimator_
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Evaluate model
        y_pred = self.supervised_model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        if hasattr(self.supervised_model, 'feature_importances_'):
            importances = self.supervised_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = [(features.columns[i], importances[i]) for i in indices[:10]]
        else:
            top_features = []
        
        # Return training results
        return {
            "classification_report": report,
            "confusion_matrix": conf_matrix,
            "top_features": top_features,
            "model": self.supervised_model
        }
    
    def create_labeled_dataset(self, prices, manual_labels=None):
        """
        Create labeled dataset for supervised learning.
        
        This combines unsupervised clustering with optional manual labels.
        
        Args:
            prices: DataFrame with OHLCV data
            manual_labels: Optional dictionary of {date: regime_label}
            
        Returns:
            tuple: (features, labels)
        """
        # Compute features
        features = self.compute_features(prices)
        
        # If no manual labels provided, use unsupervised clustering
        if manual_labels is None or len(manual_labels) == 0:
            # Train unsupervised model
            results = self.train_unsupervised(features)
            labels = results["labels"]
        else:
            # Convert manual labels to series
            label_series = pd.Series(manual_labels)
            
            # For dates without manual labels, use unsupervised approach
            missing_dates = features.index.difference(label_series.index)
            
            if len(missing_dates) > 0:
                # Train unsupervised model on unlabeled data
                missing_features = features.loc[missing_dates]
                results = self.train_unsupervised(missing_features)
                
                # Combine manual and cluster labels
                labels = pd.Series(index=features.index)
                labels.loc[label_series.index] = label_series
                labels.loc[missing_dates] = results["labels"]
            else:
                labels = label_series.loc[features.index]
        
        return features, labels
    
    def detect_regime(self, prices, date=None, use_supervised=True):
        """
        Detect market regime for a specific date.
        
        Args:
            prices: DataFrame with OHLCV data
            date: Date to detect regime for (default: latest date)
            use_supervised: Whether to use supervised model
            
        Returns:
            str: Detected market regime
        """
        if date is None:
            date = prices.index[-1]
        
        # Compute features
        features = self.compute_features(prices)
        
        # Check if date exists in features
        if date not in features.index:
            closest_date = self._find_closest_date(date, features.index)
            self.logger.warning(f"Date {date} not found in features, using closest date {closest_date}")
            date = closest_date
        
        # Get features for the specific date
        date_features = features.loc[[date]]
        
        # Detect regime
        if use_supervised and self.supervised_model is not None:
            # Use trained supervised model
            X = self.preprocess_features(date_features)
            regime = self.supervised_model.predict(X)[0]
        elif self.kmeans_model is not None and self.pca_model is not None:
            # Use unsupervised model
            X = self.preprocess_features(date_features)
            X_pca = self.pca_model.transform(X)
            cluster = self.kmeans_model.predict(X_pca)[0]
            regime = self.cluster_to_regime.get(cluster, "unknown")
        else:
            # Fallback to simple detection
            regime = self._simple_regime_detection(date_features)
        
        return regime
    
    def _find_closest_date(self, target_date, available_dates):
        """Find closest available date to target date."""
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        closest_date = min(available_dates, key=lambda x: abs((x - target_date).days))
        return closest_date
    
    def _simple_regime_detection(self, features):
        """Simple regime detection based on key features."""
        # Extract key metrics
        if len(features) == 0:
            return "unknown"
        
        # Get first row
        row = features.iloc[0]
        
        # Default to neutral
        regime = "sideways"
        
        # Check volatility
        volatility = row.get('volatility_20d', 0.15)
        high_volatility = volatility > 0.2  # 20% annualized
        low_volatility = volatility < 0.1  # 10% annualized
        
        # Check trend
        return_20d = row.get('return_20d', 0) * 100
        trend_strength = row.get('trend_strength_60d', 0)
        adx = row.get('adx_60d', 15)
        strong_trend = abs(trend_strength) > 0.7 or adx > 25
        
        # Check direction
        rsi = row.get('rsi', 50)
        bullish = return_20d > 0 and rsi > 50
        bearish = return_20d < 0 and rsi < 50
        
        # Classify
        if strong_trend and bullish:
            if high_volatility:
                regime = "volatile_bullish"
            else:
                regime = "bullish"
        elif strong_trend and bearish:
            if high_volatility:
                regime = "volatile_bearish"
            else:
                regime = "bearish"
        elif high_volatility:
            regime = "high_volatility"
        elif low_volatility:
            regime = "low_volatility"
        
        return regime
    
    def detect_regime_history(self, prices, use_supervised=True):
        """
        Detect market regimes for the entire price history.
        
        Args:
            prices: DataFrame with OHLCV data
            use_supervised: Whether to use supervised model
            
        Returns:
            Series: Detected market regimes by date
        """
        # Compute features
        features = self.compute_features(prices)
        
        # Detect regimes
        if use_supervised and self.supervised_model is not None:
            # Use trained supervised model
            X = self.preprocess_features(features)
            regimes = pd.Series(
                self.supervised_model.predict(X),
                index=features.index
            )
        elif self.kmeans_model is not None and self.pca_model is not None:
            # Use unsupervised model
            X = self.preprocess_features(features)
            X_pca = self.pca_model.transform(X)
            clusters = self.kmeans_model.predict(X_pca)
            regimes = pd.Series(
                [self.cluster_to_regime.get(cluster, "unknown") for cluster in clusters],
                index=features.index
            )
        else:
            # Perform simple detection for each date
            regimes = pd.Series(index=features.index)
            for date in features.index:
                regimes[date] = self._simple_regime_detection(features.loc[[date]])
        
        # Store historical regimes
        self.historical_regimes = regimes
        
        return regimes
    
    def detect_regime_transitions(self, regime_series):
        """
        Detect transitions between market regimes.
        
        Args:
            regime_series: Series of market regimes by date
            
        Returns:
            DataFrame: Regime transitions
        """
        if regime_series is None or len(regime_series) == 0:
            return pd.DataFrame()
        
        # Find transitions
        transitions = []
        current_regime = regime_series.iloc[0]
        current_start = regime_series.index[0]
        
        for date, regime in regime_series.iloc[1:].items():
            if regime != current_regime:
                # Record transition
                transitions.append({
                    "from_regime": current_regime,
                    "to_regime": regime,
                    "start_date": current_start,
                    "end_date": date,
                    "duration_days": (date - current_start).days
                })
                
                # Update current regime and start date
                current_regime = regime
                current_start = date
        
        # Add final regime if it hasn't ended
        transitions.append({
            "from_regime": None,
            "to_regime": current_regime,
            "start_date": current_start,
            "end_date": regime_series.index[-1],
            "duration_days": (regime_series.index[-1] - current_start).days
        })
        
        return pd.DataFrame(transitions)
    
    def save_model(self, path=None):
        """Save trained models to disk."""
        if path is None:
            path = self.model_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save models
        model_data = {
            "supervised_model": self.supervised_model,
            "kmeans_model": self.kmeans_model,
            "pca_model": self.pca_model,
            "scaler": self.scaler,
            "cluster_to_regime": getattr(self, "cluster_to_regime", {}),
            "config": self.config
        }
        
        joblib.dump(model_data, path)
        self.logger.info(f"Models saved to {path}")
        
        return path
    
    def load_model(self, path=None):
        """Load trained models from disk."""
        if path is None:
            path = self.model_path
        
        try:
            model_data = joblib.load(path)
            
            self.supervised_model = model_data["supervised_model"]
            self.kmeans_model = model_data["kmeans_model"]
            self.pca_model = model_data["pca_model"]
            self.scaler = model_data["scaler"]
            
            if "cluster_to_regime" in model_data:
                self.cluster_to_regime = model_data["cluster_to_regime"]
            
            if "config" in model_data:
                # Update config but preserve any new settings
                for key, value in model_data["config"].items():
                    if key not in self.config:
                        self.config[key] = value
            
            self.logger.info(f"Models loaded from {path}")
            return True
        except (FileNotFoundError, KeyError) as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
    
    def visualize_regimes(self, prices, regimes=None, title="Market Regimes", save_path=None):
        """
        Visualize market regimes on price chart.
        
        Args:
            prices: DataFrame with price data
            regimes: Series of regimes by date (if None, use historical_regimes)
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib.Figure: Plot figure
        """
        if regimes is None:
            regimes = self.historical_regimes
            
        if regimes is None or len(regimes) == 0:
            self.logger.warning("No regime data available for visualization")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot price
        ax.plot(prices.index, prices['close'], color='black', linewidth=1)
        
        # Define colors for regimes
        regime_colors = {
            "bullish": "green",
            "bearish": "red",
            "volatile_bullish": "lightgreen",
            "volatile_bearish": "salmon",
            "sideways": "gray",
            "low_volatility": "lightblue",
            "high_volatility": "orange",
            "transition": "purple",
            "unknown": "white"
        }
        
        # Plot regime backgrounds
        unique_regimes = regimes.unique()
        
        for regime in unique_regimes:
            # Create mask for this regime
            mask = regimes == regime
            
            # Find contiguous blocks of this regime
            blocks = []
            block_start = None
            
            for date, is_regime in mask.items():
                if is_regime and block_start is None:
                    block_start = date
                elif not is_regime and block_start is not None:
                    blocks.append((block_start, date))
                    block_start = None
            
            # Add the last block if it hasn't been closed
            if block_start is not None:
                blocks.append((block_start, regimes.index[-1]))
            
            # Plot each block
            for start, end in blocks:
                ax.axvspan(start, end, alpha=0.2, color=regime_colors.get(regime, "gray"))
        
        # Add legend
        import matplotlib.patches as mpatches
        
        patches = []
        for regime, color in regime_colors.items():
            if regime in unique_regimes:
                patch = mpatches.Patch(color=color, alpha=0.2, label=regime.replace("_", " ").title())
                patches.append(patch)
        
        ax.legend(handles=patches, loc='upper left')
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_feature_importance(self, features=None, save_path=None):
        """
        Visualize feature importance from the supervised model.
        
        Args:
            features: DataFrame with feature names (optional)
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib.Figure: Plot figure
        """
        if self.supervised_model is None or not hasattr(self.supervised_model, 'feature_importances_'):
            self.logger.warning("No trained supervised model with feature importances available")
            return None
        
        # Get feature importances
        importances = self.supervised_model.feature_importances_
        
        # Get feature names
        if hasattr(self.supervised_model, 'feature_names_in_'):
            feature_names = self.supervised_model.feature_names_in_
        elif features is not None:
            feature_names = features.columns
        else:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        # Take top 20 features
        indices = indices[:min(20, len(indices))]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title('Feature Importance for Market Regime Classification')
        ax.set_xlabel('Relative Importance')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_regime_performance(self, prices, regimes=None):
        """
        Analyze performance characteristics of each market regime.
        
        Args:
            prices: DataFrame with price data
            regimes: Series of regimes by date (if None, use historical_regimes)
            
        Returns:
            DataFrame: Performance metrics by regime
        """
        if regimes is None:
            regimes = self.historical_regimes
            
        if regimes is None or len(regimes) == 0:
            self.logger.warning("No regime data available for analysis")
            return None
        
        # Ensure price data has returns
        prices = prices.copy()
        if 'returns' not in prices.columns:
            prices['returns'] = prices['close'].pct_change()
            
        # Combine prices and regimes
        combined = pd.concat([prices, pd.DataFrame({'regime': regimes})], axis=1)
        combined = combined.dropna()
        
        # Calculate metrics by regime
        results = []
        
        for regime, group in combined.groupby('regime'):
            # Skip if insufficient data
            if len(group) < 5:
                continue
                
            # Calculate metrics
            metrics = {
                'regime': regime,
                'count': len(group),
                'avg_daily_return': group['returns'].mean() * 100,
                'annualized_return': ((1 + group['returns'].mean()) ** 252 - 1) * 100,
                'volatility': group['returns'].std() * np.sqrt(252) * 100,
                'sharpe': (group['returns'].mean() / group['returns'].std()) * np.sqrt(252) if group['returns'].std() > 0 else 0,
                'positive_days': (group['returns'] > 0).mean() * 100,
                'max_drawdown': (group['close'] / group['close'].cummax() - 1).min() * 100
            }
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def get_regime_duration_stats(self, regimes=None):
        """
        Calculate statistics on regime durations.
        
        Args:
            regimes: Series of regimes by date (if None, use historical_regimes)
            
        Returns:
            DataFrame: Duration statistics by regime
        """
        if regimes is None:
            regimes = self.historical_regimes
            
        if regimes is None or len(regimes) == 0:
            self.logger.warning("No regime data available for analysis")
            return None
        
        # Detect regime changes
        transitions = self.detect_regime_transitions(regimes)
        
        # Calculate duration stats by regime
        results = []
        
        for regime, group in transitions.groupby('to_regime'):
            if regime == "None":
                continue
                
            durations = group['duration_days']
            
            stats = {
                'regime': regime,
                'count': len(durations),
                'avg_duration': durations.mean(),
                'min_duration': durations.min(),
                'max_duration': durations.max(),
                'median_duration': durations.median(),
                'total_days': durations.sum()
            }
            
            results.append(stats)
        
        return pd.DataFrame(results)
    
    def get_regime_transition_probabilities(self, regimes=None):
        """
        Calculate probabilities of transitioning between regimes.
        
        Args:
            regimes: Series of regimes by date (if None, use historical_regimes)
            
        Returns:
            DataFrame: Transition probability matrix
        """
        if regimes is None:
            regimes = self.historical_regimes
            
        if regimes is None or len(regimes) == 0:
            self.logger.warning("No regime data available for analysis")
            return None
        
        # Create transition matrix
        transitions = pd.crosstab(
            regimes.shift(),
            regimes,
            normalize='index'
        )
        
        return transitions
    
    def get_current_regime_info(self, prices):
        """
        Get detailed information about the current market regime.
        
        Args:
            prices: DataFrame with OHLCV data
            
        Returns:
            dict: Information about the current regime
        """
        # Detect current regime
        current_regime = self.detect_regime(prices)
        
        # Compute features
        features = self.compute_features(prices)
        last_date = features.index[-1]
        last_features = features.loc[last_date]
        
        # Calculate regime duration
        if self.historical_regimes is None:
            self.detect_regime_history(prices)
            
        regime_duration = 1
        if self.historical_regimes is not None:
            # Count consecutive same regime values
            current_regime_start = None
            for date, regime in self.historical_regimes.iloc[::-1].items():
                if regime == current_regime:
                    if current_regime_start is None:
                        current_regime_start = date
                else:
                    break
                    
            if current_regime_start is not None:
                regime_duration = (last_date - current_regime_start).days + 1
        
        # Get regime performance stats
        if self.historical_regimes is not None:
            regime_performance = self.analyze_regime_performance(prices, self.historical_regimes)
            current_performance = regime_performance[regime_performance['regime'] == current_regime].to_dict(orient='records')
            if current_performance:
                current_performance = current_performance[0]
            else:
                current_performance = {}
        else:
            current_performance = {}
        
        # Create info dictionary
        info = {
            'regime': current_regime,
            'date': last_date,
            'duration_days': regime_duration,
            'key_metrics': {
                'return_20d': last_features.get('return_20d', 0) * 100,
                'volatility_20d': last_features.get('volatility_20d', 0) * 100,
                'rsi': last_features.get('rsi', 50),
                'trend_strength': last_features.get('trend_strength_60d', 0)
            },
            'historical_performance': current_performance
        }
        
        return info


if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download some market data
    data = yf.download("SPY", start="2019-01-01", end="2023-01-01")
    
    # Create regime detector
    detector = AdvancedMarketRegimeDetector()
    
    # Detect regimes
    regimes = detector.detect_regime_history(data)
    
    # Visualize regimes
    detector.visualize_regimes(data, regimes, title="SPY Market Regimes")
    
    # Analyze regime performance
    performance = detector.analyze_regime_performance(data, regimes)
    print(performance)
    
    # Save model
    detector.save_model("market_regime_model.joblib") 