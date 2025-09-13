"""
Market Microstructure Anomaly Detector Module

This module provides tools for detecting anomalies in market microstructure
data that could indicate unusual trading activity, market manipulation,
or liquidity issues. The detector helps with risk management by identifying
potentially dangerous market conditions.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyType:
    """Constants for different types of market anomalies"""
    VOLUME_SPIKE = "volume_spike"
    PRICE_SPIKE = "price_spike"
    LIQUIDITY_GAP = "liquidity_gap"
    VOLATILITY_CLUSTER = "volatility_cluster"
    BID_ASK_ANOMALY = "bid_ask_anomaly"
    MOMENTUM_BREAK = "momentum_break"
    ORDER_BOOK_IMBALANCE = "order_book_imbalance"
    TRADE_SIZE_ANOMALY = "trade_size_anomaly"

class AnomalyDetectionConfig:
    """Configuration parameters for anomaly detection"""
    def __init__(
        self,
        lookback_window: int = 100,
        contamination: float = 0.01,
        volume_threshold: float = 3.0,
        price_threshold: float = 3.0,
        volatility_threshold: float = 3.0,
        liquidity_threshold: float = 2.0,
        min_anomaly_score: float = 0.7,
        use_pca: bool = True,
        pca_components: int = 5,
        use_volume_filter: bool = True,
        use_price_filter: bool = True,
        use_liquidity_filter: bool = True,
        use_volatility_filter: bool = True
    ):
        """
        Initialize anomaly detection configuration.
        
        Args:
            lookback_window: Number of bars to use for anomaly detection
            contamination: Expected proportion of anomalies (for IsolationForest)
            volume_threshold: Z-score threshold for volume anomalies
            price_threshold: Z-score threshold for price movement anomalies
            volatility_threshold: Z-score threshold for volatility anomalies
            liquidity_threshold: Z-score threshold for liquidity anomalies
            min_anomaly_score: Minimum score to classify as anomaly (0-1)
            use_pca: Whether to use PCA for feature dimensionality reduction
            pca_components: Number of PCA components to use
            use_volume_filter: Use volume-based anomaly detection
            use_price_filter: Use price-based anomaly detection
            use_liquidity_filter: Use liquidity-based anomaly detection
            use_volatility_filter: Use volatility-based anomaly detection
        """
        self.lookback_window = lookback_window
        self.contamination = contamination
        self.volume_threshold = volume_threshold
        self.price_threshold = price_threshold
        self.volatility_threshold = volatility_threshold
        self.liquidity_threshold = liquidity_threshold
        self.min_anomaly_score = min_anomaly_score
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.use_volume_filter = use_volume_filter
        self.use_price_filter = use_price_filter
        self.use_liquidity_filter = use_liquidity_filter
        self.use_volatility_filter = use_volatility_filter

class MarketAnomalyDetector:
    """
    Detector for market microstructure anomalies.
    
    Uses machine learning and statistical methods to identify unusual patterns
    in market data that may indicate abnormal conditions or manipulation.
    """
    
    def __init__(
        self,
        config: Optional[AnomalyDetectionConfig] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the market anomaly detector.
        
        Args:
            config: Configuration parameters for anomaly detection
            model_path: Path to load a pre-trained model (optional)
        """
        self.config = config or AnomalyDetectionConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None if not self.config.use_pca else PCA(n_components=self.config.pca_components)
        self.last_trained = None
        self.feature_names = []
        self.anomaly_history = []
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
            
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for anomaly detection from price/volume data.
        
        Args:
            data: DataFrame with OHLCV and optional microstructure data
            
        Returns:
            DataFrame with engineered features for anomaly detection
        """
        if data is None or data.empty:
            logger.error("Cannot create features: data is empty")
            return pd.DataFrame()
            
        try:
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # --- Basic OHLCV features ---
            
            # Price changes
            df['return'] = df['close'].pct_change()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            df['high_low_range'] = (df['high'] - df['low']) / df['close']
            df['close_open_range'] = abs(df['close'] - df['open']) / df['open']
            
            # Volume features
            df['volume_change'] = df['volume'].pct_change()
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma']
            
            # Volatility features
            df['return_volatility'] = df['return'].rolling(20).std()
            df['range_volatility'] = df['high_low_range'].rolling(20).std()
            
            # Z-scores
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
            df['return_zscore'] = (df['return'] - df['return'].rolling(50).mean()) / df['return'].rolling(50).std()
            df['range_zscore'] = (df['high_low_range'] - df['high_low_range'].rolling(50).mean()) / df['high_low_range'].rolling(50).std()
            
            # --- Advanced microstructure features if available ---
            
            # Bid-ask spread features if available
            if all(col in df.columns for col in ['bid', 'ask']):
                df['spread'] = df['ask'] - df['bid']
                df['relative_spread'] = df['spread'] / df['close']
                df['spread_zscore'] = (df['spread'] - df['spread'].rolling(50).mean()) / df['spread'].rolling(50).std()
            
            # Order imbalance if available
            if all(col in df.columns for col in ['bid_size', 'ask_size']):
                df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'])
                df['order_imbalance_zscore'] = (df['order_imbalance'] - df['order_imbalance'].rolling(50).mean()) / df['order_imbalance'].rolling(50).std()
            
            # Trade size features if available
            if 'trade_size' in df.columns:
                df['trade_size_ma'] = df['trade_size'].rolling(20).mean()
                df['relative_trade_size'] = df['trade_size'] / df['trade_size_ma']
                df['trade_size_zscore'] = (df['trade_size'] - df['trade_size'].rolling(50).mean()) / df['trade_size'].rolling(50).std()
            
            # --- Momentum and trend features ---
            
            # Price momentum
            df['momentum_1'] = df['close'].pct_change(1)
            df['momentum_5'] = df['close'].pct_change(5)
            df['momentum_10'] = df['close'].pct_change(10)
            
            # Moving average relationships
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            df['ma_ratio'] = df['ma_5'] / df['ma_20']
            
            # Momentum breaks
            df['momentum_change'] = df['momentum_5'].pct_change()
            df['momentum_change_zscore'] = (df['momentum_change'] - df['momentum_change'].rolling(50).mean()) / df['momentum_change'].rolling(50).std()
            
            # --- Liquidity features ---
            
            # Amihud illiquidity measure (approximate)
            df['illiquidity'] = abs(df['return']) / df['volume']
            df['illiquidity_zscore'] = (df['illiquidity'] - df['illiquidity'].rolling(50).mean()) / df['illiquidity'].rolling(50).std()
            
            # --- Time-based features ---
            
            # Add time-of-day features if datetime index is available
            if isinstance(df.index, pd.DatetimeIndex):
                df['hour'] = df.index.hour
                df['minute'] = df.index.minute
                df['day_of_week'] = df.index.dayofweek
            
            # Drop NaN values
            df = df.dropna()
            
            # Select features to include
            features = df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
            
            # Drop date/time columns if they exist
            for col in ['date', 'datetime', 'timestamp']:
                if col in features.columns:
                    features = features.drop(col, axis=1)
            
            # Store feature names
            self.feature_names = list(features.columns)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return pd.DataFrame()
    
    def train(self, data: pd.DataFrame) -> bool:
        """
        Train the anomaly detection model.
        
        Args:
            data: DataFrame with price/volume and optional microstructure data
            
        Returns:
            True if training successful, False otherwise
        """
        if data is None or data.empty:
            logger.error("Cannot train detector: data is empty")
            return False
            
        try:
            # Create features
            features_df = self._create_features(data)
            if features_df.empty:
                logger.error("Failed to create features")
                return False
                
            # Scale features
            X = self.scaler.fit_transform(features_df)
            
            # Apply PCA if configured
            if self.config.use_pca and self.pca:
                X = self.pca.fit_transform(X)
                logger.info(f"PCA reduced features from {features_df.shape[1]} to {X.shape[1]} dimensions")
                
            # Train Isolation Forest
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=self.config.contamination,
                random_state=42
            )
            
            self.model.fit(X)
            self.last_trained = datetime.now()
            
            # Calculate anomaly scores on training data for reference
            scores = self.model.decision_function(X)
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            
            logger.info(f"Trained anomaly detector with {len(features_df)} samples")
            logger.info(f"Average anomaly score: {avg_score:.4f} (std: {std_score:.4f})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
            return False
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market microstructure anomalies in data.
        
        Args:
            data: DataFrame with price/volume and optional microstructure data
            
        Returns:
            DataFrame with original data plus anomaly scores and flags
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return data.copy()
            
        if data is None or data.empty:
            logger.error("Cannot detect anomalies: data is empty")
            return data.copy()
            
        try:
            # Create a copy for results
            result = data.copy()
            
            # Create features
            features_df = self._create_features(data)
            if features_df.empty:
                logger.error("Failed to create features")
                return result
                
            # Get indices to match with original data
            feature_indices = features_df.index
            
            # Scale features
            X = self.scaler.transform(features_df)
            
            # Apply PCA if configured
            if self.config.use_pca and self.pca:
                X = self.pca.transform(X)
                
            # Get anomaly scores (higher values are more normal, lower values more anomalous)
            anomaly_scores = self.model.decision_function(X)
            
            # Predictions (-1 for anomalies, 1 for normal)
            anomaly_predictions = self.model.predict(X)
            
            # Add results to data
            # Align with original data index
            result = result.loc[feature_indices]
            
            # Add anomaly scores (convert to 0-1 scale where higher means more anomalous)
            normalized_scores = 1 - (anomaly_scores - min(anomaly_scores)) / (max(anomaly_scores) - min(anomaly_scores))
            result['anomaly_score'] = normalized_scores
            
            # Add anomaly flag based on isolation forest
            result['is_anomaly'] = (anomaly_predictions == -1).astype(int)
            
            # Rule-based anomaly detection
            result = self._rule_based_anomaly_detection(result)
            
            # Store anomalies in history
            anomalies = result[result['is_anomaly'] == 1].copy()
            if not anomalies.empty:
                for idx, row in anomalies.iterrows():
                    anomaly_types = []
                    for col in result.columns:
                        if col.startswith('anomaly_') and col != 'anomaly_score' and row[col] == 1:
                            anomaly_types.append(col[8:])  # Extract type from column name
                    
                    self.anomaly_history.append({
                        'timestamp': idx,
                        'score': row['anomaly_score'],
                        'anomaly_types': anomaly_types,
                        'price': row['close'],
                        'volume': row['volume']
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return data.copy()
    
    def _rule_based_anomaly_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rule-based anomaly detection to complement ML model.
        
        Args:
            data: DataFrame with anomaly_score and is_anomaly columns
            
        Returns:
            DataFrame with additional anomaly type flags
        """
        df = data.copy()
        
        # Initialize anomaly type columns
        anomaly_types = [
            AnomalyType.VOLUME_SPIKE,
            AnomalyType.PRICE_SPIKE,
            AnomalyType.LIQUIDITY_GAP,
            AnomalyType.VOLATILITY_CLUSTER,
            AnomalyType.BID_ASK_ANOMALY,
            AnomalyType.MOMENTUM_BREAK,
            AnomalyType.ORDER_BOOK_IMBALANCE,
            AnomalyType.TRADE_SIZE_ANOMALY
        ]
        
        for anomaly_type in anomaly_types:
            df[f'anomaly_{anomaly_type}'] = 0
        
        # Volume spike anomalies
        if self.config.use_volume_filter and 'volume_zscore' in df.columns:
            df.loc[abs(df['volume_zscore']) > self.config.volume_threshold, f'anomaly_{AnomalyType.VOLUME_SPIKE}'] = 1
            
        # Price spike anomalies
        if self.config.use_price_filter and 'return_zscore' in df.columns:
            df.loc[abs(df['return_zscore']) > self.config.price_threshold, f'anomaly_{AnomalyType.PRICE_SPIKE}'] = 1
            
        # Volatility cluster anomalies
        if self.config.use_volatility_filter and 'range_zscore' in df.columns:
            df.loc[df['range_zscore'] > self.config.volatility_threshold, f'anomaly_{AnomalyType.VOLATILITY_CLUSTER}'] = 1
            
        # Liquidity gap anomalies
        if self.config.use_liquidity_filter and 'illiquidity_zscore' in df.columns:
            df.loc[df['illiquidity_zscore'] > self.config.liquidity_threshold, f'anomaly_{AnomalyType.LIQUIDITY_GAP}'] = 1
            
        # Bid-ask anomalies
        if 'spread_zscore' in df.columns:
            df.loc[df['spread_zscore'] > self.config.price_threshold, f'anomaly_{AnomalyType.BID_ASK_ANOMALY}'] = 1
            
        # Momentum break anomalies
        if 'momentum_change_zscore' in df.columns:
            df.loc[abs(df['momentum_change_zscore']) > self.config.price_threshold, f'anomaly_{AnomalyType.MOMENTUM_BREAK}'] = 1
            
        # Order book imbalance anomalies
        if 'order_imbalance_zscore' in df.columns:
            df.loc[abs(df['order_imbalance_zscore']) > self.config.price_threshold, f'anomaly_{AnomalyType.ORDER_BOOK_IMBALANCE}'] = 1
            
        # Trade size anomalies
        if 'trade_size_zscore' in df.columns:
            df.loc[abs(df['trade_size_zscore']) > self.config.volume_threshold, f'anomaly_{AnomalyType.TRADE_SIZE_ANOMALY}'] = 1
            
        # Update general anomaly flag based on any specific anomaly type
        anomaly_cols = [col for col in df.columns if col.startswith('anomaly_') and col != 'anomaly_score']
        if anomaly_cols:
            df['is_anomaly'] = df[anomaly_cols].max(axis=1)
            
        # Also mark as anomaly if score is high enough
        df.loc[df['anomaly_score'] > self.config.min_anomaly_score, 'is_anomaly'] = 1
            
        return df
    
    def plot_anomalies(self, data: pd.DataFrame, highlight_types: bool = True, 
                      figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot price data with anomalies highlighted.
        
        Args:
            data: DataFrame with anomaly detection results
            highlight_types: Whether to color-code by anomaly type
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure object
        """
        if 'is_anomaly' not in data.columns:
            logger.error("No anomaly data. Run detect_anomalies first.")
            return None
            
        try:
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, 
                                              gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Plot price
            ax1.plot(data.index, data['close'], label='Price')
            
            # Highlight anomalies
            anomalies = data[data['is_anomaly'] == 1]
            
            # Define colors for anomaly types
            anomaly_colors = {
                AnomalyType.VOLUME_SPIKE: 'purple',
                AnomalyType.PRICE_SPIKE: 'red',
                AnomalyType.LIQUIDITY_GAP: 'orange',
                AnomalyType.VOLATILITY_CLUSTER: 'darkred',
                AnomalyType.BID_ASK_ANOMALY: 'blue',
                AnomalyType.MOMENTUM_BREAK: 'green',
                AnomalyType.ORDER_BOOK_IMBALANCE: 'brown',
                AnomalyType.TRADE_SIZE_ANOMALY: 'cyan'
            }
            
            # Plot anomalies with different colors by type
            if highlight_types:
                for anomaly_type, color in anomaly_colors.items():
                    col = f'anomaly_{anomaly_type}'
                    if col in anomalies.columns:
                        type_anomalies = anomalies[anomalies[col] == 1]
                        if not type_anomalies.empty:
                            ax1.scatter(type_anomalies.index, type_anomalies['close'], 
                                      color=color, s=50, label=anomaly_type)
            else:
                # Simple highlighting all anomalies
                ax1.scatter(anomalies.index, anomalies['close'], color='red', s=50, label='Anomaly')
            
            # Plot volume
            ax2.bar(data.index, data['volume'], color='gray', alpha=0.5, label='Volume')
            
            # Highlight volume anomalies
            if f'anomaly_{AnomalyType.VOLUME_SPIKE}' in data.columns:
                volume_anomalies = data[data[f'anomaly_{AnomalyType.VOLUME_SPIKE}'] == 1]
                if not volume_anomalies.empty:
                    ax2.bar(volume_anomalies.index, volume_anomalies['volume'], color='purple', alpha=0.8, label='Volume Anomaly')
            
            # Plot anomaly scores
            ax3.plot(data.index, data['anomaly_score'], color='blue', label='Anomaly Score')
            ax3.axhline(y=self.config.min_anomaly_score, color='red', linestyle='--', label='Threshold')
            
            # Format
            ax1.set_title('Price with Market Microstructure Anomalies')
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            
            ax2.set_ylabel('Volume')
            ax2.legend(loc='upper left')
            
            ax3.set_ylabel('Anomaly Score')
            ax3.set_xlabel('Date')
            ax3.grid(True)
            ax3.legend(loc='upper left')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting anomalies: {e}")
            return None
    
    def get_anomaly_summary(self, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Get summary statistics of detected anomalies.
        
        Args:
            data: Optional DataFrame with anomaly detection results
            
        Returns:
            Dictionary with anomaly summary statistics
        """
        if data is not None and 'is_anomaly' not in data.columns:
            logger.error("No anomaly data. Run detect_anomalies first.")
            return {}
            
        try:
            # Use history if no data provided
            if data is None:
                if not self.anomaly_history:
                    return {"message": "No anomaly history available"}
                    
                # Count anomalies by type
                type_counts = {}
                for record in self.anomaly_history:
                    for anomaly_type in record['anomaly_types']:
                        type_counts[anomaly_type] = type_counts.get(anomaly_type, 0) + 1
                
                # Calculate average score
                avg_score = sum(record['score'] for record in self.anomaly_history) / len(self.anomaly_history)
                
                return {
                    "total_anomalies": len(self.anomaly_history),
                    "anomaly_types": type_counts,
                    "average_score": avg_score,
                    "last_detected": self.anomaly_history[-1]['timestamp'] if self.anomaly_history else None
                }
            else:
                # Count anomalies from provided data
                anomaly_count = data['is_anomaly'].sum()
                if anomaly_count == 0:
                    return {"message": "No anomalies detected in provided data"}
                
                # Count by type
                type_counts = {}
                for anomaly_type in [
                    AnomalyType.VOLUME_SPIKE,
                    AnomalyType.PRICE_SPIKE, 
                    AnomalyType.LIQUIDITY_GAP,
                    AnomalyType.VOLATILITY_CLUSTER,
                    AnomalyType.BID_ASK_ANOMALY,
                    AnomalyType.MOMENTUM_BREAK,
                    AnomalyType.ORDER_BOOK_IMBALANCE,
                    AnomalyType.TRADE_SIZE_ANOMALY
                ]:
                    col = f'anomaly_{anomaly_type}'
                    if col in data.columns:
                        count = data[col].sum()
                        if count > 0:
                            type_counts[anomaly_type] = count
                
                # Calculate average score for anomalies
                anomalies = data[data['is_anomaly'] == 1]
                avg_score = anomalies['anomaly_score'].mean()
                
                return {
                    "total_anomalies": anomaly_count,
                    "anomaly_rate": anomaly_count / len(data),
                    "anomaly_types": type_counts,
                    "average_score": avg_score,
                    "max_score": data['anomaly_score'].max(),
                    "detection_period": {
                        "start": data.index[0],
                        "end": data.index[-1]
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting anomaly summary: {e}")
            return {"error": str(e)}
    
    def get_latest_anomalies(self, data: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        Get the most recent anomalies from the data.
        
        Args:
            data: DataFrame with anomaly detection results
            n: Number of recent anomalies to return
            
        Returns:
            DataFrame with most recent anomalies
        """
        if 'is_anomaly' not in data.columns:
            logger.error("No anomaly data. Run detect_anomalies first.")
            return pd.DataFrame()
            
        try:
            # Get anomalies and sort by date
            anomalies = data[data['is_anomaly'] == 1].copy()
            if anomalies.empty:
                return pd.DataFrame()
                
            # Sort by index (assuming datetime index)
            anomalies = anomalies.sort_index(ascending=False)
            
            # Get last n anomalies
            recent = anomalies.head(n)
            
            # Add information about which anomaly types triggered
            for anomaly_type in [
                AnomalyType.VOLUME_SPIKE,
                AnomalyType.PRICE_SPIKE,
                AnomalyType.LIQUIDITY_GAP,
                AnomalyType.VOLATILITY_CLUSTER,
                AnomalyType.BID_ASK_ANOMALY,
                AnomalyType.MOMENTUM_BREAK,
                AnomalyType.ORDER_BOOK_IMBALANCE,
                AnomalyType.TRADE_SIZE_ANOMALY
            ]:
                col = f'anomaly_{anomaly_type}'
                if col in data.columns:
                    recent[anomaly_type] = recent[col]
            
            # Select relevant columns for output
            output_cols = ['close', 'volume', 'anomaly_score'] + [
                col for col in recent.columns if col in [
                    AnomalyType.VOLUME_SPIKE,
                    AnomalyType.PRICE_SPIKE,
                    AnomalyType.LIQUIDITY_GAP,
                    AnomalyType.VOLATILITY_CLUSTER,
                    AnomalyType.BID_ASK_ANOMALY,
                    AnomalyType.MOMENTUM_BREAK,
                    AnomalyType.ORDER_BOOK_IMBALANCE,
                    AnomalyType.TRADE_SIZE_ANOMALY
                ]
            ]
            
            return recent[output_cols]
            
        except Exception as e:
            logger.error(f"Error getting latest anomalies: {e}")
            return pd.DataFrame()
    
    def save_model(self, filepath: str) -> bool:
        """
        Save anomaly detection model to file.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            True if saved successfully, False otherwise
        """
        if self.model is None:
            logger.error("No trained model to save")
            return False
            
        try:
            state = {
                'model': self.model,
                'scaler': self.scaler,
                'pca': self.pca,
                'config': self.config,
                'feature_names': self.feature_names,
                'last_trained': self.last_trained,
                'anomaly_history': self.anomaly_history
            }
            joblib.dump(state, filepath)
            logger.info(f"Anomaly detection model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load anomaly detection model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            state = joblib.load(filepath)
            self.model = state.get('model')
            self.scaler = state.get('scaler', StandardScaler())
            self.pca = state.get('pca')
            self.config = state.get('config', self.config)
            self.feature_names = state.get('feature_names', [])
            self.last_trained = state.get('last_trained')
            self.anomaly_history = state.get('anomaly_history', [])
            logger.info(f"Anomaly detection model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
        
    def reset_anomaly_history(self) -> None:
        """Reset the anomaly history."""
        self.anomaly_history = []
        logger.info("Anomaly history reset") 