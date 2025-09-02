"""
Market Condition Classification Module

This module provides a machine learning model for classifying market conditions
based on various technical, fundamental, and sentiment indicators. It identifies
market regimes such as bullish, bearish, sideways, volatile, and trending.
"""

import os
import json
import pickle
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """
    Enumeration of market condition types.
    """
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    SIDEWAYS = "sideways"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"

class MarketConditionClassifier:
    """
    Machine learning model for classifying market conditions.
    
    This class implements a model that:
    1. Creates features from market data, economic indicators, and sentiment data
    2. Trains a model to identify different market regimes
    3. Classifies current market conditions with probability scores
    
    The classifier can identify market regimes such as bullish, bearish, sideways,
    volatile, and trending markets.
    """
    
    def __init__(
        self,
        symbol: str,
        model_dir: str = "models",
        model_type: str = "random_forest",
        features: List[str] = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the market condition classifier.
        
        Args:
            symbol: Trading symbol this model is for
            model_dir: Directory to store trained models
            model_type: Type of ML model to use (random_forest, xgboost, lightgbm)
            features: List of feature groups to use
            config: Additional configuration parameters
        """
        self.symbol = symbol
        self.model_dir = model_dir
        self.model_type = model_type
        self.features = features or ["trend", "volatility", "momentum", "volume", "breadth"]
        self.config = config or {}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Dictionary to store market condition labels
        self.condition_labels = {}
        
        # Load model if available
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load trained model from disk if available.
        """
        model_path = os.path.join(self.model_dir, f"{self.symbol}_market_condition_model.pkl")
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"Loaded market condition model for {self.symbol}")
            except Exception as e:
                logger.error(f"Error loading market condition model for {self.symbol}: {str(e)}")
        
        # Load condition labels if available
        labels_path = os.path.join(self.model_dir, f"{self.symbol}_market_condition_labels.json")
        if os.path.exists(labels_path):
            try:
                with open(labels_path, 'r') as f:
                    self.condition_labels = json.load(f)
                logger.info(f"Loaded market condition labels for {self.symbol}")
            except Exception as e:
                logger.error(f"Error loading market condition labels for {self.symbol}: {str(e)}")
    
    def _save_model(self, model) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model object
        """
        model_path = os.path.join(self.model_dir, f"{self.symbol}_market_condition_model.pkl")
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save training metadata
            metadata = {
                "symbol": self.symbol,
                "model_type": self.model_type,
                "features_used": self.features,
                "trained_at": datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.model_dir, f"{self.symbol}_market_condition_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved market condition model for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving market condition model for {self.symbol}: {str(e)}")
    
    def _save_condition_labels(self, labels_dict: Dict[str, Any]) -> None:
        """
        Save condition labels dictionary to disk.
        
        Args:
            labels_dict: Dictionary mapping dates to market conditions
        """
        labels_path = os.path.join(self.model_dir, f"{self.symbol}_market_condition_labels.json")
        try:
            with open(labels_path, 'w') as f:
                json.dump(labels_dict, f, indent=4)
            logger.info(f"Saved market condition labels for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving market condition labels for {self.symbol}: {str(e)}")
    
    def get_last_trained_date(self) -> Optional[datetime]:
        """
        Get the date when the model was last trained.
        
        Returns:
            Datetime object of last training, or None if never trained
        """
        metadata_path = os.path.join(self.model_dir, f"{self.symbol}_market_condition_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                trained_at = datetime.fromisoformat(metadata.get("trained_at", "2000-01-01T00:00:00"))
                return trained_at
            except Exception as e:
                logger.error(f"Error reading training metadata for {self.symbol}: {str(e)}")
        
        return None
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for market condition classification from market data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added market condition features
        """
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col.upper() in df.columns:
                    # Try uppercase version
                    df[col] = df[col.upper()]
                else:
                    raise ValueError(f"Required column {col} not found in data")
        
        # Create different feature groups based on configuration
        features_to_create = self.features
        
        # Trend indicators
        if "trend" in features_to_create:
            # Calculate price relation to moving averages
            for period in [20, 50, 100, 200]:
                # Moving averages
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                
                # Price relative to MA
                df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
                
                # MA direction (slope)
                df[f'sma_{period}_slope'] = df[f'sma_{period}'].diff(20) / df[f'sma_{period}'].shift(20)
                
                # MA crossovers (shorter period crosses over longer period)
                if period > 20:
                    df[f'sma_20_cross_{period}'] = np.where(
                        (df['sma_20'] > df[f'sma_{period}']) & (df['sma_20'].shift(1) <= df[f'sma_{period}'].shift(1)),
                        1,  # Golden cross (bullish)
                        np.where(
                            (df['sma_20'] < df[f'sma_{period}']) & (df['sma_20'].shift(1) >= df[f'sma_{period}'].shift(1)),
                            -1,  # Death cross (bearish)
                            0  # No cross
                        )
                    )
            
            # ADX (Average Directional Index) for trend strength
            # Calculate +DI and -DI
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff(-1).abs()
            plus_dm = np.where(plus_dm > minus_dm, np.maximum(plus_dm, 0), 0)
            minus_dm = np.where(minus_dm > plus_dm, np.maximum(minus_dm, 0), 0)
            
            # True Range
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Smooth the indicators
            period = 14
            tr_smooth = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_smooth)
            
            # Calculate DX and ADX
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            df['adx'] = dx.rolling(window=period).mean()
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            # Trend strength (ADX interpretations)
            df['strong_trend'] = np.where(df['adx'] > 25, 1, 0)
            df['very_strong_trend'] = np.where(df['adx'] > 50, 1, 0)
            df['trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)
        
        # Volatility indicators
        if "volatility" in features_to_create:
            # Historical volatility (standard deviation of returns)
            df['returns'] = df['close'].pct_change()
            
            for period in [10, 20, 50]:
                # HV calculation (annualized)
                df[f'hv_{period}'] = df['returns'].rolling(window=period).std() * np.sqrt(252)
                
                # Relative volatility (current vol vs. average vol)
                df[f'rel_vol_{period}'] = df[f'hv_{period}'] / df[f'hv_{period}'].rolling(window=100).mean()
            
            # Bollinger Band width as volatility measure
            boll_period = 20
            df['boll_mid'] = df['close'].rolling(window=boll_period).mean()
            df['boll_std'] = df['close'].rolling(window=boll_period).std()
            df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
            df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']
            df['boll_width_percentile'] = df['boll_width'].rolling(window=252).rank(pct=True)
            
            # High volatility regime
            df['high_vol_regime'] = np.where(df['boll_width_percentile'] > 0.8, 1, 0)
            
            # Low volatility regime
            df['low_vol_regime'] = np.where(df['boll_width_percentile'] < 0.2, 1, 0)
            
            # ATR (Average True Range)
            df['atr_14'] = tr.rolling(window=14).mean()
            df['atr_percent'] = df['atr_14'] / df['close'] * 100
            
            # Volatility regime based on ATR %
            df['atr_percentile'] = df['atr_percent'].rolling(window=252).rank(pct=True)
            df['high_atr_regime'] = np.where(df['atr_percentile'] > 0.8, 1, 0)
            df['low_atr_regime'] = np.where(df['atr_percentile'] < 0.2, 1, 0)
        
        # Momentum indicators
        if "momentum" in features_to_create:
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # RSI based regimes
            df['overbought'] = np.where(df['rsi_14'] > 70, 1, 0)
            df['oversold'] = np.where(df['rsi_14'] < 30, 1, 0)
            
            # MACD (Moving Average Convergence Divergence)
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # MACD based signals
            df['macd_positive'] = np.where(df['macd'] > 0, 1, 0)
            df['macd_rising'] = np.where(df['macd'] > df['macd'].shift(1), 1, 0)
            df['macd_above_signal'] = np.where(df['macd'] > df['macd_signal'], 1, 0)
            
            # Price momentum (returns over different periods)
            for period in [5, 10, 20, 60]:
                df[f'return_{period}d'] = df['close'].pct_change(periods=period)
            
            # Rate of change momentum
            df['roc_ratio'] = df['return_20d'] / df['return_60d']
            
            # Momentum regime classifications
            df['strong_momentum'] = np.where(
                (df['return_20d'] > 0) & 
                (df['macd_positive'] == 1) &
                (df['rsi_14'] > 50),
                1, 0
            )
            
            df['weak_momentum'] = np.where(
                (df['return_20d'] < 0) & 
                (df['macd_positive'] == 0) &
                (df['rsi_14'] < 50),
                1, 0
            )
        
        # Volume indicators
        if "volume" in features_to_create:
            # Volume moving averages
            for period in [20, 50]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'relative_volume_{period}'] = df['volume'] / df[f'volume_sma_{period}']
            
            # On-Balance Volume (OBV)
            df['obv_raw'] = np.where(
                df['close'] > df['close'].shift(1),
                df['volume'],
                np.where(
                    df['close'] < df['close'].shift(1),
                    -df['volume'],
                    0
                )
            )
            df['obv'] = df['obv_raw'].cumsum()
            
            # OBV moving average
            df['obv_sma'] = df['obv'].rolling(window=20).mean()
            
            # Volume trends
            df['rising_volume'] = np.where(df['volume'] > df['volume'].shift(1), 1, 0)
            df['volume_spike'] = np.where(df['relative_volume_20'] > 2, 1, 0)
            
            # Chaikin Money Flow (CMF)
            money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            df['cmf_20'] = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # Volume based regime indicators
            df['high_volume_regime'] = np.where(df['relative_volume_20'] > 1.5, 1, 0)
            df['low_volume_regime'] = np.where(df['relative_volume_20'] < 0.5, 1, 0)
            df['bullish_volume'] = np.where((df['cmf_20'] > 0.05) & (df['obv'] > df['obv_sma']), 1, 0)
            df['bearish_volume'] = np.where((df['cmf_20'] < -0.05) & (df['obv'] < df['obv_sma']), 1, 0)
        
        # Market breadth indicators (requires index/sector data)
        if "breadth" in features_to_create and 'breadth_data' in self.config:
            try:
                breadth_data = self.config['breadth_data']
                if isinstance(breadth_data, pd.DataFrame):
                    # Ensure dates align
                    if 'date' in breadth_data.columns and 'date' in df.columns:
                        breadth_data = breadth_data.set_index('date')
                        df = df.set_index('date')
                        
                        # Join breadth data
                        df = df.join(breadth_data, how='left')
                        
                        # Reset index
                        df = df.reset_index()
                    else:
                        # If no date column, assume data is already aligned
                        for col in breadth_data.columns:
                            df[col] = breadth_data[col].values
            except Exception as e:
                logger.warning(f"Error incorporating breadth data: {str(e)}")
        
        # Create composite market regime features
        # Bullish regime
        df['bullish_regime'] = np.where(
            (df['price_to_sma_50'] > 0) &
            (df['sma_50_slope'] > 0) &
            (df['macd_positive'] == 1) &
            (df['rsi_14'] > 50),
            1, 0
        )
        
        # Strong bullish regime
        df['strong_bullish_regime'] = np.where(
            (df['price_to_sma_50'] > 0.05) &
            (df['sma_50_slope'] > 0.001) &
            (df['macd_positive'] == 1) &
            (df['rsi_14'] > 60) &
            (df['bullish_volume'] == 1),
            1, 0
        )
        
        # Bearish regime
        df['bearish_regime'] = np.where(
            (df['price_to_sma_50'] < 0) &
            (df['sma_50_slope'] < 0) &
            (df['macd_positive'] == 0) &
            (df['rsi_14'] < 50),
            1, 0
        )
        
        # Strong bearish regime
        df['strong_bearish_regime'] = np.where(
            (df['price_to_sma_50'] < -0.05) &
            (df['sma_50_slope'] < -0.001) &
            (df['macd_positive'] == 0) &
            (df['rsi_14'] < 40) &
            (df['bearish_volume'] == 1),
            1, 0
        )
        
        # Sideways/consolidation regime
        df['sideways_regime'] = np.where(
            (abs(df['price_to_sma_50']) < 0.03) &
            (abs(df['sma_50_slope']) < 0.0005) &
            (df['adx'] < 20) &
            (df['low_vol_regime'] == 1),
            1, 0
        )
        
        # Volatility expansion regime
        df['volatility_expansion_regime'] = np.where(
            (df['high_vol_regime'] == 1) &
            (df['high_atr_regime'] == 1) &
            (df['adx'] > 30),
            1, 0
        )
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def _label_market_conditions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create market condition labels based on technical indicators and rules.
        
        Args:
            data: DataFrame with OHLCV data and calculated features
            
        Returns:
            DataFrame with added market condition labels
        """
        df = data.copy()
        
        # Create market condition column
        df['market_condition'] = np.nan
        
        # Label strong bullish conditions
        mask_strong_bullish = (
            (df['strong_bullish_regime'] == 1) &
            (df['return_20d'] > 0.05)
        )
        df.loc[mask_strong_bullish, 'market_condition'] = MarketCondition.STRONG_BULLISH.value
        
        # Label bullish conditions
        mask_bullish = (
            (df['bullish_regime'] == 1) &
            (df['return_20d'] > 0.02) &
            (df['market_condition'].isna())
        )
        df.loc[mask_bullish, 'market_condition'] = MarketCondition.BULLISH.value
        
        # Label slightly bullish conditions
        mask_slightly_bullish = (
            (df['bullish_regime'] == 1) &
            (df['market_condition'].isna())
        )
        df.loc[mask_slightly_bullish, 'market_condition'] = MarketCondition.SLIGHTLY_BULLISH.value
        
        # Label strong bearish conditions
        mask_strong_bearish = (
            (df['strong_bearish_regime'] == 1) &
            (df['return_20d'] < -0.05)
        )
        df.loc[mask_strong_bearish, 'market_condition'] = MarketCondition.STRONG_BEARISH.value
        
        # Label bearish conditions
        mask_bearish = (
            (df['bearish_regime'] == 1) &
            (df['return_20d'] < -0.02) &
            (df['market_condition'].isna())
        )
        df.loc[mask_bearish, 'market_condition'] = MarketCondition.BEARISH.value
        
        # Label slightly bearish conditions
        mask_slightly_bearish = (
            (df['bearish_regime'] == 1) &
            (df['market_condition'].isna())
        )
        df.loc[mask_slightly_bearish, 'market_condition'] = MarketCondition.SLIGHTLY_BEARISH.value
        
        # Label sideways conditions
        mask_sideways = (
            (df['sideways_regime'] == 1) &
            (abs(df['return_20d']) < 0.02) &
            (df['market_condition'].isna())
        )
        df.loc[mask_sideways, 'market_condition'] = MarketCondition.SIDEWAYS.value
        
        # Label high volatility conditions
        mask_high_vol = (
            (df['volatility_expansion_regime'] == 1) &
            (df['market_condition'].isna())
        )
        df.loc[mask_high_vol, 'market_condition'] = MarketCondition.HIGH_VOLATILITY.value
        
        # Label low volatility conditions
        mask_low_vol = (
            (df['low_vol_regime'] == 1) &
            (df['low_atr_regime'] == 1) &
            (df['market_condition'].isna())
        )
        df.loc[mask_low_vol, 'market_condition'] = MarketCondition.LOW_VOLATILITY.value
        
        # Label trending conditions
        mask_trend_up = (
            (df['strong_trend'] == 1) &
            (df['trend_direction'] == 1) &
            (df['market_condition'].isna())
        )
        df.loc[mask_trend_up, 'market_condition'] = MarketCondition.TRENDING_UP.value
        
        mask_trend_down = (
            (df['strong_trend'] == 1) &
            (df['trend_direction'] == -1) &
            (df['market_condition'].isna())
        )
        df.loc[mask_trend_down, 'market_condition'] = MarketCondition.TRENDING_DOWN.value
        
        # Default to neutral for any remaining unlabeled data
        df.loc[df['market_condition'].isna(), 'market_condition'] = MarketCondition.NEUTRAL.value
        
        # Store labeled conditions for later reference
        if 'date' in df.columns:
            self.condition_labels = {
                str(date): condition for date, condition in 
                zip(df['date'].astype(str), df['market_condition'])
            }
            self._save_condition_labels(self.condition_labels)
        
        return df
    
    def _init_model(self, model_type: str = None):
        """
        Initialize a machine learning model.
        
        Args:
            model_type: Type of model to initialize
            
        Returns:
            Initialized model instance
        """
        model_type = model_type or self.model_type
        
        if model_type.lower() == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            except ImportError:
                logger.warning("LightGBM not available, falling back to Random Forest")
                model_type = 'random_forest'
        
        if model_type.lower() == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to Random Forest")
                model_type = 'random_forest'
        
        # Default to Random Forest
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the market condition classifier.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with training results
        """
        # Create features
        logger.info(f"Creating features for {self.symbol} market condition classifier")
        df_features = self._create_features(data)
        
        # Label market conditions
        logger.info(f"Labeling market conditions for {self.symbol}")
        df_labeled = self._label_market_conditions(df_features)
        
        # Results dictionary
        results = {
            "success": False,
            "error": None,
            "class_distribution": None
        }
        
        try:
            # Get feature columns (exclude target variable and original OHLCV data)
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'market_condition']
            feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]
            
            # Check if we have enough data
            if len(df_labeled) < 50 or len(feature_cols) < 5:
                error_msg = f"Insufficient data for {self.symbol} market condition classifier: {len(df_labeled)} rows, {len(feature_cols)} features"
                logger.warning(error_msg)
                results["error"] = error_msg
                return results
            
            # Initialize model
            model = self._init_model()
            
            # Train model
            X = df_labeled[feature_cols]
            y = df_labeled['market_condition']
            
            # Handle imbalanced classes
            class_counts = y.value_counts()
            class_weights = {class_name: max(class_counts.values()) / count 
                             for class_name, count in class_counts.items()}
            
            # Train the model with class weights if supported
            if hasattr(model, 'class_weight'):
                model.set_params(class_weight=class_weights)
            
            model.fit(X, y)
            
            # Store model
            self.model = model
            
            # Save model
            self._save_model(model)
            
            # Update results
            results["success"] = True
            results["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
        except Exception as e:
            logger.error(f"Error training market condition classifier for {self.symbol}: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Classify current market conditions.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with market condition predictions and probabilities
        """
        # Check if model is available
        if self.model is None:
            logger.warning(f"No model available for {self.symbol} market condition classifier")
            return {
                "primary_condition": "UNKNOWN",
                "condition_probabilities": {},
                "error": "Model not trained"
            }
        
        try:
            # Create features
            df_features = self._create_features(data)
            
            # Get last row for prediction
            latest_data = df_features.iloc[-1:]
            
            # Get feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'market_condition']
            feature_cols = [col for col in df_features.columns if col not in exclude_cols]
            
            # Ensure all model features are available
            model_features = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else feature_cols
            missing_features = set(model_features) - set(feature_cols)
            
            if missing_features:
                logger.warning(f"Missing features for {self.symbol} market condition prediction: {missing_features}")
                # Fill missing features with zeros
                for feat in missing_features:
                    latest_data[feat] = 0
            
            # Make prediction
            X = latest_data[model_features]
            predicted_class = self.model.predict(X)[0]
            
            # Get probabilities for each class
            probas = self.model.predict_proba(X)[0]
            classes = self.model.classes_
            
            # Prepare results
            result = {
                "primary_condition": predicted_class,
                "condition_probabilities": {
                    str(cls): float(prob) for cls, prob in zip(classes, probas)
                },
                "all_conditions": [str(c) for c in MarketCondition],
                "date": latest_data['date'].iloc[0] if 'date' in latest_data.columns else None
            }
            
            # Add secondary conditions (next highest probability)
            sorted_probs = sorted(
                [(cls, prob) for cls, prob in zip(classes, probas)],
                key=lambda x: x[1],
                reverse=True
            )
            
            if len(sorted_probs) > 1:
                result["secondary_condition"] = sorted_probs[1][0]
                result["secondary_probability"] = float(sorted_probs[1][1])
            
            # Add market features for reference
            for feature in ['adx', 'rsi_14', 'boll_width', 'atr_percent', 'return_20d']:
                if feature in latest_data.columns:
                    result[feature] = float(latest_data[feature].iloc[0])
            
            return result
        except Exception as e:
            logger.error(f"Error predicting market condition for {self.symbol}: {str(e)}")
            return {
                "primary_condition": "ERROR",
                "condition_probabilities": {},
                "error": str(e)
            }
    
    def evaluate_accuracy(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate classification accuracy on test data.
        
        Args:
            test_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Check if model is available
        if self.model is None:
            return {"error": "Model not trained"}
        
        try:
            # Create features
            df_features = self._create_features(test_data)
            
            # Label market conditions
            df_labeled = self._label_market_conditions(df_features)
            
            # Get feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date', 'timestamp', 'market_condition']
            feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]
            
            # Ensure all model features are available
            model_features = self.model.feature_names_in_ if hasattr(self.model, 'feature_names_in_') else feature_cols
            missing_features = set(model_features) - set(feature_cols)
            
            if missing_features:
                for feat in missing_features:
                    df_labeled[feat] = 0
            
            X = df_labeled[model_features]
            y_true = df_labeled['market_condition']
            
            # Make predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, classification_report
            
            # Overall accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Detailed classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Store results
            results = {
                "accuracy": float(accuracy),
                "classification_report": report,
                "sample_count": len(y_true),
                "confusion_matrix": cm.tolist()
            }
            
            return results
        except Exception as e:
            logger.error(f"Error evaluating market condition classifier for {self.symbol}: {str(e)}")
            return {"error": str(e)}
    
    def get_condition_history(self, start_date=None, end_date=None) -> Dict[str, str]:
        """
        Get historical market conditions for a date range.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping dates to market conditions
        """
        if not self.condition_labels:
            return {}
        
        if start_date is None and end_date is None:
            return self.condition_labels
        
        filtered_labels = {}
        
        for date_str, condition in self.condition_labels.items():
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            filtered_labels[date_str] = condition
        
        return filtered_labels 