"""
Market Condition Classifier Module

This module provides tools for classifying market conditions (regimes) based
on price action, volatility, volume patterns, and other market indicators.
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

class MarketCondition:
    """Market condition enumeration with descriptive properties."""
    
    # Main condition types
    BULLISH_TREND = "bullish_trend"
    BEARISH_TREND = "bearish_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    
    # Detailed conditions
    EARLY_BULLISH = "early_bullish"
    LATE_BULLISH = "late_bullish"
    EARLY_BEARISH = "early_bearish"
    LATE_BEARISH = "late_bearish"
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    BULLISH_CONTINUATION = "bullish_continuation"
    BEARISH_CONTINUATION = "bearish_continuation"
    RANGE_BOUND = "range_bound"
    
    # Market types
    TRENDING = "trending"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"

class MarketConditionClassifier:
    """
    Machine learning classifier for market conditions.
    
    This class implements tools for:
    1. Creating features for classifying market conditions
    2. Training classifiers to identify market regimes
    3. Predicting current market conditions with probabilities
    
    The model can be configured to identify various market regimes
    and can be trained for different symbols and timeframes.
    """
    
    def __init__(
        self,
        symbol: str = None,
        lookback_window: int = 20,
        model_dir: str = "models",
        model_type: str = "lightgbm",
        config: Dict[str, Any] = None
    ):
        """
        Initialize the market condition classifier.
        
        Args:
            symbol: Trading symbol to classify
            lookback_window: Number of periods to use for feature calculation
            model_dir: Directory to store trained models
            model_type: Type of ML model to use (lightgbm, xgboost, random_forest)
            config: Additional configuration parameters
        """
        self.symbol = symbol
        self.lookback_window = lookback_window
        self.model_dir = model_dir
        self.model_type = model_type
        self.config = config or {}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.feature_columns = []
        self.feature_importance = {}
        self.scaler = None
        self.label_encoder = None
        self.market_conditions = []
        
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
                self.market_conditions = metadata.get('market_conditions', [])
                
                # Load model
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.model = model_data.get('model')
                self.scaler = model_data.get('scaler')
                self.label_encoder = model_data.get('label_encoder')
                self.feature_importance = model_data.get('feature_importance', {})
                
                logger.info(f"Loaded market condition classifier for {self.symbol}")
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
                'lookback_window': self.lookback_window,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'market_conditions': self.market_conditions,
                'created_at': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved market condition classifier for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving model for {self.symbol}: {str(e)}")
    
    def _get_model_path(self) -> str:
        """
        Get the path to the model file.
        
        Returns:
            Path to the model file
        """
        model_name = f"{self.symbol}_market_condition"
        return os.path.join(self.model_dir, f"{model_name}.pkl")
    
    def _get_metadata_path(self) -> str:
        """
        Get the path to the model metadata file.
        
        Returns:
            Path to the model metadata file
        """
        model_name = f"{self.symbol}_market_condition"
        return os.path.join(self.model_dir, f"{model_name}_metadata.json")
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for market condition classification.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with features for market condition classification
        """
        df = data.copy()
        
        # Ensure necessary columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            missing_cols_str = ', '.join(missing_cols)
            logger.error(f"Required columns missing: {missing_cols_str}")
            raise ValueError(f"Required columns missing: {missing_cols_str}")
        
        # Technical indicators for market condition
        
        # Trend indicators
        
        # Moving averages and their relationships
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            
            # Only calculate relationships for windows that we have enough data for
            if window <= len(df):
                # Price relative to moving average
                df[f'close_to_ma_{window}'] = df['close'] / df[f'ma_{window}'] - 1
                
                # Moving average slope (momentum)
                df[f'ma_{window}_slope'] = df[f'ma_{window}'].diff(5) / df[f'ma_{window}'].shift(5)
        
        # Moving average crossovers
        df['ma_5_10_cross'] = np.where(
            df['ma_5'] > df['ma_10'], 1, 
            np.where(df['ma_5'] < df['ma_10'], -1, 0)
        )
        
        df['ma_10_20_cross'] = np.where(
            df['ma_10'] > df['ma_20'], 1, 
            np.where(df['ma_10'] < df['ma_20'], -1, 0)
        )
        
        df['ma_20_50_cross'] = np.where(
            df['ma_20'] > df['ma_50'], 1, 
            np.where(df['ma_20'] < df['ma_50'], -1, 0)
        )
        
        df['ma_50_200_cross'] = np.where(
            df['ma_50'] > df['ma_200'], 1, 
            np.where(df['ma_50'] < df['ma_200'], -1, 0)
        )
        
        # Volatility indicators
        
        # Average True Range (ATR)
        for window in [5, 10, 14, 20]:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{window}'] = tr.rolling(window).mean()
            
            # ATR as percentage of price
            df[f'atr_pct_{window}'] = df[f'atr_{window}'] / df['close']
            
            # ATR trend
            df[f'atr_{window}_trend'] = df[f'atr_{window}'].diff(3) / df[f'atr_{window}'].shift(3)
        
        # Bollinger Bands
        for window in [20]:
            df[f'bb_middle_{window}'] = df['close'].rolling(window=window).mean()
            df[f'bb_std_{window}'] = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = df[f'bb_middle_{window}'] + 2 * df[f'bb_std_{window}']
            df[f'bb_lower_{window}'] = df[f'bb_middle_{window}'] - 2 * df[f'bb_std_{window}']
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
            df[f'bb_pct_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            
            # Bollinger Band width trend
            df[f'bb_width_{window}_trend'] = df[f'bb_width_{window}'].diff(5) / df[f'bb_width_{window}'].shift(5)
        
        # Momentum indicators
        
        # RSI
        for window in [6, 14]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
            # RSI trend
            df[f'rsi_{window}_trend'] = df[f'rsi_{window}'].diff(3)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_hist_trend'] = df['macd_hist'].diff(3)
        
        # Rate of Change
        for window in [5, 10, 20]:
            df[f'roc_{window}'] = df['close'].pct_change(periods=window) * 100
        
        # Volume indicators
        
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            df[f'volume_ma_{window}'] = df['volume'].rolling(window=window).mean()
            df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ma_10'] = df['obv'].rolling(window=10).mean()
        df['obv_slope'] = df['obv'].diff(5) / df['obv'].shift(5)
        
        # Price pattern indicators
        
        # Candlestick body size and wick sizes
        df['body_size'] = abs(df['close'] - df['open']) / ((df['high'] - df['low']) + 0.001)
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / ((df['high'] - df['low']) + 0.001)
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / ((df['high'] - df['low']) + 0.001)
        
        # Doji detection (small body)
        df['is_doji'] = df['body_size'] < 0.1
        
        # Hammer/Hanging man detection (small body, small/no upper wick, long lower wick)
        df['is_hammer'] = (df['body_size'] < 0.3) & (df['upper_wick'] < 0.1) & (df['lower_wick'] > 0.5)
        
        # Shooting star/Inverted hammer detection (small body, long upper wick, small/no lower wick)
        df['is_shooting_star'] = (df['body_size'] < 0.3) & (df['upper_wick'] > 0.5) & (df['lower_wick'] < 0.1)
        
        # Market type indicators
        
        # Trending vs ranging
        # Calculate ADX (Average Directional Index)
        window = 14
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/window).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/window).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
        adx = pd.Series(dx).ewm(alpha=1/window).mean()
        
        df['adx'] = adx
        df['trending'] = df['adx'] > 25
        df['strong_trending'] = df['adx'] > 40
        
        # Determine market type based on ADX and other indicators
        df['market_type'] = 'unknown'
        
        # Trending market
        df.loc[df['adx'] > 25, 'market_type'] = MarketCondition.TRENDING
        
        # Mean reverting market
        df.loc[(df['adx'] < 20) & (df['bb_width_20'] < df['bb_width_20'].rolling(50).mean() * 0.8), 'market_type'] = MarketCondition.MEAN_REVERSION
        
        # Momentum market
        df.loc[(df['adx'] > 25) & (df['roc_10'].abs() > df['roc_10'].abs().rolling(50).mean() * 1.5), 'market_type'] = MarketCondition.MOMENTUM
        
        # Convert boolean columns to int for machine learning
        for col in ['is_doji', 'is_hammer', 'is_shooting_star', 'trending', 'strong_trending']:
            df[col] = df[col].astype(int)
        
        # Add custom market regime labels if provided
        if 'market_regimes' in self.config and isinstance(self.config['market_regimes'], pd.DataFrame):
            regimes_df = self.config['market_regimes']
            
            # Join with main dataframe based on date/timestamp
            if 'date' in df.columns and 'date' in regimes_df.columns:
                df = pd.merge(df, regimes_df, on='date', how='left')
            elif 'timestamp' in df.columns and 'timestamp' in regimes_df.columns:
                df = pd.merge(df, regimes_df, on='timestamp', how='left')
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train(
        self,
        data: pd.DataFrame,
        labels: Optional[pd.Series] = None,
        test_size: float = 0.2,
        evaluation_metric: str = None,
        use_default_regime_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Train the market condition classifier.
        
        Args:
            data: DataFrame with OHLCV data
            labels: Optional series with manually labeled market conditions
            test_size: Proportion of data to use for testing
            evaluation_metric: Metric to use for evaluation
            use_default_regime_detection: Whether to use default regime detection
                if manual labels are not provided
            
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
            exclude_cols = ['date', 'timestamp', 'datetime', 'symbol', 'open', 'high', 'low', 
                          'close', 'volume', 'market_condition', 'market_type', 'market_regime']
            
            feature_cols = [col for col in df.columns if col not in exclude_cols 
                         and not col.startswith('target_')]
            
            self.feature_columns = feature_cols
            
            # Determine target labels
            if labels is not None:
                # Use provided labels
                if isinstance(labels, pd.Series):
                    if len(labels) != len(df):
                        raise ValueError("Length of labels must match length of data")
                    df['market_condition'] = labels.values
                elif isinstance(labels, dict):
                    # Labels provided as dictionary with date/timestamp keys
                    if 'date' in df.columns:
                        df['market_condition'] = df['date'].map(labels)
                    elif 'timestamp' in df.columns:
                        df['market_condition'] = df['timestamp'].map(labels)
                    else:
                        raise ValueError("Cannot map labels: no date or timestamp column found")
            elif 'market_condition' not in df.columns and use_default_regime_detection:
                # Apply default market condition detection rules
                df['market_condition'] = self._detect_market_conditions(df)
            
            if 'market_condition' not in df.columns:
                raise ValueError("No market condition labels provided or detected")
            
            # Store unique market conditions
            self.market_conditions = df['market_condition'].unique().tolist()
            
            # Prepare data
            X = df[feature_cols]
            y = df['market_condition']
            
            # Balance classes if needed
            from sklearn.utils import class_weight
            class_weights = class_weight.compute_class_weight(
                'balanced', classes=np.unique(y), y=y)
            class_weight_dict = {c: w for c, w in zip(np.unique(y), class_weights)}
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Encode target labels
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42, shuffle=False
            )
            
            # Train model based on model type
            if self.model_type.lower() == 'lightgbm':
                self._train_lightgbm(
                    X_train, X_test, y_train, y_test, 
                    class_weight_dict, evaluation_metric
                )
            elif self.model_type.lower() == 'xgboost':
                self._train_xgboost(
                    X_train, X_test, y_train, y_test,
                    class_weight_dict, evaluation_metric
                )
            else:
                # Default to random forest
                self._train_random_forest(
                    X_train, X_test, y_train, y_test,
                    class_weight_dict
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
                "market_conditions": self.market_conditions,
                "feature_importance": self.feature_importance
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _detect_market_conditions(self, df: pd.DataFrame) -> pd.Series:
        """
        Apply rule-based market condition detection.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Series with detected market conditions
        """
        conditions = pd.Series(index=df.index, dtype='object')
        
        # Initialize with default condition
        conditions[:] = MarketCondition.SIDEWAYS
        
        # Bullish trend
        mask_bullish = (
            (df['ma_5'] > df['ma_20']) & 
            (df['ma_20'] > df['ma_50']) & 
            (df['ma_20_slope'] > 0)
        )
        conditions[mask_bullish] = MarketCondition.BULLISH_TREND
        
        # Early bullish trend
        mask_early_bullish = (
            (df['ma_5'] > df['ma_20']) & 
            (df['ma_20'] < df['ma_50']) & 
            (df['ma_20_slope'] > 0)
        )
        conditions[mask_early_bullish] = MarketCondition.EARLY_BULLISH
        
        # Late bullish trend
        mask_late_bullish = (
            (df['ma_5'] > df['ma_20']) & 
            (df['ma_20'] > df['ma_50']) & 
            (df['ma_20_slope'] < 0) &
            (df['rsi_14'] > 70)
        )
        conditions[mask_late_bullish] = MarketCondition.LATE_BULLISH
        
        # Bearish trend
        mask_bearish = (
            (df['ma_5'] < df['ma_20']) & 
            (df['ma_20'] < df['ma_50']) & 
            (df['ma_20_slope'] < 0)
        )
        conditions[mask_bearish] = MarketCondition.BEARISH_TREND
        
        # Early bearish trend
        mask_early_bearish = (
            (df['ma_5'] < df['ma_20']) & 
            (df['ma_20'] > df['ma_50']) & 
            (df['ma_20_slope'] < 0)
        )
        conditions[mask_early_bearish] = MarketCondition.EARLY_BEARISH
        
        # Late bearish trend
        mask_late_bearish = (
            (df['ma_5'] < df['ma_20']) & 
            (df['ma_20'] < df['ma_50']) & 
            (df['ma_20_slope'] > 0) &
            (df['rsi_14'] < 30)
        )
        conditions[mask_late_bearish] = MarketCondition.LATE_BEARISH
        
        # High volatility
        mask_high_volatility = (
            (df['atr_pct_14'] > df['atr_pct_14'].rolling(100).mean() * 1.5) &
            (df['bb_width_20'] > df['bb_width_20'].rolling(100).mean() * 1.3)
        )
        conditions[mask_high_volatility] = MarketCondition.HIGH_VOLATILITY
        
        # Low volatility
        mask_low_volatility = (
            (df['atr_pct_14'] < df['atr_pct_14'].rolling(100).mean() * 0.7) &
            (df['bb_width_20'] < df['bb_width_20'].rolling(100).mean() * 0.7)
        )
        conditions[mask_low_volatility] = MarketCondition.LOW_VOLATILITY
        
        # Overbought
        mask_overbought = (
            (df['rsi_14'] > 70) &
            (df['bb_pct_20'] > 0.8)
        )
        conditions[mask_overbought] = MarketCondition.OVERBOUGHT
        
        # Oversold
        mask_oversold = (
            (df['rsi_14'] < 30) &
            (df['bb_pct_20'] < 0.2)
        )
        conditions[mask_oversold] = MarketCondition.OVERSOLD
        
        # Bullish reversal
        mask_bullish_reversal = (
            (df['rsi_14'] < 30) &
            (df['rsi_14_trend'] > 0) &
            (df['ma_5'] < df['ma_20']) &
            (df['ma_5'].diff() > 0) &
            (df['is_hammer'] == 1)
        )
        conditions[mask_bullish_reversal] = MarketCondition.BULLISH_REVERSAL
        
        # Bearish reversal
        mask_bearish_reversal = (
            (df['rsi_14'] > 70) &
            (df['rsi_14_trend'] < 0) &
            (df['ma_5'] > df['ma_20']) &
            (df['ma_5'].diff() < 0) &
            (df['is_shooting_star'] == 1)
        )
        conditions[mask_bearish_reversal] = MarketCondition.BEARISH_REVERSAL
        
        # Breakout
        mask_breakout = (
            (df['close'] > df['high_20'].shift()) &
            (df['volume'] > df['volume_ma_20'] * 1.5)
        )
        conditions[mask_breakout] = MarketCondition.BREAKOUT
        
        # Breakdown
        mask_breakdown = (
            (df['close'] < df['low_20'].shift()) &
            (df['volume'] > df['volume_ma_20'] * 1.5)
        )
        conditions[mask_breakdown] = MarketCondition.BREAKDOWN
        
        # Range bound
        mask_range_bound = (
            (df['adx'] < 20) &
            (df['bb_width_20'] < df['bb_width_20'].mean() * 0.8)
        )
        conditions[mask_range_bound] = MarketCondition.RANGE_BOUND
        
        return conditions
    
    def _train_lightgbm(
        self,
        X_train, X_test, y_train, y_test,
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
            'max_depth': 6,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': evaluation_metric or 'multi_logloss'
        }
        
        if class_weights:
            # Convert class weights to sample weights
            sample_weights = np.ones(len(y_train))
            for cls, weight in class_weights.items():
                sample_weights[y_train == cls] = weight
            
            train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        
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
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'multi:softprob',
            'num_class': len(np.unique(y_train)),
            'eval_metric': evaluation_metric or 'mlogloss'
        }
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        
        # Fit model
        if class_weights:
            # Convert class weights to sample weights
            sample_weights = np.ones(len(y_train))
            for cls, weight in class_weights.items():
                sample_weights[y_train == cls] = weight
        else:
            sample_weights = None
        
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
        class_weights: Dict[int, float] = None
    ) -> None:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            class_weights: Class weights for handling imbalance
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Convert class weights to sklearn format
        if class_weights:
            sklearn_class_weights = {
                int(cls): float(weight) for cls, weight in class_weights.items()
            }
        else:
            sklearn_class_weights = None
        
        # Set parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': sklearn_class_weights
        }
        
        # Train model
        self.model = RandomForestClassifier(**params)
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
        Predict market conditions.
        
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
            y_pred = self.model.predict(X_scaled)
            
            # Get predicted class names
            if self.label_encoder is not None:
                predicted_conditions = self.label_encoder.inverse_transform(y_pred)
            else:
                predicted_conditions = y_pred
                
            # Get class probabilities if available
            if hasattr(self.model, 'predict_proba') and include_probabilities:
                probabilities = self.model.predict_proba(X_scaled)
            else:
                probabilities = None
                
            # Format results
            results = []
            for i, condition in enumerate(predicted_conditions):
                entry = {
                    "market_condition": condition,
                    "confidence": 1.0  # Default confidence
                }
                
                if probabilities is not None:
                    # Find the index of the predicted class
                    if self.label_encoder is not None:
                        class_idx = np.where(self.label_encoder.classes_ == condition)[0][0]
                    else:
                        class_idx = np.where(self.model.classes_ == condition)[0][0]
                    
                    # Get confidence (probability) for the predicted class
                    entry["confidence"] = float(probabilities[i, class_idx])
                    
                    if include_probabilities:
                        # Get all class probabilities
                        probs_dict = {}
                        for j, prob in enumerate(probabilities[i]):
                            if self.label_encoder is not None:
                                condition_name = self.label_encoder.inverse_transform([j])[0]
                            else:
                                condition_name = self.model.classes_[j]
                                
                            probs_dict[condition_name] = float(prob)
                        
                        entry["probabilities"] = probs_dict
                
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
            
            # Check if market condition labels are available
            if 'market_condition' not in df.columns:
                return {
                    "status": "error",
                    "error": "No market condition labels available for evaluation"
                }
            
            # Extract features and target
            X = df[self.feature_columns]
            y_true = df['market_condition']
            
            # Encode true labels
            if self.label_encoder is not None:
                y_true_encoded = self.label_encoder.transform(y_true)
            else:
                # Try to use model classes if available
                if hasattr(self.model, 'classes_'):
                    # Create a simple mapping
                    class_to_idx = {cls: i for i, cls in enumerate(self.model.classes_)}
                    y_true_encoded = np.array([class_to_idx.get(cls, -1) for cls in y_true])
                else:
                    # Fallback to original
                    y_true_encoded = y_true
            
            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Make predictions
            y_pred_encoded = self.model.predict(X_scaled)
            
            # Convert predictions back to original labels
            if self.label_encoder is not None:
                y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            else:
                y_pred = y_pred_encoded
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate precision, recall and F1 with different averaging methods
            metrics = {}
            
            for avg in ['micro', 'macro', 'weighted']:
                metrics[f'precision_{avg}'] = precision_score(
                    y_true_encoded, y_pred_encoded, average=avg, zero_division=0)
                metrics[f'recall_{avg}'] = recall_score(
                    y_true_encoded, y_pred_encoded, average=avg, zero_division=0)
                metrics[f'f1_{avg}'] = f1_score(
                    y_true_encoded, y_pred_encoded, average=avg, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true_encoded, y_pred_encoded)
            
            # Convert confusion matrix to dictionary
            cm_dict = {}
            
            if self.label_encoder is not None:
                labels = self.label_encoder.classes_
            elif hasattr(self.model, 'classes_'):
                labels = self.model.classes_
            else:
                labels = np.unique(np.concatenate([y_true_encoded, y_pred_encoded]))
            
            for i, true_label in enumerate(labels):
                for j, pred_label in enumerate(labels):
                    if i < cm.shape[0] and j < cm.shape[1]:
                        if self.label_encoder is not None:
                            true_name = self.label_encoder.inverse_transform([true_label])[0]
                            pred_name = self.label_encoder.inverse_transform([pred_label])[0]
                        else:
                            true_name = str(true_label)
                            pred_name = str(pred_label)
                        
                        cm_dict[f"{true_name}_{pred_name}"] = int(cm[i, j])
            
            results = {
                "status": "success",
                "accuracy": float(accuracy),
                **{k: float(v) for k, v in metrics.items()},
                "confusion_matrix": cm_dict
            }
            
            if include_report:
                from sklearn.metrics import classification_report
                report = classification_report(y_true_encoded, y_pred_encoded, output_dict=True)
                results["classification_report"] = report
            
            return results
        
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
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