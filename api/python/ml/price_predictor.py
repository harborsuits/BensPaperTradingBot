import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import joblib
from datetime import datetime, timedelta
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("price_predictor")

class PricePredictionModel:
    """
    Machine learning model to predict short-term price movements.
    Uses ensemble methods with technical indicators as features.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.prediction_horizon = self.config.get('prediction_horizon', 5)  # 5 bars ahead
        self.feature_periods = self.config.get('feature_periods', [5, 10, 20, 50])
        self.model = None
        self.scaler = None
        self.last_trained = None
        self.min_training_samples = 1000
        self.model_path = self.config.get('model_path', 'models/price_predictor.joblib')
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def _create_features(self, df):
        """Create technical indicator features from price data."""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        for period in self.feature_periods:
            # Returns
            features[f'return_{period}'] = df['close'].pct_change(period)
            
            # Moving averages
            features[f'ma_{period}'] = df['close'].rolling(period).mean()
            features[f'ma_ratio_{period}'] = df['close'] / features[f'ma_{period}']
            
            # Volatility
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
            
            # Price range
            features[f'range_{period}'] = (df['high'].rolling(period).max() - 
                                          df['low'].rolling(period).min()) / df['close']
        
        # Volume-based features
        for period in self.feature_periods:
            features[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / features[f'volume_ma_{period}']
        
        # Technical indicators
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        features['macd'] = self._calculate_macd(df['close'])
        features['bollinger_upper'], features['bollinger_lower'] = self._calculate_bollinger_bands(df['close'])
        features['distance_to_upper'] = (features['bollinger_upper'] - df['close']) / df['close']
        features['distance_to_lower'] = (df['close'] - features['bollinger_lower']) / df['close']
        
        # Advanced indicators
        features['atr_14'] = self._calculate_atr(df, 14)
        features['cci_20'] = self._calculate_cci(df, 20)
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(df)
        features['obv'] = self._calculate_obv(df)
        features['obv_ratio'] = features['obv'] / features['obv'].rolling(20).mean()
        
        # Momentum indicators
        features['adx'] = self._calculate_adx(df)
        features['ppo'] = self._calculate_ppo(df['close'])
        
        # Create target variable: future return
        features['target'] = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        # Drop NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: pd.Series(x).mad())
        cci = (tp - tp_ma) / (0.015 * md)
        
        return cci
    
    def _calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator."""
        high_rolling = df['high'].rolling(k_period).max()
        low_rolling = df['low'].rolling(k_period).min()
        
        k = 100 * (df['close'] - low_rolling) / (high_rolling - low_rolling)
        d = k.rolling(d_period).mean()
        
        return k, d
    
    def _calculate_obv(self, df):
        """Calculate On-Balance Volume."""
        obv = pd.Series(0, index=df.index)
        
        # Initial value
        obv.iloc[0] = df['volume'].iloc[0]
        
        # Calculate OBV
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        return obv
    
    def _calculate_adx(self, df, period=14):
        """Calculate Average Directional Index."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _calculate_ppo(self, prices, fast=12, slow=26):
        """Calculate Percentage Price Oscillator."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        ppo = (ema_fast - ema_slow) / ema_slow * 100
        
        return ppo
    
    def train(self, historical_data):
        """
        Train the price prediction model.
        
        Args:
            historical_data: DataFrame with OHLCV data
        """
        if len(historical_data) < self.min_training_samples:
            raise ValueError(f"Insufficient data for training. Need at least {self.min_training_samples} samples.")
        
        # Prepare features
        features = self._create_features(historical_data)
        features = features.dropna()
        X = features.drop('target', axis=1)
        y = features['target']
        
        logger.info(f"Training price prediction model with {len(X)} samples")
        
        # Create and train the model pipeline
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ))
        ])
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)
            
            # Calculate metrics
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            
            fold_metrics.append({'mse': mse, 'mae': mae})
            
        logger.info(f"Cross-validation metrics: MSE={np.mean([m['mse'] for m in fold_metrics]):.6f}, MAE={np.mean([m['mae'] for m in fold_metrics]):.6f}")
        
        # Final fit on all data
        model_pipeline.fit(X, y)
        
        self.model = model_pipeline
        self.last_trained = datetime.now()
        
        # Save the model
        joblib.dump(self.model, self.model_path)
        
        # Feature importance
        feature_importance = model_pipeline.named_steps['model'].feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 important features: {', '.join(importance_df['feature'].head(5).tolist())}")
        
        return importance_df
    
    def predict(self, current_data):
        """
        Predict future price movement.
        
        Args:
            current_data: DataFrame with recent OHLCV data
            
        Returns:
            predicted_return: Predicted percentage return over the horizon
            confidence: Prediction confidence score
            direction: Predicted direction ('up', 'down', or 'sideways')
        """
        if self.model is None:
            try:
                self.model = joblib.load(self.model_path)
            except:
                raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        features = self._create_features(current_data)
        features = features.dropna()
        
        if features.empty:
            logger.warning("Not enough data to make a prediction")
            return 0, 0, 'unknown'
            
        X = features.drop('target', axis=1).iloc[-1:] 
        
        # Make prediction
        predicted_return = self.model.predict(X)[0]
        
        # Calculate prediction confidence (based on model's internal confidence)
        if hasattr(self.model.named_steps['model'], 'estimators_'):
            # For ensemble methods, use the variance of predictions from individual trees
            predictions = np.array([tree.predict(X)[0] for tree in self.model.named_steps['model'].estimators_])
            confidence = 1.0 - (np.std(predictions) / (np.abs(np.mean(predictions)) + 1e-8))
            confidence = max(0, min(confidence, 1))  # Bound between 0 and 1
        else:
            confidence = 0.5  # Default confidence
        
        # Determine direction
        if predicted_return > 0.005:  # 0.5% threshold
            direction = 'up'
        elif predicted_return < -0.005:
            direction = 'down'
        else:
            direction = 'sideways'
        
        return predicted_return, confidence, direction
    
    def predict_multiple_horizons(self, current_data, horizons=[1, 5, 10, 20]):
        """
        Predict price movements over multiple time horizons.
        
        Args:
            current_data: DataFrame with recent OHLCV data
            horizons: List of time horizons to predict
            
        Returns:
            dict: Predictions for each horizon
        """
        results = {}
        original_horizon = self.prediction_horizon
        
        for horizon in horizons:
            # Temporarily change prediction horizon
            self.prediction_horizon = horizon
            
            # Make prediction
            try:
                pred_return, confidence, direction = self.predict(current_data)
                results[horizon] = {
                    'predicted_return': pred_return,
                    'confidence': confidence,
                    'direction': direction
                }
            except Exception as e:
                logger.error(f"Error predicting horizon {horizon}: {str(e)}")
                results[horizon] = {
                    'predicted_return': 0,
                    'confidence': 0,
                    'direction': 'unknown',
                    'error': str(e)
                }
            
        # Restore original horizon
        self.prediction_horizon = original_horizon
        
        return results
    
    def should_retrain(self, max_age_days=7):
        """Check if model should be retrained based on age."""
        if self.last_trained is None:
            return True
        
        age = datetime.now() - self.last_trained
        return age.days >= max_age_days 