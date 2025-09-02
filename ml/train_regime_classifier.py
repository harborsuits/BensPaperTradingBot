#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Classifier Training Script

This script trains a machine learning model to classify market regimes based on
technical indicators and price patterns. The trained model is saved for use by
the forex strategy selector.

Features used for classification:
- Moving average relationships (fast vs slow)
- ADX (trend strength)
- Volatility measures (ATR, ATR%)
- RSI (momentum and overbought/oversold)
- MACD 
- Bollinger Band width
- Price patterns (HH/HL for uptrends, LL/LH for downtrends)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import talib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from trading_bot.strategies.strategy_template import MarketRegime

# Configuration
OUTPUT_DIR = os.path.join(project_root, 'ml/models')
MODEL_FILENAME = 'forex_regime_classifier.joblib'
RANDOM_SEED = 42
TEST_SIZE = 0.2
FEATURE_WINDOW = 20  # Look back window for regime classification

def create_features(data, window=FEATURE_WINDOW):
    """
    Extract technical features for regime classification.
    
    Args:
        data: DataFrame with OHLCV data
        window: Lookback window for feature calculation
        
    Returns:
        DataFrame with extracted features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # ============ PRICE-BASED FEATURES ============
    # Basic returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Higher timeframe returns (multi-period momentum)
    for period in [3, 5, 10, 20]:
        df[f'returns_{period}d'] = df['close'].pct_change(period) 
        df[f'returns_{period}d_normalized'] = df[f'returns_{period}d'] / (df['atr'].rolling(period).mean() / df['close'] if 'atr' in df else 0.01)
    
    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['higher_close'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
    df['lower_close'] = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
    
    # HLC average price and ranges
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['range'] = df['high'] - df['low']
    df['range_pct'] = df['range'] / df['close'] * 100
    
    # ============ MOVING AVERAGES ============
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
    
    # Exponential Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
    
    # Weighted Moving Averages (more weight to recent price)
    for period in [10, 20, 50]:
        df[f'wma_{period}'] = talib.WMA(df['close'], timeperiod=period)
    
    # Moving average slopes and relationships
    for period in [5, 10, 20, 50, 100]:
        # Calculate slope (percentage change over 5 periods)
        df[f'sma_{period}_slope'] = df[f'sma_{period}'].pct_change(5) * 100
        df[f'ema_{period}_slope'] = df[f'ema_{period}'].pct_change(5) * 100
        
        # Longer MA crossovers (e.g., 10-period crossing 20-period)
        if period < 100:
            next_ma = next(p for p in [20, 50, 100, 200] if p > period)
            df[f'sma_{period}_{next_ma}_cross'] = df[f'sma_{period}'] / df[f'sma_{next_ma}'] - 1
            df[f'ema_{period}_{next_ma}_cross'] = df[f'ema_{period}'] / df[f'ema_{next_ma}'] - 1
    
    # Price to Moving Average Relationships
    for period in [10, 20, 50, 100, 200]:
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
    
    # ============ VOLATILITY MEASURES ============
    # ATR and related
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr_percent'] = df['atr'] / df['close'] * 100
    
    # ATR ratio (current vs past)
    df['atr_ratio_10d'] = df['atr'] / df['atr'].rolling(10).mean()
    df['atr_ratio_20d'] = df['atr'] / df['atr'].rolling(20).mean()
    
    # Bollinger Bands and derivatives
    for period, dev in [(20, 2), (50, 2), (20, 1.5)]:
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=period, nbdevup=dev, nbdevdn=dev)
        df[f'bb_width_{period}_{int(dev*10)}'] = (upper - lower) / middle * 100
        df[f'bb_pos_{period}_{int(dev*10)}'] = (df['close'] - lower) / (upper - lower) if upper is not lower else 0.5
    
    # Historical volatility
    for period in [5, 10, 20, 50]:
        df[f'hist_vol_{period}'] = df['returns'].rolling(period).std() * np.sqrt(252) * 100  # Annualized
    
    # ============ TREND STRENGTH & DIRECTION ============
    # ADX (trend strength)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['adx_slope'] = df['adx'].pct_change(5) * 100
    
    # Directional indicators
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
    df['di_diff'] = df['plus_di'] - df['minus_di']
    df['di_sum'] = df['plus_di'] + df['minus_di']
    df['di_ratio'] = df['plus_di'] / df['minus_di'] if 'minus_di' in df else 1.0
    
    # Aroon indicators (trend identification and strength)
    df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'], timeperiod=25)
    df['aroon_osc'] = df['aroon_up'] - df['aroon_down']
    
    # Parabolic SAR
    df['psar'] = talib.SAR(df['high'], df['low'], 0.02, 0.2)
    df['psar_ratio'] = df['close'] / df['psar'] - 1
    
    # ============ MOMENTUM INDICATORS ============
    # RSI
    for period in [6, 14, 21]:
        df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
    
    df['rsi_slope'] = df['rsi_14'].pct_change(5) * 100 if 'rsi_14' in df else df['rsi'].pct_change(5) * 100
    
    # Stochastic Oscillator
    df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], 
                                              fastk_period=14, slowk_period=3, slowd_period=3)
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
    
    # MACD and derivatives
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['macd_ratio'] = df['macd'] / df['macd_signal'] if 'macd_signal' in df and df['macd_signal'].mean() != 0 else 0
    df['macd_hist_slope'] = df['macd_hist'].pct_change(3) * 100 if 'macd_hist' in df else 0
    
    # CCI (Commodity Channel Index)
    for period in [14, 20, 40]:
        df[f'cci_{period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
    
    # Rate of Change (ROC)
    for period in [5, 10, 20, 40]:
        df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
    
    # ============ VOLUME ANALYSIS ============
    if 'volume' in df.columns:
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # OBV (On-Balance Volume)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_sma_20'] = df['obv'].rolling(window=20).mean()
        df['obv_ratio'] = df['obv'] / df['obv_sma_20'] if 'obv_sma_20' in df else 1.0
        
        # Chaikin Money Flow
        df['cmf'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)
        
        # Up/Down Volume
        df['up_volume'] = df['volume'] * (df['close'] > df['close'].shift(1))
        df['down_volume'] = df['volume'] * (df['close'] < df['close'].shift(1))
        df['up_down_vol_ratio'] = df['up_volume'].rolling(10).sum() / df['down_volume'].rolling(10).sum().replace(0, 1)
    
    # ============ PATTERN RECOGNITION ============
    # Candlestick patterns (binary 0/1 outputs)
    pattern_functions = [
        ('doji', talib.CDLDOJI),
        ('engulfing', talib.CDLENGULFING),
        ('hammer', talib.CDLHAMMER),
        ('hanging_man', talib.CDLHANGINGMAN),
        ('shooting_star', talib.CDLSHOOTINGSTAR),
        ('evening_star', talib.CDLEVENINGSTAR),
        ('morning_star', talib.CDLMORNINGSTAR)
    ]
    
    for name, func in pattern_functions:
        df[f'pattern_{name}'] = func(df['open'], df['high'], df['low'], df['close'])
        # Normalize pattern outputs to 0 or 1
        df[f'pattern_{name}'] = (df[f'pattern_{name}'] != 0).astype(int)
    
    # Rolling window trend indicators
    for period in [5, 10, 20]:
        # Count of positive vs negative days
        df[f'pos_days_{period}'] = df['returns'].rolling(period).apply(lambda x: (x > 0).sum() / period)
        
        # Higher highs and lower lows in window
        df[f'higher_highs_{period}'] = df['higher_high'].rolling(period).sum() / period
        df[f'lower_lows_{period}'] = df['lower_low'].rolling(period).sum() / period
        
        # Trend consistency
        df[f'trend_consistency_{period}'] = (
            df['close'].rolling(period).corr(pd.Series(range(period)))
        )
    
    # ============ CYCLICAL COMPONENTS ============
    # Calculate Fourier transform based components for cyclical analysis
    if len(df) >= 100:  # Need sufficient data points
        try:
            # Detrend the price series
            price_trend = df['close'].rolling(20).mean()
            detrended = df['close'] - price_trend
            detrended = detrended.fillna(0)
            
            # Apply FFT to find dominant cycles
            from scipy import signal
            from scipy.fft import fft
            
            if len(detrended) >= 30:  # Minimum needed for meaningful FFT
                # Use the last 128 points for FFT or whatever is available
                n_points = min(128, len(detrended))
                fft_values = fft(detrended.iloc[-n_points:].values)
                power = np.abs(fft_values)**2
                freqs = np.fft.fftfreq(len(power), 1)  # Assuming daily data, frequency in cycles/day
                
                # Find dominant cycle length (in days)
                pos_mask = freqs > 0
                freqs = freqs[pos_mask]
                power = power[pos_mask]
                
                if len(power) > 0 and power.max() > 0:
                    # Normalize dominant cycle power
                    df['dominant_cycle_power'] = power.max() / detrended.std()
                    
                    # Dominant cycle length (in bars)
                    peak_idx = np.argmax(power)
                    if freqs[peak_idx] > 0:  # Avoid division by zero
                        df['dominant_cycle_length'] = 1 / freqs[peak_idx]
                    else:
                        df['dominant_cycle_length'] = 0
                        
                    # Calculate where we are in the cycle
                    if df['dominant_cycle_length'].iloc[-1] > 0:
                        cycle_len = int(df['dominant_cycle_length'].iloc[-1])
                        if cycle_len > 1 and cycle_len < len(detrended):
                            # Phase of the cycle (0 to 1, where 0 is start and 1 is end)
                            cycle_pattern = detrended.iloc[-cycle_len:]
                            # Normalize to -1 to 1
                            cycle_pattern = (cycle_pattern - cycle_pattern.mean()) / cycle_pattern.std()
                            df['cycle_phase'] = (cycle_pattern.iloc[-1] + 1) / 2  # Convert to 0-1 range
        except Exception as e:
            # Suppress errors in cyclical analysis
            pass
    
    # Clean up NaN values that result from the calculations
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df

def label_regimes(data):
    """
    Label market regimes based on technical patterns.
    This function creates the training labels.
    
    Args:
        data: DataFrame with technical indicators
        
    Returns:
        DataFrame with regime labels added
    """
    df = data.copy()
    
    # Initialize with UNKNOWN regime
    df['regime'] = MarketRegime.UNKNOWN.value
    
    # Bull Trend: Strong upward momentum, price above MAs, positive slopes
    bull_trend_mask = (
        (df['price_to_sma_50'] > 0.001) &  # Price above 50 SMA
        (df['price_to_sma_20'] > 0) &      # Price above 20 SMA
        (df['sma_20_50_cross'] > 0) &      # 20 SMA above 50 SMA
        (df['adx'] > 25) &                 # Strong trend
        (df['rsi'] > 50) &                 # Bullish momentum
        (df['pos_days_20'] > 0.6)          # Mostly positive days
    )
    df.loc[bull_trend_mask, 'regime'] = MarketRegime.BULL_TREND.value
    
    # Bear Trend: Strong downward momentum, price below MAs, negative slopes
    bear_trend_mask = (
        (df['price_to_sma_50'] < -0.001) & # Price below 50 SMA
        (df['price_to_sma_20'] < 0) &      # Price below 20 SMA
        (df['sma_20_50_cross'] < 0) &      # 20 SMA below 50 SMA
        (df['adx'] > 25) &                 # Strong trend
        (df['rsi'] < 50) &                 # Bearish momentum
        (df['pos_days_20'] < 0.4)          # Mostly negative days
    )
    df.loc[bear_trend_mask, 'regime'] = MarketRegime.BEAR_TREND.value
    
    # Consolidation: Low volatility, flat MAs, price in narrow range
    consolidation_mask = (
        (abs(df['price_to_sma_50']) < 0.0015) &  # Price near 50 SMA
        (abs(df['sma_20_slope']) < 0.1) &        # Flat 20 SMA
        (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.8) &  # Narrow Bollinger Bands
        (df['adx'] < 20)                         # Weak trend
    )
    df.loc[consolidation_mask, 'regime'] = MarketRegime.CONSOLIDATION.value
    
    # High Volatility: Expanded ranges, high ATR, wide Bollinger Bands
    high_volatility_mask = (
        (df['atr_percent'] > df['atr_percent'].rolling(50).mean() * 1.5) &  # High ATR
        (df['bb_width'] > df['bb_width'].rolling(50).mean() * 1.5) &        # Wide Bollinger Bands
        (df['adx'] > 30)                                                    # Strong directional movement
    )
    df.loc[high_volatility_mask, 'regime'] = MarketRegime.HIGH_VOLATILITY.value
    
    # Low Volatility: Contracted ranges, low ATR
    low_volatility_mask = (
        (df['atr_percent'] < df['atr_percent'].rolling(50).mean() * 0.7) &  # Low ATR
        (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.7) &        # Narrow Bollinger Bands
        ~consolidation_mask                                                 # Not in consolidation
    )
    df.loc[low_volatility_mask, 'regime'] = MarketRegime.LOW_VOLATILITY.value
    
    return df

def load_and_prepare_data():
    """
    Load historical forex data and prepare it for ML training.
    
    Returns:
        Processed DataFrame with features and labels
    """
    # In a real implementation, you'd load actual historical data here
    # For this example, we'll generate synthetic data for different regimes
    
    print("Generating synthetic training data...")
    
    # Date range for synthetic data (multiple years for robust training)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='1H')
    
    # Create the base dataframe
    df = pd.DataFrame(index=dates)
    df['datetime'] = df.index
    
    # Initialize with a starting price
    base_price = 1.2000
    current_price = base_price
    
    # Set up for alternating regimes to ensure balanced training data
    regimes = [
        {'name': 'BULL_TREND', 'duration': 30*24},  # 30 days
        {'name': 'CONSOLIDATION', 'duration': 20*24},  # 20 days
        {'name': 'BEAR_TREND', 'duration': 30*24},  # 30 days
        {'name': 'HIGH_VOLATILITY', 'duration': 15*24},  # 15 days
        {'name': 'LOW_VOLATILITY', 'duration': 15*24},  # 15 days
    ]
    
    # Generate synthetic price data with different regime characteristics
    prices = []
    volumes = []
    true_regimes = []
    
    regime_idx = 0
    days_in_regime = 0
    
    for i in range(len(df)):
        regime = regimes[regime_idx]
        days_in_regime += 1
        
        # Transition to next regime when current one is done
        if days_in_regime >= regime['duration']:
            regime_idx = (regime_idx + 1) % len(regimes)
            days_in_regime = 0
            regime = regimes[regime_idx]
            
        # Record the true regime for validation
        true_regimes.append(regime['name'])
        
        # Generate price based on current regime
        if regime['name'] == 'BULL_TREND':
            # Upward trend with small noise
            daily_change = np.random.normal(0.0004, 0.0002)
            volume = np.random.normal(150, 30)
            
        elif regime['name'] == 'BEAR_TREND':
            # Downward trend with small noise
            daily_change = np.random.normal(-0.0004, 0.0002)
            volume = np.random.normal(150, 30)
            
        elif regime['name'] == 'CONSOLIDATION':
            # Minimal movement around a value
            daily_change = np.random.normal(0, 0.0001)
            volume = np.random.normal(100, 20)
            
        elif regime['name'] == 'HIGH_VOLATILITY':
            # Larger price swings
            daily_change = np.random.normal(0, 0.0008)
            volume = np.random.normal(200, 50)
            
        elif regime['name'] == 'LOW_VOLATILITY':
            # Very small movements
            daily_change = np.random.normal(0, 0.00005)
            volume = np.random.normal(80, 15)
        
        # Update price and record
        current_price *= (1 + daily_change)
        prices.append(current_price)
        volumes.append(max(10, volume))  # Ensure positive volume
    
    # Build OHLCV data
    df['close'] = prices
    df['volume'] = volumes
    
    # Generate realistic OHLC data
    df['open'] = df['close'].shift(1)
    df['open'].iloc[0] = prices[0] * 0.9999
    
    # Generate high and low with appropriate noise
    daily_volatility = df['close'].pct_change().rolling(window=24).std().fillna(0.001)
    df['high'] = df.apply(
        lambda row: max(row['open'], row['close']) * (1 + abs(np.random.normal(0, daily_volatility[row.name] * 0.5))),
        axis=1
    )
    df['low'] = df.apply(
        lambda row: min(row['open'], row['close']) * (1 - abs(np.random.normal(0, daily_volatility[row.name] * 0.5))),
        axis=1
    )
    
    # Store true regimes for validation
    df['true_regime'] = true_regimes
    
    return df

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the ML model and display metrics.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print accuracy metrics
    print("\nModel Evaluation:")
    print("=================")
    print(classification_report(y_test, y_pred))
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[r.name for r in MarketRegime],
               yticklabels=[r.name for r in MarketRegime])
    plt.title('Confusion Matrix')
    plt.ylabel('True Regime')
    plt.xlabel('Predicted Regime')
    plt.tight_layout()
    
    # Save the confusion matrix plot
    plt.savefig(os.path.join(OUTPUT_DIR, 'regime_confusion_matrix.png'))
    
    # Feature importance analysis
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances')
        plt.bar(range(X_test.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
        
        print("\nTop 10 Most Important Features:")
        for i in range(10):
            if i < len(indices):
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    print("\nEvaluation plots saved to:", OUTPUT_DIR)

def main():
    """Main execution function for model training and evaluation."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Starting market regime classifier training...")
    
    # Load and prepare data
    data = load_and_prepare_data()
    print(f"Generated synthetic data with {len(data)} samples")
    
    # Create features
    print("Extracting technical features...")
    features_df = create_features(data)
    print(f"Generated {len(features_df)} samples with features")
    
    # Label the data
    print("Labeling market regimes...")
    labeled_df = label_regimes(features_df)
    
    # Prepare features and target for ML
    feature_columns = [col for col in labeled_df.columns if col not in 
                      ['datetime', 'open', 'high', 'low', 'close', 'volume', 
                       'regime', 'true_regime', 'higher_high', 'lower_low']]
    
    X = labeled_df[feature_columns].values
    y = labeled_df['regime'].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    # Train model with hyperparameter tuning
    print("Training Random Forest classifier with GridSearchCV...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate model
    evaluate_model(best_model, X_test, y_test, feature_columns)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    model_path = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
    scaler_path = os.path.join(OUTPUT_DIR, 'feature_scaler.joblib')
    
    # Save a dictionary with all needed components
    model_package = {
        'model': best_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'accuracy': grid_search.best_score_
    }
    
    joblib.dump(model_package, model_path)
    print(f"Model saved to {model_path}")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
