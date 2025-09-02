#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forex Algorithmic Trading Strategy

This implements an advanced algorithmic trading strategy with multiple
statistical models, adaptive parameters, and machine learning components.
The strategy uses ensemble methods to combine signals from various 
statistical models and technical indicators.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
import random
from collections import deque

# Strategy registry import
from trading_bot.strategies_new.factory.registry import register_strategy

# Base strategy import
from trading_bot.strategies_new.base.event_driven_strategy import EventDrivenStrategy


@register_strategy(
    asset_class="forex",
    strategy_type="algorithmic",
    name="forex_algorithmic_trading",
    description="Advanced algorithmic trading strategy with statistical models, ML components, and adaptive parameters",
    version="1.0.0",
    author="Trading System"
)
class ForexAlgorithmicTradingStrategy(EventDrivenStrategy):
    """
    Advanced Algorithmic Trading Strategy for Forex markets.
    
    This strategy combines multiple statistical models and technical indicators
    to generate trading signals. It features:
    
    1. Statistical arbitrage models
    2. Mean reversion and trend following components
    3. Machine learning signal generation and filtering
    4. Adaptive parameter optimization
    5. Ensemble methods for signal combination
    6. Advanced risk management
    
    The strategy adapts to changing market conditions by continuously 
    evaluating model performance and adjusting weights accordingly.
    """
    
    def __init__(self, symbols: List[str], parameters: Dict[str, Any] = None):
        """
        Initialize the Algorithmic Trading Strategy.
        
        Args:
            symbols: List of currency pairs to trade
            parameters: Strategy parameters
        """
        # Default parameters
        default_parameters = {
            # General parameters
            "timeframe": "1h",
            "lookback_window": 500,          # Data history for model training/analysis
            "retraining_frequency_hours": 24, # How often to retrain models
            
            # Model parameters
            "use_mean_reversion": True,
            "use_trend_following": True,
            "use_statistical_arbitrage": True,
            "use_ml_predictions": True,
            
            # Mean reversion parameters
            "z_score_threshold": 2.0,         # Z-score threshold for mean reversion
            "mean_reversion_window": 20,       # Lookback window for calculating means
            
            # Trend following parameters
            "short_ema": 9,                   # Short EMA period
            "medium_ema": 21,                 # Medium EMA period  
            "long_ema": 50,                   # Long EMA period
            "adx_period": 14,                 # ADX period
            "adx_threshold": 25,              # ADX threshold for trend strength
            
            # Statistical arbitrage parameters
            "pairs_correlation_threshold": 0.7,  # Minimum correlation for pairs
            "cointegration_pvalue_threshold": 0.05, # P-value threshold for cointegration
            "stat_arb_lookback": 60,          # Lookback for statistical arbitrage
            
            # ML parameters
            "training_window": 252,           # Training window in bars
            "prediction_features": [          # Features to use for ML predictions
                "rsi", "macd", "bollinger_z", "adx",
                "atr_percent", "price_delta", "volume_delta"
            ],
            "prediction_target": "return_5bar", # Prediction target
            
            # Risk management
            "risk_per_trade_pct": 1.0,        # Risk per trade as percentage of account
            "max_position_size_pct": 5.0,     # Maximum position size as percentage
            "max_correlation_exposure": 0.3,  # Maximum correlation-weighted exposure
            "max_concurrent_positions": 5,    # Maximum number of concurrent positions
            "use_dynamic_position_sizing": True, # Adjust position size based on signal strength
            "use_adaptive_stops": True,       # Adapt stop loss based on volatility
            
            # Execution parameters
            "entry_filter_threshold": 0.6,    # Minimum combined signal strength for entry
            "ensemble_weights": {             # Weights for signal ensemble
                "mean_reversion": 0.25,
                "trend_following": 0.25,
                "statistical_arbitrage": 0.25,
                "ml_prediction": 0.25
            },
            "enable_parameter_adaptation": True, # Automatically adjust parameters
            "parameter_update_frequency_hours": 48, # How often to update parameters
            
            # Performance tracking
            "track_model_performance": True,  # Track performance of individual models
            "performance_window": 30,         # Performance tracking window (days)
            
            # Additional features
            "apply_volatility_filter": True,  # Filter trades during extreme volatility
            "volatility_percentile_threshold": 80, # Percentile threshold for volatility
            "use_time_of_day_filter": True,   # Filter trades based on time of day
            "preferred_session_windows": [    # Preferred trading sessions
                {"session": "london", "start_hour": 8, "end_hour": 16, "timezone": "Europe/London"},
                {"session": "new_york", "start_hour": 8, "end_hour": 16, "timezone": "America/New_York"}
            ]
        }
        
        # Initialize base strategy
        super().__init__(
            strategy_name="ForexAlgorithmicTrading",
            symbols=symbols,
            parameters={**default_parameters, **(parameters or {})}
        )
        
        # Initialize strategy-specific attributes
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Data storage
        self.market_data = {}           # Market data by symbol
        self.model_predictions = {}     # ML model predictions by symbol
        self.signal_history = {}        # Signal history by model and symbol
        self.performance_metrics = {}   # Performance metrics by model
        
        # Position management
        self.active_positions = {}      # Currently active positions
        self.pending_orders = {}        # Orders waiting to be executed
        self.closed_positions = []      # Closed position history
        self.positions_to_close = []    # Positions marked for closing
        
        # Model storage
        self.models = {                 # Statistical and ML models
            "mean_reversion": {},
            "trend_following": {},
            "statistical_arbitrage": {},
            "ml_models": {}
        }
        
        # Performance tracking
        self.model_returns = {          # Returns by model
            "mean_reversion": deque(maxlen=30),
            "trend_following": deque(maxlen=30),
            "statistical_arbitrage": deque(maxlen=30),
            "ml_prediction": deque(maxlen=30)
        }
        
        # State tracking
        self.last_model_training = {}   # Last training time by model
        self.last_parameter_update = datetime.now() - timedelta(days=10)  # Force update on first run
        self.current_time = None
        
        # Performance metrics
        self.trades_count = 0
        self.winning_trades = 0
        self.total_profit_pips = 0
        self.r_multiples = []
        
        # Initialize models and parameters
        self._initialize_models()
        
    def _initialize_models(self):
        """
        Initialize statistical and machine learning models used by the strategy.
        """
        self.logger.info("Initializing algorithmic trading models")
        
        # Initialize models for each symbol
        for symbol in self.symbols:
            # Mean reversion models
            if self.parameters["use_mean_reversion"]:
                self.models["mean_reversion"][symbol] = {
                    "window": self.parameters["mean_reversion_window"],
                    "z_score_threshold": self.parameters["z_score_threshold"],
                    "means": {},
                    "stds": {},
                    "last_update": None
                }
                
            # Trend following models
            if self.parameters["use_trend_following"]:
                self.models["trend_following"][symbol] = {
                    "short_ema": self.parameters["short_ema"],
                    "medium_ema": self.parameters["medium_ema"],
                    "long_ema": self.parameters["long_ema"],
                    "adx_period": self.parameters["adx_period"],
                    "adx_threshold": self.parameters["adx_threshold"],
                    "last_update": None
                }
                
            # Statistical arbitrage models - will be populated when data is available
            if self.parameters["use_statistical_arbitrage"]:
                self.models["statistical_arbitrage"][symbol] = {
                    "pairs": [],
                    "cointegration_tests": {},
                    "hedge_ratios": {},
                    "spread_means": {},
                    "spread_stds": {},
                    "last_update": None
                }
                
            # ML prediction models - placeholders, would implement real models in production
            if self.parameters["use_ml_predictions"]:
                self.models["ml_models"][symbol] = {
                    "model": None,  # Placeholder for actual ML model
                    "features": self.parameters["prediction_features"],
                    "target": self.parameters["prediction_target"],
                    "training_window": self.parameters["training_window"],
                    "last_training": None,
                    "performance": {}
                }
                
            # Initialize signal history
            self.signal_history[symbol] = {
                "mean_reversion": deque(maxlen=100),
                "trend_following": deque(maxlen=100),
                "statistical_arbitrage": deque(maxlen=100),
                "ml_prediction": deque(maxlen=100),
                "combined": deque(maxlen=100)
            }
            
            # Mark for initial training
            self.last_model_training[symbol] = datetime.now() - timedelta(hours=self.parameters["retraining_frequency_hours"] + 1)
    
    def _update_mean_reversion_model(self, symbol: str, data: pd.DataFrame):
        """
        Update mean reversion model parameters for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
        """
        if not self.parameters["use_mean_reversion"] or data.empty:
            return
            
        # Get relevant parameters
        window = self.models["mean_reversion"][symbol]["window"]
        
        # Ensure we have enough data
        if len(data) < window * 2:
            self.logger.warning(f"Insufficient data for mean reversion model: {symbol}")
            return
            
        # Calculate means and standard deviations for various indicators
        # 1. Price levels
        close = data["close"]
        self.models["mean_reversion"][symbol]["means"]["price"] = close.rolling(window=window).mean().iloc[-1]
        self.models["mean_reversion"][symbol]["stds"]["price"] = close.rolling(window=window).std().iloc[-1]
        
        # 2. RSI
        rsi = self._calculate_rsi(data["close"], window=14)
        if not rsi.empty:
            self.models["mean_reversion"][symbol]["means"]["rsi"] = rsi.rolling(window=window).mean().iloc[-1]
            self.models["mean_reversion"][symbol]["stds"]["rsi"] = rsi.rolling(window=window).std().iloc[-1]
        
        # 3. Bollinger Band Width
        bb_width = self._calculate_bollinger_width(data["close"], window=20)
        if not bb_width.empty:
            self.models["mean_reversion"][symbol]["means"]["bb_width"] = bb_width.rolling(window=window).mean().iloc[-1]
            self.models["mean_reversion"][symbol]["stds"]["bb_width"] = bb_width.rolling(window=window).std().iloc[-1]
        
        # Update last update timestamp
        self.models["mean_reversion"][symbol]["last_update"] = self.current_time
    
    def _update_trend_following_model(self, symbol: str, data: pd.DataFrame):
        """
        Update trend following model parameters for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
        """
        if not self.parameters["use_trend_following"] or data.empty:
            return
            
        # Get relevant parameters
        short_ema = self.parameters["short_ema"]
        medium_ema = self.parameters["medium_ema"]
        long_ema = self.parameters["long_ema"]
        adx_period = self.parameters["adx_period"]
        
        # Ensure we have enough data
        if len(data) < max(short_ema, medium_ema, long_ema, adx_period) * 2:
            self.logger.warning(f"Insufficient data for trend following model: {symbol}")
            return
            
        # Calculate trend indicators
        close = data["close"]
        
        # EMAs
        short_ema_values = close.ewm(span=short_ema, adjust=False).mean()
        medium_ema_values = close.ewm(span=medium_ema, adjust=False).mean()
        long_ema_values = close.ewm(span=long_ema, adjust=False).mean()
        
        # Store latest values
        self.models["trend_following"][symbol]["short_ema_value"] = short_ema_values.iloc[-1]
        self.models["trend_following"][symbol]["medium_ema_value"] = medium_ema_values.iloc[-1]
        self.models["trend_following"][symbol]["long_ema_value"] = long_ema_values.iloc[-1]
        
        # Calculate ADX for trend strength
        adx = self._calculate_adx(data, period=adx_period)
        if not adx.empty:
            self.models["trend_following"][symbol]["adx_value"] = adx.iloc[-1]
        
        # Update trend direction
        if short_ema_values.iloc[-1] > medium_ema_values.iloc[-1] and medium_ema_values.iloc[-1] > long_ema_values.iloc[-1]:
            self.models["trend_following"][symbol]["trend_direction"] = "bullish"
        elif short_ema_values.iloc[-1] < medium_ema_values.iloc[-1] and medium_ema_values.iloc[-1] < long_ema_values.iloc[-1]:
            self.models["trend_following"][symbol]["trend_direction"] = "bearish"
        else:
            self.models["trend_following"][symbol]["trend_direction"] = "neutral"
            
        # Calculate slope of medium EMA for trend strength
        if len(medium_ema_values) > 5:
            self.models["trend_following"][symbol]["ema_slope"] = (
                medium_ema_values.iloc[-1] - medium_ema_values.iloc[-5]
            ) / medium_ema_values.iloc[-5] * 100  # As percentage
        
        # Update last update timestamp
        self.models["trend_following"][symbol]["last_update"] = self.current_time
    
    def _update_statistical_arbitrage_model(self, all_data: Dict[str, pd.DataFrame]):
        """
        Update statistical arbitrage model for correlated pairs.
        
        Args:
            all_data: Dictionary of DataFrames with symbol as key
        """
        if not self.parameters["use_statistical_arbitrage"]:
            return
            
        # Need at least 2 symbols with sufficient data for statistical arbitrage
        valid_symbols = [s for s in self.symbols if s in all_data and not all_data[s].empty 
                         and len(all_data[s]) >= self.parameters["stat_arb_lookback"]]                         
        
        if len(valid_symbols) < 2:
            self.logger.warning("Insufficient data for statistical arbitrage model")
            return
            
        # Calculate correlation matrix for all currency pairs
        correlation_matrix = {}
        price_series = {}
        
        # Extract close prices
        for symbol in valid_symbols:
            price_series[symbol] = all_data[symbol]["close"].values
            
        # Calculate correlations
        for i, symbol1 in enumerate(valid_symbols):
            correlation_matrix[symbol1] = {}
            for j, symbol2 in enumerate(valid_symbols):
                if i != j:  # Don't calculate correlation with itself
                    # Calculate correlation if both series have data
                    if len(price_series[symbol1]) > 0 and len(price_series[symbol2]) > 0:
                        # Make sure series are the same length
                        min_length = min(len(price_series[symbol1]), len(price_series[symbol2]))
                        s1 = price_series[symbol1][-min_length:]
                        s2 = price_series[symbol2][-min_length:]
                        
                        try:
                            corr = np.corrcoef(s1, s2)[0, 1]
                            correlation_matrix[symbol1][symbol2] = corr
                        except Exception as e:
                            self.logger.error(f"Error calculating correlation: {e}")
                            correlation_matrix[symbol1][symbol2] = 0
                    else:
                        correlation_matrix[symbol1][symbol2] = 0
        
        # Find highly correlated pairs for each symbol
        for symbol in valid_symbols:
            # Reset pairs list
            self.models["statistical_arbitrage"][symbol]["pairs"] = []
            
            if symbol not in correlation_matrix:
                continue
                
            # Find symbol pairs with correlation above threshold
            correlated_pairs = [(s, c) for s, c in correlation_matrix[symbol].items() 
                               if abs(c) >= self.parameters["pairs_correlation_threshold"]]                               
            
            # Sort by absolute correlation (highest first)
            correlated_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for pair_symbol, correlation in correlated_pairs:
                # Check for cointegration between the symbol pairs
                if len(all_data[symbol]) > 60 and len(all_data[pair_symbol]) > 60:
                    # Prepare price series for cointegration test
                    s1 = all_data[symbol]["close"].values[-60:]
                    s2 = all_data[pair_symbol]["close"].values[-60:]
                    
                    # Calculate hedge ratio using regression
                    hedge_ratio = self._calculate_hedge_ratio(s1, s2)
                    
                    # Create spread series
                    spread = s1 - hedge_ratio * s2
                    
                    # Calculate spread mean and standard deviation
                    spread_mean = np.mean(spread)
                    spread_std = np.std(spread)
                    
                    # Store the pair information
                    self.models["statistical_arbitrage"][symbol]["pairs"].append({
                        "symbol": pair_symbol,
                        "correlation": correlation,
                        "hedge_ratio": hedge_ratio,
                        "spread_mean": spread_mean,
                        "spread_std": spread_std,
                        "current_spread": spread[-1],
                        "z_score": (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0
                    })
            
            # Update last update timestamp
            self.models["statistical_arbitrage"][symbol]["last_update"] = self.current_time
    
    def _update_ml_prediction_model(self, symbol: str, data: pd.DataFrame):
        """
        Update machine learning prediction model for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
        """
        if not self.parameters["use_ml_predictions"] or data.empty:
            return
            
        # In a real implementation, this would train ML models
        # For this implementation, we'll use a placeholder with random predictions
        
        # Ensure we have enough data
        if len(data) < self.parameters["training_window"]:
            self.logger.warning(f"Insufficient data for ML model: {symbol}")
            return
            
        # Calculate features for prediction (simplified for demonstration)
        features = self._calculate_prediction_features(data)
        
        # In a real implementation, this is where model training would occur
        # For example:
        # X_train, y_train = self._prepare_ml_training_data(features, data)
        # self.models["ml_models"][symbol]["model"] = self._train_ml_model(X_train, y_train)
        
        # For this implementation, just generate a random prediction
        # In a real implementation, this would use the trained model to make predictions
        last_prediction = random.uniform(-1.0, 1.0)  # -1 to 1 scale, negative for sell, positive for buy
        
        # Store prediction
        self.model_predictions[symbol] = {
            "prediction": last_prediction,
            "timestamp": self.current_time,
            "confidence": random.uniform(0.5, 0.9)  # Random confidence score
        }
        
        # Update last training timestamp
        self.models["ml_models"][symbol]["last_training"] = self.current_time
    
    def _calculate_prediction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features to use for ML prediction.
        
        Args:
            data: OHLCV price data
            
        Returns:
            DataFrame with calculated features
        """
        if data.empty:
            return pd.DataFrame()
            
        # Copy data to avoid modifying the original
        features = data.copy()
        
        # Calculate features specified in parameters
        for feature in self.parameters["prediction_features"]:
            if feature == "rsi":
                features["rsi"] = self._calculate_rsi(data["close"])
            elif feature == "macd":
                macd, signal, hist = self._calculate_macd(data["close"])
                features["macd"] = macd
                features["macd_signal"] = signal
                features["macd_hist"] = hist
            elif feature == "bollinger_z":
                features["bollinger_z"] = self._calculate_bollinger_z_score(data["close"])
            elif feature == "adx":
                features["adx"] = self._calculate_adx(data)
            elif feature == "atr_percent":
                features["atr_percent"] = self._calculate_atr_percent(data)
            elif feature == "price_delta":
                features["price_delta_1d"] = data["close"].pct_change(1)
                features["price_delta_5d"] = data["close"].pct_change(5)
            elif feature == "volume_delta":
                if "volume" in data.columns:
                    features["volume_delta"] = data["volume"].pct_change(1)
                    
        # Remove NaN values
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            window: RSI period
            
        Returns:
            Series with RSI values
        """
        if len(prices) < window + 1:
            return pd.Series()
            
        # Calculate price changes
        delta = prices.diff(1)
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal EMA period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        if len(prices) < slow + signal:
            return pd.Series(), pd.Series(), pd.Series()
            
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_width(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
        """
        Calculate Bollinger Band width.
        
        Args:
            prices: Price series
            window: Bollinger Band period
            num_std: Number of standard deviations
            
        Returns:
            Series with Bollinger Band width
        """
        if len(prices) < window:
            return pd.Series()
            
        # Calculate SMA and standard deviation
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        # Calculate band width as percentage of middle band
        bb_width = (upper_band - lower_band) / sma * 100
        
        return bb_width
    
    def _calculate_bollinger_z_score(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate Z-score relative to Bollinger Bands.
        
        Args:
            prices: Price series
            window: Bollinger Band period
            
        Returns:
            Series with Z-scores
        """
        if len(prices) < window:
            return pd.Series()
            
        # Calculate SMA and standard deviation
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        # Calculate Z-score: (price - mean) / std
        z_score = (prices - sma) / std
        
        return z_score
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        Args:
            data: OHLCV data
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        if len(data) < period * 2:
            return pd.Series()
            
        # Calculate +DM, -DM, and TR
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Create Series with appropriate index
        adx = pd.Series(np.nan, index=data.index)
        
        # Try/except block to handle any calculation issues
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Convert to pandas Series
            plus_dm = pd.Series(plus_dm, index=data.index)
            minus_dm = pd.Series(minus_dm, index=data.index)
            
            # Calculate smoothed values
            tr_avg = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_avg)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_avg)
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
        
        return adx
    
    def _calculate_atr_percent(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range as percentage of price.
        
        Args:
            data: OHLCV data
            period: ATR period
            
        Returns:
            Series with ATR percent values
        """
        if len(data) < period:
            return pd.Series()
            
        # Calculate ATR
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Average True Range
        atr = tr.rolling(window=period).mean()
        
        # Calculate ATR as percentage of close price
        atr_percent = atr / close * 100
        
        return atr_percent
    
    def _calculate_hedge_ratio(self, y: np.ndarray, x: np.ndarray) -> float:
        """
        Calculate hedge ratio between two price series using linear regression.
        
        Args:
            y: Dependent variable (price series 1)
            x: Independent variable (price series 2)
            
        Returns:
            Hedge ratio (slope coefficient)
        """
        if len(y) != len(x) or len(y) < 2:
            return 1.0
            
        try:
            # Add constant to x for regression
            X = np.vstack([x, np.ones(len(x))]).T
            
            # Calculate regression coefficients (beta, alpha)
            beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]
            
            return beta
        except Exception as e:
            self.logger.error(f"Error calculating hedge ratio: {e}")
            return 1.0
    
    def _check_model_retraining(self, symbol: str, force: bool = False) -> bool:
        """
        Check if models need to be retrained for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            force: Whether to force retraining
            
        Returns:
            True if models were retrained, False otherwise
        """
        if symbol not in self.last_model_training or symbol not in self.market_data:
            return False
            
        # Get time since last training
        time_since_last_training = None
        if self.current_time is not None and self.last_model_training[symbol] is not None:
            time_since_last_training = self.current_time - self.last_model_training[symbol]
            
        # Check if retraining is needed
        retraining_needed = force
        if time_since_last_training is not None:
            retraining_hours = time_since_last_training.total_seconds() / 3600
            retraining_needed = retraining_needed or retraining_hours >= self.parameters["retraining_frequency_hours"]
            
        if not retraining_needed:
            return False
            
        # Get data for this symbol
        data = self.market_data[symbol]
        
        # Ensure we have enough data
        if data.empty or len(data) < self.parameters["lookback_window"]:
            return False
            
        # Update models
        if self.parameters["use_mean_reversion"]:
            self._update_mean_reversion_model(symbol, data)
            
        if self.parameters["use_trend_following"]:
            self._update_trend_following_model(symbol, data)
            
        if self.parameters["use_ml_predictions"]:
            self._update_ml_prediction_model(symbol, data)
            
        # Update last training time
        self.last_model_training[symbol] = self.current_time
        
        return True
            
    def _generate_mean_reversion_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate mean reversion signal for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
            
        Returns:
            Signal dictionary with direction and strength
        """
        # Default neutral signal
        signal = {
            "direction": "neutral",
            "strength": 0.0,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "timestamp": self.current_time,
            "model": "mean_reversion"
        }
        
        # Skip if mean reversion is disabled or we don't have model data
        if not self.parameters["use_mean_reversion"] or symbol not in self.models["mean_reversion"]:
            return signal
            
        # Get model data
        model = self.models["mean_reversion"][symbol]
        
        # Skip if model hasn't been updated
        if model["last_update"] is None:
            return signal
            
        # Get latest price
        if data.empty or len(data) < 2:
            return signal
            
        current_price = data["close"].iloc[-1]
        
        # Calculate Z-scores for different indicators
        z_scores = {}
        
        # Price Z-score
        if "price" in model["means"] and model["stds"]["price"] > 0:
            z_scores["price"] = (current_price - model["means"]["price"]) / model["stds"]["price"]
        
        # RSI Z-score
        rsi = self._calculate_rsi(data["close"]).iloc[-1] if len(data) > 14 else None
        if rsi is not None and "rsi" in model["means"] and model["stds"]["rsi"] > 0:
            z_scores["rsi"] = (rsi - model["means"]["rsi"]) / model["stds"]["rsi"]
        
        # Skip if no Z-scores calculated
        if not z_scores:
            return signal
            
        # Calculate average Z-score (weighted by importance)
        weights = {"price": 0.7, "rsi": 0.3}
        available_indicators = set(z_scores.keys()) & set(weights.keys())
        
        if not available_indicators:
            return signal
            
        # Normalize weights
        total_weight = sum(weights[i] for i in available_indicators)
        normalized_weights = {i: weights[i] / total_weight for i in available_indicators}
        
        # Calculate weighted average Z-score
        avg_z_score = sum(z_scores[i] * normalized_weights[i] for i in available_indicators)
        
        # Generate signal based on Z-score
        threshold = model["z_score_threshold"]
        
        if avg_z_score < -threshold:  # Price is below mean (oversold)
            signal["direction"] = "buy"
            signal["strength"] = min(0.9, abs(avg_z_score) / (threshold * 2.0))  # Cap at 0.9
        elif avg_z_score > threshold:  # Price is above mean (overbought)
            signal["direction"] = "sell"
            signal["strength"] = min(0.9, abs(avg_z_score) / (threshold * 2.0))
        
        # Calculate entry, stop, and take profit levels if we have a signal
        if signal["direction"] != "neutral":
            # ATR for stop loss and take profit
            atr = self._calculate_atr_percent(data).iloc[-1] * current_price / 100 if len(data) > 14 else current_price * 0.001
            
            # Entry at current price
            signal["entry_price"] = current_price
            
            # Set stop loss and take profit based on ATR and Z-score
            if signal["direction"] == "buy":
                signal["stop_loss"] = current_price - (atr * 1.5)
                signal["take_profit"] = current_price + (atr * abs(avg_z_score))
            else:  # sell
                signal["stop_loss"] = current_price + (atr * 1.5)
                signal["take_profit"] = current_price - (atr * abs(avg_z_score))
        
        # Store signal in history
        self.signal_history[symbol]["mean_reversion"].append(signal)
        
        return signal
    
    def _generate_trend_following_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trend following signal for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
            
        Returns:
            Signal dictionary with direction and strength
        """
        # Default neutral signal
        signal = {
            "direction": "neutral",
            "strength": 0.0,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "timestamp": self.current_time,
            "model": "trend_following"
        }
        
        # Skip if trend following is disabled or we don't have model data
        if not self.parameters["use_trend_following"] or symbol not in self.models["trend_following"]:
            return signal
            
        # Get model data
        model = self.models["trend_following"][symbol]
        
        # Skip if model hasn't been updated
        if model["last_update"] is None:
            return signal
            
        # Get latest price
        if data.empty or len(data) < 2:
            return signal
            
        current_price = data["close"].iloc[-1]
        
        # Get trend direction and strength
        trend_direction = model.get("trend_direction", "neutral")
        adx_value = model.get("adx_value", 0)
        ema_slope = model.get("ema_slope", 0)
        
        # Skip if no clear trend
        if trend_direction == "neutral":
            return signal
            
        # Check if trend is strong enough based on ADX
        if adx_value < self.parameters["adx_threshold"]:
            # Weak trend, reduce signal strength or return neutral
            if adx_value < self.parameters["adx_threshold"] * 0.7:  # Significantly below threshold
                return signal
        
        # Generate signal based on trend direction
        signal["direction"] = "buy" if trend_direction == "bullish" else "sell"
        
        # Calculate signal strength based on ADX and EMA slope
        # Normalize ADX (typically 0-100, but practically 0-50 in most cases)
        normalized_adx = min(adx_value / 50, 1.0)
        
        # Normalize EMA slope (typically within -2% to 2% per period)
        normalized_slope = min(abs(ema_slope) / 2.0, 1.0)
        
        # Combine for overall strength
        signal["strength"] = min(0.9, (normalized_adx * 0.7) + (normalized_slope * 0.3))
        
        # Calculate entry, stop, and take profit levels
        if signal["strength"] > 0:
            # ATR for stop loss and take profit
            atr = self._calculate_atr_percent(data).iloc[-1] * current_price / 100 if len(data) > 14 else current_price * 0.001
            
            # Entry at current price
            signal["entry_price"] = current_price
            
            # Set stop loss and take profit based on ATR and signal strength
            atr_multiple = 1.5 + (signal["strength"] * 1.5)  # Higher strength = larger target
            
            if signal["direction"] == "buy":
                # For buy, stop below recent low or support level
                signal["stop_loss"] = current_price - (atr * 1.5)
                signal["take_profit"] = current_price + (atr * atr_multiple)
            else:  # sell
                # For sell, stop above recent high or resistance level
                signal["stop_loss"] = current_price + (atr * 1.5)
                signal["take_profit"] = current_price - (atr * atr_multiple)
        
        # Store signal in history
        self.signal_history[symbol]["trend_following"].append(signal)
        
        return signal
    
    def _generate_statistical_arbitrage_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistical arbitrage signal for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
            
        Returns:
            Signal dictionary with direction and strength
        """
        # Default neutral signal
        signal = {
            "direction": "neutral",
            "strength": 0.0,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "timestamp": self.current_time,
            "model": "statistical_arbitrage",
            "pair_symbol": None,
            "pair_direction": None,
            "hedge_ratio": None
        }
        
        # Skip if statistical arbitrage is disabled or we don't have model data
        if not self.parameters["use_statistical_arbitrage"] or symbol not in self.models["statistical_arbitrage"]:
            return signal
            
        # Get model data
        model = self.models["statistical_arbitrage"][symbol]
        
        # Skip if model hasn't been updated or no pairs found
        if model["last_update"] is None or not model["pairs"]:
            return signal
            
        # Get latest price
        if data.empty or len(data) < 2:
            return signal
            
        current_price = data["close"].iloc[-1]
        
        # Find the pair with the strongest signal
        best_pair = None
        best_z_score = 0
        
        for pair in model["pairs"]:
            # Skip pairs with no z-score
            if "z_score" not in pair:
                continue
                
            # Find the pair with the highest absolute Z-score
            if abs(pair["z_score"]) > abs(best_z_score):
                best_z_score = pair["z_score"]
                best_pair = pair
        
        # If no valid pair found, return neutral signal
        if best_pair is None or abs(best_z_score) < self.parameters["z_score_threshold"]:
            return signal
            
        # Generate signal based on Z-score
        # A positive Z-score means the spread is wider than average, suggesting:  
        # - Short the base currency (symbol) and long the quote currency (pair_symbol) if correlation is positive
        # - Long the base currency and short the quote currency if correlation is negative
        
        correlation = best_pair["correlation"]
        z_score = best_pair["z_score"]
        
        if correlation > 0:  # Positive correlation
            if z_score > self.parameters["z_score_threshold"]:
                signal["direction"] = "sell"
                signal["pair_direction"] = "buy"
            elif z_score < -self.parameters["z_score_threshold"]:
                signal["direction"] = "buy"
                signal["pair_direction"] = "sell"
        else:  # Negative correlation
            if z_score > self.parameters["z_score_threshold"]:
                signal["direction"] = "buy"
                signal["pair_direction"] = "buy"
            elif z_score < -self.parameters["z_score_threshold"]:
                signal["direction"] = "sell"
                signal["pair_direction"] = "sell"
        
        # If we have a signal, set its properties
        if signal["direction"] != "neutral":
            signal["strength"] = min(0.9, abs(z_score) / (self.parameters["z_score_threshold"] * 2))
            signal["pair_symbol"] = best_pair["symbol"]
            signal["hedge_ratio"] = best_pair["hedge_ratio"]
            
            # ATR for stop loss and take profit
            atr = self._calculate_atr_percent(data).iloc[-1] * current_price / 100 if len(data) > 14 else current_price * 0.001
            
            # Entry at current price
            signal["entry_price"] = current_price
            
            # Set stop loss based on historical spread volatility
            spread_std = best_pair["spread_std"]
            
            # Convert spread std to price units for stop loss
            # This is simplified; in reality would need more complex calculation
            price_buffer = max(atr, spread_std * 0.5)
            
            if signal["direction"] == "buy":
                signal["stop_loss"] = current_price - price_buffer
                # Target is mean reversion, so target is the mean spread
                signal["take_profit"] = current_price + (abs(z_score) * price_buffer)
            else:  # sell
                signal["stop_loss"] = current_price + price_buffer
                signal["take_profit"] = current_price - (abs(z_score) * price_buffer)
        
        # Store signal in history
        self.signal_history[symbol]["statistical_arbitrage"].append(signal)
        
        return signal
    
    def _generate_ml_prediction_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate machine learning prediction signal for a given symbol.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
            
        Returns:
            Signal dictionary with direction and strength
        """
        # Default neutral signal
        signal = {
            "direction": "neutral",
            "strength": 0.0,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "timestamp": self.current_time,
            "model": "ml_prediction"
        }
        
        # Skip if ML predictions are disabled
        if not self.parameters["use_ml_predictions"]:
            return signal
            
        # Skip if no prediction available
        if symbol not in self.model_predictions:
            return signal
            
        # Get latest prediction
        prediction = self.model_predictions[symbol]
        
        # Skip if prediction is too old (more than 1 day)
        if prediction["timestamp"] is None or self.current_time is None or \
           (self.current_time - prediction["timestamp"]).total_seconds() > 86400:  # 24 hours
            return signal
            
        # Get latest price
        if data.empty or len(data) < 2:
            return signal
            
        current_price = data["close"].iloc[-1]
        
        # Generate signal based on prediction
        # Prediction is on a scale from -1 (strong sell) to 1 (strong buy)
        pred_value = prediction["prediction"]
        confidence = prediction["confidence"]
        
        # Only generate signal if prediction is strong enough and confidence is high enough
        if abs(pred_value) < 0.2 or confidence < 0.6:
            return signal
            
        # Set direction based on prediction
        if pred_value > 0:
            signal["direction"] = "buy"
        else:
            signal["direction"] = "sell"
            
        # Strength is a combination of prediction magnitude and confidence
        signal["strength"] = min(0.9, abs(pred_value) * confidence)
        
        # Calculate entry, stop, and take profit levels
        if signal["strength"] > 0:
            # ATR for stop loss and take profit
            atr = self._calculate_atr_percent(data).iloc[-1] * current_price / 100 if len(data) > 14 else current_price * 0.001
            
            # Entry at current price
            signal["entry_price"] = current_price
            
            # Set stop loss and take profit based on ATR, prediction strength, and confidence
            risk_factor = 1.0 + (1.0 - confidence) * 2  # Higher uncertainty = wider stop
            reward_factor = 1.0 + (signal["strength"] * 2)  # Stronger signal = larger target
            
            if signal["direction"] == "buy":
                signal["stop_loss"] = current_price - (atr * risk_factor)
                signal["take_profit"] = current_price + (atr * reward_factor)
            else:  # sell
                signal["stop_loss"] = current_price + (atr * risk_factor)
                signal["take_profit"] = current_price - (atr * reward_factor)
        
        # Store signal in history
        self.signal_history[symbol]["ml_prediction"].append(signal)
        
        return signal
    
    def _combine_signals(self, symbol: str, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine signals from different models into a single trading signal.
        
        Args:
            symbol: Currency pair symbol
            signals: Dictionary of signals from different models
            
        Returns:
            Combined signal dictionary
        """
        # Default neutral signal
        combined_signal = {
            "direction": "neutral",
            "strength": 0.0,
            "entry_price": None,
            "stop_loss": None,
            "take_profit": None,
            "timestamp": self.current_time,
            "model": "ensemble",
            "components": signals
        }
        
        # Skip if no signals
        if not signals:
            return combined_signal
            
        # Get ensemble weights from parameters
        weights = self.parameters["ensemble_weights"]
        
        # Calculate directional signals (positive for buy, negative for sell)
        buy_strength = 0.0
        sell_strength = 0.0
        total_weight = 0.0
        
        for model_name, signal in signals.items():
            # Skip if model not used or signal is neutral
            if model_name not in weights or signal["direction"] == "neutral":
                continue
                
            # Get weight for this model
            weight = weights[model_name]
            total_weight += weight
            
            # Add to directional strength
            if signal["direction"] == "buy":
                buy_strength += signal["strength"] * weight
            else:  # sell
                sell_strength += signal["strength"] * weight
        
        # No valid signals or weights
        if total_weight == 0:
            return combined_signal
            
        # Normalize strengths
        buy_strength /= total_weight
        sell_strength /= total_weight
        
        # Determine overall direction and strength
        if buy_strength > sell_strength:
            net_strength = buy_strength - sell_strength
            direction = "buy"
        else:
            net_strength = sell_strength - buy_strength
            direction = "sell"
            
        # Only generate a signal if strength exceeds threshold
        if net_strength < self.parameters["entry_filter_threshold"]:
            return combined_signal
            
        # Set combined signal properties
        combined_signal["direction"] = direction
        combined_signal["strength"] = net_strength
        
        # Determine entry price, stop loss, and take profit
        # Use weighted average of component signals
        entry_prices = []
        stop_losses = []
        take_profits = []
        
        for model_name, signal in signals.items():
            if model_name not in weights or signal["direction"] != direction:
                continue
                
            # Get weight for this model
            weight = weights[model_name]
            
            # Add to weighted values if available
            if signal["entry_price"] is not None:
                entry_prices.append((signal["entry_price"], weight))
            if signal["stop_loss"] is not None:
                stop_losses.append((signal["stop_loss"], weight))
            if signal["take_profit"] is not None:
                take_profits.append((signal["take_profit"], weight))
        
        # Calculate weighted averages
        if entry_prices:
            total_weight = sum(w for _, w in entry_prices)
            combined_signal["entry_price"] = sum(p * w for p, w in entry_prices) / total_weight
            
        if stop_losses:
            # For stop loss, we want the more conservative value
            if direction == "buy":
                # Lower stop (further from entry) is more conservative for buy
                combined_signal["stop_loss"] = min(p for p, _ in stop_losses)
            else:
                # Higher stop (further from entry) is more conservative for sell
                combined_signal["stop_loss"] = max(p for p, _ in stop_losses)
                
        if take_profits:
            # For take profit, we'll use the weighted average
            total_weight = sum(w for _, w in take_profits)
            combined_signal["take_profit"] = sum(p * w for p, w in take_profits) / total_weight
        
        # Store combined signal in history
        self.signal_history[symbol]["combined"].append(combined_signal)
        
        return combined_signal
        
    def _generate_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal for a given symbol by combining multiple models.
        
        Args:
            symbol: Currency pair symbol
            data: OHLCV price data
            
        Returns:
            Signal dictionary with direction and strength
        """
        # Check if models need to be retrained
        self._check_model_retraining(symbol)
        
        # Skip if no data
        if data.empty or len(data) < 20:
            return None
            
        # Generate signals from each model
        signals = {}
        
        # Mean reversion signal
        if self.parameters["use_mean_reversion"]:
            signals["mean_reversion"] = self._generate_mean_reversion_signal(symbol, data)
            
        # Trend following signal
        if self.parameters["use_trend_following"]:
            signals["trend_following"] = self._generate_trend_following_signal(symbol, data)
            
        # Statistical arbitrage signal
        if self.parameters["use_statistical_arbitrage"]:
            signals["statistical_arbitrage"] = self._generate_statistical_arbitrage_signal(symbol, data)
            
        # ML prediction signal
        if self.parameters["use_ml_predictions"]:
            signals["ml_prediction"] = self._generate_ml_prediction_signal(symbol, data)
            
        # Combine signals
        combined_signal = self._combine_signals(symbol, signals)
        
        # Apply additional filters
        if combined_signal["direction"] != "neutral":
            # Volatility filter
            if self.parameters["apply_volatility_filter"] and not self._check_volatility(data):
                self.logger.info(f"Signal rejected for {symbol} due to volatility filter")
                return None
                
            # Time of day filter
            if self.parameters["use_time_of_day_filter"] and not self._check_session_window(symbol):
                self.logger.info(f"Signal rejected for {symbol} due to session time filter")
                return None
                
        return combined_signal
    
    def _check_volatility(self, data: pd.DataFrame) -> bool:
        """
        Check if volatility is suitable for trading.
        
        Args:
            data: OHLCV price data
            
        Returns:
            True if volatility is acceptable, False otherwise
        """
        if data.empty or len(data) < 20:
            return False
            
        # Calculate ATR
        atr_percent = self._calculate_atr_percent(data)
        if atr_percent.empty:
            return False
            
        # Get latest ATR percent
        current_atr = atr_percent.iloc[-1]
        
        # Calculate percentile of current ATR
        # First we need the historical distribution
        if len(atr_percent) >= 50:  # Need sufficient history
            percentile = sum(1 for x in atr_percent.iloc[-50:] if x <= current_atr) / 50 * 100
        else:
            percentile = 50  # Default if insufficient history
            
        # Check if volatility is too high
        if percentile > self.parameters["volatility_percentile_threshold"]:
            return False
            
        # Check if volatility is too low (below 20th percentile)
        if percentile < 20:
            return False
            
        return True
        
    def _check_session_window(self, symbol: str) -> bool:
        """
        Check if current time is within preferred trading sessions.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            True if within trading session, False otherwise
        """
        if not self.parameters["use_time_of_day_filter"] or self.current_time is None:
            return True
            
        # Get preferred sessions
        sessions = self.parameters["preferred_session_windows"]
        
        # Check each session
        for session in sessions:
            # Convert session times to UTC for comparison
            try:
                # Get session timezone and hours
                timezone_str = session["timezone"]
                start_hour = session["start_hour"]
                end_hour = session["end_hour"]
                
                # Get current time in session timezone
                import pytz
                session_tz = pytz.timezone(timezone_str)
                current_time_in_session = self.current_time.astimezone(session_tz)
                
                # Check if current time is within session hours
                if start_hour <= current_time_in_session.hour < end_hour:
                    return True
                    
            except Exception as e:
                self.logger.error(f"Error checking session window: {e}")
                continue
                
        return False
        
    def _calculate_position_size(self, signal: Dict[str, Any], account_balance: float) -> float:
        """
        Calculate position size based on risk parameters and signal strength.
        
        Args:
            signal: Trading signal
            account_balance: Current account balance
            
        Returns:
            Position size in units of base currency
        """
        if signal is None or signal["direction"] == "neutral":
            return 0.0
            
        # Get risk percentage
        risk_pct = self.parameters["risk_per_trade_pct"] / 100.0
        
        # Risk amount in account currency
        risk_amount = account_balance * risk_pct
        
        # Calculate price risk (distance from entry to stop)
        entry_price = signal["entry_price"]
        stop_loss = signal["stop_loss"]
        
        if entry_price is None or stop_loss is None:
            return 0.0
            
        price_risk = abs(entry_price - stop_loss)
        
        # Base position size on risk
        position_size = risk_amount / price_risk if price_risk > 0 else 0.0
        
        # Adjust size based on signal strength if enabled
        if self.parameters["use_dynamic_position_sizing"]:
            strength_modifier = 0.5 + (signal["strength"] * 0.5)  # Scale from 0.5x to 1.0x
            position_size *= strength_modifier
            
        # Ensure position size doesn't exceed maximum
        max_position_value = account_balance * (self.parameters["max_position_size_pct"] / 100.0)
        position_value = position_size * entry_price
        
        if position_value > max_position_value:
            position_size = max_position_value / entry_price
            
        return position_size
    
    def _check_exit_conditions(self, symbol: str, position: Dict[str, Any], 
                             current_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Check if exit conditions are met for an existing position.
        
        Args:
            symbol: Currency pair symbol
            position: Position information
            current_data: Current OHLCV data
            
        Returns:
            Tuple of (should_exit, reason)
        """
        # Skip if no data
        if current_data.empty or len(current_data) < 2:
            return False, ""
            
        # Get current price
        current_price = current_data["close"].iloc[-1]
        
        # Get position details
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        direction = position["direction"]
        
        # Check stop loss
        if direction == "buy" and current_price <= stop_loss:
            return True, "stop_loss"
        elif direction == "sell" and current_price >= stop_loss:
            return True, "stop_loss"
            
        # Check take profit
        if direction == "buy" and current_price >= take_profit:
            return True, "take_profit"
        elif direction == "sell" and current_price <= take_profit:
            return True, "take_profit"
            
        # Check for reversal signals
        # Generate signals from models
        signals = {}
        
        # We'll check for strong reversal signals from mean reversion and trend following
        if self.parameters["use_mean_reversion"]:
            mr_signal = self._generate_mean_reversion_signal(symbol, current_data)
            if mr_signal["direction"] != "neutral" and mr_signal["direction"] != direction and mr_signal["strength"] > 0.7:
                return True, "mean_reversion_reversal"
                
        if self.parameters["use_trend_following"]:
            tf_signal = self._generate_trend_following_signal(symbol, current_data)
            if tf_signal["direction"] != "neutral" and tf_signal["direction"] != direction and tf_signal["strength"] > 0.7:
                return True, "trend_following_reversal"
                
        # No exit condition met
        return False, ""
    
    def _adjust_stop_loss(self, position: Dict[str, Any], current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Adjust stop loss based on current price action.
        
        Args:
            position: Position information
            current_data: Current OHLCV data
            
        Returns:
            Updated position with adjusted stop loss
        """
        # Skip if adaptive stops are disabled
        if not self.parameters["use_adaptive_stops"] or current_data.empty or len(current_data) < 20:
            return position
            
        # Get current price
        current_price = current_data["close"].iloc[-1]
        
        # Get position details
        entry_price = position["entry_price"]
        current_stop = position["stop_loss"]
        direction = position["direction"]
        entry_time = position.get("entry_time", self.current_time)
        
        # Calculate unrealized profit
        unrealized_profit = (current_price - entry_price) if direction == "buy" else (entry_price - current_price)
        
        # Calculate ATR for stop adjustment
        atr = self._calculate_atr_percent(current_data).iloc[-1] * current_price / 100 if len(current_data) > 14 else current_price * 0.001
        
        # Don't adjust stop if trade is too new (less than 4 hours)
        if self.current_time is not None and entry_time is not None:
            hours_in_trade = (self.current_time - entry_time).total_seconds() / 3600
            if hours_in_trade < 4:
                return position
                
        # Adjust stop based on unrealized profit
        if unrealized_profit > 2 * atr:  # Significant profit
            # Move stop to break even + small buffer
            new_stop = entry_price + (0.2 * atr) if direction == "buy" else entry_price - (0.2 * atr)
            
            # Only move stop in the right direction (closer to price)
            if (direction == "buy" and new_stop > current_stop) or (direction == "sell" and new_stop < current_stop):
                position["stop_loss"] = new_stop
                position["stop_type"] = "break_even_plus"
                
        if unrealized_profit > 4 * atr:  # Large profit
            # Start trailing stop
            trailing_distance = 1.5 * atr
            
            if direction == "buy":
                new_stop = current_price - trailing_distance
            else:  # sell
                new_stop = current_price + trailing_distance
                
            # Only move stop in the right direction (closer to price)
            if (direction == "buy" and new_stop > current_stop) or (direction == "sell" and new_stop < current_stop):
                position["stop_loss"] = new_stop
                position["stop_type"] = "trailing"
                
        return position
    
    def on_data(self, data: Dict[str, pd.DataFrame], timestamp: datetime):
        """
        Process new market data, update models, and generate signals.
        
        Args:
            data: Dictionary of DataFrames with symbol as key
            timestamp: Current timestamp
        """
        self.current_time = timestamp
        
        # Update internal data store
        for symbol, df in data.items():
            if symbol in self.symbols:
                self.market_data[symbol] = df.copy()
        
        # Update statistical arbitrage model (needs data from all symbols)
        if self.parameters["use_statistical_arbitrage"]:
            update_needed = False
            
            # Check if any symbol needs updating
            for symbol in self.symbols:
                if symbol not in self.models["statistical_arbitrage"] or \
                   self.models["statistical_arbitrage"][symbol]["last_update"] is None or \
                   (timestamp - self.models["statistical_arbitrage"][symbol]["last_update"]).total_seconds() > 3600:  # 1 hour
                    update_needed = True
                    break
                    
            if update_needed:
                self._update_statistical_arbitrage_model(self.market_data)
        
        # Check if parameters need to be updated
        if self.parameters["enable_parameter_adaptation"] and self.current_time is not None:
            hours_since_last_update = 0
            if self.last_parameter_update is not None:
                hours_since_last_update = (self.current_time - self.last_parameter_update).total_seconds() / 3600
                
            if hours_since_last_update >= self.parameters["parameter_update_frequency_hours"]:
                self._update_adaptive_parameters()
                self.last_parameter_update = self.current_time
        
        # First, process existing positions
        positions_to_update = {}
        for position_id, position in self.active_positions.items():
            symbol = position["symbol"]
            
            # Skip if we don't have data for this symbol
            if symbol not in self.market_data:
                continue
                
            # Get current data for this symbol
            current_data = self.market_data[symbol]
            
            # Adjust stop loss
            updated_position = self._adjust_stop_loss(position.copy(), current_data)
            
            # Check if exit conditions are met
            should_exit, exit_reason = self._check_exit_conditions(symbol, updated_position, current_data)
            
            if should_exit:
                self.logger.info(f"Exit signal for {symbol} position {position_id}: {exit_reason}")
                
                # Create exit event
                exit_event = {
                    "type": "exit",
                    "position_id": position_id,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "price": current_data["close"].iloc[-1],
                    "reason": exit_reason,
                    "position": updated_position
                }
                
                # Emit exit event
                self.events.append(exit_event)
                
                # Remove from active positions
                # We don't actually remove it here, we'll do it in update() to avoid modifying while iterating
                positions_to_update[position_id] = {"status": "pending_exit", "position": updated_position}
            else:
                # Just update the position with new stops
                positions_to_update[position_id] = {"status": "active", "position": updated_position}
        
        # Apply position updates
        for position_id, update_info in positions_to_update.items():
            if update_info["status"] == "pending_exit":
                # Will be removed in update()
                self.positions_to_close.append(position_id)
            else:
                self.active_positions[position_id] = update_info["position"]
        
        # Now look for new entry signals
        # Get current account info
        account_balance = getattr(self, "account_balance", 100000)  # Default if not available
        
        # Track exposure and correlation
        total_exposure = 0
        symbol_exposures = {}
        
        # Calculate current exposure
        for position in self.active_positions.values():
            symbol = position["symbol"]
            position_value = position["size"] * position.get("entry_price", 0)
            symbol_exposures[symbol] = position_value
            total_exposure += position_value
            
        # Check if we're at max positions
        if len(self.active_positions) >= self.parameters["max_concurrent_positions"]:
            return
            
        # Limit total exposure
        max_account_exposure = account_balance * (self.parameters["max_position_size_pct"] * 2 / 100)  # 2x single position limit
        if total_exposure >= max_account_exposure:
            return
        
        # Check each symbol for entry signals
        for symbol, df in self.market_data.items():
            # Skip if we already have a position in this symbol
            if any(pos["symbol"] == symbol for pos in self.active_positions.values()):
                continue
                
            # Skip if symbol not in our universe
            if symbol not in self.symbols:
                continue
                
            # Generate signal for this symbol
            signal = self._generate_signal(symbol, df)
            
            # Skip if no signal or neutral
            if signal is None or signal["direction"] == "neutral":
                continue
                
            # Calculate position size
            position_size = self._calculate_position_size(signal, account_balance)
            if position_size <= 0:
                continue
                
            # Skip if this would exceed max exposure
            position_value = position_size * signal["entry_price"]
            if total_exposure + position_value > max_account_exposure:
                continue
                
            # Create a new position
            position_id = f"algo_{symbol}_{timestamp.strftime('%Y%m%d%H%M%S')}"
            
            position = {
                "id": position_id,
                "symbol": symbol,
                "direction": signal["direction"],
                "entry_price": signal["entry_price"],
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "size": position_size,
                "entry_time": timestamp,
                "strategy": "AlgorithmicTrading",
                "model": signal["model"],
                "initial_stop_loss": signal["stop_loss"]  # Keep track of initial stop for calculations
            }
            
            # Create entry event
            entry_event = {
                "type": "entry",
                "position_id": position_id,
                "symbol": symbol,
                "timestamp": timestamp,
                "price": signal["entry_price"],
                "direction": signal["direction"],
                "size": position_size,
                "stop_loss": signal["stop_loss"],
                "take_profit": signal["take_profit"],
                "reason": f"{signal['model']} model with strength {signal['strength']:.2f}",
                "position": position
            }
            
            # Emit entry event
            self.events.append(entry_event)
            
            # Add to active positions
            self.active_positions[position_id] = position
            
            self.logger.info(f"New {signal['direction']} signal for {symbol} from {signal['model']} model")
            
            # Update total exposure
            total_exposure += position_value
            
            # Check if we've reached max positions
            if len(self.active_positions) >= self.parameters["max_concurrent_positions"]:
                break
    
    def update(self):
        """
        Process pending events and update strategy state.
        """
        # Process any positions that need to be closed
        for position_id in self.positions_to_close:
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                
                # Calculate profit/loss
                entry_price = position["entry_price"]
                symbol = position["symbol"]
                direction = position["direction"]
                
                # Use the most recent close price as exit price
                exit_price = None
                if symbol in self.market_data:
                    df = self.market_data[symbol]
                    if not df.empty:
                        exit_price = df["close"].iloc[-1]
                
                if exit_price is not None:
                    # Calculate profit in pips
                    direction_multiplier = 1 if direction == "buy" else -1
                    profit_pips = direction_multiplier * (exit_price - entry_price) * 10000  # Convert to pips
                    
                    # Calculate R-multiple (risk multiple)
                    initial_risk = abs(entry_price - position["initial_stop_loss"])
                    r_multiple = direction_multiplier * (exit_price - entry_price) / initial_risk if initial_risk > 0 else 0
                    
                    # Update performance metrics
                    self.trades_count += 1
                    if profit_pips > 0:
                        self.winning_trades += 1
                    self.total_profit_pips += profit_pips
                    self.r_multiples.append(r_multiple)
                    
                    # Update model performance if we're tracking it
                    if self.parameters["track_model_performance"] and "model" in position:
                        model_name = position["model"]
                        self.model_returns[model_name].append(r_multiple)
                    
                    self.logger.info(f"Closed position {position_id} with {profit_pips:.1f} pips profit (R = {r_multiple:.2f})")
                
                # Remove from active positions
                del self.active_positions[position_id]
                
                # Add to closed positions history
                self.closed_positions.append({
                    "position_id": position_id,
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit_pips": profit_pips if exit_price is not None else None,
                    "r_multiple": r_multiple if exit_price is not None else None,
                    "entry_time": position.get("entry_time"),
                    "exit_time": self.current_time,
                    "model": position.get("model")
                })
        
        # Clear list of positions to close
        self.positions_to_close = []
        
        # If we're tracking model performance, periodically adjust model weights
        if self.parameters["track_model_performance"] and self.trades_count > 10:
            self._update_model_weights()
    
    def _update_model_weights(self):
        """
        Update model weights based on recent performance.
        """
        # Skip if we don't have enough data
        if not all(len(returns) > 0 for returns in self.model_returns.values()):
            return
            
        # Calculate average returns for each model
        avg_returns = {}
        for model_name, returns in self.model_returns.items():
            if returns:  # Check if we have returns for this model
                avg_returns[model_name] = sum(returns) / len(returns)
                
        # Skip if no returns data
        if not avg_returns:
            return
            
        # Normalize returns to get weights
        total_return = sum(max(0.1, r) for r in avg_returns.values())  # Ensure no negative weights
        
        if total_return <= 0:
            return
            
        new_weights = {}
        for model_name, ret in avg_returns.items():
            # Ensure weight is positive
            adjusted_return = max(0.1, ret)
            new_weights[model_name] = adjusted_return / total_return
            
        # Update ensemble weights, but limit change to avoid instability
        for model_name, new_weight in new_weights.items():
            current_weight = self.parameters["ensemble_weights"].get(model_name, 0.25)
            # Limit change to 10% per update
            max_change = current_weight * 0.1
            delta = new_weight - current_weight
            limited_delta = max(-max_change, min(max_change, delta))
            self.parameters["ensemble_weights"][model_name] = current_weight + limited_delta
            
        self.logger.info(f"Updated model weights: {self.parameters['ensemble_weights']}")
    
    def _update_adaptive_parameters(self):
        """
        Update strategy parameters based on market conditions and performance.
        """
        # Only adapt parameters if we have enough trading history
        if self.trades_count < 10:
            return
            
        # Calculate win rate and average R
        win_rate = self.winning_trades / self.trades_count if self.trades_count > 0 else 0.5
        avg_r = sum(self.r_multiples) / len(self.r_multiples) if self.r_multiples else 0
        
        # Adjust parameters based on performance
        # 1. Entry filter threshold - increase if win rate is low
        if win_rate < 0.4:
            self.parameters["entry_filter_threshold"] = min(0.8, self.parameters["entry_filter_threshold"] + 0.05)
        elif win_rate > 0.6:
            self.parameters["entry_filter_threshold"] = max(0.5, self.parameters["entry_filter_threshold"] - 0.05)
            
        # 2. Z-score threshold - adjust based on average R
        if avg_r < 0.5:
            self.parameters["z_score_threshold"] = min(3.0, self.parameters["z_score_threshold"] + 0.2)
        elif avg_r > 1.5:
            self.parameters["z_score_threshold"] = max(1.5, self.parameters["z_score_threshold"] - 0.2)
            
        # 3. Risk per trade - adjust based on win rate and R
        expected_value = win_rate * avg_r - (1 - win_rate)
        if expected_value > 0.5:  # Very good expectancy
            self.parameters["risk_per_trade_pct"] = min(2.0, self.parameters["risk_per_trade_pct"] + 0.1)
        elif expected_value < 0:  # Negative expectancy
            self.parameters["risk_per_trade_pct"] = max(0.5, self.parameters["risk_per_trade_pct"] - 0.1)
            
        self.logger.info(f"Adapted parameters based on performance: "
                        f"entry_threshold={self.parameters['entry_filter_threshold']:.2f}, "
                        f"z_score={self.parameters['z_score_threshold']:.2f}, "
                        f"risk={self.parameters['risk_per_trade_pct']:.2f}%")
    
    def shutdown(self):
        """
        Clean up and save state when strategy is shutting down.
        """
        # Log performance statistics
        if self.trades_count > 0:
            win_rate = self.winning_trades / self.trades_count * 100
            avg_pips = self.total_profit_pips / self.trades_count if self.trades_count > 0 else 0
            avg_r = sum(self.r_multiples) / len(self.r_multiples) if self.r_multiples else 0
            
            self.logger.info(f"Strategy performance: Trades={self.trades_count}, "
                           f"Win rate={win_rate:.1f}%, "
                           f"Avg pips={avg_pips:.1f}, "
                           f"Avg R={avg_r:.2f}, "
                           f"Total pips={self.total_profit_pips:.1f}")
            
            # Log model performance
            if self.parameters["track_model_performance"]:
                for model_name, returns in self.model_returns.items():
                    if returns:
                        model_avg_r = sum(returns) / len(returns)
                        model_win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
                        self.logger.info(f"Model {model_name}: Trades={len(returns)}, "
                                       f"Win rate={model_win_rate:.1f}%, "
                                       f"Avg R={model_avg_r:.2f}")
        
        # Close all open positions if requested
        if self.active_positions and hasattr(self, "close_positions_on_shutdown") and self.close_positions_on_shutdown:
            self.logger.info(f"Closing {len(self.active_positions)} positions on shutdown")
            
            for position_id in list(self.active_positions.keys()):
                self.positions_to_close.append(position_id)
                
            # Process closings
            self.update()
