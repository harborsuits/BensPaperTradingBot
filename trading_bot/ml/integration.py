"""
Machine Learning Integration Module

This module connects the various machine learning components to the trading bot's
core systems. It provides a unified interface for using predictive models,
market condition classification, parameter optimization, and anomaly detection
in trading strategies.
"""

import pandas as pd
import numpy as np
import logging
import os
import joblib
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta

# Import ML components
from trading_bot.ml.price_prediction import PricePredictionModel
from trading_bot.ml.market_classifier import MarketConditionClassifier, MarketCondition
from trading_bot.ml.parameter_optimizer import ParameterOptimizer
from trading_bot.ml.anomaly_detector import MarketAnomalyDetector, AnomalyType

# Import adapter
from trading_bot.multi_asset_adapter import MultiAssetAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLManager:
    """
    Manages and integrates machine learning components with the trading system.
    
    This class coordinates the various ML models, handles data preparation,
    manages model lifecycles, and provides a unified interface for strategies
    to access predictions and analysis.
    """
    
    def __init__(
        self,
        adapter: MultiAssetAdapter,
        config_path: Optional[str] = None,
        models_dir: str = "models"
    ):
        """
        Initialize the ML Manager.
        
        Args:
            adapter: MultiAssetAdapter instance for market data access
            config_path: Path to ML configuration file (optional)
            models_dir: Directory to store trained models
        """
        self.adapter = adapter
        self.models_dir = models_dir
        self.config = self._load_config(config_path)
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize ML components
        self.price_prediction_models = {}  # Symbol -> model mapping
        self.market_classifier = None
        self.parameter_optimizer = None
        self.anomaly_detector = None
        
        # Feature engineering cache
        self.feature_cache = {}
        
        # Performance tracking
        self.prediction_performance = {}
        self.last_market_condition = None
        self.detected_anomalies = []
        
        # Initialize the ML components based on config
        self._initialize_components()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "enable_price_prediction": True,
            "enable_market_classifier": True,
            "enable_parameter_optimizer": True,
            "enable_anomaly_detector": True,
            "price_prediction": {
                "lookback_window": 20,
                "prediction_horizon": 5,
                "confidence_threshold": 0.65,
                "retrain_interval_days": 7
            },
            "market_classifier": {
                "lookback_window": 100,
                "ma_short": 10,
                "ma_medium": 50,
                "ma_long": 200,
                "volatility_window": 20
            },
            "parameter_optimizer": {
                "max_trials": 100,
                "timeout": 600,
                "n_jobs": -1
            },
            "anomaly_detector": {
                "lookback_window": 100,
                "contamination": 0.01,
                "volume_threshold": 3.0,
                "price_threshold": 3.0
            },
            "symbols": [
                "SPY", "QQQ", "AAPL", "MSFT", "GOOGL"
            ],
            "timeframes": [
                "1h", "1d"
            ]
        }
        
        if config_path and os.path.exists(config_path):
            try:
                import json
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Merge with defaults (to ensure all keys exist)
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
                logger.info(f"Loaded ML configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading ML configuration: {e}")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize ML components based on configuration."""
        try:
            # Initialize market condition classifier
            if self.config["enable_market_classifier"]:
                mc_config = self.config["market_classifier"]
                self.market_classifier = MarketConditionClassifier(
                    lookback_window=mc_config["lookback_window"],
                    ma_short=mc_config["ma_short"],
                    ma_medium=mc_config["ma_medium"],
                    ma_long=mc_config["ma_long"],
                    volatility_window=mc_config["volatility_window"]
                )
                
                # Load pre-trained model if exists
                model_path = os.path.join(self.models_dir, "market_classifier.joblib")
                if os.path.exists(model_path):
                    self.market_classifier.load_model(model_path)
                    logger.info("Loaded pre-trained market classifier model")
            
            # Initialize parameter optimizer
            if self.config["enable_parameter_optimizer"]:
                po_config = self.config["parameter_optimizer"]
                self.parameter_optimizer = ParameterOptimizer(
                    max_trials=po_config["max_trials"],
                    timeout=po_config["timeout"],
                    n_jobs=po_config["n_jobs"]
                )
                
                # Load pre-trained model if exists
                model_path = os.path.join(self.models_dir, "parameter_optimizer.joblib")
                if os.path.exists(model_path):
                    self.parameter_optimizer.load_model(model_path)
                    logger.info("Loaded pre-trained parameter optimizer model")
            
            # Initialize anomaly detector
            if self.config["enable_anomaly_detector"]:
                from trading_bot.ml.anomaly_detector import AnomalyDetectionConfig
                
                ad_config = self.config["anomaly_detector"]
                config_obj = AnomalyDetectionConfig(
                    lookback_window=ad_config.get("lookback_window", 100),
                    contamination=ad_config.get("contamination", 0.01),
                    volume_threshold=ad_config.get("volume_threshold", 3.0),
                    price_threshold=ad_config.get("price_threshold", 3.0),
                    volatility_threshold=ad_config.get("volatility_threshold", 3.0),
                    liquidity_threshold=ad_config.get("liquidity_threshold", 2.0)
                )
                
                self.anomaly_detector = MarketAnomalyDetector(config=config_obj)
                
                # Load pre-trained model if exists
                model_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
                if os.path.exists(model_path):
                    self.anomaly_detector.load_model(model_path)
                    logger.info("Loaded pre-trained anomaly detector model")
            
            # Initialize price prediction models for configured symbols
            if self.config["enable_price_prediction"]:
                pp_config = self.config["price_prediction"]
                
                for symbol in self.config["symbols"]:
                    self.price_prediction_models[symbol] = {}
                    
                    for timeframe in self.config["timeframes"]:
                        model = PricePredictionModel(
                            lookback_window=pp_config["lookback_window"],
                            prediction_horizon=pp_config["prediction_horizon"],
                            confidence_threshold=pp_config["confidence_threshold"]
                        )
                        
                        # Load pre-trained model if exists
                        model_path = os.path.join(self.models_dir, f"price_prediction_{symbol}_{timeframe}.joblib")
                        if os.path.exists(model_path):
                            model.load_model(model_path)
                            logger.info(f"Loaded pre-trained price prediction model for {symbol} {timeframe}")
                            
                        self.price_prediction_models[symbol][timeframe] = model
                        
            logger.info("ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
    
    def prepare_data(self, symbol: str, timeframe: str, 
                    lookback_bars: int = 500) -> pd.DataFrame:
        """
        Fetch and prepare data for ML models.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for data
            lookback_bars: Number of historical bars to fetch
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            # Use adapter to fetch data
            ohlcv = self.adapter.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=lookback_bars
            )
            
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Check if we have this data in cache
            cache_key = f"{symbol}_{timeframe}_{len(ohlcv)}"
            if cache_key in self.feature_cache:
                cache_time, cached_data = self.feature_cache[cache_key]
                # Use cache if it's less than 10 minutes old and has the same number of rows
                if datetime.now() - cache_time < timedelta(minutes=10) and len(cached_data) == len(ohlcv):
                    return cached_data
            
            # Add technical indicators
            data = self._add_technical_indicators(ohlcv)
            
            # Store in cache
            self.feature_cache[cache_key] = (datetime.now(), data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            import talib as ta
        except ImportError:
            logger.warning("TA-Lib not installed. Using pandas for indicators.")
            return self._add_pandas_indicators(data)
            
        try:
            df = data.copy()
            
            # Ensure columns are properly named
            if 'open' not in df.columns:
                df.columns = [c.lower() for c in df.columns]
            
            # Moving Averages
            df['sma_10'] = ta.SMA(df['close'], timeperiod=10)
            df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
            df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
            df['sma_200'] = ta.SMA(df['close'], timeperiod=200)
            df['ema_10'] = ta.EMA(df['close'], timeperiod=10)
            df['ema_20'] = ta.EMA(df['close'], timeperiod=20)
            
            # Momentum Indicators
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            df['willr'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Volatility Indicators
            df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = ta.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
            )
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Volume Indicators
            df['obv'] = ta.OBV(df['close'], df['volume'])
            df['ad'] = ta.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # Trend Indicators
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Oscillators
            df['stoch_k'], df['stoch_d'] = ta.STOCH(
                df['high'], df['low'], df['close'], 
                fastk_period=5, slowk_period=3, slowd_period=3
            )
            
            # Custom indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma']
            
            # Normalized price features
            df['norm_open'] = df['open'] / df['close'].shift(1) - 1
            df['norm_high'] = df['high'] / df['close'].shift(1) - 1
            df['norm_low'] = df['low'] / df['close'].shift(1) - 1
            df['norm_close'] = df['close'] / df['close'].shift(1) - 1
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return data
    
    def _add_pandas_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using pandas (fallback if TA-Lib is not available).
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            df = data.copy()
            
            # Ensure columns are properly named
            if 'open' not in df.columns:
                df.columns = [c.lower() for c in df.columns]
            
            # Moving Averages
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['sma_200'] = df['close'].rolling(200).mean()
            
            # Exponential Moving Averages
            df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            # Price changes
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Volatility
            df['volatility'] = df['returns'].rolling(20).std()
            df['range'] = (df['high'] - df['low']) / df['close']
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['relative_volume'] = df['volume'] / df['volume_ma']
            
            # Simple RSI implementation
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bbands_middle'] = df['close'].rolling(20).mean()
            df['bbands_std'] = df['close'].rolling(20).std()
            df['bbands_upper'] = df['bbands_middle'] + (df['bbands_std'] * 2)
            df['bbands_lower'] = df['bbands_middle'] - (df['bbands_std'] * 2)
            
            # Rate of change
            df['roc'] = df['close'].pct_change(10) * 100
            
            # Normalized price features
            df['norm_open'] = df['open'] / df['close'].shift(1) - 1
            df['norm_high'] = df['high'] / df['close'].shift(1) - 1
            df['norm_low'] = df['low'] / df['close'].shift(1) - 1
            df['norm_close'] = df['close'] / df['close'].shift(1) - 1
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding pandas indicators: {e}")
            return data
    
    def get_price_prediction(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get price movement prediction for a symbol.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with prediction details
        """
        if not self.config["enable_price_prediction"]:
            return {"enabled": False}
            
        try:
            # Check if we have a model for this symbol/timeframe
            if symbol not in self.price_prediction_models or timeframe not in self.price_prediction_models[symbol]:
                return {"error": f"No prediction model for {symbol} {timeframe}"}
                
            model = self.price_prediction_models[symbol][timeframe]
            
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=max(500, model.lookback_window * 3)
            )
            
            if data.empty:
                return {"error": "No data available for prediction"}
                
            # Make prediction
            prediction = model.predict(data)
            
            # Update performance tracking
            self._update_prediction_performance(symbol, timeframe, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting price prediction for {symbol} {timeframe}: {e}")
            return {"error": str(e)}
    
    def _update_prediction_performance(self, symbol: str, timeframe: str, prediction: Dict[str, Any]):
        """Update prediction performance tracking."""
        if "prediction" not in prediction or "confidence" not in prediction:
            return
            
        key = f"{symbol}_{timeframe}"
        if key not in self.prediction_performance:
            self.prediction_performance[key] = {
                "predictions": [],
                "accuracy": 0.0,
                "total": 0,
                "correct": 0
            }
            
        perf = self.prediction_performance[key]
        perf["predictions"].append({
            "timestamp": datetime.now(),
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "price": prediction.get("current_price"),
            "target_price": prediction.get("target_price"),
            "verified": False,
            "actual": None
        })
        
        # Keep only the last 100 predictions
        if len(perf["predictions"]) > 100:
            perf["predictions"] = perf["predictions"][-100:]
    
    def verify_prediction_accuracy(self):
        """Verify previous predictions against actual outcomes."""
        try:
            # For each symbol/timeframe
            for key, perf in self.prediction_performance.items():
                symbol, timeframe = key.split("_")
                
                # Get current data
                data = self.prepare_data(symbol, timeframe, 50)
                if data.empty:
                    continue
                    
                # For each unverified prediction
                for pred in [p for p in perf["predictions"] if not p["verified"]]:
                    pred_time = pred["timestamp"]
                    horizon = self.price_prediction_models[symbol][timeframe].prediction_horizon
                    
                    # Check if enough time has passed to verify
                    if timeframe == "1h":
                        if datetime.now() - pred_time < timedelta(hours=horizon):
                            continue
                    elif timeframe == "1d":
                        if datetime.now() - pred_time < timedelta(days=horizon):
                            continue
                            
                    # Find actual outcome
                    try:
                        current_price = data['close'].iloc[-1]
                        pred_price = pred["price"]
                        
                        # Calculate actual direction
                        actual_direction = "up" if current_price > pred_price else "down"
                        pred["actual"] = actual_direction
                        pred["verified"] = True
                        
                        # Update accuracy
                        perf["total"] += 1
                        if actual_direction == pred["prediction"]:
                            perf["correct"] += 1
                            
                        perf["accuracy"] = perf["correct"] / perf["total"]
                    except Exception as e:
                        logger.warning(f"Error verifying prediction: {e}")
                        
            logger.info("Prediction accuracy verification completed")
            
        except Exception as e:
            logger.error(f"Error verifying predictions: {e}")
    
    def train_price_prediction_model(self, symbol: str, timeframe: str,
                                     force: bool = False) -> bool:
        """
        Train or update a price prediction model.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for model
            force: Force retraining even if recently trained
            
        Returns:
            True if training successful, False otherwise
        """
        if not self.config["enable_price_prediction"]:
            return False
            
        try:
            # Check if model exists
            if symbol not in self.price_prediction_models:
                self.price_prediction_models[symbol] = {}
                
            # Create new model if needed
            pp_config = self.config["price_prediction"]
            if timeframe not in self.price_prediction_models[symbol]:
                model = PricePredictionModel(
                    lookback_window=pp_config["lookback_window"],
                    prediction_horizon=pp_config["prediction_horizon"],
                    confidence_threshold=pp_config["confidence_threshold"]
                )
                self.price_prediction_models[symbol][timeframe] = model
            else:
                model = self.price_prediction_models[symbol][timeframe]
            
            # Check if training is needed
            if not force and model.last_trained:
                days_since_train = (datetime.now() - model.last_trained).days
                if days_since_train < pp_config["retrain_interval_days"]:
                    logger.info(f"Skipping training for {symbol} {timeframe}, last trained {days_since_train} days ago")
                    return True
            
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=2000  # Get more data for training
            )
            
            if data.empty:
                logger.warning(f"No data available for training {symbol} {timeframe}")
                return False
                
            # Train model
            logger.info(f"Training price prediction model for {symbol} {timeframe}")
            success = model.train(data)
            
            if success:
                # Save model
                model_path = os.path.join(self.models_dir, f"price_prediction_{symbol}_{timeframe}.joblib")
                model.save_model(model_path)
                logger.info(f"Saved price prediction model to {model_path}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error training price prediction model for {symbol} {timeframe}: {e}")
            return False
    
    def get_market_condition(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get current market condition classification.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with market condition details
        """
        if not self.config["enable_market_classifier"] or not self.market_classifier:
            return {"enabled": False}
            
        try:
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=max(500, self.market_classifier.lookback_window * 3)
            )
            
            if data.empty:
                return {"error": "No data available for market classification"}
                
            # Train if first time
            if not hasattr(self.market_classifier, 'model') or self.market_classifier.model is None:
                logger.info(f"Training market classifier for {symbol} {timeframe}")
                self.market_classifier.train(data)
                model_path = os.path.join(self.models_dir, "market_classifier.joblib")
                self.market_classifier.save_model(model_path)
                
            # Get prediction
            condition = self.market_classifier.predict(data)
            
            # Store last condition
            self.last_market_condition = {
                "symbol": symbol,
                "timeframe": timeframe,
                "condition": condition["condition"],
                "timestamp": datetime.now()
            }
            
            return condition
            
        except Exception as e:
            logger.error(f"Error getting market condition for {symbol} {timeframe}: {e}")
            return {"error": str(e)}
    
    def get_historical_market_conditions(self, symbol: str, timeframe: str, 
                                        lookback_bars: int = 100) -> Dict[str, Any]:
        """
        Get historical market conditions over a period.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            lookback_bars: Number of bars to analyze
            
        Returns:
            Dictionary with historical market conditions
        """
        if not self.config["enable_market_classifier"] or not self.market_classifier:
            return {"enabled": False}
            
        try:
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=lookback_bars
            )
            
            if data.empty:
                return {"error": "No data available for historical market classification"}
                
            # Get historical conditions
            historical = self.market_classifier.get_historical_conditions(data)
            
            return {
                "historical_conditions": historical,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback_bars": lookback_bars
            }
            
        except Exception as e:
            logger.error(f"Error getting historical market conditions: {e}")
            return {"error": str(e)}
    
    def visualize_market_conditions(self, symbol: str, timeframe: str,
                                  lookback_bars: int = 100) -> Dict[str, Any]:
        """
        Generate a visualization of market conditions.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            lookback_bars: Number of bars to visualize
            
        Returns:
            Dictionary with visualization info
        """
        if not self.config["enable_market_classifier"] or not self.market_classifier:
            return {"enabled": False}
            
        try:
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=lookback_bars
            )
            
            if data.empty:
                return {"error": "No data available for visualization"}
                
            # Generate plot
            fig = self.market_classifier.plot_market_conditions(data)
            
            if fig:
                # Save figure to file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    filename = tmp.name
                
                fig.savefig(filename)
                
                return {
                    "visualization": filename,
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            else:
                return {"error": "Failed to generate visualization"}
            
        except Exception as e:
            logger.error(f"Error visualizing market conditions: {e}")
            return {"error": str(e)}
    
    def optimize_parameters(self, strategy_name: str, parameter_space: Dict[str, Any],
                         symbol: str, timeframe: str, market_condition: Optional[str] = None,
                         metric: str = "sharpe_ratio", max_trials: Optional[int] = None) -> Dict[str, Any]:
        """
        Optimize strategy parameters for current or specific market condition.
        
        Args:
            strategy_name: Name of the strategy to optimize
            parameter_space: Dictionary of parameters and their ranges
            symbol: Instrument symbol
            timeframe: Timeframe for optimization
            market_condition: Specific market condition or None for current
            metric: Metric to optimize (e.g., "sharpe_ratio", "return", "win_rate")
            max_trials: Maximum number of trials to run
            
        Returns:
            Dictionary with optimized parameters
        """
        if not self.config["enable_parameter_optimizer"] or not self.parameter_optimizer:
            return {"enabled": False}
            
        try:
            # Get market condition if not specified
            if market_condition is None:
                condition_result = self.get_market_condition(symbol, timeframe)
                if "error" in condition_result:
                    return {"error": f"Could not determine market condition: {condition_result['error']}"}
                market_condition = condition_result["condition"]
                
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=500
            )
            
            if data.empty:
                return {"error": "No data available for parameter optimization"}
                
            # Run optimization
            logger.info(f"Optimizing parameters for {strategy_name} in {market_condition} market")
            
            if max_trials is not None:
                self.parameter_optimizer.max_trials = max_trials
                
            result = self.parameter_optimizer.optimize(
                strategy_name=strategy_name,
                parameter_space=parameter_space,
                data=data,
                market_condition=market_condition,
                metric=metric
            )
            
            # Save model
            model_path = os.path.join(self.models_dir, "parameter_optimizer.joblib")
            self.parameter_optimizer.save_model(model_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {e}")
            return {"error": str(e)}
    
    def get_optimal_parameters(self, strategy_name: str, symbol: str,
                            timeframe: str) -> Dict[str, Any]:
        """
        Get optimal parameters for current market condition.
        
        Args:
            strategy_name: Name of the strategy
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with optimal parameters
        """
        if not self.config["enable_parameter_optimizer"] or not self.parameter_optimizer:
            return {"enabled": False}
            
        try:
            # Get current market condition
            condition_result = self.get_market_condition(symbol, timeframe)
            if "error" in condition_result:
                return {"error": f"Could not determine market condition: {condition_result['error']}"}
                
            market_condition = condition_result["condition"]
            
            # Get optimal parameters
            params = self.parameter_optimizer.get_optimal_parameters(
                strategy_name=strategy_name,
                market_condition=market_condition
            )
            
            return {
                "strategy": strategy_name,
                "market_condition": market_condition,
                "parameters": params,
                "symbol": symbol,
                "timeframe": timeframe
            }
            
        except Exception as e:
            logger.error(f"Error getting optimal parameters: {e}")
            return {"error": str(e)}
    
    def detect_market_anomalies(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Detect market microstructure anomalies.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.config["enable_anomaly_detector"] or not self.anomaly_detector:
            return {"enabled": False}
            
        try:
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=500
            )
            
            if data.empty:
                return {"error": "No data available for anomaly detection"}
                
            # Train model if first time
            if not hasattr(self.anomaly_detector, 'model') or self.anomaly_detector.model is None:
                logger.info(f"Training anomaly detector for {symbol} {timeframe}")
                self.anomaly_detector.train(data)
                model_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
                self.anomaly_detector.save_model(model_path)
                
            # Detect anomalies
            result_df = self.anomaly_detector.detect_anomalies(data)
            
            # Prepare results
            anomalies = result_df[result_df['is_anomaly'] == 1]
            
            if len(anomalies) > 0:
                # Get recent anomalies
                recent_anomalies = self.anomaly_detector.get_latest_anomalies(result_df, n=5)
                
                # Add to detected anomalies list
                for idx, row in recent_anomalies.iterrows():
                    anomaly_record = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": idx,
                        "price": row["close"],
                        "score": row["anomaly_score"],
                        "types": [col for col in row.index if col.startswith("anomaly_") and row[col] == 1],
                        "detected_at": datetime.now()
                    }
                    self.detected_anomalies.append(anomaly_record)
                
                # Keep only most recent 100 anomalies
                if len(self.detected_anomalies) > 100:
                    self.detected_anomalies = self.detected_anomalies[-100:]
                    
                # Get summary
                summary = self.anomaly_detector.get_anomaly_summary(result_df)
                
                return {
                    "anomalies_detected": len(anomalies),
                    "recent_anomalies": recent_anomalies.to_dict(orient='records'),
                    "summary": summary,
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            else:
                return {
                    "anomalies_detected": 0,
                    "message": "No anomalies detected",
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"error": str(e)}
    
    def visualize_anomalies(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Generate a visualization of detected anomalies.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with visualization info
        """
        if not self.config["enable_anomaly_detector"] or not self.anomaly_detector:
            return {"enabled": False}
            
        try:
            # Prepare data
            data = self.prepare_data(
                symbol=symbol, 
                timeframe=timeframe,
                lookback_bars=200
            )
            
            if data.empty:
                return {"error": "No data available for visualization"}
                
            # Detect anomalies if not already detected
            if 'is_anomaly' not in data.columns:
                result_df = self.anomaly_detector.detect_anomalies(data)
            else:
                result_df = data
                
            # Generate plot
            fig = self.anomaly_detector.plot_anomalies(result_df)
            
            if fig:
                # Save figure to file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    filename = tmp.name
                
                fig.savefig(filename)
                
                return {
                    "visualization": filename,
                    "symbol": symbol,
                    "timeframe": timeframe
                }
            else:
                return {"error": "Failed to generate visualization"}
            
        except Exception as e:
            logger.error(f"Error visualizing anomalies: {e}")
            return {"error": str(e)}
    
    def get_ml_summary(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Get a summary of all ML component statuses and predictions.
        
        Args:
            symbol: Instrument symbol
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary with ML summary
        """
        try:
            summary = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "price_prediction": {"enabled": self.config["enable_price_prediction"]},
                    "market_classifier": {"enabled": self.config["enable_market_classifier"]},
                    "parameter_optimizer": {"enabled": self.config["enable_parameter_optimizer"]},
                    "anomaly_detector": {"enabled": self.config["enable_anomaly_detector"]}
                }
            }
            
            # Add price prediction
            if self.config["enable_price_prediction"]:
                prediction = self.get_price_prediction(symbol, timeframe)
                if "error" not in prediction:
                    summary["components"]["price_prediction"]["prediction"] = prediction["prediction"]
                    summary["components"]["price_prediction"]["confidence"] = prediction["confidence"]
                    summary["components"]["price_prediction"]["target_price"] = prediction.get("target_price")
                    
                    # Add model performance if available
                    key = f"{symbol}_{timeframe}"
                    if key in self.prediction_performance:
                        perf = self.prediction_performance[key]
                        if perf["total"] > 0:
                            summary["components"]["price_prediction"]["accuracy"] = perf["accuracy"]
                            summary["components"]["price_prediction"]["predictions_verified"] = perf["total"]
            
            # Add market condition
            if self.config["enable_market_classifier"] and self.market_classifier:
                condition = self.get_market_condition(symbol, timeframe)
                if "error" not in condition:
                    summary["components"]["market_classifier"]["condition"] = condition["condition"]
                    summary["components"]["market_classifier"]["probability"] = condition.get("probability", 0.0)
                    summary["components"]["market_classifier"]["secondary_conditions"] = condition.get("secondary_conditions", [])
            
            # Add parameter optimizer info if available
            if self.config["enable_parameter_optimizer"] and self.parameter_optimizer:
                summary["components"]["parameter_optimizer"]["strategies"] = self.parameter_optimizer.get_available_strategies()
                
                # Add current condition parameters if market classifier is enabled
                if self.config["enable_market_classifier"] and self.last_market_condition:
                    condition = self.last_market_condition["condition"]
                    strategies = summary["components"]["parameter_optimizer"]["strategies"]
                    
                    if strategies:
                        strategy_params = {}
                        for strategy in strategies:
                            params = self.parameter_optimizer.get_optimal_parameters(
                                strategy_name=strategy,
                                market_condition=condition
                            )
                            if params:
                                strategy_params[strategy] = params
                                
                        summary["components"]["parameter_optimizer"]["current_parameters"] = strategy_params
            
            # Add anomaly detector info
            if self.config["enable_anomaly_detector"] and self.anomaly_detector:
                # Detect new anomalies
                anomaly_result = self.detect_market_anomalies(symbol, timeframe)
                
                if "error" not in anomaly_result:
                    summary["components"]["anomaly_detector"]["anomalies_detected"] = anomaly_result["anomalies_detected"]
                    
                    if anomaly_result["anomalies_detected"] > 0:
                        summary["components"]["anomaly_detector"]["recent_anomalies"] = anomaly_result["recent_anomalies"]
                        summary["components"]["anomaly_detector"]["summary"] = anomaly_result["summary"]
                        
                # Add total anomalies tracked
                summary["components"]["anomaly_detector"]["total_tracked"] = len(self.detected_anomalies)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting ML summary: {e}")
            return {"error": str(e)}
    
    def save_all_models(self):
        """Save all trained models to disk."""
        try:
            # Save price prediction models
            for symbol, timeframe_models in self.price_prediction_models.items():
                for timeframe, model in timeframe_models.items():
                    if hasattr(model, 'model') and model.model is not None:
                        model_path = os.path.join(self.models_dir, f"price_prediction_{symbol}_{timeframe}.joblib")
                        model.save_model(model_path)
                        logger.info(f"Saved price prediction model to {model_path}")
            
            # Save market classifier
            if self.market_classifier and hasattr(self.market_classifier, 'model') and self.market_classifier.model is not None:
                model_path = os.path.join(self.models_dir, "market_classifier.joblib")
                self.market_classifier.save_model(model_path)
                logger.info(f"Saved market classifier model to {model_path}")
                
            # Save parameter optimizer
            if self.parameter_optimizer and hasattr(self.parameter_optimizer, 'model'):
                model_path = os.path.join(self.models_dir, "parameter_optimizer.joblib")
                self.parameter_optimizer.save_model(model_path)
                logger.info(f"Saved parameter optimizer model to {model_path}")
                
            # Save anomaly detector
            if self.anomaly_detector and hasattr(self.anomaly_detector, 'model') and self.anomaly_detector.model is not None:
                model_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
                self.anomaly_detector.save_model(model_path)
                logger.info(f"Saved anomaly detector model to {model_path}")
                
            logger.info("All models saved successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False 