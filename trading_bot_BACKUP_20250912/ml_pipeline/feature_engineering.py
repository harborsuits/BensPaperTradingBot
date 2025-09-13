"""
Feature Engineering Framework

This module provides feature engineering capabilities adapted from
intelligent-trading-bot to enhance trading strategy performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class FeatureEngineeringFramework:
    """Framework for generating and managing trading features"""
    
    def __init__(self, config=None):
        """
        Initialize the feature engineering framework
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.feature_registry = {}
        self.feature_dependencies = {}
        self._register_default_features()
        logger.info("Feature Engineering Framework initialized")
    
    def _register_default_features(self):
        """Register the default technical indicator features"""
        # Price-based features
        self.register_feature("log_return", self._calculate_log_return, ["close"])
        self.register_feature("return", self._calculate_return, ["close"])
        self.register_feature("normalized_price", self._calculate_normalized_price, ["close"])
        
        # Volume-based features
        self.register_feature("volume_change", self._calculate_volume_change, ["volume"])
        self.register_feature("volume_ma_ratio", self._calculate_volume_ma_ratio, ["volume"])
        
        # Volatility features
        self.register_feature("atr", self._calculate_atr, ["high", "low", "close"])
        self.register_feature("bollinger_upper", self._calculate_bollinger_upper, ["close"])
        self.register_feature("bollinger_lower", self._calculate_bollinger_lower, ["close"])
        self.register_feature("bollinger_width", self._calculate_bollinger_width, ["bollinger_upper", "bollinger_lower"])
        
        # Momentum indicators
        self.register_feature("rsi", self._calculate_rsi, ["close"])
        self.register_feature("macd", self._calculate_macd, ["close"])
        self.register_feature("macd_signal", self._calculate_macd_signal, ["macd"])
        self.register_feature("macd_histogram", self._calculate_macd_histogram, ["macd", "macd_signal"])
        
        # Trend indicators
        self.register_feature("sma_20", self._calculate_sma, ["close"], {"window": 20})
        self.register_feature("sma_50", self._calculate_sma, ["close"], {"window": 50})
        self.register_feature("sma_200", self._calculate_sma, ["close"], {"window": 200})
        self.register_feature("ema_12", self._calculate_ema, ["close"], {"window": 12})
        self.register_feature("ema_26", self._calculate_ema, ["close"], {"window": 26})
        
        # Advanced features
        self.register_feature("stochastic_k", self._calculate_stoch_k, ["high", "low", "close"])
        self.register_feature("stochastic_d", self._calculate_stoch_d, ["stochastic_k"])
        self.register_feature("adx", self._calculate_adx, ["high", "low", "close"])
    
    def register_feature(self, name: str, calculation_function, dependencies: List[str], params: Dict[str, Any] = None):
        """
        Register a new feature with the framework
        
        Args:
            name: Name of the feature
            calculation_function: Function that calculates the feature
            dependencies: List of feature names that this feature depends on
            params: Optional parameters for the calculation function
        """
        self.feature_registry[name] = calculation_function
        self.feature_dependencies[name] = {
            "dependencies": dependencies,
            "params": params or {}
        }
        logger.debug(f"Registered feature: {name}")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all registered features for the given data
        
        Args:
            data: DataFrame with at least OHLCV price data
            
        Returns:
            DataFrame with all generated features
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Generate all registered features
        for feature_name in self._get_feature_generation_order():
            if feature_name in df.columns:
                continue  # Feature already exists
                
            # Get dependencies and parameters
            deps = self.feature_dependencies[feature_name]["dependencies"]
            params = self.feature_dependencies[feature_name]["params"]
            
            # Check if all dependencies are available
            if not all(dep in df.columns for dep in deps):
                missing = [dep for dep in deps if dep not in df.columns]
                logger.warning(f"Cannot generate {feature_name}, missing dependencies: {missing}")
                continue
                
            # Calculate the feature
            try:
                calculation_func = self.feature_registry[feature_name]
                df[feature_name] = calculation_func(df, **params)
                logger.debug(f"Generated feature: {feature_name}")
            except Exception as e:
                logger.error(f"Error generating feature {feature_name}: {e}")
        
        return df
    
    def _get_feature_generation_order(self) -> List[str]:
        """
        Get the order in which features should be generated based on dependencies
        
        Returns:
            List of feature names in dependency order
        """
        # Simple topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(feature):
            if feature in temp_visited:
                raise ValueError(f"Circular dependency detected for {feature}")
            if feature in visited:
                return
                
            temp_visited.add(feature)
            
            deps = self.feature_dependencies.get(feature, {}).get("dependencies", [])
            for dep in deps:
                if dep in self.feature_dependencies:
                    visit(dep)
            
            temp_visited.remove(feature)
            visited.add(feature)
            order.append(feature)
        
        for feature in self.feature_dependencies:
            if feature not in visited:
                visit(feature)
                
        return order
    
    # Feature calculation methods
    
    def _calculate_log_return(self, df: pd.DataFrame) -> pd.Series:
        """Calculate log return"""
        return np.log(df["close"] / df["close"].shift(1))
    
    def _calculate_return(self, df: pd.DataFrame) -> pd.Series:
        """Calculate percentage return"""
        return df["close"].pct_change()
    
    def _calculate_normalized_price(self, df: pd.DataFrame) -> pd.Series:
        """Normalize price to the first value"""
        return df["close"] / df["close"].iloc[0]
    
    def _calculate_volume_change(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume change percentage"""
        return df["volume"].pct_change()
    
    def _calculate_volume_ma_ratio(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate ratio of volume to its moving average"""
        vol_ma = df["volume"].rolling(window=window).mean()
        return df["volume"] / vol_ma
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def _calculate_bollinger_upper(self, df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate upper Bollinger Band"""
        sma = df["close"].rolling(window=window).mean()
        std_dev = df["close"].rolling(window=window).std()
        return sma + (std_dev * std)
    
    def _calculate_bollinger_lower(self, df: pd.DataFrame, window: int = 20, std: float = 2.0) -> pd.Series:
        """Calculate lower Bollinger Band"""
        sma = df["close"].rolling(window=window).mean()
        std_dev = df["close"].rolling(window=window).std()
        return sma - (std_dev * std)
    
    def _calculate_bollinger_width(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Band width"""
        return (df["bollinger_upper"] - df["bollinger_lower"]) / df["close"]
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD line"""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow
    
    def _calculate_macd_signal(self, df: pd.DataFrame, signal: int = 9) -> pd.Series:
        """Calculate MACD signal line"""
        return df["macd"].ewm(span=signal, adjust=False).mean()
    
    def _calculate_macd_histogram(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD histogram"""
        return df["macd"] - df["macd_signal"]
    
    def _calculate_sma(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df["close"].rolling(window=window).mean()
    
    def _calculate_ema(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df["close"].ewm(span=window, adjust=False).mean()
    
    def _calculate_stoch_k(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Stochastic Oscillator %K"""
        low_min = df["low"].rolling(window=window).min()
        high_max = df["high"].rolling(window=window).max()
        return 100 * ((df["close"] - low_min) / (high_max - low_min))
    
    def _calculate_stoch_d(self, df: pd.DataFrame, window: int = 3) -> pd.Series:
        """Calculate Stochastic Oscillator %D"""
        return df["stochastic_k"].rolling(window=window).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        # +DM and -DM
        high_diff = df["high"].diff()
        low_diff = df["low"].diff().mul(-1)
        
        plus_dm = ((high_diff > 0) & (high_diff > low_diff)) * high_diff
        minus_dm = ((low_diff > 0) & (low_diff > high_diff)) * low_diff
        
        # True Range
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.DataFrame({
            "hl": high_low,
            "hc": high_close, 
            "lc": low_close
        }).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=window).mean()
        
        # Smooth +DM and -DM
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # Directional movement index
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        
        # Average directional index
        adx = dx.rolling(window=window).mean()
        return adx
