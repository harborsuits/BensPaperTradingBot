"""
Regime Features Calculator

This module calculates various features for market regime detection, including trend strength,
volatility metrics, momentum indicators, and range-bound indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class FeatureType(str, Enum):
    """Types of features used for regime detection."""
    TREND = "trend"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    RANGE = "range"
    TECHNICAL = "technical"
    STATISTICAL = "statistical"

class RegimeFeaturesCalculator:
    """
    Calculates features for market regime detection.
    
    Features include:
    - Trend strength (ADX, linear regression slope)
    - Volatility metrics (ATR, historical volatility)
    - Momentum indicators (RSI, MACD)
    - Range-bound indicators (Bollinger %B, Keltner channels)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize features calculator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Configure which features to calculate
        self.enabled_features = self.config.get("enabled_features", {
            FeatureType.TREND: True,
            FeatureType.VOLATILITY: True,
            FeatureType.MOMENTUM: True,
            FeatureType.RANGE: True,
            FeatureType.TECHNICAL: True,
            FeatureType.STATISTICAL: True
        })
        
        # Feature parameters
        self.adx_period = self.config.get("adx_period", 14)
        self.atr_period = self.config.get("atr_period", 14)
        self.volatility_period = self.config.get("volatility_period", 20)
        self.regression_period = self.config.get("regression_period", 20)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.bollinger_period = self.config.get("bollinger_period", 20)
        self.bollinger_std = self.config.get("bollinger_std", 2.0)
        self.keltner_period = self.config.get("keltner_period", 20)
        self.keltner_atr_mult = self.config.get("keltner_atr_mult", 1.5)
        self.macd_fast = self.config.get("macd_fast", 12)
        self.macd_slow = self.config.get("macd_slow", 26)
        self.macd_signal = self.config.get("macd_signal", 9)
        self.autocorr_lag = self.config.get("autocorr_lag", 10)
        
        logger.info("Regime Features Calculator initialized")
    
    def calculate(self, symbol: str, timeframe: str, 
                 price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate regime detection features for a symbol and timeframe.
        
        Args:
            symbol: Symbol to calculate features for
            timeframe: Timeframe to calculate features for
            price_data: DataFrame with price data
            
        Returns:
            Dict of calculated feature values
        """
        features = {}
        
        try:
            # Ensure data is sufficient
            if len(price_data) < 50:
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(price_data)} bars")
                return features
            
            # Calculate trend features
            if self.enabled_features.get(FeatureType.TREND, True):
                trend_features = self._calculate_trend_features(price_data)
                features.update(trend_features)
                
            # Calculate volatility features
            if self.enabled_features.get(FeatureType.VOLATILITY, True):
                volatility_features = self._calculate_volatility_features(price_data)
                features.update(volatility_features)
                
            # Calculate momentum features
            if self.enabled_features.get(FeatureType.MOMENTUM, True):
                momentum_features = self._calculate_momentum_features(price_data)
                features.update(momentum_features)
                
            # Calculate range features
            if self.enabled_features.get(FeatureType.RANGE, True):
                range_features = self._calculate_range_features(price_data)
                features.update(range_features)
                
            # Calculate technical features
            if self.enabled_features.get(FeatureType.TECHNICAL, True):
                technical_features = self._calculate_technical_features(price_data)
                features.update(technical_features)
                
            # Calculate statistical features
            if self.enabled_features.get(FeatureType.STATISTICAL, True):
                stat_features = self._calculate_statistical_features(price_data)
                features.update(stat_features)
                
            return features
            
        except Exception as e:
            logger.error(f"Error calculating features for {symbol} {timeframe}: {str(e)}")
            return features
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trend-related features.
        
        Args:
            df: Price data
            
        Returns:
            Dict of trend features
        """
        features = {}
        
        try:
            # Calculate ADX (Average Directional Index)
            features["adx"] = self._calculate_adx(df, self.adx_period)
            
            # Calculate linear regression slope
            features["regression_slope"] = self._calculate_linear_regression_slope(
                df["close"].values, self.regression_period
            )
            
            # Calculate directional slope (positive or negative)
            features["directional_slope"] = 1.0 if features["regression_slope"] > 0 else -1.0
            
            # Calculate slope strength (absolute value normalized)
            features["slope_strength"] = min(abs(features["regression_slope"]) * 100, 1.0)
            
            # Calculate price relative to moving averages
            features["price_vs_ma50"] = df["close"].iloc[-1] / df["close"].rolling(50).mean().iloc[-1] - 1
            features["price_vs_ma200"] = df["close"].iloc[-1] / df["close"].rolling(200).mean().iloc[-1] - 1
            
            # Calculate moving average crossovers
            ma20 = df["close"].rolling(20).mean()
            ma50 = df["close"].rolling(50).mean()
            features["ma_crossover"] = 1.0 if ma20.iloc[-1] > ma50.iloc[-1] else -1.0
            
            # Calculate directional movement
            up_moves = df["high"] - df["high"].shift(1)
            down_moves = df["low"].shift(1) - df["low"]
            up_moves[up_moves < 0] = 0
            down_moves[down_moves < 0] = 0
            
            features["directional_movement"] = up_moves.rolling(14).sum().iloc[-1] / (
                up_moves.rolling(14).sum().iloc[-1] + down_moves.rolling(14).sum().iloc[-1]
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating trend features: {str(e)}")
            return features
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate volatility-related features.
        
        Args:
            df: Price data
            
        Returns:
            Dict of volatility features
        """
        features = {}
        
        try:
            # Calculate ATR (Average True Range)
            atr = self._calculate_atr(df, self.atr_period)
            features["atr"] = atr
            
            # Calculate normalized ATR (as percentage of price)
            features["atr_pct"] = atr / df["close"].iloc[-1]
            
            # Calculate historical volatility
            features["hist_volatility"] = self._calculate_historical_volatility(
                df["close"].values, self.volatility_period
            )
            
            # Calculate volatility ratio (recent vs longer-term)
            vol_recent = self._calculate_historical_volatility(df["close"].values, 10)
            vol_longer = self._calculate_historical_volatility(df["close"].values, 30)
            features["volatility_ratio"] = vol_recent / vol_longer if vol_longer > 0 else 1.0
            
            # Calculate Bollinger Band width
            bb_upper, bb_lower = self._calculate_bollinger_bands(
                df["close"].values, self.bollinger_period, self.bollinger_std
            )
            features["bb_width"] = (bb_upper - bb_lower) / df["close"].iloc[-1]
            
            # Calculate high-low range
            features["hl_range_pct"] = (df["high"] - df["low"]).rolling(10).mean().iloc[-1] / df["close"].iloc[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating volatility features: {str(e)}")
            return features
    
    def _calculate_momentum_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate momentum-related features.
        
        Args:
            df: Price data
            
        Returns:
            Dict of momentum features
        """
        features = {}
        
        try:
            # Calculate RSI
            features["rsi"] = self._calculate_rsi(df["close"].values, self.rsi_period)
            
            # Calculate MACD
            macd, signal, hist = self._calculate_macd(
                df["close"].values, self.macd_fast, self.macd_slow, self.macd_signal
            )
            features["macd"] = macd
            features["macd_signal"] = signal
            features["macd_hist"] = hist
            
            # Calculate rate of change
            features["roc_5"] = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
            features["roc_10"] = (df["close"].iloc[-1] / df["close"].iloc[-11] - 1) * 100
            features["roc_20"] = (df["close"].iloc[-1] / df["close"].iloc[-21] - 1) * 100
            
            # Calculate price velocity and acceleration
            close = df["close"].values
            velocity = close[-1] - close[-2]
            acceleration = (close[-1] - close[-2]) - (close[-2] - close[-3])
            features["price_velocity"] = velocity / close[-2]  # Normalized
            features["price_acceleration"] = acceleration / close[-3]  # Normalized
            
            # Calculate stochastic oscillator
            features["stochastic_k"] = self._calculate_stochastic_k(
                df["high"].values, df["low"].values, df["close"].values, 14
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating momentum features: {str(e)}")
            return features
    
    def _calculate_range_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate range-related features.
        
        Args:
            df: Price data
            
        Returns:
            Dict of range features
        """
        features = {}
        
        try:
            # Calculate Bollinger %B
            bb_upper, bb_lower = self._calculate_bollinger_bands(
                df["close"].values, self.bollinger_period, self.bollinger_std
            )
            bb_mid = (bb_upper + bb_lower) / 2
            features["bollinger_b"] = (df["close"].iloc[-1] - bb_lower) / (bb_upper - bb_lower)
            
            # Calculate Keltner Channel %K
            keltner_upper, keltner_lower = self._calculate_keltner_channels(
                df, self.keltner_period, self.keltner_atr_mult
            )
            keltner_mid = (keltner_upper + keltner_lower) / 2
            features["keltner_k"] = (df["close"].iloc[-1] - keltner_lower) / (keltner_upper - keltner_lower)
            
            # Calculate distance from moving average
            ma20 = df["close"].rolling(20).mean().iloc[-1]
            features["ma_distance"] = (df["close"].iloc[-1] - ma20) / ma20
            
            # Calculate range oscillator
            high_max = df["high"].rolling(20).max().iloc[-1]
            low_min = df["low"].rolling(20).min().iloc[-1]
            range_width = high_max - low_min
            features["range_position"] = (df["close"].iloc[-1] - low_min) / range_width if range_width > 0 else 0.5
            
            # Calculate breakout potential
            upper_touches = np.sum((df["high"].rolling(2).max() >= bb_upper).iloc[-10:])
            lower_touches = np.sum((df["low"].rolling(2).min() <= bb_lower).iloc[-10:])
            features["upper_band_touches"] = upper_touches / 10.0
            features["lower_band_touches"] = lower_touches / 10.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating range features: {str(e)}")
            return features
    
    def _calculate_technical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate additional technical features.
        
        Args:
            df: Price data
            
        Returns:
            Dict of technical features
        """
        features = {}
        
        try:
            # Calculate OBV (On-Balance Volume)
            if "volume" in df.columns:
                features["obv_trend"] = self._calculate_obv_trend(df)
            
            # Calculate Chaikin Money Flow
            if "volume" in df.columns:
                features["cmf"] = self._calculate_chaikin_money_flow(df, 20)
            
            # Calculate moving average convergence/divergence
            ma20 = df["close"].rolling(20).mean()
            ma50 = df["close"].rolling(50).mean()
            ma200 = df["close"].rolling(200).mean()
            
            features["ma_convergence"] = (ma20.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1]
            
            # Calculate dynamic support/resistance levels
            pivot = (df["high"].iloc[-2] + df["low"].iloc[-2] + df["close"].iloc[-2]) / 3
            r1 = 2 * pivot - df["low"].iloc[-2]
            s1 = 2 * pivot - df["high"].iloc[-2]
            
            features["dist_to_resistance"] = (r1 - df["close"].iloc[-1]) / df["close"].iloc[-1]
            features["dist_to_support"] = (df["close"].iloc[-1] - s1) / df["close"].iloc[-1]
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {str(e)}")
            return features
    
    def _calculate_statistical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate statistical features.
        
        Args:
            df: Price data
            
        Returns:
            Dict of statistical features
        """
        features = {}
        
        try:
            # Calculate autocorrelation
            close_returns = df["close"].pct_change().dropna()
            features["autocorrelation"] = self._calculate_autocorrelation(
                close_returns.values, self.autocorr_lag
            )
            
            # Calculate variance ratio
            features["variance_ratio"] = self._calculate_variance_ratio(close_returns.values, 1, 5)
            
            # Calculate mean reversion score
            features["mean_reversion"] = self._calculate_mean_reversion_score(df["close"].values)
            
            # Calculate kurtosis
            if len(close_returns) > 30:
                features["returns_kurtosis"] = close_returns.rolling(30).kurt().iloc[-1]
            
            # Calculate skewness
            if len(close_returns) > 30:
                features["returns_skew"] = close_returns.rolling(30).skew().iloc[-1]
                
            # Calculate Hurst exponent
            features["hurst_exponent"] = self._calculate_hurst_exponent(df["close"].values)
            
            return features
            
        except Exception as e:
            logger.error(f"Error calculating statistical features: {str(e)}")
            return features
    
    # ===== Feature calculation methods =====
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        try:
            # Simple implementation of ADX
            high = df["high"]
            low = df["low"]
            close = df["close"]
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame([tr1, tr2, tr3]).max()
            
            atr = tr.rolling(period).mean()
            
            plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean()
            
            return adx.iloc[-1]
        except:
            return 0.0
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            return atr.iloc[-1]
        except:
            return 0.0
    
    def _calculate_linear_regression_slope(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate linear regression slope of price series."""
        try:
            if len(prices) < period:
                return 0.0
                
            x = np.arange(period)
            y = prices[-period:]
            
            slope, _ = np.polyfit(x, y, 1)
            
            # Normalize by average price
            slope = slope / np.mean(y)
            
            return slope
        except:
            return 0.0
    
    def _calculate_historical_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate historical volatility (standard deviation of returns)."""
        try:
            if len(prices) < period + 1:
                return 0.0
                
            returns = np.diff(np.log(prices))[-period:]
            return np.std(returns) * np.sqrt(252)  # Annualized
        except:
            return 0.0
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        try:
            if len(prices) < period + 1:
                return 50.0
                
            deltas = np.diff(prices)[-period-1:]
            gain = np.where(deltas > 0, deltas, 0)
            loss = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])
            
            for i in range(period, len(deltas)):
                avg_gain = (avg_gain * (period - 1) + gain[i]) / period
                avg_loss = (avg_loss * (period - 1) + loss[i]) / period
            
            if avg_loss == 0:
                return 100.0
                
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        except:
            return 50.0
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float]:
        """Calculate Bollinger Bands."""
        try:
            if len(prices) < period:
                return prices[-1], prices[-1]
                
            ma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            upper = ma + std_dev * std
            lower = ma - std_dev * std
            
            return upper, lower
        except:
            return prices[-1], prices[-1]
    
    def _calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5) -> Tuple[float, float]:
        """Calculate Keltner Channels."""
        try:
            if len(df) < period:
                return df["close"].iloc[-1], df["close"].iloc[-1]
                
            typical_price = (df["high"] + df["low"] + df["close"]) / 3
            ma = typical_price.rolling(period).mean().iloc[-1]
            atr = self._calculate_atr(df, period)
            
            upper = ma + atr_mult * atr
            lower = ma - atr_mult * atr
            
            return upper, lower
        except:
            return df["close"].iloc[-1], df["close"].iloc[-1]
    
    def _calculate_macd(self, prices: np.ndarray, fast_period: int = 12, 
                       slow_period: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD, signal line, and histogram."""
        try:
            if len(prices) < slow_period + signal_period:
                return 0.0, 0.0, 0.0
                
            # Function to calculate EMA
            def ema(data, period):
                alpha = 2 / (period + 1)
                ema_values = np.zeros_like(data)
                ema_values[0] = data[0]
                for i in range(1, len(data)):
                    ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
                return ema_values
            
            fast_ema = ema(prices, fast_period)
            slow_ema = ema(prices, slow_period)
            
            macd_line = fast_ema - slow_ema
            signal_line = ema(macd_line, signal_period)
            histogram = macd_line - signal_line
            
            return macd_line[-1], signal_line[-1], histogram[-1]
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_stochastic_k(self, high: np.ndarray, low: np.ndarray, 
                              close: np.ndarray, period: int = 14) -> float:
        """Calculate Stochastic %K."""
        try:
            if len(high) < period:
                return 50.0
                
            recent_high = np.max(high[-period:])
            recent_low = np.min(low[-period:])
            
            if recent_high == recent_low:
                return 50.0
                
            k = 100 * (close[-1] - recent_low) / (recent_high - recent_low)
            return k
        except:
            return 50.0
    
    def _calculate_obv_trend(self, df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume trend."""
        try:
            close = df["close"]
            volume = df["volume"]
            
            obv = np.zeros(len(close))
            
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close[i] < close[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            
            # Calculate OBV slope over last 20 periods
            obv_slope = self._calculate_linear_regression_slope(obv[-20:], 20)
            
            return obv_slope
        except:
            return 0.0
    
    def _calculate_chaikin_money_flow(self, df: pd.DataFrame, period: int = 20) -> float:
        """Calculate Chaikin Money Flow."""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]
            volume = df["volume"]
            
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow_volume = money_flow_multiplier * volume
            
            cmf = money_flow_volume.rolling(period).sum() / volume.rolling(period).sum()
            
            return cmf.iloc[-1]
        except:
            return 0.0
    
    def _calculate_autocorrelation(self, returns: np.ndarray, lag: int = 10) -> float:
        """Calculate autocorrelation of returns."""
        try:
            if len(returns) <= lag:
                return 0.0
                
            # Pearson correlation between series and lagged series
            n = len(returns) - lag
            series1 = returns[lag:lag+n]
            series2 = returns[:n]
            
            return np.corrcoef(series1, series2)[0, 1]
        except:
            return 0.0
    
    def _calculate_variance_ratio(self, returns: np.ndarray, short_period: int = 1, 
                                long_period: int = 5) -> float:
        """Calculate variance ratio for random walk testing."""
        try:
            if len(returns) < long_period * 20:
                return 1.0
                
            # Variance of returns over different periods
            var_short = np.var(returns)
            
            # Create long-period returns
            long_returns = np.zeros(len(returns) // long_period)
            for i in range(len(long_returns)):
                long_returns[i] = np.sum(returns[i*long_period:(i+1)*long_period])
            
            var_long = np.var(long_returns) / long_period
            
            if var_short == 0:
                return 1.0
                
            # For random walk, ratio should be close to 1
            # > 1 indicates trend, < 1 indicates mean reversion
            return var_long / var_short
        except:
            return 1.0
    
    def _calculate_mean_reversion_score(self, prices: np.ndarray) -> float:
        """Calculate mean reversion score."""
        try:
            if len(prices) < 60:
                return 0.5
                
            # Calculate z-score of price relative to 30-day moving average
            ma30 = np.mean(prices[-30:])
            std30 = np.std(prices[-30:])
            
            if std30 == 0:
                return 0.5
                
            z_score = (prices[-1] - ma30) / std30
            
            # Normalize to 0-1 range (higher means more mean reversion potential)
            mean_reversion_score = 1 / (1 + np.exp(-(-z_score)))  # Sigmoid function
            
            return mean_reversion_score
        except:
            return 0.5
    
    def _calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent to determine if series is trending, mean-reverting, or random."""
        try:
            if len(prices) < 100:
                return 0.5
                
            # Calculate log returns
            returns = np.diff(np.log(prices))
            
            # Calculate Hurst exponent
            tau = [10, 20, 30, 40, 50]
            lagvec = []
            
            for lag in tau:
                # Calculate price difference
                pp = np.zeros(len(returns) - lag)
                for i in range(len(pp)):
                    pp[i] = np.sum(returns[i:i+lag])
                
                # Calculate range
                pmax = np.max(pp)
                pmin = np.min(pp)
                prange = pmax - pmin
                
                # Calculate standard deviation
                pstd = np.std(pp)
                
                # Calculate rescaled range
                if pstd > 0:
                    rs = prange / pstd
                else:
                    rs = 0
                
                lagvec.append([np.log(lag), np.log(rs)])
            
            # Linear regression to estimate Hurst exponent
            x = np.array([item[0] for item in lagvec])
            y = np.array([item[1] for item in lagvec])
            
            hurst_exponent, _ = np.polyfit(x, y, 1)
            
            return hurst_exponent
        except:
            return 0.5  # Default to random walk
