#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Analysis Module for Volatility Strategies

This module handles all market analysis functions for volatility-based options strategies,
including historical and implied volatility calculations, volatility surface analysis,
and event detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class VolatilityAnalyzer:
    """
    Analyzes market volatility for options strategies.
    
    This class provides efficient, vectorized calculations for:
    - Historical volatility (multiple timeframes)
    - Implied volatility analysis
    - Volatility term structure
    - Volatility skew
    - Event-based volatility forecasting
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None, cache_size: int = 128):
        """
        Initialize the volatility analyzer.
        
        Args:
            lookback_periods: Dictionary mapping period names to number of days
                e.g. {'short': 10, 'medium': 30, 'long': 60}
            cache_size: Size of LRU cache for expensive calculations
        """
        self.lookback_periods = lookback_periods or {
            'short': 10,
            'medium': 30, 
            'long': 60,
            'extended': 252  # ~1 trading year
        }
        
        # Set up caching for expensive calculations
        self.cache_size = cache_size
        self._configure_caching()
        
        # Volatility history tracking
        self.historical_vol_data = {}
        self.implied_vol_data = {}
        
    def _configure_caching(self) -> None:
        """Configure LRU caching for expensive calculations"""
        # Apply LRU cache decorators to methods that benefit from caching
        self.calculate_historical_volatility = lru_cache(maxsize=self.cache_size)(
            self._calculate_historical_volatility
        )
        
    def _calculate_historical_volatility(
        self, 
        symbol: str,
        period: int = 20,
        price_tuple: tuple = None,
        annualize: bool = True
    ) -> float:
        """
        Private implementation of historical volatility calculation.
        This method should be called through the cached public method.
        
        Args:
            symbol: Symbol to identify the cache key
            period: Number of trading days to calculate volatility over
            price_tuple: Tuple of price data (must be hashable for caching)
            annualize: Whether to annualize the volatility
            
        Returns:
            Historical volatility value
        """
        if not price_tuple or len(price_tuple) < period + 1:
            return 0.0
            
        # Convert tuple to numpy array for calculations
        prices = np.array(price_tuple)
        
        # Calculate log returns
        log_returns = np.diff(np.log(prices))
        
        # Calculate standard deviation
        vol = np.std(log_returns, ddof=1)
        
        # Annualize if requested
        if annualize:
            vol = vol * np.sqrt(252)
            
        return vol
    
    def calculate_historical_volatility(
        self, 
        price_data: pd.DataFrame,
        period: int = 20,
        annualize: bool = True,
        return_type: str = 'latest'
    ) -> Union[float, pd.Series]:
        """
        Calculate historical volatility using vectorized operations.
        
        Args:
            price_data: DataFrame with 'close' price column
            period: Number of trading days to calculate volatility over
            annualize: Whether to annualize the volatility
            return_type: 'latest' for most recent value, 'series' for full series
            
        Returns:
            Historical volatility (annualized by default)
        """
        if len(price_data) < period + 1:
            logger.warning(f"Insufficient data for {period}-day volatility calculation")
            return 0.0 if return_type == 'latest' else pd.Series()
            
        # Calculate log returns (more accurate than pct_change for volatility)
        log_returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
        
        # Calculate rolling standard deviation
        rolling_std = log_returns.rolling(window=period).std()
        
        # Annualize if requested (√252 is the typical annualization factor for daily data)
        if annualize:
            rolling_std = rolling_std * np.sqrt(252)
            
        # Store in historical data
        symbol = getattr(price_data, 'name', 'unknown')
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0] if not symbol.empty else 'unknown'
            
        self.historical_vol_data[f"{symbol}_{period}d"] = rolling_std
            
        # Return based on specified return type
        if return_type == 'latest':
            return rolling_std.iloc[-1] if not rolling_std.empty else 0.0
        else:  # 'series'
            return rolling_std
    
    def calculate_volatility_percentile(
        self,
        current_volatility: float,
        historical_volatility: pd.Series
    ) -> float:
        """
        Calculate the percentile of current volatility compared to historical data.
        
        Args:
            current_volatility: Current volatility value
            historical_volatility: Series of historical volatility values
                
        Returns:
            Volatility percentile (0-100)
        """
        if historical_volatility.empty:
            return 50.0  # Default to median if no data
            
        # Calculate percentile using numpy for efficiency
        hist_vol_array = historical_volatility.dropna().values
        
        if len(hist_vol_array) == 0:
            return 50.0
            
        # Calculate percentile
        lower_count = (hist_vol_array < current_volatility).sum()
        return (lower_count / len(hist_vol_array)) * 100
    
    def calculate_iv_rank(
        self,
        current_iv: float,
        historical_iv: pd.Series
    ) -> float:
        """
        Calculate IV Rank (current IV relative to 52-week range).
        
        IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100
        
        Args:
            current_iv: Current implied volatility
            historical_iv: Series of historical implied volatility
                
        Returns:
            IV Rank (0-100)
        """
        if historical_iv.empty:
            return 50.0
            
        iv_min = historical_iv.min()
        iv_max = historical_iv.max()
        
        if iv_max == iv_min:  # Avoid division by zero
            return 50.0
            
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        return min(max(iv_rank, 0), 100)  # Ensure result is between 0-100
    
    def detect_volatility_regime(
        self,
        historical_volatility: pd.Series,
        vix_data: pd.Series = None,
        lookback_days: int = 60
    ) -> Dict[str, Any]:
        """
        Detect the current volatility regime.
        
        Args:
            historical_volatility: Series of historical volatility values
            vix_data: Series of VIX data (optional)
            lookback_days: Number of days to look back
                
        Returns:
            Dictionary with volatility regime information
        """
        if historical_volatility.empty:
            return {'regime': 'unknown', 'confidence': 0.0}
            
        # Get recent volatility data
        recent_vol = historical_volatility.tail(lookback_days)
        
        if recent_vol.empty:
            return {'regime': 'unknown', 'confidence': 0.0}
            
        # Calculate key metrics
        current_vol = recent_vol.iloc[-1]
        mean_vol = recent_vol.mean()
        std_vol = recent_vol.std()
        
        if std_vol == 0:
            return {'regime': 'neutral', 'confidence': 0.5}
            
        # Calculate z-score for current volatility
        z_score = (current_vol - mean_vol) / std_vol
        
        # Determine regime
        if z_score > 1.0:
            regime = 'high_volatility'
            confidence = min(0.5 + abs(z_score) / 6, 0.95)  # Cap at 0.95
        elif z_score < -1.0:
            regime = 'low_volatility'
            confidence = min(0.5 + abs(z_score) / 6, 0.95)
        else:
            regime = 'neutral'
            confidence = 0.5 - abs(z_score) / 2  # Lower confidence closer to boundaries
            
        # Incorporate VIX if available
        if vix_data is not None and not vix_data.empty:
            current_vix = vix_data.iloc[-1]
            
            # VIX-based regime thresholds
            if current_vix > 25:
                vix_regime = 'high_volatility'
            elif current_vix < 15:
                vix_regime = 'low_volatility'
            else:
                vix_regime = 'neutral'
                
            # Adjust confidence if VIX and HV regimes agree
            if vix_regime == regime:
                confidence = min(confidence + 0.1, 0.95)
            else:
                confidence = max(confidence - 0.1, 0.05)
                
        return {
            'regime': regime,
            'confidence': confidence,
            'z_score': z_score,
            'current_vol': current_vol,
            'mean_vol': mean_vol,
            'std_vol': std_vol
        }
    
    def calculate_iv_hv_spread(
        self,
        implied_volatility: Union[float, pd.Series],
        historical_volatility: Union[float, pd.Series]
    ) -> Union[float, pd.Series]:
        """
        Calculate the spread between implied and historical volatility.
        
        A positive spread indicates options are relatively expensive.
        A negative spread indicates options are relatively cheap.
        
        Args:
            implied_volatility: Current implied volatility or series
            historical_volatility: Current historical volatility or series
                
        Returns:
            IV-HV spread as float or series
        """
        # Handle both single values and series
        if isinstance(implied_volatility, (float, int)) and isinstance(historical_volatility, (float, int)):
            return implied_volatility - historical_volatility
            
        # Convert to series if needed
        if isinstance(implied_volatility, (float, int)):
            implied_volatility = pd.Series([implied_volatility] * len(historical_volatility),
                                         index=historical_volatility.index)
                                         
        if isinstance(historical_volatility, (float, int)):
            historical_volatility = pd.Series([historical_volatility] * len(implied_volatility),
                                            index=implied_volatility.index)
                                            
        # Calculate spread
        return implied_volatility - historical_volatility
