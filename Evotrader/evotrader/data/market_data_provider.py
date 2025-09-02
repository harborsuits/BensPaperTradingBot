"""Market data providers for EvoTrader."""

import abc
import random
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta


class MarketDataProvider(abc.ABC):
    """Abstract base class for market data providers."""
    
    @abc.abstractmethod
    def get_data(self, day: int) -> Dict[str, Any]:
        """
        Get market data for a specific day.
        
        Args:
            day: Simulation day (0-indexed)
            
        Returns:
            Dict containing market data per symbol
        """
        pass
    
    @abc.abstractmethod
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols.
        
        Returns:
            List of symbol strings
        """
        pass


class RandomWalkDataProvider(MarketDataProvider):
    """
    Generates synthetic market data using random walk process.
    
    This provider creates realistic-looking price series with
    volatility clustering, trends, and other market-like properties.
    """
    
    def __init__(self, 
                symbols: List[str] = None, 
                initial_prices: Dict[str, float] = None,
                seed: int = 42,
                volatility: float = 0.015,
                drift: float = 0.0001,
                mean_reversion: float = 0.001,
                correlation_matrix: Optional[np.ndarray] = None):
        """
        Initialize random walk data provider.
        
        Args:
            symbols: List of symbols to generate data for
            initial_prices: Starting prices for each symbol
            seed: Random seed for reproducibility
            volatility: Base volatility of price movements
            drift: Upward drift in price (market bias)
            mean_reversion: Strength of mean reversion
            correlation_matrix: Correlation between symbols
        """
        self.logger = logging.getLogger(__name__)
        
        # Default symbols if none provided
        if symbols is None:
            symbols = ["BTC/USD", "ETH/USD", "XRP/USD", "LTC/USD"]
        self.symbols = symbols
        
        # Default initial prices if none provided
        if initial_prices is None:
            initial_prices = {
                "BTC/USD": 30000.0,
                "ETH/USD": 2000.0,
                "XRP/USD": 0.5,
                "LTC/USD": 100.0,
            }
            # Fill any missing symbols with random prices
            for symbol in symbols:
                if symbol not in initial_prices:
                    initial_prices[symbol] = random.uniform(10.0, 1000.0)
        
        self.initial_prices = initial_prices
        self.current_prices = initial_prices.copy()
        self.seed = seed
        self.volatility = volatility
        self.drift = drift
        self.mean_reversion = mean_reversion
        
        # Price history for calculating moving averages etc.
        self.price_history = {symbol: [price] for symbol, price in initial_prices.items()}
        
        # Volatility state (for volatility clustering)
        self.current_volatility = {symbol: volatility for symbol in symbols}
        
        # Create correlation matrix if not provided
        if correlation_matrix is None:
            # Default: modest positive correlation between assets
            n_symbols = len(symbols)
            self.correlation_matrix = np.ones((n_symbols, n_symbols)) * 0.3
            np.fill_diagonal(self.correlation_matrix, 1.0)
        else:
            self.correlation_matrix = correlation_matrix
            
        # Initialize random state
        self.random_state = np.random.RandomState(seed)
        
        self.logger.info(f"Initialized random walk data provider with {len(symbols)} symbols")
        
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.symbols
        
    def get_data(self, day: int) -> Dict[str, Any]:
        """
        Generate market data for a specific day.
        
        Args:
            day: Simulation day (0-indexed)
            
        Returns:
            Dict containing market data per symbol
        """
        # Re-seed for reproducibility if needed
        if day == 0:
            self.random_state = np.random.RandomState(self.seed)
            
            # Initialize trending periods for more realistic price action
            self.trend_phases = {}
            self.trend_strengths = {}
            for symbol in self.symbols:
                # Each symbol will have its own trend cycle
                self.trend_phases[symbol] = self.random_state.randint(0, 100)  # Random starting phase
                self.trend_strengths[symbol] = self.random_state.uniform(0.0005, 0.003)  # Random trend strength
        
        # Generate correlated returns across all symbols
        correlated_returns = self._generate_correlated_returns()
        
        # Update prices for each symbol
        market_data = {}
        
        for i, symbol in enumerate(self.symbols):
            # Get current price and volatility
            current_price = self.current_prices[symbol]
            current_vol = self.current_volatility[symbol]
            
            # Update volatility (volatility clustering)
            # More dramatic volatility changes to create better trading opportunities
            vol_shock = 1.0
            if self.random_state.random() < 0.03:  # 3% chance of volatility shock
                vol_shock = self.random_state.uniform(1.5, 3.0)  # Dramatic increase
                
            vol_change = self.random_state.normal(0, 0.2) * vol_shock
            new_vol = max(0.01, current_vol * (1 + vol_change))  # Higher minimum volatility
            self.current_volatility[symbol] = new_vol
            
            # Add exaggerated trending component - sine wave with random period
            # Faster phase advance for more frequent crossovers
            self.trend_phases[symbol] = (self.trend_phases[symbol] + 3) % 360  # Move 3x faster
            trend_component = np.sin(np.radians(self.trend_phases[symbol])) * self.trend_strengths[symbol] * 4.0  # 4x stronger trends
            
            # Calculate return with drift, volatility, and trend
            price_return = correlated_returns[i] * new_vol * 2.0  # Double the volatility impact
            price_return += self.drift * 5.0  # 5x increased drift for more directional movement
            price_return += trend_component  # Add enhanced trending component
            
            # Add frequent price shocks for trading opportunities
            if self.random_state.random() < 0.10:  # 10% chance of price shock (was 2%)
                # Bigger moves up or down
                shock_direction = 1 if self.random_state.random() < 0.5 else -1
                price_return += shock_direction * self.random_state.uniform(0.05, 0.15)  # 5-15% shocks
            
            # Add mean reversion component (pull toward initial price)
            distance_from_start = current_price / self.initial_prices[symbol] - 1.0
            mean_reversion_effect = -distance_from_start * self.mean_reversion * 1.5
            price_return += mean_reversion_effect
            
            # Update price
            new_price = current_price * (1 + price_return)
            self.current_prices[symbol] = max(0.001, new_price)  # Ensure positive price
            
            # Generate realistic OHLC data
            daily_high = self.current_prices[symbol] * (1 + abs(self.random_state.normal(0, new_vol * 0.5)))
            daily_low = self.current_prices[symbol] * (1 - abs(self.random_state.normal(0, new_vol * 0.5)))
            daily_open = self.price_history[symbol][-1] if self.price_history[symbol] else self.current_prices[symbol]
            
            # Update price history with closing price
            self.price_history[symbol].append(new_price)
            
            # Add to market data
            market_data[symbol] = {
                "price": new_price,
                "open": daily_open,  # Opening price
                "high": daily_high,  # Daily high
                "low": daily_low,    # Daily low
                "timestamp": time.time(),
                "volume": random.uniform(1000, 100000),  # Higher volume range
            }
            
            # Add moving averages that strategies commonly use
            for period in [5, 8, 10, 20, 21, 50, 100, 200]:
                if len(self.price_history[symbol]) >= period:
                    market_data[symbol][f"sma_{period}"] = self._calculate_sma(symbol, period)
            
            # Add EMA indicators
            for period in [8, 12, 21, 26, 50, 200]:
                if len(self.price_history[symbol]) >= period:
                    market_data[symbol][f"ema_{period}"] = self._calculate_ema(symbol, period)
            
            # Add technical indicators if we have enough history
            if len(self.price_history[symbol]) >= 14:
                market_data[symbol]["rsi_14"] = self._calculate_rsi(symbol, 14)
                
                if len(self.price_history[symbol]) >= 20:
                    bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(symbol, 20, 2.0)
                    market_data[symbol]["bb_middle"] = bb_middle
                    market_data[symbol]["bb_upper"] = bb_upper
                    market_data[symbol]["bb_lower"] = bb_lower
                    market_data[symbol]["bb_width"] = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
                
                # Calculate price change and percent for momentum strategies
                if len(self.price_history[symbol]) >= 2:
                    prev_price = self.price_history[symbol][-2]
                    curr_price = self.price_history[symbol][-1]
                    price_change = curr_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100 if prev_price > 0 else 0
                    
                    market_data[symbol]["price_change"] = price_change
                    market_data[symbol]["price_change_pct"] = price_change_pct
        
        return market_data
        
    def _generate_correlated_returns(self) -> np.ndarray:
        """
        Generate correlated random returns based on correlation matrix.
        
        Returns:
            Array of correlated returns
        """
        # Generate uncorrelated random returns
        uncorrelated_returns = self.random_state.randn(len(self.symbols))
        
        # Apply Cholesky decomposition to get correlated returns
        L = np.linalg.cholesky(self.correlation_matrix)
        return L @ uncorrelated_returns
    
    def _calculate_sma(self, symbol: str, window: int) -> Optional[float]:
        """
        Calculate simple moving average.
        
        Args:
            symbol: Symbol to calculate for
            window: SMA window length
            
        Returns:
            SMA value or None if not enough history
        """
        history = self.price_history[symbol]
        if len(history) < window:
            return None
        return sum(history[-window:]) / window
    
    def _calculate_ema(self, symbol: str, window: int, smoothing: float = 2.0) -> Optional[float]:
        """
        Calculate Exponential Moving Average.
        
        Args:
            symbol: Symbol to calculate for
            window: EMA window length
            smoothing: Smoothing factor
            
        Returns:
            EMA value or None if not enough history
        """
        history = self.price_history[symbol]
        if len(history) < window:
            return None
            
        # Calculate SMA for first EMA value
        sma = sum(history[:window]) / window
        
        # Calculate alpha (smoothing factor)
        alpha = smoothing / (window + 1.0)
        
        # Initialize EMA with SMA and calculate for remaining values
        ema = sma
        for price in history[window:]:
            ema = price * alpha + ema * (1 - alpha)
            
        return ema
    
    def _calculate_bollinger_bands(self, symbol: str, window: int = 20, std_dev: float = 2.0) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            symbol: Symbol to calculate for
            window: Lookback window
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (middle band, upper band, lower band)
        """
        history = self.price_history[symbol]
        if len(history) < window:
            return None, None, None
            
        # Calculate middle band (SMA)
        middle = self._calculate_sma(symbol, window)
        
        # Calculate standard deviation
        recent_history = history[-window:]
        std = np.std(recent_history)
        
        # Calculate upper and lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return middle, upper, lower
    
    def _calculate_rsi(self, symbol: str, window: int) -> Optional[float]:
        """
        Calculate Relative Strength Index.
        
        Args:
            symbol: Symbol to calculate for
            window: RSI window length
            
        Returns:
            RSI value or None if not enough history
        """
        history = self.price_history[symbol]
        if len(history) <= window:
            return None
            
        # Calculate price changes
        price_changes = [history[i+1] - history[i] for i in range(len(history)-1)]
        
        # Get the last 'window' changes
        changes = price_changes[-(window+1):]
        
        # Separate gains and losses
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / window
        avg_loss = sum(losses) / window
        
        if avg_loss == 0:
            return 100.0  # No losses, RSI = 100
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class HistoricalDataProvider(MarketDataProvider):
    """
    Provides market data from historical CSV or JSON files.
    
    This is a stub implementation that would be expanded to read
    real historical data from files in a production system.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize with path to historical data.
        
        Args:
            data_dir: Directory containing historical data files
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__)
        self.symbols = []
        self.data_by_day = {}
        
        self.logger.info(f"Initialized historical data provider from {data_dir}")
        self.logger.warning("Historical data provider is a stub and not fully implemented")
        
    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return self.symbols
        
    def get_data(self, day: int) -> Dict[str, Any]:
        """
        Get historical market data for a specific day.
        
        Args:
            day: Simulation day (0-indexed)
            
        Returns:
            Dict containing market data per symbol
        """
        if day in self.data_by_day:
            return self.data_by_day[day]
        
        # Fallback to synthetic data generation
        self.logger.warning(f"No historical data for day {day}, generating synthetic data")
        provider = RandomWalkDataProvider()
        return provider.get_data(day)
