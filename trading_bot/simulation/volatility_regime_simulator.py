#!/usr/bin/env python3
"""
Volatility Regime Simulator

This module creates synthetic market data with different volatility regimes
to simulate realistic market conditions for testing trading strategies.

Features:
- Multiple volatility states (low, normal, high, extreme)
- Realistic regime transitions
- Correlation preservation across assets
- Configurable regime parameters
- Historical pattern overlays
"""

import os
import logging
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.stats import norm

# Configure logging
logger = logging.getLogger(__name__)

class VolatilityRegime(str, Enum):
    """Volatility regime types."""
    LOW = "low"              # Low volatility, steady market
    NORMAL = "normal"        # Normal market conditions
    HIGH = "high"            # High volatility, typically during earnings or events
    EXTREME = "extreme"      # Extreme volatility, crisis conditions

class MarketTrend(str, Enum):
    """Market trend directions."""
    BULLISH = "bullish"      # Upward trend
    BEARISH = "bearish"      # Downward trend
    SIDEWAYS = "sideways"    # Range-bound, no clear direction

class VolatilityRegimeSimulator:
    """
    Simulates market data with different volatility regimes.
    
    Features:
    - Generate synthetic OHLCV data with realistic volatility patterns
    - Transition between different market states
    - Support for multi-asset simulation with correlation
    - Configurable parameters for each volatility regime
    """
    
    def __init__(self, 
                 assets: List[str],
                 base_prices: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None,
                 correlation_matrix: Optional[np.ndarray] = None):
        """
        Initialize the volatility regime simulator.
        
        Args:
            assets: List of asset symbols to simulate
            base_prices: Initial prices for each asset (None = use defaults)
            seed: Random seed for reproducibility
            correlation_matrix: Asset return correlation matrix
        """
        self.assets = assets
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Default base prices if not provided
        self.base_prices = base_prices or {
            "AAPL": 170.0,
            "MSFT": 290.0,
            "AMZN": 130.0,
            "GOOGL": 140.0,
            "TSLA": 180.0,
            "BTC/USD": 35000.0,
            "ETH/USD": 2000.0,
            "EUR/USD": 1.1,
            "SPY": 420.0,
            "QQQ": 350.0
        }
        
        # Use defaults for assets not in the dictionary
        for symbol in self.assets:
            if symbol not in self.base_prices:
                self.base_prices[symbol] = 100.0
        
        # Correlation matrix (default if not provided)
        if correlation_matrix is None:
            # Create default correlation matrix
            n_assets = len(assets)
            correlation_matrix = np.eye(n_assets)
            
            # Add some realistic correlations for equity-like assets
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    # Default moderate correlation
                    corr = 0.5
                    
                    # Higher correlation for similar assets (rough estimate)
                    if assets[i][:3] == assets[j][:3]:
                        corr = 0.8
                    
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        
        # Ensure correlation matrix dimensions match assets
        if correlation_matrix.shape[0] != len(assets) or correlation_matrix.shape[1] != len(assets):
            raise ValueError(f"Correlation matrix dimensions ({correlation_matrix.shape}) don't match number of assets ({len(assets)})")
        
        self.correlation_matrix = correlation_matrix
        
        # Perform Cholesky decomposition for correlated returns
        self.cholesky = np.linalg.cholesky(correlation_matrix)
        
        # Volatility parameters for each regime
        self.volatility_params = {
            VolatilityRegime.LOW: {
                "annualized_vol": 0.10,        # 10% annual volatility
                "mean_reversion_strength": 0.02,  # Strong mean reversion
                "jumps_per_year": 2,           # Very few jumps
                "jump_size_mean": 0.005,       # Small jumps
                "jump_size_std": 0.005,        # Low jump dispersion
                "intraday_pattern": "weak",    # Weak intraday pattern
                "auto_correlation": 0.1,       # Low auto-correlation
                "trend_strength": 0.0005,      # Weak trend
                "mean_reversion_level": 0.0,   # Mean reversion level (relative to trend)
                "bid_ask_spread_bps": 1.0,     # 1 basis point spread
            },
            VolatilityRegime.NORMAL: {
                "annualized_vol": 0.20,        # 20% annual volatility
                "mean_reversion_strength": 0.01,  # Normal mean reversion
                "jumps_per_year": 10,          # Occasional jumps
                "jump_size_mean": 0.01,        # Normal jumps
                "jump_size_std": 0.01,         # Normal jump dispersion
                "intraday_pattern": "normal",  # Normal intraday pattern
                "auto_correlation": 0.05,      # Some auto-correlation
                "trend_strength": 0.001,       # Normal trend
                "mean_reversion_level": 0.0,   # Mean reversion level
                "bid_ask_spread_bps": 2.0,     # 2 basis point spread
            },
            VolatilityRegime.HIGH: {
                "annualized_vol": 0.35,        # 35% annual volatility
                "mean_reversion_strength": 0.005, # Weak mean reversion
                "jumps_per_year": 30,          # Frequent jumps
                "jump_size_mean": 0.02,        # Larger jumps
                "jump_size_std": 0.02,         # Higher jump dispersion
                "intraday_pattern": "strong",  # Strong intraday pattern
                "auto_correlation": 0.02,      # Lower auto-correlation
                "trend_strength": 0.002,       # Stronger trend
                "mean_reversion_level": 0.0,   # Mean reversion level
                "bid_ask_spread_bps": 5.0,     # 5 basis point spread
            },
            VolatilityRegime.EXTREME: {
                "annualized_vol": 0.60,        # 60% annual volatility
                "mean_reversion_strength": 0.001, # Very weak mean reversion
                "jumps_per_year": 60,          # Very frequent jumps
                "jump_size_mean": 0.03,        # Large jumps
                "jump_size_std": 0.04,         # High jump dispersion
                "intraday_pattern": "extreme", # Extreme intraday pattern
                "auto_correlation": -0.01,     # Negative auto-correlation
                "trend_strength": 0.003,       # Strong trend
                "mean_reversion_level": 0.0,   # Mean reversion level
                "bid_ask_spread_bps": 10.0,    # 10 basis point spread
            }
        }
        
        # Trend parameters
        self.trend_params = {
            MarketTrend.BULLISH: {
                "drift_multiplier": 3.0,       # Strong positive drift
                "up_probability": 0.6,         # Higher probability of up moves
                "skew": 0.5,                   # Positive skew
            },
            MarketTrend.BEARISH: {
                "drift_multiplier": -3.0,      # Strong negative drift
                "up_probability": 0.4,         # Lower probability of up moves
                "skew": -0.5,                  # Negative skew
            },
            MarketTrend.SIDEWAYS: {
                "drift_multiplier": 0.0,       # No drift
                "up_probability": 0.5,         # Equal probability of up/down
                "skew": 0.0,                   # No skew
            }
        }
        
        # Initialize current state
        self.current_regime = VolatilityRegime.NORMAL
        self.current_trend = MarketTrend.SIDEWAYS
        self.current_prices = {symbol: price for symbol, price in self.base_prices.items() if symbol in self.assets}
        
        # Regime transition probabilities (rows=from, cols=to)
        # low, normal, high, extreme
        self.regime_transition_matrix = np.array([
            [0.95, 0.05, 0.00, 0.00],  # From LOW
            [0.05, 0.90, 0.05, 0.00],  # From NORMAL
            [0.00, 0.10, 0.85, 0.05],  # From HIGH
            [0.00, 0.00, 0.20, 0.80]   # From EXTREME
        ])
        
        # Trend transition probabilities (rows=from, cols=to)
        # bullish, bearish, sideways
        self.trend_transition_matrix = np.array([
            [0.80, 0.05, 0.15],  # From BULLISH
            [0.05, 0.80, 0.15],  # From BEARISH
            [0.15, 0.15, 0.70]   # From SIDEWAYS
        ])
        
        # Additional state variables
        self.prev_returns = np.zeros(len(assets))
        self.regime_duration = 0
        self.trend_duration = 0
        self.regime_history = []
        self.trend_history = []
        
        logger.info(f"Initialized VolatilityRegimeSimulator with {len(assets)} assets")
        logger.info(f"Initial regime: {self.current_regime}, trend: {self.current_trend}")
    
    def set_regime_parameters(self, regime: VolatilityRegime, parameters: Dict[str, Any]):
        """
        Set parameters for a specific volatility regime.
        
        Args:
            regime: Volatility regime to update
            parameters: Dictionary of parameters to set
        """
        if regime not in self.volatility_params:
            raise ValueError(f"Invalid regime: {regime}")
        
        # Update parameters
        self.volatility_params[regime].update(parameters)
        
        logger.info(f"Updated parameters for {regime} regime")
    
    def set_trend_parameters(self, trend: MarketTrend, parameters: Dict[str, Any]):
        """
        Set parameters for a specific market trend.
        
        Args:
            trend: Market trend to update
            parameters: Dictionary of parameters to set
        """
        if trend not in self.trend_params:
            raise ValueError(f"Invalid trend: {trend}")
        
        # Update parameters
        self.trend_params[trend].update(parameters)
        
        logger.info(f"Updated parameters for {trend} trend")
    
    def set_current_state(self, regime: VolatilityRegime, trend: MarketTrend):
        """
        Manually set the current market state.
        
        Args:
            regime: Volatility regime to set
            trend: Market trend to set
        """
        self.current_regime = regime
        self.current_trend = trend
        
        # Reset durations
        self.regime_duration = 0
        self.trend_duration = 0
        
        # Record in history
        self.regime_history.append(regime)
        self.trend_history.append(trend)
        
        logger.info(f"Manually set state to regime={regime}, trend={trend}")
    
    def _update_regimes(self):
        """Update volatility regime and market trend based on transition probabilities."""
        # Update regime duration
        self.regime_duration += 1
        self.trend_duration += 1
        
        # Volatility regime transition
        # Less likely to change if duration is short
        regime_change_prob = min(1.0, self.regime_duration / 20)  # Full probability after 20 periods
        
        if np.random.random() < regime_change_prob:
            # Get transition probabilities for current regime
            regime_idx = list(VolatilityRegime).index(self.current_regime)
            transition_probs = self.regime_transition_matrix[regime_idx]
            
            # Sample new regime
            new_regime_idx = np.random.choice(len(transition_probs), p=transition_probs)
            new_regime = list(VolatilityRegime)[new_regime_idx]
            
            if new_regime != self.current_regime:
                logger.info(f"Volatility regime changing from {self.current_regime} to {new_regime}")
                self.current_regime = new_regime
                self.regime_duration = 0
        
        # Market trend transition
        # Less likely to change if duration is short
        trend_change_prob = min(1.0, self.trend_duration / 30)  # Full probability after 30 periods
        
        if np.random.random() < trend_change_prob:
            # Get transition probabilities for current trend
            trend_idx = list(MarketTrend).index(self.current_trend)
            transition_probs = self.trend_transition_matrix[trend_idx]
            
            # Sample new trend
            new_trend_idx = np.random.choice(len(transition_probs), p=transition_probs)
            new_trend = list(MarketTrend)[new_trend_idx]
            
            if new_trend != self.current_trend:
                logger.info(f"Market trend changing from {self.current_trend} to {new_trend}")
                self.current_trend = new_trend
                self.trend_duration = 0
        
        # Record current state in history
        self.regime_history.append(self.current_regime)
        self.trend_history.append(self.current_trend)
    
    def generate_returns(self, with_jumps: bool = True) -> np.ndarray:
        """
        Generate correlated returns for all assets based on current regime.
        
        Args:
            with_jumps: Whether to include price jumps
            
        Returns:
            Array of returns for each asset
        """
        n_assets = len(self.assets)
        
        # Get parameters for current regime
        vol_params = self.volatility_params[self.current_regime]
        trend_params = self.trend_params[self.current_trend]
        
        # Convert annualized volatility to per-period volatility
        # Assuming 252 trading days per year and 78 periods per day (5-minute bars)
        periods_per_year = 252 * 78
        period_vol = vol_params["annualized_vol"] / np.sqrt(periods_per_year)
        
        # Generate uncorrelated random returns
        uncorrelated_returns = np.random.normal(0, 1, n_assets)
        
        # Apply correlation structure
        correlated_returns = np.dot(self.cholesky, uncorrelated_returns) * period_vol
        
        # Apply trend bias (drift)
        drift = trend_params["drift_multiplier"] * period_vol / 10
        correlated_returns += drift
        
        # Apply skew
        if trend_params["skew"] != 0:
            skew_factor = np.random.normal(trend_params["skew"] * period_vol, period_vol / 2)
            correlated_returns += skew_factor
        
        # Apply mean reversion
        if vol_params["mean_reversion_strength"] > 0:
            mean_reversion = -vol_params["mean_reversion_strength"] * self.prev_returns
            correlated_returns += mean_reversion
        
        # Apply auto-correlation
        if vol_params["auto_correlation"] != 0:
            auto_corr = vol_params["auto_correlation"] * self.prev_returns
            correlated_returns += auto_corr
        
        # Apply jumps
        if with_jumps:
            # Calculate jump probability for this period
            jump_prob = vol_params["jumps_per_year"] / periods_per_year
            
            # Check for jumps in each asset
            for i in range(n_assets):
                if np.random.random() < jump_prob:
                    # Jump direction (more likely to be in trend direction)
                    if trend_params["up_probability"] > np.random.random():
                        jump_sign = 1.0  # Up
                    else:
                        jump_sign = -1.0  # Down
                    
                    # Jump size
                    jump_size = np.random.normal(
                        vol_params["jump_size_mean"], 
                        vol_params["jump_size_std"]
                    )
                    
                    # Apply jump
                    correlated_returns[i] += jump_sign * abs(jump_size)
        
        # Store returns for next iteration
        self.prev_returns = correlated_returns.copy()
        
        return correlated_returns
    
    def generate_ohlc(self, returns: np.ndarray, timestamp: datetime) -> Dict[str, pd.DataFrame]:
        """
        Generate OHLC data for all assets based on returns.
        
        Args:
            returns: Array of returns for each asset
            timestamp: Timestamp for the generated data
            
        Returns:
            Dictionary of DataFrames with OHLC data for each asset
        """
        result = {}
        
        # Process each asset
        for i, symbol in enumerate(self.assets):
            # Get parameters for current regime
            vol_params = self.volatility_params[self.current_regime]
            
            # Get current price
            current_price = self.current_prices[symbol]
            
            # Calculate new price
            new_price = current_price * (1 + returns[i])
            
            # Calculate OHLC
            # In real markets, typically high > open and close > low, so we bias random values
            intraday_vol_factor = {
                "weak": 0.1,
                "normal": 0.2,
                "strong": 0.4,
                "extreme": 0.7
            }.get(vol_params["intraday_pattern"], 0.2)
            
            # Intraday volatility as a fraction of return
            intraday_vol = abs(returns[i]) * intraday_vol_factor
            if intraday_vol < period_vol * 0.1:
                intraday_vol = period_vol * 0.1  # Minimum intraday volatility
            
            # OHLC with realistic relationships
            if returns[i] >= 0:
                # Positive return: open near low, close near high
                open_price = current_price * (1 + np.random.uniform(0, intraday_vol * 0.5))
                close_price = new_price
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, intraday_vol))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, intraday_vol))
            else:
                # Negative return: open near high, close near low
                open_price = current_price * (1 - np.random.uniform(0, intraday_vol * 0.5))
                close_price = new_price
                high_price = max(open_price, close_price) * (1 + np.random.uniform(0, intraday_vol))
                low_price = min(open_price, close_price) * (1 - np.random.uniform(0, intraday_vol))
            
            # Ensure high >= max(open, close) and low <= min(open, close)
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Generate volume
            # Volume tends to be higher in high volatility regimes
            base_volume = 1000
            vol_multiplier = {
                VolatilityRegime.LOW: 0.5,
                VolatilityRegime.NORMAL: 1.0,
                VolatilityRegime.HIGH: 2.0,
                VolatilityRegime.EXTREME: 4.0
            }.get(self.current_regime, 1.0)
            
            # Volume also tends to be higher with larger price moves
            move_multiplier = 1.0 + 5.0 * abs(returns[i])
            
            # Random component with log-normal distribution
            random_component = np.random.lognormal(0, 0.5)
            
            volume = base_volume * vol_multiplier * move_multiplier * random_component
            
            # Calculate bid and ask prices based on spread
            spread_bps = vol_params["bid_ask_spread_bps"]
            spread_factor = spread_bps / 10000  # Convert bps to factor
            
            bid_price = close_price * (1 - spread_factor / 2)
            ask_price = close_price * (1 + spread_factor / 2)
            
            # Create DataFrame
            df = pd.DataFrame({
                'open': [open_price],
                'high': [high_price],
                'low': [low_price],
                'close': [close_price],
                'volume': [volume],
                'bid': [bid_price],
                'ask': [ask_price]
            }, index=[timestamp])
            
            # Store result
            result[symbol] = df
            
            # Update current price
            self.current_prices[symbol] = close_price
        
        return result
    
    def generate_data(self, 
                    periods: int = 100, 
                    start_time: Optional[datetime] = None,
                    time_delta: timedelta = timedelta(minutes=5),
                    with_regime_changes: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate market data for multiple periods.
        
        Args:
            periods: Number of periods to generate
            start_time: Start time for the data (None = now)
            time_delta: Time between data points
            with_regime_changes: Whether to allow regime changes
            
        Returns:
            Dictionary of DataFrames with OHLC data for each asset
        """
        # Use current time if not specified
        if start_time is None:
            start_time = datetime.now().replace(microsecond=0)
        
        # Initialize result
        result = {symbol: [] for symbol in self.assets}
        
        # Generate data for each period
        for i in range(periods):
            # Current timestamp
            timestamp = start_time + (i * time_delta)
            
            # Update regimes if enabled
            if with_regime_changes:
                self._update_regimes()
            
            # Generate returns
            returns = self.generate_returns()
            
            # Generate OHLC data
            ohlc_data = self.generate_ohlc(returns, timestamp)
            
            # Append to result
            for symbol, df in ohlc_data.items():
                result[symbol].append(df)
        
        # Combine data into DataFrames
        for symbol in self.assets:
            result[symbol] = pd.concat(result[symbol])
        
        logger.info(f"Generated {periods} periods of market data with {len(self.assets)} assets")
        return result
    
    def get_regime_history(self) -> pd.DataFrame:
        """
        Get the history of volatility regimes and market trends.
        
        Returns:
            DataFrame with regime and trend history
        """
        # Create DataFrame with history
        df = pd.DataFrame({
            "regime": self.regime_history,
            "trend": self.trend_history
        })
        
        return df
    
    def plot_regime_history(self, save_path: Optional[str] = None):
        """
        Plot the history of volatility regimes and market trends.
        
        Args:
            save_path: Path to save the plot (None = don't save)
        """
        if not self.regime_history:
            logger.warning("No regime history available to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot regimes
        regime_values = [list(VolatilityRegime).index(r) for r in self.regime_history]
        plt.subplot(2, 1, 1)
        plt.plot(regime_values, '-o', markersize=4)
        plt.yticks(
            range(len(VolatilityRegime)),
            [r.value for r in VolatilityRegime]
        )
        plt.title("Volatility Regime History")
        plt.grid(True)
        
        # Plot trends
        trend_values = [list(MarketTrend).index(t) for t in self.trend_history]
        plt.subplot(2, 1, 2)
        plt.plot(trend_values, '-o', markersize=4)
        plt.yticks(
            range(len(MarketTrend)),
            [t.value for t in MarketTrend]
        )
        plt.title("Market Trend History")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved regime history plot to {save_path}")
        
        plt.show()

    def plot_simulated_data(self, 
                          data: Dict[str, pd.DataFrame],
                          symbols: Optional[List[str]] = None,
                          save_path: Optional[str] = None):
        """
        Plot the simulated price data.
        
        Args:
            data: Dictionary of DataFrames with OHLC data
            symbols: List of symbols to plot (None = all)
            save_path: Path to save the plot (None = don't save)
        """
        if not data:
            logger.warning("No data available to plot")
            return
        
        # Use all symbols if not specified
        if symbols is None:
            symbols = list(data.keys())
        
        # Limit to available symbols
        symbols = [s for s in symbols if s in data]
        
        if not symbols:
            logger.warning("No valid symbols to plot")
            return
        
        # Create figure
        n_symbols = len(symbols)
        fig, axes = plt.subplots(n_symbols, 1, figsize=(12, 4 * n_symbols))
        
        # Handle single symbol case
        if n_symbols == 1:
            axes = [axes]
        
        # Plot each symbol
        for i, symbol in enumerate(symbols):
            df = data[symbol]
            
            # Plot close price
            axes[i].plot(df.index, df['close'], label=f"{symbol} Close")
            
            # Add volume as bars on a secondary y-axis
            volume_ax = axes[i].twinx()
            volume_ax.bar(df.index, df['volume'], alpha=0.3, label="Volume", color='gray')
            volume_ax.set_ylabel("Volume")
            
            # Add labels
            axes[i].set_title(f"{symbol} Price and Volume")
            axes[i].grid(True)
            
            # Create legend with both axes
            lines1, labels1 = axes[i].get_legend_handles_labels()
            lines2, labels2 = volume_ax.get_legend_handles_labels()
            axes[i].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis for datetime
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved price plot to {save_path}")
        
        plt.show()
    
    def overlay_historical_pattern(self, 
                                 source_data: pd.DataFrame,
                                 target_symbol: str,
                                 scaling_factor: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        Overlay a historical price pattern on simulated data.
        
        Args:
            source_data: DataFrame with historical OHLCV data
            target_symbol: Symbol to apply the pattern to
            scaling_factor: Scale the pattern magnitude
            
        Returns:
            Dictionary with updated OHLC data
        """
        if target_symbol not in self.current_prices:
            raise ValueError(f"Invalid target symbol: {target_symbol}")
        
        if len(source_data) < 2:
            raise ValueError("Source data too short for overlay")
        
        # Calculate returns from source data
        source_returns = source_data['close'].pct_change().dropna().values
        
        # Scale returns
        scaled_returns = source_returns * scaling_factor
        
        # Number of periods to generate
        periods = len(scaled_returns)
        
        # Use current time if not specified
        start_time = datetime.now().replace(microsecond=0)
        
        # Initialize result
        result = {symbol: [] for symbol in self.assets}
        
        # Current prices
        current_prices = self.current_prices.copy()
        
        # Generate data for each period
        for i in range(periods):
            # Current timestamp
            timestamp = start_time + (i * timedelta(minutes=5))
            
            # Generate correlated returns for all assets
            returns = self.generate_returns()
            
            # Override return for target symbol
            target_idx = self.assets.index(target_symbol)
            returns[target_idx] = scaled_returns[i]
            
            # Generate OHLC data
            ohlc_data = self.generate_ohlc(returns, timestamp)
            
            # Append to result
            for symbol, df in ohlc_data.items():
                result[symbol].append(df)
        
        # Combine data into DataFrames
        for symbol in self.assets:
            result[symbol] = pd.concat(result[symbol])
        
        logger.info(f"Generated {periods} periods with historical overlay for {target_symbol}")
        return result

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create simulator
    assets = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    simulator = VolatilityRegimeSimulator(assets)
    
    # Set initial state
    simulator.set_current_state(VolatilityRegime.NORMAL, MarketTrend.BULLISH)
    
    # Generate data
    data = simulator.generate_data(periods=100)
    
    # Plot regime history
    simulator.plot_regime_history()
    
    # Plot price data
    simulator.plot_simulated_data(data)
    
    print("Simulation completed successfully!") 