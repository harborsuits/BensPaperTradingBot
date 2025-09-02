#!/usr/bin/env python3
"""
Standalone Synthetic Market Generator

This self-contained script demonstrates the core functionality of generating 
synthetic market data for testing trading strategies. It includes:

1. Single asset random walk generation with realistic OHLCV data
2. Market regime application (bullish, bearish, volatile, etc.)
3. Correlated asset price generation with sector-based relationships

This can be used with our A/B Testing Framework for strategy evaluation 
under different market conditions.
"""

import os
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta


class MarketRegimeType(str, Enum):
    """Types of market regimes for synthetic data generation."""
    BULLISH = "bullish"           # Steadily rising market
    BEARISH = "bearish"           # Steadily falling market
    SIDEWAYS = "sideways"         # Range-bound, low-volatility market
    VOLATILE = "volatile"         # High-volatility, unpredictable moves
    TRENDING = "trending"         # Strong directional movement
    MEAN_REVERTING = "mean_reverting"  # Price returns to mean
    CRASH = "crash"               # Sharp market decline
    RECOVERY = "recovery"         # Post-crash recovery


class SyntheticMarketGenerator:
    """Generates synthetic market data for testing trading strategies."""
    
    def __init__(self, seed=None):
        """Initialize the generator with optional random seed."""
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_random_walk(
        self, 
        days=252, 
        base_price=100.0,
        volatility=0.01,
        drift=0.0001
    ):
        """Generate a basic random walk price series."""
        # Generate daily returns
        daily_returns = np.random.normal(drift, volatility, days)
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        # Calculate price series
        prices = base_price * cumulative_returns
        
        # Generate dates
        end_date = datetime.now()
        dates = []
        current_date = end_date
        
        # Skip weekends
        while len(dates) < days:
            if current_date.weekday() < 5:  # Monday=0, Sunday=6
                dates.append(current_date)
            current_date = current_date - timedelta(days=1)
        
        # Reverse to get ascending order
        dates = dates[::-1]
        
        # Create price DataFrame with OHLCV structure
        df = pd.DataFrame(index=dates)
        
        # Close prices are directly from our simulation
        df['close'] = prices
        
        # Generate other OHLCV columns
        df['open'] = np.roll(df['close'], 1)
        df.loc[df.index[0], 'open'] = base_price
        
        # Calculate high and low from open and close
        for i in range(len(df)):
            min_price = min(df['open'].iloc[i], df['close'].iloc[i])
            max_price = max(df['open'].iloc[i], df['close'].iloc[i])
            
            # High is above the max of open and close
            high_offset = abs(np.random.normal(0, volatility * max_price))
            df.loc[df.index[i], 'high'] = max_price + high_offset
            
            # Low is below the min of open and close
            low_offset = abs(np.random.normal(0, volatility * min_price))
            df.loc[df.index[i], 'low'] = min_price - low_offset
        
        # Generate volume (higher in more volatile periods)
        avg_volume = 1000000  # Base volume level
        daily_volatility = np.abs(df['close'].pct_change())
        
        # Volume correlates with volatility but has its own randomness
        df['volume'] = avg_volume * (1 + 5 * daily_volatility) * np.random.lognormal(
            0, 0.5, days
        )
        df.loc[df.index[0], 'volume'] = avg_volume
        
        return df
    
    def apply_trend(self, df, trend_strength=0.001, trend_direction=1):
        """Apply a trend to an existing price series."""
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Generate trend component
        days = len(df)
        trend = np.linspace(0, days * trend_strength, days) * trend_direction
        
        # Apply trend to prices
        for col in ['open', 'high', 'low', 'close']:
            if col in result.columns:
                # Apply exponential trend factor
                result[col] = result[col] * np.exp(trend)
        
        return result
    
    def apply_volatility(self, df, volatility_factor=2.0):
        """Apply increased or decreased volatility to the price series."""
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate returns of closing prices
        returns = result['close'].pct_change().fillna(0)
        
        # Calculate new returns with modified volatility
        new_returns = returns * volatility_factor
        
        # Recalculate prices from new returns
        base_price = result['close'].iloc[0]
        new_closes = base_price * np.cumprod(1 + new_returns)
        
        # Apply new closing prices
        result['close'] = new_closes
        
        # Adjust other price columns to maintain relative relationships
        for i in range(len(result)):
            if i > 0:  # Skip first row as it's already correct
                idx = result.index[i]
                
                # Calculate ratio of new close to old close
                price_ratio = result.loc[idx, 'close'] / df.loc[idx, 'close']
                
                # Apply same ratio to other price columns
                for col in ['open', 'high', 'low']:
                    if col in result.columns:
                        result.loc[idx, col] = df.loc[idx, col] * price_ratio
        
        return result
    
    def generate_regime_scenario(self, regime, days=252, base_price=100.0, volatility=0.01):
        """Generate a specific market regime scenario."""
        # Start with a basic random walk
        df = self.generate_random_walk(
            days=days,
            base_price=base_price,
            volatility=volatility
        )
        
        # Apply regime-specific modifications
        if regime == MarketRegimeType.BULLISH:
            # Strong upward trend
            df = self.apply_trend(df, trend_strength=0.001, trend_direction=1)
        
        elif regime == MarketRegimeType.BEARISH:
            # Strong downward trend
            df = self.apply_trend(df, trend_strength=0.001, trend_direction=-1)
        
        elif regime == MarketRegimeType.VOLATILE:
            # High volatility
            df = self.apply_volatility(df, volatility_factor=2.5)
        
        elif regime == MarketRegimeType.CRASH:
            # Normal market followed by sharp crash
            crash_start = int(days * 0.7)  # Crash in last 30% of series
            
            # Apply uptrend before crash
            df = self.apply_trend(df, trend_strength=0.0005, trend_direction=1)
            
            # Apply crash
            crash_series = pd.Series(
                index=range(crash_start, days),
                data=np.linspace(1.0, 0.6, days - crash_start)  # 40% drop
            )
            
            # Apply crash to prices
            for i in range(crash_start, days):
                crash_factor = crash_series.iloc[i - crash_start]
                df.iloc[i, df.columns.get_indexer(['open', 'high', 'low', 'close'])] *= crash_factor
        
        return df


class CorrelationStructure:
    """Defines correlation structures between assets."""
    
    def __init__(self, assets, base_correlation=0.3, seed=None):
        """Initialize the correlation structure."""
        self.assets = assets
        self.num_assets = len(assets)
        self.seed = seed
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize correlation matrix with base correlation
        self.correlation_matrix = np.ones((self.num_assets, self.num_assets)) * base_correlation
        
        # Set diagonal to 1 (perfect self-correlation)
        np.fill_diagonal(self.correlation_matrix, 1.0)
    
    def set_pairwise_correlation(self, asset1, asset2, correlation):
        """Set correlation between a pair of assets."""
        try:
            idx1 = self.assets.index(asset1)
            idx2 = self.assets.index(asset2)
            
            # Set correlation symmetrically
            self.correlation_matrix[idx1, idx2] = correlation
            self.correlation_matrix[idx2, idx1] = correlation
            
        except ValueError:
            print(f"Asset not found: {asset1} or {asset2}")
    
    def set_sector_structure(self, sectors, intra_sector_correlation=0.7, inter_sector_correlation=0.3):
        """Set up a sector-based correlation structure."""
        # First reset to baseline
        self.correlation_matrix.fill(inter_sector_correlation)
        np.fill_diagonal(self.correlation_matrix, 1.0)
        
        # Set correlations within each sector
        for sector, sector_assets in sectors.items():
            for i, asset1 in enumerate(sector_assets):
                if asset1 not in self.assets:
                    continue
                    
                for asset2 in sector_assets[i+1:]:
                    if asset2 not in self.assets:
                        continue
                        
                    self.set_pairwise_correlation(asset1, asset2, intra_sector_correlation)
    
    def get_correlation_dataframe(self):
        """Get the correlation matrix as a labeled DataFrame."""
        return pd.DataFrame(
            self.correlation_matrix,
            index=self.assets,
            columns=self.assets
        )
    
    def ensure_positive_definite(self):
        """Ensure the correlation matrix is positive definite."""
        # Check eigenvalues
        eigenvalues = np.linalg.eigvals(self.correlation_matrix)
        
        if np.any(eigenvalues <= 0):
            # Add a small positive value to diagonal if needed
            min_eigenvalue = min(eigenvalues)
            if min_eigenvalue <= 0:
                adjustment = abs(min_eigenvalue) + 1e-6
                np.fill_diagonal(self.correlation_matrix, 1.0 + adjustment)
                
                # Re-normalize to ensure diagonal is exactly 1
                D = np.diag(1 / np.sqrt(np.diag(self.correlation_matrix)))
                self.correlation_matrix = D @ self.correlation_matrix @ D


class CorrelatedMarketGenerator:
    """Generates correlated synthetic market data for multiple assets."""
    
    def __init__(self, seed=None):
        """Initialize the correlated market generator."""
        self.seed = seed
        self.single_asset_generator = SyntheticMarketGenerator(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_correlated_series(
        self,
        correlation_structure,
        days=252,
        base_prices=None,
        volatilities=None,
        drifts=None
    ):
        """Generate correlated price series for multiple assets."""
        # Get assets and correlation matrix
        assets = correlation_structure.assets
        correlation_matrix = correlation_structure.correlation_matrix
        
        # Ensure the correlation matrix is valid for Cholesky decomposition
        correlation_structure.ensure_positive_definite()
        
        # Set default parameters if not provided
        if base_prices is None:
            base_prices = {asset: 100.0 for asset in assets}
        
        if volatilities is None:
            volatilities = {asset: 0.01 for asset in assets}
        
        if drifts is None:
            drifts = {asset: 0.0001 for asset in assets}
        
        # Calculate Cholesky decomposition of correlation matrix
        cholesky = np.linalg.cholesky(correlation_matrix)
        
        # Generate uncorrelated random returns
        uncorrelated_returns = np.random.normal(0, 1, (days, len(assets)))
        
        # Convert to correlated returns
        correlated_returns = uncorrelated_returns @ cholesky.T
        
        # Apply asset-specific drift and volatility
        for i, asset in enumerate(assets):
            vol = volatilities.get(asset, 0.01)
            drift = drifts.get(asset, 0.0001)
            
            # Scale and shift
            correlated_returns[:, i] = correlated_returns[:, i] * vol + drift
        
        # Generate dates
        end_date = datetime.now()
        
        # Skip weekends
        dates = []
        current_date = end_date
        while len(dates) < days:
            if current_date.weekday() < 5:  # Monday=0, Sunday=6
                dates.append(current_date)
            current_date = current_date - timedelta(days=1)
        
        # Reverse to get ascending order
        dates = dates[::-1]
        
        # Calculate price series for each asset
        price_data = {}
        
        for i, asset in enumerate(assets):
            # Get parameters for this asset
            base_price = base_prices.get(asset, 100.0)
            
            # Calculate cumulative returns
            cum_returns = np.cumprod(1 + correlated_returns[:, i])
            
            # Calculate prices
            prices = base_price * cum_returns
            
            # Create DataFrame
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            
            # Generate OHLC data based on close prices
            df['open'] = np.roll(df['close'], 1)
            df.loc[df.index[0], 'open'] = base_price
            
            # Add random variation to create high and low
            daily_volatility = volatilities.get(asset, 0.01)
            
            for j in range(len(df)):
                max_price = max(df['open'].iloc[j], df['close'].iloc[j])
                min_price = min(df['open'].iloc[j], df['close'].iloc[j])
                
                high_pct = np.random.uniform(0.005, 0.015) + daily_volatility
                low_pct = np.random.uniform(0.005, 0.015) + daily_volatility
                
                df.loc[df.index[j], 'high'] = max_price * (1 + high_pct)
                df.loc[df.index[j], 'low'] = min_price * (1 - low_pct)
            
            # Generate volume - higher volume often correlates with volatility
            avg_volume = 1000000  # Base daily volume
            daily_price_change = np.abs(df['close'].pct_change())
            
            df['volume'] = avg_volume * (1 + 5 * daily_price_change) * np.random.lognormal(
                0, 0.5, days
            )
            df.loc[df.index[0], 'volume'] = avg_volume
            
            # Store in result dictionary
            price_data[asset] = df
        
        return price_data
    
    def save_to_csv(self, price_data, directory):
        """Save correlated price data to CSV files."""
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each asset's data
        for asset, df in price_data.items():
            filename = os.path.join(directory, f"{asset}.csv")
            df.to_csv(filename)
        
        print(f"Saved {len(price_data)} synthetic price series to {directory}")


# Example usage function
def run_example():
    """Run example demonstrating the synthetic market generators."""
    # Create output directory
    output_dir = os.path.join(os.path.expanduser("~"), ".trading_bot", "synthetic_data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n1. Testing Single Asset Generation")
    print("---------------------------------")
    
    # Create generator
    generator = SyntheticMarketGenerator(seed=42)
    
    # Generate a simple random walk
    df = generator.generate_random_walk(days=252, base_price=100.0, volatility=0.01)
    
    # Print summary
    print(f"Generated {len(df)} days of price data")
    print(f"Starting price: ${df['close'].iloc[0]:.2f}")
    print(f"Ending price: ${df['close'].iloc[-1]:.2f}")
    print(f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    # Save data
    df.to_csv(os.path.join(output_dir, "basic_price_series.csv"))
    
    print("\n2. Testing Market Regimes")
    print("------------------------")
    
    # Generate different regime scenarios
    regimes = [
        MarketRegimeType.BULLISH,
        MarketRegimeType.BEARISH,
        MarketRegimeType.VOLATILE,
        MarketRegimeType.CRASH
    ]
    
    for regime in regimes:
        # Generate the data
        df = generator.generate_regime_scenario(
            regime=regime,
            days=252,
            base_price=100.0,
            volatility=0.015
        )
        
        # Print summary
        returns = df['close'].pct_change().dropna()
        print(f"{regime.value}: Volatility: {returns.std()*100:.2f}%, "
              f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
        
        # Save data
        df.to_csv(os.path.join(output_dir, f"regime_{regime.value}.csv"))
    
    print("\n3. Testing Correlated Assets")
    print("--------------------------")
    
    # Create correlated generator
    correlated_generator = CorrelatedMarketGenerator(seed=42)
    
    # Define asset list
    assets = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    # Create correlation structure
    correlation = CorrelationStructure(assets, seed=42)
    
    # Set up sector structure
    sectors = {
        'Equity': ['SPY', 'QQQ', 'IWM'],
        'Safe Haven': ['GLD', 'TLT']
    }
    
    correlation.set_sector_structure(
        sectors,
        intra_sector_correlation=0.8,
        inter_sector_correlation=-0.3  # Negative correlation between equities and safe havens
    )
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation.get_correlation_dataframe().round(2))
    
    # Generate correlated price data
    price_data = correlated_generator.generate_correlated_series(
        correlation,
        days=252,
        volatilities={'SPY': 0.01, 'QQQ': 0.015, 'IWM': 0.012, 'GLD': 0.008, 'TLT': 0.007}
    )
    
    # Print summary for each asset
    print("\nGenerated correlated price series:")
    for asset, df in price_data.items():
        returns = df['close'].pct_change().dropna()
        print(f"{asset}: {len(df)} days, Volatility: {returns.std()*100:.2f}%, "
              f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
    
    # Save data
    correlated_generator.save_to_csv(price_data, os.path.join(output_dir, "correlated"))
    
    # Calculate actual correlation of returns
    returns = pd.DataFrame({
        asset: df['close'].pct_change().dropna()
        for asset, df in price_data.items()
    })
    
    return_corr = returns.corr()
    print("\nActual Return Correlation:")
    print(return_corr.round(2))
    
    print(f"\nAll data saved to: {output_dir}")


if __name__ == "__main__":
    print("Standalone Synthetic Market Generator")
    print("====================================")
    
    try:
        run_example()
        print("\nExample completed successfully!")
    except Exception as e:
        print(f"\nError running example: {e}")
        import traceback
        traceback.print_exc()
