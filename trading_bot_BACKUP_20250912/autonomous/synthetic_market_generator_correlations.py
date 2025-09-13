#!/usr/bin/env python3
"""
Synthetic Market Generator - Correlations

This module extends the SyntheticMarketGenerator with capabilities for 
generating correlated price movements across multiple assets. This is critical
for testing strategies that rely on correlations, sector rotation, and
cross-asset relationships.

It builds upon our synthetic_market_generator.py implementation, introducing:
1. Multi-asset price generation with configurable correlation matrices
2. Sector-based correlation structures
3. Dynamic correlation regimes that can shift over time
4. Event-based correlation shocks

This directly integrates with our correlation monitoring and regime detection
system for comprehensive strategy testing.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Set

# Import base generator
from trading_bot.autonomous.synthetic_market_generator import (
    SyntheticMarketGenerator, PriceSeriesGenerator, MarketRegimeType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrelationStructure:
    """
    Defines and manages correlation structures between assets.
    
    This class provides methods to create, modify, and apply correlation
    structures to synthetic price data.
    """
    
    def __init__(
        self,
        assets: List[str],
        base_correlation: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize the correlation structure.
        
        Args:
            assets: List of asset symbols
            base_correlation: Default correlation between assets
            seed: Random seed for reproducibility
        """
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
    
    def set_pairwise_correlation(
        self,
        asset1: str,
        asset2: str,
        correlation: float
    ):
        """
        Set correlation between a pair of assets.
        
        Args:
            asset1: First asset symbol
            asset2: Second asset symbol
            correlation: Correlation coefficient (-1 to 1)
        """
        try:
            idx1 = self.assets.index(asset1)
            idx2 = self.assets.index(asset2)
            
            # Set correlation symmetrically
            self.correlation_matrix[idx1, idx2] = correlation
            self.correlation_matrix[idx2, idx1] = correlation
            
        except ValueError:
            logger.error(f"Asset not found in correlation structure: {asset1} or {asset2}")
    
    def set_group_correlation(
        self,
        group1: List[str],
        group2: List[str],
        correlation: float
    ):
        """
        Set correlation between two groups of assets.
        
        Args:
            group1: First group of asset symbols
            group2: Second group of asset symbols
            correlation: Correlation coefficient (-1 to 1)
        """
        for asset1 in group1:
            for asset2 in group2:
                self.set_pairwise_correlation(asset1, asset2, correlation)
    
    def set_sector_structure(
        self,
        sectors: Dict[str, List[str]],
        intra_sector_correlation: float = 0.7,
        inter_sector_correlation: float = 0.3
    ):
        """
        Set up a sector-based correlation structure.
        
        Args:
            sectors: Dictionary mapping sector names to lists of assets
            intra_sector_correlation: Correlation within the same sector
            inter_sector_correlation: Correlation between different sectors
        """
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
                        
                    self.set_pairwise_correlation(
                        asset1, asset2, intra_sector_correlation
                    )
    
    def apply_random_variation(self, variation: float = 0.1):
        """
        Apply random variation to the correlation matrix.
        
        Args:
            variation: Maximum amount of random variation
        """
        # Generate random variations
        random_variation = np.random.uniform(
            -variation, variation, 
            (self.num_assets, self.num_assets)
        )
        
        # Make it symmetric
        random_variation = (random_variation + random_variation.T) / 2
        
        # Set diagonal to 0 (no variation in self-correlation)
        np.fill_diagonal(random_variation, 0.0)
        
        # Apply variation
        self.correlation_matrix += random_variation
        
        # Ensure correlations remain in valid range [-1, 1]
        self.correlation_matrix = np.clip(self.correlation_matrix, -1.0, 1.0)
        
        # Ensure diagonal is 1
        np.fill_diagonal(self.correlation_matrix, 1.0)
    
    def ensure_positive_definite(self):
        """
        Ensure the correlation matrix is positive definite
        (valid for Cholesky decomposition).
        """
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
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Get the current correlation matrix.
        
        Returns:
            Correlation matrix as numpy array
        """
        return self.correlation_matrix
    
    def get_correlation_dataframe(self) -> pd.DataFrame:
        """
        Get the correlation matrix as a labeled DataFrame.
        
        Returns:
            Correlation matrix as pandas DataFrame
        """
        return pd.DataFrame(
            self.correlation_matrix,
            index=self.assets,
            columns=self.assets
        )


class CorrelatedMarketGenerator:
    """
    Generates correlated synthetic market data for multiple assets.
    
    This class builds on the SyntheticMarketGenerator to create price
    series for multiple assets with realistic correlation structures.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the correlated market generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.market_generator = SyntheticMarketGenerator(seed=seed)
    
    def generate_correlated_series(
        self,
        correlation_structure: CorrelationStructure,
        days: int = 252,
        base_prices: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        drifts: Optional[Dict[str, float]] = None,
        include_weekends: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate correlated price series for multiple assets.
        
        Args:
            correlation_structure: Correlation structure defining relationships
            days: Number of days to generate
            base_prices: Dictionary of starting prices by asset
            volatilities: Dictionary of volatilities by asset
            drifts: Dictionary of drift terms by asset
            include_weekends: Whether to include weekend days
            
        Returns:
            Dictionary mapping asset symbols to price DataFrames
        """
        # Get assets and correlation matrix
        assets = correlation_structure.assets
        correlation_matrix = correlation_structure.get_correlation_matrix()
        
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
        
        if include_weekends:
            # Simple date range including all days
            dates = [end_date - timedelta(days=i) for i in range(days)]
        else:
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
            
            # High is typically 0.5% to 1.5% above the higher of open/close
            # Low is typically 0.5% to 1.5% below the lower of open/close
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
    
    def generate_changing_correlation_regime(
        self,
        correlation_structure: CorrelationStructure,
        target_correlation_structure: CorrelationStructure,
        days: int = 252,
        regime_change_start: int = 126,  # Day when correlation begins changing
        regime_change_duration: int = 20,  # Days to transition between regimes
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate price series with changing correlation regimes.
        
        Args:
            correlation_structure: Initial correlation structure
            target_correlation_structure: Target correlation structure
            days: Total number of days
            regime_change_start: Day when correlation begins changing
            regime_change_duration: Number of days for correlation transition
            **kwargs: Additional arguments for generate_correlated_series
            
        Returns:
            Dictionary mapping asset symbols to price DataFrames
        """
        # Check that both structures have the same assets
        if set(correlation_structure.assets) != set(target_correlation_structure.assets):
            raise ValueError("Initial and target correlation structures must have the same assets")
        
        assets = correlation_structure.assets
        
        # Generate first part with initial correlation
        if regime_change_start > 0:
            initial_data = self.generate_correlated_series(
                correlation_structure,
                days=regime_change_start,
                **kwargs
            )
        else:
            # Create empty initial data structures
            initial_data = {asset: pd.DataFrame() for asset in assets}
        
        # Generate data for transition period
        current_corr = correlation_structure.get_correlation_matrix().copy()
        target_corr = target_correlation_structure.get_correlation_matrix().copy()
        
        transition_data = {}
        
        if regime_change_duration > 0:
            # Create transition structures
            for day in range(regime_change_duration):
                # Calculate interpolated correlation
                progress = (day + 1) / regime_change_duration
                interpolated_corr = (1 - progress) * current_corr + progress * target_corr
                
                # Create temporary correlation structure
                temp_structure = CorrelationStructure(assets, seed=self.seed)
                temp_structure.correlation_matrix = interpolated_corr
                
                # Generate one day of data
                day_data = self.generate_correlated_series(
                    temp_structure,
                    days=1,
                    **kwargs
                )
                
                # Store in transition data
                for asset in assets:
                    if asset not in transition_data:
                        transition_data[asset] = day_data[asset].copy()
                    else:
                        transition_data[asset] = pd.concat([
                            transition_data[asset],
                            day_data[asset]
                        ])
        
        # Generate final part with target correlation
        remaining_days = days - regime_change_start - regime_change_duration
        if remaining_days > 0:
            final_data = self.generate_correlated_series(
                target_correlation_structure,
                days=remaining_days,
                **kwargs
            )
        else:
            # Create empty final data structures
            final_data = {asset: pd.DataFrame() for asset in assets}
        
        # Combine the parts for each asset
        result = {}
        for asset in assets:
            # Get parts for this asset
            initial = initial_data.get(asset, pd.DataFrame())
            transition = transition_data.get(asset, pd.DataFrame())
            final = final_data.get(asset, pd.DataFrame())
            
            # Combine
            result[asset] = pd.concat([initial, transition, final])
            
            # Ensure the dates are connected properly
            # This may require adjusting dates for a seamless series
            if len(result[asset]) > 0:
                # Reset index to sequential dates
                old_index = result[asset].index
                
                # Create new date index
                end_date = datetime.now()
                
                if kwargs.get("include_weekends", False):
                    # Simple date range including all days
                    new_dates = [end_date - timedelta(days=i) for i in range(len(result[asset]))]
                else:
                    # Skip weekends
                    new_dates = []
                    current_date = end_date
                    while len(new_dates) < len(result[asset]):
                        if current_date.weekday() < 5:  # Monday=0, Sunday=6
                            new_dates.append(current_date)
                        current_date = current_date - timedelta(days=1)
                
                # Reverse to get ascending order
                new_dates = new_dates[::-1]
                
                # Reindex
                result[asset].index = new_dates
        
        return result
    
    def apply_market_regime(
        self,
        price_data: Dict[str, pd.DataFrame],
        regime: MarketRegimeType,
        regime_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply a market regime to correlated price data.
        
        Args:
            price_data: Dictionary of price DataFrames by asset
            regime: Market regime to apply
            regime_params: Additional parameters for the regime
            
        Returns:
            Modified price data with regime applied
        """
        # Create a copy to avoid modifying the original
        result = {asset: df.copy() for asset, df in price_data.items()}
        
        # Set default parameters if not provided
        if regime_params is None:
            regime_params = {}
        
        # Apply regime to each asset, with some variation between assets
        generator = self.market_generator.generator
        
        for asset, df in result.items():
            # Apply asset-specific variations
            if regime == MarketRegimeType.BULLISH:
                # Upward trend with asset-specific strength
                trend_strength = regime_params.get('trend_strength', 0.001)
                # Add some variation between assets
                asset_trend = trend_strength * np.random.uniform(0.7, 1.3)
                result[asset] = generator.apply_trend(
                    df, trend_strength=asset_trend, trend_direction=1
                )
            
            elif regime == MarketRegimeType.BEARISH:
                # Downward trend with asset-specific strength
                trend_strength = regime_params.get('trend_strength', 0.001)
                # Add some variation between assets
                asset_trend = trend_strength * np.random.uniform(0.7, 1.3)
                result[asset] = generator.apply_trend(
                    df, trend_strength=asset_trend, trend_direction=-1
                )
            
            elif regime == MarketRegimeType.VOLATILE:
                # High volatility with asset-specific factor
                vol_factor = regime_params.get('volatility_factor', 2.0)
                # Add some variation between assets
                asset_vol = vol_factor * np.random.uniform(0.8, 1.2)
                result[asset] = generator.apply_volatility_regime(
                    df, volatility_factor=asset_vol
                )
            
            elif regime == MarketRegimeType.MEAN_REVERTING:
                # Mean reversion with asset-specific strength
                strength = regime_params.get('reversion_strength', 0.3)
                window = regime_params.get('window', 15)
                # Add some variation between assets
                asset_strength = strength * np.random.uniform(0.8, 1.2)
                result[asset] = generator.apply_mean_reversion(
                    df, reversion_strength=asset_strength, window=window
                )
            
            # Add other regime types as needed
        
        return result
    
    def save_to_csv(
        self,
        price_data: Dict[str, pd.DataFrame],
        directory: str
    ):
        """
        Save correlated price data to CSV files.
        
        Args:
            price_data: Dictionary of price DataFrames by asset
            directory: Output directory
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save each asset's data
        for asset, df in price_data.items():
            filename = os.path.join(directory, f"{asset}.csv")
            df.to_csv(filename)
        
        logger.info(f"Saved {len(price_data)} synthetic price series to {directory}")
    
    def load_from_csv(
        self,
        directory: str,
        assets: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load correlated price data from CSV files.
        
        Args:
            directory: Input directory
            assets: List of asset symbols to load (None = all)
            
        Returns:
            Dictionary of price DataFrames by asset
        """
        result = {}
        
        # List CSV files in directory
        files = [f for f in os.listdir(directory) if f.endswith('.csv')]
        
        # Filter by requested assets if specified
        if assets is not None:
            files = [f for f in files if f.split('.')[0] in assets]
        
        # Load each file
        for file in files:
            asset = file.split('.')[0]
            filename = os.path.join(directory, file)
            
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            result[asset] = df
        
        logger.info(f"Loaded {len(result)} synthetic price series from {directory}")
        return result


# Singleton instance
_correlated_market_generator = None


def get_correlated_market_generator() -> CorrelatedMarketGenerator:
    """
    Get singleton instance of CorrelatedMarketGenerator.
    
    Returns:
        CorrelatedMarketGenerator instance
    """
    global _correlated_market_generator
    
    if _correlated_market_generator is None:
        _correlated_market_generator = CorrelatedMarketGenerator()
    
    return _correlated_market_generator


if __name__ == "__main__":
    # Example usage
    generator = get_correlated_market_generator()
    
    # Define asset list
    assets = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    
    # Create correlation structure
    correlation = CorrelationStructure(assets)
    
    # Set up sector structure
    sectors = {
        'Equity': ['SPY', 'QQQ', 'IWM'],
        'Safe Haven': ['GLD', 'TLT']
    }
    
    correlation.set_sector_structure(
        sectors,
        intra_sector_correlation=0.8,
        inter_sector_correlation=0.1
    )
    
    # Apply some random variation
    correlation.apply_random_variation(0.1)
    
    # Generate correlated price data
    price_data = generator.generate_correlated_series(
        correlation,
        days=252,
        volatilities={'SPY': 0.01, 'QQQ': 0.015, 'IWM': 0.012, 'GLD': 0.008, 'TLT': 0.007}
    )
    
    # Print summary
    for asset, df in price_data.items():
        returns = df['close'].pct_change().dropna()
        print(f"{asset}: {len(df)} days, Volatility: {returns.std()*100:.2f}%, "
              f"Return: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.2f}%")
