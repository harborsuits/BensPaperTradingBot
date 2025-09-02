#!/usr/bin/env python3
"""
Market Data Generator for Testing

This module provides functionality to generate synthetic market data for testing
strategies, optimizers, and the autonomous engine without external dependencies.
"""

import random
import math
import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

class MarketDataGenerator:
    """Generates synthetic market data for testing."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the market data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Default parameters
        self.default_params = {
            'start_price': 100.0,
            'volatility': 0.2,           # Annual volatility
            'drift': 0.05,               # Annual drift (expected return)
            'trading_days': 252,         # Trading days per year
            'mean_reversion': 0.1,       # Mean reversion strength
            'jump_probability': 0.01,    # Probability of a price jump
            'max_jump_size': 0.05,       # Maximum jump size as fraction of price
            'correlation': 0.3,          # Default correlation between assets
        }
    
    def generate_price_series(self, 
                             days: int = 252, 
                             params: Optional[Dict[str, float]] = None) -> List[float]:
        """
        Generate a synthetic price series using a geometric Brownian motion model
        with occasional jumps and mean reversion.
        
        Args:
            days: Number of days to generate
            params: Custom parameters to override defaults
            
        Returns:
            List of daily prices
        """
        # Merge default params with any custom params
        p = self.default_params.copy()
        if params:
            p.update(params)
            
        prices = [p['start_price']]
        daily_drift = p['drift'] / p['trading_days']
        daily_vol = p['volatility'] / math.sqrt(p['trading_days'])
        
        for _ in range(days):
            # Get previous price
            prev_price = prices[-1]
            
            # Calculate random shock
            random_shock = np.random.normal(0, 1) * daily_vol
            
            # Add mean reversion component
            reversion = p['mean_reversion'] * (p['start_price'] - prev_price) / p['start_price']
            
            # Calculate price change
            change = daily_drift + random_shock + reversion
            
            # Add occasional jumps
            if random.random() < p['jump_probability']:
                # Jump direction (up or down)
                jump_direction = 1 if random.random() > 0.5 else -1
                jump_size = random.random() * p['max_jump_size'] * jump_direction
                change += jump_size
                
            # Calculate new price and ensure it's positive
            new_price = prev_price * (1 + change)
            new_price = max(new_price, 0.01)  # Ensure price doesn't go to zero
            
            prices.append(new_price)
            
        return prices
    
    def generate_ohlc(self, 
                     days: int = 252, 
                     params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate OHLC (Open, High, Low, Close) data.
        
        Args:
            days: Number of days to generate
            params: Custom parameters to override defaults
            
        Returns:
            DataFrame with OHLC data
        """
        # Generate closing prices
        closes = self.generate_price_series(days, params)
        
        # Generate rest of OHLC based on close prices
        dates = []
        opens = []
        highs = []
        lows = []
        volumes = []
        
        # Start date (use yesterday to avoid future dates)
        start_date = datetime.datetime.now().date() - datetime.timedelta(days=days)
        
        for i in range(days):
            # Date
            current_date = start_date + datetime.timedelta(days=i)
            # Skip weekends
            if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                continue
                
            dates.append(current_date)
            
            # Close price for this day
            close = closes[i]
            
            # Open typically near previous close
            prev_close = closes[i-1] if i > 0 else close
            opens.append(prev_close * (1 + np.random.normal(0, 0.005)))
            
            # High and Low
            daily_range = close * np.random.uniform(0.01, 0.03)  # 1-3% range
            high = max(close, opens[-1]) + daily_range / 2
            low = min(close, opens[-1]) - daily_range / 2
            low = max(low, 0.01)  # Ensure low doesn't go negative
            
            highs.append(high)
            lows.append(low)
            
            # Volume
            avg_volume = 1000000  # Base average volume
            # More volume on volatile days
            volatility_factor = abs(close - opens[-1]) / opens[-1]
            volume = int(avg_volume * (1 + 10 * volatility_factor) * np.random.lognormal(0, 0.5))
            volumes.append(volume)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': closes[:len(dates)],
            'Volume': volumes
        })
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        return df
    
    def generate_correlated_assets(self, 
                                   symbols: List[str], 
                                   days: int = 252,
                                   correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate correlated price data for multiple assets.
        
        Args:
            symbols: List of symbols to generate data for
            days: Number of days to generate
            correlation_matrix: Custom correlation matrix (must match symbols length)
            
        Returns:
            Dictionary of DataFrames, one for each symbol
        """
        n = len(symbols)
        
        # If no correlation matrix provided, create one with default correlation
        if correlation_matrix is None:
            # Start with identity matrix
            correlation_matrix = np.eye(n)
            # Fill upper and lower triangles with default correlation
            correlation_matrix[np.triu_indices(n, 1)] = self.default_params['correlation']
            correlation_matrix[np.tril_indices(n, -1)] = self.default_params['correlation']
            
        # Ensure the matrix is valid (positive semi-definite)
        # This is a simple approach - in practice you might need more sophisticated adjustments
        eigen_vals, _ = np.linalg.eigh(correlation_matrix)
        if np.any(eigen_vals < 0):
            # Make it positive semi-definite
            nearest_corr = correlation_matrix.copy()
            eigen_vals[eigen_vals < 0] = 0.0001
            nearest_corr = nearest_corr.dot(np.diag(eigen_vals)).dot(nearest_corr.T)
            # Re-normalize to ensure diagonal is 1
            d = np.sqrt(np.diag(nearest_corr))
            nearest_corr = nearest_corr / np.outer(d, d)
            correlation_matrix = nearest_corr
            
        # Generate correlated random shocks
        random_shocks = np.random.multivariate_normal(
            mean=np.zeros(n),
            cov=correlation_matrix,
            size=days
        )
        
        # Generate asset prices
        asset_data = {}
        
        for i, symbol in enumerate(symbols):
            # Custom parameters for this asset
            params = self.default_params.copy()
            # Randomize parameters slightly for each asset
            params['start_price'] = random.uniform(50, 200)
            params['volatility'] = random.uniform(0.15, 0.35)
            params['drift'] = random.uniform(0.03, 0.08)
            
            # Generate base time series
            closes = [params['start_price']]
            daily_drift = params['drift'] / params['trading_days']
            daily_vol = params['volatility'] / math.sqrt(params['trading_days'])
            
            for j in range(days-1):
                prev_price = closes[-1]
                # Use pre-generated correlated shock
                random_shock = random_shocks[j, i] * daily_vol
                
                # Mean reversion
                reversion = params['mean_reversion'] * (params['start_price'] - prev_price) / params['start_price']
                
                # Calculate price change
                change = daily_drift + random_shock + reversion
                
                # Add occasional jumps
                if random.random() < params['jump_probability']:
                    jump_direction = 1 if random.random() > 0.5 else -1
                    jump_size = random.random() * params['max_jump_size'] * jump_direction
                    change += jump_size
                    
                # Calculate new price
                new_price = prev_price * (1 + change)
                new_price = max(new_price, 0.01)
                
                closes.append(new_price)
            
            # Generate OHLC
            df = self._create_ohlc_from_closes(closes, days)
            asset_data[symbol] = df
            
        return asset_data
    
    def _create_ohlc_from_closes(self, closes: List[float], days: int) -> pd.DataFrame:
        """Helper to create OHLC data from a list of close prices."""
        # Start date (use yesterday to avoid future dates)
        start_date = datetime.datetime.now().date() - datetime.timedelta(days=days)
        
        dates = []
        opens = []
        highs = []
        lows = []
        volumes = []
        filtered_closes = []
        
        day_counter = 0
        for i in range(days):
            # Date
            current_date = start_date + datetime.timedelta(days=i)
            # Skip weekends
            if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                continue
                
            dates.append(current_date)
            
            # Get close for this trading day
            if day_counter < len(closes):
                close = closes[day_counter]
                filtered_closes.append(close)
                
                # Open price based on previous close
                prev_close = closes[day_counter-1] if day_counter > 0 else close
                opens.append(prev_close * (1 + np.random.normal(0, 0.005)))
                
                # High and Low
                daily_range = close * np.random.uniform(0.01, 0.03)
                high = max(close, opens[-1]) + daily_range / 2
                low = min(close, opens[-1]) - daily_range / 2
                low = max(low, 0.01)  # Ensure low doesn't go negative
                
                highs.append(high)
                lows.append(low)
                
                # Volume
                avg_volume = 1000000
                volatility_factor = abs(close - opens[-1]) / opens[-1]
                volume = int(avg_volume * (1 + 10 * volatility_factor) * np.random.lognormal(0, 0.5))
                volumes.append(volume)
                
                day_counter += 1
            
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': filtered_closes,
            'Volume': volumes
        })
        
        # Set Date as index
        df.set_index('Date', inplace=True)
        
        return df
    
    def generate_options_data(self, 
                             underlying: pd.DataFrame, 
                             strike_range: List[float] = [0.8, 1.2],
                             expiry_days: List[int] = [30, 60, 90],
                             risk_free_rate: float = 0.02) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        """
        Generate options data for an underlying asset.
        
        Args:
            underlying: DataFrame with OHLC data for the underlying
            strike_range: Range of strikes as multipliers of current price [min, max]
            expiry_days: List of days to expiration
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Nested dictionary: {expiry: {strike: {call: df, put: df}}}
        """
        from scipy.stats import norm
        
        # Black-Scholes formula for option pricing
        def black_scholes_call(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            
        def black_scholes_put(S, K, T, r, sigma):
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        # Implied volatility surface parameters
        def get_implied_vol(S, K, days_to_expiry):
            # Create a simple volatility smile
            # ATM volatility
            atm_vol = 0.2
            # Skew - lower strikes have higher vol (put skew)
            moneyness = K / S
            # Smile component - further from ATM, higher the vol
            smile = 0.1 * abs(moneyness - 1)**2
            # Term structure - longer dated options less volatile
            term_effect = np.exp(-0.5 * days_to_expiry / 365)
            
            vol = atm_vol + smile * term_effect
            # Add some noise
            vol += np.random.normal(0, 0.01)
            return max(vol, 0.05)  # Ensure minimum vol
            
        options_data = {}
        
        # Current price of underlying
        current_price = underlying['Close'].iloc[-1]
        
        # Calculate strike prices
        min_strike = current_price * strike_range[0]
        max_strike = current_price * strike_range[1]
        num_strikes = 11  # Number of strikes to generate
        strike_prices = np.linspace(min_strike, max_strike, num_strikes)
        
        # For each expiration
        for days in expiry_days:
            expiry = f"{days}d"
            options_data[expiry] = {}
            
            # For each strike
            for strike in strike_prices:
                strike_label = f"{strike:.2f}"
                options_data[expiry][strike_label] = {}
                
                # Calculate option data for each date in the underlying
                call_data = []
                put_data = []
                
                for date, row in underlying.iterrows():
                    # Stock price
                    S = row['Close']
                    # Time to expiry in years
                    remaining_days = max(1, days - len(call_data))
                    T = remaining_days / 365
                    
                    # Get implied volatility
                    vol = get_implied_vol(S, strike, remaining_days)
                    
                    # Calculate option prices
                    call_price = black_scholes_call(S, strike, T, risk_free_rate, vol)
                    put_price = black_scholes_put(S, strike, T, risk_free_rate, vol)
                    
                    # Add some bid-ask spread
                    spread = 0.05 * call_price
                    call_bid = call_price - spread/2
                    call_ask = call_price + spread/2
                    
                    spread = 0.05 * put_price
                    put_bid = put_price - spread/2
                    put_ask = put_price + spread/2
                    
                    # Option greeks (simplified)
                    d1 = (np.log(S/strike) + (risk_free_rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                    
                    # Delta
                    call_delta = norm.cdf(d1)
                    put_delta = call_delta - 1
                    
                    # Gamma
                    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
                    
                    # Theta (daily)
                    call_theta = -(S * vol * norm.pdf(d1)) / (2 * np.sqrt(T) * 365)
                    put_theta = call_theta + risk_free_rate * strike * np.exp(-risk_free_rate * T) / 365
                    
                    # Vega (for 1% change in vol)
                    vega = S * np.sqrt(T) * norm.pdf(d1) / 100
                    
                    # Create call data
                    call_data.append({
                        'Date': date,
                        'Underlying': S,
                        'Strike': strike,
                        'DTE': remaining_days,
                        'Bid': call_bid,
                        'Ask': call_ask,
                        'Mid': call_price,
                        'ImpliedVol': vol,
                        'Delta': call_delta,
                        'Gamma': gamma,
                        'Theta': call_theta,
                        'Vega': vega
                    })
                    
                    # Create put data
                    put_data.append({
                        'Date': date,
                        'Underlying': S,
                        'Strike': strike,
                        'DTE': remaining_days,
                        'Bid': put_bid,
                        'Ask': put_ask,
                        'Mid': put_price,
                        'ImpliedVol': vol,
                        'Delta': put_delta,
                        'Gamma': gamma,
                        'Theta': put_theta,
                        'Vega': vega
                    })
                
                # Convert to DataFrames
                call_df = pd.DataFrame(call_data)
                put_df = pd.DataFrame(put_data)
                
                if not call_df.empty and not put_df.empty:
                    call_df.set_index('Date', inplace=True)
                    put_df.set_index('Date', inplace=True)
                    
                    # Store dataframes
                    options_data[expiry][strike_label]['call'] = call_df
                    options_data[expiry][strike_label]['put'] = put_df
                
        return options_data
    
    def generate_test_dataset(self, 
                              num_stocks: int = 10, 
                              days: int = 252,
                              include_options: bool = True) -> Dict[str, Any]:
        """
        Generate a complete test dataset with stocks and options.
        
        Args:
            num_stocks: Number of stocks to generate
            days: Number of days of data to generate
            include_options: Whether to include options data
            
        Returns:
            Dictionary with stock and options data
        """
        # Generate stock symbols
        symbols = [f"TEST{i:02d}" for i in range(1, num_stocks+1)]
        
        # Generate correlated stock data
        stocks = self.generate_correlated_assets(symbols, days)
        
        dataset = {
            'stocks': stocks,
            'metadata': {
                'generated_date': datetime.datetime.now().isoformat(),
                'days': days,
                'num_stocks': num_stocks,
                'include_options': include_options
            }
        }
        
        # Generate options data if requested
        if include_options:
            options = {}
            for symbol, stock_data in stocks.items():
                options[symbol] = self.generate_options_data(stock_data)
            dataset['options'] = options
            
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: str) -> None:
        """
        Save the dataset to a file.
        
        Args:
            dataset: Dataset to save
            filepath: Path to save to
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
    
    @staticmethod
    def load_dataset(filepath: str) -> Dict[str, Any]:
        """
        Load a dataset from a file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            The loaded dataset
        """
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# Example usage
if __name__ == "__main__":
    generator = MarketDataGenerator(seed=42)
    
    # Generate a full test dataset
    dataset = generator.generate_test_dataset(
        num_stocks=5,
        days=252,
        include_options=True
    )
    
    # Save to file
    generator.save_dataset(dataset, "test_market_data.pkl")
    
    # Show summary
    print("Generated test market data:")
    print(f"Stocks: {len(dataset['stocks'])}")
    if 'options' in dataset:
        print(f"Stocks with options: {len(dataset['options'])}")
        # Sample the first stock
        first_stock = list(dataset['options'].keys())[0]
        print(f"Sample options for {first_stock}:")
        for expiry in dataset['options'][first_stock]:
            num_strikes = len(dataset['options'][first_stock][expiry])
            print(f"  {expiry}: {num_strikes} strikes")
