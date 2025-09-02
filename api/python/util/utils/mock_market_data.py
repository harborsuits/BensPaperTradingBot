"""
Mock Market Data Generator

This module generates realistic mock market data for backtesting purposes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("mock_market_data")

def generate_price_series(initial_price=100.0, days=180, volatility=0.01, drift=0.0002, 
                         regime_changes=True, seed=None):
    """
    Generate a realistic price series with specified parameters.
    
    Args:
        initial_price: Starting price
        days: Number of days to generate
        volatility: Daily volatility parameter
        drift: Daily drift parameter (positive for uptrend, negative for downtrend)
        regime_changes: Whether to include market regime changes
        seed: Random seed for reproducibility
        
    Returns:
        Dict with OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate daily returns with geometric brownian motion
    daily_returns = np.exp(drift + volatility * np.random.normal(0, 1, days))
    
    # Add regime changes if requested
    if regime_changes:
        # Generate 2-3 regime changes
        num_regimes = np.random.randint(2, 4)
        regime_points = sorted(np.random.choice(range(days), num_regimes, replace=False))
        
        # For each regime, adjust the drift
        current_idx = 0
        for regime_idx in regime_points:
            # How many days in this regime
            regime_length = regime_idx - current_idx
            
            # Randomly select a regime type
            regime_type = np.random.choice(['bullish', 'bearish', 'volatile', 'sideways'])
            
            if regime_type == 'bullish':
                regime_drift = 0.001  # Strong positive drift
                regime_vol = 0.008    # Lower volatility
            elif regime_type == 'bearish':
                regime_drift = -0.001  # Negative drift
                regime_vol = 0.012     # Higher volatility
            elif regime_type == 'volatile':
                regime_drift = 0.0     # No drift
                regime_vol = 0.025     # High volatility
            else:  # sideways
                regime_drift = 0.0     # No drift
                regime_vol = 0.005     # Low volatility
            
            # Replace the returns for this regime
            regime_returns = np.exp(regime_drift + regime_vol * np.random.normal(0, 1, regime_length))
            daily_returns[current_idx:regime_idx] = regime_returns
            
            current_idx = regime_idx
        
        # Final regime
        if current_idx < days:
            regime_length = days - current_idx
            regime_type = np.random.choice(['bullish', 'bearish', 'volatile', 'sideways'])
            
            if regime_type == 'bullish':
                regime_drift = 0.001
                regime_vol = 0.008
            elif regime_type == 'bearish':
                regime_drift = -0.001
                regime_vol = 0.012
            elif regime_type == 'volatile':
                regime_drift = 0.0
                regime_vol = 0.025
            else:  # sideways
                regime_drift = 0.0
                regime_vol = 0.005
            
            regime_returns = np.exp(regime_drift + regime_vol * np.random.normal(0, 1, regime_length))
            daily_returns[current_idx:] = regime_returns
    
    # Calculate price series
    prices = initial_price * np.cumprod(daily_returns)
    
    # Generate OHLC
    opens = prices * np.exp(volatility * 0.5 * np.random.normal(0, 1, days))
    highs = np.maximum(opens, prices) * np.exp(volatility * 0.5 * np.random.normal(0.5, 0.5, days))
    lows = np.minimum(opens, prices) * np.exp(volatility * 0.5 * np.random.normal(-0.5, 0.5, days))
    closes = prices
    
    # Generate volume
    avg_volume = 1000000  # 1M shares per day
    volume_variation = 0.3  # 30% variation in volume
    volumes = avg_volume * np.exp(volume_variation * np.random.normal(0, 1, days))
    volumes = volumes.astype(int)
    
    # Ensure highs are always highest, lows always lowest
    highs = np.maximum(np.maximum(opens, closes), highs)
    lows = np.minimum(np.minimum(opens, closes), lows)
    
    return {
        'open': opens.tolist(),
        'high': highs.tolist(),
        'low': lows.tolist(),
        'close': closes.tolist(),
        'volume': volumes.tolist()
    }

def generate_options_data(underlying_price, days_to_expiration=30, strike_spacing=5.0):
    """
    Generate options data for a given underlying price.
    
    Args:
        underlying_price: Current price of the underlying
        days_to_expiration: Days to expiration
        strike_spacing: Spacing between strike prices
        
    Returns:
        Dict with options data
    """
    # Generate strike prices (5-10 strikes above and below current price)
    atm_strike = round(underlying_price / strike_spacing) * strike_spacing
    num_strikes_up = np.random.randint(5, 11)
    num_strikes_down = np.random.randint(5, 11)
    
    strikes = [atm_strike + i * strike_spacing for i in range(-num_strikes_down, num_strikes_up+1)]
    
    # Calculate implied volatility
    base_iv = 0.20 + 0.10 * np.random.random()  # Base IV between 20% and 30%
    iv_skew = 0.01 * np.random.random() * 10    # IV skew factor
    
    # Calculate IV for each strike (smile pattern)
    ivs = {}
    for strike in strikes:
        moneyness = strike / underlying_price - 1.0
        skew_adjustment = iv_skew * (moneyness ** 2)  # Parabolic skew (smile)
        ivs[str(strike)] = base_iv + skew_adjustment
    
    # Calculate IV rank and percentile
    iv_rank = np.random.random()  # 0-1
    iv_percentile = iv_rank * 100
    
    # Generate option prices
    call_prices = {}
    put_prices = {}
    call_deltas = {}
    put_deltas = {}
    prob_otm_call = {}
    prob_otm_put = {}
    
    for strike in strikes:
        strike_iv = ivs[str(strike)]
        
        # Simple Black-Scholes approximation
        time_factor = days_to_expiration / 365
        vol_factor = strike_iv * np.sqrt(time_factor)
        
        # Call price
        moneyness = underlying_price / strike
        if moneyness > 1.0:
            intrinsic = underlying_price - strike
        else:
            intrinsic = 0
        
        time_value = underlying_price * 0.4 * vol_factor * np.exp(-0.5 * ((np.log(moneyness) / vol_factor) ** 2))
        call_price = max(intrinsic + time_value, 0.01)
        call_prices[str(strike)] = call_price
        
        # Put price
        if moneyness < 1.0:
            intrinsic = strike - underlying_price
        else:
            intrinsic = 0
        
        time_value = underlying_price * 0.4 * vol_factor * np.exp(-0.5 * ((np.log(moneyness) / vol_factor) ** 2))
        put_price = max(intrinsic + time_value, 0.01)
        put_prices[str(strike)] = put_price
        
        # Approximate delta
        d1 = (np.log(moneyness) + (0.5 * strike_iv**2) * time_factor) / (strike_iv * np.sqrt(time_factor))
        norm_d1 = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * d1))  # Poor man's normal CDF
        
        call_delta = norm_d1
        put_delta = norm_d1 - 1
        
        call_deltas[str(strike)] = call_delta
        put_deltas[str(strike)] = put_delta
        
        # Probabilities
        prob_otm_call[str(strike)] = 1 - norm_d1
        prob_otm_put[str(strike)] = norm_d1
    
    # Construct options data
    options_data = {
        'strikes': strikes,
        'iv_percentile': iv_percentile,
        'iv_rank': iv_rank,
        'prices': {
            'call': call_prices,
            'put': put_prices
        },
        'deltas': {
            'call': call_deltas,
            'put': put_deltas
        },
        'probability_otm': {
            'call': prob_otm_call,
            'put': prob_otm_put
        }
    }
    
    return options_data

def generate_mock_market_data(symbols=None, start_date=None, end_date=None, days=180, 
                            include_options=True, seed=None):
    """
    Generate mock market data for a set of symbols.
    
    Args:
        symbols: List of symbols to generate data for
        start_date: Start date
        end_date: End date
        days: Number of days (if start_date and end_date not provided)
        include_options: Whether to include options data
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of market data by date
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default symbols if not provided
    if symbols is None:
        symbols = ['SPY', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA']
    
    # Generate date range
    if start_date is None or end_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create a date range, excluding weekends
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday to Friday
            all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Generate market data for each symbol
    market_data = {}
    
    # Different starting prices for different symbols
    base_prices = {
        'SPY': 450.0,
        'AAPL': 180.0,
        'MSFT': 350.0,
        'GOOGL': 140.0,
        'AMZN': 180.0,
        'META': 350.0,
        'TSLA': 240.0,
        'NVDA': 450.0
    }
    
    # Market regime definition
    regime_periods = []
    current_idx = 0
    while current_idx < len(all_dates):
        # Regime length between 15 and 40 days
        regime_length = np.random.randint(15, 41)
        end_idx = min(current_idx + regime_length, len(all_dates))
        
        # Randomly select a regime type
        regime_type = np.random.choice(['bullish', 'bearish', 'volatile', 'sideways'])
        
        # Add this regime period
        regime_periods.append({
            'start_idx': current_idx,
            'end_idx': end_idx,
            'regime': regime_type
        })
        
        current_idx = end_idx
    
    # Generate price and options data for each date
    for date_idx, date in enumerate(all_dates):
        date_str = date.strftime('%Y-%m-%d')
        market_data[date_str] = {}
        
        # Find current regime
        current_regime = None
        for period in regime_periods:
            if period['start_idx'] <= date_idx < period['end_idx']:
                current_regime = period['regime']
                break
        
        # Add market regime to the data
        market_context = {
            'market_regime': current_regime,
            'trend_strength': np.random.random(),
            'volatility_index': 10 + 20 * np.random.random()
        }
        market_data[date_str]['market_context'] = market_context
        
        # Generate data for each symbol
        for symbol in symbols:
            # If this is the first date, generate price series for the entire period
            if date_idx == 0:
                # Generate daily data with appropriate parameters based on regime
                initial_price = base_prices.get(symbol, 100.0)
                
                # Different volatility and drift for different symbols
                base_volatility = 0.01 + 0.005 * np.random.random()
                base_drift = 0.0002 + 0.0002 * np.random.random()
                
                price_series = generate_price_series(
                    initial_price=initial_price,
                    days=len(all_dates),
                    volatility=base_volatility,
                    drift=base_drift,
                    regime_changes=True,
                    seed=seed+hash(symbol) if seed is not None else None
                )
                
                # Store the series for future dates
                market_data[f"__series_{symbol}"] = price_series
            
            # Get the price data for this date
            price_series = market_data[f"__series_{symbol}"]
            
            # Create daily data
            daily_data = {
                'open': price_series['open'][date_idx],
                'high': price_series['high'][date_idx],
                'low': price_series['low'][date_idx],
                'close': price_series['close'][date_idx],
                'volume': price_series['volume'][date_idx],
                'date': date_str
            }
            
            # Add options data if requested
            if include_options:
                days_to_expiration = 30  # Standard 30 days
                options_data = generate_options_data(
                    underlying_price=daily_data['close'],
                    days_to_expiration=days_to_expiration
                )
                daily_data['options'] = options_data
            
            # Store the data
            market_data[date_str][symbol] = daily_data
    
    # Remove temporary series data
    for symbol in symbols:
        if f"__series_{symbol}" in market_data:
            del market_data[f"__series_{symbol}"]
    
    logger.info(f"Generated mock market data for {len(symbols)} symbols and {len(all_dates)} trading days")
    return market_data 