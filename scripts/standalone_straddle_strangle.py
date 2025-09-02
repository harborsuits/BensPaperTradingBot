#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Straddle Strangle Strategy Implementation

This is a self-contained implementation of the Straddle/Strangle strategy 
with no dependencies on the larger trading framework, to avoid circular imports.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandaloneStraddleStrangleStrategy:
    """
    Standalone implementation of the Straddle/Strangle Options Strategy.
    
    This strategy involves buying both call and put options to profit from significant
    price movements in either direction. It's a volatility strategy that performs well
    in environments with large price swings or before major market events.
    
    Key characteristics:
    - Unlimited profit potential in either direction
    - Limited risk (premium paid)
    - Requires significant price movement to be profitable
    - Benefits from volatility expansion
    - Suffers from time decay if the underlying doesn't move
    - High cost compared to directional strategies
    """
    
    def __init__(self, strategy_id=None, name=None, parameters=None):
        """
        Initialize the Straddle/Strangle strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
        """
        self.strategy_id = strategy_id or "straddle_strangle_01"
        self.name = name or "Straddle/Strangle Strategy"
        
        # Default parameters
        self.default_params = {
            'volatility_threshold': 0.20,        # Historical volatility threshold
            'implied_volatility_rank_min': 30,   # Minimum IV rank to consider
            'atm_threshold': 0.03,               # How close to ATM for straddle
            'strangle_width_pct': 0.05,          # Strike width for strangle as % of price
            'min_dte': 20,                       # Minimum days to expiration
            'max_dte': 45,                       # Maximum days to expiration
            'profit_target_pct': 35,             # Profit target as percent of premium
            'stop_loss_pct': 60,                 # Stop loss as percent of premium
            'max_positions': 5,                  # Maximum positions
            'position_size_pct': 5,              # Position size as portfolio percentage
            'strategy_variant': 'adaptive',      # 'straddle', 'strangle', or 'adaptive'
            'iv_percentile_threshold': 30,       # IV percentile threshold for strategy selection
            'vix_threshold': 18,                 # VIX threshold for strategy selection
            'event_window_days': 5               # Days before earnings/events to consider
        }
        
        # Apply any provided parameters
        self.params = self.default_params.copy()
        if parameters:
            self.params.update(parameters)
            
        # Strategy-specific tracking
        self.positions = {}                    # Track current positions
        self.universe = ["SPY", "QQQ", "AAPL", "MSFT", "TSLA"]  # Default universe
        
        logger.info(f"Initialized {self.name} with ID {self.strategy_id}")
        
    def set_universe(self, symbols: List[str]):
        """Set the universe of tradable assets"""
        self.universe = symbols
        logger.info(f"Universe set to {len(symbols)} symbols")
        return True
        
    def generate_signals(self, market_data, option_chains=None):
        """
        Generate straddle/strangle signals based on market data and option chains.
        
        Args:
            market_data: Market data for analysis
            option_chains: Option chain data
            
        Returns:
            List of signal dictionaries
        """
        logger.info("Generating signals...")
        
        signals = []
        
        # Iterate through our universe
        for symbol in self.universe:
            logger.info(f"Analyzing {symbol}...")
            
            # Get historical data for the symbol
            if isinstance(market_data, dict):
                historical_data = market_data.get(symbol)
            else:
                # Try to get data from a MarketData object
                try:
                    historical_data = market_data.get_historical_data(symbol)
                except (AttributeError, TypeError):
                    logger.warning(f"Could not get historical data for {symbol}")
                    continue
                    
            if historical_data is None or len(historical_data) < 20:
                logger.warning(f"Insufficient historical data for {symbol}")
                continue
                
            # Check if we have option chain data
            symbol_options = None
            if option_chains:
                try:
                    symbol_options = option_chains.get_chain_for_symbol(symbol)
                except (AttributeError, TypeError):
                    logger.warning(f"Could not get option chain for {symbol}")
            
            if not symbol_options:
                logger.warning(f"No option chain data for {symbol}")
                continue
                
            # Calculate historical volatility
            hist_volatility = self._get_historical_volatility(historical_data)
            
            # Check if volatility conditions are met
            if hist_volatility < self.params['volatility_threshold']:
                logger.info(f"{symbol} historical volatility {hist_volatility:.2%} below threshold")
                continue
                
            # Decide on straddle or strangle based on strategy variant
            strategy_variant = self._determine_strategy_variant(hist_volatility)
            
            # Get current price
            current_price = historical_data.iloc[-1]['close']
            
            # Find suitable options
            if strategy_variant == 'straddle':
                strategy_data = self._find_straddle(symbol, symbol_options, current_price)
            else:  # strangle
                strategy_data = self._find_strangle(symbol, symbol_options, current_price)
                
            if not strategy_data:
                logger.info(f"Could not find suitable {strategy_variant} for {symbol}")
                continue
                
            # Create signal
            signal = self._create_signal(symbol, strategy_data, current_price, strategy_variant)
            
            if signal:
                signals.append(signal)
                
        logger.info(f"Generated {len(signals)} signals")
        return signals
    
    def _get_historical_volatility(self, data, days=20):
        """
        Calculate historical volatility for a symbol.
        
        Args:
            data: Historical price data
            days: Number of days to calculate over
            
        Returns:
            Historical volatility (annualized)
        """
        if isinstance(data, pd.DataFrame):
            if len(data) < days:
                return 0
                
            # Calculate daily returns
            returns = data['close'].pct_change().dropna().tail(days)
            
            # Calculate annualized volatility
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252)  # Trading days in a year
            
            return annualized_vol
        else:
            return 0
        
    def _determine_strategy_variant(self, volatility):
        """
        Determine whether to use straddle or strangle based on volatility and settings.
        
        Args:
            volatility: Current historical volatility
            
        Returns:
            'straddle' or 'strangle'
        """
        if self.params['strategy_variant'] == 'straddle':
            return 'straddle'
        elif self.params['strategy_variant'] == 'strangle':
            return 'strangle'
        else:  # adaptive
            # For high volatility, prefer strangles (cheaper)
            # For moderate volatility, prefer straddles (more precise)
            if volatility > 0.30:  # Very high volatility
                return 'strangle'
            else:
                return 'straddle'
                
    def _find_straddle(self, symbol, option_chain, current_price):
        """
        Find suitable options for a straddle strategy.
        
        Args:
            symbol: Symbol to trade
            option_chain: Option chain data
            current_price: Current price of the underlying
            
        Returns:
            Dictionary with selected option data
        """
        # In a real implementation, we would:
        # 1. Find options with strike prices close to current price
        # 2. Filter by our DTE requirements
        # 3. Check open interest and volume
        # 4. Select the best pair
        
        # Simplified mock implementation for validation
        result = {
            'strategy_type': 'straddle',
            'symbol': symbol,
            'current_price': current_price,
            'expiration': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'strike': round(current_price, 1),
            'call': {
                'symbol': f"{symbol}C{round(current_price, 1)}",
                'price': current_price * 0.05,  # Mock price
                'delta': 0.50,
                'implied_volatility': 0.25
            },
            'put': {
                'symbol': f"{symbol}P{round(current_price, 1)}",
                'price': current_price * 0.05,  # Mock price
                'delta': -0.50,
                'implied_volatility': 0.25
            },
            'total_premium': current_price * 0.10,  # Combined premium
            'max_loss': current_price * 0.10,       # Premium paid
            'break_even_down': current_price - (current_price * 0.10),
            'break_even_up': current_price + (current_price * 0.10),
            'days_to_expiration': 30
        }
        
        return result
        
    def _find_strangle(self, symbol, option_chain, current_price):
        """
        Find suitable options for a strangle strategy.
        
        Args:
            symbol: Symbol to trade
            option_chain: Option chain data
            current_price: Current price of the underlying
            
        Returns:
            Dictionary with selected option data
        """
        # In a real implementation, we would:
        # 1. Find OTM call and put options based on width parameter
        # 2. Filter by our DTE requirements
        # 3. Check open interest and volume
        # 4. Select the best pair
        
        # Simplified mock implementation for validation
        width = self.params['strangle_width_pct']
        call_strike = round(current_price * (1 + width), 1)
        put_strike = round(current_price * (1 - width), 1)
        
        result = {
            'strategy_type': 'strangle',
            'symbol': symbol,
            'current_price': current_price,
            'expiration': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'call_strike': call_strike,
            'put_strike': put_strike,
            'call': {
                'symbol': f"{symbol}C{call_strike}",
                'price': current_price * 0.03,  # Mock price
                'delta': 0.30,
                'implied_volatility': 0.25
            },
            'put': {
                'symbol': f"{symbol}P{put_strike}",
                'price': current_price * 0.03,  # Mock price
                'delta': -0.30,
                'implied_volatility': 0.25
            },
            'total_premium': current_price * 0.06,  # Combined premium
            'max_loss': current_price * 0.06,       # Premium paid
            'break_even_down': put_strike - (current_price * 0.06),
            'break_even_up': call_strike + (current_price * 0.06),
            'days_to_expiration': 30
        }
        
        return result
        
    def _create_signal(self, symbol, strategy_data, current_price, strategy_variant):
        """
        Create a trading signal for straddle/strangle strategy.
        
        Args:
            symbol: Symbol to trade
            strategy_data: Strategy data from _find_straddle or _find_strangle
            current_price: Current price
            strategy_variant: 'straddle' or 'strangle'
        """
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'strategy_id': self.strategy_id,
            'strategy_type': strategy_variant,
            'action': 'BUY',
            'confidence': 0.70,  # Would normally be calculated based on various factors
            'direction': 'NEUTRAL',
            'current_price': current_price,
            'options': strategy_data,
            'risk_reward_ratio': 3.0,  # Would normally be calculated
            'metadata': {
                'volatility': strategy_data.get('implied_volatility', 0),
                'days_to_expiration': strategy_data.get('days_to_expiration', 0),
                'reason': f"High volatility setup using {strategy_variant} strategy",
                'signal_version': '1.0'
            }
        }
        
        return signal

    def get_health_status(self):
        """
        Get comprehensive health status of the strategy.
        
        Returns:
            Dictionary with health metrics
        """
        return {
            'strategy_id': self.strategy_id,
            'strategy_name': self.name,
            'strategy_variant': self.params.get('strategy_variant', 'adaptive'),
            'active_positions': len(self.positions),
            'status': 'healthy',
            'last_updated': datetime.now().isoformat()
        }
        
    def get_parameters(self):
        """Get current strategy parameters"""
        return self.params


def validate_strategy():
    """Test the standalone strategy implementation"""
    logger.info("Testing standalone strategy...")
    
    # Create the strategy
    strategy = StandaloneStraddleStrangleStrategy(
        strategy_id="test_strategy",
        name="Test Straddle/Strangle",
        parameters={
            'volatility_threshold': 0.15,  # Lower for testing
            'strategy_variant': 'adaptive'
        }
    )
    
    # Create mock market data
    market_data = {
        "SPY": create_mock_data("SPY", 450.0),
        "QQQ": create_mock_data("QQQ", 380.0),
        "AAPL": create_mock_data("AAPL", 175.0),
        "TSLA": create_mock_data("TSLA", 250.0)
    }
    
    # Create mock option chains
    option_chains = MockOptionChains(market_data)
    
    # Generate signals
    signals = strategy.generate_signals(market_data, option_chains)
    
    # Check health status
    health = strategy.get_health_status()
    logger.info(f"Strategy health: {health}")
    
    # Report results
    if signals:
        logger.info(f"Strategy generated {len(signals)} signals:")
        for i, signal in enumerate(signals, 1):
            logger.info(f"Signal {i}: {signal['symbol']} {signal['strategy_type']} - {signal['action']}")
    else:
        logger.warning("No signals generated")
    
    logger.info("Strategy validation completed")
    return True


def create_mock_data(symbol, base_price):
    """Create mock price data for testing"""
    np.random.seed(hash(symbol) % 100)
    
    # Generate 100 days of data with realistic volatility
    days = 100
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    
    # Add some volatility based on the symbol
    volatility = {
        "SPY": 0.010,
        "QQQ": 0.015,
        "AAPL": 0.020,
        "TSLA": 0.035
    }.get(symbol, 0.015)
    
    # Generate daily returns with some autocorrelation
    daily_returns = np.random.normal(0.0005, volatility, days)
    
    # Add some autocorrelation to simulate trending
    for i in range(1, days):
        daily_returns[i] += 0.2 * daily_returns[i-1]
    
    # Create cumulative returns and prices
    cum_returns = np.cumprod(1 + daily_returns[::-1])  # Reverse to have older dates first
    prices = base_price * cum_returns
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 - np.random.normal(0, 0.003, days)),
        'high': prices * (1 + np.random.normal(0.005, 0.003, days)),
        'low': prices * (1 - np.random.normal(0.005, 0.003, days)),
        'close': prices,
        'volume': np.random.normal(1000000, 300000, days).astype(int)
    })
    
    # Make sure volume is positive
    df['volume'] = df['volume'].abs()
    
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    return df


class MockOptionChains:
    """Simple mock option chains for testing"""
    
    def __init__(self, market_data):
        self.market_data = market_data
        
    def get_chain_for_symbol(self, symbol):
        """Get option chain for a symbol"""
        if symbol not in self.market_data:
            return None
            
        data = self.market_data[symbol]
        current_price = data.iloc[-1]['close']
        
        # Create a simple option chain with a few strikes around current price
        chain = {
            'calls': [],
            'puts': [],
            'expirations': [
                (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
            ],
            'underlying_price': current_price
        }
        
        # Generate strikes
        for strike_pct in [0.9, 0.95, 1.0, 1.05, 1.1]:
            strike = round(current_price * strike_pct, 1)
            
            # Create call
            call = {
                'symbol': f"{symbol}C{strike}",
                'option_type': 'call',
                'strike': strike,
                'expiration': chain['expirations'][0],
                'bid': round(max(0, (current_price - strike) * 0.8 + current_price * 0.02), 2),
                'ask': round(max(0, (current_price - strike) + current_price * 0.03), 2),
                'implied_volatility': 0.25,
                'volume': 500,
                'open_interest': 2000,
                'delta': min(1.0, max(0, 0.5 + (current_price - strike) / (current_price * 0.1)))
            }
            
            # Create put
            put = {
                'symbol': f"{symbol}P{strike}",
                'option_type': 'put',
                'strike': strike,
                'expiration': chain['expirations'][0],
                'bid': round(max(0, (strike - current_price) * 0.8 + current_price * 0.02), 2),
                'ask': round(max(0, (strike - current_price) + current_price * 0.03), 2),
                'implied_volatility': 0.25,
                'volume': 500,
                'open_interest': 2000,
                'delta': max(-1.0, min(0, -0.5 + (current_price - strike) / (current_price * 0.1)))
            }
            
            chain['calls'].append(call)
            chain['puts'].append(put)
            
        return chain


if __name__ == "__main__":
    validate_strategy()
