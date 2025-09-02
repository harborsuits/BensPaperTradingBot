#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Straddle/Strangle options strategy.
This script loads the strategy and tests its core functionality.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any

# Add the project root to the path so we can import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from trading_bot.strategies.options.volatility_spreads.straddle_strangle_strategy import StraddleStrangleStrategy
from trading_bot.strategies.factory.strategy_registry import StrategyRegistry
from trading_bot.market.market_data import MarketData
from trading_bot.market.option_chains import OptionChains

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockMarketData(MarketData):
    """Mock market data provider for testing."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.prices = {
            'AAPL': 175.50,
            'MSFT': 330.25,
            'GOOGL': 140.75,
            'AMZN': 178.25,
            'TSLA': 245.75
        }
        self.historical_data = self._generate_mock_historical_data()
    
    def get_all_symbols(self):
        return self.symbols
    
    def get_latest_quote(self, symbol):
        if symbol in self.prices:
            return {
                'symbol': symbol,
                'price': self.prices[symbol],
                'bid': self.prices[symbol] - 0.05,
                'ask': self.prices[symbol] + 0.05,
                'volume': 5000000,
                'timestamp': datetime.now().isoformat()
            }
        return None
    
    def get_historical_data(self, symbol, period=90):
        if symbol in self.historical_data:
            return self.historical_data[symbol].tail(period)
        return None
    
    def get_iv_percentile(self, symbol):
        # Return mock IV percentile data
        return 65  # Moderately high IV
    
    def get_adx(self, symbol):
        # Return mock ADX data
        return 18  # Fairly low trend strength, good for straddle/strangle
    
    def _generate_mock_historical_data(self):
        """Generate mock historical price data for testing."""
        data = {}
        for symbol in self.symbols:
            # Generate 100 days of mock data with some randomness
            base_price = self.prices[symbol]
            dates = [date.today() - timedelta(days=i) for i in range(100)]
            dates.reverse()
            
            # Create price series with some random walk
            prices = [base_price]
            for i in range(1, 100):
                change = np.random.normal(0, 0.015)  # 1.5% daily volatility
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                'close': [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices],
                'volume': [int(np.random.uniform(1000000, 10000000)) for _ in range(100)]
            })
            df.set_index('date', inplace=True)
            data[symbol] = df
            
        return data


class MockOptionChains(OptionChains):
    """Mock option chains provider for testing."""
    
    def __init__(self, market_data):
        self.market_data = market_data
        self.symbols = market_data.symbols
        self.expiration_dates = self._generate_expiration_dates()
        self.chains = self._generate_option_chains()
    
    def has_symbol(self, symbol):
        return symbol in self.symbols
    
    def get_chain(self, symbol):
        if symbol in self.chains:
            return self.chains[symbol]
        return None
    
    def _generate_expiration_dates(self):
        """Generate a set of expiration dates."""
        today = date.today()
        dates = [
            today + timedelta(days=30),  # 30 DTE
            today + timedelta(days=45),  # 45 DTE
            today + timedelta(days=60),  # 60 DTE
            today + timedelta(days=90)   # 90 DTE
        ]
        return dates
    
    def _generate_option_chains(self):
        """Generate mock option chains for testing."""
        chains = {}
        
        for symbol in self.symbols:
            current_price = self.market_data.prices[symbol]
            symbol_chains = MockSymbolChain(symbol, current_price, self.expiration_dates)
            chains[symbol] = symbol_chains
            
        return chains


class MockSymbolChain:
    """Mock option chain for a specific symbol."""
    
    def __init__(self, symbol, current_price, expiration_dates):
        self.symbol = symbol
        self.current_price = current_price
        self.expiration_dates = expiration_dates
        self.strikes = self._generate_strikes()
        self.calls = {}
        self.puts = {}
        
        # Generate option data for each expiration
        for expiry in expiration_dates:
            self.calls[expiry] = self._generate_calls(expiry)
            self.puts[expiry] = self._generate_puts(expiry)
    
    def get_expiration_dates(self):
        return self.expiration_dates
    
    def get_calls(self, expiration_date):
        if expiration_date in self.calls:
            return self.calls[expiration_date]
        return []
    
    def get_puts(self, expiration_date):
        if expiration_date in self.puts:
            return self.puts[expiration_date]
        return []
    
    def _generate_strikes(self):
        """Generate strike prices around the current price."""
        base_price = self.current_price
        # Create strikes at 5% intervals from -20% to +20%
        strike_pcts = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
        strikes = [round(base_price * pct, 1) for pct in strike_pcts]
        return strikes
    
    def _generate_calls(self, expiration_date):
        """Generate call option data for a specific expiration."""
        days_to_expiry = (expiration_date - date.today()).days
        calls = []
        
        for strike in self.strikes:
            # Calculate option price based on intrinsic value + time value
            intrinsic = max(0, self.current_price - strike)
            time_value = self.current_price * 0.01 * (days_to_expiry / 30)  # 1% per month
            iv = 0.4 + np.random.uniform(-0.1, 0.1)  # IV around 40%
            
            # Calculate rough delta
            if strike < self.current_price:
                delta = 0.5 + (self.current_price - strike) / self.current_price / 2
                delta = min(0.95, delta)
            else:
                delta = 0.5 - (strike - self.current_price) / self.current_price / 2
                delta = max(0.05, delta)
            
            option_price = intrinsic + time_value
            
            call = {
                'symbol': f"{self.symbol}_{expiration_date.strftime('%y%m%d')}C{int(strike)}",
                'underlying': self.symbol,
                'expiration': expiration_date,
                'strike': strike,
                'option_type': 'call',
                'bid': option_price - 0.05,
                'ask': option_price + 0.05,
                'last': option_price,
                'volume': int(np.random.uniform(100, 1000)),
                'open_interest': int(np.random.uniform(500, 5000)),
                'implied_volatility': iv,
                'delta': delta,
                'gamma': 0.02,
                'theta': -0.02 * (30 / days_to_expiry),
                'vega': 0.1
            }
            calls.append(call)
        
        return calls
    
    def _generate_puts(self, expiration_date):
        """Generate put option data for a specific expiration."""
        days_to_expiry = (expiration_date - date.today()).days
        puts = []
        
        for strike in self.strikes:
            # Calculate option price based on intrinsic value + time value
            intrinsic = max(0, strike - self.current_price)
            time_value = self.current_price * 0.01 * (days_to_expiry / 30)  # 1% per month
            iv = 0.45 + np.random.uniform(-0.1, 0.1)  # Put IV slightly higher
            
            # Calculate rough delta
            if strike > self.current_price:
                delta = -0.5 - (strike - self.current_price) / self.current_price / 2
                delta = max(-0.95, delta)
            else:
                delta = -0.5 + (self.current_price - strike) / self.current_price / 2
                delta = min(-0.05, delta)
            
            option_price = intrinsic + time_value
            
            put = {
                'symbol': f"{self.symbol}_{expiration_date.strftime('%y%m%d')}P{int(strike)}",
                'underlying': self.symbol,
                'expiration': expiration_date,
                'strike': strike,
                'option_type': 'put',
                'bid': option_price - 0.05,
                'ask': option_price + 0.05,
                'last': option_price,
                'volume': int(np.random.uniform(100, 1000)),
                'open_interest': int(np.random.uniform(500, 5000)),
                'implied_volatility': iv,
                'delta': delta,
                'gamma': 0.02,
                'theta': -0.02 * (30 / days_to_expiry),
                'vega': 0.1
            }
            puts.append(put)
        
        return puts


def test_strategy_registration():
    """Test that the strategy is properly registered."""
    logger.info("Testing strategy registration...")
    registry = StrategyRegistry()
    strategies = registry.get_all_strategies()
    
    # Check if our strategy is in the registry
    strategy_names = [s.__name__ for s in strategies.values()]
    assert 'StraddleStrangleStrategy' in strategy_names, "Strategy not found in registry"
    
    logger.info("✅ Strategy is properly registered in the StrategyRegistry")


def test_universe_definition():
    """Test the universe definition logic."""
    logger.info("Testing universe definition...")
    
    # Create market data
    market_data = MockMarketData()
    
    # Create strategy instance
    strategy = StraddleStrangleStrategy(strategy_id='test_straddle', name='Test Straddle')
    
    # Define universe
    universe = strategy.define_universe(market_data)
    symbols = universe.get_symbols()
    
    logger.info(f"Universe contains {len(symbols)} symbols: {symbols}")
    assert len(symbols) > 0, "Universe should contain at least one symbol"


def test_signal_generation():
    """Test signal generation for both straddle and strangle variants."""
    logger.info("Testing signal generation...")
    
    # Create mock data
    market_data = MockMarketData()
    option_chains = MockOptionChains(market_data)
    
    # Test straddle variant
    logger.info("Testing straddle strategy...")
    straddle_strategy = StraddleStrangleStrategy(
        strategy_id='test_straddle',
        name='Test Straddle',
        parameters={'strategy_variant': 'straddle'}
    )
    
    straddle_signals = straddle_strategy.generate_signals(market_data, option_chains)
    logger.info(f"Generated {len(straddle_signals)} straddle signals")
    
    for signal in straddle_signals:
        logger.info(f"Straddle signal for {signal['symbol']}: {len(signal['option_legs'])} legs, " +
                  f"confidence: {signal['confidence']}, investment: ${signal['investment_amount']:.2f}")
    
    # Test strangle variant
    logger.info("Testing strangle strategy...")
    strangle_strategy = StraddleStrangleStrategy(
        strategy_id='test_strangle',
        name='Test Strangle',
        parameters={'strategy_variant': 'strangle'}
    )
    
    strangle_signals = strangle_strategy.generate_signals(market_data, option_chains)
    logger.info(f"Generated {len(strangle_signals)} strangle signals")
    
    for signal in strangle_signals:
        logger.info(f"Strangle signal for {signal['symbol']}: {len(signal['option_legs'])} legs, " +
                  f"confidence: {signal['confidence']}, investment: ${signal['investment_amount']:.2f}")


def test_exit_strategy():
    """Test exit signal generation."""
    logger.info("Testing exit strategy...")
    
    # Create mock data
    market_data = MockMarketData()
    option_chains = MockOptionChains(market_data)
    
    # Create strategy with positions already populated
    strategy = StraddleStrangleStrategy(
        strategy_id='test_exit',
        name='Test Exit',
        parameters={'strategy_variant': 'straddle'}
    )
    
    # First generate some signals and store positions
    signals = strategy.generate_signals(market_data, option_chains)
    
    if not signals:
        logger.warning("No signals generated, can't test exit strategy")
        return
    
    # Now test exit signals
    exit_signals = strategy.on_exit_signal(market_data, option_chains)
    logger.info(f"Generated {len(exit_signals)} exit signals")
    
    # Modify a position to trigger exit conditions
    if strategy.straddle_positions:
        symbol = list(strategy.straddle_positions.keys())[0]
        # Set time_stop to yesterday to trigger exit
        strategy.straddle_positions[symbol]['time_stop'] = date.today() - timedelta(days=1)
        
        # Test again with modified position
        exit_signals = strategy.on_exit_signal(market_data, option_chains)
        logger.info(f"Generated {len(exit_signals)} exit signals after modification")
        
        for signal in exit_signals:
            logger.info(f"Exit signal for {signal['symbol']}: reason: {signal['reason']}, " +
                      f"P&L: ${signal['profit_loss']:.2f} ({signal['profit_loss_pct']:.2f}%)")


def main():
    """Run all tests."""
    logger.info("Starting Straddle/Strangle strategy tests...")
    
    # Run tests
    test_strategy_registration()
    test_universe_definition()
    test_signal_generation()
    test_exit_strategy()
    
    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    main()
