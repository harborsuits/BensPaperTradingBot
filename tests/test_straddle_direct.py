#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple direct test for the Straddle/Strangle strategy.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directly import the strategy class to avoid complex dependency chains
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Mock the base class and dependencies
class AssetClass:
    OPTIONS = "options"

class StrategyType:
    VOLATILITY = "volatility"

class MarketRegime:
    VOLATILE = "volatile"

class TimeFrame:
    SWING = "swing"

def register_strategy(metadata):
    """Mock decorator for strategy registration."""
    def decorator(cls):
        return cls
    return decorator

class Universe:
    """Mock Universe class."""
    def __init__(self):
        self.symbols = []
        
    def add_symbol(self, symbol):
        self.symbols.append(symbol)
        
    def get_symbols(self):
        return self.symbols

class OptionsBaseStrategy:
    """Mock base strategy class."""
    
    DEFAULT_PARAMS = {
        'strategy_name': 'options_template_strategy',
    }
    
    def __init__(self, strategy_id=None, name=None, parameters=None):
        self.strategy_id = strategy_id or 'mock_strategy'
        self.name = name or 'Mock Strategy'
        self.parameters = parameters or {}
        self.straddle_positions = {}
        self.closed_positions = []
        
    def get_account_value(self):
        """Mock account value."""
        return 100000.0  # $100k account

class MarketData:
    """Mock MarketData class."""
    
    def __init__(self):
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.prices = {
            'AAPL': 175.50,
            'MSFT': 330.25,
            'GOOGL': 140.75,
            'AMZN': 178.25,
            'TSLA': 245.75
        }
        
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
        """Return mock historical data."""
        data = pd.DataFrame()
        price = self.prices.get(symbol, 100)
        
        # Create 100 days of mock data
        dates = [date.today() - timedelta(days=i) for i in range(100)]
        dates.reverse()
        
        # Create simple price series with moderate volatility
        prices = []
        current = price
        for _ in range(100):
            change = np.random.normal(0, 0.015)  # 1.5% daily volatility
            current = current * (1 + change)
            prices.append(current)
        
        data['date'] = dates
        data['open'] = prices
        data['high'] = [p * 1.01 for p in prices]  # 1% higher than open
        data['low'] = [p * 0.99 for p in prices]   # 1% lower than open
        data['close'] = [p * (1 + np.random.uniform(-0.005, 0.005)) for p in prices]
        data['volume'] = [int(np.random.uniform(1000000, 10000000)) for _ in range(100)]
        
        data.set_index('date', inplace=True)
        
        return data
    
    def get_iv_percentile(self, symbol):
        """Return mock IV percentile."""
        return 65
    
    def get_adx(self, symbol):
        """Return mock ADX value."""
        return 18

class OptionChains:
    """Mock OptionChains class."""
    
    def __init__(self):
        self.market_data = MarketData()
        self.symbols = self.market_data.symbols
    
    def has_symbol(self, symbol):
        return symbol in self.symbols
    
    def get_chain(self, symbol):
        if symbol in self.symbols:
            return MockChain(symbol, self.market_data.prices.get(symbol, 100))
        return None

class MockChain:
    """Mock option chain for a symbol."""
    
    def __init__(self, symbol, price):
        self.symbol = symbol
        self.price = price
        self.expirations = [
            date.today() + timedelta(days=30),
            date.today() + timedelta(days=45),
            date.today() + timedelta(days=60)
        ]
    
    def get_expiration_dates(self):
        return self.expirations
    
    def get_calls(self, expiration):
        """Generate mock call options."""
        strikes = [
            self.price * 0.9,
            self.price * 0.95,
            self.price,
            self.price * 1.05,
            self.price * 1.1
        ]
        
        calls = []
        for strike in strikes:
            days_to_expiry = (expiration - date.today()).days
            intrinsic = max(0, self.price - strike)
            time_value = self.price * 0.01 * (days_to_expiry / 30)
            option_price = intrinsic + time_value
            
            call = {
                'strike': strike,
                'bid': option_price - 0.05,
                'ask': option_price + 0.05,
                'implied_volatility': 0.4,
                'volume': 500,
                'open_interest': 2000,
                'delta': 0.5 if strike == self.price else (0.7 if strike < self.price else 0.3)
            }
            calls.append(call)
        
        return calls
    
    def get_puts(self, expiration):
        """Generate mock put options."""
        strikes = [
            self.price * 0.9,
            self.price * 0.95,
            self.price,
            self.price * 1.05,
            self.price * 1.1
        ]
        
        puts = []
        for strike in strikes:
            days_to_expiry = (expiration - date.today()).days
            intrinsic = max(0, strike - self.price)
            time_value = self.price * 0.01 * (days_to_expiry / 30)
            option_price = intrinsic + time_value
            
            put = {
                'strike': strike,
                'bid': option_price - 0.05,
                'ask': option_price + 0.05,
                'implied_volatility': 0.45,
                'volume': 450,
                'open_interest': 1800,
                'delta': -0.5 if strike == self.price else (-0.3 if strike > self.price else -0.7)
            }
            puts.append(put)
        
        return puts

# Now import the actual strategy
# This assumes these mock classes match the signatures expected by the strategy
sys.modules['trading_bot.strategies.base.options_base_strategy'] = type('', (), {'OptionsBaseStrategy': OptionsBaseStrategy})
sys.modules['trading_bot.strategies.factory.strategy_registry'] = type('', (), {
    'register_strategy': register_strategy,
    'StrategyType': StrategyType,
    'AssetClass': AssetClass,
    'MarketRegime': MarketRegime,
    'TimeFrame': TimeFrame
})
sys.modules['trading_bot.market.market_data'] = type('', (), {'MarketData': MarketData})
sys.modules['trading_bot.market.universe'] = type('', (), {'Universe': Universe})
sys.modules['trading_bot.market.option_chains'] = type('', (), {'OptionChains': OptionChains})

# Now execute the direct import of our strategy
exec(open("trading_bot/strategies/options/volatility_spreads/straddle_strangle_strategy.py").read())

# Get the StraddleStrangleStrategy class from the executed file
StraddleStrangleStrategy = locals().get('StraddleStrangleStrategy')

def test_strategy():
    """Test the straddle/strangle strategy directly."""
    logger.info("Testing Straddle/Strangle strategy...")
    
    # Create test instances
    market_data = MarketData()
    option_chains = OptionChains()
    
    # Test straddle variant
    straddle = StraddleStrangleStrategy(
        strategy_id='test_straddle',
        name='Test Straddle',
        parameters={'strategy_variant': 'straddle'}
    )
    
    logger.info("Testing straddle universe definition...")
    universe = straddle.define_universe(market_data)
    symbols = universe.get_symbols()
    logger.info(f"Straddle universe contains {len(symbols)} symbols: {symbols}")
    
    logger.info("Testing straddle signal generation...")
    signals = straddle.generate_signals(market_data, option_chains)
    logger.info(f"Generated {len(signals)} straddle signals")
    
    for signal in signals:
        logger.info(f"Signal for {signal['symbol']}: {signal['strategy']}")
        logger.info(f"  - Legs: {len(signal['option_legs'])}")
        logger.info(f"  - Confidence: {signal['confidence']}")
        logger.info(f"  - Investment: ${signal['investment_amount']}")
        for leg in signal['option_legs']:
            logger.info(f"  - {leg['option_type']} @ {leg['strike']} for {leg['price']}")
        logger.info(f"  - Breakeven: {signal['breakeven_lower']} to {signal['breakeven_upper']}")
    
    # Test strangle variant
    strangle = StraddleStrangleStrategy(
        strategy_id='test_strangle',
        name='Test Strangle',
        parameters={'strategy_variant': 'strangle'}
    )
    
    logger.info("Testing strangle signal generation...")
    signals = strangle.generate_signals(market_data, option_chains)
    logger.info(f"Generated {len(signals)} strangle signals")
    
    for signal in signals:
        logger.info(f"Signal for {signal['symbol']}: {signal['strategy']}")
        logger.info(f"  - Legs: {len(signal['option_legs'])}")
        logger.info(f"  - Confidence: {signal['confidence']}")
        logger.info(f"  - Investment: ${signal['investment_amount']}")
        for leg in signal['option_legs']:
            logger.info(f"  - {leg['option_type']} @ {leg['strike']} for {leg['price']}")
        logger.info(f"  - Breakeven: {signal['breakeven_lower']} to {signal['breakeven_upper']}")
    
    # Test exit signal generation
    exit_signals = straddle.on_exit_signal(market_data, option_chains)
    logger.info(f"Generated {len(exit_signals)} exit signals")
    
    # Force an exit condition
    if straddle.straddle_positions:
        symbol = list(straddle.straddle_positions.keys())[0]
        straddle.straddle_positions[symbol]['time_stop'] = date.today() - timedelta(days=1)
        
        logger.info("Testing exit with forced time stop...")
        exit_signals = straddle.on_exit_signal(market_data, option_chains)
        logger.info(f"Generated {len(exit_signals)} exit signals after time stop")
        
        for signal in exit_signals:
            logger.info(f"Exit signal for {signal['symbol']}: {signal['reason']}")
            logger.info(f"  - P&L: ${signal['profit_loss']} ({signal['profit_loss_pct']}%)")
    
    logger.info("Test completed!")

if __name__ == "__main__":
    test_strategy()
