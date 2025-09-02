"""
Unit tests for the Straddle strategy implementation.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from trading_bot.strategies.options_spreads.straddle_strategy import StraddleStrategy
from trading_bot.config.straddle_config import STRADDLE_CONFIG


class TestStraddleStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.config = STRADDLE_CONFIG.copy()
        
        # Create a mock data source
        self.mock_data_source = MagicMock()
        
        # Create a mock broker
        self.mock_broker = MagicMock()
        
        # Create the strategy instance
        self.strategy = StraddleStrategy(
            data_source=self.mock_data_source,
            broker=self.mock_broker,
            config=self.config
        )
        
        # Create sample price data
        self.create_sample_data()
        
    def create_sample_data(self):
        """Create sample historical price and options data for testing."""
        # Create historical price data for SPY
        dates = pd.date_range(start=datetime.now() - timedelta(days=60), 
                              end=datetime.now(), freq='D')
        
        # Generate sample price data
        self.price_data = pd.DataFrame({
            'date': dates,
            'open': np.linspace(400, 420, len(dates)) + np.random.normal(0, 2, len(dates)),
            'high': np.linspace(405, 425, len(dates)) + np.random.normal(0, 2, len(dates)),
            'low': np.linspace(395, 415, len(dates)) + np.random.normal(0, 2, len(dates)),
            'close': np.linspace(400, 420, len(dates)) + np.random.normal(0, 2, len(dates)),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        })
        
        # Generate IV data
        self.iv_data = pd.DataFrame({
            'date': dates,
            'iv_rank': np.random.uniform(30, 80, len(dates)),
            'iv_percentile': np.random.uniform(30, 80, len(dates))
        })
        
        # Create sample options chain data
        today = datetime.now()
        expirations = [today + timedelta(days=d) for d in [15, 30, 45, 60]]
        
        self.options_chain = []
        current_price = 410
        
        for exp in expirations:
            for strike in range(380, 440, 5):
                # Call option
                self.options_chain.append({
                    'symbol': 'SPY',
                    'option_type': 'call',
                    'strike': strike,
                    'expiration': exp,
                    'bid': max(0, 0.9 * (current_price - strike + 10)),
                    'ask': max(0, 1.1 * (current_price - strike + 10)),
                    'volume': np.random.randint(100, 1000),
                    'open_interest': np.random.randint(500, 5000),
                    'delta': 0.5 - 0.05 * ((strike - current_price) / 5),
                    'gamma': 0.05,
                    'theta': -0.03,
                    'vega': 0.10,
                    'implied_volatility': np.random.uniform(0.15, 0.35)
                })
                
                # Put option
                self.options_chain.append({
                    'symbol': 'SPY',
                    'option_type': 'put',
                    'strike': strike,
                    'expiration': exp,
                    'bid': max(0, 0.9 * (strike - current_price + 10)),
                    'ask': max(0, 1.1 * (strike - current_price + 10)),
                    'volume': np.random.randint(100, 1000),
                    'open_interest': np.random.randint(500, 5000),
                    'delta': -0.5 + 0.05 * ((strike - current_price) / 5),
                    'gamma': 0.05,
                    'theta': -0.03,
                    'vega': 0.10,
                    'implied_volatility': np.random.uniform(0.15, 0.35)
                })
        
        self.options_chain = pd.DataFrame(self.options_chain)
        
    def test_strategy_initialization(self):
        """Test strategy initialization and parameter loading."""
        self.assertEqual(self.strategy.config, self.config)
        self.assertEqual(self.strategy.name, "Straddle")
        self.assertEqual(len(self.strategy.config["watchlist"]["symbols"]), 10)
        
    def test_filter_expirations(self):
        """Test the expiration date filtering logic."""
        today = datetime.now()
        test_expirations = [
            today + timedelta(days=10),
            today + timedelta(days=25),
            today + timedelta(days=40),
            today + timedelta(days=60)
        ]
        
        # Configure the strategy
        self.strategy.config["strike_selection"]["min_days_to_expiration"] = 20
        self.strategy.config["strike_selection"]["max_days_to_expiration"] = 45
        
        # Filter expirations
        filtered = self.strategy._filter_expirations(test_expirations)
        
        # We expect the 25-day and 40-day expirations to pass
        self.assertEqual(len(filtered), 2)
        self.assertIn(test_expirations[1], filtered)
        self.assertIn(test_expirations[2], filtered)
        
    @patch('trading_bot.strategies.options_spreads.straddle_strategy.StraddleStrategy._get_option_chain')
    def test_find_best_straddle(self, mock_get_chain):
        """Test finding the best straddle setup."""
        # Mock the option chain retrieval
        mock_get_chain.return_value = self.options_chain
        
        # Configure strategy to use ATM strike selection
        self.strategy.config["strike_selection"]["method"] = "ATM"
        
        # Set the current price
        current_price = 410
        
        # Call the method to find the best straddle
        symbol = "SPY"
        straddle = self.strategy._find_best_straddle(symbol, current_price)
        
        # Verify we got a valid straddle setup
        self.assertIsNotNone(straddle)
        self.assertEqual(straddle['symbol'], symbol)
        
        # Verify that both call and put options are included
        self.assertIn('call_option', straddle)
        self.assertIn('put_option', straddle)
        
        # Check that the strike prices are the same for call and put
        self.assertEqual(straddle['call_option']['strike'], straddle['put_option']['strike'])
        
        # For ATM, the strike should be closest to current price
        self.assertAlmostEqual(straddle['call_option']['strike'], 410, delta=5)
        
    @patch('trading_bot.strategies.options_spreads.straddle_strategy.StraddleStrategy._get_current_price')
    @patch('trading_bot.strategies.options_spreads.straddle_strategy.StraddleStrategy._get_iv_rank')
    @patch('trading_bot.strategies.options_spreads.straddle_strategy.StraddleStrategy._find_best_straddle')
    def test_generate_signals(self, mock_find_straddle, mock_get_iv_rank, mock_get_price):
        """Test signal generation logic."""
        # Mock the dependencies
        mock_get_price.return_value = 410
        mock_get_iv_rank.return_value = 70  # High IV rank
        
        # Set up the mock straddle return value
        mock_straddle = {
            'symbol': 'SPY',
            'strike': 410,
            'expiration': datetime.now() + timedelta(days=30),
            'call_option': {
                'symbol': 'SPY220630C00410000',
                'strike': 410,
                'bid': 10,
                'ask': 11,
                'volume': 500,
                'delta': 0.50
            },
            'put_option': {
                'symbol': 'SPY220630P00410000',
                'strike': 410,
                'bid': 9,
                'ask': 10,
                'volume': 450,
                'delta': -0.50
            }
        }
        mock_find_straddle.return_value = mock_straddle
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        # Verify we got a signal
        self.assertGreater(len(signals), 0)
        
        # Verify signal properties
        signal = signals[0]
        self.assertEqual(signal.symbol, 'SPY')
        self.assertEqual(signal.direction, 'straddle')  # Should be 'straddle' for this strategy
        
    def test_evaluate_straddle_opportunity(self):
        """Test the evaluation of a straddle opportunity."""
        # Create a sample straddle configuration
        straddle = {
            'symbol': 'SPY',
            'strike': 410,
            'expiration': datetime.now() + timedelta(days=30),
            'call_option': {
                'strike': 410,
                'bid': 10,
                'ask': 11,
                'volume': 500,
                'implied_volatility': 0.25
            },
            'put_option': {
                'strike': 410,
                'bid': 9,
                'ask': 10,
                'volume': 450,
                'implied_volatility': 0.25
            }
        }
        
        # Set high IV rank
        iv_rank = 75
        
        # Evaluate the opportunity
        score, reason = self.strategy._evaluate_straddle_opportunity(straddle, iv_rank)
        
        # The score should be positive for a high IV rank
        self.assertGreater(score, 0)
        
        # Test with low IV rank
        low_iv_rank = 35
        score, reason = self.strategy._evaluate_straddle_opportunity(straddle, low_iv_rank)
        
        # The score should be lower or negative for a low IV rank
        self.assertLess(score, 75)  # Less than high IV score

if __name__ == '__main__':
    unittest.main() 