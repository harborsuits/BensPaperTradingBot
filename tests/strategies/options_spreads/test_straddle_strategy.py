import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from trading_bot.strategies.options_spreads.straddle_strategy import StraddleStrategy
from trading_bot.models.signal import Signal, SignalType, OptionContract, OptionType


class TestStraddleStrategy(unittest.TestCase):
    
    def setUp(self):
        # Mock data source and broker
        self.mock_data_source = MagicMock()
        self.mock_broker = MagicMock()
        
        # Create test config with smaller watchlist for faster testing
        self.test_config = {
            "watchlist": {
                "symbols": ["SPY", "AAPL"],
                "include_earnings": True,
                "include_economic_events": False
            },
            "entry_criteria": {
                "min_iv_rank": 60,
                "min_option_volume": 100,
                "max_bid_ask_spread_pct": 5.0,
                "min_score": 70
            },
            "strike_selection": {
                "method": "ATM",
                "target_delta": 0.50,
                "min_days_to_expiration": 20,
                "max_days_to_expiration": 45,
                "expiration_preference": "MIDDLE"
            },
            "risk_management": {
                "position_sizing_pct": 5.0,
                "max_concurrent_positions": 5,
                "target_return_pct": 50.0,
                "max_loss_pct": 25.0,
                "adjust_for_earnings": True
            }
        }
        
        # Initialize strategy with mocks
        self.strategy = StraddleStrategy(
            data_source=self.mock_data_source,
            broker=self.mock_broker,
            config=self.test_config
        )
        
    def _create_mock_option_chain(self):
        """Helper method to create a mock option chain dataframe"""
        # Create sample dates
        today = datetime.now().date()
        exp_dates = [today + timedelta(days=days) for days in [21, 35, 49]]
        
        # Create option chain mock data
        calls = []
        puts = []
        
        for exp in exp_dates:
            for strike in [398, 399, 400, 401, 402]:
                # Add call options
                calls.append({
                    'expiration': exp,
                    'strike': strike,
                    'type': 'call',
                    'bid': 5.5,
                    'ask': 5.7,
                    'volume': 500,
                    'open_interest': 1000,
                    'implied_volatility': 0.25,
                    'delta': 0.5 if strike == 400 else (0.6 if strike < 400 else 0.4)
                })
                
                # Add put options
                puts.append({
                    'expiration': exp,
                    'strike': strike,
                    'type': 'put',
                    'bid': 5.2,
                    'ask': 5.4,
                    'volume': 450,
                    'open_interest': 950,
                    'implied_volatility': 0.26,
                    'delta': -0.5 if strike == 400 else (-0.4 if strike < 400 else -0.6)
                })
        
        # Combine into single dataframe
        options_df = pd.DataFrame(calls + puts)
        return options_df
    
    @patch('trading_bot.strategies.options_spreads.straddle_strategy.StraddleStrategy._get_iv_rank')
    @patch('trading_bot.strategies.options_spreads.straddle_strategy.StraddleStrategy._get_current_price')
    @patch('trading_bot.strategies.options_spreads.straddle_strategy.StraddleStrategy._get_option_chain')
    def test_generate_signals(self, mock_get_chain, mock_get_price, mock_get_iv):
        """Test the signal generation process"""
        # Set up mocks
        mock_get_price.return_value = 400.0
        mock_get_iv.return_value = 75.0  # High IV rank
        mock_get_chain.return_value = self._create_mock_option_chain()
        
        # Run the strategy
        signals = self.strategy.generate_signals()
        
        # Verify that signals were generated
        self.assertTrue(len(signals) > 0, "Should generate at least one signal")
        
        # Verify signal properties
        for signal in signals:
            self.assertEqual(signal.signal_type, SignalType.ENTRY)
            self.assertEqual(len(signal.option_legs), 2, "Should have two option legs (call and put)")
            
            # Check that we have one call and one put
            option_types = [leg.option_type for leg in signal.option_legs]
            self.assertIn(OptionType.CALL, option_types)
            self.assertIn(OptionType.PUT, option_types)
            
            # All legs should have the same strike and expiration
            strikes = set(leg.strike for leg in signal.option_legs)
            expirations = set(leg.expiration for leg in signal.option_legs)
            self.assertEqual(len(strikes), 1, "All legs should have the same strike")
            self.assertEqual(len(expirations), 1, "All legs should have the same expiration")
    
    def test_filter_expirations(self):
        """Test the expiration date filtering logic"""
        today = datetime.now().date()
        expirations = [
            today + timedelta(days=10),  # Too close
            today + timedelta(days=25),  # Within range
            today + timedelta(days=40),  # Within range
            today + timedelta(days=60)   # Too far
        ]
        
        filtered = self.strategy._filter_expirations(expirations)
        
        # Should only include the middle two expirations
        self.assertEqual(len(filtered), 2)
        self.assertIn(today + timedelta(days=25), filtered)
        self.assertIn(today + timedelta(days=40), filtered)
    
    def test_select_strikes_atm(self):
        """Test ATM strike selection method"""
        current_price = 400.25
        option_chain = self._create_mock_option_chain()
        
        strikes = self.strategy._select_strikes(option_chain, current_price, method="ATM")
        
        # Should select the strike closest to current price
        self.assertEqual(len(strikes), 1)
        self.assertEqual(strikes[0], 400.0)
    
    def test_evaluate_straddle_opportunity(self):
        """Test the evaluation of a straddle opportunity"""
        symbol = "SPY"
        expiration = datetime.now().date() + timedelta(days=30)
        strike = 400.0
        call_price = 5.6  # Mid of bid-ask
        put_price = 5.3   # Mid of bid-ask
        current_price = 400.0
        iv_rank = 75.0
        
        score, signal = self.strategy._evaluate_straddle_opportunity(
            symbol, expiration, strike, call_price, put_price, current_price, iv_rank
        )
        
        # Verify the score is calculated
        self.assertGreater(score, 0)
        
        # Verify the signal has correct properties
        self.assertEqual(signal.symbol, symbol)
        self.assertEqual(signal.signal_type, SignalType.ENTRY)
        self.assertEqual(len(signal.option_legs), 2)
        
        # Verify the call leg
        call_leg = next(leg for leg in signal.option_legs if leg.option_type == OptionType.CALL)
        self.assertEqual(call_leg.strike, strike)
        self.assertEqual(call_leg.expiration, expiration)
        
        # Verify the put leg
        put_leg = next(leg for leg in signal.option_legs if leg.option_type == OptionType.PUT)
        self.assertEqual(put_leg.strike, strike)
        self.assertEqual(put_leg.expiration, expiration)


if __name__ == '__main__':
    unittest.main() 