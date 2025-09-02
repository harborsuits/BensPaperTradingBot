"""
Option Spreads Strategy.

This module implements various option spread strategies based on implied volatility and market direction.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("option_spreads")

class OptionSpreadsStrategy:
    def __init__(self, params=None):
        self.params = params or {
            "iv_percentile_threshold": 70,  # High IV percentile for selling premium
            "iv_rank_threshold": 0.6,      # High IV rank for selling premium
            "days_to_expiration": 30,      # Target 30 days to expiration
            "delta_target": 0.30,          # Delta target for short options
            "risk_per_trade": 0.01,        # Risk 1% of portfolio per trade
            "probability_otm": 0.70        # Target 70% probability of profit
        }
    
    def generate_signals(self, market_data, context):
        """Generate option spread signals based on implied volatility and market direction"""
        signals = []
        
        for symbol, data in market_data.items():
            # Skip if no options data available
            if 'options' not in data:
                continue
            
            options_data = data['options']
            
            # Get price and volatility data
            current_price = data['close'][-1]
            
            # Check if we have the necessary IV data
            if 'iv_percentile' not in options_data or 'iv_rank' not in options_data:
                continue
            
            iv_percentile = options_data['iv_percentile']
            iv_rank = options_data['iv_rank']
            
            # High IV environment - good for selling premium
            if iv_percentile > self.params['iv_percentile_threshold'] and iv_rank > self.params['iv_rank_threshold']:
                # Determine market regime and appropriate strategy
                market_regime = context.get('market_regime', 'neutral')
                
                # In bullish market - Put Credit Spread (Sell put spread)
                if market_regime == 'bullish':
                    short_strike = self._find_strike_by_delta(options_data, 'put', self.params['delta_target'])
                    long_strike = self._find_next_strike(options_data, 'put', short_strike, direction='down')
                    
                    if short_strike and long_strike:
                        # Generate put credit spread signal
                        max_risk = short_strike - long_strike  # Per contract
                        credit_received = self._calculate_credit(options_data, 'put', short_strike, long_strike)
                        
                        signals.append({
                            'symbol': symbol,
                            'direction': 'long',  # Bullish position
                            'strategy_type': 'put_credit_spread',
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'days_to_expiration': self.params['days_to_expiration'],
                            'credit_received': credit_received,
                            'max_risk': max_risk,
                            'probability_otm': options_data.get('probability_otm', {}).get('put', {}).get(str(short_strike), 0.5),
                            'signal_type': 'option_spreads',
                            'confidence': self._calculate_confidence(iv_percentile, iv_rank, market_regime, 'bullish')
                        })
                
                # In bearish market - Call Credit Spread (Sell call spread)
                elif market_regime == 'bearish':
                    short_strike = self._find_strike_by_delta(options_data, 'call', self.params['delta_target'])
                    long_strike = self._find_next_strike(options_data, 'call', short_strike, direction='up')
                    
                    if short_strike and long_strike:
                        # Generate call credit spread signal
                        max_risk = long_strike - short_strike  # Per contract
                        credit_received = self._calculate_credit(options_data, 'call', short_strike, long_strike)
                        
                        signals.append({
                            'symbol': symbol,
                            'direction': 'short',  # Bearish position
                            'strategy_type': 'call_credit_spread',
                            'short_strike': short_strike,
                            'long_strike': long_strike,
                            'days_to_expiration': self.params['days_to_expiration'],
                            'credit_received': credit_received,
                            'max_risk': max_risk,
                            'probability_otm': options_data.get('probability_otm', {}).get('call', {}).get(str(short_strike), 0.5),
                            'signal_type': 'option_spreads',
                            'confidence': self._calculate_confidence(iv_percentile, iv_rank, market_regime, 'bearish')
                        })
                
                # In neutral or sideways market - Iron Condor (Sell call spread and put spread)
                elif market_regime in ['neutral', 'sideways']:
                    # Call side
                    call_short_strike = self._find_strike_by_delta(options_data, 'call', self.params['delta_target'])
                    call_long_strike = self._find_next_strike(options_data, 'call', call_short_strike, direction='up')
                    
                    # Put side
                    put_short_strike = self._find_strike_by_delta(options_data, 'put', self.params['delta_target'])
                    put_long_strike = self._find_next_strike(options_data, 'put', put_short_strike, direction='down')
                    
                    if call_short_strike and call_long_strike and put_short_strike and put_long_strike:
                        # Generate iron condor signal
                        call_max_risk = call_long_strike - call_short_strike
                        put_max_risk = put_short_strike - put_long_strike
                        max_risk = max(call_max_risk, put_max_risk)  # Per contract
                        
                        call_credit = self._calculate_credit(options_data, 'call', call_short_strike, call_long_strike)
                        put_credit = self._calculate_credit(options_data, 'put', put_short_strike, put_long_strike)
                        total_credit = call_credit + put_credit
                        
                        signals.append({
                            'symbol': symbol,
                            'direction': 'neutral',  # Neutral position
                            'strategy_type': 'iron_condor',
                            'call_short_strike': call_short_strike,
                            'call_long_strike': call_long_strike,
                            'put_short_strike': put_short_strike,
                            'put_long_strike': put_long_strike,
                            'days_to_expiration': self.params['days_to_expiration'],
                            'credit_received': total_credit,
                            'max_risk': max_risk,
                            'call_probability_otm': options_data.get('probability_otm', {}).get('call', {}).get(str(call_short_strike), 0.5),
                            'put_probability_otm': options_data.get('probability_otm', {}).get('put', {}).get(str(put_short_strike), 0.5),
                            'signal_type': 'option_spreads',
                            'confidence': self._calculate_confidence(iv_percentile, iv_rank, market_regime, 'neutral')
                        })
                
                # In volatile market - Long Strangle (Buy call and put)
                elif market_regime == 'volatile':
                    # For long volatility strategies, we want lower IV percentile for better entry
                    if iv_percentile < 30:
                        call_strike = self._find_strike_by_delta(options_data, 'call', 0.25)
                        put_strike = self._find_strike_by_delta(options_data, 'put', 0.25)
                        
                        if call_strike and put_strike:
                            # Generate long strangle signal
                            call_cost = options_data.get('prices', {}).get('call', {}).get(str(call_strike), 0)
                            put_cost = options_data.get('prices', {}).get('put', {}).get(str(put_strike), 0)
                            total_cost = call_cost + put_cost
                            
                            signals.append({
                                'symbol': symbol,
                                'direction': 'volatile',  # Volatility position
                                'strategy_type': 'long_strangle',
                                'call_strike': call_strike,
                                'put_strike': put_strike,
                                'days_to_expiration': self.params['days_to_expiration'] * 2,  # Longer DTE for long volatility
                                'total_cost': total_cost,
                                'max_risk': total_cost,  # Max risk is the premium paid
                                'signal_type': 'option_spreads',
                                'confidence': self._calculate_confidence(iv_percentile, iv_rank, market_regime, 'volatile')
                            })
        
        return signals
    
    def calculate_position_size(self, signal, portfolio_value):
        """Calculate position size (number of spreads) based on risk per trade"""
        # For option spreads, position size is number of contracts/spreads
        strategy_type = signal.get('strategy_type')
        
        if strategy_type in ['put_credit_spread', 'call_credit_spread', 'iron_condor']:
            max_risk = signal['max_risk'] * 100  # Convert to dollars (per spread)
            dollar_risk = portfolio_value * self.params['risk_per_trade']
            
            # Number of spreads = Dollar risk / Max risk per spread
            return max(1, int(dollar_risk / max_risk)) if max_risk > 0 else 1
        
        elif strategy_type == 'long_strangle':
            total_cost = signal['total_cost'] * 100  # Convert to dollars (per strangle)
            dollar_risk = portfolio_value * self.params['risk_per_trade']
            
            # Number of strangles = Dollar risk / Total cost per strangle
            return max(1, int(dollar_risk / total_cost)) if total_cost > 0 else 1
        
        # Default case
        return 1
    
    def _find_strike_by_delta(self, options_data, option_type, target_delta):
        """Find the strike price closest to the target delta"""
        if 'deltas' not in options_data or option_type not in options_data['deltas']:
            return None
        
        deltas = options_data['deltas'][option_type]
        strikes = options_data['strikes']
        
        # For calls, find strike with delta closest to target
        # For puts, delta is negative, so we need absolute value
        closest_strike = None
        min_diff = float('inf')
        
        for i, strike in enumerate(strikes):
            strike_str = str(strike)
            if strike_str not in deltas:
                continue
            
            delta = deltas[strike_str]
            delta_abs = abs(delta) if option_type == 'put' else delta
            
            diff = abs(delta_abs - target_delta)
            if diff < min_diff:
                min_diff = diff
                closest_strike = strike
        
        return closest_strike
    
    def _find_next_strike(self, options_data, option_type, current_strike, direction='up'):
        """Find the next strike price up or down from the current strike"""
        strikes = sorted(options_data['strikes'])
        
        try:
            current_index = strikes.index(current_strike)
            
            if direction == 'up':
                # Find next strike up
                if current_index < len(strikes) - 1:
                    return strikes[current_index + 1]
            else:
                # Find next strike down
                if current_index > 0:
                    return strikes[current_index - 1]
        except ValueError:
            pass
        
        return None
    
    def _calculate_credit(self, options_data, option_type, short_strike, long_strike):
        """Calculate the credit received for a spread"""
        if 'prices' not in options_data or option_type not in options_data['prices']:
            return 0
        
        prices = options_data['prices'][option_type]
        short_price = prices.get(str(short_strike), 0)
        long_price = prices.get(str(long_strike), 0)
        
        # Credit received = Short option premium - Long option premium
        return max(0, short_price - long_price)
    
    def _calculate_confidence(self, iv_percentile, iv_rank, market_regime, strategy_direction):
        # Base confidence
        confidence = 0.5
        
        # Adjust based on IV environment
        if strategy_direction in ['bullish', 'bearish', 'neutral']:
            # For premium selling strategies, higher IV is better
            confidence += min((iv_percentile / 100), 0.25)
            confidence += min(iv_rank, 0.25)
        else:  # Long volatility strategy
            # For long volatility strategies, lower IV is better entry
            confidence += min((100 - iv_percentile) / 100, 0.25)
            confidence += min(1 - iv_rank, 0.25)
        
        # Adjust based on market regime alignment
        if (strategy_direction == 'bullish' and market_regime == 'bullish') or \
           (strategy_direction == 'bearish' and market_regime == 'bearish') or \
           (strategy_direction == 'neutral' and market_regime in ['neutral', 'sideways']) or \
           (strategy_direction == 'volatile' and market_regime == 'volatile'):
            confidence += 0.15
        elif (strategy_direction == 'bullish' and market_regime == 'bearish') or \
             (strategy_direction == 'bearish' and market_regime == 'bullish'):
            confidence -= 0.15
        
        # Ensure confidence is between 0.1 and 0.9
        return max(0.1, min(0.9, confidence)) 