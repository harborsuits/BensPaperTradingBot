"""
Volatility Breakout Strategy.

This module implements a volatility breakout trading strategy based on expanding volatility.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("volatility_breakout")

class VolatilityBreakoutStrategy:
    def __init__(self, params=None):
        self.params = params or {
            "atr_period": 14,
            "volatility_threshold": 1.2,  # Minimum volatility increase factor
            "range_period": 5,
            "range_factor": 1.0,          # Breakout beyond recent range
            "risk_per_trade": 0.015       # Risk 1.5% of portfolio per trade
        }
    
    def generate_signals(self, market_data, context):
        """Generate volatility breakout signals based on expanding volatility and price action"""
        signals = []
        
        for symbol, data in market_data.items():
            # Calculate indicators
            atr = self._calculate_atr(data, self.params['atr_period'])
            
            # Get price data
            prices = data['close']
            highs = data['high']
            lows = data['low']
            
            # Calculate recent volatility increase
            current_atr = atr[-1]
            prev_atr = atr[-5]  # ATR 5 days ago
            
            # Volatility increase check
            volatility_increase = current_atr / max(prev_atr, 0.0001) > self.params['volatility_threshold']
            
            if volatility_increase:
                # Calculate recent trading range
                range_period = self.params['range_period']
                recent_highs = highs[-range_period:-1]
                recent_lows = lows[-range_period:-1]
                
                range_high = max(recent_highs)
                range_low = min(recent_lows)
                range_size = range_high - range_low
                
                current_price = prices[-1]
                
                # Bullish breakout: price breaks above recent range with increased volatility
                if current_price > range_high + (range_size * self.params['range_factor']):
                    # Generate long signal
                    stop_loss = max(range_low, current_price - (current_atr * 2))
                    target = current_price + (current_atr * 3)  # Using ATR for target setting
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': 'long',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_amount': current_price - stop_loss,
                        'signal_type': 'volatility_breakout',
                        'confidence': self._calculate_confidence(
                            current_price, range_high, range_low, current_atr, prev_atr, context, 'long'
                        )
                    })
                
                # Bearish breakdown: price breaks below recent range with increased volatility
                elif current_price < range_low - (range_size * self.params['range_factor']):
                    # Generate short signal
                    stop_loss = min(range_high, current_price + (current_atr * 2))
                    target = current_price - (current_atr * 3)  # Using ATR for target setting
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': 'short',
                        'entry_price': current_price,
                        'stop_loss': stop_loss,
                        'target': target,
                        'risk_amount': stop_loss - current_price,
                        'signal_type': 'volatility_breakout',
                        'confidence': self._calculate_confidence(
                            current_price, range_high, range_low, current_atr, prev_atr, context, 'short'
                        )
                    })
        
        return signals
    
    def calculate_position_size(self, signal, portfolio_value):
        """Calculate position size based on risk per trade"""
        risk_amount = signal['risk_amount']
        dollar_risk = portfolio_value * self.params['risk_per_trade']
        
        # Position size = Dollar risk / Risk amount per share
        return dollar_risk / risk_amount if risk_amount > 0 else 0
    
    def _calculate_atr(self, data, period):
        # Implementation of Average True Range
        highs = data['high']
        lows = data['low']
        closes = data['close']
        
        tr_values = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr_values.append(max(tr1, tr2, tr3))
        
        # Calculate ATR
        atr_values = [sum(tr_values[:period]) / period]
        for i in range(period, len(tr_values)):
            atr_values.append((atr_values[-1] * (period-1) + tr_values[i]) / period)
        
        # Pad beginning to match input length
        return [atr_values[0]] * (len(closes) - len(atr_values)) + atr_values
    
    def _calculate_confidence(self, current_price, range_high, range_low, current_atr, prev_atr, context, direction):
        # Base confidence
        confidence = 0.5
        
        # Adjust based on the volatility increase
        volatility_ratio = current_atr / max(prev_atr, 0.0001)
        confidence += min((volatility_ratio - 1) / 2, 0.2)
        
        # Adjust based on breakout strength
        if direction == 'long':
            breakout_strength = (current_price - range_high) / current_atr
            confidence += min(breakout_strength / 2, 0.2)
        else:
            breakout_strength = (range_low - current_price) / current_atr
            confidence += min(breakout_strength / 2, 0.2)
        
        # Adjust based on market regime - volatility breakouts work best in volatile regimes
        market_regime = context.get('market_regime', 'neutral')
        if market_regime == 'volatile':
            confidence += 0.15
        elif market_regime == 'sideways':
            confidence -= 0.05
        
        # Ensure confidence is between 0.1 and 0.9
        return max(0.1, min(0.9, confidence)) 