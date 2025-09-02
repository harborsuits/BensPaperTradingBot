"""
Breakout Swing Strategy.

This module implements a breakout swing trading strategy using price levels and volume confirmation.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("breakout_swing")

class BreakoutSwingStrategy:
    def __init__(self, params=None):
        self.params = params or {
            "lookback_period": 20,
            "breakout_factor": 1.5,  # ATR multiplier for breakout confirmation
            "volume_factor": 1.5,    # Volume increase required for confirmation
            "risk_per_trade": 0.015  # Risk 1.5% of portfolio per trade (slightly higher for breakouts)
        }
    
    def generate_signals(self, market_data, context):
        """Generate breakout swing signals based on price levels and volume confirmation"""
        signals = []
        
        for symbol, data in market_data.items():
            # Calculate indicators
            atr = self._calculate_atr(data, 14)
            
            # Get price and volume data
            prices = data['close']
            highs = data['high']
            lows = data['low']
            volumes = data['volume']
            
            # Current values
            current_price = prices[-1]
            current_volume = volumes[-1]
            
            # Calculate recent resistance and support levels
            lookback_period = self.params['lookback_period']
            recent_highs = highs[-lookback_period:-1]
            recent_lows = lows[-lookback_period:-1]
            
            resistance_level = max(recent_highs)
            support_level = min(recent_lows)
            
            # Calculate average volume
            avg_volume = sum(volumes[-lookback_period:-1]) / (lookback_period - 1)
            
            # Volume confirmation
            volume_confirmed = current_volume > avg_volume * self.params['volume_factor']
            
            # Breakout identification
            breakout_threshold = atr[-1] * self.params['breakout_factor']
            
            # Bullish breakout: price breaks above resistance with volume confirmation
            if current_price > resistance_level + breakout_threshold and volume_confirmed:
                # Generate long signal
                stop_loss = max(support_level, current_price - (atr[-1] * 2))
                target = current_price + (current_price - stop_loss) * 1.5  # 1.5:1 reward-risk
                
                signals.append({
                    'symbol': symbol,
                    'direction': 'long',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_amount': current_price - stop_loss,
                    'signal_type': 'breakout_swing',
                    'confidence': self._calculate_confidence(
                        current_price, resistance_level, breakout_threshold, 
                        current_volume, avg_volume, context, 'long'
                    )
                })
            
            # Bearish breakdown: price breaks below support with volume confirmation
            elif current_price < support_level - breakout_threshold and volume_confirmed:
                # Generate short signal
                stop_loss = min(resistance_level, current_price + (atr[-1] * 2))
                target = current_price - (stop_loss - current_price) * 1.5  # 1.5:1 reward-risk
                
                signals.append({
                    'symbol': symbol,
                    'direction': 'short',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_amount': stop_loss - current_price,
                    'signal_type': 'breakout_swing',
                    'confidence': self._calculate_confidence(
                        current_price, support_level, breakout_threshold,
                        current_volume, avg_volume, context, 'short'
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
    
    def _calculate_confidence(self, current_price, level, threshold, current_volume, avg_volume, context, direction):
        # Base confidence
        confidence = 0.5
        
        # Adjust based on the strength of the breakout
        if direction == 'long':
            # Distance above resistance
            breakout_strength = (current_price - level) / threshold
            confidence += min(breakout_strength / 4, 0.2)
        else:
            # Distance below support
            breakout_strength = (level - current_price) / threshold
            confidence += min(breakout_strength / 4, 0.2)
        
        # Adjust based on volume confirmation
        volume_ratio = current_volume / avg_volume
        confidence += min((volume_ratio - 1) / 2, 0.2)
        
        # Adjust based on market regime
        market_regime = context.get('market_regime', 'neutral')
        if (direction == 'long' and market_regime == 'bullish') or \
           (direction == 'short' and market_regime == 'bearish'):
            confidence += 0.1
        elif (direction == 'long' and market_regime == 'bearish') or \
             (direction == 'short' and market_regime == 'bullish'):
            confidence -= 0.1
        
        # Ensure confidence is between 0.1 and 0.9
        return max(0.1, min(0.9, confidence)) 