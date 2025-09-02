"""
Momentum Strategy.

This module implements a momentum trading strategy based on RSI and Rate of Change.
"""

import logging
from typing import Dict, List, Any, Optional

# Configure logging
logger = logging.getLogger("momentum")

class MomentumStrategy:
    def __init__(self, params=None):
        self.params = params or {
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "rate_of_change_period": 10,
            "risk_per_trade": 0.01  # Risk 1% of portfolio per trade
        }
    
    def generate_signals(self, market_data, context):
        """Generate momentum signals based on RSI and Rate of Change"""
        signals = []
        
        for symbol, data in market_data.items():
            # Calculate indicators
            rsi = self._calculate_rsi(data['close'], self.params['rsi_period'])
            roc = self._calculate_rate_of_change(data['close'], self.params['rate_of_change_period'])
            
            # Current price
            current_price = data['close'][-1]
            
            # RSI values
            current_rsi = rsi[-1]
            prev_rsi = rsi[-2]
            
            # ROC values
            current_roc = roc[-1]
            
            # Volatility for stop loss calculation
            atr = self._calculate_atr(data, 14)
            
            # Long conditions: RSI crossing up from oversold and positive ROC
            if prev_rsi < self.params['rsi_oversold'] and current_rsi > self.params['rsi_oversold'] and current_roc > 0:
                # Generate long signal
                stop_loss = current_price - (atr[-1] * 2)
                target = current_price + (atr[-1] * 3)  # 1.5:1 reward-risk
                
                signals.append({
                    'symbol': symbol,
                    'direction': 'long',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_amount': current_price - stop_loss,
                    'signal_type': 'momentum',
                    'confidence': self._calculate_confidence(data, context, rsi[-1], roc[-1], 'long')
                })
            
            # Short conditions: RSI crossing down from overbought and negative ROC
            elif prev_rsi > self.params['rsi_overbought'] and current_rsi < self.params['rsi_overbought'] and current_roc < 0:
                # Generate short signal
                stop_loss = current_price + (atr[-1] * 2)
                target = current_price - (atr[-1] * 3)  # 1.5:1 reward-risk
                
                signals.append({
                    'symbol': symbol,
                    'direction': 'short',
                    'entry_price': current_price,
                    'stop_loss': stop_loss,
                    'target': target,
                    'risk_amount': stop_loss - current_price,
                    'signal_type': 'momentum',
                    'confidence': self._calculate_confidence(data, context, rsi[-1], roc[-1], 'short')
                })
        
        return signals
    
    def calculate_position_size(self, signal, portfolio_value):
        """Calculate position size based on risk per trade"""
        risk_amount = signal['risk_amount']
        dollar_risk = portfolio_value * self.params['risk_per_trade']
        
        # Position size = Dollar risk / Risk amount per share
        return dollar_risk / risk_amount if risk_amount > 0 else 0
    
    def _calculate_rsi(self, prices, period):
        # Implementation of Relative Strength Index
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # First average
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Subsequent averages
        rsi_values = []
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        # Pad beginning to match input length
        return [rsi_values[0]] * (len(prices) - len(rsi_values)) + rsi_values
    
    def _calculate_rate_of_change(self, prices, period):
        # Rate of Change calculation: (Current Price / Price n periods ago) - 1
        roc = []
        for i in range(len(prices)):
            if i < period:
                roc.append(0)
            else:
                roc.append((prices[i] / prices[i-period] - 1) * 100)
        return roc
    
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
    
    def _calculate_confidence(self, data, context, rsi, roc, direction):
        # Base confidence
        confidence = 0.5
        
        # Adjust for RSI extremes
        if direction == 'long':
            # Stronger signal when RSI is more oversold
            confidence += (30 - min(rsi, 30)) / 30 * 0.2
        else:
            # Stronger signal when RSI is more overbought
            confidence += (max(rsi, 70) - 70) / 30 * 0.2
        
        # Adjust based on ROC magnitude
        confidence += min(abs(roc) / 10, 0.2)
        
        # Adjust based on market regime
        market_regime = context.get('market_regime', 'neutral')
        if direction == 'long' and market_regime == 'bullish':
            confidence += 0.1
        elif direction == 'short' and market_regime == 'bearish':
            confidence += 0.1
        elif (direction == 'long' and market_regime == 'bearish') or \
             (direction == 'short' and market_regime == 'bullish'):
            confidence -= 0.1
        
        # Ensure confidence is between 0.1 and 0.9
        return max(0.1, min(0.9, confidence)) 