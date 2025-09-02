#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Signal Quality Enhancer

This module implements advanced signal quality enhancements such as:
1. Signal Strength Metadata - volatility, news sensitivity, liquidity
2. Multi-Timeframe Confirmation - confirming signals on higher timeframes
3. Volume Spike Confirmation - requiring volume confirmation for breakouts
4. Market Breadth Context - considering sector/index strength
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Union, List, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.factory.strategy_template import Signal, SignalType

logger = logging.getLogger(__name__)

class SignalQualityEnhancer:
    """
    Signal Quality Enhancer
    
    This class provides methods to enhance and validate trading signals:
    - Adds metadata to signals about market conditions
    - Confirms signals across multiple timeframes
    - Validates volume conditions
    - Adds market breadth context
    """
    
    # Default parameters - can be overridden via constructor
    DEFAULT_PARAMS = {
        # Volume confirmation
        'volume_spike_threshold': 1.5,  # Minimum volume spike multiple of average
        'volume_lookback_period': 20,   # Periods to consider for volume average
        
        # Multi-timeframe confirmation
        'require_timeframe_confirmation': True,
        'timeframe_confirmation_levels': [2, 4],  # Multiple of base timeframe (e.g., 5min → 10min, 20min)
        'confirmation_threshold': 0.7,  # Required agreement threshold (0.0-1.0)
        
        # Volatility thresholds
        'high_volatility_threshold': 1.5,  # ATR multiple for high volatility
        'low_volatility_threshold': 0.7,   # ATR multiple for low volatility
        
        # Liquidity thresholds
        'min_liquidity_score': 0.4,  # Minimum liquidity score (0.0-1.0)
        
        # News sensitivity
        'news_impact_window_hours': 24,  # Hours to consider recent news
        
        # Market breadth
        'require_market_breadth_check': True,
        'market_breadth_threshold': 0.6,  # Required breadth score (0.0-1.0)
    }
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize Signal Quality Enhancer.
        
        Args:
            parameters: Custom parameters to override defaults
        """
        # Set parameters
        self.parameters = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.parameters.update(parameters)
            
        logger.info(f"Signal Quality Enhancer initialized with parameters: {self.parameters}")
        
        # Initialize caches for performance
        self._volatility_cache = {}
        self._liquidity_cache = {}
        self._news_impact_cache = {}
        self._breadth_cache = {}
        
    def enhance_signal(self, 
                     signal: Signal, 
                     data: Dict[str, pd.DataFrame],
                     market_data: Optional[Dict[str, Any]] = None,
                     news_data: Optional[Dict[str, Any]] = None) -> Signal:
        """
        Enhance a signal with quality metadata and confirmations.
        
        Args:
            signal: The original trading signal
            data: Dictionary of timeframe data (key = timeframe, value = DataFrame)
            market_data: Optional market context data
            news_data: Optional recent news data
        
        Returns:
            Enhanced signal with additional metadata
        """
        # Create a copy of the signal to avoid modifying the original
        enhanced_signal = signal.copy()
        
        # Initialize metadata if it doesn't exist
        if enhanced_signal.metadata is None:
            enhanced_signal.metadata = {}
            
        # Add signal strength metadata
        self._add_volatility_metadata(enhanced_signal, data)
        self._add_liquidity_metadata(enhanced_signal, data, market_data)
        self._add_news_sensitivity(enhanced_signal, news_data)
        
        # Add confirmation metadata
        if self.parameters['require_timeframe_confirmation']:
            confirmed = self._confirm_on_multiple_timeframes(enhanced_signal, data)
            enhanced_signal.metadata['timeframe_confirmed'] = confirmed
            
        # Add volume confirmation
        volume_confirmed = self._confirm_volume_spike(enhanced_signal, data)
        enhanced_signal.metadata['volume_confirmed'] = volume_confirmed
        
        # Add market breadth context for stocks
        if (self.parameters['require_market_breadth_check'] and 
            market_data is not None and 
            signal.asset_class == 'stock'):
            breadth_aligned = self._check_market_breadth(enhanced_signal, market_data)
            enhanced_signal.metadata['breadth_aligned'] = breadth_aligned
            
        # Calculate overall signal quality score
        enhanced_signal.metadata['quality_score'] = self._calculate_quality_score(enhanced_signal)
        
        return enhanced_signal
    
    def is_valid_signal(self, signal: Signal) -> bool:
        """
        Check if a signal passes all quality thresholds.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal passes all checks, False otherwise
        """
        # Signal must have metadata
        if not hasattr(signal, 'metadata') or signal.metadata is None:
            return False
            
        # Check timeframe confirmation if required
        if (self.parameters['require_timeframe_confirmation'] and 
            ('timeframe_confirmed' not in signal.metadata or 
             not signal.metadata['timeframe_confirmed'])):
            logger.debug(f"Signal rejected: Failed timeframe confirmation for {signal.symbol}")
            return False
            
        # Check volume confirmation for breakouts
        if (signal.signal_type in [SignalType.BREAKOUT_LONG, SignalType.BREAKOUT_SHORT] and 
            ('volume_confirmed' not in signal.metadata or 
             not signal.metadata['volume_confirmed'])):
            logger.debug(f"Signal rejected: Failed volume confirmation for {signal.symbol} breakout")
            return False
            
        # Check market breadth alignment for stocks
        if (signal.asset_class == 'stock' and 
            self.parameters['require_market_breadth_check'] and 
            ('breadth_aligned' not in signal.metadata or 
             not signal.metadata['breadth_aligned'])):
            logger.debug(f"Signal rejected: Failed market breadth alignment for {signal.symbol}")
            return False
            
        # Check overall quality score
        min_quality = 0.5  # Minimum quality threshold
        if ('quality_score' not in signal.metadata or 
            signal.metadata['quality_score'] < min_quality):
            logger.debug(f"Signal rejected: Quality score too low ({signal.metadata.get('quality_score', 0)}) for {signal.symbol}")
            return False
            
        return True
    
    def _add_volatility_metadata(self, signal: Signal, data: Dict[str, pd.DataFrame]) -> None:
        """
        Add volatility metadata to the signal.
        
        Args:
            signal: Signal to enhance
            data: Dictionary of timeframe data
        """
        # Get base timeframe data
        base_tf = list(data.keys())[0]  # Assume first timeframe is base
        base_data = data[base_tf]
        
        # Calculate ATR if we have enough data
        if len(base_data) >= 14:
            # Calculate TR (True Range)
            high = base_data['high'].values
            low = base_data['low'].values
            close = base_data['close'].values
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR (14-period simple average)
            atr = np.mean(tr[-14:])
            
            # Calculate current volatility compared to average
            volatility_ratio = tr[-1] / atr if atr > 0 else 1.0
            
            # Classify volatility
            if volatility_ratio >= self.parameters['high_volatility_threshold']:
                volatility_class = 'high'
            elif volatility_ratio <= self.parameters['low_volatility_threshold']:
                volatility_class = 'low'
            else:
                volatility_class = 'normal'
            
            # Add to signal metadata
            signal.metadata['volatility'] = {
                'atr': atr,
                'current_tr': tr[-1],
                'volatility_ratio': volatility_ratio,
                'volatility_class': volatility_class
            }
            
            # Cache for future reference
            self._volatility_cache[signal.symbol] = {
                'timestamp': datetime.now(),
                'data': signal.metadata['volatility']
            }
    
    def _add_liquidity_metadata(self, 
                              signal: Signal, 
                              data: Dict[str, pd.DataFrame],
                              market_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add liquidity metadata to the signal.
        
        Args:
            signal: Signal to enhance
            data: Dictionary of timeframe data
            market_data: Optional market context data
        """
        # Get base timeframe data
        base_tf = list(data.keys())[0]  # Assume first timeframe is base
        base_data = data[base_tf]
        
        # Calculate liquidity metrics if we have volume data
        if 'volume' in base_data.columns and len(base_data) > 0:
            # Calculate average volume
            avg_volume = base_data['volume'].tail(20).mean()
            current_volume = base_data['volume'].iloc[-1]
            
            # Calculate volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Calculate spread if available in market data
            spread = None
            if market_data and 'spread' in market_data.get(signal.symbol, {}):
                spread = market_data[signal.symbol]['spread']
            
            # Calculate bid-ask depth if available
            depth = None
            if market_data and 'depth' in market_data.get(signal.symbol, {}):
                depth = market_data[signal.symbol]['depth']
            
            # Calculate liquidity score (0.0-1.0)
            # Higher is better (more liquid)
            liquidity_score = 0.5  # Default mid-range
            
            if volume_ratio > 0:
                # Adjust score based on volume (0.0-0.7 range)
                liquidity_score = min(0.7, (volume_ratio / 3))
            
            # Adjust for spread if available (can reduce score by up to 0.3)
            if spread is not None:
                # Normalize spread to 0-0.3 range (higher spread = lower score)
                spread_factor = max(0, 0.3 - (0.3 * min(1, spread / 10)))
                liquidity_score += spread_factor
            
            # Add to signal metadata
            signal.metadata['liquidity'] = {
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'volume_ratio': volume_ratio,
                'spread': spread,
                'depth': depth,
                'liquidity_score': liquidity_score
            }
            
            # Cache for future reference
            self._liquidity_cache[signal.symbol] = {
                'timestamp': datetime.now(),
                'data': signal.metadata['liquidity']
            }
    
    def _add_news_sensitivity(self, 
                            signal: Signal, 
                            news_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add news sensitivity metadata to the signal.
        
        Args:
            signal: Signal to enhance
            news_data: Optional recent news data
        """
        if news_data is None or signal.symbol not in news_data:
            # Default values when no news data is available
            signal.metadata['news_sensitivity'] = {
                'recent_news_count': 0,
                'has_significant_news': False,
                'news_sentiment': 'neutral',
                'news_impact_score': 0.0
            }
            return
            
        # Process news data
        symbol_news = news_data[signal.symbol]
        
        # Count recent news
        recent_news_count = len(symbol_news.get('recent', []))
        
        # Check for significant news
        has_significant_news = any(item.get('significance', 0) > 0.7 
                                 for item in symbol_news.get('recent', []))
        
        # Calculate aggregate sentiment
        sentiments = [item.get('sentiment', 0) for item in symbol_news.get('recent', [])]
        avg_sentiment = sum(sentiments) / max(1, len(sentiments))
        
        # Classify sentiment
        if avg_sentiment > 0.2:
            sentiment_class = 'positive'
        elif avg_sentiment < -0.2:
            sentiment_class = 'negative'
        else:
            sentiment_class = 'neutral'
            
        # Calculate news impact score (0.0-1.0)
        # Higher means more impactful news
        impact_score = min(1.0, (recent_news_count / 5) + 
                         (0.5 if has_significant_news else 0))
        
        # Add to signal metadata
        signal.metadata['news_sensitivity'] = {
            'recent_news_count': recent_news_count,
            'has_significant_news': has_significant_news,
            'news_sentiment': sentiment_class,
            'news_impact_score': impact_score
        }
        
        # Cache for future reference
        self._news_impact_cache[signal.symbol] = {
            'timestamp': datetime.now(),
            'data': signal.metadata['news_sensitivity']
        }
    
    def _confirm_on_multiple_timeframes(self, 
                                      signal: Signal, 
                                      data: Dict[str, pd.DataFrame]) -> bool:
        """
        Confirm signal on multiple timeframes.
        
        Args:
            signal: Signal to confirm
            data: Dictionary of timeframe data
        
        Returns:
            True if signal is confirmed on higher timeframes
        """
        # Get base timeframe and data
        timeframes = list(data.keys())
        if len(timeframes) < 2:
            # Not enough timeframes available for confirmation
            return False
            
        base_tf = timeframes[0]
        confirmations = 0
        total_checks = 0
        
        # Check each higher timeframe
        for tf_level in self.parameters['timeframe_confirmation_levels']:
            # Find closest matching timeframe
            target_tf = None
            for tf in timeframes[1:]:  # Skip base timeframe
                if tf == base_tf * tf_level:
                    target_tf = tf
                    break
            
            if target_tf is None or target_tf not in data:
                continue  # Skip if timeframe not available
                
            # Get higher timeframe data
            higher_tf_data = data[target_tf]
            if len(higher_tf_data) < 2:
                continue  # Not enough data
                
            # Perform confirmation check based on signal type
            confirmed = False
            if signal.signal_type in [SignalType.BUY, SignalType.BREAKOUT_LONG]:
                # For bullish signals, check if higher timeframe is also bullish
                
                # Simple trend direction check (close > previous close)
                current_close = higher_tf_data['close'].iloc[-1]
                prev_close = higher_tf_data['close'].iloc[-2]
                
                # Simple moving average direction check if available
                trend_up = current_close > prev_close
                
                # Check if recent candle is bullish
                recent_bullish = higher_tf_data['close'].iloc[-1] > higher_tf_data['open'].iloc[-1]
                
                # Consider confirmed if both checks pass
                confirmed = trend_up and recent_bullish
                
            elif signal.signal_type in [SignalType.SELL, SignalType.BREAKOUT_SHORT]:
                # For bearish signals, check if higher timeframe is also bearish
                
                # Simple trend direction check (close < previous close)
                current_close = higher_tf_data['close'].iloc[-1]
                prev_close = higher_tf_data['close'].iloc[-2]
                
                # Check trend direction
                trend_down = current_close < prev_close
                
                # Check if recent candle is bearish
                recent_bearish = higher_tf_data['close'].iloc[-1] < higher_tf_data['open'].iloc[-1]
                
                # Consider confirmed if both checks pass
                confirmed = trend_down and recent_bearish
            
            # Count confirmation
            if confirmed:
                confirmations += 1
            total_checks += 1
            
        # Add detailed confirmation data to signal
        signal.metadata['timeframe_confirmation'] = {
            'confirmations': confirmations,
            'total_checks': total_checks,
            'confirmation_ratio': confirmations / max(1, total_checks)
        }
        
        # Return overall confirmation result
        return (confirmations / max(1, total_checks)) >= self.parameters['confirmation_threshold']
    
    def _confirm_volume_spike(self, 
                            signal: Signal, 
                            data: Dict[str, pd.DataFrame]) -> bool:
        """
        Confirm volume spike for breakout signals.
        
        Args:
            signal: Signal to confirm
            data: Dictionary of timeframe data
        
        Returns:
            True if volume spike is confirmed
        """
        # Only apply to breakout signals and signals with entry_type='breakout'
        is_breakout = (
            signal.signal_type in [SignalType.BREAKOUT_LONG, SignalType.BREAKOUT_SHORT] or
            signal.metadata.get('entry_type') == 'breakout'
        )
        
        if not is_breakout:
            return True  # Not a breakout signal, so no volume confirmation needed
            
        # Get base timeframe data
        base_tf = list(data.keys())[0]  # Assume first timeframe is base
        base_data = data[base_tf]
        
        # Check if we have volume data
        if 'volume' not in base_data.columns or len(base_data) < self.parameters['volume_lookback_period']:
            return False  # Not enough data
            
        # Calculate average volume
        avg_volume = base_data['volume'].tail(self.parameters['volume_lookback_period']).mean()
        current_volume = base_data['volume'].iloc[-1]
        
        # Calculate volume ratio
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Add volume data to signal metadata
        signal.metadata['volume_data'] = {
            'avg_volume': avg_volume,
            'current_volume': current_volume,
            'volume_ratio': volume_ratio,
            'volume_threshold': self.parameters['volume_spike_threshold'],
            'volume_confirmed': volume_ratio >= self.parameters['volume_spike_threshold']
        }
        
        # Return confirmation result
        return volume_ratio >= self.parameters['volume_spike_threshold']
    
    def _check_market_breadth(self, 
                            signal: Signal, 
                            market_data: Dict[str, Any]) -> bool:
        """
        Check market breadth alignment for stock signals.
        
        Args:
            signal: Signal to check
            market_data: Market context data
        
        Returns:
            True if signal aligns with market breadth
        """
        # Only apply to stock signals
        if signal.asset_class != 'stock':
            return True  # Not a stock signal, so no breadth check needed
            
        # Extract market breadth data if available
        if 'market_breadth' not in market_data:
            return True  # No breadth data available
            
        breadth_data = market_data['market_breadth']
        
        # Get relevant sector data
        sector = market_data.get(signal.symbol, {}).get('sector', None)
        if sector is None:
            # No sector data, use overall market breadth
            sector = 'market'
            
        # Get breadth metrics
        advance_decline_ratio = breadth_data.get(sector, {}).get('advance_decline_ratio', 1.0)
        up_volume_ratio = breadth_data.get(sector, {}).get('up_volume_ratio', 0.5)
        
        # Calculate breadth score (0.0-1.0)
        breadth_score = 0.5  # Default neutral
        
        if advance_decline_ratio > 1.0:
            # Bullish breadth (more advances than declines)
            breadth_score = 0.5 + min(0.5, (advance_decline_ratio - 1.0) / 4)
        else:
            # Bearish breadth (more declines than advances)
            breadth_score = 0.5 - min(0.5, (1.0 - advance_decline_ratio) / 4)
            
        # Adjust based on up/down volume
        if up_volume_ratio > 0.5:
            # More up volume than down volume (bullish)
            breadth_score += (up_volume_ratio - 0.5) * 0.5
        else:
            # More down volume than up volume (bearish)
            breadth_score -= (0.5 - up_volume_ratio) * 0.5
            
        # Ensure score stays in 0.0-1.0 range
        breadth_score = max(0, min(1, breadth_score))
        
        # Check if signal aligns with breadth
        aligned = False
        if signal.signal_type in [SignalType.BUY, SignalType.BREAKOUT_LONG]:
            # For bullish signals, breadth should be bullish (score > threshold)
            aligned = breadth_score >= self.parameters['market_breadth_threshold']
        elif signal.signal_type in [SignalType.SELL, SignalType.BREAKOUT_SHORT]:
            # For bearish signals, breadth should be bearish (score < 1-threshold)
            aligned = breadth_score <= (1.0 - self.parameters['market_breadth_threshold'])
        
        # Add breadth data to signal metadata
        signal.metadata['market_breadth'] = {
            'sector': sector,
            'advance_decline_ratio': advance_decline_ratio,
            'up_volume_ratio': up_volume_ratio,
            'breadth_score': breadth_score,
            'aligned': aligned
        }
        
        # Cache for future reference
        self._breadth_cache[signal.symbol] = {
            'timestamp': datetime.now(),
            'data': signal.metadata['market_breadth']
        }
        
        return aligned
    
    def _calculate_quality_score(self, signal: Signal) -> float:
        """
        Calculate overall signal quality score.
        
        Args:
            signal: Signal to score
            
        Returns:
            Quality score (0.0-1.0)
        """
        score_components = []
        
        # Add volatility component
        if 'volatility' in signal.metadata:
            volatility = signal.metadata['volatility']
            # Prefer normal volatility, penalize extremes
            vol_score = 0.8
            if volatility.get('volatility_class') == 'high':
                vol_score = 0.6  # High volatility is okay but not ideal
            elif volatility.get('volatility_class') == 'low':
                vol_score = 0.4  # Low volatility is least preferred
            score_components.append(vol_score)
        
        # Add liquidity component
        if 'liquidity' in signal.metadata:
            liquidity = signal.metadata['liquidity']
            # Higher liquidity is better
            liq_score = min(1.0, liquidity.get('liquidity_score', 0.5))
            score_components.append(liq_score)
        
        # Add news component
        if 'news_sensitivity' in signal.metadata:
            news = signal.metadata['news_sensitivity']
            # Prefer no significant news (less unexpected price movements)
            news_score = 0.8
            if news.get('has_significant_news', False):
                # Check sentiment alignment with signal
                if ((signal.signal_type in [SignalType.BUY, SignalType.BREAKOUT_LONG] and 
                     news.get('news_sentiment') == 'positive') or
                    (signal.signal_type in [SignalType.SELL, SignalType.BREAKOUT_SHORT] and 
                     news.get('news_sentiment') == 'negative')):
                    news_score = 0.9  # Aligned sentiment is good
                elif ((signal.signal_type in [SignalType.BUY, SignalType.BREAKOUT_LONG] and 
                       news.get('news_sentiment') == 'negative') or
                      (signal.signal_type in [SignalType.SELL, SignalType.BREAKOUT_SHORT] and 
                       news.get('news_sentiment') == 'positive')):
                    news_score = 0.3  # Contradictory sentiment is bad
                else:
                    news_score = 0.6  # Neutral sentiment with significant news
            score_components.append(news_score)
        
        # Add timeframe confirmation component
        if 'timeframe_confirmation' in signal.metadata:
            confirmation = signal.metadata['timeframe_confirmation']
            # Higher confirmation ratio is better
            conf_score = confirmation.get('confirmation_ratio', 0) * 0.8 + 0.2
            score_components.append(conf_score)
        
        # Add volume component for breakouts
        if 'volume_data' in signal.metadata:
            volume = signal.metadata['volume_data']
            # Higher volume ratio is better for breakouts
            vol_ratio = volume.get('volume_ratio', 0)
            vol_score = min(1.0, vol_ratio / self.parameters['volume_spike_threshold'])
            score_components.append(vol_score)
        
        # Add market breadth component
        if 'market_breadth' in signal.metadata:
            breadth = signal.metadata['market_breadth']
            # Alignment with breadth is good
            if breadth.get('aligned', False):
                breadth_score = 0.9
            else:
                breadth_score = 0.3
            score_components.append(breadth_score)
        
        # Calculate overall score (average of components)
        if not score_components:
            return 0.5  # Default score if no components available
            
        return sum(score_components) / len(score_components)
