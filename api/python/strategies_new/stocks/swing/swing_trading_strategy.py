#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Swing Trading Strategy

A professional-grade swing trading implementation that leverages the modular,
event-driven architecture. This strategy is designed to capture medium-term
price movements over several days to weeks.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.options.base.strategy_adjustments import StrategyAdjustments

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="SwingTradingStrategy",
    market_type="stocks",
    description="A medium-term trading strategy that aims to capture significant price swings over several days to weeks",
    timeframes=["1d", "1h", "4h"],
    parameters={
        "trend_lookback_period": {"description": "Lookback period for trend determination", "type": "integer"},
        "min_swing_potential": {"description": "Minimum potential swing magnitude as % of price", "type": "float"},
        "use_max_drawdown_filter": {"description": "Whether to filter by max drawdown", "type": "boolean"}
    }
)
class SwingTradingStrategy(StocksBaseStrategy, StrategyAdjustments):
    """
    Swing Trading Strategy
    
    This strategy captures medium-term price movements, typically over a period
    of days to weeks. It identifies stocks at the beginning of potential swings
    and holds them until the swing completes.
    
    Features:
    - Multiple swing identification techniques including technical and price action
    - Adaptive position sizing based on volatility and conviction level
    - Support for both long and short swings
    - Event-driven design that reacts to market events and news catalysts
    """
    
    def __init__(self, session, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Swing Trading strategy.
        
        Args:
            session: Trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize base class
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Strategy identification
            'strategy_name': 'Swing Trading',
            'strategy_id': 'swing_trading',
            
            # Swing trading specific parameters
            'trend_lookback_period': 20,    # Days to look back for trend
            'min_swing_potential': 3.0,     # Minimum swing potential in %
            'min_reward_risk_ratio': 2.0,   # Minimum reward/risk ratio
            'min_win_rate': 50,             # Minimum expected win rate
            
            # Technical filters
            'use_rsi_filter': True,         # Filter by RSI
            'rsi_entry_lower': 30,          # RSI oversold threshold for longs
            'rsi_entry_upper': 70,          # RSI overbought threshold for shorts
            'use_adr_filter': True,         # Filter by average daily range
            'min_adr_percent': 1.5,         # Minimum ADR as % of price
            'use_volume_filter': True,      # Filter by volume
            'min_volume_multiple': 1.5,     # Minimum volume as multiple of avg
            
            # Risk management
            'max_position_size_pct': 5.0,   # Max position size as % of portfolio
            'stop_loss_pct': 3.0,           # Initial stop loss as % of entry price
            'trailing_stop_pct': 2.0,       # Trailing stop as % of high/low
            'profit_target_pct': 10.0,      # Profit target as % of entry price
            'use_max_drawdown_filter': True, # Filter by maximum drawdown
            'max_drawdown_pct': 10.0,        # Maximum acceptable drawdown
            
            # Entry/exit timing
            'allow_partial_entries': True,  # Allow scaling into positions
            'allow_partial_exits': True,    # Allow scaling out of positions
            'max_days_in_trade': 20,        # Maximum days to hold a position
            'min_days_in_trade': 2,         # Minimum days to hold a position
            
            # Market condition filters
            'require_bullish_market': False, # Whether to only enter long in bullish market
            'require_bearish_market': False, # Whether to only enter short in bearish market
            
            # Swing pattern recognition
            'patterns_to_recognize': [     # Swing patterns to recognize
                'double_bottom',
                'double_top',
                'head_and_shoulders',
                'inverse_head_and_shoulders',
                'triangle_breakout',
                'support_bounce',
                'resistance_breakdown'
            ],
            'pattern_recognition_threshold': 70, # Pattern recognition confidence threshold
        }
        
        # Apply defaults for any missing parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
                
        # Initialize strategy state
        self.swing_patterns_detected = {}
        self.current_market_bias = "neutral"  # "bullish", "bearish", or "neutral"
        self.last_swing_scan_time = None
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Swing Trading strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        # Start with parent class indicators
        indicators = super().calculate_indicators(data)
        
        if data.empty or len(data) < self.parameters['trend_lookback_period']:
            return indicators
        
        try:
            # Calculate trend strength indicators
            lookback = self.parameters['trend_lookback_period']
            
            # Moving averages
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
            indicators['sma_200'] = data['close'].rolling(window=200).mean()
            
            # Determine trend direction and strength
            indicators['trend_direction'] = np.where(
                indicators['sma_20'] > indicators['sma_50'], 
                1, 
                np.where(indicators['sma_20'] < indicators['sma_50'], -1, 0)
            )
            
            # Calculate Average True Range (ATR) for volatility
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr_14'] = true_range.rolling(window=14).mean()
            
            # ADR as percentage of price
            indicators['adr_percent'] = indicators['atr_14'] / data['close'] * 100
            
            # Volume analysis
            indicators['volume_sma_20'] = data['volume'].rolling(window=20).mean()
            indicators['relative_volume'] = data['volume'] / indicators['volume_sma_20']
            
            # RSI for momentum
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            indicators['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Swing potential calculation
            recent_high = data['high'].rolling(window=lookback).max()
            recent_low = data['low'].rolling(window=lookback).min()
            price_range = recent_high - recent_low
            indicators['swing_potential'] = price_range / data['close'] * 100
            
            # Scoring for swing trade quality
            swing_quality_score = 0
            
            # Component 1: Trend alignment (20%)
            if indicators['trend_direction'].iloc[-1] == 1:  # Uptrend
                swing_quality_score += 20
            elif indicators['trend_direction'].iloc[-1] == -1:  # Downtrend
                swing_quality_score += 10  # Downtrends less predictable
            
            # Component 2: Volatility suitable for swings (20%)
            avg_adr = indicators['adr_percent'].iloc[-5:].mean()
            if 1.5 <= avg_adr <= 3.0:  # Ideal range for swing trading
                swing_quality_score += 20
            elif avg_adr > 3.0:
                swing_quality_score += 10  # Too volatile
            elif avg_adr > 1.0:
                swing_quality_score += 15  # Still acceptable
            
            # Component 3: Volume confirmation (20%)
            avg_rel_volume = indicators['relative_volume'].iloc[-5:].mean()
            if avg_rel_volume >= 1.5:
                swing_quality_score += 20
            elif avg_rel_volume >= 1.0:
                swing_quality_score += 15
            
            # Component 4: RSI in good swing range (20%)
            latest_rsi = indicators['rsi_14'].iloc[-1]
            if 30 <= latest_rsi <= 70:
                swing_quality_score += 20
            elif 20 <= latest_rsi <= 80:
                swing_quality_score += 10
            
            # Component 5: Moving average alignment (20%)
            current_price = data['close'].iloc[-1]
            if (current_price > indicators['sma_20'].iloc[-1] > 
                indicators['sma_50'].iloc[-1] > indicators['sma_200'].iloc[-1]):
                # Strong uptrend - good for long swings
                swing_quality_score += 20
            elif (current_price < indicators['sma_20'].iloc[-1] < 
                  indicators['sma_50'].iloc[-1] < indicators['sma_200'].iloc[-1]):
                # Strong downtrend - good for short swings
                swing_quality_score += 20
            elif (abs(current_price - indicators['sma_20'].iloc[-1]) / current_price < 0.02):
                # Price near MA - potential reversal point
                swing_quality_score += 15
            
            indicators['swing_quality_score'] = swing_quality_score
            
        except Exception as e:
            logger.error(f"Error calculating swing trading indicators: {str(e)}")
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for Swing Trading strategy.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "direction": None,  # "long" or "short"
            "exit_positions": [],
            "conviction": 0.0,  # 0.0 to 1.0
            "stop_price": None,
            "target_price": None,
            "pattern_detected": None
        }
        
        try:
            if data.empty or len(data) < 50:
                return signals
                
            current_price = data['close'].iloc[-1]
            
            # Check for minimum swing potential
            min_potential = self.parameters['min_swing_potential']
            if 'swing_potential' in indicators:
                recent_potential = indicators['swing_potential'].iloc[-1]
                if recent_potential < min_potential:
                    logger.info(f"Insufficient swing potential: {recent_potential:.2f}% < {min_potential:.2f}%")
                    return signals
            
            # Check for swing quality score
            if 'swing_quality_score' in indicators:
                quality_score = indicators['swing_quality_score']
                if quality_score < 60:  # Threshold for reasonable quality
                    logger.info(f"Low swing quality score: {quality_score} < 60")
                    return signals
            
            # Check market conditions
            market_bullish = (
                indicators['sma_20'].iloc[-1] > indicators['sma_50'].iloc[-1] and
                indicators['sma_50'].iloc[-1] > indicators['sma_200'].iloc[-1]
            )
            market_bearish = (
                indicators['sma_20'].iloc[-1] < indicators['sma_50'].iloc[-1] and
                indicators['sma_50'].iloc[-1] < indicators['sma_200'].iloc[-1]
            )
            
            # Update market bias
            if market_bullish:
                self.current_market_bias = "bullish"
            elif market_bearish:
                self.current_market_bias = "bearish"
            else:
                self.current_market_bias = "neutral"
            
            # Look for long swing entry
            long_entry = False
            long_conviction = 0.0
            
            # RSI oversold and bouncing
            if ('rsi_14' in indicators and 
                indicators['rsi_14'].iloc[-1] < self.parameters['rsi_entry_lower'] and
                indicators['rsi_14'].iloc[-1] > indicators['rsi_14'].iloc[-2]):
                long_entry = True
                long_conviction += 0.3
                signals["pattern_detected"] = "oversold_bounce"
            
            # Support level bounce
            recent_lows = data['low'].iloc[-20:].rolling(window=5).min()
            support_level = recent_lows.min()
            if (abs(support_level - current_price) / current_price < 0.02 and
                data['close'].iloc[-1] > data['open'].iloc[-1]):  # Bullish candle
                long_entry = True
                long_conviction += 0.3
                signals["pattern_detected"] = "support_bounce"
            
            # Look for short swing entry
            short_entry = False
            short_conviction = 0.0
            
            # RSI overbought and rolling over
            if ('rsi_14' in indicators and 
                indicators['rsi_14'].iloc[-1] > self.parameters['rsi_entry_upper'] and
                indicators['rsi_14'].iloc[-1] < indicators['rsi_14'].iloc[-2]):
                short_entry = True
                short_conviction += 0.3
                signals["pattern_detected"] = "overbought_rollover"
            
            # Resistance level rejection
            recent_highs = data['high'].iloc[-20:].rolling(window=5).max()
            resistance_level = recent_highs.max()
            if (abs(resistance_level - current_price) / current_price < 0.02 and
                data['close'].iloc[-1] < data['open'].iloc[-1]):  # Bearish candle
                short_entry = True
                short_conviction += 0.3
                signals["pattern_detected"] = "resistance_rejection"
            
            # Final decision based on market bias and entry signals
            if long_entry and (self.current_market_bias != "bearish" or not self.parameters['require_bullish_market']):
                signals["entry"] = True
                signals["direction"] = "long"
                signals["conviction"] = long_conviction
                
                # Calculate stop and target prices for long position
                atr = indicators['atr_14'].iloc[-1] if 'atr_14' in indicators else current_price * 0.02
                signals["stop_price"] = current_price - (atr * 2)
                signals["target_price"] = current_price + (atr * self.parameters['min_reward_risk_ratio'] * 2)
                
            elif short_entry and (self.current_market_bias != "bullish" or not self.parameters['require_bearish_market']):
                signals["entry"] = True
                signals["direction"] = "short"
                signals["conviction"] = short_conviction
                
                # Calculate stop and target prices for short position
                atr = indicators['atr_14'].iloc[-1] if 'atr_14' in indicators else current_price * 0.02
                signals["stop_price"] = current_price + (atr * 2)
                signals["target_price"] = current_price - (atr * self.parameters['min_reward_risk_ratio'] * 2)
                
            # Check for exit signals on existing positions
            for position in self.positions:
                if self._check_exit_conditions(position, data, indicators):
                    signals["exit_positions"].append(position.position_id)
                
        except Exception as e:
            logger.error(f"Error generating swing trading signals: {str(e)}")
        
        return signals
    
    def _check_exit_conditions(self, position, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open position.
        
        Args:
            position: The position to check
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Boolean indicating whether to exit
        """
        try:
            if not hasattr(position, 'entry_price') or not hasattr(position, 'direction'):
                return False
                
            current_price = data['close'].iloc[-1]
            
            # Exit based on duration
            days_in_trade = (datetime.now().date() - position.entry_time.date()).days
            max_days = self.parameters['max_days_in_trade']
            min_days = self.parameters['min_days_in_trade']
            
            if days_in_trade >= max_days:
                logger.info(f"Exit signal: maximum days in trade reached ({max_days})")
                return True
                
            # Don't exit if we haven't held the minimum duration, unless hitting stop loss
            if days_in_trade < min_days:
                # Check for stop loss hit (always honor stops)
                if position.direction == "long" and current_price <= position.stop_price:
                    logger.info(f"Exit signal: stop loss hit despite minimum hold period")
                    return True
                elif position.direction == "short" and current_price >= position.stop_price:
                    logger.info(f"Exit signal: stop loss hit despite minimum hold period")
                    return True
                    
                # Otherwise, honor minimum hold period
                return False
            
            # Normal exit conditions after minimum hold period
            
            # 1. Profit target reached
            if position.direction == "long" and current_price >= position.target_price:
                logger.info(f"Exit signal: profit target reached")
                return True
            elif position.direction == "short" and current_price <= position.target_price:
                logger.info(f"Exit signal: profit target reached")
                return True
            
            # 2. Stop loss hit
            if position.direction == "long" and current_price <= position.stop_price:
                logger.info(f"Exit signal: stop loss hit")
                return True
            elif position.direction == "short" and current_price >= position.stop_price:
                logger.info(f"Exit signal: stop loss hit")
                return True
            
            # 3. Trend reversal
            if 'trend_direction' in indicators:
                trend = indicators['trend_direction'].iloc[-1]
                
                if position.direction == "long" and trend == -1:
                    logger.info(f"Exit signal: trend reversal from up to down")
                    return True
                elif position.direction == "short" and trend == 1:
                    logger.info(f"Exit signal: trend reversal from down to up")
                    return True
            
            # 4. RSI extreme in opposite direction
            if 'rsi_14' in indicators:
                rsi = indicators['rsi_14'].iloc[-1]
                
                if position.direction == "long" and rsi > self.parameters['rsi_entry_upper']:
                    logger.info(f"Exit signal: RSI overbought at {rsi:.1f}")
                    return True
                elif position.direction == "short" and rsi < self.parameters['rsi_entry_lower']:
                    logger.info(f"Exit signal: RSI oversold at {rsi:.1f}")
                    return True
            
            # 5. Volume spike in opposite direction
            if 'relative_volume' in indicators:
                rel_volume = indicators['relative_volume'].iloc[-1]
                
                if rel_volume > 2.0:
                    # Significant volume spike, check if it's a reversal candle
                    if (position.direction == "long" and 
                        data['close'].iloc[-1] < data['open'].iloc[-1]):  # Bearish candle
                        logger.info(f"Exit signal: high volume reversal candle")
                        return True
                    elif (position.direction == "short" and 
                          data['close'].iloc[-1] > data['open'].iloc[-1]):  # Bullish candle
                        logger.info(f"Exit signal: high volume reversal candle")
                        return True
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {str(e)}")
        
        return False
    
    def _execute_signals(self):
        """
        Execute Swing Trading specific trading signals.
        """
        try:
            if not self.signals:
                return
                
            # Handle entry signals
            if self.signals.get("entry", False):
                direction = self.signals.get("direction")
                if not direction:
                    return
                    
                # Calculate position size based on conviction and risk
                conviction = self.signals.get("conviction", 0.5)
                base_position_size = self.parameters['max_position_size_pct'] / 100
                position_size = base_position_size * conviction
                
                # Get stop and target prices
                stop_price = self.signals.get("stop_price")
                target_price = self.signals.get("target_price")
                pattern = self.signals.get("pattern_detected")
                
                # Create the position
                if direction == "long":
                    self.enter_long_position(
                        size=position_size,
                        stop_price=stop_price,
                        target_price=target_price,
                        metadata={"pattern": pattern}
                    )
                    logger.info(f"Entered long swing trade with size {position_size:.2%}, " +
                               f"stop at {stop_price:.2f}, target at {target_price:.2f}")
                elif direction == "short":
                    self.enter_short_position(
                        size=position_size,
                        stop_price=stop_price,
                        target_price=target_price,
                        metadata={"pattern": pattern}
                    )
                    logger.info(f"Entered short swing trade with size {position_size:.2%}, " +
                               f"stop at {stop_price:.2f}, target at {target_price:.2f}")
            
            # Handle exit signals
            for position_id in self.signals.get("exit_positions", []):
                self.exit_position(position_id)
                
        except Exception as e:
            logger.error(f"Error executing swing trading signals: {str(e)}")
    
    def register_events(self):
        """Register for events relevant to Swing Trading."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add Swing Trading specific event subscriptions
        EventBus.subscribe(EventType.VOLATILITY_SPIKE, self.on_event)
        EventBus.subscribe(EventType.MARKET_REGIME_CHANGE, self.on_event)
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.ECONOMIC_INDICATOR, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Swing Trading strategy.
        
        Args:
            event: The event to process
        """
        try:
            # Let parent class handle common events first
            super().on_event(event)
            
            # Add Swing Trading specific event handling
            if event.type == EventType.VOLATILITY_SPIKE:
                spike_pct = event.data.get('percentage', 0)
                
                if spike_pct > 20:
                    logger.info(f"Volatility spike of {spike_pct}% detected")
                    
                    # Close existing positions if volatility is too high
                    if self.parameters['use_max_drawdown_filter']:
                        for position in self.positions:
                            if position.status == "open":
                                logger.info(f"Closing position due to high volatility")
                                self.exit_position(position.position_id, "high_volatility")
            
            elif event.type == EventType.MARKET_REGIME_CHANGE:
                new_regime = event.data.get('new_regime')
                old_regime = event.data.get('old_regime')
                logger.info(f"Market regime changed from {old_regime} to {new_regime}")
                
                # Update internal market bias
                if new_regime == "bullish" or new_regime == "strongly_bullish":
                    self.current_market_bias = "bullish"
                elif new_regime == "bearish" or new_regime == "strongly_bearish":
                    self.current_market_bias = "bearish"
                else:
                    self.current_market_bias = "neutral"
                
                # Adjust strategy parameters based on new regime
                if new_regime == "strongly_bullish":
                    # Focus on long swings only
                    self.parameters['require_bullish_market'] = True
                elif new_regime == "strongly_bearish":
                    # Focus on short swings only
                    self.parameters['require_bearish_market'] = True
                else:
                    # Allow both long and short swings
                    self.parameters['require_bullish_market'] = False
                    self.parameters['require_bearish_market'] = False
            
            elif event.type == EventType.EARNINGS_ANNOUNCEMENT:
                symbol = event.data.get('symbol')
                days_to_event = event.data.get('days_to_event')
                
                # For swing trades, avoid holding through earnings
                if days_to_event is not None and days_to_event <= 5:
                    for position in self.positions:
                        if position.symbol == symbol and position.status == "open":
                            logger.info(f"Closing position due to upcoming earnings in {days_to_event} days")
                            self.exit_position(position.position_id, "earnings_announcement")
                
        except Exception as e:
            logger.error(f"Error processing event: {str(e)}")
