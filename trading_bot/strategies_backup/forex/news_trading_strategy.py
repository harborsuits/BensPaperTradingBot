#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex News Trading Strategy

This module implements a news trading strategy for forex markets,
using economic calendar data to identify and trade potential market
movements around high-impact economic events.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)

class ForexNewsTrading(ForexBaseStrategy):
    """News trading strategy for forex markets.
    
    This strategy trades around high-impact economic events by:
    1. Identifying upcoming high-impact news events
    2. Determining potential market reaction based on forecast vs previous
    3. Analyzing pre-news price action for positioning
    4. Executing trades based on market reaction to actual news release
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # News event parameters
        'news_impact_threshold': 'high',  # 'low', 'medium', 'high'
        'pre_news_window_minutes': 120,   # Minutes before news to start monitoring
        'post_news_window_minutes': 60,   # Minutes after news to continue monitoring
        'avoid_trading_window_minutes': 5, # Minutes before news to avoid entering new positions
        
        # Position sizing and risk
        'news_risk_factor': 0.75,         # Scale normal position size by this factor for news trades
        'max_events_per_day': 3,          # Maximum number of news events to trade per day
        
        # Entry parameters
        'breakout_atr_multiple': 0.5,     # ATR multiple for breakout confirmation
        'min_volume_spike_factor': 2.0,   # Minimum volume increase to confirm breakout
        'min_pre_news_consolidation': 30, # Minimum minutes of pre-news consolidation 
        
        # Exit parameters
        'take_profit_pips': 40,           # Fixed take profit in pips
        'stop_loss_pips': 25,             # Fixed stop loss in pips
        'trailing_activation_pips': 20,   # Pips of profit before activating trailing stop
        'use_volatility_based_exits': True, # Use ATR for dynamic exits instead of fixed pips
        
        # Filter parameters
        'min_adr_pips': 70,               # Minimum Average Daily Range in pips
        'required_liquidity': 'high',     # 'low', 'medium', 'high' - liquidity requirement

        # Session preferences (inherited from ForexBaseStrategy)
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
    }
    
    # Event importance mapping
    EVENT_IMPORTANCE = {
        'low': 1,
        'medium': 2,
        'high': 3
    }
    
    def __init__(self, name: str = "Forex News Trading", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex news trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMETERS)
            metadata: Strategy metadata
        """
        # Merge default parameters with ForexBaseStrategy defaults
        forex_params = self.DEFAULT_FOREX_PARAMS.copy()
        forex_params.update(self.DEFAULT_PARAMETERS)
        
        # Override with user-provided parameters if any
        if parameters:
            forex_params.update(parameters)
        
        # Initialize the base strategy
        super().__init__(name=name, parameters=forex_params, metadata=metadata)
        
        # Register with the event system
        self.event_bus = EventBus()
        
        # Strategy state
        self.current_signals = {}          # Current trading signals
        self.active_news_events = {}       # Active news events being monitored
        self.recent_news_trades = {}       # Recent trades made for news events
        self.news_event_calendar = []      # Upcoming economic events
        self.last_calendar_update = None   # Timestamp of last calendar update
        
        # Set economic calendar flag
        self.parameters['use_economic_calendar'] = True
        
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals based on economic news events.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # First, ensure we have up-to-date economic calendar data
        self._update_economic_calendar(current_time)
        
        # Identify relevant news events for the current time window
        relevant_events = self._get_relevant_news_events(current_time)
        
        if not relevant_events:
            logger.debug(f"No relevant news events found for {current_time}")
            return signals
        
        # Process each symbol
        for symbol, ohlcv in data.items():
            # Skip if we don't have enough data
            if len(ohlcv) < 30:  # Need at least 30 bars for proper analysis
                continue
                
            # Calculate indicators for this symbol
            indicators = self._calculate_news_indicators(ohlcv)
            
            # Find news events relevant to this currency pair
            pair_events = self._filter_events_for_pair(symbol, relevant_events)
            
            if not pair_events:
                continue
                
            # Evaluate potential setups based on upcoming or recent news
            signal = self._evaluate_news_setup(symbol, ohlcv, indicators, pair_events, current_time)
            
            if signal:
                signals[symbol] = signal
                # Store in current signals
                self.current_signals[symbol] = signal
        
        # Publish event with active news events
        if self.active_news_events:
            event_data = {
                'strategy_name': self.name,
                'active_news_events': self.active_news_events,
                'events_count': len(self.active_news_events),
                'timestamp': current_time
            }
            
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'news_trading'}
            )
            self.event_bus.publish(event)
        
        return signals
    
    def _update_economic_calendar(self, current_time: datetime) -> None:
        """
        Update the economic calendar with upcoming events.
        
        Args:
            current_time: Current timestamp
        """
        # Check if we need to update the calendar
        should_update = (
            self.last_calendar_update is None or 
            (current_time - self.last_calendar_update).total_seconds() > 3600  # Update hourly
        )
        
        if not should_update:
            return
            
        logger.info("Updating economic calendar")
        
        # In a real implementation, this would call an external API or service
        # For now, we'll create a placeholder with some sample data
        
        # Placeholder for economic calendar data - would be replaced with real API call
        self.news_event_calendar = [
            {
                'datetime': current_time + timedelta(minutes=30),
                'currency': 'USD',
                'event': 'Non-Farm Payrolls',
                'impact': 'high',
                'forecast': '200K',
                'previous': '180K'
            },
            {
                'datetime': current_time + timedelta(hours=2),
                'currency': 'EUR',
                'event': 'ECB Interest Rate Decision',
                'impact': 'high',
                'forecast': '3.75%',
                'previous': '3.75%'
            },
            {
                'datetime': current_time + timedelta(hours=4),
                'currency': 'GBP',
                'event': 'GDP',
                'impact': 'high',
                'forecast': '0.2%',
                'previous': '0.1%'
            }
        ]
        
        self.last_calendar_update = current_time
        logger.debug(f"Updated economic calendar with {len(self.news_event_calendar)} events")
    
    def _get_relevant_news_events(self, current_time: datetime) -> List[Dict[str, Any]]:
        """
        Get news events relevant to the current time window.
        
        Args:
            current_time: Current timestamp
        
        Returns:
            List of relevant news events
        """
        # Calculate time window boundaries
        pre_window_mins = self.parameters['pre_news_window_minutes']
        post_window_mins = self.parameters['post_news_window_minutes']
        
        window_start = current_time - timedelta(minutes=post_window_mins)
        window_end = current_time + timedelta(minutes=pre_window_mins)
        
        # Filter events within the time window and with sufficient impact
        min_impact = self.parameters['news_impact_threshold']
        min_impact_level = self.EVENT_IMPORTANCE.get(min_impact, 1)
        
        relevant_events = []
        
        for event in self.news_event_calendar:
            event_time = event['datetime']
            event_impact = event['impact']
            impact_level = self.EVENT_IMPORTANCE.get(event_impact, 0)
            
            if (window_start <= event_time <= window_end and 
                impact_level >= min_impact_level):
                
                # Calculate time relation to current time
                minutes_to_event = (event_time - current_time).total_seconds() / 60
                event_info = event.copy()
                event_info['minutes_to_event'] = minutes_to_event
                
                relevant_events.append(event_info)
        
        return relevant_events
    
    def _filter_events_for_pair(self, symbol: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter news events relevant to a specific currency pair.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EUR/USD')
            events: List of news events
            
        Returns:
            List of events relevant to the pair
        """
        if '/' not in symbol:
            return []
            
        # Extract base and quote currencies
        base, quote = symbol.split('/')
        
        # Filter events for either currency in the pair
        relevant_events = [event for event in events 
                          if event['currency'] == base or event['currency'] == quote]
        
        return relevant_events
    
    def _calculate_news_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for news trading.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate ATR for volatility measurement
        atr_period = 14
        high_low = ohlcv['high'] - ohlcv['low']
        high_close = np.abs(ohlcv['high'] - ohlcv['close'].shift())
        low_close = np.abs(ohlcv['low'] - ohlcv['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        indicators['atr'] = true_range.rolling(atr_period).mean()
        
        # Calculate recent volume statistics
        indicators['volume_sma'] = ohlcv['volume'].rolling(20).mean()
        indicators['volume_ratio'] = ohlcv['volume'] / indicators['volume_sma']
        
        # Calculate volatility zones
        indicators['high_band'] = ohlcv['high'].rolling(20).max()
        indicators['low_band'] = ohlcv['low'].rolling(20).min()
        indicators['midpoint'] = (indicators['high_band'] + indicators['low_band']) / 2
        
        # Calculate momentum indicators
        indicators['rsi'] = self._calculate_rsi(ohlcv['close'], 14)
        
        # Calculate price change velocity
        indicators['price_velocity'] = ohlcv['close'].pct_change(5) * 100
        
        return indicators
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).fillna(0)
        loss = -delta.where(delta < 0, 0).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _evaluate_news_setup(self, 
                           symbol: str, 
                           ohlcv: pd.DataFrame, 
                           indicators: Dict[str, Any],
                           events: List[Dict[str, Any]], 
                           current_time: datetime) -> Optional[Signal]:
        """
        Evaluate potential trade setups based on news events.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            events: List of relevant news events
            current_time: Current timestamp
            
        Returns:
            Signal object if a trade setup is found, None otherwise
        """
        # Sort events by time (closest first)
        events.sort(key=lambda x: abs(x['minutes_to_event']))
        closest_event = events[0]
        
        minutes_to_event = closest_event['minutes_to_event']
        event_name = closest_event['event']
        event_currency = closest_event['currency']
        event_impact = closest_event['impact']
        
        logger.debug(f"Evaluating {symbol} for {event_name} ({event_currency}) in {minutes_to_event:.1f} minutes")
        
        # Different setups based on timing relative to the event
        if minutes_to_event > 0:
            # Pre-news setup
            return self._evaluate_pre_news_setup(symbol, ohlcv, indicators, closest_event, current_time)
        else:
            # Post-news setup
            return self._evaluate_post_news_setup(symbol, ohlcv, indicators, closest_event, current_time)
    
    def _evaluate_pre_news_setup(self, 
                               symbol: str, 
                               ohlcv: pd.DataFrame, 
                               indicators: Dict[str, Any],
                               event: Dict[str, Any], 
                               current_time: datetime) -> Optional[Signal]:
        """
        Evaluate pre-news trade setups.
        
        This method looks for consolidation patterns before high-impact news.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            event: News event information
            current_time: Current timestamp
            
        Returns:
            Signal object if a valid pre-news setup is found
        """
        minutes_to_event = event['minutes_to_event']
        avoid_window = self.parameters['avoid_trading_window_minutes']
        
        # Don't enter new positions too close to the news release
        if 0 < minutes_to_event < avoid_window:
            logger.debug(f"Too close to {event['event']} - avoiding new entries for {symbol}")
            return None
        
        # Check if we're in a pre-news consolidation
        atr = indicators['atr'].iloc[-1]
        recent_range_pips = (ohlcv['high'].iloc[-10:].max() - ohlcv['low'].iloc[-10:].min()) / self.parameters['pip_value']
        
        # Define consolidation as narrow recent range relative to ATR
        is_consolidating = recent_range_pips < (atr / self.parameters['pip_value']) * 3
        
        if not is_consolidating:
            return None
            
        # Calculate support and resistance levels
        support = ohlcv['low'].iloc[-20:].min()
        resistance = ohlcv['high'].iloc[-20:].max()
        current_price = ohlcv['close'].iloc[-1]
        
        # Determine straddle entry levels
        buy_entry = resistance + (atr * 0.5)
        sell_entry = support - (atr * 0.5)
        
        # Calculate risk parameters
        stop_pips = self.parameters['stop_loss_pips']
        target_pips = self.parameters['take_profit_pips']
        
        # Create signal for pending entries
        confidence = min(0.7, 0.4 + (self.EVENT_IMPORTANCE.get(event['impact'], 1) / 10))
        
        # Adjust confidence based on forecast vs previous
        if 'forecast' in event and 'previous' in event:
            try:
                forecast = float(event['forecast'].replace('%', '').replace('K', '000').replace('M', '000000'))
                previous = float(event['previous'].replace('%', '').replace('K', '000').replace('M', '000000'))
                
                # Higher confidence if forecast significantly different from previous
                if abs(forecast - previous) / max(0.01, abs(previous)) > 0.1:  # >10% change
                    confidence += 0.1
            except (ValueError, TypeError):
                pass  # Not all forecasts can be parsed as numbers
        
        # Create signal with dual pending orders (straddle)
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.PENDING,  # Pending order
            direction=0,  # Neutral - we'll place orders in both directions
            confidence=confidence,
            entry_price=current_price,  # Current price as reference
            stop_loss=0,  # Will be set when orders are triggered
            take_profit=0,  # Will be set when orders are triggered
            metadata={
                'strategy': self.name,
                'setup_type': 'pre_news_straddle',
                'event': event['event'],
                'event_time': event['datetime'].isoformat(),
                'event_impact': event['impact'],
                'minutes_to_event': minutes_to_event,
                'buy_entry': buy_entry,
                'sell_entry': sell_entry,
                'stop_pips': stop_pips,
                'target_pips': target_pips,
                'atr': atr
            }
        )
        
        # Register this as an active news event
        self.active_news_events[symbol] = {
            'event': event,
            'setup_type': 'pre_news_straddle',
            'timestamp': current_time.isoformat()
        }
        
        return signal
    
    def _evaluate_post_news_setup(self, 
                                symbol: str, 
                                ohlcv: pd.DataFrame, 
                                indicators: Dict[str, Any],
                                event: Dict[str, Any], 
                                current_time: datetime) -> Optional[Signal]:
        """
        Evaluate post-news trade setups.
        
        This method looks for breakouts after news releases.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            event: News event information
            current_time: Current timestamp
            
        Returns:
            Signal object if a valid post-news setup is found
        """
        minutes_since_event = -event['minutes_to_event']  # Convert to positive
        post_window = self.parameters['post_news_window_minutes']
        
        # Only consider recent news
        if minutes_since_event > post_window:
            return None
            
        # Check for volume spike
        volume_ratio = indicators['volume_ratio'].iloc[-1]
        min_volume_spike = self.parameters['min_volume_spike_factor']
        
        if volume_ratio < min_volume_spike:
            return None
            
        # Check for price breakout
        atr = indicators['atr'].iloc[-1]
        pre_news_high = ohlcv['high'].iloc[-int(minutes_since_event+10):-int(minutes_since_event)].max()
        pre_news_low = ohlcv['low'].iloc[-int(minutes_since_event+10):-int(minutes_since_event)].min()
        current_price = ohlcv['close'].iloc[-1]
        
        breakout_threshold = atr * self.parameters['breakout_atr_multiple']
        
        # Detect direction of breakout
        is_upside_breakout = current_price > (pre_news_high + breakout_threshold)
        is_downside_breakout = current_price < (pre_news_low - breakout_threshold)
        
        if not (is_upside_breakout or is_downside_breakout):
            return None
            
        # Calculate entry, stop, and target
        direction = 1 if is_upside_breakout else -1
        entry_price = current_price
        
        # Calculate stop and target in pips
        stop_pips = self.parameters['stop_loss_pips']
        target_pips = self.parameters['take_profit_pips']
        
        pip_value = self.parameters['pip_value']
        stop_loss = entry_price - (direction * stop_pips * pip_value)
        take_profit = entry_price + (direction * target_pips * pip_value)
        
        # Calculate confidence based on various factors
        base_confidence = 0.5
        
        # Higher confidence for high impact events
        impact_boost = (self.EVENT_IMPORTANCE.get(event['impact'], 1) - 1) * 0.1
        
        # Higher confidence for stronger volume spike
        volume_boost = min(0.2, (volume_ratio / 10))
        
        # Higher confidence for stronger breakout
        breakout_strength = abs(current_price - (pre_news_high if is_upside_breakout else pre_news_low)) / atr
        breakout_boost = min(0.2, breakout_strength / 10)
        
        # Combine confidence factors
        confidence = min(0.95, base_confidence + impact_boost + volume_boost + breakout_boost)
        
        # Create the signal
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.MARKET_OPEN,  # Immediate entry
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': self.name,
                'setup_type': 'post_news_breakout',
                'event': event['event'],
                'event_time': event['datetime'].isoformat(),
                'event_impact': event['impact'],
                'minutes_since_event': minutes_since_event,
                'volume_ratio': volume_ratio,
                'breakout_strength': breakout_strength,
                'atr': atr
            }
        )
        
        # Register this as a recent news trade
        self.recent_news_trades[symbol] = {
            'event': event,
            'setup_type': 'post_news_breakout',
            'direction': direction,
            'entry_price': entry_price,
            'timestamp': current_time.isoformat()
        }
        
        return signal
    
    def get_compatibility_score(self, market_regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # News trading can work in various market regimes, but excels in certain conditions
        compatibility_map = {
            # News trading thrives on volatility - best regimes
            MarketRegime.VOLATILE_BREAKOUT: 0.95,  # Excellent for news breakouts
            MarketRegime.VOLATILE_REVERSAL: 0.85,  # Good for news reactions
            
            # Can work in trending markets if news aligns with trend
            MarketRegime.TRENDING_UP: 0.70,
            MarketRegime.TRENDING_DOWN: 0.70,
            
            # Less ideal but still viable
            MarketRegime.RANGING: 0.60,  # News can break ranges
            MarketRegime.CHOPPY: 0.50,   # News can create clear direction
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.60   # Generally viable
        }
        
        # Return the compatibility score or default to 0.6 if regime unknown
        return compatibility_map.get(market_regime, 0.6)
    
    def optimize_for_regime(self, market_regime: MarketRegime) -> Dict[str, Any]:
        """
        Optimize strategy parameters for the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Dictionary of optimized parameters
        """
        # Start with current parameters
        optimized_params = self.parameters.copy()
        
        # Adjust parameters based on regime
        if market_regime in [MarketRegime.VOLATILE_BREAKOUT, MarketRegime.VOLATILE_REVERSAL]:
            # For volatile regimes, optimize for breakouts
            optimized_params['breakout_atr_multiple'] = 0.3  # Lower threshold to catch breakouts
            optimized_params['take_profit_pips'] = 60       # Larger targets
            optimized_params['stop_loss_pips'] = 30         # Wider stops
            optimized_params['min_volume_spike_factor'] = 1.5  # Less strict volume requirement
            
        elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # For trending regimes, optimize for trend continuation
            optimized_params['breakout_atr_multiple'] = 0.5  # Standard threshold
            optimized_params['take_profit_pips'] = 50        # Good targets
            optimized_params['stop_loss_pips'] = 25          # Standard stops
            optimized_params['min_volume_spike_factor'] = 2.0  # Standard volume requirement
            
        elif market_regime == MarketRegime.RANGING:
            # For ranging regimes, be more selective
            optimized_params['breakout_atr_multiple'] = 0.7  # Higher threshold
            optimized_params['take_profit_pips'] = 35        # Smaller targets
            optimized_params['stop_loss_pips'] = 20          # Tighter stops
            optimized_params['min_volume_spike_factor'] = 2.5  # Stricter volume requirement
            
        elif market_regime == MarketRegime.CHOPPY:
            # For choppy markets, be very selective
            optimized_params['breakout_atr_multiple'] = 0.8  # High threshold
            optimized_params['take_profit_pips'] = 30        # Small targets
            optimized_params['stop_loss_pips'] = 15          # Tight stops
            optimized_params['min_volume_spike_factor'] = 3.0  # Very strict volume requirement
            optimized_params['news_impact_threshold'] = 'high'  # Only highest impact news
            
        # Log the optimization
        logger.info(f"Optimized {self.name} for {market_regime} regime")
        
        return optimized_params
