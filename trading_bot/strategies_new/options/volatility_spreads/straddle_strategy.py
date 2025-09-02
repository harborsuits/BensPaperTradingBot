#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Straddle Strategy

A professional-grade straddle implementation that leverages the modular,
event-driven architecture. This strategy profits from significant price 
movement in either direction.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsSession
from trading_bot.strategies_new.options.base.spread_types import OptionType
from trading_bot.strategies_new.options.base.volatility_spread_engine import VolatilitySpreadEngine, VolatilitySpreadType
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="StraddleStrategy",
    market_type="options",
    description="A strategy that simultaneously buys a call and a put option at the same strike price and expiration, profiting when the underlying makes a large move in either direction",
    timeframes=["1d", "1w"],
    parameters={
        "is_long": {"description": "Whether to use a long straddle (True) or short straddle (False)", "type": "boolean"},
        "target_days_to_expiry": {"description": "Ideal days to expiry for straddle entry", "type": "integer"},
        "min_volatility_score": {"description": "Minimum score for expected volatility (0-100)", "type": "float"},
        "use_earnings_events": {"description": "Whether to target earnings announcements", "type": "boolean"}
    }
)
class StraddleStrategy(VolatilitySpreadEngine, AccountAwareMixin):
    """
    Straddle Strategy
    
    This strategy simultaneously buys a call and a put option at the same strike price 
    and expiration, profiting when the underlying makes a large move in either direction.
    
    Features:
    - Adapts the legacy Straddle implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the VolatilitySpreadEngine for core volatility spread mechanics
    - Implements custom filtering and volatility analysis for optimal entry timing
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Straddle strategy.
        
        Args:
            session: Options trading session
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize base engine
        super().__init__(session, data_pipeline, parameters)
        # Initialize account awareness functionality
        AccountAwareMixin.__init__(self)
        
        # Strategy-specific default parameters
        default_params = {
            # Strategy identification
            'strategy_name': 'Long Straddle',
            'strategy_id': 'long_straddle',
            
            # Straddle specific parameters
            'is_long': True,                # Long straddle by default
            'use_atm_strike': True,         # Use strike closest to current price
            'target_days_to_expiry': 45,    # Ideal DTE for straddle entry
            
            # Market condition preferences
            'min_volatility_score': 75,     # Minimum score for expected volatility (0-100)
            'prefer_low_iv': True,          # Long straddles often perform better with lower IV
            'max_iv_percentile': 40,        # Maximum IV percentile for entry (for long straddles)
            'expect_volatility_expansion': True,  # Expect volatility to increase
            
            # Event-based criteria
            'use_earnings_events': True,    # Target earnings announcements
            'use_economic_events': True,    # Target economic announcements
            'days_before_event': 5,         # Enter this many days before known catalyst events
            
            # Risk parameters
            'max_risk_per_trade_pct': 2.0,  # Max risk as % of account
            'target_profit_pct': 100,       # Target profit as % of initial debit
            'stop_loss_pct': 50,            # Stop loss as % of initial debit
            'max_positions': 2,             # Maximum concurrent positions
            
            # Exit conditions
            'iv_exit_threshold': 25,        # Exit if IV increases by this percentage
            'exit_days_before_expiry': 10,  # Exit this many days before expiry
            'use_delta_neutral_adjustment': False,  # Use delta neutral adjustments
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set state based on long/short straddle
        self.spread_type = VolatilitySpreadType.LONG_STRADDLE if self.parameters['is_long'] else VolatilitySpreadType.SHORT_STRADDLE
        
        # Register for market events
        self.register_events()
        
        # Upcoming events that may cause volatility
        self.upcoming_events = []
        
        logger.info(f"Initialized Straddle Strategy for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Straddle strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        # Get base indicators from parent class
        indicators = super().calculate_indicators(data)
        
        if data.empty or len(data) < 50:
            return indicators
        
        # Calculate indicators specific to volatility assessment
        
        # Historical volatility over different periods
        if 'returns' not in data.columns:
            data['returns'] = data['close'].pct_change()
        
        if len(data) >= 10:
            indicators['volatility_10d'] = data['returns'].rolling(window=10).std() * np.sqrt(252)
        
        if len(data) >= 30:
            indicators['volatility_30d'] = data['returns'].rolling(window=30).std() * np.sqrt(252)
        
        # Volatility of volatility - measure of volatility stability
        if 'volatility_10d' in indicators and len(indicators['volatility_10d']) >= 10:
            vol_of_vol = indicators['volatility_10d'].rolling(window=10).std()
            indicators['vol_of_vol'] = vol_of_vol
        
        # Current volatility relative to recent history
        if 'volatility_30d' in indicators:
            vol_30d = indicators['volatility_30d'].iloc[-1]
            vol_30d_min = indicators['volatility_30d'].rolling(window=20).min().iloc[-1]
            vol_30d_max = indicators['volatility_30d'].rolling(window=20).max().iloc[-1]
            
            # Normalize to 0-1 range
            vol_range = vol_30d_max - vol_30d_min
            if vol_range > 0:
                vol_normalized = (vol_30d - vol_30d_min) / vol_range
                indicators['vol_normalized'] = vol_normalized
        
        # Bollinger Band width as volatility expansion/contraction indicator
        if 'ma_20' not in indicators:
            indicators['ma_20'] = data['close'].rolling(window=20).mean()
        
        std_20 = data['close'].rolling(window=20).std()
        upper_band = indicators['ma_20'] + (std_20 * 2)
        lower_band = indicators['ma_20'] - (std_20 * 2)
        indicators['bb_width'] = (upper_band - lower_band) / indicators['ma_20']
        
        # BB width rate of change - contracting BB width often precedes volatility expansion
        if len(indicators['bb_width']) > 5:
            indicators['bb_width_roc'] = indicators['bb_width'].pct_change(periods=5)
        
        # ATR rate of change - increasing ATR signals increasing volatility
        if 'atr' not in indicators:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            indicators['atr'] = tr.rolling(window=14).mean()
        
        if len(indicators['atr']) > 5:
            indicators['atr_roc'] = indicators['atr'].pct_change(periods=5)
        
        # Calculate a volatility score (0-100)
        volatility_score = 50  # Start neutral
        
        # Check if volatility is historically low
        if 'vol_normalized' in indicators:
            vol_norm = indicators['vol_normalized']
            
            if vol_norm < 0.2:  # Low volatility relative to recent range
                volatility_score += 20
            elif vol_norm > 0.8:  # High volatility relative to recent range
                volatility_score -= 20
        
        # Check for BB width contraction (precedes volatility expansion)
        if 'bb_width_roc' in indicators:
            bb_width_roc = indicators['bb_width_roc'].iloc[-1]
            
            if bb_width_roc < -0.05:  # BB width contracting
                volatility_score += 15
            elif bb_width_roc > 0.10:  # BB width expanding rapidly
                volatility_score -= 10
        
        # Check for ATR changes
        if 'atr_roc' in indicators:
            atr_roc = indicators['atr_roc'].iloc[-1]
            
            if atr_roc > 0.05:  # ATR increasing
                volatility_score += 10
        
        # Check for upcoming events
        if self.upcoming_events:
            closest_event_days = min(event['days_away'] for event in self.upcoming_events)
            if closest_event_days <= self.parameters['days_before_event']:
                volatility_score += 25
                
                # Extra points for earnings (typically high volatility events)
                if any(event['type'] == 'earnings' for event in self.upcoming_events 
                       if event['days_away'] <= self.parameters['days_before_event']):
                    volatility_score += 10
        
        # Ensure score is within 0-100 range
        volatility_score = max(0, min(100, volatility_score))
        indicators['volatility_score'] = volatility_score
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for Straddles based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "exit_positions": [],
            "signal_strength": 0.0,
            "is_long": self.parameters['is_long']
        }
        
        if data.empty or not indicators:
            return signals
        
        # Check if we already have too many positions
        if len(self.positions) >= self.parameters['max_positions']:
            logger.info("Maximum number of positions reached, no new entries.")
            # Still check for exits
        else:
            # Check IV environment
            iv_check_passed = True
            if self.session.current_iv is not None:
                iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
                iv_percentile = iv_metrics.get('iv_percentile', 50)
                
                # For long straddles, we prefer low IV (with expectation of expansion)
                if self.parameters['is_long'] and self.parameters['prefer_low_iv']:
                    max_iv = self.parameters['max_iv_percentile']
                    
                    if iv_percentile > max_iv:
                        iv_check_passed = False
                        logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} above max {max_iv}")
                
                # For short straddles, we prefer high IV (with expectation of contraction)
                elif not self.parameters['is_long']:
                    min_iv = 100 - self.parameters['max_iv_percentile']  # Inverse relationship
                    
                    if iv_percentile < min_iv:
                        iv_check_passed = False
                        logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} below min {min_iv}")
            
            # Check volatility score
            if 'volatility_score' in indicators and iv_check_passed:
                volatility_score = indicators['volatility_score']
                min_score = self.parameters['min_volatility_score']
                
                if volatility_score >= min_score:
                    # Generate entry signal
                    signals["entry"] = True
                    signals["signal_strength"] = volatility_score / 100.0
                    signals["is_long"] = self.parameters['is_long']  # Use configured direction
                    
                    logger.info(f"Straddle entry signal: volatility score {volatility_score}, "
                                f"direction {'long' if signals['is_long'] else 'short'}, "
                                f"strength {signals['signal_strength']:.2f}")
                else:
                    logger.info(f"Volatility score {volatility_score} below threshold {min_score}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.status == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def _check_exit_conditions(self, position, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open straddle position.
        
        Args:
            position: The position to check
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Boolean indicating whether to exit
        """
        # Get current market data
        if data.empty:
            return False
        
        is_long_position = position.spread_type == VolatilitySpreadType.LONG_STRADDLE
        
        # Days to expiration check
        if hasattr(position, 'call_leg') and 'expiration' in position.call_leg:
            expiry_date = position.call_leg.get('expiration')
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            days_to_expiry = (expiry_date - datetime.now().date()).days
            exit_days = self.parameters['exit_days_before_expiry']
            
            if days_to_expiry <= exit_days:
                logger.info(f"Exit signal: approaching expiration ({days_to_expiry} days left)")
                return True
        
        # IV change check
        if self.session.current_iv is not None and hasattr(position, 'entry_iv'):
            entry_iv = position.entry_iv
            current_iv = self.session.current_iv
            
            iv_change_pct = (current_iv / entry_iv - 1) * 100
            iv_threshold = self.parameters['iv_exit_threshold']
            
            # For long straddles, exit if IV expanded significantly (captured the move)
            if is_long_position and iv_change_pct > iv_threshold:
                logger.info(f"Exit signal: IV increased by {iv_change_pct:.2f}% (above threshold {iv_threshold}%)")
                return True
            
            # For short straddles, exit if IV contracted significantly (captured the premium)
            if not is_long_position and iv_change_pct < -iv_threshold:
                logger.info(f"Exit signal: IV decreased by {abs(iv_change_pct):.2f}% (above threshold {iv_threshold}%)")
                return True
        
        # Check for volatility score deterioration for long straddles
        if is_long_position and 'volatility_score' in indicators:
            vol_score = indicators['volatility_score']
            if vol_score < 30:  # Major deterioration in volatility expectations
                logger.info(f"Exit signal: volatility score dropped to {vol_score}")
                return True
        
        # Check if expected catalyst event has passed
        if self.upcoming_events and hasattr(position, 'entry_reason'):
            if 'event' in position.entry_reason:
                event_passed = True
                for event in self.upcoming_events:
                    # If the event is still upcoming and relevant to our position
                    if event['days_away'] <= 0 and event['symbol'] == self.session.symbol:
                        event_passed = False
                        break
                
                if event_passed:
                    logger.info(f"Exit signal: catalyst event has passed")
                    return True
        
        # Profit target and stop loss are handled by the base VolatilitySpreadEngine
        
        return False
    
    def update_events(self, events: List[Dict[str, Any]]):
        """
        Update the list of upcoming events that may cause volatility.
        
        Args:
            events: List of event dictionaries with keys:
                   - symbol: ticker symbol
                   - type: event type (earnings, economic, etc.)
                   - date: event date
                   - days_away: days until event
        """
        self.upcoming_events = events
        
        # Log upcoming events
        if events:
            logger.info(f"Updated upcoming events for {self.session.symbol}: {len(events)} events")
            for event in sorted(events, key=lambda x: x.get('days_away', 0)):
                logger.info(f"  {event.get('type', 'unknown')} in {event.get('days_away')} days")
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Straddle specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Set is_long based on signal
            is_long = self.signals.get("is_long", self.parameters['is_long'])
            
            # Construct and open straddle position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_straddle method from the parent VolatilitySpreadEngine
                straddle_position = self.construct_straddle(
                    self.session.option_chain, 
                    underlying_price,
                    is_long
                )
                
                if straddle_position:
                    # Record entry reason and IV
                    if self.session.current_iv is not None:
                        straddle_position.entry_iv = self.session.current_iv
                    
                    # Add information about entry reason (e.g., upcoming event)
                    entry_reason = "volatility_signal"
                    if self.upcoming_events:
                        closest_event = min(self.upcoming_events, key=lambda x: x.get('days_away', float('inf')))
                        if closest_event.get('days_away', float('inf')) <= self.parameters['days_before_event']:
                            entry_reason = f"event_{closest_event.get('type', 'unknown')}"
                    
                    straddle_position.entry_reason = entry_reason
                    
                    # Add position
                    self.positions.append(straddle_position)
                    logger.info(f"Opened {is_long and 'long' or 'short'} straddle position {straddle_position.position_id}")
                else:
                    logger.warning("Failed to construct valid straddle")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for i, position in enumerate(self.positions):
                if position.position_id == position_id and position.status == "open":
                    # Get current prices for both legs
                    # In a real implementation, would get actual prices from option chain
                    # For now, just set placeholder values
                    call_exit_price = position.call_leg['entry_price'] * 1.1  # Placeholder
                    put_exit_price = position.put_leg['entry_price'] * 1.1    # Placeholder
                    
                    # Close the position
                    position.close_position(call_exit_price, put_exit_price, "signal_generated")
                    logger.info(f"Closed straddle position {position_id} based on signals")
    
    def register_events(self):
        """Register for events relevant to Straddles."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add straddle specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.ECONOMIC_INDICATOR, self.on_event)
        EventBus.subscribe(EventType.VOLATILITY_SPIKE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Straddle strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Straddle specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT:
            symbol = event.data.get('symbol')
            if symbol == self.session.symbol and self.parameters['use_earnings_events']:
                days_to_event = event.data.get('days_to_event', 0)
                
                # Add to upcoming events
                if days_to_event > 0:
                    self.upcoming_events.append({
                        'symbol': symbol,
                        'type': 'earnings',
                        'date': event.data.get('date'),
                        'days_away': days_to_event
                    })
                    
                    logger.info(f"Added earnings event for {symbol} in {days_to_event} days")
                
                # Remove from upcoming events if it's passed
                elif days_to_event <= 0:
                    self.upcoming_events = [e for e in self.upcoming_events 
                                           if not (e['symbol'] == symbol and e['type'] == 'earnings')]
                    
                    logger.info(f"Removed passed earnings event for {symbol}")
        
        elif event.type == EventType.ECONOMIC_INDICATOR:
            if self.parameters['use_economic_events']:
                indicator = event.data.get('indicator')
                impact = event.data.get('impact', 'low')
                days_to_event = event.data.get('days_to_event', 0)
                
                # Only consider high impact events
                if impact.lower() in ['high', 'medium'] and days_to_event > 0:
                    self.upcoming_events.append({
                        'symbol': 'MARKET',  # Affects the whole market
                        'type': f"economic_{indicator}",
                        'date': event.data.get('date'),
                        'days_away': days_to_event,
                        'impact': impact
                    })
                    
                    logger.info(f"Added {impact} impact economic event ({indicator}) in {days_to_event} days")
        
        elif event.type == EventType.VOLATILITY_SPIKE:
            magnitude = event.data.get('magnitude', 0)
            
            if magnitude > 20:  # Significant spike
                logger.info(f"Volatility spike detected: {magnitude}%")
                
                # For long straddles, consider closing to capture profit
                if self.parameters['is_long']:
                    for position in self.positions:
                        if position.status == "open" and position.spread_type == VolatilitySpreadType.LONG_STRADDLE:
                            call_exit_price = position.call_leg['entry_price'] * 1.5  # Estimated price after volatility spike
                            put_exit_price = position.put_leg['entry_price'] * 1.5    # Estimated price after volatility spike
                            position.close_position(call_exit_price, put_exit_price, "volatility_spike")
                            logger.info(f"Closed long straddle position {position.position_id} due to volatility spike")
                
                # For short straddles, consider closing to limit losses
                else:
                    for position in self.positions:
                        if position.status == "open" and position.spread_type == VolatilitySpreadType.SHORT_STRADDLE:
                            call_exit_price = position.call_leg['entry_price'] * 2.0  # Estimated price after volatility spike
                            put_exit_price = position.put_leg['entry_price'] * 2.0    # Estimated price after volatility spike
                            position.close_position(call_exit_price, put_exit_price, "volatility_spike")
                            logger.info(f"Closed short straddle position {position.position_id} due to volatility spike")
