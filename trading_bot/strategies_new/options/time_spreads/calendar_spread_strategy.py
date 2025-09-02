#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calendar Spread Strategy

A professional-grade calendar spread implementation that leverages the modular,
event-driven architecture. This strategy profits from time decay differences 
between option expiration cycles.
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
from trading_bot.strategies_new.options.base.time_spread_engine import TimeSpreadEngine, TimeSpreadType
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="CalendarSpreadStrategy",
    market_type="options",
    description="A strategy that sells a near-term option and buys a longer-term option at the same strike price, profiting from time decay differences between expiration cycles",
    timeframes=["1d", "1w"],
    parameters={
        "option_type": {"description": "Option type (call or put) for calendar spread", "type": "string"},
        "front_month_min_dte": {"description": "Minimum days to expiration for front month", "type": "integer"},
        "back_month_min_dte": {"description": "Minimum days to expiration for back month", "type": "integer"},
        "min_days_between": {"description": "Minimum days between front and back month expirations", "type": "integer"}
    }
)
class CalendarSpreadStrategy(TimeSpreadEngine, AccountAwareMixin):
    """
    Calendar Spread Strategy
    
    This strategy involves selling a near-term option and buying a longer-term
    option at the same strike price. It profits from time decay differences
    between the two expiration cycles.
    
    Features:
    - Adapts the legacy Calendar Spread implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the TimeSpreadEngine for time-based spread mechanics
    - Implements custom filtering and time-decay analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Calendar Spread strategy.
        
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
            'strategy_name': 'Calendar Spread',
            'strategy_id': 'calendar_spread',
            
            # Calendar specific parameters
            'option_type': 'call',         # 'call' or 'put' calendar
            'use_atm_strike': True,        # Use strike closest to current price
            'use_iv_skew': True,           # Consider IV skew between months
            'front_month_min_dte': 20,     # Minimum DTE for front month
            'front_month_max_dte': 35,     # Maximum DTE for front month
            'back_month_min_dte': 40,      # Minimum DTE for back month
            'back_month_max_dte': 90,      # Maximum DTE for back month
            'min_days_between': 20,        # Minimum days between expirations
            'max_days_between': 60,        # Maximum days between expirations
            'min_iv_skew': 0.02,           # Minimum IV difference between months
            
            # Market condition preferences
            'min_stability_score': 65,     # Minimum score for price stability (0-100)
            'prefer_high_iv': True,        # Calendars benefit from high IV environments
            'min_iv_percentile': 40,       # Minimum IV percentile for entry
            'max_iv_percentile': 90,       # Maximum IV percentile for entry
            
            # Risk parameters
            'max_risk_per_trade_pct': 1.5, # Max risk as % of account
            'target_profit_pct': 50,       # Target profit as % of max loss
            'stop_loss_pct': 35,           # Stop loss as % of max loss
            'max_positions': 3,            # Maximum concurrent positions
            
            # Exit parameters
            'days_before_front_expiry': 7, # Exit this many days before front expiry
            'manage_front_expiry': True,   # Actively manage positions near front expiry
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set state for calendar spread
        self.spread_type = TimeSpreadType.CALENDAR
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Calendar Spread Strategy for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Calendar Spread strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        # Get base indicators from parent class
        indicators = super().calculate_indicators(data)
        
        if data.empty or len(data) < 50:
            return indicators
        
        # Calculate indicators specific to price stability assessment
        
        # Recent price volatility (standard deviation of close prices)
        if len(data) >= 20:
            price_std = data['close'].rolling(window=20).std()
            price_mean = data['close'].rolling(window=20).mean()
            indicators['price_volatility'] = price_std / price_mean * 100  # As percentage
        
        # Price channeling - identify range-bound markets
        if len(data) >= 50:
            # Upper and lower channels using recent highs and lows
            upper_channel = data['high'].rolling(window=20).max()
            lower_channel = data['low'].rolling(window=20).min()
            mid_channel = (upper_channel + lower_channel) / 2
            
            # Channel width as percentage
            channel_width = (upper_channel - lower_channel) / mid_channel * 100
            indicators['channel_width_pct'] = channel_width
            
            # Where is price in the channel (0 = bottom, 1 = top)
            current_price = data['close'].iloc[-1]
            channel_position = (current_price - lower_channel.iloc[-1]) / (upper_channel.iloc[-1] - lower_channel.iloc[-1])
            indicators['channel_position'] = channel_position
        
        # Rate of change measures
        if len(data) >= 20:
            indicators['roc_5d'] = data['close'].pct_change(periods=5) * 100
            indicators['roc_20d'] = data['close'].pct_change(periods=20) * 100
            
            # Average rate of change over last 5 days
            indicators['avg_daily_roc'] = data['close'].pct_change().rolling(window=5).mean() * 100
        
        # ADX to measure trend strength
        if 'adx' not in indicators:
            # Calculate DI+, DI-, and ADX
            high = data['high']
            low = data['low']
            close = data['close']
            
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr14 = tr.rolling(window=14).mean()
            
            # Plus Directional Movement
            plus_dm = high.diff()
            plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -low.diff()), 0)
            
            # Minus Directional Movement
            minus_dm = low.shift(1) - low
            minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > high - high.shift(1)), 0)
            
            # Directional Indicators
            plus_di14 = 100 * (plus_dm.rolling(window=14).sum() / atr14.rolling(window=14).sum())
            minus_di14 = 100 * (minus_dm.rolling(window=14).sum() / atr14.rolling(window=14).sum())
            
            # Directional Index
            dx = 100 * abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
            
            # Average Directional Index
            indicators['adx'] = dx.rolling(window=14).mean()
        
        # Calculate stability score (0-100)
        stability_score = 50  # Start neutral
        
        # Lower ADX is better for calendar spreads (less trending)
        if 'adx' in indicators:
            adx = indicators['adx'].iloc[-1]
            
            if adx < 20:  # Weak trend - good for range-bound strategies
                stability_score += 20
            elif adx < 30:
                stability_score += 10
            elif adx > 40:  # Strong trend - not ideal for calendar spreads
                stability_score -= 20
        
        # Lower price volatility is better for calendar spreads
        if 'price_volatility' in indicators:
            price_vol = indicators['price_volatility'].iloc[-1]
            
            if price_vol < 2:  # Very stable price
                stability_score += 20
            elif price_vol < 4:
                stability_score += 10
            elif price_vol > 8:  # Highly volatile price
                stability_score -= 20
        
        # Recent price change should be moderate
        if 'roc_5d' in indicators:
            roc_5d = indicators['roc_5d'].iloc[-1]
            
            if abs(roc_5d) < 2:  # Very little price change
                stability_score += 15
            elif abs(roc_5d) > 7:  # Large price change
                stability_score -= 15
        
        # Price in middle of channel is ideal
        if 'channel_position' in indicators:
            channel_pos = indicators['channel_position']
            
            if 0.4 <= channel_pos <= 0.6:  # Near middle
                stability_score += 10
            elif channel_pos < 0.2 or channel_pos > 0.8:  # Near edges
                stability_score -= 10
        
        # Ensure score is within 0-100 range
        stability_score = max(0, min(100, stability_score))
        indicators['stability_score'] = stability_score
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for Calendar Spreads based on market conditions.
        
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
            "option_type": self.parameters['option_type']
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
                
                min_iv = self.parameters['min_iv_percentile']
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile < min_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} below min {min_iv}")
                elif iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} above max {max_iv}")
            
            # Check stability score
            if 'stability_score' in indicators and iv_check_passed:
                stability_score = indicators['stability_score']
                min_score = self.parameters['min_stability_score']
                
                if stability_score >= min_score:
                    # Generate entry signal
                    signals["entry"] = True
                    signals["signal_strength"] = stability_score / 100.0
                    
                    # Determine if we should use call or put calendar
                    if 'channel_position' in indicators:
                        channel_pos = indicators['channel_position']
                        
                        if channel_pos < 0.4:
                            # Price in lower part of channel, prefer put calendar
                            signals["option_type"] = "put"
                        elif channel_pos > 0.6:
                            # Price in upper part of channel, prefer call calendar
                            signals["option_type"] = "call"
                        else:
                            # Price in middle, use default option type
                            signals["option_type"] = self.parameters['option_type']
                    
                    logger.info(f"Calendar spread entry signal: stability score {stability_score}, "
                                f"type {signals['option_type']}, strength {signals['signal_strength']:.2f}")
                else:
                    logger.info(f"Stability score {stability_score} below threshold {min_score}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.status == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position.position_id)
        
        return signals
    
    def _check_exit_conditions(self, position, data: pd.DataFrame, indicators: Dict[str, Any]) -> bool:
        """
        Check exit conditions for an open calendar spread position.
        
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
        
        current_price = data['close'].iloc[-1]
        
        # Days to front expiration check
        if hasattr(position, 'front_leg') and 'expiration' in position.front_leg:
            front_exp = position.front_leg.get('expiration')
            if isinstance(front_exp, str):
                front_exp = datetime.strptime(front_exp, '%Y-%m-%d').date()
            
            days_to_expiry = (front_exp - datetime.now().date()).days
            exit_days = self.parameters['days_before_front_expiry']
            
            if days_to_expiry <= exit_days:
                logger.info(f"Exit signal: approaching front expiration ({days_to_expiry} days left)")
                return True
        
        # Check strike proximity for maximum time decay benefit
        if hasattr(position, 'front_leg') and 'strike' in position.front_leg:
            strike = position.front_leg['strike']
            
            # Calculate distance from strike as percentage
            distance_pct = abs(current_price - strike) / strike * 100
            
            # If price moves far from strike with limited time left
            if hasattr(position, 'front_leg') and 'dte' in position.front_leg:
                dte = position.front_leg['dte']
                
                if distance_pct > 8 and dte < 14:
                    logger.info(f"Exit signal: price moved {distance_pct:.2f}% from strike with only {dte} days left")
                    return True
        
        # Check stability score deterioration
        if 'stability_score' in indicators:
            stability = indicators['stability_score']
            if stability < 30:  # Major deterioration in stability
                logger.info(f"Exit signal: stability score dropped to {stability}")
                return True
        
        # Check for trend development (bad for calendar spreads)
        if 'adx' in indicators:
            adx = indicators['adx'].iloc[-1]
            adx_prev = indicators['adx'].iloc[-2] if len(indicators['adx']) > 1 else 0
            
            if adx > 30 and adx > adx_prev * 1.2:  # ADX increasing by more than 20%
                logger.info(f"Exit signal: ADX increasing ({adx_prev:.1f} to {adx:.1f}), trend developing")
                return True
        
        # Check for volatility expansion
        if 'hist_volatility_20d' in indicators and hasattr(position, 'entry_time'):
            days_in_trade = (datetime.now() - position.entry_time).days
            
            if days_in_trade > 0:
                entry_vol = self.get_volatility_at_time(position.entry_time)
                current_vol = indicators['hist_volatility_20d'].iloc[-1]
                
                if entry_vol and current_vol and (current_vol / entry_vol > 1.5):
                    logger.info(f"Exit signal: volatility expanded by more than 50%")
                    return True
        
        # Profit target and stop loss are handled by the base TimeSpreadEngine
        
        return False
    
    def get_volatility_at_time(self, timestamp: datetime) -> Optional[float]:
        """
        Get the historical volatility at a specific time.
        
        Args:
            timestamp: The time to look up
            
        Returns:
            Historical volatility or None if not available
        """
        # This is a simplified implementation
        # In a real system, would query historical volatility data
        return 0.20  # Placeholder
    
    def filter_option_chains(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom filters for Calendar Spread option selection.
        
        Args:
            option_chain: Option chain data
            
        Returns:
            Filtered option chain
        """
        # Apply base filters first
        filtered_chain = super().filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            return filtered_chain
        
        # Additional filters specific to Calendar Spreads
        
        # We need good liquidity for calendar spreads
        if 'open_interest' in filtered_chain.columns and 'volume' in filtered_chain.columns:
            min_oi = 100
            min_volume = 10
            filtered_chain = filtered_chain[
                (filtered_chain['open_interest'] >= min_oi) |
                (filtered_chain['volume'] >= min_volume)
            ]
        
        # Filter by option type
        option_type = self.parameters['option_type']
        filtered_chain = filtered_chain[filtered_chain['option_type'] == option_type]
        
        # Filter by time to expiration
        if 'expiration_date' in filtered_chain.columns:
            today = datetime.now().date()
            front_min_dte = self.parameters['front_month_min_dte']
            front_max_dte = self.parameters['front_month_max_dte']
            back_min_dte = self.parameters['back_month_min_dte']
            back_max_dte = self.parameters['back_month_max_dte']
            
            # Only keep expirations within our target ranges
            valid_expirations = []
            
            for exp in filtered_chain['expiration_date'].unique():
                if isinstance(exp, str):
                    exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                else:
                    exp_date = exp
                
                dte = (exp_date - today).days
                
                if (front_min_dte <= dte <= front_max_dte) or (back_min_dte <= dte <= back_max_dte):
                    valid_expirations.append(exp)
            
            if valid_expirations:
                filtered_chain = filtered_chain[filtered_chain['expiration_date'].isin(valid_expirations)]
        
        return filtered_chain
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Calendar Spread specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Set option type based on signal
            option_type = self.signals.get("option_type", self.parameters['option_type'])
            if option_type == "call":
                option_type_enum = OptionType.CALL
            else:
                option_type_enum = OptionType.PUT
            
            # Construct and open calendar spread position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_calendar_spread method from the parent TimeSpreadEngine
                calendar_position = self.construct_calendar_spread(
                    self.session.option_chain, 
                    underlying_price,
                    option_type_enum
                )
                
                if calendar_position:
                    # Add position
                    self.positions.append(calendar_position)
                    logger.info(f"Opened {option_type} calendar spread position {calendar_position.position_id}")
                else:
                    logger.warning("Failed to construct valid calendar spread")
        
        # Handle exit signals
        for position_id in self.signals.get("exit_positions", []):
            for position in self.positions:
                if position.position_id == position_id and position.status == "open":
                    # Get current prices for both legs
                    # In a real implementation, would get actual prices from option chain
                    # For now, just set placeholder values
                    front_exit_price = position.front_leg['entry_price'] * 0.5  # Placeholder (decreased value for short)
                    back_exit_price = position.back_leg['entry_price'] * 1.1    # Placeholder (increased value for long)
                    
                    # Close the position
                    position.close_position(front_exit_price, back_exit_price, "signal_generated")
                    logger.info(f"Closed calendar spread position {position_id} based on signals")
    
    def register_events(self):
        """Register for events relevant to Calendar Spreads."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add calendar spread specific event subscriptions
        EventBus.subscribe(EventType.VOLATILITY_CHANGE, self.on_event)
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Calendar Spread strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Calendar Spread specific event handling
        if event.type == EventType.VOLATILITY_CHANGE:
            symbol = event.data.get('symbol')
            if symbol == self.session.symbol:
                change_pct = event.data.get('change_pct', 0)
                
                if change_pct < -30:  # Major IV contraction
                    logger.info(f"IV contracted by {abs(change_pct):.2f}%, adjusting calendar parameters")
                    
                    # Consider rolling positions in IV collapse
                    if self.parameters['manage_front_expiry']:
                        for position in self.positions:
                            if position.status == "open":
                                # Check for significant IV skew change
                                if hasattr(position, 'iv_skew') and position.iv_skew:
                                    current_iv_skew = self.get_current_iv_skew(position)
                                    if current_iv_skew and current_iv_skew < position.iv_skew * 0.5:
                                        logger.info(f"IV skew collapsed, closing position {position.position_id}")
                                        self.spread_manager.close_position(position.position_id, "iv_skew_collapse")
        
        elif event.type == EventType.EARNINGS_ANNOUNCEMENT:
            symbol = event.data.get('symbol')
            if symbol == self.session.symbol:
                days_to_earnings = event.data.get('days_to_event', 0)
                
                # Be cautious with calendar spreads through earnings
                if 0 < days_to_earnings <= 5:
                    logger.info(f"Earnings approaching in {days_to_earnings} days, adjusting calendar risk")
                    
                    # Reduce risk or close positions based on proximity to earnings
                    for position in self.positions:
                        if position.status == "open":
                            # Only close if front month expires after earnings
                            if hasattr(position, 'front_leg') and 'expiration' in position.front_leg:
                                front_exp = position.front_leg.get('expiration')
                                if isinstance(front_exp, str):
                                    front_exp = datetime.strptime(front_exp, '%Y-%m-%d').date()
                                
                                earnings_date = datetime.now().date() + timedelta(days=days_to_earnings)
                                
                                if front_exp > earnings_date:
                                    logger.info(f"Closing calendar position {position.position_id} to avoid earnings risk")
                                    self.spread_manager.close_position(position.position_id, "earnings_risk")
    
    def get_current_iv_skew(self, position) -> Optional[float]:
        """
        Get the current IV skew between months for a position.
        
        Args:
            position: The position to check
            
        Returns:
            IV skew or None if not available
        """
        # This is a simplified implementation
        # In a real system, would query current IV data for both expirations
        return 0.01  # Placeholder
