#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strangle Strategy

A professional-grade strangle implementation that leverages the modular,
event-driven architecture. This strategy profits from significant price 
movement in either direction using OTM options.
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
    name="StrangleStrategy",
    market_type="options",
    description="A strategy that simultaneously buys a put and a call option at different out-of-the-money strikes, profiting when the underlying makes a large move in either direction",
    timeframes=["1d", "1w"],
    parameters={
        "is_long": {"description": "Whether this is a long or short strangle strategy", "type": "boolean"},
        "call_delta_target": {"description": "Target delta for OTM call", "type": "float"},
        "put_delta_target": {"description": "Target delta for OTM put (negative)", "type": "float"},
        "target_days_to_expiry": {"description": "Ideal DTE for strangle entry", "type": "integer"}
    }
)
class StrangleStrategy(VolatilitySpreadEngine, AccountAwareMixin):
    """
    Strangle Strategy
    
    This strategy simultaneously buys a put and a call option at different strikes
    (both out of the money), profiting when the underlying makes a large move in 
    either direction.
    
    Features:
    - Adapts the legacy Strangle implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the VolatilitySpreadEngine for core volatility spread mechanics
    - Implements custom filtering and volatility analysis for optimal entry timing
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Strangle strategy.
        
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
            'strategy_name': 'Long Strangle',
            'strategy_id': 'long_strangle',
            
            # Strangle specific parameters
            'is_long': True,                # Long strangle by default
            'call_delta_target': 0.30,      # Target delta for OTM call
            'put_delta_target': -0.30,      # Target delta for OTM put (negative)
            'strike_width_pct': 5.0,        # Min width between strikes as % of underlying
            'target_days_to_expiry': 45,    # Ideal DTE for strangle entry
            
            # Market condition preferences
            'min_volatility_score': 75,     # Minimum score for expected volatility (0-100)
            'prefer_low_iv': True,          # Long strangles often perform better with lower IV
            'max_iv_percentile': 40,        # Maximum IV percentile for entry (for long strangles)
            
            # Risk parameters
            'max_risk_per_trade_pct': 2.0,  # Max risk as % of account
            'target_profit_pct': 100,       # Target profit as % of initial debit
            'stop_loss_pct': 50,            # Stop loss as % of initial debit
            'max_positions': 2,             # Maximum concurrent positions
        }
        
        # Update with default parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set state based on long/short strangle
        self.spread_type = VolatilitySpreadType.LONG_STRANGLE if self.parameters['is_long'] else VolatilitySpreadType.SHORT_STRANGLE
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Strangle Strategy for {session.symbol}")
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for Strangles based on market conditions.
        
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
        else:
            # Check IV environment for long strangles (prefer low IV with expectation of expansion)
            iv_check_passed = True
            if self.session.current_iv is not None and self.parameters['is_long'] and self.parameters['prefer_low_iv']:
                iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
                iv_percentile = iv_metrics.get('iv_percentile', 50)
                
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f} above max {max_iv}")
            
            # Check volatility score
            if 'volatility_score' in indicators and iv_check_passed:
                volatility_score = indicators['volatility_score']
                min_score = self.parameters['min_volatility_score']
                
                if volatility_score >= min_score:
                    # Generate entry signal
                    signals["entry"] = True
                    signals["signal_strength"] = volatility_score / 100.0
                    signals["is_long"] = self.parameters['is_long']
                    
                    logger.info(f"Strangle entry signal: volatility score {volatility_score}, "
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
        Check exit conditions for an open strangle position.
        
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
        
        is_long_position = position.spread_type == VolatilitySpreadType.LONG_STRANGLE
        
        # Days to expiration check
        if hasattr(position, 'call_leg') and 'expiration' in position.call_leg:
            expiry_date = position.call_leg.get('expiration')
            if isinstance(expiry_date, str):
                expiry_date = datetime.strptime(expiry_date, '%Y-%m-%d').date()
            
            days_to_expiry = (expiry_date - datetime.now().date()).days
            
            # Exit long strangles before expiration to avoid theta decay
            if is_long_position and days_to_expiry <= 10:
                logger.info(f"Exit signal: approaching expiration ({days_to_expiry} days left)")
                return True
        
        # Check for volatility score deterioration for long strangles
        if is_long_position and 'volatility_score' in indicators:
            vol_score = indicators['volatility_score']
            if vol_score < 30:  # Major deterioration in volatility expectations
                logger.info(f"Exit signal: volatility score dropped to {vol_score}")
                return True
        
        # Profit target and stop loss are handled by the base VolatilitySpreadEngine
        
        return False
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Strangle specific trading signals.
        """
        if not self.signals:
            return
        
        # Handle entry signals
        if self.signals.get("entry", False):
            # Set is_long based on signal
            is_long = self.signals.get("is_long", self.parameters['is_long'])
            
            # Construct and open strangle position
            underlying_price = self.session.current_price
            
            if underlying_price and self.session.option_chain is not None:
                # Use the construct_strangle method from the parent VolatilitySpreadEngine
                strangle_position = self.construct_strangle(
                    self.session.option_chain, 
                    underlying_price,
                    is_long
                )
                
                if strangle_position:
                    # Add position
                    self.positions.append(strangle_position)
                    logger.info(f"Opened {is_long and 'long' or 'short'} strangle position {strangle_position.position_id}")
                else:
                    logger.warning("Failed to construct valid strangle")
        
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
                    logger.info(f"Closed strangle position {position_id} based on signals")
    
    def register_events(self):
        """Register for events relevant to Strangles."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add strangle specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.VOLATILITY_SPIKE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Strangle strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Strangle specific event handling
        if event.type == EventType.VOLATILITY_SPIKE:
            magnitude = event.data.get('magnitude', 0)
            
            if magnitude > 20:  # Significant spike
                logger.info(f"Volatility spike detected: {magnitude}%")
                
                # For long strangles, consider closing to capture profit
                if self.parameters['is_long']:
                    for position in self.positions:
                        if position.status == "open" and position.spread_type == VolatilitySpreadType.LONG_STRANGLE:
                            logger.info(f"Closing long strangle position {position.position_id} due to volatility spike")
                            # Close position logic would go here
