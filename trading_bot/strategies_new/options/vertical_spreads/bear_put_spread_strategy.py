#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bear Put Spread Strategy

A professional-grade bear put spread implementation that leverages the modular,
event-driven architecture. This strategy is designed to profit from moderately
bearish market movements with defined risk and reward.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.strategies_new.options.base.options_base_strategy import OptionsSession
from trading_bot.strategies_new.options.base.spread_types import OptionType, VerticalSpreadType
from trading_bot.strategies_new.options.base.vertical_spread_engine import VerticalSpreadEngine
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="BearPutSpreadStrategy",
    market_type="options",
    description="A strategy that buys a put option at a higher strike price and sells another put at a lower strike, profiting from moderately bearish movement",
    timeframes=["1d", "1w"],
    parameters={
        "delta_lower_long": {"description": "Target delta for long put leg (lower bound of abs value)", "type": "float"},
        "delta_upper_long": {"description": "Target delta for long put leg (upper bound of abs value)", "type": "float"},
        "preferred_width_pct": {"description": "Preferred width between strikes as % of stock price", "type": "float"},
        "min_bearish_score": {"description": "Minimum bearish score to enter (0-100)", "type": "float"}
    }
)
class BearPutSpreadStrategy(VerticalSpreadEngine, AccountAwareMixin):
    """
    Bear Put Spread Strategy
    
    This strategy buys a put option at a higher strike price and sells another 
    put option at a lower strike price with the same expiration. It profits 
    when the underlying security price falls moderately.
    
    Features:
    - Adapts the legacy Bear Put Spread implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the VerticalSpreadEngine for core vertical spread mechanics
    - Implements custom filtering and market condition analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Bear Put Spread strategy.
        
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
            'strategy_name': 'Bear Put Spread',
            'strategy_id': 'bear_put_spread',
            
            # Bear Put specific parameters
            'delta_lower_long': 0.45,  # Target delta for long put leg (absolute value)
            'delta_upper_long': 0.60,  # Target delta for long put leg (absolute value)
            'delta_lower_short': 0.20, # Target delta for short put leg (absolute value)
            'delta_upper_short': 0.35, # Target delta for short put leg (absolute value)
            
            # Market condition filters
            'min_bearish_score': 60,   # Minimum bearish score to enter (0-100)
            'use_macd_filter': True,   # Use MACD for trend confirmation
            'use_rsi_filter': True,    # Use RSI for overbought/oversold filter
            'min_rsi': 30,             # Minimum RSI value
            'max_rsi': 70,             # Maximum RSI value (avoid not oversold)
            
            # Spread construction parameters
            'preferred_width_pct': 5,  # Preferred width between strikes as % of stock price
            'max_spread_cost_pct': 30, # Maximum cost of spread as % of width
            
            # Risk management
            'max_loss_per_trade_pct': 2.0,  # Max loss as % of account
            'target_profit_pct': 50,        # Target profit as % of max profit
            'stop_loss_pct': 75,            # Stop loss as % of max loss
            'max_positions': 3,             # Maximum concurrent positions
        }
        
        # Update with default parameters for Bear Put Spreads 
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set the spread type for this strategy
        self.spread_type = VerticalSpreadType.BEAR_PUT
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Bear Put Spread Strategy for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Bear Put Spread strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        # Get base indicators from parent class
        indicators = super().calculate_indicators(data)
        
        if data.empty or len(data) < 50:
            return indicators
        
        # Calculate MACD if not already present
        if 'macd' not in indicators and self.parameters['use_macd_filter']:
            ema12 = data['close'].ewm(span=12).mean()
            ema26 = data['close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            
            indicators['macd'] = macd_line
            indicators['macd_signal'] = signal_line
            indicators['macd_histogram'] = macd_line - signal_line
        
        # Calculate RSI if not already present
        if 'rsi' not in indicators and self.parameters['use_rsi_filter']:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate bearish score (0-100)
        if len(data) >= 50:
            bearish_score = 50  # Start neutral
            
            # Price relative to moving averages
            if 'ma_20' in indicators and 'ma_50' in indicators:
                ma20 = indicators['ma_20'].iloc[-1]
                ma50 = indicators['ma_50'].iloc[-1]
                current_price = data['close'].iloc[-1]
                
                if current_price < ma20:
                    bearish_score += 10
                if current_price < ma50:
                    bearish_score += 10
                if ma20 < ma50:
                    bearish_score += 10
            
            # MACD signal
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                
                if macd < 0:
                    bearish_score += 5
                if macd < macd_signal:
                    bearish_score += 10
            
            # RSI momentum
            if 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                
                if 40 < rsi < 60:
                    bearish_score += 5  # Neutral, slight bearish bias
                elif 30 <= rsi < 40:
                    bearish_score += 10  # Bearish momentum, not oversold
                elif rsi <= 30:
                    bearish_score -= 10  # Oversold, bullish signal
            
            # Trend strength from ADX
            if 'adx' in indicators:
                adx = indicators['adx'].iloc[-1]
                
                if adx > 20:
                    bearish_score += 5
                if adx > 30:
                    bearish_score += 5
            
            # Volume confirmation
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                
                if current_volume > avg_volume and data['close'].iloc[-1] < data['close'].iloc[-2]:
                    bearish_score += 5
            
            # Volatility assessment for options strategies
            if 'hist_volatility_20d' in indicators:
                vol = indicators['hist_volatility_20d'].iloc[-1]
                
                if 0.15 <= vol <= 0.35:  # Moderate volatility is ideal for spreads
                    bearish_score += 5
            
            # Ensure score is within 0-100 range
            bearish_score = max(0, min(100, bearish_score))
            indicators['bearish_score'] = bearish_score
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for Bear Put Spreads based on market conditions.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            "entry": False,
            "exit_positions": [],
            "signal_strength": 0.0
        }
        
        if data.empty or not indicators:
            return signals
        
        # Check if we already have too many positions
        if len(self.positions) >= self.parameters['max_positions']:
            logger.info("Maximum number of positions reached, no new entries.")
            # Still check for exits
        else:
            # Check bearish score
            if 'bearish_score' in indicators:
                bearish_score = indicators['bearish_score']
                min_score = self.parameters['min_bearish_score']
                
                if bearish_score >= min_score:
                    # Check RSI filter if enabled
                    rsi_check_passed = True
                    if self.parameters['use_rsi_filter'] and 'rsi' in indicators:
                        rsi = indicators['rsi'].iloc[-1]
                        min_rsi = self.parameters['min_rsi']
                        max_rsi = self.parameters['max_rsi']
                        
                        if rsi < min_rsi:  # Too oversold
                            rsi_check_passed = False
                            logger.info(f"RSI filter failed: current RSI {rsi:.2f} below min {min_rsi} (too oversold)")
                    
                    # Check MACD filter if enabled
                    macd_check_passed = True
                    if self.parameters['use_macd_filter'] and 'macd' in indicators and 'macd_signal' in indicators:
                        macd = indicators['macd'].iloc[-1]
                        macd_signal = indicators['macd_signal'].iloc[-1]
                        
                        if macd > 0 or macd > macd_signal:
                            macd_check_passed = False
                            logger.info(f"MACD filter failed: MACD {macd:.4f}, Signal {macd_signal:.4f}")
                    
                    # Generate entry signal if all checks pass
                    if rsi_check_passed and macd_check_passed:
                        signals["entry"] = True
                        signals["signal_strength"] = bearish_score / 100.0
                        logger.info(f"Bear Put Spread entry signal: score {bearish_score}, strength {signals['signal_strength']:.2f}")
        
        # Check for exits on existing positions
        for position in self.positions:
            if position.status == "open":
                exit_signal = self._check_exit_conditions(position, data, indicators)
                if exit_signal:
                    signals["exit_positions"].append(position.position_id)
        
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
        # Get current market data
        if data.empty:
            return False
        
        current_price = data['close'].iloc[-1]
        
        # If bearish score drops significantly, consider exit
        if 'bearish_score' in indicators:
            bearish_score = indicators['bearish_score']
            if bearish_score < 40:  # Much less bearish than entry threshold
                logger.info(f"Exit signal: bearish score dropped to {bearish_score}")
                return True
        
        # Check trend change
        if 'ma_20' in indicators and 'ma_50' in indicators:
            ma20 = indicators['ma_20'].iloc[-1]
            ma50 = indicators['ma_50'].iloc[-1]
            
            if ma20 > ma50 and current_price > ma20:
                logger.info("Exit signal: bullish trend developing")
                return True
        
        # Check if MACD turned bullish
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            
            if macd > 0 and macd > macd_signal:
                logger.info("Exit signal: MACD turned bullish")
                return True
        
        # Check if RSI is oversold (market might reverse)
        if 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            if rsi <= 30:
                logger.info(f"Exit signal: RSI indicates oversold ({rsi:.2f})")
                return True
        
        # Profit target and stop loss are handled by the base VerticalSpreadEngine
        
        return False
    
    def filter_option_chains(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom filters for Bear Put Spread option selection.
        
        Args:
            option_chain: Option chain data
            
        Returns:
            Filtered option chain
        """
        # Apply base filters first
        filtered_chain = super().filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            return filtered_chain
        
        # Additional filters specific to Bear Put Spreads
        
        # Delta filter for selecting appropriate strikes (note: put deltas are negative)
        if 'delta' in filtered_chain.columns:
            # Only include options that meet our delta criteria
            # For puts, we need to use absolute value since delta is negative
            filtered_chain = filtered_chain[
                ((filtered_chain['option_type'] == 'put') & 
                 ((filtered_chain['delta'].abs() >= self.parameters['delta_lower_long']) & 
                  (filtered_chain['delta'].abs() <= self.parameters['delta_upper_long']))) |
                ((filtered_chain['option_type'] == 'put') & 
                 ((filtered_chain['delta'].abs() >= self.parameters['delta_lower_short']) & 
                  (filtered_chain['delta'].abs() <= self.parameters['delta_upper_short'])))
            ]
        
        # Open interest and volume filter for liquidity
        if 'open_interest' in filtered_chain.columns and 'volume' in filtered_chain.columns:
            min_oi = 100
            min_volume = 10
            filtered_chain = filtered_chain[
                (filtered_chain['open_interest'] >= min_oi) |
                (filtered_chain['volume'] >= min_volume)
            ]
        
        return filtered_chain
    
    def _execute_signals(self):

        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Trade execution aborted: Insufficient buying power")
            return
        """
        Execute Bear Put Spread specific trading signals.
        """
        # Use base class implementation for standard vertical spread execution
        super()._execute_signals()
    
    def register_events(self):
        """Register for events relevant to Bear Put Spreads."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add any Bear Put Spread specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.ECONOMIC_INDICATOR, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Bear Put Spread strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Bear Put Spread specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT:
            symbol = event.data.get('symbol')
            if symbol == self.session.symbol:
                logger.info(f"Earnings announcement for {symbol}, adjusting strategy...")
                # Implement strategy-specific adjustments for earnings
        
        elif event.type == EventType.ECONOMIC_INDICATOR:
            indicator = event.data.get('indicator')
            value = event.data.get('value')
            impact = event.data.get('impact', 'neutral')
            
            logger.info(f"Economic indicator: {indicator}, value: {value}, impact: {impact}")
            # Implement strategy-specific adjustments for economic indicators
