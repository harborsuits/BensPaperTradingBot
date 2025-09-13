#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bull Put Spread Strategy

A professional-grade bull put spread implementation that leverages the modular,
event-driven architecture. This strategy is designed to collect premium in 
moderately bullish markets with defined risk and reward.
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
    name="BullPutSpreadStrategy",
    market_type="options",
    description="A strategy that sells a put option at a higher strike price and buys another put option at a lower strike price with the same expiration, profiting from premium collection in moderately bullish markets",
    timeframes=["1d", "1w"],
    parameters={
        "delta_lower_short": {"description": "Target delta for short put leg (lower bound)", "type": "float"},
        "delta_upper_short": {"description": "Target delta for short put leg (upper bound)", "type": "float"},
        "delta_lower_long": {"description": "Target delta for long put leg (lower bound)", "type": "float"},
        "delta_upper_long": {"description": "Target delta for long put leg (upper bound)", "type": "float"}
    }
)
class BullPutSpreadStrategy(VerticalSpreadEngine, AccountAwareMixin):
    """
    Bull Put Spread Strategy
    
    This strategy sells a put option at a higher strike price and buys another 
    put option at a lower strike price with the same expiration. It profits 
    from premium collection when the underlying security stays above the short strike.
    
    Features:
    - Adapts the legacy Bull Put Spread implementation to the new architecture
    - Uses event-driven signal generation and position management
    - Leverages the VerticalSpreadEngine for core vertical spread mechanics
    - Implements custom filtering and market condition analysis
    """
    
    def __init__(self, session: OptionsSession, data_pipeline: DataPipeline, 
                 parameters: Dict[str, Any] = None):
        """
        Initialize the Bull Put Spread strategy.
        
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
            'strategy_name': 'Bull Put Spread',
            'strategy_id': 'bull_put_spread',
            
            # Bull Put specific parameters
            'delta_lower_short': 0.25,  # Target delta for short put leg (absolute value)
            'delta_upper_short': 0.35,  # Target delta for short put leg (absolute value)
            'delta_lower_long': 0.10,   # Target delta for long put leg (absolute value)
            'delta_upper_long': 0.20,   # Target delta for long put leg (absolute value)
            
            # Market condition filters
            'min_bullish_score': 55,    # Minimum bullish score to enter (0-100)
            'use_macd_filter': True,    # Use MACD for trend confirmation
            'use_rsi_filter': True,     # Use RSI for overbought/oversold filter
            'min_rsi': 40,              # Minimum RSI value (avoid oversold)
            'max_rsi': 70,              # Maximum RSI value (avoid overbought)
            
            # IV preferences (bull put spreads often perform better in higher IV)
            'min_iv_percentile': 40,    # Minimum IV percentile to enter
            'max_iv_percentile': 85,    # Maximum IV percentile to enter
            
            # Spread construction parameters
            'preferred_width_pct': 5,   # Preferred width between strikes as % of stock price
            'min_credit_per_width': 0.15, # Minimum credit as % of width between strikes
            
            # Risk management
            'max_risk_per_trade_pct': 2.0,  # Max risk as % of account
            'profit_target_pct': 50,        # Target profit as % of max credit
            'stop_loss_pct': 150,           # Stop loss as % of credit (e.g., 150% = 1.5x initial credit)
            'max_positions': 3,             # Maximum concurrent positions
            'probability_otm_threshold': 70, # Minimum probability of OTM for short strike
        }
        
        # Update with default parameters for Bull Put Spreads 
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Set the spread type for this strategy
        self.spread_type = VerticalSpreadType.BULL_PUT
        
        # Register for market events
        self.register_events()
        
        logger.info(f"Initialized Bull Put Spread Strategy for {session.symbol}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Bull Put Spread strategy.
        
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
        
        # Calculate support and resistance levels for put strike placement
        if len(data) >= 50:
            # Simple support using recent lows
            recent_lows = data['low'].rolling(window=20).min()
            recent_closes = data['close'].rolling(window=5).mean()
            
            # Find potential support levels below current price
            current_price = data['close'].iloc[-1]
            support_level = recent_lows.iloc[-1]
            
            # Distance of support from current price as percentage
            support_distance = (current_price - support_level) / current_price * 100
            indicators['support_distance_pct'] = support_distance
            
            # Track distance from main moving averages
            if 'ma_50' in indicators:
                ma50 = indicators['ma_50'].iloc[-1]
                ma50_distance = (current_price - ma50) / current_price * 100
                indicators['ma50_distance_pct'] = ma50_distance
        
        # Calculate bullish score (0-100)
        if len(data) >= 50:
            bullish_score = 50  # Start neutral
            
            # Price relative to moving averages
            if 'ma_20' in indicators and 'ma_50' in indicators:
                ma20 = indicators['ma_20'].iloc[-1]
                ma50 = indicators['ma_50'].iloc[-1]
                current_price = data['close'].iloc[-1]
                
                if current_price > ma20:
                    bullish_score += 10
                if current_price > ma50:
                    bullish_score += 10
                if ma20 > ma50:
                    bullish_score += 10
            
            # MACD signal
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                
                if macd > 0:
                    bullish_score += 5
                if macd > macd_signal:
                    bullish_score += 10
            
            # RSI momentum
            if 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                
                if 40 < rsi < 60:
                    bullish_score += 5  # Neutral, slight bullish bias
                elif 60 <= rsi < 70:
                    bullish_score += 10  # Bullish momentum, not overbought
                elif rsi >= 70:
                    bullish_score -= 10  # Overbought, bearish signal
            
            # Support distance
            if 'support_distance_pct' in indicators:
                support_dist = indicators['support_distance_pct']
                
                if 5 <= support_dist <= 15:  # Good distance for put selling
                    bullish_score += 5
            
            # Trend strength
            if 'adx' in indicators:
                adx = indicators['adx'].iloc[-1]
                
                if adx > 20:
                    bullish_score += 5
                if adx > 30:
                    bullish_score += 5
            
            # Volume confirmation
            if 'volume' in data.columns:
                avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                
                if current_volume > avg_volume and data['close'].iloc[-1] > data['close'].iloc[-2]:
                    bullish_score += 5
            
            # Ensure score is within 0-100 range
            bullish_score = max(0, min(100, bullish_score))
            indicators['bullish_score'] = bullish_score
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for Bull Put Spreads based on market conditions.
        
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
            # Check IV environment
            iv_check_passed = True
            if self.session.current_iv is not None and self.session.symbol in self.iv_history:
                iv_metrics = self.calculate_implied_volatility_metrics(self.iv_history[self.session.symbol])
                iv_percentile = iv_metrics.get('iv_percentile', 50)
                
                min_iv = self.parameters['min_iv_percentile']
                max_iv = self.parameters['max_iv_percentile']
                
                if iv_percentile < min_iv or iv_percentile > max_iv:
                    iv_check_passed = False
                    logger.info(f"IV filter failed: current IV percentile {iv_percentile:.2f}, range: {min_iv}-{max_iv}")
            
            # Check bullish score
            if 'bullish_score' in indicators and iv_check_passed:
                bullish_score = indicators['bullish_score']
                min_score = self.parameters['min_bullish_score']
                
                if bullish_score >= min_score:
                    # Check RSI filter if enabled
                    rsi_check_passed = True
                    if self.parameters['use_rsi_filter'] and 'rsi' in indicators:
                        rsi = indicators['rsi'].iloc[-1]
                        min_rsi = self.parameters['min_rsi']
                        max_rsi = self.parameters['max_rsi']
                        
                        if rsi < min_rsi or rsi > max_rsi:
                            rsi_check_passed = False
                            logger.info(f"RSI filter failed: current RSI {rsi:.2f}, range: {min_rsi}-{max_rsi}")
                    
                    # Check MACD filter if enabled
                    macd_check_passed = True
                    if self.parameters['use_macd_filter'] and 'macd' in indicators and 'macd_signal' in indicators:
                        macd = indicators['macd'].iloc[-1]
                        macd_signal = indicators['macd_signal'].iloc[-1]
                        
                        if macd < 0 or macd < macd_signal:
                            macd_check_passed = False
                            logger.info(f"MACD filter failed: MACD {macd:.4f}, Signal {macd_signal:.4f}")
                    
                    # Generate entry signal if all checks pass
                    if rsi_check_passed and macd_check_passed:
                        signals["entry"] = True
                        signals["signal_strength"] = bullish_score / 100.0
                        logger.info(f"Bull Put Spread entry signal: score {bullish_score}, strength {signals['signal_strength']:.2f}")
        
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
        
        # For credit spreads like bull puts, check price relative to short strike
        if hasattr(position, 'short_leg') and 'strike' in position.short_leg:
            short_strike = position.short_leg['strike']
            
            # Price is moving dangerously close to short strike
            cushion_pct = 0.02  # 2% cushion
            danger_level = short_strike * (1 + cushion_pct)
            
            if current_price <= danger_level:
                logger.info(f"Exit signal: price {current_price:.2f} near or below short strike {short_strike:.2f}")
                return True
        
        # If bullish score drops significantly, consider exit
        if 'bullish_score' in indicators:
            bullish_score = indicators['bullish_score']
            if bullish_score < 40:  # Much less bullish than entry threshold
                logger.info(f"Exit signal: bullish score dropped to {bullish_score}")
                return True
        
        # Check trend change
        if 'ma_20' in indicators and 'ma_50' in indicators:
            ma20 = indicators['ma_20'].iloc[-1]
            ma50 = indicators['ma_50'].iloc[-1]
            
            if ma20 < ma50 and current_price < ma20:
                logger.info("Exit signal: bearish trend developing")
                return True
        
        # Check if MACD turned bearish
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            
            if macd < 0 and macd < macd_signal:
                logger.info("Exit signal: MACD turned bearish")
                return True
        
        # Profit target and stop loss are handled by the base VerticalSpreadEngine
        
        return False
    
    def filter_option_chains(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Apply custom filters for Bull Put Spread option selection.
        
        Args:
            option_chain: Option chain data
            
        Returns:
            Filtered option chain
        """
        # Apply base filters first
        filtered_chain = super().filter_option_chains(option_chain)
        
        if filtered_chain is None or filtered_chain.empty:
            return filtered_chain
        
        # Additional filters specific to Bull Put Spreads
        
        # Delta filter for selecting appropriate strikes (note: put deltas are negative)
        if 'delta' in filtered_chain.columns:
            # Only include put options that meet our delta criteria
            # For puts, we need to use absolute value since delta is negative
            filtered_chain = filtered_chain[
                ((filtered_chain['option_type'] == 'put') & 
                 ((filtered_chain['delta'].abs() >= self.parameters['delta_lower_short']) & 
                  (filtered_chain['delta'].abs() <= self.parameters['delta_upper_short']))) |
                ((filtered_chain['option_type'] == 'put') & 
                 ((filtered_chain['delta'].abs() >= self.parameters['delta_lower_long']) & 
                  (filtered_chain['delta'].abs() <= self.parameters['delta_upper_long'])))
            ]
        
        # Filter by probability OTM if available (for selling puts, we want high probability OTM)
        if 'prob_otm' in filtered_chain.columns:
            otm_threshold = self.parameters['probability_otm_threshold']
            
            # Keep options that meet our probability threshold
            short_leg_filter = (
                (filtered_chain['option_type'] == 'put') & 
                (filtered_chain['delta'].abs() >= self.parameters['delta_lower_short']) & 
                (filtered_chain['delta'].abs() <= self.parameters['delta_upper_short']) &
                (filtered_chain['prob_otm'] >= otm_threshold)
            )
            
            # For long leg, we don't have strict probability requirements
            long_leg_filter = (
                (filtered_chain['option_type'] == 'put') & 
                (filtered_chain['delta'].abs() >= self.parameters['delta_lower_long']) & 
                (filtered_chain['delta'].abs() <= self.parameters['delta_upper_long'])
            )
            
            filtered_chain = filtered_chain[short_leg_filter | long_leg_filter]
        
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
        Execute Bull Put Spread specific trading signals.
        """
        # Use base class implementation for standard vertical spread execution
        super()._execute_signals()
    
    def register_events(self):
        """Register for events relevant to Bull Put Spreads."""
        # Register for common event types from the parent class
        super().register_events()
        
        # Add any Bull Put Spread specific event subscriptions
        EventBus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self.on_event)
        EventBus.subscribe(EventType.ECONOMIC_INDICATOR, self.on_event)
        EventBus.subscribe(EventType.VOLATILITY_SPIKE, self.on_event)
    
    def on_event(self, event: Event):
        """
        Process incoming events for the Bull Put Spread strategy.
        
        Args:
            event: The event to process
        """
        # Let parent class handle common events first
        super().on_event(event)
        
        # Add additional Bull Put Spread specific event handling
        if event.type == EventType.EARNINGS_ANNOUNCEMENT:
            symbol = event.data.get('symbol')
            if symbol == self.session.symbol:
                logger.info(f"Earnings announcement for {symbol}, adjusting strategy...")
                # Consider closing positions before earnings to avoid gap risk
                for position in self.positions:
                    if position.status == "open":
                        # Close positions with upcoming earnings
                        days_to_earnings = event.data.get('days_to_event', 0)
                        if days_to_earnings <= 5:
                            self.spread_manager.close_position(position.position_id, "earnings_risk")
        
        elif event.type == EventType.VOLATILITY_SPIKE:
            # Bull Put Spreads can be attractive during volatility spikes due to elevated premiums
            magnitude = event.data.get('magnitude', 0)
            if magnitude > 20:  # Significant spike
                logger.info(f"Volatility spike detected: {magnitude}%, adjusting parameters...")
                # Temporarily adjust our delta targets to be more conservative
                self.parameters['delta_lower_short'] = 0.20
                self.parameters['delta_upper_short'] = 0.30
