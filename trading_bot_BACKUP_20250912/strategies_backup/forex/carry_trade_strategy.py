#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Carry Trade Strategy

This module implements a carry trade strategy for forex markets, 
focusing on exploiting interest rate differentials between currencies
while managing directional and volatility risks.
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

class ForexCarryTradeStrategy(ForexBaseStrategy):
    """Carry trade strategy for forex markets.
    
    This strategy seeks to profit from interest rate differentials between currencies by:
    1. Buying currencies with higher interest rates and selling those with lower rates
    2. Only entering trades when market conditions favor the higher-rate currency's direction
    3. Focusing on stable pairs with consistent interest rate spreads
    4. Using technical filters to avoid adverse price movements against the carry
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Carry trade parameters
        'min_interest_rate_differential': 1.5,  # Minimum rate differential in percentage points
        'prefer_stable_differentials': True,    # Prefer pairs with stable spread history
        'interest_rate_source': 'central_bank', # Source for interest rate data
        
        # Trend alignment parameters
        'trend_alignment_required': True,       # Require trend in the same direction as carry
        'ma_period': 50,                        # MA period for trend confirmation
        'adx_period': 14,                       # ADX period for trend strength
        'min_adx_value': 20,                    # Minimum ADX value for trend confirmation
        
        # Volatility parameters
        'max_volatility_percentile': 70,        # Maximum allowed volatility (ATR percentile)
        'atr_period': 14,                       # ATR period for volatility measurement
        'volatility_lookback': 90,              # Days to look back for volatility context
        
        # Risk management parameters
        'max_carry_exposure': 0.4,              # Maximum portfolio exposure to carry trades
        'stop_loss_atr_multiple': 3.0,          # Wider stop loss for carry trades
        'trailing_stop_activation': 1.0,        # ATR multiple before activating trailing stop
        'take_partial_profits': True,           # Take partial profits along the way
        
        # Advanced parameters
        'use_swap_optimization': True,          # Optimize for swap points, not just rates
        'roll_positions_automatically': True,   # Auto-roll positions to capture daily swaps
        'hedge_with_options': False,            # Use options to hedge tail risk (advanced)
        
        # Session preferences (inherited from ForexBaseStrategy)
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
        'roll_over_sessions': [ForexSession.SYDNEY], # Best session for position rollovers
    }
    
    # Interest rates for major currencies (placeholder - would be updated from external source)
    # Actual implementation would fetch from central bank APIs or economic data providers
    INTEREST_RATES = {
        'USD': 5.25,  # Federal Reserve rate
        'EUR': 3.75,  # ECB rate
        'GBP': 5.00,  # Bank of England rate
        'JPY': 0.10,  # Bank of Japan rate
        'AUD': 4.35,  # RBA rate
        'NZD': 5.50,  # RBNZ rate
        'CAD': 5.00,  # Bank of Canada rate
        'CHF': 1.75,  # Swiss National Bank rate
    }
    
    # Carry trade favorites (pairs historically good for carry)
    CARRY_FAVORITES = [
        'AUD/JPY',
        'NZD/JPY',
        'CAD/JPY',
        'GBP/JPY',
        'AUD/CHF',
        'NZD/CHF'
    ]
    
    def __init__(self, name: str = "Forex Carry Trade", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex carry trade strategy.
        
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
        
        # Strategy specific attributes
        self.event_bus = EventBus()
        self.interest_rates = self.INTEREST_RATES.copy()
        self.interest_rate_update_time = None
        self.current_signals = {}
        self.active_carry_trades = {}
        self.last_rollover_check = None
        
        # Interest rate differential tracking
        self.rate_differentials = {}
        
        logger.info(f"Initialized {name} strategy")
    
    def update_interest_rates(self) -> None:
        """
        Update interest rates from external source.
        
        In a production environment, this would fetch the latest
        interest rates from a data provider or central bank APIs.
        """
        # This is a placeholder - in a real implementation, fetch from an API
        # For example:
        # api_client = EconomicDataProvider()
        # self.interest_rates = api_client.get_current_interest_rates()
        
        # For now, just use our static rates
        self.interest_rate_update_time = datetime.now()
        logger.info("Updated interest rate data")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals based on carry opportunities.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Update interest rates if needed (once per day)
        if (self.interest_rate_update_time is None or 
            (current_time - self.interest_rate_update_time).total_seconds() > 86400):
            self.update_interest_rates()
        
        signals = {}
        
        # Calculate interest rate differentials for all pairs
        self.rate_differentials = self.analyze_interest_rate_differentials(list(data.keys()))
        
        # Sort pairs by rate differential (highest first)
        sorted_pairs = sorted(
            self.rate_differentials.items(),
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Consider rollovers for existing positions
        self._check_position_rollovers(current_time)
        
        # Process each pair in order of differential
        for symbol, differential in sorted_pairs:
            # Skip if differential is too small
            min_differential = self.parameters['min_interest_rate_differential']
            if abs(differential) < min_differential:
                continue
            
            # Get OHLCV data for this pair
            if symbol not in data:
                continue
                
            ohlcv = data[symbol]
            
            # Skip if we don't have enough data
            if len(ohlcv) < max(self.parameters['ma_period'], self.parameters['adx_period']) + 10:
                continue
            
            # Calculate indicators
            indicators = self._calculate_carry_trade_indicators(ohlcv, differential)
            
            # Evaluate carry trade opportunity
            signal = self._evaluate_carry_opportunity(symbol, ohlcv, indicators, differential, current_time)
            
            if signal:
                signals[symbol] = signal
                # Store in current signals
                self.current_signals[symbol] = signal
        
        # Publish event with active carry trades
        if self.active_carry_trades:
            event_data = {
                'strategy_name': self.name,
                'active_carry_trades': self.active_carry_trades,
                'trades_count': len(self.active_carry_trades),
                'timestamp': current_time.isoformat()
            }
            
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'carry_trade'}
            )
            self.event_bus.publish(event)
        
        return signals
    
    def _calculate_carry_trade_indicators(self, 
                                       ohlcv: pd.DataFrame, 
                                       differential: float) -> Dict[str, Any]:
        """
        Calculate indicators for carry trade evaluation.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            differential: Interest rate differential
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Direction of the carry trade (buy currency with higher rate)
        indicators['carry_direction'] = 1 if differential > 0 else -1
        
        # Calculate trend indicators
        ma_period = self.parameters['ma_period']
        indicators['ma'] = ohlcv['close'].rolling(window=ma_period).mean()
        
        # Determine trend direction
        indicators['above_ma'] = ohlcv['close'] > indicators['ma']
        indicators['ma_slope'] = indicators['ma'].diff(5) / indicators['ma'].shift(5) * 100
        
        # Calculate ADX for trend strength
        adx_period = self.parameters['adx_period']
        tr1 = abs(ohlcv['high'] - ohlcv['low'])
        tr2 = abs(ohlcv['high'] - ohlcv['close'].shift())
        tr3 = abs(ohlcv['low'] - ohlcv['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional movement
        up_move = ohlcv['high'] - ohlcv['high'].shift()
        down_move = ohlcv['low'].shift() - ohlcv['low']
        
        pos_dm = up_move.copy()
        pos_dm[up_move <= down_move] = 0
        pos_dm[up_move <= 0] = 0
        
        neg_dm = down_move.copy()
        neg_dm[down_move <= up_move] = 0
        neg_dm[down_move <= 0] = 0
        
        # Smooth the indicators
        tr_smooth = tr.rolling(window=adx_period).mean()
        pos_di = 100 * (pos_dm.rolling(window=adx_period).mean() / tr_smooth)
        neg_di = 100 * (neg_dm.rolling(window=adx_period).mean() / tr_smooth)
        
        dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
        indicators['adx'] = dx.rolling(window=adx_period).mean()
        indicators['pos_di'] = pos_di
        indicators['neg_di'] = neg_di
        
        # Calculate ATR for volatility
        atr_period = self.parameters['atr_period']
        high_low = ohlcv['high'] - ohlcv['low']
        high_close = np.abs(ohlcv['high'] - ohlcv['close'].shift())
        low_close = np.abs(ohlcv['low'] - ohlcv['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        indicators['atr'] = true_range.rolling(atr_period).mean()
        
        # Calculate volatility percentile
        lookback = min(self.parameters['volatility_lookback'], len(ohlcv))
        indicators['atr_percentile'] = indicators['atr'].rolling(lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=True
        )
        
        # Daily swap points estimation (placeholder - would come from broker in real implementation)
        # For now, approximate it based on interest rate differential
        points_per_10_percent = 10  # Rough approximation
        indicators['daily_swap_points'] = differential * points_per_10_percent
        
        return indicators
    
    def _evaluate_carry_opportunity(self, 
                                  symbol: str, 
                                  ohlcv: pd.DataFrame, 
                                  indicators: Dict[str, Any],
                                  differential: float,
                                  current_time: datetime) -> Optional[Signal]:
        """
        Evaluate if a currency pair offers a good carry trade opportunity.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            differential: Interest rate differential
            current_time: Current timestamp
            
        Returns:
            Signal object if a valid carry opportunity is found
        """
        # Get latest values
        current_price = ohlcv['close'].iloc[-1]
        atr = indicators['atr'].iloc[-1]
        adx = indicators['adx'].iloc[-1]
        carry_direction = indicators['carry_direction']
        
        # Check if volatility is acceptable
        max_vol_percentile = self.parameters['max_volatility_percentile']
        if indicators['atr_percentile'].iloc[-1] > max_vol_percentile:
            logger.debug(f"Rejecting {symbol} - volatility too high")
            return None
        
        # Check if trend alignment is required and present
        need_trend_alignment = self.parameters['trend_alignment_required']
        
        if need_trend_alignment:
            # Check if price is above MA and MA is sloping in carry direction
            ma_slope = indicators['ma_slope'].iloc[-1]
            above_ma = indicators['above_ma'].iloc[-1]
            min_adx = self.parameters['min_adx_value']
            
            trend_aligned = (
                (carry_direction > 0 and above_ma and ma_slope > 0) or
                (carry_direction < 0 and not above_ma and ma_slope < 0)
            )
            
            trend_strong_enough = adx > min_adx
            
            if not (trend_aligned and trend_strong_enough):
                logger.debug(f"Rejecting {symbol} - trend not aligned with carry direction")
                return None
        
        # Calculate risk parameters
        stop_loss_multiple = self.parameters['stop_loss_atr_multiple']
        stop_loss = current_price - (carry_direction * atr * stop_loss_multiple)
        
        # For carry trades, take profit is often not fixed but trailing
        # We'll set a reference take profit level, but use trailing stops in practice
        take_profit = None
        
        # Calculate confidence based on various factors
        base_confidence = 0.5
        
        # Higher confidence for larger differentials
        differential_factor = min(0.3, abs(differential) / 10)
        
        # Higher confidence if pair is a known good carry pair
        favorite_boost = 0.1 if symbol in self.CARRY_FAVORITES else 0
        
        # Higher confidence if volatility is low
        volatility_factor = max(0, 0.2 * (1 - indicators['atr_percentile'].iloc[-1] / 100))
        
        # Higher confidence if trend is aligned and strong
        trend_factor = 0 if not need_trend_alignment else min(0.2, adx / 100)
        
        # Combine confidence factors
        confidence = min(0.95, base_confidence + differential_factor + favorite_boost + volatility_factor + trend_factor)
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            signal_type=SignalType.MARKET_OPEN,
            direction=carry_direction,
            confidence=confidence,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                'strategy': self.name,
                'setup_type': 'carry_trade',
                'interest_differential': differential,
                'daily_swap_points': indicators['daily_swap_points'].iloc[-1],
                'adx': adx,
                'atr': atr,
                'atr_percentile': indicators['atr_percentile'].iloc[-1],
                'estimated_annual_carry': differential  # Percentage points per year
            }
        )
        
        # Register this as an active carry trade
        self.active_carry_trades[symbol] = {
            'entry_time': current_time.isoformat(),
            'direction': carry_direction,
            'interest_differential': differential,
            'entry_price': current_price
        }
        
        return signal
    
    def _check_position_rollovers(self, current_time: datetime) -> None:
        """
        Check if any positions need to be rolled over to capture swaps.
        
        In forex, positions held at 5pm EST can receive/pay overnight interest.
        This method checks if it's approaching rollover time and emits events.
        
        Args:
            current_time: Current timestamp
        """
        if not self.parameters['roll_positions_automatically']:
            return
            
        # Check if it's time to roll positions (approaching 5pm EST / 10pm UTC)
        is_rollover_time = (
            current_time.hour == 21 and current_time.minute >= 50 or
            current_time.hour == 22 and current_time.minute <= 10
        )
        
        # Only check once per day around rollover time
        should_check = (
            is_rollover_time and
            (self.last_rollover_check is None or 
             (current_time - self.last_rollover_check).total_seconds() > 43200)  # 12 hours
        )
        
        if not should_check:
            return
            
        self.last_rollover_check = current_time
        
        # If we have active carry trades, emit a rollover event
        if self.active_carry_trades:
            event_data = {
                'strategy_name': self.name,
                'action': 'position_rollover',
                'active_trades': self.active_carry_trades,
                'rollover_time': current_time.replace(hour=22, minute=0, second=0).isoformat()
            }
            
            event = Event(
                event_type=EventType.POSITION_MANAGEMENT,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'carry_trade'}
            )
            self.event_bus.publish(event)
            
            logger.info(f"Preparing rollover for {len(self.active_carry_trades)} carry trade positions")
    
    def get_compatibility_score(self, market_regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Carry trades work best in stable, low volatility environments
        compatibility_map = {
            # Stable markets are best for carry trades
            MarketRegime.RANGING: 0.90,       # Excellent for stable carry collection
            MarketRegime.TRENDING_UP: 0.85,   # Good if carry matches trend (we filter for this)
            MarketRegime.TRENDING_DOWN: 0.80, # Same but slightly lower confidence
            
            # Choppy is fine if volatility isn't too high
            MarketRegime.CHOPPY: 0.65,        # Still workable
            
            # Worst regimes for carry
            MarketRegime.VOLATILE_BREAKOUT: 0.40,  # Dangerous for carry trades
            MarketRegime.VOLATILE_REVERSAL: 0.35,  # Very risky for carry positions
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.50        # Average compatibility
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
        if market_regime == MarketRegime.RANGING:
            # For ranging, stable markets, maximize carry potential
            optimized_params['min_interest_rate_differential'] = 1.0  # Lower threshold
            optimized_params['trend_alignment_required'] = False      # Don't need trend
            optimized_params['max_volatility_percentile'] = 80        # Allow higher volatility
            optimized_params['stop_loss_atr_multiple'] = 4.0          # Wider stops
            
        elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # For trending markets, emphasize trend alignment
            optimized_params['min_interest_rate_differential'] = 1.5  # Standard threshold
            optimized_params['trend_alignment_required'] = True       # Require trend alignment
            optimized_params['min_adx_value'] = 25                    # Stronger trend required
            optimized_params['max_volatility_percentile'] = 70        # Standard volatility
            optimized_params['stop_loss_atr_multiple'] = 3.0          # Standard stops
            
        elif market_regime == MarketRegime.CHOPPY:
            # For choppy markets, be more selective
            optimized_params['min_interest_rate_differential'] = 2.0  # Higher threshold
            optimized_params['trend_alignment_required'] = True       # Require trend alignment
            optimized_params['max_volatility_percentile'] = 60        # Lower volatility
            optimized_params['stop_loss_atr_multiple'] = 2.5          # Tighter stops
            
        elif market_regime in [MarketRegime.VOLATILE_BREAKOUT, MarketRegime.VOLATILE_REVERSAL]:
            # For volatile markets, be very conservative or avoid
            optimized_params['min_interest_rate_differential'] = 3.0  # Much higher threshold
            optimized_params['trend_alignment_required'] = True       # Require trend alignment
            optimized_params['min_adx_value'] = 35                    # Strong trend required
            optimized_params['max_volatility_percentile'] = 50        # Only low volatility
            optimized_params['stop_loss_atr_multiple'] = 2.0          # Tight stops
            optimized_params['max_carry_exposure'] = 0.2              # Reduce exposure
        
        # Log the optimization
        logger.info(f"Optimized {self.name} for {market_regime} regime")
        
        return optimized_params
