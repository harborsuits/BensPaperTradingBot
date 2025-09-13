#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Position Trading Strategy

This module implements a long-term position trading strategy for forex markets,
focusing on fundamental analysis and long-term trends over weeks to months.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta, time
import uuid

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)

class ForexPositionTradingStrategy(ForexBaseStrategy):
    """Position trading strategy for forex markets.
    
    This strategy focuses on long-term price movements by:
    1. Analyzing fundamental economic data
    2. Trading with the longer-term trend (weeks to months)
    3. Using higher timeframes (4H, Daily, Weekly)
    4. Weathering short-term volatility for long-term trends
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Time parameters
        'preferred_timeframes': [TimeFrame.HOUR_4, TimeFrame.DAY_1, TimeFrame.WEEK_1],
        'min_hold_period_days': 5,            # Minimum holding period
        'max_hold_period_days': 90,           # Maximum position duration (3 months)
        'entry_timeframes': [TimeFrame.HOUR_4, TimeFrame.DAY_1],  # For entry timing
        
        # Technical indicators
        'long_term_ma_period': 50,            # Long-term moving average (daily)
        'medium_term_ma_period': 20,          # Medium-term moving average
        'weekly_ema_period': 10,              # Weekly EMA for trend confirmation
        'daily_atr_period': 14,               # ATR period for volatility
        'monthly_pivot_lookback': 3,          # Months to look back for pivots
        
        # Fundamental parameters
        'interest_rate_threshold': 0.5,       # Minimum interest rate differential (%)
        'use_interest_rate_direction': True,  # Consider interest rate direction
        'economic_importance_threshold': 3,   # Minimum economic event importance (1-5 scale)
        'cot_report_use': True,               # Use COT report data if available
        'inflation_impact_factor': 1.0,       # How much inflation affects decision (0-1)
        'gdp_growth_impact_factor': 1.0,      # How much GDP growth affects decision (0-1)
        
        # Entry parameters
        'trend_confirmation_lookback': 20,    # Bars to confirm trend direction
        'entry_signal_threshold': 0.7,        # Minimum signal strength for entry
        'pullback_entry_percentage': 0.3,     # Enter on pullbacks of this % of trend
        
        # Position management
        'initial_stop_atr_multiple': 2.0,     # Initial stop loss as ATR multiple
        'trailing_stop_atr_multiple': 3.0,    # Trailing stop as ATR multiple
        'take_profit_atr_multiple': 6.0,      # Take profit as ATR multiple
        'partial_exits': True,                # Take partial profits
        'partial_exit_levels': [0.4, 0.6, 0.8],  # Levels for partial exits
        'partial_exit_sizes': [0.2, 0.3, 0.2],   # Size to exit at each level
        
        # Risk parameters
        'max_risk_per_trade_percent': 0.02,   # Max risk per trade (2%)
        'max_open_trades': 5,                 # Maximum concurrent open trades
        'position_size_adjustment': 1.0,      # Size adjustment based on conviction
    }
    
    def __init__(self, name: str = "Forex Position Trading", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex position trading strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters (will be merged with DEFAULT_PARAMETERS)
            metadata: Strategy metadata
        """
        # Merge default parameters with ForexBaseStrategy defaults
        forex_params = self.DEFAULT_PARAMETERS.copy()
        
        # Override with user-provided parameters if any
        if parameters:
            forex_params.update(parameters)
        
        # Initialize the base strategy
        super().__init__(name=name, parameters=forex_params, metadata=metadata)
        
        # Register with the event system
        self.event_bus = EventBus()
        
        # Strategy state
        self.active_positions = {}            # Active long-term positions
        self.fundamental_data = {}            # Cached fundamental data
        self.interest_rate_data = {}          # Interest rate data by currency
        self.economic_calendar = {}           # Major economic events
        self.monthly_pivots = {}              # Monthly pivot levels
        self.longterm_trend = {}              # Long-term trend state by pair
        
        logger.info(f"Initialized {self.name} strategy")
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals based on long-term position trading analysis.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        # Process each symbol
        for symbol, ohlcv in data.items():
            # Skip if we don't have enough data
            if len(ohlcv) < max(self.parameters['long_term_ma_period'] * 2, 100):
                logger.debug(f"Insufficient data for {symbol}, skipping position analysis")
                continue
            
            # Calculate technical indicators for position trading
            indicators = self._calculate_position_indicators(ohlcv)
            
            # Update long-term trend data
            self._update_longterm_trend(symbol, ohlcv, indicators)
            
            # Calculate monthly pivot levels if needed
            if symbol not in self.monthly_pivots or self._should_update_pivots(current_time):
                self._calculate_monthly_pivots(symbol, ohlcv)
            
            # Update fundamental data (in real implementation, this would pull from APIs)
            self._update_fundamental_data(symbol, current_time)
            
            # Evaluate for potential position trades
            signal = self._evaluate_position_setup(symbol, ohlcv, indicators, current_time)
            
            if signal:
                signals[symbol] = signal
                
                # Create event for new position signal
                event_data = {
                    'strategy_name': self.name,
                    'symbol': symbol,
                    'signal_type': signal.signal_type.name,
                    'entry_price': signal.entry_price,
                    'timestamp': current_time.isoformat(),
                    'timeframe': 'POSITION',  # Long-term position
                    'expected_hold_time': f"{self.parameters['min_hold_period_days']}-{self.parameters['max_hold_period_days']} days"
                }
                
                event = Event(
                    event_type=EventType.SIGNAL_GENERATED,
                    source=self.name,
                    data=event_data,
                    metadata={'strategy_type': 'forex', 'category': 'position_trading'}
                )
                self.event_bus.publish(event)
        
        # Update active positions and check for exits
        self._update_active_positions(data, current_time)
        
        return signals
    
    def _calculate_position_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for position trading.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary with calculated indicators
        """
        indicators = {}
        
        # Moving Averages on different timeframes
        indicators['ema_medium'] = self.calculate_ema(ohlcv, self.parameters['medium_term_ma_period'])
        indicators['sma_long'] = self.calculate_sma(ohlcv, self.parameters['long_term_ma_period'])
        
        # Trend strength indicators
        indicators['adx'] = self.calculate_adx(ohlcv, 14)  # Standard 14-period ADX
        
        # Volatility
        indicators['atr'] = self.calculate_atr(ohlcv, self.parameters['daily_atr_period'])
        
        # Momentum indicators for additional trend confirmation
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self.calculate_macd(ohlcv, 12, 26, 9)
        
        # RSI for overbought/oversold
        indicators['rsi'] = self.calculate_rsi(ohlcv, 14)
        
        # Calculate longer-term trend slope (using linear regression)
        indicators['trend_slope'] = self.calculate_slope(ohlcv['close'], self.parameters['trend_confirmation_lookback'])
        
        # Higher timeframe trend (simulate weekly by taking every 5 daily bars)
        if len(ohlcv) >= 50:  # Need enough data for weekly analysis
            # Simple approach: pick every 5th bar to simulate weekly
            weekly_data = ohlcv.iloc[::5].copy()
            indicators['weekly_ema'] = self.calculate_ema(weekly_data, self.parameters['weekly_ema_period'])
            
            # For trend direction on weekly
            if len(indicators['weekly_ema']) > 2:
                indicators['weekly_trend'] = 1 if indicators['weekly_ema'][-1] > indicators['weekly_ema'][-2] else -1
            else:
                indicators['weekly_trend'] = 0
        else:
            indicators['weekly_trend'] = 0  # Neutral if not enough data
        
        return indicators
    
    def _update_longterm_trend(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        Update the long-term trend state for this symbol.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
        """
        # Initialize trend state if needed
        if symbol not in self.longterm_trend:
            self.longterm_trend[symbol] = {
                'direction': 0,          # 1 for up, -1 for down, 0 for neutral
                'strength': 0,           # 0-1 scale
                'duration_days': 0,      # How many days in this trend
                'last_change': None      # When the trend last changed
            }
        
        # Determine trend direction
        trend_direction = 0
        trend_strength = 0
        
        # Check moving average alignment
        if indicators['ema_medium'][-1] > indicators['sma_long'][-1]:
            trend_direction += 1
        elif indicators['ema_medium'][-1] < indicators['sma_long'][-1]:
            trend_direction -= 1
        
        # Check weekly trend if available
        trend_direction += indicators['weekly_trend']
        
        # Check trend slope
        if indicators['trend_slope'][-1] > 0:
            trend_direction += 1
            trend_strength += 0.2 * min(1.0, abs(indicators['trend_slope'][-1]))
        elif indicators['trend_slope'][-1] < 0:
            trend_direction -= 1
            trend_strength += 0.2 * min(1.0, abs(indicators['trend_slope'][-1]))
        
        # Check ADX for trend strength
        if indicators['adx'][-1] > 30:
            trend_strength += 0.4 * min(1.0, indicators['adx'][-1] / 50)  # Normalize to 0-1
        
        # Determine final trend
        final_direction = 1 if trend_direction > 1 else (-1 if trend_direction < -1 else 0)
        
        # Update trend duration
        current_direction = self.longterm_trend[symbol]['direction']
        if final_direction == current_direction:
            # Same trend continuing
            self.longterm_trend[symbol]['duration_days'] += 1
        else:
            # Trend changed
            self.longterm_trend[symbol]['duration_days'] = 1
            self.longterm_trend[symbol]['last_change'] = ohlcv.index[-1]
        
        # Update trend state
        self.longterm_trend[symbol]['direction'] = final_direction
        self.longterm_trend[symbol]['strength'] = min(1.0, trend_strength)
    
    def _should_update_pivots(self, current_time: datetime) -> bool:
        """
        Check if we should update monthly pivot levels.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if pivots should be updated, False otherwise
        """
        # Simple implementation: update on first day of month
        return current_time.day == 1
    
    def _calculate_monthly_pivots(self, symbol: str, ohlcv: pd.DataFrame) -> None:
        """
        Calculate monthly pivot levels for long-term support/resistance.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
        """
        # Ensure we have enough data
        if len(ohlcv) < 30:
            return
        
        # Simple approach: use last month's data to calculate pivots
        # In a real implementation, this would use actual monthly bars
        
        # Assume the last 20-30 bars constitute "last month"
        lookback = min(30, len(ohlcv) - 1)
        
        # Calculate classic pivot points
        high = ohlcv['high'].iloc[-lookback:-1].max()
        low = ohlcv['low'].iloc[-lookback:-1].min()
        close = ohlcv['close'].iloc[-2]
        
        # Classic pivot formula
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        # Store pivot levels
        self.monthly_pivots[symbol] = {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3,
            'calc_date': ohlcv.index[-1]
        }
    
    def _update_fundamental_data(self, symbol: str, current_time: datetime) -> None:
        """
        Update fundamental data for the currency pair.
        In a real implementation, this would fetch from economic APIs.
        
        Args:
            symbol: Currency pair symbol
            current_time: Current timestamp
        """
        # Parse base and quote currencies from symbol
        currencies = self._extract_currencies(symbol)
        if not currencies:
            return
            
        base_currency, quote_currency = currencies
        
        # Mock fundamental data structure (in real implementation, this would be fetched from APIs)
        # This is a placeholder for demo purposes
        if symbol not in self.fundamental_data:
            self.fundamental_data[symbol] = {
                'interest_rate_differential': 0.5,  # Mock differential (percentage points)
                'rate_direction': {'base': 0, 'quote': 0},  # 1=hiking, -1=cutting, 0=stable
                'gdp_growth': {'base': 2.1, 'quote': 1.8},  # Annual GDP growth %
                'inflation': {'base': 2.2, 'quote': 1.9},    # Annual inflation %
                'economic_events': [],                       # Upcoming major events
                'cot_data': {                               # Commitment of Traders data
                    'net_position': 0,                      # Net position (long-short)
                    'position_change': 0                    # Weekly change
                },
                'last_updated': current_time
            }
        
        # In a real implementation, we would refresh this data based on a schedule
        # or when new reports are released
    
    def _extract_currencies(self, symbol: str) -> Optional[Tuple[str, str]]:
        """
        Extract base and quote currencies from a symbol.
        
        Args:
            symbol: Currency pair symbol (e.g., 'EURUSD')
            
        Returns:
            Tuple of (base_currency, quote_currency) or None if can't parse
        """
        # Common forex pairs use 6-8 characters
        if len(symbol) < 6:
            return None
            
        # For standard 6-char symbols like 'EURUSD'
        if len(symbol) == 6:
            return symbol[:3], symbol[3:]
            
        # Handle symbols with separators like 'EUR/USD' or 'EUR_USD'
        for sep in ['/', '_']:
            if sep in symbol:
                parts = symbol.split(sep)
                if len(parts) == 2:
                    return parts[0], parts[1]
        
        # Default fallback for 6+ char symbols without separators
        return symbol[:3], symbol[3:6]
    
    def _evaluate_position_setup(self, symbol: str, ohlcv: pd.DataFrame, 
                                 indicators: Dict[str, Any], current_time: datetime) -> Optional[Signal]:
        """
        Evaluate if there's a valid position trade setup for this symbol.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            current_time: Current timestamp
            
        Returns:
            Signal object if a setup is found, None otherwise
        """
        # Current price data
        current_price = ohlcv['close'].iloc[-1]
        
        # Skip if we already have an active position for this symbol
        for position_id, position in self.active_positions.items():
            if position['symbol'] == symbol:
                logger.debug(f"Already have active position for {symbol}, skipping setup evaluation")
                return None
        
        # Skip if we're at max positions
        if len(self.active_positions) >= self.parameters['max_open_trades']:
            logger.debug(f"Maximum open positions ({self.parameters['max_open_trades']}) reached, skipping setup")
            return None
        
        # Check fundamental factors first (they are slower to change)
        fundamental_score = self._calculate_fundamental_score(symbol)
        
        # If fundamental factors are not favorable, skip
        if fundamental_score < 0.3:  # Minimum threshold for fundamentals
            logger.debug(f"Fundamental factors not favorable for {symbol}, score: {fundamental_score:.2f}")
            return None
        
        # Check long-term trend
        if symbol not in self.longterm_trend:
            return None
            
        trend_data = self.longterm_trend[symbol]
        trend_direction = trend_data['direction']
        trend_strength = trend_data['strength']
        
        # If trend is not clear, skip
        if trend_direction == 0 or trend_strength < 0.4:
            logger.debug(f"No clear long-term trend for {symbol}, skipping")
            return None
        
        # Calculate technical score
        technical_score = self._calculate_technical_score(symbol, ohlcv, indicators, trend_direction)
        
        # Combined score with heavy weight on fundamentals for position trading
        combined_score = (0.6 * fundamental_score) + (0.4 * technical_score)
        signal_threshold = self.parameters['entry_signal_threshold']
        
        # Check for entry conditions
        if combined_score >= signal_threshold:
            # Determine trade direction
            signal_type = SignalType.LONG if trend_direction > 0 else SignalType.SHORT
            
            # Calculate stop loss and take profit levels
            atr_value = indicators['atr'][-1]
            
            if signal_type == SignalType.LONG:
                stop_loss = current_price - (atr_value * self.parameters['initial_stop_atr_multiple'])
                take_profit = current_price + (atr_value * self.parameters['take_profit_atr_multiple'])
            else:  # SHORT
                stop_loss = current_price + (atr_value * self.parameters['initial_stop_atr_multiple'])
                take_profit = current_price - (atr_value * self.parameters['take_profit_atr_multiple'])
            
            # Create position ID
            position_id = f"POS-{symbol}-{current_time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
            
            # Store as active position
            self.active_positions[position_id] = {
                'symbol': symbol,
                'direction': signal_type.name,
                'entry_time': current_time.isoformat(),
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'score': combined_score,
                'atr': atr_value,
                'fundamental_score': fundamental_score,
                'technical_score': technical_score,
                'partial_exits_taken': [],
                'hold_duration_target': self.parameters['min_hold_period_days']
            }
            
            # Create signal with expected holding period
            expiration = current_time + timedelta(days=self.parameters['max_hold_period_days'])
            
            return Signal(
                symbol=symbol,
                signal_type=signal_type,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                quantity=None,  # Will be calculated by position sizing
                timestamp=current_time,
                expiration=expiration,
                source=self.name,
                metadata={
                    'strategy_type': 'forex_position',
                    'score': combined_score,
                    'position_id': position_id,
                    'fundamental_score': fundamental_score,
                    'technical_score': technical_score,
                    'regime': self.current_regime if hasattr(self, 'current_regime') else 'unknown',
                    'hold_duration_target': f"{self.parameters['min_hold_period_days']}-{self.parameters['max_hold_period_days']} days"
                }
            )
        
        return None
    
    def _calculate_fundamental_score(self, symbol: str) -> float:
        """
        Calculate fundamental strength score for position trading.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            Fundamental score from 0.0 (weak) to 1.0 (strong)
        """
        if symbol not in self.fundamental_data:
            return 0.5  # Neutral if we don't have data
        
        fund_data = self.fundamental_data[symbol]
        score = 0.5  # Start at neutral
        
        # Interest rate differential (key driver for currency strength)
        ir_diff = fund_data['interest_rate_differential']
        rate_impact = min(1.0, abs(ir_diff) / 2.0) * 0.3  # Up to 30% of score
        
        if ir_diff > 0:  # Positive differential favors base currency
            score += rate_impact
        elif ir_diff < 0:  # Negative differential favors quote currency
            score -= rate_impact
        
        # Interest rate direction
        base_direction = fund_data['rate_direction']['base']
        quote_direction = fund_data['rate_direction']['quote']
        direction_impact = 0.2  # 20% of score
        
        # Rising rates are bullish for currency
        if base_direction > 0:
            score += 0.1 * direction_impact
        elif base_direction < 0:
            score -= 0.1 * direction_impact
            
        if quote_direction > 0:
            score -= 0.1 * direction_impact  # Rising quote rates are bearish for the pair
        elif quote_direction < 0:
            score += 0.1 * direction_impact
        
        # GDP growth differential
        gdp_diff = fund_data['gdp_growth']['base'] - fund_data['gdp_growth']['quote']
        gdp_impact = min(1.0, abs(gdp_diff) / 3.0) * 0.2 * self.parameters['gdp_growth_impact_factor']
        
        if gdp_diff > 0:  # Stronger base economy
            score += gdp_impact
        elif gdp_diff < 0:  # Stronger quote economy
            score -= gdp_impact
        
        # Inflation differential (lower inflation is generally positive)
        inf_diff = fund_data['inflation']['quote'] - fund_data['inflation']['base']
        inf_impact = min(1.0, abs(inf_diff) / 3.0) * 0.2 * self.parameters['inflation_impact_factor']
        
        if inf_diff > 0:  # Lower base inflation (positive)
            score += inf_impact
        elif inf_diff < 0:  # Higher base inflation (negative)
            score -= inf_impact
        
        # COT data if available and enabled
        if self.parameters['cot_report_use'] and 'cot_data' in fund_data:
            cot_data = fund_data['cot_data']
            cot_impact = 0.1  # 10% of score
            
            # Positive net position is bullish
            if cot_data['net_position'] > 0:
                score += cot_impact * min(1.0, abs(cot_data['net_position']) / 100000)
            elif cot_data['net_position'] < 0:
                score -= cot_impact * min(1.0, abs(cot_data['net_position']) / 100000)
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, score))
    
    def _calculate_technical_score(self, symbol: str, ohlcv: pd.DataFrame, 
                                   indicators: Dict[str, Any], trend_direction: int) -> float:
        """
        Calculate technical strength score for position trading.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            trend_direction: Long-term trend direction (1=up, -1=down)
            
        Returns:
            Technical score from 0.0 (weak) to 1.0 (strong)
        """
        score = 0.5  # Start at neutral
        
        # For position trading, we want to enter on pullbacks in the direction of the trend
        current_price = ohlcv['close'].iloc[-1]
        
        # Check if price is near support/resistance levels
        if symbol in self.monthly_pivots:
            pivots = self.monthly_pivots[symbol]
            nearest_level = None
            min_distance = float('inf')
            
            # Find nearest pivot level
            for level_name in ['pivot', 's1', 's2', 's3', 'r1', 'r2', 'r3']:
                level = pivots[level_name]
                distance = abs(current_price - level)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_level = level_name
            
            # Score based on proximity to support/resistance
            if nearest_level:
                atr = indicators['atr'][-1] if 'atr' in indicators else (current_price * 0.01)  # Fallback 1%
                distance_ratio = min_distance / atr
                
                # Close to level = better score
                level_score = max(0, 0.3 - (0.1 * distance_ratio))
                
                # For supports and resistances, check alignment with trend
                if trend_direction > 0:  # Uptrend
                    if nearest_level.startswith('s'):  # Near support in uptrend (good)
                        score += level_score
                    elif nearest_level.startswith('r'):  # Near resistance in uptrend (not ideal)
                        pass  # Neutral
                elif trend_direction < 0:  # Downtrend
                    if nearest_level.startswith('r'):  # Near resistance in downtrend (good)
                        score += level_score
                    elif nearest_level.startswith('s'):  # Near support in downtrend (not ideal)
                        pass  # Neutral
        
        # Check for pullback entry
        lookback = min(20, len(ohlcv) - 2)
        max_price = ohlcv['high'].iloc[-lookback:-1].max() if trend_direction > 0 else ohlcv['high'].iloc[-2]
        min_price = ohlcv['low'].iloc[-lookback:-1].min() if trend_direction < 0 else ohlcv['low'].iloc[-2]
        
        # Calculate pullback percentage
        if trend_direction > 0:  # Uptrend
            pullback = (max_price - current_price) / (max_price - min_price) if (max_price - min_price) > 0 else 0
        else:  # Downtrend
            pullback = (current_price - min_price) / (max_price - min_price) if (max_price - min_price) > 0 else 0
        
        # Score based on pullback - we want to enter after a pullback in trending direction
        ideal_pullback = self.parameters['pullback_entry_percentage']
        pullback_score = 0.3 * (1.0 - min(1.0, abs(pullback - ideal_pullback) / ideal_pullback))
        score += pullback_score
        
        # Check momentum alignment
        if 'macd' in indicators and 'macd_signal' in indicators:
            # For position trades, we want MACD histogram starting to turn in trend direction
            if trend_direction > 0 and indicators['macd'][-1] > indicators['macd_signal'][-1]:
                score += 0.1
            elif trend_direction < 0 and indicators['macd'][-1] < indicators['macd_signal'][-1]:
                score += 0.1
        
        # Weekly trend alignment adds significant weight
        if indicators['weekly_trend'] == trend_direction:
            score += 0.2
        
        # ADX strength - strong trends are good for position trades
        if 'adx' in indicators:
            adx = indicators['adx'][-1]
            if adx > 30:  # Strong trend
                score += 0.1 * min(1.0, (adx - 20) / 30)
        
        # Normalize score
        return max(0.0, min(1.0, score))
    
    def _update_active_positions(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> None:
        """
        Update active positions and check for exits.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
        """
        positions_to_exit = []
        partial_exits = []
        
        for position_id, position in self.active_positions.items():
            symbol = position['symbol']
            
            # Skip if we don't have data for this symbol
            if symbol not in data:
                continue
                
            ohlcv = data[symbol]
            current_price = ohlcv['close'].iloc[-1]
            entry_time = datetime.fromisoformat(position['entry_time'])
            position_age_days = (current_time - entry_time).days
            
            # Check if position hit take profit
            take_profit = position['take_profit']
            stop_loss = position['stop_loss']
            
            # For long positions
            if position['direction'] == 'LONG':
                # Check stop loss
                if ohlcv['low'].iloc[-1] <= stop_loss:
                    positions_to_exit.append((position_id, 'stop_loss'))
                    continue
                    
                # Check take profit
                if ohlcv['high'].iloc[-1] >= take_profit:
                    positions_to_exit.append((position_id, 'take_profit'))
                    continue
                    
                # Check partial exits
                if self.parameters['partial_exits']:
                    exit_levels = self.parameters['partial_exit_levels']
                    exit_sizes = self.parameters['partial_exit_sizes']
                    
                    profit_range = take_profit - position['entry_price']
                    current_profit = current_price - position['entry_price']
                    profit_percent = current_profit / profit_range if profit_range > 0 else 0
                    
                    for i, level in enumerate(exit_levels):
                        if profit_percent >= level and i not in position['partial_exits_taken']:
                            partial_exits.append((position_id, i, exit_sizes[i]))
                
            # For short positions
            elif position['direction'] == 'SHORT':
                # Check stop loss
                if ohlcv['high'].iloc[-1] >= stop_loss:
                    positions_to_exit.append((position_id, 'stop_loss'))
                    continue
                    
                # Check take profit
                if ohlcv['low'].iloc[-1] <= take_profit:
                    positions_to_exit.append((position_id, 'take_profit'))
                    continue
                    
                # Check partial exits
                if self.parameters['partial_exits']:
                    exit_levels = self.parameters['partial_exit_levels']
                    exit_sizes = self.parameters['partial_exit_sizes']
                    
                    profit_range = position['entry_price'] - take_profit
                    current_profit = position['entry_price'] - current_price
                    profit_percent = current_profit / profit_range if profit_range > 0 else 0
                    
                    for i, level in enumerate(exit_levels):
                        if profit_percent >= level and i not in position['partial_exits_taken']:
                            partial_exits.append((position_id, i, exit_sizes[i]))
            
            # Check time-based exit
            if position_age_days >= self.parameters['max_hold_period_days']:
                positions_to_exit.append((position_id, 'time_exit'))
                continue
            
            # Check for trend reversal exit
            if symbol in self.longterm_trend:
                current_trend = self.longterm_trend[symbol]['direction']
                if (position['direction'] == 'LONG' and current_trend < 0) or \
                   (position['direction'] == 'SHORT' and current_trend > 0):
                    # Only exit on trend reversal if we've held for minimum period
                    if position_age_days >= self.parameters['min_hold_period_days']:
                        positions_to_exit.append((position_id, 'trend_reversal'))
                        continue
        
        # Process exits
        for position_id, exit_reason in positions_to_exit:
            position = self.active_positions[position_id]
            symbol = position['symbol']
            current_price = data[symbol]['close'].iloc[-1] if symbol in data else 0
            
            # Calculate profit/loss
            if position['direction'] == 'LONG':
                profit_pips = (current_price - position['entry_price']) / self.pip_value
            else:  # SHORT
                profit_pips = (position['entry_price'] - current_price) / self.pip_value
            
            # Publish exit event
            event_data = {
                'strategy_name': self.name,
                'symbol': symbol,
                'position_id': position_id,
                'exit_reason': exit_reason,
                'entry_time': position['entry_time'],
                'exit_time': current_time.isoformat(),
                'position_direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'profit_pips': profit_pips,
                'hold_days': (current_time - datetime.fromisoformat(position['entry_time'])).days
            }
            
            event = Event(
                event_type=EventType.POSITION_EXIT,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'position_trading'}
            )
            self.event_bus.publish(event)
            
            # Remove from active positions
            self.active_positions.pop(position_id, None)
        
        # Process partial exits
        for position_id, exit_index, exit_size in partial_exits:
            position = self.active_positions[position_id]
            symbol = position['symbol']
            current_price = data[symbol]['close'].iloc[-1] if symbol in data else 0
            
            # Calculate profit/loss
            if position['direction'] == 'LONG':
                profit_pips = (current_price - position['entry_price']) / self.pip_value
            else:  # SHORT
                profit_pips = (position['entry_price'] - current_price) / self.pip_value
            
            # Publish partial exit event
            event_data = {
                'strategy_name': self.name,
                'symbol': symbol,
                'position_id': position_id,
                'exit_reason': f'partial_exit_{exit_index}',
                'exit_size': exit_size,
                'entry_time': position['entry_time'],
                'exit_time': current_time.isoformat(),
                'position_direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'profit_pips': profit_pips,
                'hold_days': (current_time - datetime.fromisoformat(position['entry_time'])).days
            }
            
            event = Event(
                event_type=EventType.PARTIAL_EXIT,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'position_trading'}
            )
            self.event_bus.publish(event)
            
            # Update position to track partial exits taken
            position['partial_exits_taken'].append(exit_index)
            self.active_positions[position_id] = position
    
    def get_regime_compatibility_score(self, regime: MarketRegime) -> float:
        """
        Calculate how compatible this strategy is with the given market regime.
        
        Args:
            regime: Market regime to check compatibility with
            
        Returns:
            Compatibility score from 0.0 (incompatible) to 1.0 (highly compatible)
        """
        compatibility_scores = {
            MarketRegime.TRENDING_BULL: 0.9,    # Excellent in established uptrends
            MarketRegime.TRENDING_BEAR: 0.9,    # Excellent in established downtrends
            MarketRegime.RANGING: 0.3,          # Poor in ranging markets - needs direction
            MarketRegime.VOLATILE: 0.2,         # Poor in volatile markets - too much noise
            MarketRegime.VOLATILE_BULL: 0.4,    # Can work but with caution in volatile uptrends
            MarketRegime.VOLATILE_BEAR: 0.4,    # Can work but with caution in volatile downtrends
            MarketRegime.BREAKOUT: 0.5,         # Moderate after breakout confirms new trend
            MarketRegime.REVERSAL: 0.2,         # Poor during reversals - wants established trends
            MarketRegime.LOW_VOLATILITY: 0.7,   # Good in low volatility trending markets
            MarketRegime.UNDEFINED: 0.5,        # Moderate compatibility when regime is unclear
        }
        
        return compatibility_scores.get(regime, 0.5)  # Default moderate compatibility
    
    def optimize_for_regime(self, regime: MarketRegime) -> None:
        """
        Optimize strategy parameters for the specified market regime.
        
        Args:
            regime: Market regime to optimize for
        """
        self.current_regime = regime
        
        # Base parameters that don't change
        base_params = {
            'preferred_timeframes': self.parameters['preferred_timeframes'],
            'min_hold_period_days': self.parameters['min_hold_period_days'],
            'max_hold_period_days': self.parameters['max_hold_period_days'],
            'monthly_pivot_lookback': self.parameters['monthly_pivot_lookback']
        }
        
        # Regime-specific parameter adjustments
        if regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
            # In strong trends, focus on trend following with wider targets
            regime_params = {
                'entry_signal_threshold': 0.65,            # More permissive entries in trends
                'pullback_entry_percentage': 0.3,         # Enter on moderate pullbacks
                'initial_stop_atr_multiple': 2.0,         # Standard stop for trends
                'trailing_stop_atr_multiple': 2.5,        # Wider trailing to let trends run
                'take_profit_atr_multiple': 6.0,          # Extended take profit in trends
                'partial_exits': True,
                'partial_exit_levels': [0.4, 0.6, 0.8],   # Progressive profit taking
                'partial_exit_sizes': [0.2, 0.3, 0.2],    # Total 70% in partials, 30% at final
                'position_size_adjustment': 1.2,          # Slightly larger positions in trends
                'max_open_trades': 5                      # Full allocation in strong trends
            }
            
        elif regime in [MarketRegime.VOLATILE_BULL, MarketRegime.VOLATILE_BEAR]:
            # In volatile trends, more caution with tighter risk
            regime_params = {
                'entry_signal_threshold': 0.8,             # More stringent entries
                'pullback_entry_percentage': 0.4,          # Deeper pullbacks before entry
                'initial_stop_atr_multiple': 2.5,          # Wider stops for volatility
                'trailing_stop_atr_multiple': 2.0,         # Tighter trailing to lock profits
                'take_profit_atr_multiple': 5.0,           # Moderate take profit
                'partial_exits': True,
                'partial_exit_levels': [0.3, 0.5, 0.7],    # Earlier partial exits
                'partial_exit_sizes': [0.3, 0.3, 0.2],     # Take more profits early (80% total)
                'position_size_adjustment': 0.8,           # Smaller positions in volatility
                'max_open_trades': 3                       # Fewer trades in volatile conditions
            }
            
        elif regime == MarketRegime.LOW_VOLATILITY:
            # In low volatility trending markets, more patient approach
            regime_params = {
                'entry_signal_threshold': 0.6,             # More permissive entries in low vol
                'pullback_entry_percentage': 0.2,          # Smaller pullbacks are significant
                'initial_stop_atr_multiple': 3.0,          # Wider stops relative to small ATR
                'trailing_stop_atr_multiple': 4.0,         # Very wide trailing in low volatility
                'take_profit_atr_multiple': 8.0,           # Extended targets in low vol
                'partial_exits': True,
                'partial_exit_levels': [0.5, 0.7, 0.9],    # Later partial exits
                'partial_exit_sizes': [0.2, 0.2, 0.2],     # Keep 40% to final target
                'position_size_adjustment': 1.1,           # Slightly larger positions
                'max_open_trades': 4                       # Standard allocation
            }
            
        elif regime == MarketRegime.BREAKOUT:
            # For breakouts, focus on confirmed new trends
            regime_params = {
                'entry_signal_threshold': 0.7,             # Moderate threshold
                'pullback_entry_percentage': 0.25,         # Shallow pullbacks after breakout
                'initial_stop_atr_multiple': 2.2,          # Moderate stops
                'trailing_stop_atr_multiple': 2.5,         # Moderate trailing
                'take_profit_atr_multiple': 5.5,           # Good profit target for breakouts
                'partial_exits': True,
                'partial_exit_levels': [0.4, 0.6, 0.8],    # Standard partial exits
                'partial_exit_sizes': [0.2, 0.3, 0.2],     # Standard sizing
                'position_size_adjustment': 1.0,           # Standard position size
                'max_open_trades': 4                       # Standard allocation
            }
            
        elif regime == MarketRegime.RANGING:
            # In ranging markets, more conservative approach
            regime_params = {
                'entry_signal_threshold': 0.85,            # Very stringent entries
                'pullback_entry_percentage': 0.1,          # Minimal pullbacks in ranges
                'initial_stop_atr_multiple': 2.0,          # Standard stops
                'trailing_stop_atr_multiple': 1.5,         # Tighter trailing in ranges
                'take_profit_atr_multiple': 3.0,           # Reduced profit targets in ranges
                'partial_exits': True,
                'partial_exit_levels': [0.3, 0.5, 0.7],    # Earlier profit taking
                'partial_exit_sizes': [0.3, 0.3, 0.3],     # Take all profits in partials
                'position_size_adjustment': 0.7,           # Much smaller positions
                'max_open_trades': 2                       # Minimal allocation in ranges
            }
        
        else:  # UNDEFINED, REVERSAL or other
            # Balanced, more conservative approach for unclear regimes
            regime_params = {
                'entry_signal_threshold': 0.75,            # More stringent entries
                'pullback_entry_percentage': 0.3,          # Standard pullbacks
                'initial_stop_atr_multiple': 2.0,          # Standard stops
                'trailing_stop_atr_multiple': 2.0,         # Standard trailing
                'take_profit_atr_multiple': 4.0,           # Moderate targets
                'partial_exits': True,
                'partial_exit_levels': [0.4, 0.6, 0.8],    # Standard partial exits
                'partial_exit_sizes': [0.2, 0.3, 0.2],     # Standard sizing
                'position_size_adjustment': 0.9,           # Slightly reduced position size
                'max_open_trades': 3                       # Moderate allocation
            }
        
        # Combine base and regime-specific parameters
        optimized_params = {**self.parameters.copy(), **base_params, **regime_params}
        
        # Update strategy parameters
        self.parameters.update(optimized_params)
        
        logger.info(f"Optimized Position Trading strategy for {regime.name} regime")
        
    def optimize(self, data: Dict[str, pd.DataFrame], parameter_ranges: Optional[Dict[str, List[Any]]] = None):
        """
        Optimize strategy parameters using historical data.
        
        Args:
            data: Dictionary mapping symbols to historical OHLCV data
            parameter_ranges: Optional parameter ranges to optimize (if None, use defaults)
        """
        if not parameter_ranges:
            # Default parameter ranges to search
            parameter_ranges = {
                'long_term_ma_period': [40, 50, 60, 80],
                'medium_term_ma_period': [15, 20, 25, 30],
                'pullback_entry_percentage': [0.2, 0.3, 0.4],
                'initial_stop_atr_multiple': [1.5, 2.0, 2.5, 3.0],
                'take_profit_atr_multiple': [4.0, 5.0, 6.0, 7.0],
            }
        
        logger.info(f"Starting parameter optimization for {self.name}")
        
        # Implement grid search or other optimization method here
        # This would be a more extensive implementation in a real system
        
        # For now, we'll just log that optimization would happen
        # and set some reasonable default parameters
        
        logger.info(f"Completed parameter optimization for {self.name}")
        
        # In a real implementation, we would:
        # 1. Generate all parameter combinations
        # 2. Backtest each combination
        # 3. Score results based on Sharpe ratio, profit factor, max drawdown
        # 4. Select best parameters
        # 5. Update self.parameters
