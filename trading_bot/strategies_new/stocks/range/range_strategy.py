#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Range Trading Strategy

This module implements a stock range trading strategy that aims to profit from price
movements within established support and resistance levels. The strategy is account-aware,
ensuring it complies with account balance requirements, regulatory constraints, and risk management.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.validation.account_aware_mixin import AccountAwareMixin
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.position import Position, PositionStatus
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="RangeStrategy",
    market_type="stocks",
    description="A strategy that trades within established price ranges, buying at support and selling at resistance",
    timeframes=["1d", "4h", "1h"],
    parameters={
        "lookback_period": {"description": "Period for identifying support/resistance levels", "type": "int"},
        "range_strength_threshold": {"description": "Minimum strength required for range validation", "type": "float"},
        "rsi_upper": {"description": "Upper RSI threshold for overbought condition", "type": "int"},
        "rsi_lower": {"description": "Lower RSI threshold for oversold condition", "type": "int"}
    }
)
class RangeStrategy(StocksBaseStrategy, AccountAwareMixin):
    """
    Stock Range Trading Strategy
    
    This strategy:
    1. Identifies established price ranges using support and resistance levels
    2. Buys near support levels when price is oversold
    3. Sells near resistance levels when price is overbought
    4. Uses oscillators and mean reversion indicators for confirmation
    5. Adapts to different market volatility levels
    6. Incorporates account awareness for regulatory compliance and risk management
    
    The range strategy works best in sideways, non-trending markets and is designed for
    short to medium-term trades.
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Range Trading strategy.
        
        Args:
            session: Stock trading session with symbol, timeframe, etc.
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize parent classes
        StocksBaseStrategy.__init__(self, session, data_pipeline, parameters)
        AccountAwareMixin.__init__(self)
        
        # Default parameters for range trading
        default_params = {
            # Strategy identification
            'strategy_name': 'Range Trading',
            'strategy_id': 'range_trading',
            'is_day_trade': False,  # Can be intraday or multi-day
            
            # Range identification parameters
            'lookback_period': 20,
            'min_range_bars': 10,
            'range_strength_threshold': 0.7,  # Minimum strength required for range validation
            'min_touches': 2,  # Minimum number of touches for support/resistance validation
            
            # Oscillator parameters
            'rsi_period': 14,
            'rsi_upper': 70,
            'rsi_lower': 30,
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            
            # Trade execution
            'entry_buffer_pct': 0.01,  # 1% buffer from exact support/resistance
            'position_hold_time': 5,  # Hold position for at least this many bars
            'max_positions': 3,
            
            # Risk management
            'risk_per_trade': 0.01,  # 1% risk per trade
            'reward_risk_ratio': 1.5,  # Minimum reward:risk ratio
            'max_range_width_atr': 5,  # Maximum range width in ATR
            'max_position_size_pct': 0.10,  # Maximum 10% of account in a single position
        }
        
        # Update with user-provided parameters
        if parameters:
            default_params.update(parameters)
        self.parameters = default_params
        
        # Strategy state
        self.support_levels = []
        self.resistance_levels = []
        self.current_range = {
            'support': None,
            'resistance': None,
            'strength': 0.0,
            'detected_at': None
        }
        
        logger.info(f"Initialized {self.name} for {session.symbol} on {session.timeframe}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Range Trading strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < max(self.parameters['lookback_period'], 
                                        self.parameters['rsi_period'],
                                        self.parameters['bollinger_period']):
            return indicators
        
        try:
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.parameters['rsi_period']).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.parameters['rsi_period']).mean()
            
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate Bollinger Bands
            middle_band = data['close'].rolling(window=self.parameters['bollinger_period']).mean()
            std_dev = data['close'].rolling(window=self.parameters['bollinger_period']).std()
            upper_band = middle_band + (std_dev * self.parameters['bollinger_std'])
            lower_band = middle_band - (std_dev * self.parameters['bollinger_std'])
            
            indicators['bb_middle'] = middle_band
            indicators['bb_upper'] = upper_band
            indicators['bb_lower'] = lower_band
            indicators['bb_width'] = (upper_band - lower_band) / middle_band
            
            # Calculate ATR for volatility assessment
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean()
            
            # Identify support and resistance levels
            self._identify_support_resistance(data)
            
            # Calculate distance from current price to support/resistance
            if self.current_range['support'] is not None and self.current_range['resistance'] is not None:
                current_price = data['close'].iloc[-1]
                support = self.current_range['support']
                resistance = self.current_range['resistance']
                
                indicators['distance_to_support'] = (current_price - support) / support
                indicators['distance_to_resistance'] = (resistance - current_price) / current_price
                indicators['range_position'] = (current_price - support) / (resistance - support)
                
        except Exception as e:
            logger.error(f"Error calculating range trading indicators: {str(e)}")
        
        return indicators
    
    def _identify_support_resistance(self, data: pd.DataFrame) -> None:
        """
        Identify support and resistance levels based on price action.
        
        Args:
            data: Market data DataFrame with OHLCV columns
        """
        try:
            # Use only the lookback period for analysis
            lookback = self.parameters['lookback_period']
            if len(data) < lookback:
                return
                
            analysis_data = data.tail(lookback)
            
            # Find local minima (support) and maxima (resistance)
            local_min = []
            local_max = []
            
            for i in range(1, len(analysis_data)-1):
                # Local minimum (support)
                if (analysis_data['low'].iloc[i] < analysis_data['low'].iloc[i-1] and 
                    analysis_data['low'].iloc[i] < analysis_data['low'].iloc[i+1]):
                    local_min.append((i, analysis_data['low'].iloc[i]))
                
                # Local maximum (resistance)
                if (analysis_data['high'].iloc[i] > analysis_data['high'].iloc[i-1] and 
                    analysis_data['high'].iloc[i] > analysis_data['high'].iloc[i+1]):
                    local_max.append((i, analysis_data['high'].iloc[i]))
            
            # Group similar price levels (within 0.5% of each other)
            support_clusters = self._cluster_price_levels([price for _, price in local_min])
            resistance_clusters = self._cluster_price_levels([price for _, price in local_max])
            
            # Get the strongest levels (most touches)
            self.support_levels = [level for level, _ in support_clusters[:3]] if support_clusters else []
            self.resistance_levels = [level for level, _ in resistance_clusters[:3]] if resistance_clusters else []
            
            # Find the current trading range
            if self.support_levels and self.resistance_levels:
                self._identify_current_range(data)
                
        except Exception as e:
            logger.error(f"Error identifying support/resistance levels: {str(e)}")
    
    def _cluster_price_levels(self, price_levels: List[float], tolerance: float = 0.005) -> List[Tuple[float, int]]:
        """
        Cluster similar price levels together and count touches.
        
        Args:
            price_levels: List of price levels to cluster
            tolerance: Percentage tolerance for clustering
            
        Returns:
            List of tuples with (price_level, touch_count) sorted by touch count
        """
        if not price_levels:
            return []
            
        # Sort price levels
        sorted_levels = sorted(price_levels)
        
        # Cluster similar levels
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            current_price = sorted_levels[i]
            prev_price = current_cluster[-1]
            
            # If within tolerance, add to current cluster
            if abs(current_price - prev_price) / prev_price <= tolerance:
                current_cluster.append(current_price)
            else:
                # New cluster
                avg_price = sum(current_cluster) / len(current_cluster)
                clusters.append((avg_price, len(current_cluster)))
                current_cluster = [current_price]
        
        # Add the last cluster
        if current_cluster:
            avg_price = sum(current_cluster) / len(current_cluster)
            clusters.append((avg_price, len(current_cluster)))
        
        # Sort clusters by touch count (descending)
        return sorted(clusters, key=lambda x: x[1], reverse=True)
    
    def _identify_current_range(self, data: pd.DataFrame) -> None:
        """
        Identify the current trading range based on support and resistance levels.
        
        Args:
            data: Market data DataFrame with OHLCV columns
        """
        current_price = data['close'].iloc[-1]
        
        # Find closest support below current price
        below_supports = [s for s in self.support_levels if s < current_price]
        support = min(below_supports, key=lambda x: current_price - x) if below_supports else None
        
        # Find closest resistance above current price
        above_resistances = [r for r in self.resistance_levels if r > current_price]
        resistance = min(above_resistances, key=lambda x: x - current_price) if above_resistances else None
        
        if support is not None and resistance is not None:
            # Calculate range width relative to ATR
            atr = data['high'].iloc[-1] - data['low'].iloc[-1]  # Simplified ATR
            range_width = resistance - support
            range_width_atr = range_width / atr if atr > 0 else 0
            
            # Calculate range strength based on:
            # 1. Number of times price has respected the range
            # 2. Width of range relative to ATR
            # 3. Duration of the range
            
            # Simplified range strength score (0-1)
            touch_count = self._count_range_touches(data, support, resistance)
            range_strength = min(1.0, touch_count / self.parameters['min_touches'])
            
            # Check if range is valid
            if (range_strength >= self.parameters['range_strength_threshold'] and 
                range_width_atr <= self.parameters['max_range_width_atr']):
                
                self.current_range = {
                    'support': support,
                    'resistance': resistance,
                    'strength': range_strength,
                    'width': range_width,
                    'width_atr': range_width_atr,
                    'detected_at': len(data) - 1  # Index where range was detected
                }
                
                logger.info(f"Identified trading range: Support={support:.2f}, "
                           f"Resistance={resistance:.2f}, Strength={range_strength:.2f}")
            else:
                logger.debug(f"Potential range does not meet criteria: "
                            f"Support={support:.2f}, Resistance={resistance:.2f}, "
                            f"Strength={range_strength:.2f}, Width ATR={range_width_atr:.2f}")
    
    def _count_range_touches(self, data: pd.DataFrame, support: float, resistance: float) -> int:
        """
        Count how many times price has touched support or resistance levels.
        
        Args:
            data: Market data DataFrame
            support: Support level
            resistance: Resistance level
            
        Returns:
            Number of touches
        """
        # Define touch area (0.5% buffer)
        support_lower = support * 0.995
        support_upper = support * 1.005
        resistance_lower = resistance * 0.995
        resistance_upper = resistance * 1.005
        
        # Count support touches
        support_touches = sum(1 for i in range(len(data)) if 
                             data['low'].iloc[i] >= support_lower and 
                             data['low'].iloc[i] <= support_upper)
        
        # Count resistance touches
        resistance_touches = sum(1 for i in range(len(data)) if 
                                data['high'].iloc[i] >= resistance_lower and 
                                data['high'].iloc[i] <= resistance_upper)
        
        return support_touches + resistance_touches
        
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the Range Trading strategy.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {
            'entry': False,
            'exit': False,
            'direction': None,
            'stop_loss': None,
            'take_profit': None,
            'strength': 0.0,
            'positions_to_close': []
        }
        
        if data.empty or not indicators or 'rsi' not in indicators:
            return signals
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Check if we have a valid range
            if (self.current_range['support'] is None or 
                self.current_range['resistance'] is None or 
                self.current_range['strength'] < self.parameters['range_strength_threshold']):
                logger.debug("No valid trading range detected")
                return signals
            
            support = self.current_range['support']
            resistance = self.current_range['resistance']
            
            # Exit signals
            for position in self.positions:
                if position.status == PositionStatus.OPEN:
                    # Check stop loss
                    if position.direction == 'long' and current_price <= position.stop_loss:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Stop loss triggered for long position {position.position_id}")
                    
                    elif position.direction == 'short' and current_price >= position.stop_loss:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Stop loss triggered for short position {position.position_id}")
                    
                    # Check take profit
                    elif position.direction == 'long' and current_price >= position.take_profit:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Take profit reached for long position {position.position_id}")
                    
                    elif position.direction == 'short' and current_price <= position.take_profit:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Take profit reached for short position {position.position_id}")
                    
                    # Exit when price breaks out of the range
                    if position.direction == 'long' and current_price > resistance:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Range breakout exit for long position {position.position_id}")
                    
                    elif position.direction == 'short' and current_price < support:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Range breakout exit for short position {position.position_id}")
            
            # Don't generate entry signals if we already have max positions
            if len([p for p in self.positions if p.status == PositionStatus.OPEN]) >= self.parameters['max_positions']:
                return signals
            
            # Entry signals
            rsi = indicators['rsi'].iloc[-1]
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            range_position = indicators.get('range_position', 0.5)  # Default to middle of range
            
            # Entry buffer (don't buy exactly at support/resistance)
            entry_buffer = self.parameters['entry_buffer_pct']
            
            # Long entry conditions (near support, oversold)
            if (current_price <= support * (1 + entry_buffer) and 
                rsi < self.parameters['rsi_lower'] and 
                current_price < bb_lower and 
                range_position < 0.3):  # Bottom 30% of range
                
                signals['entry'] = True
                signals['direction'] = 'long'
                
                # Calculate strength based on how close to support and how oversold
                price_factor = 1 - (current_price - support) / support
                rsi_factor = 1 - (rsi / self.parameters['rsi_lower'])
                signals['strength'] = min(1.0, (price_factor + rsi_factor) / 2)
                
                # Calculate stop loss and take profit
                stop_buffer = support * 0.98  # 2% below support
                take_profit = support + (resistance - support) * 0.8  # 80% to resistance
                
                signals['stop_loss'] = stop_buffer
                signals['take_profit'] = take_profit
                
                logger.info(f"Long range entry signal generated at {current_price:.2f}, "
                           f"Support: {support:.2f}, SL: {stop_buffer:.2f}, TP: {take_profit:.2f}")
            
            # Short entry conditions (near resistance, overbought)
            elif (current_price >= resistance * (1 - entry_buffer) and 
                  rsi > self.parameters['rsi_upper'] and 
                  current_price > bb_upper and 
                  range_position > 0.7):  # Top 30% of range
                
                signals['entry'] = True
                signals['direction'] = 'short'
                
                # Calculate strength based on how close to resistance and how overbought
                price_factor = 1 - (resistance - current_price) / resistance
                rsi_factor = (rsi - self.parameters['rsi_upper']) / (100 - self.parameters['rsi_upper'])
                signals['strength'] = min(1.0, (price_factor + rsi_factor) / 2)
                
                # Calculate stop loss and take profit
                stop_buffer = resistance * 1.02  # 2% above resistance
                take_profit = resistance - (resistance - support) * 0.8  # 80% to support
                
                signals['stop_loss'] = stop_buffer
                signals['take_profit'] = take_profit
                
                logger.info(f"Short range entry signal generated at {current_price:.2f}, "
                           f"Resistance: {resistance:.2f}, SL: {stop_buffer:.2f}, TP: {take_profit:.2f}")
                
        except Exception as e:
            logger.error(f"Error generating range trading signals: {str(e)}")
        
        return signals
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate position size based on risk parameters and account constraints.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in number of shares
        """
        if data.empty:
            return 0
        
        try:
            # Get current price
            current_price = data['close'].iloc[-1]
            
            # Get stop loss from signals
            if not self.signals or 'stop_loss' not in self.signals:
                return 0
                
            stop_loss = self.signals['stop_loss']
            
            # Calculate risk amount based on account equity and risk per trade
            account_equity = self.get_account_equity()
            risk_amount = account_equity * self.parameters['risk_per_trade']
            
            # Calculate stop loss distance
            stop_loss_distance = abs(current_price - stop_loss)
            
            # Avoid division by zero
            if stop_loss_distance == 0:
                return 0
            
            # Calculate position size based on risk (risk amount / stop loss distance)
            position_size = risk_amount / stop_loss_distance
            
            # Convert to number of shares
            shares = position_size / current_price
            
            # Apply account aware constraints
            max_shares, max_notional = self.calculate_max_position_size(
                price=current_price,
                is_day_trade=self.parameters['is_day_trade'],
                risk_percent=self.parameters['risk_per_trade']
            )
            
            # Use the smaller of our calculated position sizes
            position_size = min(shares, max_shares)
            
            # Check if position would exceed max position size as percentage of account
            max_position_value = account_equity * self.parameters['max_position_size_pct']
            max_position_shares = max_position_value / current_price
            position_size = min(position_size, max_position_shares)
            
            logger.info(f"Calculated position size: {position_size:.2f} shares (${position_size * current_price:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0
    
    def _execute_signals(self) -> None:
        """
        Execute the trading signals with account awareness checks.
        
        This method ensures we check for:
        1. Account balance requirements
        2. Position size limits
        3. Range validity and strength
        """
        # Ensure account status is up to date
        self.check_account_status()
        
        # Verify account has sufficient buying power
        buying_power = self.get_buying_power(day_trade=self.parameters['is_day_trade'])
        if buying_power <= 0:
            logger.warning("Insufficient buying power for range trading strategy")
            return
        
        # Verify that we have a valid range
        if (self.current_range['support'] is None or 
            self.current_range['resistance'] is None or 
            self.current_range['strength'] < self.parameters['range_strength_threshold']):
            logger.debug("No valid trading range detected, skipping execution")
            return
        
        # Execute exit signals first
        if self.signals.get('exit', False):
            for position_id in self.signals.get('positions_to_close', []):
                self._close_position(position_id)
                logger.info(f"Closed position {position_id}")
                
        # Execute entry signals
        if self.signals.get('entry', False):
            direction = self.signals.get('direction')
            if not direction:
                return
                
            # Calculate position size
            position_size = self.calculate_position_size(direction, self.market_data, self.indicators)
            
            # Validate trade size
            symbol = self.session.symbol
            current_price = self.market_data['close'].iloc[-1] if not self.market_data.empty else 0
            
            if not self.validate_trade_size(symbol, position_size, current_price, is_day_trade=self.parameters['is_day_trade']):
                logger.warning(f"Trade validation failed for {symbol}, size: {position_size}")
                return
                
            # Open position if size > 0
            if position_size > 0:
                stop_loss = self.signals.get('stop_loss')
                take_profit = self.signals.get('take_profit')
                
                # Only execute if we have valid stop loss and take profit
                if stop_loss and take_profit:
                    self._open_position(direction, position_size)
                    logger.info(f"Opened {direction} position of {position_size:.2f} shares")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the range trading strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            'ranging': 0.95,              # Excellent in range-bound markets
            'trending': 0.30,             # Poor in trending markets
            'strong_trend': 0.10,         # Very poor in strong trends
            'choppy': 0.75,               # Good in choppy markets
            'low_volatility': 0.80,       # Very good in low volatility
            'high_volatility': 0.40,      # Poor in high volatility
            'extreme_volatility': 0.20,   # Very poor in extreme volatility
            'bullish': 0.50,              # Moderate in general bullish markets
            'bearish': 0.50,              # Moderate in general bearish markets
            'sector_rotation': 0.60,      # Above average during sector rotations
            'earnings_season': 0.40,      # Below average during earnings season
            'news_driven': 0.30,          # Poor in news-driven markets
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.50)
