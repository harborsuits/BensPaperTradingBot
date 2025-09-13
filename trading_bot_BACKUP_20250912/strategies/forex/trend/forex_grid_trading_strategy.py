#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Grid Trading Strategy

This module implements a grid trading strategy for forex markets,
placing multiple buy and sell orders at predetermined price levels,
profiting from price movements in both directions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from trading_bot.strategies.factory.strategy_registry import register_strategy

from trading_bot.strategies.base.forex_base import ForexBaseStrategy, ForexSession
from trading_bot.strategies.strategy_template import Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.event_system import EventBus
from trading_bot.event_system.event_types import EventType, Event

logger = logging.getLogger(__name__)


@register_strategy({
    'asset_class': 'forex',
    'strategy_type': 'trend',
    'compatible_market_regimes': ['all_weather'],
    'timeframe': 'daily',
    'regime_compatibility_scores': {}
})
class ForexGridTradingStrategy(ForexBaseStrategy):
    """Grid trading strategy for forex markets.
    
    This strategy implements a price grid system by:
    1. Defining upper and lower boundaries of a trading range
    2. Creating a grid of evenly-spaced price levels within the range
    3. Placing buy orders at support levels and sell orders at resistance levels
    4. Managing multiple positions simultaneously with set profit targets
    """
    
    # Default strategy parameters
    DEFAULT_PARAMETERS = {
        # Grid setup parameters
        'grid_levels': 10,                  # Number of levels in the grid
        'grid_spacing_pips': 20,            # Spacing between grid levels in pips
        'dynamic_grid_sizing': True,        # Adjust grid spacing based on volatility
        'atr_grid_factor': 0.5,             # Multiplier for ATR-based grid spacing
        
        # Grid boundaries
        'use_auto_boundaries': True,        # Automatically determine grid boundaries
        'boundary_lookback_periods': 20,    # Periods to analyze for auto-boundaries
        'manual_upper_boundary': None,      # Manual upper boundary (price)
        'manual_lower_boundary': None,      # Manual lower boundary (price)
        'boundary_padding_pips': 10,        # Extra padding beyond detected boundaries
        
        # Trade management
        'profit_pips_per_grid': 15,         # Target profit per grid level in pips
        'max_active_positions': 15,         # Maximum number of positions to hold
        'adjust_position_size': True,       # Scale position size based on grid level
        'position_size_factor': 0.8,        # Position size multiplier per level from center
        
        # Grid filters
        'use_trend_filter': True,           # Filter grid operation based on trend
        'trend_indicator': 'ema',           # Indicator for trend filter ('ema', 'adx', 'macd')
        'trend_period': 50,                 # Period for trend indicator
        'min_adx_threshold': 20,            # Minimum ADX for trend consideration
        
        # Risk management
        'max_grid_exposure': 0.3,           # Maximum account exposure to grid (0-1)
        'grid_reset_drawdown': 0.05,        # Drawdown level to reset the grid (5%)
        'use_global_stop_loss': True,       # Use a global stop loss for the entire grid
        'global_stop_atr_multiple': 3.0,    # Global stop loss as ATR multiple from entry
        
        # Volatility parameters
        'atr_period': 14,                   # ATR period for volatility measurement
        'min_volatility_percentile': 20,    # Minimum volatility (ATR percentile)
        'max_volatility_percentile': 80,    # Maximum volatility (ATR percentile)
        
        # Session preferences
        'trading_sessions': [ForexSession.LONDON, ForexSession.NEWYORK],
    }
    
    def __init__(self, name: str = "Forex Grid Trading", 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the forex grid trading strategy.
        
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
        self.active_grids = {}  # Active grid configurations by symbol
        self.active_positions = {}  # Active positions within grids
        self.last_grid_updates = {}  # Last update times for each grid
        self.current_signals = {}  # Current signals
        
        logger.info(f"Initialized {self.name} strategy")
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: datetime) -> Dict[str, Signal]:
        """
        Generate trade signals based on grid levels.
        
        Args:
            data: Dictionary mapping symbols to OHLCV DataFrames
            current_time: Current timestamp
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        signals = {}
        
        for symbol, ohlcv in data.items():
            # Skip if we don't have enough data
            required_lookback = max(
                self.parameters['boundary_lookback_periods'],
                self.parameters['trend_period'],
                self.parameters['atr_period']
            ) + 10
            
            if len(ohlcv) < required_lookback:
                logger.debug(f"Insufficient data for {symbol}, skipping")
                continue
            
            # Calculate indicators
            indicators = self._calculate_grid_indicators(ohlcv)
            
            # First, check if we need to initialize or update the grid
            self._update_grid(symbol, ohlcv, indicators, current_time)
            
            # Generate signals based on current price and grid levels
            grid_signals = self._evaluate_grid_signals(symbol, ohlcv, indicators, current_time)
            
            if grid_signals:
                for signal in grid_signals:
                    signals[f"{symbol}_{signal.metadata['grid_level']}"] = signal
                    # Also store in current signals
                    self.current_signals[f"{symbol}_{signal.metadata['grid_level']}"] = signal
        
        # Publish event with active grids
        if self.active_grids:
            event_data = {
                'strategy_name': self.name,
                'active_grids': self.active_grids,
                'active_positions': self.active_positions,
                'timestamp': current_time.isoformat()
            }
            
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,
                source=self.name,
                data=event_data,
                metadata={'strategy_type': 'forex', 'category': 'grid_trading'}
            )
            self.event_bus.publish(event)
        
        return signals
    
    def _calculate_grid_indicators(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for grid trading.
        
        Args:
            ohlcv: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate ATR for volatility measurement and dynamic grid sizing
        atr_period = self.parameters['atr_period']
        high_low = ohlcv['high'] - ohlcv['low']
        high_close = np.abs(ohlcv['high'] - ohlcv['close'].shift())
        low_close = np.abs(ohlcv['low'] - ohlcv['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        indicators['atr'] = true_range.rolling(atr_period).mean()
        
        # Calculate volatility percentile
        lookback = min(100, len(ohlcv))
        indicators['atr_percentile'] = indicators['atr'].rolling(lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=True
        )
        
        # Calculate trend indicator based on parameter
        trend_indicator = self.parameters['trend_indicator']
        trend_period = self.parameters['trend_period']
        
        if trend_indicator == 'ema':
            indicators['trend_line'] = ohlcv['close'].ewm(span=trend_period, adjust=False).mean()
            indicators['trend_direction'] = np.where(
                indicators['trend_line'].diff(5) > 0, 1, 
                np.where(indicators['trend_line'].diff(5) < 0, -1, 0)
            )
        elif trend_indicator == 'adx':
            # Calculate directional movement
            up_move = ohlcv['high'] - ohlcv['high'].shift()
            down_move = ohlcv['low'].shift() - ohlcv['low']
            
            pos_dm = up_move.copy()
            pos_dm[up_move <= down_move] = 0
            pos_dm[up_move <= 0] = 0
            
            neg_dm = down_move.copy()
            neg_dm[down_move <= up_move] = 0
            neg_dm[down_move <= 0] = 0
            
            # Smooth the indicators
            tr_smooth = true_range.rolling(window=trend_period).mean()
            pos_di = 100 * (pos_dm.rolling(window=trend_period).mean() / tr_smooth)
            neg_di = 100 * (neg_dm.rolling(window=trend_period).mean() / tr_smooth)
            
            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
            indicators['adx'] = dx.rolling(window=trend_period).mean()
            indicators['trend_strength'] = indicators['adx']
            indicators['trend_direction'] = np.where(pos_di > neg_di, 1, -1)
        elif trend_indicator == 'macd':
            # Calculate MACD
            ema_fast = ohlcv['close'].ewm(span=12, adjust=False).mean()
            ema_slow = ohlcv['close'].ewm(span=26, adjust=False).mean()
            indicators['macd'] = ema_fast - ema_slow
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            indicators['trend_direction'] = np.where(
                indicators['macd'] > indicators['macd_signal'], 1, -1
            )
        
        # Calculate support and resistance levels
        lookback = self.parameters['boundary_lookback_periods']
        indicators['upper_boundary'] = ohlcv['high'].rolling(lookback).max()
        indicators['lower_boundary'] = ohlcv['low'].rolling(lookback).min()
        
        return indicators
    
    def _update_grid(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, Any], current_time: datetime) -> None:
        """
        Initialize or update the grid for a symbol.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            current_time: Current timestamp
        """
        # Check if we need to initialize or update
        last_update = self.last_grid_updates.get(symbol)
        update_needed = (
            symbol not in self.active_grids or
            last_update is None or
            (current_time - last_update).total_seconds() > 14400  # Update every 4 hours
        )
        
        if not update_needed:
            return
        
        logger.info(f"Updating grid for {symbol}")
        
        # Get current price and ATR
        current_price = ohlcv['close'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        
        # Check volatility conditions
        atr_percentile = indicators['atr_percentile'].iloc[-1]
        min_vol = self.parameters['min_volatility_percentile']
        max_vol = self.parameters['max_volatility_percentile']
        
        if atr_percentile < min_vol:
            logger.debug(f"Volatility too low for {symbol} ({atr_percentile:.1f}%), skipping grid update")
            return
            
        if atr_percentile > max_vol:
            logger.debug(f"Volatility too high for {symbol} ({atr_percentile:.1f}%), skipping grid update")
            return
        
        # Determine grid boundaries
        if self.parameters['use_auto_boundaries']:
            # Use detected boundaries
            upper_boundary = indicators['upper_boundary'].iloc[-1]
            lower_boundary = indicators['lower_boundary'].iloc[-1]
            
            # Add padding
            padding_pips = self.parameters['boundary_padding_pips'] * self.parameters['pip_value']
            upper_boundary += padding_pips
            lower_boundary -= padding_pips
        else:
            # Use manual boundaries if provided
            upper_boundary = self.parameters['manual_upper_boundary']
            lower_boundary = self.parameters['manual_lower_boundary']
            
            # If manual boundaries not provided, use current price ± range
            if upper_boundary is None or lower_boundary is None:
                range_size = current_atr * 4  # 4 ATRs range
                upper_boundary = current_price + range_size / 2
                lower_boundary = current_price - range_size / 2
        
        # Calculate grid spacing
        if self.parameters['dynamic_grid_sizing']:
            # Base grid spacing on ATR
            atr_factor = self.parameters['atr_grid_factor']
            grid_spacing = current_atr * atr_factor
        else:
            # Use fixed grid spacing in pips
            grid_spacing = self.parameters['grid_spacing_pips'] * self.parameters['pip_value']
        
        # Ensure grid fits between boundaries
        grid_levels = self.parameters['grid_levels']
        total_grid_range = upper_boundary - lower_boundary
        min_grid_spacing = total_grid_range / (grid_levels - 1)
        
        # Use the larger of calculated or minimum spacing
        grid_spacing = max(grid_spacing, min_grid_spacing)
        
        # Calculate grid levels
        grid_prices = []
        for i in range(grid_levels):
            level_price = lower_boundary + (i * grid_spacing)
            # Ensure we don't exceed upper boundary
            if level_price <= upper_boundary:
                grid_prices.append(level_price)
        
        # Sort grid from lowest to highest price
        grid_prices.sort()
        
        # Create grid configuration
        grid_config = {
            'upper_boundary': upper_boundary,
            'lower_boundary': lower_boundary,
            'grid_spacing': grid_spacing,
            'grid_prices': grid_prices,
            'current_price': current_price,
            'atr': current_atr,
            'last_update': current_time.isoformat(),
            'grid_levels': len(grid_prices)
        }
        
        # Store grid configuration
        self.active_grids[symbol] = grid_config
        self.last_grid_updates[symbol] = current_time
        
        logger.info(f"Updated {symbol} grid with {len(grid_prices)} levels")
    
    def _evaluate_grid_signals(self, symbol: str, ohlcv: pd.DataFrame, indicators: Dict[str, Any], 
                            current_time: datetime) -> List[Signal]:
        """
        Evaluate price position relative to grid levels and generate signals.
        
        Args:
            symbol: Currency pair symbol
            ohlcv: DataFrame with OHLCV price data
            indicators: Dictionary of technical indicators
            current_time: Current timestamp
            
        Returns:
            List of Signal objects for grid levels
        """
        # Check if we have an active grid for this symbol
        if symbol not in self.active_grids:
            return []
            
        grid_config = self.active_grids[symbol]
        grid_prices = grid_config['grid_prices']
        current_price = ohlcv['close'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        
        # Count active positions for this symbol
        active_count = sum(1 for pos in self.active_positions.values() 
                         if pos['symbol'] == symbol)
        
        max_positions = self.parameters['max_active_positions']
        if active_count >= max_positions:
            logger.debug(f"Maximum positions ({max_positions}) reached for {symbol}")
            return []
        
        # Check trend filter if enabled
        if self.parameters['use_trend_filter']:
            trend_direction = indicators.get('trend_direction', [0])[-1]
            
            # If using ADX, check strength
            if self.parameters['trend_indicator'] == 'adx':
                adx_value = indicators.get('adx', [0])[-1]
                min_adx = self.parameters['min_adx_threshold']
                
                if adx_value < min_adx:
                    # Weak trend, consider as ranging
                    trend_direction = 0
        else:
            # No trend filter, allow both directions
            trend_direction = 0
        
        signals = []
        
        # Find the two grid levels that surround the current price
        for i, price_level in enumerate(grid_prices):
            # Skip if we're at the last level
            if i == len(grid_prices) - 1:
                continue
                
            next_level = grid_prices[i + 1]
            
            # Check if price is between these levels
            is_between_levels = price_level <= current_price <= next_level
            
            if not is_between_levels:
                continue
                
            # Calculate distances to levels as percentage of grid spacing
            grid_spacing = grid_config['grid_spacing']
            distance_to_lower = (current_price - price_level) / grid_spacing
            distance_to_upper = (next_level - current_price) / grid_spacing
            
            # Determine which level(s) to activate based on trend and distance
            levels_to_activate = []
            
            # In an uptrend, prefer buy at support
            if trend_direction > 0:
                # Buy at lower level if close enough
                if distance_to_lower < 0.3:
                    levels_to_activate.append((i, 1))  # (level_index, direction)
            
            # In a downtrend, prefer sell at resistance
            elif trend_direction < 0:
                # Sell at upper level if close enough
                if distance_to_upper < 0.3:
                    levels_to_activate.append((i + 1, -1))  # (level_index, direction)
            
            # In ranging or unknown, activate both levels
            else:
                # Buy at lower level if close enough
                if distance_to_lower < 0.3:
                    levels_to_activate.append((i, 1))
                
                # Sell at upper level if close enough
                if distance_to_upper < 0.3:
                    levels_to_activate.append((i + 1, -1))
            
            # Generate signals for levels to activate
            for level_index, direction in levels_to_activate:
                # Skip if position already exists for this level
                level_key = f"{symbol}_level_{level_index}"
                if level_key in self.active_positions:
                    continue
                
                # Grid level price
                level_price = grid_prices[level_index]
                
                # Calculate take profit
                profit_pips = self.parameters['profit_pips_per_grid']
                take_profit = level_price + (direction * profit_pips * self.parameters['pip_value'])
                
                # Calculate stop loss if global stop loss is enabled
                if self.parameters['use_global_stop_loss']:
                    stop_multiple = self.parameters['global_stop_atr_multiple']
                    stop_loss = level_price - (direction * current_atr * stop_multiple)
                else:
                    # No stop loss for individual grid positions
                    stop_loss = None
                
                # Calculate confidence based on trend alignment and level position
                base_confidence = 0.5
                
                # Higher confidence if trend agrees with direction
                trend_alignment = trend_direction * direction
                trend_boost = 0.2 if trend_alignment > 0 else 0
                
                # Higher confidence for levels closer to boundaries (mean reversion)
                level_position = level_index / (len(grid_prices) - 1)  # 0 to 1
                boundary_proximity = min(level_position, 1 - level_position) * 2  # 0 at boundaries, 1 at center
                boundary_boost = 0.2 * (1 - boundary_proximity)  # Higher at boundaries
                
                # Combine confidence factors
                confidence = min(0.95, base_confidence + trend_boost + boundary_boost)
                
                # Calculate position size scaling if enabled
                position_scale = 1.0
                if self.parameters['adjust_position_size']:
                    # Center level index
                    center_index = (len(grid_prices) - 1) / 2
                    # Distance from center (0 to 1)
                    distance_from_center = abs(level_index - center_index) / center_index
                    # Scale factor (higher away from center)
                    position_scale = 1.0 + (distance_from_center * (1.0 - self.parameters['position_size_factor']))
                
                # Create signal
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.LIMIT,
                    direction=direction,
                    confidence=confidence,
                    entry_price=level_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    size_factor=position_scale,
                    metadata={
                        'strategy': self.name,
                        'setup_type': 'grid_level',
                        'grid_level': level_index,
                        'grid_price': level_price,
                        'grid_direction': 'buy' if direction > 0 else 'sell',
                        'grid_confidence': confidence,
                        'position_scale': position_scale,
                        'current_price': current_price,
                        'level_key': level_key
                    }
                )
                
                signals.append(signal)
                
                # Register this as a pending position
                self.active_positions[level_key] = {
                    'symbol': symbol,
                    'level_index': level_index,
                    'direction': direction,
                    'entry_price': level_price,
                    'take_profit': take_profit,
                    'stop_loss': stop_loss,
                    'status': 'pending',
                    'timestamp': current_time.isoformat()
                }
        
        return signals
    
    def get_compatibility_score(self, market_regime: MarketRegime) -> float:
        """
        Calculate compatibility score with the given market regime.
        
        Args:
            market_regime: The current market regime
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Grid trading works well in ranging markets but can also profit in trending markets
        compatibility_map = {
            # Best regimes for grid trading
            MarketRegime.RANGING: 0.95,        # Excellent for grid trading
            MarketRegime.CHOPPY: 0.85,         # Very good for grid trading
            
            # Medium compatibility
            MarketRegime.TRENDING_UP: 0.70,    # Good if grid aligned with trend
            MarketRegime.TRENDING_DOWN: 0.70,  # Good if grid aligned with trend
            
            # Worst regimes for grid trading
            MarketRegime.VOLATILE_BREAKOUT: 0.50,  # Can work but higher risk
            MarketRegime.VOLATILE_REVERSAL: 0.55,  # Can work but higher risk
            
            # Default for unknown regimes
            MarketRegime.UNKNOWN: 0.70         # Generally viable
        }
        
        # Return the compatibility score or default to 0.7 if regime unknown
        return compatibility_map.get(market_regime, 0.7)
    
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
        if market_regime in [MarketRegime.RANGING, MarketRegime.CHOPPY]:
            # For ranging markets, maximize grid coverage
            optimized_params['grid_levels'] = 12               # More levels
            optimized_params['dynamic_grid_sizing'] = False    # Fixed grid size
            optimized_params['grid_spacing_pips'] = 15         # Tighter spacing
            optimized_params['use_trend_filter'] = False       # No trend filter
            optimized_params['max_active_positions'] = 20      # More positions
            optimized_params['adjust_position_size'] = True    # Scale positions
            optimized_params['use_global_stop_loss'] = False   # No global stop
            
        elif market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # For trending markets, align with trend and wider spacing
            optimized_params['grid_levels'] = 8                # Fewer levels
            optimized_params['dynamic_grid_sizing'] = True     # Dynamic grid size
            optimized_params['atr_grid_factor'] = 0.7          # Wider spacing
            optimized_params['use_trend_filter'] = True        # Use trend filter
            optimized_params['trend_indicator'] = 'ema'        # EMA for trends
            optimized_params['max_active_positions'] = 12      # Moderate positions
            optimized_params['adjust_position_size'] = False   # Equal sizing
            optimized_params['use_global_stop_loss'] = True    # Use global stop
            
        elif market_regime in [MarketRegime.VOLATILE_BREAKOUT, MarketRegime.VOLATILE_REVERSAL]:
            # For volatile markets, be conservative
            optimized_params['grid_levels'] = 6                # Few levels
            optimized_params['dynamic_grid_sizing'] = True     # Dynamic grid size
            optimized_params['atr_grid_factor'] = 1.0          # Very wide spacing
            optimized_params['use_trend_filter'] = True        # Use trend filter
            optimized_params['trend_indicator'] = 'adx'        # ADX for strength
            optimized_params['min_adx_threshold'] = 25         # Higher threshold
            optimized_params['max_active_positions'] = 8       # Fewer positions
            optimized_params['adjust_position_size'] = True    # Scale positions
            optimized_params['position_size_factor'] = 0.6     # More scaling
            optimized_params['use_global_stop_loss'] = True    # Use global stop
            optimized_params['global_stop_atr_multiple'] = 2.0 # Tighter stop
        
        # Log the optimization
        logger.info(f"Optimized {self.name} for {market_regime} regime")
        
        return optimized_params
