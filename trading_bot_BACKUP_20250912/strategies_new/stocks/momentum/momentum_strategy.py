#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Momentum Trading Strategy

This module implements a stock momentum trading strategy that aims to capitalize on the 
continuation of existing market trends. The strategy is account-aware, ensuring it 
complies with account balance requirements, regulatory constraints, and risk management.
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
    name="MomentumStrategy",
    market_type="stocks",
    description="A strategy that capitalizes on the continuation of existing market trends by identifying stocks with strong momentum",
    timeframes=["1d", "4h", "1h"],
    parameters={
        "momentum_period": {"description": "Lookback period for momentum calculation", "type": "int"},
        "roc_threshold": {"description": "Rate of Change threshold for momentum confirmation", "type": "float"},
        "atr_multiplier": {"description": "ATR multiplier for stop loss calculation", "type": "float"},
        "max_positions": {"description": "Maximum number of concurrent positions", "type": "int"}
    }
)
class MomentumStrategy(StocksBaseStrategy, AccountAwareMixin):
    """
    Stock Momentum Trading Strategy
    
    This strategy:
    1. Identifies stocks with strong directional momentum using rate of change (ROC) and ADX
    2. Confirms momentum with volume analysis and moving average relationships
    3. Uses careful position sizing based on volatility and account constraints
    4. Adapts to different market regimes (trending, choppy, volatile)
    5. Incorporates account awareness for regulatory compliance and risk management
    
    The momentum strategy works best in trending markets and is designed for
    medium-term trades ranging from days to weeks.
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Momentum strategy.
        
        Args:
            session: Stock trading session with symbol, timeframe, etc.
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize parent classes
        StocksBaseStrategy.__init__(self, session, data_pipeline, parameters)
        AccountAwareMixin.__init__(self)
        
        # Default parameters for momentum trading
        default_params = {
            # Strategy identification
            'strategy_name': 'Momentum',
            'strategy_id': 'momentum',
            'is_day_trade': False,  # Typically not day trading
            
            # Momentum parameters
            'momentum_period': 20,
            'roc_threshold': 3.0,  # 3% minimum rate of change for entry
            'rsi_period': 14,
            'rsi_threshold': 50,  # Must be above 50 for uptrend momentum
            'adx_period': 14,
            'adx_threshold': 20,  # Minimum ADX for trend strength
            
            # Moving averages
            'ema_short': 20,
            'ema_medium': 50,
            'ema_long': 200,
            
            # Volume confirmation
            'volume_threshold': 1.5,  # Volume should be above average
            'volume_lookback': 20,
            
            # Trade execution
            'entry_delay': 1,  # Wait 1 bar after signal before entry
            'max_positions': 5,  # Maximum number of concurrent positions
            'trail_stop_pct': 0.08,  # 8% trailing stop
            'atr_multiplier': 2.5,  # ATR multiplier for stop loss
            'atr_period': 14,  # ATR calculation period
            'profit_target_multiplier': 3.0,  # Risk:reward ratio
            
            # Risk management
            'risk_per_trade': 0.01,  # 1% risk per trade
            'max_sector_exposure': 0.25,  # Maximum 25% exposure to a single sector
            'max_correlated_positions': 3,  # Maximum number of positions with high correlation
            'max_position_size_pct': 0.10,  # Maximum 10% of account in a single position
        }
        
        # Update with user-provided parameters
        if parameters:
            default_params.update(parameters)
        self.parameters = default_params
        
        # Strategy state
        self.momentum_scores = {}
        self.sector_exposure = {}
        self.is_active = True
        
        logger.info(f"Initialized {self.name} for {session.symbol} on {session.timeframe}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Momentum strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < max(self.parameters['momentum_period'], 
                                        self.parameters['adx_period'],
                                        self.parameters['ema_long']):
            return indicators
        
        try:
            # Calculate Rate of Change (ROC)
            indicators['roc'] = 100 * (data['close'] / data['close'].shift(self.parameters['momentum_period']) - 1)
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.parameters['rsi_period']).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.parameters['rsi_period']).mean()
            
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate EMAs
            indicators['ema_short'] = data['close'].ewm(span=self.parameters['ema_short'], adjust=False).mean()
            indicators['ema_medium'] = data['close'].ewm(span=self.parameters['ema_medium'], adjust=False).mean()
            indicators['ema_long'] = data['close'].ewm(span=self.parameters['ema_long'], adjust=False).mean()
            
            # Calculate ADX (Average Directional Index)
            # True Range
            data['tr0'] = abs(data['high'] - data['low'])
            data['tr1'] = abs(data['high'] - data['close'].shift())
            data['tr2'] = abs(data['low'] - data['close'].shift())
            data['tr'] = data[['tr0', 'tr1', 'tr2']].max(axis=1)
            
            # Directional Movement
            data['up_move'] = data['high'] - data['high'].shift()
            data['down_move'] = data['low'].shift() - data['low']
            
            data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
            data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
            
            # Smoothed Directional Indicators
            adx_period = self.parameters['adx_period']
            data['tr_smoothed'] = data['tr'].rolling(window=adx_period).mean()
            data['plus_di'] = 100 * (data['plus_dm'].rolling(window=adx_period).mean() / data['tr_smoothed'])
            data['minus_di'] = 100 * (data['minus_dm'].rolling(window=adx_period).mean() / data['tr_smoothed'])
            
            # Directional Index
            data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
            
            # ADX
            indicators['adx'] = data['dx'].rolling(window=adx_period).mean()
            indicators['plus_di'] = data['plus_di']
            indicators['minus_di'] = data['minus_di']
            
            # Calculate ATR for volatility assessment
            indicators['atr'] = data['tr'].rolling(window=self.parameters['atr_period']).mean()
            
            # Volume analysis
            indicators['volume_sma'] = data['volume'].rolling(window=self.parameters['volume_lookback']).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
            
            # Trend strength metrics
            indicators['trend_strength'] = self._calculate_trend_strength(data, indicators)
            
            # Calculate momentum score (0-100)
            indicators['momentum_score'] = self._calculate_momentum_score(data, indicators)
            
            # Trend direction
            indicators['trend_direction'] = np.where(
                indicators['ema_short'] > indicators['ema_medium'], 
                'bullish', 
                'bearish'
            )
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {str(e)}")
        
        return indicators
    
    def _calculate_trend_strength(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> pd.Series:
        """
        Calculate trend strength score combining multiple indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Series with trend strength scores (0-100)
        """
        trend_strength = pd.Series(50.0, index=data.index)  # Neutral starting point
        
        # ADX contribution (0-30 points)
        if 'adx' in indicators:
            adx_contrib = indicators['adx'] * 0.75  # Scale to get max ~22.5 points
            adx_contrib = adx_contrib.clip(0, 30)
            trend_strength += adx_contrib
        
        # Moving average alignment (0-30 points)
        if all(k in indicators for k in ['ema_short', 'ema_medium', 'ema_long']):
            # Initialize alignment score
            ma_alignment = pd.Series(0.0, index=data.index)
            
            # Short above medium
            short_above_medium = np.where(
                indicators['ema_short'] > indicators['ema_medium'],
                10.0,  # Strong bullish alignment
                0.0    # Not aligned
            )
            
            # Medium above long
            medium_above_long = np.where(
                indicators['ema_medium'] > indicators['ema_long'],
                10.0,  # Strong bullish alignment
                0.0    # Not aligned
            )
            
            # Short/medium slope (rising)
            short_rising = np.where(
                indicators['ema_short'] > indicators['ema_short'].shift(5),
                10.0,  # Rising
                0.0    # Falling
            )
            
            ma_alignment = pd.Series(short_above_medium, index=data.index) + \
                           pd.Series(medium_above_long, index=data.index) + \
                           pd.Series(short_rising, index=data.index)
            
            trend_strength += ma_alignment
        
        # Volume confirmation (0-20 points)
        if 'volume_ratio' in indicators:
            volume_contrib = (indicators['volume_ratio'] - 1) * 10
            volume_contrib = volume_contrib.clip(0, 20)
            trend_strength += volume_contrib
        
        # Momentum confirmation (0-20 points)
        if 'roc' in indicators:
            roc_contrib = indicators['roc'] * 2
            roc_contrib = roc_contrib.clip(0, 20)
            trend_strength += roc_contrib
        
        # Clip final values to 0-100 range
        return trend_strength.clip(0, 100)
    
    def _calculate_momentum_score(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> pd.Series:
        """
        Calculate a comprehensive momentum score.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Series with momentum scores (0-100)
        """
        momentum_score = pd.Series(0.0, index=data.index)
        
        # Rate of Change contribution (0-30 points)
        if 'roc' in indicators:
            roc_factor = indicators['roc'] * 3
            roc_factor = roc_factor.clip(0, 30)
            momentum_score += roc_factor
        
        # RSI contribution (0-20 points)
        if 'rsi' in indicators:
            # RSI above 50 indicates bullish momentum
            rsi_factor = (indicators['rsi'] - 50) * 0.4  # Scale to get max 20 points
            rsi_factor = rsi_factor.clip(0, 20)
            momentum_score += rsi_factor
        
        # Trend strength contribution (0-30 points)
        if 'trend_strength' in indicators:
            trend_factor = indicators['trend_strength'] * 0.3
            momentum_score += trend_factor
        
        # Volume confirmation (0-20 points)
        if 'volume_ratio' in indicators:
            volume_factor = (indicators['volume_ratio'] - 1) * 10
            volume_factor = volume_factor.clip(0, 20)
            momentum_score += volume_factor
        
        # Clip final values to 0-100 range
        return momentum_score.clip(0, 100)
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the Momentum strategy.
        
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
        
        if data.empty or not indicators or 'momentum_score' not in indicators:
            return signals
        
        try:
            current_price = data['close'].iloc[-1]
            
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
                    
                    # Exit on trend reversal
                    elif position.direction == 'long' and indicators['ema_short'].iloc[-1] < indicators['ema_medium'].iloc[-1]:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Trend reversal exit for long position {position.position_id}")
                    
                    elif position.direction == 'short' and indicators['ema_short'].iloc[-1] > indicators['ema_medium'].iloc[-1]:
                        signals['exit'] = True
                        signals['positions_to_close'].append(position.position_id)
                        logger.info(f"Trend reversal exit for short position {position.position_id}")
            
            # Don't generate entry signals if we already have max positions
            if len([p for p in self.positions if p.status == PositionStatus.OPEN]) >= self.parameters['max_positions']:
                return signals
            
            # Entry signals
            momentum_score = indicators['momentum_score'].iloc[-1]
            roc = indicators['roc'].iloc[-1]
            adx = indicators['adx'].iloc[-1]
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            
            # Long momentum entry conditions
            if (roc > self.parameters['roc_threshold'] and 
                adx > self.parameters['adx_threshold'] and
                indicators['ema_short'].iloc[-1] > indicators['ema_medium'].iloc[-1] and
                indicators['ema_medium'].iloc[-1] > indicators['ema_long'].iloc[-1] and
                indicators['plus_di'].iloc[-1] > indicators['minus_di'].iloc[-1] and
                volume_ratio > self.parameters['volume_threshold']):
                
                signals['entry'] = True
                signals['direction'] = 'long'
                signals['strength'] = min(1.0, momentum_score / 100)
                
                # Calculate stop loss using ATR
                atr = indicators['atr'].iloc[-1]
                stop_loss = current_price - (atr * self.parameters['atr_multiplier'])
                take_profit = current_price + (atr * self.parameters['atr_multiplier'] * self.parameters['profit_target_multiplier'])
                
                signals['stop_loss'] = stop_loss
                signals['take_profit'] = take_profit
                
                logger.info(f"Long momentum entry signal generated at {current_price:.2f}, "
                           f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            
            # Short momentum entry conditions
            elif (roc < -self.parameters['roc_threshold'] and 
                  adx > self.parameters['adx_threshold'] and
                  indicators['ema_short'].iloc[-1] < indicators['ema_medium'].iloc[-1] and
                  indicators['ema_medium'].iloc[-1] < indicators['ema_long'].iloc[-1] and
                  indicators['plus_di'].iloc[-1] < indicators['minus_di'].iloc[-1] and
                  volume_ratio > self.parameters['volume_threshold']):
                
                signals['entry'] = True
                signals['direction'] = 'short'
                signals['strength'] = min(1.0, momentum_score / 100)
                
                # Calculate stop loss using ATR
                atr = indicators['atr'].iloc[-1]
                stop_loss = current_price + (atr * self.parameters['atr_multiplier'])
                take_profit = current_price - (atr * self.parameters['atr_multiplier'] * self.parameters['profit_target_multiplier'])
                
                signals['stop_loss'] = stop_loss
                signals['take_profit'] = take_profit
                
                logger.info(f"Short momentum entry signal generated at {current_price:.2f}, "
                           f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                
        except Exception as e:
            logger.error(f"Error generating momentum signals: {str(e)}")
        
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
            
            # Calculate risk amount based on account equity and risk per trade
            account_equity = self.get_account_equity()
            risk_amount = account_equity * self.parameters['risk_per_trade']
            
            # Calculate stop loss distance using ATR
            atr = indicators['atr'].iloc[-1]
            stop_loss_distance = atr * self.parameters['atr_multiplier']
            
            # Calculate position size based on risk (risk amount / stop loss distance)
            position_size = risk_amount / stop_loss_distance
            
            # Convert to number of shares
            shares = position_size / current_price
            
            # Apply account aware constraints
            max_shares, max_notional = self.calculate_max_position_size(
                price=current_price,
                is_day_trade=False,  # Momentum is typically not day trading
                risk_percent=self.parameters['risk_per_trade']
            )
            
            # Use the smaller of our calculated position sizes
            position_size = min(shares, max_shares)
            
            # Check if position would exceed max position size as percentage of account
            max_position_value = account_equity * self.parameters['max_position_size_pct']
            max_position_shares = max_position_value / current_price
            position_size = min(position_size, max_position_shares)
            
            # Adjust for sector exposure
            symbol = self.session.symbol
            sector = self._get_symbol_sector(symbol)
            if sector:
                current_sector_exposure = self._get_sector_exposure(sector)
                remaining_sector_capacity = max(0, self.parameters['max_sector_exposure'] - current_sector_exposure)
                
                # Calculate how many shares would put us at the sector limit
                sector_limit_shares = (remaining_sector_capacity * account_equity) / current_price
                position_size = min(position_size, sector_limit_shares)
            
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
        3. Sector exposure limits
        """
        # Ensure account status is up to date
        self.check_account_status()
        
        # Verify account has sufficient buying power
        buying_power = self.get_buying_power()
        if buying_power <= 0:
            logger.warning("Insufficient buying power for momentum strategy")
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
            
            if not self.validate_trade_size(symbol, position_size, current_price, is_day_trade=False):
                logger.warning(f"Trade validation failed for {symbol}, size: {position_size}")
                return
                
            # Open position if size > 0
            if position_size > 0:
                stop_loss = self.signals.get('stop_loss')
                take_profit = self.signals.get('take_profit')
                
                # Only execute if we have valid stop loss and take profit
                if stop_loss and take_profit:
                    self._open_position(direction, position_size)
                    
                    # Update sector exposure
                    symbol = self.session.symbol
                    sector = self._get_symbol_sector(symbol)
                    if sector:
                        self._update_sector_exposure(sector, position_size * current_price)
                    
                    logger.info(f"Opened {direction} position of {position_size:.2f} shares")
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """
        Get the sector for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sector name or empty string if not found
        """
        # This would use an actual data source or API in production
        # Simplified placeholder mapping for demonstration
        sector_map = {
            'AAPL': 'technology',
            'MSFT': 'technology',
            'GOOGL': 'technology',
            'AMZN': 'consumer_cyclical',
            'TSLA': 'automotive',
            'JPM': 'financial',
            'BAC': 'financial',
            'WMT': 'consumer_defensive',
            'PFE': 'healthcare',
            'XOM': 'energy'
        }
        return sector_map.get(symbol, "")
    
    def _get_sector_exposure(self, sector: str) -> float:
        """
        Get current exposure to a sector as percentage of account.
        
        Args:
            sector: Sector name
            
        Returns:
            Current exposure as decimal (0.0-1.0)
        """
        return self.sector_exposure.get(sector, 0.0)
    
    def _update_sector_exposure(self, sector: str, position_value: float) -> None:
        """
        Update the sector exposure after opening a new position.
        
        Args:
            sector: Sector name
            position_value: Dollar value of the position
        """
        current_exposure = self.sector_exposure.get(sector, 0.0)
        account_equity = self.get_account_equity()
        
        if account_equity > 0:
            new_exposure = current_exposure + (position_value / account_equity)
            self.sector_exposure[sector] = new_exposure
            logger.info(f"Updated {sector} exposure to {new_exposure:.2%}")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the momentum strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            'trending': 0.95,              # Excellent in trending markets
            'strong_trend': 0.90,          # Excellent in strong trends
            'choppy': 0.40,                # Poor in choppy markets
            'ranging': 0.30,               # Poor in range-bound markets
            'low_volatility': 0.60,        # Moderate in low volatility
            'high_volatility': 0.75,       # Good in high volatility if trending
            'extreme_volatility': 0.40,    # Moderate in extreme volatility
            'bullish': 0.85,               # Very good in bullish markets
            'bearish': 0.70,               # Good in bearish markets (for shorts)
            'sector_rotation': 0.65,       # Good during sector rotations
            'earnings_season': 0.70,       # Good during earnings season
            'news_driven': 0.60,           # Moderate in news-driven markets
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.60)
