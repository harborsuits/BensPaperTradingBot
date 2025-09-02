#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalping Strategy

This module implements a stock scalping strategy that aims to profit from small price changes,
typically entering and exiting positions within minutes or seconds. The strategy is 
account-aware, ensuring it complies with Pattern Day Trader (PDT) rules and other 
account-related constraints.
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
    name="ScalpingStrategy",
    market_type="stocks",
    description="A fast-paced trading strategy that aims to profit from small price movements over very short time periods",
    timeframes=["1m", "3m", "5m"],
    parameters={
        "rsi_length": {"description": "Length of RSI indicator", "type": "int"},
        "rsi_overbought": {"description": "RSI overbought threshold", "type": "float"},
        "rsi_oversold": {"description": "RSI oversold threshold", "type": "float"},
        "volume_threshold": {"description": "Minimum volume threshold as percentage of average volume", "type": "float"},
        "stop_loss_ticks": {"description": "Number of ticks for stop loss", "type": "int"},
        "take_profit_ticks": {"description": "Number of ticks for take profit", "type": "int"}
    }
)
class ScalpingStrategy(StocksBaseStrategy, AccountAwareMixin):
    """
    Stock Scalping Strategy
    
    This strategy:
    1. Looks for quick price reversals at support/resistance levels
    2. Uses technical indicators like RSI and volume for entry signals
    3. Implements tight stop losses and take profits
    4. Maintains account awareness to respect PDT rules and risk limits
    5. Adapts parameters based on market volatility
    
    The strategy is designed for very short-term trades measured in minutes, 
    and is particularly careful about trade frequency to avoid PDT rule violations.
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Scalping strategy.
        
        Args:
            session: Stock trading session with symbol, timeframe, etc.
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize parent classes
        StocksBaseStrategy.__init__(self, session, data_pipeline, parameters)
        AccountAwareMixin.__init__(self)
        
        # Default parameters for scalping
        default_params = {
            # Strategy identification
            'strategy_name': 'Scalping',
            'strategy_id': 'scalping',
            'is_day_trade': True,  # Scalping is always day trading
            
            # Technical indicators
            'rsi_length': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'ema_fast': 9,
            'ema_slow': 20,
            'volume_threshold': 1.2,  # Minimum volume as multiple of average
            
            # Trade execution
            'tick_size': 0.01,  # Default tick size for most stocks
            'stop_loss_ticks': 10,  # Number of ticks for stop loss
            'take_profit_ticks': 15,  # Number of ticks for take profit
            'max_positions': 1,  # Maximum number of concurrent positions
            'max_trades_per_day': 2,  # Conservative to avoid PDT issues
            
            # Risk management
            'risk_per_trade': 0.01,  # 1% per trade
            'max_risk_per_day': 0.03,  # 3% max per day
            'max_position_size_pct': 0.05,  # Max 5% of account in a single position
        }
        
        # Update with user-provided parameters
        if parameters:
            default_params.update(parameters)
        self.parameters = default_params
        
        # Strategy state
        self.daily_trade_count = 0
        self.daily_risk_used = 0.0
        self.vwap_data = []
        
        logger.info(f"Initialized {self.name} for {session.symbol} on {session.timeframe}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for the Scalping strategy.
        
        Args:
            data: Market data DataFrame with OHLCV columns
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        if data.empty or len(data) < self.parameters['rsi_length']:
            return indicators
        
        try:
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.parameters['rsi_length']).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.parameters['rsi_length']).mean()
            
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate EMAs
            indicators['ema_fast'] = data['close'].ewm(span=self.parameters['ema_fast'], adjust=False).mean()
            indicators['ema_slow'] = data['close'].ewm(span=self.parameters['ema_slow'], adjust=False).mean()
            
            # Calculate VWAP (Volume Weighted Average Price)
            self._calculate_vwap(data)
            indicators['vwap'] = self.vwap_data
            
            # Calculate Volume analysis
            indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
            
            # Calculate price action metrics
            indicators['price_range'] = data['high'] - data['low']
            indicators['body_size'] = abs(data['close'] - data['open'])
            indicators['upper_wick'] = data['high'] - data[['open', 'close']].max(axis=1)
            indicators['lower_wick'] = data[['open', 'close']].min(axis=1) - data['low']
            
            # Market microstructure
            indicators['micro_trend'] = np.where(
                indicators['ema_fast'] > indicators['ema_slow'], 
                'bullish', 
                'bearish'
            )
            
            # Momentum
            indicators['momentum'] = data['close'].diff(3)
            
        except Exception as e:
            logger.error(f"Error calculating scalping indicators: {str(e)}")
        
        return indicators
    
    def _calculate_vwap(self, data: pd.DataFrame) -> None:
        """
        Calculate VWAP (Volume Weighted Average Price).
        
        Args:
            data: Market data DataFrame with OHLCV columns
        """
        try:
            # Reset VWAP data at market open
            current_time = datetime.now()
            market_open = datetime(current_time.year, current_time.month, current_time.day, 9, 30)
            
            if current_time.time() < market_open.time():
                self.vwap_data = []
            
            # Calculate typical price
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            
            # Calculate VWAP
            cumulative_tp_vol = (typical_price * data['volume']).cumsum()
            cumulative_vol = data['volume'].cumsum()
            
            # Avoid division by zero
            cumulative_vol = np.where(cumulative_vol == 0, 1, cumulative_vol)
            
            self.vwap_data = cumulative_tp_vol / cumulative_vol
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            self.vwap_data = []
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals for the Scalping strategy.
        
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
            
            # Check if we've reached our daily trade limit
            if self.daily_trade_count >= self.parameters['max_trades_per_day']:
                logger.info(f"Reached daily trade limit ({self.parameters['max_trades_per_day']}), no new entries")
                return signals
            
            # Check if we've reached our daily risk limit
            if self.daily_risk_used >= self.parameters['max_risk_per_day']:
                logger.info(f"Reached daily risk limit ({self.parameters['max_risk_per_day']*100:.1f}%), no new entries")
                return signals
            
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
            
            # Don't generate entry signals if we already have max positions
            if len([p for p in self.positions if p.status == PositionStatus.OPEN]) >= self.parameters['max_positions']:
                return signals
            
            # Entry signals
            rsi = indicators['rsi'].iloc[-1]
            rsi_prev = indicators['rsi'].iloc[-2] if len(indicators['rsi']) > 1 else None
            
            ema_fast = indicators['ema_fast'].iloc[-1]
            ema_slow = indicators['ema_slow'].iloc[-1]
            
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            
            # Check volume threshold
            if volume_ratio < self.parameters['volume_threshold']:
                logger.debug(f"Volume ratio {volume_ratio:.2f} below threshold {self.parameters['volume_threshold']:.2f}")
                return signals
            
            # Long entry conditions
            if (rsi < self.parameters['rsi_oversold'] and 
                rsi_prev is not None and 
                rsi > rsi_prev and  # RSI turning up from oversold
                ema_fast > ema_fast.shift(1).iloc[-1] and  # Fast EMA turning up
                volume_ratio > self.parameters['volume_threshold']):
                
                signals['entry'] = True
                signals['direction'] = 'long'
                signals['strength'] = min(1.0, (self.parameters['rsi_oversold'] - rsi) / 10 + volume_ratio / 2)
                
                # Calculate stop loss and take profit
                stop_loss = current_price - (self.parameters['stop_loss_ticks'] * self.parameters['tick_size'])
                take_profit = current_price + (self.parameters['take_profit_ticks'] * self.parameters['tick_size'])
                
                signals['stop_loss'] = stop_loss
                signals['take_profit'] = take_profit
                
                logger.info(f"Long entry signal generated at {current_price:.2f}, "
                           f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
            
            # Short entry conditions
            elif (rsi > self.parameters['rsi_overbought'] and 
                  rsi_prev is not None and 
                  rsi < rsi_prev and  # RSI turning down from overbought
                  ema_fast < ema_fast.shift(1).iloc[-1] and  # Fast EMA turning down
                  volume_ratio > self.parameters['volume_threshold']):
                
                signals['entry'] = True
                signals['direction'] = 'short'
                signals['strength'] = min(1.0, (rsi - self.parameters['rsi_overbought']) / 10 + volume_ratio / 2)
                
                # Calculate stop loss and take profit
                stop_loss = current_price + (self.parameters['stop_loss_ticks'] * self.parameters['tick_size'])
                take_profit = current_price - (self.parameters['take_profit_ticks'] * self.parameters['tick_size'])
                
                signals['stop_loss'] = stop_loss
                signals['take_profit'] = take_profit
                
                logger.info(f"Short entry signal generated at {current_price:.2f}, "
                           f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
                
        except Exception as e:
            logger.error(f"Error generating scalping signals: {str(e)}")
        
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
            
            # Calculate stop loss distance in dollars
            stop_loss_distance = self.parameters['stop_loss_ticks'] * self.parameters['tick_size']
            
            # Calculate position size based on risk (risk amount / stop loss distance)
            position_size = risk_amount / stop_loss_distance
            
            # Convert to number of shares
            shares = position_size / current_price
            
            # Apply account aware constraints
            max_shares, max_notional = self.calculate_max_position_size(
                price=current_price,
                is_day_trade=True,  # Scalping is always day trading
                risk_percent=self.parameters['risk_per_trade']
            )
            
            # Check PDT rule compliance
            if not self.check_pdt_rule_compliance(is_day_trade=True):
                logger.warning("PDT rule would be violated, reducing position size to zero")
                return 0
            
            # Use the smaller of our calculated position sizes
            position_size = min(shares, max_shares)
            
            # Check if position would exceed max position size as percentage of account
            max_position_value = account_equity * self.parameters['max_position_size_pct']
            max_position_shares = max_position_value / current_price
            position_size = min(position_size, max_position_shares)
            
            # Track daily risk
            self.daily_risk_used += self.parameters['risk_per_trade']
            
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
        2. PDT rule compliance
        3. Position size limits
        4. Daily trade and risk limits
        """
        # Ensure account status is up to date
        self.check_account_status()
        
        # Verify account has sufficient buying power
        buying_power = self.get_buying_power(day_trade=True)
        if buying_power <= 0:
            logger.warning("Insufficient buying power for scalping strategy")
            return
        
        # Verify PDT rule compliance
        if not self.check_pdt_rule_compliance(is_day_trade=True):
            logger.warning("Cannot execute trade due to PDT rule restrictions")
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
            
            if not self.validate_trade_size(symbol, position_size, current_price, is_day_trade=True):
                logger.warning(f"Trade validation failed for {symbol}, size: {position_size}")
                return
                
            # Open position if size > 0
            if position_size > 0:
                stop_loss = self.signals.get('stop_loss')
                take_profit = self.signals.get('take_profit')
                
                # Only execute if we have valid stop loss and take profit
                if stop_loss and take_profit:
                    self._open_position(direction, position_size)
                    
                    # Track day trade
                    self.daily_trade_count += 1
                    self.record_day_trade()
                    
                    logger.info(f"Opened {direction} position of {position_size:.2f} shares")
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible the scalping strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime description
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        compatibility_map = {
            'ranging': 0.90,              # Excellent in range-bound markets
            'choppy': 0.85,               # Very good in choppy markets
            'trending': 0.65,             # Good in trending markets
            'strong_trend': 0.50,         # Moderate in strong trends
            'low_volatility': 0.40,       # Poor in low volatility (not enough price movement)
            'high_volatility': 0.75,      # Good in high volatility (more opportunities)
            'extreme_volatility': 0.25,   # Poor in extreme volatility (too risky)
            'news_driven': 0.30,          # Poor during major news events
            'opening_hour': 0.80,         # Very good during market open
            'closing_hour': 0.70,         # Good during market close
            'lunch_hour': 0.30,           # Poor during lunch hour (low volume)
            'overnight': 0.00,            # Not compatible with overnight holding
            'gap_up': 0.60,               # Moderate after gap up
            'gap_down': 0.60,             # Moderate after gap down
        }
        
        # Default to moderate compatibility if regime not recognized
        return compatibility_map.get(market_regime, 0.50)
    
    def on_market_open(self) -> None:
        """Handle market open event."""
        # Reset daily counters
        self.daily_trade_count = 0
        self.daily_risk_used = 0.0
        
        # Reset VWAP data
        self.vwap_data = []
        
        logger.info("Market open: Reset Scalping strategy daily counters and VWAP data")
    
    def on_market_close(self) -> None:
        """Handle market close event."""
        # Close all open positions at market close (scalping doesn't hold overnight)
        for position in self.positions:
            if position.status == PositionStatus.OPEN:
                self._close_position(position.position_id)
                logger.info(f"Market close: Closed position {position.position_id}")
