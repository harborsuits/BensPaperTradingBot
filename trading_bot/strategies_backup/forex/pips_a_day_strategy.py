#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forex Pips-a-Day Strategy

A target-based forex strategy that aims to achieve specific daily profit
objectives in pips, adapting to market conditions and using systematic
risk management to secure consistent returns.

Key Features:
1. Daily profit target management
2. Adaptive entry point selection
3. Risk-reward optimization based on volatility
4. Intraday scaling in/out of positions
5. System-based trade management
6. Automatic position closure on target achievement

Author: Ben Dickinson
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pytz
from enum import Enum

from trading_bot.strategies.base.forex_base import (
    ForexBaseStrategy, 
    MarketRegime, 
    MarketSession, 
    TradeDirection
)
from trading_bot.utils.event_bus import EventBus

logger = logging.getLogger(__name__)

class TargetStatus(Enum):
    """Daily target status tracking"""
    NOT_STARTED = "not_started"    # Day's trading not started
    IN_PROGRESS = "in_progress"    # Working toward target
    TARGET_REACHED = "target_reached"  # Daily target achieved
    TARGET_EXCEEDED = "target_exceeded"  # Exceeded daily target
    LOSS_LIMIT_REACHED = "loss_limit_reached"  # Daily loss limit reached

class EntryQuality(Enum):
    """Entry quality classification"""
    PREMIUM = "premium"    # High-probability setup
    STANDARD = "standard"  # Standard setup
    MARGINAL = "marginal"  # Lower-probability setup
    NOT_VALID = "not_valid"  # Setup doesn't meet criteria

class PipsADayStrategy(ForexBaseStrategy):
    """
    Pips-a-Day Strategy
    
    A target-based forex strategy focused on achieving specific daily pip profit
    objectives through systematic trading rules and rigid risk management.
    
    This strategy excels at:
    - Securing consistent daily returns
    - Adapting targets to volatility conditions
    - Systematic scaling in and out of positions
    - Achieving predictable performance metrics
    - Clear daily performance measurement
    """
    
    def __init__(self, 
                 parameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize the Pips-a-Day Strategy
        
        Args:
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Call the parent constructor first
        super().__init__("pips_a_day", parameters, metadata)
        
        # Default parameters specific to pips-a-day strategy
        default_params = {
            # Core target settings
            'daily_pip_target': 50,       # Target pips per day
            'daily_loss_limit': -30,      # Max loss per day in pips
            'multi_target_strategy': True, # Use multiple targets
            'base_currency_pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'], # Primary focus pairs
            
            # Target adjustment
            'volatility_based_targets': True, # Adjust targets based on volatility
            'target_volatility_factor': 1.5,  # Target = ATR * this factor
            'min_daily_target': 20,       # Minimum target even in low volatility
            'max_daily_target': 100,      # Maximum target even in high volatility
            
            # Entry criteria
            'min_entry_quality': 'standard',  # Minimum setup quality
            'premium_entry_boost': 1.5,   # Increase position size for premium entries
            'required_confirmation_count': 2, # Number of confirming indicators
            'max_entries_per_day': 5,     # Maximum new positions per day
            
            # Trade management
            'scale_out_levels': [0.3, 0.7, 1.0], # Take partial profits at these target %
            'scale_in_enabled': True,     # Allow scaling into positions
            'max_scale_in_count': 2,      # Maximum scale-ins per position
            'scale_in_threshold': -10,    # Pips drawdown to consider scaling in
            
            # Stop loss and take profit
            'base_stop_loss_pips': 20,    # Base stop loss in pips
            'min_reward_risk_ratio': 1.5, # Minimum reward:risk ratio
            'atr_stop_loss_factor': 1.0,  # SL = ATR * this factor (if used)
            'use_adaptive_stops': True,   # Adjust stops based on volatility
            'move_to_breakeven_pct': 0.5, # Move to breakeven at this % of target
            
            # Timeframes and data
            'analysis_timeframes': ['1h', '4h', 'D'], # For multi-timeframe analysis
            'primary_execution_timeframe': '1h', # Main execution timeframe
            'min_data_lookback': 50,      # Minimum bars of historical data needed
            
            # Session settings
            'trade_asian_session': True,
            'trade_european_session': True,
            'trade_us_session': True,
            
            # Performance tracking
            'track_daily_performance': True, # Track daily performance metrics
            'weekly_target_modifier': 0.8,   # Reduce target if weekly goal reached
            'monthly_target_modifier': 0.6,  # Reduce target if monthly goal reached
            
            # Indicators
            'entry_indicators': ['rsi', 'macd', 'atr', 'bollinger'],
            'filter_indicators': ['moving_averages', 'adx', 'keltner'],
            'confirmation_indicators': ['volume', 'momentum', 'divergence'],
            
            # Filters
            'news_filter_enabled': True,     # Filter out high-impact news
            'news_filter_window_hours': 1,   # Hours around news to avoid trading
            'min_daily_range_pips': 30,      # Minimum daily range to consider trading
            'correlation_filter_enabled': True, # Avoid highly correlated positions
        }
        
        # Update default parameters with provided ones
        if self.parameters is None:
            self.parameters = {}
            
        # Apply defaults for any missing parameters
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Initialize strategy state
        self._initialize_state()
        
        logger.info(f"Initialized {self.__class__.__name__} with parameters: {self.parameters}")
        
    def _initialize_state(self):
        """Initialize strategy state variables"""
        # Daily tracking
        self.daily_stats = {}          # Track daily performance by date
        self.current_day_pip_total = 0  # Running pip total for current day
        self.current_day_positions = 0  # Count of positions for current day
        self.current_day_status = TargetStatus.NOT_STARTED  # Current day status
        
        # Active positions and targets
        self.active_positions = {}     # Track active positions
        self.daily_targets = {}        # Daily pip targets by symbol
        self.daily_achieved = {}       # Tracking of daily pips achieved
        
        # Performance tracking
        self.weekly_pips = 0           # Running total for the week
        self.monthly_pips = 0          # Running total for the month
        self.last_updated_day = None   # Last day we updated stats
        
        # Session tracking
        self.current_session = None    # Current trading session
        self.session_pips = {          # Pips by session
            MarketSession.ASIAN: 0,
            MarketSession.EUROPEAN: 0,
            MarketSession.US: 0
        }
        
        # Target adjustment factors
        self.volatility_factors = {}   # Volatility adjustment factors by symbol
        
        # Register event listeners
        self._register_events()
    
    def _register_events(self):
        """Register strategy event listeners"""
        event_bus = EventBus.get_instance()
        event_bus.subscribe("market_data_update", self._on_market_data_update)
        event_bus.subscribe("session_change", self._on_session_change)
        event_bus.subscribe("day_change", self._on_day_change)
        event_bus.subscribe("news_announcement", self._on_news_announcement)
        event_bus.subscribe("trade_closed", self._on_trade_closed)
        
    def _on_market_data_update(self, data):
        """Handle market data updates"""
        if 'symbol' not in data or 'timeframe' not in data:
            return
            
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        ohlcv = data.get('ohlcv')
        
        # We primarily care about our execution timeframe
        if timeframe == self.parameters['primary_execution_timeframe']:
            self._update_volatility_factors(symbol, ohlcv)
            self._check_daily_targets(symbol, ohlcv)
            
    def _on_session_change(self, data):
        """Handle trading session changes"""
        new_session = data.get('new_session')
        old_session = data.get('old_session')
        timestamp = data.get('timestamp')
        
        # Update current session
        self.current_session = new_session
        
        logger.info(f"Trading session changed from {old_session} to {new_session} at {timestamp}")
        
        # Check if we should trade in this session
        if self._is_tradable_session(new_session):
            logger.info(f"Session {new_session} is tradable, continuing normal operations")
        else:
            logger.info(f"Session {new_session} is not tradable, will not generate new signals")
            
        # Publish session update event with pips data
        EventBus.get_instance().publish('pips_session_update', {
            'strategy': self.__class__.__name__,
            'old_session': old_session.name if hasattr(old_session, 'name') else old_session,
            'new_session': new_session.name if hasattr(new_session, 'name') else new_session,
            'session_pips': self.session_pips.get(new_session, 0),
            'day_pips': self.current_day_pip_total,
            'target_status': self.current_day_status.value,
            'timestamp': timestamp
        })
        
    def _on_day_change(self, data):
        """Handle day change events"""
        previous_date = data.get('previous_date')
        new_date = data.get('new_date')
        
        # Reset daily tracking
        self._reset_daily_stats()
        
        # Calculate new daily targets
        self._calculate_daily_targets()
        
        logger.info(f"Day changed from {previous_date} to {new_date}, daily pip targets reset")
        
        # Check if we need to update weekly/monthly totals
        if new_date.weekday() == 0:  # Monday
            self.weekly_pips = 0
            logger.info("New week started, weekly pip counter reset")
            
        if new_date.day == 1:  # First day of month
            self.monthly_pips = 0
            logger.info("New month started, monthly pip counter reset")
            
    def _on_news_announcement(self, data):
        """Handle economic news announcement"""
        if self.parameters['news_filter_enabled']:
            impact = data.get('impact', 'low')
            # Only filter for medium and high impact news
            if impact.lower() in ['medium', 'high']:
                symbol = data.get('symbol')
                window_hours = self.parameters['news_filter_window_hours']
                
                # Add this symbol to the news filter list with expiry time
                news_time = pd.to_datetime(data.get('time'))
                expiry_time = news_time + pd.Timedelta(hours=window_hours)
                
                logger.info(f"Filtering {symbol} until {expiry_time} due to {impact} impact news")
                
                # Store in news filter with expiry time
                if not hasattr(self, 'news_filter'):
                    self.news_filter = {}
                
                self.news_filter[symbol] = {
                    'expiry': expiry_time,
                    'impact': impact
                }
                
    def _on_trade_closed(self, data):
        """Handle trade closed events"""
        symbol = data.get('symbol')
        pips = data.get('pips_profit', 0)
        position_id = data.get('position_id')
        
        # Update daily pip totals
        self.current_day_pip_total += pips
        
        # Update session pips if we have a current session
        if self.current_session and self.current_session in self.session_pips:
            self.session_pips[self.current_session] += pips
            
        # Update daily achieved for this symbol
        if symbol not in self.daily_achieved:
            self.daily_achieved[symbol] = 0
        self.daily_achieved[symbol] += pips
        
        # Update weekly and monthly totals
        self.weekly_pips += pips
        self.monthly_pips += pips
        
        # Remove from active positions if tracked
        if position_id in self.active_positions:
            del self.active_positions[position_id]
            
        # Check if we've hit daily target or loss limit
        self._check_daily_status()
        
        # Log the trade result
        if pips > 0:
            logger.info(f"Closed trade on {symbol} with profit of {pips} pips. Daily total: {self.current_day_pip_total} pips")
        else:
            logger.info(f"Closed trade on {symbol} with loss of {pips} pips. Daily total: {self.current_day_pip_total} pips")
            
        # Publish updated pip status
        EventBus.get_instance().publish('pips_update', {
            'strategy': self.__class__.__name__,
            'symbol': symbol,
            'trade_pips': pips,
            'daily_pips': self.current_day_pip_total,
            'weekly_pips': self.weekly_pips,
            'monthly_pips': self.monthly_pips,
            'target_status': self.current_day_status.value,
            'daily_target': self.parameters['daily_pip_target'],
            'timestamp': pd.Timestamp.now()
        })
        
    def _is_tradable_session(self, session):
        """Check if we should trade in the given session"""
        if session == MarketSession.ASIAN and self.parameters['trade_asian_session']:
            return True
        elif session == MarketSession.EUROPEAN and self.parameters['trade_european_session']:
            return True
        elif session == MarketSession.US and self.parameters['trade_us_session']:
            return True
        return False
    
    def _reset_daily_stats(self):
        """Reset daily statistics for a new trading day"""
        # Store the previous day's stats if we have any
        if self.current_day_pip_total != 0 or self.current_day_positions > 0:
            today = pd.Timestamp.now().strftime('%Y-%m-%d')
            self.daily_stats[today] = {
                'pips_total': self.current_day_pip_total,
                'positions': self.current_day_positions,
                'target_status': self.current_day_status.value,
                'targets': self.daily_targets.copy(),
                'achieved': self.daily_achieved.copy()
            }
            
        # Reset counters for the new day
        self.current_day_pip_total = 0
        self.current_day_positions = 0
        self.current_day_status = TargetStatus.NOT_STARTED
        self.daily_achieved = {}
        
        # Reset session pips
        for session in self.session_pips:
            self.session_pips[session] = 0
            
    def _calculate_daily_targets(self):
        """Calculate daily pip targets, potentially adjusting for volatility"""
        base_target = self.parameters['daily_pip_target']
        
        # Check for weekly/monthly target modifiers
        if self.weekly_pips > (base_target * 5):
            # If we've already hit the weekly target, reduce daily target
            modifier = self.parameters['weekly_target_modifier']
            logger.info(f"Weekly target reached ({self.weekly_pips} pips), applying modifier {modifier}")
            base_target *= modifier
            
        if self.monthly_pips > (base_target * 20):
            # If we've already hit the monthly target, reduce daily target
            modifier = self.parameters['monthly_target_modifier']
            logger.info(f"Monthly target reached ({self.monthly_pips} pips), applying modifier {modifier}")
            base_target *= modifier
        
        # Set the targets for each symbol
        for symbol in self.parameters['base_currency_pairs']:
            if self.parameters['volatility_based_targets'] and symbol in self.volatility_factors:
                # Adjust target based on volatility
                vol_factor = self.volatility_factors[symbol]
                symbol_target = base_target * vol_factor
                
                # Apply min/max constraints
                symbol_target = max(self.parameters['min_daily_target'], 
                                  min(self.parameters['max_daily_target'], symbol_target))
                
                logger.debug(f"Volatility-adjusted target for {symbol}: {symbol_target:.1f} pips (factor: {vol_factor:.2f})")
            else:
                symbol_target = base_target
            
            # Store the target
            if not hasattr(self, 'daily_targets'):
                self.daily_targets = {}
                
            self.daily_targets[symbol] = symbol_target
        
        # Log the new targets
        logger.info(f"Daily pip targets calculated: {self.daily_targets}")
        
    def _update_volatility_factors(self, symbol: str, ohlcv_data: pd.DataFrame):
        """Update volatility adjustment factors for a symbol"""
        if symbol not in self.parameters['base_currency_pairs'] or ohlcv_data is None or ohlcv_data.empty:
            return
            
        # Calculate ATR (Average True Range)
        if len(ohlcv_data) >= 14:  # Need at least 14 bars
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            close = ohlcv_data['close'].shift(1)
            
            # Calculate true range
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            # True range is the max of these
            tr = pd.DataFrame({
                'tr1': tr1,
                'tr2': tr2,
                'tr3': tr3
            }).max(axis=1)
            
            # ATR is the moving average of true range
            atr = tr.rolling(window=14).mean().iloc[-1]
            
            # Calculate pip-based ATR
            if 'JPY' in symbol:
                # For JPY pairs, 1 pip = 0.01
                atr_pips = atr * 100
            else:
                # For other pairs, 1 pip = 0.0001
                atr_pips = atr * 10000
                
            # Calculate volatility factor (normalize against base target)
            base_target = self.parameters['daily_pip_target']
            vol_factor = (atr_pips * self.parameters['target_volatility_factor']) / base_target
            
            # Update volatility factor
            self.volatility_factors[symbol] = vol_factor
            
            logger.debug(f"Updated volatility factor for {symbol}: {vol_factor:.2f} (ATR: {atr_pips:.1f} pips)")
            
    def _check_daily_targets(self, symbol: str, ohlcv_data: pd.DataFrame):
        """Check if daily range is sufficient for trading"""
        # Skip if we've already reached the target or loss limit
        if self.current_day_status in [TargetStatus.TARGET_REACHED, TargetStatus.LOSS_LIMIT_REACHED]:
            return False
            
        # Check min daily range
        if ohlcv_data is not None and not ohlcv_data.empty:
            # Get today's data
            today = pd.Timestamp.now().date()
            today_data = ohlcv_data[ohlcv_data.index.date == today]
            
            if not today_data.empty:
                daily_high = today_data['high'].max()
                daily_low = today_data['low'].min()
                
                # Calculate daily range in pips
                if 'JPY' in symbol:
                    daily_range_pips = (daily_high - daily_low) * 100
                else:
                    daily_range_pips = (daily_high - daily_low) * 10000
                    
                # Check if range is sufficient
                return daily_range_pips >= self.parameters['min_daily_range_pips']
        
        return False
        
    def _check_daily_status(self):
        """Check if daily target or loss limit has been reached"""
        # Get base target and loss limit
        base_target = self.parameters['daily_pip_target']
        loss_limit = self.parameters['daily_loss_limit']
        
        # Check if we've hit the target
        if self.current_day_pip_total >= base_target and self.current_day_status != TargetStatus.TARGET_REACHED:
            self.current_day_status = TargetStatus.TARGET_REACHED
            logger.info(f"Daily pip target of {base_target} pips reached! Current total: {self.current_day_pip_total} pips")
            
            # Publish target reached event
            EventBus.get_instance().publish('daily_target_reached', {
                'strategy': self.__class__.__name__,
                'target': base_target,
                'achieved': self.current_day_pip_total,
                'timestamp': pd.Timestamp.now()
            })
            
        # Check if we've exceeded the target
        elif self.current_day_pip_total > base_target * 1.5 and self.current_day_status != TargetStatus.TARGET_EXCEEDED:
            self.current_day_status = TargetStatus.TARGET_EXCEEDED
            logger.info(f"Daily pip target exceeded by 50%! Current total: {self.current_day_pip_total} pips")
            
            # Publish target exceeded event
            EventBus.get_instance().publish('daily_target_exceeded', {
                'strategy': self.__class__.__name__,
                'target': base_target,
                'achieved': self.current_day_pip_total,
                'timestamp': pd.Timestamp.now()
            })
            
        # Check if we've hit the loss limit
        elif self.current_day_pip_total <= loss_limit and self.current_day_status != TargetStatus.LOSS_LIMIT_REACHED:
            self.current_day_status = TargetStatus.LOSS_LIMIT_REACHED
            logger.warning(f"Daily loss limit of {loss_limit} pips reached! Current total: {self.current_day_pip_total} pips")
            
            # Publish loss limit reached event
            EventBus.get_instance().publish('daily_loss_limit_reached', {
                'strategy': self.__class__.__name__,
                'limit': loss_limit,
                'current': self.current_day_pip_total,
                'timestamp': pd.Timestamp.now()
            })
            
        # If we're making progress but haven't hit any threshold yet
        elif self.current_day_pip_total != 0 and self.current_day_status == TargetStatus.NOT_STARTED:
            self.current_day_status = TargetStatus.IN_PROGRESS
        
        return self.current_day_status
        
    def _should_skip_for_news(self, symbol: str, current_time: pd.Timestamp) -> bool:
        """Check if we should skip trading due to news filter"""
        if not self.parameters['news_filter_enabled'] or not hasattr(self, 'news_filter'):
            return False
            
        # Check if symbol is in news filter
        if symbol in self.news_filter:
            news_info = self.news_filter[symbol]
            expiry_time = news_info['expiry']
            
            # If current time is before expiry, skip trading
            if current_time < expiry_time:
                return True
                
            # If expired, remove from filter
            else:
                del self.news_filter[symbol]
                
        return False
        
    def should_generate_signals(self) -> bool:
        """Check if the strategy should generate signals"""
        # Skip if we've reached the target or loss limit
        if self.current_day_status in [TargetStatus.TARGET_REACHED, TargetStatus.TARGET_EXCEEDED, 
                                    TargetStatus.LOSS_LIMIT_REACHED]:
            return False
            
        # Skip if we're already at max positions for the day
        if self.current_day_positions >= self.parameters['max_entries_per_day']:
            return False
            
        # Continue if we're still working toward the target
        return True
    
    def _evaluate_setup(self, symbol: str, data: pd.DataFrame, current_time: pd.Timestamp) -> Tuple[EntryQuality, TradeDirection, float]:
        """Evaluate a potential setup for a symbol
        
        Args:
            symbol: Currency pair
            data: OHLCV DataFrame
            current_time: Current timestamp
            
        Returns:
            Tuple of (setup quality, trade direction, setup strength)
        """
        # Default values
        quality = EntryQuality.NOT_VALID
        direction = TradeDirection.FLAT
        strength = 0.0
        
        # Check if we have enough data
        if data is None or len(data) < self.parameters['min_data_lookback']:
            return quality, direction, strength
            
        # Get required indicators
        entry_indicators = self.parameters['entry_indicators']
        filter_indicators = self.parameters['filter_indicators']
        confirmation_indicators = self.parameters['confirmation_indicators']
        
        # Extract indicators based on configuration
        indicator_values = {}
        
        # RSI
        if 'rsi' in entry_indicators:
            rsi = self._calculate_rsi(data['close'], window=14)
            indicator_values['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
        # MACD
        if 'macd' in entry_indicators or 'macd' in confirmation_indicators:
            macd, signal, hist = self._calculate_macd(data['close'])
            indicator_values['macd'] = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
            indicator_values['macd_signal'] = signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0
            indicator_values['macd_hist'] = hist.iloc[-1] if not pd.isna(hist.iloc[-1]) else 0
            
        # ATR
        if 'atr' in entry_indicators:
            atr = self._calculate_atr(data, window=14)
            indicator_values['atr'] = atr
            
        # Bollinger Bands
        if 'bollinger' in entry_indicators:
            upper, middle, lower, bandwidth = self._calculate_bollinger_bands(data['close'])
            indicator_values['bollinger_upper'] = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else data['close'].iloc[-1]
            indicator_values['bollinger_middle'] = middle.iloc[-1] if not pd.isna(middle.iloc[-1]) else data['close'].iloc[-1]
            indicator_values['bollinger_lower'] = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else data['close'].iloc[-1]
            indicator_values['bollinger_bandwidth'] = bandwidth.iloc[-1] if not pd.isna(bandwidth.iloc[-1]) else 0
            
        # Moving Averages
        if 'moving_averages' in filter_indicators:
            sma20 = data['close'].rolling(window=20).mean()
            sma50 = data['close'].rolling(window=50).mean()
            ema20 = data['close'].ewm(span=20, adjust=False).mean()
            
            indicator_values['sma20'] = sma20.iloc[-1] if not pd.isna(sma20.iloc[-1]) else data['close'].iloc[-1]
            indicator_values['sma50'] = sma50.iloc[-1] if not pd.isna(sma50.iloc[-1]) else data['close'].iloc[-1]
            indicator_values['ema20'] = ema20.iloc[-1] if not pd.isna(ema20.iloc[-1]) else data['close'].iloc[-1]
            
        # ADX
        if 'adx' in filter_indicators:
            adx, di_plus, di_minus = self._calculate_adx(data)
            indicator_values['adx'] = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
            indicator_values['di_plus'] = di_plus.iloc[-1] if not pd.isna(di_plus.iloc[-1]) else 0
            indicator_values['di_minus'] = di_minus.iloc[-1] if not pd.isna(di_minus.iloc[-1]) else 0
        
        # Count confirmations for trade direction
        long_confirmations = 0
        short_confirmations = 0
        
        # Check RSI
        if 'rsi' in entry_indicators and 'rsi' in indicator_values:
            rsi_value = indicator_values['rsi']
            # Bullish RSI
            if rsi_value < 40 and rsi_value > 20:
                long_confirmations += 1
            # Bearish RSI
            elif rsi_value > 60 and rsi_value < 80:
                short_confirmations += 1
                
        # Check MACD
        if 'macd' in entry_indicators and 'macd_hist' in indicator_values:
            macd_hist = indicator_values['macd_hist']
            # Bullish MACD
            if macd_hist > 0 and macd_hist > indicator_values.get('macd_hist_prev', 0):
                long_confirmations += 1
            # Bearish MACD
            elif macd_hist < 0 and macd_hist < indicator_values.get('macd_hist_prev', 0):
                short_confirmations += 1
                
        # Check Bollinger Bands
        if 'bollinger' in entry_indicators:
            current_price = data['close'].iloc[-1]
            upper = indicator_values['bollinger_upper']
            lower = indicator_values['bollinger_lower']
            bandwidth = indicator_values['bollinger_bandwidth']
            
            # Bullish Bollinger
            if current_price < lower * 1.01 and bandwidth > 0.1:  # Price near lower band
                long_confirmations += 1
            # Bearish Bollinger
            elif current_price > upper * 0.99 and bandwidth > 0.1:  # Price near upper band
                short_confirmations += 1
                
        # Check Moving Averages
        if 'moving_averages' in filter_indicators:
            current_price = data['close'].iloc[-1]
            sma20 = indicator_values['sma20']
            sma50 = indicator_values['sma50']
            
            # Bullish MA setup
            if current_price > sma20 and sma20 > sma50:
                long_confirmations += 1
            # Bearish MA setup
            elif current_price < sma20 and sma20 < sma50:
                short_confirmations += 1
                
        # Check ADX for trend strength
        if 'adx' in filter_indicators:
            adx_value = indicator_values['adx']
            di_plus = indicator_values['di_plus']
            di_minus = indicator_values['di_minus']
            
            # Strong bullish trend
            if adx_value > 25 and di_plus > di_minus:
                long_confirmations += 1
            # Strong bearish trend
            elif adx_value > 25 and di_minus > di_plus:
                short_confirmations += 1
        
        # Determine setup quality based on confirmations
        required_confirmations = self.parameters['required_confirmation_count']
        
        if long_confirmations > short_confirmations and long_confirmations >= required_confirmations:
            direction = TradeDirection.LONG
            # Calculate strength based on number of confirmations
            strength = min(1.0, long_confirmations / len(entry_indicators))
            
            # Determine quality
            if long_confirmations >= required_confirmations + 2:
                quality = EntryQuality.PREMIUM
            elif long_confirmations >= required_confirmations + 1:
                quality = EntryQuality.STANDARD
            else:
                quality = EntryQuality.MARGINAL
                
        elif short_confirmations > long_confirmations and short_confirmations >= required_confirmations:
            direction = TradeDirection.SHORT
            # Calculate strength based on number of confirmations
            strength = min(1.0, short_confirmations / len(entry_indicators))
            
            # Determine quality
            if short_confirmations >= required_confirmations + 2:
                quality = EntryQuality.PREMIUM
            elif short_confirmations >= required_confirmations + 1:
                quality = EntryQuality.STANDARD
            else:
                quality = EntryQuality.MARGINAL
        
        # Check if setup meets minimum quality requirement
        min_quality = EntryQuality(self.parameters['min_entry_quality'])
        if quality.value < min_quality.value:
            quality = EntryQuality.NOT_VALID
            direction = TradeDirection.FLAT
            strength = 0.0
            
        return quality, direction, strength
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window, min_periods=1).mean()
        avg_loss = losses.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD, Signal, and Histogram"""
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd = ema_fast - ema_slow
        
        # Calculate signal line
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def _calculate_atr(self, data, window=14):
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        # Calculate true range components
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        # True range is the max of these
        tr = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # ATR is the moving average of true range
        atr = tr.rolling(window=window).mean().iloc[-1]
        
        # Convert to pips
        if 'JPY' in data.columns[0] if isinstance(data.columns[0], str) else False:
            atr_pips = atr * 100
        else:
            atr_pips = atr * 10000
            
        return atr_pips
    
    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        # Calculate middle band (SMA)
        middle_band = prices.rolling(window=window).mean()
        
        # Calculate standard deviation
        std_dev = prices.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        # Calculate bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        
        return upper_band, middle_band, lower_band, bandwidth
    
    def _calculate_adx(self, data, window=14):
        """Calculate Average Directional Index (ADX)"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate +DM and -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        
        # Calculate smoothed TR, +DM, and -DM
        tr_smoothed = tr.rolling(window=window).sum()
        plus_dm_smoothed = plus_dm.rolling(window=window).sum()
        minus_dm_smoothed = minus_dm.rolling(window=window).sum()
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
        minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
        
        # Calculate DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.rolling(window=window).mean()
        
        return adx, plus_di, minus_di
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals based on pip target strategy
        
        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            current_time: Current timestamp
            
        Returns:
            Dictionary of signal information
        """
        signals = {}
        
        # Check if we should generate signals at all
        if not self.should_generate_signals():
            return signals
            
        # Check if we're in an active trading session
        if not self.is_active_trading_session(current_time):
            return signals
            
        # Process each symbol in our base currency pairs
        for symbol in self.parameters['base_currency_pairs']:
            # Skip if we don't have data for this symbol
            if symbol not in data:
                continue
                
            # Get data for primary execution timeframe
            primary_tf = self.parameters['primary_execution_timeframe']
            if primary_tf not in data[symbol]:
                logger.warning(f"Missing {primary_tf} data for {symbol}")
                continue
                
            # Get OHLCV data
            ohlcv_data = data[symbol][primary_tf]
            
            # Skip if we don't have enough data
            if len(ohlcv_data) < self.parameters['min_data_lookback']:
                logger.debug(f"Insufficient data for {symbol}, have {len(ohlcv_data)} bars, need {self.parameters['min_data_lookback']}")
                continue
                
            # Check if we should skip due to news filter
            if self._should_skip_for_news(symbol, current_time):
                logger.info(f"Skipping {symbol} due to news filter")
                continue
                
            # Check if daily range is sufficient for trading
            if not self._check_daily_targets(symbol, ohlcv_data):
                logger.debug(f"Daily range for {symbol} is insufficient for trading")
                continue
                
            # Check correlation with existing positions if enabled
            if self.parameters['correlation_filter_enabled'] and self.active_positions:
                # Simple implementation - in production would check actual correlation
                if symbol in [pos['symbol'] for pos in self.active_positions.values()]:
                    logger.debug(f"Skipping {symbol} due to correlation filter - already have position")
                    continue
            
            # Evaluate setup quality
            quality, direction, strength = self._evaluate_setup(symbol, ohlcv_data, current_time)
            
            # Skip if not a valid setup
            if quality == EntryQuality.NOT_VALID or direction == TradeDirection.FLAT:
                continue
                
            # Calculate ATR for stop loss and take profit
            atr = self._calculate_atr(ohlcv_data)
            
            # Get current price
            current_price = ohlcv_data['close'].iloc[-1]
            
            # Calculate position size
            position_size = self._calculate_position_size(symbol, quality, strength)
            
            # Calculate stop loss and take profit levels
            if direction == TradeDirection.LONG:
                if self.parameters['use_adaptive_stops']:
                    stop_loss = current_price - (atr * self.parameters['atr_stop_loss_factor'])
                else:
                    stop_loss_pips = self.parameters['base_stop_loss_pips']
                    stop_loss = current_price - (stop_loss_pips / 10000 if 'JPY' not in symbol else stop_loss_pips / 100)
                    
                # Calculate take profit based on reward:risk ratio
                risk = current_price - stop_loss
                take_profit = current_price + (risk * self.parameters['min_reward_risk_ratio'])
                
            else:  # SHORT
                if self.parameters['use_adaptive_stops']:
                    stop_loss = current_price + (atr * self.parameters['atr_stop_loss_factor'])
                else:
                    stop_loss_pips = self.parameters['base_stop_loss_pips']
                    stop_loss = current_price + (stop_loss_pips / 10000 if 'JPY' not in symbol else stop_loss_pips / 100)
                    
                # Calculate take profit based on reward:risk ratio
                risk = stop_loss - current_price
                take_profit = current_price - (risk * self.parameters['min_reward_risk_ratio'])
            
            # Generate signal ID
            signal_id = f"pips_{symbol}_{quality.value}_{current_time.strftime('%Y%m%d%H%M')}"
            
            # Calculate pip distance to target
            if 'JPY' in symbol:
                pips_to_target = abs(take_profit - current_price) * 100
            else:
                pips_to_target = abs(take_profit - current_price) * 10000
                
            # Create signal object
            signal = {
                'id': signal_id,
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strength': strength,
                'position_size': position_size,
                'setup_quality': quality.value,
                'timeframe': primary_tf,
                'target_pips': pips_to_target,
                'atr_pips': atr,
                'daily_pips_total': self.current_day_pip_total,
                'daily_pip_target': self.parameters['daily_pip_target'],
                'time': current_time
            }
            
            # Add scale out levels if configured
            if self.parameters['scale_out_levels']:
                signal['scale_out_levels'] = []
                for level in self.parameters['scale_out_levels']:
                    if direction == TradeDirection.LONG:
                        price_level = current_price + (level * (take_profit - current_price))
                    else:
                        price_level = current_price - (level * (current_price - take_profit))
                    
                    signal['scale_out_levels'].append({
                        'percent': level * 100,
                        'price': price_level
                    })
            
            # Store the signal
            signals[symbol] = signal
            
            # Log signal generation
            logger.info(f"Generated pips-a-day signal for {symbol}: {direction.name} setup with {quality.value} quality")
            
            # Increment the position counter
            self.current_day_positions += 1
            
            # Add to active positions
            self.active_positions[signal_id] = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'setup_quality': quality.value,
                'entry_time': current_time,
                'scale_out_status': [False] * len(self.parameters['scale_out_levels']) 
                                  if self.parameters['scale_out_levels'] else []
            }
            
            # Publish signal event
            EventBus.get_instance().publish('pips_a_day_signal', {
                'id': signal_id,
                'symbol': symbol,
                'direction': direction.name,
                'setup_quality': quality.value,
                'strength': strength,
                'timeframe': primary_tf,
                'target_pips': pips_to_target,
                'current_day_pips': self.current_day_pip_total,
                'daily_target': self.parameters['daily_pip_target'],
                'timestamp': current_time.isoformat()
            })
            
        return signals
    
    def _calculate_position_size(self, symbol: str, quality: EntryQuality, strength: float) -> float:
        """Calculate position size based on setup quality"""
        # Base position size
        base_size = self.parameters['base_position_size']
        
        # Adjust for quality
        if quality == EntryQuality.PREMIUM and self.parameters['premium_entry_boost'] > 1.0:
            base_size *= self.parameters['premium_entry_boost']
        elif quality == EntryQuality.MARGINAL:
            base_size *= 0.7  # Reduce size for marginal setups
            
        # Adjust for strength
        base_size *= (0.5 + 0.5 * strength)  # Scale from 50-100% based on strength
        
        # Check for daily target progress
        daily_target = self.parameters['daily_pip_target']
        if self.current_day_pip_total > 0:
            # If we're making good progress, reduce risk slightly
            progress_pct = min(0.8, self.current_day_pip_total / daily_target)
            base_size *= (1.0 - progress_pct * 0.3)  # Reduce by up to 24% at 80% of target
            
        # Apply any symbol-specific adjustments if needed
        # For example, adjust for higher volatility pairs
        if symbol in ['GBPJPY', 'AUDJPY'] and 'JPY' in symbol:
            base_size *= 0.8  # Reduce size for volatile JPY crosses
            
        return base_size
    
    def on_trade_closed(self, trade_data: Dict[str, Any]):
        """Handle a closed trade event
        
        Args:
            trade_data: Dictionary with trade details
        """
        # Check if this is one of our trades
        if trade_data.get('strategy_name') != self.__class__.__name__:
            return
            
        # Extract trade details
        symbol = trade_data.get('symbol')
        direction = trade_data.get('direction')
        entry_price = trade_data.get('entry_price')
        exit_price = trade_data.get('exit_price')
        pips_gained = trade_data.get('pips')
        
        if not symbol or not direction or pips_gained is None:
            logger.warning(f"Incomplete trade data received: {trade_data}")
            return
            
        # Update pip counters
        self.current_day_pip_total += pips_gained
        self.weekly_pips += pips_gained
        self.monthly_pips += pips_gained
        
        # Update session pips if we know which session
        current_session = self.get_current_session(pd.Timestamp.now())
        if current_session in self.session_pips:
            self.session_pips[current_session] += pips_gained
            
        # Update symbol-specific achieved values
        if symbol not in self.daily_achieved:
            self.daily_achieved[symbol] = 0
        self.daily_achieved[symbol] += pips_gained
        
        # Remove from active positions if present
        if trade_data.get('id') in self.active_positions:
            del self.active_positions[trade_data.get('id')]
            
        # Check if we've hit the daily target
        self._check_daily_status()
        
        # Log the trade result
        logger.info(f"Pips-a-Day trade closed: {symbol} {direction} for {pips_gained:.1f} pips. "  
                   f"Daily total: {self.current_day_pip_total:.1f} pips, target: {self.parameters['daily_pip_target']} pips")
        
        # Publish pip update event
        EventBus.get_instance().publish('pips_day_update', {
            'strategy': self.__class__.__name__,
            'symbol': symbol,
            'pips_gained': pips_gained,
            'daily_total': self.current_day_pip_total,
            'daily_target': self.parameters['daily_pip_target'],
            'target_status': self.current_day_status.value,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    def check_scale_out_levels(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp):
        """Check if any scale-out levels have been reached"""
        if not self.parameters['scale_out_enabled'] or not self.active_positions:
            return
            
        # For each active position
        for signal_id, position in list(self.active_positions.items()):
            symbol = position['symbol']
            direction = position['direction']
            entry_price = position['entry_price']
            
            # Skip if we don't have data for this symbol
            if symbol not in data:
                continue
                
            # Get data for our execution timeframe
            primary_tf = self.parameters['primary_execution_timeframe']
            if primary_tf not in data[symbol]:
                continue
                
            # Get current price
            ohlcv_data = data[symbol][primary_tf]
            if ohlcv_data.empty:
                continue
                
            current_price = ohlcv_data['close'].iloc[-1]
            
            # No scale out levels defined or already triggered
            if not self.parameters['scale_out_levels'] or not position.get('scale_out_status'):
                continue
                
            # Check each scale out level
            for i, level in enumerate(self.parameters['scale_out_levels']):
                # Skip if already triggered
                if position['scale_out_status'][i]:
                    continue
                    
                # Calculate target price
                if direction == TradeDirection.LONG:
                    target_price = entry_price + (level * (position['take_profit'] - entry_price))
                    # Check if we've reached the level
                    if current_price >= target_price:
                        position['scale_out_status'][i] = True
                        self._trigger_scale_out(signal_id, symbol, direction, current_price, level, i)
                else:  # SHORT
                    target_price = entry_price - (level * (entry_price - position['take_profit']))
                    # Check if we've reached the level
                    if current_price <= target_price:
                        position['scale_out_status'][i] = True
                        self._trigger_scale_out(signal_id, symbol, direction, current_price, level, i)
    
    def _trigger_scale_out(self, signal_id: str, symbol: str, direction: TradeDirection, 
                           current_price: float, level: float, level_index: int):
        """Handle a scale-out trigger"""
        # Get scale out percent
        scale_amount = self.parameters['scale_out_percents'][level_index]
        
        # Calculate pip gain
        position = self.active_positions[signal_id]
        entry_price = position['entry_price']
        
        if 'JPY' in symbol:
            pip_factor = 100
        else:
            pip_factor = 10000
            
        if direction == TradeDirection.LONG:
            pips_gained = (current_price - entry_price) * pip_factor
        else:  # SHORT
            pips_gained = (entry_price - current_price) * pip_factor
            
        # Update pip counters (proportional to scale amount)
        pips_this_exit = pips_gained * (scale_amount / 100)
        self.current_day_pip_total += pips_this_exit
        
        if symbol not in self.daily_achieved:
            self.daily_achieved[symbol] = 0
        self.daily_achieved[symbol] += pips_this_exit
        
        # Log the scale out
        logger.info(f"Pips-a-Day scale out triggered: {symbol} {direction.name} at {level*100:.0f}% target "  
                   f"for {pips_this_exit:.1f} pips. Daily total: {self.current_day_pip_total:.1f} pips")
        
        # Publish scale out event
        EventBus.get_instance().publish('pips_day_scale_out', {
            'strategy': self.__class__.__name__,
            'signal_id': signal_id,
            'symbol': symbol,
            'direction': direction.name,
            'scale_level': level * 100,
            'scale_amount': scale_amount,
            'pips_gained': pips_this_exit,
            'daily_total': self.current_day_pip_total,
            'daily_target': self.parameters['daily_pip_target'],
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        # Check if we've hit the daily target
        self._check_daily_status()
    
    def calculate_regime_compatibility(self, market_regime: MarketRegime) -> float:
        """Calculate compatibility score for a given market regime
        
        Args:
            market_regime: MarketRegime enum value
            
        Returns:
            Compatibility score 0.0-1.0
        """
        # Compatibility matrix
        compatibility = {
            MarketRegime.TRENDING_BULL: 0.6,    # Good in trending
            MarketRegime.TRENDING_BEAR: 0.6,    # Good in trending
            MarketRegime.RANGE_BOUND: 0.9,      # Excellent in range-bound
            MarketRegime.BREAKOUT: 0.5,         # Moderate in breakout
            MarketRegime.VOLATILE: 0.75,        # Very good in volatile
            MarketRegime.CHOPPY: 0.7,           # Good in choppy
            MarketRegime.LOW_VOLATILITY: 0.85,  # Very good in low volatility
            MarketRegime.HIGH_VOLATILITY: 0.65, # Good in high volatility
            MarketRegime.NORMAL: 0.9,           # Excellent in normal conditions
        }
        
        return compatibility.get(market_regime, 0.5)  # Default 0.5 for unknown regimes
    
    def optimize_for_regime(self, market_regime: MarketRegime) -> None:
        """Optimize strategy parameters for the given market regime
        
        Args:
            market_regime: Market regime to optimize for
        """
        if market_regime == MarketRegime.TRENDING_BULL or market_regime == MarketRegime.TRENDING_BEAR:
            # In trending, increase reward-risk and scale out more aggressively
            self.parameters['min_reward_risk_ratio'] = 2.0
            self.parameters['scale_out_levels'] = [0.33, 0.67, 0.9]
            self.parameters['scale_out_percents'] = [25, 25, 50]
            self.parameters['target_volatility_factor'] = 1.5
            
        elif market_regime == MarketRegime.RANGE_BOUND or market_regime == MarketRegime.LOW_VOLATILITY:
            # In range-bound, take smaller profits with higher win rate
            self.parameters['min_reward_risk_ratio'] = 1.2
            self.parameters['scale_out_levels'] = [0.25, 0.5, 0.75]
            self.parameters['scale_out_percents'] = [25, 25, 50]
            self.parameters['daily_pip_target'] *= 0.8  # More conservative target
            self.parameters['target_volatility_factor'] = 1.0
            
        elif market_regime == MarketRegime.VOLATILE or market_regime == MarketRegime.HIGH_VOLATILITY:
            # In volatile, use wider stops, higher targets
            self.parameters['min_reward_risk_ratio'] = 1.8
            self.parameters['scale_out_levels'] = [0.45, 0.8, 1.0]
            self.parameters['scale_out_percents'] = [30, 30, 40]
            self.parameters['atr_stop_loss_factor'] = 1.5
            self.parameters['target_volatility_factor'] = 2.0
            
        elif market_regime == MarketRegime.BREAKOUT:
            # In breakout, focus on catching strong moves
            self.parameters['min_reward_risk_ratio'] = 2.5
            self.parameters['scale_out_levels'] = [0.5, 0.8, 1.1]
            self.parameters['scale_out_percents'] = [20, 30, 50]
            self.parameters['target_volatility_factor'] = 2.0
            
        elif market_regime == MarketRegime.CHOPPY:
            # In choppy, more conservative with faster exits
            self.parameters['min_reward_risk_ratio'] = 1.3
            self.parameters['scale_out_levels'] = [0.2, 0.4, 0.6]
            self.parameters['scale_out_percents'] = [40, 30, 30]
            self.parameters['target_volatility_factor'] = 0.8
            
        elif market_regime == MarketRegime.NORMAL:
            # In normal conditions, use balanced approach
            self.parameters['min_reward_risk_ratio'] = 1.5
            self.parameters['scale_out_levels'] = [0.3, 0.6, 0.9]
            self.parameters['scale_out_percents'] = [20, 30, 50]
            self.parameters['target_volatility_factor'] = 1.25
            
        # Log optimization
        logger.info(f"Optimized Pips-a-Day strategy for {market_regime.name} regime")
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the strategy"""
        metrics = super().get_performance_metrics()
        
        # Add pips-a-day specific metrics
        metrics.update({
            'daily_pip_target': self.parameters['daily_pip_target'],
            'current_day_pips': self.current_day_pip_total,
            'weekly_pips': self.weekly_pips,
            'monthly_pips': self.monthly_pips,
            'target_achievement_rate': len([x for x in self.daily_stats.values() 
                                         if x['target_status'] in 
                                         [TargetStatus.TARGET_REACHED.value, TargetStatus.TARGET_EXCEEDED.value]]) / 
                                   max(1, len(self.daily_stats)),
            'daily_stats': self.daily_stats,
            'session_pips': self.session_pips
        })
        
        return metrics
