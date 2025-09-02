#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short Selling Strategy

A specialized strategy focused on short selling opportunities with enhanced
risk management for the unique challenges of short positions.
"""

import logging
import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.core.signals import Signal, SignalType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.position import Position, PositionStatus
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="ShortSellingStrategy",
    market_type="stocks",
    description="A strategy for short selling overvalued stocks using technical, fundamental, and sentiment data",
    timeframes=["15m", "1h", "4h", "1d", "1w"],
    parameters={
        "entry_approach": {"description": "Entry approach (technical, fundamental, combined)", "type": "string"},
        "risk_control_level": {"description": "Risk control strictness (conservative, moderate, aggressive)", "type": "string"},
        "min_short_interest": {"description": "Minimum short interest percentage for consideration", "type": "float"},
        "max_borrow_rate": {"description": "Maximum borrow rate acceptable for shorting", "type": "float"}
    }
)
class ShortSellingStrategy(StocksBaseStrategy):
    """
    Short Selling Strategy
    
    A specialized strategy focused on identifying optimal short selling opportunities
    with enhanced risk management tailored to short positions.
    
    Features:
    - Multiple short criteria (technical, fundamental, relative strength)
    - Short squeeze detection and risk management
    - Borrow availability monitoring
    - Hard-to-borrow fee consideration
    - Advanced risk controls specific to short selling
    - Multiple exit strategies (time-based, profit target, trailing stop)
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Short Selling Strategy.
        
        Args:
            session: StocksSession for the specific symbol and timeframe
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize the base strategy
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Entry parameters
            'strategy_mode': 'technical',      # 'technical', 'fundamental', or 'combined'
            'rsi_overbought': 70,              # RSI threshold for overbought condition
            'rsi_period': 14,                  # RSI calculation period
            'min_down_days': 2,                # Minimum consecutive down days for entry
            'require_bearish_pattern': True,   # Require bearish candlestick pattern
            
            # Short squeeze protection
            'enable_short_squeeze_protection': True,  # Enable protection against short squeezes
            'max_short_interest': 20,          # Maximum short interest percentage (avoid heavily shorted)
            'min_float': 20000000,             # Minimum float size in shares
            'days_to_cover_threshold': 5,      # Maximum days to cover for safe shorting
            
            # Risk management
            'max_risk_per_trade_percent': 1.0, # Max risk per trade as % of account
            'position_size_factor': 0.8,       # Smaller size factor for short positions (vs long)
            'tight_stop_percent': 3.0,         # Initial tight stop percentage
            'max_loss_percent': 7.0,           # Maximum loss percentage before mandatory exit
            'max_positions': 5,                # Maximum concurrent short positions
            
            # Exit parameters
            'profit_target_percent': 15.0,     # Target profit percentage
            'max_hold_period_days': 20,        # Maximum days to hold short position
            'trailing_stop_activation': 5.0,   # Profit percentage to activate trailing stop
            'trailing_stop_percent': 3.0,      # Trailing stop percentage once activated
            
            # Advanced parameters
            'avoid_earnings_days_before': 5,   # Avoid shorting this many days before earnings
            'avoid_earnings_days_after': 2,    # Avoid shorting this many days after earnings
            'respect_trend': True,             # Only short in bearish broader trends
            'avoid_dividend_dates': True,      # Avoid shorting before dividend dates
            'borrow_rate_threshold': 5.0,      # Maximum borrow rate percentage to consider
            'require_catalyst': False,         # Require a bearish catalyst for entry
            
            # Fundamental parameters (if used)
            'pe_ratio_threshold': 50,          # High P/E ratio threshold for overvaluation
            'debt_to_equity_threshold': 2.0,   # High debt to equity threshold
            'negative_earnings_growth': True,  # Prefer companies with negative earnings growth
            
            # Market regime parameters
            'bear_market_bias': 1.5,           # Position size multiplier in bear markets
            'avoid_bull_markets': False,       # Avoid shorting in strong bull markets
            
            # Monitoring parameters
            'monitor_sec_filings': True,       # Monitor SEC filings for material events
            'monitor_short_volume': True,      # Monitor daily short volume data
            
            # External data
            'earnings_calendar': {},           # Dict of symbols -> upcoming earnings dates
            'market_regime': 'neutral',        # Current market regime: 'bull', 'bear', or 'neutral'
            'sector_performance': {},          # Sector performance data
            'borrow_data': {},                 # Short borrow availability and fee data
        }
        
        # Update parameters with defaults for any missing keys
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Strategy state
        self.short_signals = {}              # Track active short signals
        self.short_squeeze_watchlist = set() # Symbols at risk of short squeeze
        self.borrow_status = {}              # Track borrow status for symbols
        self.days_held = {}                  # Track days positions have been held
        self.sector_trend = 'neutral'        # Track sector trend
        self.total_short_positions = 0       # Count of current short positions
        self.tickers_to_avoid = set()        # Tickers to avoid shorting
        
        # Register for market events if event bus is available
        if self.event_bus:
            self.register_for_events(self.event_bus)
        
        logger.info(f"Initialized Short Selling Strategy for {session.symbol} on {session.timeframe}")
    
    def register_for_events(self, event_bus: EventBus) -> None:
        """
        Register for relevant market events.
        
        Args:
            event_bus: EventBus to register with
        """
        # First register for common events via base class
        super().register_for_events(event_bus)
        
        # Register for short-selling specific events
        event_bus.subscribe(EventType.MARKET_OPEN, self._on_market_open)
        event_bus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self._on_earnings_announcement)
        event_bus.subscribe(EventType.SEC_FILING, self._on_sec_filing)
        event_bus.subscribe(EventType.DIVIDEND_ANNOUNCEMENT, self._on_dividend_announcement)
        event_bus.subscribe(EventType.HARD_TO_BORROW_UPDATE, self._on_borrow_update)
        event_bus.subscribe(EventType.SHORT_INTEREST_UPDATE, self._on_short_interest_update)
        
        logger.debug(f"Short Selling Strategy registered for events")
    
    def _on_market_open(self, event: Event) -> None:
        """
        Handle market open event.
        
        Update borrow status and check existing positions.
        
        Args:
            event: Market open event
        """
        # Update day counters for positions
        for position in self.positions:
            if position.symbol == self.session.symbol and position.status == PositionStatus.OPEN:
                if position.id in self.days_held:
                    self.days_held[position.id] += 1
                else:
                    self.days_held[position.id] = 1
                
                # Check if position has been held too long
                max_days = self.parameters['max_hold_period_days']
                if self.days_held[position.id] >= max_days:
                    logger.info(f"Position {position.id} held for {self.days_held[position.id]} days, max is {max_days}")
                    self._close_position(position.id, "Max hold time reached")
        
        # Check for short opportunities at market open
        self._check_for_short_opportunities()
    
    def _on_earnings_announcement(self, event: Event) -> None:
        """
        Handle earnings announcement events.
        
        Avoid shorting around earnings dates and possibly close positions.
        
        Args:
            event: Earnings announcement event
        """
        # Check if the event data contains our symbol
        symbols = event.data.get('symbols', [])
        symbol = self.session.symbol
        
        if symbol in symbols:
            # Get days to announcement
            days_to_announcement = event.data.get('days_to_announcement', 0)
            
            # Add to avoid list if earnings are coming up
            if days_to_announcement <= self.parameters['avoid_earnings_days_before']:
                logger.info(f"Adding {symbol} to avoid list due to upcoming earnings")
                self.tickers_to_avoid.add(symbol)
                
                # Close positions if earnings are very soon
                if days_to_announcement <= 2:
                    for position in self.positions:
                        if position.symbol == symbol and position.status == PositionStatus.OPEN:
                            self._close_position(position.id, "Closing before earnings")
            
            # Recent earnings - keep in avoid list for a few days after
            if days_to_announcement == 0:
                logger.info(f"Earnings today for {symbol}, avoiding for {self.parameters['avoid_earnings_days_after']} days")
                self.tickers_to_avoid.add(symbol)
                
                # Close any open positions
                for position in self.positions:
                    if position.symbol == symbol and position.status == PositionStatus.OPEN:
                        self._close_position(position.id, "Earnings announcement")
    
    def _on_sec_filing(self, event: Event) -> None:
        """
        Handle SEC filing events.
        
        Monitor for material events that might affect short positions.
        
        Args:
            event: SEC filing event
        """
        if not self.parameters['monitor_sec_filings']:
            return
            
        # Check if the filing is for our symbol
        symbol = event.data.get('symbol')
        if symbol != self.session.symbol:
            return
            
        # Check filing type and content for relevant information
        filing_type = event.data.get('filing_type')
        
        # List of potentially bullish filing types (risk for shorts)
        bullish_filings = ['8-K_ITEM5.02', '8-K_ITEM7.01', '8-K_ITEM8.01']
        
        # Check for material events that might be risky for shorts
        if filing_type in bullish_filings:
            logger.warning(f"Potentially bullish SEC filing {filing_type} for {symbol}")
            
            # Check keywords in the filing content
            keywords = ['acquisition', 'merger', 'buyout', 'takeover', 'positive', 'exceeds', 
                        'above expectations', 'guidance increase', 'buyback']
            
            filing_content = event.data.get('content', '').lower()
            found_keywords = [k for k in keywords if k in filing_content]
            
            if found_keywords:
                logger.warning(f"Bullish keywords found in filing: {found_keywords}")
                
                # Close positions on potentially serious bullish events
                for position in self.positions:
                    if position.symbol == symbol and position.status == PositionStatus.OPEN:
                        self._close_position(position.id, f"Bullish SEC filing: {filing_type}")
    
    def _on_dividend_announcement(self, event: Event) -> None:
        """
        Handle dividend announcement events.
        
        Avoid shorting before ex-dividend dates (short seller must pay the dividend).
        
        Args:
            event: Dividend announcement event
        """
        if not self.parameters['avoid_dividend_dates']:
            return
            
        # Check if the event is for our symbol
        symbol = event.data.get('symbol')
        if symbol != self.session.symbol:
            return
            
        # Get ex-dividend date
        ex_date = event.data.get('ex_date')
        if not ex_date:
            return
            
        # Convert to datetime if it's a string
        if isinstance(ex_date, str):
            try:
                ex_date = datetime.strptime(ex_date, '%Y-%m-%d').date()
            except ValueError:
                return
                
        # Calculate days to ex-date
        today = datetime.now().date()
        days_to_ex = (ex_date - today).days
        
        # Add to avoid list if ex-date is coming up
        if days_to_ex >= 0 and days_to_ex <= 5:
            logger.info(f"Adding {symbol} to avoid list due to upcoming ex-dividend date")
            self.tickers_to_avoid.add(symbol)
            
            # Close positions if ex-date is very soon
            if days_to_ex <= 2:
                for position in self.positions:
                    if position.symbol == symbol and position.status == PositionStatus.OPEN:
                        self._close_position(position.id, "Closing before ex-dividend date")
    
    def _on_borrow_update(self, event: Event) -> None:
        """
        Handle updates to borrow availability and rates.
        
        Update internal state and possibly close positions if borrow rates spike.
        
        Args:
            event: Borrow update event
        """
        # Check if the update is for our symbol
        symbol = event.data.get('symbol')
        if symbol != self.session.symbol:
            return
            
        # Extract borrow data
        borrow_rate = event.data.get('borrow_rate', 0.0)
        shares_available = event.data.get('shares_available', 0)
        hard_to_borrow = event.data.get('hard_to_borrow', False)
        
        # Update our borrow status
        self.borrow_status[symbol] = {
            'borrow_rate': borrow_rate,
            'shares_available': shares_available,
            'hard_to_borrow': hard_to_borrow,
            'updated_at': datetime.now()
        }
        
        # Log the update
        logger.info(f"Borrow update for {symbol}: rate={borrow_rate}%, " +
                   f"shares_available={shares_available}, hard_to_borrow={hard_to_borrow}")
        
        # Check if borrow rate exceeds our threshold
        rate_threshold = self.parameters['borrow_rate_threshold']
        if borrow_rate > rate_threshold:
            logger.warning(f"Borrow rate for {symbol} ({borrow_rate}%) exceeds threshold ({rate_threshold}%)")
            
            # Add to avoid list
            self.tickers_to_avoid.add(symbol)
            
            # Close positions if borrow rate spikes significantly
            if borrow_rate > rate_threshold * 2:
                for position in self.positions:
                    if position.symbol == symbol and position.status == PositionStatus.OPEN:
                        self._close_position(position.id, f"Borrow rate spike: {borrow_rate}%")
    
    def _on_short_interest_update(self, event: Event) -> None:
        """
        Handle updates to short interest data.
        
        Update short squeeze risk assessment.
        
        Args:
            event: Short interest update event
        """
        # Check if the update is for our symbol
        symbol = event.data.get('symbol')
        if symbol != self.session.symbol:
            return
            
        # Extract short interest data
        short_interest_percent = event.data.get('short_interest_percent', 0.0)
        days_to_cover = event.data.get('days_to_cover', 0.0)
        float_shares = event.data.get('float_shares', 0)
        
        # Check for short squeeze risk
        max_short_interest = self.parameters['max_short_interest']
        min_float = self.parameters['min_float']
        days_to_cover_threshold = self.parameters['days_to_cover_threshold']
        
        short_squeeze_risk = False
        
        # Assess short squeeze risk based on multiple factors
        if short_interest_percent > max_short_interest:
            logger.warning(f"High short interest for {symbol}: {short_interest_percent}% > {max_short_interest}%")
            short_squeeze_risk = True
        
        if float_shares < min_float:
            logger.warning(f"Small float for {symbol}: {float_shares} < {min_float}")
            short_squeeze_risk = True
            
        if days_to_cover > days_to_cover_threshold:
            logger.warning(f"High days to cover for {symbol}: {days_to_cover} > {days_to_cover_threshold}")
            short_squeeze_risk = True
            
        # Update short squeeze watchlist
        if short_squeeze_risk:
            logger.warning(f"Adding {symbol} to short squeeze watchlist")
            self.short_squeeze_watchlist.add(symbol)
            
            # Add to avoid list
            self.tickers_to_avoid.add(symbol)
            
            # Close positions if very high risk
            if short_interest_percent > max_short_interest * 1.5:
                for position in self.positions:
                    if position.symbol == symbol and position.status == PositionStatus.OPEN:
                        self._close_position(position.id, f"High short squeeze risk: {short_interest_percent}% SI")
        else:
            # Remove from watchlist if previously added
            if symbol in self.short_squeeze_watchlist:
                logger.info(f"Removing {symbol} from short squeeze watchlist")
                self.short_squeeze_watchlist.remove(symbol)
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated event.
        
        Check for exit signals on existing positions and update indicators.
        
        Args:
            event: Market data updated event
        """
        # Let the base class handle common functionality first
        super()._on_market_data_updated(event)
        
        # Check if the event data is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Update our indicators if we have enough data
        if len(self.market_data) > 20:
            self.indicators = self.calculate_indicators(self.market_data)
            
        # Check for position exit signals
        self.check_exit_conditions()
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed event.
        
        Check for new short opportunities.
        
        Args:
            event: Timeframe completed event
        """
        # Let the base class handle common functionality first
        super()._on_timeframe_completed(event)
        
        # Check if the event data is for our symbol and timeframe
        if (event.data.get('symbol') != self.session.symbol or 
            event.data.get('timeframe') != self.session.timeframe):
            return
        
        # Check for short opportunities
        self._check_for_short_opportunities()
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators for short selling analysis.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        if len(data) < 20:  # Need at least 20 bars for meaningful indicators
            return {}
            
        indicators = {}
        
        # Calculate Relative Strength Index (RSI)
        try:
            rsi_period = self.parameters['rsi_period']
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
        
        # Calculate Moving Averages
        try:
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
            indicators['sma_200'] = data['close'].rolling(window=200).mean()
            
            # Calculate Moving Average slopes (rate of change)
            if len(data) >= 25:
                indicators['sma_20_slope'] = indicators['sma_20'].diff(5) / indicators['sma_20'].shift(5) * 100
        except Exception as e:
            logger.error(f"Error calculating moving averages: {e}")
        
        # Calculate Bollinger Bands
        try:
            window = 20
            std_dev = 2
            
            sma = data['close'].rolling(window=window).mean()
            rolling_std = data['close'].rolling(window=window).std()
            
            indicators['bollinger_upper'] = sma + (rolling_std * std_dev)
            indicators['bollinger_lower'] = sma - (rolling_std * std_dev)
            indicators['bollinger_mid'] = sma
            
            # Calculate %B (position within Bollinger Bands)
            indicators['bollinger_pct_b'] = (data['close'] - indicators['bollinger_lower']) / \
                                           (indicators['bollinger_upper'] - indicators['bollinger_lower'])
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
        
        # Calculate Average True Range (ATR) for volatility assessment
        try:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr'] = tr.rolling(window=14).mean()
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
        
        # Calculate MACD
        try:
            exp1 = data['close'].ewm(span=12, adjust=False).mean()
            exp2 = data['close'].ewm(span=26, adjust=False).mean()
            indicators['macd'] = exp1 - exp2
            indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
        
        # Calculate On-Balance Volume (OBV)
        try:
            obv = pd.Series(0, index=data.index)
            for i in range(1, len(data)):
                if data['close'].iloc[i] > data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
                elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            indicators['obv'] = obv
            
            # Calculate OBV moving average for trend detection
            if len(data) >= 20:
                indicators['obv_ma'] = obv.rolling(window=20).mean()
        except Exception as e:
            logger.error(f"Error calculating OBV: {e}")
            
        # Calculate Stochastic Oscillator
        try:
            k_period = 14
            d_period = 3
            
            low_min = data['low'].rolling(window=k_period).min()
            high_max = data['high'].rolling(window=k_period).max()
            
            # Fast %K
            indicators['stoch_k'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
            
            # Slow %D (3-day SMA of %K)
            indicators['stoch_d'] = indicators['stoch_k'].rolling(window=d_period).mean()
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {e}")
        
        # Calculate bearish pattern detection
        if len(data) >= 3 and self.parameters['require_bearish_pattern']:
            indicators['bearish_pattern'] = self._detect_bearish_patterns(data)
        
        return indicators
    
    def _detect_bearish_patterns(self, data: pd.DataFrame) -> pd.Series:
        """
        Detect bearish candlestick patterns.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Series with boolean values indicating presence of bearish patterns
        """
        # Initialize result series with False values
        bearish_patterns = pd.Series(False, index=data.index)
        
        # Need at least 3 bars for pattern detection
        if len(data) < 3:
            return bearish_patterns
        
        # Define pattern detection thresholds
        body_size_threshold = 0.5  # Ratio of body to range for significant candlestick
        doji_threshold = 0.1      # Ratio for doji (very small body)
        
        # Calculate candlestick properties
        open_prices = data['open']
        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        
        price_range = high_prices - low_prices
        body_size = (close_prices - open_prices).abs()
        body_to_range_ratio = body_size / price_range
        is_bearish = close_prices < open_prices
        
        # Pattern: Bearish Engulfing
        for i in range(1, len(data)):
            if (is_bearish.iloc[i] and not is_bearish.iloc[i-1] and 
                close_prices.iloc[i] < open_prices.iloc[i-1] and 
                open_prices.iloc[i] > close_prices.iloc[i-1] and
                body_to_range_ratio.iloc[i] > body_size_threshold):
                bearish_patterns.iloc[i] = True
        
        # Pattern: Evening Star (3-bar pattern)
        for i in range(2, len(data)):
            # First day: bullish candle
            first_bullish = close_prices.iloc[i-2] > open_prices.iloc[i-2]
            first_significant = body_to_range_ratio.iloc[i-2] > body_size_threshold
            
            # Second day: small body (doji or small candle) gapping up
            second_small_body = body_to_range_ratio.iloc[i-1] < doji_threshold * 2
            second_gap_up = open_prices.iloc[i-1] > close_prices.iloc[i-2]
            
            # Third day: bearish candle that closes well into first day's body
            third_bearish = close_prices.iloc[i] < open_prices.iloc[i]
            third_close_into_first = close_prices.iloc[i] < (open_prices.iloc[i-2] + 
                                                          (close_prices.iloc[i-2] - open_prices.iloc[i-2]) / 2)
            
            if (first_bullish and first_significant and second_small_body and 
                second_gap_up and third_bearish and third_close_into_first):
                bearish_patterns.iloc[i] = True
        
        # Pattern: Shooting Star
        for i in range(1, len(data)):
            upper_shadow = high_prices.iloc[i] - max(open_prices.iloc[i], close_prices.iloc[i])
            lower_shadow = min(open_prices.iloc[i], close_prices.iloc[i]) - low_prices.iloc[i]
            
            if (upper_shadow > body_size.iloc[i] * 2 and 
                lower_shadow < body_size.iloc[i] / 2 and
                body_to_range_ratio.iloc[i] < 0.3 and
                open_prices.iloc[i] > close_prices.iloc[i-1]):
                bearish_patterns.iloc[i] = True
        
        # Pattern: Dark Cloud Cover
        for i in range(1, len(data)):
            first_bullish = close_prices.iloc[i-1] > open_prices.iloc[i-1]
            second_bearish = close_prices.iloc[i] < open_prices.iloc[i]
            gap_up_open = open_prices.iloc[i] > close_prices.iloc[i-1]
            deep_close = close_prices.iloc[i] < (open_prices.iloc[i-1] + 
                                           (close_prices.iloc[i-1] - open_prices.iloc[i-1]) / 2)
            
            if first_bullish and second_bearish and gap_up_open and deep_close:
                bearish_patterns.iloc[i] = True
        
        return bearish_patterns
    
    def _check_for_short_opportunities(self) -> None:
        """
        Check for short selling opportunities based on technical and fundamental criteria.
        """
        # Skip if we're already at max positions
        if self.total_short_positions >= self.parameters['max_positions']:
            logger.info(f"Maximum short positions reached: {self.total_short_positions}")
            return
        
        # Skip if symbol is on our avoid list
        symbol = self.session.symbol
        if symbol in self.tickers_to_avoid:
            logger.debug(f"Skipping {symbol} - on avoid list")
            return
        
        # Skip if symbol is on short squeeze watchlist
        if symbol in self.short_squeeze_watchlist:
            logger.debug(f"Skipping {symbol} - on short squeeze watchlist")
            return
        
        # Check borrow availability if we have that data
        if symbol in self.borrow_status:
            borrow_data = self.borrow_status[symbol]
            
            # Skip if hard to borrow
            if borrow_data.get('hard_to_borrow', False):
                logger.debug(f"Skipping {symbol} - hard to borrow")
                return
                
            # Skip if borrow rate is too high
            if borrow_data.get('borrow_rate', 0) > self.parameters['borrow_rate_threshold']:
                logger.debug(f"Skipping {symbol} - borrow rate too high: {borrow_data['borrow_rate']}%")
                return
                
            # Skip if not enough shares available to borrow
            if borrow_data.get('shares_available', 0) <= 0:
                logger.debug(f"Skipping {symbol} - no shares available to borrow")
                return
        
        # Skip if we're already in a position for this symbol
        for position in self.positions:
            if position.symbol == symbol and position.status == PositionStatus.OPEN:
                logger.debug(f"Already in position for {symbol}, skipping")
                return
        
        # Make sure we have market data
        if len(self.market_data) < 20:  # Need sufficient data for analysis
            return
        
        # Update indicators if not already done
        if not hasattr(self, 'indicators') or not self.indicators:
            self.indicators = self.calculate_indicators(self.market_data)
        
        # Check entry conditions based on strategy mode
        strategy_mode = self.parameters['strategy_mode']
        short_signal = False
        signal_metadata = {}
        
        if strategy_mode in ['technical', 'combined']:
            short_signal, tech_metadata = self._check_technical_short_entry()
            signal_metadata.update(tech_metadata)
            
        if strategy_mode in ['fundamental', 'combined'] and (not short_signal or strategy_mode == 'combined'):
            fund_signal, fund_metadata = self._check_fundamental_short_entry()
            short_signal = short_signal or fund_signal
            signal_metadata.update(fund_metadata)
        
        # If we have a signal, generate a short position
        if short_signal:
            self._generate_short_signal(signal_metadata)
    
    def _check_technical_short_entry(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for technical short entry signals.
        
        Returns:
            Tuple of (signal_generated, metadata)
        """
        # Default to no signal
        short_signal = False
        metadata = {
            'trigger_type': 'technical',
            'signals': []
        }
        
        # Check for overbought RSI
        if 'rsi' in self.indicators and len(self.indicators['rsi']) > 0:
            current_rsi = self.indicators['rsi'].iloc[-1]
            rsi_overbought = self.parameters['rsi_overbought']
            
            if not pd.isna(current_rsi) and current_rsi > rsi_overbought:
                short_signal = True
                metadata['signals'].append(f"RSI overbought: {current_rsi:.1f} > {rsi_overbought}")
                metadata['rsi'] = current_rsi
        
        # Check for price above upper Bollinger Band
        if 'bollinger_upper' in self.indicators and 'bollinger_pct_b' in self.indicators:
            pct_b = self.indicators['bollinger_pct_b'].iloc[-1]
            
            if not pd.isna(pct_b) and pct_b > 1.0:  # Price above upper band
                short_signal = True
                metadata['signals'].append(f"Price above upper Bollinger Band: {pct_b:.2f}")
                metadata['bollinger_pct_b'] = pct_b
        
        # Check for bearish MACD crossover
        if 'macd' in self.indicators and 'macd_signal' in self.indicators:
            current_macd = self.indicators['macd'].iloc[-1]
            current_signal = self.indicators['macd_signal'].iloc[-1]
            prev_macd = self.indicators['macd'].iloc[-2] if len(self.indicators['macd']) > 1 else np.nan
            prev_signal = self.indicators['macd_signal'].iloc[-2] if len(self.indicators['macd_signal']) > 1 else np.nan
            
            if (not pd.isna(current_macd) and not pd.isna(current_signal) and 
                not pd.isna(prev_macd) and not pd.isna(prev_signal)):
                # Bearish crossover (MACD crosses below signal line)
                if prev_macd > prev_signal and current_macd < current_signal:
                    short_signal = True
                    metadata['signals'].append("Bearish MACD crossover")
                    metadata['macd_crossover'] = True
        
        # Check for consecutive down days
        min_down_days = self.parameters['min_down_days']
        if len(self.market_data) >= min_down_days + 1:
            down_days = 0
            for i in range(1, min_down_days + 1):
                if self.market_data['close'].iloc[-i] < self.market_data['close'].iloc[-i-1]:
                    down_days += 1
            
            if down_days >= min_down_days:
                short_signal = True
                metadata['signals'].append(f"Consecutive down days: {down_days}")
                metadata['down_days'] = down_days
        
        # Check for bearish candlestick patterns
        if self.parameters['require_bearish_pattern'] and 'bearish_pattern' in self.indicators:
            current_pattern = self.indicators['bearish_pattern'].iloc[-1]
            
            if current_pattern:
                short_signal = True
                metadata['signals'].append("Bearish candlestick pattern")
                metadata['bearish_pattern'] = True
        
        # Check for bearish trend based on moving averages
        if self.parameters['respect_trend'] and 'sma_20' in self.indicators and 'sma_50' in self.indicators:
            sma_20 = self.indicators['sma_20'].iloc[-1]
            sma_50 = self.indicators['sma_50'].iloc[-1]
            
            if not pd.isna(sma_20) and not pd.isna(sma_50) and sma_20 < sma_50:
                metadata['bearish_trend'] = True
                metadata['signals'].append("Bearish trend (20 SMA < 50 SMA)")
            else:
                # If we require a bearish trend and don't have one, cancel the signal
                if self.parameters['respect_trend']:
                    short_signal = False
                    metadata['signals'].append("Signal rejected: not in bearish trend")
        
        # Calculate signal strength based on number of confirming indicators
        signal_count = len(metadata['signals'])
        metadata['signal_strength'] = min(0.9, signal_count / 5)  # Cap at 0.9
        
        # For a technical short, we need at least 2 confirming signals
        if signal_count < 2:
            short_signal = False
            metadata['signals'].append(f"Not enough confirming signals: {signal_count} < 2")
        
        return short_signal, metadata
    
    def _check_fundamental_short_entry(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for fundamental short entry signals.
        
        Returns:
            Tuple of (signal_generated, metadata)
        """
        # Default to no signal
        short_signal = False
        metadata = {
            'trigger_type': 'fundamental',
            'signals': []
        }
        
        # This would typically connect to a fundamental data service
        # For now, use placeholder data from parameters
        
        # Check P/E ratio for overvaluation
        pe_threshold = self.parameters['pe_ratio_threshold']
        symbol_data = self.parameters.get('fundamental_data', {}).get(self.session.symbol, {})
        
        pe_ratio = symbol_data.get('pe_ratio', None)
        if pe_ratio and pe_ratio > pe_threshold:
            short_signal = True
            metadata['signals'].append(f"High P/E ratio: {pe_ratio:.1f} > {pe_threshold}")
            metadata['pe_ratio'] = pe_ratio
        
        # Check debt to equity ratio
        debt_equity_threshold = self.parameters['debt_to_equity_threshold']
        debt_equity = symbol_data.get('debt_to_equity', None)
        
        if debt_equity and debt_equity > debt_equity_threshold:
            short_signal = True
            metadata['signals'].append(f"High debt-to-equity: {debt_equity:.1f} > {debt_equity_threshold}")
            metadata['debt_to_equity'] = debt_equity
        
        # Check for negative earnings growth
        if self.parameters['negative_earnings_growth']:
            earnings_growth = symbol_data.get('earnings_growth', None)
            
            if earnings_growth and earnings_growth < 0:
                short_signal = True
                metadata['signals'].append(f"Negative earnings growth: {earnings_growth:.1f}%")
                metadata['earnings_growth'] = earnings_growth
        
        # Check sector performance (avoiding shorting strong sectors)
        sector = symbol_data.get('sector', None)
        if sector and sector in self.parameters.get('sector_performance', {}):
            sector_perf = self.parameters['sector_performance'][sector]
            
            # Avoid shorting in strong sectors
            if sector_perf > 5.0:  # If sector is up more than 5%
                short_signal = False
                metadata['signals'].append(f"Signal rejected: strong sector performance ({sector_perf:.1f}%)")
        
        # Calculate signal strength based on number of confirming indicators
        signal_count = len([s for s in metadata['signals'] if 'rejected' not in s])
        metadata['signal_strength'] = min(0.9, signal_count / 3)  # Cap at 0.9
        
        return short_signal, metadata
    
    def _generate_short_signal(self, metadata: Dict[str, Any]) -> None:
        """
        Generate a short signal based on analysis.
        
        Args:
            metadata: Signal metadata with analysis results
        """
        # Get current price and market data
        current_price = self.market_data['close'].iloc[-1]
        current_time = datetime.now()
        
        # Create a unique signal ID
        signal_id = str(uuid.uuid4())
        
        # Calculate stop loss based on ATR
        stop_loss = None
        if 'atr' in self.indicators and len(self.indicators['atr']) > 0:
            atr_value = self.indicators['atr'].iloc[-1]
            stop_loss = current_price + (atr_value * 3)  # 3 ATR stop for short positions
        else:
            # Fallback if ATR not available
            stop_loss = current_price * (1 + self.parameters['tight_stop_percent'] / 100)
        
        # Calculate target price based on risk-reward
        risk = stop_loss - current_price
        reward = risk * 2  # 1:2 risk-reward ratio
        target_price = current_price - reward
        
        # Set signal confidence based on metadata
        confidence = metadata.get('signal_strength', 0.5)
        
        # Create signal object
        signal = Signal(
            id=signal_id,
            symbol=self.session.symbol,
            signal_type=SignalType.SHORT,
            entry_price=current_price,
            stop_loss=stop_loss,
            target_price=target_price,
            timestamp=current_time,
            expiration=current_time + timedelta(days=1),
            confidence=confidence,
            metadata=metadata
        )
        
        # Store the signal
        self.short_signals[signal_id] = signal
        
        # Log signal generation
        trigger_types = metadata.get('signals', [])
        triggers_str = ', '.join(trigger_types[:3])  # Show first 3 triggers
        
        logger.info(f"Generated SHORT signal for {self.session.symbol} " +
                   f"at {current_price:.2f} with stop at {stop_loss:.2f} " +
                   f"(triggers: {triggers_str})")
        
        # Act on the signal
        self._act_on_signal(signal)
    
    def _act_on_signal(self, signal: Signal) -> None:
        """
        Act on a generated short signal.
        
        Args:
            signal: Signal object with trade details
        """
        # Skip if we're already in a position for this symbol
        for position in self.positions:
            if position.symbol == signal.symbol and position.status == PositionStatus.OPEN:
                logger.info(f"Already in position for {signal.symbol}, skipping signal")
                return
        
        # Calculate position size based on risk parameters
        account_balance = 100000.0  # Example, in real implementation this would be retrieved
        position_size = self._calculate_position_size(signal, account_balance)
        
        # Create a new position
        position_id = str(uuid.uuid4())
        position = Position(
            id=position_id,
            symbol=signal.symbol,
            direction='short',  # Always short for this strategy
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            target_price=signal.target_price,
            size=position_size,
            entry_time=datetime.now(),
            status=PositionStatus.PENDING,
            metadata={
                'strategy': 'short_selling',
                'signal_id': signal.id,
                'triggers': signal.metadata.get('signals', []),
                'strategy_mode': self.parameters['strategy_mode'],
                'highest_price': signal.entry_price  # For tracking trailing stop
            }
        )
        
        # In a real implementation, this would send the order to a broker
        # and the position would be updated once the order is filled
        logger.info(f"Opening SHORT position for {position.symbol} " +
                   f"at {position.entry_price:.2f} with size {position.size:.2f}")
        
        # Add position to our tracker
        self.positions.append(position)
        
        # Update position status to OPEN (in real implementation this would happen after fill)
        position.status = PositionStatus.OPEN
        
        # Initialize days held counter
        self.days_held[position.id] = 0
        
        # Increment total short positions counter
        self.total_short_positions += 1
        
        # Emit an event for position opened if event bus is available
        if self.event_bus:
            event = Event(
                event_type=EventType.POSITION_OPENED,
                timestamp=datetime.now(),
                data={
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'size': position.size,
                    'strategy': 'short_selling'
                }
            )
            self.event_bus.emit(event)
    
    def _calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """
        Calculate position size based on risk parameters, with adjustments specific to short selling.
        
        Args:
            signal: Signal with entry and stop loss prices
            account_balance: Current account balance
            
        Returns:
            Position size in shares
        """
        # Calculate risk amount based on max risk per trade
        max_risk_percent = self.parameters['max_risk_per_trade_percent'] / 100.0
        risk_amount = account_balance * max_risk_percent
        
        # Apply position size factor for shorts (typically smaller than longs)
        position_size_factor = self.parameters['position_size_factor']
        risk_amount *= position_size_factor
        
        # Apply market regime bias if in bear market
        if self.parameters['market_regime'] == 'bear':
            risk_amount *= self.parameters['bear_market_bias']
        
        # Calculate risk per share
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        
        # Ensure stop loss is provided
        if stop_loss is None:
            # Fallback if stop loss not provided
            stop_loss = entry_price * (1 + self.parameters['tight_stop_percent'] / 100)
        
        # Calculate risk per share
        risk_per_share = abs(stop_loss - entry_price)
        
        # Ensure minimum risk per share to avoid division by zero
        if risk_per_share < 0.01 or risk_per_share < entry_price * 0.005:
            risk_per_share = max(0.01, entry_price * 0.005)  # At least 0.5% of price
        
        # Calculate number of shares based on risk
        shares = risk_amount / risk_per_share
        
        # Round to nearest lot size (typically 100 for stocks)
        lot_size = self.session.lot_size if hasattr(self.session, 'lot_size') else 100
        shares = max(lot_size, round(shares / lot_size) * lot_size)
        
        # Limit to max position size (example: 15% of account for shorts - more conservative)
        max_position_value = account_balance * 0.15
        max_shares = max_position_value / entry_price
        shares = min(shares, max_shares)
        
        # Ensure minimum position size
        min_shares = lot_size
        shares = max(min_shares, shares)
        
        return shares
    
    def _close_position(self, position_id: str, reason: str = "Unspecified") -> None:
        """
        Close an open short position.
        
        Args:
            position_id: ID of the position to close
            reason: Reason for closing the position
        """
        # Find the position in our tracker
        position = None
        for p in self.positions:
            if p.id == position_id:
                position = p
                break
        
        if position is None or position.status != PositionStatus.OPEN:
            logger.warning(f"Cannot close position {position_id}: not found or not open")
            return
        
        # Get current price (simulated - in real implementation would get from market)
        current_price = self.market_data['close'].iloc[-1] if len(self.market_data) > 0 else position.entry_price
        
        # Calculate P&L (for shorts, profit when price goes down)
        pnl = (position.entry_price - current_price) * position.size
        
        # Update position status
        position.status = PositionStatus.CLOSED
        position.exit_price = current_price
        position.exit_time = datetime.now()
        position.pnl = pnl
        position.metadata['exit_reason'] = reason
        
        # Update days held if tracked
        if position.id in self.days_held:
            position.metadata['days_held'] = self.days_held[position.id]
            del self.days_held[position.id]  # Clean up
        
        # Decrement total short positions counter
        self.total_short_positions -= 1
        
        logger.info(f"Closed SHORT position for {position.symbol} " +
                  f"at {position.exit_price:.2f} with P&L {position.pnl:.2f} " +
                  f"(reason: {reason})")
        
        # Emit an event for position closed if event bus is available
        if self.event_bus:
            event = Event(
                event_type=EventType.POSITION_CLOSED,
                timestamp=datetime.now(),
                data={
                    'position_id': position.id,
                    'symbol': position.symbol,
                    'direction': position.direction,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'pnl': position.pnl,
                    'reason': reason,
                    'strategy': 'short_selling'
                }
            )
            self.event_bus.emit(event)
    
    def check_exit_conditions(self) -> None:
        """
        Check exit conditions for open short positions.
        Should be called on each price update.
        """
        if not self.positions or len(self.market_data) == 0:
            return
        
        current_price = self.market_data['close'].iloc[-1]
        
        # Check each open position
        for position in self.positions:
            if position.status != PositionStatus.OPEN:
                continue
                
            # Skip if not for our symbol
            if position.symbol != self.session.symbol:
                continue
            
            # Only process short positions
            if position.direction != 'short':
                continue
            
            # Check stop loss - for shorts, this is when price goes UP above stop
            if current_price >= position.stop_loss:
                self._close_position(position.id, "Stop loss triggered")
                continue
            
            # Check target price - for shorts, this is when price goes DOWN to target
            if current_price <= position.target_price:
                self._close_position(position.id, "Profit target reached")
                continue
            
            # Check trailing stop if activated
            # For shorts, we track the lowest price seen and adjust stop downward
            if 'lowest_price' not in position.metadata:
                position.metadata['lowest_price'] = position.entry_price
            
            # Update lowest price seen during trade
            if current_price < position.metadata['lowest_price']:
                position.metadata['lowest_price'] = current_price
                
                # Check if we should activate trailing stop
                price_move = position.entry_price - position.metadata['lowest_price']
                initial_risk = position.stop_loss - position.entry_price
                
                # If price has moved in our favor by certain amount, activate trailing stop
                activation_pct = self.parameters['trailing_stop_activation']
                if price_move >= (position.entry_price * activation_pct / 100):
                    # Calculate trailing stop distance 
                    if 'atr' in self.indicators and len(self.indicators['atr']) > 0:
                        trailing_distance = self.indicators['atr'].iloc[-1] * 2  # 2 ATR
                    else:
                        # Use percentage-based trailing stop if ATR not available
                        trailing_distance = current_price * (self.parameters['trailing_stop_percent'] / 100)
                    
                    # New stop price - for shorts, this moves down as price moves down
                    new_stop = position.metadata['lowest_price'] + trailing_distance
                    
                    # Only update if new stop is better (lower) than current stop
                    if new_stop < position.stop_loss:
                        old_stop = position.stop_loss
                        position.stop_loss = new_stop
                        logger.info(f"Updated trailing stop for SHORT position {position.id} from {old_stop:.2f} to {new_stop:.2f}")
            
            # Check for max loss threshold - close position if loss exceeds threshold
            max_loss_percent = self.parameters['max_loss_percent']
            current_loss_percent = ((current_price / position.entry_price) - 1) * 100
            
            if current_loss_percent >= max_loss_percent:
                self._close_position(position.id, f"Max loss threshold exceeded: {current_loss_percent:.2f}% > {max_loss_percent}%")
                continue
            
            # Special exit checks for short selling
            
            # Check for bullish reversal pattern
            if hasattr(self, 'indicators') and self.indicators:
                # Bullish trend developing - consider closing short position
                if 'sma_20' in self.indicators and 'sma_50' in self.indicators:
                    sma_20 = self.indicators['sma_20'].iloc[-1]
                    sma_50 = self.indicators['sma_50'].iloc[-1]
                    
                    # If shorter-term MA crosses above longer-term, market could be turning bullish
                    if sma_20 > sma_50 and self.indicators['sma_20'].iloc[-2] <= self.indicators['sma_50'].iloc[-2]:
                        self._close_position(position.id, "Bullish trend developing (20 SMA crossed above 50 SMA)")
                        continue
                        
                # RSI shows oversold conditions - exit short before bounce
                if 'rsi' in self.indicators and len(self.indicators['rsi']) > 0:
                    current_rsi = self.indicators['rsi'].iloc[-1]
                    if current_rsi < 30:  # Oversold
                        self._close_position(position.id, f"Oversold RSI: {current_rsi:.1f}")
                        continue
