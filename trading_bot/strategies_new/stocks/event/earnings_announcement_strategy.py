#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Earnings Announcement Strategy

A sophisticated strategy for trading around earnings announcements.
This strategy identifies upcoming earnings events and trades based on:
- Pre-earnings momentum and implied volatility changes
- Post-earnings price reactions
- Historical earnings reaction patterns
- Market sentiment and analyst expectations
"""

import logging
import numpy as np
import pandas as pd
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, time

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
    name="EarningsAnnouncementStrategy",
    market_type="stocks",
    description="A strategy that trades around earnings announcements, capturing price movements driven by surprises, expectations, and post-announcement price action",
    timeframes=["15m", "1h", "4h", "1d"],
    parameters={
        "trade_timing": {"description": "When to trade (pre_earnings, post_earnings, both)", "type": "string"},
        "trade_direction": {"description": "Direction bias (directional, straddle, expectations_based)", "type": "string"},
        "historical_lookback": {"description": "Number of past earnings to analyze for patterns", "type": "integer"},
        "use_analyst_expectations": {"description": "Whether to incorporate analyst expectations", "type": "boolean"}
    }
)
class EarningsAnnouncementStrategy(StocksBaseStrategy):
    """
    Earnings Announcement Strategy
    
    This strategy specializes in trading around corporate earnings announcements
    using historical patterns, implied volatility changes, and price action.
    
    Features:
    - Pre-earnings momentum identification
    - Post-earnings reaction trading
    - Implied volatility-based position sizing
    - Historical earnings surprise analysis
    - Event horizon risk management
    - Adaptive entry/exit timing based on market conditions
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the Earnings Announcement Strategy.
        
        Args:
            session: StocksSession for the specific symbol and timeframe
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize the base strategy
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Earnings timing parameters
            'days_before_earnings': 5,     # Start monitoring this many days before earnings
            'days_after_earnings': 3,      # Continue monitoring this many days after earnings
            'entry_day_threshold': 2,      # Enter no later than this many days before earnings
            
            # Trade timing parameters
            'pre_earnings_entry': True,    # Whether to enter before earnings
            'post_earnings_entry': True,   # Whether to enter after earnings announcement
            'hold_through_earnings': False, # Whether to hold positions through earnings
            
            # Position sizing and risk parameters
            'pre_earnings_size_percent': 50,  # Percentage of normal position size pre-earnings
            'post_earnings_size_percent': 100, # Percentage of normal position size post-earnings
            'max_risk_per_trade_percent': 1.0, # Max risk per trade as % of account
            'max_earnings_trades_per_month': 10, # Maximum earnings trades per month
            
            # Technical parameters
            'iv_rank_threshold': 70,        # Implied volatility rank threshold for consideration
            'pre_earnings_momentum_days': 10, # Days to measure momentum before earnings
            'momentum_threshold': 5.0,      # Minimum momentum in percent for entry consideration
            'volume_threshold': 1.5,        # Minimum volume vs average for confirmation
            
            # Analysis parameters
            'historical_earnings_lookback': 8, # Number of historical earnings to analyze
            'surprise_threshold_percent': 10,  # Earnings surprise threshold in percent
            'historical_reaction_weight': 0.7, # Weight given to historical reactions vs current technicals
            
            # Trading approach
            'strategy_mode': 'adaptive',    # 'bullish', 'bearish', 'adaptive'
            'analyst_beat_threshold': 60,   # Percentage of analysts expecting a beat for bullish bias
            
            # External data sources
            'earnings_calendar': {},        # Dict of symbols -> upcoming earnings dates
            'analyst_expectations': {},     # Dict of symbols -> analyst expectations
            'historical_earnings_data': {}, # Dict of symbols -> historical earnings data
        }
        
        # Update parameters with defaults for any missing keys
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
        
        # Strategy state
        self.earnings_trades_this_month = 0  # Counter for earnings trades
        self.last_month_checked = datetime.now().month
        self.upcoming_earnings = None  # Will store next earnings date when detected
        self.earnings_countdown = None  # Days until next earnings
        self.historical_reactions = []  # Will store historical post-earnings reactions
        self.earnings_trade_stage = None  # 'pre_earnings', 'post_earnings', or None
        self.earnings_signals = {}  # Stores active earnings-related signals
        
        # Register for market events if event bus is available
        if self.event_bus:
            self.register_for_events(self.event_bus)
        
        logger.info(f"Initialized Earnings Announcement Strategy for {session.symbol} on {session.timeframe}")
    
    def register_for_events(self, event_bus: EventBus) -> None:
        """
        Register for relevant market events.
        
        Args:
            event_bus: EventBus to register with
        """
        # First register for common events via base class
        super().register_for_events(event_bus)
        
        # Register for earnings-specific events
        event_bus.subscribe(EventType.EARNINGS_ANNOUNCEMENT, self._on_earnings_announcement)
        event_bus.subscribe(EventType.ANALYST_RATING_CHANGE, self._on_analyst_rating_change)
        event_bus.subscribe(EventType.MARKET_OPEN, self._on_market_open)
        event_bus.subscribe(EventType.MARKET_CLOSE, self._on_market_close)
        
        logger.debug(f"Earnings Announcement Strategy registered for events")
    
    def _on_market_open(self, event: Event) -> None:
        """
        Handle market open event.
        
        Updates earnings countdown and checks for trade opportunities.
        
        Args:
            event: Market open event
        """
        # Reset monthly counter if in a new month
        current_month = datetime.now().month
        if current_month != self.last_month_checked:
            self.earnings_trades_this_month = 0
            self.last_month_checked = current_month
            logger.info(f"New month {current_month}, reset earnings trade counter")
        
        # Update earnings date information
        self._update_earnings_date()
        
        # Check if we're in the earnings window
        if self.earnings_countdown is not None:
            if 0 <= self.earnings_countdown <= self.parameters['days_before_earnings']:
                logger.info(f"In pre-earnings window: {self.earnings_countdown} days until earnings")
                self.earnings_trade_stage = 'pre_earnings'
            elif -self.parameters['days_after_earnings'] <= self.earnings_countdown < 0:
                logger.info(f"In post-earnings window: {-self.earnings_countdown} days after earnings")
                self.earnings_trade_stage = 'post_earnings'
            else:
                self.earnings_trade_stage = None
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
    
    def _on_market_close(self, event: Event) -> None:
        """
        Handle market close event.
        
        Manages positions before earnings announcements which typically occur after market close.
        
        Args:
            event: Market close event
        """
        # Update earnings date information
        self._update_earnings_date()
        
        # Check if earnings are expected today after market close
        if self.earnings_countdown is not None and self.earnings_countdown <= 0.5:
            # If we're not holding through earnings, close positions
            if not self.parameters['hold_through_earnings']:
                for position in self.positions:
                    if position.status == PositionStatus.OPEN:
                        self._close_position(position.id)
                        logger.info(f"Closed position {position.id} before earnings announcement")
    
    def _on_earnings_announcement(self, event: Event) -> None:
        """
        Handle earnings announcement events.
        
        Process earnings data and prepare for post-earnings trading.
        
        Args:
            event: Earnings announcement event
        """
        # Check if the event is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Extract earnings data
        earnings_data = event.data.get('earnings_data', {})
        actual_eps = earnings_data.get('actual_eps')
        expected_eps = earnings_data.get('expected_eps')
        previous_eps = earnings_data.get('previous_eps')
        
        # Calculate earnings surprise if we have the data
        surprise_percent = None
        if actual_eps is not None and expected_eps is not None and expected_eps != 0:
            surprise_percent = ((actual_eps - expected_eps) / abs(expected_eps)) * 100
            
        # Log earnings results
        logger.info(f"Earnings announcement for {self.session.symbol}: " +
                   f"Actual EPS: {actual_eps}, Expected: {expected_eps}, " +
                   f"Surprise: {surprise_percent:.2f}% if surprise_percent else 'N/A'}")
        
        # Reset earnings date as we've just had the announcement
        self.upcoming_earnings = None
        self.earnings_countdown = 0  # Just happened
        self.earnings_trade_stage = 'post_earnings'
        
        # Store this reaction for future analysis
        if surprise_percent is not None:
            # Store pre-announcement price
            pre_earnings_price = self.market_data['close'].iloc[-1] if not self.market_data.empty else None
            
            # We'll update with post-announcement price reaction when we have it
            self.historical_reactions.append({
                'date': datetime.now(),
                'actual_eps': actual_eps,
                'expected_eps': expected_eps,
                'surprise_percent': surprise_percent,
                'pre_earnings_price': pre_earnings_price,
                'post_earnings_price': None,  # Will update this on next market data
                'reaction_percent': None      # Will calculate once we have post price
            })
    
    def _on_analyst_rating_change(self, event: Event) -> None:
        """
        Handle analyst rating change events.
        
        Update internal analyst sentiment for decision making.
        
        Args:
            event: Analyst rating change event
        """
        # Check if the event is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
            
        # Extract analyst data
        analyst_data = event.data.get('analyst_data', {})
        rating_change = analyst_data.get('rating_change')
        new_rating = analyst_data.get('new_rating')
        old_rating = analyst_data.get('old_rating')
        price_target = analyst_data.get('price_target')
        
        # Log analyst changes
        logger.info(f"Analyst rating change for {self.session.symbol}: " +
                   f"{old_rating} to {new_rating}, PT: {price_target}")
        
        # Update our analyst expectations data
        symbol = self.session.symbol
        if symbol not in self.parameters['analyst_expectations']:
            self.parameters['analyst_expectations'][symbol] = []
            
        # Add this analyst's view
        self.parameters['analyst_expectations'][symbol].append({
            'date': datetime.now(),
            'firm': analyst_data.get('firm'),
            'analyst': analyst_data.get('analyst'),
            'rating': new_rating,
            'price_target': price_target
        })
        
        # Recalculate analyst sentiment if we're approaching earnings
        if self.earnings_trade_stage == 'pre_earnings':
            self._analyze_analyst_sentiment()
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated event.
        
        Update internal data and check for earnings-related trading opportunities.
        
        Args:
            event: Market data updated event
        """
        # Let the base class handle common functionality first
        super()._on_market_data_updated(event)
        
        # Check if the event data is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Update post-earnings price reaction if we just had earnings
        if self.historical_reactions and self.historical_reactions[-1]['post_earnings_price'] is None:
            latest_reaction = self.historical_reactions[-1]
            pre_price = latest_reaction['pre_earnings_price']
            
            if pre_price is not None:
                # Update with first post-earnings price
                post_price = self.market_data['close'].iloc[-1]
                reaction_percent = ((post_price / pre_price) - 1) * 100
                
                latest_reaction['post_earnings_price'] = post_price
                latest_reaction['reaction_percent'] = reaction_percent
                
                logger.info(f"Post-earnings reaction for {self.session.symbol}: {reaction_percent:.2f}%")
        
        # Check for earnings-related trade opportunities
        if self.earnings_trade_stage:
            self._check_for_trade_opportunities()
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed event.
        
        Calculate indicators and check entry conditions on completed timeframes.
        
        Args:
            event: Timeframe completed event
        """
        # Let the base class handle common functionality first
        super()._on_timeframe_completed(event)
        
        # Check if the event data is for our symbol and timeframe
        if (event.data.get('symbol') != self.session.symbol or 
            event.data.get('timeframe') != self.session.timeframe):
            return
        
        # Calculate indicators
        self.indicators = self.calculate_indicators(self.market_data)
        
        # Update earnings date
        self._update_earnings_date()
        
        # Check for earnings-related trade opportunities
        if self.earnings_trade_stage:
            self._check_for_trade_opportunities()
    
    def _update_earnings_date(self) -> None:
        """Update information about upcoming earnings date."""
        symbol = self.session.symbol
        
        # Check if we have earnings calendar data
        if symbol in self.parameters['earnings_calendar']:
            earnings_date = self.parameters['earnings_calendar'][symbol]
            
            # Convert string to datetime if needed
            if isinstance(earnings_date, str):
                try:
                    earnings_date = datetime.strptime(earnings_date, '%Y-%m-%d')
                except ValueError:
                    logger.error(f"Invalid earnings date format for {symbol}: {earnings_date}")
                    return
                    
            # Store upcoming earnings date
            self.upcoming_earnings = earnings_date
            
            # Calculate days until earnings
            today = datetime.now().date()
            days_until_earnings = (earnings_date.date() - today).days
            self.earnings_countdown = days_until_earnings
            
            # Log earnings information
            if 0 <= days_until_earnings <= self.parameters['days_before_earnings']:
                logger.info(f"Upcoming earnings for {symbol} in {days_until_earnings} days on {earnings_date.date()}")
        else:
            # No earnings data, reset values
            self.upcoming_earnings = None
            self.earnings_countdown = None
    
    def _analyze_historical_earnings_reactions(self) -> Dict[str, Any]:
        """
        Analyze historical earnings reactions to inform current trading decisions.
        
        Returns:
            Dictionary with analysis results
        """
        symbol = self.session.symbol
        results = {
            'reaction_bias': 'neutral',  # 'bullish', 'bearish', or 'neutral'
            'avg_surprise': 0.0,         # Average earnings surprise percentage
            'avg_reaction': 0.0,         # Average post-earnings price reaction percentage
            'surprise_reaction_correlation': 0.0,  # Correlation between surprise and reaction
            'reaction_consistency': 0.0,  # Consistency of reactions (0.0-1.0)
            'confidence': 0.0            # Confidence in the analysis (0.0-1.0)
        }
        
        # Get historical earnings data
        historical_data = self.parameters['historical_earnings_data'].get(symbol, [])
        
        # Combine with our recorded reactions
        all_reactions = historical_data + self.historical_reactions
        
        # Need sufficient data for meaningful analysis
        if len(all_reactions) < 2:
            logger.warning(f"Insufficient historical earnings data for {symbol}")
            return results
            
        # Calculate average surprise and reaction
        surprises = [r.get('surprise_percent', 0) for r in all_reactions if r.get('surprise_percent') is not None]
        reactions = [r.get('reaction_percent', 0) for r in all_reactions if r.get('reaction_percent') is not None]
        
        if surprises and reactions:
            avg_surprise = sum(surprises) / len(surprises)
            avg_reaction = sum(reactions) / len(reactions)
            
            results['avg_surprise'] = avg_surprise
            results['avg_reaction'] = avg_reaction
            
            # Determine bias based on average reaction
            if avg_reaction > 1.0:  # 1% threshold for bullish
                results['reaction_bias'] = 'bullish'
            elif avg_reaction < -1.0:  # -1% threshold for bearish
                results['reaction_bias'] = 'bearish'
            
            # Calculate correlation if we have enough paired data
            paired_data = [(r.get('surprise_percent', 0), r.get('reaction_percent', 0)) 
                        for r in all_reactions 
                        if r.get('surprise_percent') is not None and r.get('reaction_percent') is not None]
            
            if len(paired_data) >= 3:  # Need at least 3 points for correlation
                surprise_values = [p[0] for p in paired_data]
                reaction_values = [p[1] for p in paired_data]
                
                # Calculate correlation coefficient
                if len(set(surprise_values)) > 1 and len(set(reaction_values)) > 1:
                    surprise_mean = sum(surprise_values) / len(surprise_values)
                    reaction_mean = sum(reaction_values) / len(reaction_values)
                    
                    numerator = sum((s - surprise_mean) * (r - reaction_mean) for s, r in paired_data)
                    denominator_s = sum((s - surprise_mean) ** 2 for s in surprise_values) ** 0.5
                    denominator_r = sum((r - reaction_mean) ** 2 for r in reaction_values) ** 0.5
                    
                    if denominator_s > 0 and denominator_r > 0:
                        correlation = numerator / (denominator_s * denominator_r)
                        results['surprise_reaction_correlation'] = correlation
            
            # Calculate consistency - what percentage of reactions are in the same direction
            positive_reactions = sum(1 for r in reactions if r > 0)
            negative_reactions = sum(1 for r in reactions if r < 0)
            
            if reactions:
                consistency = max(positive_reactions, negative_reactions) / len(reactions)
                results['reaction_consistency'] = consistency
            
            # Calculate confidence based on sample size and consistency
            sample_size_factor = min(1.0, len(all_reactions) / 8)  # Max confidence at 8+ earnings
            results['confidence'] = sample_size_factor * (0.5 + results['reaction_consistency'] / 2)
        
        logger.debug(f"Historical earnings analysis for {symbol}: {results}")
        return results
    
    def _analyze_analyst_sentiment(self) -> Dict[str, Any]:
        """
        Analyze current analyst sentiment toward the stock.
        
        Returns:
            Dictionary with analysis results
        """
        symbol = self.session.symbol
        results = {
            'sentiment': 'neutral',  # 'bullish', 'bearish', or 'neutral'
            'consensus_rating': None,  # Average/consensus rating
            'avg_price_target': None,  # Average price target
            'price_target_upside': None,  # % upside to avg price target
            'buy_percentage': 0.0,     # % of analysts with Buy ratings
            'confidence': 0.0          # Confidence in the analysis (0.0-1.0)
        }
        
        # Get analyst expectations
        analysts_data = self.parameters['analyst_expectations'].get(symbol, [])
        
        # Need sufficient data for meaningful analysis
        if not analysts_data:
            logger.warning(f"No analyst data available for {symbol}")
            return results
            
        # Count ratings by category
        rating_counts = {'buy': 0, 'hold': 0, 'sell': 0}
        price_targets = []
        
        # Process each analyst's data
        for analyst in analysts_data:
            rating = analyst.get('rating', '').lower()
            
            # Normalize ratings into buy/hold/sell
            if rating in ['buy', 'outperform', 'overweight', 'strong buy']:
                rating_counts['buy'] += 1
            elif rating in ['sell', 'underperform', 'underweight', 'strong sell']:
                rating_counts['sell'] += 1
            elif rating in ['hold', 'neutral', 'market perform', 'sector perform']:
                rating_counts['hold'] += 1
                
            # Collect price targets
            if analyst.get('price_target') is not None:
                price_targets.append(analyst['price_target'])
        
        # Calculate consensus metrics
        total_ratings = sum(rating_counts.values())
        
        if total_ratings > 0:
            buy_percentage = rating_counts['buy'] / total_ratings * 100
            sell_percentage = rating_counts['sell'] / total_ratings * 100
            hold_percentage = rating_counts['hold'] / total_ratings * 100
            
            results['buy_percentage'] = buy_percentage
            
            # Determine sentiment based on buy percentage
            if buy_percentage >= self.parameters['analyst_beat_threshold']:
                results['sentiment'] = 'bullish'
            elif sell_percentage > buy_percentage:
                results['sentiment'] = 'bearish'
            
            # Calculate current price vs. target
            if price_targets and not self.market_data.empty:
                avg_price_target = sum(price_targets) / len(price_targets)
                current_price = self.market_data['close'].iloc[-1]
                upside_percent = ((avg_price_target / current_price) - 1) * 100
                
                results['avg_price_target'] = avg_price_target
                results['price_target_upside'] = upside_percent
                
                # Adjust sentiment based on upside
                if upside_percent > 15 and results['sentiment'] != 'bearish':
                    results['sentiment'] = 'bullish'
                elif upside_percent < -5 and results['sentiment'] != 'bullish':
                    results['sentiment'] = 'bearish'
            
            # Calculate confidence based on sample size and consensus
            sample_size_factor = min(1.0, total_ratings / 5)  # Max confidence at 5+ analysts
            consensus_strength = max(buy_percentage, sell_percentage, hold_percentage) / 100
            results['confidence'] = sample_size_factor * consensus_strength
        
        logger.debug(f"Analyst sentiment analysis for {symbol}: {results}")
        return results
    
    def _calculate_pre_earnings_momentum(self) -> Dict[str, Any]:
        """
        Calculate price momentum leading into earnings.
        
        Returns:
            Dictionary with momentum metrics
        """
        results = {
            'momentum': 0.0,          # Price change percentage
            'direction': 'neutral',    # 'bullish', 'bearish', or 'neutral'
            'volume_trend': 0.0,       # Volume trend (positive = increasing)
            'volatility_change': 0.0,  # Change in volatility
            'signal_strength': 0.0     # Overall signal strength (0.0-1.0)
        }
        
        # Need sufficient data
        momentum_days = self.parameters['pre_earnings_momentum_days']
        if len(self.market_data) <= momentum_days:
            logger.warning(f"Insufficient data for pre-earnings momentum calculation")
            return results
            
        # Calculate price change
        start_price = self.market_data['close'].iloc[-momentum_days-1]
        current_price = self.market_data['close'].iloc[-1]
        
        if start_price > 0:
            momentum_pct = ((current_price / start_price) - 1) * 100
            results['momentum'] = momentum_pct
            
            # Determine direction
            if momentum_pct > 2.0:  # 2% threshold for bullish
                results['direction'] = 'bullish'
            elif momentum_pct < -2.0:  # -2% threshold for bearish
                results['direction'] = 'bearish'
        
        # Calculate volume trend
        recent_volume = self.market_data['volume'].iloc[-5:].mean()
        previous_volume = self.market_data['volume'].iloc[-momentum_days-5:-5].mean()
        
        if previous_volume > 0:
            volume_change = ((recent_volume / previous_volume) - 1) * 100
            results['volume_trend'] = volume_change
        
        # Calculate volatility change
        recent_volatility = self.market_data['high'].iloc[-5:].div(self.market_data['low'].iloc[-5:]).std()
        previous_volatility = self.market_data['high'].iloc[-momentum_days-5:-5].div(self.market_data['low'].iloc[-momentum_days-5:-5]).std()
        
        if previous_volatility > 0:
            volatility_change = ((recent_volatility / previous_volatility) - 1) * 100
            results['volatility_change'] = volatility_change
        
        # Calculate signal strength based on momentum and volume
        momentum_factor = min(1.0, abs(results['momentum']) / 10)  # Max strength at 10% momentum
        volume_factor = min(1.0, max(0.0, results['volume_trend']) / 50)  # Max strength at 50% volume increase
        
        results['signal_strength'] = (momentum_factor * 0.7) + (volume_factor * 0.3)
        
        logger.debug(f"Pre-earnings momentum for {self.session.symbol}: {results}")
        return results
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators for earnings trading.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Basic price indicators
        if not data.empty:
            indicators['current_price'] = data['close'].iloc[-1]
            
            # Calculate moving averages
            for period in [5, 10, 20, 50]:
                if len(data) >= period:
                    indicators[f'sma_{period}'] = data['close'].rolling(window=period).mean().iloc[-1]
            
            # Calculate RSI
            if len(data) >= 14:
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # Calculate Bollinger Bands
            if len(data) >= 20:
                sma_20 = data['close'].rolling(window=20).mean()
                std_20 = data['close'].rolling(window=20).std()
                
                indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
                indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
                indicators['bb_width'] = ((indicators['bb_upper'] - indicators['bb_lower']) / sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) and sma_20.iloc[-1] > 0 else 0
            
            # Calculate ATR for volatility
            if len(data) >= 14:
                high_low = data['high'] - data['low']
                high_close = (data['high'] - data['close'].shift()).abs()
                low_close = (data['low'] - data['close'].shift()).abs()
                
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                atr = true_range.rolling(window=14).mean()
                indicators['atr'] = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else (data['high'].iloc[-1] - data['low'].iloc[-1])
                
                # Calculate ATR as percentage of price (normalized volatility)
                if indicators['current_price'] > 0:
                    indicators['atr_percent'] = (indicators['atr'] / indicators['current_price']) * 100
            
            # Calculate pre-earnings momentum metrics
            if self.earnings_trade_stage == 'pre_earnings':
                momentum_metrics = self._calculate_pre_earnings_momentum()
                indicators.update(momentum_metrics)
            
            # Calculate historical earnings metrics
            historical_metrics = self._analyze_historical_earnings_reactions()
            indicators.update({
                'historical_reaction_bias': historical_metrics['reaction_bias'],
                'avg_earnings_reaction': historical_metrics['avg_reaction'],
                'historical_confidence': historical_metrics['confidence']
            })
            
            # Add implied volatility if available (would come from options data in a real implementation)
            indicators['iv_rank'] = 50  # Placeholder, would be calculated from options data
            
            # Analyst sentiment
            analyst_metrics = self._analyze_analyst_sentiment()
            indicators.update({
                'analyst_sentiment': analyst_metrics['sentiment'],
                'analyst_buy_percentage': analyst_metrics['buy_percentage'],
                'price_target_upside': analyst_metrics['price_target_upside']
            })
        
        return indicators
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on earnings events and indicators.
        
        Args:
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Dictionary of trading signals
        """
        signals = {}
        symbol = self.session.symbol
        
        # Don't generate signals if we're outside the earnings window
        if not self.earnings_trade_stage:
            return signals
        
        # Check if we've reached the maximum earnings trades for the month
        if self.earnings_trades_this_month >= self.parameters['max_earnings_trades_per_month']:
            logger.info(f"Reached maximum earnings trades for the month ({self.parameters['max_earnings_trades_per_month']})")
            return signals
        
        # Generate signal based on the earnings stage
        if self.earnings_trade_stage == 'pre_earnings':
            # Only generate pre-earnings signals if configured to do so
            if not self.parameters['pre_earnings_entry']:
                return signals
                
            # Only enter if we're not too close to earnings
            if self.earnings_countdown is not None and self.earnings_countdown < self.parameters['entry_day_threshold']:
                logger.info(f"Too close to earnings for new pre-earnings entry: {self.earnings_countdown} days")
                return signals
                
            # Determine direction based on pre-earnings momentum, historical bias, and analyst sentiment
            trade_direction = self._determine_trade_direction(indicators)
            
            # Check if signal strength is sufficient
            momentum = indicators.get('momentum', 0)
            if abs(momentum) < self.parameters['momentum_threshold']:
                logger.debug(f"Insufficient momentum for pre-earnings entry: {momentum:.2f}%")
                return signals
                
            # Create a unique signal ID
            signal_id = str(uuid.uuid4())
            
            # Determine signal type
            signal_type = SignalType.LONG if trade_direction == 'long' else SignalType.SHORT
            
            # Get current price
            current_price = indicators.get('current_price', data['close'].iloc[-1])
            
            # Create signal object
            signal = Signal(
                id=signal_id,
                symbol=symbol,
                signal_type=signal_type,
                timeframe=self.session.timeframe,
                price=current_price,
                confidence=0.7,  # Moderate confidence for pre-earnings
                metadata={
                    'strategy': self.name,
                    'trade_stage': 'pre_earnings',
                    'days_to_earnings': self.earnings_countdown,
                    'momentum': momentum,
                    'analyst_sentiment': indicators.get('analyst_sentiment', 'neutral'),
                    'historical_bias': indicators.get('historical_reaction_bias', 'neutral')
                }
            )
            
            signals[signal_id] = signal
            self.earnings_signals[signal_id] = signal
            
            # Log signal generation
            logger.info(f"Generated {trade_direction} pre-earnings signal for {symbol}: {momentum:.2f}% momentum, {self.earnings_countdown} days to earnings")
            
        elif self.earnings_trade_stage == 'post_earnings':
            # Only generate post-earnings signals if configured to do so
            if not self.parameters['post_earnings_entry']:
                return signals
                
            # For post-earnings, we want to trade the reaction
            # Check if we have reaction data from the most recent earnings
            if self.historical_reactions and self.historical_reactions[-1].get('reaction_percent') is not None:
                reaction = self.historical_reactions[-1]['reaction_percent']
                
                # Only trade significant reactions
                if abs(reaction) >= self.parameters['surprise_threshold_percent']:
                    # Determine direction - for post earnings we trade in the direction of the reaction
                    trade_direction = 'long' if reaction > 0 else 'short'
                    
                    # Create a unique signal ID
                    signal_id = str(uuid.uuid4())
                    
                    # Determine signal type
                    signal_type = SignalType.LONG if trade_direction == 'long' else SignalType.SHORT
                    
                    # Get current price
                    current_price = indicators.get('current_price', data['close'].iloc[-1])
                    
                    # Create signal object
                    signal = Signal(
                        id=signal_id,
                        symbol=symbol,
                        signal_type=signal_type,
                        timeframe=self.session.timeframe,
                        price=current_price,
                        confidence=0.8,  # High confidence for post-earnings
                        metadata={
                            'strategy': self.name,
                            'trade_stage': 'post_earnings',
                            'days_after_earnings': -self.earnings_countdown,
                            'earnings_reaction': reaction,
                            'earnings_surprise': self.historical_reactions[-1].get('surprise_percent')
                        }
                    )
                    
                    signals[signal_id] = signal
                    self.earnings_signals[signal_id] = signal
                    
                    # Log signal generation
                    logger.info(f"Generated {trade_direction} post-earnings signal for {symbol}: {reaction:.2f}% price reaction")
                    
                    # Increment earnings trades counter
                    self.earnings_trades_this_month += 1
        
        return signals
    
    def _determine_trade_direction(self, indicators: Dict[str, Any]) -> str:
        """
        Determine the optimal trade direction based on various factors.
        
        Args:
            indicators: Pre-calculated indicators
            
        Returns:
            Trade direction ('long' or 'short')
        """
        # Default to the strategy mode if it's explicitly set
        strategy_mode = self.parameters['strategy_mode']
        if strategy_mode == 'bullish':
            return 'long'
        elif strategy_mode == 'bearish':
            return 'short'
        
        # For adaptive mode, weigh multiple factors
        # 1. Pre-earnings momentum
        momentum_direction = indicators.get('direction', 'neutral')
        
        # 2. Historical earnings reaction bias
        historical_bias = indicators.get('historical_reaction_bias', 'neutral')
        
        # 3. Analyst sentiment
        analyst_sentiment = indicators.get('analyst_sentiment', 'neutral')
        
        # 4. Technical indicators
        rsi = indicators.get('rsi', 50)
        technical_bias = 'neutral'
        if rsi > 70:
            technical_bias = 'bearish'  # Overbought
        elif rsi < 30:
            technical_bias = 'bullish'  # Oversold
        
        # Count the signals in each direction
        bullish_count = 0
        bearish_count = 0
        
        # Weight the factors
        if momentum_direction == 'bullish':
            bullish_count += 2  # High weight for recent momentum
        elif momentum_direction == 'bearish':
            bearish_count += 2
            
        if historical_bias == 'bullish':
            bullish_count += 1.5  # Medium-high weight for historical patterns
        elif historical_bias == 'bearish':
            bearish_count += 1.5
            
        if analyst_sentiment == 'bullish':
            bullish_count += 1  # Medium weight for analyst views
        elif analyst_sentiment == 'bearish':
            bearish_count += 1
            
        if technical_bias == 'bullish':
            bullish_count += 0.5  # Lower weight for technicals
        elif technical_bias == 'bearish':
            bearish_count += 0.5
        
        # Determine direction based on weighted counts
        if bullish_count > bearish_count + 1:  # Need significant edge
            return 'long'
        elif bearish_count > bullish_count + 1:  # Need significant edge
            return 'short'
        else:
            # If it's close, look at momentum as tiebreaker
            return 'long' if momentum_direction != 'bearish' else 'short'
    
    def calculate_position_size(self, direction: str, data: pd.DataFrame, indicators: Dict[str, Any]) -> float:
        """
        Calculate appropriate position size based on earnings stage and risk parameters.
        
        Args:
            direction: Direction of the trade ('long' or 'short')
            data: Market data DataFrame
            indicators: Pre-calculated indicators
            
        Returns:
            Position size in number of shares
        """
        symbol = self.session.symbol
        
        # Get account balance (placeholder - would come from the actual broker in production)
        account_balance = 100000.0  # Example value
        
        # Get current price
        current_price = indicators.get('current_price', data['close'].iloc[-1])
        
        # Get ATR for volatility-based position sizing
        atr = indicators.get('atr', data['high'].iloc[-1] - data['low'].iloc[-1])
        
        # Calculate stop loss distance based on ATR and expected earnings volatility
        atr_multiplier = 2.0  # Default
        
        # Adjust multiplier based on historical earnings volatility
        avg_reaction = abs(indicators.get('avg_earnings_reaction', 0))
        if avg_reaction > 10:  # High volatility stock
            atr_multiplier = 3.0
        elif avg_reaction > 5:  # Medium volatility
            atr_multiplier = 2.5
        
        stop_distance = atr * atr_multiplier
        
        # Calculate risk amount based on max risk per trade
        max_risk_percent = self.parameters['max_risk_per_trade_percent'] / 100.0
        risk_amount = account_balance * max_risk_percent
        
        # Calculate base position size based on stop distance
        base_shares = risk_amount / stop_distance
        
        # Adjust position size based on earnings stage
        if self.earnings_trade_stage == 'pre_earnings':
            # Reduce position size for pre-earnings trades due to higher uncertainty
            size_percent = self.parameters['pre_earnings_size_percent'] / 100.0
            shares = base_shares * size_percent
        else:  # post_earnings
            # Use full or adjusted size for post-earnings trades
            size_percent = self.parameters['post_earnings_size_percent'] / 100.0
            shares = base_shares * size_percent
        
        # Convert to nearest lot size (typically 100 shares for stocks)
        lot_size = self.session.lot_size
        shares = round(shares / lot_size) * lot_size
        
        # Ensure we have at least one lot
        if shares < lot_size:
            shares = lot_size
        
        logger.info(f"Calculated position size for {symbol}: {shares} shares ({direction}) for {self.earnings_trade_stage} trade")
        return shares
    
    def regime_compatibility(self, market_regime: str) -> float:
        """
        Calculate how compatible this strategy is with the current market regime.
        
        Args:
            market_regime: Current market regime ('trending', 'ranging', 'volatile', etc.)
            
        Returns:
            Compatibility score between 0.0 and 1.0
        """
        # Earnings strategies tend to work best in volatile markets
        # and during earnings seasons.
        compatibility_map = {
            'trending': 0.6,     # Moderate compatibility with trends
            'ranging': 0.5,      # Average compatibility with ranges
            'volatile': 0.9,     # High compatibility with volatility
            'calm': 0.3          # Low compatibility with calm markets
        }
        
        # Boost compatibility if we're in earnings season (Q1, Q2, Q3, Q4 reports)
        current_month = datetime.now().month
        earnings_season_months = [1, 4, 7, 10]  # Primary earnings seasons
        secondary_months = [2, 5, 8, 11]  # Tail end of earnings seasons
        
        base_compatibility = compatibility_map.get(market_regime, 0.5)
        
        if current_month in earnings_season_months:
            # Boost during peak earnings season
            return min(1.0, base_compatibility * 1.3)
        elif current_month in secondary_months:
            # Slight boost during secondary earnings months
            return min(1.0, base_compatibility * 1.1)
        else:
            return base_compatibility
    
    def __str__(self) -> str:
        """String representation of the strategy."""
        earnings_info = f", {self.earnings_countdown} days to earnings" if self.earnings_countdown else ""
        return f"EarningsAnnouncementStrategy(symbol={self.session.symbol}, mode={self.parameters['strategy_mode']}{earnings_info})"
    
    def __repr__(self) -> str:
        """Detailed representation of the strategy."""
        return f"EarningsAnnouncementStrategy(symbol={self.session.symbol}, " \
               f"timeframe={self.session.timeframe}, mode={self.parameters['strategy_mode']}, " \
               f"pre_entry={self.parameters['pre_earnings_entry']}, post_entry={self.parameters['post_earnings_entry']})"
