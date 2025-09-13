#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
News Sentiment Trading Strategy

A sophisticated strategy for trading based on news and social media sentiment.
This strategy analyzes news articles, financial reports, and social media posts to:
- Identify significant sentiment shifts as trading signals
- Quantify news impact on price movements
- Trade based on sentiment divergence from price action
- Detect and capitalize on market overreactions
"""

import logging
import numpy as np
import pandas as pd
import uuid
import copy
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from trading_bot.core.events import Event, EventType, EventBus
from trading_bot.core.constants import TimeFrame, MarketType
from trading_bot.core.signals import Signal, SignalType
from trading_bot.data.data_pipeline import DataPipeline
from trading_bot.core.position import Position, PositionStatus
from trading_bot.strategies_new.stocks.base.stocks_base_strategy import StocksBaseStrategy, StocksSession
from trading_bot.strategies_new.stocks.event.news_sentiment_analyzer import NewsSentimentAnalyzer
from trading_bot.strategies_new.factory.registry import register_strategy

# Configure logging
logger = logging.getLogger(__name__)

@register_strategy(
    name="NewsSentimentStrategy",
    market_type="stocks",
    description="A strategy that trades based on news, social media, and analyst sentiment analysis",
    timeframes=["1h", "4h", "1d"],
    parameters={
        "sentiment_sources": {"description": "Sources of sentiment data to analyze", "type": "list"},
        "source_weights": {"description": "Weights for each sentiment source", "type": "dict"},
        "trade_mode": {"description": "Trading mode (trend_following, contrarian, combined)", "type": "string"},
        "min_sentiment_volume": {"description": "Minimum number of sentiment data points required", "type": "integer"}
    }
)
class NewsSentimentStrategy(StocksBaseStrategy):
    """
    News Sentiment Trading Strategy
    
    This strategy analyzes news sentiment and media coverage to identify
    trading opportunities based on significant sentiment shifts, news
    impact, sentiment-price divergence, and market overreactions.
    
    Features:
    - Multi-source sentiment analysis (news, social media, analyst reports)
    - Sentiment trend identification and scoring
    - Sentiment-price correlation analysis
    - News volume impact assessment
    - Contrarian trading against sentiment extremes
    - Event-driven reaction trading
    """
    
    def __init__(self, session: StocksSession, data_pipeline: DataPipeline, parameters: Dict[str, Any] = None):
        """
        Initialize the News Sentiment Trading Strategy.
        
        Args:
            session: StocksSession for the specific symbol and timeframe
            data_pipeline: Data processing pipeline
            parameters: Strategy parameters (will override defaults)
        """
        # Initialize the base strategy
        super().__init__(session, data_pipeline, parameters)
        
        # Strategy-specific default parameters
        default_params = {
            # Sentiment sources and weights
            'sentiment_sources': ['news', 'social_media', 'analyst_reports'],
            'source_weights': {'news': 0.5, 'social_media': 0.3, 'analyst_reports': 0.2},
            
            # Sentiment thresholds
            'bullish_threshold': 0.6,      # Minimum score to consider bullish (0-1)
            'bearish_threshold': 0.4,      # Maximum score to consider bearish (0-1)
            'neutral_band': 0.1,           # Neutral zone around 0.5 (0.5 +/- this value)
            'significant_change': 0.15,    # Minimum change to be considered significant
            
            # Trading parameters
            'min_sentiment_volume': 5,     # Minimum news volume to consider reliable
            'lookback_period_days': 3,     # Days to look back for sentiment analysis
            'sentiment_smoothing': 7,      # Rolling window for sentiment smoothing
            'entry_delay_bars': 2,         # Bars to wait after sentiment signal before entry
            'max_hold_period_days': 5,     # Maximum days to hold a sentiment-based position
            
            # Additional filters
            'min_avg_volume': 500000,      # Minimum average trading volume
            'min_price': 5.0,              # Minimum price for tradable stocks
            'max_sentiment_trades_per_day': 3,  # Maximum news sentiment trades per day
            'require_volume_confirmation': True, # Require increased volume for confirmation
            'volume_confirmation_threshold': 1.5, # Minimum volume vs average for confirmation
            
            # Contrarian settings
            'enable_contrarian_mode': True, # Enable trading against extreme sentiment
            'contrarian_threshold': 0.8,    # Threshold for extreme sentiment (0-1)
            'contrarian_min_lookback': 30,  # Minimum bars to look back for contrarian signals
            'contrarian_lookback_days': 5,  # Days to analyze for contrarian setup
            
            # Risk management
            'position_size_pct': 0.05,     # Position size as percentage of portfolio
            'max_loss_pct': 0.03,          # Maximum loss percentage per trade
            'trailing_stop_enabled': True,  # Use trailing stops
            'trailing_stop_activation': 0.02,  # Profit required to activate trailing stop (2%)
            'trailing_stop_distance': 0.015,   # Trailing stop distance (1.5%)
            
            # Trading modes
            'trade_mode': 'combined',       # 'trend_following', 'contrarian', or 'combined'
            'react_to_earnings': True,      # React to earnings announcements
            'react_to_guidance': True,      # React to guidance changes
            'trade_on_sentiment_reversals': True,  # Trade on sentiment trend reversals
            'enable_overnight_holding': False,    # Allow holding positions overnight
        }
            'strategy_mode': 'adaptive',    # 'trend_following', 'contrarian', or 'adaptive'
            'position_scaling': True,       # Whether to scale positions by sentiment strength
            
            # External data sources (would be populated in real implementation)
            'sentiment_data': {},           # Dict of symbols -> sentiment data
            'news_events': [],              # List of recent news events
            'social_sentiment': {},         # Dict of symbols -> social sentiment scores
        }
        
        # Update parameters with defaults for any missing keys
        for key, value in default_params.items():
            if key not in self.parameters:
                self.parameters[key] = value
                
        if parameters:
            self.parameters.update(parameters)
            
        # Initialize sentiment data structures
        self.sentiment_history = []  # Raw sentiment data from various sources
        self.sentiment_data = {      # Aggregated and processed sentiment metrics
            'current_score': 0.5,    # Current aggregate sentiment score (0-1)
            'previous_score': 0.5,   # Previous sentiment score for comparison
            'daily_change': 0.0,     # Day-over-day sentiment change
            'weekly_change': 0.0,    # Week-over-week sentiment change
            'volume': 0,             # Number of sentiment data points
            'trend': 'neutral',      # Current sentiment trend
            'news_buzz': 0,          # Relative news volume
            'extremes': {
                'bullish': 0.0,       # Most bullish recent sentiment
                'bearish': 1.0,       # Most bearish recent sentiment
            },
            'source_scores': {       # Sentiment by source
                'news': 0.5,
                'social_media': 0.5,
                'analyst_reports': 0.5
            },
            'divergence': 0.0,       # Divergence from price action
            'reversal_signal': None, # Potential sentiment reversal
            'last_updated': datetime.now()
        }
        
        # Trading state tracking
        self.signals_generated = []  # Track sentiment-based signals
        self.trades_today = 0        # Count of trades today
        self.last_sentiment_reversal = None  # Last detected sentiment reversal
        self.entry_opportunities = []  # Potential entry opportunities
        self.pending_signals = {}    # Signals waiting for confirmation
        
        # Strategy state
        self.sentiment_trades_today = 0  # Counter for sentiment trades
        
        # Create the sentiment analyzer
        self.sentiment_analyzer = NewsSentimentAnalyzer(self.parameters)
        
        # Pending signals and opportunities tracking
        self.pending_signals = {}      # Signals waiting for confirmation
        self.entry_opportunities = []  # Potential entry opportunities
        self.sentiment_signals = {}  # Stores active sentiment-related signals
        self.sentiment_trend = 'neutral'  # 'bullish', 'bearish', or 'neutral'
        self.news_volume = 0  # Volume of recent news/mentions
        
        # Register for market events if event bus is available
        if self.event_bus:
            self.register_for_events(self.event_bus)
        
        logger.info(f"Initialized News Sentiment Strategy for {session.symbol} on {session.timeframe}")
    
    def register_for_events(self, event_bus: EventBus) -> None:
        """
        Register for relevant market events.
        
        Args:
            event_bus: EventBus to register with
        """
        # First register for common events via base class
        super().register_for_events(event_bus)
        
        # Register for sentiment-specific events
        event_bus.subscribe(EventType.NEWS_RELEASE, self._on_news_release)
        event_bus.subscribe(EventType.SOCIAL_SENTIMENT_UPDATE, self._on_social_sentiment_update)
        event_bus.subscribe(EventType.ANALYST_RATING_CHANGE, self._on_analyst_rating_change)
        event_bus.subscribe(EventType.MARKET_OPEN, self._on_market_open)
        
        logger.debug(f"News Sentiment Strategy registered for events")
    
    def _on_market_open(self, event: Event) -> None:
        """
        Handle market open event.
        
        Reset daily counters and update sentiment data.
        
        Args:
            event: Market open event
        """
        # Reset daily counter if it's a new trading day
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.sentiment_trades_today = 0
            self.last_trade_date = current_date
            logger.info(f"New trading day {current_date}, reset sentiment trade counter")
        
        # Update sentiment data at market open
        self._update_sentiment_data()
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
    
    def _on_news_release(self, event: Event) -> None:
        """
        Handle news release events.
        
        Process news articles and update sentiment data.
        
        Args:
            event: News release event
        """
        # Check if the event is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Extract news data
        news_data = event.data.get('news_data', {})
        news_text = news_data.get('text', '')
        news_headline = news_data.get('headline', '')
        news_source = news_data.get('source', 'unknown')
        
        # We'd use NLP to determine sentiment in real implementation
        # For this example, we'll just use a random value with a bit of realism
        # In reality, this would use a proper sentiment analysis model
        
        # Simulate sentiment score (0-1 scale, 0.5 = neutral)
        sentiment_score = 0.5  # Neutral default
        
        # Use positive/negative word detection (simple approach)
        positive_words = ['beat', 'exceed', 'positive', 'growth', 'up', 'higher', 'bullish', 'strong']
        negative_words = ['miss', 'below', 'negative', 'decline', 'down', 'lower', 'bearish', 'weak']
        
        # Check for positive/negative words in headline and adjust sentiment
        text_lower = (news_headline + " " + news_text).lower()
        pos_count = sum(word in text_lower for word in positive_words)
        neg_count = sum(word in text_lower for word in negative_words)
        
        if pos_count > neg_count:
            sentiment_score = 0.5 + min(0.4, 0.1 * (pos_count - neg_count))
        elif neg_count > pos_count:
            sentiment_score = 0.5 - min(0.4, 0.1 * (neg_count - pos_count))
        
        # Add sentiment data to the analyzer
        self.sentiment_analyzer.add_sentiment_item(
            source='news',
            sentiment=sentiment_score,
            metadata={
                'headline': news_headline,
                'source_name': news_source
            }
        )
        
        logger.info(f"Processed news for {self.session.symbol}: " +
                   f"{news_headline}, sentiment: {sentiment_score:.2f}")
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
    
    def _on_social_sentiment_update(self, event: Event) -> None:
        """
        Handle social media sentiment update events.
        
        Update sentiment data with social media signals.
        
        Args:
            event: Social sentiment update event
        """
        # Check if the event is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Extract social sentiment data
        social_data = event.data.get('social_data', {})
        sentiment_score = social_data.get('sentiment', 0.5)  # 0-1 scale
        sentiment_volume = social_data.get('volume', 1)  # Number of posts/mentions
        sentiment_source = social_data.get('source', 'general')  # e.g., twitter, reddit
        
        # Add sentiment data to the analyzer
        self.sentiment_analyzer.add_sentiment_item(
            source='social_media',
            sentiment=sentiment_score,
            metadata={
                'volume': sentiment_volume,
                'platform': sentiment_source
            }
        )
        
        # Log social sentiment update
        logger.info(f"Social sentiment update for {self.session.symbol}: " +
                   f"{sentiment_source} sentiment: {sentiment_score:.2f}, " +
                   f"volume: {sentiment_volume}")
        
        # Only check for trades on significant volume of mentions
        if sentiment_volume > 5:  # Arbitrary threshold
            self._check_for_trade_opportunities()
    
    def _on_analyst_rating_change(self, event: Event) -> None:
        """
        Handle analyst rating change events.
        
        Update sentiment data with analyst updates.
        
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
        
        # Convert rating to sentiment score (0-1)
        sentiment_score = 0.5  # Neutral default
        
        # Normalize ratings into sentiment score
        if new_rating.lower() in ['buy', 'outperform', 'overweight', 'strong buy']:
            sentiment_score = 0.8  # Bullish
        elif new_rating.lower() in ['sell', 'underperform', 'underweight', 'strong sell']:
            sentiment_score = 0.2  # Bearish
        elif new_rating.lower() in ['hold', 'neutral', 'market perform', 'sector perform']:
            sentiment_score = 0.5  # Neutral
            
        # Adjust for upgrade/downgrade
        if rating_change == 'upgrade':
            sentiment_score += 0.1
        elif rating_change == 'downgrade':
            sentiment_score -= 0.1
            
        # Ensure score stays in 0-1 range
        sentiment_score = max(0.0, min(1.0, sentiment_score))
        
        # Add sentiment data to the analyzer
        self.sentiment_analyzer.add_sentiment_item(
            source='analyst_reports',
            sentiment=sentiment_score,
            metadata={
                'rating': new_rating,
                'price_target': price_target,
                'rating_change': rating_change,
                'old_rating': old_rating
            }
        )
        
        # Log analyst change
        logger.info(f"Analyst rating change for {self.session.symbol}: " +
                   f"{old_rating} to {new_rating}, PT: {price_target}, " +
                   f"sentiment score: {sentiment_score:.2f}")
        
        # Check for trading opportunities on analyst changes
        self._check_for_trade_opportunities()
    
    def _on_market_data_updated(self, event: Event) -> None:
        """
        Handle market data updated event.
        
        Update internal data and check for sentiment-based trading opportunities.
        
        Args:
            event: Market data updated event
        """
        # Let the base class handle common functionality first
        super()._on_market_data_updated(event)
        
        # Check if the event data is for our symbol
        if event.data.get('symbol') != self.session.symbol:
            return
        
        # Periodically update sentiment
        if len(self.market_data) % 10 == 0:  # Every 10 bars
            self._update_sentiment_data()
            
        # Check for sentiment-based trade opportunities
        self._check_for_trade_opportunities()
    
    def _on_timeframe_completed(self, event: Event) -> None:
        """
        Handle timeframe completed event.
        
        Calculate indicators and check entry conditions.
        
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
        
        # Update sentiment-price divergence with new market data
        if not self.market_data.empty:
            self.sentiment_analyzer.calculate_sentiment_price_divergence(
                self.sentiment_analyzer.sentiment_data['current_score'],
                self.market_data
            )
        
        # Check for trade opportunities
        self._check_for_trade_opportunities()
    
    def _check_for_trade_opportunities(self) -> None:
        """
        Check for sentiment-based trading opportunities.
        """
        # Skip if strategy not active
        if not self.active:
            return
            
        # Skip if no market data or insufficient sentiment data
        if self.market_data.empty or self.sentiment_analyzer.sentiment_data['volume'] < self.parameters['min_sentiment_volume']:
            return
            
        # Check if we've reached our daily trade limit
        if self.sentiment_trades_today >= self.parameters['max_sentiment_trades_per_day']:
            logger.info(f"Trade limit reached for today: {self.sentiment_trades_today}/{self.parameters['max_sentiment_trades_per_day']}")
            return
            
        # Verify stock meets minimum criteria
        if not self._verify_stock_criteria():
            return
        
        # Get sentiment signals from the analyzer
        signals = self.sentiment_analyzer.get_sentiment_signals(
            symbol=self.session.symbol,
            price_data=self.market_data
        )
        
        # Process signals
        for signal in signals:
            # Create a unique signal ID
            signal_id = str(uuid.uuid4())
            
            # Add to pending signals - wait for confirmation
            if signal['type'] in ['long', 'short']:
                self.pending_signals[signal_id] = {
                    'id': signal_id,
                    'type': signal['type'],
                    'confidence': signal['confidence'],
                    'timestamp': datetime.now(),
                    'expiration': datetime.now() + timedelta(hours=24),
                    'source': signal['source'],
                    'reason': signal['reason']
                }
                
                logger.info(f"Generated pending {signal['type'].upper()} signal for {self.session.symbol} " +
                           f"based on {signal['source']} with confidence {signal['confidence']:.2f}")
        
        # Process pending signals for confirmation
        self._process_pending_signals()
    
    def _verify_stock_criteria(self) -> bool:
        """
        Verify that the stock meets minimum criteria for trading.
        
        Returns:
            True if stock passes all filters, False otherwise
        """
        # Check price minimum
        if 'close' in self.market_data and len(self.market_data['close']) > 0:
            current_price = self.market_data['close'].iloc[-1]
            if current_price < self.parameters['min_price']:
                return False
        else:
            return False  # No price data
            
        # Check volume minimum
        if 'volume' in self.market_data and len(self.market_data['volume']) >= 20:
            avg_volume = self.market_data['volume'].iloc[-20:].mean()
            if avg_volume < self.parameters['min_avg_volume']:
                return False
        else:
            return False  # No volume data or not enough history
            
        return True
    
    def _process_pending_signals(self) -> None:
        """
        Process pending signals, looking for confirmation.
        """
        # Skip if no pending signals
        if not self.pending_signals:
            return
            
        now = datetime.now()
        signals_to_remove = []
        
        # Check each pending signal
        for signal_id, signal in self.pending_signals.items():
            # Check if signal expired
            if now > signal['expiration']:
                signals_to_remove.append(signal_id)
                continue
                
            # Check for confirmation
            if self._confirm_signal(signal):
                # Signal confirmed - act on it
                self._act_on_confirmed_signal(signal)
                signals_to_remove.append(signal_id)
        
        # Remove processed signals
        for signal_id in signals_to_remove:
            if signal_id in self.pending_signals:
                del self.pending_signals[signal_id]
    
    def _confirm_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Confirm if a pending signal should be acted upon.
        
        Args:
            signal: The signal to confirm
            
        Returns:
            True if signal is confirmed, False otherwise
        """
        # Check for volume confirmation if required
        if self.parameters['require_volume_confirmation']:
            if 'volume' not in self.market_data or len(self.market_data['volume']) < 20:
                return False
                
            recent_volume = self.market_data['volume'].iloc[-1]
            avg_volume = self.market_data['volume'].iloc[-20:-1].mean()  # Exclude current bar
            
            volume_ratio = recent_volume / avg_volume if avg_volume else 0
            
            # Volume must be above threshold
            if volume_ratio < self.parameters['volume_confirmation_threshold']:
                return False
        
        # Additional confirmation logic could be added here
        # For example, confirming trend direction with price action
        
        if 'close' in self.market_data and len(self.market_data['close']) >= 20:
            # For LONG signals, confirm upward momentum
            if signal['type'] == 'long':
                # Simple check: price above short-term moving average
                sma10 = self.market_data['close'].rolling(window=10).mean().iloc[-1]
                current_price = self.market_data['close'].iloc[-1]
                
                if current_price < sma10:
                    return False  # Price below short MA, don't confirm bullish signal
            
            # For SHORT signals, confirm downward momentum
            elif signal['type'] == 'short':
                # Simple check: price below short-term moving average
                sma10 = self.market_data['close'].rolling(window=10).mean().iloc[-1]
                current_price = self.market_data['close'].iloc[-1]
                
                if current_price > sma10:
                    return False  # Price above short MA, don't confirm bearish signal
        
        return True
    
    def _act_on_confirmed_signal(self, signal: Dict[str, Any]) -> None:
        """
        Act on a confirmed signal by generating a trading signal.
        
        Args:
            signal: The confirmed signal to act on
        """
        # Get current price from market data
        if 'close' not in self.market_data or len(self.market_data['close']) == 0:
            logger.warning(f"Cannot act on signal for {self.session.symbol}: No market data")
            return
            
        current_price = self.market_data['close'].iloc[-1]
        
        # Calculate stop loss and target prices
        stop_loss = None
        target_price = None
        
        # For long signals
        if signal['type'] == 'long':
            # Calculate stop loss as a percentage below entry
            risk_pct = self.parameters['max_loss_pct']
            stop_loss = current_price * (1 - risk_pct)
            
            # Target price based on 2:1 reward-to-risk ratio
            price_risk = current_price - stop_loss
            target_price = current_price + (price_risk * 2)
            
            # Create the trading signal
            trade_signal = Signal(
                id=signal['id'],
                symbol=self.session.symbol,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timestamp=datetime.now(),
                expiration=signal['expiration'],
                confidence=signal['confidence'],
                metadata={
                    'strategy': 'news_sentiment',
                    'source': signal['source'],
                    'reason': signal['reason'],
                    'sentiment_score': self.sentiment_analyzer.sentiment_data['current_score'],
                    'sentiment_volume': self.sentiment_analyzer.sentiment_data['volume'],
                    'sentiment_trend': self.sentiment_analyzer.sentiment_data['trend']
                }
            )
            
        # For short signals
        elif signal['type'] == 'short':
            # Calculate stop loss as a percentage above entry
            risk_pct = self.parameters['max_loss_pct']
            stop_loss = current_price * (1 + risk_pct)
            
            # Target price based on 2:1 reward-to-risk ratio
            price_risk = stop_loss - current_price
            target_price = current_price - (price_risk * 2)
            
            # Create the trading signal
            trade_signal = Signal(
                id=signal['id'],
                symbol=self.session.symbol,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                stop_loss=stop_loss,
                target_price=target_price,
                timestamp=datetime.now(),
                expiration=signal['expiration'],
                confidence=signal['confidence'],
                metadata={
                    'strategy': 'news_sentiment',
                    'source': signal['source'],
                    'reason': signal['reason'],
                    'sentiment_score': self.sentiment_analyzer.sentiment_data['current_score'],
                    'sentiment_volume': self.sentiment_analyzer.sentiment_data['volume'],
                    'sentiment_trend': self.sentiment_analyzer.sentiment_data['trend']
                }
            )
        else:
            logger.warning(f"Unknown signal type: {signal['type']}")
            return
            
        # Emit signal through event bus if available
        if self.event_bus:
            event = Event(
                event_type=EventType.SIGNAL_GENERATED,
                timestamp=datetime.now(),
                data={
                    'signal': trade_signal.to_dict(),
                    'strategy': 'news_sentiment',
                    'timeframe': self.session.timeframe.value
                }
            )
            self.event_bus.emit(event)
        
        # Track the signal and update counters
        self.sentiment_trades_today += 1
        
        logger.info(f"Generated {signal['type'].upper()} signal for {self.session.symbol} at {current_price:.2f} " +
                  f"(SL: {stop_loss:.2f}, TP: {target_price:.2f})")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate technical indicators to supplement sentiment analysis.
        
        Args:
            data: Market data for the symbol
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Skip if not enough data
        if len(data) < 30:
            return indicators
            
        # Extract price and volume data
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume'] if 'volume' in data else None
        
        # Calculate simple moving averages
        indicators['sma20'] = close.rolling(window=20).mean()
        indicators['sma50'] = close.rolling(window=50).mean()
        
        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR (Average True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(window=14).mean()
        
        # Calculate VWAP if volume data available
        if volume is not None and len(volume) > 0:
            # Calculate Cumulative (Price * Volume)
            cumulative_pv = (((high + low + close) / 3) * volume).cumsum()
            # Calculate Cumulative Volume
            cumulative_volume = volume.cumsum()
            # Calculate VWAP
            indicators['vwap'] = cumulative_pv / cumulative_volume
        
        return indicators
    
    def get_sentiment_data(self) -> Dict[str, Any]:
        """
        Get current sentiment data and analysis.
        
        Returns:
            Dictionary with sentiment data and analysis
        """
        # Get the sentiment report from the analyzer
        result = self.sentiment_analyzer.get_sentiment_report()
        
        # Add additional strategy context
        result['symbol'] = self.session.symbol
        result['pending_signals'] = len(self.pending_signals)
        result['trades_today'] = self.sentiment_trades_today
        result['max_trades_per_day'] = self.parameters['max_sentiment_trades_per_day']
        result['strategy_mode'] = self.parameters['trade_mode']
        
        return result
