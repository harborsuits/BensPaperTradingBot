"""
Forex News Trading Strategy

This strategy focuses on trading currency pairs around significant economic news events.
It monitors economic calendars, analyzes news sentiment, and executes trades based on
expected and actual impact of news events. 

The strategy can operate in either:
1. Announcement-spike mode: Entering positions immediately after news events
2. Pre-event positioning: Taking positions before the news based on expectations
3. Post-event reaction: Trading based on the market's reaction to news

Features:
- Integration with economic calendar data sources
- News sentiment analysis and impact categorization  
- Session-aware trading parameters
- Volatility-based position sizing and risk management
- Specialized exit conditions for news event trades
"""

import re
import pytz
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np

from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.models.signal import Signal
from trading_bot.services.economic_calendar import EconomicCalendar, NewsEvent


@register_strategy(
    asset_class="forex",
    strategy_type="news",
    name="ForexNewsTrading",
    description="Trades forex pairs based on economic news events and their market impact",
    parameters={
        "default": {
            # Base news event parameters
            "high_impact_only": True,  # Only trade high-impact news
            "news_look_ahead_hours": 24,  # Look ahead window for detecting upcoming news
            "news_look_back_hours": 6,   # Look back window for recent news impact
            
            # Calendar sources and settings
            "economic_calendar_sources": ["forexfactory", "investing"],  # Sources for news data
            "preferred_event_types": [
                "Interest Rate Decision",
                "Non-Farm Payrolls",
                "GDP",
                "CPI",
                "Retail Sales",
                "PMI",
                "Employment Change",
                "Trade Balance"
            ],
            
            # Currency importance ranking (higher = more significant)
            "currency_importance": {
                "USD": 10,
                "EUR": 9,
                "GBP": 8,
                "JPY": 7,
                "AUD": 6,
                "CAD": 6,
                "NZD": 5,
                "CHF": 5
            },
            
            # Trading parameters
            "pre_event_minutes": 30,     # Minutes before event to consider pre-positioning
            "post_event_minutes": 60,    # Minutes after event to monitor for trading opportunities
            "trade_direction_mode": "deviation",  # "deviation", "trend", or "reversal"
            "min_volatility_multiple": 1.5,  # Minimum volatility increase to consider trading
            
            # Risk parameters  
            "pre_event_risk_per_trade": 0.005,  # 0.5% account risk for pre-event trades
            "post_event_risk_per_trade": 0.01,  # 1% account risk for post-event trades
            "min_stop_loss_pips": 15,
            "max_stop_loss_pips": 100,
            "min_risk_reward_ratio": 1.5,
            "default_take_profit_pips": 50,
            
            # General parameters
            "min_position_size": 0.01,  # Minimum position size in lots
            "max_position_size": 5.0,   # Maximum position size in lots
            "timezone": pytz.UTC,
            
            # News-specific filters
            "min_surprise_score": 0.7,   # Minimum surprise (actual vs expected) to consider trading
            "prevent_trading_on_conflict": True,  # Don't trade when multiple events conflict
            "max_event_count_per_day": 3,  # Maximum number of events to trade per day
        }
    }
)
class ForexNewsTradingStrategy(ForexBaseStrategy):
    """
    A strategy that trades forex pairs around significant economic news events.
    
    This strategy monitors economic calendars, analyzes news sentiment, and 
    executes trades based on expected and actual impact of news events.
    
    It supports multiple trading modes including announcement-spike trading,
    pre-event positioning, and post-event reaction trading.
    """
    
    def __init__(self, session=None):
        """
        Initialize the news trading strategy.
        
        Args:
            session: Trading session object with configuration
        """
        super().__init__(session)
        self.name = "ForexNewsTrading"
        self.description = "Trades forex pairs based on economic news events"
        self.logger = logging.getLogger(__name__)
        
        # Initialize economic calendar integration
        self.economic_calendar = EconomicCalendar(
            sources=self.parameters["economic_calendar_sources"]
        )
        
        # Track active news events
        self.active_events = {}  # Symbol -> NewsEvent mapping
        self.upcoming_events = []  # List of upcoming events
        self.recent_events = []  # List of recent events
        
        # News impact tracking
        self.event_impact_history = {}  # Event type -> historical impact data
        
        # Cache for currency pair -> currencies mapping
        self.pair_currencies = {}  # EURUSD -> (EUR, USD)
        
    def initialize(self) -> None:
        """Initialize strategy and load any required data."""
        super().initialize()
        
        # Initialize economic calendar
        self.economic_calendar.initialize()
        
        # Update upcoming events
        self._update_upcoming_events()
        
        # Initialize historical impact data
        self._initialize_impact_history()
        
    def _update_upcoming_events(self) -> None:
        """Update the list of upcoming economic events."""
        now = datetime.now(self.parameters["timezone"])
        look_ahead = now + timedelta(hours=self.parameters["news_look_ahead_hours"])
        
        # Get events from economic calendar
        self.upcoming_events = self.economic_calendar.get_events(
            start_time=now,
            end_time=look_ahead,
            impact_level="high" if self.parameters["high_impact_only"] else None
        )
        
        # Also update recent events
        look_back = now - timedelta(hours=self.parameters["news_look_back_hours"])
        self.recent_events = self.economic_calendar.get_events(
            start_time=look_back,
            end_time=now,
            impact_level="high" if self.parameters["high_impact_only"] else None
        )
        
        self.logger.info(f"Updated upcoming events: {len(self.upcoming_events)} events found")
        self.logger.info(f"Updated recent events: {len(self.recent_events)} events found")
    
    def _initialize_impact_history(self) -> None:
        """Initialize historical impact data for different event types."""
        # This would normally load from a database or file
        # For now, we'll use a simple default impact model
        
        # Default impact model (based on average pips movement) for major event types
        default_impacts = {
            "Interest Rate Decision": {
                "USD": 80, "EUR": 70, "GBP": 65, "JPY": 60,
                "AUD": 50, "CAD": 45, "NZD": 40, "CHF": 35
            },
            "Non-Farm Payrolls": {
                "USD": 90, "EUR": 30, "GBP": 25, "JPY": 20,
                "AUD": 15, "CAD": 20, "NZD": 10, "CHF": 10
            },
            "GDP": {
                "USD": 60, "EUR": 50, "GBP": 45, "JPY": 40,
                "AUD": 35, "CAD": 35, "NZD": 30, "CHF": 25
            },
            "CPI": {
                "USD": 55, "EUR": 45, "GBP": 40, "JPY": 35,
                "AUD": 30, "CAD": 30, "NZD": 25, "CHF": 20
            },
            "Retail Sales": {
                "USD": 50, "EUR": 40, "GBP": 35, "JPY": 30,
                "AUD": 30, "CAD": 25, "NZD": 20, "CHF": 15
            },
            "PMI": {
                "USD": 40, "EUR": 40, "GBP": 35, "JPY": 25,
                "AUD": 25, "CAD": 20, "NZD": 15, "CHF": 15
            },
            "Employment Change": {
                "USD": 70, "EUR": 50, "GBP": 45, "JPY": 35,
                "AUD": 40, "CAD": 35, "NZD": 30, "CHF": 20
            },
            "Trade Balance": {
                "USD": 30, "EUR": 25, "GBP": 25, "JPY": 25,
                "AUD": 30, "CAD": 25, "NZD": 20, "CHF": 15
            }
        }
        
        # Initialize event impact history with default values
        self.event_impact_history = default_impacts
    
    def _parse_currency_pair(self, pair: str) -> Tuple[str, str]:
        """
        Parse a currency pair into its component currencies.
        
        Args:
            pair: Currency pair symbol (e.g., "EURUSD")
            
        Returns:
            Tuple of (base_currency, quote_currency)
        """
        # Check cache first
        if pair in self.pair_currencies:
            return self.pair_currencies[pair]
            
        # Standard currency pairs are 6 characters (EURUSD, GBPJPY)
        if len(pair) >= 6:
            base = pair[:3]
            quote = pair[3:6]
            self.pair_currencies[pair] = (base, quote)
            return base, quote
        
        # If non-standard format, return empty
        return ("", "")
    
    def _find_relevant_events(self, symbol: str) -> List[NewsEvent]:
        """
        Find relevant news events for a specific currency pair.
        
        Args:
            symbol: Currency pair symbol
            
        Returns:
            List of relevant news events
        """
        base, quote = self._parse_currency_pair(symbol)
        if not base or not quote:
            return []
            
        # Find events for either currency in the pair
        relevant_events = []
        now = datetime.now(self.parameters["timezone"])
        pre_event_window = timedelta(minutes=self.parameters["pre_event_minutes"])
        post_event_window = timedelta(minutes=self.parameters["post_event_minutes"])
        
        # Check both upcoming and recent events
        for event in self.upcoming_events + self.recent_events:
            # Check if event is for one of the currencies in the pair
            if event.currency == base or event.currency == quote:
                # Check if event is in our preferred list
                if event.event_type in self.parameters["preferred_event_types"]:
                    # Check timing - if it's within our pre or post window, or upcoming
                    event_time = event.time
                    is_pre_event = now > (event_time - pre_event_window) and now < event_time
                    is_post_event = now > event_time and now < (event_time + post_event_window)
                    is_upcoming = now < (event_time - pre_event_window)
                    
                    if is_pre_event or is_post_event or is_upcoming:
                        relevant_events.append(event)
        
        # Sort by time (nearest first)
        relevant_events.sort(key=lambda e: abs((e.time - now).total_seconds()))
        
        return relevant_events
    
    def _calculate_event_importance(self, event: NewsEvent) -> float:
        """
        Calculate the importance of a news event based on currency and event type.
        
        Args:
            event: News event
            
        Returns:
            Importance score (0-10, higher = more important)
        """
        # Base importance on currency
        currency_importance = self.parameters["currency_importance"].get(event.currency, 5)
        
        # Adjust based on event type
        event_multiplier = 1.0
        preferred_events = self.parameters["preferred_event_types"]
        if event.event_type in preferred_events:
            # Prioritize based on position in preferred list
            event_position = preferred_events.index(event.event_type)
            event_multiplier = 1.0 - (event_position / (len(preferred_events) * 2))
        
        # Adjust based on impact level provided by data source
        impact_multiplier = 1.0
        if event.impact == "high":
            impact_multiplier = 1.5
        elif event.impact == "medium":
            impact_multiplier = 1.0
        elif event.impact == "low":
            impact_multiplier = 0.5
        
        # Calculate final importance score
        importance = (currency_importance / 10) * event_multiplier * impact_multiplier
        
        # Normalize to 0-10 scale
        return min(10, importance * 10)
    
    def _calculate_surprise_score(self, event: NewsEvent) -> float:
        """
        Calculate how surprising a news event was (actual vs expected).
        
        Args:
            event: News event with actual and expected values
            
        Returns:
            Surprise score (0-1, higher = more surprising)
        """
        # Can only calculate if we have both actual and expected values
        if event.actual is None or event.expected is None:
            return 0.0
            
        try:
            # Convert to float if possible
            actual = float(event.actual) if isinstance(event.actual, (int, float, str)) else 0
            expected = float(event.expected) if isinstance(event.expected, (int, float, str)) else 0
            
            # Avoid division by zero
            if expected == 0:
                if actual == 0:
                    return 0.0  # No surprise
                else:
                    return 0.8  # Significant surprise (arbitrary value)
            
            # Calculate percentage difference
            percent_diff = abs(actual - expected) / abs(expected)
            
            # Normalize to 0-1 range with a cap at 100% difference
            return min(1.0, percent_diff)
        except (ValueError, TypeError):
            # For non-numeric data, use string comparison
            if isinstance(event.actual, str) and isinstance(event.expected, str):
                if event.actual.lower() == event.expected.lower():
                    return 0.0  # No surprise
                else:
                    return 0.8  # Some surprise (arbitrary value)
            
            return 0.0  # Default if we can't calculate
    
    def _predict_event_impact(self, event: NewsEvent, symbol: str) -> Tuple[str, float]:
        """
        Predict the likely impact of a news event on a currency pair.
        
        Args:
            event: News event
            symbol: Currency pair
            
        Returns:
            Tuple of (predicted_direction, confidence)
            where direction is 'buy', 'sell', or 'neutral'
        """
        base, quote = self._parse_currency_pair(symbol)
        if not base or not quote or event.currency not in [base, quote]:
            return ("neutral", 0.0)
        
        # Default direction and confidence
        direction = "neutral"
        confidence = 0.0
        
        # Check if we have actual and expected values for comparison
        if event.actual is not None and event.expected is not None:
            try:
                actual = float(event.actual) if isinstance(event.actual, (int, float, str)) else None
                expected = float(event.expected) if isinstance(event.expected, (int, float, str)) else None
                
                if actual is not None and expected is not None:
                    # Determine if the actual value is better or worse than expected
                    # This is event-specific logic
                    is_better = None
                    
                    # Economic indicators where higher is generally better
                    higher_better_events = [
                        "GDP", "Retail Sales", "Employment Change", "Consumer Confidence", "PMI"
                    ]
                    
                    # Economic indicators where lower is generally better
                    lower_better_events = [
                        "Unemployment Rate", "CPI", "Inflation", "Trade Deficit"
                    ]
                    
                    if event.event_type in higher_better_events:
                        is_better = actual > expected
                    elif event.event_type in lower_better_events:
                        is_better = actual < expected
                    elif "Interest Rate" in event.event_type:
                        # Special case for interest rates - depends on market expectations
                        # Higher rates typically strengthen a currency
                        is_better = actual > expected
                    
                    # Determine direction based on whether the news is better/worse and which
                    # currency in the pair is affected
                    if is_better is not None:
                        # Better than expected news is good for the affected currency
                        if event.currency == base:
                            # Good for base = buy the pair
                            direction = "buy" if is_better else "sell"
                        else:  # event.currency == quote
                            # Good for quote = sell the pair
                            direction = "sell" if is_better else "buy"
                        
                        # Confidence based on surprise score
                        surprise_score = self._calculate_surprise_score(event)
                        importance = self._calculate_event_importance(event) / 10.0
                        confidence = surprise_score * importance
            except (ValueError, TypeError):
                # Default to neutral if we can't parse the values
                pass
        
        return (direction, confidence)
    
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals based on news events and market data.
        
        Args:
            data_dict: Dictionary of market data for each symbol
            
        Returns:
            Dictionary of signals
        """
        signals = {}
        
        # Ensure we have up-to-date news data
        self._update_upcoming_events()
        
        # Track events considered for trading to avoid duplicates
        considered_events = set()
        
        # Check each symbol for tradable news events
        for symbol, data in data_dict.items():
            if data.empty or len(data) < 20:
                continue
                
            # Find relevant events for this pair
            relevant_events = self._find_relevant_events(symbol)
            if not relevant_events:
                continue
                
            # Assess each relevant event
            for event in relevant_events:
                # Skip if we've already considered this event
                event_key = f"{event.currency}_{event.event_type}_{event.time.isoformat()}"
                if event_key in considered_events:
                    continue
                    
                considered_events.add(event_key)
                
                # Determine if we should trade this event
                should_trade, trade_type = self._should_trade_event(event, symbol, data)
                if not should_trade:
                    continue
                    
                # Predict impact and direction
                direction, confidence = self._predict_event_impact(event, symbol)
                if direction == "neutral" or confidence < self.parameters["min_surprise_score"]:
                    continue
                    
                # Check for conflicting events
                if self.parameters["prevent_trading_on_conflict"] and self._has_conflicting_events(event):
                    self.logger.info(f"Skipping {symbol} due to conflicting events for {event.event_type}")
                    continue
                    
                # Calculate volatility-adjusted risk parameters
                stop_loss_pips, take_profit_pips = self._calculate_news_risk_parameters(
                    data, event, trade_type
                )
                
                # Calculate position size
                position_size = self._calculate_news_position_size(
                    symbol, data, stop_loss_pips, event, trade_type
                )
                
                # Create signal
                current_price = data["close"].iloc[-1]
                point_value = 0.0001 if "JPY" not in symbol else 0.01
                
                # Set stop loss and take profit prices
                if direction == "buy":
                    stop_loss_price = current_price - (stop_loss_pips * point_value)
                    take_profit_price = current_price + (take_profit_pips * point_value)
                else:  # direction == "sell"
                    stop_loss_price = current_price + (stop_loss_pips * point_value)
                    take_profit_price = current_price - (take_profit_pips * point_value)
                
                # Create the signal
                signal = Signal(
                    symbol=symbol,
                    signal_type=direction,
                    entry_price=current_price,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    size=position_size,
                    timestamp=datetime.now(self.parameters["timezone"]),
                    timeframe=data.index.freq or "1H",  # Use actual timeframe or default
                    strategy=self.name,
                    strength=confidence,
                    metadata={
                        "event_currency": event.currency,
                        "event_type": event.event_type,
                        "event_time": event.time.isoformat(),
                        "trade_type": trade_type,  # "pre_event" or "post_event"
                        "expected": str(event.expected) if event.expected else "",
                        "actual": str(event.actual) if event.actual else "",
                        "surprise_score": self._calculate_surprise_score(event),
                        "importance": self._calculate_event_importance(event),
                        "expiry_minutes": self.parameters["post_event_minutes"] * 1.5  # 50% extra buffer
                    }
                )
                
                # Add signal to results
                signals[symbol] = signal
                
                self.logger.info(f"Generated {direction} signal for {symbol} based on {event.currency} {event.event_type}")
                
                # Limit the number of events per day
                if len(signals) >= self.parameters["max_event_count_per_day"]:
                    self.logger.info(f"Reached maximum event count for today ({self.parameters['max_event_count_per_day']})")
                    break
        
        return signals
    
    def _should_trade_event(self, event: NewsEvent, symbol: str, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determine if we should trade this event and what type of trade.
        
        Args:
            event: News event
            symbol: Currency pair
            data: Market data
            
        Returns:
            Tuple of (should_trade, trade_type)
            where trade_type is 'pre_event' or 'post_event'
        """
        now = datetime.now(self.parameters["timezone"])
        pre_event_window = timedelta(minutes=self.parameters["pre_event_minutes"])
        post_event_window = timedelta(minutes=self.parameters["post_event_minutes"])
        
        # Determine timing relative to the event
        is_pre_event = now > (event.time - pre_event_window) and now < event.time
        is_post_event = now > event.time and now < (event.time + post_event_window)
        
        # Pre-event trades require expected data
        if is_pre_event and event.expected is not None:
            return (True, "pre_event")
            
        # Post-event trades require actual data
        if is_post_event and event.actual is not None:
            # Check for minimum volatility increase for post-event trades
            if self._check_volatility_increase(data, event):
                return (True, "post_event")
        
        return (False, "")
    
    def _check_volatility_increase(self, data: pd.DataFrame, event: NewsEvent) -> bool:
        """
        Check if there's been a significant volatility increase after the news event.
        
        Args:
            data: Market data
            event: News event
            
        Returns:
            True if volatility increased significantly
        """
        # Calculate pre-event volatility (ATR-like)
        if len(data) < 20:
            return False
            
        # Get event time index
        event_idx = None
        for i, time in enumerate(data.index):
            if time >= event.time:
                event_idx = i
                break
                
        if event_idx is None or event_idx < 10 or event_idx >= len(data) - 3:
            return False  # Not enough data points
        
        # Calculate pre-event volatility (last 10 bars before event)
        pre_event_data = data.iloc[event_idx-10:event_idx]
        pre_event_volatility = (pre_event_data["high"] - pre_event_data["low"]).mean()
        
        # Calculate post-event volatility (up to 3 bars after event)
        post_event_data = data.iloc[event_idx:event_idx+3]
        post_event_volatility = (post_event_data["high"] - post_event_data["low"]).mean()
        
        # Check if volatility increased by the required multiple
        volatility_ratio = post_event_volatility / pre_event_volatility if pre_event_volatility > 0 else 0
        return volatility_ratio >= self.parameters["min_volatility_multiple"]
    
    def _has_conflicting_events(self, event: NewsEvent) -> bool:
        """
        Check if there are conflicting events that might affect the trade.
        
        Args:
            event: News event
            
        Returns:
            True if there are conflicting events
        """
        # Look for other high-impact events in a narrow time window
        event_time = event.time
        conflict_window = timedelta(minutes=30)  # 30 min before and after
        
        for other_event in self.upcoming_events + self.recent_events:
            # Skip the same event
            if other_event.currency == event.currency and other_event.event_type == event.event_type:
                continue
                
            # Check if it's in the conflict window
            time_diff = abs((other_event.time - event_time).total_seconds())
            if time_diff <= conflict_window.total_seconds():
                # Check if it's high impact
                if other_event.impact == "high":
                    return True
        
        return False
    
    def _calculate_news_risk_parameters(self, data: pd.DataFrame, event: NewsEvent, 
                                      trade_type: str) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels for news trades.
        
        Args:
            data: Market data
            event: News event
            trade_type: 'pre_event' or 'post_event'
            
        Returns:
            Tuple of (stop_loss_pips, take_profit_pips)
        """
        # Calculate volatility-based stop loss
        atr = self._calculate_atr(data, 14)  # 14-period ATR
        symbol = data.name if hasattr(data, 'name') else ""
        point_value = 0.0001 if "JPY" not in symbol else 0.01
        atr_pips = atr / point_value
        
        # Base stop loss on historical impact for this event type
        event_impact = 0
        if event.event_type in self.event_impact_history and event.currency in self.event_impact_history[event.event_type]:
            event_impact = self.event_impact_history[event.event_type][event.currency]
        
        # Higher of ATR-based or event-based stop loss
        base_stop_loss = max(atr_pips * 2, event_impact * 0.8)
        
        # Adjust based on trade type
        if trade_type == "pre_event":
            # Pre-event trades need wider stops to handle volatility spike
            stop_loss_pips = base_stop_loss * 1.5
        else:  # post_event
            # Post-event can have tighter stops as we've already seen the reaction
            stop_loss_pips = base_stop_loss
        
        # Apply min/max constraints
        stop_loss_pips = max(self.parameters["min_stop_loss_pips"], 
                            min(self.parameters["max_stop_loss_pips"], stop_loss_pips))
        
        # Calculate take profit based on risk-reward ratio
        take_profit_pips = max(
            stop_loss_pips * self.parameters["min_risk_reward_ratio"],
            self.parameters["default_take_profit_pips"]
        )
        
        # Round to 0.1 pip precision
        stop_loss_pips = round(stop_loss_pips * 10) / 10
        take_profit_pips = round(take_profit_pips * 10) / 10
        
        return stop_loss_pips, take_profit_pips
    
    def _calculate_news_position_size(self, symbol: str, data: pd.DataFrame, 
                                    stop_loss_pips: float, event: NewsEvent, 
                                    trade_type: str) -> float:
        """
        Calculate position size for news trades based on risk amount and stop loss distance.
        
        Args:
            symbol: Currency pair
            data: Market data
            stop_loss_pips: Stop loss in pips
            event: News event
            trade_type: 'pre_event' or 'post_event'
            
        Returns:
            Position size in lots
        """
        # Get risk percentage based on trade type
        if trade_type == "pre_event":
            risk_pct = self.parameters["pre_event_risk_per_trade"]
        else:  # post_event
            risk_pct = self.parameters["post_event_risk_per_trade"]
        
        # Adjust risk based on event importance
        importance = self._calculate_event_importance(event) / 10.0  # Normalize to 0-1
        adjusted_risk_pct = risk_pct * importance
        
        # Calculate risk amount
        account_balance = self.session.account_balance
        risk_amount = account_balance * adjusted_risk_pct
        
        # Calculate position size using standard method
        current_price = data["close"].iloc[-1]
        point_value = 0.0001 if "JPY" not in symbol else 0.01
        stop_loss_price_diff = stop_loss_pips * point_value
        
        # Calculate pip value for standard lot (100,000 units)
        standard_lot = 100000
        pip_value_in_quote = standard_lot * point_value
        
        # Calculate position size in standard lots
        if stop_loss_pips > 0 and pip_value_in_quote > 0:
            position_size_in_standard_lots = risk_amount / (stop_loss_pips * pip_value_in_quote)
        else:
            position_size_in_standard_lots = 0
        
        # Apply position size limits
        min_lot = self.parameters["min_position_size"]
        max_lot = self.parameters["max_position_size"]
        position_size_in_standard_lots = max(min(position_size_in_standard_lots, max_lot), min_lot)
        
        return position_size_in_standard_lots
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range for volatility assessment.
        
        Args:
            data: OHLCV data
            period: ATR period
            
        Returns:
            ATR value
        """
        if len(data) < period:
            return 0.0
            
        high_low = data["high"] - data["low"]
        high_close = abs(data["high"] - data["close"].shift(1))
        low_close = abs(data["low"] - data["close"].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Check if a news-based position should be exited due to changing conditions.
        
        Args:
            position: Current position information
            data: Latest market data
            
        Returns:
            True if position should be exited, False otherwise
        """
        if data.empty or len(data) < 5:
            return False
            
        # Extract position metadata
        symbol = position["symbol"]
        entry_time_str = position.get("metadata", {}).get("entry_time")
        event_time_str = position.get("metadata", {}).get("event_time")
        trade_type = position.get("metadata", {}).get("trade_type")
        expiry_minutes = position.get("metadata", {}).get("expiry_minutes", 
                                                       self.parameters["post_event_minutes"])
        
        if not all([entry_time_str, event_time_str, trade_type]):
            return False  # Missing required metadata
        
        # Parse timestamps
        try:
            entry_time = datetime.fromisoformat(entry_time_str)
            event_time = datetime.fromisoformat(event_time_str)
        except (ValueError, TypeError):
            return False  # Invalid timestamps
        
        now = datetime.now(self.parameters["timezone"])
        
        # Check time-based exit conditions
        if trade_type == "pre_event":
            # For pre-event trades, exit after the event has occurred
            if now > event_time:
                self.logger.info(f"Exiting pre-event trade for {symbol} as event time has passed")
                return True
        else:  # post_event
            # For post-event trades, exit after expiry time
            expiry_time = entry_time + timedelta(minutes=float(expiry_minutes))
            if now > expiry_time:
                self.logger.info(f"Exiting post-event trade for {symbol} as expiry time has passed")
                return True
                
        # Check if volatility has decreased significantly (for post-event trades)
        if trade_type == "post_event":
            original_atr = position.get("metadata", {}).get("original_atr", 0)
            current_atr = self._calculate_atr(data)
            
            if original_atr > 0 and current_atr < original_atr * 0.5:
                self.logger.info(f"Exiting post-event trade for {symbol} as volatility has returned to normal")
                return True
        
        return False
        
    def _update_upcoming_events(self) -> None:
        """Update the list of upcoming economic events."""
        now = datetime.now(self.parameters["timezone"])
        look_ahead = now + timedelta(hours=self.parameters["news_look_ahead_hours"])
        
        # Get events from economic calendar
        self.upcoming_events = self.economic_calendar.get_events(
            start_time=now,
            end_time=look_ahead,
            impact_level="high" if self.parameters["high_impact_only"] else None
        )
        
        # Also update recent events
        look_back = now - timedelta(hours=self.parameters["news_look_back_hours"])
        self.recent_events = self.economic_calendar.get_events(
            start_time=look_back,
            end_time=now,
            impact_level="high" if self.parameters["high_impact_only"] else None
        )
        
        self.logger.info(f"Updated upcoming events: {len(self.upcoming_events)} events found")
        self.logger.info(f"Updated recent events: {len(self.recent_events)} events found")
