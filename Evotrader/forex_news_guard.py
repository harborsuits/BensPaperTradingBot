#!/usr/bin/env python3
"""
Forex News Guard - Risk Management Module

This module detects and filters trades around high-impact economic news events.
It integrates with BenBot for decision making and acts as a safety filter
for strategies evolved by EvoTrader.
"""

import datetime
import pandas as pd
import numpy as np
import requests
import json
import os
import logging
import yaml
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_news_guard')


class NewsImpact(Enum):
    """Enum representing the impact level of economic news events."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class NewsEvent:
    """Dataclass representing an economic news event."""
    title: str
    datetime: datetime.datetime
    currency: str
    impact: NewsImpact
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    url: Optional[str] = None
    
    def affects_pair(self, pair: str) -> bool:
        """Check if this news event affects the given currency pair."""
        if not pair or len(pair) != 6:
            return False
        
        base = pair[0:3]
        quote = pair[3:6]
        return self.currency == base or self.currency == quote
    
    def time_until(self, from_time: Optional[datetime.datetime] = None) -> datetime.timedelta:
        """Calculate time until this news event."""
        if from_time is None:
            from_time = datetime.datetime.now()
        return self.datetime - from_time
    
    def time_since(self, from_time: Optional[datetime.datetime] = None) -> datetime.timedelta:
        """Calculate time since this news event."""
        if from_time is None:
            from_time = datetime.datetime.now()
        return from_time - self.datetime
    
    def is_imminent(self, minutes: int = 30, from_time: Optional[datetime.datetime] = None) -> bool:
        """Check if news event is coming up within specified minutes."""
        if from_time is None:
            from_time = datetime.datetime.now()
        return 0 < (self.datetime - from_time).total_seconds() <= minutes * 60
    
    def is_recent(self, minutes: int = 30, from_time: Optional[datetime.datetime] = None) -> bool:
        """Check if news event occurred within last specified minutes."""
        if from_time is None:
            from_time = datetime.datetime.now()
        return 0 < (from_time - self.datetime).total_seconds() <= minutes * 60


class NewsGuard:
    """
    Forex News Guard
    
    Detects and filters trades around high-impact economic news events.
    Integrates with BenBot for decision making and control signals.
    Acts as a safety layer for EvoTrader strategies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the News Guard.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Default configuration
        self.config = {
            'data_sources': {
                'api_key': None,
                'use_mocks': True,  # Use mock data when no API key available
                'cache_expiry_hours': 12,
                'endpoints': {
                    'forex_factory': 'https://api.forexfactory.com/calendar',
                    'investing': 'https://api.investing.com/calendar'
                }
            },
            'filters': {
                'high_impact_buffer_minutes': 30,
                'medium_impact_buffer_minutes': 15,
                'low_impact_buffer_minutes': 0,
                'post_news_resume_minutes': 15,
                'currency_specific_buffers': {
                    'USD': 45,  # Extra buffer for USD news
                    'EUR': 30
                },
                'event_specific_buffers': {
                    'Non-Farm Payrolls': 60,
                    'FOMC': 120,
                    'ECB Rate Decision': 60
                }
            },
            'benbot_integration': {
                'signal_endpoint': 'http://localhost:8080/benbot/signal',
                'notify_on_news': True,
                'allow_override': False  # Whether BenBot can override news restrictions
            },
            'logging': {
                'level': 'INFO',
                'file': 'news_guard.log',
                'notify_events': ['HIGH']
            }
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Update default config with user settings
                    self._deep_update(self.config, user_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Initialize news cache
        self.news_cache = []
        self.last_update = None
        self.update_news_events()
        
        logger.info("Forex News Guard initialized")
        
        # Check if BenBot integration is available
        self.benbot_available = self._check_benbot_connection()
        if self.benbot_available:
            logger.info("BenBot integration confirmed")
        else:
            logger.warning("BenBot integration unavailable - running in standalone mode")
    
    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Recursively update a nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    def _check_benbot_connection(self) -> bool:
        """Check if BenBot integration is available."""
        if not self.config['benbot_integration']['signal_endpoint']:
            return False
        
        try:
            # Simple ping to BenBot
            endpoint = self.config['benbot_integration']['signal_endpoint'].rstrip('/') + '/ping'
            response = requests.get(endpoint, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def update_news_events(self, force: bool = False) -> List[NewsEvent]:
        """
        Update the internal news events cache.
        
        Args:
            force: Force update even if cache is still valid
            
        Returns:
            List of news events
        """
        now = datetime.datetime.now()
        
        # Check if we need to update
        if not force and self.last_update and \
           (now - self.last_update).total_seconds() < self.config['data_sources']['cache_expiry_hours'] * 3600:
            return self.news_cache
        
        events = []
        
        # Try to get news from API if key is available
        if self.config['data_sources']['api_key']:
            try:
                events = self._fetch_live_news_events()
                logger.info(f"Fetched {len(events)} news events from live API")
            except Exception as e:
                logger.error(f"Error fetching news data: {e}")
                # Fall back to mock data
                events = self._get_mock_news_events()
        else:
            # Use mock data
            events = self._get_mock_news_events()
            logger.info(f"Using {len(events)} mock news events (no API key)")
        
        # Update cache
        self.news_cache = events
        self.last_update = now
        
        # Notify BenBot of high impact events if configured
        if self.benbot_available and self.config['benbot_integration']['notify_on_news']:
            high_impact = [e for e in events if e.impact == NewsImpact.HIGH and e.is_imminent(60)]
            if high_impact:
                self._notify_benbot_of_events(high_impact)
        
        return events
    
    def _fetch_live_news_events(self) -> List[NewsEvent]:
        """Fetch live news events from the configured API."""
        events = []
        
        # Date range (next 7 days)
        now = datetime.datetime.now()
        start_date = now.strftime('%Y-%m-%d')
        end_date = (now + datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Try ForexFactory API first
        try:
            endpoint = self.config['data_sources']['endpoints']['forex_factory']
            params = {
                'api_key': self.config['data_sources']['api_key'],
                'from': start_date,
                'to': end_date
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('events', []):
                    impact_map = {'low': NewsImpact.LOW, 'medium': NewsImpact.MEDIUM, 'high': NewsImpact.HIGH}
                    
                    event = NewsEvent(
                        title=item.get('title', 'Unknown Event'),
                        datetime=datetime.datetime.fromisoformat(item.get('date', now.isoformat())),
                        currency=item.get('currency', 'Unknown'),
                        impact=impact_map.get(item.get('impact', 'low').lower(), NewsImpact.LOW),
                        forecast=item.get('forecast'),
                        previous=item.get('previous'),
                        actual=item.get('actual'),
                        url=item.get('url')
                    )
                    events.append(event)
        except Exception as e:
            logger.error(f"Error fetching from ForexFactory: {e}")
            # Fall back to mock data
            return self._get_mock_news_events()
        
        return events
    
    def _get_mock_news_events(self) -> List[NewsEvent]:
        """Generate mock news events for testing."""
        events = []
        now = datetime.datetime.now()
        
        # Some important upcoming events
        upcoming_events = [
            {
                'title': 'US Non-Farm Payrolls',
                'days_ahead': 2,
                'hour': 13,
                'minute': 30,
                'currency': 'USD',
                'impact': NewsImpact.HIGH
            },
            {
                'title': 'ECB Interest Rate Decision',
                'days_ahead': 1,
                'hour': 12,
                'minute': 45,
                'currency': 'EUR',
                'impact': NewsImpact.HIGH
            },
            {
                'title': 'UK GDP',
                'days_ahead': 3,
                'hour': 9,
                'minute': 0,
                'currency': 'GBP',
                'impact': NewsImpact.MEDIUM
            },
            {
                'title': 'Bank of Japan Policy Rate',
                'days_ahead': 5,
                'hour': 3,
                'minute': 0,
                'currency': 'JPY',
                'impact': NewsImpact.HIGH
            },
            {
                'title': 'Australia Employment Change',
                'days_ahead': 4,
                'hour': 1,
                'minute': 30,
                'currency': 'AUD',
                'impact': NewsImpact.MEDIUM
            },
            {
                'title': 'Canada CPI',
                'days_ahead': 2,
                'hour': 13,
                'minute': 30,
                'currency': 'CAD',
                'impact': NewsImpact.MEDIUM
            }
        ]
        
        # Add a very close event for testing
        imminent_event = {
            'title': 'US ISM Manufacturing PMI',
            'days_ahead': 0,
            'hour': now.hour,
            'minute': now.minute + 10,  # 10 minutes from now
            'currency': 'USD',
            'impact': NewsImpact.HIGH
        }
        upcoming_events.append(imminent_event)
        
        # Add a recent event for testing
        recent_event = {
            'title': 'US Federal Reserve Rate Decision',
            'days_ahead': 0,
            'hour': now.hour,
            'minute': now.minute - 15,  # 15 minutes ago
            'currency': 'USD',
            'impact': NewsImpact.HIGH
        }
        upcoming_events.append(recent_event)
        
        # Convert to NewsEvent objects
        for event in upcoming_events:
            event_date = now + datetime.timedelta(days=event['days_ahead'])
            event_datetime = datetime.datetime(
                event_date.year, event_date.month, event_date.day,
                event['hour'], event['minute']
            )
            
            events.append(NewsEvent(
                title=event['title'],
                datetime=event_datetime,
                currency=event['currency'],
                impact=event['impact'],
                forecast='0.2%',
                previous='0.1%'
            ))
        
        return events
    
    def _notify_benbot_of_events(self, events: List[NewsEvent]) -> None:
        """Notify BenBot of upcoming high-impact events."""
        if not self.benbot_available:
            return
        
        try:
            endpoint = self.config['benbot_integration']['signal_endpoint'].rstrip('/') + '/news'
            
            events_data = []
            for event in events:
                events_data.append({
                    'title': event.title,
                    'datetime': event.datetime.isoformat(),
                    'currency': event.currency,
                    'impact': event.impact.name,
                    'time_until_minutes': event.time_until().total_seconds() / 60
                })
            
            payload = {
                'source': 'EvoTrader',
                'module': 'NewsGuard',
                'events': events_data,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            response = requests.post(endpoint, json=payload, timeout=5)
            if response.status_code != 200:
                logger.warning(f"Failed to notify BenBot: {response.status_code}")
        except Exception as e:
            logger.error(f"Error notifying BenBot: {e}")
    
    def get_upcoming_events(self, 
                           hours: int = 24, 
                           min_impact: NewsImpact = NewsImpact.MEDIUM,
                           currency: Optional[str] = None) -> List[NewsEvent]:
        """
        Get upcoming news events filtered by criteria.
        
        Args:
            hours: Hours ahead to look for events
            min_impact: Minimum impact level to include
            currency: Filter by currency
            
        Returns:
            List of filtered upcoming events
        """
        # Ensure cache is up-to-date
        self.update_news_events()
        
        now = datetime.datetime.now()
        cutoff = now + datetime.timedelta(hours=hours)
        
        filtered = []
        for event in self.news_cache:
            # Skip events in the past
            if event.datetime < now:
                continue
                
            # Skip events too far in the future
            if event.datetime > cutoff:
                continue
                
            # Skip events below minimum impact
            if event.impact.value < min_impact.value:
                continue
                
            # Filter by currency if specified
            if currency and event.currency != currency:
                continue
                
            filtered.append(event)
        
        # Sort by datetime
        filtered.sort(key=lambda e: e.datetime)
        
        return filtered
    
    def can_trade(self, 
                 pair: str,
                 timestamp: Optional[datetime.datetime] = None,
                 allow_benbot_override: bool = None) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed for the given pair at the specified time.
        
        Args:
            pair: Currency pair (e.g., 'EURUSD')
            timestamp: Time to check (default: now)
            allow_benbot_override: Whether to allow BenBot to override restrictions
                                  (defaults to config setting)
            
        Returns:
            Tuple of (can_trade, reason)
            - can_trade: True if trading is allowed, False if restricted
            - reason: String explaining why trading is restricted, or None if allowed
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        if allow_benbot_override is None:
            allow_benbot_override = self.config['benbot_integration']['allow_override']
        
        # First check if BenBot provides an override
        if self.benbot_available and allow_benbot_override:
            override = self._check_benbot_override(pair, timestamp)
            if override is not None:
                return override
        
        # Ensure we have the latest news
        self.update_news_events()
        
        # Extract currencies from the pair
        if not pair or len(pair) != 6:
            return False, "Invalid currency pair format"
        
        base = pair[0:3]
        quote = pair[3:6]
        
        # Check upcoming events
        for event in self.news_cache:
            # Skip past events
            if event.datetime < timestamp:
                continue
            
            # Check if event affects this pair
            if event.currency not in [base, quote]:
                continue
            
            # Get appropriate buffer based on impact and currency
            buffer_minutes = 0
            
            # First check if there's a specific buffer for this event
            event_title_lower = event.title.lower()
            for key, minutes in self.config['filters']['event_specific_buffers'].items():
                if key.lower() in event_title_lower:
                    buffer_minutes = minutes
                    break
            
            # If no event-specific buffer, use impact-based buffer
            if buffer_minutes == 0:
                if event.impact == NewsImpact.HIGH:
                    buffer_minutes = self.config['filters']['high_impact_buffer_minutes']
                elif event.impact == NewsImpact.MEDIUM:
                    buffer_minutes = self.config['filters']['medium_impact_buffer_minutes']
                elif event.impact == NewsImpact.LOW:
                    buffer_minutes = self.config['filters']['low_impact_buffer_minutes']
            
            # Add extra buffer for specific currencies if configured
            currency_extra = self.config['filters']['currency_specific_buffers'].get(event.currency, 0)
            buffer_minutes += currency_extra
            
            # Calculate time until event
            time_until = event.datetime - timestamp
            minutes_until = time_until.total_seconds() / 60
            
            # Check if we're within the buffer window
            if minutes_until <= buffer_minutes:
                return False, f"Upcoming {event.impact.name} impact {event.currency} news: {event.title} in {int(minutes_until)} minutes"
        
        # Check recent events
        for event in self.news_cache:
            # Skip future events
            if event.datetime > timestamp:
                continue
            
            # Skip low impact events for post-news restrictions
            if event.impact == NewsImpact.LOW:
                continue
                
            # Check if event affects this pair
            if event.currency not in [base, quote]:
                continue
            
            # Calculate time since event
            time_since = timestamp - event.datetime
            minutes_since = time_since.total_seconds() / 60
            
            # Check if we're still in the post-news restricted period
            resume_minutes = self.config['filters']['post_news_resume_minutes']
            if minutes_since <= resume_minutes:
                return False, f"Recent {event.impact.name} impact {event.currency} news: {event.title} {int(minutes_since)} minutes ago"
        
        # No restrictions found
        return True, None
    
    def _check_benbot_override(self, pair: str, timestamp: datetime.datetime) -> Optional[Tuple[bool, Optional[str]]]:
        """
        Check if BenBot provides an override for news restrictions.
        
        Args:
            pair: Currency pair
            timestamp: Timestamp to check
            
        Returns:
            Override decision or None if no override
        """
        if not self.benbot_available:
            return None
        
        try:
            endpoint = self.config['benbot_integration']['signal_endpoint'].rstrip('/') + '/override'
            
            payload = {
                'source': 'EvoTrader',
                'module': 'NewsGuard',
                'pair': pair,
                'timestamp': timestamp.isoformat(),
                'restriction_type': 'news'
            }
            
            response = requests.post(endpoint, json=payload, timeout=2)
            if response.status_code == 200:
                data = response.json()
                override = data.get('override', False)
                reason = data.get('reason')
                
                if override:
                    logger.info(f"BenBot override: Allowing trade for {pair} despite news restrictions: {reason}")
                    return True, None
                
            # No override if we reach here
            return None
            
        except Exception as e:
            logger.error(f"Error checking BenBot override: {e}")
            return None
    
    def annotate_dataframe(self, 
                          df: pd.DataFrame, 
                          pair: str) -> pd.DataFrame:
        """
        Annotate a DataFrame with news event information.
        
        Args:
            df: DataFrame with datetime index
            pair: Currency pair to check for news
            
        Returns:
            DataFrame with news annotations
        """
        # Create a copy to avoid modifying the input
        result = df.copy()
        
        # Add news columns
        result['has_high_impact_news'] = False
        result['has_medium_impact_news'] = False
        result['news_event'] = None
        result['minutes_to_news'] = None
        result['minutes_from_news'] = None
        
        # Process each row
        for idx, timestamp in enumerate(result.index):
            # Get the nearest upcoming and recent news events
            upcoming = self._get_nearest_news(pair, timestamp, future=True)
            recent = self._get_nearest_news(pair, timestamp, future=False)
            
            # Mark high impact news
            if (upcoming and upcoming.impact == NewsImpact.HIGH and 
                upcoming.time_until(timestamp).total_seconds() / 60 <= self.config['filters']['high_impact_buffer_minutes']):
                result.loc[timestamp, 'has_high_impact_news'] = True
                result.loc[timestamp, 'news_event'] = upcoming.title
                result.loc[timestamp, 'minutes_to_news'] = upcoming.time_until(timestamp).total_seconds() / 60
            
            # Mark medium impact news
            elif (upcoming and upcoming.impact == NewsImpact.MEDIUM and 
                 upcoming.time_until(timestamp).total_seconds() / 60 <= self.config['filters']['medium_impact_buffer_minutes']):
                result.loc[timestamp, 'has_medium_impact_news'] = True
                result.loc[timestamp, 'news_event'] = upcoming.title
                result.loc[timestamp, 'minutes_to_news'] = upcoming.time_until(timestamp).total_seconds() / 60
            
            # Check recent news
            if (recent and recent.impact != NewsImpact.LOW and 
                recent.time_since(timestamp).total_seconds() / 60 <= self.config['filters']['post_news_resume_minutes']):
                if recent.impact == NewsImpact.HIGH:
                    result.loc[timestamp, 'has_high_impact_news'] = True
                else:
                    result.loc[timestamp, 'has_medium_impact_news'] = True
                    
                result.loc[timestamp, 'news_event'] = recent.title
                result.loc[timestamp, 'minutes_from_news'] = recent.time_since(timestamp).total_seconds() / 60
        
        return result
    
    def _get_nearest_news(self, 
                         pair: str, 
                         timestamp: datetime.datetime, 
                         future: bool = True) -> Optional[NewsEvent]:
        """
        Find the nearest news event for a currency pair.
        
        Args:
            pair: Currency pair
            timestamp: Reference timestamp
            future: If True, find next upcoming event; if False, find most recent
            
        Returns:
            Nearest NewsEvent or None if not found
        """
        if not pair or len(pair) != 6:
            return None
        
        base = pair[0:3]
        quote = pair[3:6]
        
        nearest = None
        min_diff = float('inf')
        
        for event in self.news_cache:
            # Skip events that don't affect this pair
            if event.currency not in [base, quote]:
                continue
            
            # Skip future events when looking for past events
            if not future and event.datetime > timestamp:
                continue
                
            # Skip past events when looking for future events
            if future and event.datetime < timestamp:
                continue
            
            # Calculate time difference
            if future:
                diff = (event.datetime - timestamp).total_seconds()
            else:
                diff = (timestamp - event.datetime).total_seconds()
            
            # Update nearest if this is closer
            if 0 <= diff < min_diff:
                nearest = event
                min_diff = diff
        
        return nearest


# Module execution
if __name__ == "__main__":
    import argparse
    from tabulate import tabulate
    
    parser = argparse.ArgumentParser(description="Forex News Guard")
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--check", 
        type=str,
        help="Check if trading is currently allowed for a given pair (e.g., EURUSD)"
    )
    
    parser.add_argument(
        "--upcoming", 
        action="store_true",
        help="Show upcoming news events"
    )
    
    parser.add_argument(
        "--hours", 
        type=int,
        default=24,
        help="Hours ahead to look for events"
    )
    
    parser.add_argument(
        "--impact", 
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum impact level to include"
    )
    
    parser.add_argument(
        "--currency", 
        type=str,
        help="Filter by currency (e.g., USD, EUR)"
    )
    
    args = parser.parse_args()
    
    # Map impact string to enum
    impact_map = {
        "low": NewsImpact.LOW,
        "medium": NewsImpact.MEDIUM,
        "high": NewsImpact.HIGH
    }
    
    # Initialize news guard
    news_guard = NewsGuard(args.config)
    
    # Check if trading is allowed
    if args.check:
        can_trade, reason = news_guard.can_trade(args.check)
        if can_trade:
            print(f"✅ Trading allowed for {args.check}")
        else:
            print(f"❌ Trading restricted for {args.check}: {reason}")
    
    # Show upcoming events
    if args.upcoming:
        events = news_guard.get_upcoming_events(
            hours=args.hours,
            min_impact=impact_map[args.impact],
            currency=args.currency
        )
        
        if events:
            table_data = []
            for event in events:
                time_until = event.time_until()
                hours, remainder = divmod(time_until.total_seconds(), 3600)
                minutes, _ = divmod(remainder, 60)
                
                table_data.append([
                    event.datetime.strftime('%Y-%m-%d %H:%M'),
                    event.currency,
                    event.impact.name,
                    event.title,
                    f"{int(hours)}h {int(minutes)}m"
                ])
            
            print("\nUpcoming Economic News Events:")
            print(tabulate(
                table_data,
                headers=["Date/Time (UTC)", "Currency", "Impact", "Event", "Time Until"],
                tablefmt="grid"
            ))
        else:
            print(f"No upcoming {args.impact}+ impact events found in the next {args.hours} hours" + 
                 (f" for {args.currency}" if args.currency else ""))
    
    # Default output if no arguments
    if not (args.check or args.upcoming):
        # Show a summary of near-term high impact events
        print("Forex News Guard - Event Summary")
        print("================================")
        
        high_events = news_guard.get_upcoming_events(hours=24, min_impact=NewsImpact.HIGH)
        medium_events = news_guard.get_upcoming_events(hours=12, min_impact=NewsImpact.MEDIUM)
        
        if high_events:
            print(f"\nHigh Impact Events (next 24h): {len(high_events)}")
            for event in high_events[:5]:  # Show top 5
                time_until = event.time_until()
                hours, remainder = divmod(time_until.total_seconds(), 3600)
                minutes, _ = divmod(remainder, 60)
                
                print(f"- {event.currency}: {event.title} in {int(hours)}h {int(minutes)}m")
                
            if len(high_events) > 5:
                print(f"  ... and {len(high_events) - 5} more")
        else:
            print("\nNo high impact events in the next 24 hours")
            
        if medium_events:
            print(f"\nMedium Impact Events (next 12h): {len(medium_events)}")
            for event in medium_events[:3]:  # Show top 3
                time_until = event.time_until()
                hours, remainder = divmod(time_until.total_seconds(), 3600)
                minutes, _ = divmod(remainder, 60)
                
                print(f"- {event.currency}: {event.title} in {int(hours)}h {int(minutes)}m")
                
            if len(medium_events) > 3:
                print(f"  ... and {len(medium_events) - 3} more")
        else:
            print("\nNo medium impact events in the next 12 hours")
            
        print("\nUse --upcoming for a complete list of events")
        print("Use --check PAIR to verify if trading is allowed")
