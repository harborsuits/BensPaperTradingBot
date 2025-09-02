#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Event Driven Trading Strategies Package

This package contains implementations of event and earnings-driven strategies:
- Earnings Announcement Strategy
- Economic Release Trading Strategy
- News Sentiment Trading Strategy
- Event-based Momentum Strategy
"""

# Import strategies for easy access
from trading_bot.strategies_new.stocks.event.earnings_announcement_strategy import EarningsAnnouncementStrategy
from trading_bot.strategies_new.stocks.event.news_sentiment_strategy import NewsSentimentStrategy

__all__ = ['EarningsAnnouncementStrategy', 'NewsSentimentStrategy']
