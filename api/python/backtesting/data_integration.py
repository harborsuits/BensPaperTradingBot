#!/usr/bin/env python3
"""
Data Integration Layer for Autonomous ML Backtesting

This module integrates various data sources including:
- News and sentiment data
- Historical price data
- Technical indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyzes news sentiment across multiple dimensions"""
    
    def __init__(self):
        self.political_keywords = {
            'positive': ['policy', 'reform', 'growth', 'support', 'agreement', 'partnership', 'innovation'],
            'negative': ['regulation', 'restriction', 'tariff', 'sanction', 'conflict', 'investigation', 'lawsuit']
        }
        
        self.social_keywords = {
            'positive': ['sustainability', 'ethical', 'diversity', 'inclusion', 'community', 'responsible', 'green'],
            'negative': ['controversy', 'scandal', 'protest', 'discrimination', 'backlash', 'criticism', 'violation']
        }
        
        self.economic_keywords = {
            'positive': ['profit', 'growth', 'revenue', 'expansion', 'earnings', 'dividend', 'upgrade'],
            'negative': ['loss', 'debt', 'inflation', 'recession', 'downgrade', 'bankruptcy', 'decline']
        }
    
    def analyze_text(self, text, keyword_dict):
        """
        Analyze text for keyword matches
        
        Args:
            text: Text to analyze
            keyword_dict: Dictionary of positive/negative keywords
            
        Returns:
            float: Sentiment score (-1 to 1)
        """
        if not text or not isinstance(text, str):
            return 0.0
            
        text = text.lower()
        positive_count = sum(1 for keyword in keyword_dict['positive'] if keyword in text)
        negative_count = sum(1 for keyword in keyword_dict['negative'] if keyword in text)
        
        if positive_count + negative_count == 0:
            return 0.0
            
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def analyze_dimensions(self, news_data):
        """
        Analyze news across political, social, and economic dimensions
        
        Args:
            news_data: List of news articles from news fetcher
            
        Returns:
            dict: Multi-dimensional sentiment analysis
        """
        if not news_data:
            return {
                'overall_sentiment': 0,
                'political_sentiment': 0,
                'social_sentiment': 0,
                'economic_sentiment': 0,
                'news_count': 0,
                'analyzed_articles': []
            }
        
        analyzed_articles = []
        political_scores = []
        social_scores = []
        economic_scores = []
        
        for article in news_data:
            # Combine headline and summary for analysis
            headline = article.get('headline', article.get('title', ''))
            summary = article.get('summary', article.get('description', ''))
            combined_text = f"{headline} {summary}"
            
            # Analyze each dimension
            political_score = self.analyze_text(combined_text, self.political_keywords)
            social_score = self.analyze_text(combined_text, self.social_keywords)
            economic_score = self.analyze_text(combined_text, self.economic_keywords)
            
            # Calculate overall score with economic weighing more heavily
            overall_score = (political_score + social_score + economic_score * 2) / 4
            
            political_scores.append(political_score)
            social_scores.append(social_score)
            economic_scores.append(economic_score)
            
            analyzed_articles.append({
                'headline': headline,
                'source': article.get('source', 'Unknown'),
                'url': article.get('url', ''),
                'timestamp': article.get('datetime', article.get('published_at', '')),
                'political_score': political_score,
                'social_score': social_score,
                'economic_score': economic_score,
                'overall_score': overall_score
            })
        
        return {
            'overall_sentiment': sum([a['overall_score'] for a in analyzed_articles]) / len(analyzed_articles) if analyzed_articles else 0,
            'political_sentiment': sum(political_scores) / len(political_scores) if political_scores else 0,
            'social_sentiment': sum(social_scores) / len(social_scores) if social_scores else 0,
            'economic_sentiment': sum(economic_scores) / len(economic_scores) if economic_scores else 0,
            'news_count': len(news_data),
            'analyzed_articles': analyzed_articles
        }


class DataIntegrationLayer:
    """Integrates multiple data sources for ML backtesting"""
    
    def __init__(self, news_fetcher, market_data_api=None, indicator_calculator=None):
        """
        Initialize with data source components
        
        Parameters:
            news_fetcher: Existing NewsFetcher class
            market_data_api: API for historical price data
            indicator_calculator: Technical indicator calculation engine
        """
        self.news_fetcher = news_fetcher
        self.market_data_api = market_data_api
        self.indicator_calculator = indicator_calculator
        self.sentiment_analyzer = SentimentAnalyzer()
        logger.info("DataIntegrationLayer initialized")
    
    def get_comprehensive_data(self, ticker=None, timeframe="1y", sectors=None):
        """
        Gather all data needed for strategy development
        
        Args:
            ticker: Stock symbol or None for market-wide analysis
            timeframe: Time period for historical data (e.g., "1y", "6m", "1m")
            sectors: List of sectors to analyze or None for all
            
        Returns:
            dict: Combined dataset with news, price data, and indicators
        """
        logger.info(f"Getting comprehensive data for {ticker if ticker else 'market'}, timeframe: {timeframe}")
        
        # Get news data with sentiment analysis
        news_data = self._get_news_data(ticker, sectors)
        
        # Analyze news sentiment (political, social, economic dimensions)
        sentiment_data = self.sentiment_analyzer.analyze_dimensions(news_data)
        
        # Get market data
        price_data = self._get_market_data(ticker, timeframe)
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(price_data)
        
        # Combine all data
        return {
            "timestamp": datetime.now(),
            "ticker": ticker,
            "timeframe": timeframe,
            "news": news_data,
            "sentiment": sentiment_data,
            "price_data": price_data,
            "indicators": indicators
        }
        
    def _get_news_data(self, ticker=None, sectors=None):
        """
        Get news data from the news fetcher
        
        Args:
            ticker: Stock symbol or None for market news
            sectors: List of sectors to filter by
            
        Returns:
            list: News articles
        """
        try:
            if ticker:
                logger.info(f"Fetching news for {ticker}")
                # Use existing news fetcher with ticker
                news_data = self.news_fetcher.get_news(ticker=ticker, max_items=50)
            else:
                logger.info("Fetching market news")
                # Use existing news fetcher for market news
                news_data = self.news_fetcher.get_news(max_items=50)
                
            logger.info(f"Received {len(news_data) if news_data else 0} news items")
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
            
    def _get_market_data(self, ticker, timeframe):
        """
        Get historical market data
        
        Args:
            ticker: Stock symbol
            timeframe: Time period for data
            
        Returns:
            DataFrame: Historical price data
        """
        if self.market_data_api is None:
            # Fallback to generate mock data if no API provided
            return self._generate_mock_price_data(ticker, timeframe)
            
        try:
            logger.info(f"Fetching market data for {ticker}, timeframe: {timeframe}")
            return self.market_data_api.get_historical_data(ticker, timeframe=timeframe)
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            # Fallback to mock data
            return self._generate_mock_price_data(ticker, timeframe)
            
    def _calculate_indicators(self, price_data):
        """
        Calculate technical indicators
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            dict: Technical indicators
        """
        if self.indicator_calculator is None or price_data is None:
            # Return basic mock indicators if no calculator provided
            return self._generate_mock_indicators()
            
        try:
            logger.info("Calculating technical indicators")
            return self.indicator_calculator.calculate_all(price_data)
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return self._generate_mock_indicators()
            
    def _generate_mock_price_data(self, ticker, timeframe):
        """Generate mock price data for testing"""
        logger.warning(f"Generating mock price data for {ticker}")
        
        # Determine number of days based on timeframe
        days = 252  # Default to 1 year of trading days
        if timeframe == "1m":
            days = 21
        elif timeframe == "3m":
            days = 63
        elif timeframe == "6m":
            days = 126
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate price data with some randomness but with a trend
        n = len(dates)
        base_price = 100
        
        # Add a trend component
        trend = np.linspace(0, 20, n) 
        
        # Add some randomness
        np.random.seed(42)  # For reproducibility
        random_walk = np.random.normal(0, 1, n).cumsum()
        
        # Combine for final price series
        close_prices = base_price + trend + random_walk
        
        # Create OHLCV data
        data = {
            'date': dates,
            'open': close_prices * np.random.uniform(0.99, 1.01, n),
            'high': close_prices * np.random.uniform(1.01, 1.03, n),
            'low': close_prices * np.random.uniform(0.97, 0.99, n),
            'close': close_prices,
            'volume': np.random.randint(100000, 1000000, n)
        }
        
        return pd.DataFrame(data).set_index('date')
        
    def _generate_mock_indicators(self):
        """Generate mock technical indicators"""
        logger.warning("Generating mock technical indicators")
        
        return {
            'rsi': {
                'value': 55.5,
                'signal': 'neutral',
                'description': 'RSI(14) is in neutral territory'
            },
            'macd': {
                'value': 1.2,
                'signal': 'bullish',
                'description': 'MACD is above signal line'
            },
            'moving_averages': {
                'sma_20': 105.5,
                'sma_50': 102.3,
                'sma_200': 95.7,
                'signal': 'bullish',
                'description': 'Price above all major MAs'
            },
            'bollinger_bands': {
                'upper': 110.2,
                'middle': 105.5,
                'lower': 100.8,
                'signal': 'neutral',
                'description': 'Price near middle band'
            },
            'volume': {
                'value': 450000,
                'signal': 'neutral',
                'description': 'Volume near average'
            }
        } 