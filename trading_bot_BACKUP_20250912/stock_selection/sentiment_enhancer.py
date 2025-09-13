#!/usr/bin/env python3
import logging
import pandas as pd
import numpy as np
import re
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

# Local imports
from trading_bot.stock_selection.sentiment_analyzer import SentimentAnalyzer
from trading_bot.news.api_manager import NewsApiManager

logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analyzer that builds upon the base SentimentAnalyzer
    with improved financial term recognition and news integration.
    """
    
    def __init__(self, custom_terms_file: Optional[str] = None):
        """Initialize the enhanced sentiment analyzer
        
        Args:
            custom_terms_file: Optional path to custom financial terms
        """
        # Initialize base sentiment analyzer
        self.base_analyzer = SentimentAnalyzer(custom_terms_file)
        
        # Add additional financial terms
        self._add_enhanced_financial_terms()
        
        # Initialize news API manager
        self.news_api = NewsApiManager()
        
        logger.info("Enhanced sentiment analyzer initialized")
    
    def _add_enhanced_financial_terms(self):
        """Add enhanced financial domain-specific terms to improve sentiment detection"""
        
        enhanced_financial_terms = {
            # Market condition terms
            'bull_market': 3.0,
            'bear_market': -3.0,
            'correction': -2.0,
            'rally': 2.5,
            'crash': -3.5,
            'boom': 3.0,
            'bubble': -1.5,
            'recession': -3.0,
            'depression': -3.5,
            'expansion': 2.5,
            'contraction': -2.0,
            
            # Earnings terms
            'beat_estimates': 2.5,
            'miss_estimates': -2.5,
            'guidance_raised': 3.0,
            'guidance_lowered': -3.0,
            'top_line': 1.0,
            'bottom_line': 1.0,
            'revenue_growth': 2.0,
            'revenue_decline': -2.0,
            
            # Analyst actions
            'upgrade': 2.5,
            'downgrade': -2.5,
            'outperform': 2.0,
            'underperform': -2.0,
            'buy_rating': 2.0,
            'sell_rating': -2.0,
            'hold_rating': 0.0,
            'price_target_raised': 2.0,
            'price_target_lowered': -2.0,
            
            # Corporate actions
            'acquisition': 1.5,
            'merger': 1.0,
            'spinoff': 0.5,
            'restructuring': -1.0,
            'cost_cutting': 1.5,
            'layoffs': -2.0,
            'stock_split': 1.0,
            'stock_buyback': 2.0,
            'dividend_increase': 2.5,
            'dividend_cut': -2.5,
            'ipo': 2.0,
            'delisting': -3.0,
            
            # Regulatory and legal
            'lawsuit': -2.0,
            'settlement': -0.5,
            'fine': -2.0,
            'investigation': -2.0,
            'regulatory_approval': 2.5,
            'regulatory_rejection': -2.5,
            'patent_granted': 2.0,
            'patent_expired': -1.0,
            'recall': -2.5,
            
            # Technology and innovation
            'breakthrough': 3.0,
            'innovation': 2.0,
            'disruption': 1.5,
            'obsolete': -2.0,
            'cutting_edge': 2.0,
            'next_generation': 1.5,
            'revolutionary': 2.0,
            
            # Market sentiment
            'confidence': 2.0,
            'uncertainty': -1.5,
            'volatility': -1.0,
            'stability': 1.5,
            'momentum': 1.0,
            'optimism': 2.0,
            'pessimism': -2.0,
            'fear': -2.0,
            'greed': -1.0,
            'panic': -3.0,
            'exuberance': 1.5
        }
        
        # Add enhanced terms to the VADER lexicon
        for word, score in enhanced_financial_terms.items():
            self.base_analyzer.vader.lexicon[word] = score
            
        logger.info(f"Added {len(enhanced_financial_terms)} enhanced financial terms to sentiment lexicon")
    
    def analyze_ticker_sentiment(self, ticker: str, news_days: int = 7) -> Dict[str, Any]:
        """Analyze sentiment for a specific ticker
        
        Args:
            ticker: Stock ticker symbol
            news_days: Number of days of news to fetch
            
        Returns:
            Dictionary with sentiment analysis results
        """
        logger.info(f"Analyzing sentiment for {ticker}")
        
        # Fetch news articles
        articles = self.fetch_ticker_news(ticker, news_days)
        logger.info(f"Fetched {len(articles)} news articles for {ticker}")
        
        # Analyze social media if available - for now using an empty list
        social_posts = []
        
        # Get sentiment analysis from base analyzer
        sentiment_results = self.base_analyzer.calculate_ticker_sentiment(ticker, articles, social_posts)
        
        # Add enhanced metrics
        enhanced_results = self._enhance_sentiment_results(sentiment_results, ticker)
        
        # Extract key phrases
        key_phrases = self.base_analyzer.extract_key_phrases(articles, ticker)
        enhanced_results['key_phrases'] = key_phrases
        
        # Add timestamp
        enhanced_results['timestamp'] = datetime.now().isoformat()
        enhanced_results['news_days'] = news_days
        
        return enhanced_results
    
    def fetch_ticker_news(self, ticker: str, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch news articles for a ticker
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days of news to fetch
            
        Returns:
            List of news articles
        """
        # First check local cache to minimize API calls
        cache_key = f"{ticker.lower()}_news"
        if hasattr(self, 'news_cache') and cache_key in self.news_cache:
            cache_entry = self.news_cache[cache_key]
            cache_time = cache_entry.get('timestamp', 0)
            # Use cached results if less than 12 hours old
            if time.time() - cache_time < 43200:  # 12 hours in seconds
                logger.info(f"Using cached news for {ticker} (cache age: {(time.time() - cache_time) / 3600:.1f} hours)")
                return cache_entry.get('articles', [])
                
        # Try to get news from the API manager
        try:
            # Be conservative with API calls - only ask for what we need
            max_results = 10  # Limit to 10 articles to conserve API usage
            articles = self.news_api.fetch_news(ticker, max_results=max_results)
            
            # If no articles found, try with company name if available
            if not articles:
                # This would be better with a company name lookup, but for now just try common variations
                expanded_ticker = self._expand_ticker_name(ticker)
                if expanded_ticker != ticker:
                    logger.info(f"No articles found for {ticker}, trying with expanded name: {expanded_ticker}")
                    articles = self.news_api.fetch_news(expanded_ticker, max_results=max_results)
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            articles = []
        
        # Filter articles by date if needed
        if days > 0 and articles:
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_articles = []
            
            for article in articles:
                pub_date = article.get('published_at', '')
                try:
                    if pub_date:
                        date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        if date_obj >= cutoff_date:
                            filtered_articles.append(article)
                except (ValueError, TypeError):
                    # If date parsing fails, include the article anyway
                    filtered_articles.append(article)
            
            articles = filtered_articles
        
        # Store in cache
        if not hasattr(self, 'news_cache'):
            self.news_cache = {}
        self.news_cache[cache_key] = {
            'timestamp': time.time(),
            'articles': articles
        }
        
        return articles
    
    def _enhance_sentiment_results(self, 
                                  sentiment_results: Dict[str, Any], 
                                  ticker: str) -> Dict[str, Any]:
        """Enhance sentiment results with additional metrics
        
        Args:
            sentiment_results: Base sentiment results
            ticker: Stock ticker symbol
            
        Returns:
            Enhanced sentiment results
        """
        # Start with the base results
        enhanced = sentiment_results.copy()
        
        # Add confidence score (higher when more news sources available)
        news_count = sentiment_results.get('news_count', 0)
        if news_count > 15:
            confidence = 0.9
        elif news_count > 10:
            confidence = 0.8
        elif news_count > 5:
            confidence = 0.7
        elif news_count > 2:
            confidence = 0.6
        else:
            confidence = 0.5
        
        enhanced['confidence'] = confidence
        
        # Calculate weighted sentiment impact (adjusted by confidence)
        sentiment_score = sentiment_results.get('sentiment_score', 0.5)
        raw_sentiment = sentiment_results.get('raw_sentiment', 0)
        
        sentiment_impact = raw_sentiment * confidence
        enhanced['sentiment_impact'] = sentiment_impact
        
        # Add source diversity score
        sources = set()
        for article in sentiment_results.get('top_positive', []) + sentiment_results.get('top_negative', []):
            source = article.get('source', '')
            if source:
                sources.add(source)
        
        source_diversity = min(1.0, len(sources) / 5.0)  # Normalize to 0-1
        enhanced['source_diversity'] = source_diversity
        
        # Add recency factor (how recent are the most impactful articles)
        recency_factor = self._calculate_recency_factor(sentiment_results)
        enhanced['recency_factor'] = recency_factor
        
        # Calculate composite score that combines all factors
        composite_score = (
            sentiment_score * 0.5 +
            confidence * 0.2 +
            source_diversity * 0.1 +
            recency_factor * 0.2
        )
        
        enhanced['composite_score'] = composite_score
        
        # Classify sentiment
        if composite_score >= 0.7:
            classification = "very_positive"
        elif composite_score >= 0.6:
            classification = "positive"
        elif composite_score >= 0.45:
            classification = "neutral"
        elif composite_score >= 0.3:
            classification = "negative"
        else:
            classification = "very_negative"
            
        enhanced['classification'] = classification
        
        return enhanced
    
    def _calculate_recency_factor(self, sentiment_results: Dict[str, Any]) -> float:
        """Calculate a recency factor for sentiment
        
        Args:
            sentiment_results: Sentiment analysis results
            
        Returns:
            Recency factor (0-1)
        """
        # Initialize with a neutral value
        recency_factor = 0.5
        
        # Get top positive and negative articles
        top_articles = sentiment_results.get('top_positive', []) + sentiment_results.get('top_negative', [])
        
        if not top_articles:
            return recency_factor
        
        # Calculate weighted age of articles
        now = datetime.now()
        total_weight = 0
        weighted_recency = 0
        
        for i, article in enumerate(top_articles):
            # Articles are sorted by sentiment impact, so earlier ones have more weight
            weight = 1.0 / (i + 1)
            
            # Parse publication date
            pub_date = article.get('published', '')
            try:
                if pub_date:
                    pub_date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    age_days = (now - pub_date_obj).total_seconds() / (24 * 3600)
                    
                    # Newer articles get higher recency score
                    article_recency = 1.0 if age_days < 1 else (1.0 / age_days) if age_days < 7 else 0.1
                    
                    weighted_recency += article_recency * weight
                    total_weight += weight
            except (ValueError, TypeError):
                # Skip articles with invalid dates
                continue
        
        # Calculate final recency factor
        if total_weight > 0:
            recency_factor = weighted_recency / total_weight
            
        return recency_factor
    
    def _expand_ticker_name(self, ticker: str) -> str:
        """Expand ticker to company name for better news search
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Expanded name or original ticker
        """
        # Simple mapping of common tickers to company names
        ticker_map = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'GOOG': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta Facebook',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA',
            'JPM': 'JPMorgan',
            'BAC': 'Bank of America',
            'WMT': 'Walmart',
            'DIS': 'Disney',
            'NFLX': 'Netflix',
            'INTC': 'Intel',
            'AMD': 'Advanced Micro Devices',
            'IBM': 'IBM',
            'GS': 'Goldman Sachs',
            'BA': 'Boeing',
        }
        
        return ticker_map.get(ticker.upper(), ticker)
    
    def analyze_multiple_tickers(self, 
                               tickers: List[str], 
                               news_days: int = 7) -> Dict[str, Dict[str, Any]]:
        """Analyze sentiment for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            news_days: Number of days of news to analyze
            
        Returns:
            Dictionary mapping tickers to their sentiment analysis
        """
        results = {}
        
        for ticker in tickers:
            try:
                results[ticker] = self.analyze_ticker_sentiment(ticker, news_days)
                logger.info(f"Completed sentiment analysis for {ticker}")
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {ticker}: {str(e)}")
                results[ticker] = {'error': str(e)}
        
        return results
    
    def get_market_sentiment(self, index_tickers: List[str] = None) -> Dict[str, Any]:
        """Get overall market sentiment
        
        Args:
            index_tickers: Optional list of index tickers to use
            
        Returns:
            Dictionary with market sentiment metrics
        """
        if index_tickers is None:
            index_tickers = ['SPY', 'QQQ', 'DIA', 'IWM']
        
        # Fetch news for 'market' keyword
        try:
            market_news = self.news_api.fetch_news('market', max_results=20)
        except Exception as e:
            logger.error(f"Error fetching market news: {str(e)}")
            market_news = self.news_api.generate_mock_news('market', count=10)
        
        # Analyze sentiment on market news
        market_sentiment = self.base_analyzer.analyze_news_articles(market_news)
        
        # Also analyze index-specific news
        index_sentiments = {}
        for ticker in index_tickers:
            ticker_news = self.fetch_ticker_news(ticker, days=3)
            index_sentiments[ticker] = self.base_analyzer.analyze_news_articles(ticker_news, ticker)
        
        # Combine all sentiments
        overall_sentiment = market_sentiment.get('overall_sentiment', 0)
        sentiment_counts = {
            'positive': market_sentiment.get('positive_count', 0),
            'negative': market_sentiment.get('negative_count', 0),
            'neutral': market_sentiment.get('neutral_count', 0)
        }
        
        # Add index sentiments 
        for ticker, sentiment in index_sentiments.items():
            overall_sentiment += sentiment.get('overall_sentiment', 0)
            sentiment_counts['positive'] += sentiment.get('positive_count', 0)
            sentiment_counts['negative'] += sentiment.get('negative_count', 0)
            sentiment_counts['neutral'] += sentiment.get('neutral_count', 0)
        
        # Average the sentiment values
        if len(index_tickers) > 0:
            overall_sentiment = overall_sentiment / (len(index_tickers) + 1)
        
        # Calculate sentiment ratio
        total_articles = sum(sentiment_counts.values())
        sentiment_ratio = 0.5
        if total_articles > 0:
            sentiment_ratio = sentiment_counts['positive'] / total_articles
        
        # Classify market sentiment
        if overall_sentiment > 0.3:
            market_mood = "bullish"
        elif overall_sentiment > 0.1:
            market_mood = "mildly_bullish"
        elif overall_sentiment > -0.1:
            market_mood = "neutral"
        elif overall_sentiment > -0.3:
            market_mood = "mildly_bearish"
        else:
            market_mood = "bearish"
        
        # Final market sentiment results
        results = {
            'overall_sentiment': overall_sentiment,
            'sentiment_ratio': sentiment_ratio,
            'market_mood': market_mood,
            'article_counts': sentiment_counts,
            'total_articles': total_articles,
            'index_sentiments': {ticker: data.get('overall_sentiment', 0) for ticker, data in index_sentiments.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        return results


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the enhanced sentiment analyzer
    analyzer = EnhancedSentimentAnalyzer()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT', 'TSLA']
    
    print("Testing Enhanced Sentiment Analyzer")
    print("===================================\n")
    
    # Get market sentiment
    print("Market Sentiment:")
    market_sentiment = analyzer.get_market_sentiment()
    print(f"  Mood: {market_sentiment['market_mood']}")
    print(f"  Overall Score: {market_sentiment['overall_sentiment']:.4f}")
    print(f"  Positive/Total Ratio: {market_sentiment['sentiment_ratio']:.2f}")
    print(f"  Total Articles: {market_sentiment['total_articles']}")
    print("\n")
    
    # Analyze each ticker
    for ticker in test_tickers:
        print(f"Analyzing {ticker}:")
        sentiment = analyzer.analyze_ticker_sentiment(ticker)
        
        print(f"  Classification: {sentiment['classification']}")
        print(f"  Composite Score: {sentiment['composite_score']:.4f}")
        print(f"  Confidence: {sentiment['confidence']:.2f}")
        print(f"  News Count: {sentiment['news_count']}")
        print(f"  Key Phrases: {', '.join(sentiment['key_phrases'][:3])}" if sentiment['key_phrases'] else "  No key phrases found")
        print("\n") 