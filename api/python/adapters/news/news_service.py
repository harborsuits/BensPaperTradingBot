"""
News service that integrates all available news adaptors into a unified interface.
Provides methods for fetching economic news digest, market intelligence, 
symbol-specific news, and sentiment analysis.
"""
import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# Import all available news adapters
try:
    from trading_bot.adapters.news.gnews_adapter import GNewsAdapter
    GNEWS_AVAILABLE = True
except ImportError:
    GNEWS_AVAILABLE = False

try:
    from trading_bot.adapters.news.newsdata_adapter import NewsDataAdapter
    NEWSDATA_AVAILABLE = True
except ImportError:
    NEWSDATA_AVAILABLE = False

try:
    from trading_bot.adapters.news.currents_adapter import CurrentsAdapter
    CURRENTS_AVAILABLE = True
except ImportError:
    CURRENTS_AVAILABLE = False

try:
    from trading_bot.adapters.news.mediastack_adapter import MediaStackAdapter
    MEDIASTACK_AVAILABLE = True
except ImportError:
    MEDIASTACK_AVAILABLE = False

try:
    from trading_bot.adapters.news.finnhub_adapter import FinnhubNewsAdapter
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

try:
    from trading_bot.adapters.news.nytimes_adapter import NYTimesAdapter
    NYTIMES_AVAILABLE = True
except ImportError:
    NYTIMES_AVAILABLE = False

# Additional adapter imports can be added here


class NewsService:
    """
    Unified news service that integrates all available news sources.
    Handles API key management, caching, and fallback between different providers.
    """
    
    def __init__(self, api_keys: Dict[str, str], cache_expiry_minutes: int = 30):
        """
        Initialize the news service with API keys.
        
        Args:
            api_keys: Dictionary of API keys for different services
            cache_expiry_minutes: Cache expiry time in minutes
        """
        self.api_keys = api_keys
        self.adapters = {}
        self.cache = {}
        self.cache_expiry_minutes = cache_expiry_minutes
        self.logger = logging.getLogger('news_service')
        
        # Initialize all available adapters
        self._initialize_adapters()
        
    def _initialize_adapters(self):
        """Initialize all available news adapters with their respective API keys."""
        # GNews Adapter
        if GNEWS_AVAILABLE and 'gnews' in self.api_keys:
            try:
                self.adapters['gnews'] = GNewsAdapter(self.api_keys['gnews'])
                self.logger.info("GNews adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize GNews adapter: {e}")
        
        # NewsData.io Adapter
        if NEWSDATA_AVAILABLE and 'newsdata' in self.api_keys:
            try:
                self.adapters['newsdata'] = NewsDataAdapter(self.api_keys['newsdata'])
                self.logger.info("NewsData adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize NewsData adapter: {e}")
        
        # Currents Adapter
        if CURRENTS_AVAILABLE and 'currents' in self.api_keys:
            try:
                self.adapters['currents'] = CurrentsAdapter(self.api_keys['currents'])
                self.logger.info("Currents adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Currents adapter: {e}")
        
        # MediaStack Adapter
        if MEDIASTACK_AVAILABLE and 'mediastack' in self.api_keys:
            try:
                self.adapters['mediastack'] = MediaStackAdapter(self.api_keys['mediastack'])
                self.logger.info("MediaStack adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize MediaStack adapter: {e}")
        
        # Finnhub Adapter
        if FINNHUB_AVAILABLE and 'finnhub' in self.api_keys:
            try:
                self.adapters['finnhub'] = FinnhubNewsAdapter(self.api_keys['finnhub'])
                self.logger.info("Finnhub adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Finnhub adapter: {e}")
        
        # NY Times Adapter
        if NYTIMES_AVAILABLE and 'nytimes' in self.api_keys:
            try:
                self.adapters['nytimes'] = NYTimesAdapter(self.api_keys['nytimes'])
                self.logger.info("NY Times adapter initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize NY Times adapter: {e}")
        
        # Check if any adapters were initialized
        if not self.adapters:
            self.logger.warning("No news adapters could be initialized. Check API keys.")
            
    def get_available_sources(self) -> List[str]:
        """Return a list of available news sources."""
        return list(self.adapters.keys())
    
    def _format_time(self, time_str: str) -> str:
        """
        Format a time string to a consistent format.
        
        Args:
            time_str: Input time string in various formats
            
        Returns:
            Formatted time string
        """
        if not time_str:
            return "Unknown"
        
        try:
            # Try to parse the time string into a datetime object
            formats = [
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]
            
            dt = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(time_str, fmt)
                    break
                except ValueError:
                    continue
            
            if dt:
                # Format the datetime object
                now = datetime.now()
                diff = now - dt
                
                if diff.days == 0:
                    if diff.seconds < 3600:
                        minutes = diff.seconds // 60
                        return f"{minutes} min{'s' if minutes != 1 else ''} ago"
                    else:
                        hours = diff.seconds // 3600
                        return f"{hours} hour{'s' if hours != 1 else ''} ago"
                elif diff.days < 7:
                    return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
                else:
                    return dt.strftime("%Y-%m-%d")
            else:
                return time_str
                
        except Exception:
            return time_str
    
    def _map_sentiment_score(self, score: float) -> str:
        """Map a sentiment score to a text label."""
        if score > 0.3:
            return "Positive"
        elif score < -0.3:
            return "Negative"
        else:
            return "Neutral"
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if a cache entry is still valid."""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key].get('timestamp')
        if not cache_time:
            return False
        
        expiry_time = cache_time + timedelta(minutes=self.cache_expiry_minutes)
        return datetime.now() < expiry_time
    
    def get_economic_digest(self) -> Dict[str, Any]:
        """
        Get an economic news digest with high impact, medium impact news,
        and market sector shifts.
        
        Returns:
            Dictionary with news digest information
        """
        # Check cache first
        cache_key = 'economic_digest'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        # Initialize the digest structure
        digest = {
            "summary": "Market is showing mixed signals with cautious optimism.",
            "high_impact": [],
            "medium_impact": [],
            "market_shifts": []
        }
        
        # Try to get news from each adapter
        for source_name, adapter in self.adapters.items():
            try:
                # Each adapter should have a get_economic_news method
                if hasattr(adapter, 'get_economic_news'):
                    news = adapter.get_economic_news()
                    
                    # Add source information to each news item
                    for item in news:
                        item['source'] = source_name
                    
                    # Categorize news by impact
                    for item in news:
                        impact = item.get('impact', 'medium').lower()
                        if impact == 'high':
                            digest['high_impact'].append(item)
                        else:
                            digest['medium_impact'].append(item)
            except Exception as e:
                self.logger.error(f"Error fetching economic news from {source_name}: {e}")
        
        # Add market sector information
        digest['market_shifts'] = self._get_sector_performance()
        
        # Update summary based on news sentiment
        digest['summary'] = self._generate_market_summary(digest)
        
        # Cache the results
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': digest
        }
        
        return digest
    
    def _generate_market_summary(self, digest: Dict[str, Any]) -> str:
        """Generate a market summary based on news sentiment."""
        # Count positive and negative high-impact news
        positive_count = 0
        negative_count = 0
        
        for news in digest['high_impact']:
            sentiment = news.get('sentiment', '').lower()
            if sentiment == 'positive':
                positive_count += 1
            elif sentiment == 'negative':
                negative_count += 1
        
        # Determine market mood
        if positive_count > negative_count * 2:
            mood = "bullish"
        elif negative_count > positive_count * 2:
            mood = "bearish"
        elif positive_count > negative_count:
            mood = "cautiously optimistic"
        elif negative_count > positive_count:
            mood = "cautiously pessimistic"
        else:
            mood = "mixed"
        
        # Generate market environment description
        environment_options = [
            f"Market environment is {mood} with moderate volatility.",
            f"Market showing {mood} signals with potential for short-term movements.",
            f"Overall sentiment appears {mood} based on recent economic indicators."
        ]
        
        # Generate policy outlook
        policy_options = [
            "Policy outlook remains accommodative.",
            "Central banks maintaining current policy stance.",
            "Policy changes expected in response to economic data.",
            "Monetary policy tightening anticipated."
        ]
        
        return f"{random.choice(environment_options)} {random.choice(policy_options)}"
    
    def _get_sector_performance(self) -> List[Dict[str, str]]:
        """Get sector performance data."""
        sectors = [
            "Technology", "Healthcare", "Financials", "Consumer Discretionary",
            "Energy", "Utilities", "Materials", "Industrials", "Real Estate"
        ]
        
        results = []
        for sector in sectors:
            # In a real implementation, this would fetch actual sector performance
            # For now, generate random values for demonstration
            change = round(random.uniform(-2.0, 2.0), 1)
            direction = "+" if change >= 0 else ""
            
            if change > 1.0:
                driver = "Strong earnings"
            elif change > 0.5:
                driver = "Positive sentiment"
            elif change > -0.5:
                driver = "Mixed earnings"
            elif change > -1.0:
                driver = "Sector rotation"
            else:
                driver = "Weak outlook"
            
            results.append({
                "sector": sector,
                "change": f"{direction}{change}%",
                "driver": driver
            })
        
        # Sort by absolute change (descending)
        results.sort(key=lambda x: abs(float(x["change"].replace("%", "").replace("+", ""))), reverse=True)
        
        return results[:5]  # Return top 5 movers
    
    def get_news_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get news for a specific symbol.
        
        Args:
            symbol: Stock symbol to get news for
            
        Returns:
            List of news items for the symbol
        """
        # Check cache first
        cache_key = f'symbol_news_{symbol}'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        all_news = []
        
        # Try to get news from each adapter
        for source_name, adapter in self.adapters.items():
            try:
                # Each adapter should have a get_symbol_news method
                if hasattr(adapter, 'get_symbol_news'):
                    news = adapter.get_symbol_news(symbol, max_items=5)
                    
                    # Add source information to each news item
                    for item in news:
                        item['source'] = source_name
                    
                    all_news.extend(news)
            except Exception as e:
                self.logger.error(f"Error fetching news for {symbol} from {source_name}: {e}")
        
        # Sort news by published date (newest first)
        all_news.sort(key=lambda x: x.get('time_published', ''), reverse=True)
        
        # Cache the results
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': all_news[:15]  # Limit to 15 items
        }
        
        return all_news[:15]
    
    def get_additional_news_from_source(self, source: str, max_items: int = 5) -> List[Dict[str, Any]]:
        """
        Get additional news from a specific source.
        
        Args:
            source: Source name
            max_items: Maximum number of items to return
            
        Returns:
            List of news items
        """
        # Check cache first
        cache_key = f'additional_news_{source}'
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        news = []
        
        # Try to get news from the specified adapter
        if source in self.adapters:
            try:
                adapter = self.adapters[source]
                
                # Each adapter should have a get_latest_news method
                if hasattr(adapter, 'get_latest_news'):
                    news = adapter.get_latest_news(max_items=max_items)
                    
                    # Add source information to each news item
                    for item in news:
                        item['source'] = source
            except Exception as e:
                self.logger.error(f"Error fetching additional news from {source}: {e}")
        
        # Cache the results
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': news
        }
        
        return news
    
    def get_sentiment_analysis(self, symbol: str = None) -> Dict[str, Any]:
        """
        Get sentiment analysis for the market or a specific symbol.
        
        Args:
            symbol: Optional stock symbol for specific sentiment
            
        Returns:
            Sentiment analysis data
        """
        # For now, return mock sentiment data
        # In a real implementation, this would aggregate sentiment from news articles
        
        if symbol:
            cache_key = f'sentiment_{symbol}'
        else:
            cache_key = 'market_sentiment'
            
        # Check cache first
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        # Generate sentiment data
        bullish_pct = random.randint(30, 80)
        bearish_pct = 100 - bullish_pct
        
        # Source-specific sentiment
        sources = self.get_available_sources()
        source_sentiment = {}
        
        for source in sources:
            # Generate random sentiment between -0.8 and 0.8
            sentiment = round(random.uniform(-0.8, 0.8), 2)
            source_sentiment[source] = sentiment
        
        sentiment_data = {
            "bullish_articles_pct": bullish_pct,
            "bearish_articles_pct": bearish_pct,
            "overall_score": round(2 * (bullish_pct / 100) - 1, 2),  # Scale to -1 to 1
            "source_sentiment": source_sentiment
        }
        
        # Cache the results
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': sentiment_data
        }
        
        return sentiment_data
