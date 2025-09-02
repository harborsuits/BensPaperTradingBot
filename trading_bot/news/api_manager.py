import os
import json
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsApiManager:
    """
    Manages multiple news API providers with intelligent cycling to prevent rate limiting
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the news API manager
        
        Args:
            config_path: Path to the API configuration file (optional)
        """
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'config', 
            'news_api_config.json'
        )
        self.api_data = self._load_or_create_config()
        self._update_usage_stats()
        logger.info(f"NewsApiManager initialized with {len(self.api_data['apis'])} APIs")
        
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load or create API configuration"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    logger.info(f"Loaded API configuration from {self.config_path}")
                    return config
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading config: {e}, creating new configuration")
        
        # Create default configuration
        config_dir = os.path.dirname(self.config_path)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        default_config = self._create_default_config()
        self._save_config(default_config)
        logger.info(f"Created new API configuration at {self.config_path}")
        return default_config
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default API configuration"""
        return {
            'apis': {
                'nytimes': {
                    'api_key': '',
                    'base_url': 'https://api.nytimes.com/svc/search/v2/articlesearch.json',
                    'daily_limit': 500,
                    'current_usage': 0,
                    'cooldown_minutes': 10,
                    'priority': 4,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                },
                'finnhub': {
                    'api_key': '',
                    'base_url': 'https://finnhub.io/api/v1/company-news',
                    'daily_limit': 60,
                    'current_usage': 0,
                    'cooldown_minutes': 5,
                    'priority': 3,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                },
                'newsapi': {
                    'api_key': '',
                    'base_url': 'https://newsapi.org/v2/everything',
                    'daily_limit': 100,
                    'current_usage': 0,
                    'cooldown_minutes': 15,
                    'priority': 5,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                },
                'marketaux': {
                    'api_key': '',
                    'base_url': 'https://api.marketaux.com/v1/news/all',
                    'daily_limit': 100,
                    'current_usage': 0,
                    'cooldown_minutes': 10,
                    'priority': 6,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                },
                'newsdata': {
                    'api_key': '',
                    'base_url': 'https://newsdata.io/api/1/news',
                    'daily_limit': 200,
                    'current_usage': 0,
                    'cooldown_minutes': 10,
                    'priority': 7,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                },
                'gnews': {
                    'api_key': '',
                    'base_url': 'https://gnews.io/api/v4/search',
                    'daily_limit': 100,
                    'current_usage': 0,
                    'cooldown_minutes': 10,
                    'priority': 8,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                },
                'alpha_vantage': {
                    'api_key': '',
                    'base_url': 'https://www.alphavantage.co/query',
                    'daily_limit': 25,
                    'current_usage': 0,
                    'cooldown_minutes': 10,
                    'priority': 9,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                },
                'mediastack': {
                    'api_key': '',
                    'base_url': 'http://api.mediastack.com/v1/news',
                    'daily_limit': 1000,
                    'current_usage': 0,
                    'cooldown_minutes': 5,
                    'priority': 1,
                    'enabled': True,
                    'last_used': 0,
                    'error_count': 0
                },
                'currents': {
                    'api_key': '',
                    'base_url': 'https://api.currentsapi.services/v1/search',
                    'daily_limit': 600,
                    'current_usage': 0,
                    'cooldown_minutes': 10,
                    'priority': 2,
                    'enabled': False,
                    'last_used': 0,
                    'error_count': 0
                }
            },
            'settings': {
                'timeout_seconds': 10,
                'auto_disable_on_error': True,
                'error_threshold': 3,
                'reset_usage_daily': True,
                'auto_fallback': True,
                'max_retries': 2
            },
            'cache': {
                'enabled': True,
                'max_age_minutes': 60,
                'entries': {}
            }
        }
    
    def _save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save API configuration to file"""
        if config is None:
            config = self.api_data
            
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _update_usage_stats(self) -> None:
        """Reset usage counters if it's a new day"""
        current_date = datetime.now().strftime('%Y-%m-%d')
        if self.api_data.get('date') != current_date:
            logger.info("New day detected, resetting API usage counters")
            for api_name in self.api_data['apis']:
                self.api_data['apis'][api_name]['current_usage'] = 0
                self.api_data['apis'][api_name]['error_count'] = 0
                self.api_data['apis'][api_name]['enabled'] = True
            
            # Clear old cache entries
            self.api_data['cache']['entries'] = {}
            self.api_data['date'] = current_date
            self._save_config()
    
    def select_api(self, query: str) -> Optional[str]:
        """
        Select the best API to use based on current usage and cooldown periods
        
        Args:
            query: Search query (used to check cache)
            
        Returns:
            Selected API name or None if no suitable API is available
        """
        # Check if query is in cache
        cache_results = self._get_from_cache(query)
        if cache_results:
            logger.info(f"Found cached results for query '{query}'")
            return None
            
        # Sort APIs by priority
        sorted_apis = sorted(
            [(name, info) for name, info in self.api_data['apis'].items() if info.get('api_key')],
            key=lambda x: x[1].get('priority', 999)
        )
        
        current_time = time.time()
        selected_api = None
        
        # First pass: find an API that's not in cooldown and hasn't reached daily limit
        for api_name, api_info in sorted_apis:
            # Skip if API is disabled or missing API key
            if not api_info.get('enabled', True) or not api_info.get('api_key'):
                continue
                
            # Skip if in cooldown period
            last_used = api_info.get('last_used', 0)
            cooldown = api_info.get('cooldown_minutes', 0) * 60
            if current_time - last_used < cooldown:
                continue
                
            # Skip if reached daily limit
            current_usage = api_info.get('current_usage', 0)
            daily_limit = api_info.get('daily_limit', float('inf'))
            if current_usage >= daily_limit:
                continue
                
            # Found a suitable API
            selected_api = api_name
            break
            
        # If no available API, check if auto fallback is enabled
        if not selected_api and self.api_data['settings'].get('auto_fallback'):
            # Second pass: find any API with valid key, ignoring limits
            for api_name, api_info in sorted_apis:
                if api_info.get('api_key'):
                    selected_api = api_name
                    logger.warning(f"Using {api_name} as fallback despite limits")
                    break
        
        return selected_api
    
    def fetch_news(self, query: str, max_results: int = 10, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch news articles for a query
        
        Args:
            query: Search query (e.g., ticker symbol, company name, topic)
            max_results: Maximum number of results to return
            force_refresh: If True, bypass cache and fetch new results
            
        Returns:
            List of news article dictionaries with standardized fields
        """
        if not query:
            logger.warning("Empty query provided to fetch_news")
            return []
            
        # Check cache first (unless force_refresh is True)
        if not force_refresh:
            cache_results = self._get_from_cache(query)
            if cache_results:
                logger.info(f"Returning {len(cache_results)} cached results for '{query}'")
                return cache_results[:max_results]
        
        # Select API to use
        api_name = self.select_api(query)
        if not api_name:
            logger.warning("No suitable API available for fetching news")
            return []
            
        logger.info(f"Using {api_name} API to fetch news for '{query}'")
        api_config = self.api_data['apis'][api_name]
        
        # Get the appropriate fetch method
        fetch_method = getattr(self, f"_fetch_from_{api_name.lower()}", None)
        if not fetch_method:
            logger.error(f"No fetch method available for {api_name}")
            return []
            
        # Try to fetch news
        results = []
        retry_count = 0
        max_retries = self.api_data['settings'].get('max_retries', 2)
        
        while retry_count <= max_retries:
            try:
                results = fetch_method(query, api_config, max_results)
                
                # Log the API call
                self._log_api_call(api_name)
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                error_message = str(e)
                status_code = 0
                
                # Extract status code if present in error message
                if 'API error:' in error_message:
                    try:
                        status_code = int(error_message.split('API error:')[1].strip())
                    except (ValueError, IndexError):
                        pass
                
                # Handle the error
                logger.error(f"Error fetching from {api_name}: {error_message}")
                self._handle_api_error(api_name, status_code)
                
                # Check if we should retry
                retry_count += 1
                if retry_count <= max_retries:
                    logger.info(f"Retrying with {api_name} ({retry_count}/{max_retries})")
                    time.sleep(2)  # Brief delay before retry
                else:
                    # Try with next available API if auto fallback is enabled
                    if self.api_data['settings'].get('auto_fallback'):
                        # Temporarily disable the failed API
                        api_config['enabled'] = False
                        
                        # Try to select another API
                        next_api = self.select_api(query)
                        if next_api:
                            logger.info(f"Falling back to {next_api} API after {api_name} failed")
                            api_name = next_api
                            api_config = self.api_data['apis'][api_name]
                            fetch_method = getattr(self, f"_fetch_from_{api_name.lower()}", None)
                            
                            # Reset retry count for new API
                            retry_count = 0
                        else:
                            logger.error("No fallback APIs available")
                            break
                    else:
                        logger.error("Max retries reached and auto fallback disabled")
                        break
        
        # If successful, cache results
        if results and self.api_data['cache']['enabled']:
            self._add_to_cache(query, results)
            
        return results[:max_results]
    
    def _get_from_cache(self, query: str, ignore_expiry: bool = False) -> List[Dict[str, Any]]:
        """
        Get results from cache if they exist and are not expired
        
        Args:
            query: Search query
            ignore_expiry: If True, return results even if they're expired
            
        Returns:
            List of cached news articles
        """
        cache = self.api_data['cache']
        query_key = query.lower().strip()
        
        if query_key not in cache['entries']:
            return []
        
        entry = cache['entries'][query_key]
        timestamp = entry.get('timestamp', 0)
        max_age = cache['max_age_minutes'] * 60
        
        # Check if entry is expired (unless ignore_expiry is True)
        if not ignore_expiry and time.time() - timestamp > max_age:
            logger.debug(f"Cache entry for '{query}' expired")
            return []
        
        return entry.get('results', [])
    
    def _add_to_cache(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Add results to cache"""
        query_key = query.lower().strip()
        
        self.api_data['cache']['entries'][query_key] = {
            'timestamp': time.time(),
            'results': results
        }
        
        # Save the updated cache
        self._save_config()
    
    def _log_api_call(self, api_name: str) -> None:
        """Log an API call and update usage statistics"""
        if api_name not in self.api_data['apis']:
            return
        
        api_info = self.api_data['apis'][api_name]
        api_info['current_usage'] += 1
        api_info['last_used'] = time.time()
        
        logger.info(f"API call to {api_name}. Total today: {api_info['current_usage']}")
        self._save_config()
    
    def _handle_api_error(self, api_name: str, status_code: int = 0) -> None:
        """
        Handle API error by updating error counters and disabling if needed
        
        Args:
            api_name: Name of the API that encountered an error
            status_code: HTTP status code (if available)
        """
        if api_name not in self.api_data['apis']:
            return
        
        api_info = self.api_data['apis'][api_name]
        error_count = api_info.get('error_count', 0) + 1
        api_info['error_count'] = error_count
        
        # Log specific error types
        if status_code:
            if status_code == 401:
                logger.error(f"{api_name} authentication failed (401): Invalid API key or unauthorized access")
                # Flag as an auth error specifically
                api_info['auth_error'] = True
            elif status_code == 403:
                logger.error(f"{api_name} access forbidden (403): API key may be restricted or IP blocked")
                api_info['auth_error'] = True
            elif status_code == 429:
                logger.warning(f"{api_name} rate limited (429): Exceeding request limits")
                # Increase cooldown for rate-limited APIs
                api_info['cooldown_minutes'] = api_info.get('cooldown_minutes', 10) * 2
                logger.info(f"Increased cooldown for {api_name} to {api_info['cooldown_minutes']} minutes")
            elif status_code == 402:
                logger.warning(f"{api_name} payment required (402): Free tier limit reached")
                api_info['payment_required'] = True
        
        # Automatically disable API if it exceeds error threshold
        if (self.api_data['settings']['auto_disable_on_error'] and 
            error_count >= self.api_data['settings']['error_threshold']):
            logger.warning(f"Disabling {api_name} API due to repeated errors")
            api_info['enabled'] = False
        
        self._save_config()
    
    def _fetch_from_nytimes(self, query: str, api_config: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Fetch news from New York Times API"""
        params = {
            'q': query,
            'api-key': api_config['api_key'],
            'sort': 'newest',
            'page': 0
        }
        
        response = requests.get(
            api_config['base_url'],
            params=params,
            timeout=self.api_data['settings']['timeout_seconds']
        )
        
        if response.status_code != 200:
            logger.warning(f"NY Times API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        articles = data.get('response', {}).get('docs', [])
        
        # Transform to standard format
        results = []
        for article in articles:
            # Extract publication date
            pub_date = article.get('pub_date', '')
            try:
                if pub_date:
                    pub_date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    formatted_date = pub_date_obj.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    formatted_date = ''
            except ValueError:
                formatted_date = pub_date
                
            # Calculate sentiment
            headline = article.get('headline', {}).get('main', '')
            abstract = article.get('abstract', '')
            sentiment = self._calculate_sentiment(f"{headline} {abstract}")
            
            results.append({
                'title': headline,
                'description': abstract,
                'url': article.get('web_url', ''),
                'source': 'The New York Times',
                'published_at': formatted_date,
                'image_url': self._get_nytimes_image(article),
                'sentiment': sentiment,
                'relevance': self._calculate_relevance(query, headline, abstract),
                'categories': [section.get('name', '') for section in article.get('section_name', [])]
            })
        
        return results
    
    def _fetch_from_finnhub(self, ticker: str, api_config: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Fetch news from Finnhub API"""
        # Format dates for Finnhub (requires YYYY-MM-DD format)
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        
        params = {
            'symbol': ticker,
            'from': week_ago.strftime('%Y-%m-%d'),
            'to': today.strftime('%Y-%m-%d'),
            'token': api_config['api_key']
        }
        
        response = requests.get(
            api_config['base_url'],
            params=params,
            timeout=self.api_data['settings']['timeout_seconds']
        )
        
        if response.status_code != 200:
            logger.warning(f"Finnhub API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        articles = response.json()
        
        # Transform to standard format
        results = []
        for article in articles:
            # Format timestamp
            timestamp = article.get('datetime', 0)
            if timestamp:
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            else:
                date_str = ''
                
            title = article.get('headline', '')
            summary = article.get('summary', '')
            sentiment = self._calculate_sentiment(f"{title} {summary}")
            
            results.append({
                'title': title,
                'description': summary,
                'url': article.get('url', ''),
                'source': article.get('source', 'Finnhub'),
                'published_at': date_str,
                'image_url': article.get('image', ''),
                'sentiment': sentiment,
                'relevance': self._calculate_relevance(ticker, title, summary),
                'categories': [article.get('category', '')]
            })
        
        return results
    
    def _fetch_from_newsapi(self, query: str, api_config: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Fetch news from News API"""
        params = {
            'q': query,
            'apiKey': api_config['api_key'],
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': max_results
        }
        
        response = requests.get(
            api_config['base_url'],
            params=params,
            timeout=self.api_data['settings']['timeout_seconds']
        )
        
        if response.status_code != 200:
            logger.warning(f"News API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        articles = data.get('articles', [])
        
        # Transform to standard format
        results = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            sentiment = self._calculate_sentiment(f"{title} {description}")
            
            results.append({
                'title': title,
                'description': description,
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'NewsAPI'),
                'published_at': article.get('publishedAt', ''),
                'image_url': article.get('urlToImage', ''),
                'sentiment': sentiment,
                'relevance': self._calculate_relevance(query, title, description),
                'categories': []  # News API doesn't provide categories
            })
        
        return results
    
    def _fetch_from_marketaux(self, query: str, api_config: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Fetch news from Marketaux API"""
        params = {
            'api_token': api_config['api_key'],
            'symbols': query,
            'limit': max_results,
            'language': 'en'
        }
        
        response = requests.get(
            api_config['base_url'],
            params=params,
            timeout=self.api_data['settings']['timeout_seconds']
        )
        
        if response.status_code != 200:
            logger.warning(f"Marketaux API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        articles = data.get('data', [])
        
        # Transform to standard format
        results = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            sentiment = self._calculate_sentiment(f"{title} {description}")
            
            results.append({
                'title': title,
                'description': description,
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'published_at': article.get('published_at', ''),
                'image_url': article.get('image_url', ''),
                'sentiment': sentiment,
                'relevance': self._calculate_relevance(query, title, description),
                'categories': [entity.get('name', '') for entity in article.get('entities', [])]
            })
        
        return results
    
    def _fetch_from_newsdata(self, query: str, api_config: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Fetch news from NewsData API"""
        params = {
            'apikey': api_config['api_key'],
            'q': query,
            'language': 'en',
            'size': max_results
        }
        
        response = requests.get(
            api_config['base_url'],
            params=params,
            timeout=self.api_data['settings']['timeout_seconds']
        )
        
        if response.status_code != 200:
            logger.warning(f"NewsData API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        articles = data.get('results', [])
        
        # Transform to standard format
        results = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            sentiment = self._calculate_sentiment(f"{title} {description}")
            
            results.append({
                'title': title,
                'description': description,
                'url': article.get('link', ''),
                'source': article.get('source_id', ''),
                'published_at': article.get('pubDate', ''),
                'image_url': article.get('image_url', ''),
                'sentiment': sentiment,
                'relevance': self._calculate_relevance(query, title, description),
                'categories': article.get('category', [])
            })
        
        return results
    
    def _fetch_from_gnews(self, query: str, api_config: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Fetch news from GNews API"""
        params = {
            'token': api_config['api_key'],
            'q': query,
            'lang': 'en',
            'max': max_results
        }
        
        response = requests.get(
            api_config['base_url'],
            params=params,
            timeout=self.api_data['settings']['timeout_seconds']
        )
        
        if response.status_code != 200:
            logger.warning(f"GNews API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        articles = data.get('articles', [])
        
        # Transform to standard format
        results = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            sentiment = self._calculate_sentiment(f"{title} {description}")
            
            results.append({
                'title': title,
                'description': description,
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', ''),
                'published_at': article.get('publishedAt', ''),
                'image_url': article.get('image', ''),
                'sentiment': sentiment,
                'relevance': self._calculate_relevance(query, title, description),
                'categories': []  # GNews doesn't provide categories
            })
        
        return results
    
    def _fetch_from_alpha_vantage(self, symbol: str, api_config: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """Fetch news from Alpha Vantage API"""
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'apikey': api_config['api_key'],
            'limit': max_results
        }
        
        response = requests.get(
            api_config['base_url'],
            params=params,
            timeout=self.api_data['settings']['timeout_seconds']
        )
        
        if response.status_code != 200:
            logger.warning(f"Alpha Vantage API error: {response.status_code}")
            raise Exception(f"API error: {response.status_code}")
        
        data = response.json()
        feed = data.get('feed', [])
        
        # Transform to standard format
        results = []
        for article in feed:
            title = article.get('title', '')
            summary = article.get('summary', '')
            
            # Alpha Vantage provides sentiment scores
            av_sentiment = article.get('overall_sentiment_score', 0)
            if av_sentiment:
                # Convert Alpha Vantage sentiment (-1 to 1) to our format (-1 to 1)
                sentiment = av_sentiment
            else:
                sentiment = self._calculate_sentiment(f"{title} {summary}")
            
            results.append({
                'title': title,
                'description': summary,
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'published_at': article.get('time_published', ''),
                'image_url': '',  # Alpha Vantage doesn't provide images
                'sentiment': sentiment,
                'relevance': self._calculate_relevance(symbol, title, summary),
                'categories': article.get('topics', [])
            })
        
        return results
    
    def _get_nytimes_image(self, article: Dict[str, Any]) -> str:
        """Extract image URL from NY Times article"""
        multimedia = article.get('multimedia', [])
        if multimedia:
            for media in multimedia:
                if media.get('type') == 'image':
                    return f"https://www.nytimes.com/{media.get('url', '')}"
        return ''
    
    def _calculate_sentiment(self, text: str) -> float:
        """
        Calculate a sentiment score for text
        
        Returns a score between -1.0 (negative) and 1.0 (positive)
        """
        if not text:
            return 0.0
            
        # Simple word lists for basic sentiment analysis
        positive_words = [
            'up', 'gain', 'gains', 'positive', 'profit', 'profits', 'growth', 'increase',
            'increases', 'increased', 'higher', 'bull', 'bullish', 'opportunity', 'opportunities',
            'success', 'successful', 'win', 'winning', 'advantage', 'advantages', 'strong',
            'strength', 'strengthen', 'strengthened', 'grow', 'growing', 'grew', 'risen',
            'rises', 'rise', 'outperform', 'outperforms', 'outperformed', 'beat', 'beats',
            'progress', 'progressive', 'improve', 'improved', 'improving', 'exceed', 'exceeds',
            'exceeded', 'expectations', 'surpass', 'surpasses', 'surpassed', 'potential', 'rally',
            'rallies', 'bullish', 'uptrend', 'upside', 'optimistic', 'recovery', 'rebound',
            'momentum', 'attractive', 'promising', 'breakthrough', 'innovation', 'innovative'
        ]
        
        negative_words = [
            'down', 'loss', 'losses', 'negative', 'deficit', 'decline', 'declines', 'decreased',
            'lower', 'bear', 'bearish', 'risk', 'risks', 'risky', 'fail', 'fails', 'failed',
            'failure', 'weak', 'weakness', 'weakened', 'fall', 'falling', 'fell', 'drop',
            'drops', 'dropped', 'shrink', 'shrinks', 'shrinking', 'shrank', 'underperform',
            'underperforms', 'underperformed', 'miss', 'misses', 'missed', 'concern', 'concerns',
            'concerning', 'worry', 'worries', 'worried', 'struggle', 'struggles', 'struggling',
            'difficult', 'difficulty', 'difficulties', 'problem', 'problems', 'problematic',
            'bearish', 'downside', 'slowdown', 'disappointment', 'disappoint', 'disappoints',
            'disappointed', 'disappointing', 'pessimistic', 'pessimism', 'warning', 'warn',
            'warns', 'warned', 'downturn', 'downtrend', 'cut', 'cuts', 'challenge', 'challenged'
        ]
        
        # Add specific financial sector terms
        financial_positive = [
            'dividend', 'dividends', 'buyback', 'buybacks', 'acquisition', 'acquisitions',
            'merger', 'mergers', 'invested', 'investment', 'investing', 'profitability',
            'profitable', 'earnings', 'revenue', 'revenues', 'sales', 'cash flow', 'cashflow',
            'guidance', 'upgrade', 'upgrades', 'upgraded', 'buy', 'buying', 'accumulate'
        ]
        
        financial_negative = [
            'debt', 'debts', 'indebted', 'bankruptcy', 'bankrupt', 'liquidation', 'liquidate',
            'investigation', 'investigations', 'investigated', 'lawsuit', 'lawsuits', 'sue',
            'sued', 'fine', 'fines', 'penalty', 'penalties', 'downgrade', 'downgrades',
            'downgraded', 'sell', 'selling', 'short', 'shorting', 'restructuring', 'layoff',
            'layoffs', 'fire', 'fired', 'firing', 'expense', 'expenses', 'costly', 'cost-cutting'
        ]
        
        # Combine word lists
        positive_words.extend(financial_positive)
        negative_words.extend(financial_negative)
            
        text = text.lower()
        words = text.split()
        
        # Count positive and negative words with intensity modifiers
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            # Check for intensifiers
            multiplier = 1.0
            if i > 0:
                if words[i-1] in ['very', 'highly', 'extremely', 'significantly']:
                    multiplier = 2.0
                elif words[i-1] in ['somewhat', 'slightly', 'marginally']:
                    multiplier = 0.5
            
            if word in positive_words:
                positive_count += multiplier
            elif word in negative_words:
                negative_count += multiplier
                
            # Handle negation
            if i > 0 and words[i-1] in ['not', "n't", 'never', 'no', 'neither', 'nor']:
                if word in positive_words:
                    positive_count -= multiplier
                    negative_count += multiplier
                elif word in negative_words:
                    negative_count -= multiplier
                    positive_count += multiplier
        
        # Calculate sentiment score (-1 to 1)
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0
            
        return (positive_count - negative_count) / total_count
    
    def _calculate_relevance(self, query: str, title: str, content: str) -> float:
        """
        Calculate relevance score of article to query
        
        Returns a score between 0.0 (not relevant) and 1.0 (highly relevant)
        """
        if not query or not (title or content):
            return 0.0
            
        query = query.lower()
        full_text = (title + " " + content).lower()
        
        # Direct match in title is highly relevant
        if query in title.lower():
            return 1.0
            
        # Count occurrences in full text
        query_terms = query.split()
        hits = 0
        
        for term in query_terms:
            # Skip very short terms
            if len(term) <= 2:
                continue
                
            hits += full_text.count(term)
        
        # Calculate score based on number of hits
        word_count = len(full_text.split())
        if word_count == 0:
            return 0.0
            
        density = hits / word_count
        # Cap at 1.0
        return min(density * 50, 1.0)  # Multiply by 50 to scale up the usually small density values
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get current API usage statistics"""
        return {
            api_name: api_info['current_usage'] 
            for api_name, api_info in self.api_data['apis'].items()
        }
    
    def reset_api_errors(self, api_name: Optional[str] = None) -> None:
        """
        Reset error count for specific API or all APIs
        
        Args:
            api_name: Name of API to reset, or None to reset all
        """
        if api_name:
            if api_name in self.api_data['apis']:
                self.api_data['apis'][api_name]['error_count'] = 0
                self.api_data['apis'][api_name]['enabled'] = True
                logger.info(f"Reset errors for {api_name} API")
        else:
            # Reset all APIs
            for name in self.api_data['apis']:
                self.api_data['apis'][name]['error_count'] = 0
                self.api_data['apis'][name]['enabled'] = True
            logger.info("Reset errors for all APIs")
        
        self._save_config()
    
    def reset_daily_usage(self) -> None:
        """Reset daily usage counters for all APIs"""
        for api_name in self.api_data['apis']:
            self.api_data['apis'][api_name]['current_usage'] = 0
            
        logger.info("Reset daily usage counters for all APIs")
        self._save_config()
        
    def reset_cooldowns(self) -> None:
        """Reset cooldown timers for all APIs"""
        for api_name in self.api_data['apis']:
            self.api_data['apis'][api_name]['last_used'] = 0
            
        logger.info("Reset cooldown timers for all APIs")
        self._save_config()

    def validate_api_keys(self) -> Dict[str, bool]:
        """
        Validate all API keys by sending a minimal test request
        
        Returns:
            Dictionary mapping API names to validation status
        """
        validation_results = {}
        test_query = "market"  # Generic query for testing
        
        logger.info("Starting API key validation")
        
        for api_name, api_info in self.api_data['apis'].items():
            if not api_info.get('api_key') or not api_info.get('enabled', True):
                validation_results[api_name] = False
                continue
                
            try:
                # Use each API's fetch method with a minimal request
                fetch_method = getattr(self, f"_fetch_from_{api_name.lower()}", None)
                if not fetch_method:
                    validation_results[api_name] = False
                    continue
                    
                # Set very low timeout for validation
                original_timeout = self.api_data['settings']['timeout_seconds']
                self.api_data['settings']['timeout_seconds'] = 5
                
                # Attempt minimal fetch
                try:
                    results = fetch_method(test_query, api_info, 1)
                    # If we get here, the API key is valid
                    validation_results[api_name] = True
                    logger.info(f"API key for {api_name} is valid")
                    
                    # Reset any error counts
                    api_info['error_count'] = 0
                    api_info['auth_error'] = False
                    
                except Exception as e:
                    error_message = str(e)
                    if '401' in error_message or '403' in error_message:
                        logger.warning(f"API key for {api_name} is invalid: {error_message}")
                        validation_results[api_name] = False
                        api_info['auth_error'] = True
                    else:
                        # Other errors might not be key-related
                        logger.warning(f"Error validating {api_name}, might be service issue: {error_message}")
                        validation_results[api_name] = None
                
                # Restore original timeout
                self.api_data['settings']['timeout_seconds'] = original_timeout
                
            except Exception as e:
                logger.error(f"Unexpected error validating {api_name}: {str(e)}")
                validation_results[api_name] = False
        
        # Update configuration after validation
        self._save_config()
        return validation_results 