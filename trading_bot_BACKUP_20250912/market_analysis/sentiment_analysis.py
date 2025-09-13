"""
Market Sentiment Analysis Module

This module provides functionality to analyze market sentiment from various sources
including news articles, social media, and analyst reports. It uses natural language
processing techniques to extract sentiment and key topics.
"""

import logging
import re
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field, validator

logger = logging.getLogger("market_analysis.sentiment")

class SentimentSource(str, Enum):
    """Sources of sentiment data"""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ANALYST_REPORTS = "analyst_reports"
    EARNINGS_CALLS = "earnings_calls"
    FINANCIAL_BLOGS = "financial_blogs"
    REDDIT = "reddit"
    TWITTER = "twitter"
    STOCKTWITS = "stocktwits"

class SentimentTopic(str, Enum):
    """Common sentiment topics"""
    ECONOMIC_OUTLOOK = "economic_outlook"
    FEDERAL_RESERVE = "federal_reserve"
    INTEREST_RATES = "interest_rates"
    INFLATION = "inflation"
    EARNINGS = "earnings"
    GEOPOLITICAL = "geopolitical"
    SECTOR_SPECIFIC = "sector_specific"
    TECHNICAL_ANALYSIS = "technical_analysis"
    MARKET_BREADTH = "market_breadth"
    VOLATILITY = "volatility"

class SentimentInfluence(BaseModel):
    """A factor influencing market sentiment"""
    topic: str  # Can be one of SentimentTopic or custom
    sentiment_score: float  # -1.0 to 1.0
    weight: float  # 0.0 to 1.0
    source: SentimentSource
    description: str
    reported_time: Optional[str] = None
    
    @validator('sentiment_score')
    def validate_score(cls, v):
        """Ensure sentiment_score is between -1 and 1"""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Sentiment score must be between -1.0 and 1.0, got {v}")
        return v
    
    @validator('weight')
    def validate_weight(cls, v):
        """Ensure weight is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {v}")
        return v

class MarketSentimentResult(BaseModel):
    """Result of market sentiment analysis"""
    overall_sentiment: float  # -1.0 to 1.0
    bullish_factors: List[SentimentInfluence] = []
    bearish_factors: List[SentimentInfluence] = []
    key_topics: List[str] = []
    source_breakdown: Dict[str, float] = {}  # Sentiment by source
    topic_breakdown: Dict[str, float] = {}  # Sentiment by topic
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @validator('overall_sentiment')
    def validate_overall(cls, v):
        """Ensure overall_sentiment is between -1 and 1"""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Overall sentiment must be between -1.0 and 1.0, got {v}")
        return v

class NewsItem(BaseModel):
    """A news article with sentiment analysis"""
    title: str
    source: str
    url: Optional[str] = None
    published_at: str
    summary: Optional[str] = None
    sentiment: float = 0.0  # -1.0 to 1.0
    relevance: float = 1.0  # 0.0 to 1.0
    topics: List[str] = []
    tickers: List[str] = []
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        """Ensure sentiment is between -1 and 1"""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Sentiment must be between -1.0 and 1.0, got {v}")
        return v

    @validator('relevance')
    def validate_relevance(cls, v):
        """Ensure relevance is between 0 and 1"""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Relevance must be between 0.0 and 1.0, got {v}")
        return v

class SocialMediaPost(BaseModel):
    """A social media post with sentiment analysis"""
    content: str
    source: SentimentSource
    user: Optional[str] = None
    posted_at: str
    sentiment: float = 0.0  # -1.0 to 1.0
    followers: Optional[int] = None
    engagement: Optional[int] = None
    topics: List[str] = []
    tickers: List[str] = []
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        """Ensure sentiment is between -1 and 1"""
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Sentiment must be between -1.0 and 1.0, got {v}")
        return v


class SentimentAnalyzer:
    """Base class for sentiment analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("sentiment_analyzer")
        
        # Configure source weights
        self.source_weights = {
            SentimentSource.NEWS: config.get("news_weight", 0.4) if config else 0.4,
            SentimentSource.SOCIAL_MEDIA: config.get("social_media_weight", 0.2) if config else 0.2,
            SentimentSource.ANALYST_REPORTS: config.get("analyst_weight", 0.25) if config else 0.25,
            SentimentSource.EARNINGS_CALLS: config.get("earnings_weight", 0.3) if config else 0.3,
            SentimentSource.FINANCIAL_BLOGS: config.get("blogs_weight", 0.15) if config else 0.15,
            SentimentSource.REDDIT: config.get("reddit_weight", 0.1) if config else 0.1,
            SentimentSource.TWITTER: config.get("twitter_weight", 0.1) if config else 0.1,
            SentimentSource.STOCKTWITS: config.get("stocktwits_weight", 0.1) if config else 0.1,
        }
        
        # Topic detection regex patterns
        self.topic_patterns = {
            SentimentTopic.FEDERAL_RESERVE: 
                r'fed|federal reserve|powell|fomc|monetary policy',
            SentimentTopic.INTEREST_RATES: 
                r'interest rates?|rate hikes?|basis points|yields?|treasury',
            SentimentTopic.INFLATION: 
                r'inflation|cpi|ppi|prices? ris(e|ing)|deflation',
            SentimentTopic.EARNINGS: 
                r'earnings|eps|revenue|profit|income statement|quarterly results?',
            SentimentTopic.GEOPOLITICAL: 
                r'geopolitic|war|conflict|trade tensions?|sanctions|tariffs?',
            SentimentTopic.ECONOMIC_OUTLOOK: 
                r'gdp|economic (outlook|forecast|growth)|recession|economy',
        }
        
        # Initialize keyword-based sentiment dictionary
        self.sentiment_keywords = {
            # Bullish keywords
            'buy': 0.5, 'bullish': 0.7, 'upside': 0.5, 'outperform': 0.6, 
            'strong': 0.4, 'positive': 0.4, 'growth': 0.3, 'beat': 0.5,
            'opportunity': 0.3, 'rally': 0.6, 'upgrade': 0.7, 'surge': 0.6,
            'exceeded': 0.5, 'higher': 0.3, 'record': 0.4, 'gains': 0.4,
            'recovery': 0.3, 'confidence': 0.3, 'impressive': 0.4,
            
            # Bearish keywords
            'sell': -0.5, 'bearish': -0.7, 'downside': -0.5, 'underperform': -0.6,
            'weak': -0.4, 'negative': -0.4, 'decline': -0.4, 'miss': -0.5,
            'risk': -0.3, 'sell-off': -0.6, 'downgrade': -0.7, 'plunge': -0.6,
            'missed': -0.5, 'lower': -0.3, 'losses': -0.4, 'concern': -0.3,
            'warning': -0.4, 'disappointing': -0.5, 'slowdown': -0.4,
            'recession': -0.6, 'bearmarket': -0.7, 'crash': -0.8, 'bubble': -0.5,
            'inflation': -0.3, 'uncertainty': -0.3, 'fear': -0.5, 'unemployment': -0.4,
            'downturn': -0.5, 'volatility': -0.3, 'unstable': -0.4,
            
            # Context modifiers
            'not ': -1.0,  # Negation modifier - multiply sentiment by this
            'very ': 1.5,  # Intensity modifier
            'extremely ': 2.0,  # Strong intensity modifier
            'slightly ': 0.5,  # Diminishing modifier
        }
    
    def analyze_market_sentiment(
        self,
        news_items: List[NewsItem] = None,
        social_posts: List[SocialMediaPost] = None,
        additional_data: Dict[str, Any] = None
    ) -> MarketSentimentResult:
        """
        Analyze market sentiment from various sources
        
        Args:
            news_items: List of news articles
            social_posts: List of social media posts
            additional_data: Any additional sentiment data
            
        Returns:
            Market sentiment analysis result
        """
        news_items = news_items or []
        social_posts = social_posts or []
        additional_data = additional_data or {}
        
        # Analyze news sentiment
        news_sentiment = self._analyze_news_sentiment(news_items)
        
        # Analyze social media sentiment
        social_sentiment = self._analyze_social_sentiment(social_posts)
        
        # Get any analyst sentiment from additional data
        analyst_sentiment = self._extract_analyst_sentiment(additional_data)
        
        # Combine all sentiment sources with weights
        sentiment_sources = []
        
        if news_sentiment:
            sentiment_sources.append((news_sentiment, self.source_weights[SentimentSource.NEWS]))
            
        if social_sentiment:
            sentiment_sources.append((social_sentiment, self.source_weights[SentimentSource.SOCIAL_MEDIA]))
            
        for source, (sentiment, factors) in analyst_sentiment.items():
            weight = self.source_weights.get(source, 0.2)
            sentiment_sources.append((sentiment, weight))
        
        # Calculate overall sentiment (weighted average)
        total_weight = sum(weight for _, weight in sentiment_sources)
        if total_weight > 0:
            overall = sum(sentiment * weight for sentiment, weight in sentiment_sources) / total_weight
        else:
            overall = 0.0
        
        # Extract bullish and bearish factors
        bullish_factors = []
        bearish_factors = []
        
        # From news
        for item in news_items:
            # Only include significant news items (high relevance and strong sentiment)
            if item.relevance > 0.7 and abs(item.sentiment) > 0.3:
                # Create a sentiment influence from this news item
                influence = SentimentInfluence(
                    topic=self._detect_topic(item.title, item.summary or ""),
                    sentiment_score=item.sentiment,
                    weight=item.relevance,
                    source=SentimentSource.NEWS,
                    description=item.title,
                    reported_time=item.published_at
                )
                
                if item.sentiment > 0:
                    bullish_factors.append(influence)
                else:
                    bearish_factors.append(influence)
        
        # From social media
        for post in social_posts:
            # Only include significant posts (strong sentiment)
            if abs(post.sentiment) > 0.5:
                # Weight by engagement if available
                weight = 0.5
                if post.engagement is not None and post.followers is not None:
                    # Higher weight for posts with high engagement relative to follower count
                    if post.followers > 0:
                        engagement_ratio = min(1.0, post.engagement / post.followers)
                        weight = 0.3 + (0.7 * engagement_ratio)
                
                influence = SentimentInfluence(
                    topic=self._detect_topic(post.content),
                    sentiment_score=post.sentiment,
                    weight=weight,
                    source=post.source,
                    description=post.content[:100] + "..." if len(post.content) > 100 else post.content,
                    reported_time=post.posted_at
                )
                
                if post.sentiment > 0:
                    bullish_factors.append(influence)
                else:
                    bearish_factors.append(influence)
        
        # From analyst sentiment
        for source, (_, factors) in analyst_sentiment.items():
            for factor in factors:
                if factor.sentiment_score > 0:
                    bullish_factors.append(factor)
                else:
                    bearish_factors.append(factor)
        
        # Sort factors by absolute impact (sentiment * weight)
        bullish_factors.sort(key=lambda x: x.sentiment_score * x.weight, reverse=True)
        bearish_factors.sort(key=lambda x: abs(x.sentiment_score * x.weight), reverse=True)
        
        # Limit to top factors
        max_factors = 5
        bullish_factors = bullish_factors[:max_factors]
        bearish_factors = bearish_factors[:max_factors]
        
        # Extract key topics from all news and social posts
        all_topics = []
        for item in news_items:
            all_topics.extend(item.topics)
        for post in social_posts:
            all_topics.extend(post.topics)
        
        # Count topic frequencies
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Get top topics
        key_topics = sorted(topic_counts.keys(), key=lambda x: topic_counts[x], reverse=True)[:8]
        
        # Calculate sentiment by source
        source_sentiment = {}
        for item in news_items:
            if item.source not in source_sentiment:
                source_sentiment[item.source] = []
            source_sentiment[item.source].append(item.sentiment)
        
        # Average sentiment by source
        source_breakdown = {}
        for source, sentiments in source_sentiment.items():
            if sentiments:
                source_breakdown[source] = sum(sentiments) / len(sentiments)
        
        # Calculate sentiment by topic
        topic_sentiment = {}
        for item in news_items:
            for topic in item.topics:
                if topic not in topic_sentiment:
                    topic_sentiment[topic] = []
                topic_sentiment[topic].append(item.sentiment)
        
        # Average sentiment by topic
        topic_breakdown = {}
        for topic, sentiments in topic_sentiment.items():
            if sentiments:
                topic_breakdown[topic] = sum(sentiments) / len(sentiments)
        
        # Create the result
        return MarketSentimentResult(
            overall_sentiment=overall,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            key_topics=key_topics,
            source_breakdown=source_breakdown,
            topic_breakdown=topic_breakdown
        )
    
    def analyze_news_item(self, title: str, content: str = None, source: str = None) -> NewsItem:
        """
        Analyze sentiment for a single news item
        
        Args:
            title: News headline
            content: News content/body
            source: News source
            
        Returns:
            NewsItem with sentiment analysis
        """
        text = f"{title} {content or ''}"
        
        # Extract sentiment using keyword-based approach
        sentiment = self._extract_sentiment_from_text(text)
        
        # Extract topics
        topics = []
        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                topics.append(topic.value)
        
        # Extract ticker symbols - simple regex for cashtags
        tickers = re.findall(r'\$([A-Z]{1,5})', text)
        
        # Create news item
        return NewsItem(
            title=title,
            source=source or "Unknown",
            published_at=datetime.now().isoformat(),
            summary=content[:200] + "..." if content and len(content) > 200 else content,
            sentiment=sentiment,
            relevance=1.0,  # Default to high relevance
            topics=topics,
            tickers=tickers
        )
    
    def _extract_sentiment_from_text(self, text: str) -> float:
        """
        Extract sentiment score from text using keyword-based approach
        Returns a score from -1.0 (bearish) to 1.0 (bullish)
        """
        text = text.lower()
        total_score = 0
        matches = 0
        
        # Check for keywords
        for keyword, score in self.sentiment_keywords.items():
            if keyword in text:
                # Count occurrences
                occurrences = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
                
                # Check for negation (e.g., "not bullish")
                negated = 0
                for i in range(len(text) - len(keyword)):
                    if text[i:i+len(keyword)] == keyword:
                        # Look for negation words before the keyword
                        start = max(0, i - 5)  # Look up to 5 characters before
                        prefix = text[start:i]
                        if "not " in prefix or "n't " in prefix:
                            negated += 1
                
                # Calculate contribution
                regular_occurrences = occurrences - negated
                negated_contribution = negated * score * -0.8  # Negation doesn't completely reverse
                regular_contribution = regular_occurrences * score
                
                total_score += regular_contribution + negated_contribution
                matches += occurrences
        
        # Normalize score between -1 and 1
        if matches > 0:
            normalized_score = total_score / (matches * 0.8)  # Scale factor to avoid extreme values
            return max(-1.0, min(1.0, normalized_score))
        
        return 0.0  # Neutral if no matches
    
    def _detect_topic(self, *texts) -> str:
        """Detect the main topic from a text"""
        combined_text = " ".join([t for t in texts if t])
        combined_text = combined_text.lower()
        
        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, combined_text, re.IGNORECASE):
                return topic.value
        
        return SentimentTopic.SECTOR_SPECIFIC.value  # Default
    
    def _analyze_news_sentiment(self, news_items: List[NewsItem]) -> float:
        """Calculate average sentiment from news items, weighted by relevance"""
        if not news_items:
            return 0.0
        
        total_weight = sum(item.relevance for item in news_items)
        if total_weight == 0:
            return 0.0
        
        weighted_sentiment = sum(item.sentiment * item.relevance for item in news_items)
        return weighted_sentiment / total_weight
    
    def _analyze_social_sentiment(self, posts: List[SocialMediaPost]) -> float:
        """Calculate average sentiment from social media posts, weighted by engagement"""
        if not posts:
            return 0.0
        
        total_weight = 0
        weighted_sentiment = 0
        
        for post in posts:
            # Weight by engagement if available, otherwise use 1.0
            weight = 1.0
            if post.engagement is not None:
                weight = min(5.0, 1.0 + (post.engagement / 1000))  # Cap at 5.0
            
            weighted_sentiment += post.sentiment * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sentiment / total_weight
    
    def _extract_analyst_sentiment(self, additional_data: Dict[str, Any]) -> Dict[SentimentSource, Tuple[float, List[SentimentInfluence]]]:
        """Extract analyst sentiment from additional data sources"""
        result = {}
        
        # Extract analyst reports if available
        if 'analyst_reports' in additional_data:
            reports = additional_data['analyst_reports']
            factors = []
            
            # Calculate average sentiment
            sentiments = []
            for report in reports:
                sentiment = report.get('sentiment', 0)
                sentiments.append(sentiment)
                
                # Create a sentiment influence
                if abs(sentiment) > 0.2:  # Only include significant sentiment
                    factors.append(SentimentInfluence(
                        topic=report.get('topic', SentimentTopic.SECTOR_SPECIFIC.value),
                        sentiment_score=sentiment,
                        weight=report.get('weight', 0.5),
                        source=SentimentSource.ANALYST_REPORTS,
                        description=report.get('title', 'Analyst Report'),
                        reported_time=report.get('date')
                    ))
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            result[SentimentSource.ANALYST_REPORTS] = (avg_sentiment, factors)
        
        # Extract earnings call sentiment if available
        if 'earnings_calls' in additional_data:
            calls = additional_data['earnings_calls']
            factors = []
            
            # Calculate average sentiment
            sentiments = []
            for call in calls:
                sentiment = call.get('sentiment', 0)
                sentiments.append(sentiment)
                
                # Create a sentiment influence
                if abs(sentiment) > 0.2:  # Only include significant sentiment
                    factors.append(SentimentInfluence(
                        topic=SentimentTopic.EARNINGS.value,
                        sentiment_score=sentiment,
                        weight=call.get('weight', 0.7),  # Earnings calls typically high weight
                        source=SentimentSource.EARNINGS_CALLS,
                        description=f"{call.get('company', 'Company')} Earnings Call",
                        reported_time=call.get('date')
                    ))
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            result[SentimentSource.EARNINGS_CALLS] = (avg_sentiment, factors)
        
        return result


class KeywordSentimentAnalyzer(SentimentAnalyzer):
    """A simple keyword-based sentiment analyzer"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Custom keyword dictionary can be provided in config
        custom_keywords = config.get('custom_keywords', {}) if config else {}
        self.sentiment_keywords.update(custom_keywords)


class RuleBasedSentimentAnalyzer(SentimentAnalyzer):
    """Rule-based sentiment analyzer with more advanced text processing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Add more advanced rules for sentiment analysis
        self.price_movement_patterns = {
            r'up \d+%': 0.5,
            r'down \d+%': -0.5,
            r'gained \d+%': 0.5,
            r'lost \d+%': -0.5,
            r'jumped \d+%': 0.6,
            r'plunged \d+%': -0.6,
            r'soared \d+%': 0.7,
            r'crashed \d+%': -0.7,
        }
        
        # Patterns for intensifiers and sentiment modifiers
        self.intensifiers = {
            r'very': 1.5,
            r'extremely': 2.0,
            r'slightly': 0.5,
            r'somewhat': 0.7,
            r'highly': 1.8,
            r'strongly': 1.7
        }
        
        self.negators = {
            r'not': -1.0,
            r'never': -1.0,
            r"n't": -1.0,
            r"isn't": -1.0,
            r"aren't": -1.0,
            r"wasn't": -1.0,
        }
    
    def _extract_sentiment_from_text(self, text: str) -> float:
        """
        Enhanced sentiment extraction using rules and patterns
        """
        text = text.lower()
        total_score = 0
        matches = 0
        
        # Check for keyword matches
        keyword_score, keyword_matches = self._get_keyword_sentiment(text)
        total_score += keyword_score
        matches += keyword_matches
        
        # Check for price movement patterns
        for pattern, score in self.price_movement_patterns.items():
            if re.search(pattern, text):
                # Extract the percentage to adjust the score
                match = re.search(r'(\d+)%', text)
                if match:
                    percentage = int(match.group(1))
                    # Scale score based on percentage (higher % = stronger sentiment)
                    adjusted_score = score * min(2.0, 1.0 + (percentage / 10))
                    total_score += adjusted_score
                    matches += 1
                else:
                    total_score += score
                    matches += 1
        
        # Normalize score
        if matches > 0:
            normalized_score = total_score / (matches * 0.8)  # Scale factor to avoid extreme values
            return max(-1.0, min(1.0, normalized_score))
        
        return 0.0
    
    def _get_keyword_sentiment(self, text: str) -> Tuple[float, int]:
        """Extract sentiment from keywords with support for negation and intensifiers"""
        score = 0
        matches = 0
        
        # Process text as a list of words to handle negation and intensifiers
        words = text.split()
        
        for i, word in enumerate(words):
            # Check if word is a sentiment keyword
            if word in self.sentiment_keywords:
                keyword_score = self.sentiment_keywords[word]
                matches += 1
                
                # Look for negators and intensifiers in the preceding words (up to 3 words back)
                modifier = 1.0
                for j in range(max(0, i-3), i):
                    prev_word = words[j]
                    
                    # Check for negators
                    for negator, neg_value in self.negators.items():
                        if prev_word == negator or prev_word.endswith(negator):
                            modifier *= neg_value
                    
                    # Check for intensifiers
                    for intensifier, int_value in self.intensifiers.items():
                        if prev_word == intensifier:
                            modifier *= int_value
                
                score += keyword_score * modifier
        
        return score, matches


# Factory function
def create_sentiment_analyzer(method: str = "keyword", config: Dict[str, Any] = None) -> SentimentAnalyzer:
    """Create a sentiment analyzer based on the specified method"""
    if method == "keyword":
        return KeywordSentimentAnalyzer(config)
    elif method == "rule_based":
        return RuleBasedSentimentAnalyzer(config)
    else:
        logger.warning(f"Unknown sentiment analysis method: {method}, using keyword")
        return KeywordSentimentAnalyzer(config)
