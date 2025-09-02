import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes sentiment from financial news articles and social media content.
    Provides detailed sentiment scoring for stocks based on multiple sources.
    """
    
    def __init__(self, custom_words_file: Optional[str] = None):
        """
        Initialize the sentiment analyzer
        
        Args:
            custom_words_file: Optional path to file with custom financial 
                               sentiment words and their scores
        """
        # Initialize VADER sentiment analyzer
        self.vader = SentimentIntensityAnalyzer()
        
        # Add custom financial sentiment lexicon
        self._add_financial_terms()
        
        # Load custom words if file provided
        if custom_words_file:
            self._load_custom_words(custom_words_file)
        
        # For tracking processed articles
        self.processed_articles = set()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
        logger.info("SentimentAnalyzer initialized")
    
    def _add_financial_terms(self):
        """Add finance-specific terms to the sentiment analyzer"""
        financial_terms = {
            # Positive terms
            'bullish': 3.0,
            'outperform': 2.5,
            'beat': 2.0,
            'exceeded': 2.0, 
            'upgrade': 2.0,
            'buy': 1.5,
            'growth': 1.5,
            'profit': 1.5,
            'positive': 1.0,
            'raised': 1.0,
            'higher': 0.8,
            'gain': 0.8,
            'increase': 0.7,
            'recovery': 0.7,
            'improved': 0.7,
            'strong': 0.7,
            'opportunity': 0.6,
            'momentum': 0.6,
            'upside': 0.5,
            
            # Negative terms
            'bearish': -3.0,
            'underperform': -2.5,
            'miss': -2.0,
            'missed': -2.0,
            'downgrade': -2.0,
            'sell': -1.5,
            'loss': -1.5,
            'negative': -1.0,
            'lowered': -1.0,
            'decline': -1.0,
            'weak': -0.8,
            'drop': -0.8,
            'fell': -0.8,
            'decrease': -0.7,
            'disappointing': -1.2,
            'below': -0.6,
            'concern': -0.8,
            'risk': -0.6,
            'volatile': -0.5,
            'caution': -0.5,
            'pressure': -0.5,
            'slowdown': -0.8,
            'recession': -1.0,
            'bankruptcy': -3.0,
            'lawsuit': -1.5,
            'investigation': -1.2,
            'penalty': -1.2,
            'fine': -1.0,
            'debt': -0.7,
            'inflation': -0.6,
            'layoff': -1.5,
            'restructuring': -0.8,
            'delay': -0.7,
            'suspend': -1.0,
            'warning': -1.0
        }
        
        # Add words to VADER lexicon
        for word, score in financial_terms.items():
            self.vader.lexicon[word] = score
    
    def _load_custom_words(self, filepath: str):
        """
        Load custom sentiment words from file
        
        Args:
            filepath: Path to the file with words and scores
        """
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        word, score = parts[0].strip(), float(parts[1].strip())
                        self.vader.lexicon[word] = score
            logger.info(f"Loaded custom words from {filepath}")
        except Exception as e:
            logger.error(f"Error loading custom words: {str(e)}")
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment scores (compound, pos, neg, neu)
        """
        if not text or not isinstance(text, str):
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}
        
        # Clean text
        clean_text = self._preprocess_text(text)
        
        # Get sentiment scores
        sentiment = self.vader.polarity_scores(clean_text)
        
        return sentiment
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Original text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_news_articles(self, 
                             articles: List[Dict[str, Any]], 
                             ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles for a stock
        
        Args:
            articles: List of news articles as dictionaries
            ticker: Optional stock ticker to filter for specific mentions
            
        Returns:
            Dictionary with overall sentiment and article-level sentiment
        """
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'articles': []
            }
        
        # Track sentiment of all articles
        sentiments = []
        article_results = []
        
        # Counts
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        # Process each article
        for article in articles:
            # Skip already processed articles
            article_id = article.get('id', article.get('url', ''))
            if article_id in self.processed_articles:
                continue
            
            self.processed_articles.add(article_id)
            
            # Combine title and content for analysis
            title = article.get('title', '')
            content = article.get('content', article.get('description', ''))
            full_text = f"{title}. {content}"
            
            # Get sentiment
            sentiment = self.analyze_text(full_text)
            
            # Weight title more heavily
            title_sentiment = self.analyze_text(title)
            compound_score = sentiment['compound'] * 0.7 + title_sentiment['compound'] * 0.3
            
            # Adjust sentiment if ticker is mentioned directly
            if ticker and ticker.lower() in full_text.lower():
                # Extract sentences containing ticker for targeted analysis
                ticker_sentences = self._extract_ticker_sentences(full_text, ticker)
                if ticker_sentences:
                    ticker_sentiment = self.analyze_text(" ".join(ticker_sentences))
                    # Blend with overall sentiment, emphasizing ticker-specific sentiment
                    compound_score = compound_score * 0.4 + ticker_sentiment['compound'] * 0.6
            
            # Categorize sentiment
            sentiment_category = 'neutral'
            if compound_score >= 0.2:
                sentiment_category = 'positive'
                positive_count += 1
            elif compound_score <= -0.2:
                sentiment_category = 'negative'
                negative_count += 1
            else:
                neutral_count += 1
                
            # Append to results
            article_result = {
                'title': title,
                'published': article.get('published', article.get('publishedAt', '')),
                'source': article.get('source', {}).get('name', article.get('source', '')),
                'url': article.get('url', ''),
                'sentiment': sentiment_category,
                'sentiment_score': compound_score
            }
            article_results.append(article_result)
            sentiments.append(compound_score)
        
        # Calculate overall sentiment
        overall_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        # Sort articles by sentiment (most positive first)
        article_results = sorted(article_results, key=lambda x: x['sentiment_score'], reverse=True)
        
        return {
            'overall_sentiment': overall_sentiment,
            'article_count': len(article_results),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'articles': article_results
        }
    
    def _extract_ticker_sentences(self, text: str, ticker: str) -> List[str]:
        """
        Extract sentences that mention the ticker
        
        Args:
            text: The text to analyze
            ticker: The ticker symbol to look for
            
        Returns:
            List of sentences mentioning the ticker
        """
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Find sentences containing ticker (case insensitive)
        ticker_pattern = re.compile(fr'\b{re.escape(ticker)}\b', re.IGNORECASE)
        ticker_sentences = [s for s in sentences if ticker_pattern.search(s)]
        
        return ticker_sentences
    
    def calculate_ticker_sentiment(self, 
                                  ticker: str, 
                                  articles: List[Dict[str, Any]], 
                                  social_posts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Calculate overall sentiment for a ticker combining news and social media
        
        Args:
            ticker: The ticker symbol
            articles: List of news articles
            social_posts: Optional list of social media posts
            
        Returns:
            Dictionary with sentiment scores and analysis
        """
        # Analyze news articles
        news_analysis = self.analyze_news_articles(articles, ticker)
        
        # Analyze social media if provided
        social_analysis = None
        if social_posts:
            social_analysis = self.analyze_social_media(social_posts, ticker)
        
        # Calculate weighted sentiment score
        news_weight = 0.7  # News articles more reliable than social media
        social_weight = 0.3
        
        overall_score = news_analysis['overall_sentiment']
        if social_analysis:
            overall_score = (news_analysis['overall_sentiment'] * news_weight + 
                            social_analysis['overall_sentiment'] * social_weight)
        
        # Map to -1 to 1 range to 0 to 1 range
        normalized_score = (overall_score + 1) / 2
        
        # Calculate buzz/volume score based on number of mentions
        article_count = news_analysis['article_count']
        social_count = social_analysis['post_count'] if social_analysis else 0
        total_mentions = article_count + social_count
        
        # Log scale for mentions to avoid extreme values
        buzz_factor = np.log1p(total_mentions) / 5  # Scale factor
        buzz_score = min(1.0, buzz_factor)
        
        # Final sentiment score combines sentiment and buzz
        final_score = normalized_score * 0.8 + buzz_score * 0.2
        
        # Get top positive and negative articles
        top_positive = [a for a in news_analysis['articles'] if a['sentiment'] == 'positive'][:3]
        top_negative = [a for a in news_analysis['articles'] if a['sentiment'] == 'negative'][:3]
        
        # Calculate sentiment momentum (if timestamps available)
        sentiment_momentum = self._calculate_sentiment_momentum(news_analysis['articles'])
        
        return {
            'ticker': ticker,
            'sentiment_score': final_score,
            'raw_sentiment': overall_score,
            'buzz_score': buzz_score,
            'news_count': article_count,
            'social_count': social_count,
            'positive_news': news_analysis['positive_count'],
            'negative_news': news_analysis['negative_count'],
            'neutral_news': news_analysis['neutral_count'],
            'sentiment_momentum': sentiment_momentum,
            'top_positive': top_positive,
            'top_negative': top_negative,
            'sentiment_summary': self._generate_sentiment_summary(
                ticker, overall_score, news_analysis, social_analysis
            )
        }
    
    def _calculate_sentiment_momentum(self, articles: List[Dict[str, Any]]) -> float:
        """
        Calculate sentiment momentum based on chronological trend
        
        Args:
            articles: List of articles with timestamps and sentiment
            
        Returns:
            Momentum score from -1 to 1
        """
        if not articles or len(articles) < 2:
            return 0.0
        
        # Try to sort articles by date if available
        dated_articles = []
        for article in articles:
            pub_date = article.get('published', '')
            try:
                if isinstance(pub_date, str) and pub_date:
                    date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    dated_articles.append((date_obj, article['sentiment_score']))
            except (ValueError, TypeError):
                continue
        
        if len(dated_articles) < 2:
            return 0.0
        
        # Sort by date
        dated_articles.sort(key=lambda x: x[0])
        
        # Calculate weighted average of older vs newer articles
        older_half = dated_articles[:len(dated_articles)//2]
        newer_half = dated_articles[len(dated_articles)//2:]
        
        older_avg = sum(score for _, score in older_half) / len(older_half)
        newer_avg = sum(score for _, score in newer_half) / len(newer_half)
        
        # Momentum is the difference
        return newer_avg - older_avg
    
    def _generate_sentiment_summary(self, 
                                   ticker: str, 
                                   overall_score: float, 
                                   news_analysis: Dict[str, Any],
                                   social_analysis: Optional[Dict[str, Any]]) -> str:
        """
        Generate a natural language summary of sentiment analysis
        
        Args:
            ticker: Ticker symbol
            overall_score: Overall sentiment score
            news_analysis: News analysis results
            social_analysis: Social media analysis results
            
        Returns:
            Summary text
        """
        if overall_score >= 0.5:
            sentiment_desc = "very positive"
        elif overall_score >= 0.2:
            sentiment_desc = "positive"
        elif overall_score > -0.2:
            sentiment_desc = "neutral"
        elif overall_score > -0.5:
            sentiment_desc = "negative"
        else:
            sentiment_desc = "very negative"
        
        article_count = news_analysis['article_count']
        positive_pct = (news_analysis['positive_count'] / article_count * 100) if article_count > 0 else 0
        negative_pct = (news_analysis['negative_count'] / article_count * 100) if article_count > 0 else 0
        
        momentum = news_analysis.get('sentiment_momentum', 0)
        momentum_phrase = ""
        if abs(momentum) >= 0.1:
            direction = "improving" if momentum > 0 else "deteriorating"
            momentum_phrase = f" and {direction} recently"
        
        summary = f"Sentiment for {ticker} is {sentiment_desc}{momentum_phrase}. "
        
        if article_count > 0:
            summary += f"Based on {article_count} news articles, {positive_pct:.0f}% are positive and {negative_pct:.0f}% are negative. "
        
        if social_analysis and social_analysis.get('post_count', 0) > 0:
            social_count = social_analysis['post_count']
            social_sentiment = "positive" if social_analysis['overall_sentiment'] > 0.2 else "negative" if social_analysis['overall_sentiment'] < -0.2 else "neutral"
            summary += f"Social media sentiment is {social_sentiment} across {social_count} posts."
        
        return summary
    
    def analyze_social_media(self, 
                            posts: List[Dict[str, Any]], 
                            ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment from social media posts
        
        Args:
            posts: List of social media posts
            ticker: Optional ticker symbol to filter mentions
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not posts:
            return {
                'overall_sentiment': 0.0,
                'post_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'posts': []
            }
        
        # Process each post
        sentiments = []
        post_results = []
        
        # Counts
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for post in posts:
            # Get text content
            content = post.get('content', post.get('text', ''))
            if not content:
                continue
                
            # Get sentiment
            sentiment = self.analyze_text(content)
            compound_score = sentiment['compound']
            
            # Adjust for ticker mentions
            if ticker and ticker.lower() in content.lower():
                # Extract ticker context
                ticker_mentions = self._extract_ticker_context(content, ticker)
                if ticker_mentions:
                    ticker_sentiment = self.analyze_text(' '.join(ticker_mentions))
                    # Weight ticker mentions more heavily
                    compound_score = compound_score * 0.3 + ticker_sentiment['compound'] * 0.7
            
            # Categorize
            sentiment_category = 'neutral'
            if compound_score >= 0.2:
                sentiment_category = 'positive'
                positive_count += 1
            elif compound_score <= -0.2:
                sentiment_category = 'negative'
                negative_count += 1
            else:
                neutral_count += 1
            
            # Store result
            post_result = {
                'content': content[:100] + '...' if len(content) > 100 else content,
                'source': post.get('source', 'social media'),
                'timestamp': post.get('timestamp', post.get('created_at', '')),
                'sentiment': sentiment_category,
                'sentiment_score': compound_score
            }
            post_results.append(post_result)
            sentiments.append(compound_score)
        
        # Calculate overall
        overall_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        # Sort by sentiment
        post_results = sorted(post_results, key=lambda x: x['sentiment_score'], reverse=True)
        
        return {
            'overall_sentiment': overall_sentiment,
            'post_count': len(post_results),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'posts': post_results[:20]  # Limit to top 20
        }
    
    def _extract_ticker_context(self, text: str, ticker: str, window: int = 5) -> List[str]:
        """
        Extract context around ticker mentions
        
        Args:
            text: The text to analyze
            ticker: The ticker symbol
            window: Number of words to include before and after
            
        Returns:
            List of context snippets
        """
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Find ticker positions
        ticker_positions = [i for i, token in enumerate(tokens) if token.lower() == ticker.lower()]
        
        # Extract context
        contexts = []
        for pos in ticker_positions:
            start = max(0, pos - window)
            end = min(len(tokens), pos + window + 1)
            context = ' '.join(tokens[start:end])
            contexts.append(context)
        
        return contexts
    
    def extract_key_phrases(self, articles: List[Dict[str, Any]], ticker: str) -> List[str]:
        """
        Extract key phrases related to the ticker from articles
        
        Args:
            articles: List of news articles
            ticker: Ticker symbol
            
        Returns:
            List of key phrases
        """
        if not articles:
            return []
        
        # Extract all ticker sentences
        all_ticker_sentences = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('content', article.get('description', ''))
            full_text = f"{title}. {content}"
            
            ticker_sentences = self._extract_ticker_sentences(full_text, ticker)
            all_ticker_sentences.extend(ticker_sentences)
        
        if not all_ticker_sentences:
            return []
        
        # Process sentences to extract phrases
        phrases = []
        for sentence in all_ticker_sentences:
            # Tokenize
            tokens = word_tokenize(sentence.lower())
            
            # Filter stopwords
            filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 1]
            
            # Find adjectives and nouns near ticker
            try:
                ticker_idx = next((i for i, token in enumerate(filtered_tokens) if token.lower() == ticker.lower()), -1)
                if ticker_idx >= 0:
                    start = max(0, ticker_idx - 3)
                    end = min(len(filtered_tokens), ticker_idx + 4)
                    phrase = ' '.join(filtered_tokens[start:end])
                    phrases.append(phrase)
            except:
                continue
        
        # Return unique phrases
        return list(set(phrases))[:10]  # Top 10 unique phrases 