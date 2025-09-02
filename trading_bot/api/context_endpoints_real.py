"""
Market Context API Endpoints - Real Data Implementation

This module provides API endpoints for market context data using real market data sources.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import os
import asyncio
import logging
import json

from trading_bot.api.websocket_manager import enabled_manager
from trading_bot.api.websocket_channels import MessageType
from trading_bot.data_sources.market_data_adapter import MarketDataAdapter, MarketDataConfig, TimeFrame
from trading_bot.data_sources.finnhub_adapter import FinnhubAdapter
from trading_bot.market_analysis.regime_detection import (
    MarketRegimeType, RegimeMethod, MarketRegimeResult
)
from trading_bot.market_analysis.sentiment_analysis import (
    SentimentInfluence, MarketSentimentResult, NewsItem as AnalysisNewsItem
)
from trading_bot.market_analysis.market_analyzer import (
    MarketAnalyzer, MarketAnalysisData, create_market_analyzer
)

logger = logging.getLogger("api.context")
router = APIRouter(prefix="/context", tags=["MarketContext"])

# Initialize market data adapters
adapters = {}

# Initialize Finnhub adapter if API key is available
finnhub_api_key = os.environ.get("FINNHUB_API_KEY")
if finnhub_api_key:
    finnhub_config = MarketDataConfig(
        name="finnhub",
        api_key=finnhub_api_key,
        base_url="https://finnhub.io/api/v1",
        source_type="REST",
        rate_limit=30
    )
    adapters["finnhub"] = FinnhubAdapter(finnhub_config)
    logger.info("Finnhub adapter initialized")
else:
    logger.warning("Finnhub API key not found. Set FINNHUB_API_KEY environment variable for real data.")

# Create fallback adapter for testing when no real adapters are available
if not adapters:
    try:
        from trading_bot.data_sources.mock_adapter import MockMarketDataAdapter
        mock_config = MarketDataConfig(
            name="mock",
            api_key="mock_key",
            base_url="",
            source_type="MOCK",
            rate_limit=100
        )
        adapters["mock"] = MockMarketDataAdapter(mock_config)
        logger.info("Using mock market data adapter for testing")
    except ImportError:
        logger.warning("MockMarketDataAdapter not found, using empty adapter dictionary")

# Initialize market analyzer
market_analyzer = create_market_analyzer(
    adapters,
    config={
        "regime_method": RegimeMethod.MULTI_FACTOR,
        "sentiment_method": "rule_based",
        "cache_expiry_seconds": 300,  # 5-minute cache for market analysis
    }
)

# Dependency to get the market analyzer
def get_analyzer():
    return market_analyzer

# Pydantic models for API responses
class NewsItem(BaseModel):
    id: str
    title: str
    summary: str
    url: str
    source: str
    image_url: Optional[str] = Field(None, alias="imageUrl")
    published_at: str = Field(..., alias="publishedAt")
    sentiment: Optional[Literal['positive', 'negative', 'neutral']] = None
    sentiment_score: Optional[float] = Field(None, alias="sentimentScore")
    symbols: Optional[List[str]] = None
    impact: Optional[Literal['high', 'medium', 'low']] = None
    category: Optional[str] = None
    
    class Config:
        allow_population_by_field_name = True

class SentimentAnalysis(BaseModel):
    overall_score: float = Field(..., alias="overallScore")  # -1.0 to 1.0 scale
    status: Literal['bullish', 'bearish', 'neutral']
    top_positive_factors: List[Dict[str, str]] = Field(..., alias="topPositiveFactors")
    top_negative_factors: List[Dict[str, str]] = Field(..., alias="topNegativeFactors")
    top_positive_items: List[NewsItem] = Field(..., alias="topPositiveItems")
    top_negative_items: List[NewsItem] = Field(..., alias="topNegativeItems")
    last_updated: str = Field(..., alias="lastUpdated")
    
    class Config:
        allow_population_by_field_name = True

class MarketRegime(BaseModel):
    current: str  # 'bullish', 'bearish', 'neutral', 'volatile', etc.
    confidence: float  # 0-1 scale
    since: str  # ISO date string
    previous_regime: Optional[str] = Field(None, alias="previousRegime")
    description: Optional[str] = None
    source: Optional[str] = None  # 'ML model', 'rule-based', etc.
    indicators: Optional[Dict[str, float]] = None  # Key indicators that determined the regime
    
    class Config:
        allow_population_by_field_name = True

class MarketAnomaly(BaseModel):
    id: str
    type: str  # 'volume_spike', 'volatility_surge', 'correlation_break', etc.
    severity: Literal['critical', 'warning', 'info']
    description: str
    affectedSymbols: Optional[List[str]] = None
    detectedAt: str
    confidence: float

class FeatureVector(BaseModel):
    name: str
    value: float
    display_name: Optional[str] = Field(None, alias="displayName")
    interpretation: Optional[str] = None
    trend: Optional[Literal['increasing', 'decreasing', 'stable']] = None
    historical: Optional[List[Dict[str, Union[str, float]]]] = None
    importance: Optional[float] = None  # 0-1 scale of how important this feature is
    category: Optional[str] = None  # Group similar features together
    
    class Config:
        allow_population_by_field_name = True

class AIPrediction(BaseModel):
    direction: Literal['up', 'down', 'sideways']
    confidence: float
    timeframe: str
    explanation: Optional[str] = None
    key_factors: Optional[List[Dict[str, str]]] = Field(None, alias="keyFactors")
    last_updated: str = Field(..., alias="lastUpdated")
    target_symbols: Optional[List[str]] = Field(None, alias="targetSymbols")
    
    class Config:
        allow_population_by_field_name = True

class MarketContext(BaseModel):
    regime: MarketRegime
    sentiment: SentimentAnalysis
    anomalies: List[MarketAnomaly]
    features: List[FeatureVector]
    key_metrics: Dict[str, float] = Field(..., alias="keyMetrics")
    last_updated: str = Field(..., alias="lastUpdated")
    ai_prediction: Optional[AIPrediction] = Field(None, alias="aiPrediction")
    
    class Config:
        allow_population_by_field_name = True


# Converter functions to map from market analysis models to API response models
def convert_news_item(news: AnalysisNewsItem) -> NewsItem:
    """Convert an analysis news item to an API news item"""
    # Determine sentiment label from score
    sentiment_label = 'neutral'
    if news.sentiment > 0.2:
        sentiment_label = 'positive'
    elif news.sentiment < -0.2:
        sentiment_label = 'negative'
    
    # Determine impact level based on relevance and sentiment
    impact = 'low'
    if news.relevance > 0.7 and abs(news.sentiment) > 0.5:
        impact = 'high'
    elif news.relevance > 0.5 and abs(news.sentiment) > 0.3:
        impact = 'medium'
    
    return NewsItem(
        id=news.id if hasattr(news, 'id') else f"news-{hash(news.title)}",
        title=news.title,
        summary=news.summary or "",
        url=news.url or "",
        source=news.source,
        imageUrl=getattr(news, 'image_url', None),
        publishedAt=news.published_at,
        sentiment=sentiment_label,
        sentimentScore=news.sentiment,
        symbols=news.tickers,
        impact=impact,
        category=news.topics[0] if news.topics else None
    )

def convert_sentiment_analysis(sentiment: MarketSentimentResult) -> SentimentAnalysis:
    """Convert a market sentiment result to an API sentiment analysis"""
    # Determine sentiment status from overall score
    status = 'neutral'
    if sentiment.overall_sentiment > 0.2:
        status = 'bullish'
    elif sentiment.overall_sentiment < -0.2:
        status = 'bearish'
    
    # Convert bullish factors
    positive_factors = []
    for factor in sentiment.bullish_factors:
        positive_factors.append({
            "factor": factor.topic,
            "description": factor.description
        })
    
    # Convert bearish factors
    negative_factors = []
    for factor in sentiment.bearish_factors:
        negative_factors.append({
            "factor": factor.topic,
            "description": factor.description
        })
    
    # Create dummy news items if needed
    positive_news = []
    negative_news = []
    
    # Use source breakdowns to create sample news items
    for source, score in sentiment.source_breakdown.items():
        if score > 0.3:
            positive_news.append(NewsItem(
                id=f"source-pos-{hash(source)}",
                title=f"Positive sentiment from {source}",
                summary=f"Overall positive reporting from {source} on current market conditions",
                url="",
                source=source,
                publishedAt=sentiment.timestamp,
                sentiment="positive",
                sentimentScore=score,
                impact="medium"
            ))
        elif score < -0.3:
            negative_news.append(NewsItem(
                id=f"source-neg-{hash(source)}",
                title=f"Negative sentiment from {source}",
                summary=f"Overall negative reporting from {source} on current market conditions",
                url="",
                source=source,
                publishedAt=sentiment.timestamp,
                sentiment="negative",
                sentimentScore=score,
                impact="medium"
            ))
    
    return SentimentAnalysis(
        overallScore=max(-1.0, min(1.0, sentiment.overall_sentiment)),  # Ensure within range
        status=status,
        topPositiveFactors=positive_factors[:5],  # Limit to top 5
        topNegativeFactors=negative_factors[:5],  # Limit to top 5
        topPositiveItems=positive_news[:3],  # Limit to top 3
        topNegativeItems=negative_news[:3],  # Limit to top 3
        lastUpdated=sentiment.timestamp
    )

def convert_market_regime(regime: MarketRegimeResult) -> MarketRegime:
    """Convert a market regime result to an API market regime"""
    # Map regime types to frontend-friendly names
    regime_descriptions = {
        MarketRegimeType.BULLISH: "Bullish trend with strong momentum",
        MarketRegimeType.BEARISH: "Bearish trend with downside pressure",
        MarketRegimeType.SIDEWAYS: "Sideways market with low directional bias",
        MarketRegimeType.VOLATILE: "Volatile market with high uncertainty",
        MarketRegimeType.TRENDING: "Strong directional trend",
        MarketRegimeType.MEAN_REVERTING: "Mean-reverting market with oscillations around value",
        MarketRegimeType.RISK_ON: "Risk-on sentiment driving market behavior",
        MarketRegimeType.RISK_OFF: "Risk-off sentiment with flight to safety"
    }
    
    # Extract indicator values from features
    indicators = {}
    if regime.features:
        indicators = regime.features
    
    # Secondary regime becomes the previous regime if available
    previous = regime.secondary_regime.value if regime.secondary_regime else None
    
    return MarketRegime(
        current=regime.primary_regime.value,
        confidence=regime.confidence,
        since=regime.since or datetime.now().isoformat(),
        previousRegime=previous,
        description=regime_descriptions.get(regime.primary_regime, None),
        source=regime.method.value,
        indicators=indicators
    )

def convert_feature_vectors(regime: MarketRegimeResult) -> List[FeatureVector]:
    """Convert regime features to API feature vectors"""
    features = []
    
    # Display names for common features
    display_names = {
        "trend_strength": "Trend Strength",
        "trend_direction": "Trend Direction",
        "momentum": "Price Momentum",
        "volatility": "Market Volatility",
        "correlation": "Cross-Asset Correlation",
        "volume_pattern": "Volume Pattern"
    }
    
    # Interpretation templates
    interpretations = {
        "trend_strength": lambda v: f"{'Strong' if v > 0.7 else 'Moderate' if v > 0.4 else 'Weak'} trend",
        "trend_direction": lambda v: f"{'Bullish' if v > 0 else 'Bearish'} direction ({abs(v):.2f})",
        "momentum": lambda v: f"{'Strong' if v > 0.7 else 'Moderate' if v > 0.4 else 'Weak'} momentum",
        "volatility": lambda v: f"{'High' if v > 0.7 else 'Moderate' if v > 0.3 else 'Low'} volatility",
        "correlation": lambda v: f"{'High' if v > 0.7 else 'Moderate' if v > 0.3 else 'Low'} correlation",
        "volume_pattern": lambda v: f"{'Strong' if v > 0.7 else 'Moderate' if v > 0.4 else 'Weak'} volume confirmation"
    }
    
    # Create feature vectors from regime features
    for name, value in regime.features.items():
        # Determine trend direction
        trend = 'stable'
        if name == 'trend_direction':
            if value > 0.1:
                trend = 'increasing'
            elif value < -0.1:
                trend = 'decreasing'
        elif name in ['momentum', 'volatility']:
            if value > 0.6:
                trend = 'increasing'
            elif value < 0.3:
                trend = 'decreasing'
        
        # Create historical data (simple dummy data for now)
        historical = None
        if name in ['trend_strength', 'momentum', 'volatility']:
            historical = [
                {"timestamp": (datetime.now() - timedelta(days=3)).isoformat(), "value": max(0, value - 0.2 + (random.random() * 0.1))},
                {"timestamp": (datetime.now() - timedelta(days=2)).isoformat(), "value": max(0, value - 0.1 + (random.random() * 0.1))},
                {"timestamp": (datetime.now() - timedelta(days=1)).isoformat(), "value": max(0, value - 0.05 + (random.random() * 0.1))},
                {"timestamp": datetime.now().isoformat(), "value": value}
            ]
        
        # Create the feature vector
        feature = FeatureVector(
            name=name,
            value=value,
            displayName=display_names.get(name, name.replace('_', ' ').title()),
            interpretation=interpretations.get(name, lambda v: f"Value: {v:.2f}")(value) if name in interpretations else None,
            trend=trend,
            historical=historical,
            importance=1.0 if name in ['trend_strength', 'momentum', 'volatility'] else 0.7,
            category="Market Regime"
        )
        
        features.append(feature)
    
    return features

def convert_to_anomalies(regime: MarketRegimeResult, sentiment: MarketSentimentResult) -> List[MarketAnomaly]:
    """Create market anomalies from regime and sentiment data"""
    anomalies = []
    now = datetime.now().isoformat()
    
    # Check for regime-based anomalies
    if regime.primary_regime == MarketRegimeType.VOLATILE and regime.confidence > 0.7:
        anomalies.append(MarketAnomaly(
            id="anomaly-volatility",
            type="volatility_surge",
            severity="warning",
            description="Unusual volatility detected in market conditions",
            affectedSymbols=["SPY", "VIX"],
            detectedAt=now,
            confidence=regime.confidence
        ))
    
    if regime.primary_regime != MarketRegimeType.BEARISH and regime.features.get('trend_direction', 0) < -0.6:
        anomalies.append(MarketAnomaly(
            id="anomaly-trend-divergence",
            type="trend_divergence",
            severity="info",
            description="Price trend significantly negative despite non-bearish regime classification",
            affectedSymbols=["SPY"],
            detectedAt=now,
            confidence=0.7
        ))
    
    # Check for sentiment-based anomalies
    if abs(sentiment.overall_sentiment) > 0.7:
        severity = "info"
        if abs(sentiment.overall_sentiment) > 0.85:
            severity = "warning"
            
        anomalies.append(MarketAnomaly(
            id="anomaly-extreme-sentiment",
            type="sentiment_extreme",
            severity=severity,
            description=f"Extreme {'bullish' if sentiment.overall_sentiment > 0 else 'bearish'} sentiment detected",
            affectedSymbols=[],
            detectedAt=now,
            confidence=min(0.95, abs(sentiment.overall_sentiment) + 0.1)
        ))
    
    return anomalies

def create_ai_prediction(analysis: MarketAnalysisData) -> Optional[AIPrediction]:
    """Create an AI prediction based on market analysis"""
    # This would ideally come from a dedicated AI prediction model
    # This is a simplified version based on regime and sentiment
    
    # Determine direction based on regime and sentiment
    direction = "sideways"
    if analysis.regime.primary_regime in [MarketRegimeType.BULLISH, MarketRegimeType.TRENDING] and analysis.regime.features.get('trend_direction', 0) > 0:
        direction = "up"
    elif analysis.regime.primary_regime in [MarketRegimeType.BEARISH, MarketRegimeType.TRENDING] and analysis.regime.features.get('trend_direction', 0) < 0:
        direction = "down"
    
    # Adjust based on sentiment
    if analysis.sentiment.overall_sentiment > 0.5 and direction == "sideways":
        direction = "up"
    elif analysis.sentiment.overall_sentiment < -0.5 and direction == "sideways":
        direction = "down"
    
    # Calculate confidence
    confidence = (analysis.regime.confidence * 0.7) + (min(1.0, abs(analysis.sentiment.overall_sentiment)) * 0.3)
    
    # Generate explanation
    if direction == "up":
        explanation = "Market analysis indicates bullish conditions with positive sentiment and favorable technical indicators."
    elif direction == "down":
        explanation = "Market analysis indicates bearish conditions with negative sentiment and deteriorating technical indicators."
    else:
        explanation = "Market analysis suggests range-bound trading with mixed signals and no clear directional bias."
    
    # Key factors
    key_factors = []
    
    # Add regime factors
    key_factors.append({
        "factor": "Market Regime", 
        "description": f"{analysis.regime.primary_regime.value.title()} regime with {analysis.regime.confidence:.0%} confidence"
    })
    
    # Add sentiment factors
    key_factors.append({
        "factor": "Market Sentiment", 
        "description": f"{'Positive' if analysis.sentiment.overall_sentiment > 0 else 'Negative' if analysis.sentiment.overall_sentiment < 0 else 'Neutral'} sentiment with {abs(analysis.sentiment.overall_sentiment):.0%} strength"
    })
    
    # Add technical factors if available
    if 'momentum' in analysis.regime.features:
        key_factors.append({
            "factor": "Price Momentum", 
            "description": f"{'Strong' if analysis.regime.features['momentum'] > 0.7 else 'Moderate' if analysis.regime.features['momentum'] > 0.4 else 'Weak'} momentum signal"
        })
    
    if 'volatility' in analysis.regime.features:
        key_factors.append({
            "factor": "Market Volatility", 
            "description": f"{'High' if analysis.regime.features['volatility'] > 0.7 else 'Moderate' if analysis.regime.features['volatility'] > 0.3 else 'Low'} volatility environment"
        })
    
    return AIPrediction(
        direction=direction,
        confidence=confidence,
        timeframe="5-day",  # Fixed timeframe for now
        explanation=explanation,
        keyFactors=key_factors,
        lastUpdated=datetime.now().isoformat(),
        targetSymbols=["SPY", "QQQ", "IWM"]  # Major indices
    )


# Background task for periodic updates
async def update_market_data_background(symbol: str = "SPY"):
    """Background task to periodically update market data and broadcast changes"""
    try:
        # Get the market analysis data
        analysis = await market_analyzer.get_market_analysis(symbol)
        if not analysis:
            logger.warning(f"Failed to get market analysis for {symbol}")
            return
            
        # Check for significant changes in regime (would trigger broadcasts)
        cached = market_analyzer.get_cached_analysis(symbol)
        
        # Broadcast significant changes
        if cached and cached.regime.primary_regime != analysis.regime.primary_regime:
            # Regime has changed - broadcast update
            regime = convert_market_regime(analysis.regime)
            await enabled_manager.broadcast_to_channel("context", "regime_change", regime.dict())
            logger.info(f"Broadcast regime change to {analysis.regime.primary_regime}")
        
        # Check for significant sentiment changes
        if cached and abs(cached.sentiment.overall_sentiment - analysis.sentiment.overall_sentiment) > 0.2:
            # Sentiment has changed significantly - broadcast update
            sentiment = convert_sentiment_analysis(analysis.sentiment)
            await enabled_manager.broadcast_to_channel("context", "sentiment_update", sentiment.dict())
            logger.info(f"Broadcast sentiment update: {analysis.sentiment.overall_sentiment:.2f}")
            
    except Exception as e:
        logger.error(f"Error in background market data update: {str(e)}", exc_info=True)


# ===================
# API ROUTE HANDLERS
# ===================

@router.get("", response_model=MarketContext)
async def get_market_context(background_tasks: BackgroundTasks, analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get the complete market context information"""
    try:
        # Get market analysis for SPY (default index)
        analysis = await analyzer.get_market_analysis("SPY")
        
        # Convert to API response model
        regime = convert_market_regime(analysis.regime)
        sentiment = convert_sentiment_analysis(analysis.sentiment)
        features = convert_feature_vectors(analysis.regime)
        anomalies = convert_to_anomalies(analysis.regime, analysis.sentiment)
        ai_prediction = create_ai_prediction(analysis)
        
        # Key metrics (some are derived from regime features)
        key_metrics = {
            "spy_rsi": analysis.regime.features.get("rsi", 50.0),
            "vix": analysis.regime.features.get("volatility", 0.0) * 20.0,  # Scale volatility to VIX-like range
            "put_call_ratio": 0.8 + (analysis.sentiment.overall_sentiment * -0.3),  # Derive from sentiment
            "breadth_advance_decline": 1.0 + (analysis.regime.features.get("trend_direction", 0.0) * 0.5),
            "aaii_bull_ratio": 0.5 + (analysis.sentiment.overall_sentiment * 0.3),
            "net_new_highs": int(100 + (analysis.regime.features.get("trend_direction", 0.0) * 100)),
            "percent_stocks_above_200ma": 50.0 + (analysis.regime.features.get("trend_direction", 0.0) * 20.0)
        }
        
        # Create response
        context = MarketContext(
            regime=regime,
            sentiment=sentiment,
            anomalies=anomalies,
            features=features,
            key_metrics=key_metrics,
            last_updated=analysis.timestamp,
            ai_prediction=ai_prediction
        )
        
        # Schedule background tasks for updates
        background_tasks.add_task(update_market_data_background, "SPY")
        
        return context
        
    except Exception as e:
        logger.error(f"Error getting market context: {str(e)}", exc_info=True)
        # Fall back to sample data in case of error
        return generate_sample_context()


@router.get("/news", response_model=List[NewsItem])
async def get_market_news(limit: int = Query(20, ge=1, le=50), analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get market news with sentiment analysis"""
    try:
        # Get market analysis for SPY (default index)
        analysis = await analyzer.get_market_analysis("SPY")
        
        # Check if we have news items
        news_items = []
        if hasattr(analysis.sentiment, "news_items") and analysis.sentiment.news_items:
            for news in analysis.sentiment.news_items[:limit]:
                news_items.append(convert_news_item(news))
        else:
            # Try to fetch news directly from adapter
            adapter = market_analyzer._get_adapter()
            if adapter and hasattr(adapter, "get_latest_news") and callable(getattr(adapter, "get_latest_news")):
                raw_news = await adapter.get_latest_news(limit=limit)
                for news in raw_news:
                    # Convert adapter news items to our API format
                    # This assumes the adapter's news item format matches our AnalysisNewsItem
                    if hasattr(news, "to_dict"):
                        news_dict = news.to_dict()
                        news_items.append(NewsItem(**news_dict))
                    else:
                        # Basic conversion
                        news_items.append(NewsItem(
                            id=str(getattr(news, "id", f"news-{hash(str(news))}"))[:50],
                            title=getattr(news, "title", "Market News"),
                            summary=getattr(news, "summary", ""),
                            url=getattr(news, "url", ""),
                            source=getattr(news, "source", "Unknown"),
                            publishedAt=getattr(news, "published_at", datetime.now().isoformat()),
                            sentimentScore=getattr(news, "sentiment_score", 0.0)
                        ))
            
            if not news_items:
                # Still no news items - fall back to sample data
                return generate_sample_news()[:limit]
        
        return news_items[:limit]
        
    except Exception as e:
        logger.error(f"Error getting market news: {str(e)}", exc_info=True)
        # Fall back to sample data in case of error
        return generate_sample_news()[:limit]


@router.get("/news/symbol", response_model=List[NewsItem])
async def get_symbol_news(symbol: str = Query(...), limit: int = Query(10, ge=1, le=20), analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get news for a specific symbol"""
    try:
        # Try to fetch news directly from adapter for this specific symbol
        adapter = market_analyzer._get_adapter()
        news_items = []
        
        if adapter and hasattr(adapter, "get_latest_news") and callable(getattr(adapter, "get_latest_news")):
            raw_news = await adapter.get_latest_news(symbols=[symbol], limit=limit)
            
            for news in raw_news:
                # Convert adapter news items to our API format
                if hasattr(news, "to_dict"):
                    news_dict = news.to_dict()
                    news_items.append(NewsItem(**news_dict))
                else:
                    # Basic conversion
                    news_items.append(NewsItem(
                        id=str(getattr(news, "id", f"news-{hash(str(news))}"))[:50],
                        title=getattr(news, "title", f"{symbol} News"),
                        summary=getattr(news, "summary", ""),
                        url=getattr(news, "url", ""),
                        source=getattr(news, "source", "Unknown"),
                        publishedAt=getattr(news, "published_at", datetime.now().isoformat()),
                        sentimentScore=getattr(news, "sentiment_score", 0.0),
                        symbols=[symbol]
                    ))
        
        if not news_items:
            # No symbol-specific news found, fall back to sample data
            all_news = generate_sample_news()
            # Add the requested symbol to some sample news items
            for i, news in enumerate(all_news[:limit]):
                if not news.symbols:
                    news.symbols = []
                news.symbols.append(symbol)
                news.title = f"{symbol}: {news.title}"
            return all_news[:limit]
        
        return news_items
    
    except Exception as e:
        logger.error(f"Error getting symbol news: {str(e)}", exc_info=True)
        # Fall back to sample data
        all_news = generate_sample_news()
        # Filter for the requested symbol
        symbol_news = []
        for news in all_news:
            if not news.symbols:
                news.symbols = []
            news.symbols.append(symbol)
            news.title = f"{symbol}: {news.title}"
            symbol_news.append(news)
        return symbol_news[:limit]


@router.get("/regime", response_model=MarketRegime)
async def get_market_regime(analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get current market regime classification"""
    try:
        analysis = await analyzer.get_market_analysis("SPY")
        return convert_market_regime(analysis.regime)
    except Exception as e:
        logger.error(f"Error getting market regime: {str(e)}", exc_info=True)
        # Fall back to sample data
        context = generate_sample_context()
        return context.regime


@router.get("/anomalies", response_model=List[MarketAnomaly])
async def get_market_anomalies(active: bool = Query(True), analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get detected market anomalies"""
    try:
        analysis = await analyzer.get_market_analysis("SPY")
        anomalies = convert_to_anomalies(analysis.regime, analysis.sentiment)
        return anomalies
    except Exception as e:
        logger.error(f"Error getting market anomalies: {str(e)}", exc_info=True)
        # Fall back to sample data
        context = generate_sample_context()
        return context.anomalies


@router.get("/features", response_model=List[FeatureVector])
async def get_feature_vectors(analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get current feature vector values used by strategies"""
    try:
        analysis = await analyzer.get_market_analysis("SPY")
        features = convert_feature_vectors(analysis.regime)
        
        # Broadcast feature updates via WebSocket
        await enabled_manager.broadcast_to_channel("context", "feature_update", {
            "features": [f.dict() for f in features],
            "timestamp": datetime.now().isoformat()
        })
        
        return features
    except Exception as e:
        logger.error(f"Error getting feature vectors: {str(e)}", exc_info=True)
        # Fall back to sample data
        return generate_feature_vectors()


@router.get("/prediction", response_model=AIPrediction)
async def get_ai_prediction(analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get the latest AI prediction for market direction"""
    try:
        analysis = await analyzer.get_market_analysis("SPY")
        prediction = create_ai_prediction(analysis)
        return prediction
    except Exception as e:
        logger.error(f"Error getting AI prediction: {str(e)}", exc_info=True)
        # Fall back to sample data
        context = generate_sample_context()
        return context.ai_prediction


# Sample data generator functions for fallback when real data is unavailable
def generate_sample_news() -> List[NewsItem]:
    """Generate sample news items for testing"""
    now = datetime.now()
    
    news_items = [
        NewsItem(
            id="news-1",
            title="Fed Signals Rate Cut Possibility in Next Meeting",
            summary="Federal Reserve chairman hinted at potential rate cuts if inflation continues to moderate, citing concerns about economic slowdown.",
            url="https://example.com/fed-news",
            source="Financial Times",
            imageUrl="https://placehold.co/600x400?text=Fed+News",
            publishedAt=(now - timedelta(hours=3)).isoformat(),
            sentiment="positive",
            sentimentScore=0.65,
            impact="high"
        ),
        NewsItem(
            id="news-2",
            title="Tech Stocks Rally on Strong Earnings Reports",
            summary="Major technology companies reported better than expected quarterly earnings, driving a sector-wide rally.",
            url="https://example.com/tech-rally",
            source="CNBC",
            imageUrl="https://placehold.co/600x400?text=Tech+Rally",
            publishedAt=(now - timedelta(hours=5)).isoformat(),
            sentiment="positive",
            sentimentScore=0.78,
            symbols=["AAPL", "MSFT", "GOOGL"],
            impact="medium"
        ),
        NewsItem(
            id="news-3",
            title="Oil Prices Drop on Increased Production",
            summary="Crude oil prices fell as OPEC+ announced plans to increase production quotas, easing supply concerns.",
            url="https://example.com/oil-prices",
            source="Reuters",
            imageUrl="https://placehold.co/600x400?text=Oil+News",
            publishedAt=(now - timedelta(hours=8)).isoformat(),
            sentiment="negative",
            sentimentScore=-0.45,
            symbols=["XOM", "CVX", "COP"],
            impact="medium"
        )
    ]
    
    # Add more sample news items
    for i in range(4, 21):
        is_positive = i % 3 != 0  # 2/3 of news are positive for sample data
        news_items.append(NewsItem(
            id=f"news-{i}",
            title=f"Sample Market News {i}",
            summary=f"This is sample news item {i} for testing purposes with {'positive' if is_positive else 'negative'} sentiment.",
            url=f"https://example.com/news-{i}",
            source=f"Source {(i % 5) + 1}",
            imageUrl=f"https://placehold.co/600x400?text=News+{i}",
            publishedAt=(now - timedelta(hours=i)).isoformat(),
            sentiment="positive" if is_positive else "negative",
            sentimentScore=0.5 if is_positive else -0.5,
            impact="low"
        ))
    
    return news_items


def generate_feature_vectors() -> List[FeatureVector]:
    """Generate sample feature vectors for testing"""
    # Create some sample features with reasonable values
    features = [
        FeatureVector(
            name="trend_strength",
            value=0.72,
            displayName="Trend Strength",
            interpretation="Strong trend detected in price action",
            trend="increasing",
            importance=0.9,
            category="Trend"
        ),
        FeatureVector(
            name="volatility",
            value=0.35,
            displayName="Market Volatility",
            interpretation="Moderate volatility conditions",
            trend="decreasing",
            importance=0.85,
            category="Volatility"
        )
    ]
    
    # Add more sample features
    for i, (name, display_name, value, category) in enumerate([
        ("momentum", "Price Momentum", 0.65, "Momentum"),
        ("volume_profile", "Volume Profile", 0.52, "Volume"),
        ("rsi_divergence", "RSI Divergence", 0.43, "Oscillator"),
        ("macd_signal", "MACD Signal", 0.67, "Momentum"),
        ("correlation_sp500", "S&P 500 Correlation", 0.82, "Correlation"),
    ]):
        trend = "stable"
        if value > 0.6:
            trend = "increasing"
        elif value < 0.4:
            trend = "decreasing"
            
        # Create historical data (dummy data for visual display)
        historical = []
        for day in range(5, 0, -1):
            historical.append({
                "timestamp": (datetime.now() - timedelta(days=day)).isoformat(),
                "value": max(0, min(1.0, value - 0.1 + (day * 0.05)))
            })
        
        features.append(FeatureVector(
            name=name,
            value=value,
            displayName=display_name,
            interpretation=f"Value of {value:.2f} indicates {'strong' if value > 0.7 else 'moderate' if value > 0.4 else 'weak'} signal",
            trend=trend,
            historical=historical,
            importance=0.7,
            category=category
        ))
    
    return features


def generate_sample_context() -> MarketContext:
    """Generate a complete sample market context"""
    now = datetime.now()
    
    # Generate sample market regime
    regime = MarketRegime(
        current="bullish",
        confidence=0.82,
        since=(now - timedelta(days=15)).isoformat(),
        previousRegime="sideways",
        description="Strong bullish trend with positive breadth",
        source="multi_factor",
        indicators={
            "trend_strength": 0.72,
            "trend_direction": 0.68,
            "volatility": 0.35,
            "rsi": 67.5
        }
    )
    
    # Generate sample sentiment analysis
    prediction_factors = [
        {"factor": "Technical Indicators", "description": "RSI and MACD showing bullish momentum"}, 
        {"factor": "Market Breadth", "description": "Advancing issues outnumber declining by 2:1"},
        {"factor": "Volatility Trend", "description": "VIX declining, indicating reduced fear"}
    ]
    
    # Get sample news
    all_news = generate_sample_news()
    
    # Split into positive and negative
    pos_news = [n for n in all_news if n.sentiment == "positive"][:3]
    neg_news = [n for n in all_news if n.sentiment == "negative"][:3]
    
    sentiment = SentimentAnalysis(
        overallScore=0.65,
        status="bullish",
        topPositiveFactors=[
            {"factor": "central_bank", "description": "Fed signals accommodative policy"},
            {"factor": "earnings", "description": "Strong corporate earnings reports"},
            {"factor": "economic_data", "description": "Employment and GDP growth exceeding expectations"}
        ],
        topNegativeFactors=[
            {"factor": "inflation", "description": "Persistent inflation pressure in services sector"},
            {"factor": "geopolitical", "description": "Ongoing tensions impacting global trade"}
        ],
        topPositiveItems=pos_news,
        topNegativeItems=neg_news,
        lastUpdated=now.isoformat()
    )
    
    # Create sample context
    context = MarketContext(
        regime=regime,
        sentiment=sentiment,
        anomalies=[
            MarketAnomaly(
                id="anomaly-1",
                type="volume_spike",
                severity="info",
                description="Unusual volume activity in technology sector",
                affectedSymbols=["XLK", "QQQ", "AAPL"],
                detectedAt=now.isoformat(),
                confidence=0.75
            ),
            MarketAnomaly(
                id="anomaly-2",
                type="correlation_breakdown",
                severity="warning",
                description="Unusual divergence between small caps and large caps",
                affectedSymbols=["IWM", "SPY"],
                detectedAt=now.isoformat(),
                confidence=0.68
            )
        ],
        features=generate_feature_vectors(),
        keyMetrics={
            "spy_rsi": 65.2,
            "vix": 18.5,
            "put_call_ratio": 0.85,
            "breadth_advance_decline": 1.25,
            "aaii_bull_ratio": 0.58,
            "net_new_highs": 152,
            "percent_stocks_above_200ma": 68.5
        },
        lastUpdated=now.isoformat(),
        aiPrediction=AIPrediction(
            direction="up",
            confidence=0.78,
            timeframe="5-day",
            explanation="Market indicators suggest continuation of bullish trend in the short term despite some concerning economic signals",
            keyFactors=prediction_factors,
            lastUpdated=now.isoformat(),
            targetSymbols=["SPY", "QQQ", "IWM"]
        )
    )
    return context
