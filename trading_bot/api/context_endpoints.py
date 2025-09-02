from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import os
import random
import asyncio
import logging
import json

from trading_bot.api.websocket_manager import enabled_manager
from trading_bot.api.websocket_channels import MessageType

# Import market analysis modules if available
try:
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
    REAL_DATA_AVAILABLE = True
except ImportError:
    # If market analysis modules aren't available, fall back to mock data
    REAL_DATA_AVAILABLE = False

logger = logging.getLogger("api.context")
router = APIRouter(prefix="/context", tags=["MarketContext"])

# Initialize real data if available
market_analyzer = None
if REAL_DATA_AVAILABLE:
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
            from trading_bot.data_sources.market_data_adapter import DataSourceType
            mock_config = MarketDataConfig(
                source_type=DataSourceType.INTERNAL_DB,
                api_key="mock_key",
                base_url="",
                rate_limit=100
            )
            adapters["mock"] = MockMarketDataAdapter(mock_config)
            logger.info("Using mock market data adapter for testing")
        except ImportError:
            logger.warning("MockMarketDataAdapter not found, using empty adapter dictionary")

    # Initialize market analyzer if adapters are available
    if adapters:
        market_analyzer = create_market_analyzer(
            adapters,
            config={
                "regime_method": RegimeMethod.MULTI_FACTOR,
                "sentiment_method": "rule_based",
                "cache_expiry_seconds": 300,  # 5-minute cache for market analysis
            }
        )
        logger.info("Market analyzer initialized with real data sources")
    else:
        logger.warning("No market data adapters available, using sample data")
else:
    logger.warning("Market analysis modules not available, using sample data")

# Dependency to get the market analyzer if available
def get_analyzer():
    return market_analyzer

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
    overall_score: float = Field(..., alias="overallScore")  # 0 to 1 scale
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

# Sample data generator functions
def generate_sample_news() -> List[NewsItem]:
    now = datetime.utcnow()
    
    news_items = [
        NewsItem(
            id="news-1",
            title="Fed Signals Rate Cut Possibility in Next Meeting",
            summary="Federal Reserve officials hinted at potential rate cuts in response to cooling inflation data, which could boost market sentiment.",
            url="https://example.com/fed-news",
            source="Financial Times",
            image_url="https://example.com/images/fed.jpg",
            published_at=now.isoformat(),
            sentiment="positive",
            sentiment_score=0.75,
            symbols=["SPY", "QQQ", "TLT"],
            impact="high",
            category="monetary_policy"
        ),
        NewsItem(
            id="news-2",
            title="Tech Earnings Beat Expectations",
            summary="Major tech companies reported earnings above analyst expectations, driving market optimism and potential sector rotation.",
            url="https://example.com/tech-earnings",
            source="CNBC",
            published_at=(now - timedelta(hours=2)).isoformat(),
            sentiment="positive",
            sentiment_score=0.85,
            symbols=["AAPL", "MSFT", "GOOGL"],
            impact="medium",
            category="earnings"
        ),
        NewsItem(
            id="news-3",
            title="Oil Prices Drop on Supply Concerns",
            summary="Crude oil prices fell as OPEC+ considers increasing production amid global demand uncertainty and potential economic slowdown.",
            url="https://example.com/oil-markets",
            source="Bloomberg",
            published_at=(now - timedelta(hours=5)).isoformat(),
            sentiment="negative",
            sentiment_score=0.35,  # Using 0-1 scale where 0.5 is neutral
            symbols=["USO", "XOP", "XLE"],
            impact="medium",
            category="commodities"
        ),
        NewsItem(
            id="news-4",
            title="Retail Sales Miss Expectations",
            summary="Consumer spending shows signs of weakening as retail sales data disappoints economists, raising concerns about economic growth.",
            url="https://example.com/retail-sales",
            source="Reuters",
            published_at=(now - timedelta(hours=8)).isoformat(),
            sentiment="negative",
            sentiment_score=0.25,
            symbols=["XRT", "WMT", "TGT"],
            impact="medium",
            category="economic_data"
        ),
        NewsItem(
            id="news-5",
            title="Treasury Yields Rise Sharply",
            summary="Bond market selloff intensifies as yields reach new highs amid inflation concerns and changing expectations for Fed policy.",
            url="https://example.com/treasury-yields",
            source="Bloomberg",
            published_at=(now - timedelta(hours=10)).isoformat(),
            sentiment="negative",
            sentiment_score=0.30,
            symbols=["TLT", "IEF", "AGG"],
            impact="high",
            category="fixed_income"
        ),
        NewsItem(
            id="news-6",
            title="Market Volatility Index Jumps 15%",
            summary="The VIX surges as investors hedge against uncertainty following mixed economic signals and geopolitical tensions.",
            url="https://example.com/vix-jump",
            source="MarketWatch",
            published_at=(now - timedelta(hours=3)).isoformat(),
            sentiment="negative",
            sentiment_score=0.25,
            symbols=["VIX", "UVXY", "SVXY"],
            impact="high",
            category="volatility"
        ),
        NewsItem(
            id="news-7",
            title="AI Sector Receives Major Investment Boost",
            summary="Venture capital firms announce significant funding rounds for artificial intelligence startups, signaling continued growth.",
            url="https://example.com/ai-investment",
            source="TechCrunch",
            published_at=(now - timedelta(hours=7)).isoformat(),
            sentiment="positive",
            sentiment_score=0.80,
            symbols=["NVDA", "AMD", "GOOGL"],
            impact="medium",
            category="technology"
        ),
        NewsItem(
            id="news-8",
            title="Employment Numbers Exceed Forecasts",
            summary="Non-farm payroll data shows stronger than expected job growth, suggesting economic resilience despite rate pressures.",
            url="https://example.com/jobs-report",
            source="Wall Street Journal",
            published_at=(now - timedelta(hours=6)).isoformat(),
            sentiment="positive",
            sentiment_score=0.70,
            symbols=["SPY", "DIA", "XLF"],
            impact="high",
            category="economic_data"
        )
    ]
    return news_items

def generate_feature_vectors() -> List[FeatureVector]:
    now = datetime.utcnow()
    
    # Create timeseries data points for historical values
    def create_historical_data(values: List[float]) -> List[Dict[str, Union[str, float]]]:
        timestamps = [
            (now - timedelta(hours=4)).isoformat(),
            (now - timedelta(hours=3)).isoformat(),
            (now - timedelta(hours=2)).isoformat(),
            (now - timedelta(hours=1)).isoformat(),
        ]
        return [{'timestamp': ts, 'value': val} for ts, val in zip(timestamps, values)]
    
    features = [
        # Market dynamics category
        FeatureVector(
            name="market_momentum",
            display_name="Market Momentum",
            value=0.75,
            interpretation="Strong bullish momentum indicating potential trend continuation",
            trend="increasing",
            historical=create_historical_data([0.65, 0.68, 0.72, 0.75]),
            importance=0.85,
            category="market_dynamics"
        ),
        FeatureVector(
            name="volatility_index",
            display_name="Volatility (VIX)",
            value=18.5,
            interpretation="Moderately low volatility suggesting market confidence",
            trend="decreasing",
            historical=create_historical_data([22.5, 21.2, 19.8, 18.5]),
            importance=0.78,
            category="market_dynamics"
        ),
        FeatureVector(
            name="breadth_indicator",
            display_name="Market Breadth",
            value=0.62,
            interpretation="Healthy market breadth with broad participation",
            trend="stable",
            historical=create_historical_data([0.61, 0.63, 0.60, 0.62]),
            importance=0.70,
            category="market_dynamics"
        ),
        
        # Risk metrics category
        FeatureVector(
            name="liquidity_factor",
            display_name="Market Liquidity",
            value=0.85,
            interpretation="High market liquidity suggesting ease of position entering/exiting",
            trend="increasing",
            historical=create_historical_data([0.78, 0.80, 0.83, 0.85]),
            importance=0.65,
            category="risk_metrics"
        ),
        FeatureVector(
            name="yield_curve",
            display_name="Yield Curve Slope",
            value=-0.15,
            interpretation="Slight inversion indicating potential economic caution",
            trend="stable",
            historical=create_historical_data([-0.12, -0.14, -0.15, -0.15]),
            importance=0.75,
            category="risk_metrics"
        ),
        FeatureVector(
            name="credit_spread",
            display_name="Credit Spread",
            value=3.25,
            interpretation="Moderate credit spreads indicating balanced risk perception",
            trend="stable",
            historical=create_historical_data([3.30, 3.28, 3.26, 3.25]),
            importance=0.68,
            category="risk_metrics"
        ),
        
        # Technical indicators category
        FeatureVector(
            name="rsi_spy",
            display_name="RSI (S&P 500)",
            value=65.2,
            interpretation="Approaching overbought but still in healthy territory",
            trend="increasing",
            historical=create_historical_data([58.5, 61.3, 63.8, 65.2]),
            importance=0.72,
            category="technical_indicators"
        ),
        FeatureVector(
            name="macd_signal",
            display_name="MACD Signal",
            value=0.35,
            interpretation="Positive MACD showing upward momentum",
            trend="increasing",
            historical=create_historical_data([0.15, 0.22, 0.30, 0.35]),
            importance=0.68,
            category="technical_indicators"
        ),
        FeatureVector(
            name="avg_true_range",
            display_name="Average True Range",
            value=2.45,
            interpretation="Normal volatility levels in daily price action",
            trend="decreasing",
            historical=create_historical_data([2.85, 2.70, 2.55, 2.45]),
            importance=0.60,
            category="technical_indicators"
        ),
        
        # Sentiment indicators category
        FeatureVector(
            name="put_call_ratio",
            display_name="Put/Call Ratio",
            value=0.85,
            interpretation="Balanced options positioning with slight bearish tilt",
            trend="decreasing",
            historical=create_historical_data([0.95, 0.92, 0.88, 0.85]),
            importance=0.75,
            category="sentiment_indicators"
        ),
        FeatureVector(
            name="aaii_bull_ratio",
            display_name="AAII Bull Ratio",
            value=0.58,
            interpretation="Retail investors moderately bullish, not at extreme",
            trend="stable",
            historical=create_historical_data([0.57, 0.59, 0.58, 0.58]),
            importance=0.65,
            category="sentiment_indicators"
        )
    ]
    return features

def generate_sample_context() -> MarketContext:
    news = generate_sample_news()
    
    now = datetime.utcnow()
    
    # Separate positive and negative news
    pos_news = [n for n in news if n.sentiment == "positive"]
    neg_news = [n for n in news if n.sentiment == "negative"]
    
    # Positive and negative factors that influence market sentiment
    positive_factors = [
        {"factor": "Strong earnings reports", "impact": "high"},
        {"factor": "Fed policy accommodation", "impact": "high"},
        {"factor": "Improving economic data", "impact": "medium"},
        {"factor": "Increased retail participation", "impact": "medium"},
    ]
    
    negative_factors = [
        {"factor": "Rising inflation concerns", "impact": "high"},
        {"factor": "Geopolitical tensions", "impact": "medium"},
        {"factor": "Supply chain disruptions", "impact": "medium"},
        {"factor": "Treasury yield volatility", "impact": "medium"},
    ]
    
    # Indicators used for determining the market regime
    regime_indicators = {
        "trend_strength": 0.78,
        "momentum": 0.82,
        "volatility": 0.35,  # Lower is better
        "breadth": 0.65,
        "liquidity": 0.72
    }
    
    # AI prediction key factors
    prediction_factors = [
        {"factor": "Positive earnings momentum", "contribution": "strong"},
        {"factor": "Improving technical indicators", "contribution": "moderate"},
        {"factor": "Supportive monetary policy", "contribution": "strong"},
        {"factor": "Healthy market internals", "contribution": "moderate"}
    ]
    
    context = MarketContext(
        regime=MarketRegime(
            current="bullish",
            confidence=0.82,
            since=(now - timedelta(days=25)).isoformat(),
            previous_regime="neutral",
            description="Strong uptrend with broad participation across sectors",
            source="ML model",
            indicators=regime_indicators
        ),
        sentiment=SentimentAnalysis(
            overall_score=0.65,  # Using 0-1 scale where 0.5 is neutral
            status="bullish",
            top_positive_factors=positive_factors,
            top_negative_factors=negative_factors,
            top_positive_items=pos_news[:2],
            top_negative_items=neg_news[:2],
            last_updated=now.isoformat()
        ),
        anomalies=[
            MarketAnomaly(
                id="anomaly-1",
                type="volume_spike",
                severity="info",
                description="Unusual trading volume in technology sector",
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
        key_metrics={
            "spy_rsi": 65.2,
            "vix": 18.5,
            "put_call_ratio": 0.85,
            "breadth_advance_decline": 1.25,
            "aaii_bull_ratio": 0.58,
            "net_new_highs": 152,
            "percent_stocks_above_200ma": 68.5
        },
        last_updated=now.isoformat(),
        ai_prediction=AIPrediction(
            direction="up",
            confidence=0.78,
            timeframe="5-day",
            explanation="Market indicators suggest continuation of bullish trend in the short term despite some concerning economic signals",
            key_factors=prediction_factors,
            last_updated=now.isoformat(),
            target_symbols=["SPY", "QQQ", "IWM"]
        )
    )
    return context

@router.get("", response_model=MarketContext)
async def get_market_context(background_tasks: BackgroundTasks, analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get the complete market context information"""
    # Use real data if market analyzer is available, otherwise fall back to sample data
    if analyzer is not None and REAL_DATA_AVAILABLE:
        try:
            # Get market analysis for SPY (default index)
            analysis = await analyzer.get_market_analysis("SPY")
            
            # Convert to API response model
            from trading_bot.api.context_endpoints_real import (
                convert_market_regime, convert_sentiment_analysis, convert_feature_vectors,
                convert_to_anomalies, create_ai_prediction, update_market_data_background
            )
            
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
            logger.error(f"Error getting market context with real data: {str(e)}", exc_info=True)
            # Fall back to sample data in case of error
    
    # If no analyzer or error, use sample data
    return generate_sample_context()

@router.get("/news", response_model=List[NewsItem])
async def get_market_news(limit: int = Query(20, ge=1, le=50), analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get market news with sentiment analysis"""
    # Use real data if market analyzer is available
    if analyzer is not None and REAL_DATA_AVAILABLE:
        try:
            # Import news conversion function
            from trading_bot.api.context_endpoints_real import convert_news_item
            
            # First try to get news from market analyzer
            analysis = await analyzer.get_market_analysis("SPY")
            
            # Check if we have news items
            news_items = []
            if hasattr(analysis.sentiment, "news_items") and analysis.sentiment.news_items:
                for news in analysis.sentiment.news_items[:limit]:
                    news_items.append(convert_news_item(news))
            else:
                # Try to fetch news directly from adapter
                adapter = analyzer._get_adapter()
                if adapter and hasattr(adapter, "get_latest_news") and callable(getattr(adapter, "get_latest_news")):
                    raw_news = await adapter.get_latest_news(limit=limit)
                    for news in raw_news:
                        # Convert adapter news items to our API format
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
                
                if news_items:
                    return news_items[:limit]
        except Exception as e:
            logger.error(f"Error getting news with real data: {str(e)}", exc_info=True)
            # Fall back to sample data in case of error
    
    # If no analyzer or error, use sample data
    news_items = generate_sample_news()
    return news_items[:min(limit, len(news_items))]

@router.get("/news/symbol", response_model=List[NewsItem])
async def get_symbol_news(symbol: str = Query(...), limit: int = Query(10, ge=1, le=20), analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get news for a specific symbol"""
    # Use real data if market analyzer is available
    if analyzer is not None and REAL_DATA_AVAILABLE:
        try:
            # Try to fetch news directly from adapter for this specific symbol
            adapter = analyzer._get_adapter()
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
            
            if news_items:
                return news_items[:limit]
        except Exception as e:
            logger.error(f"Error getting symbol news with real data: {str(e)}", exc_info=True)
    
    # If no analyzer, no symbol-specific news found, or error, use sample data
    all_news = generate_sample_news()
    
    # Filter for the requested symbol or add it if no matches
    symbol_news = []
    for item in all_news:
        if item.symbols and symbol.upper() in [s.upper() for s in item.symbols]:
            symbol_news.append(item)
    
    # If no news for this symbol, modify some sample news to include this symbol
    if not symbol_news:
        for i, news in enumerate(all_news[:limit]):
            modified_news = NewsItem(
                **news.dict(),
                title=f"{symbol}: {news.title}"
            )
            if not modified_news.symbols:
                modified_news.symbols = []
            if symbol not in modified_news.symbols:
                modified_news.symbols.append(symbol)
            symbol_news.append(modified_news)
    
    return symbol_news[:min(limit, len(symbol_news))]

@router.get("/regime", response_model=MarketRegime)
async def get_market_regime(analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get current market regime classification"""
    # Use real data if market analyzer is available
    if analyzer is not None and REAL_DATA_AVAILABLE:
        try:
            # Import conversion function
            from trading_bot.api.context_endpoints_real import convert_market_regime
            
            # Get market analysis and extract regime
            analysis = await analyzer.get_market_analysis("SPY")
            regime = convert_market_regime(analysis.regime)
            return regime
        except Exception as e:
            logger.error(f"Error getting market regime with real data: {str(e)}", exc_info=True)
    
    # If no analyzer or error, use sample data
    context = generate_sample_context()
    return context.regime

@router.get("/anomalies", response_model=List[MarketAnomaly])
async def get_market_anomalies(active: bool = Query(True), analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get detected market anomalies"""
    # Use real data if market analyzer is available
    if analyzer is not None and REAL_DATA_AVAILABLE:
        try:
            # Import conversion function
            from trading_bot.api.context_endpoints_real import convert_to_anomalies
            
            # Get market analysis and detect anomalies
            analysis = await analyzer.get_market_analysis("SPY")
            anomalies = convert_to_anomalies(analysis.regime, analysis.sentiment)
            
            # Filter by active status if requested
            if active:
                # All anomalies from real-time analysis are considered active
                return anomalies
            else:
                # Simulate some resolved anomalies by returning an empty list
                # In a real implementation, we would track resolved anomalies separately
                return []
                
        except Exception as e:
            logger.error(f"Error getting market anomalies with real data: {str(e)}", exc_info=True)
    
    # If no analyzer or error, use sample data
    context = generate_sample_context()
    return context.anomalies

@router.get("/features", response_model=List[FeatureVector])
async def get_feature_vectors(analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get current feature vector values used by strategies"""
    # Use real data if market analyzer is available
    if analyzer is not None and REAL_DATA_AVAILABLE:
        try:
            # Import conversion function
            from trading_bot.api.context_endpoints_real import convert_feature_vectors
            
            # Get market analysis and extract features
            analysis = await analyzer.get_market_analysis("SPY")
            features = convert_feature_vectors(analysis.regime)
            
            # Broadcast feature updates via WebSocket
            await enabled_manager.broadcast_to_channel("context", "feature_update", {
                "features": [f.dict() for f in features],
                "timestamp": datetime.now().isoformat()
            })
            
            return features
        except Exception as e:
            logger.error(f"Error getting feature vectors with real data: {str(e)}", exc_info=True)
    
    # If no analyzer or error, use sample data
    features = generate_feature_vectors()
    # Broadcast feature updates via WebSocket
    await enabled_manager.broadcast_to_channel("context", "feature_update", features)
    return features

@router.get("/prediction", response_model=AIPrediction)
async def get_ai_prediction(analyzer: MarketAnalyzer = Depends(get_analyzer)):
    """Get the latest AI prediction for market direction"""
    # Use real data if market analyzer is available
    if analyzer is not None and REAL_DATA_AVAILABLE:
        try:
            # Import prediction creation function
            from trading_bot.api.context_endpoints_real import create_ai_prediction
            
            # Get market analysis and create prediction
            analysis = await analyzer.get_market_analysis("SPY")
            prediction = create_ai_prediction(analysis)
            return prediction
                
        except Exception as e:
            logger.error(f"Error getting AI prediction with real data: {str(e)}", exc_info=True)
    
    # If no analyzer or error, use sample data
    context = generate_sample_context()
    return context.aiPrediction

# Broadcast functions that could be called from a background task or scheduled job
async def broadcast_news_update(news_item: NewsItem):
    """Broadcast a news update to all connected clients"""
    await enabled_manager.broadcast_to_channel("context", "news_update", news_item.dict())

async def broadcast_regime_change(regime: MarketRegime):
    """Broadcast a regime change to all connected clients"""
    await enabled_manager.broadcast_to_channel("context", "regime_change", regime.dict())

async def broadcast_anomaly(anomaly: MarketAnomaly):
    """Broadcast a detected anomaly to all connected clients"""
    await enabled_manager.broadcast_to_channel("context", "anomaly_detected", anomaly.dict())

# These functions could be called when new data is processed
