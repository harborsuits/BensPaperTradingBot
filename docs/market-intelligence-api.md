# Market Intelligence API

The Market Intelligence API provides institutional-grade financial insights, market data, and news integration for the BensBot Trading System.

## Overview

The Market Intelligence API delivers:

1. **Real-time Market Data**: Quotes, historical data, and technical indicators
2. **Financial News**: Multi-source news integration with impact analysis
3. **Sentiment Analysis**: Natural language processing of news and social media
4. **Pattern Recognition**: Technical chart pattern identification
5. **Portfolio Analysis**: Risk and exposure analytics

## API Architecture

The Market Intelligence API is built on FastAPI with:

- **JWT Authentication**: Secure access control
- **Rate Limiting**: Protect against abuse
- **Documentation**: Interactive OpenAPI documentation
- **Caching**: Efficient data storage and retrieval
- **Fallback Mechanisms**: Robust error handling and recovery

## News Integration

The API integrates with multiple financial news sources:

### Primary News Sources
- **NewsData.io**: Comprehensive global financial news
- **Alpha Vantage**: Economic and earnings news

### Secondary News Sources
- **Marketaux**: Financial news with sentiment analysis
- **GNews**: Aggregated news content

### Fallback News Sources
- **MediaStack**: Global news coverage
- **Currents**: Real-time news stream
- **NYTimes**: Financial and business reporting

All news is presented in a professional institutional-grade format with dark blue card backgrounds, white text, source logos, and actionable insights.

## News Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/news/market` | GET | Get latest market news with impact categorization |
| `/api/v1/news/symbol/{symbol}` | GET | Get symbol-specific news |
| `/api/v1/news/sector/{sector}` | GET | Get sector-specific news |
| `/api/v1/news/impact` | GET | Get high-impact news affecting the market |
| `/api/v1/news/portfolio` | GET | Get news impacting your portfolio holdings |

### News Response Format

```json
{
  "news_items": [
    {
      "title": "Federal Reserve Raises Interest Rates by 25 Basis Points",
      "summary": "The Federal Reserve raised its benchmark interest rate by 25 basis points to a target range of 5.25% to 5.50%, marking the highest level in 22 years.",
      "source": "Alpha Vantage",
      "source_logo_url": "https://static.alphaapi.com/logo.png",
      "published_at": "2023-07-26T14:00:00Z",
      "url": "https://www.example.com/article",
      "impact": {
        "level": "high",
        "markets_affected": ["equities", "bonds", "forex"],
        "direction": "negative",
        "confidence": 0.87
      },
      "portfolio_impact": {
        "affected_holdings": ["AAPL", "MSFT", "SPY"],
        "estimated_pct_impact": -0.43,
        "recommendation": "Consider increasing bond allocation as rates peak"
      },
      "sentiment": {
        "score": -0.32,
        "magnitude": 0.76,
        "aspects": {
          "market": -0.54,
          "economy": -0.21,
          "growth": -0.45
        }
      }
    }
  ],
  "source_distribution": {
    "Alpha Vantage": 4,
    "NewsData": 7,
    "Marketaux": 2,
    "GNews": 1
  },
  "sector_performance": {
    "technology": -0.42,
    "financials": 0.65,
    "healthcare": -0.18,
    "consumer_discretionary": -0.31,
    "energy": 0.22
  }
}
```

## Market Data Integration

### Market Data Sources
- **Tradier**: Primary source for real-time quotes and historical data
- **Alpha Vantage**: Technical indicators and fundamental data
- **Finnhub**: Alternative data and sentiment
- **Alpaca**: Additional market data with options support

### Market Data Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/market/quotes/{symbols}` | GET | Get real-time quotes for symbols |
| `/api/v1/market/history/{symbol}` | GET | Get historical price data |
| `/api/v1/market/indicators/{symbol}` | GET | Get technical indicators |
| `/api/v1/market/fundamentals/{symbol}` | GET | Get fundamental data |
| `/api/v1/market/options/{symbol}` | GET | Get options chain data |
| `/api/v1/market/screener` | POST | Screen for symbols meeting criteria |

## AI-Enhanced Analysis

The Market Intelligence API leverages multiple AI models for enhanced insights:

### AI Model Integrations
- **OpenAI**: Primary and secondary models for market analysis
- **Claude (Anthropic)**: Deep market context generation
- **Mistral**: Efficient pattern recognition
- **Cohere**: Semantic search and embeddings
- **Gemini**: Alternative analysis provider
- **HuggingFace**: Specialized financial models

### AI Analysis Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analysis/market-context` | GET | Get AI-generated market context |
| `/api/v1/analysis/symbol/{symbol}` | GET | Get AI analysis for a symbol |
| `/api/v1/analysis/news/{article_id}` | GET | Get AI breakdown of news impact |
| `/api/v1/analysis/portfolio` | GET | Get AI-generated portfolio insights |
| `/api/v1/analysis/recommend` | GET | Get AI trade recommendations |

## Configuration

The Market Intelligence API is configured through the typed settings system:

```python
class MarketIntelligenceSettings(BaseModel):
    """Market Intelligence API configuration."""
    host: str = "0.0.0.0"
    port: int = 8001
    enable_cache: bool = True
    cache_expiry_seconds: int = 60
    rate_limit_requests: int = 120
    rate_limit_period_seconds: int = 60
    news_sources_priority: List[str] = Field(
        default_factory=lambda: ["newsdata", "alpha_vantage", "marketaux", "gnews", "mediastack", "currents", "nytimes"]
    )
    enable_sentiment_analysis: bool = True
    enable_portfolio_impact: bool = True
    news_max_age_hours: int = 24
    max_news_items_per_request: int = 20
    default_impact_threshold: str = "medium"  # low, medium, high
    ai_provider: str = "openai"  # openai, claude, mistral
```

## Usage Example

### Python

```python
import requests

# Set up authentication
auth_response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={"username": "user", "password": "password"}
)
token = auth_response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Get high-impact news
news = requests.get(
    "http://localhost:8001/api/v1/news/impact?level=high",
    headers=headers
).json()

# Get AAPL analysis with news and AI insights
analysis = requests.get(
    "http://localhost:8001/api/v1/analysis/symbol/AAPL?include_news=true&include_ai=true",
    headers=headers
).json()

# Get portfolio impact from recent news
portfolio_impact = requests.get(
    "http://localhost:8001/api/v1/news/portfolio",
    headers=headers
).json()
```

### JavaScript

```javascript
// Authentication
const authResponse = await fetch('http://localhost:8000/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ username: 'user', password: 'password' })
});
const { access_token } = await authResponse.json();
const headers = { 'Authorization': `Bearer ${access_token}` };

// Get market context with AI analysis
const marketContext = await fetch('http://localhost:8001/api/v1/analysis/market-context', {
  headers
});
const contextData = await marketContext.json();

// Display in professional UI format (dark blue background, white text)
renderMarketContext(contextData);
```

## Error Handling

The API implements robust error handling with fallbacks:

```python
@app.exception_handler(NewsAPIException)
async def handle_news_api_error(request: Request, exc: NewsAPIException):
    """Handle failure of a news API by trying fallback sources."""
    logger.warning(f"News API error: {str(exc)}")
    
    # Try fallback news source
    try:
        news_service = get_fallback_news_service()
        results = await news_service.get_market_news()
        return results
    except Exception as fallback_error:
        logger.error(f"Fallback news source also failed: {str(fallback_error)}")
        
        # Return cached results if available
        cached_news = get_cached_news()
        if cached_news:
            return {
                "news_items": cached_news,
                "source": "cache",
                "cached_at": get_cache_timestamp(),
                "error": str(exc)
            }
            
        # Last resort - return error with friendly message
        return JSONResponse(
            status_code=503,
            content={
                "error": "News services temporarily unavailable",
                "message": "Our news providers are experiencing issues. Please try again later.",
                "technical_details": str(exc)
            }
        )
```

## Security

The Market Intelligence API implements several security measures:

1. **JWT Authentication**: Token-based authentication
2. **Rate Limiting**: Prevent abuse and DoS
3. **API Key Encryption**: Keys stored with secure encryption
4. **CORS Protection**: Configurable origins
5. **Input Validation**: Prevent injection attacks
