# News Integration

The BensBot Trading System features comprehensive news integration with multiple financial news sources, delivering actionable market insights directly to your dashboard.

## Features Overview

### Dashboard Tab News Feed

The dashboard provides a professional news digest from multiple financial news APIs with:

- **Impact-categorized news cards**
  - High impact news (significant market movers)
  - Medium impact news (sector or industry relevant)
  - Low impact news (general market information)

- **Professional UI design**
  - Dark blue card backgrounds with white text
  - Source logos to identify news providers
  - Clean, institutional-grade layout

- **Actionable insights**
  - Portfolio impact assessment with likelihood scores
  - Suggested action plans based on news content
  - Sector performance tables alongside news items

### Symbol-Specific News

The News/Prediction tab shows targeted news for the currently selected symbol:

- **Real-time updates** from multiple financial news APIs
- **Symbol relevance scoring** to filter for highest-impact news
- **Sentiment analysis** of news content (positive/negative/neutral)
- **Price impact assessment** with probability indicators
- **"Read More" links** to access full articles at the source

## Technical Implementation

### News API Integration

The system integrates with multiple financial news APIs in a fallback chain:

1. **Primary sources**: NewsData.io and Alpha Vantage
2. **Secondary sources**: Marketaux, GNews
3. **Fallback sources**: MediaStack, Currents, NYTimes

This multi-API approach ensures:
- Continuous news coverage even if one API is unavailable
- Diverse perspectives from different news sources
- Rich content with varied metadata

### Robust Error Handling

The news integration system includes:

- **Diagnostic logging** of API request/response cycles
- **Automatic fallback** to alternative sources on failure
- **Cache management** to prevent API overuse
- **User-friendly error messages** when all sources are unavailable

### Configuration

News APIs are configured in the typed settings system:

```yaml
api:
  api_keys:
    newsdata: "YOUR_NEWSDATA_KEY"
    marketaux: "YOUR_MARKETAUX_KEY"
    gnews: "YOUR_GNEWS_KEY"
    mediastack: "YOUR_MEDIASTACK_KEY"
    currents: "YOUR_CURRENTS_KEY"
    nytimes: "YOUR_NYTIMES_KEY"

data:
  news_refresh_seconds: 300  # News refresh interval
  max_news_items: 25  # Maximum news items per source
  enable_sentiment_analysis: true  # Analyze news sentiment
```

## News Processing Pipeline

1. **Collection**: Gather news from multiple sources
2. **Deduplication**: Remove duplicate stories across sources
3. **Relevance Scoring**: Rank news by market impact
4. **Sentiment Analysis**: Determine positive/negative sentiment
5. **Impact Assessment**: Calculate potential portfolio impact
6. **Presentation**: Display in professional card format

## Sample Implementation

```python
from trading_bot.news import NewsAggregator

# Initialize news aggregator with fallback chain
news_aggregator = NewsAggregator(
    primary_sources=["newsdata", "alpha_vantage"],
    secondary_sources=["marketaux", "gnews"],
    fallback_sources=["mediastack", "currents", "nytimes"],
    config=settings
)

# Get general market news
market_news = news_aggregator.get_market_news(
    max_items=10,
    categories=["economy", "markets", "business"]
)

# Get symbol-specific news
symbol_news = news_aggregator.get_symbol_news(
    symbol="AAPL",
    max_items=15,
    days_back=7
)
```

## Customization

The news integration system can be customized with:

- **Source priority**: Change the order of API fallback
- **Refresh intervals**: Adjust how frequently news updates
- **Category filters**: Focus on specific news categories
- **Impact thresholds**: Set minimum impact level for display
- **Display settings**: Adjust card style and information density
