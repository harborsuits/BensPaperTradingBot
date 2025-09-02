# News Sentiment Integration Guide

This document explains the News Sentiment Analysis system that aggregates data from multiple outlets, clusters similar stories, and provides sentiment analysis with partisan and information density metrics.

## Overview

The News Sentiment Analysis system provides a comprehensive view of financial news by:

1. Fetching data from multiple news outlets per category (markets, politics, tech, crypto, macro)
2. Clustering similar stories to reduce duplication
3. Computing sentiment, partisanship, information density, and finance relevance for each article
4. Aggregating metrics at the cluster level and outlet level
5. Presenting a clean, unified view in the UI

## Features

- **Multi-Source Aggregation**: Pulls from multiple outlets per category
- **Deduplication**: Clusters similar stories using fuzzy matching
- **Sentiment Analysis**: Provides sentiment scores (-1 to 1) for each article and cluster
- **Partisanship Detection**: Measures political bias and partisan spread across sources
- **Information Density**: Scores articles based on factual content vs. hype
- **Finance Relevance**: Detects finance-specific terms and tickers
- **Category Filtering**: Organizes news by category (markets, politics, tech, crypto, macro)
- **Outlet Analysis**: Provides metrics per news outlet (avg sentiment, bias, info quality)

## Architecture

### Backend Components

1. **Configuration**: `news_sources.yaml` defines outlets, categories, and bias priors
2. **Service Layer**: `news_service.py` handles fetching, extraction, clustering, and analysis
3. **API Layer**: FastAPI router in `routers/news.py` exposes the `/api/news/sentiment` endpoint

### Frontend Components

1. **Schema**: `news.ts` defines TypeScript types with Zod validation
2. **Service**: `news.ts` service for API calls
3. **Hook**: `useNewsSentiment.ts` React Query hook with visibility awareness
4. **UI**: `NewsSentimentBoard.tsx` component for displaying clusters and metrics
5. **MSW**: Mock handler for local development and testing

## Setup

### 1. Install Required Python Packages

```bash
pip install -r requirements-news.txt
```

Required packages:
- feedparser: RSS feed parsing
- trafilatura: Web content extraction
- vaderSentiment: Sentiment analysis
- rapidfuzz: Fuzzy matching for clustering
- httpx: Async HTTP client
- pyyaml: YAML configuration parsing

### 2. Configure News Sources

Edit `trading_bot/news_sources.yaml` to add or modify news sources:

```yaml
sources:
  - name: "Reuters"
    domain: "reuters.com"
    category: "markets"
    bias_prior: 0.1
    reliability_prior: 0.85
    rss: "https://www.reuters.com/finance/rss"

  - name: "Bloomberg"
    domain: "bloomberg.com"
    category: "markets"
    bias_prior: 0.2
    reliability_prior: 0.85
    rss: "https://www.bloomberg.com/feeds/podcasts/etf-report.xml"
```

Each source requires:
- `name`: Display name of the source
- `domain`: Domain for URL canonicalization
- `category`: Primary category (markets, politics, tech, crypto, macro)
- `bias_prior`: Prior bias score (-1 to 1, negative = left, positive = right)
- `reliability_prior`: Prior reliability score (0 to 1)
- `rss`: URL of the RSS feed

### 3. Restart the Backend

```bash
python -m uvicorn trading_bot.api.app:app --reload --port 8000
```

## Using the News Sentiment API

### API Endpoint

```
GET /api/news/sentiment?category=markets&query=fed&per_source=5
```

Parameters:
- `category`: News category (markets, politics, tech, crypto, macro)
- `query`: Optional filter for headlines containing this text
- `per_source`: Maximum number of articles to process per source (default: 5)

Response:
```json
{
  "category": "markets",
  "clusters": [
    {
      "headline": "Fed signals potential rate cuts in coming months",
      "url": "https://example.com/fed-signals",
      "sentiment": 0.45,
      "partisan_spread": 0.2,
      "informational": 0.85,
      "finance": 0.9,
      "sources": ["Reuters", "Bloomberg", "CNBC"],
      "articles": [...]
    },
    ...
  ],
  "outlets": {
    "Reuters": {
      "count": 3,
      "avg_sent": 0.1,
      "avg_partisan": 0.05,
      "avg_info": 0.72
    },
    ...
  }
}
```

### React Hook

```tsx
import { useNewsSentiment } from '@/hooks/useNewsSentiment';

function NewsDashboard() {
  const { data, isLoading } = useNewsSentiment('markets');
  
  if (isLoading) return <div>Loading...</div>;
  
  return (
    <div>
      {data?.clusters.map(cluster => (
        <div key={cluster.url}>
          <h3>{cluster.headline}</h3>
          <p>Sentiment: {cluster.sentiment.toFixed(2)}</p>
        </div>
      ))}
    </div>
  );
}
```

## Metrics Explained

### Sentiment Score

Ranges from -1 (very negative) to 1 (very positive). Calculated using VADER sentiment analysis or a fallback lexicon-based approach.

### Partisan Score

Ranges from -1 (strongly left-leaning) to 1 (strongly right-leaning). Based on the presence of partisan terms and the source's bias prior.

### Partisan Spread

Measures the range of partisan scores within a cluster. Higher values indicate stories with divergent political framing.

### Information Score

Ranges from 0 (low information) to 1 (high information). Based on:
- Presence of numbers and statistics
- Proper nouns and technical terms
- Direct quotes
- Absence of hype words

### Finance Score

Ranges from 0 (not finance-related) to 1 (highly finance-related). Based on:
- Finance-specific terminology (Fed, CPI, earnings, etc.)
- Presence of ticker symbols
- Finance-specific entities

## Customization

### Adding New Categories

1. Add sources with the new category in `news_sources.yaml`
2. Update the `CATEGORIES` array in `NewsSentimentBoard.tsx`

### Adjusting Clustering Threshold

Modify the `threshold` parameter in the `_cluster` function in `news_service.py` (default: 85). Higher values require more similarity for clustering.

### Changing Polling Frequency

Adjust the `refetchInterval` in `useNewsSentiment.ts` (default: 300000ms / 5 minutes).

## Troubleshooting

### No Data Showing

1. Check that the backend is running and the `/api/news/sentiment` endpoint is accessible
2. Verify that the RSS feeds in `news_sources.yaml` are valid and accessible
3. Check browser console for errors
4. Try a different category

### Slow Performance

1. Reduce the `per_source` parameter to fetch fewer articles
2. Add more specific categories to your news sources
3. Increase the TTL cache duration in `news_service.py`

### Missing Required Libraries

If you see errors about missing libraries:

```bash
pip install feedparser trafilatura vaderSentiment rapidfuzz httpx pyyaml
```
