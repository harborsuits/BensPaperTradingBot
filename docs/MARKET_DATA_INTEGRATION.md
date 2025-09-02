# Market Data Integration Guide

This document explains how to set up and use the market data integration with Alpaca API in the trading dashboard.

## Overview

We've integrated Alpaca Trading API to provide real-time market data and paper trading capabilities to your dashboard. This integration includes:

1. A Python service wrapper for Alpaca API (`alpaca_service.py`)
2. FastAPI endpoints for batched quotes and historical bars
3. React hooks for consuming the live data with proper rate limiting
4. Components for displaying quotes, charts, and executing paper trades

## Features

- **Batched Quotes**: Get multiple symbols in a single API request
- **Rate Limiting**: Stays under API rate limits (180 requests/minute)
- **TTL Caching**: Short-lived caching to prevent duplicate requests
- **Graceful Error Handling**: Returns stale data when possible instead of errors
- **Visibility-Aware Polling**: Pauses API requests when tab is in background
- **Circuit Breaker**: Prevents API hammering during outages

## Setup

### 1. Configure Alpaca API Keys

Create a `.env` file at the root of your project with your Alpaca API keys:

```
ALPACA_KEY_ID=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_IS_PAPER=true
```

You can get API keys by signing up for a free account at [Alpaca](https://app.alpaca.markets/signup).

### 2. Install Dependencies

```bash
pip install httpx aiolimiter cachetools
```

### 3. Configure React Dashboard

Create a `.env.local` file in the `new-trading-dashboard` directory:

```
VITE_API_URL=http://localhost:8000
VITE_USE_MOCK=false
```

Alternatively, you can run the provided script:

```bash
./setup_market_data.sh
```

### 4. Start the Backend

```bash
python -m uvicorn trading_bot.api.app:app --reload --port 8000
```

### 5. Start the Frontend

```bash
cd new-trading-dashboard
npm run dev
```

## Using Market Data Components

### 1. Fetch a Single Quote

```tsx
import { useQuote } from '../hooks/useQuotes';

function StockPrice({ symbol }) {
  const { quote, isLoading } = useQuote(symbol);
  
  if (isLoading) return <p>Loading...</p>;
  
  // Get mid price (or ask/bid if mid not available)
  const price = quote?.quote?.ap && quote?.quote?.bp
    ? (quote.quote.ap + quote.quote.bp) / 2
    : quote?.quote?.ap || quote?.quote?.bp;
    
  return <p>{price ? `$${price.toFixed(2)}` : '—'}</p>;
}
```

### 2. Fetch Multiple Quotes Efficiently

```tsx
import { usePrices } from '../hooks/useQuotes';

function WatchList({ symbols }) {
  const { prices, isLoading } = usePrices(symbols);
  
  return (
    <div>
      {symbols.map(symbol => (
        <div key={symbol}>
          <span>{symbol}</span>
          <span>{prices[symbol] ? `$${prices[symbol].toFixed(2)}` : '—'}</span>
        </div>
      ))}
    </div>
  );
}
```

### 3. Display a Price Chart

```tsx
import { MarketChart } from '@/components/market/MarketChart';

function StockChart({ symbol }) {
  return (
    <MarketChart
      symbol={symbol}
      timeframe="1Day"
      height={300}
      title={`${symbol} Chart`}
    />
  );
}
```

### 4. Paper Trading

```tsx
import { PaperTradeButton } from '@/components/market/PaperTradeButton';

function TradingPanel({ symbol }) {
  return <PaperTradeButton symbol={symbol} />;
}
```

## API Endpoints

### Quotes Endpoint

```
GET /api/quotes?symbols=AAPL,MSFT,GOOGL
```

Response:
```json
{
  "quotes": {
    "AAPL": {
      "symbol": "AAPL",
      "quote": {
        "ap": 185.25,
        "bp": 185.20,
        "as": 100,
        "bs": 200,
        "t": "2023-11-15T12:34:56Z"
      },
      "stale": false
    },
    "MSFT": { ... },
    "GOOGL": { ... }
  }
}
```

### Bars Endpoint

```
GET /api/bars?symbol=AAPL&timeframe=1Day&limit=30
```

Response:
```json
{
  "bars": [
    {
      "t": "2023-11-15T00:00:00Z",
      "o": 185.20,
      "h": 187.50,
      "l": 184.75,
      "c": 186.30,
      "v": 45000000
    },
    ...
  ],
  "symbol": "AAPL",
  "stale": false
}
```

## Advanced Configuration

### Environment Variables

- `ALPACA_KEY_ID`: Your Alpaca API key
- `ALPACA_SECRET_KEY`: Your Alpaca API secret
- `ALPACA_IS_PAPER`: Set to "true" for paper trading (default), "false" for live trading
- `ALPACA_DATA_BASE`: Override the data API URL (default: "https://data.alpaca.markets/v2")
- `ALPACA_PAPER_BASE`: Override the paper API URL (default: "https://paper-api.alpaca.markets/v2")
- `ALPACA_LIVE_BASE`: Override the live API URL (default: "https://api.alpaca.markets/v2")

### Rate Limiting

The server implements rate limiting to stay under Alpaca's limits:
- 180 requests per minute for quotes
- TTL caching of 0.9 seconds for quotes
- TTL caching of 5 seconds for bars

### Circuit Breaker

The server includes a circuit breaker pattern to prevent API hammering during outages:
- Opens after 5 consecutive failures
- Half-opens after 5 minutes to test if API is back
- Closes on successful API call

## Troubleshooting

### No Data Showing

1. Check your Alpaca API keys in `.env`
2. Verify the backend is running (`python -m uvicorn trading_bot.api.app:app --port 8000`)
3. Check the browser console for errors
4. Try enabling mock data temporarily (`VITE_USE_MOCK=true`)

### Rate Limit Errors

If you see rate limit errors:
1. Reduce the number of symbols you're requesting
2. Increase the polling interval in the hooks
3. Check if multiple components are requesting the same data

### API Connection Issues

If the API connection is unstable:
1. Check your internet connection
2. Verify Alpaca services are operational
3. The circuit breaker may be open - wait 5 minutes for it to reset
