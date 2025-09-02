# BenBot Trading API Documentation

This document outlines all available API endpoints for the BenBot trading platform. These endpoints serve as the integration points for the React frontend.

## Authentication

```
POST /auth/login
```
- Request body: `{ "username": string, "password": string }`
- Response: `{ "access_token": string, "token_type": string }`

## WebSocket Connection

Connect to the WebSocket server at `/ws`. You can provide an authentication token:

```
ws://localhost:8000/ws?token={your_jwt_token}
```

### WebSocket Subscription

After connecting, subscribe to channels by sending:

```json
{
  "type": "subscription",
  "action": "subscribe",
  "channel": "trades"
}
```

Available channels:
- `trades`: Trade execution events
- `orders`: Order status updates
- `positions`: Position updates
- `context`: Market context updates (regime, sentiment, features)
- `strategies`: Strategy updates
- `logs`: System log messages
- `alerts`: System alerts
- `evotester`: EvoTester progress updates

## Market Context

### Complete Context

```
GET /api/context
```
- Returns all market context data including regime, sentiment, features, anomalies, and predictions

### Market Regime

```
GET /api/context/regime
```
- Returns the current market regime classification

### News & Sentiment

```
GET /api/context/news
```
- Query params: 
  - `limit`: Number of news items to return (default 20, max 50)

```
GET /api/context/news/symbol
```
- Query params:
  - `symbol`: Stock symbol
  - `limit`: Number of news items to return (default 10, max 20)

### Market Features

```
GET /api/context/features
```
- Returns feature vectors used by trading strategies

### Market Anomalies

```
GET /api/context/anomalies
```
- Query params:
  - `active`: Boolean to filter active anomalies only (default true)

### AI Predictions

```
GET /api/context/prediction
```
- Returns AI-generated market direction predictions

## Strategy Management

### Get All Strategies

```
GET /api/strategies
```
- Query params:
  - `status`: Optional filter by status ("active", "inactive", "all")

### Get Strategy Details

```
GET /api/strategies/{strategy_id}
```
- Returns detailed information about a specific strategy

### Update Strategy

```
PUT /api/strategies/{strategy_id}
```
- Request body: `{ "enabled": boolean, "weight": number, ... }`
- Updates a strategy's configuration

### Strategy Ranking

```
GET /api/strategies/ranking
```
- Returns strategies ranked by performance or suitability for current market

### Strategy Insights

```
GET /api/strategies/insights
```
- Returns insights about strategy performance in current market conditions

## Portfolio & Trading

### Portfolio Positions

```
GET /api/positions
```
- Query params:
  - `account`: Account type ("live" or "paper", default "live")

### Orders

```
GET /api/orders
```
- Query params:
  - `account`: Account type ("live" or "paper", default "live")
  - `status`: Filter by status ("open", "filled", "canceled", "all")

### Trades History

```
GET /api/trades
```
- Query params:
  - `account`: Account type ("live" or "paper", default "live")
  - `limit`: Number of trades to return (default 20)

### Place Order (Manual Trading)

```
POST /api/orders
```
- Request body: `{ "symbol": string, "side": string, "quantity": number, "order_type": string, "price": number, ... }`
- Creates a new order

### Cancel Order

```
DELETE /api/orders/{order_id}
```
- Cancels an existing order

## Trade Decisions

### Recent Decisions

```
GET /api/decisions/latest
```
- Returns the latest trade decisions and their scores

### Decision History

```
GET /api/decisions
```
- Query params:
  - `date`: Optional date filter (YYYY-MM-DD)
  - `limit`: Number of decisions to return (default 20)

## Logging & Notifications

### System Logs

```
GET /api/logs
```
- Query params:
  - `level`: Log level filter ("info", "warning", "error", "all")
  - `limit`: Number of logs to return (default 50)
  - `component`: Filter by system component

### System Alerts

```
GET /api/alerts
```
- Query params:
  - `limit`: Number of alerts to return (default 20)

### System Status

```
GET /api/system/status
```
- Returns system health metrics

## EvoTester

### Start EvoTester

```
POST /api/evotester/start
```
- Request body: EvoTester configuration
- Starts a new evolutionary testing session
- Returns session ID

### Stop EvoTester

```
POST /api/evotester/{session_id}/stop
```
- Stops an ongoing EvoTester session

### EvoTester Status

```
GET /api/evotester/{session_id}/status
```
- Returns the current status of an EvoTester session

### EvoTester Results

```
GET /api/evotester/{session_id}/result
```
- Returns results of a completed EvoTester session

### Recent EvoTester Sessions

```
GET /api/evotester/recent
```
- Query params:
  - `limit`: Number of sessions to return (default 10)

## Data Ingestion Status

```
GET /api/data/status
```
- Returns the status of data ingestion processes

## Event Bus Status

```
GET /api/events/status
```
- Returns the status of the event system

## System Configuration

```
GET /api/system/config
```
- Returns current system configuration

## Backtesting

```
POST /api/backtest/run
```
- Request body: Backtest configuration
- Runs a backtest with specified parameters

```
GET /api/backtest/{backtest_id}/result
```
- Returns results of a completed backtest

## Market Data

```
GET /api/market/prices
```
- Query params:
  - `symbols`: Comma-separated list of symbols
  - `timeframe`: Timeframe (e.g., "1d", "1h")
  - `limit`: Number of candles to return

## AI Assistant

```
POST /api/assistant/chat
```
- Request body: `{ "message": string, "context": object, "conversation_id": string }`
- Returns AI assistant response

## How to Use this API

1. First, authenticate by making a POST request to `/auth/login`
2. Use the returned JWT token in the Authorization header for subsequent requests:
   - HTTP: `Authorization: Bearer {token}`
   - WebSocket: Connect to `/ws?token={token}`
3. Subscribe to WebSocket channels for real-time updates
4. Use REST endpoints for initial data loading and actions
