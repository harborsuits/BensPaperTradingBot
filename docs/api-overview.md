# API Overview

The BensBot Trading System provides a comprehensive RESTful API to access all trading functionality, market data, and system management features.

## API Architecture

The API layer consists of multiple components:

1. **Main FastAPI App** - Core trading system API endpoints
2. **Market Intelligence API** - Market data, news, and analytics
3. **Authentication Service** - User authentication and authorization
4. **Backtesting API** - Strategy backtesting and optimization

All APIs are built on modern, high-performance frameworks (FastAPI and Flask) with proper type annotations, OpenAPI documentation, and rate limiting.

## API Settings

API configuration is managed through the typed settings system:

```python
class APISettings(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    enable_docs: bool = True
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    rate_limit_requests: int = 100
    rate_limit_period_seconds: int = 60
    jwt_secret: Optional[str] = None
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    api_keys: Dict[str, str] = Field(default_factory=dict)
    
    @validator('port')
    def validate_port(cls, v):
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v
```

## Core API Endpoints

### Trading Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/strategies` | GET | List available trading strategies |
| `/api/v1/strategies/{strategy_id}` | GET | Get strategy details |
| `/api/v1/strategies/{strategy_id}/activate` | POST | Activate a strategy |
| `/api/v1/strategies/{strategy_id}/deactivate` | POST | Deactivate a strategy |
| `/api/v1/orders` | GET | List orders |
| `/api/v1/orders` | POST | Place a new order |
| `/api/v1/orders/{order_id}` | GET | Get order details |
| `/api/v1/orders/{order_id}/cancel` | DELETE | Cancel an order |
| `/api/v1/positions` | GET | List current positions |
| `/api/v1/positions/{symbol}` | GET | Get position details |

### Account Information

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/account` | GET | Get account overview |
| `/api/v1/account/balance` | GET | Get account balance |
| `/api/v1/account/history` | GET | Get account history |
| `/api/v1/account/performance` | GET | Get account performance metrics |

### Market Data

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/market/quotes` | GET | Get quotes for symbols |
| `/api/v1/market/quotes/{symbol}` | GET | Get quote for a symbol |
| `/api/v1/market/history/{symbol}` | GET | Get price history for a symbol |
| `/api/v1/market/options/{symbol}` | GET | Get option chain for a symbol |
| `/api/v1/market/calendar` | GET | Get market calendar |
| `/api/v1/market/status` | GET | Get current market status |

### System Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/system/status` | GET | Get system status |
| `/api/v1/system/logs` | GET | Get system logs |
| `/api/v1/system/settings` | GET | Get system settings |
| `/api/v1/system/settings` | PUT | Update system settings |

## Market Intelligence API

The Market Intelligence API provides advanced market analysis:

### News and Sentiment

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/news` | GET | Get latest market news |
| `/api/v1/news/{symbol}` | GET | Get symbol-specific news |
| `/api/v1/sentiment/{symbol}` | GET | Get sentiment analysis for a symbol |
| `/api/v1/sentiment/news/{article_id}` | GET | Get sentiment for a specific news article |

### Analysis and Indicators

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analysis/{symbol}` | GET | Get technical analysis for a symbol |
| `/api/v1/indicators/{symbol}` | GET | Get technical indicators for a symbol |
| `/api/v1/patterns/{symbol}` | GET | Get pattern recognition for a symbol |
| `/api/v1/correlations/{symbol}` | GET | Get correlated assets |

## Backtesting API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/backtest` | POST | Run a backtest |
| `/api/v1/backtest/{backtest_id}` | GET | Get backtest results |
| `/api/v1/backtest/{backtest_id}/performance` | GET | Get detailed performance metrics |
| `/api/v1/backtest/{backtest_id}/trades` | GET | Get trades from backtest |
| `/api/v1/backtest/optimize` | POST | Run parameter optimization |

## Authentication

The API uses JWT (JSON Web Tokens) for authentication:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register a new user |
| `/api/auth/login` | POST | Log in and get tokens |
| `/api/auth/refresh` | POST | Refresh access token |
| `/api/auth/logout` | POST | Log out (revoke tokens) |

## API Security

The API implements several security measures:

1. **JWT Authentication** - Secure token-based auth
2. **Rate Limiting** - Prevent abuse and DDoS
3. **CORS Protection** - Control cross-origin requests
4. **Input Validation** - Prevent injection attacks
5. **Role-Based Access Control** - Limit actions by user role

## OpenAPI Documentation

API documentation is available at:

- Main API: `/docs` or `/redoc`
- Market Intelligence API: `/market-api/docs`
- Backtest API: `/backtest-api/docs`

## Client Usage Examples

### Python

```python
import requests

# Authentication
auth_response = requests.post(
    "http://localhost:8000/api/auth/login",
    json={"username": "user", "password": "password"}
)
token = auth_response.json()["access_token"]

# Get account balance
headers = {"Authorization": f"Bearer {token}"}
balance = requests.get(
    "http://localhost:8000/api/v1/account/balance",
    headers=headers
).json()

# Place an order
order = requests.post(
    "http://localhost:8000/api/v1/orders",
    headers=headers,
    json={
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10,
        "order_type": "market"
    }
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

// Get account balance
const balanceResponse = await fetch('http://localhost:8000/api/v1/account/balance', {
  headers: { 'Authorization': `Bearer ${access_token}` }
});
const balance = await balanceResponse.json();

// Place an order
const orderResponse = await fetch('http://localhost:8000/api/v1/orders', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${access_token}`
  },
  body: JSON.stringify({
    symbol: 'AAPL',
    side: 'buy',
    quantity: 10,
    order_type: 'market'
  })
});
const order = await orderResponse.json();
```
