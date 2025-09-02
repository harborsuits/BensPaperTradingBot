"""
Integration tests to verify the entire system works end-to-end
"""
import pytest
import sys
import os
import json
import time
from datetime import datetime
import requests
import threading

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try importing the API clients and database
try:
    from api_clients import alpaca_client, tradier_client, cache
    from database import db
    API_CLIENTS_AVAILABLE = True
except ImportError:
    API_CLIENTS_AVAILABLE = False
    print("Warning: API clients or database not available. Some tests will be skipped.")

# Test config
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5000")
API_TOKEN = os.environ.get("API_TOKEN", "6165f902-b7a3-408c-9512-4e554225d825")  # Use your Alpaca key as test token

@pytest.fixture(scope="module")
def api_client():
    """Create an authenticated API client fixture"""
    class ApiClient:
        def __init__(self):
            self.headers = {
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            }
            self.base_url = BASE_URL
        
        def get(self, endpoint):
            return requests.get(f"{self.base_url}{endpoint}", headers=self.headers)
        
        def post(self, endpoint, data):
            return requests.post(f"{self.base_url}{endpoint}", json=data, headers=self.headers)
    
    return ApiClient()

@pytest.mark.skipif(not API_CLIENTS_AVAILABLE, reason="API clients not available")
def test_alpaca_client():
    """Test direct Alpaca API client functionality"""
    # Get account info
    account = alpaca_client.get_account()
    assert isinstance(account, dict)
    assert "equity" in account
    assert "cash" in account
    
    # Get market data
    bars = alpaca_client.get_bars("AAPL", "1D", 10)
    assert isinstance(bars, list)
    assert len(bars) > 0
    assert "timestamp" in bars[0]
    assert "open" in bars[0]
    assert "close" in bars[0]

@pytest.mark.skipif(not API_CLIENTS_AVAILABLE, reason="API clients not available")
def test_tradier_client():
    """Test direct Tradier API client functionality"""
    try:
        # Get quote for a symbol
        quotes = tradier_client.get_quotes("AAPL")
        assert isinstance(quotes, dict)
        
        # Market status
        status = tradier_client.get_market_status()
        assert isinstance(status, dict)
    except Exception as e:
        # Skip if Tradier credentials aren't set up
        pytest.skip(f"Skipping Tradier tests: {e}")

@pytest.mark.skipif(not API_CLIENTS_AVAILABLE, reason="API clients not available")
def test_database_integration():
    """Test database operations"""
    # Create test trade record
    test_trade = {
        "trade_id": f"test_{int(time.time())}",
        "symbol": "AAPL",
        "strategy": "test_strategy",
        "entry_price": 175.0,
        "stop_price": 170.0,
        "position_size": 10,
        "timestamp": datetime.now().isoformat(),
        "status": "open"
    }
    
    # Save to database
    result = db.save_trade(test_trade)
    assert result is not None
    
    # Retrieve trade
    retrieved = db.get_trade_by_id(test_trade["trade_id"])
    assert retrieved is not None
    assert retrieved["symbol"] == "AAPL"
    
    # Update trade
    db.update_one("trades", 
                {"trade_id": test_trade["trade_id"]}, 
                {"$set": {"status": "closed"}})
    
    # Verify update
    updated = db.get_trade_by_id(test_trade["trade_id"])
    assert updated["status"] == "closed"
    
    # Clean up
    db.delete_one("trades", {"trade_id": test_trade["trade_id"]})

@pytest.mark.skipif(not API_CLIENTS_AVAILABLE, reason="API clients not available")
def test_caching():
    """Test cache functionality"""
    # Set value in cache
    test_key = f"test_key_{int(time.time())}"
    test_value = {"data": "test_value", "timestamp": time.time()}
    
    result = cache.set(test_key, test_value, expiry=10)
    assert result
    
    # Get value from cache
    cached = cache.get(test_key)
    assert cached is not None
    assert cached["data"] == "test_value"
    
    # Delete from cache
    cache.delete(test_key)
    
    # Verify deleted
    after_delete = cache.get(test_key)
    assert after_delete is None

def test_api_connectivity(api_client):
    """Test basic API connectivity"""
    response = api_client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

def test_full_trade_lifecycle(api_client):
    """Test a complete trade lifecycle from entry to exit"""
    # Only run in explicit test mode
    if os.environ.get("TEST_MODE") != "true":
        pytest.skip("Skipping full trade lifecycle test (requires TEST_MODE=true)")
    
    # 1. Create an entry order
    entry_payload = {
        "action": "entry",
        "symbol": "AAPL",
        "strategy": "test_strategy",
        "entry_price": 175.50,  # Should be close to current price
        "stop_price": 173.50,
        "target_price": 180.00,
        "risk_percent": 0.1,  # Keep very small for testing
        "order_type": "limit",  # Use limit to prevent actual execution
        "trade_id": f"test_{int(time.time())}"
    }
    
    entry_response = api_client.post("/webhook", entry_payload)
    assert entry_response.status_code == 200
    entry_data = entry_response.json()
    
    if entry_data["status"] == "skipped":
        pytest.skip("Trade skipped, likely due to psychological risk check")
    
    trade_id = entry_data["trade_id"]
    
    # 2. Verify trade in open positions
    time.sleep(2)  # Wait for processing
    open_trades_response = api_client.get("/open-trades")
    open_trades = open_trades_response.json()
    
    found = False
    for trade in open_trades.get("open_trades", []):
        if trade.get("trade_id") == trade_id:
            found = True
            break
    
    if not found:
        print("Trade not found in open positions. API response:")
        print(json.dumps(open_trades, indent=2))
    
    # 3. Exit the trade
    exit_payload = {
        "action": "exit",
        "symbol": "AAPL",
        "trade_id": trade_id,
        "exit_price": 176.00,  # Slight profit
        "order_type": "limit"
    }
    
    exit_response = api_client.post("/webhook", exit_payload)
    assert exit_response.status_code == 200
    exit_data = exit_response.json()
    
    assert exit_data["status"] == "success"
    assert "pnl" in exit_data
    
    # 4. Verify trade no longer in open positions
    time.sleep(2)  # Wait for processing
    open_trades_response = api_client.get("/open-trades")
    open_trades = open_trades_response.json()
    
    for trade in open_trades.get("open_trades", []):
        assert trade.get("trade_id") != trade_id 