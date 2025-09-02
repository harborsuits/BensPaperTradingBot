import pytest
import requests
import os
import json
import time
from datetime import datetime, timedelta

# Test configuration
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5000")
API_TOKEN = os.environ.get("API_TOKEN", "6165f902-b7a3-408c-9512-4e554225d825")  # Use your Alpaca key as test token

@pytest.fixture
def api_client():
    """Create an authenticated API client fixture"""
    class ApiClient:
        def __init__(self):
            self.headers = {
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            }
        
        def get(self, endpoint):
            return requests.get(f"{BASE_URL}{endpoint}", headers=self.headers)
        
        def post(self, endpoint, data):
            return requests.post(f"{BASE_URL}{endpoint}", json=data, headers=self.headers)
    
    return ApiClient()

def test_api_status(api_client):
    """Test API status endpoint"""
    response = api_client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert data["status"] == "success"
    assert "service" in data
    assert "version" in data
    assert "uptime" in data
    assert "trading_allowed" in data
    assert "paper_trading" in data
    assert "account" in data
    assert "market" in data

def test_account_summary(api_client):
    """Test account summary endpoint"""
    response = api_client.get("/account/summary")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "account" in data
    
    account = data["account"]
    assert "balance" in account
    assert "equity" in account
    assert isinstance(account["balance"], (int, float))
    assert isinstance(account["equity"], (int, float))

def test_open_trades(api_client):
    """Test open trades endpoint"""
    response = api_client.get("/open-trades")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "open_trades" in data
    assert "count" in data
    assert isinstance(data["open_trades"], list)
    
    # If any trades, validate structure
    if data["open_trades"]:
        trade = data["open_trades"][0]
        assert "symbol" in trade
        assert "entry_price" in trade
        assert "position_size" in trade
        assert "current_price" in trade
        assert "unrealized_pnl" in trade

def test_journal_metrics(api_client):
    """Test journal metrics endpoint"""
    response = api_client.get("/journal/metrics")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "metrics" in data
    assert "overall" in data["metrics"]
    assert "top_performers" in data["metrics"]
    assert "underperformers" in data["metrics"]
    assert "day_performance" in data["metrics"]

def test_chart_data(api_client):
    """Test chart data endpoint"""
    response = api_client.get("/dashboard/chart-data?type=equity_curve")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "chart_data" in data

def test_recommendations(api_client):
    """Test recommendations endpoint"""
    response = api_client.get("/journal/recommendations")
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    
    # If any recommendations, validate structure
    if data["recommendations"]:
        rec = data["recommendations"][0]
        assert "suggestion" in rec
        assert "focus" in rec
        assert "confidence" in rec

def test_webhook_entry_processing():
    """Test webhook entry signal processing"""
    # This should use the API client to send a test webhook entry signal
    # Then verify the trade was processed correctly
    
    client = requests.Session()
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # Test entry payload
    entry_payload = {
        "action": "entry",
        "symbol": "AAPL",
        "strategy": "test_strategy",
        "entry_price": 175.50,
        "stop_price": 173.50,
        "target_price": 180.00,
        "risk_percent": 0.5,
        "order_type": "market",
        "trade_id": f"test_{int(time.time())}"
    }
    
    # Only run this test if we're explicitly in test mode to avoid accidental orders
    if os.environ.get("TEST_MODE") == "true":
        response = client.post(f"{BASE_URL}/webhook", json=entry_payload, headers=headers)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] in ["success", "skipped"]
        if data["status"] == "success":
            assert "trade_id" in data
            assert "position_sizing" in data 