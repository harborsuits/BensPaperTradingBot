from locust import HttpUser, task, between, events
import random
import json
import time
import os

class TradingBotUser(HttpUser):
    """
    Simulates a user interacting with the trading bot API
    For performance and load testing
    """
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Setup before starting tests"""
        # Get test API token from environment variable or use default
        self.api_token = os.environ.get("API_TOKEN", "6165f902-b7a3-408c-9512-4e554225d825")  # Your test Alpaca key
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
    
    @task(5)
    def get_api_status(self):
        """Test API status endpoint - highest frequency task"""
        self.client.get("/api/status", headers=self.headers, name="API Status")
    
    @task(4)
    def get_account_summary(self):
        """Test account summary endpoint"""
        self.client.get("/account/summary", headers=self.headers, name="Account Summary")
    
    @task(4)
    def get_open_trades(self):
        """Test open trades endpoint"""
        self.client.get("/open-trades", headers=self.headers, name="Open Trades")
    
    @task(3)
    def get_journal_metrics(self):
        """Test metrics endpoint"""
        periods = ["day", "week", "month", "year", "all"]
        period = random.choice(periods)
        self.client.get(f"/journal/metrics?period={period}", 
                       headers=self.headers, 
                       name="Journal Metrics")
    
    @task(3)
    def get_recommendations(self):
        """Test recommendations endpoint"""
        confidence = random.uniform(0.5, 0.9)
        self.client.get(f"/journal/recommendations?min_confidence={confidence}", 
                       headers=self.headers, 
                       name="Trade Recommendations")
    
    @task(2)
    def get_chart_data(self):
        """Test chart data endpoint"""
        chart_types = ["equity_curve", "win_loss", "strategy_performance", 
                      "pnl_distribution", "day_of_week", "drawdown"]
        chart_type = random.choice(chart_types)
        days = random.choice([7, 30, 90, 180, 365])
        
        self.client.get(f"/dashboard/chart-data?type={chart_type}&days={days}", 
                       headers=self.headers, 
                       name=f"Chart Data - {chart_type}")
    
    @task(1)
    def submit_webhook(self):
        """Simulate webhook call - lowest frequency task due to impact"""
        # Only do this if explicitly in test mode
        if self.environment.parsed_options.test_mode:
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            strategies = ["trend_following", "mean_reversion", "breakout", "momentum"]
            
            # Create test webhook entry
            entry_payload = {
                "action": "entry",
                "symbol": random.choice(symbols),
                "strategy": random.choice(strategies),
                "entry_price": random.uniform(100, 200),
                "stop_price": random.uniform(95, 99),
                "target_price": random.uniform(201, 210),
                "risk_percent": random.uniform(0.2, 1.0),
                "order_type": "limit",  # Use limit to prevent actual execution
                "trade_id": f"test_{int(time.time())}_{random.randint(1000, 9999)}"
            }
            
            # Submit webhook
            self.client.post("/webhook", 
                           json=entry_payload, 
                           headers=self.headers, 
                           name="Webhook - Entry")

    def on_stop(self):
        """Cleanup after tests"""
        pass

# Custom command line arguments for Locust
@events.init_command_line_parser.add_listener
def on_init_command_line_parser(parser):
    parser.add_argument(
        '--test-mode',
        dest='test_mode',
        action='store_true',
        default=False,
        help="Enable test mode to allow webhook submissions"
    )

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Starting stress test")
    if environment.parsed_options.test_mode:
        print("TEST MODE ENABLED - Will send webhook submissions")
    else:
        print("Test mode disabled - Will NOT send webhook submissions")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Stress test completed") 