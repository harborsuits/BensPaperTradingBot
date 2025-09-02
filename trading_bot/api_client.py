"""
API Client

A robust client for communicating with the trading bot API.
Features error handling, retry logic, request tracking, and performance monitoring.
"""

import os
import json
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

class APIEndpoint(str, Enum):
    """Enumeration of available API endpoints."""
    # Trading endpoints
    WEBHOOK = "webhook"
    OPEN_TRADES = "open-trades"
    EXIT_TRADE = "exit_trade/{trade_id}"
    
    # Account endpoints
    UPDATE_ACCOUNT_SIZE = "update-account-size"
    
    # Journal endpoints
    JOURNAL_TRADES = "journal/trades"
    JOURNAL_METRICS = "journal/metrics"
    JOURNAL_TRADE = "journal/trade/{trade_id}"
    JOURNAL_EQUITY_CURVE = "journal/equity_curve"
    JOURNAL_PATTERNS = "journal/patterns"
    JOURNAL_RECOMMENDATIONS = "journal/recommendations"
    
    # Psychological assessment endpoints
    PSYCH_ASSESSMENT = "psychological-assessment"
    CHECKLIST_QUESTIONS = "checklist-questions"
    TRADE_PATTERN_ANALYSIS = "trade-pattern-analysis"
    RECENT_STATISTICS = "recent-statistics"
    
    # Cooldown and strategy endpoints
    COOLDOWN_STATUS = "cooldown_status"
    STRATEGY_STATUS = "strategy_status"
    STRATEGY_STATUS_SPECIFIC = "strategy_status/{strategy_name}"
    UPDATE_MARKET = "update_market"
    
    # Optimizer endpoints
    OPTIMIZERS_STRATEGY_ALLOCATION = "optimizers/strategy-allocation"
    OPTIMIZERS_UPDATE_STRATEGY_PERFORMANCE = "optimizers/update-strategy-performance"
    OPTIMIZERS_CONFIDENCE_SCORE = "optimizers/confidence-score"
    OPTIMIZERS_GENERATE_REPORT = "optimizers/generate-report"
    OPTIMIZERS_TRADE_REPLAY = "optimizers/trade-replay/{trade_id}"
    
    # Report endpoints
    REPORTS_GENERATE_DAILY = "reports/generate-daily"
    REPORTS_GENERATE_WEEKLY = "reports/generate-weekly"
    REPORTS_GET_LATEST = "reports/get-latest"
    
    # Replay endpoints
    REPLAY_LOAD_TRADES = "replay/load-trades"
    REPLAY_CURRENT_TRADE = "replay/current-trade"
    REPLAY_NAVIGATE = "replay/navigate"
    REPLAY_TRADE_CONTEXT = "replay/trade-context"
    REPLAY_SIMILAR_TRADES = "replay/similar-trades"

@dataclass
class APIResponse:
    """Container for API response data and metadata."""
    endpoint: str
    success: bool
    status_code: int
    data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    latency_ms: float
    timestamp: datetime
    request_params: Optional[Dict[str, Any]] = None
    request_data: Optional[Dict[str, Any]] = None

class APIClient:
    """
    A robust client for interacting with the trading bot API.
    
    Features:
    - Automatic retries for transient errors
    - Comprehensive error handling
    - Request tracking and metrics
    - Authentication support
    - Configurable timeouts and retries
    """
    
    def __init__(self, 
                base_url: str = None,
                auth_token: str = None,
                timeout: int = 10,
                max_retries: int = 3,
                retry_backoff_factor: float = 0.5,
                retry_status_forcelist: List[int] = None,
                log_level: int = logging.INFO):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            auth_token: Authentication token (if required)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_backoff_factor: Backoff factor for retries
            retry_status_forcelist: List of status codes to retry
            log_level: Logging level
        """
        self.base_url = base_url or os.environ.get("BOT_API_URL", "http://localhost:5000")
        self.auth_token = auth_token
        self.timeout = timeout
        
        # Setup logger
        self.logger = logging.getLogger("APIClient")
        self.logger.setLevel(log_level)
        
        # Request history
        self.request_history = []
        self.max_history_size = 100
        self.error_count = 0
        self.request_count = 0
        
        # Performance metrics
        self.performance_metrics = {
            "latency_avg_ms": 0,
            "success_rate": 0,
            "error_rate_by_endpoint": {},
            "latency_by_endpoint": {}
        }
        
        # Configure session with retries
        self.session = requests.Session()
        
        # Setup retry strategy
        retry_status_forcelist = retry_status_forcelist or [408, 429, 500, 502, 503, 504]
        retries = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=retry_status_forcelist,
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        
        # Mount the adapter to the session
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        
        # Set authentication header if provided
        if self.auth_token:
            self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
        
        self.logger.info(f"Initialized API client for {self.base_url}")
    
    def _make_request(self,
                    method: str,
                    endpoint: str,
                    params: Dict[str, Any] = None,
                    data: Dict[str, Any] = None,
                    timeout: int = None,
                    placeholders: Dict[str, str] = None) -> APIResponse:
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint to call
            params: Query parameters
            data: Request body data (for POST/PUT)
            timeout: Request timeout in seconds (overrides default)
            placeholders: Placeholder values for endpoint formatting
            
        Returns:
            APIResponse object with response data and metadata
        """
        # Format endpoint with placeholders if provided
        formatted_endpoint = endpoint
        if placeholders:
            formatted_endpoint = endpoint.format(**placeholders)
        
        # Construct full URL
        url = f"{self.base_url}/{formatted_endpoint}"
        
        # Initialize response data
        self.request_count += 1
        start_time = time.time()
        request_timestamp = datetime.now()
        
        # Initialize API response with default values
        response = APIResponse(
            endpoint=endpoint,
            success=False,
            status_code=None,
            data=None,
            error_message=None,
            latency_ms=0,
            timestamp=request_timestamp,
            request_params=params,
            request_data=data
        )
        
        try:
            # Set content type for POST/PUT requests
            headers = {"Content-Type": "application/json"}
            
            # Make the request
            if method.upper() == "GET":
                r = self.session.get(url, params=params, timeout=timeout or self.timeout, headers=headers)
            elif method.upper() == "POST":
                r = self.session.post(url, params=params, json=data, timeout=timeout or self.timeout, headers=headers)
            elif method.upper() == "PUT":
                r = self.session.put(url, params=params, json=data, timeout=timeout or self.timeout, headers=headers)
            elif method.upper() == "DELETE":
                r = self.session.delete(url, params=params, json=data, timeout=timeout or self.timeout, headers=headers)
            else:
                self.error_count += 1
                response.error_message = f"Unsupported HTTP method: {method}"
                return response
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            response.status_code = r.status_code
            response.latency_ms = latency_ms
            
            # Check if request was successful
            r.raise_for_status()
            
            # Try to parse JSON response
            try:
                response_data = r.json()
                response.data = response_data
                response.success = True
            except json.JSONDecodeError:
                # Not a JSON response
                response.error_message = "Response is not valid JSON"
                response.success = False
                self.error_count += 1
            
        except requests.exceptions.HTTPError as e:
            # HTTP error (status codes 4xx, 5xx)
            response.error_message = f"HTTP Error: {e}"
            self.error_count += 1
            self.logger.error(f"HTTP Error ({endpoint}): {e}")
            
            # Try to parse error response
            try:
                response.data = r.json()
            except (json.JSONDecodeError, UnboundLocalError):
                pass
            
        except requests.exceptions.ConnectionError as e:
            # Connection error
            response.error_message = f"Connection Error: {e}"
            self.error_count += 1
            self.logger.error(f"Connection Error ({endpoint}): {e}")
            
        except requests.exceptions.Timeout as e:
            # Timeout
            response.error_message = f"Timeout Error: {e}"
            self.error_count += 1
            self.logger.error(f"Timeout Error ({endpoint}): {e}")
            
        except requests.exceptions.RequestException as e:
            # Other request errors
            response.error_message = f"Request Error: {e}"
            self.error_count += 1
            self.logger.error(f"Request Error ({endpoint}): {e}")
        
        # Update request history
        self.request_history.append(response)
        if len(self.request_history) > self.max_history_size:
            self.request_history = self.request_history[-self.max_history_size:]
        
        # Update performance metrics
        self._update_performance_metrics(response)
        
        return response
    
    def _update_performance_metrics(self, response: APIResponse) -> None:
        """
        Update performance metrics with the latest request data.
        
        Args:
            response: APIResponse object from the latest request
        """
        # Update success rate
        self.performance_metrics["success_rate"] = (self.request_count - self.error_count) / self.request_count if self.request_count > 0 else 0
        
        # Update average latency
        latency_sum = sum(r.latency_ms for r in self.request_history)
        self.performance_metrics["latency_avg_ms"] = latency_sum / len(self.request_history) if self.request_history else 0
        
        # Update endpoint-specific metrics
        endpoint = response.endpoint
        
        # Latency by endpoint
        if endpoint not in self.performance_metrics["latency_by_endpoint"]:
            self.performance_metrics["latency_by_endpoint"][endpoint] = []
        
        # Keep only the last 20 latency values
        latency_values = self.performance_metrics["latency_by_endpoint"][endpoint]
        latency_values.append(response.latency_ms)
        if len(latency_values) > 20:
            latency_values = latency_values[-20:]
        self.performance_metrics["latency_by_endpoint"][endpoint] = latency_values
        
        # Error rate by endpoint
        if endpoint not in self.performance_metrics["error_rate_by_endpoint"]:
            self.performance_metrics["error_rate_by_endpoint"][endpoint] = {"total": 0, "errors": 0}
        
        endpoint_metrics = self.performance_metrics["error_rate_by_endpoint"][endpoint]
        endpoint_metrics["total"] += 1
        if not response.success:
            endpoint_metrics["errors"] += 1
        
        endpoint_metrics["rate"] = endpoint_metrics["errors"] / endpoint_metrics["total"] if endpoint_metrics["total"] > 0 else 0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for API requests.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate additional metrics
        metrics = self.performance_metrics.copy()
        
        # Calculate average latency by endpoint
        avg_latency_by_endpoint = {}
        for endpoint, latency_values in metrics["latency_by_endpoint"].items():
            if latency_values:
                avg_latency_by_endpoint[endpoint] = sum(latency_values) / len(latency_values)
            else:
                avg_latency_by_endpoint[endpoint] = 0
        
        metrics["avg_latency_by_endpoint"] = avg_latency_by_endpoint
        
        # Add request counts
        metrics["total_requests"] = self.request_count
        metrics["total_errors"] = self.error_count
        
        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()
        
        return metrics
    
    def clear_history(self) -> None:
        """Clear request history and reset metrics."""
        self.request_history = []
        self.error_count = 0
        self.request_count = 0
        
        # Reset performance metrics
        self.performance_metrics = {
            "latency_avg_ms": 0,
            "success_rate": 0,
            "error_rate_by_endpoint": {},
            "latency_by_endpoint": {}
        }
        
        self.logger.info("Cleared request history and reset metrics")
    
    def get_request_history(self, 
                          limit: int = 10,
                          endpoint_filter: str = None,
                          success_filter: Optional[bool] = None) -> List[APIResponse]:
        """
        Get recent request history with optional filtering.
        
        Args:
            limit: Maximum number of responses to return
            endpoint_filter: Filter by endpoint
            success_filter: Filter by success status
            
        Returns:
            List of APIResponse objects
        """
        filtered_history = self.request_history
        
        # Apply filters
        if endpoint_filter:
            filtered_history = [r for r in filtered_history if r.endpoint == endpoint_filter]
        
        if success_filter is not None:
            filtered_history = [r for r in filtered_history if r.success == success_filter]
        
        # Sort by timestamp (most recent first) and limit
        sorted_history = sorted(filtered_history, key=lambda x: x.timestamp, reverse=True)
        
        return sorted_history[:limit]
    
    def get(self, 
           endpoint: str, 
           params: Dict[str, Any] = None,
           placeholders: Dict[str, str] = None) -> APIResponse:
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            placeholders: Placeholder values for endpoint formatting
            
        Returns:
            APIResponse object with response data and metadata
        """
        return self._make_request("GET", endpoint, params=params, placeholders=placeholders)
    
    def post(self, 
            endpoint: str, 
            data: Dict[str, Any] = None,
            params: Dict[str, Any] = None,
            placeholders: Dict[str, str] = None) -> APIResponse:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint to call
            data: Request body data
            params: Query parameters
            placeholders: Placeholder values for endpoint formatting
            
        Returns:
            APIResponse object with response data and metadata
        """
        return self._make_request("POST", endpoint, params=params, data=data, placeholders=placeholders)
    
    def put(self, 
           endpoint: str, 
           data: Dict[str, Any] = None,
           params: Dict[str, Any] = None,
           placeholders: Dict[str, str] = None) -> APIResponse:
        """
        Make a PUT request to the API.
        
        Args:
            endpoint: API endpoint to call
            data: Request body data
            params: Query parameters
            placeholders: Placeholder values for endpoint formatting
            
        Returns:
            APIResponse object with response data and metadata
        """
        return self._make_request("PUT", endpoint, params=params, data=data, placeholders=placeholders)
    
    def delete(self, 
              endpoint: str, 
              data: Dict[str, Any] = None,
              params: Dict[str, Any] = None,
              placeholders: Dict[str, str] = None) -> APIResponse:
        """
        Make a DELETE request to the API.
        
        Args:
            endpoint: API endpoint to call
            data: Request body data
            params: Query parameters
            placeholders: Placeholder values for endpoint formatting
            
        Returns:
            APIResponse object with response data and metadata
        """
        return self._make_request("DELETE", endpoint, params=params, data=data, placeholders=placeholders)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the API.
        
        Returns:
            Dictionary with health check results
        """
        # Endpoints to check
        test_endpoints = [
            APIEndpoint.OPEN_TRADES,
            APIEndpoint.RECENT_STATISTICS,
            APIEndpoint.STRATEGY_STATUS
        ]
        
        results = {}
        overall_status = "healthy"
        
        # Test each endpoint
        for endpoint in test_endpoints:
            start_time = time.time()
            response = self.get(endpoint)
            latency = (time.time() - start_time) * 1000
            
            status = "healthy" if response.success else "unhealthy"
            if status == "unhealthy":
                overall_status = "unhealthy"
            
            results[endpoint] = {
                "status": status,
                "latency_ms": latency,
                "status_code": response.status_code,
                "error": response.error_message
            }
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "endpoints": results
        }
    
    def update_auth_token(self, token: str) -> None:
        """
        Update the authentication token.
        
        Args:
            token: New authentication token
        """
        self.auth_token = token
        # Update session headers
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.logger.info("Updated authentication token")
    
    # Specific API endpoint methods
    
    def get_open_trades(self) -> APIResponse:
        """
        Get list of open trades.
        
        Returns:
            APIResponse with open trades data
        """
        return self.get(APIEndpoint.OPEN_TRADES)
    
    def submit_trade_signal(self, trade_data: Dict[str, Any]) -> APIResponse:
        """
        Submit a trade signal to the webhook.
        
        Args:
            trade_data: Trade signal data
            
        Returns:
            APIResponse with submission result
        """
        return self.post(APIEndpoint.WEBHOOK, data=trade_data)
    
    def exit_trade(self, trade_id: str, exit_price: Optional[float] = None) -> APIResponse:
        """
        Exit an open trade.
        
        Args:
            trade_id: ID of the trade to exit
            exit_price: Optional exit price (default: current market price)
            
        Returns:
            APIResponse with exit result
        """
        data = {}
        if exit_price is not None:
            data["exit_price"] = exit_price
        
        return self.post(
            APIEndpoint.EXIT_TRADE,
            data=data,
            placeholders={"trade_id": trade_id}
        )
    
    def update_account_size(self, account_size: float) -> APIResponse:
        """
        Update the account size.
        
        Args:
            account_size: New account size
            
        Returns:
            APIResponse with update result
        """
        return self.post(
            APIEndpoint.UPDATE_ACCOUNT_SIZE,
            data={"account_size": account_size}
        )
    
    def get_journal_trades(self, 
                         strategy: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> APIResponse:
        """
        Get trade journal entries.
        
        Args:
            strategy: Optional strategy filter
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            
        Returns:
            APIResponse with trade journal entries
        """
        params = {}
        if strategy:
            params["strategy"] = strategy
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return self.get(APIEndpoint.JOURNAL_TRADES, params=params)
    
    def get_journal_metrics(self,
                          strategy: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> APIResponse:
        """
        Get performance metrics from the journal.
        
        Args:
            strategy: Optional strategy filter
            start_date: Optional start date filter (ISO format)
            end_date: Optional end date filter (ISO format)
            
        Returns:
            APIResponse with performance metrics
        """
        params = {}
        if strategy:
            params["strategy"] = strategy
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        return self.get(APIEndpoint.JOURNAL_METRICS, params=params)
    
    def get_trade_recommendations(self) -> APIResponse:
        """
        Get trade recommendations based on journal analysis.
        
        Returns:
            APIResponse with trade recommendations
        """
        return self.get(APIEndpoint.JOURNAL_RECOMMENDATIONS)
    
    def get_psychological_assessment(self, assessment_data: Dict[str, Any]) -> APIResponse:
        """
        Get psychological risk assessment.
        
        Args:
            assessment_data: Assessment data including symbol and optional responses
            
        Returns:
            APIResponse with psychological assessment
        """
        return self.post(APIEndpoint.PSYCH_ASSESSMENT, data=assessment_data)
    
    def get_strategy_status(self) -> APIResponse:
        """
        Get status of all trading strategies.
        
        Returns:
            APIResponse with strategy status
        """
        return self.get(APIEndpoint.STRATEGY_STATUS)
    
    def generate_daily_report(self, report_data: Dict[str, Any]) -> APIResponse:
        """
        Generate a daily trading report.
        
        Args:
            report_data: Report generation data
            
        Returns:
            APIResponse with report generation result
        """
        return self.post(APIEndpoint.REPORTS_GENERATE_DAILY, data=report_data)
    
    def generate_weekly_report(self, report_data: Dict[str, Any]) -> APIResponse:
        """
        Generate a weekly trading report.
        
        Args:
            report_data: Report generation data
            
        Returns:
            APIResponse with report generation result
        """
        return self.post(APIEndpoint.REPORTS_GENERATE_WEEKLY, data=report_data)

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create API client
    client = APIClient(base_url="http://localhost:5000")
    
    # Example: Get open trades
    response = client.get_open_trades()
    if response.success:
        print(f"Open trades: {len(response.data.get('open_trades', []))}")
    else:
        print(f"Error getting open trades: {response.error_message}")
    
    # Example: Get performance metrics
    response = client.get_journal_metrics()
    if response.success:
        metrics = response.data.get('metrics', {})
        print(f"Win rate: {metrics.get('win_rate', 0) * 100:.1f}%")
    else:
        print(f"Error getting metrics: {response.error_message}")
    
    # Get API client performance metrics
    performance = client.get_performance_metrics()
    print(f"API client performance: {performance['success_rate'] * 100:.1f}% success rate, " +
          f"{performance['latency_avg_ms']:.1f}ms average latency") 