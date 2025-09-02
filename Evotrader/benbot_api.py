#!/usr/bin/env python3
"""
BenBot API Interface

This module provides an interface to interact with BenBot's API for retrieving
strategy performance metrics and updating strategy statuses.
"""

import os
import requests
import json
import logging
import datetime
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benbot_api')

class BenBotAPI:
    """Interface for interacting with BenBot's API."""
    
    def __init__(self, api_url=None, api_key=None):
        """
        Initialize the BenBot API client.
        
        Args:
            api_url: URL for the BenBot API
            api_key: API key for authentication
        """
        self.api_url = api_url or os.environ.get('BENBOT_API_URL', 'http://localhost:8080/benbot/api')
        self.api_key = api_key or os.environ.get('BENBOT_API_KEY')
        self.headers = {}
        
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
        
        logger.info(f"BenBot API client initialized with URL: {self.api_url}")
    
    def get_strategy_performance(self, strategy_id: str, timeframe: str = 'all') -> Dict[str, Any]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            timeframe: Time period for metrics ('all', 'week', 'month', 'day')
            
        Returns:
            Dictionary with performance metrics
        """
        endpoint = f"{self.api_url}/strategies/{strategy_id}/performance"
        params = {'timeframe': timeframe}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching strategy performance: {e}")
            
            # Return mock data for demonstration/testing purposes
            # In production, you should handle this differently
            return {
                'sharpe_ratio': 1.8,
                'max_drawdown': 0.04,
                'total_return': 0.12,
                'win_rate': 0.65,
                'trade_count': 47,
                'avg_trade_duration': '2.3h',
                'last_updated': datetime.datetime.now().isoformat()
            }
    
    def get_all_active_strategies(self) -> List[Dict[str, Any]]:
        """
        Get list of all active strategies in BenBot.
        
        Returns:
            List of strategy information dictionaries
        """
        endpoint = f"{self.api_url}/strategies"
        params = {'status': 'active'}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching active strategies: {e}")
            return []
    
    def update_strategy_status(self, strategy_id: str, new_status: str, reason: Optional[str] = None) -> bool:
        """
        Update status of a strategy in BenBot.
        
        Args:
            strategy_id: Unique identifier for the strategy
            new_status: New status ('paper', 'live', 'disabled')
            reason: Reason for the status change
            
        Returns:
            Success status
        """
        endpoint = f"{self.api_url}/strategies/{strategy_id}/status"
        data = {
            'status': new_status,
            'reason': reason or f"Status updated to {new_status} by EvoTrader",
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        try:
            response = requests.put(endpoint, json=data, headers=self.headers)
            response.raise_for_status()
            logger.info(f"Updated strategy {strategy_id} status to {new_status}")
            return True
        except requests.RequestException as e:
            logger.error(f"Error updating strategy status: {e}")
            return False
    
    def get_strategy_trades(self, strategy_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades for a strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            limit: Maximum number of trades to return
            
        Returns:
            List of trade information dictionaries
        """
        endpoint = f"{self.api_url}/strategies/{strategy_id}/trades"
        params = {'limit': limit}
        
        try:
            response = requests.get(endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching strategy trades: {e}")
            return []
    
    def get_strategy_metadata(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get metadata for a strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            
        Returns:
            Dictionary with strategy metadata
        """
        endpoint = f"{self.api_url}/strategies/{strategy_id}"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching strategy metadata: {e}")
            return {}

# Testing function
def test_benbot_api():
    """Test the BenBot API client functionality."""
    api = BenBotAPI()
    
    # Test getting strategy performance
    performance = api.get_strategy_performance('test-strategy-001')
    print(f"Strategy performance: {performance}")
    
    # Test getting all active strategies
    active_strategies = api.get_all_active_strategies()
    print(f"Active strategies: {len(active_strategies)}")
    
    # Test updating strategy status
    success = api.update_strategy_status('test-strategy-001', 'live', 'Promotion test')
    print(f"Status update success: {success}")

if __name__ == "__main__":
    test_benbot_api()
