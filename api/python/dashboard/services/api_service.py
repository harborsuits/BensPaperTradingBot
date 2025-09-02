"""
API Service for Streamlit Dashboard

This service handles all API calls to control the trading bot system,
providing methods for pausing/resuming strategies, approving strategies,
closing positions, and other administrative actions.
"""
import requests
import logging
import json
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIService:
    """
    Service for interacting with the trading bot's control API.
    Handles all operations that modify the state of the trading bot.
    """
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the API service.
        
        Args:
            api_base_url: Base URL for the trading bot API
        """
        self.api_base_url = api_base_url
        
        # Check connection
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check connection to the API."""
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            if response.status_code == 200:
                logger.info("Successfully connected to trading bot API")
                return True
            else:
                logger.warning(f"API connection issue: Status code {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to API: {str(e)}")
            # For demo purposes, we'll continue even if API is not available
            return False
    
    def _api_post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint
            data: POST data
            
        Returns:
            Dict: Response data or error info
        """
        try:
            url = f"{self.api_base_url}{endpoint}"
            response = requests.post(url, json=data, timeout=10)
            
            if response.status_code in [200, 201, 202]:
                return response.json()
            else:
                error_message = f"API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        error_message = f"API error: {error_data['detail']}"
                except:
                    pass
                
                logger.error(error_message)
                return {"success": False, "error": error_message}
        
        except Exception as e:
            error_message = f"Failed to call API {endpoint}: {str(e)}"
            logger.error(error_message)
            return {"success": False, "error": error_message}
    
    # Strategy control methods
    def pause_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Pause a strategy.
        
        Args:
            strategy_id: ID of the strategy to pause
            
        Returns:
            Dict: Response with success status
        """
        data = {"strategy_id": strategy_id}
        result = self._api_post("/api/strategies/pause", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info(f"[DEMO] Pausing strategy {strategy_id}")
            return {"success": True, "message": f"Strategy {strategy_id} paused successfully"}
        
        return result
    
    def resume_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Resume a paused strategy.
        
        Args:
            strategy_id: ID of the strategy to resume
            
        Returns:
            Dict: Response with success status
        """
        data = {"strategy_id": strategy_id}
        result = self._api_post("/api/strategies/resume", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info(f"[DEMO] Resuming strategy {strategy_id}")
            return {"success": True, "message": f"Strategy {strategy_id} resumed successfully"}
        
        return result
    
    def close_strategy_positions(self, strategy_id: str) -> Dict[str, Any]:
        """
        Close all positions for a strategy.
        
        Args:
            strategy_id: ID of the strategy
            
        Returns:
            Dict: Response with success status
        """
        data = {"strategy_id": strategy_id}
        result = self._api_post("/api/strategies/close-positions", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info(f"[DEMO] Closing positions for strategy {strategy_id}")
            return {"success": True, "message": f"All positions for strategy {strategy_id} closed successfully"}
        
        return result
    
    # Approval workflow methods
    def request_strategy_approval(self, strategy_id: str) -> Dict[str, Any]:
        """
        Request approval to promote a strategy from paper to live.
        
        Args:
            strategy_id: ID of the strategy to request approval for
            
        Returns:
            Dict: Response with approval request status
        """
        data = {"strategy_id": strategy_id}
        result = self._api_post("/api/approval/request", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info(f"[DEMO] Requesting approval for strategy {strategy_id}")
            return {
                "success": True, 
                "message": f"Approval requested for strategy {strategy_id}",
                "request_id": "demo_req_123"
            }
        
        return result
    
    def approve_strategy(self, request_id: str, position_handling: str = "close") -> Dict[str, Any]:
        """
        Approve a strategy for live trading.
        
        Args:
            request_id: ID of the approval request
            position_handling: How to handle paper positions ('close', 'mirror', 'wait')
            
        Returns:
            Dict: Response with approval status
        """
        data = {
            "request_id": request_id,
            "approved": True,
            "position_handling": position_handling
        }
        result = self._api_post("/api/approval/approve", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info(f"[DEMO] Approving strategy request {request_id} with position handling: {position_handling}")
            return {
                "success": True, 
                "message": f"Strategy approved successfully",
                "strategy_id": "demo_strategy_id",
                "status": "LIVE"
            }
        
        return result
    
    def reject_strategy(self, request_id: str, reason: str = "") -> Dict[str, Any]:
        """
        Reject a strategy for live trading.
        
        Args:
            request_id: ID of the approval request
            reason: Reason for rejection
            
        Returns:
            Dict: Response with rejection status
        """
        data = {
            "request_id": request_id,
            "approved": False,
            "reason": reason
        }
        result = self._api_post("/api/approval/approve", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info(f"[DEMO] Rejecting strategy request {request_id} with reason: {reason}")
            return {
                "success": True, 
                "message": f"Strategy rejected successfully",
                "strategy_id": "demo_strategy_id",
                "status": "PAPER_TRADE"
            }
        
        return result
    
    # Emergency control methods
    def pause_all_strategies(self) -> Dict[str, Any]:
        """
        Emergency pause for all strategies.
        
        Returns:
            Dict: Response with pause status
        """
        result = self._api_post("/api/system/pause-all", {})
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info("[DEMO] Emergency pause for all strategies")
            return {
                "success": True, 
                "message": "All strategies paused successfully",
                "paused_count": 4
            }
        
        return result
    
    def resume_all_strategies(self) -> Dict[str, Any]:
        """
        Resume all paused strategies.
        
        Returns:
            Dict: Response with resume status
        """
        result = self._api_post("/api/system/resume-all", {})
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info("[DEMO] Resuming all strategies")
            return {
                "success": True, 
                "message": "All strategies resumed successfully",
                "resumed_count": 4
            }
        
        return result
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Emergency close all positions across all strategies.
        
        Returns:
            Dict: Response with close status
        """
        result = self._api_post("/api/system/close-all-positions", {})
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info("[DEMO] Emergency close all positions")
            return {
                "success": True, 
                "message": "All positions closed successfully",
                "closed_count": 8
            }
        
        return result
    
    def toggle_trading(self, enabled: bool) -> Dict[str, Any]:
        """
        Enable or disable all trading system-wide.
        
        Args:
            enabled: Whether to enable trading
            
        Returns:
            Dict: Response with toggle status
        """
        data = {"enabled": enabled}
        result = self._api_post("/api/system/toggle-trading", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            status = "enabled" if enabled else "disabled"
            logger.info(f"[DEMO] Trading {status}")
            return {
                "success": True, 
                "message": f"Trading {status} successfully"
            }
        
        return result
    
    # Position management methods
    def close_position(self, broker_id: str, symbol: str) -> Dict[str, Any]:
        """
        Close a specific position.
        
        Args:
            broker_id: ID of the broker
            symbol: Symbol of the position
            
        Returns:
            Dict: Response with close status
        """
        data = {
            "broker_id": broker_id,
            "symbol": symbol
        }
        result = self._api_post("/api/portfolio/close-position", data)
        
        # If API is unavailable in demo mode, simulate success
        if not result:
            logger.info(f"[DEMO] Closing position for {symbol} on {broker_id}")
            return {
                "success": True, 
                "message": f"Position for {symbol} closed successfully"
            }
        
        return result
