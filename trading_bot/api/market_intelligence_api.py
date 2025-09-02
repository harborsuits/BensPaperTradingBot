"""
Market Intelligence API - FastAPI External Access Layer
This module provides a secure REST API for external access to the Market Intelligence system.
"""

import os
import sys
import json
import time
import logging
import datetime
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# FastAPI imports
from fastapi import FastAPI, Request, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our components
from trading_bot.market_intelligence_controller import get_market_intelligence_controller
from trading_bot.ml_pipeline.backtest_feedback_loop import get_backtest_executor
from trading_bot.triggers.market_intelligence_triggers import get_webhook_trigger

# Models for API requests and responses
class SymbolRequest(BaseModel):
    """Request model for symbol-related endpoints."""
    symbols: List[str] = Field(..., description="List of stock symbols")

class StrategyRequest(BaseModel):
    """Request model for strategy-related endpoints."""
    strategy_id: str = Field(..., description="Strategy ID")

class BacktestRequest(BaseModel):
    """Request model for backtest endpoints."""
    symbol: str = Field(..., description="Stock symbol")
    strategy: str = Field(..., description="Strategy ID")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional parameters")

class TriggerRequest(BaseModel):
    """Request model for trigger endpoints."""
    action: str = Field(..., description="Action to trigger")
    symbols: Optional[List[str]] = Field(None, description="List of stock symbols")
    force: Optional[bool] = Field(False, description="Force update")

# API configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Import typed settings
from trading_bot.config.typed_settings import APISettings, TradingBotSettings, load_config

# Load API keys from typed settings if available
api_settings = None
api_keys = ["test_key_1", "test_key_2"]  # Default fallback keys

try:
    config = load_config()
    api_settings = config.api
    if "market_intelligence_api" in api_settings.api_keys:
        api_keys = api_settings.api_keys["market_intelligence_api"]
    logger.info("Loaded API settings from typed config")
except Exception as e:
    # Try legacy config as fallback
    try:
        from config import API_KEYS
        api_keys = API_KEYS.get("market_intelligence_api", api_keys)
    except ImportError:
        pass  # Use default fallback keys
    logger.warning(f"Could not load typed API settings: {str(e)}. Using fallback.")
    api_settings = APISettings()

# Set up logging
logger = logging.getLogger("MarketIntelligenceAPI")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Create FastAPI app
app = FastAPI(
    title="Market Intelligence API",
    description="External access to the Market Intelligence system",
    version="1.0.0"
)

# Add CORS middleware with settings from config
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication dependency
async def get_api_key(
    api_key_header: str = Security(api_key_header),
):
    if api_key_header in api_keys:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
    )

# Get client IP middleware
@app.middleware("http")
async def add_client_ip(request: Request, call_next):
    # Get client IP
    client_host = request.client.host if request.client else "unknown"
    request.state.client_ip = client_host
    
    # Call next middleware or route handler
    response = await call_next(request)
    return response

# Rate limiting middleware
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    # Simple in-memory rate limiting
    # In production, use Redis or another distributed cache
    client_ip = request.state.client_ip
    
    # Check rate limit using settings from config
    if not hasattr(app, "rate_limit_store"):
        app.rate_limit_store = {}
    
    current_time = time.time()
    time_window = api_settings.rate_limit_period_seconds
    max_requests = api_settings.rate_limit_requests
    
    # Clean up old entries
    for ip in list(app.rate_limit_store.keys()):
        if current_time - app.rate_limit_store[ip]["timestamp"] > time_window:
            del app.rate_limit_store[ip]
    
    # Check current client
    if client_ip in app.rate_limit_store:
        entry = app.rate_limit_store[client_ip]
        if entry["count"] >= max_requests:
            if current_time - entry["timestamp"] <= time_window:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "retry_after": time_window}
                )
        
        # Update count
        entry["count"] += 1
    else:
        # New client
        app.rate_limit_store[client_ip] = {"count": 1, "timestamp": current_time}
    
    # Continue processing
    return await call_next(request)

# Routes
@app.get("/")
async def root(api_key: APIKey = Depends(get_api_key)):
    """Root endpoint."""
    return {
        "name": "Market Intelligence API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health():
    """Health check endpoint (no authentication required)."""
    return {
        "status": "ok",
        "timestamp": datetime.datetime.now().isoformat()
    }

@app.get("/market/context")
async def get_market_context(api_key: APIKey = Depends(get_api_key)):
    """Get the full market context."""
    try:
        controller = get_market_intelligence_controller()
        context = controller.market_context.get_market_context()
        
        # Remove any sensitive information
        if "config" in context:
            del context["config"]
        
        return context
    
    except Exception as e:
        logger.error(f"Error getting market context: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.get("/market/summary")
async def get_market_summary(api_key: APIKey = Depends(get_api_key)):
    """Get a summary of the current market conditions."""
    try:
        controller = get_market_intelligence_controller()
        summary = controller.get_market_summary()
        return summary
    
    except Exception as e:
        logger.error(f"Error getting market summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.get("/market/regime")
async def get_market_regime(api_key: APIKey = Depends(get_api_key)):
    """Get the current market regime."""
    try:
        controller = get_market_intelligence_controller()
        context = controller.market_context.get_market_context()
        regime = context.get("market", {}).get("regime", "unknown")
        
        return {
            "regime": regime,
            "timestamp": datetime.datetime.now().isoformat(),
            "confidence": context.get("market", {}).get("regime_confidence", 0.0)
        }
    
    except Exception as e:
        logger.error(f"Error getting market regime: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.post("/symbols/data")
async def get_symbol_data(
    request: SymbolRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """Get data for specific symbols."""
    try:
        controller = get_market_intelligence_controller()
        context = controller.market_context.get_market_context()
        
        # Extract data for requested symbols
        symbols_data = {}
        for symbol in request.symbols:
            if symbol in context.get("symbols", {}):
                symbols_data[symbol] = context["symbols"][symbol]
            else:
                symbols_data[symbol] = {"error": "Symbol not found in context"}
        
        return symbols_data
    
    except Exception as e:
        logger.error(f"Error getting symbol data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.get("/strategies/list")
async def list_strategies(api_key: APIKey = Depends(get_api_key)):
    """Get list of available strategies."""
    try:
        controller = get_market_intelligence_controller()
        strategies = controller.symbolranker.get_available_strategies()
        
        return {
            "strategies": strategies,
            "count": len(strategies)
        }
    
    except Exception as e:
        logger.error(f"Error listing strategies: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.get("/pairs/top")
async def get_top_pairs(
    limit: int = 10,
    strategy: Optional[str] = None,
    api_key: APIKey = Depends(get_api_key)
):
    """Get top symbol-strategy pairs."""
    try:
        controller = get_market_intelligence_controller()
        pairs = controller.get_top_symbol_strategy_pairs(limit=limit, strategy=strategy)
        
        return {
            "pairs": pairs,
            "count": len(pairs),
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error getting top pairs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.post("/backtest/run")
async def run_backtest(
    request: BacktestRequest,
    api_key: APIKey = Depends(get_api_key)
):
    """Run a backtest for a symbol-strategy pair."""
    try:
        # Get backtest executor
        executor = get_backtest_executor()
        
        # Run backtest
        result = executor.backtest_pair(
            request.symbol,
            request.strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            params=request.params
        )
        
        return {
            "symbol": request.symbol,
            "strategy": request.strategy,
            "backtest_result": result,
            "timestamp": datetime.datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.get("/backtest/performance")
async def get_backtest_performance(
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    limit: int = 10,
    api_key: APIKey = Depends(get_api_key)
):
    """Get backtest performance for symbol-strategy pairs."""
    try:
        # Get backtest feedback system
        from trading_bot.ml_pipeline.backtest_feedback_loop import get_backtest_feedback_system
        feedback_system = get_backtest_feedback_system()
        
        if symbol and strategy:
            # Get performance for specific pair
            performance = feedback_system.get_pair_performance(symbol, strategy)
            
            if not performance:
                return {
                    "error": "No performance data found for this pair",
                    "symbol": symbol,
                    "strategy": strategy
                }
            
            return {
                "symbol": symbol,
                "strategy": strategy,
                "performance": performance
            }
        else:
            # Get top performing pairs
            top_pairs = feedback_system.get_top_performing_pairs(limit=limit)
            
            return {
                "top_pairs": top_pairs,
                "count": len(top_pairs)
            }
    
    except Exception as e:
        logger.error(f"Error getting backtest performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

@app.post("/trigger")
async def trigger_action(
    request: TriggerRequest,
    request_obj: Request,
    api_key: APIKey = Depends(get_api_key)
):
    """Trigger an action in the Market Intelligence system."""
    try:
        # Get webhook trigger
        webhook_trigger = get_webhook_trigger()
        
        # Prepare request data
        request_data = {
            "action": request.action,
            "symbols": request.symbols,
            "force": request.force
        }
        
        # Prepare headers
        headers = {
            "X-API-Key": api_key
        }
        
        # Get client IP
        client_ip = request_obj.state.client_ip
        
        # Handle webhook
        result = webhook_trigger.handle_webhook(
            request_data=request_data,
            request_headers=headers,
            client_ip=client_ip
        )
        
        if result.get("status") == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Unknown error")
            )
        
        return result
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Error triggering action: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )

# Run the FastAPI app with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=api_settings.host, 
        port=api_settings.port,
        debug=api_settings.debug
    )
