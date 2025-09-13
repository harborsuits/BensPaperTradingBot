#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot Daemon Service

This module provides a daemon service that:
1. Continuously runs the AutonomousOrchestrator
2. Provides REST API endpoints for UI interaction
3. Exposes health check endpoints for monitoring
4. Handles real-time data via WebSockets (with REST fallback)
"""

import os
import json
import time
import logging
import signal
import argparse
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# FastAPI for REST endpoints
import fastapi
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Redis for pub/sub and caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import trading bot components
from trading_bot.orchestration.autonomous_orchestrator import AutonomousOrchestrator
from trading_bot.brokers.risk_enforced_executor import RiskEnforcedExecutor
from trading_bot.brokers.tradier_client_enhanced import TradierClient
from trading_bot.risk.risk_manager import RiskManager
from trading_bot.data.real_time_data_manager import RealTimeDataManager
from trading_bot.notification_manager.notification_manager import NotificationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_daemon.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_daemon")

# Create FastAPI app
app = FastAPI(title="Trading Bot API", description="REST API for Trading Bot Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global component references
orchestrator = None
data_manager = None
trade_executor = None
risk_manager = None
notification_manager = None
redis_client = None

# Status tracking
daemon_status = {
    "status": "initializing",
    "last_step_time": None,
    "uptime_seconds": 0,
    "start_time": None,
    "errors": [],
    "active_strategies": []
}

# Message queue for inter-thread communication
message_queue = queue.Queue()

# Request models
class OrderRequest(BaseModel):
    symbol: str
    side: str
    quantity: Optional[int] = None
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    risk_pct: Optional[float] = None
    order_type: Optional[str] = "market"
    duration: Optional[str] = "day"
    strategy_name: Optional[str] = "manual"
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class BacktestRequest(BaseModel):
    strategy_id: str
    start_date: str
    end_date: str
    symbols: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)

# Helper functions
def load_config():
    """Load configuration from environment variables or config files"""
    # Look for config files in descending priority
    config_paths = [
        os.environ.get("TRADING_CONFIG_PATH"),
        "./config.yaml",
        "./config.json",
        "./trading_bot/config/config.yaml",
        "./trading_bot/config/config.json"
    ]
    
    for path in config_paths:
        if path and os.path.exists(path):
            logger.info(f"Loading config from {path}")
            if path.endswith(".yaml") or path.endswith(".yml"):
                import yaml
                with open(path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                with open(path, 'r') as f:
                    return json.load(f)
    
    # Fall back to environment variables if no config file found
    logger.warning("No config file found, using environment variables")
    return {
        "tradier": {
            "api_key": os.environ.get("TRADIER_API_KEY"),
            "account_id": os.environ.get("TRADIER_ACCOUNT_ID"),
            "sandbox": os.environ.get("TRADIER_SANDBOX", "true").lower() == "true"
        },
        "redis": {
            "host": os.environ.get("REDIS_HOST", "localhost"),
            "port": int(os.environ.get("REDIS_PORT", "6379")),
            "password": os.environ.get("REDIS_PASSWORD", "")
        },
        "orchestrator": {
            "step_interval_seconds": int(os.environ.get("STEP_INTERVAL_SECONDS", "30"))
        }
    }

def initialize_redis(config):
    """Initialize Redis connection if available"""
    global redis_client, REDIS_AVAILABLE
    
    if not REDIS_AVAILABLE:
        logger.warning("Redis not available, caching and pub/sub will be disabled")
        return None
    
    try:
        redis_config = config.get("redis", {})
        client = redis.Redis(
            host=redis_config.get("host", "localhost"),
            port=redis_config.get("port", 6379),
            password=redis_config.get("password", ""),
            socket_timeout=5,
            retry_on_timeout=True
        )
        client.ping()  # Test connection
        logger.info("Redis connection established")
        return client
    except Exception as e:
        logger.error(f"Redis connection failed: {str(e)}")
        REDIS_AVAILABLE = False
        return None

def initialize_components(config):
    """Initialize all trading system components"""
    global orchestrator, data_manager, trade_executor, risk_manager, notification_manager
    
    try:
        # Initialize tradier client
        tradier_config = config.get("tradier", {})
        tradier_client = TradierClient(
            api_key=tradier_config.get("api_key"),
            account_id=tradier_config.get("account_id"),
            sandbox=tradier_config.get("sandbox", True)
        )
        
        # Initialize risk manager
        risk_config = config.get("risk", {})
        risk_manager = RiskManager(
            max_position_pct=risk_config.get("max_position_pct", 0.05),
            max_risk_pct=risk_config.get("max_risk_pct", 0.01),
            max_correlation=risk_config.get("max_correlation", 0.7),
            max_sector_exposure=risk_config.get("max_sector_exposure", 0.25),
            max_open_trades=risk_config.get("max_open_trades", 5)
        )
        
        # Initialize trade executor with risk enforcement
        executor_config = config.get("executor", {})
        trade_executor = RiskEnforcedExecutor(
            tradier_client=tradier_client,
            risk_manager=risk_manager,
            max_position_pct=executor_config.get("max_position_pct", 0.05),
            max_risk_pct=executor_config.get("max_risk_pct", 0.01),
            order_type=executor_config.get("default_order_type", "market"),
            order_duration=executor_config.get("default_duration", "day")
        )
        
        # Initialize real-time data manager
        data_config = config.get("data", {})
        data_manager = RealTimeDataManager(
            tradier_client=tradier_client,
            redis_client=redis_client if REDIS_AVAILABLE else None,
            use_websocket=data_config.get("use_websocket", False),
            default_cache_seconds=data_config.get("default_cache_seconds", 10)
        )
        
        # Initialize notification manager
        notification_config = config.get("notifications", {})
        notification_manager = NotificationManager(
            telegram_token=notification_config.get("telegram_token"),
            telegram_chat_id=notification_config.get("telegram_chat_id"),
            slack_webhook=notification_config.get("slack_webhook"),
            email_config=notification_config.get("email")
        )
        
        # Initialize orchestrator
        orchestrator_config = config.get("orchestrator", {})
        orchestrator = AutonomousOrchestrator(
            trade_executor=trade_executor,
            data_manager=data_manager,
            risk_manager=risk_manager,
            notification_manager=notification_manager,
            config=orchestrator_config
        )
        
        logger.info("All components initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Component initialization failed: {str(e)}")
        daemon_status["errors"].append(f"Initialization failed: {str(e)}")
        daemon_status["status"] = "error"
        return False

def update_status_in_redis():
    """Update trading status in Redis for UI consumption"""
    if not redis_client:
        return
        
    try:
        status_data = {
            **daemon_status,
            "portfolio": get_portfolio_summary(),
            "timestamp": datetime.now().isoformat()
        }
        redis_client.set("trading:status", json.dumps(status_data))
        redis_client.publish("trading:status_update", json.dumps({
            "type": "status_update",
            "data": status_data
        }))
    except Exception as e:
        logger.error(f"Failed to update Redis status: {str(e)}")

def get_portfolio_summary():
    """Get current portfolio summary for status updates"""
    if not trade_executor or not trade_executor.client:
        return {}
        
    try:
        return trade_executor.client.get_account_summary()
    except Exception as e:
        logger.error(f"Failed to get portfolio summary: {str(e)}")
        return {"error": str(e)}

def orchestrator_thread_func():
    """Main orchestrator thread function"""
    global daemon_status
    
    logger.info("Orchestrator thread starting")
    daemon_status["status"] = "running"
    daemon_status["start_time"] = datetime.now().isoformat()
    
    # Get step interval from config, default to 30 seconds
    config = load_config()
    step_interval = config.get("orchestrator", {}).get("step_interval_seconds", 30)
    
    while daemon_status["status"] != "stopping":
        try:
            # Run one step of the orchestrator
            logger.info("Running orchestrator step")
            orchestrator.step()
            daemon_status["last_step_time"] = datetime.now().isoformat()
            daemon_status["active_strategies"] = orchestrator.get_active_strategies()
            daemon_status["uptime_seconds"] = (datetime.now() - datetime.fromisoformat(daemon_status["start_time"])).total_seconds()
            
            # Update status in Redis
            update_status_in_redis()
            
            # Check message queue for commands
            process_message_queue()
            
        except Exception as e:
            error_msg = f"Error in orchestrator step: {str(e)}"
            logger.error(error_msg)
            daemon_status["errors"].append(error_msg)
            
        # Sleep until next step
        time.sleep(step_interval)
    
    logger.info("Orchestrator thread stopping")
    daemon_status["status"] = "stopped"

def process_message_queue():
    """Process messages from the message queue"""
    try:
        # Non-blocking queue check
        while not message_queue.empty():
            message = message_queue.get_nowait()
            logger.info(f"Processing message: {message.get('type')}")
            
            if message.get("type") == "execute_trade":
                handle_trade_execution(message.get("data"))
            elif message.get("type") == "run_backtest":
                handle_backtest(message.get("data"))
                
            message_queue.task_done()
    except queue.Empty:
        pass
    except Exception as e:
        logger.error(f"Error processing message queue: {str(e)}")

def handle_trade_execution(trade_data):
    """Execute a trade based on message data"""
    if not trade_executor:
        logger.error("Trade executor not initialized")
        return {"error": "Trade executor not initialized"}
    
    try:
        result = trade_executor.execute_trade(**trade_data)
        
        # If trade was successful, notify
        if result.get("status") != "rejected":
            notification_manager.send_notification(
                f"Trade executed: {trade_data.get('side')} {trade_data.get('quantity', '?')} {trade_data.get('symbol')}",
                level="info",
                metadata=result
            )
        return result
    except Exception as e:
        error_msg = f"Trade execution failed: {str(e)}"
        logger.error(error_msg)
        notification_manager.send_notification(
            f"Trade failed: {trade_data.get('symbol')} - {str(e)}",
            level="error"
        )
        return {"error": error_msg}

def handle_backtest(backtest_data):
    """Run a backtest based on message data"""
    if not orchestrator:
        logger.error("Orchestrator not initialized")
        return {"error": "Orchestrator not initialized"}
    
    try:
        # This would typically delegate to a separate backtest module
        result = orchestrator.run_backtest(**backtest_data)
        
        # Store results in Redis if available
        if redis_client:
            redis_client.set(
                f"backtest:{backtest_data.get('strategy_id')}:{datetime.now().isoformat()}",
                json.dumps(result)
            )
            redis_client.publish("trading:backtest_complete", json.dumps({
                "type": "backtest_complete",
                "data": {
                    "strategy_id": backtest_data.get("strategy_id"),
                    "summary": result.get("summary")
                }
            }))
        
        return result
    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def shutdown_handler():
    """Handle graceful shutdown"""
    global daemon_status
    
    logger.info("Shutdown requested, stopping services...")
    daemon_status["status"] = "stopping"
    
    # Allow orchestrator to complete current step (max 5 seconds)
    shutdown_wait = 5
    for _ in range(shutdown_wait):
        if daemon_status["status"] == "stopped":
            break
        time.sleep(1)
    
    # Clean up resources
    if orchestrator:
        orchestrator.shutdown()
    
    if redis_client:
        redis_client.close()
    
    logger.info("Shutdown complete")

# API routes
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint for monitoring"""
    if daemon_status["status"] in ["running", "initializing"]:
        return {"status": "healthy", "details": daemon_status}
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "details": daemon_status}
        )

@app.get("/api/status")
def get_status():
    """Get current trading system status"""
    return {
        **daemon_status,
        "portfolio": get_portfolio_summary(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/portfolio")
def get_portfolio():
    """Get current portfolio data"""
    if not trade_executor or not trade_executor.client:
        raise HTTPException(status_code=503, detail="Trade executor not available")
    
    try:
        return trade_executor.client.get_account_summary()
    except Exception as e:
        logger.error(f"Failed to get portfolio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/positions")
def get_positions():
    """Get current positions"""
    if not trade_executor:
        raise HTTPException(status_code=503, detail="Trade executor not available")
    
    try:
        return trade_executor.get_open_trades()
    except Exception as e:
        logger.error(f"Failed to get positions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/execute_trade")
def execute_trade(order: OrderRequest, background_tasks: BackgroundTasks):
    """Execute a trade via API"""
    if not trade_executor:
        raise HTTPException(status_code=503, detail="Trade executor not available")
    
    # Queue the trade for execution in the orchestrator thread
    message_queue.put({
        "type": "execute_trade",
        "data": order.dict()
    })
    
    return {
        "status": "trade_queued",
        "message": f"Trade for {order.symbol} queued for execution",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/run_backtest")
def run_backtest(backtest: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest via API"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    
    # Queue the backtest
    message_queue.put({
        "type": "run_backtest",
        "data": backtest.dict()
    })
    
    return {
        "status": "backtest_queued",
        "message": f"Backtest for strategy {backtest.strategy_id} queued",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/market_status")
def get_market_status():
    """Get current market status"""
    if not trade_executor or not trade_executor.client:
        raise HTTPException(status_code=503, detail="Trade executor not available")
    
    try:
        return {
            "market_open": trade_executor.client.is_market_open(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get market status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for the trading daemon"""
    parser = argparse.ArgumentParser(description='Trading Bot Daemon Service')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind API server')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind API server')
    args = parser.parse_args()
    
    # Set config path in environment if provided
    if args.config:
        os.environ["TRADING_CONFIG_PATH"] = args.config
    
    # Load configuration
    config = load_config()
    
    # Initialize Redis
    global redis_client
    redis_client = initialize_redis(config)
    
    # Initialize components
    if not initialize_components(config):
        logger.error("Failed to initialize components, exiting")
        return 1
    
    # Start orchestrator thread
    thread = threading.Thread(target=orchestrator_thread_func, daemon=True)
    thread.start()
    
    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda sig, frame: shutdown_handler())
    
    # Start API server using uvicorn
    import uvicorn
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
