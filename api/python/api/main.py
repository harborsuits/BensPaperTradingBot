"""
Main FastAPI application for the trading bot.
Integrates all API endpoints including strategy approval and monitoring.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.staging_environment import create_staging_environment
from trading_bot.api.strategy_approval_endpoints import router as approval_router
from trading_bot.api.strategy_monitoring_endpoints import router as monitoring_router

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("trading_bot_api")

# Create FastAPI app
app = FastAPI(
    title="Trading Bot API",
    description="API for managing trading strategies, including paper-to-live transitions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers for strategy approval and monitoring
app.include_router(approval_router)
app.include_router(monitoring_router)

# Environment configuration
ENV_CONFIG = {
    "environment": "staging",  # Options: development, staging, production
    "staging_config_path": "./config/staging_config.json"
}

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Trading Bot API")
    
    # Initialize service registry if needed
    service_registry = ServiceRegistry.get_instance()
    
    # Activate staging environment if in staging mode
    if ENV_CONFIG.get("environment") == "staging":
        try:
            logger.info("Activating staging environment")
            staging_env = create_staging_environment(ENV_CONFIG.get("staging_config_path"))
            logger.info("Staging environment activated successfully")
        except Exception as e:
            logger.error(f"Error activating staging environment: {str(e)}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Trading Bot API")
    
    # Get staging environment if active
    service_registry = ServiceRegistry.get_instance()
    staging_env = service_registry.get_service("staging_environment")
    
    if staging_env:
        logger.info("Deactivating staging environment")
        staging_env.deactivate()

# Root endpoint
@app.get("/", tags=["status"])
async def root() -> Dict[str, Any]:
    """Get API status."""
    return {
        "status": "online",
        "environment": ENV_CONFIG.get("environment"),
        "staging_active": is_staging_active()
    }

# Helper function to check if staging is active
def is_staging_active() -> bool:
    """Check if staging environment is active."""
    service_registry = ServiceRegistry.get_instance()
    staging_env = service_registry.get_service("staging_environment")
    
    if staging_env:
        status = staging_env.get_staging_status()
        return status.get("active", False)
    
    return False
