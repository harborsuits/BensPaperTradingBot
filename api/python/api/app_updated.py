#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Dashboard FastAPI Backend

This module serves as the main entry point for the FastAPI backend of the
trading dashboard. It handles API routes, WebSocket connections, and data access.
"""

import os
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routers
try:
    from trading_bot.api.coinbase_market_data import router as coinbase_router
    coinbase_available = True
except ImportError:
    coinbase_available = False
    logging.warning("Coinbase API module not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trading Bot API",
    description="API for the Trading Bot Dashboard",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"}
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "online",
        "name": "Trading Bot API",
        "version": "1.0.0"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# API version endpoint
@app.get("/version")
async def get_version():
    return {"version": "1.0.0"}

# Include routers
if coinbase_available:
    app.include_router(coinbase_router)
    logger.info("Coinbase API endpoints registered")

# Add any existing routers here...

# This is where you would include your existing routers, like:
# from trading_bot.api.data import router as data_router
# app.include_router(data_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
