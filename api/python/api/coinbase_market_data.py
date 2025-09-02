#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coinbase Market Data API Endpoints

This module provides FastAPI endpoints to access Coinbase market data
for the trading dashboard.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

# Import Coinbase broker
from trading_bot.brokers.coinbase_cloud_broker import CoinbaseCloudBroker
from trading_bot.brokers.coinbase_cloud_client import CoinbaseCloudBrokerageClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/coinbase", tags=["coinbase"])

# Models for responses
class TickerData(BaseModel):
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    high_24h: float
    low_24h: float
    timestamp: str

class HistoricalCandle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class OrderBookEntry(BaseModel):
    price: float
    size: float

class OrderBook(BaseModel):
    bids: List[OrderBookEntry]
    asks: List[OrderBookEntry]
    timestamp: str

# Initialize Coinbase broker (lazy initialization)
_broker = None

def get_broker():
    """Get or initialize the Coinbase broker instance"""
    global _broker
    if _broker is None:
        try:
            # BenbotReal credentials
            api_key_name = "organizations/1781cc1d-57ec-4e92-aa78-7a403caa11c5/apiKeys/8ef865bf-2217-47ec-9fa9-237c0637d335"
            private_key = """-----BEGIN EC PRIVATE KEY-----
MHcCAQEEIMs6tEqZbbC6ziEaK/MxCl/YBLJ1/uL0AybaAdvWJfr4oAoGCCqGSM49
AwEHoUQDQgAEdZJ/L8mrFCNNKLQRo3r52YRm4oAWlKc341TYsymeyXiG6DGPdFEX
WHezb1iJMTCwBBpsJCxwYnfKieCZrbiJig==
-----END EC PRIVATE KEY-----"""
            
            # Initialize broker in read-only mode for safety
            broker = CoinbaseCloudBroker(api_key_name=api_key_name, private_key=private_key, sandbox=False)
            
            # Wrap in the brokerage client
            _broker = CoinbaseCloudBrokerageClient(broker)
            logger.info("Coinbase broker initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Coinbase broker: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Coinbase broker: {str(e)}")
    
    return _broker

@router.get("/products", response_model=List[str])
async def get_products(broker = Depends(get_broker)):
    """Get available trading products from Coinbase"""
    try:
        success, products = broker.get_available_products()
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to get products from Coinbase")
            
        # Extract product IDs
        product_ids = [p.get('id') for p in products if 'id' in p]
        return product_ids
    except Exception as e:
        logger.error(f"Error getting products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ticker/{symbol}", response_model=TickerData)
async def get_ticker(symbol: str, broker = Depends(get_broker)):
    """Get ticker data for a specific symbol"""
    try:
        # Get ticker data
        success, ticker = broker.get_ticker(symbol)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to get ticker for {symbol}")
        
        # Get 24h stats for additional information
        success, stats = broker.get_product_stats(symbol)
        
        if not success:
            stats = {}
        
        # Combine ticker and stats
        response = {
            "symbol": symbol,
            "price": float(ticker.get("price", 0)),
            "volume_24h": float(stats.get("volume", 0)),
            "change_24h": float(stats.get("open", 0)) - float(ticker.get("price", 0)),
            "high_24h": float(stats.get("high", 0)),
            "low_24h": float(stats.get("low", 0)),
            "timestamp": ticker.get("time", datetime.now().isoformat())
        }
        
        return response
    except Exception as e:
        logger.error(f"Error getting ticker for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/candles/{symbol}", response_model=List[HistoricalCandle])
async def get_candles(
    symbol: str, 
    timeframe: str = Query("1h", description="Timeframe (1m, 5m, 15m, 1h, 6h, 1d)"),
    limit: int = Query(100, ge=1, le=300, description="Number of candles to return"),
    broker = Depends(get_broker)
):
    """Get historical candles for a specific symbol"""
    try:
        # Convert timeframe to seconds
        timeframe_seconds = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "6h": 21600,
            "1d": 86400
        }.get(timeframe, 3600)
        
        # Calculate start and end times
        end = datetime.now()
        start = end - timedelta(seconds=timeframe_seconds * limit)
        
        # Get candles
        candles = broker.get_historical_candles(symbol, timeframe_seconds, start, end)
        
        if not candles or (isinstance(candles, tuple) and not candles[0]):
            raise HTTPException(status_code=500, detail=f"Failed to get candles for {symbol}")
            
        # Format candles
        formatted_candles = []
        for candle in candles:
            if isinstance(candle, list) and len(candle) >= 6:
                # Format: [timestamp, low, high, open, close, volume]
                formatted_candles.append({
                    "timestamp": datetime.fromtimestamp(candle[0]).isoformat(),
                    "open": float(candle[3]),
                    "high": float(candle[2]),
                    "low": float(candle[1]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })
            elif isinstance(candle, dict):
                # Format: {timestamp, open, high, low, close, volume}
                formatted_candles.append({
                    "timestamp": candle.get("timestamp", datetime.now().isoformat()),
                    "open": float(candle.get("open", 0)),
                    "high": float(candle.get("high", 0)),
                    "low": float(candle.get("low", 0)),
                    "close": float(candle.get("close", 0)),
                    "volume": float(candle.get("volume", 0))
                })
                
        return formatted_candles
    except Exception as e:
        logger.error(f"Error getting candles for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orderbook/{symbol}", response_model=OrderBook)
async def get_orderbook(
    symbol: str,
    level: int = Query(2, ge=1, le=3, description="Order book detail level (1-3)"),
    broker = Depends(get_broker)
):
    """Get order book for a specific symbol"""
    try:
        # Get order book
        success, orderbook = broker.get_order_book(symbol, level)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to get order book for {symbol}")
            
        # Format bids and asks
        bids = [{"price": float(bid[0]), "size": float(bid[1])} for bid in orderbook.get("bids", [])]
        asks = [{"price": float(ask[0]), "size": float(ask[1])} for ask in orderbook.get("asks", [])]
        
        return {
            "bids": bids,
            "asks": asks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting order book for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-summary", response_model=Dict[str, Any])
async def get_market_summary(broker = Depends(get_broker)):
    """Get a summary of the crypto market"""
    try:
        # Get major cryptocurrencies
        major_symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "BNB-USD"]
        summary = {}
        
        for symbol in major_symbols:
            try:
                # Get ticker data
                success, ticker = broker.get_ticker(symbol)
                
                if success:
                    # Get 24h stats
                    _, stats = broker.get_product_stats(symbol)
                    
                    # Add to summary
                    summary[symbol] = {
                        "price": float(ticker.get("price", 0)),
                        "volume_24h": float(stats.get("volume", 0)) if stats else 0,
                        "change_24h_pct": ((float(ticker.get("price", 0)) / float(stats.get("open", ticker.get("price", 0)))) - 1) * 100 if stats else 0,
                        "last_updated": ticker.get("time", datetime.now().isoformat())
                    }
            except Exception as e:
                logger.warning(f"Error getting data for {symbol}: {str(e)}")
                continue
        
        return {
            "coins": summary,
            "market_status": "open",  # Crypto market is always open
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting market summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this to your FastAPI app:
# from trading_bot.api.coinbase_market_data import router as coinbase_router
# app.include_router(coinbase_router)
