#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtester API Server

Simple Flask server that provides endpoints for the backtesting pipeline.
This serves as a backend for the backtesting page in the trading dashboard.
"""

import os
import json
import logging
import random
import datetime
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BacktesterAPI")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Mock database for storing backtesting data
MOCK_DB = {
    "current_test": {
        "id": "bt-123456",
        "status": "running",
        "progress": 65,
        "eta": "00:15:30",
        "startedAt": (datetime.datetime.now() - datetime.timedelta(minutes=30, seconds=15)).isoformat(),
        "elapsed": "00:30:15",
        "testPeriod": "2020-01-01 to 2024-12-31",
        "symbols": ["SPY", "QQQ", "IWM"],
        "parameters": {
            "lookbackPeriod": 20,
            "overBoughtThreshold": 0.8,
            "overSoldThreshold": 0.2
        },
        "executionStage": "Executing strategy logic",
        "results": {
            "totalReturn": 12.8,
            "annualizedReturn": 8.4,
            "sharpeRatio": 1.2,
            "sortino": 1.8,
            "maxDrawdown": 8.2,
            "winRate": 62,
            "profitFactor": 1.7,
            "expectancy": 0.87,
            "totalTrades": 245,
            "winningTrades": 152,
            "losingTrades": 93,
            "averageWinning": 1.2,
            "averageLosing": 0.6,
            "averageDuration": "3.5 days"
        },
        "newsSentiment": {
            "positiveEvents": 78,
            "negativeEvents": 53,
            "neutralEvents": 114,
            "totalEvents": 245,
            "averageSentimentScore": 0.23,
            "topEventsByImpact": [
                {
                    "headline": "Positive earnings surprise for AAPL",
                    "date": "2023-08-15",
                    "sentiment": "positive",
                    "impact": 0.85
                },
                {
                    "headline": "MSFT announces major cloud expansion",
                    "date": "2023-10-22",
                    "sentiment": "positive",
                    "impact": 0.76
                },
                {
                    "headline": "GOOGL faces new regulatory challenges",
                    "date": "2023-11-14",
                    "sentiment": "negative",
                    "impact": -0.67
                }
            ],
            "sentimentBySymbol": {
                "AAPL": 0.45,
                "MSFT": 0.38,
                "AMZN": 0.12,
                "GOOGL": -0.21,
                "META": 0.31,
                "TSLA": 0.19
            }
        }
    },
    "strategy_queue": [
        {
            "id": "ST-101",
            "name": "ML-Enhanced Mean Reversion",
            "description": "Machine learning model that identifies overbought/oversold conditions",
            "status": "In Queue",
            "priority": "High",
            "estimatedStart": "2025-05-06T15:30:00.000Z",
            "assets": ["SPY", "QQQ", "IWM"],
            "parameters": {},
            "complexity": "High",
            "createdAt": "2025-05-06T10:15:00.000Z",
            "updatedAt": "2025-05-06T11:15:00.000Z"
        },
        {
            "id": "ST-102",
            "name": "Fixed Income Tactical",
            "description": "Tactical rotation model for fixed income assets based on yield curve dynamics",
            "status": "In Queue",
            "priority": "Medium",
            "estimatedStart": "2025-05-06T17:15:00.000Z",
            "assets": ["TLT", "IEF", "HYG"],
            "parameters": {},
            "complexity": "Medium",
            "createdAt": "2025-05-06T09:15:00.000Z",
            "updatedAt": "2025-05-06T10:30:00.000Z"
        },
        {
            "id": "ST-103",
            "name": "Global Macro Rotation",
            "description": "Systematic global macro strategy with allocation shifts based on economic regime",
            "status": "In Queue",
            "priority": "Low",
            "estimatedStart": "2025-05-07T10:00:00.000Z",
            "assets": ["SPY", "EFA", "EEM", "AGG"],
            "parameters": {},
            "complexity": "High",
            "createdAt": "2025-05-06T08:00:00.000Z",
            "updatedAt": "2025-05-06T09:00:00.000Z"
        }
    ],
    "processing_stats": {
        "cpu": 72,
        "memory": 68,
        "disk": 45,
        "concurrentTests": 3,
        "completedToday": 42,
        "averageDuration": "00:45:32",
        "queueLength": 7
    },
    "deployments": [
        {
            "id": "deploy-987654",
            "strategy_id": "news_sentiment_20251204_121530",
            "status": "deployed",
            "timestamp": int(time.time() * 1000) - 3600000, # 1 hour ago
            "metadata": {
                "creator": "auto_deployment_pipeline",
                "symbols": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"],
                "sentiment_threshold": 0.3,
                "impact_threshold": 0.5,
                "backtest_id": "bt-1638262626",
                "backtest_results": {
                    "profit_factor": 2.4,
                    "win_rate": 0.68,
                    "trade_count": 42,
                    "max_drawdown_pct": -0.08,
                    "total_return": 0.32
                }
            }
        },
        {
            "id": "deploy-876543",
            "strategy_id": "news_sentiment_20251203_154512",
            "status": "pending",
            "timestamp": int(time.time() * 1000) - 1800000, # 30 minutes ago
            "metadata": {
                "creator": "manual_deployment",
                "symbols": ["SPY", "QQQ", "IWM", "DIA"],
                "sentiment_threshold": 0.35,
                "impact_threshold": 0.6,
                "backtest_id": "bt-1638262627"
            }
        }
    ]
}

# Health check endpoint
@app.route('/health', methods=['GET'])
def healthcheck():
    return jsonify({"status": "healthy", "timestamp": datetime.datetime.now().isoformat()})

# Frontend-compatible API endpoints
@app.route('/api/current_test', methods=['GET'])
def get_current_test():
    """Get current backtesting information - matches frontend naming convention"""
    logger.info("Fetching current test data")
    
    # Simulate some progress changes
    current_test = MOCK_DB["current_test"].copy()
    progress_increment = random.randint(1, 5)
    current_progress = current_test["progress"]
    
    # Update progress (cap at 99% to keep it running)
    new_progress = min(current_progress + progress_increment, 99)
    current_test["progress"] = new_progress
    
    # Update elapsed time
    elapsed_parts = current_test["elapsed"].split(":")
    elapsed_minutes = int(elapsed_parts[0]) * 60 + int(elapsed_parts[1])
    elapsed_minutes += random.randint(1, 3)
    new_hours = elapsed_minutes // 60
    new_minutes = elapsed_minutes % 60
    current_test["elapsed"] = f"{new_hours:02d}:{new_minutes:02d}:15"
    
    # Update ETA based on progress
    remaining = 100 - new_progress
    eta_minutes = int(remaining * 0.5)
    eta_hours = eta_minutes // 60
    eta_minutes = eta_minutes % 60
    current_test["eta"] = f"{eta_hours:02d}:{eta_minutes:02d}:30"
    
    # Save updates
    MOCK_DB["current_test"] = current_test
    
    return jsonify(current_test)

@app.route('/api/strategy_queue', methods=['GET'])
def get_strategy_queue():
    """Get strategy queue information - matches frontend naming convention"""
    logger.info("Fetching strategy queue")
    return jsonify(MOCK_DB["strategy_queue"])

@app.route('/api/processing_stats', methods=['GET'])
def get_processing_stats():
    """Get ML processing statistics - matches frontend naming convention"""
    logger.info("Fetching processing stats")
    
    # Simulate some changes in stats
    stats = MOCK_DB["processing_stats"].copy()
    
    # Randomly fluctuate CPU, memory, and disk within reasonable bounds
    stats["cpu"] = max(min(stats["cpu"] + random.randint(-5, 5), 95), 50)
    stats["memory"] = max(min(stats["memory"] + random.randint(-3, 3), 90), 55)
    stats["disk"] = max(min(stats["disk"] + random.randint(-1, 1), 75), 40)
    
    # Update queue length occasionally
    if random.random() < 0.3:
        stats["queueLength"] = max(stats["queueLength"] + random.choice([-1, 0, 1]), 0)
    
    # Increment completed tests occasionally
    if random.random() < 0.2:
        stats["completedToday"] += 1
    
    # Save updates
    MOCK_DB["processing_stats"] = stats
    
    return jsonify(stats)

@app.route('/api/ml-backtesting/control', methods=['POST'])
def control_backtest():
    """Control the current backtest (pause, resume, cancel)"""
    try:
        data = request.json
        action = data.get("action")
        logger.info(f"Controlling backtest: {action}")
        
        if action not in ["pause", "resume", "cancel"]:
            return jsonify({
                "success": False,
                "error": "Invalid action. Must be 'pause', 'resume', or 'cancel'"
            }), 400
        
        return jsonify({
            "success": True,
            "message": f"Backtest {action}d successfully"
        })
    except Exception as e:
        logger.error(f"Error controlling backtest: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error controlling backtest: {str(e)}"
        }), 500

@app.route('/api/submit-strategy', methods=['POST'])
def submit_strategy():
    """Submit a new strategy for backtesting"""
    try:
        data = request.json
        logger.info(f"Submitting new strategy: {data.get('name')}")
        
        # Create a new strategy entry
        new_strategy = {
            "id": f"ST-{random.randint(100000, 999999)}",
            "name": data.get("name", "New Strategy"),
            "description": data.get("description", ""),
            "status": "In Queue",
            "priority": data.get("priority", "Medium"),
            "estimatedStart": (datetime.datetime.now() + datetime.timedelta(minutes=random.randint(30, 120))).isoformat(),
            "assets": data.get("assets", []),
            "parameters": data.get("parameters", {}),
            "complexity": data.get("complexity", "Medium"),
            "createdAt": datetime.datetime.now().isoformat(),
            "updatedAt": datetime.datetime.now().isoformat()
        }
        
        # Add to queue
        MOCK_DB["strategy_queue"].append(new_strategy)
        
        return jsonify({
            "success": True,
            "message": "Strategy submitted successfully",
            "strategy": new_strategy
        })
    except Exception as e:
        logger.error(f"Error submitting strategy: {str(e)}")
        return jsonify({"success": False, "error": f"Error submitting strategy: {str(e)}"}), 500

@app.route('/api/deployments', methods=['GET'])
def get_deployments():
    """Get deployment status information"""
    logger.info("Fetching deployment statuses")
    return jsonify(MOCK_DB["deployments"])

if __name__ == '__main__':
    # Get port from environment or use default (now 5002)
    port = int(os.environ.get('PORT', 5002))
    debug = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'
    
    logger.info(f"Starting Backtester API server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
