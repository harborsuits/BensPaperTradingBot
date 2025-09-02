"""
Alerts API routes.
Provides endpoints for system alerts and notifications.
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import random
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger("TradingBotAPI.Alerts")

router = APIRouter()

class AlertItem(BaseModel):
    id: str
    message: str
    type: str
    timestamp: str
    source: str
    read: bool
    category: Optional[str] = None
    severity: str
    details: Optional[Dict[str, Any]] = None

@router.get("/alerts", response_model=List[AlertItem])
async def get_alerts(limit: int = 20):
    """Get system alerts and notifications."""
    try:
        # In a real implementation, this would pull data from your alert system
        # For now, we'll create realistic data that matches the frontend expectations
        
        # Create sample alerts
        alerts = []
        
        # Alert types and categories
        alert_types = ["system", "trade", "strategy", "market", "account"]
        severities = ["critical", "high", "medium", "low", "info"]
        
        # Alert templates for different types
        alert_templates = {
            "system": [
                {"message": "System health check {STATUS}", "severity": ["info", "medium", "high"]},
                {"message": "API rate limit at {PERCENT}% for {BROKER}", "severity": ["medium", "high"]},
                {"message": "Database backup {STATUS}", "severity": ["info", "critical"]},
                {"message": "System resource usage high: {RESOURCE} at {PERCENT}%", "severity": ["medium", "high"]}
            ],
            "trade": [
                {"message": "{SYMBOL} trade {ACTION} at ${PRICE}", "severity": ["info"]},
                {"message": "Stop loss triggered for {SYMBOL} at ${PRICE}", "severity": ["medium"]},
                {"message": "Take profit reached for {SYMBOL} at ${PRICE}", "severity": ["info"]},
                {"message": "Failed to execute {ACTION} for {SYMBOL}", "severity": ["high"]}
            ],
            "strategy": [
                {"message": "{STRATEGY} generated new signal for {SYMBOL}", "severity": ["info"]},
                {"message": "{STRATEGY} performance below threshold: {PERCENT}%", "severity": ["medium", "high"]},
                {"message": "Strategy rotation activated: {OLD_STRATEGY} → {NEW_STRATEGY}", "severity": ["info"]},
                {"message": "Backtest completed for {STRATEGY} with {RESULT}", "severity": ["info"]}
            ],
            "market": [
                {"message": "Market regime change detected: {OLD_REGIME} → {NEW_REGIME}", "severity": ["medium"]},
                {"message": "Unusual volatility detected in {SYMBOL}", "severity": ["medium", "high"]},
                {"message": "Earnings announcement for {SYMBOL} scheduled for {DATE}", "severity": ["info"]},
                {"message": "Significant news impact detected for {SYMBOL}", "severity": ["medium"]}
            ],
            "account": [
                {"message": "Account balance below {THRESHOLD}", "severity": ["high"]},
                {"message": "Margin usage at {PERCENT}%", "severity": ["info", "medium", "high", "critical"]},
                {"message": "Account connection status: {STATUS} for {BROKER}", "severity": ["info", "critical"]},
                {"message": "Portfolio allocation updated", "severity": ["info"]}
            ]
        }
        
        # Generate random alerts
        for i in range(min(limit, 20)):
            # Select random alert type and template
            alert_type = random.choice(alert_types)
            template = random.choice(alert_templates[alert_type])
            
            # Determine severity (either a specific one or choose from options)
            severity = template["severity"] if isinstance(template["severity"], str) else random.choice(template["severity"])
            
            # Generate placeholder values
            placeholders = {
                "STATUS": random.choice(["completed", "failed", "pending", "in progress"]),
                "PERCENT": random.randint(50, 99),
                "BROKER": random.choice(["Tradier", "Alpaca", "ETrade"]),
                "RESOURCE": random.choice(["CPU", "Memory", "Disk", "Network"]),
                "SYMBOL": random.choice(["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]),
                "ACTION": random.choice(["executed", "canceled", "modified", "rejected"]),
                "PRICE": round(random.uniform(50, 500), 2),
                "STRATEGY": random.choice([
                    "GapTradingStrategy", 
                    "TrendFollowingStrategy", 
                    "MACDStrategy", 
                    "RSIStrategy", 
                    "VolumeSurgeStrategy"
                ]),
                "OLD_STRATEGY": random.choice(["TrendFollowingStrategy", "MACDStrategy"]),
                "NEW_STRATEGY": random.choice(["RSIStrategy", "VolumeSurgeStrategy"]),
                "RESULT": random.choice(["positive results", "mixed results", "below expectations"]),
                "OLD_REGIME": random.choice(["Bullish", "Bearish", "Sideways"]),
                "NEW_REGIME": random.choice(["Bullish", "Bearish", "Volatile"]),
                "DATE": (datetime.now() + timedelta(days=random.randint(1, 14))).strftime("%Y-%m-%d"),
                "THRESHOLD": f"${random.randrange(5000, 50000, 5000)}",
            }
            
            # Process message template
            message = template["message"]
            for key, value in placeholders.items():
                if f"{{{key}}}" in message:
                    message = message.replace(f"{{{key}}}", str(value))
            
            # Create timestamp (more recent for higher severity)
            if severity in ["critical", "high"]:
                minutes_ago = random.randint(1, 60)
            else:
                minutes_ago = random.randint(60, 1440)  # 1-24 hours
            
            timestamp = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
            
            # Create alert with appropriate details
            details = {}
            if alert_type == "trade":
                details = {
                    "symbol": placeholders["SYMBOL"] if "SYMBOL" in placeholders else random.choice(["AAPL", "MSFT"]),
                    "price": placeholders["PRICE"] if "PRICE" in placeholders else round(random.uniform(50, 500), 2),
                    "quantity": random.randint(10, 100)
                }
            elif alert_type == "strategy":
                details = {
                    "strategy": placeholders["STRATEGY"] if "STRATEGY" in placeholders else "DefaultStrategy",
                    "metrics": {
                        "performance": f"{random.uniform(-5, 15):.2f}%",
                        "trades": random.randint(1, 10)
                    }
                }
            
            # Determine if alert has been read (older alerts more likely to be read)
            read = random.random() < (minutes_ago / 1440)
            
            alert = {
                "id": str(uuid.uuid4()),
                "message": message,
                "type": alert_type,
                "timestamp": timestamp,
                "source": random.choice(["System", "TradingEngine", "RiskManager", "MarketMonitor"]),
                "read": read,
                "category": random.choice(["notification", "warning", "error"]) if random.random() < 0.7 else None,
                "severity": severity,
                "details": details if details else None
            }
            
            alerts.append(alert)
        
        # Sort by timestamp (most recent first)
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return alerts
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")
