"""
Logging API endpoints.
Provides endpoints for system logs and alert notifications.
"""
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import random
import json

# Import WebSocket manager
from trading_bot.api.websocket_manager import enabled_manager

# Import auth service
from trading_bot.auth.service import AuthService

logger = logging.getLogger("TradingBotAPI.Logging")

router = APIRouter()

class LogEvent(BaseModel):
    """A log event from the trading system."""
    id: str
    timestamp: str
    level: str  # INFO, WARNING, ERROR, DEBUG
    source: str  # Component that generated the log
    message: str
    details: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    requires_action: bool = False
    related_symbol: Optional[str] = None
    category: Optional[str] = None  # System, Trading, Data, etc.

class AlertNotification(BaseModel):
    """An alert notification that requires user attention."""
    id: str
    timestamp: str
    severity: str  # critical, warning, info
    title: str
    message: str
    source: str
    acknowledged: bool = False
    action_taken: bool = False
    related_symbol: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Sample data generator functions
def generate_sample_logs(level: str = None, limit: int = 100) -> List[LogEvent]:
    """Generate sample log events for demo purposes."""
    log_levels = ["INFO", "WARNING", "ERROR", "DEBUG"]
    if level:
        log_levels = [level.upper()]
    
    sources = ["DataIngestion", "StrategyManager", "PortfolioManager", "TradeExecution", "System"]
    categories = ["System", "Trading", "Data", "Authentication", "API"]
    
    now = datetime.now()
    
    logs = []
    for i in range(limit):
        level = random.choice(log_levels)
        source = random.choice(sources)
        category = random.choice(categories)
        
        # Generate appropriate message based on level and source
        if level == "ERROR":
            if source == "DataIngestion":
                message = "Failed to fetch data from API provider"
                details = {"provider": "Alpha Vantage", "status_code": 429, "retry_after": 60}
            elif source == "TradeExecution":
                message = "Order execution failed"
                details = {"order_id": f"order-{random.randint(1000, 9999)}", "reason": "Insufficient funds"}
            else:
                message = f"Error in {source} process"
                details = {"error_code": random.randint(400, 599)}
        elif level == "WARNING":
            if source == "StrategyManager":
                message = "Strategy signal quality below threshold"
                details = {"strategy": f"Strategy-{random.randint(1, 10)}", "quality": round(random.uniform(0.3, 0.5), 2)}
            else:
                message = f"Warning in {source} process"
                details = {"warning_code": random.randint(100, 399)}
        else:  # INFO or DEBUG
            messages = [
                f"{source} process completed successfully",
                f"{category} operation performed",
                f"Configuration updated for {source}",
                f"New data received from {random.choice(['market', 'API', 'user'])}"
            ]
            message = random.choice(messages)
            details = {"duration": round(random.uniform(0.1, 5.0), 2)}
        
        # Decide if this log requires action
        requires_action = level in ["ERROR", "WARNING"] and random.random() < 0.3
        
        # Create log event
        log = LogEvent(
            id=f"log-{i}",
            timestamp=(now - timedelta(minutes=random.randint(0, 60))).isoformat(),
            level=level,
            source=source,
            message=message,
            details=details,
            acknowledged=random.random() > 0.7,
            requires_action=requires_action,
            related_symbol=random.choice(["AAPL", "MSFT", "GOOGL", None, None]),
            category=category
        )
        
        logs.append(log)
    
    # Sort by timestamp, newest first
    logs.sort(key=lambda x: x.timestamp, reverse=True)
    
    return logs

def generate_sample_alerts(limit: int = 20) -> List[AlertNotification]:
    """Generate sample alert notifications for demo purposes."""
    severities = ["critical", "warning", "info"]
    sources = ["DataIngestion", "StrategyManager", "PortfolioManager", "TradeExecution", "System"]
    
    now = datetime.now()
    
    alerts = []
    for i in range(limit):
        severity = random.choice(severities)
        source = random.choice(sources)
        
        # Generate appropriate alert based on severity and source
        if severity == "critical":
            if source == "DataIngestion":
                title = "Data Feed Disruption"
                message = "Critical market data feed disruption detected"
                details = {"provider": "Primary Market Data", "outage_duration": f"{random.randint(5, 30)} minutes"}
            elif source == "TradeExecution":
                title = "Trade Execution Failure"
                message = "Multiple trade executions have failed"
                details = {"failed_count": random.randint(3, 10), "impact": "High"}
            else:
                title = f"Critical {source} Alert"
                message = f"Critical issue detected in {source}"
                details = {"system_impact": "High"}
        elif severity == "warning":
            if source == "PortfolioManager":
                title = "Portfolio Drift Warning"
                message = "Portfolio allocation has drifted beyond thresholds"
                details = {"drift_percentage": f"{random.randint(5, 15)}%", "rebalance_recommended": True}
            else:
                title = f"Warning from {source}"
                message = f"Warning condition detected in {source}"
                details = {"importance": "Medium"}
        else:  # info
            titles = [
                f"{source} Notification",
                "System Update Available",
                "Market Information",
                "Trading Session Update"
            ]
            title = random.choice(titles)
            message = f"Informational update from {source}"
            details = {"priority": "Low"}
        
        # Create alert notification
        alert = AlertNotification(
            id=f"alert-{i}",
            timestamp=(now - timedelta(minutes=random.randint(0, 120))).isoformat(),
            severity=severity,
            title=title,
            message=message,
            source=source,
            acknowledged=random.random() > 0.5,
            action_taken=random.random() > 0.7,
            related_symbol=random.choice(["AAPL", "MSFT", "GOOGL", None, None, None]),
            details=details
        )
        
        alerts.append(alert)
    
    # Sort by timestamp and severity, newest and most critical first
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda x: (severity_rank.get(x.severity, 999), x.timestamp), reverse=True)
    
    return alerts

@router.get("/logs", response_model=List[LogEvent])
async def get_logs(
    level: str = Query(None, description="Filter logs by level (INFO, WARNING, ERROR, DEBUG)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get system logs filtered by level and paginated.
    """
    try:
        # In a real implementation, this would query your logs database or service
        all_logs = generate_sample_logs(level=level, limit=limit+offset)
        
        # Apply pagination
        paginated_logs = all_logs[offset:offset+limit]
        
        return paginated_logs
    except Exception as e:
        logger.error(f"Error fetching logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching logs: {str(e)}")

@router.get("/alerts", response_model=List[AlertNotification])
async def get_alerts(
    severity: str = Query(None, description="Filter alerts by severity (critical, warning, info)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of alerts to return"),
    acknowledged: bool = Query(None, description="Filter by acknowledged status"),
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Get system alerts filtered by severity and acknowledged status.
    """
    try:
        # In a real implementation, this would query your alerts database or service
        all_alerts = generate_sample_alerts(limit=100)  # Generate a larger set to filter from
        
        # Apply filters
        filtered_alerts = all_alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity.lower()]
        
        if acknowledged is not None:
            filtered_alerts = [a for a in filtered_alerts if a.acknowledged == acknowledged]
        
        # Apply limit
        result = filtered_alerts[:limit]
        
        return result
    except Exception as e:
        logger.error(f"Error fetching alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user = Depends(AuthService.get_current_active_user),
):
    """
    Acknowledge an alert.
    """
    try:
        # In a real implementation, this would update the alert in your database
        all_alerts = generate_sample_alerts(limit=100)
        
        # Find the alert
        for alert in all_alerts:
            if alert.id == alert_id:
                # Update the alert
                alert.acknowledged = True
                
                # Broadcast the updated alert to all connected clients
                await enabled_manager.broadcast_to_channel("alerts", "alert_update", alert.dict())
                
                return {"status": "success", "message": f"Alert {alert_id} acknowledged"}
        
        # Alert not found
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error acknowledging alert: {str(e)}")

# Broadcast functions for real-time updates
async def broadcast_log_event(log_event: LogEvent):
    """Broadcast a log event to all connected clients subscribed to the logs channel."""
    try:
        await enabled_manager.broadcast_to_channel("logs", "log_event", log_event.dict())
    except Exception as e:
        logger.error(f"Error broadcasting log event: {str(e)}")

async def broadcast_alert(alert: AlertNotification):
    """Broadcast an alert to all connected clients subscribed to the alerts channel."""
    try:
        await enabled_manager.broadcast_to_channel("alerts", "new_alert", alert.dict())
    except Exception as e:
        logger.error(f"Error broadcasting alert: {str(e)}")
