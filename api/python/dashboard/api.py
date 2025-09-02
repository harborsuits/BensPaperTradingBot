#!/usr/bin/env python3
"""
BensBot Dashboard API

FastAPI backend for the trading dashboard that provides data access and control
endpoints for the frontend, with direct access to the MongoDB persistence layer.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, HTTPException, Depends, status, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pymongo
import redis
import jwt
from passlib.context import CryptContext

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dashboard_api")

# Initialize FastAPI app
app = FastAPI(
    title="BensBot Trading Dashboard API",
    description="API for the BensBot trading dashboard",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change_this_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Database connections
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "bensbot_trading")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Initialize MongoDB connection
try:
    mongo_client = pymongo.MongoClient(MONGODB_URI)
    mongo_db = mongo_client[MONGODB_DATABASE]
    # Test connection
    mongo_client.admin.command('ping')
    logger.info("Connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    mongo_client = None
    mongo_db = None

# Initialize Redis connection
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logger.info("Connected to Redis")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {str(e)}")
    redis_client = None

# Models
class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None


class User(BaseModel):
    """User model."""
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = False


class UserInDB(User):
    """User model with password hash."""
    hashed_password: str


class Order(BaseModel):
    """Order model."""
    id: str
    broker_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None


class Position(BaseModel):
    """Position model."""
    symbol: str
    broker_id: str
    quantity: float
    avg_entry_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = None
    last_updated: datetime


class MarginStatus(BaseModel):
    """Margin status model."""
    broker_id: str
    maintenance_margin: float
    margin_used: float
    margin_available: float
    margin_percentage: float
    equity: float
    last_updated: datetime
    status: str = "ok"  # ok, warning, call


class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status model."""
    active: bool
    reason: Optional[str] = None
    triggered_at: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    circuit_type: Optional[str] = None


class TradingStatus(BaseModel):
    """Trading status model."""
    paused: bool
    reason: Optional[str] = None
    paused_at: Optional[datetime] = None
    resume_at: Optional[datetime] = None
    manual_override: bool = False


class StrategyStatus(BaseModel):
    """Strategy status model."""
    id: str
    name: str
    enabled: bool
    status: str  # active, idle, error
    error_message: Optional[str] = None
    last_trade: Optional[datetime] = None
    performance: Dict[str, Any] = Field(default_factory=dict)
    symbols: List[str] = Field(default_factory=list)
    active_orders: int = 0
    active_positions: int = 0


class BrokerPerformance(BaseModel):
    """Broker performance model."""
    broker_id: str
    name: str
    latency_ms: float
    success_rate: float
    fill_rate: float
    execution_quality: float
    last_updated: datetime
    recent_errors: List[str] = Field(default_factory=list)


class SystemStats(BaseModel):
    """System statistics model."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_rx_bytes: float
    network_tx_bytes: float
    uptime_seconds: int
    last_updated: datetime


# Helper functions
def get_user(db, username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username == os.getenv("ADMIN_USERNAME", "admin"):
        # Default admin user
        return UserInDB(
            username=username,
            email="admin@example.com",
            full_name="Administrator",
            disabled=False,
            hashed_password=pwd_context.hash(os.getenv("ADMIN_PASSWORD", "adminpassword"))
        )
    
    # Check database for user
    if mongo_db:
        user_data = mongo_db.users.find_one({"username": username})
        if user_data:
            return UserInDB(**user_data)
    
    return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password."""
    user = get_user(mongo_db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(mongo_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Authentication endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get JWT token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info."""
    return current_user


# Database health check
@app.get("/health")
async def health_check():
    """Check the health of the API and its dependencies."""
    mongodb_ok = mongo_client is not None
    redis_ok = redis_client is not None
    
    if not mongodb_ok or not redis_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "mongodb": mongodb_ok,
                "redis": redis_ok,
                "message": "One or more dependencies unavailable"
            }
        )
    
    return {
        "status": "ok",
        "mongodb": mongodb_ok,
        "redis": redis_ok,
        "timestamp": datetime.utcnow().isoformat()
    }


# Order endpoints
@app.get("/orders", response_model=List[Order])
async def get_orders(
    status: Optional[str] = Query(None, description="Filter by order status"),
    broker_id: Optional[str] = Query(None, description="Filter by broker ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of orders to return"),
    skip: int = Query(0, ge=0, description="Number of orders to skip"),
    current_user: User = Depends(get_current_active_user)
):
    """Get orders with optional filtering."""
    if not mongo_db:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    # Build query
    query = {}
    if status:
        query["status"] = status
    if broker_id:
        query["broker_id"] = broker_id
    if symbol:
        query["symbol"] = symbol
    
    # Execute query
    orders = list(mongo_db.orders.find(query).sort("created_at", -1).skip(skip).limit(limit))
    
    # Convert MongoDB _id to str and format dates
    for order in orders:
        order["id"] = str(order.pop("_id"))
        if isinstance(order.get("created_at"), float):
            order["created_at"] = datetime.fromtimestamp(order["created_at"])
        if isinstance(order.get("updated_at"), float):
            order["updated_at"] = datetime.fromtimestamp(order["updated_at"])
    
    return orders


@app.get("/orders/{order_id}", response_model=Order)
async def get_order(
    order_id: str = Path(..., description="Order ID"),
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific order by ID."""
    if not mongo_db:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    # Find order
    order = mongo_db.orders.find_one({"_id": order_id})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    # Format order
    order["id"] = str(order.pop("_id"))
    if isinstance(order.get("created_at"), float):
        order["created_at"] = datetime.fromtimestamp(order["created_at"])
    if isinstance(order.get("updated_at"), float):
        order["updated_at"] = datetime.fromtimestamp(order["updated_at"])
    
    return order


# Position endpoints
@app.get("/positions", response_model=List[Position])
async def get_positions(
    broker_id: Optional[str] = Query(None, description="Filter by broker ID"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    current_user: User = Depends(get_current_active_user)
):
    """Get all open positions."""
    if not mongo_db:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    # Build query
    query = {}
    if broker_id:
        query["broker_id"] = broker_id
    if symbol:
        query["symbol"] = symbol
    
    # Execute query
    positions = list(mongo_db.positions.find(query))
    
    # Format positions
    for position in positions:
        position.pop("_id", None)
        if isinstance(position.get("last_updated"), float):
            position["last_updated"] = datetime.fromtimestamp(position["last_updated"])
    
    return positions


# Risk management endpoints
@app.get("/risk/margin-status", response_model=List[MarginStatus])
async def get_margin_status(
    broker_id: Optional[str] = Query(None, description="Filter by broker ID"),
    current_user: User = Depends(get_current_active_user)
):
    """Get margin status for all brokers or a specific broker."""
    if not mongo_db:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    query = {}
    if broker_id:
        query["broker_id"] = broker_id
    
    margin_statuses = list(mongo_db.margin_status.find(query).sort("last_updated", -1))
    
    # Format margin statuses
    for status in margin_statuses:
        status.pop("_id", None)
        if isinstance(status.get("last_updated"), float):
            status["last_updated"] = datetime.fromtimestamp(status["last_updated"])
    
    return margin_statuses


@app.get("/risk/circuit-breaker", response_model=CircuitBreakerStatus)
async def get_circuit_breaker_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get current circuit breaker status."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection unavailable")
    
    # Check Redis for circuit breaker status
    active = redis_client.get("bensbot:circuit_breaker:active") == "1"
    
    status = CircuitBreakerStatus(active=active)
    
    if active:
        reason = redis_client.get("bensbot:circuit_breaker:reason")
        if reason:
            status.reason = reason
        
        triggered_at = redis_client.get("bensbot:circuit_breaker:triggered_at")
        if triggered_at:
            status.triggered_at = datetime.fromtimestamp(float(triggered_at))
        
        cooldown_until = redis_client.get("bensbot:circuit_breaker:cooldown_until")
        if cooldown_until:
            status.cooldown_until = datetime.fromtimestamp(float(cooldown_until))
        
        circuit_type = redis_client.get("bensbot:circuit_breaker:type")
        if circuit_type:
            status.circuit_type = circuit_type
    
    return status


@app.get("/trading/status", response_model=TradingStatus)
async def get_trading_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get current trading status."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection unavailable")
    
    # Check Redis for trading status
    paused = redis_client.get("bensbot:trading:paused") == "1"
    
    status = TradingStatus(paused=paused)
    
    if paused:
        reason = redis_client.get("bensbot:trading:pause_reason")
        if reason:
            status.reason = reason
        
        paused_at = redis_client.get("bensbot:trading:paused_at")
        if paused_at:
            status.paused_at = datetime.fromtimestamp(float(paused_at))
        
        resume_at = redis_client.get("bensbot:trading:resume_at")
        if resume_at:
            status.resume_at = datetime.fromtimestamp(float(resume_at))
        
        manual_override = redis_client.get("bensbot:trading:manual_override") == "1"
        status.manual_override = manual_override
    
    return status


@app.post("/trading/resume")
async def resume_trading(
    current_user: User = Depends(get_current_active_user)
):
    """Manually resume trading after pause or circuit breaker."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection unavailable")
    
    # Check if trading is actually paused
    paused = redis_client.get("bensbot:trading:paused") == "1"
    if not paused:
        raise HTTPException(status_code=400, detail="Trading is not paused")
    
    # Resume trading
    redis_client.set("bensbot:trading:paused", "0")
    redis_client.set("bensbot:trading:manual_override", "1")
    redis_client.set("bensbot:circuit_breaker:active", "0")
    
    # Publish an event to notify the trading system
    redis_client.publish("bensbot:events", json.dumps({
        "type": "trading_resumed",
        "data": {
            "manual": True,
            "user": current_user.username,
            "timestamp": datetime.utcnow().timestamp()
        }
    }))
    
    return {"status": "success", "message": "Trading resumed"}


@app.post("/trading/pause")
async def pause_trading(
    reason: str = Body(..., embed=True),
    duration_minutes: Optional[int] = Body(60, embed=True),
    current_user: User = Depends(get_current_active_user)
):
    """Manually pause trading."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection unavailable")
    
    # Check if trading is already paused
    paused = redis_client.get("bensbot:trading:paused") == "1"
    if paused:
        raise HTTPException(status_code=400, detail="Trading is already paused")
    
    # Pause trading
    now = datetime.utcnow()
    redis_client.set("bensbot:trading:paused", "1")
    redis_client.set("bensbot:trading:pause_reason", reason)
    redis_client.set("bensbot:trading:paused_at", now.timestamp())
    redis_client.set("bensbot:trading:manual_override", "1")
    
    if duration_minutes:
        resume_at = now + timedelta(minutes=duration_minutes)
        redis_client.set("bensbot:trading:resume_at", resume_at.timestamp())
    
    # Publish an event to notify the trading system
    redis_client.publish("bensbot:events", json.dumps({
        "type": "trading_paused",
        "data": {
            "manual": True,
            "reason": reason,
            "user": current_user.username,
            "duration_minutes": duration_minutes,
            "timestamp": now.timestamp()
        }
    }))
    
    return {"status": "success", "message": "Trading paused"}


# Strategy endpoints
@app.get("/strategies", response_model=List[StrategyStatus])
async def get_strategies(
    current_user: User = Depends(get_current_active_user)
):
    """Get all strategy statuses."""
    if not mongo_db:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    # Get strategies
    strategies = list(mongo_db.strategies.find())
    
    # Format strategies
    for strategy in strategies:
        strategy.pop("_id", None)
        if isinstance(strategy.get("last_trade"), float):
            strategy["last_trade"] = datetime.fromtimestamp(strategy["last_trade"])
    
    return strategies


@app.post("/strategies/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: str = Path(..., description="Strategy ID"),
    enabled: bool = Body(..., embed=True),
    current_user: User = Depends(get_current_active_user)
):
    """Toggle a strategy's enabled status."""
    if not mongo_db:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    # Check if strategy exists
    strategy = mongo_db.strategies.find_one({"id": strategy_id})
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    
    # Update strategy
    mongo_db.strategies.update_one(
        {"id": strategy_id},
        {"$set": {"enabled": enabled}}
    )
    
    # Publish an event to notify the trading system
    if redis_client:
        redis_client.publish("bensbot:events", json.dumps({
            "type": "strategy_toggled",
            "data": {
                "strategy_id": strategy_id,
                "enabled": enabled,
                "user": current_user.username,
                "timestamp": datetime.utcnow().timestamp()
            }
        }))
    
    return {"status": "success", "message": f"Strategy {'enabled' if enabled else 'disabled'}"}


# Broker endpoints
@app.get("/brokers/performance", response_model=List[BrokerPerformance])
async def get_broker_performance(
    current_user: User = Depends(get_current_active_user)
):
    """Get performance metrics for all brokers."""
    if not mongo_db:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    # Get broker performance
    performances = list(mongo_db.broker_performance.find())
    
    # Format performances
    for perf in performances:
        perf.pop("_id", None)
        if isinstance(perf.get("last_updated"), float):
            perf["last_updated"] = datetime.fromtimestamp(perf["last_updated"])
    
    return performances


# System metrics
@app.get("/system/stats", response_model=SystemStats)
async def get_system_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Get system resource statistics."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis connection unavailable")
    
    # Get stats from Redis
    cpu_usage = redis_client.get("bensbot:system:cpu_usage")
    memory_usage = redis_client.get("bensbot:system:memory_usage")
    disk_usage = redis_client.get("bensbot:system:disk_usage")
    network_rx = redis_client.get("bensbot:system:network_rx_bytes")
    network_tx = redis_client.get("bensbot:system:network_tx_bytes")
    uptime = redis_client.get("bensbot:system:uptime_seconds")
    last_updated = redis_client.get("bensbot:system:last_updated")
    
    # If any stat is missing, use default values
    if not all([cpu_usage, memory_usage, disk_usage, uptime, last_updated]):
        # Generate some placeholder data
        now = datetime.utcnow()
        return SystemStats(
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_rx_bytes=0.0,
            network_tx_bytes=0.0,
            uptime_seconds=0,
            last_updated=now
        )
    
    return SystemStats(
        cpu_usage=float(cpu_usage),
        memory_usage=float(memory_usage),
        disk_usage=float(disk_usage),
        network_rx_bytes=float(network_rx or 0),
        network_tx_bytes=float(network_tx or 0),
        uptime_seconds=int(uptime),
        last_updated=datetime.fromtimestamp(float(last_updated))
    )


# When the app is run directly
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run("api:app", host=host, port=port, reload=True)
