"""
Brokerage API Integration

This package provides functionality for connecting to various brokerage APIs,
handling order execution, position management, and account monitoring.
"""

# Import key components for easy access
from .brokerage_client import (
    BrokerageClient,
    OrderType,
    OrderSide, 
    TimeInForce,
    BrokerConnectionStatus,
    BrokerAPIError,
    BrokerAuthError,
    BrokerConnectionError,
    OrderExecutionError,
    BROKER_IMPLEMENTATIONS
)

# Optional broker imports – keep optional deps from breaking API startup
try:
    from .alpaca_client import AlpacaClient
except Exception:  # ImportError or any runtime import issue from optional deps
    AlpacaClient = None  # type: ignore
from .connection_monitor import ConnectionMonitor, ConnectionAlert
from .order_selector import (
    OrderSelector, 
    MarketCondition, 
    ExecutionSpeed, 
    PriceAggression
)
from .broker_registry import BrokerRegistry, get_broker_registry

# Export key components
__all__ = [
    'BrokerageClient',
    'OrderType',
    'OrderSide',
    'TimeInForce',
    'BrokerConnectionStatus',
    'BrokerAPIError',
    'BrokerAuthError',
    'BrokerConnectionError',
    'OrderExecutionError',
    'ConnectionMonitor',
    'ConnectionAlert',
    'OrderSelector',
    'MarketCondition',
    'ExecutionSpeed',
    'PriceAggression',
    'BrokerRegistry',
    'get_broker_registry',
    'BROKER_IMPLEMENTATIONS'
]

# Export AlpacaClient only if available
if 'AlpacaClient' in globals() and AlpacaClient is not None:  # type: ignore[name-defined]
    __all__.append('AlpacaClient')