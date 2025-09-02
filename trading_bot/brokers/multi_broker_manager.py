"""
Multi-Broker Manager

Provides a unified interface for managing multiple broker connections simultaneously,
with specific support for Tradier, Alpaca, TradeStation, and E*TRADE.

Features:
- Secure credential management via pluggable CredentialStore
- Comprehensive audit trail for all operations
- Thread-safe broker operations with automatic failover
- Asset-type based routing to appropriate brokers
"""

import logging
import time
import random
import threading
from typing import Dict, List, Optional, Union, Any, Tuple, Type
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import uuid

from .broker_interface import (
    BrokerInterface, BrokerCredentials, BrokerAccount, Position, Order, Quote, Bar,
    AssetType, OrderType, OrderSide, OrderStatus, TimeInForce,
    BrokerAuthenticationError, BrokerConnectionError, BrokerOrderError
)
from .credential_store import CredentialStore, AuthMethod, EncryptedFileStore, YamlFileStore, AuthenticatorFactory
from .trade_audit_log import TradeAuditLog, AuditEventType, SqliteAuditLog, JsonFileAuditLog

# Configure logging
logger = logging.getLogger(__name__)


class MultiBrokerManager:
    """
    Manages multiple broker connections for cross-platform trading.
    
    Features:
    - Thread-safe broker connections
    - Automatic failover between brokers
    - Asset class routing to appropriate brokers
    - Unified interface for trading operations
    - Support for paper and live trading modes
    """
    
    def __init__(self, credential_store: Optional[CredentialStore] = None, 
                 audit_log: Optional[TradeAuditLog] = None,
                 max_retries: int = 3, retry_delay: int = 2, auto_failover: bool = True):
        """
        Initialize the multi-broker manager.
        
        Args:
            credential_store: Secure storage for broker credentials
            audit_log: Audit logger for recording all broker operations
            max_retries: Maximum number of operation retry attempts
            retry_delay: Delay between retries in seconds
            auto_failover: Whether to automatically failover to another broker on failure
        """
        self.brokers: Dict[str, BrokerInterface] = {}
        self.primary_broker_id: Optional[str] = None
        self.active_broker_id: Optional[str] = None
        
        # Initialize credential store if not provided
        self.credential_store = credential_store
        
        # Initialize audit log if not provided
        self.audit_log = audit_log
        
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_failover = auto_failover
        
        # Asset-to-broker routing configuration
        self.asset_routing: Dict[AssetType, str] = {}
        
        # For thread safety
        self._lock = threading.RLock()
        
        # For broker monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = 60  # seconds
        
        # Cache for broker data
        self.account_info_cache: Dict[str, List[BrokerAccount]] = {}
        self.positions_cache: Dict[str, List[Position]] = {}
        self.quotes_cache: Dict[str, Dict[str, Quote]] = {}
        self.last_cache_update = datetime.now()
        self.cache_ttl = 5  # seconds
        
        logger.info("MultiBrokerManager initialized")
        
        # Log initialization event if audit log is available
        if self.audit_log:
            self.log_event(
                AuditEventType.SYSTEM_ERROR,
                details={
                    "action": "manager_initialized",
                    "max_retries": max_retries,
                    "retry_delay": retry_delay,
                    "auto_failover": auto_failover
                }
            )

    def add_broker(self, broker_id: str, broker: BrokerInterface, credentials: BrokerCredentials, 
                  make_primary: bool = False,
                  auth_method: AuthMethod = AuthMethod.API_KEY) -> bool:
        """
        Add a broker to the manager.
        
        Args:
            broker_id: Unique identifier for this broker instance
            broker: Broker implementation
            credentials: Authentication credentials for this broker
            make_primary: Whether to make this the primary broker
            auth_method: Authentication method to use for this broker
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if broker_id in self.brokers:
                logger.warning(f"Broker with ID '{broker_id}' already exists, replacing")
            
            # Store broker instance
            self.brokers[broker_id] = broker
            
            # Store credentials in credential store if available
            if self.credential_store:
                try:
                    self.credential_store.store_credentials(
                        broker_id, 
                        credentials, 
                        auth_method
                    )
                except Exception as e:
                    logger.error(f"Failed to store credentials for broker '{broker_id}': {str(e)}")
                    return False
            
            if make_primary or not self.primary_broker_id:
                self.primary_broker_id = broker_id
                self.active_broker_id = broker_id
                logger.info(f"Set '{broker_id}' as primary broker")
            
            logger.info(f"Added broker '{broker_id}' ({broker.get_broker_name()})")
            
            # Log broker addition event
            if self.audit_log:
                self.log_event(
                    AuditEventType.BROKER_OPERATION,
                    details={
                        "action": "broker_added",
                        "broker_name": broker.get_broker_name(),
                        "is_primary": make_primary or not self.primary_broker_id
                    },
                    broker_id=broker_id
                )
            
            return True

    def connect_all(self) -> Dict[str, bool]:
        """
        Connect to all registered brokers.
        
        Returns:
            Dict[str, bool]: Map of broker IDs to connection success status
        """
        results = {}
        
        with self._lock:
            for broker_id, broker in self.brokers.items():
                try:
                    # Get credentials from store if available, otherwise use direct credentials
                    if self.credential_store:
                        try:
                            credentials = self.credential_store.get_credentials(broker_id)
                            auth_method = self.credential_store.get_auth_method(broker_id)
                            
                            # Create appropriate authenticator
                            authenticator = AuthenticatorFactory.create_authenticator(
                                auth_method, 
                                self.credential_store, 
                                broker_id
                            )
                            
                            # Authenticate
                            auth_result = authenticator.authenticate()
                            if not auth_result.get('authenticated', False):
                                logger.error(f"Authentication failed for broker '{broker_id}'")
                                results[broker_id] = False
                                continue
                                
                            # Connect with authenticated credentials
                            success = broker.connect(credentials)
                        except KeyError:
                            logger.error(f"No credentials found for broker '{broker_id}'")
                            results[broker_id] = False
                            continue
                        except Exception as e:
                            logger.error(f"Error retrieving credentials for '{broker_id}': {str(e)}")
                            results[broker_id] = False
                            continue
                    else:
                        # Fall back to direct credentials if no credential store
                        logger.warning(f"No credential store available, using direct credentials for '{broker_id}'")
                        # In this fallback mode, we'd need to have the credentials passed directly
                        # through other means, like a configuration mechanism
                        credentials = self._get_fallback_credentials(broker_id)
                        if not credentials:
                            logger.error(f"No fallback credentials for '{broker_id}'")
                            results[broker_id] = False
                            continue
                        success = broker.connect(credentials)
                    
                    results[broker_id] = success
                    logger.info(f"Connection to '{broker_id}' {'succeeded' if success else 'failed'}")
                    
                    # Log connection event
                    if self.audit_log:
                        event_type = AuditEventType.BROKER_CONNECTED if success else AuditEventType.BROKER_ERROR
                        self.log_event(
                            event_type,
                            details={
                                "action": "connect",
                                "result": "success" if success else "failure"
                            },
                            broker_id=broker_id
                        )
                        
                except Exception as e:
                    results[broker_id] = False
                    logger.error(f"Error connecting to '{broker_id}': {str(e)}")
                    
                    # Log connection error
                    if self.audit_log:
                        self.log_event(
                            AuditEventType.BROKER_ERROR,
                            details={
                                "action": "connect",
                                "error": str(e)
                            },
                            broker_id=broker_id
                        )
        
        return results
        
    def _get_fallback_credentials(self, broker_id: str) -> Optional[BrokerCredentials]:
        """
        Get credentials for a broker when no credential store is available.
        This is a fallback method and should be replaced with a more secure approach.
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            Optional[BrokerCredentials]: Credentials or None if not found
        """
        # This would be replaced with your fallback credential retrieval mechanism
        # For example, from environment variables or a config file
        logger.warning("Using fallback credential retrieval mechanism")
        return None

    def disconnect_all(self) -> Dict[str, bool]:
        """
        Disconnect from all brokers.
        
        Returns:
            Dict[str, bool]: Map of broker IDs to disconnection success status
        """
        results = {}
        
        with self._lock:
            for broker_id, broker in self.brokers.items():
                try:
                    if broker.is_connected():
                        success = broker.disconnect()
                        results[broker_id] = success
                        logger.info(f"Disconnection from '{broker_id}' {'succeeded' if success else 'failed'}")
                    else:
                        results[broker_id] = True
                except Exception as e:
                    results[broker_id] = False
                    logger.error(f"Error disconnecting from '{broker_id}': {str(e)}")
        
        return results

    def set_asset_routing(self, asset_type: AssetType, broker_id: str) -> bool:
        """
        Configure which broker should handle which asset types.
        
        Args:
            asset_type: Type of asset
            broker_id: ID of broker to handle this asset type
            
        Returns:
            bool: Success status
        """
        with self._lock:
            if broker_id not in self.brokers:
                logger.error(f"Cannot route {asset_type.name} to unknown broker '{broker_id}'")
                return False
            
            # Verify broker supports this asset type
            if asset_type not in self.brokers[broker_id].get_supported_asset_types():
                logger.warning(f"Broker '{broker_id}' may not support {asset_type.name}")
            
            self.asset_routing[asset_type] = broker_id
            logger.info(f"Routing {asset_type.name} to broker '{broker_id}'")
            return True

    def get_broker_for_asset(self, asset_type: AssetType) -> Optional[str]:
        """
        Get the appropriate broker ID for an asset type.
        
        Args:
            asset_type: Type of asset
            
        Returns:
            Optional[str]: Broker ID or None if not configured
        """
        return self.asset_routing.get(asset_type)

    def start_monitoring(self) -> bool:
        """
        Start background thread to monitor broker connections.
        
        Returns:
            bool: Success status
        """
        with self._lock:
            if self.monitoring_thread and self.monitoring_active:
                logger.warning("Monitoring already active")
                return True
            
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="BrokerMonitoringThread"
            )
            self.monitoring_thread.start()
            logger.info("Started broker connection monitoring")
            return True

    def stop_monitoring(self) -> bool:
        """
        Stop the broker monitoring thread.
        
        Returns:
            bool: Success status
        """
        with self._lock:
            if not self.monitoring_active:
                logger.warning("Monitoring not active")
                return True
            
            self.monitoring_active = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
                self.monitoring_thread = None
            
            logger.info("Stopped broker connection monitoring")
            return True

    def _monitoring_loop(self) -> None:
        """Background thread for monitoring broker connections."""
        logger.info("Broker monitoring thread started")
        
        while self.monitoring_active:
            try:
                # Check each broker connection
                for broker_id, broker in self.brokers.items():
                    try:
                        connected = broker.is_connected()
                        if not connected:
                            logger.warning(f"Broker '{broker_id}' disconnected, attempting reconnect")
                            broker.connect(self.broker_credentials[broker_id])
                    except Exception as e:
                        logger.error(f"Error checking connection for '{broker_id}': {str(e)}")
                
                # Sleep until next check
                for _ in range(self.monitoring_interval):
                    if not self.monitoring_active:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)  # Brief pause on error
        
        logger.info("Broker monitoring thread stopped")

    def _execute_with_failover(self, operation_func, *args, broker_id=None, asset_type=None, **kwargs):
        """
        Execute an operation with retry and failover logic.
        
        Args:
            operation_func: Function to call on the broker
            *args: Arguments to pass to the function
            broker_id: Specific broker to use (optional)
            asset_type: Asset type to route to appropriate broker (optional)
            **kwargs: Keyword arguments for the function
            
        Returns:
            Operation result
            
        Raises:
            BrokerConnectionError: If all brokers fail
        """
        if not broker_id and asset_type and asset_type in self.asset_routing:
            broker_id = self.asset_routing[asset_type]
        
        # Use active broker if no specific broker requested
        if not broker_id:
            broker_id = self.active_broker_id
        
        if not broker_id or broker_id not in self.brokers:
            raise BrokerConnectionError("No suitable broker available")
        
        # Try the specified broker first
        primary_broker = self.brokers[broker_id]
        
        # Keep track of tried brokers to avoid circular fallback
        tried_brokers = set()
        current_broker_id = broker_id
        
        for attempt in range(self.max_retries):
            if current_broker_id in tried_brokers:
                # Already tried this broker
                continue
            
            tried_brokers.add(current_broker_id)
            current_broker = self.brokers[current_broker_id]
            
            try:
                # Ensure broker is connected
                if not current_broker.is_connected():
                    success = current_broker.connect(self.broker_credentials[current_broker_id])
                    if not success:
                        logger.warning(f"Failed to connect to broker '{current_broker_id}'")
                        if self.auto_failover and len(self.brokers) > 1:
                            current_broker_id = self._get_next_broker(current_broker_id, tried_brokers)
                            continue
                        else:
                            time.sleep(self.retry_delay)
                            continue
                
                # Execute the operation
                result = operation_func(current_broker, *args, **kwargs)
                
                # Update active broker on success if different
                if current_broker_id != self.active_broker_id:
                    with self._lock:
                        self.active_broker_id = current_broker_id
                        logger.info(f"Switched active broker to '{current_broker_id}'")
                
                return result
                
            except (BrokerConnectionError, BrokerAuthenticationError) as e:
                logger.warning(f"Connection error with broker '{current_broker_id}': {str(e)}")
                
                # Try another broker if failover enabled
                if self.auto_failover and len(self.brokers) > 1:
                    current_broker_id = self._get_next_broker(current_broker_id, tried_brokers)
                    logger.info(f"Failing over to broker '{current_broker_id}'")
                else:
                    time.sleep(self.retry_delay)
            
            except BrokerOrderError as e:
                # Don't retry order errors, they're likely valid errors
                logger.error(f"Order error with broker '{current_broker_id}': {str(e)}")
                raise
            
            except Exception as e:
                logger.error(f"Unexpected error with broker '{current_broker_id}': {str(e)}")
                time.sleep(self.retry_delay)
        
        # All retries and brokers failed
        raise BrokerConnectionError(f"All brokers failed after {self.max_retries} attempts")

    def _get_next_broker(self, current_broker_id: str, excluded_brokers: set) -> str:
        """Get the next available broker ID, excluding those already tried."""
        available_brokers = [bid for bid in self.brokers.keys() if bid not in excluded_brokers]
        
        if not available_brokers:
            # If all brokers have been tried, go back to primary
            return self.primary_broker_id
        
        # Prioritize the primary broker if available
        if self.primary_broker_id in available_brokers:
            return self.primary_broker_id
        
        # Otherwise, pick the first available
        return available_brokers[0]

    def refresh_cache(self) -> None:
        """Force refresh of cached broker data."""
        with self._lock:
            self.account_info_cache.clear()
            self.positions_cache.clear()
            self.quotes_cache.clear()
            self.last_cache_update = datetime.now()

    def get_all_accounts(self) -> Dict[str, List[BrokerAccount]]:
        """
        Get account information from all connected brokers.
        
        Returns:
            Dict[str, List[BrokerAccount]]: Map of broker IDs to their account information
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.brokers)) as executor:
            futures = {}
            
            # Start requests to each broker in parallel
            for broker_id, broker in self.brokers.items():
                if broker.is_connected():
                    futures[broker_id] = executor.submit(
                        self._execute_with_failover,
                        lambda b, *a, **k: b.get_account_info(),
                        broker_id=broker_id
                    )
            
            # Collect results as they complete
            for broker_id, future in futures.items():
                try:
                    results[broker_id] = future.result()
                    self.account_info_cache[broker_id] = results[broker_id]
                except Exception as e:
                    logger.error(f"Failed to get accounts from '{broker_id}': {str(e)}")
                    results[broker_id] = []
        
        self.last_cache_update = datetime.now()
        return results

    def get_all_positions(self) -> Dict[str, List[Position]]:
        """
        Get positions from all connected brokers.
        
        Returns:
            Dict[str, List[Position]]: Map of broker IDs to their positions
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.brokers)) as executor:
            futures = {}
            
            # Start requests to each broker in parallel
            for broker_id, broker in self.brokers.items():
                if broker.is_connected():
                    futures[broker_id] = executor.submit(
                        self._execute_with_failover,
                        lambda b, *a, **k: b.get_positions(),
                        broker_id=broker_id
                    )
            
            # Collect results as they complete
            for broker_id, future in futures.items():
                try:
                    results[broker_id] = future.result()
                    self.positions_cache[broker_id] = results[broker_id]
                    
                    # Log position retrieval
                    if self.audit_log:
                        self.log_event(
                            AuditEventType.POSITION_UPDATED,
                            details={
                                "action": "get_positions",
                                "position_count": len(results[broker_id]),
                                "positions": [{
                                    "symbol": p.symbol,
                                    "quantity": p.quantity,
                                    "avg_price": p.avg_price
                                } for p in results[broker_id][:5]]  # Log limited positions to avoid huge logs
                            },
                            broker_id=broker_id
                        )
                        
                except Exception as e:
                    logger.error(f"Failed to get positions from '{broker_id}': {str(e)}")
                    results[broker_id] = []
                    
                    # Log error
                    if self.audit_log:
                        self.log_event(
                            AuditEventType.BROKER_ERROR,
                            details={
                                "action": "get_positions",
                                "error": str(e)
                            },
                            broker_id=broker_id
                        )
        
        self.last_cache_update = datetime.now()
        return results

    def place_order(self, order: Order) -> Order:
        """
        Place an order with the appropriate broker based on asset type.
        
        Args:
            order: Order to place
            
        Returns:
            Order: Updated order with broker's response
            
        Raises:
            BrokerOrderError: If order placement fails
        """
        # Determine which broker to use
        broker_id = None
        if order.broker_id:
            broker_id = order.broker_id
        elif order.asset_type in self.asset_routing:
            broker_id = self.asset_routing[order.asset_type]
        
        # Log order submission event before execution
        if self.audit_log:
            self.log_event(
                AuditEventType.ORDER_SUBMITTED,
                details={
                    "symbol": order.symbol,
                    "quantity": order.quantity,
                    "order_type": order.order_type.value if hasattr(order.order_type, 'value') else str(order.order_type),
                    "order_side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                    "price": order.price,
                    "time_in_force": order.time_in_force.value if hasattr(order.time_in_force, 'value') else str(order.time_in_force),
                    "asset_type": order.asset_type.value if hasattr(order.asset_type, 'value') else str(order.asset_type),
                    "target_broker": broker_id
                },
                broker_id=broker_id,
                order_id=order.order_id if hasattr(order, 'order_id') else None
            )
        
        # Execute the order placement
        try:
            result = self._execute_with_failover(
                lambda b, o: b.place_order(o),
                order,
                broker_id=broker_id,
                asset_type=order.asset_type
            )
            
            # Log successful order result
            if self.audit_log:
                self.log_event(
                    AuditEventType.ORDER_FILLED if result.status == OrderStatus.FILLED else AuditEventType.ORDER_SUBMITTED,
                    details={
                        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                        "filled_quantity": result.filled_quantity if hasattr(result, 'filled_quantity') else 0,
                        "fill_price": result.fill_price if hasattr(result, 'fill_price') else None,
                        "broker_order_id": result.broker_order_id if hasattr(result, 'broker_order_id') else None
                    },
                    broker_id=broker_id,
                    order_id=result.order_id if hasattr(result, 'order_id') else None
                )
            
            return result
            
        except BrokerOrderError as e:
            # Log order error
            if self.audit_log:
                self.log_event(
                    AuditEventType.ORDER_REJECTED,
                    details={
                        "error": str(e),
                        "status": "rejected",
                        "reason": getattr(e, 'reason', 'Unknown rejection reason')
                    },
                    broker_id=broker_id,
                    order_id=order.order_id if hasattr(order, 'order_id') else None
                )
            # Re-raise the exception
            raise

    def cancel_order(self, order_id: str, broker_id: Optional[str] = None) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            broker_id: Specific broker to use (optional)
            
        Returns:
            bool: Success status
            
        Raises:
            BrokerOrderError: If cancellation fails
        """
        try:
            # Log order cancellation attempt
            if self.audit_log:
                self.log_event(
                    AuditEventType.ORDER_CANCELLED,
                    details={
                        "action": "cancel_attempt",
                    },
                    broker_id=broker_id,
                    order_id=order_id
                )
                
            result = self._execute_with_failover(
                lambda b, oid: b.cancel_order(oid),
                order_id,
                broker_id=broker_id
            )
            
            # Log successful cancellation
            if self.audit_log:
                self.log_event(
                    AuditEventType.ORDER_CANCELLED,
                    details={
                        "action": "cancel_success",
                        "result": result
                    },
                    broker_id=broker_id,
                    order_id=order_id
                )
                
            return result
            
        except Exception as e:
            # Log cancellation error
            if self.audit_log:
                self.log_event(
                    AuditEventType.BROKER_ERROR,
                    details={
                        "action": "cancel_order",
                        "error": str(e)
                    },
                    broker_id=broker_id,
                    order_id=order_id
                )
            raise

    def get_order_status(self, order_id: str, broker_id: Optional[str] = None) -> Order:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to check
            broker_id: Specific broker to use (optional)
            
        Returns:
            Order: Updated order information
            
        Raises:
            BrokerOrderError: If status retrieval fails
        """
        return self._execute_with_failover(
            lambda b, oid: b.get_order_status(oid),
            order_id,
            broker_id=broker_id
        )

    def get_quote(self, symbol: str, asset_type: AssetType = AssetType.STOCK) -> Quote:
        """
        Get a quote for a symbol.
        
        Args:
            symbol: Symbol to get quote for
            asset_type: Type of asset
            
        Returns:
            Quote: Quote information
            
        Raises:
            BrokerConnectionError: If quote retrieval fails
        """
        # Use cached quote if available and recent
        if (symbol in self.quotes_cache.get(self.active_broker_id, {}) and 
            (datetime.now() - self.last_cache_update).total_seconds() < self.cache_ttl):
            return self.quotes_cache[self.active_broker_id][symbol]
        
        # Otherwise fetch fresh quote
        broker_id = self.asset_routing.get(asset_type)
        quote = self._execute_with_failover(
            lambda b, s, a: b.get_quote(s, a),
            symbol, asset_type,
            broker_id=broker_id,
            asset_type=asset_type
        )
        
        # Cache the result
        if self.active_broker_id not in self.quotes_cache:
            self.quotes_cache[self.active_broker_id] = {}
        self.quotes_cache[self.active_broker_id][symbol] = quote
        
        return quote

    def get_bars(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        asset_type: AssetType = AssetType.STOCK
    ) -> List[Bar]:
        """
        Get historical bars for a symbol.
        
        Args:
            symbol: Symbol to get bars for
            timeframe: Timeframe for the bars
            start: Start datetime
            end: End datetime (defaults to now)
            limit: Maximum number of bars to return
            asset_type: Type of asset
            
        Returns:
            List[Bar]: List of historical bars
            
        Raises:
            BrokerConnectionError: If bar retrieval fails
        """
        broker_id = self.asset_routing.get(asset_type)
        return self._execute_with_failover(
            lambda b, s, tf, st, e, l, a: b.get_bars(s, tf, st, e, l, a),
            symbol, timeframe, start, end, limit, asset_type,
            broker_id=broker_id,
            asset_type=asset_type
        )

    def get_available_brokers(self) -> Dict[str, str]:
        """
        Get list of available brokers with their connection status.
        
        Returns:
            Dict[str, str]: Map of broker IDs to connection status
        """
        results = {}
        
        with self._lock:
            for broker_id, broker in self.brokers.items():
                try:
                    connected = broker.is_connected()
                    results[broker_id] = "connected" if connected else "disconnected"
                except Exception:
                    results[broker_id] = "error"
        
        return results

    def is_primary_broker_available(self) -> bool:
        """
        Check if the primary broker is available.
        
        Returns:
            bool: True if primary broker is connected
        """
        if not self.primary_broker_id or self.primary_broker_id not in self.brokers:
            return False
        
        try:
            is_connected = self.brokers[self.primary_broker_id].is_connected()
            
            # Log broker status check
            if self.audit_log:
                self.log_event(
                    AuditEventType.BROKER_OPERATION,
                    details={
                        "action": "check_connection",
                        "is_connected": is_connected
                    },
                    broker_id=self.primary_broker_id
                )
                
            return is_connected
        except Exception as e:
            # Log error
            if self.audit_log:
                self.log_event(
                    AuditEventType.BROKER_ERROR,
                    details={
                        "action": "check_connection",
                        "error": str(e)
                    },
                    broker_id=self.primary_broker_id
                )
                
            return False

    def reset_to_primary_broker(self) -> bool:
        """
        Reset the active broker to the primary broker.
        
        Returns:
            bool: Success status
        """
        with self._lock:
            if not self.primary_broker_id or self.primary_broker_id not in self.brokers:
                logger.error("No primary broker defined")
                return False
            
            try:
                primary_broker = self.brokers[self.primary_broker_id]
                if not primary_broker.is_connected():
                    # Get credentials from credential store if available
                    if self.credential_store:
                        try:
                            credentials = self.credential_store.get_credentials(self.primary_broker_id)
                            success = primary_broker.connect(credentials)
                        except Exception as e:
                            logger.error(f"Error getting credentials for primary broker: {str(e)}")
                            return False
                    else:
                        # Fallback to direct credentials
                        credentials = self._get_fallback_credentials(self.primary_broker_id)
                        if not credentials:
                            logger.error("No credentials available for primary broker")
                            return False
                        success = primary_broker.connect(credentials)
                        
                    if not success:
                        logger.error("Failed to connect to primary broker")
                        return False
                
                self.active_broker_id = self.primary_broker_id
                logger.info(f"Reset to primary broker '{self.primary_broker_id}'")
                
                # Log the reset event
                if self.audit_log:
                    self.log_event(
                        AuditEventType.BROKER_OPERATION,
                        details={"action": "reset_to_primary_broker"},
                        broker_id=self.primary_broker_id
                    )
                    
                return True
            except Exception as e:
                logger.error(f"Error resetting to primary broker: {str(e)}")
                
                # Log the error
                if self.audit_log:
                    self.log_event(
                        AuditEventType.SYSTEM_ERROR,
                        details={
                            "action": "reset_to_primary_broker",
                            "error": str(e)
                        }
                    )
                    
                return False

    def place_order_idempotently(self, order, idempotency_key=None):
        """
        Place an order with idempotency check to prevent duplicate orders.
        
        Args:
            order: Order to place
            idempotency_key: Optional custom idempotency key
            
        Returns:
            Dict: Order result
        """
        from redis import Redis
        import json
        
        # Get Redis connection from the repository
        redis_client = None
        try:
            from trading_bot.persistence.connection_manager import ConnectionManager
            connection_mgr = ConnectionManager()
            redis_client = connection_mgr.get_redis_connection()
        except Exception as e:
            self.logger.warning(f"Redis not available for idempotency check: {str(e)}")
            # Fall back to placing order directly
            return self.place_order(order)
        
        # Generate idempotency key if not provided
        if not idempotency_key:
            # Use order properties to create a unique key
            # Include timestamp to allow identical orders separated by time
            idempotency_key = f"{order.symbol}-{order.side}-{order.quantity}-{int(time.time())}"
        
        cache_key = f"order:idempotency:{idempotency_key}"
        
        # Check if we've already placed this order
        try:
            existing_order_id = redis_client.get(cache_key)
            if existing_order_id:
                existing_order_id = existing_order_id.decode('utf-8')
                self.logger.info(f"Found existing order with idempotency key {idempotency_key}: {existing_order_id}")
                
                # Return the existing order status
                return self.get_order_status(existing_order_id)
        except Exception as e:
            self.logger.warning(f"Error checking order idempotency: {str(e)}")
            # Fall back to placing order directly
            return self.place_order(order)
        
        # Place the order
        result = self.place_order(order)
        
        # If successful, store in Redis with TTL
        if result.get('success') and result.get('order_id'):
            try:
                redis_client.setex(
                    cache_key,
                    86400,  # 24 hour TTL
                    result.get('order_id')
                )
                self.logger.info(f"Stored order idempotency key {idempotency_key} -> {result.get('order_id')}")
            except Exception as e:
                self.logger.warning(f"Error storing order idempotency key: {str(e)}")
        
        return result
