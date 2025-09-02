"""
Smart Order Routing

Provides optimized broker selection based on execution quality, fees,
liquidity, and other factors to ensure best execution across multiple brokers.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from datetime import datetime, timedelta
import threading
import json
import statistics
from enum import Enum

from trading_bot.event_system import EventBus, Event
from trading_bot.brokers.broker_health_metrics import BrokerHealthManager

# Configure logging
logger = logging.getLogger(__name__)


class RoutingFactor(Enum):
    """Factors that influence routing decisions"""
    HEALTH = "health"               # Broker health and reliability
    FEES = "fees"                   # Trading fees and commissions
    EXECUTION_QUALITY = "execution" # Historical execution quality 
    LIQUIDITY = "liquidity"         # Available liquidity
    LATENCY = "latency"             # Response time and latency
    SMART = "smart"                 # Combined intelligent scoring


class BrokerExecutionStats:
    """
    Tracks execution statistics for a broker to evaluate performance
    """
    
    def __init__(self, broker_id: str, window_size: int = 100):
        """
        Initialize execution stats tracker for a broker
        
        Args:
            broker_id: Unique identifier for the broker
            window_size: Number of executions to track in sliding window
        """
        self.broker_id = broker_id
        self.window_size = window_size
        
        # Execution tracking
        self._executions = []  # List of execution records
        self._latencies = []   # List of execution latencies
        self._slippages = []   # List of price slippages
        
        # Fee structures
        self._fee_structure = {}  # Asset type -> fee info
        
        # For thread safety
        self._lock = threading.RLock()
        
        # Default values for assets without specific fee info
        self._default_fees = {
            "flat_fee": 0.0,         # Flat fee per trade
            "percentage_fee": 0.0,    # Percentage fee
            "min_fee": 0.0,           # Minimum fee
            "max_fee": float('inf')   # Maximum fee
        }
    
    def record_execution(self, order_id: str, symbol: str, asset_type: str,
                       submitted_price: float, executed_price: float,
                       quantity: float, side: str, latency_ms: float,
                       timestamp: datetime) -> None:
        """
        Record an order execution
        
        Args:
            order_id: Unique identifier for the order
            symbol: Asset symbol
            asset_type: Type of asset (e.g., 'stock', 'option', 'crypto')
            submitted_price: Price when order was submitted
            executed_price: Actual execution price
            quantity: Order quantity
            side: 'buy' or 'sell'
            latency_ms: Execution latency in milliseconds
            timestamp: Execution timestamp
        """
        with self._lock:
            # Calculate slippage as percentage (positive is unfavorable)
            slippage_pct = 0.0
            if submitted_price > 0:
                if side.lower() == 'buy':
                    slippage_pct = (executed_price - submitted_price) / submitted_price * 100
                else:  # sell
                    slippage_pct = (submitted_price - executed_price) / submitted_price * 100
            
            execution_record = {
                'order_id': order_id,
                'symbol': symbol,
                'asset_type': asset_type,
                'submitted_price': submitted_price,
                'executed_price': executed_price,
                'quantity': quantity,
                'side': side,
                'latency_ms': latency_ms,
                'slippage_pct': slippage_pct,
                'timestamp': timestamp
            }
            
            self._executions.append(execution_record)
            self._latencies.append(latency_ms)
            self._slippages.append(slippage_pct)
            
            # Keep only window_size records
            if len(self._executions) > self.window_size:
                self._executions = self._executions[-self.window_size:]
                self._latencies = self._latencies[-self.window_size:]
                self._slippages = self._slippages[-self.window_size:]
    
    def set_fee_structure(self, asset_type: str, flat_fee: float = 0.0,
                        percentage_fee: float = 0.0, min_fee: float = 0.0,
                        max_fee: Optional[float] = None) -> None:
        """
        Set fee structure for a specific asset type
        
        Args:
            asset_type: Type of asset (e.g., 'stock', 'option', 'crypto')
            flat_fee: Flat fee per trade
            percentage_fee: Percentage fee (0.01 = 1%)
            min_fee: Minimum fee per trade
            max_fee: Maximum fee per trade (None for no cap)
        """
        with self._lock:
            self._fee_structure[asset_type] = {
                "flat_fee": flat_fee,
                "percentage_fee": percentage_fee,
                "min_fee": min_fee,
                "max_fee": float('inf') if max_fee is None else max_fee
            }
    
    def set_default_fees(self, flat_fee: float = 0.0, percentage_fee: float = 0.0,
                       min_fee: float = 0.0, max_fee: Optional[float] = None) -> None:
        """
        Set default fee structure for asset types without specific fees
        
        Args:
            flat_fee: Flat fee per trade
            percentage_fee: Percentage fee (0.01 = 1%)
            min_fee: Minimum fee per trade
            max_fee: Maximum fee per trade (None for no cap)
        """
        with self._lock:
            self._default_fees = {
                "flat_fee": flat_fee,
                "percentage_fee": percentage_fee,
                "min_fee": min_fee,
                "max_fee": float('inf') if max_fee is None else max_fee
            }
    
    def get_average_latency(self) -> float:
        """
        Get average execution latency
        
        Returns:
            float: Average latency in milliseconds
        """
        with self._lock:
            if not self._latencies:
                return 0.0
            return sum(self._latencies) / len(self._latencies)
    
    def get_average_slippage(self) -> float:
        """
        Get average price slippage
        
        Returns:
            float: Average slippage as percentage
        """
        with self._lock:
            if not self._slippages:
                return 0.0
            return sum(self._slippages) / len(self._slippages)
    
    def get_execution_quality_score(self) -> float:
        """
        Calculate overall execution quality score (0.0-1.0, higher is better)
        
        Returns:
            float: Execution quality score
        """
        with self._lock:
            if not self._executions:
                # No data, return neutral score
                return 0.5
            
            # Calculate average latency
            avg_latency = self.get_average_latency()
            
            # Calculate average slippage
            avg_slippage = self.get_average_slippage()
            
            # Calculate latency score (lower is better)
            # Map latency from 0ms->1.0, 500ms->0.5, 1000ms+->0.0
            latency_score = max(0.0, 1.0 - (avg_latency / 1000.0))
            
            # Calculate slippage score (lower is better)
            # Map slippage from 0%->1.0, 0.1%->0.5, 0.2%+->0.0
            slippage_score = max(0.0, 1.0 - (abs(avg_slippage) / 0.2))
            
            # Weight scores (slippage is more important than latency)
            return slippage_score * 0.7 + latency_score * 0.3
    
    def calculate_fee(self, asset_type: str, price: float, quantity: float) -> float:
        """
        Calculate fee for a potential trade
        
        Args:
            asset_type: Type of asset
            price: Asset price
            quantity: Trade quantity
            
        Returns:
            float: Estimated fee
        """
        with self._lock:
            # Get fee structure for this asset type, or use default
            fee_info = self._fee_structure.get(asset_type, self._default_fees)
            
            # Calculate trade value
            trade_value = price * quantity
            
            # Calculate fee
            fee = fee_info["flat_fee"] + (trade_value * fee_info["percentage_fee"])
            
            # Apply min/max
            fee = max(fee_info["min_fee"], min(fee_info["max_fee"], fee))
            
            return fee
    
    def get_asset_types_with_fees(self) -> List[str]:
        """
        Get list of asset types with specific fee structures
        
        Returns:
            List[str]: List of asset types
        """
        with self._lock:
            return list(self._fee_structure.keys())
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get summary of execution statistics
        
        Returns:
            Dict: Summary of execution stats
        """
        with self._lock:
            return {
                'broker_id': self.broker_id,
                'executions_tracked': len(self._executions),
                'avg_latency_ms': self.get_average_latency(),
                'avg_slippage_pct': self.get_average_slippage(),
                'execution_quality_score': self.get_execution_quality_score(),
                'asset_types_with_fees': self.get_asset_types_with_fees()
            }


class LiquidityManager:
    """
    Manages liquidity information for different brokers and assets
    """
    
    def __init__(self):
        """Initialize the liquidity manager"""
        self._liquidity_info = {}  # broker_id -> asset_type -> symbol -> liquidity info
        self._lock = threading.RLock()
    
    def update_liquidity(self, broker_id: str, asset_type: str, symbol: str,
                       bid_size: float, ask_size: float,
                       daily_volume: Optional[float] = None) -> None:
        """
        Update liquidity information for a symbol
        
        Args:
            broker_id: Broker identifier
            asset_type: Asset type
            symbol: Asset symbol
            bid_size: Size available at the bid
            ask_size: Size available at the ask
            daily_volume: Optional daily trading volume
        """
        with self._lock:
            # Ensure nested dictionaries exist
            if broker_id not in self._liquidity_info:
                self._liquidity_info[broker_id] = {}
            
            if asset_type not in self._liquidity_info[broker_id]:
                self._liquidity_info[broker_id][asset_type] = {}
            
            # Update or create liquidity entry
            self._liquidity_info[broker_id][asset_type][symbol] = {
                'bid_size': bid_size,
                'ask_size': ask_size,
                'daily_volume': daily_volume,
                'timestamp': datetime.now()
            }
    
    def get_liquidity_score(self, broker_id: str, asset_type: str, symbol: str,
                          side: str, size: float) -> float:
        """
        Calculate liquidity score for an order (0.0-1.0, higher is better)
        
        Args:
            broker_id: Broker identifier
            asset_type: Asset type
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            size: Order size
            
        Returns:
            float: Liquidity score
        """
        with self._lock:
            # Return neutral score if no data
            if (broker_id not in self._liquidity_info or
                asset_type not in self._liquidity_info[broker_id] or
                symbol not in self._liquidity_info[broker_id][asset_type]):
                return 0.5
            
            liquidity = self._liquidity_info[broker_id][asset_type][symbol]
            
            # Check if data is too old (over 1 hour)
            if (datetime.now() - liquidity['timestamp']).total_seconds() > 3600:
                return 0.5
            
            # Get relevant size based on side
            available_size = liquidity['ask_size'] if side.lower() == 'buy' else liquidity['bid_size']
            
            # Calculate score based on what portion of the order can be filled
            if size <= 0 or available_size <= 0:
                return 0.5
            
            fill_ratio = min(1.0, available_size / size)
            
            # Map fill ratio to score (0.0-1.0)
            # 100% fill -> 1.0, 50% fill -> 0.5, 0% fill -> 0.0
            return fill_ratio
    
    def get_all_liquidity_info(self) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
        """
        Get all liquidity information
        
        Returns:
            Dict: All liquidity information
        """
        with self._lock:
            return self._liquidity_info


class SmartOrderRouter:
    """
    Intelligent order router that selects the optimal broker for each order
    based on multiple factors including health, fees, execution quality, and liquidity
    """
    
    def __init__(self, health_manager: BrokerHealthManager):
        """
        Initialize the smart order router
        
        Args:
            health_manager: Broker health manager instance
        """
        self.health_manager = health_manager
        self.event_bus = EventBus()
        
        # Broker execution statistics
        self.broker_stats = {}  # broker_id -> BrokerExecutionStats
        
        # Liquidity management
        self.liquidity_manager = LiquidityManager()
        
        # Routing weights (importance of different factors)
        self.routing_weights = {
            RoutingFactor.HEALTH: 0.3,        # Health is most important
            RoutingFactor.EXECUTION_QUALITY: 0.25,  # Then execution quality
            RoutingFactor.FEES: 0.2,          # Then fees
            RoutingFactor.LIQUIDITY: 0.15,    # Then liquidity
            RoutingFactor.LATENCY: 0.1        # Then latency
        }
        
        # For thread safety
        self._lock = threading.RLock()
        
        # Register for health events
        self.event_bus.subscribe("broker_health_updated", self._on_health_updated)
        
        logger.info("Initialized SmartOrderRouter")
    
    def register_broker(self, broker_id: str) -> None:
        """
        Register a broker for smart routing
        
        Args:
            broker_id: Unique identifier for the broker
        """
        with self._lock:
            # Ensure broker is registered with health manager
            if broker_id not in self.health_manager.brokers:
                self.health_manager.register_broker(broker_id)
            
            # Create execution stats for broker
            if broker_id not in self.broker_stats:
                self.broker_stats[broker_id] = BrokerExecutionStats(broker_id)
                logger.info(f"Registered broker '{broker_id}' for smart routing")
    
    def set_routing_weights(self, weights: Dict[RoutingFactor, float]) -> None:
        """
        Set weights for different routing factors
        
        Args:
            weights: Dict mapping factors to weights (should sum to 1.0)
        """
        with self._lock:
            # Validate weights
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                # Normalize weights
                normalized = {factor: weight / total for factor, weight in weights.items()}
                logger.warning(f"Routing weights did not sum to 1.0, normalized: {normalized}")
                self.routing_weights = normalized
            else:
                self.routing_weights = weights.copy()
            
            logger.info(f"Updated routing weights: {self.routing_weights}")
    
    def record_execution(self, broker_id: str, order_id: str, symbol: str, 
                       asset_type: str, submitted_price: float, executed_price: float,
                       quantity: float, side: str, latency_ms: float,
                       timestamp: Optional[datetime] = None) -> None:
        """
        Record an order execution for analysis
        
        Args:
            broker_id: Broker that executed the order
            order_id: Unique identifier for the order
            symbol: Asset symbol
            asset_type: Type of asset
            submitted_price: Price when order was submitted
            executed_price: Actual execution price
            quantity: Order quantity
            side: 'buy' or 'sell'
            latency_ms: Execution latency in milliseconds
            timestamp: Execution timestamp (defaults to now)
        """
        with self._lock:
            # Register broker if needed
            if broker_id not in self.broker_stats:
                self.register_broker(broker_id)
            
            # Record execution
            self.broker_stats[broker_id].record_execution(
                order_id=order_id,
                symbol=symbol,
                asset_type=asset_type,
                submitted_price=submitted_price,
                executed_price=executed_price,
                quantity=quantity,
                side=side,
                latency_ms=latency_ms,
                timestamp=timestamp or datetime.now()
            )
            
            # Publish event
            self.event_bus.publish(Event(
                "order_execution_recorded",
                {
                    "broker_id": broker_id,
                    "order_id": order_id,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "submitted_price": submitted_price,
                    "executed_price": executed_price,
                    "quantity": quantity,
                    "side": side,
                    "latency_ms": latency_ms,
                    "timestamp": (timestamp or datetime.now()).isoformat()
                }
            ))
    
    def update_broker_fees(self, broker_id: str, asset_type: str, flat_fee: float = 0.0,
                         percentage_fee: float = 0.0, min_fee: float = 0.0,
                         max_fee: Optional[float] = None) -> None:
        """
        Update fee structure for a broker and asset type
        
        Args:
            broker_id: Broker identifier
            asset_type: Asset type
            flat_fee: Flat fee per trade
            percentage_fee: Percentage fee (0.01 = 1%)
            min_fee: Minimum fee per trade
            max_fee: Maximum fee per trade (None for no cap)
        """
        with self._lock:
            # Register broker if needed
            if broker_id not in self.broker_stats:
                self.register_broker(broker_id)
            
            # Update fee structure
            self.broker_stats[broker_id].set_fee_structure(
                asset_type=asset_type,
                flat_fee=flat_fee,
                percentage_fee=percentage_fee,
                min_fee=min_fee,
                max_fee=max_fee
            )
    
    def update_liquidity(self, broker_id: str, asset_type: str, symbol: str,
                       bid_size: float, ask_size: float,
                       daily_volume: Optional[float] = None) -> None:
        """
        Update liquidity information
        
        Args:
            broker_id: Broker identifier
            asset_type: Asset type
            symbol: Asset symbol
            bid_size: Size available at the bid
            ask_size: Size available at the ask
            daily_volume: Optional daily trading volume
        """
        self.liquidity_manager.update_liquidity(
            broker_id=broker_id,
            asset_type=asset_type,
            symbol=symbol,
            bid_size=bid_size,
            ask_size=ask_size,
            daily_volume=daily_volume
        )
    
    def get_broker_scores(self, asset_type: str, symbol: str, side: str,
                        quantity: float, price: float) -> Dict[str, Dict[str, float]]:
        """
        Get scores for all brokers for a potential order
        
        Args:
            asset_type: Asset type
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Current asset price
            
        Returns:
            Dict: Map of broker IDs to their factor scores and total score
        """
        with self._lock:
            scores = {}
            
            for broker_id, stats in self.broker_stats.items():
                # Skip brokers that aren't registered with health manager
                if broker_id not in self.health_manager.brokers:
                    continue
                
                # Calculate factor scores (all 0.0-1.0, higher is better)
                health_score = 0.0
                if self.health_manager.is_broker_healthy(broker_id):
                    # Convert broker rank to score (1st place -> 1.0, last place -> 0.1)
                    broker_ranks = self.health_manager.rank_brokers_by_health()
                    if broker_id in broker_ranks:
                        rank = broker_ranks.index(broker_id)
                        rank_score = 1.0 - (rank / len(broker_ranks) * 0.9)
                        health_score = rank_score
                else:
                    # Unhealthy brokers get a very low score
                    health_score = 0.05
                
                # Get execution quality score
                execution_score = stats.get_execution_quality_score()
                
                # Calculate fee score
                fee = stats.calculate_fee(asset_type, price, quantity)
                
                # Get average fee across all brokers for comparison
                all_fees = [b.calculate_fee(asset_type, price, quantity) for b in self.broker_stats.values()]
                avg_fee = sum(all_fees) / len(all_fees) if all_fees else 0.0
                max_fee = max(all_fees) if all_fees else 0.0
                
                # Convert fee to score (lowest fee -> 1.0, highest fee -> 0.0)
                fee_score = 1.0
                if max_fee > 0 and len(all_fees) > 1:
                    fee_score = 1.0 - (fee / max_fee)
                
                # Get liquidity score
                liquidity_score = self.liquidity_manager.get_liquidity_score(
                    broker_id, asset_type, symbol, side, quantity
                )
                
                # Get latency score (from execution stats)
                avg_latency = stats.get_average_latency()
                latency_score = max(0.0, 1.0 - (avg_latency / 1000.0))
                
                # Calculate weighted score
                factor_scores = {
                    RoutingFactor.HEALTH.value: health_score,
                    RoutingFactor.EXECUTION_QUALITY.value: execution_score,
                    RoutingFactor.FEES.value: fee_score,
                    RoutingFactor.LIQUIDITY.value: liquidity_score,
                    RoutingFactor.LATENCY.value: latency_score
                }
                
                total_score = (
                    health_score * self.routing_weights[RoutingFactor.HEALTH] +
                    execution_score * self.routing_weights[RoutingFactor.EXECUTION_QUALITY] +
                    fee_score * self.routing_weights[RoutingFactor.FEES] +
                    liquidity_score * self.routing_weights[RoutingFactor.LIQUIDITY] +
                    latency_score * self.routing_weights[RoutingFactor.LATENCY]
                )
                
                # Store scores
                scores[broker_id] = {
                    **factor_scores,
                    'total_score': total_score
                }
            
            return scores
    
    def get_best_broker(self, asset_type: str, symbol: str, side: str,
                      quantity: float, price: float,
                      routing_mode: RoutingFactor = RoutingFactor.SMART) -> Optional[str]:
        """
        Get the best broker for an order
        
        Args:
            asset_type: Asset type
            symbol: Asset symbol
            side: Order side ('buy' or 'sell')
            quantity: Order quantity
            price: Current asset price
            routing_mode: Routing factor to optimize for
            
        Returns:
            Optional[str]: Best broker ID or None if no suitable broker
        """
        with self._lock:
            # Get scores for all brokers
            broker_scores = self.get_broker_scores(
                asset_type, symbol, side, quantity, price
            )
            
            if not broker_scores:
                return None
            
            # Select based on routing mode
            if routing_mode == RoutingFactor.SMART:
                # Use weighted total score
                return max(broker_scores.items(), key=lambda x: x[1]['total_score'])[0]
            else:
                # Use specific factor score
                factor = routing_mode.value
                return max(broker_scores.items(), key=lambda x: x[1][factor])[0]
    
    def _on_health_updated(self, event_data: Dict[str, Any]) -> None:
        """Handle broker health update event"""
        # Can be used to trigger re-evaluation of routing decisions
        pass
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing statistics and configuration
        
        Returns:
            Dict: Routing statistics
        """
        with self._lock:
            return {
                'registered_brokers': list(self.broker_stats.keys()),
                'routing_weights': {f.value: w for f, w in self.routing_weights.items()},
                'broker_stats': {broker_id: stats.get_stats_summary() 
                              for broker_id, stats in self.broker_stats.items()}
            }
