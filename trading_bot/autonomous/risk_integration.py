#!/usr/bin/env python3
"""
Risk Integration for Autonomous Trading Engine

This module integrates the autonomous trading engine with the risk management system,
ensuring all autonomously executed strategies adhere to proper risk controls.
It serves as a bridge between strategy generation/optimization and safe execution.
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Import risk manager
from trading_bot.risk.risk_manager import RiskManager, RiskLevel, StopLossType

# Import event system
from trading_bot.event_system import EventBus, Event, EventType

# Import autonomous components
from trading_bot.autonomous.autonomous_engine import AutonomousEngine, StrategyCandidate

logger = logging.getLogger(__name__)

class AutonomousRiskManager:
    """
    Integrates risk management with the autonomous trading engine.
    
    This class serves as the bridge between the autonomous engine that generates 
    and optimizes strategies and the risk management system that ensures safe execution.
    It provides:
    
    1. Risk-aware strategy deployment
    2. Dynamic risk allocation across strategies
    3. Circuit breakers for autonomous trading
    4. Correlation analysis to prevent overexposure
    5. Centralized risk monitoring for all autonomous strategies
    """
    
    def __init__(self, 
                 risk_config: Optional[Dict[str, Any]] = None,
                 data_dir: Optional[str] = None):
        """
        Initialize the autonomous risk manager.
        
        Args:
            risk_config: Configuration for risk parameters
            data_dir: Directory for storing risk data
        """
        self.risk_manager = RiskManager(config=risk_config)
        self.engine = None
        
        # Event handling
        self.event_bus = EventBus()
        self._register_event_handlers()
        
        # State tracking
        self.deployed_strategies = {}  # strategy_id -> deployment info
        self.strategy_allocations = {}  # strategy_id -> allocation percentage
        self.risk_metrics = {}  # strategy_id -> risk metrics
        self.circuit_breakers = {
            "portfolio_drawdown": 15.0,  # Maximum portfolio drawdown percentage
            "strategy_drawdown": 25.0,   # Maximum strategy drawdown percentage
            "daily_loss": 5.0,           # Maximum daily loss percentage
            "trade_frequency": 20,       # Maximum trades per day
            "correlation_threshold": 0.7  # Maximum correlation between strategies
        }
        
        # Data directory
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), 
                                                "trading_data", 
                                                "autonomous_risk")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing state if available
        self._load_state()
        
        logger.info("Autonomous Risk Manager initialized")
    
    def connect_engine(self, engine: AutonomousEngine) -> None:
        """
        Connect to the autonomous trading engine.
        
        Args:
            engine: Autonomous trading engine instance
        """
        self.engine = engine
        logger.info("Connected to autonomous engine")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events."""
        # Strategy events
        self.event_bus.register(EventType.STRATEGY_OPTIMISED, self._handle_optimized_strategy)
        self.event_bus.register(EventType.STRATEGY_DEPLOYED, self._handle_deployed_strategy)
        
        # Risk events
        self.event_bus.register(EventType.RISK_LEVEL_CHANGED, self._handle_risk_level_change)
        self.event_bus.register(EventType.CIRCUIT_BREAKER_TRIGGERED, self._handle_circuit_breaker)
        
        # Trade events
        self.event_bus.register(EventType.TRADE_EXECUTED, self._handle_trade_executed)
        self.event_bus.register(EventType.POSITION_CLOSED, self._handle_position_closed)
        
        logger.info("Event handlers registered")
    
    def deploy_strategy(self, strategy_id: str, allocation_percentage: float = 5.0,
                        risk_level: Optional[RiskLevel] = None,
                        stop_loss_type: Optional[StopLossType] = None) -> bool:
        """
        Deploy a strategy with risk controls.
        
        Args:
            strategy_id: ID of the strategy to deploy
            allocation_percentage: Percentage of capital to allocate
            risk_level: Risk level override for this strategy
            stop_loss_type: Stop loss type override for this strategy
            
        Returns:
            bool: True if deployment successful
        """
        if not self.engine:
            logger.error("No autonomous engine connected")
            return False
        
        # Get strategy details
        strategy = None
        for candidate in self.engine.get_top_candidates():
            if candidate.strategy_id == strategy_id:
                strategy = candidate
                break
        
        if not strategy:
            logger.error(f"Strategy {strategy_id} not found or not approved")
            return False
        
        # Check if we're already at maximum portfolio risk
        current_risk = self.risk_manager.get_risk_metrics()
        if current_risk.get("total_portfolio_risk", 0) > 90:
            logger.warning("Maximum portfolio risk reached, cannot deploy strategy")
            return False
        
        # Check for correlated strategies
        if self._is_highly_correlated(strategy):
            logger.warning(f"Strategy {strategy_id} is highly correlated with existing strategies")
            return False
        
        # Configure risk parameters for this strategy
        risk_params = {
            "allocation_percentage": min(allocation_percentage, 20.0),  # Cap at 20%
            "risk_level": risk_level or RiskLevel.MEDIUM,
            "stop_loss_type": stop_loss_type or StopLossType.VOLATILITY,
            "drawdown_limit": self.circuit_breakers["strategy_drawdown"]
        }
        
        # Store deployment information
        self.deployed_strategies[strategy_id] = {
            "strategy": strategy.to_dict(),
            "risk_params": risk_params,
            "deploy_time": datetime.now().isoformat(),
            "status": "active",
            "performance": {
                "trades": 0,
                "profit_loss": 0.0,
                "win_rate": 0.0,
                "max_drawdown": 0.0
            }
        }
        
        # Set allocation
        self.strategy_allocations[strategy_id] = allocation_percentage
        
        # Initialize risk metrics
        self.risk_metrics[strategy_id] = {
            "current_drawdown": 0.0,
            "daily_profit_loss": 0.0,
            "trade_count_today": 0,
            "position_count": 0,
            "largest_position_size": 0.0,
            "last_updated": datetime.now().isoformat()
        }
        
        # Save state
        self._save_state()
        
        # Deploy through the engine
        success = self.engine.deploy_strategy(strategy_id)
        
        # Emit deployment event with risk parameters
        if success:
            self._emit_event(EventType.STRATEGY_DEPLOYED_WITH_RISK, {
                "strategy_id": strategy_id,
                "risk_params": risk_params,
                "allocation_percentage": allocation_percentage
            })
        
        return success
    
    def adjust_allocation(self, strategy_id: str, new_allocation: float) -> bool:
        """
        Adjust the capital allocation for a deployed strategy.
        
        Args:
            strategy_id: Strategy ID
            new_allocation: New allocation percentage
            
        Returns:
            bool: True if adjustment successful
        """
        if strategy_id not in self.deployed_strategies:
            logger.error(f"Strategy {strategy_id} not deployed")
            return False
            
        # Cap allocation
        new_allocation = min(new_allocation, 20.0)
        
        # Update allocation
        self.strategy_allocations[strategy_id] = new_allocation
        
        # Update deployment info
        self.deployed_strategies[strategy_id]["risk_params"]["allocation_percentage"] = new_allocation
        
        # Save state
        self._save_state()
        
        logger.info(f"Adjusted allocation for {strategy_id} to {new_allocation}%")
        return True
    
    def set_circuit_breakers(self, breakers: Dict[str, Any]) -> None:
        """
        Set circuit breaker thresholds for autonomous trading.
        
        Args:
            breakers: Dictionary of circuit breaker thresholds
        """
        valid_breakers = {
            k: v for k, v in breakers.items() if k in self.circuit_breakers
        }
        
        self.circuit_breakers.update(valid_breakers)
        logger.info(f"Updated circuit breakers: {valid_breakers}")
        
        # Save state
        self._save_state()
    
    def check_circuit_breakers(self, market_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if any circuit breakers should be triggered.
        
        Args:
            market_data: Current market data
            
        Returns:
            (bool, List[str]): Tuple of (should_halt, reasons)
        """
        should_halt = False
        reasons = []
        
        # Check risk metrics
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        # Portfolio drawdown check
        current_drawdown = risk_metrics.get("current_drawdown_pct", 0)
        if current_drawdown > self.circuit_breakers["portfolio_drawdown"]:
            should_halt = True
            reasons.append(f"Portfolio drawdown ({current_drawdown:.2f}%) exceeds threshold")
        
        # Daily loss check
        daily_profit_loss = risk_metrics.get("daily_profit_loss_pct", 0)
        if daily_profit_loss < -self.circuit_breakers["daily_loss"]:
            should_halt = True
            reasons.append(f"Daily loss ({-daily_profit_loss:.2f}%) exceeds threshold")
        
        # Trading frequency check
        total_trades_today = sum(metrics.get("trade_count_today", 0) 
                               for metrics in self.risk_metrics.values())
        if total_trades_today > self.circuit_breakers["trade_frequency"]:
            should_halt = True
            reasons.append(f"Trade frequency ({total_trades_today}) exceeds threshold")
        
        # Strategy-specific checks
        for strategy_id, metrics in self.risk_metrics.items():
            # Strategy drawdown check
            if metrics.get("current_drawdown", 0) > self.circuit_breakers["strategy_drawdown"]:
                should_halt = True
                reasons.append(f"Strategy {strategy_id} drawdown exceeds threshold")
        
        if should_halt:
            # Emit circuit breaker event
            self._emit_event(EventType.CIRCUIT_BREAKER_TRIGGERED, {
                "reasons": reasons,
                "timestamp": datetime.now().isoformat(),
                "current_metrics": risk_metrics
            })
        
        return should_halt, reasons
    
    def pause_strategy(self, strategy_id: str, reason: str = "Manual pause") -> bool:
        """
        Pause a deployed strategy.
        
        Args:
            strategy_id: Strategy ID
            reason: Reason for pausing
            
        Returns:
            bool: True if successful
        """
        if strategy_id not in self.deployed_strategies:
            logger.error(f"Strategy {strategy_id} not deployed")
            return False
        
        # Update status
        self.deployed_strategies[strategy_id]["status"] = "paused"
        self.deployed_strategies[strategy_id]["pause_reason"] = reason
        self.deployed_strategies[strategy_id]["pause_time"] = datetime.now().isoformat()
        
        # Save state
        self._save_state()
        
        # Emit event
        self._emit_event(EventType.STRATEGY_PAUSED, {
            "strategy_id": strategy_id,
            "reason": reason
        })
        
        logger.info(f"Paused strategy {strategy_id}: {reason}")
        return True
    
    def resume_strategy(self, strategy_id: str) -> bool:
        """
        Resume a paused strategy.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            bool: True if successful
        """
        if strategy_id not in self.deployed_strategies:
            logger.error(f"Strategy {strategy_id} not deployed")
            return False
        
        if self.deployed_strategies[strategy_id]["status"] != "paused":
            logger.error(f"Strategy {strategy_id} is not paused")
            return False
        
        # Update status
        self.deployed_strategies[strategy_id]["status"] = "active"
        
        # Remove pause info
        if "pause_reason" in self.deployed_strategies[strategy_id]:
            del self.deployed_strategies[strategy_id]["pause_reason"]
        if "pause_time" in self.deployed_strategies[strategy_id]:
            del self.deployed_strategies[strategy_id]["pause_time"]
        
        # Save state
        self._save_state()
        
        # Emit event
        self._emit_event(EventType.STRATEGY_RESUMED, {
            "strategy_id": strategy_id
        })
        
        logger.info(f"Resumed strategy {strategy_id}")
        return True
    
    def calculate_position_size(self, strategy_id: str, symbol: str, 
                               entry_price: float, stop_price: float,
                               market_data: Dict[str, Any]) -> float:
        """
        Calculate risk-adjusted position size for a trade.
        
        Args:
            strategy_id: Strategy ID
            symbol: Trading symbol
            entry_price: Entry price
            stop_price: Stop-loss price
            market_data: Market data
            
        Returns:
            float: Position size (quantity)
        """
        if strategy_id not in self.deployed_strategies:
            logger.error(f"Strategy {strategy_id} not deployed")
            return 0
        
        # Get strategy allocation
        allocation = self.strategy_allocations.get(strategy_id, 5.0) / 100.0
        
        # Get account value
        account_value = self.risk_manager.portfolio_value
        
        # Calculate allocation amount
        allocation_amount = account_value * allocation
        
        # Calculate risk per trade (as percentage of allocation)
        risk_per_trade = 0.02  # 2% risk per trade
        
        # Calculate risk amount
        risk_amount = allocation_amount * risk_per_trade
        
        # Calculate trade risk (distance to stop)
        if entry_price > stop_price:  # Long position
            trade_risk_pct = (entry_price - stop_price) / entry_price
        else:  # Short position
            trade_risk_pct = (stop_price - entry_price) / entry_price
        
        # Avoid division by zero
        if trade_risk_pct == 0:
            trade_risk_pct = 0.01  # Default to 1%
        
        # Calculate position size in dollars
        position_size_dollars = risk_amount / trade_risk_pct
        
        # Calculate position size in shares
        position_size = position_size_dollars / entry_price
        
        logger.info(f"Calculated position size for {strategy_id} on {symbol}: {position_size:.2f} shares")
        return position_size
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive risk report for all autonomous strategies.
        
        Returns:
            Dict: Risk report
        """
        # Get overall risk metrics
        overall_metrics = self.risk_manager.get_risk_metrics()
        
        # Get strategy-specific metrics
        strategy_metrics = {
            strategy_id: {
                "allocation": self.strategy_allocations.get(strategy_id, 0),
                "status": self.deployed_strategies[strategy_id]["status"],
                "risk_metrics": self.risk_metrics.get(strategy_id, {}),
                "performance": self.deployed_strategies[strategy_id].get("performance", {})
            }
            for strategy_id in self.deployed_strategies
        }
        
        # Calculate correlations
        correlations = self._calculate_correlations()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_metrics": overall_metrics,
            "strategy_metrics": strategy_metrics,
            "correlations": correlations,
            "circuit_breakers": self.circuit_breakers,
            "total_strategies": len(self.deployed_strategies),
            "active_strategies": sum(1 for s in self.deployed_strategies.values() 
                                    if s["status"] == "active")
        }
    
    def _is_highly_correlated(self, strategy: StrategyCandidate) -> bool:
        """
        Check if a strategy is highly correlated with existing strategies.
        
        Args:
            strategy: Strategy candidate
            
        Returns:
            bool: True if highly correlated
        """
        # Skip if no deployed strategies
        if not self.deployed_strategies:
            return False
        
        # Simple heuristic: check if we already have a strategy of this type
        # trading the same symbols
        for existing_id, existing_info in self.deployed_strategies.items():
            existing_strategy = existing_info["strategy"]
            
            # Check type match
            if existing_strategy["strategy_type"] == strategy.strategy_type:
                # Check symbol overlap
                existing_symbols = set(existing_strategy["symbols"])
                new_symbols = set(strategy.symbols)
                
                if existing_symbols.intersection(new_symbols):
                    logger.warning(f"Strategy {strategy.strategy_id} has symbol overlap with {existing_id}")
                    return True
        
        return False
    
    def _calculate_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between deployed strategies.
        
        Returns:
            Dict: Correlation matrix
        """
        # This would ideally use actual returns data
        # For now, we'll use a placeholder implementation
        correlations = {}
        
        # Get list of active strategies
        active_strategies = [
            strategy_id for strategy_id, info in self.deployed_strategies.items()
            if info["status"] == "active"
        ]
        
        # For each pair, assign a correlation value
        for i, strategy_1 in enumerate(active_strategies):
            correlations[strategy_1] = {}
            
            for strategy_2 in active_strategies:
                if strategy_1 == strategy_2:
                    correlations[strategy_1][strategy_2] = 1.0
                else:
                    # In a real implementation, this would use actual return data
                    # For now, assign a random correlation
                    correlations[strategy_1][strategy_2] = 0.2
        
        return correlations
    
    def _handle_optimized_strategy(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle a strategy optimization event."""
        strategy_id = event_data.get("strategy_id")
        
        if not strategy_id:
            return
        
        logger.info(f"Strategy optimized: {strategy_id}")
        
        # We don't need to do anything here yet, but in the future, we could
        # automatically deploy strategies that meet certain criteria
    
    def _handle_deployed_strategy(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle a strategy deployment event."""
        strategy_id = event_data.get("strategy_id")
        
        if not strategy_id:
            return
        
        logger.info(f"Strategy deployed: {strategy_id}")
        
        # This is for events from external systems - we already track deployments
        # that go through our deploy_strategy method
    
    def _handle_risk_level_change(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle a risk level change event."""
        old_level = event_data.get("old_level")
        new_level = event_data.get("new_level")
        
        logger.info(f"Risk level changed from {old_level} to {new_level}")
        
        # If risk level increases, reduce allocations
        if new_level in [RiskLevel.HIGH, RiskLevel.EXTREME]:
            self._reduce_all_allocations(0.7)  # Reduce all by 30%
        elif new_level == RiskLevel.CRITICAL:
            self._reduce_all_allocations(0.5)  # Reduce all by 50%
    
    def _handle_circuit_breaker(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle a circuit breaker event."""
        reasons = event_data.get("reasons", [])
        
        logger.warning(f"Circuit breaker triggered: {reasons}")
        
        # Pause all strategies when circuit breaker triggered
        for strategy_id in self.deployed_strategies:
            if self.deployed_strategies[strategy_id]["status"] == "active":
                self.pause_strategy(strategy_id, reason="Circuit breaker: " + ", ".join(reasons))
    
    def _handle_trade_executed(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle a trade execution event."""
        strategy_id = event_data.get("strategy_id")
        symbol = event_data.get("symbol")
        quantity = event_data.get("quantity", 0)
        price = event_data.get("price", 0)
        
        if not strategy_id or strategy_id not in self.risk_metrics:
            return
        
        # Update risk metrics
        self.risk_metrics[strategy_id]["trade_count_today"] += 1
        self.risk_metrics[strategy_id]["position_count"] += 1
        self.risk_metrics[strategy_id]["largest_position_size"] = max(
            self.risk_metrics[strategy_id]["largest_position_size"],
            quantity * price
        )
        self.risk_metrics[strategy_id]["last_updated"] = datetime.now().isoformat()
        
        # Update strategy performance
        if strategy_id in self.deployed_strategies:
            self.deployed_strategies[strategy_id]["performance"]["trades"] += 1
        
        # Save state
        self._save_state()
    
    def _handle_position_closed(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Handle a position closed event."""
        strategy_id = event_data.get("strategy_id")
        pnl = event_data.get("profit_loss", 0)
        
        if not strategy_id or strategy_id not in self.risk_metrics:
            return
        
        # Update risk metrics
        self.risk_metrics[strategy_id]["position_count"] = max(
            0, self.risk_metrics[strategy_id]["position_count"] - 1
        )
        self.risk_metrics[strategy_id]["daily_profit_loss"] += pnl
        self.risk_metrics[strategy_id]["last_updated"] = datetime.now().isoformat()
        
        # Update strategy performance
        if strategy_id in self.deployed_strategies:
            perf = self.deployed_strategies[strategy_id]["performance"]
            perf["profit_loss"] = perf.get("profit_loss", 0) + pnl
            
            # Update win rate
            win = pnl > 0
            wins = perf.get("wins", 0) + (1 if win else 0)
            total = perf.get("trades", 0)
            perf["win_rate"] = (wins / total) * 100 if total > 0 else 0
            perf["wins"] = wins
        
        # Save state
        self._save_state()
    
    def _reduce_all_allocations(self, factor: float) -> None:
        """Reduce all strategy allocations by a factor."""
        for strategy_id in self.strategy_allocations:
            current = self.strategy_allocations[strategy_id]
            new_allocation = current * factor
            self.strategy_allocations[strategy_id] = new_allocation
            
            if strategy_id in self.deployed_strategies:
                self.deployed_strategies[strategy_id]["risk_params"]["allocation_percentage"] = new_allocation
        
        # Save state
        self._save_state()
        
        logger.info(f"Reduced all allocations by factor {factor}")
    
    def _load_state(self) -> None:
        """Load state from disk."""
        file_path = os.path.join(self.data_dir, "autonomous_risk_state.json")
        
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.deployed_strategies = state.get("deployed_strategies", {})
            self.strategy_allocations = state.get("strategy_allocations", {})
            self.risk_metrics = state.get("risk_metrics", {})
            self.circuit_breakers = state.get("circuit_breakers", self.circuit_breakers)
            
            logger.info("Loaded risk state from disk")
        except Exception as e:
            logger.error(f"Error loading risk state: {e}")
    
    def _save_state(self) -> None:
        """Save state to disk."""
        file_path = os.path.join(self.data_dir, "autonomous_risk_state.json")
        
        try:
            state = {
                "deployed_strategies": self.deployed_strategies,
                "strategy_allocations": self.strategy_allocations,
                "risk_metrics": self.risk_metrics,
                "circuit_breakers": self.circuit_breakers,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.debug("Saved risk state to disk")
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the event bus."""
        try:
            event = Event(
                event_type=event_type,
                source="AutonomousRiskManager",
                data=data,
                timestamp=datetime.now()
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")


# Singleton instance for global access
_autonomous_risk_manager = None

def get_autonomous_risk_manager(risk_config=None, data_dir=None) -> AutonomousRiskManager:
    """
    Get the singleton instance of the AutonomousRiskManager.
    
    Args:
        risk_config: Risk configuration (only used on first call)
        data_dir: Data directory (only used on first call)
        
    Returns:
        AutonomousRiskManager: The singleton instance
    """
    global _autonomous_risk_manager
    
    if _autonomous_risk_manager is None:
        _autonomous_risk_manager = AutonomousRiskManager(risk_config, data_dir)
    
    return _autonomous_risk_manager
