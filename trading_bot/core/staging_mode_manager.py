"""
Staging Mode Manager for enforcing paper-only trading during testing phase.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta

from trading_bot.core.service_registry import ServiceRegistry
from trading_bot.core.constants import StrategyPhase, StrategyStatus
from trading_bot.brokers.paper_broker_factory import create_paper_broker

logger = logging.getLogger(__name__)

class StagingModeManager:
    """
    Manages a staging environment for safely testing strategies before live deployment.
    
    This class enforces:
    - All strategies use paper trading only
    - Enhanced logging and monitoring
    - Extended testing periods for all strategies
    - Validation checkpoints before promotion
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the staging environment manager.
        
        Args:
            config_path: Path to staging configuration file
        """
        self.service_registry = ServiceRegistry.get_instance()
        self.config = self._load_config(config_path)
        self.staging_start_time = datetime.now()
        self.forced_paper_strategies: Set[str] = set()
        self.original_brokers: Dict[str, str] = {}
        self.test_duration_days = self.config.get("test_duration_days", 14)
        self.min_trades_required = self.config.get("min_trades_required", 30)
        
        # Register with service registry
        self.service_registry.register_service("staging_mode_manager", self)
        
        logger.info(f"Staging environment initialized with {self.test_duration_days} days test duration")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load staging configuration from file or use defaults."""
        default_config = {
            "test_duration_days": 14,
            "min_trades_required": 30,
            "reporting_frequency_hours": 24,
            "memory_alert_threshold_mb": 500,
            "cpu_alert_threshold_pct": 70,
            "max_acceptable_error_rate": 0.01,
            "risk_tolerance_multiplier": 0.8,
            "enable_stress_testing": True,
            "validation_checkpoints": {
                "min_sharpe_ratio": 0.8,
                "max_drawdown_pct": -10.0,
                "min_win_rate": 0.4,
                "max_daily_loss_pct": -3.0,
                "resource_utilization_threshold": 80
            }
        }
        
        if not config_path:
            logger.info("Using default staging configuration")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                
            # Merge with defaults to ensure all keys exist
            merged_config = {**default_config, **custom_config}
            
            # If custom validation checkpoints provided, merge those too
            if "validation_checkpoints" in custom_config:
                merged_config["validation_checkpoints"] = {
                    **default_config["validation_checkpoints"],
                    **custom_config["validation_checkpoints"]
                }
                
            logger.info(f"Loaded custom staging configuration from {config_path}")
            return merged_config
        except Exception as e:
            logger.error(f"Error loading staging configuration: {str(e)}")
            return default_config
    
    def enable_staging_mode(self) -> None:
        """
        Enable staging mode for the entire trading system.
        
        This forces all strategies to use paper trading regardless of their settings.
        """
        logger.info("Enabling staging mode - all strategies will be forced to paper trading")
        
        # Get necessary services
        workflow = self.service_registry.get_service("strategy_trial_workflow")
        broker_router = self.service_registry.get_service("strategy_broker_router")
        
        if not workflow or not broker_router:
            logger.error("Required services not available")
            return
        
        # Store current broker assignments for all strategies
        all_strategies = workflow.get_all_strategies()
        
        for strategy in all_strategies:
            strategy_id = strategy.get("id")
            
            if not strategy_id:
                continue
                
            # Remember original broker
            original_broker = broker_router.get_broker_id_for_strategy(strategy_id)
            if original_broker:
                self.original_brokers[strategy_id] = original_broker
            
            # Force to paper trading
            self._force_paper_trading(strategy_id, broker_router)
            
            # Add to tracked list
            self.forced_paper_strategies.add(strategy_id)
        
        logger.info(f"Forced {len(self.forced_paper_strategies)} strategies to paper trading")
    
    def _force_paper_trading(self, strategy_id: str, broker_router) -> None:
        """Force a strategy to use paper trading."""
        # Create paper broker if needed
        paper_broker_id = f"paper_staging_{strategy_id}"
        
        # Check if this paper broker already exists
        if not broker_router.get_broker(paper_broker_id):
            # Create a new paper broker
            create_paper_broker(
                broker_id=paper_broker_id,
                name=f"Staging Paper Broker - {strategy_id}",
                initial_balance=100000.0
            )
        
        # Assign strategy to paper broker
        broker_router.assign_strategy_to_broker(strategy_id, paper_broker_id)
        logger.info(f"Assigned strategy {strategy_id} to paper broker {paper_broker_id}")
    
    def disable_staging_mode(self) -> None:
        """
        Disable staging mode and restore original broker assignments.
        """
        logger.info("Disabling staging mode")
        
        broker_router = self.service_registry.get_service("strategy_broker_router")
        if not broker_router:
            logger.error("Broker router service not available")
            return
        
        # Restore original brokers
        for strategy_id, original_broker in self.original_brokers.items():
            broker_router.assign_strategy_to_broker(strategy_id, original_broker)
            logger.info(f"Restored strategy {strategy_id} to original broker {original_broker}")
        
        # Clear tracking lists
        self.forced_paper_strategies.clear()
        self.original_brokers.clear()
        
        logger.info("Staging mode disabled")
    
    def is_in_staging_mode(self) -> bool:
        """Check if staging mode is currently active."""
        return len(self.forced_paper_strategies) > 0
    
    def get_staging_config(self) -> Dict[str, Any]:
        """Get the current staging configuration."""
        return self.config
    
    def get_staging_duration(self) -> timedelta:
        """Get the current duration of the staging period."""
        return datetime.now() - self.staging_start_time
    
    def has_met_minimum_duration(self) -> bool:
        """Check if the minimum staging duration has been met."""
        days_in_staging = (datetime.now() - self.staging_start_time).days
        return days_in_staging >= self.test_duration_days
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update staging configuration."""
        self.config.update(new_config)
        logger.info("Staging configuration updated")
        
        # Update derived properties
        if "test_duration_days" in new_config:
            self.test_duration_days = new_config["test_duration_days"]
        
        if "min_trades_required" in new_config:
            self.min_trades_required = new_config["min_trades_required"]
