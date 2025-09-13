#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Integration Module for BensBot

This module provides initialization and integration functions
for the enhanced reliability and efficiency components:
1. Persistence Layer (MongoDB)
2. Watchdog & Fault Tolerance
3. Dynamic Capital Scaling
4. Strategy Lifecycle Management
5. Execution Quality Modeling
6. Event-Driven Communication
7. Strategy Intelligence Recording
8. Live Data Integration
"""
import logging
import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime

from trading_bot.data.persistence import PersistenceManager
from trading_bot.core.watchdog import ServiceWatchdog
from trading_bot.risk.capital_manager import DynamicCapitalManager
from trading_bot.core.strategy_manager import StrategyPerformanceManager
from trading_bot.execution.execution_model import ExecutionQualityModel
from trading_bot.core.event_bus import EventBus, get_global_event_bus, Event
from trading_bot.core.constants import EventType, HealthStatus
from trading_bot.core.strategy_intelligence_recorder import StrategyIntelligenceRecorder
from trading_bot.risk.risk_management_engine import RiskManagementEngine
from trading_bot.risk.risk_based_strategy_rotation import RiskBasedStrategyRotation

# Import conditionally to allow for it to be implemented later
try:
    from trading_bot.data.live_data_source import LiveDataSource
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False
    # Define a stub class for type checking
    class LiveDataSource:
        pass

logger = logging.getLogger(__name__)

class EnhancedComponents:
    """
    Container for all enhanced components integrated into the main system.
    Provides initialization and access to these components.
    """
    
    def __init__(self):
        """Initialize enhanced components container."""
        self.persistence: Optional[PersistenceManager] = None
        self.watchdog: Optional[ServiceWatchdog] = None
        self.capital_manager: Optional[DynamicCapitalManager] = None
        self.strategy_manager: Optional[StrategyPerformanceManager] = None
        self.execution_model: Optional[ExecutionQualityModel] = None
        self.event_bus: Optional[EventBus] = None
        self.intelligence_recorder: Optional[StrategyIntelligenceRecorder] = None
        self.risk_engine: Optional[RiskManagementEngine] = None
        self.risk_strategy_rotation: Optional[RiskBasedStrategyRotation] = None
        self.live_data_source: Optional[LiveDataSource] = None
        self.initialized = False
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize all enhanced components with the provided configuration.
        
        Args:
            config: Configuration dictionary for all enhanced components
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize event bus first as others depend on it
            self.event_bus = get_global_event_bus()
            logger.info("Initialized global event bus")
            
            # Initialize persistence
            mongo_uri = config.get('mongodb_uri', os.environ.get('MONGODB_URI', 'mongodb://localhost:27017'))
            db_name = config.get('mongodb_database', os.environ.get('MONGODB_DATABASE', 'bensbot'))
            
            self.persistence = PersistenceManager(mongo_uri, db_name)
            logger.info(f"Initialized persistence manager with database {db_name}")
            
            # Initialize strategy intelligence recorder
            self.intelligence_recorder = StrategyIntelligenceRecorder(self.persistence, self.event_bus)
            logger.info("Initialized strategy intelligence recorder")
            
            # Initialize sample data if needed
            if config.get('initialize_mock_data', False):
                self.intelligence_recorder.initialize_mock_data()
                logger.info("Initialized mock intelligence data")
            
            # Initialize watchdog
            self.watchdog = ServiceWatchdog(
                check_interval=config.get('watchdog_check_interval', 30),
                persistence_manager=self.persistence
            )
            self.register_watchdog_event_handlers()
            logger.info("Initialized service watchdog")
            
            # Initialize risk management engine
            risk_config = config.get('risk_management', {
                'max_portfolio_risk': 0.05,  # 5% maximum portfolio risk
                'correlation_threshold': 0.7,  # Alert on correlations above 0.7
                'max_position_size': 0.2,  # No position can be > 20% of portfolio
                'drawdown_threshold': 0.1,  # Alert on 10% drawdowns
                'risk_per_trade': 0.01,  # Risk 1% per trade
            })
            self.risk_engine = RiskManagementEngine(risk_config, self.persistence)
            self.register_risk_engine_event_handlers()
            logger.info("Initialized risk management engine")
            
            # Initialize risk-based strategy rotation
            if self.strategy_manager:
                rotation_config = config.get('risk_strategy_rotation', {
                    'max_active_strategies': 5,
                    'risk_factor_weights': {
                        'market_beta': 0.2,
                        'sector_exposure': 0.2,
                        'volatility': 0.3,
                        'correlation': 0.2,
                        'liquidity': 0.1
                    }
                })
                self.risk_strategy_rotation = RiskBasedStrategyRotation(
                    strategy_manager=self.strategy_manager,
                    event_bus=self.event_bus,
                    config=rotation_config
                )
                logger.info("Initialized risk-based strategy rotation system")
            else:
                logger.warning("Strategy manager not available, risk-based strategy rotation disabled")
            
            # Initialize capital manager
            initial_capital = config.get('initial_capital', 10000.0)
            risk_params = config.get('risk_parameters', {})
            self.capital_manager = DynamicCapitalManager(
                initial_capital=initial_capital,
                persistence_manager=self.persistence,
                **risk_params
            )
            self.register_capital_manager_event_handlers()
            logger.info(f"Initialized dynamic capital manager with {initial_capital} capital")
            
            # Initialize strategy manager
            strategy_metrics = config.get('strategy_metrics', {})
            self.strategy_manager = StrategyPerformanceManager(
                persistence_manager=self.persistence,
                **strategy_metrics
            )
            self.register_strategy_manager_event_handlers()
            logger.info("Initialized strategy performance manager")
            
            # Initialize execution model
            execution_params = config.get('execution_parameters', {})
            self.execution_model = ExecutionQualityModel(
                persistence_manager=self.persistence,
                **execution_params
            )
            self.register_execution_model_event_handlers()
            logger.info("Initialized execution quality model")
            
            # Initialize live data source if configured and available
            if config.get('use_live_data', False) and LIVE_DATA_AVAILABLE:
                data_source_config = config.get('live_data_config', {})
                self.live_data_source = LiveDataSource(
                    persistence_manager=self.persistence,
                    **data_source_config
                )
                self.register_live_data_event_handlers()
                logger.info("Initialized live data source")
            
            self.initialized = True
            
            # Publish system initialized event
            self.event_bus.create_and_publish(
                event_type=EventType.SYSTEM_STARTED,
                data={"components": [
                    "persistence", "watchdog", "capital_manager", 
                    "strategy_manager", "execution_model", "intelligence_recorder"
                ]},
                source="enhanced_integration"
            )
            
            logger.info("All enhanced components initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing enhanced components: {str(e)}")
            return False
    
    def register_watchdog_event_handlers(self):
        """Register event handlers for the watchdog component."""
        if not self.watchdog or not self.event_bus:
            return
        
        # Define status change callback
        def on_health_status_change(service_name: str, status: str, details: str):
            self.event_bus.create_and_publish(
                event_type=EventType.HEALTH_STATUS_CHANGED,
                data={
                    "service_name": service_name,
                    "status": status,
                    "details": details,
                    "timestamp": datetime.now()
                },
                source="watchdog"
            )
        
        # Set the callback in watchdog service
        self.watchdog.set_status_change_callback(on_health_status_change)
        
        # Subscribe to health check events
        def on_health_check_event(event: Event):
            if self.watchdog:
                result = self.watchdog.run_health_check()
                # Publish results
                self.event_bus.create_and_publish(
                    event_type=EventType.HEALTH_CHECK,
                    data={
                        "results": result,
                        "timestamp": datetime.now()
                    },
                    source="watchdog"
                )
        
        self.event_bus.subscribe(EventType.HEALTH_CHECK, on_health_check_event)
    
    def register_capital_manager_event_handlers(self):
        """Register event handlers for the capital manager component."""
        if not self.capital_manager or not self.event_bus:
            return
            
        # Add method to capital manager to publish capital events
        def publish_capital_event(self, adjustment_amount: float, reason: str):
            self.event_bus.create_and_publish(
                event_type=EventType.CAPITAL_ADJUSTED,
                data={
                    "previous_capital": self.total_capital - adjustment_amount,
                    "new_capital": self.total_capital,
                    "adjustment": adjustment_amount,
                    "reason": reason,
                    "timestamp": datetime.now()
                },
                source="capital_manager"
            )
            
        # Patch method to the original capital manager's publish method
        # This is a bit of a hack, but it avoids modifying the original class
        self.capital_manager.publish_capital_event = publish_capital_event.__get__(self.capital_manager)
        
        # Override adjust_capital method to publish events
        original_adjust_capital = self.capital_manager.adjust_capital
        
        def adjusted_adjust_capital(self, amount: float, reason: str = "") -> float:
            result = original_adjust_capital(amount, reason)
            self.publish_capital_event(amount, reason)
            return result
        
        self.capital_manager.adjust_capital = adjusted_adjust_capital.__get__(self.capital_manager)
    
    def register_strategy_manager_event_handlers(self):
        """Register event handlers for the strategy manager component."""
        if not self.strategy_manager or not self.event_bus:
            return
            
        # Add method to strategy manager to publish strategy lifecycle events
        def publish_strategy_event(self, strategy_id: str, event_type: EventType, data: Dict):
            event_data = {"strategy_id": strategy_id, "timestamp": datetime.now()}
            event_data.update(data)
            self.event_bus.create_and_publish(
                event_type=event_type,
                data=event_data,
                source="strategy_manager"
            )
            
        # Patch method to strategy manager
        self.strategy_manager.publish_strategy_event = publish_strategy_event.__get__(self.strategy_manager)
        
        # Override promote_strategy method to publish events
        original_promote = self.strategy_manager.promote_strategy
        
        def adjusted_promote_strategy(self, strategy_id: str, reason: str = ""):
            result = original_promote(strategy_id, reason)
            self.publish_strategy_event(
                strategy_id, 
                EventType.STRATEGY_PROMOTED, 
                {"reason": reason}
            )
            return result
        
        self.strategy_manager.promote_strategy = adjusted_promote_strategy.__get__(self.strategy_manager)
        
        # Override retire_strategy method to publish events
        original_retire = self.strategy_manager.retire_strategy
        
        def adjusted_retire_strategy(self, strategy_id: str, reason: str = ""):
            result = original_retire(strategy_id, reason)
            self.publish_strategy_event(
                strategy_id, 
                EventType.STRATEGY_RETIRED, 
                {"reason": reason}
            )
            return result
        
        self.strategy_manager.retire_strategy = adjusted_retire_strategy.__get__(self.strategy_manager)
    
    def register_execution_model_event_handlers(self):
        """Register event handlers for the execution model component."""
        if not self.execution_model or not self.event_bus:
            return
            
        # Add method to execution model to publish execution quality events
        def publish_execution_quality(self, metrics: Dict):
            metrics_list = [
                {"metric": key, "expected": value["expected"], "actual": value["actual"]}
                for key, value in metrics.items()
            ]
            self.event_bus.create_and_publish(
                event_type=EventType.EXECUTION_QUALITY_MEASURED,
                data={
                    "metrics": metrics_list,
                    "timestamp": datetime.now()
                },
                source="execution_model"
            )
            
        # Patch method to execution model
        self.execution_model.publish_execution_quality = publish_execution_quality.__get__(self.execution_model)
        
        # Override measure_execution_quality method to publish events
        original_measure = self.execution_model.measure_execution_quality
        
        def adjusted_measure_execution_quality(self, order_id: str):
            metrics = original_measure(order_id)
            self.publish_execution_quality(metrics)
            return metrics
        
        self.execution_model.measure_execution_quality = adjusted_measure_execution_quality.__get__(self.execution_model)
    
    def register_live_data_event_handlers(self):
        """Register event handlers for the live data source component."""
        if not self.live_data_source or not self.event_bus:
            return
            
        # Add event publishing method to live data source
        def publish_data_event(self, event_type: EventType, data: Dict):
            self.event_bus.create_and_publish(
                event_type=event_type,
                data=data,
                source="live_data_source"
            )
            
        # Patch method to live data source
        self.live_data_source.publish_data_event = publish_data_event.__get__(self.live_data_source)
    
    def register_risk_engine_event_handlers(self):
        """Register event handlers for the risk management engine component."""
        if not self.risk_engine or not self.event_bus:
            return
            
        # Register the risk engine's event handlers
        self.risk_engine.register_event_handlers()
        
        # Register for system updates that impact risk management
        self.event_bus.subscribe(EventType.TRADE_EXECUTED, 
                               lambda event: self.handle_trade_risk_update(event))
        self.event_bus.subscribe(EventType.POSITION_UPDATED, 
                               lambda event: self.handle_position_risk_update(event))
        self.event_bus.subscribe(EventType.PORTFOLIO_UPDATED, 
                               lambda event: self.handle_portfolio_risk_update(event))
        
        logger.info("Risk management engine event handlers registered")
        
    def handle_trade_risk_update(self, event: Event):
        """Handle trade execution events for risk updates."""
        if not self.risk_engine or not event.data:
            return
            
        trade_data = event.data
        if all(k in trade_data for k in ['symbol', 'quantity', 'price']):
            # A new trade was executed, assess its risk
            if 'stop_loss' in trade_data:
                stop_loss_price = trade_data['stop_loss']
            else:
                # If no stop loss provided, use a default of 5% from entry price
                stop_loss_price = trade_data['price'] * 0.95 if trade_data['quantity'] > 0 else trade_data['price'] * 1.05
                
            self.risk_engine.assess_position_risk(
                symbol=trade_data['symbol'],
                position_size=trade_data['quantity'],
                entry_price=trade_data['price'],
                stop_loss_price=stop_loss_price
            )
            
    def handle_position_risk_update(self, event: Event):
        """Handle position update events for risk assessment."""
        if not self.risk_engine or not event.data:
            return
            
        # When positions are updated, reassess portfolio risk
        self.risk_engine.assess_portfolio_risk()
        
    def handle_portfolio_risk_update(self, event: Event):
        """Handle portfolio update events for risk assessment."""
        if not self.risk_engine or not event.data:
            return
            
        # Check for drawdown when portfolio value updated
        if 'portfolio_value' in event.data:
            self.risk_engine.monitor_drawdown(event.data['portfolio_value'])
    
    def start_services(self) -> bool:
        """
        Start all services that need to run continuously.
        
        Returns:
            True if all services started successfully, False otherwise
        """
        success = True
        
        try:
            # Start watchdog
            if self.watchdog:
                self.watchdog.start()
                logger.info("Service watchdog started")
                
            # Load states if persistence is available
            if self.persistence and hasattr(self.persistence, 'is_connected') and self.persistence.is_connected():
                self.load_all_states()
                logger.info("All component states loaded")
            
            # Start live data if available
            if self.live_data_source and hasattr(self.live_data_source, 'start'):
                self.live_data_source.start()
                logger.info("Live data service started")
                
            # Start risk engine if available
            if self.risk_engine and hasattr(self.risk_engine, 'start'):
                self.risk_engine.start()
                logger.info("Risk management engine started")
                
            # Register risk engine with watchdog
            if self.risk_engine and self.watchdog:
                self.register_with_watchdog('risk_management', lambda: True)
                logger.info("Risk management engine registered with watchdog")
                
            # Publish system started event
            if self.event_bus:
                self.event_bus.create_and_publish(
                    event_type=EventType.SYSTEM_STARTED,
                    data={"timestamp": datetime.now()},
                    source="enhanced_integration"
                )
                
        except Exception as e:
            logger.error(f"Error starting services: {str(e)}")
            success = False
            
        return success
    
    def stop_services(self) -> bool:
        """
        Stop all running services.
        
        Returns:
            True if all services stopped successfully, False otherwise
        """
        try:
            # Stop the watchdog service
            if self.watchdog:
                self.watchdog.stop()
                logger.info("Watchdog service stopped")
            
            # Stop live data source if initialized
            if self.live_data_source and hasattr(self.live_data_source, 'stop'):
                self.live_data_source.stop()
                logger.info("Live data source stopped")
                
            # Stop risk engine if initialized
            if self.risk_engine and hasattr(self.risk_engine, 'stop'):
                self.risk_engine.stop()
                logger.info("Risk management engine stopped")
            
            # Publish service stop event
            if self.event_bus:
                self.event_bus.create_and_publish(
                    event_type=EventType.SYSTEM_STOPPED,
                    data={"timestamp": datetime.now()},
                    source="enhanced_integration"
                )
            
            return True
        
        except Exception as e:
            logger.error(f"Error stopping enhanced services: {str(e)}")
            return False
    
    def register_with_watchdog(self, service_name: str, health_check_func) -> bool:
        """
        Register a service with the watchdog for health monitoring.
        
        Args:
            service_name: Name of the service to monitor
            health_check_func: Function that returns True if service is healthy
        
        Returns:
            True if registration was successful, False otherwise
        """
        if not self.watchdog:
            logger.error("Cannot register service: watchdog not initialized")
            return False
        
        try:
            self.watchdog.register_service(service_name, health_check_func)
            logger.info(f"Registered service '{service_name}' with watchdog")
            
            # Publish registration event
            if self.event_bus:
                self.event_bus.create_and_publish(
                    event_type=EventType.HEALTH_CHECK,
                    data={
                        "service_name": service_name, 
                        "action": "registered",
                        "timestamp": datetime.now()
                    },
                    source="enhanced_integration"
                )
            
            return True
        except Exception as e:
            logger.error(f"Error registering service with watchdog: {str(e)}")
            return False
    
    def save_all_states(self) -> bool:
        """
        Save states for all components to persistence.
        
        Returns:
            True if all states saved successfully, False otherwise
        """
        if not self.persistence or not hasattr(self.persistence, 'is_connected') or not self.persistence.is_connected():
            logger.warning("Cannot save states: Persistence not available")
            return False
            
        success = True
        
        try:
            # Save capital manager state
            if self.capital_manager and hasattr(self.capital_manager, 'save_state'):
                self.capital_manager.save_state()
                logger.info("Capital manager state saved")
            
            # Save strategy manager state
            if self.strategy_manager and hasattr(self.strategy_manager, 'save_state'):
                self.strategy_manager.save_state()
                logger.info("Strategy manager state saved")
                
            # Save risk engine state if it has a save_state method
            if self.risk_engine and hasattr(self.risk_engine, 'save_state'):
                self.risk_engine.save_state()
                logger.info("Risk management engine state saved")
            
            # Publish state saved event
            if self.event_bus:
                self.event_bus.create_and_publish(
                    event_type=EventType.SYSTEM_STATE_SAVED,
                    data={"timestamp": datetime.now()},
                    source="enhanced_integration"
                )
                
        except Exception as e:
            logger.error(f"Error saving component states: {str(e)}")
            success = False
            
        return success
    
    def load_all_states(self) -> bool:
        """
        Load states for all components from persistence.
        
        Returns:
            True if all states loaded successfully, False otherwise
        """
        if not self.persistence or not hasattr(self.persistence, 'is_connected') or not self.persistence.is_connected():
            logger.warning("Cannot load states: Persistence not available")
            return False
            
        success = True
        
        try:
            # Load capital manager state
            if self.capital_manager and hasattr(self.capital_manager, 'load_state'):
                state = self.persistence.load_strategy_state('capital_manager')
                if state:
                    self.capital_manager.load_state(state)
                    logger.info("Capital manager state loaded")
            
            # Load strategy manager state
            if self.strategy_manager and hasattr(self.strategy_manager, 'load_state'):
                state = self.persistence.load_strategy_state('strategy_manager')
                if state:
                    self.strategy_manager.load_state(state)
                    logger.info("Strategy manager state loaded")
                    
            # Load risk engine state if it has a load_state method
            if self.risk_engine and hasattr(self.risk_engine, 'load_state'):
                state = self.persistence.load_strategy_state('risk_management_engine')
                if state:
                    self.risk_engine.load_state(state)
                    logger.info("Risk management engine state loaded")
            
            # Publish state loaded event
            if self.event_bus:
                self.event_bus.create_and_publish(
                    event_type=EventType.SYSTEM_STATE_LOADED,
                    data={"timestamp": datetime.now()},
                    source="enhanced_integration"
                )
                
        except Exception as e:
            logger.error(f"Error loading component states: {str(e)}")
            success = False
            
        return success

# Create a demonstration function
def create_demo_intelligence_data():
    """
    Create a demonstration of strategy intelligence data for testing the dashboard.
    This creates and initializes the components needed for strategy intelligence.
    
    Returns:
        EnhancedComponents instance with mock data initialized
    """
    # Create configuration
    config = {
        'mongodb_uri': os.environ.get('MONGODB_URI', 'mongodb://localhost:27017'),
        'mongodb_database': os.environ.get('MONGODB_DATABASE', 'bensbot'),
        'initialize_mock_data': True  # This will generate mock data
    }
    
    # Initialize components
    components = EnhancedComponents()
    components.initialize(config)
    
    return components


def example_usage():
    """Example of how to use the enhanced components."""
    # Example configuration
    config = {
        'mongodb_uri': os.environ.get('MONGODB_URI', 'mongodb://localhost:27017'),
        'mongodb_database': os.environ.get('MONGODB_DATABASE', 'bensbot'),
    }
    
    # Initialize components
    components = EnhancedComponents()
    components.initialize(config)
    
    # Start services
    components.start_services()
    
    # When shutting down
    components.stop_services()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()
