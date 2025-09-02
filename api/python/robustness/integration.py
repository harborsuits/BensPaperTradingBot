"""
Robustness Integration Module

This module provides a centralized way to apply robustness safeguards to all core
components of the BensBot trading system. It initializes the system safeguards
and connects them to each component, ensuring comprehensive error detection,
recovery, and protection throughout the system.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Union, Tuple

# Import safeguard components
from trading_bot.robustness.system_safeguards import SystemSafeguards, ComponentType
from trading_bot.robustness.position_manager_safeguards import PositionManagerSafeguards, recover_position_manager, create_position_manager_validator
from trading_bot.robustness.trade_accounting_safeguards import TradeAccountingSafeguards, recover_trade_accounting, create_trade_accounting_validator
from trading_bot.robustness.exit_manager_safeguards import ExitManagerSafeguards, recover_exit_manager, create_exit_manager_validator
from trading_bot.robustness.capital_allocator_safeguards import CapitalAllocatorSafeguards, recover_capital_allocator, create_capital_allocator_validator

# Import core system components
from trading_bot.position.position_manager import PositionManager
from trading_bot.accounting.trade_accounting import TradeAccounting
from trading_bot.accounting.pnl_calculator import PnLCalculator
from trading_bot.accounting.performance_metrics import PerformanceMetrics
from trading_bot.strategy.exit_manager import ExitStrategyManager
from trading_bot.portfolio.capital_allocator import CapitalAllocator
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager

logger = logging.getLogger(__name__)

class RobustnessManager:
    """
    Central manager for system-wide robustness components.
    
    This class initializes and manages safeguards for all core components,
    provides a unified interface for health monitoring, and coordinates
    recovery procedures across the trading system.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the robustness manager.
        
        Args:
            config_path: Path to robustness configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize system safeguards
        self.system_safeguards = SystemSafeguards(self.config.get('system', {}))
        
        # Component safeguards
        self.position_safeguards: Optional[PositionManagerSafeguards] = None
        self.accounting_safeguards: Optional[TradeAccountingSafeguards] = None
        self.exit_safeguards: Optional[ExitManagerSafeguards] = None
        self.allocator_safeguards: Optional[CapitalAllocatorSafeguards] = None
        
        # Component references
        self.position_manager: Optional[PositionManager] = None
        self.trade_accounting: Optional[TradeAccounting] = None
        self.pnl_calculator: Optional[PnLCalculator] = None 
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self.exit_manager: Optional[ExitStrategyManager] = None
        self.capital_allocator: Optional[CapitalAllocator] = None
        self.broker_manager: Optional[MultiBrokerManager] = None
        
        # State tracking
        self.initialized_components = set()
        self.safeguard_status = {}
        
        logger.info("Robustness manager initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load robustness configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict with configuration
        """
        default_config = {
            'system': {
                'monitoring_interval_seconds': 30,
                'error_threshold': 5,
                'recovery_attempt_limit': 3,
                'recovery_cooldown_seconds': 300,
                'memory_threshold_mb': 1000,
                'cpu_threshold_pct': 80
            },
            'position_manager': {
                'checkpoint_interval_minutes': 30,
                'max_reconciliation_hours': 24
            },
            'trade_accounting': {
                'backup_interval_hours': 6,
                'max_error_history': 20,
                'db_backup_directory': 'db_backups',
                'max_backup_history': 10,
                'auto_fix_reconciliation': True
            },
            'exit_manager': {
                'exit_error_threshold': 5,
                'monitoring_error_threshold': 5,
                'max_exit_retries': 3,
                'max_seconds_between_checks': 300
            },
            'capital_allocator': {
                'max_allocation_change': 0.20,
                'min_allocation': 0.01,
                'max_allocation': 0.50,
                'smoothing_enabled': True,
                'smoothing_factor': 0.25,
                'max_hours_since_allocation': 24,
                'normal_reserved_capital': 0.10,
                'elevated_reserved_capital': 0.20,
                'high_reserved_capital': 0.30,
                'extreme_reserved_capital': 0.50
            }
        }
        
        if not config_path or not os.path.exists(config_path):
            logger.info("Using default robustness configuration")
            return default_config
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Merge with defaults
            for section in default_config:
                if section not in config:
                    config[section] = default_config[section]
                else:
                    for key, value in default_config[section].items():
                        if key not in config[section]:
                            config[section][key] = value
            
            logger.info(f"Loaded robustness configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}, using defaults")
            return default_config
    
    def initialize_position_safeguards(self, position_manager: PositionManager) -> None:
        """
        Initialize position manager safeguards.
        
        Args:
            position_manager: The position manager to enhance
        """
        if position_manager is None:
            logger.error("Cannot initialize position safeguards: position_manager is None")
            return
            
        try:
            self.position_manager = position_manager
            self.position_safeguards = PositionManagerSafeguards(
                position_manager,
                config=self.config.get('position_manager', {})
            )
            
            # Register with system safeguards
            self.system_safeguards.register_component(
                ComponentType.POSITION_MANAGER,
                position_manager,
                validation_function=create_position_manager_validator(self.position_safeguards),
                recovery_function=lambda: recover_position_manager(self.position_safeguards)
            )
            
            self.initialized_components.add('position_manager')
            self.safeguard_status['position_manager'] = {
                'initialized': True,
                'timestamp': self.position_safeguards._create_state_backup()
            }
            
            logger.info("Position manager safeguards initialized")
            
        except Exception as e:
            logger.error(f"Error initializing position safeguards: {str(e)}")
    
    def initialize_accounting_safeguards(
        self, 
        trade_accounting: TradeAccounting,
        pnl_calculator: Optional[PnLCalculator] = None,
        performance_metrics: Optional[PerformanceMetrics] = None
    ) -> None:
        """
        Initialize trade accounting safeguards.
        
        Args:
            trade_accounting: The trade accounting component to enhance
            pnl_calculator: Optional PnL calculator component
            performance_metrics: Optional performance metrics component
        """
        if trade_accounting is None:
            logger.error("Cannot initialize accounting safeguards: trade_accounting is None")
            return
            
        try:
            self.trade_accounting = trade_accounting
            self.pnl_calculator = pnl_calculator
            self.performance_metrics = performance_metrics
            
            self.accounting_safeguards = TradeAccountingSafeguards(
                trade_accounting,
                pnl_calculator,
                performance_metrics,
                config=self.config.get('trade_accounting', {})
            )
            
            # Create backup directory if specified
            backup_dir = self.config.get('trade_accounting', {}).get('db_backup_directory')
            if backup_dir and not os.path.exists(backup_dir):
                os.makedirs(backup_dir, exist_ok=True)
            
            # Register with system safeguards
            self.system_safeguards.register_component(
                ComponentType.TRADE_ACCOUNTING,
                trade_accounting,
                validation_function=create_trade_accounting_validator(self.accounting_safeguards),
                recovery_function=lambda: recover_trade_accounting(self.accounting_safeguards)
            )
            
            self.initialized_components.add('trade_accounting')
            self.safeguard_status['trade_accounting'] = {
                'initialized': True,
                'backup_created': self.accounting_safeguards.create_database_backup().get('status')
            }
            
            logger.info("Trade accounting safeguards initialized")
            
        except Exception as e:
            logger.error(f"Error initializing accounting safeguards: {str(e)}")
    
    def initialize_exit_safeguards(self, exit_manager: ExitStrategyManager) -> None:
        """
        Initialize exit strategy manager safeguards.
        
        Args:
            exit_manager: The exit strategy manager to enhance
        """
        if exit_manager is None:
            logger.error("Cannot initialize exit safeguards: exit_manager is None")
            return
            
        try:
            self.exit_manager = exit_manager
            self.exit_safeguards = ExitManagerSafeguards(
                exit_manager,
                config=self.config.get('exit_manager', {})
            )
            
            # Register with system safeguards
            self.system_safeguards.register_component(
                ComponentType.EXIT_MANAGER,
                exit_manager,
                validation_function=create_exit_manager_validator(self.exit_safeguards),
                recovery_function=lambda: recover_exit_manager(self.exit_safeguards)
            )
            
            self.initialized_components.add('exit_manager')
            self.safeguard_status['exit_manager'] = {
                'initialized': True
            }
            
            logger.info("Exit strategy manager safeguards initialized")
            
        except Exception as e:
            logger.error(f"Error initializing exit safeguards: {str(e)}")
    
    def initialize_allocator_safeguards(self, capital_allocator: CapitalAllocator) -> None:
        """
        Initialize capital allocator safeguards.
        
        Args:
            capital_allocator: The capital allocator to enhance
        """
        if capital_allocator is None:
            logger.error("Cannot initialize allocator safeguards: capital_allocator is None")
            return
            
        try:
            self.capital_allocator = capital_allocator
            self.allocator_safeguards = CapitalAllocatorSafeguards(
                capital_allocator,
                config=self.config.get('capital_allocator', {})
            )
            
            # Register with system safeguards
            self.system_safeguards.register_component(
                ComponentType.CAPITAL_ALLOCATOR,
                capital_allocator,
                validation_function=create_capital_allocator_validator(self.allocator_safeguards),
                recovery_function=lambda: recover_capital_allocator(self.allocator_safeguards)
            )
            
            self.initialized_components.add('capital_allocator')
            self.safeguard_status['capital_allocator'] = {
                'initialized': True,
                'risk_level': self.allocator_safeguards._get_active_risk_level()
            }
            
            logger.info("Capital allocator safeguards initialized")
            
        except Exception as e:
            logger.error(f"Error initializing allocator safeguards: {str(e)}")
    
    def register_broker_manager(self, broker_manager: MultiBrokerManager) -> None:
        """
        Register broker manager for system monitoring.
        
        Args:
            broker_manager: The broker manager to monitor
        """
        if broker_manager is None:
            logger.error("Cannot register broker manager: broker_manager is None")
            return
            
        try:
            self.broker_manager = broker_manager
            
            # Register with system safeguards
            self.system_safeguards.register_component(
                ComponentType.BROKER_MANAGER,
                broker_manager
            )
            
            self.initialized_components.add('broker_manager')
            logger.info("Broker manager registered with system safeguards")
            
        except Exception as e:
            logger.error(f"Error registering broker manager: {str(e)}")
    
    def start_monitoring(self) -> bool:
        """
        Start the system monitoring thread.
        
        Returns:
            bool: Success status
        """
        try:
            # Perform initial health check
            self.check_system_health()
            
            # Start monitoring thread
            return self.system_safeguards.start_monitoring()
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop the system monitoring thread.
        
        Returns:
            bool: Success status
        """
        try:
            return self.system_safeguards.stop_monitoring()
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")
            return False
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check health of all system components.
        
        Returns:
            Dict with health status
        """
        return self.system_safeguards.get_system_health()
    
    def create_checkpoint(self) -> Dict[str, Any]:
        """
        Create checkpoints/backups for all components.
        
        Returns:
            Dict with checkpoint results
        """
        results = {
            'timestamp': str(self._get_current_time()),
            'components': {}
        }
        
        try:
            # Position manager checkpoint
            if 'position_manager' in self.initialized_components and self.position_safeguards:
                try:
                    checkpoint = self.position_safeguards.create_checkpoint()
                    results['components']['position_manager'] = {
                        'status': 'success',
                        'timestamp': checkpoint.get('timestamp')
                    }
                except Exception as e:
                    results['components']['position_manager'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Trade accounting backup
            if 'trade_accounting' in self.initialized_components and self.accounting_safeguards:
                try:
                    backup_result = self.accounting_safeguards.create_database_backup()
                    results['components']['trade_accounting'] = {
                        'status': backup_result.get('status'),
                        'backup_path': backup_result.get('backup_path')
                    }
                except Exception as e:
                    results['components']['trade_accounting'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Capital allocator version
            if 'capital_allocator' in self.initialized_components and self.allocator_safeguards:
                try:
                    version_name = f"checkpoint_{self._get_current_time().strftime('%Y%m%d%H%M%S')}"
                    self.allocator_safeguards._capture_current_allocation(version_name)
                    results['components']['capital_allocator'] = {
                        'status': 'success',
                        'version': version_name
                    }
                except Exception as e:
                    results['components']['capital_allocator'] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            logger.info("Created system checkpoint")
            return results
            
        except Exception as e:
            logger.error(f"Error creating system checkpoint: {str(e)}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results
    
    def recover_system(self) -> Dict[str, Any]:
        """
        Attempt to recover all components from error states.
        
        Returns:
            Dict with recovery results
        """
        results = {
            'timestamp': str(self._get_current_time()),
            'components': {}
        }
        
        try:
            system_health = self.check_system_health()
            
            # Identify components in critical state
            critical_components = []
            for component, health in system_health.get('components', {}).items():
                if health.get('status') == 'critical':
                    critical_components.append(component)
            
            # Attempt recovery for critical components
            for component in critical_components:
                if component == ComponentType.POSITION_MANAGER:
                    if 'position_manager' in self.initialized_components and self.position_safeguards:
                        try:
                            success = recover_position_manager(self.position_safeguards)
                            results['components']['position_manager'] = {
                                'status': 'success' if success else 'failed'
                            }
                        except Exception as e:
                            results['components']['position_manager'] = {
                                'status': 'error',
                                'error': str(e)
                            }
                
                elif component == ComponentType.TRADE_ACCOUNTING:
                    if 'trade_accounting' in self.initialized_components and self.accounting_safeguards:
                        try:
                            success = recover_trade_accounting(self.accounting_safeguards)
                            results['components']['trade_accounting'] = {
                                'status': 'success' if success else 'failed'
                            }
                        except Exception as e:
                            results['components']['trade_accounting'] = {
                                'status': 'error',
                                'error': str(e)
                            }
                
                elif component == ComponentType.EXIT_MANAGER:
                    if 'exit_manager' in self.initialized_components and self.exit_safeguards:
                        try:
                            success = recover_exit_manager(self.exit_safeguards)
                            results['components']['exit_manager'] = {
                                'status': 'success' if success else 'failed'
                            }
                        except Exception as e:
                            results['components']['exit_manager'] = {
                                'status': 'error',
                                'error': str(e)
                            }
                
                elif component == ComponentType.CAPITAL_ALLOCATOR:
                    if 'capital_allocator' in self.initialized_components and self.allocator_safeguards:
                        try:
                            success = recover_capital_allocator(self.allocator_safeguards)
                            results['components']['capital_allocator'] = {
                                'status': 'success' if success else 'failed'
                            }
                        except Exception as e:
                            results['components']['capital_allocator'] = {
                                'status': 'error',
                                'error': str(e)
                            }
            
            # Reset circuit breakers
            for component, breaker in system_health.get('circuit_breakers', {}).items():
                if breaker.get('tripped'):
                    try:
                        success = self.system_safeguards.reset_circuit_breaker(component)
                        if component not in results['components']:
                            results['components'][component] = {}
                        results['components'][component]['circuit_breaker_reset'] = success
                    except Exception as e:
                        if component not in results['components']:
                            results['components'][component] = {}
                        results['components'][component]['circuit_breaker_reset_error'] = str(e)
            
            # Update overall status
            results['status'] = 'recovery_attempted'
            logger.info(f"System recovery attempted: {len(critical_components)} components recovered")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during system recovery: {str(e)}")
            results['status'] = 'error'
            results['error'] = str(e)
            return results
    
    def _get_current_time(self):
        """Get current datetime."""
        from datetime import datetime
        return datetime.now()
    
    def create_robustness_config_file(self, output_path: str) -> bool:
        """
        Create a default robustness configuration file.
        
        Args:
            output_path: Path to save the configuration file
            
        Returns:
            bool: Success status
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logger.info(f"Created robustness configuration at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating configuration file: {str(e)}")
            return False

# Convenience function to setup complete robustness for all components
def initialize_system_robustness(
    position_manager: Optional[PositionManager] = None,
    trade_accounting: Optional[TradeAccounting] = None,
    pnl_calculator: Optional[PnLCalculator] = None,
    performance_metrics: Optional[PerformanceMetrics] = None,
    exit_manager: Optional[ExitStrategyManager] = None, 
    capital_allocator: Optional[CapitalAllocator] = None,
    broker_manager: Optional[MultiBrokerManager] = None,
    config_path: Optional[str] = None,
    auto_start_monitoring: bool = True
) -> RobustnessManager:
    """
    Initialize robustness safeguards for all provided components.
    
    Args:
        position_manager: Optional position manager component
        trade_accounting: Optional trade accounting component
        pnl_calculator: Optional PnL calculator component
        performance_metrics: Optional performance metrics component
        exit_manager: Optional exit strategy manager component
        capital_allocator: Optional capital allocator component
        broker_manager: Optional broker manager component
        config_path: Optional path to configuration file
        auto_start_monitoring: Whether to automatically start monitoring
        
    Returns:
        Initialized robustness manager
    """
    try:
        # Initialize robustness manager
        manager = RobustnessManager(config_path)
        
        # Initialize component safeguards
        if position_manager is not None:
            manager.initialize_position_safeguards(position_manager)
            
        if trade_accounting is not None:
            manager.initialize_accounting_safeguards(
                trade_accounting, 
                pnl_calculator, 
                performance_metrics
            )
            
        if exit_manager is not None:
            manager.initialize_exit_safeguards(exit_manager)
            
        if capital_allocator is not None:
            manager.initialize_allocator_safeguards(capital_allocator)
            
        if broker_manager is not None:
            manager.register_broker_manager(broker_manager)
            
        # Start monitoring if requested
        if auto_start_monitoring:
            manager.start_monitoring()
            
        logger.info("System robustness initialized successfully")
        return manager
        
    except Exception as e:
        logger.error(f"Error initializing system robustness: {str(e)}")
        # Return manager even if incomplete
        return manager
