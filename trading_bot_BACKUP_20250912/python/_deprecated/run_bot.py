#!/usr/bin/env python
"""
BensBot Main Runner

This script initializes and runs the BensBot trading system,
utilizing the market regime detection and adaptation system.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import time
import signal
import threading

# Add parent directory to path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core components
from trading_bot.core.simple_config import (
    load_config, get_nested_value, ConfigError, ConfigFileNotFoundError,
    ConfigParseError, ConfigValidationError
)
from trading_bot.core.config_watcher import init_config_watcher
from trading_bot.core.event_bus import EventBus
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.accounting.trade_accounting import TradeAccounting

# Import market regime components
from trading_bot.analytics.market_regime.bootstrap import setup_market_regime_system
from trading_bot.analytics.market_regime.integration import MarketRegimeManager

# Set up default logging, will be updated from config later
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'trading_{time.strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger("benbot")

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

class TradingSystem:
    """
    Main trading system for BensBot.
    
    Orchestrates the initialization, execution, and monitoring
    of the trading system and its components.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trading system.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("Initializing BensBot trading system")
        
        # Create required directories
        os.makedirs('logs', exist_ok=True)
        
        # Load configuration
        self.config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Create data directory from config
        data_dir = self.config.get("data_dir", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Configure logging based on config
        self._configure_logging()
        
        # Initialize config watcher if hot reload is enabled
        self.config_watcher = None
        if self.config.get("enable_config_hot_reload", False):
            try:
                reload_interval = self.config.get("config_reload_interval_seconds", 60)
                self.config_watcher = init_config_watcher(
                    config_path=config_path,
                    reload_callback=self._handle_config_reload,
                    interval_seconds=reload_interval
                )
                logger.info(f"Initialized configuration watcher with {reload_interval}s interval")
            except Exception as e:
                logger.warning(f"Could not initialize config watcher: {str(e)}")
                logger.warning("Hot reload will be disabled")
        
        # Initialize core components
        self.event_bus = EventBus()
        logger.info("Initialized event bus")
        
        # Initialize broker manager
        self._init_broker_manager()
        
        # Initialize trade accounting
        self._init_trade_accounting()
        
        # Initialize market regime system
        self.market_regime_manager = None
        if self.config.get("enable_market_regime_system", False):
            self._init_market_regime_system()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        
        # System state
        self.running = False
        self.trading_mode = None
        
        # Register components
        self.components = {}
        
        logger.info("Trading system initialization complete")
        
    def _configure_logging(self):
        """Configure logging based on loaded configuration"""
        log_level_str = self.config.get("log_level", "INFO")
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level = log_levels.get(log_level_str, logging.INFO)
        
        # Update root logger and our logger
        logging.getLogger().setLevel(log_level)
        logger.setLevel(log_level)
        logger.info(f"Log level set to {log_level_str}")
    
    def _handle_config_reload(self, new_config):
        """Handle configuration reload during runtime"""
        logger.info("Configuration has been updated, applying changes...")
        
        try:
            # Update our config reference
            self.config = new_config
            
            # Update logging configuration
            self._configure_logging()
            
            # Update system safeguards if they changed
            if "system_safeguards" in new_config:
                logger.info("Updated system safeguards configuration")
            
            # Note: We don't update broker connections or other core components
            # that require restart for safety reasons
            logger.info("Configuration reload complete")
            
        except Exception as e:
            logger.error(f"Error applying configuration changes: {str(e)}")
            logger.warning("Some configuration changes may not have been applied")

    
    def _init_broker_manager(self):
        """Initialize the broker manager"""
        try:
            # Load broker configuration
            broker_config_path = self.config.get("broker_config_path", "config/broker_config.json")
            if os.path.exists(broker_config_path):
                try:
                    with open(broker_config_path, 'r') as f:
                        broker_config = json.load(f)
                    logger.info(f"Loaded broker configuration from {broker_config_path}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing broker config file: {str(e)}")
                    broker_config = {}
            else:
                logger.warning(f"Broker configuration file not found: {broker_config_path}")
                broker_config = {}
            
            # Initialize broker manager with configuration
            self.broker_manager = MultiBrokerManager(self.event_bus, broker_config)
            logger.info("Initialized broker manager")
        except Exception as e:
            logger.error(f"Error initializing broker manager: {str(e)}")
            self.broker_manager = MultiBrokerManager(self.event_bus, {})
            logger.info("Initialized broker manager with empty configuration")
    
    def _init_trade_accounting(self):
        """Initialize trade accounting system"""
        try:
            # Initialize trade accounting
            self.trade_accounting = TradeAccounting(
                data_dir=os.path.join(self.config.data_dir, "accounting"),
                event_bus=self.event_bus
            )
            logger.info("Initialized trade accounting")
        except Exception as e:
            logger.error(f"Error initializing trade accounting: {str(e)}")
            raise
    
    def _init_market_regime_system(self):
        """Initialize market regime detection and adaptation system"""
        try:
            # Check if market regime config file exists
            market_regime_config_path = self.config.get("market_regime_config_path", "config/market_regime_config.json")
            if not os.path.exists(market_regime_config_path):
                logger.warning(f"Market regime configuration file not found: {market_regime_config_path}")
                logger.warning("Market regime system will not be initialized")
                return
                
            # Setup market regime system
            watched_symbols = self.config.get("watched_symbols", [])
            if not watched_symbols:
                logger.warning("No watched symbols specified in configuration")
                logger.warning("Market regime system will not be initialized")
                return
                
            market_regime_config = setup_market_regime_system(
                config_path=market_regime_config_path,
                watched_symbols=watched_symbols,
                event_bus=self.event_bus
            )
            
            # Initialize market regime manager
            self.market_regime_manager = MarketRegimeManager(
                market_regime_config=market_regime_config,
                event_bus=self.event_bus
            )
            
            logger.info("Initialized market regime system")
            self.register_component("market_regime_manager", self.market_regime_manager)
        except Exception as e:
            logger.error(f"Error initializing market regime system: {str(e)}")
            logger.warning("Trading will continue without market regime adaptation")
            logger.warning("Continuing without market regime system")
    
    def register_component(self, name: str, component):
        """
        Register a component with the trading system.
        
        Args:
            name: Name of the component
            component: Component instance
        """
        setattr(self, name, component)
        logger.debug(f"Registered component: {name}")
    
    def start_live_trading(self):
        """Start live trading"""
        logger.info("Starting live trading")
        self.trading_mode = "live"
        self._start_trading()
    
    def start_paper_trading(self):
        """Start paper trading"""
        logger.info("Starting paper trading")
        self.trading_mode = "paper"
        self._start_trading()
    
    def _start_trading(self):
        """Start the trading system"""
        try:
            # Start components
            self.running = True
            
            # Start market regime system if initialized
            if self.market_regime_manager:
                self.market_regime_manager.start()
                logger.info("Started market regime system")
            
            # Start broker connections
            self.broker_manager.connect_all()
            logger.info("Connected to brokers")
            
            # Start trade accounting
            self.trade_accounting.start()
            logger.info("Started trade accounting")
            
            # Main trading loop
            self._run_trading_loop()
            
        except Exception as e:
            logger.error(f"Error starting trading: {str(e)}")
            self.shutdown()
            raise
    
    def _run_trading_loop(self):
        """Run the main trading loop"""
        logger.info("Entering main trading loop")
        
        try:
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
            heartbeat_thread.daemon = True
            heartbeat_thread.start()
            
            # Main loop
            while self.running and not SHUTDOWN_REQUESTED:
                # Process any pending work
                self.event_bus.process_events()
                
                # Sleep to prevent busy waiting
                time.sleep(0.1)
                
            logger.info("Exiting main trading loop")
            
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}")
            self.shutdown()
    
    def _heartbeat_loop(self):
        """Heartbeat thread to monitor system health"""
        logger.info("Starting heartbeat monitoring")
        
        heartbeat_interval = 60  # seconds
        last_heartbeat = time.time()
        
        while self.running and not SHUTDOWN_REQUESTED:
            try:
                current_time = time.time()
                
                if current_time - last_heartbeat >= heartbeat_interval:
                    # Record heartbeat
                    self._log_heartbeat()
                    last_heartbeat = current_time
                    
                    # Check system health
                    self._check_system_health()
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                # Don't crash the heartbeat thread
        
        logger.info("Stopped heartbeat monitoring")
    
    def _log_heartbeat(self):
        """Log system heartbeat"""
        logger.debug(f"Heartbeat - {datetime.now().isoformat()} - System running in {self.trading_mode} mode")
        
        # Log some basic statistics if available
        try:
            # Check broker connections
            if hasattr(self, 'broker_manager'):
                broker_status = self.broker_manager.get_connection_status()
                if broker_status:
                    logger.debug(f"Broker status: {broker_status}")
            
            # Check positions if available
            if hasattr(self, 'broker_manager'):
                positions = self.broker_manager.get_all_positions()
                if positions:
                    position_count = sum(len(pos_list) for pos_list in positions.values())
                    logger.debug(f"Current positions: {position_count}")
        except Exception as e:
            logger.error(f"Error logging heartbeat stats: {str(e)}")
    
    def _check_system_health(self):
        """Check system health and apply circuit breakers if needed"""
        try:
            # Check circuit breakers
            if hasattr(self.config, 'system_safeguards'):
                safeguards = self.config.system_safeguards
                
                # Check drawdown
                max_drawdown = safeguards.circuit_breakers.max_drawdown_percent
                # TODO: Implement drawdown check
                
                # Check daily loss
                max_daily_loss = safeguards.circuit_breakers.max_daily_loss_percent
                # TODO: Implement daily loss check
                
                # Check consecutive losses
                max_consecutive_losses = safeguards.circuit_breakers.consecutive_loss_count
                # TODO: Implement consecutive loss check
        
        except Exception as e:
            logger.error(f"Error checking system health: {str(e)}")
    
    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signal"""
        global SHUTDOWN_REQUESTED
        
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        logger.info(f"Received {signal_name} signal, initiating graceful shutdown")
        
        SHUTDOWN_REQUESTED = True
        self.shutdown()


def print_resolved_config(config_path):
    """Print resolved configuration with environment overrides"""
    try:
        config = load_config(config_path)
        # Print nicely formatted JSON
        print(json.dumps(config, indent=2, sort_keys=True))
        
        # Find and print active environment overrides
        env_overrides = []
        for key in os.environ:
            if key.startswith("BENBOT_"):
                env_overrides.append(f"{key}={os.environ[key]}")
        
        if env_overrides:
            print("\n=== Active Environment Overrides ===")
            for override in sorted(env_overrides):
                print(override)
        
        return 0
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1


def validate_broker_connections(config_path, broker_config_path, test_connections=True):
    """Validate broker connections"""
    try:
        # Import broker validation module
        from trading_bot.scripts.validate_brokers import validate_all_brokers
        
        # Load main config first to ensure it's valid
        config = load_config(config_path)
        
        # Get broker config path from main config if not specified
        if not broker_config_path:
            broker_config_path = config.get("broker_config_path", "config/broker_config.json")
            logger.info(f"Using broker config path from main config: {broker_config_path}")
        
        # Run broker validation
        logger.info(f"Validating broker connections from {broker_config_path}")
        success = validate_all_brokers(
            config_path=broker_config_path,
            test_connections=test_connections,
            quick_check=False
        )
        
        if success:
            logger.info("✅ All broker validations passed")
            return 0
        else:
            logger.error("❌ Some broker validations failed")
            return 1
            
    except Exception as e:
        logger.error(f"Error validating broker connections: {str(e)}")
        return 1


def validate_schema(config_path, schema_path):
    """Validate configuration against schema without running the bot"""
    try:
        # First check if schema exists
        if not os.path.exists(schema_path):
            logger.error(f"Schema file not found: {schema_path}")
            return 1
            
        # Then load and validate config
        config = load_config(config_path)
        logger.info(f"✅ Configuration validated successfully: {config_path}")
        
        # Print validation summary
        print(f"\n=== Validation Summary ===\n")
        print(f"✅ Configuration is valid")
        print(f"- Config file: {config_path}")
        print(f"- Schema: {schema_path}")
        print(f"- Watched symbols: {config.get('watched_symbols', [])}")
        print(f"- Initial capital: {config.get('initial_capital')}")
        print(f"- Trading hours: {config.get('trading_hours', {})}")
        
        # Check referenced files
        referenced_files = [
            config.get('market_regime_config_path'),
            config.get('broker_config_path'),
            config.get('market_data_config_path')
        ]
        
        missing_files = [f for f in referenced_files if f and not os.path.exists(f)]
        if missing_files:
            print(f"\n⚠️  Warning: Referenced files not found:")
            for f in missing_files:
                print(f"  - {f}")
        
        return 0
    except ConfigFileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")
        return 1
    except ConfigParseError as e:
        logger.error(f"Configuration parse error: {str(e)}")
        return 1
    except ConfigValidationError as e:
        logger.error(f"Configuration validation error: {str(e)}")
        return 1
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1


def main():
    """Main entry point"""
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="BensBot Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_bot.py --print-config                   # Print current config with env overrides
  python run_bot.py --validate-schema                # Validate config against schema
  python run_bot.py --validate-broker-connections    # Test broker connections
  python run_bot.py --paper                          # Run in paper trading mode
  python run_bot.py --live                           # Run in live trading mode
  python run_bot.py --config custom_config.json      # Use a different config file
        """
    )
    parser.add_argument("--config", default="config/system_config.json", help="Path to configuration file")
    parser.add_argument("--schema", default="config/system_config.schema.json", help="Path to schema file")
    parser.add_argument("--broker-config", default="", help="Path to broker configuration file (default: from system config)")
    parser.add_argument("--live", action="store_true", help="Run in live trading mode")
    parser.add_argument("--paper", action="store_true", help="Run in paper trading mode (default)")
    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit")
    parser.add_argument("--validate-schema", action="store_true", help="Validate config against schema and exit")
    parser.add_argument("--validate-broker-connections", action="store_true", help="Validate broker connections and exit")
    parser.add_argument("--skip-connection-tests", action="store_true", help="Skip broker connection tests when validating")
    parser.add_argument("--no-hot-reload", action="store_true", help="Disable configuration hot-reload")
    args = parser.parse_args()
    
    # Special mode: just print config
    if args.print_config:
        return print_resolved_config(args.config)
    
    # Special mode: just validate schema
    if args.validate_schema:
        return validate_schema(args.config, args.schema)
    
    # Special mode: validate broker connections
    if args.validate_broker_connections:
        return validate_broker_connections(args.config, args.broker_config, not args.skip_connection_tests)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
        
        # Initialize the trading system
        system = TradingSystem(args.config)
        
        # Start trading based on mode
        if args.live:
            system.start_live_trading()
        elif args.paper:
            system.start_paper_trading()
        else:
            # Default to paper trading
            system.start_paper_trading()
        
        return 0
        
    except ConfigFileNotFoundError as e:
        logger.error(f"Configuration file not found: {str(e)}")
        logger.error(f"Please check that {args.config} exists.")
        return 1
    except ConfigParseError as e:
        logger.error(f"Configuration parse error: {str(e)}")
        logger.error("Please check that your configuration is valid JSON.")
        return 1
    except ConfigValidationError as e:
        logger.error(f"Configuration validation error: {str(e)}")
        logger.error("Please check your configuration against the schema.")
        return 1
    except ConfigError as e:
        logger.error(f"Configuration error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
