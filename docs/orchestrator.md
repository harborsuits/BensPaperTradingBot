# Main Orchestrator

The Main Orchestrator is the central coordination component of the BensBot Trading System, responsible for integrating all modules and maintaining the trading loop.

## Overview

The Main Orchestrator:

1. Initializes all system components
2. Manages the main trading loop
3. Coordinates data flow between components
4. Handles system state and lifecycle events
5. Implements fault tolerance and recovery

## Architecture

The orchestrator follows a modular design pattern:

```
MainOrchestrator
├── DataManager
├── StrategyManager
├── RiskManager
├── BrokerAdapter
├── NotificationManager
└── StateManager
```

## Configuration

The orchestrator is configured through the typed settings system:

```python
class OrchestratorSettings(BaseModel):
    """Orchestration engine configuration."""
    loop_interval_seconds: int = 60
    market_hours_only: bool = True
    enable_strategy_rotation: bool = True
    rotation_time: str = "16:30"  # 4:30 PM, after market close
    enable_emergency_shutdown: bool = True
    max_consecutive_errors: int = 5
    recovery_backoff_seconds: int = 60
    logging_level: str = "INFO"
```

## Main Trading Loop

The orchestrator implements the main trading loop:

```python
def run(self):
    """Main trading loop."""
    self.initialize()
    
    while not self.should_stop:
        try:
            # Check if we should be trading now
            if self.is_trading_time():
                # Update market data
                self.update_market_data()
                
                # Generate trading signals
                signals = self.generate_signals()
                
                # Apply risk checks
                approved_signals = self.apply_risk_checks(signals)
                
                # Execute approved signals
                self.execute_signals(approved_signals)
                
                # Update portfolio state
                self.update_portfolio_state()
                
                # Send notifications
                self.send_status_updates()
            
            # Sleep until next cycle
            self.wait_for_next_cycle()
            
        except Exception as e:
            self.handle_loop_error(e)
    
    self.shutdown()
```

## Component Initialization

The orchestrator initializes all system components:

```python
def initialize(self):
    """Initialize all system components."""
    self.logger.info("Initializing MainOrchestrator...")
    
    # Load typed settings
    self.settings = load_config()
    
    # Initialize components
    self.data_manager = self._init_data_manager()
    self.strategy_manager = self._init_strategy_manager()
    self.risk_manager = self._init_risk_manager()
    self.broker = self._init_broker()
    self.notifier = self._init_notification_manager()
    
    # Initial system state
    self.portfolio = self.broker.get_portfolio()
    self.market_state = self.data_manager.get_market_state()
    
    self.logger.info("MainOrchestrator initialization complete.")
```

## Market Intelligence Integration

The orchestrator integrates with the Market Intelligence API to enrich trading decisions with:

1. **Real-time news** from multiple sources like Alpha Vantage, NewsData.io, and others
2. **Sentiment analysis** of news and social media
3. **Market context** and sector performance

This integration provides professional, institutional-level analysis with:
- High/medium/low impact classification
- Portfolio impact assessments
- Actionable trade recommendations

## Error Handling and Recovery

The orchestrator implements robust error handling:

```python
def handle_loop_error(self, error):
    """Handle errors in the main loop."""
    self.consecutive_errors += 1
    
    self.logger.error(f"Error in main loop: {str(error)}", exc_info=True)
    
    # Send error notification
    self.notifier.send_alert(
        level="error",
        title="Trading loop error",
        message=f"Error in main trading loop: {str(error)}",
        details=traceback.format_exc()
    )
    
    # Check for emergency shutdown
    if (self.settings.enable_emergency_shutdown and 
            self.consecutive_errors >= self.settings.max_consecutive_errors):
        self.logger.critical(f"Emergency shutdown triggered after {self.consecutive_errors} consecutive errors")
        self.should_stop = True
        self.notifier.send_alert(level="critical", title="EMERGENCY SHUTDOWN", 
                                 message="Trading system emergency shutdown triggered")
    
    # Exponential backoff for recovery
    backoff = self.settings.recovery_backoff_seconds * (2 ** (self.consecutive_errors - 1))
    backoff = min(backoff, 3600)  # Cap at 1 hour
    self.logger.info(f"Backing off for {backoff} seconds before next cycle")
    time.sleep(backoff)
```

## Scheduling and Timing

The orchestrator controls when trading activities occur:

- **Market Hours Filtering**: Option to trade only during market hours
- **Scheduled Activities**: End-of-day reporting, strategy rotation, etc.
- **Custom Trading Windows**: Support for pre-market, after-hours, or specific time windows

## API Integration

The orchestrator provides a REST API for monitoring and control:

- `/api/orchestrator/status` - Get current system status
- `/api/orchestrator/control/start` - Start the trading loop
- `/api/orchestrator/control/stop` - Stop the trading loop
- `/api/orchestrator/control/pause` - Pause trading (maintain state)
- `/api/orchestrator/metrics` - Get performance metrics

## Notification System

The orchestrator sends notifications through multiple channels:

- **Trade Notifications**: Alerts for executed trades
- **Risk Warnings**: Notifications about risk limit breaches
- **System Status**: Regular status updates
- **Error Alerts**: Critical system errors

Notifications use the secure Telegram integration configured through the typed settings system.

## Usage Example

```python
from trading_bot.config.typed_settings import load_config
from trading_bot.core.main_orchestrator import MainOrchestrator

# Load settings from file and environment variables
settings = load_config("config.yaml")

# Create and start the orchestrator
orchestrator = MainOrchestrator(settings)
orchestrator.run()
```
