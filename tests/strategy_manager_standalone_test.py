#!/usr/bin/env python3
"""
Standalone test script for the Enhanced Strategy Manager implementation.
This script includes minimal versions of needed components to avoid dependency issues.
"""

import logging
import sys
import time
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union
import uuid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("test")

# Minimal implementation of required classes
class EventType(Enum):
    """Minimal event types required for testing."""
    MARKET_DATA_UPDATE = auto()
    QUOTE_UPDATE = auto()
    TRADE_UPDATE = auto()
    ORDER_FILLED = auto()
    STRATEGY_STARTED = auto()
    STRATEGY_STOPPED = auto()

class AssetType(Enum):
    """Asset types."""
    STOCKS = "stocks"
    OPTIONS = "options"
    FOREX = "forex"
    CRYPTO = "crypto"

class StrategyType(Enum):
    """Strategy types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    PATTERN = "pattern"

class StrategyState(Enum):
    """Strategy states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class Event:
    """Simple event implementation."""
    def __init__(self, event_type, data=None, source=None):
        self.event_type = event_type
        self.data = data or {}
        self.source = source
        self.timestamp = datetime.now()

class EventBus:
    """Simple event bus implementation."""
    def __init__(self):
        self.subscribers = {}
    
    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event):
        if event.event_type in self.subscribers:
            for callback in self.subscribers[event.event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")

# Global event bus
_global_event_bus = EventBus()

def get_global_event_bus():
    """Get the global event bus instance."""
    return _global_event_bus

class Strategy:
    """Base class for trading strategies."""
    def __init__(self, strategy_id, name, description="", symbols=None, asset_type=None,
                 timeframe="1d", parameters=None, risk_limits=None, broker_id=None,
                 enabled=True):
        self.strategy_id = strategy_id
        self.name = name
        self.description = description
        self.symbols = symbols or []
        self.asset_type = asset_type
        self.timeframe = timeframe
        self.parameters = parameters or {}
        self.risk_limits = risk_limits or {}
        self.broker_id = broker_id
        self.enabled = enabled
        self.state = StrategyState.STOPPED
        self.strategy_type = StrategyType.MOMENTUM
        self.running_since = None
        self.event_bus = get_global_event_bus()
    
    def start(self):
        """Start the strategy."""
        self.state = StrategyState.RUNNING
        self.running_since = datetime.now()
        logger.info(f"Started strategy: {self.name}")
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_STARTED,
            data={"strategy_id": self.strategy_id, "name": self.name},
            source="strategy"
        ))
    
    def stop(self):
        """Stop the strategy."""
        self.state = StrategyState.STOPPED
        self.running_since = None
        logger.info(f"Stopped strategy: {self.name}")
        self.event_bus.publish(Event(
            event_type=EventType.STRATEGY_STOPPED,
            data={"strategy_id": self.strategy_id, "name": self.name},
            source="strategy"
        ))
    
    def pause(self):
        """Pause the strategy."""
        self.state = StrategyState.PAUSED
        logger.info(f"Paused strategy: {self.name}")
    
    def reset(self):
        """Reset the strategy."""
        self.state = StrategyState.STOPPED
        logger.info(f"Reset strategy: {self.name}")
    
    def is_running(self):
        """Check if the strategy is running."""
        return self.state == StrategyState.RUNNING
    
    def generate_signal(self, data):
        """Generate trading signal from market data."""
        raise NotImplementedError("Subclasses must implement generate_signal")

class SignalAction(Enum):
    """Signal actions."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"

class SignalStrength(Enum):
    """Signal strength levels."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class Signal:
    """Trading signal representation."""
    def __init__(self, strategy_id, symbol, action, strength=SignalStrength.MODERATE,
                 confidence=0.5, quantity=None, allocation=None, target_price=None,
                 stop_loss=None, metadata=None):
        self.id = str(uuid.uuid4())
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.action = action
        self.strength = strength
        self.confidence = confidence
        self.quantity = quantity
        self.allocation = allocation
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
        self.processed = False
        self.result = None
    
    def to_dict(self):
        """Convert signal to dictionary."""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "action": self.action.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "quantity": self.quantity,
            "allocation": self.allocation,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
            "result": self.result
        }

class StrategyEnsemble:
    """Ensemble of multiple strategies."""
    def __init__(self, ensemble_id, name, combination_method="weighted",
                 min_consensus=0.5, auto_adjust_weights=True, description=""):
        self.ensemble_id = ensemble_id
        self.name = name
        self.description = description
        self.combination_method = combination_method
        self.min_consensus = min_consensus
        self.auto_adjust_weights = auto_adjust_weights
        self.strategies = {}  # {strategy_id: (strategy, weight)}
        self.symbols = []
        self.state = StrategyState.STOPPED
    
    def add_strategy(self, strategy, weight=1.0):
        """Add a strategy to the ensemble."""
        self.strategies[strategy.strategy_id] = (strategy, weight)
        # Update symbols
        for symbol in strategy.symbols:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
    
    def update_strategy_weight(self, strategy, weight):
        """Update a strategy's weight."""
        if strategy.strategy_id in self.strategies:
            self.strategies[strategy.strategy_id] = (strategy, weight)
    
    def start(self):
        """Start all strategies in the ensemble."""
        for strategy_id, (strategy, _) in self.strategies.items():
            if strategy.state != StrategyState.RUNNING:
                strategy.start()
        self.state = StrategyState.RUNNING
    
    def stop(self):
        """Stop all strategies in the ensemble."""
        for strategy_id, (strategy, _) in self.strategies.items():
            if strategy.state == StrategyState.RUNNING:
                strategy.stop()
        self.state = StrategyState.STOPPED
    
    def is_running(self):
        """Check if the ensemble is running."""
        return self.state == StrategyState.RUNNING
    
    def generate_combined_signal(self, symbol, signals):
        """Generate a combined signal from component signals."""
        if not signals:
            return None
        
        if self.combination_method == "weighted":
            return self._combine_weighted(symbol, signals)
        elif self.combination_method == "unanimous":
            return self._combine_unanimous(symbol, signals)
        elif self.combination_method == "majority":
            return self._combine_majority(symbol, signals)
        else:
            return signals[0]  # Default to first signal
    
    def _combine_weighted(self, symbol, signals):
        """Combine signals using weighted averaging."""
        # This is a simplified implementation
        total_weight = 0
        weighted_confidence = 0
        
        for signal in signals:
            strategy_id = signal.strategy_id
            if strategy_id in self.strategies:
                _, weight = self.strategies[strategy_id]
                weighted_confidence += signal.confidence * weight
                total_weight += weight
        
        if total_weight > 0:
            final_confidence = weighted_confidence / total_weight
            
            # Determine final action based on confidence
            if final_confidence > 0.7:
                action = SignalAction.BUY
            elif final_confidence < 0.3:
                action = SignalAction.SELL
            else:
                action = SignalAction.HOLD
            
            # Create a new signal for the ensemble
            combined_signal = Signal(
                strategy_id=self.ensemble_id,
                symbol=symbol,
                action=action,
                confidence=final_confidence,
                strength=SignalStrength.MODERATE,
                allocation=0.05,  # Default allocation
                metadata={"ensemble": True, "component_signals": len(signals)}
            )
            
            return combined_signal
        
        return None

class StrategyPerformanceManager:
    """Manages strategy performance tracking and evaluation."""
    def __init__(self):
        self.strategies = {}
        self.metrics = {}
    
    def register_strategy(self, strategy_id, strategy):
        """Register a strategy for performance tracking."""
        self.strategies[strategy_id] = strategy
        self.metrics[strategy_id] = {
            "win_rate": 0.5,  # Default values for testing
            "profit_factor": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "daily_returns": {}
        }
    
    def get_strategy_metrics(self, strategy_id):
        """Get performance metrics for a strategy."""
        return self.metrics.get(strategy_id, {})
    
    def evaluate_all(self):
        """Evaluate all registered strategies."""
        results = {}
        for strategy_id, strategy in self.strategies.items():
            # Mock evaluation results
            results[strategy_id] = {
                "status": "active",
                "metrics": self.metrics.get(strategy_id, {}),
                "action": None
            }
        return results
    
    def save_state(self):
        """Save performance manager state."""
        logger.info("Saving performance manager state")

# Mock broker manager for testing
class MockBrokerManager:
    """Simple mock broker manager for testing."""
    def __init__(self):
        self.brokers = {"mock_broker": "Mock Broker"}
        self.active_broker_id = "mock_broker"
        self.primary_broker_id = "mock_broker"
    
    def connect_all(self):
        """Connect all brokers."""
        logger.info("Connected to mock broker")
        return {"mock_broker": True}
    
    def get_all_positions(self):
        """Get all positions."""
        return {"mock_broker": {}}
    
    def get_position(self, symbol):
        """Get position for symbol."""
        return None
    
    def get_all_accounts(self):
        """Get all accounts."""
        return {"mock_broker": {"buying_power": 10000}}
    
    def get_quote(self, symbol):
        """Get quote for symbol."""
        return None
    
    def get_broker_for_asset_type(self, asset_type):
        """Get broker for asset type."""
        return "mock_broker"

# Minimal implementation of the strategy manager
class MinimalEnhancedStrategyManager:
    """Minimal implementation of the Enhanced Strategy Manager for testing."""
    def __init__(self, broker_manager=None, performance_manager=None, config=None):
        self.broker_manager = broker_manager
        self.performance_manager = performance_manager or StrategyPerformanceManager()
        self.config = config or {}
        self.event_bus = get_global_event_bus()
        self.strategies = {}
        self.active_strategies = {}
        self.ensembles = {}
        self.signals = []
        self.pending_signals = {}
        self.signal_history = []
        self.is_running = False
    
    def load_strategies(self, strategy_configs):
        """Load strategies from configuration."""
        logger.info(f"Loading {len(strategy_configs)} strategies")
        
        for config in strategy_configs:
            strategy_id = config.get("strategy_id")
            name = config.get("name", "Unnamed Strategy")
            
            # Create a test strategy
            strategy = TestStrategy(
                strategy_id=strategy_id,
                name=name,
                description=config.get("description", ""),
                symbols=config.get("symbols", []),
                asset_type=config.get("asset_type"),
                timeframe=config.get("timeframe", "1d"),
                parameters=config.get("parameters", {}),
                broker_id=config.get("broker_id")
            )
            
            # Add to collections
            self.strategies[strategy_id] = strategy
            if config.get("enabled", True):
                self.active_strategies[strategy_id] = strategy
            
            # Register with performance manager
            self.performance_manager.register_strategy(strategy_id, strategy)
            
            logger.info(f"Loaded strategy: {name} ({strategy_id})")
    
    def create_ensembles(self, ensemble_configs):
        """Create ensembles from configuration."""
        logger.info(f"Creating {len(ensemble_configs)} ensembles")
        
        for config in ensemble_configs:
            ensemble_id = config.get("ensemble_id")
            name = config.get("name", "Unnamed Ensemble")
            strategy_weights = config.get("strategies", {})
            
            # Create ensemble
            ensemble = StrategyEnsemble(
                ensemble_id=ensemble_id,
                name=name,
                combination_method=config.get("combination_method", "weighted"),
                min_consensus=config.get("min_consensus", 0.5),
                auto_adjust_weights=config.get("auto_adjust_weights", True),
                description=config.get("description", "")
            )
            
            # Add strategies
            for strategy_id, weight in strategy_weights.items():
                if strategy_id in self.strategies:
                    ensemble.add_strategy(self.strategies[strategy_id], weight)
            
            # Add to collection
            self.ensembles[ensemble_id] = ensemble
            
            logger.info(f"Created ensemble: {name} ({ensemble_id}) with {len(ensemble.strategies)} strategies")
    
    def start_strategies(self):
        """Start all strategies."""
        logger.info("Starting strategies")
        self.is_running = True
        
        for strategy_id, strategy in self.active_strategies.items():
            strategy.start()
        
        for ensemble_id, ensemble in self.ensembles.items():
            ensemble.start()
    
    def stop_strategies(self):
        """Stop all strategies."""
        logger.info("Stopping strategies")
        self.is_running = False
        
        for strategy_id, strategy in self.active_strategies.items():
            strategy.stop()
        
        for ensemble_id, ensemble in self.ensembles.items():
            ensemble.stop()
    
    def evaluate_performance(self):
        """Evaluate strategy performance."""
        logger.info("Evaluating performance")
        return {
            "strategies": {},
            "ensembles": {},
            "actions_taken": []
        }
    
    def get_active_strategies(self):
        """Get active strategies."""
        result = []
        for strategy_id, strategy in self.active_strategies.items():
            result.append({
                "strategy_id": strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "symbols": strategy.symbols,
                "asset_type": strategy.asset_type,
                "state": strategy.state.value,
                "broker_id": strategy.broker_id
            })
        return result
    
    def on_market_data(self, event):
        """Handle market data events."""
        if not self.is_running:
            return
        
        data = event.data
        symbol = data.get("symbol")
        if not symbol:
            return
        
        logger.info(f"Received market data for {symbol}")
        
        # Generate signals from active strategies
        signals_to_process = []
        
        for strategy_id, strategy in self.active_strategies.items():
            if symbol in strategy.symbols and strategy.is_running():
                signal = strategy.generate_signal(data)
                if signal:
                    signals_to_process.append(signal)
                    self.signal_history.append(signal)
        
        logger.info(f"Generated {len(signals_to_process)} signals")

# Test strategy implementation for minimal test
class TestStrategy(Strategy):
    """Test strategy implementation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def generate_signal(self, data):
        """Generate a test signal."""
        symbol = data.get("symbol")
        if not symbol or symbol not in self.symbols:
            return None
        
        # Create a simple signal for testing
        logger.info(f"Strategy {self.strategy_id} generating signal for {symbol}")
        
        # Alternating buy/sell based on timestamp
        action = SignalAction.BUY if datetime.now().second % 2 == 0 else SignalAction.SELL
        
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            action=action,
            strength=SignalStrength.MODERATE,
            confidence=0.7,
            allocation=0.05,
            metadata={"test": True}
        )

def main():
    """Main test function."""
    try:
        logger.info("Starting Enhanced Strategy Manager Standalone Test")
        
        # Create event bus
        event_bus = get_global_event_bus()
        
        # Create mock broker manager
        broker_manager = MockBrokerManager()
        
        # Create performance manager
        performance_manager = StrategyPerformanceManager()
        
        # Create strategy manager
        strategy_manager = MinimalEnhancedStrategyManager(
            broker_manager=broker_manager,
            performance_manager=performance_manager
        )
        
        # Register event handlers
        event_bus.subscribe(EventType.MARKET_DATA_UPDATE, strategy_manager.on_market_data)
        
        # Load test strategies
        test_strategies = [
            {
                "strategy_id": "test_strategy_1",
                "name": "Test Strategy 1",
                "symbols": ["AAPL", "MSFT"],
                "asset_type": "stocks",
                "broker_id": "mock_broker",
                "enabled": True
            },
            {
                "strategy_id": "test_strategy_2",
                "name": "Test Strategy 2",
                "symbols": ["BTC-USD"],
                "asset_type": "crypto",
                "broker_id": "mock_broker",
                "enabled": True
            }
        ]
        
        strategy_manager.load_strategies(test_strategies)
        
        # Create test ensemble
        test_ensembles = [
            {
                "ensemble_id": "test_ensemble_1",
                "name": "Test Ensemble",
                "combination_method": "weighted",
                "strategies": {
                    "test_strategy_1": 0.6,
                    "test_strategy_2": 0.4
                }
            }
        ]
        
        strategy_manager.create_ensembles(test_ensembles)
        
        # Start strategy manager
        strategy_manager.start_strategies()
        
        # Simulate market data events
        logger.info("Simulating market data events")
        for symbol in ["AAPL", "MSFT", "BTC-USD"]:
            # Create mock data
            mock_data = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": 100.0,
                "volume": 1000
            }
            
            # Publish event
            event_bus.publish(Event(
                event_type=EventType.MARKET_DATA_UPDATE,
                data=mock_data,
                source="test"
            ))
            
            # Small delay
            time.sleep(0.5)
        
        # Check results
        logger.info(f"Generated signals: {len(strategy_manager.signal_history)}")
        for i, signal in enumerate(strategy_manager.signal_history):
            logger.info(f"Signal {i+1}: {signal.symbol} - {signal.action.value}")
        
        # Stop strategy manager
        strategy_manager.stop_strategies()
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
