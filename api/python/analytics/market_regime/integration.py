"""
Market Regime Integration

This module integrates the market regime detection system with the rest of the
trading system, enabling adaptive trading based on market conditions.
"""

import logging
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from datetime import datetime, timedelta
import time
import pandas as pd

# Import local modules
from trading_bot.analytics.market_regime.detector import MarketRegimeDetector, MarketRegimeType
from trading_bot.analytics.market_regime.adaptation import ParameterOptimizer
from trading_bot.analytics.market_regime.performance import RegimePerformanceTracker
from trading_bot.analytics.market_regime.strategy_selector import StrategySelector

# Import system components
from trading_bot.core.event_bus import EventBus, Event
from trading_bot.brokers.multi_broker_manager import MultiBrokerManager
from trading_bot.accounting.trade_accounting import TradeAccounting
from trading_bot.portfolio.capital_allocator import CapitalAllocator

logger = logging.getLogger(__name__)

class MarketRegimeManager:
    """
    Central manager for market regime-based trading adaptation.
    
    Integrates all regime-related components and orchestrates the adaptation
    of the trading system in response to changing market conditions.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        broker_manager: MultiBrokerManager,
        trade_accounting: TradeAccounting,
        capital_allocator: Optional[CapitalAllocator] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize market regime manager.
        
        Args:
            event_bus: System event bus
            broker_manager: Broker manager for market data
            trade_accounting: Trade accounting system for performance data
            capital_allocator: Capital allocator for strategy allocation
            config: Configuration parameters
        """
        self.config = config or {}
        self.event_bus = event_bus
        self.broker_manager = broker_manager
        self.trade_accounting = trade_accounting
        self.capital_allocator = capital_allocator
        
        # Initialize components
        self.detector = MarketRegimeDetector(self.event_bus, config=self.config.get("detector", {}))
        self.parameter_optimizer = ParameterOptimizer(config=self.config.get("parameter_optimizer", {}))
        self.performance_tracker = RegimePerformanceTracker(config=self.config.get("performance_tracker", {}))
        self.strategy_selector = StrategySelector(
            self.performance_tracker, 
            config=self.config.get("strategy_selector", {})
        )
        
        # Tracked symbols
        self.tracked_symbols: Set[str] = set()
        
        # Active regimes by symbol
        self.active_regimes: Dict[str, Dict[str, MarketRegimeType]] = {}
        
        # Regime history
        self.regime_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        
        # Integration state
        self.initialized = False
        self.monitoring_active = False
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Monitoring settings
        self.monitoring_interval = self.config.get("monitoring_interval_seconds", 300)  # 5 minutes default
        self.update_interval = self.config.get("update_interval_seconds", 3600)  # 1 hour default
        self.min_data_points = self.config.get("min_data_points", 100)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("Market Regime Manager initialized")
    
    def _register_event_handlers(self) -> None:
        """Register event handlers with the event bus."""
        try:
            # Register for regime change events
            self.event_bus.register("market_regime_change", self._handle_regime_change)
            
            # Register for trade events
            self.event_bus.register("trade_closed", self._handle_trade_closed)
            
            # Register for price update events (for data collection)
            self.event_bus.register("price_update", self._handle_price_update)
            
            logger.info("Registered market regime event handlers")
            
        except Exception as e:
            logger.error(f"Error registering event handlers: {str(e)}")
    
    def initialize(self, symbols: List[str]) -> bool:
        """
        Initialize the market regime system with a list of symbols to track.
        
        Args:
            symbols: List of symbols to track
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                if self.initialized:
                    logger.warning("MarketRegimeManager already initialized")
                    return True
                
                logger.info(f"Initializing MarketRegimeManager with {len(symbols)} symbols")
                
                # Add symbols to track
                for symbol in symbols:
                    self.tracked_symbols.add(symbol)
                    
                    # Initialize regime history
                    if symbol not in self.regime_history:
                        self.regime_history[symbol] = {}
                
                # Initialize detector with symbols
                for symbol in self.tracked_symbols:
                    # Add symbol to detector (this will start data collection)
                    self.detector.add_symbol(symbol)
                
                self.initialized = True
                
                # Start monitoring thread if not already running
                if not self.monitoring_active:
                    self._start_monitoring()
                
                logger.info("MarketRegimeManager initialization complete")
                return True
                
            except Exception as e:
                logger.error(f"Error initializing market regime manager: {str(e)}")
                return False
    
    def _start_monitoring(self) -> None:
        """Start the regime monitoring thread."""
        try:
            if self.monitoring_active:
                return
            
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="RegimeMonitoringThread",
                daemon=True
            )
            self._monitoring_thread.start()
            self.monitoring_active = True
            
            logger.info("Started market regime monitoring thread")
            
        except Exception as e:
            logger.error(f"Error starting monitoring thread: {str(e)}")
    
    def _stop_monitoring(self) -> None:
        """Stop the regime monitoring thread."""
        try:
            if not self.monitoring_active:
                return
            
            logger.info("Stopping market regime monitoring thread")
            self._stop_event.set()
            
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=10)
                
            self.monitoring_active = False
            
            logger.info("Stopped market regime monitoring thread")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring thread: {str(e)}")
    
    def _monitoring_loop(self) -> None:
        """Background thread for regime monitoring and system adaptation."""
        last_update_time = datetime.now() - timedelta(hours=1)  # Force update on first run
        
        try:
            while not self._stop_event.is_set():
                try:
                    current_time = datetime.now()
                    
                    # Check for regime changes
                    for symbol in self.tracked_symbols:
                        # Get current regimes for all timeframes
                        current_regimes = self.detector.get_current_regimes(symbol)
                        
                        for timeframe, regime_info in current_regimes.items():
                            regime_type = regime_info.get('regime')
                            confidence = regime_info.get('confidence', 0.0)
                            
                            # Initialize if needed
                            if symbol not in self.active_regimes:
                                self.active_regimes[symbol] = {}
                            
                            # Check for regime change
                            if timeframe not in self.active_regimes[symbol] or self.active_regimes[symbol][timeframe] != regime_type:
                                # Regime has changed, update active regimes
                                old_regime = self.active_regimes[symbol].get(timeframe)
                                self.active_regimes[symbol][timeframe] = regime_type
                                
                                # Log transition
                                self._log_regime_transition(symbol, timeframe, old_regime, regime_type, confidence)
                    
                    # Check if it's time for a full update
                    if (current_time - last_update_time).total_seconds() > self.update_interval:
                        self._perform_system_update()
                        last_update_time = current_time
                    
                    # Sleep until next check
                    time.sleep(self.monitoring_interval)
                    
                except Exception as e:
                    logger.error(f"Error in regime monitoring loop: {str(e)}")
                    time.sleep(60)  # Sleep on error to avoid rapid retries
            
        except Exception as e:
            logger.error(f"Fatal error in monitoring thread: {str(e)}")
        finally:
            logger.info("Regime monitoring thread exiting")
    
    def _log_regime_transition(
        self, symbol: str, timeframe: str, 
        old_regime: Optional[MarketRegimeType], 
        new_regime: MarketRegimeType,
        confidence: float
    ) -> None:
        """
        Log regime transition to history.
        
        Args:
            symbol: Symbol
            timeframe: Timeframe
            old_regime: Previous regime type
            new_regime: New regime type
            confidence: Confidence level
        """
        try:
            # Initialize if needed
            if symbol not in self.regime_history:
                self.regime_history[symbol] = {}
            
            if timeframe not in self.regime_history[symbol]:
                self.regime_history[symbol][timeframe] = []
            
            # Create transition record
            transition = {
                'timestamp': datetime.now(),
                'old_regime': old_regime.value if old_regime else None,
                'new_regime': new_regime.value,
                'confidence': confidence
            }
            
            # Add to history
            self.regime_history[symbol][timeframe].append(transition)
            
            # Limit history size
            max_history = self.config.get("max_regime_history", 100)
            if len(self.regime_history[symbol][timeframe]) > max_history:
                self.regime_history[symbol][timeframe] = self.regime_history[symbol][timeframe][-max_history:]
            
            # Log the transition
            if old_regime:
                logger.info(f"Regime transition for {symbol} {timeframe}: {old_regime} -> {new_regime} (confidence: {confidence:.2f})")
            else:
                logger.info(f"Initial regime for {symbol} {timeframe}: {new_regime} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error logging regime transition: {str(e)}")
    
    def _perform_system_update(self) -> None:
        """Perform a full system update based on current regimes."""
        with self._lock:
            try:
                logger.info("Performing full system update based on market regimes")
                
                # Update strategy scores for all regimes
                for regime_type in MarketRegimeType:
                    self.strategy_selector.update_strategy_scores(regime_type)
                
                # Update strategy selection and allocation for each symbol
                for symbol in self.tracked_symbols:
                    self._update_symbol_strategies(symbol)
                
                # Update capital allocation if available
                if self.capital_allocator:
                    self._update_capital_allocation()
                
                logger.info("Completed system update")
                
            except Exception as e:
                logger.error(f"Error performing system update: {str(e)}")
    
    def _update_symbol_strategies(self, symbol: str) -> None:
        """
        Update strategies for a symbol based on current regimes.
        
        Args:
            symbol: Symbol to update
        """
        try:
            # Get primary timeframe regime
            primary_timeframe = self.config.get("primary_timeframe", "1d")
            
            # Get current regime for primary timeframe
            current_regimes = self.detector.get_current_regimes(symbol)
            
            if primary_timeframe not in current_regimes:
                logger.warning(f"No regime data for {symbol} {primary_timeframe}")
                return
            
            # Get regime and confidence
            regime_info = current_regimes[primary_timeframe]
            regime_type = regime_info.get('regime')
            confidence = regime_info.get('confidence', 0.0)
            
            if not regime_type:
                logger.warning(f"No regime type for {symbol} {primary_timeframe}")
                return
            
            # Get preferred timeframe for this regime
            trading_timeframe = self.strategy_selector.get_preferred_timeframe(
                symbol, regime_type, default_timeframe=primary_timeframe
            )
            
            # Select strategies for this symbol and regime
            selected_strategies = self.strategy_selector.select_strategies(
                symbol, regime_type, trading_timeframe
            )
            
            # Log selection
            strategy_ids = [s['strategy_id'] for s in selected_strategies]
            logger.info(f"Selected strategies for {symbol} ({regime_type}): {strategy_ids}")
            
            # Emit strategy selection event
            self.event_bus.emit("strategy_selection_update", {
                'symbol': symbol,
                'regime_type': regime_type.value,
                'timeframe': trading_timeframe,
                'strategies': selected_strategies,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating symbol strategies: {str(e)}")
    
    def _update_capital_allocation(self) -> None:
        """Update capital allocation based on current regimes and strategy selection."""
        try:
            if not self.capital_allocator:
                return
            
            logger.info("Updating capital allocation based on regime data")
            
            # Prepare allocation instruction
            allocations = {}
            
            for symbol in self.tracked_symbols:
                # Get active strategies for this symbol
                active_strategies = self.strategy_selector.get_active_strategies(symbol)
                
                # Skip if no active strategies
                if not active_strategies:
                    continue
                
                # Get strategy weights
                strategy_weights = {}
                total_weight = 0.0
                
                for strategy_id in active_strategies:
                    weight = self.strategy_selector.get_strategy_weight(symbol, strategy_id)
                    if weight > 0:
                        strategy_weights[strategy_id] = weight
                        total_weight += weight
                
                # Normalize weights
                if total_weight > 0:
                    strategy_weights = {sid: w/total_weight for sid, w in strategy_weights.items()}
                
                # Store in allocations
                allocations[symbol] = {
                    'strategies': strategy_weights,
                    'timestamp': datetime.now()
                }
            
            # Update allocations through capital allocator
            if allocations:
                # Use the regime allocator method if available
                if hasattr(self.capital_allocator, 'update_regime_allocations'):
                    self.capital_allocator.update_regime_allocations(allocations)
                    logger.info(f"Updated regime-based allocations for {len(allocations)} symbols")
                
        except Exception as e:
            logger.error(f"Error updating capital allocation: {str(e)}")
    
    def _handle_regime_change(self, event: Event) -> None:
        """
        Handle market regime change event.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            symbol = data.get('symbol')
            timeframe = data.get('timeframe')
            new_regime = data.get('new_regime')
            confidence = data.get('confidence', 0.0)
            
            if not symbol or not timeframe or not new_regime:
                return
            
            # Convert string to enum if needed
            if isinstance(new_regime, str):
                try:
                    new_regime = MarketRegimeType(new_regime)
                except ValueError:
                    logger.warning(f"Invalid regime type: {new_regime}")
                    return
            
            # Get old regime
            old_regime = None
            if symbol in self.active_regimes and timeframe in self.active_regimes[symbol]:
                old_regime = self.active_regimes[symbol][timeframe]
            
            # Update active regimes
            if symbol not in self.active_regimes:
                self.active_regimes[symbol] = {}
            
            self.active_regimes[symbol][timeframe] = new_regime
            
            # Log transition
            self._log_regime_transition(symbol, timeframe, old_regime, new_regime, confidence)
            
            # If this is the primary timeframe, update strategies
            primary_timeframe = self.config.get("primary_timeframe", "1d")
            if timeframe == primary_timeframe:
                self._update_symbol_strategies(symbol)
            
        except Exception as e:
            logger.error(f"Error handling regime change event: {str(e)}")
    
    def _handle_trade_closed(self, event: Event) -> None:
        """
        Handle trade closed event to update strategy performance.
        
        Args:
            event: Event object
        """
        try:
            data = event.data
            trade = data.get('trade')
            
            if not trade:
                return
            
            # Extract trade data
            strategy_id = trade.get('strategy_id')
            symbol = trade.get('symbol')
            
            if not strategy_id or not symbol:
                return
            
            # Get performance metrics for this trade
            profit_loss = trade.get('realized_pnl', 0.0)
            win = profit_loss > 0
            
            # Get primary timeframe
            primary_timeframe = self.config.get("primary_timeframe", "1d")
            
            # Get regime at trade close time
            regime_type = None
            
            if symbol in self.active_regimes and primary_timeframe in self.active_regimes[symbol]:
                regime_type = self.active_regimes[symbol][primary_timeframe]
            else:
                # Try to get from detector
                current_regimes = self.detector.get_current_regimes(symbol)
                if primary_timeframe in current_regimes:
                    regime_info = current_regimes[primary_timeframe]
                    regime_type = regime_info.get('regime')
            
            if not regime_type:
                logger.warning(f"Cannot determine regime for trade: {trade}")
                return
            
            # Calculate performance metrics
            metrics = {
                'profit_loss': profit_loss,
                'win': 1.0 if win else 0.0,
                'win_rate': 1.0 if win else 0.0,
                'profit_factor': 2.0 if win else 0.5,  # Simplified for single trade
                'return_pct': trade.get('return_pct', 0.0)
            }
            
            # Add returns if available
            if 'return_pct' in trade:
                metrics['returns'] = trade['return_pct'] / 100.0  # Convert percentage to decimal
            
            # Update performance tracker
            self.performance_tracker.update_performance(
                strategy_id, regime_type, metrics, symbol, primary_timeframe
            )
            
            logger.debug(f"Updated performance for strategy {strategy_id} in regime {regime_type}")
            
        except Exception as e:
            logger.error(f"Error handling trade closed event: {str(e)}")
    
    def _handle_price_update(self, event: Event) -> None:
        """
        Handle price update event for data collection.
        
        Args:
            event: Event object
        """
        # This event is already handled by the detector, so we don't need to do anything here
        pass
    
    def get_parameter_set(
        self, strategy_id: str, symbol: str, timeframe: str
    ) -> Dict[str, Any]:
        """
        Get optimized parameters for a strategy based on current regime.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Symbol
            timeframe: Timeframe
            
        Returns:
            Dict of optimized parameters
        """
        try:
            # Get current regime
            current_regimes = self.detector.get_current_regimes(symbol)
            
            if timeframe not in current_regimes:
                logger.warning(f"No regime data for {symbol} {timeframe}")
                return {}
            
            # Get regime and confidence
            regime_info = current_regimes[timeframe]
            regime_type = regime_info.get('regime')
            confidence = regime_info.get('confidence', 0.0)
            
            if not regime_type:
                logger.warning(f"No regime type for {symbol} {timeframe}")
                return {}
            
            # Get optimal parameters
            params = self.parameter_optimizer.get_optimal_parameters(
                strategy_id, regime_type, symbol, timeframe, confidence
            )
            
            return params
            
        except Exception as e:
            logger.error(f"Error getting parameter set: {str(e)}")
            return {}
    
    def get_regime_performance_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of regime performance.
        
        Returns:
            Dict with summary information
        """
        try:
            result = {
                'symbols': {},
                'regimes': {},
                'strategies': {}
            }
            
            # Summarize by symbol
            for symbol in self.tracked_symbols:
                if symbol in self.active_regimes:
                    result['symbols'][symbol] = {
                        'regimes': {tf: regime.value for tf, regime in self.active_regimes[symbol].items()},
                        'active_strategies': self.strategy_selector.get_active_strategies(symbol)
                    }
            
            # Summarize by regime
            for regime_type in MarketRegimeType:
                # Count symbols in this regime (primary timeframe)
                primary_timeframe = self.config.get("primary_timeframe", "1d")
                symbols_in_regime = [
                    symbol for symbol in self.tracked_symbols
                    if symbol in self.active_regimes and 
                    primary_timeframe in self.active_regimes[symbol] and
                    self.active_regimes[symbol][primary_timeframe] == regime_type
                ]
                
                # Get top strategies for this regime
                top_strategies = self.performance_tracker.get_best_strategies_for_regime(
                    regime_type, metric_name="profit_factor"
                )
                
                result['regimes'][regime_type.value] = {
                    'symbols_count': len(symbols_in_regime),
                    'symbols': symbols_in_regime,
                    'top_strategies': [{'strategy_id': s[0], 'score': s[1]} for s in top_strategies[:5]]
                }
                
            # Summarize by strategy
            for strategy_id in self.strategy_configs:
                # Get performance summary
                summary = self.performance_tracker.get_regime_performance_summary(strategy_id)
                
                # Convert to serializable format
                serializable_summary = {}
                for regime, metrics in summary.items():
                    serializable_summary[regime.value] = metrics
                
                result['strategies'][strategy_id] = serializable_summary
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating regime performance summary: {str(e)}")
            return {'error': str(e)}
    
    def shutdown(self) -> None:
        """Shutdown the market regime manager."""
        try:
            logger.info("Shutting down MarketRegimeManager")
            
            # Stop monitoring thread
            self._stop_monitoring()
            
            # Shutdown detector
            self.detector.shutdown()
            
            logger.info("MarketRegimeManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down market regime manager: {str(e)}")

# Helper function to easily create and initialize the system
def initialize_regime_system(
    event_bus: EventBus,
    broker_manager: MultiBrokerManager,
    trade_accounting: TradeAccounting,
    capital_allocator: Optional[CapitalAllocator] = None,
    symbols: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> MarketRegimeManager:
    """
    Initialize and start the market regime system.
    
    Args:
        event_bus: System event bus
        broker_manager: Broker manager
        trade_accounting: Trade accounting
        capital_allocator: Optional capital allocator
        symbols: Optional list of symbols to track
        config: Optional configuration
        
    Returns:
        Initialized MarketRegimeManager
    """
    try:
        logger.info("Initializing market regime system")
        
        # Create manager
        manager = MarketRegimeManager(
            event_bus, broker_manager, trade_accounting, capital_allocator, config
        )
        
        # Initialize with symbols
        if symbols:
            manager.initialize(symbols)
        
        logger.info("Market regime system initialized successfully")
        return manager
        
    except Exception as e:
        logger.error(f"Error initializing regime system: {str(e)}")
        raise
