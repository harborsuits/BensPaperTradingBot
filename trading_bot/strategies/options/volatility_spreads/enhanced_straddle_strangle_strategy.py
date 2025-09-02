#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Straddle/Strangle Strategy Implementation

This is the main implementation of the enhanced Straddle/Strangle options strategy,
integrating the modular components for market analysis, option selection, risk management,
and the volatility spread factory.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging
import uuid
from enum import Enum

# Import our modular components
from .components.market_analysis import VolatilityAnalyzer
from .components.option_selection import OptionSelector
from .components.risk_management import VolatilityRiskManager
from .components.volatility_spread_factory import VolatilitySpreadFactory

# Define strategy enumerations locally to avoid circular imports
class StrategyType(Enum):
    VOLATILITY = "volatility"

class AssetClass(Enum):
    OPTIONS = "options"

class MarketRegime(Enum):
    VOLATILE = "volatile"
    EVENT_DRIVEN = "event_driven"

class TimeFrame(Enum):
    SWING = "swing"

# Define a local register_strategy function to avoid circular imports
def register_strategy(metadata):
    def decorator(cls):
        # Store metadata on the class itself for later registration
        cls.strategy_metadata = metadata
        return cls
    return decorator

logger = logging.getLogger(__name__)


@register_strategy({
    'asset_class': AssetClass.OPTIONS.value,
    'strategy_type': StrategyType.VOLATILITY.value,
    'timeframe': TimeFrame.SWING.value,
    'compatible_market_regimes': [MarketRegime.VOLATILE.value, MarketRegime.EVENT_DRIVEN.value],
    'description': 'Enhanced Straddle/Strangle strategy for profiting from significant price movements in either direction',
    'risk_level': 'high',
    'typical_holding_period': '14-30 days',
    'performance_metrics': {
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0,
        'avg_profit_per_trade': 0.0
    },
    'broker_compatibility': ['tradier', 'alpaca'],
    'tags': ['volatility', 'neutral', 'event-driven', 'options', 'non-directional']
})
class EnhancedStraddleStrangleStrategy:
    """
    Enhanced Straddle/Strangle Options Strategy
    
    This strategy involves buying both call and put options to profit from significant
    price movements in either direction. It uses advanced analytics to:
    - Determine optimal strike selection
    - Analyze volatility surface for ideal entry timing
    - Apply sophisticated risk management
    - Adapt between straddle and strangle based on market conditions
    
    Key advantages over basic implementation:
    - Vectorized volatility calculations
    - Vega-based position sizing
    - Dynamic exit conditions based on implied volatility changes
    - Event-driven trade timing
    - Volatility regime detection
    """
    
    # Default strategy parameters
    DEFAULT_PARAMS = {
        'strategy_name': 'enhanced_straddle_strangle',
        'strategy_version': '2.0.0',
        'asset_class': 'options',
        'strategy_type': 'volatility',
        'timeframe': 'swing',
        
        # Volatility parameters
        'volatility_threshold': 0.20,        # Historical volatility threshold
        'implied_volatility_rank_min': 30,   # Minimum IV rank to consider
        
        # Strategy configuration
        'strategy_variant': 'adaptive',      # 'straddle', 'strangle', or 'adaptive'
        'atm_threshold': 0.03,               # How close to ATM for straddle
        'strangle_width_pct': 0.05,          # Strike width for strangle as % of price
        'iv_percentile_threshold': 30,       # IV percentile threshold for strategy selection
        'vix_threshold': 18,                 # VIX threshold for strategy selection
        
        # Expiration parameters
        'min_dte': 20,                       # Minimum days to expiration
        'max_dte': 45,                       # Maximum days to expiration
        'target_dte': 30,                    # Target days to expiration
        'avoid_earnings': True,              # Avoid expiration near earnings
        
        # Risk management parameters
        'profit_target_pct': 35,             # Profit target as percent of premium
        'stop_loss_pct': 60,                 # Stop loss as percent of premium
        'max_positions': 5,                  # Maximum positions
        'position_size_pct': 5,              # Position size as portfolio percentage
        'max_capital_at_risk': 20,           # Maximum capital at risk (% of portfolio)
        
        # Option chain filters
        'min_open_interest': 100,            # Minimum open interest for liquidity
        'max_bid_ask_spread_pct': 0.10,      # Maximum bid-ask spread as percentage
        
        # Event parameters
        'event_window_days': 5,              # Days before earnings/events to consider
        'avoid_event_window': True,          # Avoid entering positions close to known events
        
        # Advanced parameters
        'use_iv_surface': True,              # Use IV surface for strike selection
        'use_vega_sizing': True,             # Use vega-based position sizing
        'use_dynamic_exits': True,           # Use dynamic exit conditions
        'use_volatility_regime': True,       # Use volatility regime detection
        'cache_volatility_data': True        # Cache volatility calculations for efficiency
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None,
                 broker_adapter = None,
                 event_bus = None):
        """
        Initialize the Enhanced Straddle/Strangle strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
            broker_adapter: Optional broker adapter for direct API access
            event_bus: Optional event bus for publishing/subscribing to events
        """
        # Basic initialization
        self.strategy_id = strategy_id or f"straddle_strangle_{uuid.uuid4().hex[:8]}"
        self.name = name or "Enhanced Straddle/Strangle Strategy"
        
        # Apply parameters
        self.params = self.DEFAULT_PARAMS.copy()
        if parameters:
            self.params.update(parameters)
        
        # Initialize component classes
        self.volatility_analyzer = VolatilityAnalyzer()
        self.option_selector = OptionSelector(
            min_open_interest=self.params['min_open_interest'],
            max_bid_ask_spread_pct=self.params['max_bid_ask_spread_pct'],
            min_days_to_expiration=self.params['min_dte'],
            max_days_to_expiration=self.params['max_dte']
        )
        self.risk_manager = VolatilityRiskManager(
            max_position_size_pct=self.params['position_size_pct'] / 100,
            profit_target_pct=self.params['profit_target_pct'] / 100,
            stop_loss_pct=self.params['stop_loss_pct'] / 100
        )
        
        # Initialize the volatility spread factory
        self.spread_factory = VolatilitySpreadFactory(
            volatility_analyzer=self.volatility_analyzer,
            option_selector=self.option_selector,
            risk_manager=self.risk_manager,
            default_params=self.params
        )
        
        # Trading data
        self.universe = []                    # Universe of tradable symbols
        self.active_positions = {}            # Currently active positions
        self.historical_positions = []        # Historical position data
        self.volatility_metrics = {}          # Cached volatility metrics by symbol
        self.option_chains = {}               # Cached option chain data by symbol
        
        # External integrations
        self.broker_adapter = broker_adapter  # Direct broker API access if needed
        self.event_bus = event_bus            # Event pub/sub system
        
        # Performance metrics
        self.performance_metrics = {
            'trades_total': 0,
            'trades_won': 0,
            'trades_lost': 0,
            'profit_total': 0.0,
            'loss_total': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0
        }
        
        # Health monitoring
        self.health_metrics = {
            'last_run_time': None,
            'last_signal_time': None,
            'errors': [],
            'warnings': [],
            'status': 'initialized'
        }
        
        # Initialization
        logger.info(f"Initialized {self.name} (ID: {self.strategy_id}) with variant {self.params['strategy_variant']}")
        if self.event_bus:
            self._subscribe_to_events()
            
    def set_universe(self, symbols: List[str]) -> None:
        """
        Set the universe of tradable assets for this strategy instance.
        
        Args:
            symbols: List of symbols to trade
        """
        self.universe = symbols
        logger.info(f"Set universe with {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        
    def define_universe(self, market_data: Any) -> List[str]:
        """
        Define the universe of tradable assets for straddle/strangle strategy.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            List of selected symbols
        """
        logger.info("Defining universe for straddle/strangle strategy")
        
        # Start with all available symbols
        all_symbols = market_data.get_symbols() if hasattr(market_data, 'get_symbols') else []
        
        if not all_symbols:
            logger.warning("No symbols available in market data")
            return []
            
        # Filter criteria
        filtered_symbols = []
        
        for symbol in all_symbols:
            try:
                # Get historical data
                hist_data = market_data.get_historical_data(symbol) if hasattr(market_data, 'get_historical_data') else None
                
                if hist_data is None or len(hist_data) < 20:
                    continue
                    
                # Extract current price and average volume
                if isinstance(hist_data, pd.DataFrame) and 'close' in hist_data.columns:
                    current_price = hist_data['close'].iloc[-1]
                    avg_volume = hist_data['volume'].mean() if 'volume' in hist_data.columns else 0
                else:
                    continue
                    
                # Apply price and volume filters
                if (current_price >= self.params.get('min_stock_price', 20.0) and 
                    current_price <= self.params.get('max_stock_price', 1000.0) and
                    avg_volume >= self.params.get('min_avg_volume', 500000)):
                    
                    # Calculate historical volatility
                    historical_vol = self.volatility_analyzer.calculate_historical_volatility(
                        hist_data, period=20, return_type='latest'
                    )
                    
                    # Only include symbols with sufficient volatility
                    if historical_vol >= self.params['volatility_threshold']:
                        filtered_symbols.append(symbol)
                        
                        # Store volatility metrics for future use
                        self.volatility_metrics[symbol] = {
                            'historical_volatility': historical_vol,
                            'last_updated': datetime.now().isoformat()
                        }
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                
        # Sort by volatility (highest first)
        if filtered_symbols and self.volatility_metrics:
            filtered_symbols.sort(
                key=lambda x: self.volatility_metrics.get(x, {}).get('historical_volatility', 0),
                reverse=True
            )
            
        # Cap the universe size
        max_universe_size = self.params.get('max_universe_size', 20)
        if len(filtered_symbols) > max_universe_size:
            filtered_symbols = filtered_symbols[:max_universe_size]
            
        logger.info(f"Defined universe with {len(filtered_symbols)} symbols")
        return filtered_symbols
        
    def generate_signals(self, market_data: Any, option_chains: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Generate straddle/strangle signals based on market data and option chains.
        
        Args:
            market_data: Market data for analysis
            option_chains: Option chain data
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        logger.info("Generating signals for straddle/strangle strategy")
        
        # Use provided universe or define one
        symbols = self.universe
        if not symbols and hasattr(self, 'define_universe'):
            symbols = self.define_universe(market_data)
            
        if not symbols:
            logger.warning("No symbols in universe for signal generation")
            return []
            
        # Track current time for logging
        generation_start = datetime.now()
        
        # Analyze each symbol
        for symbol in symbols:
            try:
                logger.info(f"Analyzing {symbol} for straddle/strangle opportunities")
                
                # Get market data for this symbol
                hist_data = self._get_historical_data(market_data, symbol)
                if hist_data is None or (isinstance(hist_data, pd.DataFrame) and hist_data.empty):
                    logger.warning(f"No historical data for {symbol}")
                    continue
                    
                # Get current price
                current_price = self._get_current_price(hist_data, symbol)
                if not current_price or current_price <= 0:
                    logger.warning(f"Invalid price for {symbol}: {current_price}")
                    continue
                    
                # Get option chain
                symbol_options = self._get_option_chain(option_chains, symbol)
                if not symbol_options or (isinstance(symbol_options, pd.DataFrame) and symbol_options.empty):
                    logger.warning(f"No option chain data for {symbol}")
                    continue
                    
                # Analyze volatility if not already cached
                volatility_data = self._analyze_volatility(symbol, hist_data)
                    
                # Check if volatility meets threshold
                if volatility_data.get('historical_volatility', 0) < self.params['volatility_threshold']:
                    logger.info(f"{symbol} historical volatility {volatility_data.get('historical_volatility', 0):.2%} below threshold")
                    continue
                    
                # Check for upcoming events if configured to avoid them
                if self.params.get('avoid_event_window') and self._has_upcoming_event(symbol):
                    logger.info(f"Skipping {symbol} due to upcoming event within window")
                    continue
                    
                # Generate trade using the volatility spread factory
                trade = self.spread_factory.generate_volatility_trade(
                    symbol=symbol,
                    current_price=current_price,
                    option_chain=symbol_options,
                    market_data=hist_data,
                    volatility_data=volatility_data,
                    strategy_type=self.params['strategy_variant']
                )
                
                if not trade:
                    logger.info(f"No viable trade found for {symbol}")
                    continue
                    
                # Convert trade to signal format
                signal = self._create_signal_from_trade(trade)
                
                if signal:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                self.health_metrics['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'error': str(e),
                    'component': 'generate_signals'
                })
                
        # Log signal generation results
        generation_time = (datetime.now() - generation_start).total_seconds()
        logger.info(f"Generated {len(signals)} signals in {generation_time:.2f} seconds")
        
        # Update metrics
        self.health_metrics['last_run_time'] = datetime.now().isoformat()
        if signals:
            self.health_metrics['last_signal_time'] = datetime.now().isoformat()
            
        # Publish to event bus if available
        if self.event_bus and signals:
            self._publish_signals_event(signals)
            
        return signals
        
    def _get_historical_data(self, market_data: Any, symbol: str) -> pd.DataFrame:
        """
        Get historical data for a symbol with error handling.
        
        Args:
            market_data: Market data object
            symbol: Symbol to get data for
            
        Returns:
            DataFrame with historical data or None if unavailable
        """
        try:
            # Try different methods to get historical data
            if hasattr(market_data, 'get_historical_data'):
                return market_data.get_historical_data(symbol)
            elif hasattr(market_data, 'history') and callable(market_data.history):
                return market_data.history(symbol)
            elif isinstance(market_data, dict) and symbol in market_data:
                return market_data[symbol]
            else:
                logger.warning(f"Unsupported market_data type for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None
            
    def _get_current_price(self, hist_data: pd.DataFrame, symbol: str) -> float:
        """
        Extract current price from historical data with error handling.
        
        Args:
            hist_data: Historical data for the symbol
            symbol: Symbol for logging
            
        Returns:
            Current price or 0 if unavailable
        """
        try:
            if isinstance(hist_data, pd.DataFrame):
                if 'close' in hist_data.columns:
                    return hist_data['close'].iloc[-1]
                elif 'Close' in hist_data.columns:
                    return hist_data['Close'].iloc[-1]
            return 0
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {e}")
            return 0
            
    def _get_option_chain(self, option_chains: Any, symbol: str) -> Any:
        """
        Get option chain for a symbol with error handling.
        
        Args:
            option_chains: Option chain object or dictionary
            symbol: Symbol to get options for
            
        Returns:
            Option chain data or None if unavailable
        """
        try:
            if option_chains is None:
                return None
                
            # Try different methods to get option chains
            if hasattr(option_chains, 'get_chain_for_symbol'):
                return option_chains.get_chain_for_symbol(symbol)
            elif hasattr(option_chains, 'get_chain') and callable(option_chains.get_chain):
                return option_chains.get_chain(symbol)
            elif isinstance(option_chains, dict) and symbol in option_chains:
                return option_chains[symbol]
            else:
                # For direct DataFrame option chains
                if isinstance(option_chains, pd.DataFrame):
                    if 'symbol' in option_chains.columns:
                        return option_chains[option_chains['symbol'] == symbol]
                    elif 'underlying' in option_chains.columns:
                        return option_chains[option_chains['underlying'] == symbol]
                        
                logger.warning(f"Unsupported option_chains type for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {e}")
            return None
            
    def _analyze_volatility(self, symbol: str, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive volatility analysis for a symbol.
        
        Args:
            symbol: Symbol to analyze
            hist_data: Historical price data
            
        Returns:
            Dictionary with volatility metrics
        """
        # Check if we have recently cached data and should use it
        if (symbol in self.volatility_metrics and 
            self.params.get('cache_volatility_data', True) and
            'last_updated' in self.volatility_metrics[symbol]):
            
            last_updated = datetime.fromisoformat(self.volatility_metrics[symbol]['last_updated'])
            cache_age = (datetime.now() - last_updated).seconds / 60  # minutes
            
            if cache_age < 30:  # Use cache if less than 30 minutes old
                return self.volatility_metrics[symbol]
        
        # Calculate historical volatility for multiple periods
        vol_20d = self.volatility_analyzer.calculate_historical_volatility(
            hist_data, period=20, return_type='latest'
        )
        
        vol_60d = self.volatility_analyzer.calculate_historical_volatility(
            hist_data, period=60, return_type='latest'
        )
        
        # Calculate full volatility series for percentile calculation
        vol_series = self.volatility_analyzer.calculate_historical_volatility(
            hist_data, period=20, return_type='series'
        )
        
        # Calculate percentile of current volatility
        vol_percentile = self.volatility_analyzer.calculate_volatility_percentile(
            vol_20d, vol_series
        )
        
        # Detect volatility regime
        vol_regime = self.volatility_analyzer.detect_volatility_regime(
            vol_series
        )
        
        # Store all calculated metrics
        self.volatility_metrics[symbol] = {
            'historical_volatility': vol_20d,
            'historical_volatility_60d': vol_60d,
            'volatility_percentile': vol_percentile,
            'regime': vol_regime.get('regime'),
            'regime_confidence': vol_regime.get('confidence'),
            'last_updated': datetime.now().isoformat()
        }
        
        return self.volatility_metrics[symbol]
    
    def _has_upcoming_event(self, symbol: str) -> bool:
        """
        Check if a symbol has an upcoming event within the configured window.
        
        Args:
            symbol: Symbol to check for events
            
        Returns:
            True if there is an upcoming event, False otherwise
        """
        # This would integrate with an economic/earnings calendar service
        # For now, return a placeholder implementation
        return False
        
    def _create_signal_from_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a trade generated by the factory into a standardized signal format.
        
        Args:
            trade: Trade configuration from the factory
            
        Returns:
            Signal dictionary
        """
        if not trade or 'option_data' not in trade:
            return None
            
        # Extract key data from trade
        symbol = trade.get('symbol')
        strategy_type = trade.get('strategy_type')
        option_data = trade.get('option_data', {})
        position_size = trade.get('position_size', {})
        exit_conditions = trade.get('exit_conditions', {})
        volatility_data = trade.get('volatility_data', {})
        
        # Calculate signal confidence based on multiple factors
        base_confidence = 0.7  # Starting confidence level
        
        # Adjust based on volatility metrics
        if 'volatility_percentile' in volatility_data:
            vol_percentile = volatility_data['volatility_percentile']
            if vol_percentile > 80:
                confidence_adj = 0.1  # High vol percentile is good for volatility strategies
            elif vol_percentile < 20:
                confidence_adj = -0.1
            else:
                confidence_adj = (vol_percentile - 50) / 300  # Small adjustment based on percentile
            base_confidence += confidence_adj
        
        # Adjust based on strategy economics
        if 'expected_value_ratio' in exit_conditions:
            ev_ratio = exit_conditions['expected_value_ratio']
            if ev_ratio > 0.2:
                base_confidence += 0.1
            elif ev_ratio < 0:
                base_confidence -= 0.2
                
        # Adjust based on option metrics
        if 'required_move_pct' in option_data:
            req_move = option_data['required_move_pct']
            hist_vol = volatility_data.get('historical_volatility', 0.2)
            # If required move is less than 1 std dev, higher confidence
            if req_move < hist_vol * 0.7:
                base_confidence += 0.05
            # If required move is more than 2 std dev, lower confidence
            elif req_move > hist_vol * 1.5:
                base_confidence -= 0.1
                
        # Ensure confidence is between 0.1 and 0.95
        confidence = max(0.1, min(0.95, base_confidence))
        
        # Create the signal dictionary
        signal = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'strategy_id': self.strategy_id,
            'strategy_type': strategy_type,
            'action': 'BUY',  # For long options strategies
            'confidence': confidence,
            'direction': 'NEUTRAL',  # Straddle/strangle are direction-neutral
            'current_price': trade.get('current_price'),
            'options': option_data,
            'position_size': position_size,
            'exit_conditions': exit_conditions,
            'metadata': {
                'volatility': volatility_data.get('historical_volatility', 0),
                'volatility_percentile': volatility_data.get('volatility_percentile', 0),
                'volatility_regime': volatility_data.get('regime', 'unknown'),
                'days_to_expiration': option_data.get('days_to_expiration', 30),
                'expected_value_ratio': exit_conditions.get('expected_value_ratio', 0),
                'reason': f"Volatility setup using {strategy_type} strategy",
                'signal_version': '2.0'
            }
        }
        
        return signal
        
    def _subscribe_to_events(self) -> None:
        """
        Subscribe to relevant events from the event bus.
        """
        if not self.event_bus:
            return
            
        # Subscribe to market data events
        self.event_bus.subscribe('market_data', self._on_market_data)
        
        # Subscribe to option chain updates
        self.event_bus.subscribe('option_chain_update', self._on_option_chain_update)
        
        # Subscribe to position updates
        self.event_bus.subscribe('position_update', self._on_position_update)
        
        # Subscribe to volatility events
        self.event_bus.subscribe('volatility_alert', self._on_volatility_alert)
        
        logger.info(f"Subscribed to events for {self.name}")
        
    def _publish_signals_event(self, signals: List[Dict[str, Any]]) -> None:
        """
        Publish generated signals to the event bus.
        
        Args:
            signals: List of generated signals
        """
        if not self.event_bus:
            return
            
        self.event_bus.publish('signals_generated', {
            'strategy_id': self.strategy_id,
            'strategy_name': self.name,
            'signal_count': len(signals),
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        })
        
    def _on_market_data(self, event_data: Dict[str, Any]) -> None:
        """
        Handle incoming market data events.
        
        Args:
            event_data: Event data from the event bus
        """
        # This would be implemented to handle real-time market data
        pass
        
    def _on_option_chain_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle option chain update events.
        
        Args:
            event_data: Event data from the event bus
        """
        # This would be implemented to handle real-time option chain updates
        pass
        
    def _on_position_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position update events.
        
        Args:
            event_data: Event data from the event bus
        """
        # This would be implemented to handle position updates from the broker
        pass
        
    def _on_volatility_alert(self, event_data: Dict[str, Any]) -> None:
        """
        Handle volatility alert events.
        
        Args:
            event_data: Event data from the event bus
        """
        # This would be implemented to handle volatility spike/crash alerts
        pass
        
    # Position Management Methods
    
    def manage_positions(self, market_data: Any, option_chains: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Evaluate and manage existing positions based on current market conditions.
        
        Args:
            market_data: Current market data
            option_chains: Current option chain data
            
        Returns:
            List of position management actions
        """
        if not hasattr(self, 'positions') or not self.positions:
            return []
            
        actions = []
        logger.info(f"Managing {len(self.positions)} open positions")
        
        # Track management time for performance
        start_time = datetime.now()
        
        for position_id, position in self.positions.items():
            try:
                # Extract position details
                symbol = position.get('symbol')
                entry_price = position.get('entry_price')
                position_type = position.get('position_type')  # straddle, strangle, etc.
                entry_time = position.get('entry_time')
                options_data = position.get('options', {})  # Contains legs of the spread
                exit_conditions = position.get('exit_conditions', {})
                
                if not symbol:
                    logger.warning(f"Position {position_id} missing symbol")
                    continue
                    
                # Get current market data for the symbol
                hist_data = self._get_historical_data(market_data, symbol)
                if hist_data is None:
                    logger.warning(f"No market data for {symbol}, skipping position check")
                    continue
                    
                # Get current option chain if available
                current_options = self._get_option_chain(option_chains, symbol)
                    
                # Evaluate position metrics
                position_metrics = self._evaluate_position_metrics(position, hist_data, current_options)
                
                # Update position with latest metrics
                self.positions[position_id].update({
                    'current_metrics': position_metrics,
                    'last_updated': datetime.now().isoformat()
                })
                
                # Check exit conditions
                exit_action = self._check_exit_conditions(position, position_metrics)
                
                if exit_action:
                    exit_action.update({
                        'position_id': position_id,
                        'symbol': symbol,
                        'position_type': position_type
                    })
                    actions.append(exit_action)
                    
                    # Mark position for potential closing
                    self.positions[position_id]['pending_exit'] = True
                    self.positions[position_id]['exit_signal'] = exit_action
                    
            except Exception as e:
                logger.error(f"Error managing position {position_id}: {e}")
                self.health_metrics['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'position_id': position_id,
                    'error': str(e),
                    'component': 'manage_positions'
                })
                
        # Log management results
        management_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Generated {len(actions)} position management actions in {management_time:.2f} seconds")
        
        # Update metrics
        self.health_metrics['last_position_check'] = datetime.now().isoformat()
        
        # Publish to event bus if available
        if self.event_bus and actions:
            self.event_bus.publish('position_actions', {
                'strategy_id': self.strategy_id,
                'action_count': len(actions),
                'actions': actions,
                'timestamp': datetime.now().isoformat()
            })
            
        return actions
        
    def _evaluate_position_metrics(self, position: Dict[str, Any], market_data: pd.DataFrame, 
                                  option_chain: Optional[Any] = None) -> Dict[str, Any]:
        """
        Calculate current metrics for an open position.
        
        Args:
            position: Position data dictionary
            market_data: Current market data for the symbol
            option_chain: Current option chain data
            
        Returns:
            Dictionary of position metrics
        """
        symbol = position.get('symbol')
        options_data = position.get('options', {})
        
        # Extract current price
        current_price = self._get_current_price(market_data, symbol)
        
        # Initialize metrics
        metrics = {
            'current_price': current_price,
            'price_change_pct': 0,
            'days_held': 0,
            'current_value': 0,
            'pnl': 0,
            'pnl_pct': 0,
            'iv_at_entry': 0,
            'current_iv': 0,
            'iv_change_pct': 0,
            'theta_exposure': 0,
            'vega_exposure': 0,
            'delta_exposure': 0,
            'gamma_exposure': 0
        }
        
        # Calculate days held
        if 'entry_time' in position:
            entry_time = datetime.fromisoformat(position['entry_time']) \
                if isinstance(position['entry_time'], str) else position['entry_time']
            days_held = (datetime.now() - entry_time).days
            metrics['days_held'] = days_held
        
        # Calculate price change
        if 'entry_price' in position and current_price:
            entry_price = position['entry_price']
            price_change_pct = (current_price - entry_price) / entry_price if entry_price else 0
            metrics['price_change_pct'] = price_change_pct
        
        # Add volatility metrics if we can calculate them
        try:
            # Calculate current IV if we have market data
            hist_data = market_data
            current_iv = self.volatility_analyzer.calculate_historical_volatility(
                hist_data, period=20, return_type='latest'
            )
            
            # Get IV at entry if recorded
            iv_at_entry = position.get('volatility_at_entry', options_data.get('implied_volatility', 0))
            
            # Calculate change
            iv_change_pct = (current_iv - iv_at_entry) / iv_at_entry if iv_at_entry else 0
            
            metrics.update({
                'iv_at_entry': iv_at_entry,
                'current_iv': current_iv,
                'iv_change_pct': iv_change_pct
            })
        except Exception as e:
            logger.error(f"Error calculating volatility metrics for {symbol}: {e}")
        
        # Calculate current option values and Greeks if we have option chain data
        if option_chain is not None:
            try:
                total_value = 0
                total_theta = 0
                total_vega = 0
                total_delta = 0
                total_gamma = 0
                
                # Process each option leg in the spread
                for leg in options_data.get('legs', []):
                    # Extract leg data
                    strike = leg.get('strike')
                    option_type = leg.get('option_type')  # 'call' or 'put'
                    expiration = leg.get('expiration')
                    quantity = leg.get('quantity', 1)
                    
                    # Find current option in the chain
                    current_option = self._find_option_in_chain(
                        option_chain, strike, option_type, expiration
                    )
                    
                    if current_option:
                        # Extract current price and Greeks
                        current_premium = current_option.get('last', current_option.get('ask', 0))
                        current_theta = current_option.get('theta', 0) * quantity
                        current_vega = current_option.get('vega', 0) * quantity
                        current_delta = current_option.get('delta', 0) * quantity
                        current_gamma = current_option.get('gamma', 0) * quantity
                        
                        # Add to totals
                        total_value += current_premium * quantity
                        total_theta += current_theta
                        total_vega += current_vega
                        total_delta += current_delta
                        total_gamma += current_gamma
                        
                        # Store current leg metrics
                        leg['current_metrics'] = {
                            'current_premium': current_premium,
                            'theta': current_theta,
                            'vega': current_vega,
                            'delta': current_delta,
                            'gamma': current_gamma,
                            'last_updated': datetime.now().isoformat()
                        }
                
                # Calculate P&L
                entry_value = position.get('entry_value', 0)
                pnl = total_value - entry_value
                pnl_pct = pnl / entry_value if entry_value else 0
                
                # Update metrics with option values
                metrics.update({
                    'current_value': total_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'theta_exposure': total_theta,
                    'vega_exposure': total_vega,
                    'delta_exposure': total_delta,
                    'gamma_exposure': total_gamma
                })
                
            except Exception as e:
                logger.error(f"Error calculating Greeks for {symbol}: {e}")
        
        return metrics
        
    def _find_option_in_chain(self, option_chain: Any, strike: float, option_type: str, 
                            expiration: str) -> Optional[Dict[str, Any]]:
        """
        Find a specific option in an option chain.
        
        Args:
            option_chain: Option chain data
            strike: Strike price to find
            option_type: Option type ('call' or 'put')
            expiration: Expiration date
            
        Returns:
            Option data dictionary or None if not found
        """
        try:
            # Handle different option chain formats
            if isinstance(option_chain, pd.DataFrame):
                # Filter by strike, type, and expiration
                option_filter = (
                    (option_chain['strike'] == strike) & 
                    (option_chain['option_type'].str.lower() == option_type.lower()) & 
                    (option_chain['expiration'] == expiration)
                )
                
                if any(option_filter):
                    return option_chain[option_filter].iloc[0].to_dict()
                return None
                
            elif isinstance(option_chain, dict):
                # Check if organized by expiration
                if expiration in option_chain:
                    exp_chain = option_chain[expiration]
                    option_key = f"{strike}_{option_type.lower()}"
                    
                    if option_key in exp_chain:
                        return exp_chain[option_key]
                        
                # Check if organized by option type
                elif option_type.lower() in option_chain:
                    type_chain = option_chain[option_type.lower()]
                    
                    if expiration in type_chain and strike in type_chain[expiration]:
                        return type_chain[expiration][strike]
                        
            # Generic fallback search for nested structures
            return self._recursive_option_search(option_chain, strike, option_type, expiration)
                
        except Exception as e:
            logger.error(f"Error finding option in chain: {e}")
            return None
            
    def _recursive_option_search(self, data: Any, strike: float, option_type: str, 
                               expiration: str) -> Optional[Dict[str, Any]]:
        """
        Recursively search a nested data structure for an option matching the criteria.
        
        Args:
            data: Data structure to search
            strike: Strike price to find
            option_type: Option type ('call' or 'put')
            expiration: Expiration date
            
        Returns:
            Option data dictionary or None if not found
        """
        if isinstance(data, dict):
            # Check if this dictionary represents the option we're looking for
            if ('strike' in data and data['strike'] == strike and
                'option_type' in data and data['option_type'].lower() == option_type.lower() and
                'expiration' in data and data['expiration'] == expiration):
                return data
                
            # Otherwise search each value
            for value in data.values():
                result = self._recursive_option_search(value, strike, option_type, expiration)
                if result:
                    return result
                    
        elif isinstance(data, list) or isinstance(data, tuple):
            # Search through each item in the list
            for item in data:
                result = self._recursive_option_search(item, strike, option_type, expiration)
                if result:
                    return result
                    
        return None
        
    def _check_exit_conditions(self, position: Dict[str, Any], metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if any exit conditions are met for the position.
        
        Args:
            position: Position data dictionary
            metrics: Current position metrics
            
        Returns:
            Exit action dictionary or None if no exit needed
        """
        # Extract exit conditions from position
        exit_conditions = position.get('exit_conditions', {})
        
        # No exit conditions defined
        if not exit_conditions:
            return None
            
        # Initialize exit reason
        exit_reasons = []
        exit_scores = []
        
        # Check profit target
        profit_target = exit_conditions.get('profit_target', 0.5)  # 50% profit by default
        if metrics.get('pnl_pct', 0) >= profit_target:
            exit_reasons.append(f"Profit target of {profit_target:.1%} reached")
            exit_scores.append(0.9)  # High score for profit target
            
        # Check stop loss
        stop_loss = exit_conditions.get('stop_loss', -0.5)  # 50% loss by default
        if metrics.get('pnl_pct', 0) <= stop_loss:
            exit_reasons.append(f"Stop loss of {stop_loss:.1%} triggered")
            exit_scores.append(0.95)  # Very high score for stop loss
            
        # Check max days to hold
        max_days = exit_conditions.get('max_days', 45)  # 45 days by default
        if metrics.get('days_held', 0) >= max_days:
            exit_reasons.append(f"Maximum hold time of {max_days} days reached")
            exit_scores.append(0.8)  # High score for time-based exit
            
        # Check volatility change threshold
        vol_change_exit = exit_conditions.get('volatility_change_exit', 0.3)  # 30% IV decline
        if metrics.get('iv_change_pct', 0) <= -vol_change_exit:
            exit_reasons.append(f"Volatility declined by {-metrics.get('iv_change_pct', 0):.1%}")
            exit_scores.append(0.75)  # Medium-high score for vol change
            
        # Check absolute move in underlying
        price_move_threshold = exit_conditions.get('price_move_threshold', 0.2)  # 20% price move
        if abs(metrics.get('price_change_pct', 0)) >= price_move_threshold:
            move_direction = "up" if metrics.get('price_change_pct', 0) > 0 else "down"
            exit_reasons.append(f"Price moved {move_direction} by {abs(metrics.get('price_change_pct', 0)):.1%}")
            exit_scores.append(0.7)  # Medium score for price move
            
        # Check time decay (for short-term trades)
        if exit_conditions.get('avoid_theta_decay', False) and metrics.get('theta_exposure', 0) < -50:
            remaining_days = exit_conditions.get('theta_decay_threshold', 7)
            current_days_to_expiry = self._get_days_to_expiry(position)
            
            if current_days_to_expiry <= remaining_days:
                exit_reasons.append(f"Avoiding theta decay with {current_days_to_expiry} days to expiry")
                exit_scores.append(0.65)  # Medium score for theta decay
                
        # Check for custom exit conditions in the strategy variant
        strategy_type = position.get('position_type', position.get('strategy_type', 'unknown'))
        
        if strategy_type == 'straddle':
            # Specific exit for straddle (both sides moving against us)
            if (metrics.get('delta_exposure', 0) > 0.4 or metrics.get('delta_exposure', 0) < -0.4):
                exit_reasons.append(f"Delta exposure at {metrics.get('delta_exposure', 0):.2f}")
                exit_scores.append(0.6)  # Lower score for delta exposure
                
        elif strategy_type == 'strangle':
            # Specific exit for strangle (IV collapsed)
            if metrics.get('iv_change_pct', 0) <= -0.2 and metrics.get('vega_exposure', 0) > 100:
                exit_reasons.append(f"IV collapsed with high vega exposure")
                exit_scores.append(0.7)  # Medium score for IV collapse
        
        # If we have any exit reasons, generate the exit action
        if exit_reasons:
            # Calculate overall exit score (confidence)
            overall_score = max(exit_scores) if exit_scores else 0.5
            
            return {
                'action': 'EXIT',
                'reasons': exit_reasons,
                'confidence': overall_score,
                'metrics_at_exit': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        return None
        
    def _get_days_to_expiry(self, position: Dict[str, Any]) -> int:
        """
        Calculate the minimum days to expiry across all legs of the position.
        
        Args:
            position: Position data dictionary
            
        Returns:
            Days to expiry or a large number if not calculable
        """
        try:
            options_data = position.get('options', {})
            legs = options_data.get('legs', [])
            
            if not legs:
                return 999  # No legs found
                
            # Find the minimum days to expiry
            min_days = 999
            today = datetime.now().date()
            
            for leg in legs:
                expiration = leg.get('expiration')
                if not expiration:
                    continue
                    
                # Handle different date string formats
                try:
                    if isinstance(expiration, str):
                        # Try different formats
                        if '-' in expiration:
                            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                        elif '/' in expiration:
                            exp_date = datetime.strptime(expiration, '%m/%d/%Y').date()
                        else:
                            exp_date = datetime.strptime(expiration, '%Y%m%d').date()
                    else:
                        # Assume it's already a date or datetime
                        exp_date = expiration.date() if hasattr(expiration, 'date') else expiration
                        
                    days_to_expiry = (exp_date - today).days
                    min_days = min(min_days, days_to_expiry)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Error parsing expiration date {expiration}: {e}")
                    continue
                    
            return max(0, min_days)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating days to expiry: {e}")
            return 999  # Error case, return a large number
    
    # Performance Tracking and Reporting
    
    def calculate_strategy_performance(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for the strategy.
        
        Returns:
            Dictionary of performance metrics
        """
        performance = {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'variant': self.params['strategy_variant'],
            'timestamp': datetime.now().isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'avg_days_held': 0,
            'volatility_metrics': {},
            'position_metrics': {}
        }
        
        # Return empty performance if no closed positions
        if not hasattr(self, 'closed_positions') or not self.closed_positions:
            return performance
            
        # Extract key metrics from closed positions
        total_pnl = 0
        pnl_values = []
        days_held_values = []
        profit_trades = []
        loss_trades = []
        entry_iv_values = []
        exit_iv_values = []
        
        for position in self.closed_positions:
            # Get final position metrics
            final_metrics = position.get('final_metrics', position.get('current_metrics', {}))
            pnl = final_metrics.get('pnl', 0)
            days_held = final_metrics.get('days_held', 0)
            
            # Track profit/loss
            total_pnl += pnl
            pnl_values.append(pnl)
            days_held_values.append(days_held)
            
            # Track wins and losses
            if pnl > 0:
                profit_trades.append(pnl)
                performance['winning_trades'] += 1
            else:
                loss_trades.append(pnl)
                performance['losing_trades'] += 1
                
            # Track volatility metrics
            if 'iv_at_entry' in final_metrics:
                entry_iv_values.append(final_metrics['iv_at_entry'])
            if 'current_iv' in final_metrics:
                exit_iv_values.append(final_metrics['current_iv'])
        
        # Calculate aggregate metrics
        performance['total_trades'] = len(self.closed_positions)
        performance['total_pnl'] = total_pnl
        
        if performance['total_trades'] > 0:
            performance['win_rate'] = performance['winning_trades'] / performance['total_trades']
            performance['avg_days_held'] = sum(days_held_values) / performance['total_trades']
            
        if profit_trades:
            performance['avg_profit'] = sum(profit_trades) / len(profit_trades)
            
        if loss_trades:
            performance['avg_loss'] = sum(loss_trades) / len(loss_trades)
            
        # Calculate Sharpe ratio (if we have daily P&L data)
        if pnl_values and len(pnl_values) > 1:
            try:
                daily_returns = np.array(pnl_values)
                sharpe = daily_returns.mean() / (daily_returns.std() + 1e-10) * np.sqrt(252)
                performance['sharpe_ratio'] = float(sharpe)  # Convert from numpy to float
            except Exception as e:
                logger.error(f"Error calculating Sharpe ratio: {e}")
                
        # Calculate max drawdown
        if pnl_values:
            try:
                cumulative_pnl = np.cumsum(pnl_values)
                max_dd = 0
                peak = cumulative_pnl[0]
                
                for value in cumulative_pnl:
                    if value > peak:
                        peak = value
                    dd = (peak - value) / (peak + 1e-10)  # Avoid division by zero
                    max_dd = max(max_dd, dd)
                    
                performance['max_drawdown'] = float(max_dd)  # Convert from numpy to float
            except Exception as e:
                logger.error(f"Error calculating max drawdown: {e}")
                
        # Calculate volatility success metrics
        if entry_iv_values and exit_iv_values and len(entry_iv_values) == len(exit_iv_values):
            try:
                # Average IV change
                iv_changes = [exit - entry for exit, entry in zip(exit_iv_values, entry_iv_values)]
                avg_iv_change = sum(iv_changes) / len(iv_changes)
                
                # Correlation between IV change and P&L
                if len(iv_changes) == len(pnl_values):
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(iv_changes, pnl_values)[0, 1]
                    
                    performance['volatility_metrics'] = {
                        'avg_entry_iv': sum(entry_iv_values) / len(entry_iv_values),
                        'avg_exit_iv': sum(exit_iv_values) / len(exit_iv_values),
                        'avg_iv_change': avg_iv_change,
                        'iv_pnl_correlation': float(correlation)  # Convert from numpy to float
                    }
            except Exception as e:
                logger.error(f"Error calculating volatility metrics: {e}")
                
        # Add position metrics by strategy type
        strategy_types = {}
        for position in self.closed_positions:
            position_type = position.get('position_type', position.get('strategy_type', 'unknown'))
            if position_type not in strategy_types:
                strategy_types[position_type] = []
            strategy_types[position_type].append(position)
            
        # Calculate metrics for each strategy type
        for strategy_type, positions in strategy_types.items():
            type_pnl = sum([p.get('final_metrics', {}).get('pnl', 0) for p in positions])
            type_wins = sum([1 for p in positions if p.get('final_metrics', {}).get('pnl', 0) > 0])
            
            performance['position_metrics'][strategy_type] = {
                'count': len(positions),
                'pnl': type_pnl,
                'win_rate': type_wins / len(positions) if positions else 0
            }
            
        return performance
        
    def register_filled_position(self, trade_data: Dict[str, Any]) -> str:
        """
        Register a new filled position in the strategy's position tracking.
        
        Args:
            trade_data: Trade execution data
            
        Returns:
            Position ID of the registered position
        """
        # Generate a unique position ID
        position_id = f"{self.strategy_id}_{len(self.positions) + len(self.closed_positions) + 1}"
        
        # Extract key trade data
        symbol = trade_data.get('symbol')
        entry_price = trade_data.get('entry_price')
        options_data = trade_data.get('options', {})
        position_type = trade_data.get('strategy_type', self.params['strategy_variant'])
        
        # Create the position record
        position = {
            'position_id': position_id,
            'symbol': symbol,
            'position_type': position_type,
            'entry_price': entry_price,  # Underlying price at entry
            'entry_time': datetime.now().isoformat(),
            'entry_value': trade_data.get('total_premium', 0) * trade_data.get('quantity', 1),
            'options': options_data,
            'exit_conditions': trade_data.get('exit_conditions', {}),
            'status': 'OPEN',
            'volatility_at_entry': trade_data.get('volatility_data', {}).get('historical_volatility', 0),
            'metadata': trade_data.get('metadata', {})
        }
        
        # Store the position
        self.positions[position_id] = position
        
        # Log position creation
        logger.info(f"Registered new {position_type} position {position_id} for {symbol}")
        
        # Update strategy metrics
        self.health_metrics['open_position_count'] = len(self.positions)
        self.health_metrics['last_position_opened'] = datetime.now().isoformat()
        
        # Publish to event bus if available
        if self.event_bus:
            self.event_bus.publish('position_opened', {
                'strategy_id': self.strategy_id,
                'position_id': position_id,
                'symbol': symbol,
                'position_type': position_type,
                'entry_value': position['entry_value'],
                'timestamp': datetime.now().isoformat()
            })
            
        return position_id
        
    def register_closed_position(self, position_id: str, exit_data: Dict[str, Any]) -> bool:
        """
        Register a position as closed and move it to closed_positions.
        
        Args:
            position_id: ID of the position to close
            exit_data: Data about the exit execution
            
        Returns:
            True if position was found and closed, False otherwise
        """
        if position_id not in self.positions:
            logger.warning(f"Cannot close position {position_id}: not found")
            return False
            
        # Get the position data
        position = self.positions[position_id]
        
        # Update with exit information
        position.update({
            'status': 'CLOSED',
            'exit_time': datetime.now().isoformat(),
            'exit_price': exit_data.get('exit_price'),
            'exit_value': exit_data.get('exit_value'),
            'final_metrics': exit_data.get('metrics', {}),
            'exit_reasons': exit_data.get('reasons', [])
        })
        
        # Calculate P&L if not provided
        if 'pnl' not in position.get('final_metrics', {}) and 'exit_value' in position and 'entry_value' in position:
            pnl = position['exit_value'] - position['entry_value']
            if 'final_metrics' not in position:
                position['final_metrics'] = {}
            position['final_metrics']['pnl'] = pnl
            position['final_metrics']['pnl_pct'] = pnl / position['entry_value'] if position['entry_value'] else 0
        
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        # Log position closure
        pnl = position.get('final_metrics', {}).get('pnl', 0)
        logger.info(f"Closed position {position_id} with P&L: {pnl}")
        
        # Update strategy metrics
        self.health_metrics['open_position_count'] = len(self.positions)
        self.health_metrics['closed_position_count'] = len(self.closed_positions)
        self.health_metrics['last_position_closed'] = datetime.now().isoformat()
        
        if pnl > 0:
            self.health_metrics['profitable_trades'] = self.health_metrics.get('profitable_trades', 0) + 1
        else:
            self.health_metrics['unprofitable_trades'] = self.health_metrics.get('unprofitable_trades', 0) + 1
            
        # Recalculate strategy performance
        self.performance_metrics = self.calculate_strategy_performance()
        
        # Publish to event bus if available
        if self.event_bus:
            self.event_bus.publish('position_closed', {
                'strategy_id': self.strategy_id,
                'position_id': position_id,
                'symbol': position.get('symbol'),
                'position_type': position.get('position_type'),
                'pnl': pnl,
                'duration_days': position.get('final_metrics', {}).get('days_held', 0),
                'timestamp': datetime.now().isoformat()
            })
            
        return True
        
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the strategy's current state and performance.
        
        Returns:
            Dictionary with strategy summary information
        """
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'variant': self.params['strategy_variant'],
            'open_positions': len(self.positions),
            'closed_positions': len(self.closed_positions),
            'performance': self.performance_metrics,
            'health': self.health_metrics,
            'last_updated': datetime.now().isoformat()
        }
