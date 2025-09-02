#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Forex Arbitrage Trading Strategy

This strategy implements both triangular and statistical arbitrage for forex markets,
focusing on ultra-fast execution and precision. Key features:

1. Triangular Arbitrage: Exploits price differences between three related currency pairs
2. Statistical Arbitrage: Exploits temporary deviations between correlated pairs
3. Real-time monitoring of cross-rates to detect profitable opportunities
4. Ultra-fast execution model with robust risk controls
5. Regime-awareness with adaptive parameter optimization

Institutional-grade implementation with full event-driven architecture.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Set, Union
from datetime import datetime, timedelta
import time
import pytz
from enum import Enum
from collections import defaultdict, deque

# Import strategy base class and utilities
from trading_bot.strategies.base.forex_base import ForexBaseStrategy
from trading_bot.events.event_bus import EventBus
from trading_bot.enums.market_enums import MarketRegime, MarketSession, EntryQuality, TradeDirection

logger = logging.getLogger(__name__)

class ArbitrageType(Enum):
    """Types of arbitrage opportunities"""
    TRIANGULAR = 'triangular'
    STATISTICAL = 'statistical'
    LATENCY = 'latency'
    CROSS_VENUE = 'cross_venue'

class ArbitrageQuality(Enum):
    """Quality of arbitrage opportunities"""
    INVALID = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    PREMIUM = 4

class ArbitrageStrategy(ForexBaseStrategy):
    """
    Institutional-grade Forex Arbitrage Trading Strategy
    Implements triangular and statistical arbitrage with ultra-fast execution
    """
    
    def __init__(self, name: str = "Arbitrage Strategy", parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize the Arbitrage Trading Strategy
        
        Args:
            name: Strategy name
            parameters: Strategy parameters dictionary
        """
        # Default parameters
        default_params = {
            # Core parameters
            'lookback_window': 100,  # Lookback window for statistical calculations
            'min_data_points': 20,   # Minimum data points required
            
            # Triangular arbitrage parameters
            'triangular_enabled': True,  # Enable triangular arbitrage
            'min_tri_profit_pips': 0.5,  # Minimum profit in pips to trigger triangular arbitrage
            'min_tri_profit_percent': 0.1,  # Minimum profit percent for triangular arbitrage (0.1%)
            'triangular_sets': [
                ['EURUSD', 'GBPUSD', 'EURGBP'],
                ['USDJPY', 'EURJPY', 'EURUSD'],
                ['AUDUSD', 'NZDUSD', 'AUDNZD'],
                ['USDCAD', 'CADJPY', 'USDJPY'],
                ['GBPUSD', 'GBPJPY', 'USDJPY']
            ],
            
            # Statistical arbitrage parameters
            'statistical_enabled': True,  # Enable statistical arbitrage
            'z_score_threshold': 2.5,     # Z-score threshold to trigger signals
            'mean_reversion_period': 60,  # Period for mean reversion in minutes
            'correlation_threshold': 0.7,  # Minimum correlation to consider pairs
            'pair_combinations': [
                ['EURUSD', 'GBPUSD'],
                ['AUDUSD', 'NZDUSD'],
                ['EURJPY', 'GBPJPY'],
                ['USDCAD', 'USDCHF'],
                ['GBPUSD', 'EURGBP']
            ],
            
            # Execution parameters
            'execution_mode': 'ultra_fast',  # ultra_fast, fast, normal
            'max_slippage_pips': 0.5,       # Maximum allowed slippage in pips
            'max_spread_multiplier': 2.0,   # Maximum spread multiplier for normal conditions
            'position_timeout_seconds': 300, # Close positions if not filled within timeout
            'hedge_leg_delay_ms': 50,        # Delay between legs in milliseconds
            
            # Risk parameters
            'max_pos_size_percent': 2.0,    # Maximum position size as % of account
            'max_total_exposure_percent': 10.0, # Maximum total exposure
            'stop_loss_pips': 5.0,          # Emergency stop loss in pips
            'max_daily_loss_percent': 1.0,  # Maximum daily loss as % of account
            'max_positions': 5,             # Maximum simultaneous arbitrage positions
            
            # Filter parameters
            'min_volume_threshold': 50000,  # Minimum volume for liquid markets
            'news_filter_enabled': True,    # Enable news filter
            'news_filter_window_mins': 30,  # Filter window around high-impact news
            'exclude_illiquid_sessions': True,  # Exclude low-liquidity sessions
            
            # Performance parameters
            'profit_take_pips': {
                ArbitrageQuality.LOW.value: 0.5,
                ArbitrageQuality.MEDIUM.value: 1.0,
                ArbitrageQuality.HIGH.value: 1.5,
                ArbitrageQuality.PREMIUM.value: 2.0
            },
            
            # Timeframes for analysis
            'primary_timeframe': '1m',     # Primary timeframe for detection
            'secondary_timeframe': '5m',   # Secondary timeframe for confirmation
            
            # Other settings
            'adaptive_parameters': True,   # Dynamically adjust parameters based on market conditions
            'log_all_opportunities': False, # Log all detected opportunities, not just taken trades
        }
        
        # Update default parameters with provided parameters
        if parameters:
            default_params.update(parameters)
            
        # Initialize base class
        super().__init__(name, default_params)
        
        # Initialize arbitrage-specific attributes
        self.arbitrage_opportunities = {
            ArbitrageType.TRIANGULAR: [],
            ArbitrageType.STATISTICAL: [],
            ArbitrageType.LATENCY: [],
            ArbitrageType.CROSS_VENUE: []
        }
        
        # Active arbitrage positions
        self.active_arbitrage = {}
        
        # Historical data for z-score calculation (statistical arbitrage)
        self.pair_spread_history = {}
        
        # Performance tracking
        self.performance_metrics = {
            'opportunities_detected': 0,
            'opportunities_executed': 0,
            'successful_arbitrages': 0,
            'failed_arbitrages': 0,
            'total_pips_captured': 0,
            'avg_execution_time_ms': 0,
            'profitable_trades_percent': 0,
            'by_type': {
                ArbitrageType.TRIANGULAR.value: {'count': 0, 'pips': 0, 'profit': 0},
                ArbitrageType.STATISTICAL.value: {'count': 0, 'pips': 0, 'profit': 0},
                ArbitrageType.LATENCY.value: {'count': 0, 'pips': 0, 'profit': 0},
                ArbitrageType.CROSS_VENUE.value: {'count': 0, 'pips': 0, 'profit': 0}
            }
        }
        
        # Recent trade execution times (for performance monitoring)
        self.execution_times = deque(maxlen=100)
        
        # Market state tracking
        self.last_prices = {}
        self.spreads = {}
        self.spread_stats = {}
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Initialized {self.name} with triangular arbitrage "
                  f"{'enabled' if self.parameters['triangular_enabled'] else 'disabled'} and "
                  f"statistical arbitrage {'enabled' if self.parameters['statistical_enabled'] else 'disabled'}")

    def _register_event_handlers(self):
        """Register event handlers with the EventBus"""
        event_bus = EventBus.get_instance()
        
        # Market data events for ultra-fast response
        event_bus.subscribe('tick_data', self.on_tick_data)
        event_bus.subscribe('orderbook_update', self.on_orderbook_update)
        event_bus.subscribe('market_depth_update', self.on_market_depth_update)
        
        # Order and execution events
        event_bus.subscribe('order_filled', self.on_order_filled)
        event_bus.subscribe('order_rejected', self.on_order_rejected)
        event_bus.subscribe('position_closed', self.on_position_closed)
        
        # Session and regime events
        event_bus.subscribe('session_changed', self.on_session_changed)
        event_bus.subscribe('regime_changed', self.on_regime_changed)
        event_bus.subscribe('day_changed', self.on_day_changed)
        
        # News events
        event_bus.subscribe('economic_event', self.on_economic_event)
        
        logger.debug(f"{self.name}: Registered all event handlers")

    def on_tick_data(self, event_type: str, data: Dict[str, Any]):
        """
        Handle tick data updates - critical for arbitrage detection
        
        Args:
            event_type: Event type
            data: Tick data dictionary with symbol, bid, ask, etc.
        """
        if not data or 'symbol' not in data:
            return
            
        symbol = data['symbol']
        bid = data.get('bid')
        ask = data.get('ask')
        
        if bid is None or ask is None:
            return
            
        # Update last known prices
        self.last_prices[symbol] = {
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2,
            'spread': ask - bid,
            'timestamp': data.get('timestamp', pd.Timestamp.now())
        }
        
        # Calculate and store spread information
        self.spreads[symbol] = ask - bid
        
        # Update spread statistics if we have enough data
        if symbol in self.spread_stats:
            self.spread_stats[symbol]['current'] = ask - bid
            self.spread_stats[symbol]['values'].append(ask - bid)
            if len(self.spread_stats[symbol]['values']) > 100:
                self.spread_stats[symbol]['values'].pop(0)
            self.spread_stats[symbol]['avg'] = np.mean(self.spread_stats[symbol]['values'])
            self.spread_stats[symbol]['min'] = min(self.spread_stats[symbol]['values'])
            self.spread_stats[symbol]['max'] = max(self.spread_stats[symbol]['values'])
        else:
            self.spread_stats[symbol] = {
                'current': ask - bid,
                'values': [ask - bid],
                'avg': ask - bid,
                'min': ask - bid,
                'max': ask - bid
            }
            
        # Check for triangular arbitrage opportunities if enabled
        if self.parameters['triangular_enabled']:
            self._check_triangular_arbitrage()
            
        # Process statistical arbitrage if enabled and we have new data
        if self.parameters['statistical_enabled'] and len(self.last_prices) > 1:
            self._update_pair_spreads()
            self._check_statistical_arbitrage()
            
    def on_orderbook_update(self, event_type: str, data: Dict[str, Any]):
        """
        Handle orderbook updates for more precise arbitrage calculations
        
        Args:
            event_type: Event type
            data: Orderbook data with bids, asks, etc.
        """
        # Process orderbook data for enhanced execution decisions
        if not data or 'symbol' not in data:
            return
            
        # Analyze order book depth for liquidity assessment
        symbol = data['symbol']
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if not bids or not asks:
            return
            
        # Assess market depth and liquidity
        bid_volume = sum(item[1] for item in bids[:5]) if len(bids) >= 5 else 0
        ask_volume = sum(item[1] for item in asks[:5]) if len(asks) >= 5 else 0
        
        # Store orderbook information for execution decisions
        # This can help determine if there's enough liquidity for arbitrage execution
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
            
        self.market_data[symbol]['orderbook'] = {
            'top_bid': bids[0][0] if bids else None,
            'top_ask': asks[0][0] if asks else None,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'timestamp': data.get('timestamp', pd.Timestamp.now())
        }

    def on_market_depth_update(self, event_type: str, data: Dict[str, Any]):
        """
        Handle market depth updates for liquidity assessment
        
        Args:
            event_type: Event type
            data: Market depth data
        """
        # Process market depth for liquidity assessment
        pass

    def on_order_filled(self, event_type: str, data: Dict[str, Any]):
        """
        Handle order filled events for arbitrage leg management
        
        Args:
            event_type: Event type
            data: Order data
        """
        if not data or 'order_id' not in data:
            return
            
        order_id = data['order_id']
        
        # Check if this order is part of an active arbitrage
        for arb_id, arb_data in self.active_arbitrage.items():
            if order_id in arb_data['order_ids']:
                # Update the arbitrage status
                filled_price = data.get('filled_price')
                symbol = data.get('symbol')
                leg_index = arb_data['order_ids'].index(order_id)
                
                logger.info(f"Arbitrage {arb_id}: Leg {leg_index+1} filled for {symbol} at price {filled_price}")
                
                # Mark this leg as filled
                arb_data['legs_status'][leg_index] = 'filled'
                arb_data['legs_fill_prices'][leg_index] = filled_price
                
                # Check if all legs are filled
                if all(status == 'filled' for status in arb_data['legs_status']):
                    # All legs filled, calculate actual profit
                    self._calculate_arbitrage_result(arb_id)

    def on_order_rejected(self, event_type: str, data: Dict[str, Any]):
        """
        Handle order rejected events for arbitrage error handling
        
        Args:
            event_type: Event type
            data: Order data
        """
        if not data or 'order_id' not in data:
            return
            
        order_id = data['order_id']
        
        # Check if this order is part of an active arbitrage
        for arb_id, arb_data in list(self.active_arbitrage.items()):
            if order_id in arb_data['order_ids']:
                # Mark arbitrage as failed
                arb_data['status'] = 'failed'
                arb_data['end_time'] = pd.Timestamp.now()
                arb_data['result'] = 'rejected'
                
                # Log the failure
                symbol = data.get('symbol')
                reason = data.get('reject_reason', 'Unknown reason')
                leg_index = arb_data['order_ids'].index(order_id)
                
                logger.warning(f"Arbitrage {arb_id}: Leg {leg_index+1} rejected for {symbol}. Reason: {reason}")
                
                # Close other legs if possible
                self._close_arbitrage_position(arb_id, 'leg_rejected')
                
                # Update performance metrics
                self.performance_metrics['failed_arbitrages'] += 1

    def on_position_closed(self, event_type: str, data: Dict[str, Any]):
        """
        Handle position closed events for arbitrage completion
        
        Args:
            event_type: Event type
            data: Position data
        """
        # Process position closed events to track complete arbitrage cycles
        pass

    def on_session_changed(self, event_type: str, data: Dict[str, Any]):
        """
        Handle session change events to adjust parameters
        
        Args:
            event_type: Event type
            data: Session data
        """
        if not data or 'session' not in data:
            return
            
        session = data['session']
        
        # Adjust parameters based on session
        if session == MarketSession.ASIAN:
            # Asian session may have lower liquidity, adjust accordingly
            if self.parameters['exclude_illiquid_sessions']:
                logger.info(f"{self.name}: Adjusting parameters for Asian session")
                self.parameters['min_tri_profit_pips'] *= 1.2  # Require higher profit
                self.parameters['max_pos_size_percent'] *= 0.8  # Reduce position size
                
        elif session == MarketSession.EUROPEAN:
            # European session has high liquidity, can be more aggressive
            logger.info(f"{self.name}: Adjusting parameters for European session")
            self.parameters['min_tri_profit_pips'] /= 1.2 if self.parameters['min_tri_profit_pips'] > 0.5 else 1.0
            self.parameters['max_pos_size_percent'] = min(2.0, self.parameters['max_pos_size_percent'] * 1.2)
            
        elif session == MarketSession.US:
            # US session also has good liquidity
            logger.info(f"{self.name}: Adjusting parameters for US session")
            # Similar adjustments to European session
            
        # Publish parameter change
        EventBus.get_instance().publish('strategy_parameters_changed', {
            'strategy': self.name,
            'session': session.name,
            'parameters': self.parameters
        })

    def on_regime_changed(self, event_type: str, data: Dict[str, Any]):
        """
        Handle market regime change events
        
        Args:
            event_type: Event type
            data: Regime data
        """
        if not data or 'regime' not in data:
            return
            
        regime = data['regime']
        
        # Optimize strategy for the new regime
        self.optimize_for_regime(regime)
        
        logger.info(f"{self.name}: Optimized for {regime.name} regime")

    def on_day_changed(self, event_type: str, data: Dict[str, Any]):
        """
        Handle day change events to reset daily metrics
        
        Args:
            event_type: Event type
            data: Day change data
        """
        # Reset daily tracking
        self._reset_daily_tracking()
        
        # Calculate and store daily performance
        EventBus.get_instance().publish('strategy_daily_results', {
            'strategy': self.name,
            'date': data.get('previous_date'),
            'performance': self.get_performance_metrics()
        })
        
        logger.info(f"{self.name}: Reset daily tracking for new trading day")

    def on_economic_event(self, event_type: str, data: Dict[str, Any]):
        """
        Handle economic event notifications for risk management
        
        Args:
            event_type: Event type
            data: Economic event data
        """
        if not data or 'impact' not in data or 'currencies' not in data:
            return
            
        # Skip if news filter is disabled
        if not self.parameters['news_filter_enabled']:
            return
            
        impact = data['impact']
        currencies = data['currencies']
        event_time = data.get('time', pd.Timestamp.now())
        
        # Only care about high impact events
        if impact.lower() != 'high':
            return
            
        # Calculate filter window
        filter_minutes = self.parameters['news_filter_window_mins']
        window_start = event_time - timedelta(minutes=filter_minutes // 2)
        window_end = event_time + timedelta(minutes=filter_minutes // 2)
        
        # Add affected currency pairs to news filter
        for currency in currencies:
            for symbol in self.parameters['triangular_sets'] + self.parameters['pair_combinations']:
                if isinstance(symbol, list):
                    for s in symbol:
                        if currency in s:
                            self._add_to_news_filter(s, window_start, window_end)
                else:
                    if currency in symbol:
                        self._add_to_news_filter(symbol, window_start, window_end)
        
        logger.info(f"{self.name}: Added currencies {currencies} to news filter due to high impact event")
        
    def _add_to_news_filter(self, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
        """
        Add a currency pair to the news filter for a given time window
        
        Args:
            symbol: Currency pair symbol
            start_time: Start of filter window
            end_time: End of filter window
        """
        self.news_filter[symbol] = {
            'start': start_time,
            'end': end_time
        }
        logger.debug(f"Added {symbol} to news filter from {start_time} to {end_time}")
        
    def _check_triangular_arbitrage(self):
        """
        Check for triangular arbitrage opportunities across all configured sets
        """
        # Skip if triangular arbitrage is disabled
        if not self.parameters['triangular_enabled']:
            return
            
        # Need at least 3 prices for triangular arbitrage
        if len(self.last_prices) < 3:
            return
            
        opportunities = []
        min_profit_threshold = self.parameters['min_tri_profit_pips']
        min_profit_percent = self.parameters['min_tri_profit_percent'] / 100  # Convert to decimal
        
        # Check each triangular set
        for tri_set in self.parameters['triangular_sets']:
            # Need all 3 pairs in the set
            if not all(pair in self.last_prices for pair in tri_set):
                continue
                
            # Get the exchange rates
            pair_a, pair_b, pair_c = tri_set
            
            # Check for news filter
            if any(pair in self.news_filter for pair in tri_set):
                current_time = pd.Timestamp.now()
                filtered = False
                
                for pair in tri_set:
                    if pair in self.news_filter:
                        filter_data = self.news_filter[pair]
                        if filter_data['start'] <= current_time <= filter_data['end']:
                            filtered = True
                            break
                            
                if filtered:
                    logger.debug(f"Skipping triangular set {tri_set} due to news filter")
                    continue
            
            # Calculate arbitrage opportunity
            arb_details = self._calculate_triangular_arbitrage(pair_a, pair_b, pair_c)
            
            if arb_details['profit_pips'] > min_profit_threshold and arb_details['profit_percent'] > min_profit_percent:
                # Valid opportunity detected
                quality = self._assess_arbitrage_quality(arb_details['profit_pips'], arb_details['profit_percent'])
                
                if quality != ArbitrageQuality.INVALID:
                    # Add to opportunities
                    opportunity = {
                        'id': f"tri_{pair_a}_{pair_b}_{pair_c}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
                        'type': ArbitrageType.TRIANGULAR,
                        'pairs': [pair_a, pair_b, pair_c],
                        'profit_pips': arb_details['profit_pips'],
                        'profit_percent': arb_details['profit_percent'],
                        'direction': arb_details['direction'],
                        'prices': arb_details['prices'],
                        'quality': quality,
                        'detected_time': pd.Timestamp.now(),
                        'execution_path': arb_details['execution_path']
                    }
                    
                    opportunities.append(opportunity)
                    logger.info(f"Detected triangular arbitrage: {pair_a}/{pair_b}/{pair_c} | "  
                               f"Profit: {arb_details['profit_pips']:.2f} pips ({arb_details['profit_percent']*100:.3f}%) | "  
                               f"Quality: {quality.name}")
                    
                    # Increment detection counter
                    self.performance_metrics['opportunities_detected'] += 1
                    
                    # Publish arbitrage opportunity event
                    EventBus.get_instance().publish('arbitrage_opportunity', {
                        'strategy': self.name,
                        'id': opportunity['id'],
                        'type': 'triangular',
                        'pairs': tri_set,
                        'profit_pips': arb_details['profit_pips'],
                        'profit_percent': arb_details['profit_percent'] * 100,
                        'quality': quality.name,
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
        
        # Store opportunities and prepare for execution
        if opportunities:
            self.arbitrage_opportunities[ArbitrageType.TRIANGULAR] = opportunities
            
            # Execute the best opportunity if possible
            self._execute_best_arbitrage(ArbitrageType.TRIANGULAR)
            
    def _calculate_triangular_arbitrage(self, pair_a: str, pair_b: str, pair_c: str) -> Dict[str, Any]:
        """
        Calculate potential triangular arbitrage for three currency pairs
        
        Args:
            pair_a: First currency pair (e.g., 'EURUSD')
            pair_b: Second currency pair (e.g., 'GBPUSD')
            pair_c: Third currency pair (e.g., 'EURGBP')
            
        Returns:
            Dictionary with arbitrage details
        """
        # Extract currencies from pairs
        # Each pair in format BASE/QUOTE (e.g., EUR/USD means 1 EUR = X USD)
        currencies = set()
        
        # Helper function to extract base and quote currencies
        def extract_currencies(pair):
            # Standard format is XXXYYY where XXX is base and YYY is quote
            if len(pair) == 6:
                base = pair[:3]
                quote = pair[3:]
                return base, quote
            elif '/' in pair:
                base, quote = pair.split('/')
                return base, quote
            else:
                # Try common separators
                for sep in ['_', '-', ' ']:
                    if sep in pair:
                        base, quote = pair.split(sep)
                        return base, quote
            
            # Default fallback (may not be accurate)
            return pair[:3], pair[3:6]
        
        # Extract all currencies
        base_a, quote_a = extract_currencies(pair_a)
        base_b, quote_b = extract_currencies(pair_b)
        base_c, quote_c = extract_currencies(pair_c)
        
        currencies.update([base_a, quote_a, base_b, quote_b, base_c, quote_c])
        
        # Ensure we have 3 distinct currencies (required for triangular arbitrage)
        if len(currencies) != 3:
            return {
                'profit_pips': 0,
                'profit_percent': 0,
                'direction': None,
                'prices': {},
                'execution_path': []
            }
        
        # Get mid prices for calculation
        price_a = self.last_prices[pair_a]['mid']
        price_b = self.last_prices[pair_b]['mid']
        price_c = self.last_prices[pair_c]['mid']
        
        # We need to determine the correct calculation path
        # This depends on how the currency pairs are quoted
        
        # Try both directions and see which one offers arbitrage
        # Path 1: Starting with currency 1, convert to 2, then to 3, then back to 1
        # For execution we'll use actual bid/ask prices, but for detection mid price is fine
        
        # Identify the correct path based on currency connections
        execution_paths = []
        
        # First, map the currencies
        currency_map = {
            base_a: {'base_of': [pair_a], 'quote_of': []},
            quote_a: {'base_of': [], 'quote_of': [pair_a]},
            base_b: {'base_of': [pair_b], 'quote_of': []},
            quote_b: {'base_of': [], 'quote_of': [pair_b]},
            base_c: {'base_of': [pair_c], 'quote_of': []},
            quote_c: {'base_of': [], 'quote_of': [pair_c]}
        }
        
        # Consolidate the map (some currencies appear multiple times)
        consolidated_map = defaultdict(lambda: {'base_of': [], 'quote_of': []})
        for curr, data in currency_map.items():
            consolidated_map[curr]['base_of'].extend(data['base_of'])
            consolidated_map[curr]['quote_of'].extend(data['quote_of'])
        
        # Find all possible starting currencies (those that appear in both base and quote)
        potential_starts = []
        for curr, data in consolidated_map.items():
            if data['base_of'] and data['quote_of']:
                potential_starts.append(curr)
                
        # Construct a circular path for each potential start
        for start_curr in potential_starts:
            path = []
            current = start_curr
            visited_currencies = {current}
            visited_pairs = set()
            
            # Try to find a path that uses all 3 pairs
            while len(path) < 3:
                # Find the next pair to use
                next_pair = None
                next_curr = None
                
                # Check pairs where current currency is the base
                for pair in consolidated_map[current]['base_of']:
                    if pair not in visited_pairs:
                        # Get the quote currency
                        quote = quote_a if pair == pair_a else (quote_b if pair == pair_b else quote_c)
                        if quote not in visited_currencies or (quote == start_curr and len(path) == 2):
                            next_pair = pair
                            next_curr = quote
                            path.append((pair, 'sell', current, next_curr))
                            visited_pairs.add(pair)
                            visited_currencies.add(next_curr)
                            current = next_curr
                            break
                            
                # If no pair found where current is base, check where it's quote
                if next_pair is None:
                    for pair in consolidated_map[current]['quote_of']:
                        if pair not in visited_pairs:
                            # Get the base currency
                            base = base_a if pair == pair_a else (base_b if pair == pair_b else base_c)
                            if base not in visited_currencies or (base == start_curr and len(path) == 2):
                                next_pair = pair
                                next_curr = base
                                path.append((pair, 'buy', current, next_curr))
                                visited_pairs.add(pair)
                                visited_currencies.add(next_curr)
                                current = next_curr
                                break
                                
                # If we couldn't find a next step, this path is invalid
                if next_pair is None:
                    break
                    
            # Check if we have a valid path (3 pairs, back to start)
            if len(path) == 3 and current == start_curr:
                execution_paths.append(path)
        
        # Calculate profit for each path
        best_path = None
        best_profit_percent = -100  # Start with impossible negative value
        best_profit_pips = 0
        best_direction = None
        best_prices = {}
        
        for path in execution_paths:
            # Start with 1 unit of currency
            amount = 1.0
            
            # Apply exchange rates along the path
            for step in path:
                pair, action, from_curr, to_curr = step
                price_info = self.last_prices[pair]
                
                if action == 'buy':
                    # We're buying the base currency, so we use the ask price
                    # and divide by the exchange rate
                    rate = price_info['ask']
                    amount = amount / rate
                else:  # 'sell'
                    # We're selling the base currency, so we use the bid price
                    # and multiply by the exchange rate
                    rate = price_info['bid']
                    amount = amount * rate
                    
                # Store the price used
                best_prices[pair] = {
                    'action': action,
                    'rate': rate,
                    'from': from_curr,
                    'to': to_curr
                }
            
            # Calculate profit
            profit_percent = (amount - 1.0) * 100
            
            # Calculate profit in pips
            # Convert percent to approximate pips (rough estimate)
            profit_pips = profit_percent * 10
            
            if profit_percent > best_profit_percent:
                best_profit_percent = profit_percent
                best_profit_pips = profit_pips
                best_direction = 'forward' if profit_percent > 0 else 'reverse'
                best_path = path
        
        # Return the best arbitrage opportunity
        return {
            'profit_pips': best_profit_pips,
            'profit_percent': best_profit_percent / 100,  # Convert to decimal
            'direction': best_direction,
            'prices': best_prices,
            'execution_path': best_path
        }
        
    def _assess_arbitrage_quality(self, profit_pips: float, profit_percent: float) -> ArbitrageQuality:
        """
        Assess the quality of an arbitrage opportunity
        
        Args:
            profit_pips: Expected profit in pips
            profit_percent: Expected profit as a percentage (decimal)
            
        Returns:
            ArbitrageQuality enum value
        """
        # Convert profit_percent to basis points (1 bp = 0.01%)
        profit_bp = profit_percent * 10000
        
        # Quality thresholds (adjustable)
        if profit_pips <= 0 or profit_bp <= 0:
            return ArbitrageQuality.INVALID
        elif profit_pips < 0.5 or profit_bp < 5:  # Less than 0.5 pips or 0.05%
            return ArbitrageQuality.LOW
        elif profit_pips < 1.0 or profit_bp < 10:  # Less than 1.0 pips or 0.1%
            return ArbitrageQuality.MEDIUM
        elif profit_pips < 2.0 or profit_bp < 20:  # Less than 2.0 pips or 0.2%
            return ArbitrageQuality.HIGH
        else:  # 2.0+ pips or 0.2%+
            return ArbitrageQuality.PREMIUM
            
    def _update_pair_spreads(self):
        """
        Update the historical spread data for statistical arbitrage
        """
        # Skip if statistical arbitrage is disabled
        if not self.parameters['statistical_enabled']:
            return
            
        # Process each pair combination
        for pair_combo in self.parameters['pair_combinations']:
            if len(pair_combo) != 2:
                continue
                
            pair_a, pair_b = pair_combo
            
            # Skip if we don't have prices for both pairs
            if pair_a not in self.last_prices or pair_b not in self.last_prices:
                continue
                
            # Create a unique key for this pair combination
            combo_key = f"{pair_a}_{pair_b}"
            
            # Get the mid prices
            price_a = self.last_prices[pair_a]['mid']
            price_b = self.last_prices[pair_b]['mid']
            
            # Calculate the spread (could be ratio, difference, or other relationship)
            # Method depends on the currency pairs and their relationship
            # Try to normalize the spread based on the currency pairs
            spread = self._calculate_pair_spread(pair_a, pair_b, price_a, price_b)
            
            # Initialize history if needed
            if combo_key not in self.pair_spread_history:
                self.pair_spread_history[combo_key] = {
                    'spreads': [],
                    'timestamps': [],
                    'mean': None,
                    'std': None,
                    'correlation': None,
                    'z_scores': [],
                    'pair_a': pair_a,
                    'pair_b': pair_b
                }
                
            # Add current spread to history
            self.pair_spread_history[combo_key]['spreads'].append(spread)
            self.pair_spread_history[combo_key]['timestamps'].append(pd.Timestamp.now())
            
            # Limit history size
            max_history = self.parameters['lookback_window']
            if len(self.pair_spread_history[combo_key]['spreads']) > max_history:
                self.pair_spread_history[combo_key]['spreads'].pop(0)
                self.pair_spread_history[combo_key]['timestamps'].pop(0)
                
            # Update statistics if we have enough data
            if len(self.pair_spread_history[combo_key]['spreads']) >= self.parameters['min_data_points']:
                spreads = np.array(self.pair_spread_history[combo_key]['spreads'])
                
                # Calculate mean and standard deviation
                mean = np.mean(spreads)
                std = np.std(spreads)
                
                # Update stored stats
                self.pair_spread_history[combo_key]['mean'] = mean
                self.pair_spread_history[combo_key]['std'] = std
                
                # Calculate current z-score
                if std > 0:
                    z_score = (spread - mean) / std
                else:
                    z_score = 0
                    
                self.pair_spread_history[combo_key]['z_scores'].append(z_score)
                
                # Limit z-score history size
                if len(self.pair_spread_history[combo_key]['z_scores']) > max_history:
                    self.pair_spread_history[combo_key]['z_scores'].pop(0)
                    
                # Calculate correlation between the pairs if we have raw price history
                if pair_a in self.market_data and pair_b in self.market_data:
                    if 'prices' in self.market_data[pair_a] and 'prices' in self.market_data[pair_b]:
                        prices_a = np.array(self.market_data[pair_a]['prices'][-max_history:])
                        prices_b = np.array(self.market_data[pair_b]['prices'][-max_history:])
                        
                        if len(prices_a) == len(prices_b) and len(prices_a) >= self.parameters['min_data_points']:
                            correlation = np.corrcoef(prices_a, prices_b)[0, 1]
                            self.pair_spread_history[combo_key]['correlation'] = correlation
    
    def _calculate_pair_spread(self, pair_a: str, pair_b: str, price_a: float, price_b: float) -> float:
        """
        Calculate the normalized spread between two currency pairs
        
        Args:
            pair_a: First currency pair
            pair_b: Second currency pair
            price_a: Price of first pair
            price_b: Price of second pair
            
        Returns:
            Normalized spread value
        """
        # Extract base currencies to determine relationship
        base_a = pair_a[:3]
        base_b = pair_b[:3]
        
        # If pairs share a base currency, use price ratio
        if base_a == base_b:
            return price_a / price_b
        else:
            # Check if they share the quote currency
            quote_a = pair_a[3:6]
            quote_b = pair_b[3:6]
            
            if quote_a == quote_b:
                return price_a / price_b
            else:
                # Different base and quote, use a more generic approach
                # Normalize by average price
                avg_price = (price_a + price_b) / 2
                return (price_a - price_b) / avg_price if avg_price != 0 else 0
    
    def _check_statistical_arbitrage(self):
        """
        Check for statistical arbitrage opportunities based on historical spreads
        """
        # Skip if statistical arbitrage is disabled
        if not self.parameters['statistical_enabled']:
            return
            
        # Need at least some historical data
        if not self.pair_spread_history:
            return
            
        opportunities = []
        z_score_threshold = self.parameters['z_score_threshold']
        correlation_threshold = self.parameters['correlation_threshold']
        
        # Check each pair combination
        for combo_key, history in self.pair_spread_history.items():
            # Skip if we don't have enough data or statistics
            if 'z_scores' not in history or not history['z_scores'] or history['mean'] is None:
                continue
                
            # Get latest z-score
            z_score = history['z_scores'][-1]
            
            # Skip if correlation is below threshold (pairs should be correlated)
            if history['correlation'] is not None and abs(history['correlation']) < correlation_threshold:
                continue
                
            # Check if z-score exceeds threshold (indicating potential arbitrage)
            if abs(z_score) > z_score_threshold:
                # Determine trade direction based on z-score
                # Positive z-score: spread is wider than normal -> short spread (buy pair_b, sell pair_a)
                # Negative z-score: spread is tighter than normal -> long spread (buy pair_a, sell pair_b)
                if z_score > 0:
                    direction = 'short_spread'
                    trade_directions = {history['pair_a']: TradeDirection.SHORT, history['pair_b']: TradeDirection.LONG}
                else:
                    direction = 'long_spread'
                    trade_directions = {history['pair_a']: TradeDirection.LONG, history['pair_b']: TradeDirection.SHORT}
                    
                # Estimate profit potential based on mean reversion expectation
                # How far the spread is expected to revert
                reversion_expectation = abs(z_score) - 0.5  # Expect reversion to 0.5 standard deviations
                profit_potential = reversion_expectation * history['std']
                
                # Convert to approximate pips (can be refined based on specific pairs)
                profit_pips = profit_potential * 10000  # Rough approximation
                
                # Convert to percent for quality assessment
                profit_percent = profit_potential
                
                # Assess quality
                quality = self._assess_arbitrage_quality(profit_pips, profit_percent)
                
                if quality != ArbitrageQuality.INVALID:
                    # Create opportunity object
                    opportunity = {
                        'id': f"stat_{history['pair_a']}_{history['pair_b']}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}",
                        'type': ArbitrageType.STATISTICAL,
                        'pairs': [history['pair_a'], history['pair_b']],
                        'z_score': z_score,
                        'mean': history['mean'],
                        'std': history['std'],
                        'correlation': history['correlation'],
                        'direction': direction,
                        'trade_directions': trade_directions,
                        'profit_pips': profit_pips,
                        'profit_percent': profit_percent,
                        'quality': quality,
                        'detected_time': pd.Timestamp.now(),
                        'expected_reversion_time': pd.Timestamp.now() + timedelta(minutes=self.parameters['mean_reversion_period'])
                    }
                    
                    opportunities.append(opportunity)
                    logger.info(f"Detected statistical arbitrage: {history['pair_a']}/{history['pair_b']} | "  
                               f"Z-Score: {z_score:.2f} | Direction: {direction} | "  
                               f"Quality: {quality.name}")
                    
                    # Increment detection counter
                    self.performance_metrics['opportunities_detected'] += 1
                    
                    # Publish arbitrage opportunity event
                    EventBus.get_instance().publish('arbitrage_opportunity', {
                        'strategy': self.name,
                        'id': opportunity['id'],
                        'type': 'statistical',
                        'pairs': [history['pair_a'], history['pair_b']],
                        'z_score': z_score,
                        'direction': direction,
                        'quality': quality.name,
                        'timestamp': pd.Timestamp.now().isoformat()
                    })
        
        # Store opportunities and prepare for execution
        if opportunities:
            self.arbitrage_opportunities[ArbitrageType.STATISTICAL] = opportunities
            
            # Execute the best opportunity if possible
            self._execute_best_arbitrage(ArbitrageType.STATISTICAL)
            
    def _execute_best_arbitrage(self, arbitrage_type: ArbitrageType):
        """
        Execute the best arbitrage opportunity of the given type
        
        Args:
            arbitrage_type: Type of arbitrage to execute
        """
        # Check if we have opportunities of this type
        if not self.arbitrage_opportunities[arbitrage_type]:
            return
            
        # Check if we can take more positions
        if len(self.active_arbitrage) >= self.parameters['max_positions']:
            logger.debug(f"Maximum positions reached ({self.parameters['max_positions']}), skipping execution")
            return
            
        # Sort opportunities by quality and profit
        opportunities = sorted(
            self.arbitrage_opportunities[arbitrage_type],
            key=lambda x: (x['quality'].value if isinstance(x['quality'], ArbitrageQuality) else x['quality'], x['profit_pips']),
            reverse=True
        )
        
        # Select the best opportunity
        best_opp = opportunities[0]
        
        # Execute based on type
        if arbitrage_type == ArbitrageType.TRIANGULAR:
            self._execute_triangular_arbitrage(best_opp)
        elif arbitrage_type == ArbitrageType.STATISTICAL:
            self._execute_statistical_arbitrage(best_opp)
        # Add more arbitrage types here as needed
            
    def _execute_triangular_arbitrage(self, opportunity: Dict[str, Any]):
        """
        Execute a triangular arbitrage opportunity
        
        Args:
            opportunity: Arbitrage opportunity details
        """
        # Log the execution attempt
        logger.info(f"Executing triangular arbitrage: {opportunity['id']} with profit potential {opportunity['profit_pips']:.2f} pips")
        
        # Get execution path
        execution_path = opportunity['execution_path']
        if not execution_path:
            logger.warning(f"No execution path for arbitrage {opportunity['id']}")
            return
            
        # Determine position size based on arbitrage quality
        quality = opportunity['quality']
        max_pos_size_percent = self.parameters['max_pos_size_percent']
        
        # Adjust position size based on quality
        position_size_multiplier = 0.5  # Start with 50% of max
        if quality == ArbitrageQuality.HIGH:
            position_size_multiplier = 0.75
        elif quality == ArbitrageQuality.PREMIUM:
            position_size_multiplier = 1.0
            
        # Calculate actual position size (would be based on account equity in production)
        account_size = 100000  # Placeholder, would get from account in production
        position_size = (account_size * max_pos_size_percent / 100) * position_size_multiplier
        
        # Create execution plan for all legs
        order_ids = []
        execution_start = pd.Timestamp.now()
        
        # Process each leg of the arbitrage
        for i, step in enumerate(execution_path):
            pair, action, from_curr, to_curr = step
            
            # Determine order type and parameters
            if action == 'buy':
                order_direction = 'buy'
                price = self.last_prices[pair]['ask']  # Buy at ask
            else:  # 'sell'
                order_direction = 'sell'
                price = self.last_prices[pair]['bid']  # Sell at bid
                
            # Adjust size for currency conversions in the path
            # In real implementation, would adjust based on actual conversion rates
            adjusted_size = position_size
            
            # Generate order ID
            order_id = f"arb_{opportunity['id']}_{i}"
            
            # In production, would actually place the order here
            # For this implementation, we'll simulate the order placement
            logger.info(f"Placing {order_direction} order for {pair} at {price:.5f} with size {adjusted_size:.2f}")
            
            # Simulate order placement success
            order_ids.append(order_id)
            
            # Add small delay between legs if needed
            if i < len(execution_path) - 1 and self.parameters['hedge_leg_delay_ms'] > 0:
                time.sleep(self.parameters['hedge_leg_delay_ms'] / 1000.0)
                
        # Record active arbitrage
        self.active_arbitrage[opportunity['id']] = {
            'type': ArbitrageType.TRIANGULAR,
            'opportunity': opportunity,
            'order_ids': order_ids,
            'execution_start': execution_start,
            'legs_status': ['pending'] * len(execution_path),
            'legs_fill_prices': [None] * len(execution_path),
            'status': 'executing',
            'position_size': position_size
        }
        
        # Update performance metrics
        self.performance_metrics['opportunities_executed'] += 1
        
        # Publish execution event
        EventBus.get_instance().publish('arbitrage_execution', {
            'strategy': self.name,
            'id': opportunity['id'],
            'type': 'triangular',
            'pairs': opportunity['pairs'],
            'expected_profit_pips': opportunity['profit_pips'],
            'execution_time': execution_start.isoformat()
        })
        
    def _execute_statistical_arbitrage(self, opportunity: Dict[str, Any]):
        """
        Execute a statistical arbitrage opportunity
        
        Args:
            opportunity: Arbitrage opportunity details
        """
        # Log the execution attempt
        logger.info(f"Executing statistical arbitrage: {opportunity['id']} with expected profit {opportunity['profit_pips']:.2f} pips")
        
        # Get pairs and directions
        pairs = opportunity['pairs']
        trade_directions = opportunity['trade_directions']
        
        # Determine position size based on arbitrage quality
        quality = opportunity['quality']
        max_pos_size_percent = self.parameters['max_pos_size_percent']
        
        # Adjust position size based on quality
        position_size_multiplier = 0.5  # Start with 50% of max
        if quality == ArbitrageQuality.HIGH:
            position_size_multiplier = 0.75
        elif quality == ArbitrageQuality.PREMIUM:
            position_size_multiplier = 1.0
            
        # Calculate actual position size (would be based on account equity in production)
        account_size = 100000  # Placeholder, would get from account in production
        position_size = (account_size * max_pos_size_percent / 100) * position_size_multiplier
        
        # Create execution plan for both legs
        order_ids = []
        execution_start = pd.Timestamp.now()
        
        # Process each leg of the arbitrage
        for i, pair in enumerate(pairs):
            direction = trade_directions[pair]
            
            # Determine order type and parameters
            if direction == TradeDirection.LONG:
                order_direction = 'buy'
                price = self.last_prices[pair]['ask']  # Buy at ask
            else:  # SHORT
                order_direction = 'sell'
                price = self.last_prices[pair]['bid']  # Sell at bid
                
            # Generate order ID
            order_id = f"arb_{opportunity['id']}_{i}"
            
            # In production, would actually place the order here
            # For this implementation, we'll simulate the order placement
            logger.info(f"Placing {order_direction} order for {pair} at {price:.5f} with size {position_size:.2f}")
            
            # Simulate order placement success
            order_ids.append(order_id)
            
            # Add small delay between legs if needed
            if i < len(pairs) - 1 and self.parameters['hedge_leg_delay_ms'] > 0:
                time.sleep(self.parameters['hedge_leg_delay_ms'] / 1000.0)
                
        # Record active arbitrage
        self.active_arbitrage[opportunity['id']] = {
            'type': ArbitrageType.STATISTICAL,
            'opportunity': opportunity,
            'order_ids': order_ids,
            'execution_start': execution_start,
            'legs_status': ['pending'] * len(pairs),
            'legs_fill_prices': [None] * len(pairs),
            'status': 'executing',
            'position_size': position_size,
            'expected_exit_time': opportunity['expected_reversion_time']
        }
        
        # Update performance metrics
        self.performance_metrics['opportunities_executed'] += 1
        
        # Publish execution event
        EventBus.get_instance().publish('arbitrage_execution', {
            'strategy': self.name,
            'id': opportunity['id'],
            'type': 'statistical',
            'pairs': pairs,
            'expected_profit_pips': opportunity['profit_pips'],
            'execution_time': execution_start.isoformat(),
            'expected_exit_time': opportunity['expected_reversion_time'].isoformat()
        })
        
    def _calculate_arbitrage_result(self, arbitrage_id: str):
        """
        Calculate the actual result of a completed arbitrage
        
        Args:
            arbitrage_id: ID of the completed arbitrage
        """
        if arbitrage_id not in self.active_arbitrage:
            return
            
        arb_data = self.active_arbitrage[arbitrage_id]
        
        # Check if all legs are filled
        if not all(status == 'filled' for status in arb_data['legs_status']):
            return
            
        # Mark as completed
        arb_data['status'] = 'completed'
        arb_data['end_time'] = pd.Timestamp.now()
        
        # Calculate execution time
        execution_time_ms = (arb_data['end_time'] - arb_data['execution_start']).total_seconds() * 1000
        
        # Track execution time for performance monitoring
        self.execution_times.append(execution_time_ms)
        
        # Update average execution time
        self.performance_metrics['avg_execution_time_ms'] = np.mean(self.execution_times)
        
        # Calculate actual profit (would be based on actual fill prices in production)
        opportunity = arb_data['opportunity']
        actual_profit_pips = opportunity['profit_pips']  # Simplified, would use real calculations
        
        # Update performance metrics
        arb_type = arb_data['type'].value
        self.performance_metrics['successful_arbitrages'] += 1
        self.performance_metrics['total_pips_captured'] += actual_profit_pips
        self.performance_metrics['by_type'][arb_type]['count'] += 1
        self.performance_metrics['by_type'][arb_type]['pips'] += actual_profit_pips
        
        # Calculate profitable trades percentage
        total_trades = self.performance_metrics['successful_arbitrages'] + self.performance_metrics['failed_arbitrages']
        if total_trades > 0:
            self.performance_metrics['profitable_trades_percent'] = (
                self.performance_metrics['successful_arbitrages'] / total_trades
            ) * 100
            
        # Log the completion
        logger.info(f"Completed arbitrage {arbitrage_id} with profit of {actual_profit_pips:.2f} pips "  
                   f"in {execution_time_ms:.2f}ms")
        
        # Publish completion event
        EventBus.get_instance().publish('arbitrage_completed', {
            'strategy': self.name,
            'id': arbitrage_id,
            'type': arb_type,
            'profit_pips': actual_profit_pips,
            'execution_time_ms': execution_time_ms,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
        # Store the result for analysis
        if 'history' not in self.__dict__:
            self.history = []
            
        self.history.append({
            'id': arbitrage_id,
            'type': arb_type,
            'profit_pips': actual_profit_pips,
            'execution_time_ms': execution_time_ms,
            'timestamp': pd.Timestamp.now()
        })
        
    def _close_arbitrage_position(self, arbitrage_id: str, reason: str):
        """
        Close an arbitrage position (due to timeout, error, etc.)
        
        Args:
            arbitrage_id: ID of the arbitrage to close
            reason: Reason for closing
        """
        if arbitrage_id not in self.active_arbitrage:
            return
            
        arb_data = self.active_arbitrage[arbitrage_id]
        
        # Log the closure
        logger.info(f"Closing arbitrage {arbitrage_id} due to {reason}")
        
        # In production, would place closing orders here
        
        # Mark as closed
        arb_data['status'] = 'closed'
        arb_data['end_time'] = pd.Timestamp.now()
        arb_data['close_reason'] = reason
        
        # Publish closure event
        EventBus.get_instance().publish('arbitrage_closed', {
            'strategy': self.name,
            'id': arbitrage_id,
            'reason': reason,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    def _reset_daily_tracking(self):
        """
        Reset daily tracking metrics
        """
        # Store previous day's stats first
        yesterday = (pd.Timestamp.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Reset daily counters
        daily_metrics = {
            'opportunities_detected': self.performance_metrics['opportunities_detected'],
            'opportunities_executed': self.performance_metrics['opportunities_executed'],
            'successful_arbitrages': self.performance_metrics['successful_arbitrages'],
            'failed_arbitrages': self.performance_metrics['failed_arbitrages'],
            'total_pips_captured': self.performance_metrics['total_pips_captured'],
            'avg_execution_time_ms': self.performance_metrics['avg_execution_time_ms'],
            'profitable_trades_percent': self.performance_metrics['profitable_trades_percent'],
            'by_type': self.performance_metrics['by_type'].copy()
        }
        
        # Reset metrics
        self.performance_metrics['opportunities_detected'] = 0
        self.performance_metrics['opportunities_executed'] = 0
        self.performance_metrics['successful_arbitrages'] = 0
        self.performance_metrics['failed_arbitrages'] = 0
        self.performance_metrics['total_pips_captured'] = 0
        
        # Keep the by_type structure but reset counts
        for arb_type in self.performance_metrics['by_type']:
            self.performance_metrics['by_type'][arb_type]['count'] = 0
            self.performance_metrics['by_type'][arb_type]['pips'] = 0
            
        # Store daily stats
        if not hasattr(self, 'daily_stats'):
            self.daily_stats = {}
            
        self.daily_stats[yesterday] = daily_metrics
        
        logger.info(f"Reset daily tracking. Previous day total: {daily_metrics['total_pips_captured']:.2f} pips")
        
    def generate_signals(self, data: Dict[str, pd.DataFrame], current_time: pd.Timestamp) -> Dict[str, Any]:
        """
        Generate trading signals for the arbitrage strategy
        
        Args:
            data: Market data dictionary
            current_time: Current timestamp
            
        Returns:
            Dictionary of signal information
        """
        # For arbitrage, we don't generate signals in the traditional way
        # Instead, we continuously monitor the market and execute directly
        # Return an empty dictionary
        return {}
        
    def calculate_regime_compatibility(self, market_regime: MarketRegime) -> float:
        """
        Calculate compatibility score for a given market regime
        
        Args:
            market_regime: MarketRegime enum value
            
        Returns:
            Compatibility score 0.0-1.0
        """
        # Compatibility matrix
        compatibility = {
            MarketRegime.TRENDING_BULL: 0.5,     # Moderate in trending
            MarketRegime.TRENDING_BEAR: 0.5,     # Moderate in trending
            MarketRegime.RANGE_BOUND: 0.6,       # Good in range-bound
            MarketRegime.BREAKOUT: 0.4,          # Fair in breakout
            MarketRegime.VOLATILE: 0.85,         # Excellent in volatile
            MarketRegime.CHOPPY: 0.8,            # Very good in choppy
            MarketRegime.LOW_VOLATILITY: 0.65,   # Good in low volatility
            MarketRegime.HIGH_VOLATILITY: 0.9,   # Excellent in high volatility
            MarketRegime.NORMAL: 0.7            # Good in normal conditions
        }
        
        return compatibility.get(market_regime, 0.6)  # Default to 0.6 for unknown regimes
        
    def optimize_for_regime(self, market_regime: MarketRegime) -> None:
        """
        Optimize strategy parameters for the given market regime
        
        Args:
            market_regime: Market regime to optimize for
        """
        if market_regime == MarketRegime.TRENDING_BULL or market_regime == MarketRegime.TRENDING_BEAR:
            # In trending, focus more on statistical arbitrage
            self.parameters['triangular_enabled'] = True
            self.parameters['statistical_enabled'] = True
            self.parameters['min_tri_profit_pips'] *= 1.2  # Require more profit in trending
            self.parameters['z_score_threshold'] *= 0.9  # More sensitive in trending
            
        elif market_regime == MarketRegime.RANGE_BOUND or market_regime == MarketRegime.LOW_VOLATILITY:
            # In range-bound, both strategies work well
            self.parameters['triangular_enabled'] = True
            self.parameters['statistical_enabled'] = True
            self.parameters['min_tri_profit_pips'] *= 0.9  # Less profit required
            self.parameters['z_score_threshold'] *= 1.1  # Less sensitive in range-bound
            
        elif market_regime == MarketRegime.VOLATILE or market_regime == MarketRegime.HIGH_VOLATILITY:
            # In volatile, triangular works better
            self.parameters['triangular_enabled'] = True
            self.parameters['statistical_enabled'] = True
            self.parameters['min_tri_profit_pips'] *= 0.8  # Less profit required, more opportunities
            self.parameters['max_pos_size_percent'] *= 0.9  # Reduce size in volatile
            
        elif market_regime == MarketRegime.BREAKOUT:
            # In breakout, be more cautious
            self.parameters['triangular_enabled'] = True
            self.parameters['statistical_enabled'] = False  # Disable statistical during breakout
            self.parameters['min_tri_profit_pips'] *= 1.3  # Require more profit in breakout
            self.parameters['max_pos_size_percent'] *= 0.8  # Reduce size
            
        elif market_regime == MarketRegime.CHOPPY:
            # In choppy, statistical works well
            self.parameters['triangular_enabled'] = True
            self.parameters['statistical_enabled'] = True
            self.parameters['z_score_threshold'] *= 0.85  # More sensitive in choppy
            
        elif market_regime == MarketRegime.NORMAL:
            # In normal conditions, use balanced approach
            self.parameters['triangular_enabled'] = True
            self.parameters['statistical_enabled'] = True
            # Reset to defaults
            
        # Log the optimization
        logger.info(f"Optimized arbitrage strategy for {market_regime.name} regime")
        
        # Publish parameter change
        EventBus.get_instance().publish('strategy_parameters_changed', {
            'strategy': self.name,
            'regime': market_regime.name,
            'parameters': self.parameters
        })
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the strategy
        
        Returns:
            Dictionary of metrics
        """
        # Get base metrics
        metrics = super().get_performance_metrics()
        
        # Add arbitrage-specific metrics
        metrics.update({
            'opportunities_detected': self.performance_metrics['opportunities_detected'],
            'opportunities_executed': self.performance_metrics['opportunities_executed'],
            'successful_arbitrages': self.performance_metrics['successful_arbitrages'],
            'failed_arbitrages': self.performance_metrics['failed_arbitrages'],
            'total_pips_captured': self.performance_metrics['total_pips_captured'],
            'avg_execution_time_ms': self.performance_metrics['avg_execution_time_ms'],
            'profitable_trades_percent': self.performance_metrics['profitable_trades_percent'],
            'by_type': self.performance_metrics['by_type']
        })
        
        return metrics
