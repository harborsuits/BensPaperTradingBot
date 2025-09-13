#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Straddle/Strangle Strategy Implementation

This strategy implements straddle and strangle options strategies following the standardized
template system, ensuring compatibility with the backtester and trading dashboard.

A straddle is created by:
1. Buying a call option at a specific strike price
2. Buying a put option at the same strike price
3. Both options have the same expiration date

A strangle is created by:
1. Buying a call option at a higher strike price
2. Buying a put option at a lower strike price
3. Both options have the same expiration date
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Handle missing scipy gracefully
try:
    from scipy.stats import percentileofscore
    SCIPY_AVAILABLE = True
except ImportError:
    logging.warning("scipy not available - using fallback percentileofscore implementation")
    SCIPY_AVAILABLE = False
    
    # Fallback implementation for percentileofscore
    def percentileofscore(a, score, kind='rank'):
        """Simple implementation of percentileofscore for when scipy is not available"""
        a = np.asarray(a)
        n = len(a)
        if n == 0:
            return 0.0
            
        # Count values below score
        if kind == 'rank':
            count = (a < score).sum()
            return 100.0 * count / n
        elif kind == 'weak':
            count = (a <= score).sum()
            return 100.0 * count / n
        else:  # kind == 'strict'
            count = (a < score).sum()
            return 100.0 * count / n

from trading_bot.strategies.base.options_base_strategy import OptionsBaseStrategy
from trading_bot.market.market_data import MarketData
from trading_bot.market.universe import Universe
from trading_bot.market.option_chains import OptionChains

# Define strategy enumerations locally to avoid circular imports
from enum import Enum

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
    'description': 'Straddle/Strangle strategy for profiting from significant price movements in either direction',
    'risk_level': 'high',
    'typical_holding_period': '14-30 days',
    'performance_metrics': {
        'sharpe_ratio': 0.0,       # Will be updated by performance tracker
        'sortino_ratio': 0.0,      # Will be updated by performance tracker
        'max_drawdown': 0.0,       # Will be updated by performance tracker
        'win_rate': 0.0,           # Will be updated by performance tracker
        'avg_profit_per_trade': 0.0 # Will be updated by performance tracker
    },
    'broker_compatibility': ['tradier', 'alpaca'], # List of compatible brokers
    'tags': ['volatility', 'neutral', 'event-driven', 'options', 'non-directional']
})
class StraddleStrangleStrategy(OptionsBaseStrategy):
    """
    Straddle/Strangle Options Strategy
    
    This strategy involves buying both call and put options to profit from significant
    price movements in either direction. It's a volatility strategy that performs well
    in environments with large price swings or before major market events.
    
    Key characteristics:
    - Unlimited profit potential in either direction
    - Limited risk (premium paid)
    - Requires significant price movement to be profitable
    - Benefits from volatility expansion
    - Suffers from time decay if the underlying doesn't move
    - High cost compared to directional strategies
    """
    
    # Default parameters for straddle/strangle strategy
    DEFAULT_PARAMS = {
        'strategy_name': 'straddle_strangle',
        'strategy_version': '1.0.0',
        'asset_class': 'options',
        'strategy_type': 'volatility',
        'timeframe': 'swing',
        'market_regime': 'volatile',
        
        # Strategy type
        'strategy_variant': 'straddle',      # 'straddle' or 'strangle'
        
        # Stock selection criteria
        'min_stock_price': 30.0,              # Minimum stock price to consider
        'max_stock_price': 500.0,             # Maximum stock price to consider
        'min_option_volume': 100,             # Minimum option volume
        'min_option_open_interest': 200,      # Minimum option open interest
        'min_iv_percentile': 30,              # Minimum IV percentile
        'max_iv_percentile': 100,             # Maximum IV percentile (high IV is good for this strategy)
        
        # Technical analysis parameters
        'min_historical_days': 90,            # Days of historical data required
        'bollinger_period': 20,               # Period for Bollinger Bands
        'bollinger_std': 2.0,                 # Standard deviations for Bollinger Bands
        'min_contraction_pct': 0.05,          # Minimum contraction % in Bollinger Bands (for breakout)
        'max_adx': 20,                        # Maximum ADX (want low trend strength before breakout)
        
        # Event-based parameters
        'days_before_earnings': 5,            # Buy N days before earnings
        'days_before_fed': 2,                 # Buy N days before Fed announcements
        'use_economic_calendar': True,        # Consider economic calendar for entry
        'min_event_importance': 'high',       # Minimum importance of economic event
        
        # Option parameters
        'target_dte': 30,                     # Target days to expiration
        'min_dte': 21,                        # Minimum days to expiration
        'max_dte': 45,                        # Maximum days to expiration
        'delta_target': 0.50,                 # Target delta for straddle (ATM)
        'strangle_call_delta': 0.30,          # Target delta for strangle call (OTM)
        'strangle_put_delta': -0.30,          # Target delta for strangle put (OTM)
        
        # Risk management parameters
        'max_position_size_percent': 0.03,    # Maximum position size as % of portfolio
        'max_loss_percent': 0.50,             # Maximum loss as % of premium paid
        'profit_target_percent': 100,         # Target profit as % of premium paid
        'stop_after_event': True,             # Exit after the anticipated event
        
        # Exit parameters
        'iv_decrease_exit': 0.20,             # Exit if IV decreases by this % after entry
        'early_profit_target': 0.50,          # Take profit at 50% of target if reached before event
        'time_stop': 14,                      # Maximum days to hold if no event or movement
    }
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 parameters: Dict[str, Any] = None,
                 broker_adapter = None,
                 event_bus = None):
        """
        Initialize the Straddle/Strangle strategy.
        
        Args:
            strategy_id: Unique identifier for this strategy instance
            name: Human-readable name of the strategy
            parameters: Strategy-specific parameters that override the defaults
            broker_adapter: Optional broker adapter for direct API access
            event_bus: Optional event bus for publishing/subscribing to events
        """
        super().__init__(strategy_id, name, parameters)
        
        # Strategy-specific tracking
        self.straddle_positions = {}          # Track current straddle/strangle positions
        self.economic_calendar = {}           # Track upcoming economic events
        self.earnings_calendar = {}           # Track upcoming earnings announcements
        self.volatility_metrics = {}          # Track historical and implied volatility
        self.performance_metrics = {          # Performance tracking
            'trades_total': 0,
            'trades_won': 0,
            'trades_lost': 0,
            'profit_total': 0.0,
            'loss_total': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0
        }
        
        # System integration
        self.broker_adapter = broker_adapter  # Direct broker API access if needed
        self.event_bus = event_bus            # Event pub/sub system
        self.health_metrics = {               # Health monitoring
            'last_run_time': None,
            'last_signal_time': None,
            'errors': [],
            'warnings': [],
            'status': 'initialized'
        }
        
        # Recovery and robustness tracking
        self.state_snapshots = []             # For state recovery if needed
        self.last_reconciliation = None       # Last position reconciliation timestamp
        
        # Subscribe to relevant events if event bus is provided
        if self.event_bus:
            self._subscribe_to_events()
            
        # Initialize strategy with a state snapshot
        self._create_state_snapshot('initialization')
        
        # Log initialization
        logger.info(f"Initialized {self._get_strategy_variant()} strategy with ID {strategy_id}")
        if self.event_bus:
            self._publish_status_event('initialized')

    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of tradable assets for straddle/strangle strategy.
        
        Args:
            market_data: Market data for analysis
            
        Returns:
            Universe object with selected symbols
        """
        universe = Universe()
        
        # Get all available symbols
        all_symbols = market_data.get_all_symbols()
        
        # Filter by price range
        min_price = self.parameters.get('min_stock_price', 30.0)
        max_price = self.parameters.get('max_stock_price', 500.0)
        
        filtered_symbols = []
        for symbol in all_symbols:
            quote = market_data.get_latest_quote(symbol)
            if not quote:
                continue
                
            price = quote.get('price', 0)
            if min_price <= price <= max_price:
                filtered_symbols.append(symbol)
        
        # Get historical data and calculate technical indicators
        lookback_days = self.parameters.get('min_historical_days', 90)
        
        for symbol in filtered_symbols:
            # Check if we have upcoming events for this symbol (if event-based trading is enabled)
            has_upcoming_event = self._has_upcoming_event(symbol)
            
            # Get historical data
            historical_data = market_data.get_historical_data(symbol, period=lookback_days)
            if historical_data is None or len(historical_data) < 30:  # Need at least 30 days for indicators
                continue
                
            # Convert to DataFrame if not already
            if not isinstance(historical_data, pd.DataFrame):
                historical_data = pd.DataFrame(historical_data)
            
            # Calculate Bollinger Bands for volatility contraction/expansion analysis
            bb_period = self.parameters.get('bollinger_period', 20)
            bb_std = self.parameters.get('bollinger_std', 2.0)
            historical_data['sma'] = historical_data['close'].rolling(window=bb_period).mean()
            historical_data['std'] = historical_data['close'].rolling(window=bb_period).std()
            historical_data['upper_band'] = historical_data['sma'] + (bb_std * historical_data['std'])
            historical_data['lower_band'] = historical_data['sma'] - (bb_std * historical_data['std'])
            
            # Calculate ADX to measure trend strength (we want low ADX for non-trending market)
            adx = self._calculate_adx(historical_data, period=14)
            
            # Calculate historical volatility
            historical_data['daily_return'] = historical_data['close'].pct_change()
            recent_vol = historical_data['daily_return'].rolling(window=20).std() * np.sqrt(252)  # Annualized
            
            # Calculate Bollinger Band width (indicator of volatility contraction)
            historical_data['bb_width'] = (historical_data['upper_band'] - historical_data['lower_band']) / historical_data['sma']
            
            # Volatility contraction check (narrowing Bollinger Bands indicate potential breakout)
            current_bb_width = historical_data['bb_width'].iloc[-1]
            past_bb_width = historical_data['bb_width'].iloc[-10:-1].mean()  # Average of past 10 days
            
            # Calculate contraction percentage
            contraction_pct = (past_bb_width - current_bb_width) / past_bb_width if past_bb_width > 0 else 0
            
            # Check IV percentile if available
            iv_percentile = None
            if hasattr(market_data, 'get_iv_percentile'):
                iv_percentile = market_data.get_iv_percentile(symbol)
            
            # Filter conditions for volatility strategies
            volatility_setup = False
            
            # 1. Check for volatility contraction (potential for expansion)
            min_contraction = self.parameters.get('min_contraction_pct', 0.05)
            if contraction_pct >= min_contraction:
                volatility_setup = True
            
            # 2. Check ADX for low trend strength (easier breakout in either direction)
            max_adx_value = self.parameters.get('max_adx', 20)
            if adx <= max_adx_value:
                volatility_setup = volatility_setup and True
                
            # 3. Check IV percentile (higher is better for volatility strategy entry)
            if iv_percentile is not None:
                min_iv = self.parameters.get('min_iv_percentile', 30)
                max_iv = self.parameters.get('max_iv_percentile', 100)
                iv_ok = min_iv <= iv_percentile <= max_iv
                volatility_setup = volatility_setup and iv_ok
            
            # 4. Event-based selection (if enabled)
            if self.parameters.get('use_economic_calendar', True) and has_upcoming_event:
                # Prioritize stocks with upcoming events
                volatility_setup = True
            
            # Check option liquidity if available
            liquidity_ok = True
            if hasattr(market_data, 'get_option_volume'):
                option_volume = market_data.get_option_volume(symbol)
                option_oi = market_data.get_option_open_interest(symbol)
                
                min_volume = self.parameters.get('min_option_volume', 100)
                min_oi = self.parameters.get('min_option_open_interest', 200)
                
                liquidity_ok = option_volume >= min_volume and option_oi >= min_oi
            
            # Add to universe if it meets our criteria
            if volatility_setup and liquidity_ok:
                universe.add_symbol(symbol)
        
        return universe
    
    def generate_signals(self, market_data: MarketData, 
                        option_chains: Optional[OptionChains] = None) -> List[Dict[str, Any]]:
        """
        Generate straddle/strangle signals based on market data and option chains.
        
        Args:
            market_data: Market data for analysis
            option_chains: Option chain data
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Get universe of tradable stocks
        universe = self.define_universe(market_data)
        symbols = universe.get_symbols()
        
        for symbol in symbols:
            # Skip if we already have a position on this symbol
            if symbol in self.straddle_positions:
                continue
                
            # Get option chain for this symbol
            if not option_chains or not option_chains.has_symbol(symbol):
                continue
                
            chain = option_chains.get_chain(symbol)
            if not chain:
                continue
                
            # Find appropriate options for straddle or strangle
            strategy_variant = self.parameters.get('strategy_variant', 'straddle')
            
            if strategy_variant == 'straddle':
                strategy_data = self._find_straddle(symbol, chain, market_data)
            else:  # strangle
                strategy_data = self._find_strangle(symbol, chain, market_data)
                
            if not strategy_data:
                continue
                
            # Create signal
            signal = self._create_signal(symbol, strategy_data, market_data, strategy_variant)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _has_upcoming_event(self, symbol: str) -> bool:
        """
        Check if a symbol has an upcoming event that could cause volatility.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if there is an upcoming event, False otherwise
        """
        # Check earnings calendar
        has_earnings = symbol in self.earnings_calendar
        if has_earnings:
            earnings_date = self.earnings_calendar.get(symbol)
            days_before = self.parameters.get('days_before_earnings', 5)
            
            if isinstance(earnings_date, date):
                days_to_earnings = (earnings_date - date.today()).days
                if 0 <= days_to_earnings <= days_before:
                    return True
        
        # Check economic calendar
        for event, event_data in self.economic_calendar.items():
            # Check if event affects this symbol (simplified)
            if event_data.get('symbols') and symbol in event_data.get('symbols'):
                event_date = event_data.get('date')
                importance = event_data.get('importance', 'low')
                min_importance = self.parameters.get('min_event_importance', 'high')
                days_before = self.parameters.get('days_before_fed', 2)
                
                if importance >= min_importance and isinstance(event_date, date):
                    days_to_event = (event_date - date.today()).days
                    if 0 <= days_to_event <= days_before:
                        return True
        
        return False
    
    def _calculate_adx(self, historical_data: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate the Average Directional Index (ADX) indicator.
        
        Args:
            historical_data: Historical price data
            period: Period for ADX calculation
            
        Returns:
            Current ADX value
        """
        # Calculate True Range (TR)
        historical_data['tr1'] = abs(historical_data['high'] - historical_data['low'])
        historical_data['tr2'] = abs(historical_data['high'] - historical_data['close'].shift())
        historical_data['tr3'] = abs(historical_data['low'] - historical_data['close'].shift())
        historical_data['tr'] = historical_data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate +DM and -DM
        historical_data['plus_dm'] = 0.0
        historical_data['minus_dm'] = 0.0
        
        # +DM
        mask = (historical_data['high'] - historical_data['high'].shift() > historical_data['low'].shift() - historical_data['low'])
        mask = mask & (historical_data['high'] - historical_data['high'].shift() > 0)
        historical_data.loc[mask, 'plus_dm'] = historical_data.loc[mask, 'high'] - historical_data.loc[mask, 'high'].shift()
        
        # -DM
        mask = (historical_data['low'].shift() - historical_data['low'] > historical_data['high'] - historical_data['high'].shift())
        mask = mask & (historical_data['low'].shift() - historical_data['low'] > 0)
        historical_data.loc[mask, 'minus_dm'] = historical_data.loc[mask, 'low'].shift() - historical_data.loc[mask, 'low']
        
        # Calculate smoothed TR, +DM14, and -DM14
        historical_data['tr14'] = historical_data['tr'].rolling(window=period).sum()
        historical_data['plus_dm14'] = historical_data['plus_dm'].rolling(window=period).sum()
        historical_data['minus_dm14'] = historical_data['minus_dm'].rolling(window=period).sum()
        
        # Calculate +DI14 and -DI14
        historical_data['plus_di14'] = 100 * historical_data['plus_dm14'] / historical_data['tr14']
        historical_data['minus_di14'] = 100 * historical_data['minus_dm14'] / historical_data['tr14']
        
        # Calculate DX
        historical_data['dx'] = 100 * abs(historical_data['plus_di14'] - historical_data['minus_di14']) / (historical_data['plus_di14'] + historical_data['minus_di14'])
        
        # Calculate ADX
        historical_data['adx'] = historical_data['dx'].rolling(window=period).mean()
        
        # Return current ADX value
        return historical_data['adx'].iloc[-1] if not pd.isna(historical_data['adx'].iloc[-1]) else 0.0
        
    def _find_straddle(self, symbol: str, option_chain: Any, market_data: MarketData) -> Dict[str, Any]:
        """
        Find suitable options for a straddle strategy.
        
        Args:
            symbol: Symbol to trade
            option_chain: Option chain data
            market_data: Market data
            
        Returns:
            Dictionary with selected option data
        """
        # Get current stock price
        quote = market_data.get_latest_quote(symbol)
        if not quote:
            return None
            
        current_price = quote.get('price', 0)
        if current_price <= 0:
            return None
            
        # Get target DTE parameters
        target_dte = self.parameters.get('target_dte', 30)
        min_dte = self.parameters.get('min_dte', 21)
        max_dte = self.parameters.get('max_dte', 45)
        
        # Find appropriate expiration date
        expiration_dates = option_chain.get_expiration_dates()
        if not expiration_dates:
            return None
            
        # Filter expirations by DTE
        valid_expirations = []
        for exp_date in expiration_dates:
            if isinstance(exp_date, str):
                try:
                    exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                except ValueError:
                    continue
                    
            days_to_expiration = (exp_date - date.today()).days
            if min_dte <= days_to_expiration <= max_dte:
                valid_expirations.append((exp_date, days_to_expiration))
                
        # Sort by closest to target DTE
        if not valid_expirations:
            return None
            
        valid_expirations.sort(key=lambda x: abs(x[1] - target_dte))
        selected_expiration, days_to_expiration = valid_expirations[0]
        
        # Get calls and puts for selected expiration
        calls = option_chain.get_calls(selected_expiration)
        puts = option_chain.get_puts(selected_expiration)
        
        if not calls or not puts:
            return None
            
        # Find ATM strike
        target_delta = self.parameters.get('delta_target', 0.50)  # 0.50 for ATM options
        
        # Find closest strike to current price
        closest_strike = min([option.get('strike', 0) for option in calls + puts], 
                            key=lambda x: abs(x - current_price))
        
        # Find the call and put at this strike
        selected_call = None
        selected_put = None
        
        for call in calls:
            if call.get('strike') == closest_strike:
                selected_call = call
                break
                
        for put in puts:
            if put.get('strike') == closest_strike:
                selected_put = put
                break
                
        if not selected_call or not selected_put:
            return None
            
        # Check for sufficient volume and open interest
        min_volume = self.parameters.get('min_option_volume', 100)
        min_oi = self.parameters.get('min_option_open_interest', 200)
        
        call_volume = selected_call.get('volume', 0)
        call_oi = selected_call.get('open_interest', 0)
        put_volume = selected_put.get('volume', 0)
        put_oi = selected_put.get('open_interest', 0)
        
        if call_volume < min_volume or call_oi < min_oi or put_volume < min_volume or put_oi < min_oi:
            return None
            
        # Calculate total premium and other statistics
        call_price = selected_call.get('ask', 0)
        put_price = selected_put.get('ask', 0)
        total_premium = call_price + put_price
        
        # Check IV (ideally, we want high IV for straddles/strangles)
        call_iv = selected_call.get('implied_volatility', 0)
        put_iv = selected_put.get('implied_volatility', 0)
        avg_iv = (call_iv + put_iv) / 2 if call_iv and put_iv else 0
        
        return {
            'expiration': selected_expiration,
            'days_to_expiration': days_to_expiration,
            'strike': closest_strike,
            'call': selected_call,
            'put': selected_put,
            'call_price': call_price,
            'put_price': put_price,
            'total_premium': total_premium,
            'implied_volatility': avg_iv,
            'current_price': current_price,
            'breakeven_upper': closest_strike + total_premium,
            'breakeven_lower': closest_strike - total_premium
        }
    
    def _find_strangle(self, symbol: str, option_chain: Any, market_data: MarketData) -> Dict[str, Any]:
        """
        Find suitable options for a strangle strategy.
        
        Args:
            symbol: Symbol to trade
            option_chain: Option chain data
            market_data: Market data
            
        Returns:
            Dictionary with selected option data
        """
        # Get current stock price
        quote = market_data.get_latest_quote(symbol)
        if not quote:
            return None
            
        current_price = quote.get('price', 0)
        if current_price <= 0:
            return None
            
        # Get target DTE parameters
        target_dte = self.parameters.get('target_dte', 30)
        min_dte = self.parameters.get('min_dte', 21)
        max_dte = self.parameters.get('max_dte', 45)
        
        # Find appropriate expiration date (same as straddle)
        expiration_dates = option_chain.get_expiration_dates()
        if not expiration_dates:
            return None
            
        # Filter expirations by DTE
        valid_expirations = []
        for exp_date in expiration_dates:
            if isinstance(exp_date, str):
                try:
                    exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                except ValueError:
                    continue
                    
            days_to_expiration = (exp_date - date.today()).days
            if min_dte <= days_to_expiration <= max_dte:
                valid_expirations.append((exp_date, days_to_expiration))
                
        # Sort by closest to target DTE
        if not valid_expirations:
            return None
            
        valid_expirations.sort(key=lambda x: abs(x[1] - target_dte))
        selected_expiration, days_to_expiration = valid_expirations[0]
        
        # Get calls and puts for selected expiration
        calls = option_chain.get_calls(selected_expiration)
        puts = option_chain.get_puts(selected_expiration)
        
        if not calls or not puts:
            return None
            
        # For strangle, we want OTM options with specific deltas
        call_delta_target = self.parameters.get('strangle_call_delta', 0.30)
        put_delta_target = self.parameters.get('strangle_put_delta', -0.30)
        
        # Find calls with delta closest to call_delta_target
        if 'delta' in calls[0]:
            # If delta is directly available
            calls_sorted = sorted(calls, key=lambda x: abs(x.get('delta', 0) - call_delta_target))
            selected_call = calls_sorted[0] if calls_sorted else None
        else:
            # Estimate by finding OTM calls
            otm_calls = [c for c in calls if c.get('strike', 0) > current_price]
            if otm_calls:
                # Sort by strike price and take the one closest to current price + 5%
                target_strike = current_price * 1.05
                selected_call = min(otm_calls, key=lambda x: abs(x.get('strike', 0) - target_strike))
            else:
                selected_call = None
                
        # Find puts with delta closest to put_delta_target
        if 'delta' in puts[0]:
            # If delta is directly available
            puts_sorted = sorted(puts, key=lambda x: abs(x.get('delta', 0) - put_delta_target))
            selected_put = puts_sorted[0] if puts_sorted else None
        else:
            # Estimate by finding OTM puts
            otm_puts = [p for p in puts if p.get('strike', 0) < current_price]
            if otm_puts:
                # Sort by strike price and take the one closest to current price - 5%
                target_strike = current_price * 0.95
                selected_put = min(otm_puts, key=lambda x: abs(x.get('strike', 0) - target_strike))
            else:
                selected_put = None
                
        if not selected_call or not selected_put:
            return None
            
        # Check for sufficient volume and open interest
        min_volume = self.parameters.get('min_option_volume', 100)
        min_oi = self.parameters.get('min_option_open_interest', 200)
        
        call_volume = selected_call.get('volume', 0)
        call_oi = selected_call.get('open_interest', 0)
        put_volume = selected_put.get('volume', 0)
        put_oi = selected_put.get('open_interest', 0)
        
        if call_volume < min_volume or call_oi < min_oi or put_volume < min_volume or put_oi < min_oi:
            return None
            
        # Calculate total premium and other statistics
        call_price = selected_call.get('ask', 0)
        put_price = selected_put.get('ask', 0)
        call_strike = selected_call.get('strike', 0)
        put_strike = selected_put.get('strike', 0)
        total_premium = call_price + put_price
        
        # Check IV
        call_iv = selected_call.get('implied_volatility', 0)
        put_iv = selected_put.get('implied_volatility', 0)
        avg_iv = (call_iv + put_iv) / 2 if call_iv and put_iv else 0
        
        return {
            'expiration': selected_expiration,
            'days_to_expiration': days_to_expiration,
            'call_strike': call_strike,
            'put_strike': put_strike,
            'call': selected_call,
            'put': selected_put,
            'call_price': call_price,
            'put_price': put_price,
            'total_premium': total_premium,
            'implied_volatility': avg_iv,
            'current_price': current_price,
            'breakeven_upper': call_strike + total_premium,
            'breakeven_lower': put_strike - total_premium,
            'width': call_strike - put_strike
        }
    
    def _create_signal(self, symbol: str, strategy_data: Dict[str, Any], 
                     market_data: MarketData, strategy_variant: str) -> Dict[str, Any]:
        """
        Create a trading signal for straddle/strangle strategy.
        
        Args:
            symbol: Symbol to trade
            strategy_data: Strategy data (from _find_straddle or _find_strangle)
            market_data: Market data
            strategy_variant: 'straddle' or 'strangle'
            
        Returns:
            Signal dictionary
        """
        # Calculate position size
        account_value = self.get_account_value()
        max_position_pct = self.parameters.get('max_position_size_percent', 0.03)
        max_position_value = account_value * max_position_pct
        
        # Calculate number of contracts
        premium_per_contract = strategy_data.get('total_premium', 0) * 100  # 100 shares per contract
        
        # Ensure we're not risking too much
        num_contracts = int(max_position_value / premium_per_contract)
        if num_contracts < 1:
            num_contracts = 1  # Minimum 1 contract
            
        # Calculate actual investment amount
        actual_investment = num_contracts * premium_per_contract
        
        # Create signal details
        expiration = strategy_data.get('expiration')
        if isinstance(expiration, date):
            expiration_str = expiration.strftime('%Y-%m-%d')
        else:
            expiration_str = str(expiration)
            
        # Determine exit parameters
        profit_target_pct = self.parameters.get('profit_target_percent', 100)
        max_loss_pct = self.parameters.get('max_loss_percent', 50)
        time_stop = self.parameters.get('time_stop', 14)
        
        # Create signal object
        signal = {
            'symbol': symbol,
            'strategy': f'{strategy_variant}_options',
            'direction': 'long',  # Both straddle and strangle are long volatility
            'timestamp': datetime.now().isoformat(),
            'expiration': expiration_str,
            'days_to_expiration': strategy_data.get('days_to_expiration', 0),
            'confidence': 0.75,  # Default confidence
            'position_size': num_contracts,
            'investment_amount': actual_investment,
            'premium_per_contract': premium_per_contract / 100,  # Per share
            'iv': strategy_data.get('implied_volatility', 0),
            'current_price': strategy_data.get('current_price', 0),
            'action': 'buy',
            'option_legs': [],
            'exit_parameters': {
                'profit_target_percent': profit_target_pct,
                'max_loss_percent': max_loss_pct,
                'time_stop': time_stop,
                'entry_iv': strategy_data.get('implied_volatility', 0),
                'iv_decrease_exit': self.parameters.get('iv_decrease_exit', 0.20),
                'stop_after_event': self.parameters.get('stop_after_event', True)
            }
        }
        
        # Add option legs
        if strategy_variant == 'straddle':
            signal['option_legs'] = [
                {
                    'option_type': 'call',
                    'strike': strategy_data.get('strike', 0),
                    'price': strategy_data.get('call_price', 0),
                    'action': 'buy',
                    'position_size': num_contracts,
                    'expiration': expiration_str,
                },
                {
                    'option_type': 'put',
                    'strike': strategy_data.get('strike', 0),
                    'price': strategy_data.get('put_price', 0),
                    'action': 'buy',
                    'position_size': num_contracts,
                    'expiration': expiration_str,
                }
            ]
            signal['breakeven_upper'] = strategy_data.get('breakeven_upper', 0)
            signal['breakeven_lower'] = strategy_data.get('breakeven_lower', 0)
            
        else:  # strangle
            signal['option_legs'] = [
                {
                    'option_type': 'call',
                    'strike': strategy_data.get('call_strike', 0),
                    'price': strategy_data.get('call_price', 0),
                    'action': 'buy',
                    'position_size': num_contracts,
                    'expiration': expiration_str,
                },
                {
                    'option_type': 'put',
                    'strike': strategy_data.get('put_strike', 0),
                    'price': strategy_data.get('put_price', 0),
                    'action': 'buy',
                    'position_size': num_contracts,
                    'expiration': expiration_str,
                }
            ]
            signal['breakeven_upper'] = strategy_data.get('breakeven_upper', 0)
            signal['breakeven_lower'] = strategy_data.get('breakeven_lower', 0)
            signal['width'] = strategy_data.get('width', 0)
            
        # Adjust confidence based on various factors
        self._adjust_signal_confidence(signal, strategy_data, market_data, strategy_variant)
        
        # Store position data for tracking
        self._track_position(symbol, signal, strategy_data)
        
        return signal
    
    def _adjust_signal_confidence(self, signal: Dict[str, Any], strategy_data: Dict[str, Any],
                                market_data: MarketData, strategy_variant: str) -> None:
        """
        Adjust signal confidence based on various market factors.
        
        Args:
            signal: Signal dictionary to adjust
            strategy_data: Strategy data for the trade
            market_data: Market data
            strategy_variant: 'straddle' or 'strangle'
        """
        # Start with base confidence
        confidence = 0.75
        
        # 1. Adjust based on IV percentile (higher is better for vol strategies)
        iv_percentile = market_data.get_iv_percentile(signal['symbol']) if hasattr(market_data, 'get_iv_percentile') else 50
        
        # IV percentile booster - higher IV percentile means higher implied volatility
        if iv_percentile > 80:
            confidence += 0.10
        elif iv_percentile > 60:
            confidence += 0.05
        elif iv_percentile < 30:
            confidence -= 0.10
        elif iv_percentile < 40:
            confidence -= 0.05
            
        # 2. Adjust based on upcoming events (higher confidence if events coming)
        if self._has_upcoming_event(signal['symbol']):
            confidence += 0.15
        
        # 3. Adjust based on historical volatility vs implied volatility
        hist_vol = self._get_historical_volatility(signal['symbol'], market_data)
        implied_vol = strategy_data.get('implied_volatility', 0)
        
        # If historical vol is much lower than implied vol, this suggests options are overpriced
        if hist_vol > 0 and implied_vol > 0:
            vol_ratio = implied_vol / hist_vol
            if vol_ratio > 1.5:  # IV is 50% higher than HV - potentially overpriced
                confidence -= 0.10
            elif vol_ratio < 0.8:  # IV is lower than HV - potentially underpriced
                confidence += 0.10
        
        # 4. Adjust based on market trend strength - prefer neutral for vol strategies
        if hasattr(market_data, 'get_adx'):
            adx = market_data.get_adx(signal['symbol'])
            if adx < 15:  # Very weak trend, good for straddle/strangle
                confidence += 0.05
            elif adx > 30:  # Strong trend, less ideal for non-directional strategies
                confidence -= 0.08
        
        # 5. Strategy-specific adjustments
        if strategy_variant == 'straddle':
            # For straddle, we want the stock price to be as close as possible to strike
            current_price = strategy_data.get('current_price', 0)
            strike = strategy_data.get('strike', 0)
            if strike > 0:
                price_deviation = abs(current_price - strike) / strike
                if price_deviation < 0.01:  # Within 1% of strike - good for straddle
                    confidence += 0.05
                elif price_deviation > 0.03:  # More than 3% away from strike - less ideal
                    confidence -= 0.05
                    
        elif strategy_variant == 'strangle':
            # For strangle, we want balanced pricing between call and put
            call_price = strategy_data.get('call_price', 0)
            put_price = strategy_data.get('put_price', 0)
            if call_price > 0 and put_price > 0:
                price_ratio = max(call_price, put_price) / min(call_price, put_price)
                if price_ratio < 1.3:  # Relatively balanced
                    confidence += 0.05
                elif price_ratio > 2.0:  # Very unbalanced
                    confidence -= 0.05
        
        # Ensure confidence stays within reasonable range
        confidence = max(0.3, min(0.95, confidence))
        
        # Update signal confidence
        signal['confidence'] = round(confidence, 2)
    
    def _track_position(self, symbol: str, signal: Dict[str, Any], strategy_data: Dict[str, Any]) -> None:
        """
        Track straddle/strangle position for monitoring and exit strategies.
        
        Args:
            symbol: Symbol being traded
            signal: Signal dictionary
            strategy_data: Strategy data for the position
        """
        # Create a position tracking entry
        position = {
            'entry_time': datetime.now(),
            'symbol': symbol,
            'strategy_variant': signal.get('strategy').replace('_options', ''),
            'entry_price': strategy_data.get('current_price', 0),
            'entry_iv': strategy_data.get('implied_volatility', 0),
            'expiration': strategy_data.get('expiration'),
            'days_to_expiration': strategy_data.get('days_to_expiration', 0),
            'position_size': signal.get('position_size', 0),
            'premium_paid': signal.get('investment_amount', 0),
            'legs': signal.get('option_legs', []),
            'profit_target': signal.get('investment_amount', 0) * (1 + signal.get('exit_parameters', {}).get('profit_target_percent', 100) / 100),
            'max_loss': signal.get('investment_amount', 0) * (1 - signal.get('exit_parameters', {}).get('max_loss_percent', 50) / 100),
            'time_stop': (datetime.now() + timedelta(days=signal.get('exit_parameters', {}).get('time_stop', 14))).date(),
            'breakeven_upper': signal.get('breakeven_upper', 0),
            'breakeven_lower': signal.get('breakeven_lower', 0),
            'iv_decrease_exit': signal.get('exit_parameters', {}).get('iv_decrease_exit', 0.20),
            'stop_after_event': signal.get('exit_parameters', {}).get('stop_after_event', True),
            'current_value': signal.get('investment_amount', 0),  # Initially equal to premium paid
            'max_value': signal.get('investment_amount', 0),
            'min_value': signal.get('investment_amount', 0),
            'last_update': datetime.now(),
            'status': 'open'
        }
        
        # Add unique aspects based on strategy variant
        if position['strategy_variant'] == 'straddle':
            position['strike'] = strategy_data.get('strike', 0)
        else:  # strangle
            position['call_strike'] = strategy_data.get('call_strike', 0)
            position['put_strike'] = strategy_data.get('put_strike', 0)
            position['width'] = strategy_data.get('width', 0)
        
        # Store in tracking dict
        self.straddle_positions[symbol] = position
        
        # Log position opening
        logger.info(f"Opened {position['strategy_variant']} position on {symbol} with {position['position_size']} "
                   f"contracts, premium paid: ${position['premium_paid']:.2f}")
    
    def _get_historical_volatility(self, symbol: str, market_data: MarketData, days: int = 20) -> float:
        """
        Calculate historical volatility for a symbol.
        
        Args:
            symbol: Symbol to calculate for
            market_data: Market data
            days: Number of days to calculate over
            
        Returns:
            Historical volatility (annualized)
        """
        try:
            # Get historical data
            hist_data = market_data.get_historical_data(symbol, period=max(days+10, 30))  # Add buffer
            if hist_data is None or len(hist_data) < days:
                return 0.0
                
            # Convert to DataFrame if needed
            if not isinstance(hist_data, pd.DataFrame):
                hist_data = pd.DataFrame(hist_data)
                
            # Calculate daily returns
            hist_data['daily_return'] = hist_data['close'].pct_change()
            
            # Calculate and annualize volatility
            volatility = hist_data['daily_return'].tail(days).std() * np.sqrt(252)  # Annualized
            return volatility
        except Exception as e:
            logger.error(f"Error calculating historical volatility for {symbol}: {e}")
            return 0.0
    
    def on_exit_signal(self, market_data: MarketData, option_chains: Optional[OptionChains] = None) -> List[Dict[str, Any]]:
        """
        Generate exit signals for straddle/strangle positions based on market data and option chains.
        
        Args:
            market_data: Market data for analysis
            option_chains: Option chain data
            
        Returns:
            List of exit signal dictionaries
        """
        exit_signals = []
        
        # Check each active position for exit conditions
        for symbol, position in list(self.straddle_positions.items()):
            if position['status'] != 'open':
                continue
                
            # Get current option quotes
            current_quotes = {}
            current_value = 0
            current_iv = 0
            
            if option_chains and option_chains.has_symbol(symbol):
                chain = option_chains.get_chain(symbol)
                if chain:
                    # Get updated values for each leg
                    for leg in position['legs']:
                        option_type = leg.get('option_type')
                        strike = leg.get('strike')
                        expiration = leg.get('expiration')
                        
                        # Convert expiration if it's a string
                        if isinstance(expiration, str):
                            try:
                                expiration_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                            except ValueError:
                                expiration_date = None
                        else:
                            expiration_date = expiration
                            
                        if not expiration_date:
                            continue
                            
                        # Get the current option price
                        if option_type == 'call':
                            options = chain.get_calls(expiration_date)
                        else:  # put
                            options = chain.get_puts(expiration_date)
                            
                        for option in options:
                            if option.get('strike') == strike:
                                # Calculate current value using bid price (what we could sell for)
                                current_bid = option.get('bid', 0)
                                position_size = leg.get('position_size', 0)
                                leg_value = current_bid * position_size * 100  # 100 shares per contract
                                
                                current_value += leg_value
                                current_quotes[f"{option_type}_{strike}"] = current_bid
                                
                                # Track IV
                                current_iv += option.get('implied_volatility', 0)
                                break
                    
                    # Average IV across legs
                    if len(position['legs']) > 0:
                        current_iv /= len(position['legs'])
            
            # If we couldn't get option quotes, estimate based on stock price movement
            if not current_quotes and hasattr(market_data, 'get_latest_quote'):
                quote = market_data.get_latest_quote(symbol)
                if quote:
                    current_stock_price = quote.get('price', 0)
                    entry_price = position.get('entry_price', 0)
                    
                    if current_stock_price > 0 and entry_price > 0:
                        # Rough estimate based on stock price movement
                        price_change_pct = abs(current_stock_price - entry_price) / entry_price
                        estimated_value_factor = max(0, min(3, price_change_pct * 2))  # Cap at 3x
                        current_value = position.get('premium_paid', 0) * estimated_value_factor
            
            # Update position tracking
            position['current_value'] = current_value
            position['max_value'] = max(position['max_value'], current_value)
            position['min_value'] = min(position['min_value'], current_value)
            position['last_update'] = datetime.now()
            
            # Check exit conditions
            exit_triggered = False
            exit_reason = ''
            
            # 1. Profit target
            profit_target = position.get('profit_target', 0)
            if current_value >= profit_target:
                exit_triggered = True
                exit_reason = 'Profit target reached'
            
            # 2. Stop loss
            max_loss = position.get('max_loss', 0)
            if current_value <= max_loss:
                exit_triggered = True
                exit_reason = 'Stop loss triggered'
            
            # 3. Time stop
            time_stop_date = position.get('time_stop')
            if time_stop_date and date.today() >= time_stop_date:
                exit_triggered = True
                exit_reason = 'Time stop reached'
            
            # 4. IV decrease exit
            entry_iv = position.get('entry_iv', 0)
            iv_decrease_threshold = position.get('iv_decrease_exit', 0.20)
            if entry_iv > 0 and current_iv > 0:
                iv_decrease = (entry_iv - current_iv) / entry_iv
                if iv_decrease >= iv_decrease_threshold:
                    exit_triggered = True
                    exit_reason = 'IV decrease threshold reached'
            
            # 5. Event-based exit
            stop_after_event = position.get('stop_after_event', True)
            if stop_after_event and not self._has_upcoming_event(symbol):
                # Check if we're past earnings date or other major event
                days_in_position = (datetime.now() - position.get('entry_time')).days
                if days_in_position >= 2:  # At least 2 days have passed since entry
                    exit_triggered = True
                    exit_reason = 'Target event has passed'
            
            # 6. Proximity to expiration
            if option_chains and option_chains.has_symbol(symbol):
                expiration = position.get('expiration')
                if isinstance(expiration, str):
                    try:
                        expiration_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                    except ValueError:
                        expiration_date = None
                else:
                    expiration_date = expiration
                    
                if expiration_date:
                    days_to_expiration = (expiration_date - date.today()).days
                    if days_to_expiration <= 7:  # Close if 7 or fewer days to expiration
                        exit_triggered = True
                        exit_reason = 'Too close to expiration (increased gamma risk)'
            
            # 7. Drawdown from peak value
            if position['max_value'] > position['premium_paid']:
                # Only check drawdown if we've seen a profit at some point
                drawdown = (position['max_value'] - current_value) / position['max_value']
                if drawdown >= 0.30:  # 30% drawdown from peak
                    exit_triggered = True
                    exit_reason = '30% drawdown from peak value'
            
            # Generate exit signal if triggered
            if exit_triggered:
                exit_signal = {
                    'symbol': symbol,
                    'strategy': position.get('strategy_variant', '') + '_options',
                    'action': 'sell',
                    'direction': 'close',
                    'position_size': position.get('position_size', 0),
                    'timestamp': datetime.now().isoformat(),
                    'reason': exit_reason,
                    'entry_premium': position.get('premium_paid', 0),
                    'exit_value': current_value,
                    'profit_loss': current_value - position.get('premium_paid', 0),
                    'profit_loss_pct': (current_value / position.get('premium_paid', 1) - 1) * 100,
                    'days_held': (datetime.now() - position.get('entry_time')).days,
                    'option_legs': []
                }
                
                # Add option legs to close
                for leg in position.get('legs', []):
                    # Create a closing leg (opposite action of entry)
                    closing_leg = leg.copy()
                    closing_leg['action'] = 'sell' if leg.get('action') == 'buy' else 'buy'
                    
                    # Update price if available
                    leg_key = f"{leg.get('option_type')}_{leg.get('strike')}"
                    if leg_key in current_quotes:
                        closing_leg['price'] = current_quotes[leg_key]
                    
                    exit_signal['option_legs'].append(closing_leg)
                
                # Add to exit signals
                exit_signals.append(exit_signal)
                
                # Update position status
                position['status'] = 'closing'
                logger.info(f"Closing {position['strategy_variant']} position on {symbol}: {exit_reason}. "
                           f"P&L: ${exit_signal['profit_loss']:.2f} ({exit_signal['profit_loss_pct']:.2f}%)")
        
        return exit_signals
                
    def on_trade_update(self, trade_update: Dict[str, Any]) -> None:
        """
        Handle trade update events for straddle/strangle positions.
        
        Args:
            trade_update: Trade update information
        """
        symbol = trade_update.get('symbol')
        action = trade_update.get('action')
        status = trade_update.get('status')
        
        if symbol not in self.straddle_positions:
            return
            
        position = self.straddle_positions[symbol]
        
        # Handle trade fill confirmations
        if status == 'filled':
            if action in ['sell', 'sell_to_close'] and position['status'] == 'closing':
                # Position successfully closed
                position['status'] = 'closed'
                fill_price = trade_update.get('fill_price', 0)
                fill_quantity = trade_update.get('fill_quantity', 0)
                
                # Update final P&L
                if fill_price and fill_quantity:
                    exit_value = fill_price * fill_quantity * 100  # 100 shares per contract
                    position['exit_value'] = exit_value
                    position['profit_loss'] = exit_value - position.get('premium_paid', 0)
                    position['profit_loss_pct'] = (exit_value / position.get('premium_paid', 1) - 1) * 100
                    
                    logger.info(f"Closed {position['strategy_variant']} position on {symbol}. "
                               f"Final P&L: ${position['profit_loss']:.2f} ({position['profit_loss_pct']:.2f}%)")
                    
                    # Remove from active positions if completely closed
                    if all(leg.get('status') == 'filled' for leg in trade_update.get('option_legs', [])):
                        if not hasattr(self, 'closed_positions'):
                            self.closed_positions = []
                        self.closed_positions.append(position.copy())
                        del self.straddle_positions[symbol]
    
    def update_economic_calendar(self, calendar_data: Dict[str, Any]) -> None:
        """
        Update the economic calendar used for event-based entries and exits.
        
        Args:
            calendar_data: Economic calendar data
        """
        self.economic_calendar = calendar_data
        logger.info(f"Updated economic calendar with {len(calendar_data)} events")
    
    def update_earnings_calendar(self, earnings_data: Dict[str, Any]) -> None:
        """
        Update the earnings calendar used for event-based entries and exits.
        
        Args:
            earnings_data: Earnings calendar data
        """
        self.earnings_calendar = earnings_data
        logger.info(f"Updated earnings calendar with {len(earnings_data)} events")
        
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current straddle/strangle positions.
        
        Returns:
            List of position dictionaries
        """
        return list(self.straddle_positions.values())
    
    def get_closed_positions(self) -> List[Dict[str, Any]]:
        """
        Get all closed straddle/strangle positions.
        
        Returns:
            List of closed position dictionaries
        """
        if not hasattr(self, 'closed_positions'):
            self.closed_positions = []
        return self.closed_positions
        
    def _subscribe_to_events(self) -> None:
        """
        Subscribe to relevant events on the event bus.
        """
        if not self.event_bus:
            return
            
        # Subscribe to market data updates
        self.event_bus.subscribe(f"market_data.update", self._on_market_data_update)
        
        # Subscribe to option chain updates
        self.event_bus.subscribe(f"option_chains.update", self._on_option_chains_update)
        
        # Subscribe to calendar events
        self.event_bus.subscribe(f"economic_calendar.update", self.update_economic_calendar)
        self.event_bus.subscribe(f"earnings_calendar.update", self.update_earnings_calendar)
        
        # Subscribe to trade updates
        self.event_bus.subscribe(f"trade.update.{self.strategy_id}", self.on_trade_update)
        
        # Subscribe to risk management events
        self.event_bus.subscribe(f"risk.circuit_breaker", self._on_circuit_breaker)
        self.event_bus.subscribe(f"risk.drawdown_alert", self._on_drawdown_alert)
        
        # Subscribe to system status events
        self.event_bus.subscribe("system.reconciliation", self._on_position_reconciliation)
        self.event_bus.subscribe("system.health_check", self._on_health_check)
        
        logger.info(f"Strategy {self.strategy_id} subscribed to events")
    
    def _publish_status_event(self, status: str, details: Dict[str, Any] = None) -> None:
        """
        Publish a strategy status event to the event bus.
        
        Args:
            status: Status message
            details: Additional details about the status
        """
        if not self.event_bus:
            return
            
        event_data = {
            'strategy_id': self.strategy_id,
            'strategy_name': self.name,
            'strategy_type': self._get_strategy_variant(),
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.event_bus.publish(f"strategy.status.{self.strategy_id}", event_data)
        
    def _create_state_snapshot(self, reason: str) -> None:
        """
        Create a snapshot of the current strategy state for recovery if needed.
        
        Args:
            reason: Why the snapshot is being created
        """
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'strategy_id': self.strategy_id,
            'positions': self.straddle_positions.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'health_metrics': self.health_metrics.copy(),
            'parameters': self.params.copy()
        }
        
        self.state_snapshots.append(snapshot)
        
        # Keep only the last 10 snapshots to manage memory
        if len(self.state_snapshots) > 10:
            self.state_snapshots.pop(0)
            
    def _recover_from_snapshot(self, index: int = -1) -> bool:
        """
        Recover strategy state from a snapshot.
        
        Args:
            index: Index of the snapshot to recover from, default is the most recent
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if not self.state_snapshots or abs(index) > len(self.state_snapshots):
            logger.error(f"Cannot recover state: no snapshot available at index {index}")
            return False
            
        snapshot = self.state_snapshots[index]
        logger.info(f"Recovering strategy state from snapshot taken at {snapshot['timestamp']} for reason: {snapshot['reason']}")
        
        # Restore state
        self.straddle_positions = snapshot['positions'].copy()
        self.performance_metrics = snapshot['performance_metrics'].copy()
        self.health_metrics = snapshot['health_metrics'].copy()
        
        # Log recovery
        self._publish_status_event('recovered', {'snapshot_time': snapshot['timestamp'], 'reason': snapshot['reason']})
        
        return True
    
    def _get_strategy_variant(self) -> str:
        """
        Get the current strategy variant ('straddle' or 'strangle').
        
        Returns:
            Strategy variant name
        """
        return self.params.get('strategy_variant', 'straddle')
    
    def _on_market_data_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle market data update events.
        
        Args:
            event_data: Market data update information
        """
        # Update strategy with latest market data
        symbols = event_data.get('symbols', [])
        
        # Update volatility metrics if symbols in our universe
        for symbol in symbols:
            if symbol in self.straddle_positions or self._is_in_universe(symbol):
                self._update_volatility_metrics(symbol, event_data.get('data', {}).get(symbol, {}))
                
        # Update strategy health status
        self.health_metrics['last_market_data_update'] = datetime.now().isoformat()
    
    def _on_option_chains_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle option chains update events.
        
        Args:
            event_data: Option chains update information
        """
        # Update option metrics for any open positions
        symbols = event_data.get('symbols', [])
        option_data = event_data.get('data', {})
        
        for symbol in symbols:
            if symbol in self.straddle_positions:
                self._update_position_with_option_data(symbol, option_data.get(symbol, {}))
    
    def _on_circuit_breaker(self, event_data: Dict[str, Any]) -> None:
        """
        Handle circuit breaker events from the risk management system.
        
        Args:
            event_data: Circuit breaker event information
        """
        circuit_type = event_data.get('type')
        affected_symbols = event_data.get('symbols', [])
        severity = event_data.get('severity', 'warning')
        
        logger.warning(f"Circuit breaker triggered: {circuit_type} with severity {severity}")
        
        # If any of our positions are affected, take appropriate action
        affected_positions = [sym for sym in affected_symbols if sym in self.straddle_positions]
        
        if affected_positions and severity == 'critical':
            # Create emergency exit signals for affected positions
            logger.critical(f"Creating emergency exits for {len(affected_positions)} positions due to circuit breaker")
            self._create_emergency_exits(affected_positions, circuit_type)
            
        # Update strategy health
        self.health_metrics['warnings'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'circuit_breaker',
            'details': event_data
        })
        
        # Create a state snapshot
        self._create_state_snapshot(f"circuit_breaker_{circuit_type}")
    
    def _on_drawdown_alert(self, event_data: Dict[str, Any]) -> None:
        """
        Handle drawdown alerts from the risk management system.
        
        Args:
            event_data: Drawdown alert information
        """
        drawdown_pct = event_data.get('drawdown_pct', 0)
        affected_strategies = event_data.get('strategies', [])
        
        # Check if this strategy is affected
        if self.strategy_id in affected_strategies or 'all' in affected_strategies:
            logger.warning(f"Drawdown alert: {drawdown_pct:.2f}% drawdown detected for strategy {self.strategy_id}")
            
            # Update performance metrics
            self.performance_metrics['max_drawdown'] = max(self.performance_metrics.get('max_drawdown', 0), drawdown_pct)
            
            # If drawdown exceeds a critical threshold, take protective action
            if drawdown_pct >= self.params.get('max_drawdown_threshold', 10.0):
                logger.critical(f"Critical drawdown threshold exceeded: {drawdown_pct:.2f}%")
                self._reduce_risk_exposure(drawdown_pct)
    
    def _on_position_reconciliation(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position reconciliation events from the system.
        
        Args:
            event_data: Position reconciliation information
        """
        # Extract our positions from reconciliation data
        strategy_positions = event_data.get('positions', {}).get(self.strategy_id, {})
        
        # Compare with our local state and fix discrepancies
        self._reconcile_positions(strategy_positions)
        self.last_reconciliation = datetime.now().isoformat()
        
    def _on_health_check(self, event_data: Dict[str, Any]) -> None:
        """
        Handle system health check events.
        
        Args:
            event_data: Health check information
        """
        # Respond with our current health status
        self.health_metrics['last_run_time'] = datetime.now().isoformat()
        
        # Publish our health status
        self._publish_status_event('health_check_response', {
            'health_metrics': self.health_metrics,
            'position_count': len(self.straddle_positions),
            'errors_count': len(self.health_metrics.get('errors', [])),
            'warnings_count': len(self.health_metrics.get('warnings', []))
        })
    
    def _is_in_universe(self, symbol: str) -> bool:
        """
        Check if a symbol is in our current trading universe.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            True if in universe, False otherwise
        """
        # Implementation depends on how universe is stored
        if hasattr(self, 'universe') and self.universe:
            return symbol in self.universe.get_symbols()
        return False
    
    def _update_volatility_metrics(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """
        Update volatility metrics for a symbol.
        
        Args:
            symbol: Symbol to update
            market_data: Latest market data for the symbol
        """
        if not market_data:
            return
            
        # Initialize if needed
        if symbol not in self.volatility_metrics:
            self.volatility_metrics[symbol] = {
                'historical_volatility': 0.0,
                'implied_volatility': 0.0,
                'volatility_rank': 0.0,
                'last_updated': None
            }
            
        # Update with latest data if available
        metrics = self.volatility_metrics[symbol]
        
        if 'historical_volatility' in market_data:
            metrics['historical_volatility'] = market_data['historical_volatility']
            
        if 'implied_volatility' in market_data:
            metrics['implied_volatility'] = market_data['implied_volatility']
            
        metrics['last_updated'] = datetime.now().isoformat()
    
    def _update_position_with_option_data(self, symbol: str, option_data: Dict[str, Any]) -> None:
        """
        Update a position with the latest option data.
        
        Args:
            symbol: Symbol for the position
            option_data: Latest option data
        """
        if symbol not in self.straddle_positions or not option_data:
            return
            
        position = self.straddle_positions[symbol]
        
        # Update option legs with latest quotes
        for leg in position.get('option_legs', []):
            option_key = leg.get('option_key')
            leg_type = leg.get('option_type')
            
            # Find the option in the chain
            if option_key and option_key in option_data:
                leg_data = option_data[option_key]
                
                # Update pricing
                leg['current_price'] = leg_data.get('last', leg_data.get('mid', 0.0))
                leg['bid'] = leg_data.get('bid', 0.0)
                leg['ask'] = leg_data.get('ask', 0.0)
                leg['implied_volatility'] = leg_data.get('implied_volatility', 0.0)
                leg['delta'] = leg_data.get('delta', 0.0)
                leg['theta'] = leg_data.get('theta', 0.0)
                
                # Calculate leg P&L
                entry_price = leg.get('entry_price', 0.0)
                quantity = leg.get('quantity', 0)
                leg['current_value'] = leg['current_price'] * quantity * 100  # 100 shares per contract
                leg['profit_loss'] = (leg['current_price'] - entry_price) * quantity * 100
                leg['profit_loss_pct'] = (leg['current_price'] / entry_price - 1) * 100 if entry_price else 0
        
        # Update overall position metrics
        self._update_position_metrics(symbol)
    
    def _update_position_metrics(self, symbol: str) -> None:
        """
        Update overall position metrics based on leg data.
        
        Args:
            symbol: Symbol for the position to update
        """
        if symbol not in self.straddle_positions:
            return
            
        position = self.straddle_positions[symbol]
        legs = position.get('option_legs', [])
        
        # Calculate position level metrics
        total_current_value = sum(leg.get('current_value', 0) for leg in legs)
        total_entry_value = sum(leg.get('entry_price', 0) * leg.get('quantity', 0) * 100 for leg in legs)
        
        position['current_value'] = total_current_value
        position['profit_loss'] = total_current_value - total_entry_value
        position['profit_loss_pct'] = (total_current_value / total_entry_value - 1) * 100 if total_entry_value else 0
        
        # Calculate Greeks
        position['total_delta'] = sum(leg.get('delta', 0) * leg.get('quantity', 0) * 100 for leg in legs)
        position['total_theta'] = sum(leg.get('theta', 0) * leg.get('quantity', 0) * 100 for leg in legs)
        
        # Update last update time
        position['last_updated'] = datetime.now().isoformat()
    
    def _reconcile_positions(self, broker_positions: Dict[str, Any]) -> None:
        """
        Reconcile our position tracking with broker data.
        
        Args:
            broker_positions: Position data from the broker
        """
        # Look for positions that exist in broker data but not in our tracking
        for symbol, broker_position in broker_positions.items():
            if symbol not in self.straddle_positions:
                logger.warning(f"Position reconciliation: Found position for {symbol} in broker data that's not in our tracking")
                # Add to our tracking
                self.straddle_positions[symbol] = broker_position
                
        # Look for positions that exist in our tracking but not in broker data
        symbols_to_remove = []
        for symbol in self.straddle_positions:
            if symbol not in broker_positions:
                logger.warning(f"Position reconciliation: Position for {symbol} exists in our tracking but not in broker data")
                symbols_to_remove.append(symbol)
                
        # Remove positions not in broker data
        for symbol in symbols_to_remove:
            logger.info(f"Position reconciliation: Removing {symbol} from tracking as it no longer exists at broker")
            if not hasattr(self, 'closed_positions'):
                self.closed_positions = []
            self.closed_positions.append(self.straddle_positions[symbol])
            del self.straddle_positions[symbol]
            
        # Reconciliation complete
        logger.info(f"Position reconciliation complete. Tracking {len(self.straddle_positions)} positions.")
        
    def _create_emergency_exits(self, symbols: List[str], reason: str) -> None:
        """
        Create emergency exit signals for specified positions.
        
        Args:
            symbols: List of symbols to exit
            reason: Reason for the emergency exit
        """
        exit_signals = []
        
        for symbol in symbols:
            if symbol in self.straddle_positions:
                position = self.straddle_positions[symbol]
                
                # Create exit signal
                exit_signal = {
                    'symbol': symbol,
                    'action': 'close',
                    'strategy_id': self.strategy_id,
                    'order_type': 'market',  # Use market orders for emergency exits
                    'reason': f"Emergency exit: {reason}",
                    'timestamp': datetime.now().isoformat(),
                    'option_legs': []
                }
                
                # Add option legs
                for leg in position.get('option_legs', []):
                    closing_leg = {
                        'option_key': leg.get('option_key'),
                        'option_type': leg.get('option_type'),
                        'option_action': 'sell' if leg.get('option_action') == 'buy' else 'buy',  # Reverse the action
                        'quantity': leg.get('quantity', 0),
                        'order_type': 'market'
                    }
                    exit_signal['option_legs'].append(closing_leg)
                
                exit_signals.append(exit_signal)
                position['status'] = 'emergency_exit'
                
                logger.critical(f"Emergency exit created for {symbol} due to {reason}")
        
        # Publish exit signals to event bus
        if exit_signals and self.event_bus:
            self.event_bus.publish(f"strategy.emergency_exit.{self.strategy_id}", {
                'strategy_id': self.strategy_id,
                'exit_signals': exit_signals,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            })
    
    def _reduce_risk_exposure(self, drawdown_pct: float) -> None:
        """
        Reduce risk exposure in response to significant drawdown.
        
        Args:
            drawdown_pct: Current drawdown percentage
        """
        # Implementation depends on risk management approach
        # Could reduce position sizes, exit some positions, or adjust parameters
        
        # Example: Exit worst performing positions
        if self.straddle_positions:
            # Sort positions by performance (worst first)
            positions_by_performance = sorted(
                [(symbol, pos) for symbol, pos in self.straddle_positions.items()],
                key=lambda x: x[1].get('profit_loss_pct', 0)
            )
            
            # Determine how many positions to exit based on drawdown severity
            exit_count = max(1, int(len(positions_by_performance) * drawdown_pct / 100))
            positions_to_exit = [symbol for symbol, _ in positions_by_performance[:exit_count]]
            
            logger.warning(f"Reducing risk exposure due to {drawdown_pct:.2f}% drawdown. Exiting {exit_count} positions.")
            self._create_emergency_exits(positions_to_exit, f"drawdown_protection_{drawdown_pct:.2f}pct")
            
        # Update strategy parameters to be more conservative
        self._adjust_parameters_for_risk_reduction()
        
        # Create a state snapshot
        self._create_state_snapshot(f"risk_reduction_drawdown_{drawdown_pct:.2f}pct")
    
    def _adjust_parameters_for_risk_reduction(self) -> None:
        """
        Adjust strategy parameters to be more conservative after drawdown.
        """
        # Example: Increase profit targets, tighten stop losses, reduce position sizes
        self.params['profit_target_pct'] = min(self.params.get('profit_target_pct', 50) * 0.8, 25)  # Lower profit target for quicker exits
        self.params['stop_loss_pct'] = max(self.params.get('stop_loss_pct', 25) * 0.8, 15)  # Tighter stop loss
        self.params['position_size_pct'] = self.params.get('position_size_pct', 5) * 0.5  # Half position sizes
        
        logger.info(f"Adjusted strategy parameters for risk reduction: profit_target={self.params['profit_target_pct']}%, "
                  f"stop_loss={self.params['stop_loss_pct']}%, position_size={self.params['position_size_pct']}%")
        
        # Publish parameter adjustment event
        if self.event_bus:
            self._publish_status_event('parameters_adjusted', {
                'reason': 'risk_reduction',
                'new_parameters': {
                    'profit_target_pct': self.params['profit_target_pct'],
                    'stop_loss_pct': self.params['stop_loss_pct'],
                    'position_size_pct': self.params['position_size_pct']
                }
            })
            
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the strategy.
        
        Returns:
            Dictionary with health metrics
        """
        # Update current time
        current_time = datetime.now()
        
        # Calculate time since last run
        last_run_time = None
        if self.health_metrics.get('last_run_time'):
            last_run_time = datetime.fromisoformat(self.health_metrics['last_run_time'])
        
        time_since_last_run = None
        if last_run_time:
            time_since_last_run = (current_time - last_run_time).total_seconds()
        
        health_status = {
            'strategy_id': self.strategy_id,
            'strategy_name': self.name,
            'strategy_variant': self._get_strategy_variant(),
            'active_positions': len(self.straddle_positions),
            'closed_positions': len(self.get_closed_positions()),
            'errors_count': len(self.health_metrics.get('errors', [])),
            'warnings_count': len(self.health_metrics.get('warnings', [])),
            'time_since_last_run': time_since_last_run,
            'last_reconciliation': self.last_reconciliation,
            'status': self.health_metrics.get('status', 'unknown'),
            'performance': {
                'win_rate': self._calculate_win_rate(),
                'average_profit': self._calculate_average_profit(),
                'max_drawdown': self.performance_metrics.get('max_drawdown', 0.0)
            }
        }
        
        return health_status
    
    def _calculate_win_rate(self) -> float:
        """
        Calculate the win rate for closed trades.
        
        Returns:
            Win rate as a percentage
        """
        trades_total = self.performance_metrics.get('trades_total', 0)
        trades_won = self.performance_metrics.get('trades_won', 0)
        
        if trades_total == 0:
            return 0.0
            
        return (trades_won / trades_total) * 100
    
    def _calculate_average_profit(self) -> float:
        """
        Calculate the average profit per trade.
        
        Returns:
            Average profit per trade in dollars
        """
        trades_total = self.performance_metrics.get('trades_total', 0)
        profit_total = self.performance_metrics.get('profit_total', 0.0)
        
        if trades_total == 0:
            return 0.0
            
        return profit_total / trades_total
