#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Execution Quality Modeling for BensBot
Models slippage, spread, and latency for realistic trade execution
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketImpactModel:
    """Model market impact of orders based on order size and liquidity"""
    base_impact_bps: float = 0.5      # Base impact in basis points
    market_depth_usd: float = 1000000  # Liquidity depth in USD
    volatility_factor: float = 1.2     # Impact multiplier during high volatility
    
    def calculate_impact(self, order_size_usd: float, 
                        volatility_percentile: float = 0.5) -> float:
        """
        Calculate market impact in basis points
        
        Args:
            order_size_usd: Order size in USD
            volatility_percentile: Market volatility percentile (0-1)
            
        Returns:
            float: Impact in basis points
        """
        # Calculate relative size compared to market depth
        relative_size = order_size_usd / self.market_depth_usd
        
        # Apply volatility multiplier
        vol_multiplier = 1.0 + (volatility_percentile - 0.5) * self.volatility_factor
        
        # Non-linear impact model: impact ~ sqrt(relative_size)
        # This reflects that impact doesn't increase linearly with size
        impact_bps = self.base_impact_bps * np.sqrt(relative_size) * vol_multiplier
        
        return impact_bps

@dataclass
class LatencyModel:
    """Model execution latency based on network, broker, and market conditions"""
    base_latency_ms: float = 50.0         # Baseline latency (network + broker)
    volatility_penalty_ms: float = 20.0   # Additional latency during volatility
    congestion_factor: float = 1.5        # Multiplier during market congestion
    jitter_range: Tuple[float, float] = (0.8, 1.2)  # Random jitter range
    
    def calculate_latency(self, volatility_percentile: float = 0.5,
                         congestion_level: float = 0.0) -> float:
        """
        Calculate expected latency in milliseconds
        
        Args:
            volatility_percentile: Market volatility percentile (0-1)
            congestion_level: Market congestion level (0-1)
            
        Returns:
            float: Expected latency in milliseconds
        """
        # Calculate volatility impact
        vol_addition = self.volatility_penalty_ms * volatility_percentile
        
        # Apply congestion multiplier
        congestion_multiplier = 1.0 + (congestion_level * self.congestion_factor)
        
        # Calculate base latency
        latency_ms = (self.base_latency_ms + vol_addition) * congestion_multiplier
        
        # Add random jitter to simulate real-world variability
        jitter = random.uniform(self.jitter_range[0], self.jitter_range[1])
        
        return latency_ms * jitter

class ExecutionQualityModel:
    """
    Complete model for execution quality factors
    Models spread, slippage, latency, and market impact for realistic simulation
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the execution quality model
        
        Args:
            parameters: Dictionary of model parameters
        """
        # Default parameters
        default_params = {
            # Spread model parameters
            'min_spread_pips': {
                'major_pairs': 0.5,     # Major forex pairs (EURUSD, etc.)
                'minor_pairs': 1.0,     # Minor pairs (EURGBP, etc.)
                'exotic_pairs': 2.0     # Exotic pairs (USDTRY, etc.)
            },
            'volatility_spread_multiplier': 2.0,  # Spread expansion in volatility
            
            # Slippage parameters
            'base_slippage_pips': 0.1,   # Base slippage in normal conditions
            'slippage_market_hours': {   # Session-specific multipliers
                'asian': 1.2,            # Asian session multiplier
                'european': 1.0,         # European session multiplier
                'us': 1.0,               # US session multiplier
                'overlap': 0.9,          # Session overlap multiplier (high liquidity)
                'off_hours': 1.5         # Off-hours multiplier (low liquidity)
            },
            
            # Latency parameters
            'base_latency_ms': 45.0,
            'volatility_latency_factor': 1.5,
            
            # Market impact parameters
            'market_depth': {
                'major_pairs': 5000000,   # $5M depth for major pairs
                'minor_pairs': 2000000,   # $2M depth for minor pairs
                'exotic_pairs': 500000    # $500K depth for exotic pairs
            },
            
            # API rate limiting parameters
            'rate_limit_threshold': 0.8,  # 80% of max rate before slowdown
            'rate_limit_backoff': 1.5,    # Exponential backoff factor
            
            # Price drift parameters (market movement during execution)
            'price_drift_factor': 0.1,    # Drift factor as fraction of spread
            
            # Order type impact
            'limit_slippage_reduction': 0.8,  # Reduction factor for limit orders
            'stop_slippage_increase': 1.3     # Increase factor for stop orders
        }
        
        # Use provided parameters or defaults
        self.params = default_params.copy()
        if parameters:
            self.params.update(parameters)
            
        # Initialize component models
        self.impact_model = MarketImpactModel(
            base_impact_bps=0.5,
            market_depth_usd=self.params['market_depth']['major_pairs']
        )
        
        self.latency_model = LatencyModel(
            base_latency_ms=self.params['base_latency_ms']
        )
        
        # Initialize rate limiting tracking
        self.api_calls = {
            'count': 0,
            'last_reset': datetime.now(),
            'backoff_level': 0
        }
        
        logger.info("Execution Quality Model initialized")
        
    def get_pair_category(self, symbol: str) -> str:
        """
        Categorize a forex pair as major, minor, or exotic
        
        Args:
            symbol: Forex pair symbol (e.g., 'EURUSD')
            
        Returns:
            str: Category ('major_pairs', 'minor_pairs', or 'exotic_pairs')
        """
        # List of major forex pairs
        major_pairs = ['EURUSD', 'USDJPY', 'GBPUSD', 'USDCHF', 
                      'AUDUSD', 'USDCAD', 'NZDUSD']
        
        if symbol in major_pairs:
            return 'major_pairs'
        
        # List of major currencies
        major_currencies = ['EUR', 'USD', 'JPY', 'GBP', 'CHF', 'AUD', 'CAD', 'NZD']
        
        # Check if both currencies in the pair are major (minor pair)
        if len(symbol) == 6:  # Standard forex pair length
            base = symbol[:3]
            quote = symbol[3:]
            
            if base in major_currencies and quote in major_currencies:
                return 'minor_pairs'
            
        # Default to exotic
        return 'exotic_pairs'
        
    def calculate_expected_spread(self, symbol: str, 
                                 volatility_percentile: float = 0.5,
                                 session: str = 'european') -> float:
        """
        Calculate expected spread in pips based on pair type, 
        market conditions, and session
        
        Args:
            symbol: Forex pair symbol
            volatility_percentile: Market volatility percentile (0-1)
            session: Trading session ('asian', 'european', 'us', 'overlap', 'off_hours')
            
        Returns:
            float: Expected spread in pips
        """
        # Get pair category
        category = self.get_pair_category(symbol)
        
        # Get base spread
        base_spread = self.params['min_spread_pips'][category]
        
        # Apply session adjustment
        session_factor = 1.0
        if session == 'off_hours':
            session_factor = 1.5
        elif session == 'asian':
            session_factor = 1.2
        elif session == 'overlap':
            session_factor = 0.9
            
        # Increase spread with volatility (non-linear)
        # Higher volatility causes spreads to widen dramatically
        vol_factor = 1.0 + (volatility_percentile ** 2) * self.params['volatility_spread_multiplier']
        
        # Calculate final spread
        spread = base_spread * session_factor * vol_factor
        
        return spread
        
    def calculate_expected_slippage(self, symbol: str, 
                                  order_size_usd: float,
                                  market_session: str = 'european',
                                  volatility_percentile: float = 0.5,
                                  is_market_order: bool = True,
                                  order_type: str = 'market') -> float:
        """
        Calculate expected slippage in pips
        
        Args:
            symbol: Forex pair symbol
            order_size_usd: Order size in USD
            market_session: Trading session
            volatility_percentile: Volatility percentile (0-1)
            is_market_order: Whether this is a market order
            order_type: Order type ('market', 'limit', 'stop')
            
        Returns:
            float: Expected slippage in pips
        """
        # No slippage for limit orders that are not marketable
        if not is_market_order and order_type == 'limit':
            return 0.0
            
        # Get pair category
        category = self.get_pair_category(symbol)
        
        # Base slippage
        base_slippage = self.params['base_slippage_pips']
        
        # Session adjustment
        session_factor = self.params['slippage_market_hours'].get(market_session, 1.0)
        
        # Market impact based on order size and liquidity
        market_depth = self.params['market_depth'][category]
        self.impact_model.market_depth_usd = market_depth
        impact_bps = self.impact_model.calculate_impact(order_size_usd, volatility_percentile)
        
        # Convert basis points to pips (approximate)
        impact_pips = impact_bps / 10.0
        
        # Order type adjustment
        order_factor = 1.0
        if order_type == 'limit' and is_market_order:
            # Marketable limit orders have reduced slippage
            order_factor = self.params['limit_slippage_reduction']
        elif order_type == 'stop':
            # Stop orders typically have higher slippage
            order_factor = self.params['stop_slippage_increase']
            
        # Total expected slippage
        slippage = (base_slippage * session_factor * order_factor) + impact_pips
        
        # Apply volatility factor
        slippage *= (1.0 + volatility_percentile)
        
        return slippage
        
    def model_order_execution(self, symbol: str, 
                             order_size_usd: float,
                             price: float, 
                             is_buy: bool,
                             market_session: str = 'european',
                             volatility_percentile: float = 0.5,
                             is_market_order: bool = True,
                             order_type: str = 'market') -> Dict[str, Any]:
        """
        Model complete order execution including price, timing, and costs
        
        Args:
            symbol: Forex pair symbol
            order_size_usd: Order size in USD
            price: Current/requested price
            is_buy: True for buy orders, False for sell orders
            market_session: Trading session
            volatility_percentile: Volatility percentile (0-1)
            is_market_order: Whether this is a market order
            order_type: Order type ('market', 'limit', 'stop')
            
        Returns:
            dict: Execution details including price, timing, and costs
        """
        # Calculate spread
        spread_pips = self.calculate_expected_spread(
            symbol, volatility_percentile, market_session
        )
        
        # Calculate slippage
        slippage_pips = self.calculate_expected_slippage(
            symbol, order_size_usd, market_session, 
            volatility_percentile, is_market_order, order_type
        )
        
        # Calculate latency
        # Higher order sizes can cause more congestion
        congestion = min(1.0, volatility_percentile + (order_size_usd / 10000000))
        latency_ms = self.latency_model.calculate_latency(
            volatility_percentile, congestion
        )
        
        # Price movement during latency (random walk model)
        drift_factor = self.params['price_drift_factor']
        latency_price_drift = np.random.normal(0, spread_pips * drift_factor)
        
        # Calculate execution price
        if is_market_order:
            # Market order: price + spread + slippage + drift
            price_adjustment = (spread_pips / 2.0)  # Half spread
            
            if is_buy:
                # Buy at ask + slippage + drift
                price_adjustment += slippage_pips + latency_price_drift
            else:
                # Sell at bid - slippage + drift
                price_adjustment = -price_adjustment - slippage_pips + latency_price_drift
                
            # For FX, convert pips to price movement (depends on pair)
            # For most pairs, 1 pip = 0.0001, except JPY pairs where 1 pip = 0.01
            pip_value = 0.0001
            if 'JPY' in symbol:
                pip_value = 0.01
                
            execution_price = price + (price_adjustment * pip_value)
        else:
            # Limit order: only execute if price is favorable
            execution_price = price
            
        # Calculate total cost in pips
        total_cost_pips = spread_pips
        if is_market_order:
            total_cost_pips += slippage_pips
        
        # Build result
        result = {
            'expected_price': execution_price,
            'execution_time_ms': latency_ms,
            'spread_cost_pips': spread_pips,
            'slippage_pips': slippage_pips if is_market_order else 0,
            'total_cost_pips': total_cost_pips,
            'price_drift_pips': latency_price_drift,
            'order_type': order_type,
            'is_buy': is_buy,
            'original_price': price
        }
        
        return result
    
    def model_api_rate_limiting(self, endpoint: str = 'default') -> Dict[str, Any]:
        """
        Model API rate limiting and determine if request should be delayed
        
        Args:
            endpoint: API endpoint or category
            
        Returns:
            dict: Rate limiting information
        """
        # Track API call
        self.api_calls['count'] += 1
        current_time = datetime.now()
        
        # Reset counter if more than an hour has passed
        if (current_time - self.api_calls['last_reset']).total_seconds() > 3600:
            self.api_calls['count'] = 1
            self.api_calls['last_reset'] = current_time
            self.api_calls['backoff_level'] = 0
            
        # Check if we're approaching rate limit
        rate_limit = 60  # Default: 60 calls per hour
        rate_usage = self.api_calls['count'] / rate_limit
        
        # Determine if we need to backoff
        if rate_usage > self.params['rate_limit_threshold']:
            # Calculate backoff delay
            backoff_factor = self.params['rate_limit_backoff'] ** self.api_calls['backoff_level']
            delay_seconds = 1 * backoff_factor
            
            # Increase backoff level
            self.api_calls['backoff_level'] += 1
            
            return {
                'should_delay': True,
                'delay_seconds': delay_seconds,
                'rate_usage': rate_usage,
                'calls_count': self.api_calls['count'],
                'backoff_level': self.api_calls['backoff_level']
            }
        else:
            # No delay needed
            return {
                'should_delay': False,
                'delay_seconds': 0,
                'rate_usage': rate_usage,
                'calls_count': self.api_calls['count'],
                'backoff_level': self.api_calls['backoff_level']
            }
    
    def estimate_price_at_future_time(self, symbol: str, 
                                    current_price: float,
                                    seconds_ahead: float, 
                                    volatility_annualized: float) -> Tuple[float, float]:
        """
        Estimate price distribution at a future time using random walk model
        
        Args:
            symbol: Forex pair symbol
            current_price: Current price
            seconds_ahead: Seconds into the future
            volatility_annualized: Annualized volatility as decimal (e.g., 0.10 for 10%)
            
        Returns:
            tuple: (expected_price, standard_deviation)
        """
        # Convert annualized volatility to the time period
        # T is fraction of year (seconds / seconds_in_year)
        seconds_in_year = 365 * 24 * 60 * 60
        time_fraction = seconds_ahead / seconds_in_year
        
        # Standard deviation of price change scales with sqrt(time)
        price_std = current_price * volatility_annualized * np.sqrt(time_fraction)
        
        # Expected price (no drift in short term)
        expected_price = current_price
        
        return expected_price, price_std
    
    def get_fill_probability(self, symbol: str, 
                           current_price: float, 
                           target_price: float,
                           volatility_annualized: float,
                           seconds_horizon: float) -> float:
        """
        Calculate probability of a limit or stop order being filled
        
        Args:
            symbol: Forex pair symbol
            current_price: Current market price
            target_price: Target price for the order
            volatility_annualized: Annualized volatility
            seconds_horizon: Time horizon in seconds
            
        Returns:
            float: Probability of fill (0-1)
        """
        # Get expected price distribution at future time
        expected_price, price_std = self.estimate_price_at_future_time(
            symbol, current_price, seconds_horizon, volatility_annualized
        )
        
        # Calculate how many standard deviations away the target is
        if price_std == 0:
            # Avoid division by zero
            price_std = 0.0001 * current_price
            
        z_score = (target_price - expected_price) / price_std
        
        # Calculate probability using normal CDF
        from scipy.stats import norm
        probability = norm.cdf(z_score)
        
        # Adjust for direction (buy vs sell)
        if target_price < current_price:
            # For buy limit or sell stop
            probability = 1 - probability
            
        return probability
