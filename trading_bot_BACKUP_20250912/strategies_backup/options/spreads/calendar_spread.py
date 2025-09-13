#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calendar Spread Strategy Module

This module implements calendar spread option strategies that exploit
time decay differences between short-dated and longer-dated options.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Import utility modules
from trading_bot.utils.options_pricing import (
    black_scholes, calculate_greeks, calculate_iv_rank, implied_volatility
)
from trading_bot.utils.theta_decay import (
    calculate_calendar_spread_metrics, find_optimal_calendar_spread_dte,
    project_calendar_spread_performance, analyze_calendar_spread_risk_profile
)
from trading_bot.utils.option_chain_analysis import (
    filter_option_chain_by_liquidity, find_atm_strike,
    select_calendar_spread_strikes, apply_tiered_liquidity_filters
)
from trading_bot.utils.volatility_analysis import (
    analyze_volatility_surface, select_strategy_based_on_vol_environment,
    analyze_iv_term_structure
)

# Setup logging
logger = logging.getLogger(__name__)

class CalendarSpreadStrategy(StrategyOptimizable):
    """
    Calendar Spread Strategy designed to exploit the accelerated time decay 
    of front-month options versus longer-dated options.
    
    This strategy sells short-dated premium and hedges with a longer-dated 
    long-leg at the same strike, capturing net theta decay while keeping 
    directional exposure minimal.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Calendar Spread strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # 1. Strategy Philosophy parameters
            "use_same_strike": True,
            "net_theta_decay_target": 0.01,  # 1% per day theta decay target
            
            # 2. Underlying & Option Universe & Timeframe parameters
            "underlying_universe": ["SPY", "QQQ", "AAPL"],
            "short_leg_min_dte": 7,
            "short_leg_max_dte": 21,
            "long_leg_min_dte": 45,
            "long_leg_max_dte": 90,
            "roll_short_leg_dte": 7,
            "roll_long_leg_dte": 30,
            
            # 3. Selection Criteria for Underlying parameters
            "min_iv_rank": 30,
            "max_iv_rank": 60,
            "min_underlying_adv": 500000,
            "min_option_open_interest": 1000,
            "max_bid_ask_spread_pct": 0.15,
            
            # 4. Spread Construction parameters
            "strike_selection": "ATM",  # 'ATM', 'ITM', 'OTM'
            "strike_bias": 0,  # -1, 0, or 1 for directional bias
            "max_net_debit_pct": 1.0,  # 1% of equity per spread
            "leg_ratio": 1,  # 1:1 or consider 2:1 for more theta
            
            # 5. Expiration & Roll Timing parameters
            "roll_trigger_dte": 7,
            "early_roll_volatility_change_pct": 20,
            
            # 6. Entry Execution parameters
            "use_combo_orders": True,
            "max_slippage_pct": 5.0,  # Maximum allowable slippage in percent
            
            # 7. Exit & Adjustment Rules parameters
            "profit_target_pct": 50,  # Exit at 50% of max theoretical value
            "stop_loss_multiplier": 1.0,  # Stop loss at 1x initial debit
            "adjustment_threshold_pct": 10,  # Adjust if underlying moves 10%
            
            # 8. Position Sizing & Risk Controls parameters
            "position_size_pct": 1.0,  # 1% of equity per spread
            "max_concurrent_spreads": 5,
            "max_margin_usage_pct": 10.0,
            "max_sector_concentration": 1,  # Max 1 calendar per sector
            
            # Additional parameters
            "risk_free_rate": 0.04,  # 4% risk-free rate
            "liquidity_tier": "medium",  # 'high', 'medium', 'low'
            "option_type": "call",  # 'call' or 'put'
            "direction": "neutral"  # 'neutral', 'bullish', or 'bearish'
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Calendar Spread strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "short_leg_min_dte": [7, 10, 14],
            "short_leg_max_dte": [14, 21, 28],
            "long_leg_min_dte": [30, 45, 60],
            "long_leg_max_dte": [60, 75, 90],
            "roll_short_leg_dte": [5, 7, 10],
            "min_iv_rank": [20, 30, 40],
            "max_iv_rank": [50, 60, 70],
            "strike_bias": [-1, 0, 1],
            "profit_target_pct": [40, 50, 60, 75],
            "stop_loss_multiplier": [0.8, 1.0, 1.2],
            "position_size_pct": [0.5, 1.0, 1.5],
            "leg_ratio": [1, 2],
            "option_type": ["call", "put"],
            "direction": ["neutral", "bullish", "bearish"],
            "liquidity_tier": ["high", "medium", "low"]
        }
    
    def _extract_option_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Extract option data from DataFrame.
        
        Args:
            df: DataFrame with options data
            
        Returns:
            Dictionary with option chain data by expiration
        """
        try:
            # Check if options data is available
            if 'options' not in df.columns and not isinstance(df.get('options'), pd.DataFrame):
                logger.warning("No options data available")
                return {}
            
            options_data = df['options'] if 'options' in df.columns else df
            
            # Check if options data is organized by expiration
            if 'expiration' in options_data.columns:
                # Extract expirations and create separate DataFrames
                expirations = options_data['expiration'].unique()
                
                chains_by_expiration = {}
                for expiration in expirations:
                    exp_data = options_data[options_data['expiration'] == expiration]
                    chains_by_expiration[expiration] = exp_data
                
                return chains_by_expiration
            
            # Check if options data is already a dict organized by expiration
            elif isinstance(options_data, dict):
                return options_data
            
            else:
                logger.warning("Options data format not recognized")
                return {}
            
        except Exception as e:
            logger.error(f"Error extracting option data: {e}")
            return {}
    
    def _calculate_iv_rank(self, df: pd.DataFrame) -> float:
        """
        Calculate IV rank from historical volatility data.
        
        Args:
            df: DataFrame with historical volatility data
            
        Returns:
            IV Rank (0-100)
        """
        try:
            # Check if IV data is available
            if 'implied_volatility' not in df.columns:
                logger.warning("No implied volatility data available")
                return 50.0  # Default to middle value
            
            # Get current IV (most recent)
            current_iv = df['implied_volatility'].iloc[-1]
            
            # Get historical IV data (excluding most recent)
            historical_iv = df['implied_volatility'].iloc[:-1].tolist()
            
            # Calculate IV rank
            return calculate_iv_rank(current_iv, historical_iv)
            
        except Exception as e:
            logger.error(f"Error calculating IV rank: {e}")
            return 50.0
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate calendar spread indicators for all symbols.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV and options data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        for symbol, df in data.items():
            try:
                # Get current price
                if 'close' not in df.columns:
                    logger.warning(f"No price data available for {symbol}")
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Extract option chains by expiration
                option_chains = self._extract_option_data(df)
                
                if not option_chains:
                    logger.warning(f"No options data available for {symbol}")
                    continue
                
                # Calculate IV rank
                iv_rank = self._calculate_iv_rank(df)
                iv_rank_df = pd.DataFrame({"iv_rank": [iv_rank]})
                
                # Get parameters for filtering and analysis
                min_iv_rank = self.parameters.get("min_iv_rank", 30)
                max_iv_rank = self.parameters.get("max_iv_rank", 60)
                min_option_open_interest = self.parameters.get("min_option_open_interest", 1000)
                max_bid_ask_spread_pct = self.parameters.get("max_bid_ask_spread_pct", 0.15)
                liquidity_tier = self.parameters.get("liquidity_tier", "medium")
                option_type = self.parameters.get("option_type", "call")
                direction = self.parameters.get("direction", "neutral")
                risk_free_rate = self.parameters.get("risk_free_rate", 0.04)
                
                # Filter option chains for liquidity
                filtered_chains = {}
                for expiration, chain in option_chains.items():
                    filtered_chain = apply_tiered_liquidity_filters(chain, liquidity_tier)
                    if not filtered_chain.empty:
                        filtered_chains[expiration] = filtered_chain
                
                if not filtered_chains:
                    logger.warning(f"No liquid options available for {symbol}")
                    continue
                
                # Analyze volatility surface
                vol_surface = analyze_volatility_surface(filtered_chains, current_price)
                vol_surface_df = pd.DataFrame({"vol_surface": [vol_surface]})
                
                # Find ATM strike
                atm_strike = find_atm_strike(next(iter(filtered_chains.values())), current_price)
                
                if atm_strike is None:
                    logger.warning(f"Could not find ATM strike for {symbol}")
                    continue
                
                atm_strikes_df = pd.DataFrame({"atm_strike": [atm_strike]})
                
                # Select optimal strikes based on parameters
                strike_selection = self.parameters.get("strike_selection", "ATM")
                strike_recommendations = select_calendar_spread_strikes(
                    next(iter(filtered_chains.values())),
                    current_price,
                    iv_rank,
                    direction,
                    strike_selection
                )
                
                strike_recommendations_df = pd.DataFrame({"strike_recommendations": [strike_recommendations]})
                
                # Get DTEs from expirations
                expirations_days = {}
                for expiration in filtered_chains.keys():
                    if isinstance(expiration, str):
                        try:
                            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                            dte = (exp_date - datetime.now().date()).days
                            expirations_days[expiration] = max(0, dte)
                        except:
                            # If expiration is already processed or has DTE info
                            expirations_days[expiration] = 30  # Default value
                    else:
                        expirations_days[expiration] = 30  # Default value
                
                # Find optimal DTE combinations
                short_leg_min_dte = self.parameters.get("short_leg_min_dte", 7)
                short_leg_max_dte = self.parameters.get("short_leg_max_dte", 21)
                long_leg_min_dte = self.parameters.get("long_leg_min_dte", 45)
                long_leg_max_dte = self.parameters.get("long_leg_max_dte", 90)
                
                # Use average implied volatility from ATM options
                avg_iv = 0.2  # Default value
                if 'summary' in vol_surface and 'avg_iv' in vol_surface['summary']:
                    avg_iv = vol_surface['summary']['avg_iv']
                
                # Find optimal calendar spread parameters
                optimal_spread = find_optimal_calendar_spread_dte(
                    current_price,
                    atm_strike,
                    risk_free_rate,
                    avg_iv,
                    short_leg_min_dte,
                    short_leg_max_dte,
                    long_leg_min_dte,
                    long_leg_max_dte,
                    option_type
                )
                
                optimal_spread_df = pd.DataFrame({"optimal_spread": [optimal_spread]})
                
                # Calculate theoretical calendar spread metrics
                short_leg_dte = optimal_spread.get('short_leg_dte', short_leg_min_dte)
                long_leg_dte = optimal_spread.get('long_leg_dte', long_leg_min_dte)
                
                calendar_metrics = calculate_calendar_spread_metrics(
                    current_price,
                    atm_strike,
                    risk_free_rate,
                    avg_iv,
                    short_leg_dte,
                    long_leg_dte,
                    option_type
                )
                
                calendar_metrics_df = pd.DataFrame({"calendar_metrics": [calendar_metrics]})
                
                # Project calendar spread performance
                performance_projection = project_calendar_spread_performance(
                    current_price,
                    atm_strike,
                    risk_free_rate,
                    avg_iv,
                    short_leg_dte,
                    long_leg_dte,
                    option_type
                )
                
                performance_projection_df = pd.DataFrame({"performance_projection": [performance_projection]})
                
                # Store indicators
                indicators[symbol] = {
                    "iv_rank": iv_rank_df,
                    "atm_strikes": atm_strikes_df,
                    "vol_surface": vol_surface_df,
                    "strike_recommendations": strike_recommendations_df,
                    "optimal_spread": optimal_spread_df,
                    "calendar_metrics": calendar_metrics_df,
                    "performance_projection": performance_projection_df
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate calendar spread trading signals.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV and options data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        # Extract parameters
        min_iv_rank = self.parameters.get("min_iv_rank", 30)
        max_iv_rank = self.parameters.get("max_iv_rank", 60)
        strike_selection = self.parameters.get("strike_selection", "ATM")
        strike_bias = self.parameters.get("strike_bias", 0)
        option_type = self.parameters.get("option_type", "call")
        direction = self.parameters.get("direction", "neutral")
        profit_target_pct = self.parameters.get("profit_target_pct", 50)
        net_theta_decay_target = self.parameters.get("net_theta_decay_target", 0.01)
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest data
                latest_data = data[symbol].iloc[-1]
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                current_price = latest_data['close']
                
                # 1. Check IV rank criteria
                latest_iv_rank = symbol_indicators["iv_rank"].iloc[0]["iv_rank"]
                if latest_iv_rank < min_iv_rank or latest_iv_rank > max_iv_rank:
                    logger.info(f"{symbol}: IV rank {latest_iv_rank:.1f} outside target range ({min_iv_rank}-{max_iv_rank})")
                    continue
                
                # 2. Get calendar spread metrics
                calendar_metrics = symbol_indicators["calendar_metrics"].iloc[0]["calendar_metrics"]
                theta_decay_advantage = calendar_metrics.get("theta_decay_advantage", 0)
                
                # Check if theta decay advantage meets target
                if theta_decay_advantage < net_theta_decay_target * 100:
                    logger.info(f"{symbol}: Theta decay advantage {theta_decay_advantage:.2f}% below target {net_theta_decay_target*100:.2f}%")
                    continue
                
                # 3. Get recommended strikes
                strike_recommendations = symbol_indicators["strike_recommendations"].iloc[0]["strike_recommendations"]
                
                if not strike_recommendations:
                    logger.warning(f"{symbol}: No strike recommendations available")
                    continue
                
                # Select strike based on recommendations and bias
                selected_strike = strike_recommendations[0]['strike']  # Default to first recommendation
                
                # 4. Calculate signal confidence based on metrics
                # Base confidence on theta decay advantage, IV rank, and volatility environment
                base_confidence = 0.5
                
                # Adjust for theta decay (0-0.3)
                theta_factor = min(0.3, theta_decay_advantage / 100)
                
                # Adjust for IV rank optimality (0-0.2)
                iv_rank_optimality = 1.0 - 2.0 * abs((latest_iv_rank - 45) / 45)  # 45% IV rank is optimal
                iv_factor = max(0, min(0.2, iv_factor * 0.2))
                
                # Adjust for expected return (0-0.2)
                max_potential_return = calendar_metrics.get("max_potential_return", 0)
                return_factor = min(0.2, max_potential_return / 100)
                
                # Adjust for liquidity and execution risk (0-0.1)
                liquidity_factor = 0.05  # Default middle value
                
                # Calculate final confidence
                confidence = base_confidence + theta_factor + iv_factor + return_factor + liquidity_factor
                confidence = min(0.95, max(0.3, confidence))  # Ensure between 0.3 and 0.95
                
                # 5. Generate signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,  # For calendar spreads, always BUY the spread
                    price=current_price,
                    timestamp=latest_timestamp,
                    confidence=confidence,
                    metadata={
                        "strategy_type": "calendar_spread",
                        "option_type": option_type,
                        "direction": direction,
                        "short_leg_dte": calendar_metrics.get("short_leg_dte", 0),
                        "long_leg_dte": calendar_metrics.get("long_leg_dte", 0),
                        "strike": selected_strike,
                        "iv_rank": latest_iv_rank,
                        "net_debit": calendar_metrics.get("net_debit", 0),
                        "net_theta": calendar_metrics.get("net_theta", 0),
                        "theta_decay_advantage": theta_decay_advantage,
                        "max_potential_return": max_potential_return,
                        "profit_target": calendar_metrics.get("net_debit", 0) * (1 + profit_target_pct / 100)
                    }
                )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals


# Section implementation blueprint follows the 10 sections from the user request:

# 1. Strategy Philosophy
"""
TODO: Implement a function to calculate the theta decay advantage of the front-month option
relative to the longer-dated option to find optimal calendar spread opportunities.

Expected implementation steps:
- Calculate theoretical theta for both legs
- Determine the theta decay differential
- Identify where front-month has steepest decay relative to back-month
"""

# 2. Underlying & Option Universe & Timeframe
"""
TODO: Build a universe filter that identifies high-liquidity underlyings with appropriate
option chains for calendar spreads.

Implementation components needed:
- Filter for highly liquid underlyings (SPY, QQQ, AAPL)
- Filter options chains for appropriate DTE ranges
- Verify option volume and open interest meet minimums
"""

# 3. Selection Criteria for Underlying
"""
TODO: Calculate IV_rank for underlyings and filter based on volatility regime.

Required functions:
- IV_rank calculation module
- ADV (average daily volume) filter
- Option liquidity screener
- Bid-ask spread calculator
"""

# 4. Spread Construction
"""
TODO: Implement a spread construction algorithm that selects strikes and calculates
appropriate net debit amount.

Core logic to develop:
- ATM strike identification algorithm
- Directional bias adjuster (±1 strike)
- Net debit calculator
- Leg ratio manager
"""

# 5. Expiration & Roll Timing
"""
TODO: Create a roll timing module that identifies optimal entry/roll timing.

Key components:
- DTE tracker for front/back month options
- Time-based roll trigger
- Volatility-based early roll detector
"""

# 6. Entry Execution
"""
TODO: Build execution logic that handles combo orders and implements contingency
for legging in when needed.

Implementation needs:
- Combo order builder
- Theoretical price calculator
- Slippage monitoring
- Fallback leg-in sequencer
"""

# 7. Exit & Adjustment Rules
"""
TODO: Develop exit rules engine and adjustment strategy for managing positions.

Critical functions:
- Profit-take calculator based on percent of max theoretical value
- Time-based exit trigger
- Directional adjustment handler 
- Stop-loss monitor
"""

# 8. Position Sizing & Risk Controls
"""
TODO: Implement position sizing algorithms and risk control guardrails.

Key elements:
- Position size calculator based on account equity
- Concurrent calendar counter
- Margin requirement checker
- Sector concentration limiter
"""

# 9. Backtesting & Performance Metrics
"""
TODO: Build backtesting framework for calendar spreads with custom metrics.

Specialized metrics needed:
- Theta capture ratio calculator
- Win rate and P&L tracker
- Drawdown monitor for calendar positions
- Roll cost analyzer
- ROI calculator for calendar cycles
"""

# 10. Continuous Optimization
"""
TODO: Create a feedback loop for continuous strategy improvement.

Implementation requirements:
- Monthly performance analyzer
- IV_rank adaptation module
- Strike bias optimizer
- Optional ML module for outcome prediction
""" 