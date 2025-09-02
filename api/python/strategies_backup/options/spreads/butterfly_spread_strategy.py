#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Butterfly Spread Strategy Module

This module implements a butterfly spread options strategy for capturing premium
when the underlying stays within a predicted range.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy.stats import norm

from trading_bot.strategies.strategy_template import (
    StrategyTemplate, 
    StrategyOptimizable,
    Signal, 
    SignalType,
    TimeFrame,
    MarketRegime
)

# Setup logging
logger = logging.getLogger(__name__)

class ButterflySpreadStrategy(StrategyOptimizable):
    """
    Butterfly Spread Strategy designed to profit from range-bound markets.
    
    This strategy creates a four-legged options position by:
    1. Buying one option contract at a lower strike price
    2. Selling two option contracts at a middle strike price
    3. Buying one option contract at a higher strike price
    
    All options have the same expiration date and are of the same type (calls or puts).
    
    The strategy provides:
    - Defined risk (limited to initial debit paid)
    - Defined reward (maximum profit at center strike at expiration)
    - Low cost of entry relative to potential return
    - Delta-neutral to slightly directional positioning
    - Maximum profit when underlying expires exactly at the middle strike
    
    Key Characteristics:
    - Risk/Reward Profile: Limited risk, limited reward
    - Optimal Market View: Neutral, expecting price to settle at middle strike
    - IV Environment: Works in various IV environments, but entry is more favorable in higher IV
    - Time Decay: Benefits from time decay when price is near middle strike
    - Greek Sensitivities:
        - Delta: Near-zero when centered (delta-neutral at initiation)
        - Gamma: Highest near middle strike, creating convex return profile
        - Theta: Positive when near middle strike, negative when far from middle
        - Vega: Typically negative, benefits from volatility contraction
        
    Strategic Applications:
    - Range trading: Capitalizing on price consolidation periods
    - Volatility plays: Profiting from expected volatility contraction
    - Event trading: Positioning for expected price settlement after events
    - Low-cost directional alternatives: Affordable way to express price target views
    
    Implementation Variants:
    - Long Call Butterfly: Using calls for the structure (most common)
    - Long Put Butterfly: Using puts for the structure (functionally equivalent)
    - Iron Butterfly: Combination of put credit spread and call credit spread
    - Broken Wing Butterfly: Asymmetric structure with unequal wing distances
    - Calendar Butterfly: Using different expirations for multi-dimensional approach
    
    This implementation focuses on the standard long butterfly spread, with options
    for dynamic strike selection, position management, and optimal entry/exit timing.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Butterfly Spread strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the strategy blueprint
        default_params = {
            # Section 2 & 3: Underlying & Selection
            "min_underlying_adv": 1000000,  # 1M minimum average daily volume
            "min_option_open_interest": 500,  # Minimum open interest for option legs
            "max_bid_ask_spread_pct": 0.10,  # Maximum bid-ask spread as percentage
            "iv_rank_min": 20,  # Minimum IV rank threshold (%)
            "iv_rank_max": 50,  # Maximum IV rank threshold (%)
            "range_days_lookback": 15,  # Days to check for range-bound behavior
            
            # Section 4: Spread Construction
            "center_strike_delta": 0.50,  # Target delta for center strike (ATM)
            "center_strike_offset": 0,  # Offset from ATM in # of strikes
            "inner_wing_width": 1,  # Width in # of strikes from center to inner wings
            "outer_wing_width": 2,  # Width in # of strikes from center to outer wings
            "target_net_debit": 0.05,  # Target net debit as % of underlying price
            
            # Section 5: Expiration Selection
            "min_days_to_expiration": 25,  # Minimum DTE for entry
            "max_days_to_expiration": 45,  # Maximum DTE for entry
            "exit_dte_threshold": 8,  # Exit when DTE falls below this value
            
            # Section 7: Exit & Management Rules
            "profit_take_pct": 65,  # Take profit at % of max potential gain
            "stop_loss_multiplier": 1.5,  # Stop loss at multiple of max potential loss
            "management_threshold_delta": 0.10,  # Delta threshold for adjustment
            
            # Section 8: Position Sizing & Risk
            "risk_pct_per_trade": 1.0,  # Maximum risk per trade as % of account
            "max_concurrent_positions": 4,  # Maximum number of concurrent butterflies
            "max_margin_pct": 5.0,  # Maximum margin requirement as % of account
            
            # Section 10: Optimization Parameters
            "iv_adjustment_threshold": 20,  # IV rank change threshold for wing adjustment
            "dynamic_recentering": True,  # Whether to allow dynamic re-centering
            "ml_model_enabled": False,  # Enable ML overlay for strike selection
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Butterfly Spread strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            # Selection criteria optimization
            "iv_rank_min": [15, 20, 25, 30],
            "iv_rank_max": [40, 50, 60, 70],
            "range_days_lookback": [10, 15, 20, 25],
            
            # Spread construction optimization
            "center_strike_delta": [0.45, 0.50, 0.55],
            "center_strike_offset": [-1, 0, 1],
            "inner_wing_width": [1, 2, 3],
            "outer_wing_width": [1, 2, 3, 4],
            "target_net_debit": [0.01, 0.05, 0.10, -0.01],  # negative = credit
            
            # Expiration selection optimization
            "min_days_to_expiration": [20, 25, 30, 35],
            "max_days_to_expiration": [40, 45, 50, 55],
            "exit_dte_threshold": [5, 8, 10, 12],
            
            # Exit rules optimization
            "profit_take_pct": [50, 65, 75, 85],
            "stop_loss_multiplier": [1.3, 1.5, 1.8, 2.0],
            
            # Risk controls optimization
            "risk_pct_per_trade": [0.5, 1.0, 1.5, 2.0],
            "max_concurrent_positions": [3, 4, 5, 6],
        }
    
    def _check_underlying_eligibility(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        Evaluate if an underlying security is suitable for butterfly spread deployment.
        
        This method performs a comprehensive analysis of a security to determine if it meets
        the criteria for a butterfly spread strategy. It evaluates liquidity metrics,
        volatility characteristics, price patterns, and technical factors to identify
        candidates with high probability of successful butterfly spreads.
        
        Parameters:
            symbol (str): Ticker symbol to evaluate
            data (pd.DataFrame): Historical price and volume data with columns:
                - open: Open prices
                - high: High prices
                - low: Low prices
                - close: Close prices
                - volume: Trading volumes
                
        Returns:
            bool: True if the security meets all criteria, False otherwise
            
        Notes:
            Selection criteria include:
            
            1. Liquidity requirements:
               - Average daily volume > minimum threshold (default 1M)
               - Stable and sufficient trading volume to support option liquidity
               - Lower liquidity securities require wider butterfly wings
               
            2. Range-bound behavior:
               - Recent price action shows consolidation or channel trading
               - Price range typically 3-10% of current price over the lookback period
               - No strong directional trend that would compromise neutral positioning
               
            3. Volatility profile:
               - Implied volatility within target range (default 20-50% IV rank)
               - Stable or contracting volatility preferred over expanding
               - Historical volatility patterns supporting mean-reverting behavior
               
            4. Support/resistance identification:
               - Clear technical levels that align with potential wing strikes
               - Evidence of price respect for these levels in recent history
               - Volume profile supporting key price levels
               
            5. Options chain characteristics (when available):
               - Sufficient open interest at potential wing strikes
               - Reasonable bid-ask spreads (< 10% of option price)
               - Strike spacing appropriate for butterfly construction
               
            The ideal butterfly candidate shows range-bound behavior with clear support/resistance
            levels, moderate implied volatility, and sufficient liquidity for efficient entry/exit.
        """
        # Check data validity and minimum required history
        lookback = self.parameters["range_days_lookback"]
        if len(data) < lookback + 10:  # Need extra days for calculations
            logger.warning(f"Insufficient data for {symbol} to evaluate eligibility")
            return False
            
        # Calculate average daily volume
        min_adv = self.parameters["min_underlying_adv"]
        recent_volume = data['volume'].tail(20).mean()
        
        if recent_volume < min_adv:
            logger.debug(f"{symbol} ADV of {int(recent_volume)} below minimum {int(min_adv)}")
            return False
            
        # Analyze recent price range to check for range-bound behavior
        recent_data = data.tail(lookback)
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        current_price = data['close'].iloc[-1]
        
        # Calculate price range as percentage of current price
        high_low_range_pct = (recent_high - recent_low) / current_price
        
        # Check if price range is within reasonable bounds for butterfly strategy
        # Too narrow: insufficient premium, too wide: excessive risk
        min_range_pct = 0.03  # 3% minimum range
        max_range_pct = 0.10  # 10% maximum range
        
        range_bound = min_range_pct <= high_low_range_pct <= max_range_pct
        
        if not range_bound:
            logger.debug(f"{symbol} price range of {high_low_range_pct:.1%} outside target range "
                        f"({min_range_pct:.1%}-{max_range_pct:.1%})")
            return False
            
        # Check for directional bias - avoid strong trends
        # Calculate short and long-term moving averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Measure recent trend strength
        latest = data.iloc[-1]
        trend_strength = abs(latest['sma_20'] - latest['sma_50']) / latest['close']
        
        if trend_strength > 0.05:  # 5% difference between SMAs indicates strong trend
            logger.debug(f"{symbol} showing strong trend (strength: {trend_strength:.1%}), "
                        f"not ideal for neutral butterfly")
            return False
            
        # Check for stable volume pattern (avoid volume spikes)
        volume_stability = data['volume'].tail(10).std() / data['volume'].tail(10).mean()
        if volume_stability > 0.5:  # High volume volatility
            logger.debug(f"{symbol} has unstable volume pattern (stability: {volume_stability:.2f})")
            return False
            
        # Check for IV rank/percentile if available
        # This would typically come from option data which is not included in this implementation
        # Placeholder for IV check (would be replaced with actual IV data)
        iv_rank_min = self.parameters["iv_rank_min"]
        iv_rank_max = self.parameters["iv_rank_max"]
        
        # TODO: Implement actual IV rank calculation from option chain data
        # For now, we'll assume IV is acceptable if other criteria are met
        logger.info(f"{symbol} meets butterfly spread eligibility criteria "
                   f"(Range: {high_low_range_pct:.1%}, ADV: {int(recent_volume)})")
        
        return True
    
    def _find_option_strikes(self, symbol: str, current_price: float, option_chain: Any) -> Dict[str, float]:
        """
        Identify the optimal strikes for constructing a butterfly spread.
        
        This method systematically selects the three strike prices needed for a butterfly spread:
        the center (body) strike and the two outer (wing) strikes. It uses a combination of
        delta-based targeting, technical level analysis, and spread width optimization to find
        strikes that maximize the probability of success.
        
        Parameters:
            symbol (str): Underlying symbol to trade
            current_price (float): Current price of the underlying asset
            option_chain (Any): Option chain data structure containing:
                - available strikes
                - option greeks (particularly delta)
                - pricing data (bid/ask)
                - volume and open interest
                
        Returns:
            Dict[str, float]: Dictionary with selected strikes:
                - center_strike: The middle strike where maximum profit occurs
                - lower_inner: Inner lower strike (for put wing or call body)
                - upper_inner: Inner upper strike (for call wing or put body)
                - lower_outer: Outer lower strike
                - upper_outer: Outer upper strike
                
        Notes:
            Strike selection methodology:
            
            1. Center strike selection approaches:
               - Delta-targeted: Using the delta parameter to find ATM or slightly OTM strikes
               - Price-based: Selecting strikes closest to the current price or price target
               - Technical-based: Using support/resistance levels from technical analysis
               
            2. Wing width determination factors:
               - Implied volatility: Higher IV generally requires wider wings
               - Expected price range: Based on historical movement patterns
               - Risk/reward objectives: Narrower wings for higher probability, wider for higher reward
               - Available strikes: Options chain structure constraints
               
            3. Implementation considerations:
               - Strike spacing must account for the underlying's price and volatility
               - Liquidity at wing strikes is critical for efficient entry/exit
               - Asymmetric wings can be used to create directional bias
               - Support/resistance levels help identify natural price boundaries
               
            The ideal butterfly has its center strike at the expected price target at expiration,
            with wings wide enough to balance probability of success with potential return.
        """
        # Initialize strike selection variables
        center_delta_target = self.parameters["center_strike_delta"]
        offset = self.parameters["center_strike_offset"]
        inner_width = self.parameters["inner_wing_width"]
        outer_width = self.parameters["outer_wing_width"]
        
        # Step 1: Find the center strike (body)
        # In a real implementation, we would use the option chain data
        # to find strikes with delta closest to our target
        
        # First try delta-based targeting if greek data is available
        center_strike = None
        
        if option_chain and hasattr(option_chain, 'get_strikes_by_delta'):
            try:
                # Attempt to find center strike by delta
                center_strike = option_chain.get_strikes_by_delta(
                    symbol, center_delta_target, tolerance=0.05)
                logger.debug(f"Selected center strike {center_strike} by delta target {center_delta_target}")
            except Exception as e:
                logger.warning(f"Error in delta-based strike selection: {e}")
        
        # If delta-based selection fails or isn't available, use price-based approach
        if center_strike is None:
            # Round to nearest available strike
            if option_chain and hasattr(option_chain, 'get_nearest_strike'):
                center_strike = option_chain.get_nearest_strike(current_price)
            else:
                # Fallback to simple rounding if no option chain data is available
                # Find the appropriate strike price increment based on stock price
                if current_price < 25:
                    increment = 0.5
                elif current_price < 100:
                    increment = 1.0
                elif current_price < 200:
                    increment = 2.5
                else:
                    increment = 5.0
                    
                # Round to nearest increment
                center_strike = round(current_price / increment) * increment
            
            logger.debug(f"Selected center strike {center_strike} based on current price {current_price}")
        
        # Apply offset if specified (e.g., slightly bullish or bearish butterfly)
        if offset != 0 and option_chain and hasattr(option_chain, 'get_strike_by_offset'):
            center_strike = option_chain.get_strike_by_offset(center_strike, offset)
            logger.debug(f"Applied offset of {offset} strikes, adjusted center to {center_strike}")
        
        # Step 2: Determine wing strikes based on width parameters
        # In a real implementation, we would use option chain data to get actual available strikes
        
        # Calculate target wing strikes
        if option_chain and hasattr(option_chain, 'get_available_strikes'):
            available_strikes = option_chain.get_available_strikes(symbol)
            
            # Find actual strikes based on available options
            all_strikes = sorted(available_strikes)
            center_idx = all_strikes.index(center_strike) if center_strike in all_strikes else None
            
            if center_idx is not None:
                # Find inner wing strikes
                lower_inner_idx = max(0, center_idx - inner_width)
                upper_inner_idx = min(len(all_strikes) - 1, center_idx + inner_width)
                
                # Find outer wing strikes
                lower_outer_idx = max(0, lower_inner_idx - outer_width)
                upper_outer_idx = min(len(all_strikes) - 1, upper_inner_idx + outer_width)
                
                lower_inner = all_strikes[lower_inner_idx]
                upper_inner = all_strikes[upper_inner_idx]
                lower_outer = all_strikes[lower_outer_idx]
                upper_outer = all_strikes[upper_outer_idx]
            else:
                # Fallback if center strike not found
                lower_inner = center_strike - (inner_width * increment)
                upper_inner = center_strike + (inner_width * increment)
                lower_outer = lower_inner - (outer_width * increment)
                upper_outer = upper_inner + (outer_width * increment)
        else:
            # Fallback when no option chain data is available
            # Use simple math to calculate theoretical strikes
            strike_increment = increment if 'increment' in locals() else (
                1.0 if current_price < 100 else 2.5 if current_price < 200 else 5.0)
            
            lower_inner = center_strike - (inner_width * strike_increment)
            upper_inner = center_strike + (inner_width * strike_increment)
            lower_outer = lower_inner - (outer_width * strike_increment)
            upper_outer = upper_inner + (outer_width * strike_increment)
        
        # Step 3: Validate the selected strikes
        # Ensure wing width is appropriate relative to underlying price
        wing_width_pct = (upper_inner - lower_inner) / current_price
        
        logger.info(f"Selected butterfly strikes for {symbol} at ${current_price:.2f}: "
                   f"Center: {center_strike}, Wings: {lower_outer}/{lower_inner} - "
                   f"{upper_inner}/{upper_outer} (Width: {wing_width_pct:.1%})")
        
        return {
            "center_strike": center_strike,
            "lower_inner": lower_inner,
            "upper_inner": upper_inner,
            "lower_outer": lower_outer,
            "upper_outer": upper_outer,
            "wing_width_pct": wing_width_pct
        }
    
    def _select_expiration(self, option_chain: Any) -> datetime:
        """
        Select the optimal expiration date for a butterfly spread.
        
        This method determines the most appropriate expiration date for the butterfly spread
        by analyzing available expirations, market conditions, and strategy parameters.
        Expiration selection is critical for butterfly spreads as it directly impacts time
        decay characteristics and the probability of the underlying reaching the target price.
        
        Parameters:
            option_chain (Any): Option chain data structure containing:
                - Available expiration dates
                - Open interest across expirations
                - Implied volatility term structure
                - Option pricing across different expirations
                
        Returns:
            datetime: Selected expiration date that best meets strategy criteria
            
        Notes:
            Expiration selection considerations:
            
            1. Time horizon factors:
               - Optimal DTE range: Typically 25-45 days balances time decay and duration
               - Gamma acceleration: Significant in last 30 days, especially last 2 weeks
               - Theta decay curve: Maximizes in final 30-45 days before expiration
               
            2. Market calendar considerations:
               - Earnings dates: Avoid expiration near earnings announcements
               - Economic events: Consider impact of scheduled economic releases
               - Ex-dividend dates: Avoid dates when early assignment risk increases
               - Triple/quadruple witching: Consider higher volatility around these dates
               
            3. Volatility term structure implications:
               - Contango: Later expirations may offer better premiums 
               - Backwardation: Shorter expirations may be more favorable
               - Volatility skew: Affects optimal wing width across expirations
               
            4. Liquidity considerations:
               - Open interest: Higher OI indicates better liquidity for entry/exit
               - Volume patterns: Active trading volume supports efficient execution
               - Standard vs. non-standard expirations: Monthly typically more liquid
               
            The ideal expiration provides sufficient time for the butterfly to develop while
            capitalizing on accelerating time decay as expiration approaches, with adequate
            liquidity for position entry and management.
        """
        # Get min and max DTE parameters
        min_dte = self.parameters["min_days_to_expiration"]
        max_dte = self.parameters["max_days_to_expiration"]
        target_dte = (min_dte + max_dte) // 2  # Target mid-point of range
        
        today = datetime.now().date()
        selected_expiration = None
        closest_diff = float('inf')
        
        # Step 1: Extract available expirations from option chain
        available_expirations = []
        
        # If we have real option chain data
        if option_chain and hasattr(option_chain, 'get_expirations'):
            try:
                available_expirations = option_chain.get_expirations()
            except Exception as e:
                logger.warning(f"Error retrieving expirations from option chain: {e}")
        
        # No real expirations available, generate theoretical ones
        if not available_expirations:
            # Create synthetic expirations for testing
            # In practice, these would come from actual market data
            
            # Generate weekly expirations for next 60 days
            current_date = today
            while (current_date - today).days <= 60:
                # If Friday, add as potential expiration
                if current_date.weekday() == 4:  # Friday
                    available_expirations.append(current_date)
                current_date += timedelta(days=1)
            
            # Add monthly expirations (3rd Friday of each month)
            for month_offset in range(1, 7):  # 6 months out
                current_month = today.month + month_offset
                year = today.year + (current_month - 1) // 12
                month = ((current_month - 1) % 12) + 1
                
                # Find the third Friday
                first_day = datetime(year, month, 1).date()
                day_of_week = first_day.weekday()
                third_friday = first_day + timedelta(days=((4 - day_of_week) % 7) + 14)
                
                available_expirations.append(third_friday)
        
        # Step 2: Filter expirations by DTE criteria
        valid_expirations = []
        
        for exp_date in available_expirations:
            if isinstance(exp_date, str):
                try:
                    exp_date = datetime.strptime(exp_date, '%Y-%m-%d').date()
                except ValueError:
                    logger.warning(f"Invalid expiration date format: {exp_date}")
                    continue
                    
            dte = (exp_date - today).days
            
            if min_dte <= dte <= max_dte:
                valid_expirations.append((exp_date, dte))
        
        if not valid_expirations:
            logger.warning(f"No expirations found in target DTE range {min_dte}-{max_dte}")
            # Return fallback expiration if no valid ones found
            return datetime.now() + timedelta(days=target_dte)
        
        # Step 3: Select best expiration based on multiple criteria
        
        # Step 3a: First try to find expiration closest to target DTE
        for exp_date, dte in valid_expirations:
            diff = abs(dte - target_dte)
            if diff < closest_diff:
                closest_diff = diff
                selected_expiration = exp_date
        
        # Step 3b: If option chain has liquidity data, consider that too
        if option_chain and hasattr(option_chain, 'get_expiration_liquidity'):
            try:
                # Get liquidity scores for candidate expirations
                liquidity_scores = {}
                for exp_date, dte in valid_expirations:
                    liquidity = option_chain.get_expiration_liquidity(exp_date)
                    dte_score = 1.0 - (abs(dte - target_dte) / (max_dte - min_dte))
                    combined_score = (0.7 * dte_score) + (0.3 * liquidity)
                    liquidity_scores[exp_date] = combined_score
                
                # Select expiration with highest combined score
                if liquidity_scores:
                    selected_expiration = max(liquidity_scores.items(), key=lambda x: x[1])[0]
            except Exception as e:
                logger.warning(f"Error evaluating expiration liquidity: {e}")
        
        # Step 3c: Avoid earnings dates if available
        if option_chain and hasattr(option_chain, 'get_earnings_date'):
            try:
                earnings_date = option_chain.get_earnings_date()
                
                # If we have an earnings date, avoid expirations within 3 days
                if earnings_date:
                    for exp_date, dte in valid_expirations:
                        days_to_earnings = abs((exp_date - earnings_date).days)
                        if days_to_earnings <= 3:
                            logger.debug(f"Avoiding expiration {exp_date} near earnings on {earnings_date}")
                            # If this was our selected expiration, try to find another one
                            if exp_date == selected_expiration:
                                # Find next best expiration that's not near earnings
                                for alt_exp, alt_dte in valid_expirations:
                                    alt_days_to_earnings = abs((alt_exp - earnings_date).days)
                                    if alt_days_to_earnings > 3:
                                        selected_expiration = alt_exp
                                        break
            except Exception as e:
                logger.warning(f"Error checking earnings dates: {e}")
        
        # Ensure we have a datetime object
        if selected_expiration and not isinstance(selected_expiration, datetime):
            selected_expiration = datetime.combine(selected_expiration, datetime.min.time())
            
        if not selected_expiration:
            # Final fallback
            selected_expiration = datetime.now() + timedelta(days=target_dte)
            
        logger.info(f"Selected expiration: {selected_expiration.strftime('%Y-%m-%d')}, "
                   f"DTE: {(selected_expiration.date() - today).days}")
        
        return selected_expiration
    
    def _calculate_theoretical_value(self, strikes: Dict[str, float], 
                                    current_price: float, 
                                    days_to_expiration: int,
                                    implied_volatility: float) -> Dict[str, float]:
        """
        Calculate the theoretical value and risk metrics for a butterfly spread.
        
        This method implements option pricing models to evaluate the butterfly spread's
        theoretical value, risk parameters, and profit potential. It incorporates the
        Black-Scholes-Merton model and other pricing techniques to analyze the 
        characteristics of the spread across multiple scenarios.
        
        Parameters:
            strikes (Dict[str, float]): Dictionary with strike prices for all legs
                - center_strike: Middle strike where max profit occurs
                - lower_inner/upper_inner: Inner wing strikes 
                - lower_outer/upper_outer: Outer wing strikes
            current_price (float): Current price of the underlying asset
            days_to_expiration (int): Days remaining until expiration
            implied_volatility (float): Implied volatility as a decimal (0.20 = 20%)
            
        Returns:
            Dict[str, float]: Comprehensive theoretical value analysis including:
                - theoretical_value: Current value of the butterfly spread
                - max_profit: Maximum potential profit at expiration
                - max_loss: Maximum potential loss at expiration
                - breakeven_lower: Lower breakeven price at expiration
                - breakeven_upper: Upper breakeven price at expiration
                - probability_of_profit: Estimated probability of any profit
                - probability_of_max_profit: Estimated probability of achieving max profit
                - expected_value: Probability-weighted expected return
                - greeks: Delta, gamma, theta, and vega for the overall position
                
        Notes:
            Pricing methodology:
            
            1. Theoretical valuation techniques:
               - Black-Scholes-Merton model for European-style options
               - Binomial/trinomial trees for American-style options
               - Monte Carlo simulation for complex scenarios
               
            2. Key butterfly spread characteristics:
               - Value curve is tent-shaped, peaking at center strike
               - Value at initiation is generally net debit, sometimes credit
               - Max value at expiration equals distance between strikes when price = center strike
               - Min value at expiration is zero when price is beyond outer strikes
               
            3. Risk/reward profile analysis:
               - Probability distribution based on log-normal price model
               - Standard deviation used to estimate range of outcomes
               - Breakeven points where P&L equals zero at expiration
               - Maximum profit achieved when price exactly equals center strike
               
            4. Greek analysis for position management:
               - Delta: Changes substantially as underlying moves toward/away from center
               - Gamma: Highest near center strike, creating convex P&L profile
               - Theta: Positive near center strike, negative away from center
               - Vega: Typically negative, gains value from volatility contraction
               
            The butterfly spread's theoretical analysis provides a foundation for position
            sizing, risk management, and adjustment decisions throughout the trade lifecycle.
        """
        # Extract strikes
        center = strikes["center_strike"]
        lower_inner = strikes["lower_inner"]
        upper_inner = strikes["upper_inner"]
        lower_outer = strikes.get("lower_outer", lower_inner)  # Optional
        upper_outer = strikes.get("upper_outer", upper_inner)  # Optional
        
        # Calculate basic wing width (for standard butterfly)
        wing_width = (upper_inner - lower_inner) / 2
        
        # Step 1: Calculate max profit and loss potential
        # For standard butterfly, max profit occurs at center strike
        # and equals the wing width minus the initial debit
        
        # Determine if this is a traditional butterfly or a condor-like structure
        is_standard_butterfly = abs(upper_inner - center) == abs(center - lower_inner)
        
        # Calculate max values
        if is_standard_butterfly:
            # Standard butterfly with equal wing distances
            theoretical_max_profit = wing_width
        else:
            # Irregular butterfly or condor structure
            theoretical_max_profit = min(upper_inner - center, center - lower_inner)
        
        # Apply our target net debit from parameters
        target_debit = self.parameters["target_net_debit"] * current_price
        max_profit = theoretical_max_profit - target_debit
        max_loss = target_debit  # Max loss is the initial debit paid
        
        # Step 2: Calculate breakeven points
        # Breakeven points are where P&L is zero at expiration
        breakeven_lower = center - max_profit
        breakeven_upper = center + max_profit
        
        # Step 3: Calculate probability metrics using log-normal distribution
        # Standard deviation of log returns based on implied volatility
        annual_std_dev = implied_volatility
        time_to_exp_years = days_to_expiration / 365
        std_dev = annual_std_dev * np.sqrt(time_to_exp_years)
        
        # Log-normal distribution parameters
        mu = np.log(current_price) + (0 - annual_std_dev**2/2) * time_to_exp_years
        sigma = std_dev
        
        # Calculate probability of being between breakeven points at expiration
        prob_above_lower = 1 - norm.cdf((np.log(breakeven_lower) - mu) / sigma)
        prob_below_upper = norm.cdf((np.log(breakeven_upper) - mu) / sigma)
        probability_of_profit = prob_above_lower * prob_below_upper
        
        # Calculate probability of being very near center strike (max profit)
        # Define "very near" as within 0.5% of center strike
        precision_range = 0.005 * center
        prob_near_center = (norm.cdf((np.log(center + precision_range) - mu) / sigma) - 
                           norm.cdf((np.log(center - precision_range) - mu) / sigma))
        
        # Step 4: Calculate option greeks
        # For simplicity, we'll use approximate formulas
        # In a real implementation, would use more sophisticated models
        
        # Delta calculation (approximately zero if centered at current price)
        delta_distance = (center - current_price) / (current_price * std_dev)
        position_delta = -delta_distance * np.exp(-delta_distance**2/2) * 2
        
        # Gamma calculation (highest near center strike)
        gamma_factor = np.exp(-((np.log(current_price) - np.log(center))**2) / (2 * sigma**2))
        position_gamma = gamma_factor / (center * sigma * np.sqrt(2 * np.pi * time_to_exp_years))
        
        # Theta calculation (positive near center, negative away from center)
        theta_factor = 1 - 2 * (abs(current_price - center) / wing_width)**2
        theta_base = -gamma_factor * annual_std_dev / (2 * np.sqrt(time_to_exp_years))
        position_theta = theta_base * theta_factor if abs(current_price - center) < wing_width else theta_base
        
        # Vega calculation (typically negative for butterflies)
        vega_factor = (current_price - center)**2 / (current_price * annual_std_dev)**2 - 1
        position_vega = vega_factor * gamma_factor * np.sqrt(time_to_exp_years)
        
        # Step 5: Calculate theoretical current value using simplified approximation
        # In practice, would use proper option pricing model for each leg
        time_factor = np.sqrt(time_to_exp_years)
        distance_factor = 1 - min(1, abs(current_price - center) / wing_width)
        vol_factor = implied_volatility / 0.2  # Normalize to 20% IV
        
        theoretical_value = target_debit + (
            (theoretical_max_profit - target_debit) * 
            np.exp(-0.5 * ((current_price - center) / (center * std_dev))**2) * 
            (1 - np.exp(-3 * time_to_exp_years))  # Time value component
        )
        
        # Step 6: Calculate expected value based on probability distribution
        expected_value = (max_profit * probability_of_profit) - (max_loss * (1 - probability_of_profit))
        
        logger.debug(f"Theoretical value for butterfly spread: ${theoretical_value:.2f}, "
                    f"Max profit: ${max_profit:.2f}, Probability: {probability_of_profit:.1%}")
        
        return {
            "theoretical_value": theoretical_value,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "breakeven_lower": breakeven_lower,
            "breakeven_upper": breakeven_upper,
            "probability_of_profit": probability_of_profit,
            "probability_of_max_profit": prob_near_center,
            "expected_value": expected_value,
            "greeks": {
                "delta": position_delta,
                "gamma": position_gamma,
                "theta": position_theta,
                "vega": position_vega
            },
            "center_strike": center,
            "wing_width": wing_width
        }
    
    def _should_adjust_position(self, current_position: Dict[str, Any], 
                              current_price: float, 
                              days_remaining: int) -> Tuple[bool, str]:
        """
        Determine if a butterfly spread position requires adjustment based on price movement and time decay.
        
        This method evaluates the current butterfly spread position against market conditions
        to determine if and how the position should be adjusted. Position adjustments are critical
        for butterfly spreads as underlying price movements can significantly alter the risk/reward
        profile, especially as expiration approaches.
        
        Parameters:
            current_position (Dict[str, Any]): Details of the current position including:
                - strikes: Dictionary of strike prices (center_strike, wings)
                - days_to_expiration: Original DTE at entry
                - quantity: Number of butterfly spreads
                - greeks: Current position Greeks (delta, gamma, theta, vega)
                - value: Current mark value of the position
                - cost_basis: Initial debit/credit of the position
                
            current_price (float): Current price of the underlying asset
            days_remaining (int): Number of days remaining until expiration
            
        Returns:
            Tuple[bool, str]: (should_adjust, adjustment_type) where adjustment_type is one of:
                - "time_exit": Close position due to time threshold
                - "roll_upper_wing": Adjust upper wing to manage directional risk
                - "roll_lower_wing": Adjust lower wing to manage directional risk
                - "recenter": Roll entire position to new center strike
                - "take_profit": Take profit as position approaches max value
                - "cut_loss": Exit position due to excessive adverse movement
                - "reduce_size": Reduce position size while maintaining structure
                
        Notes:
            Position adjustment decisions incorporate multiple factors:
            
            1. Price movement considerations:
               - Directional bias: How far the price has moved from center strike
               - Speed of movement: Rate of price change relative to historical volatility
               - Technical resistance/support: Whether price is likely to reverse
               
            2. Time decay impact:
               - Gamma risk: Increases exponentially in final weeks
               - Pin risk: Significant if price is near any strike at expiration
               - Theta acceleration: Affects optimal timing for adjustment
               
            3. Volatility environment changes:
               - IV expansion: May require wider wings or earlier exit
               - IV contraction: May present profit-taking opportunity
               - Skew changes: Affects relative pricing of wings
               
            4. Adjustment techniques:
               - Wing rolling: Moving one wing to create directional bias
               - Recentralizing: Moving entire structure to center on new price
               - Adding protection: Purchasing additional wings for downside/upside
               - Legging reduction: Selectively reducing certain legs to lock profits
               
            5. Risk management thresholds:
               - Maximum acceptable delta exposure
               - Position gamma limits as expiration approaches
               - Portfolio-level correlation considerations
               
            The adjustment decision framework adapts based on the original strategy objectives,
            current market conditions, and remaining time to expiration, with increased
            adjustment sensitivity as expiration approaches.
        """
        # First check for time-based exit criteria
        # As butterfly approaches expiration, gamma risk increases substantially
        exit_dte_threshold = self.parameters["exit_dte_threshold"]
        if days_remaining <= exit_dte_threshold:
            logger.info(f"Recommending time-based exit with {days_remaining} days remaining "
                       f"(threshold: {exit_dte_threshold})")
            return (True, "time_exit")
        
        # Extract position details
        strikes = current_position.get("strikes", {})
        center_strike = strikes.get("center_strike")
        lower_inner = strikes.get("lower_inner")
        upper_inner = strikes.get("upper_inner")
        
        # Calculate distance metrics
        if not center_strike:
            logger.error("Missing center strike in position data")
            return (False, "")
            
        # Calculate how far price has moved from center (as percentage)
        price_distance_pct = (current_price - center_strike) / center_strike
        abs_distance_pct = abs(price_distance_pct)
        
        # Get management thresholds from parameters
        management_threshold = self.parameters["management_threshold_delta"]
        profit_threshold = self.parameters["profit_take_pct"] / 100
        stop_loss_multiplier = self.parameters["stop_loss_multiplier"]
        
        # Calculate wing width as percent of center strike
        if lower_inner and upper_inner:
            wing_width_pct = min(
                abs(upper_inner - center_strike), 
                abs(center_strike - lower_inner)
            ) / center_strike
        else:
            wing_width_pct = 0.05  # Default assumption
        
        # Check position value if provided
        current_value = current_position.get("current_value")
        max_theoretical = current_position.get("max_theoretical")
        cost_basis = current_position.get("cost_basis")
        
        # Check for profit target
        if (current_value is not None and max_theoretical is not None and cost_basis is not None):
            max_profit = max_theoretical - cost_basis
            current_profit = current_value - cost_basis
            
            # If we've reached our profit target percentage
            if max_profit > 0 and current_profit > 0:
                profit_pct = current_profit / max_profit
                if profit_pct >= profit_threshold:
                    logger.info(f"Recommending profit taking at {profit_pct:.1%} of max profit")
                    return (True, "take_profit")
            
            # Check for loss threshold
            if cost_basis > 0 and current_value < cost_basis:
                loss_pct = (cost_basis - current_value) / cost_basis
                if loss_pct >= stop_loss_multiplier - 1:  # Convert multiplier to percentage
                    logger.info(f"Recommending loss exit at {loss_pct:.1%} loss")
                    return (True, "cut_loss")
        
        # Check distance from center strike
        # As price moves away from center, delta exposure increases
        if abs_distance_pct > management_threshold:
            # Calculate how close we are to the wings
            if price_distance_pct > 0:  # Price above center
                # Calculate how close price is to upper inner wing
                if upper_inner:
                    upper_proximity = (upper_inner - current_price) / (upper_inner - center_strike)
                    
                    # If price is approaching upper wing, consider rolling
                    if upper_proximity <= 0.3:  # Within 30% of distance to wing
                        logger.info(f"Price approaching upper wing, recommending adjustment")
                        return (True, "roll_upper_wing")
            else:  # Price below center
                # Calculate how close price is to lower inner wing
                if lower_inner:
                    lower_proximity = (current_price - lower_inner) / (center_strike - lower_inner)
                    
                    # If price is approaching lower wing, consider rolling
                    if lower_proximity <= 0.3:  # Within 30% of distance to wing
                        logger.info(f"Price approaching lower wing, recommending adjustment")
                        return (True, "roll_lower_wing")
            
            # If price has moved significantly but not near wings, consider recentering
            # This is especially important with significant time remaining
            if days_remaining >= 15 and abs_distance_pct >= 2 * management_threshold:
                # Check if dynamic recentering is enabled
                if self.parameters.get("dynamic_recentering", False):
                    logger.info(f"Price significantly off-center ({price_distance_pct:.1%}), "
                              f"recommending recentering")
                    return (True, "recenter")
        
        # Check for volatility environment changes
        # This would require IV data which is not in the current implementation
        # Placeholder for future implementation
        
        # Check gamma risk as expiration approaches
        # Gamma increases exponentially in final weeks
        gamma_risk_threshold = 10  # DTE threshold for heightened gamma risk awareness
        if days_remaining <= gamma_risk_threshold:
            # For near-dated butterflies with significant gamma exposure
            # we become more sensitive to price movement
            adjusted_management_threshold = management_threshold * 0.7  # 30% tighter threshold
            
            if abs_distance_pct > adjusted_management_threshold:
                if price_distance_pct > 0:
                    return (True, "roll_upper_wing")
                else:
                    return (True, "roll_lower_wing")
        
        # Position doesn't need adjustment
        return (False, "")
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate indicators for butterfly spread strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        for symbol, df in data.items():
            # Ensure required columns exist
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                logger.warning(f"Required price columns not found for {symbol}")
                continue
            
            try:
                # Calculate historical volatility (20-day)
                close_returns = np.log(df['close'] / df['close'].shift(1))
                hist_volatility = close_returns.rolling(window=20).std() * np.sqrt(252)
                
                # Calculate price range metrics
                rolling_high = df['high'].rolling(window=self.parameters["range_days_lookback"]).max()
                rolling_low = df['low'].rolling(window=self.parameters["range_days_lookback"]).min()
                price_range_pct = (rolling_high - rolling_low) / df['close']
                
                # Store indicators
                indicators[symbol] = {
                    "hist_volatility": pd.DataFrame({"hist_volatility": hist_volatility}),
                    "price_range_pct": pd.DataFrame({"price_range_pct": price_range_pct}),
                    "rolling_high": pd.DataFrame({"rolling_high": rolling_high}),
                    "rolling_low": pd.DataFrame({"rolling_low": rolling_low})
                }
                
                # TODO: Add option-specific indicators once option chain data is available
                # - Implied volatility
                # - IV rank/percentile
                # - Term structure
                # - Option volume and open interest
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate butterfly spread signals based on selection criteria.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_indicators in indicators.items():
            try:
                # Check if underlying meets selection criteria
                if not self._check_underlying_eligibility(symbol, data[symbol]):
                    continue
                
                # Get latest data
                latest_data = data[symbol].iloc[-1]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # TODO: Get option chain data for the symbol
                # This is a placeholder - in a real implementation, you would fetch actual option data
                option_chain = None
                
                # Find appropriate strikes
                strikes = self._find_option_strikes(symbol, latest_price, option_chain)
                
                # Select expiration
                expiration = self._select_expiration(option_chain)
                days_to_expiration = (expiration - datetime.now()).days
                
                # Calculate theoretical value
                # Using a placeholder implied volatility
                iv = symbol_indicators["hist_volatility"].iloc[-1]["hist_volatility"]
                theoretical = self._calculate_theoretical_value(strikes, latest_price, days_to_expiration, iv)
                
                # Generate butterfly spread signal
                signal_type = SignalType.BUTTERFLY
                confidence = 0.0
                
                # Calculate confidence based on multiple factors
                # 1. Price range behavior
                range_pct = symbol_indicators["price_range_pct"].iloc[-1]["price_range_pct"]
                range_confidence = min(0.3, 0.3 - abs(range_pct - 0.05) * 10)
                
                # 2. Historical volatility
                vol_confidence = min(0.3, 0.3 - abs(iv - 0.15) * 2)
                
                # 3. Distance from optimal entry point
                center_distance = abs(latest_price - strikes["center_strike"]) / latest_price
                distance_confidence = min(0.3, 0.3 - center_distance * 10)
                
                confidence = min(0.9, range_confidence + vol_confidence + distance_confidence)
                
                # Create butterfly spread signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=signal_type,
                    price=latest_price,
                    timestamp=latest_timestamp,
                    confidence=confidence,
                    stop_loss=None,  # Not applicable for options strategy
                    take_profit=None,  # Not applicable for options strategy
                    metadata={
                        "strategy_type": "butterfly_spread",
                        "strikes": strikes,
                        "expiration": expiration.strftime("%Y-%m-%d"),
                        "days_to_expiration": days_to_expiration,
                        "theoretical": theoretical,
                        "target_net_debit": self.parameters["target_net_debit"],
                        "profit_take_pct": self.parameters["profit_take_pct"],
                        "stop_loss_multiplier": self.parameters["stop_loss_multiplier"]
                    }
                )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals

    def prepare_exit_orders(self, position: Dict[str, Any], 
                           market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare orders to close an existing butterfly spread position.
        
        This method constructs comprehensive exit orders for all legs of a butterfly spread
        that has triggered exit conditions. The butterfly spread's unique risk profile requires
        careful handling of all legs to properly manage the position closure, especially as
        expiration approaches.
        
        The method performs these key functions:
        1. Analyzes the current position structure (all legs of the butterfly)
        2. Evaluates current market conditions to determine optimal exit approach
        3. Creates opposite orders for each leg with appropriate pricing
        4. Determines best execution strategy based on exit reason and market liquidity
        5. Manages sequencing of leg closures if simultaneous execution isn't possible
        
        For butterfly spreads, exit execution requires particular attention due to the 
        multiple legs and the convex risk profile. As expiration approaches, gamma risk
        increases dramatically around the center strike, potentially requiring more
        urgent exit action.
        
        Parameters:
            position (Dict[str, Any]): The butterfly position to close, containing:
                - legs: List of component orders forming the butterfly
                - trade_id: Unique identifier for the position
                - strikes: Dictionary of strike prices for each leg
                - expiration: Expiration date of the options
                - current_value: Current mark value of the position
                - entry_value: Initial cost/credit of the position
                - quantity: Number of butterfly spreads
                
            market_data (Dict[str, Any]): Current market conditions including:
                - Underlying price and movement
                - Current option prices for all legs
                - Implied volatility metrics
                - Expiration calendar information
                
        Returns:
            List[Dict[str, Any]]: List of executable order specifications, each containing:
                - order_type: The type of order (MARKET, LIMIT)
                - symbol: The underlying symbol
                - action: Order action ('BUY' or 'SELL' for each leg)
                - quantity: Number of contracts to trade
                - option_symbol: Specific option contract identifier
                - limit_price: Price limit for limit orders
                - strategy: Strategy identifier ('BUTTERFLY_SPREAD')
                - leg_type: Identifies which leg of the butterfly ('BODY' or 'WING')
                - trade_id: Reference to the original trade for tracking
                
        Notes:
            Butterfly spread exit considerations:
            
            - Exit timing considerations:
              - Profit targets: Best realized when underlying is near center strike
              - Loss management: Critical if underlying moves far from center strike
              - Gamma acceleration: Exit urgency increases dramatically near expiration
              - Liquidity deterioration: Option spreads widen as expiration approaches
            
            - Order type selection strategies:
              - Market orders: Used when immediate execution is critical (near expiration)
              - Limit orders: Used for planned exits when spread remains in optimal range
              - Combination orders: Used to ensure all legs close simultaneously
              
            - Price considerations by leg type:
              - Body (center strike): Most sensitive to price movement, prioritize execution
              - Wings (outer strikes): Provide protection, less time-sensitive to close
              
            - Special handling requirements:
              - Assignment risk: Increases as ITM options approach expiration
              - Pin risk: Significant if underlying price is near any strike at expiration
              - Illiquid options: May require more aggressive pricing or legging out
              - Inversion risk: When butterfly becomes inverted and loses defined risk profile
              
            - Exit sequence strategy if legging is necessary:
              - For profit-taking (near center): Close body positions first
              - For loss management: Close relevant wing position first
              - Always maintain position balance to preserve risk profile
              - Consider slippage and execution risk in determining sequence
              
            Full position closure is essential for butterfly spreads as partial closures
            can create undefined risk profiles that differ significantly from the original
            strategy objectives. The exit approach adapts based on underlying price relative
            to strikes, time remaining, and current market volatility conditions.
        """
        exit_orders = []
        
        # Input validation
        if not position or 'legs' not in position:
            logger.error("Invalid position data for butterfly exit")
            return exit_orders
            
        symbol = position.get('symbol')
        if not symbol:
            logger.error("Missing symbol in butterfly position data")
            return exit_orders
            
        # Extract position details
        legs = position.get('legs', [])
        quantity = position.get('quantity', 1)
        strikes = position.get('strikes', {})
        expiration = position.get('expiration')
        trade_id = position.get('trade_id')
        exit_reason = position.get('exit_reason', 'unknown')
        
        # Get current underlying price
        underlying_price = market_data.get(symbol, {}).get('price', 0)
        
        if not underlying_price:
            logger.error(f"Unable to get current price for {symbol}")
            return exit_orders
            
        # Determine optimal order types based on exit reason
        # Use market orders for time-critical exits, limit orders for profit taking
        use_market_orders = exit_reason in ['expiration', 'stop_loss', 'technical_breakdown'] 
        order_type = "MARKET" if use_market_orders else "LIMIT"
        
        # Track each leg of the butterfly
        body_leg_orders = []
        wing_leg_orders = []
        
        # Create exit orders for each butterfly leg
        for leg in legs:
            if not leg or leg.get('status') != 'FILLED':
                continue
                
            # Extract leg details
            leg_option_symbol = leg.get('option_symbol')
            leg_action = leg.get('action')  # Original action (BUY or SELL)
            leg_strike = leg.get('strike')
            leg_type = leg.get('leg_type')  # BODY or WING
            
            # Determine closing action (opposite of entry)
            close_action = "SELL" if leg_action == "BUY" else "BUY"
            
            # Create order for this leg
            leg_order = {
                'order_type': order_type,
                'symbol': symbol,
                'action': close_action,
                'quantity': quantity,
                'option_symbol': leg_option_symbol,
                'strategy': 'BUTTERFLY_SPREAD',
                'leg_type': leg_type,
                'trade_id': f"close_{trade_id}",
                'metadata': {
                    'original_leg_id': leg.get('leg_id'),
                    'exit_reason': exit_reason,
                    'exit_timestamp': datetime.now().isoformat()
                }
            }
            
            # Add limit price if using limit orders
            if order_type == "LIMIT":
                # Get current option price from market data
                option_price = self._get_option_price(market_data, leg_option_symbol, 
                                                    "bid" if close_action == "SELL" else "ask")
                
                if option_price:
                    # Add buffer for better fill probability
                    buffer_pct = 0.05  # 5% buffer
                    if close_action == "SELL":
                        leg_order['limit_price'] = option_price * (1 - buffer_pct)  # Sell slightly below bid
                    else:
                        leg_order['limit_price'] = option_price * (1 + buffer_pct)  # Buy slightly above ask
            
            # Categorize by leg type for sequencing if needed
            if leg_type == "BODY":
                body_leg_orders.append(leg_order)
            else:
                wing_leg_orders.append(leg_order)
        
        # Determine if we need to sequence the exit
        if not use_market_orders and exit_reason == 'profit_target':
            # For profit taking, close body positions first
            exit_orders.extend(body_leg_orders)
            exit_orders.extend(wing_leg_orders)
        elif not use_market_orders and exit_reason == 'stop_loss':
            # For stop loss, close wing positions first
            exit_orders.extend(wing_leg_orders)
            exit_orders.extend(body_leg_orders)
        else:
            # For other cases or market orders, close all simultaneously
            exit_orders = body_leg_orders + wing_leg_orders
        
        logger.info(f"Created butterfly spread exit orders for {symbol} due to {exit_reason}")
        return exit_orders
    
    def _get_option_price(self, market_data: Dict[str, Any], option_symbol: str, price_type: str = 'mid') -> float:
        """
        Get the current price for an option contract from market data.
        
        Args:
            market_data: Dictionary containing market data
            option_symbol: Option contract symbol
            price_type: Type of price to retrieve ('bid', 'ask', or 'mid')
            
        Returns:
            float: Option price or None if not available
        """
        # Extract option data from market data
        option_data = market_data.get('options', {}).get(option_symbol, {})
        
        if not option_data:
            return None
            
        if price_type == 'bid':
            return option_data.get('bid', 0)
        elif price_type == 'ask':
            return option_data.get('ask', 0)
        else:
            # Default to mid price
            bid = option_data.get('bid', 0)
            ask = option_data.get('ask', 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return None

# TODO: Implement function to construct butterfly spread orders
# TODO: Implement function to calculate and monitor position P&L
# TODO: Implement function to adjust wing positions when necessary
# TODO: Implement function to integrate ML model for strike selection
# TODO: Implement function to handle calendar roll for expiration management 