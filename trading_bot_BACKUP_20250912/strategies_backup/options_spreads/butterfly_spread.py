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
    
    This strategy creates a four-legged options position that maximizes profit
    when the underlying price stays near the center strike at expiration.
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
        Check if underlying meets selection criteria.
        
        Args:
            symbol: Symbol to check
            data: DataFrame with price and volume data
            
        Returns:
            Boolean indicating if underlying is eligible
        """
        # TODO: Implement liquidity checks (ADV, option OI, bid-ask spreads)
        
        # Calculate recent price range to check for range-bound behavior
        lookback = self.parameters["range_days_lookback"]
        if len(data) < lookback:
            logger.warning(f"Insufficient data for {symbol} to check range-bound behavior")
            return False
            
        recent_data = data.iloc[-lookback:]
        high_low_range = (recent_data['high'].max() - recent_data['low'].min()) / data.iloc[-1]['close']
        
        # Check if price range is within reasonable bounds (not too volatile, not too stable)
        range_bound = 0.03 <= high_low_range <= 0.10
        
        # TODO: Implement IV rank check (requires option chain data)
        iv_in_range = True  # Placeholder
        
        return range_bound and iv_in_range
    
    def _find_option_strikes(self, symbol: str, current_price: float, option_chain: Any) -> Dict[str, float]:
        """
        Find appropriate strikes for butterfly spread.
        
        Args:
            symbol: Underlying symbol
            current_price: Current price of underlying
            option_chain: Option chain data structure (placeholder)
            
        Returns:
            Dictionary with selected strikes for each leg
        """
        # TODO: Implement strike selection logic
        # - Select center strike based on delta target or offset from ATM
        # - Calculate inner wing strikes based on width
        # - Calculate outer wing strikes based on width
        
        # Placeholder implementation
        center_strike = round(current_price, 0)  # Round to nearest whole dollar
        inner_wing_width = self.parameters["inner_wing_width"]
        outer_wing_width = self.parameters["outer_wing_width"]
        
        # Calculate strikes
        lower_inner = center_strike - inner_wing_width
        upper_inner = center_strike + inner_wing_width
        lower_outer = lower_inner - outer_wing_width
        upper_outer = upper_inner + outer_wing_width
        
        return {
            "center_strike": center_strike,
            "lower_inner": lower_inner,
            "upper_inner": upper_inner,
            "lower_outer": lower_outer,
            "upper_outer": upper_outer
        }
    
    def _select_expiration(self, option_chain: Any) -> datetime:
        """
        Select appropriate expiration based on DTE criteria.
        
        Args:
            option_chain: Option chain data structure (placeholder)
            
        Returns:
            Selected expiration date
        """
        # TODO: Implement expiration selection logic
        # - Find all available expirations
        # - Filter by DTE criteria (25-45 days)
        # - Select optimal expiration
        
        # Placeholder implementation
        min_dte = self.parameters["min_days_to_expiration"]
        max_dte = self.parameters["max_days_to_expiration"]
        target_dte = (min_dte + max_dte) // 2
        
        # For now, just return a date target_dte days from now
        return datetime.now() + timedelta(days=target_dte)
    
    def _calculate_theoretical_value(self, strikes: Dict[str, float], 
                                    current_price: float, 
                                    days_to_expiration: int,
                                    implied_volatility: float) -> Dict[str, float]:
        """
        Calculate theoretical value of butterfly spread.
        
        Args:
            strikes: Dictionary with selected strikes
            current_price: Current price of underlying
            days_to_expiration: Days to expiration
            implied_volatility: Implied volatility
            
        Returns:
            Dictionary with theoretical values
        """
        # TODO: Implement option pricing model for butterfly spread
        # - Use Black-Scholes or binomial model to price each leg
        # - Calculate net value of butterfly spread
        # - Calculate max profit, max loss, breakeven points
        
        # Placeholder implementation
        center = strikes["center_strike"]
        width = strikes["upper_inner"] - center
        
        # Simplistic calculation (not accurate pricing)
        max_profit = width - self.parameters["target_net_debit"]
        max_loss = self.parameters["target_net_debit"]
        
        # Calculate rough probability of profit using normal distribution
        # (This is a very simplified model)
        std_dev = implied_volatility * current_price * np.sqrt(days_to_expiration / 365)
        lower_breakeven = center - width + self.parameters["target_net_debit"]
        upper_breakeven = center + width - self.parameters["target_net_debit"]
        
        return {
            "theoretical_value": self.parameters["target_net_debit"],
            "max_profit": max_profit,
            "max_loss": max_loss,
            "lower_breakeven": lower_breakeven,
            "upper_breakeven": upper_breakeven,
            "center_strike": center
        }
    
    def _should_adjust_position(self, current_position: Dict[str, Any], 
                              current_price: float, 
                              days_remaining: int) -> Tuple[bool, str]:
        """
        Determine if position should be adjusted based on price movement.
        
        Args:
            current_position: Current position details
            current_price: Current price of underlying
            days_remaining: Days remaining to expiration
            
        Returns:
            Tuple of (should_adjust, adjustment_type)
        """
        # TODO: Implement position adjustment logic
        # - Check if price is approaching inner wings
        # - Check if DTE is approaching exit threshold
        # - Calculate if adjustment would be beneficial
        
        # Placeholder implementation
        center = current_position["strikes"]["center_strike"]
        distance_from_center = abs(current_price - center) / center
        management_threshold = self.parameters["management_threshold_delta"]
        
        if days_remaining <= self.parameters["exit_dte_threshold"]:
            return (True, "time_exit")
        
        if distance_from_center > management_threshold:
            if current_price > center:
                return (True, "roll_upper_wing")
            else:
                return (True, "roll_lower_wing")
                
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

# TODO: Implement function to construct butterfly spread orders
# TODO: Implement function to calculate and monitor position P&L
# TODO: Implement function to adjust wing positions when necessary
# TODO: Implement function to integrate ML model for strike selection
# TODO: Implement function to handle calendar roll for expiration management 