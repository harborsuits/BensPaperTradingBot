#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Strategy Rotator

This script integrates the advanced options strategies with the strategy rotation system,
allowing the system to dynamically select and rotate between different options strategies
based on market regimes and volatility conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional, Tuple

from trading_bot.strategies.advanced_options_strategies import (
    ImpliedVolatilitySurface,
    OptionsPositionManager,
    VolatilityBasedStrikeSelector,
    GammaScalper,
    AdvancedOptionsStrategyManager
)

# If strategy rotator is available, import it
try:
    from trading_bot.ai_scoring.strategy_rotator import StrategyRotator
except ImportError:
    StrategyRotator = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("options_strategy_rotator")


class OptionsStrategyRotator:
    """
    Manages the rotation between different options strategies based on market conditions.
    
    This class extends the base strategy rotation system with options-specific capabilities:
    1. Detects volatility regime and trend/momentum conditions
    2. Selects appropriate options strategies based on current market environment
    3. Uses IV surface analysis to optimize trade entry and exit parameters
    4. Manages positions including adjustments, rolls, and scaling
    """
    
    def __init__(self, config=None):
        """
        Initialize the options strategy rotator.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Initialize advanced options components
        self.options_manager = AdvancedOptionsStrategyManager()
        self.active_positions = {}
        self.position_history = []
        
        # Initialize base strategy rotator if available
        self.base_rotator = StrategyRotator() if StrategyRotator else None
        
        # Strategy allocations
        self.current_allocations = {}
        
        # Define options strategies with appropriate market regimes
        self.options_strategies = {
            # High volatility strategies
            "long_put_spreads": {
                "suitable_regimes": ["bearish", "volatile"],
                "min_iv_percentile": 60,
                "suitable_correlation": "high_positive"
            },
            "long_call_spreads": {
                "suitable_regimes": ["bullish", "volatile"],
                "min_iv_percentile": 40,
                "suitable_correlation": "high_positive"
            },
            "short_iron_condors": {
                "suitable_regimes": ["sideways", "low_volatility"],
                "max_iv_percentile": 60,
                "suitable_correlation": "uncorrelated"
            },
            "long_straddles": {
                "suitable_regimes": ["volatile"],
                "min_iv_percentile": 20,
                "max_iv_percentile": 80,
                "suitable_correlation": "any"
            },
            "calendar_spreads": {
                "suitable_regimes": ["sideways", "low_volatility"],
                "max_iv_percentile": 40,
                "suitable_correlation": "low_negative"
            },
            "butterflies": {
                "suitable_regimes": ["sideways"],
                "max_iv_percentile": 50,
                "suitable_correlation": "uncorrelated"
            }
        }
    
    def analyze_market_environment(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current market environment for options trading.
        
        Args:
            market_data: Dictionary with current market data
            
        Returns:
            Dict with market environment analysis
        """
        # Extract key market indicators
        vix = market_data.get('vix', 15.0)
        market_trend = market_data.get('market_trend', 0.0)
        sector_dispersion = market_data.get('sector_dispersion', 0.5)
        iv_percentile = market_data.get('iv_percentile', 50)
        iv_term_structure = market_data.get('iv_term_structure', 0.0)
        
        # Determine volatility regime
        if vix < 15:
            vol_regime = "low_volatility"
        elif vix < 25:
            vol_regime = "normal_volatility"
        elif vix < 35:
            vol_regime = "high_volatility"
        else:
            vol_regime = "extreme_volatility"
            
        # Determine trend regime
        if market_trend > 0.8:
            trend_regime = "strong_bullish"
        elif market_trend > 0.3:
            trend_regime = "bullish"
        elif market_trend > -0.3:
            trend_regime = "sideways"
        elif market_trend > -0.8:
            trend_regime = "bearish"
        else:
            trend_regime = "strong_bearish"
            
        # Determine IV environment
        if iv_percentile < 20:
            iv_environment = "very_low"
        elif iv_percentile < 40:
            iv_environment = "low"
        elif iv_percentile < 60:
            iv_environment = "medium"
        elif iv_percentile < 80:
            iv_environment = "high"
        else:
            iv_environment = "very_high"
            
        # Analyze IV term structure
        if iv_term_structure > 0.1:
            # Contango (longer-dated options have higher IVs)
            term_structure = "contango"
        elif iv_term_structure < -0.1:
            # Backwardation (shorter-dated options have higher IVs)
            term_structure = "backwardation"
        else:
            term_structure = "flat"
            
        # Determine optimal strategy type
        if vol_regime in ["high_volatility", "extreme_volatility"]:
            if trend_regime in ["bullish", "strong_bullish"]:
                primary_strategy = "long_call_spreads"
            elif trend_regime in ["bearish", "strong_bearish"]:
                primary_strategy = "long_put_spreads"
            else:
                primary_strategy = "long_straddles"
        else:
            if trend_regime == "sideways":
                primary_strategy = "short_iron_condors" if iv_environment in ["medium", "high", "very_high"] else "butterflies"
            elif trend_regime in ["bullish", "strong_bullish"]:
                primary_strategy = "long_call_spreads"
            elif trend_regime in ["bearish", "strong_bearish"]:
                primary_strategy = "long_put_spreads"
            else:
                primary_strategy = "calendar_spreads"
        
        # Compile analysis
        analysis = {
            "vol_regime": vol_regime,
            "trend_regime": trend_regime,
            "iv_environment": iv_environment,
            "term_structure": term_structure,
            "sector_dispersion": sector_dispersion,
            "primary_strategy": primary_strategy,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Market environment analysis: {analysis['vol_regime']}, {analysis['trend_regime']}, {analysis['iv_environment']}")
        
        return analysis
    
    def get_strategy_allocations(self, market_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine the optimal allocation across options strategies.
        
        Args:
            market_analysis: Analysis of current market environment
            
        Returns:
            Dictionary mapping strategy names to allocation percentages
        """
        # Define base allocations based on primary strategy
        primary_strategy = market_analysis['primary_strategy']
        vol_regime = market_analysis['vol_regime']
        trend_regime = market_analysis['trend_regime']
        
        # Start with base allocation
        allocations = {strategy: 0.0 for strategy in self.options_strategies.keys()}
        
        # Set primary strategy allocation
        allocations[primary_strategy] = 0.6
        
        # Determine secondary strategy based on regime
        if vol_regime in ["high_volatility", "extreme_volatility"]:
            # In high volatility, add long options strategies
            if "long_straddles" != primary_strategy:
                allocations["long_straddles"] = 0.3
                
            # Add small hedging position
            if trend_regime == "bullish":
                allocations["long_put_spreads"] = 0.1
            elif trend_regime == "bearish":
                allocations["long_call_spreads"] = 0.1
            else:
                allocations["butterflies"] = 0.1
        else:
            # In lower volatility, more diversified approach
            if trend_regime == "sideways":
                if "short_iron_condors" != primary_strategy:
                    allocations["short_iron_condors"] = 0.2
                allocations["calendar_spreads"] = 0.2
            elif trend_regime in ["bullish", "strong_bullish"]:
                allocations["long_call_spreads"] = 0.3
                allocations["calendar_spreads"] = 0.1
            elif trend_regime in ["bearish", "strong_bearish"]:
                allocations["long_put_spreads"] = 0.3
                allocations["calendar_spreads"] = 0.1
            else:
                allocations["butterflies"] = 0.2
                allocations["calendar_spreads"] = 0.2
                
        # Normalize allocations to ensure they sum to 1.0
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocations = {k: v / total_allocation for k, v in allocations.items()}
            
        self.current_allocations = allocations
        
        logger.info(f"Strategy allocations: Primary {primary_strategy} at {allocations[primary_strategy]:.0%}")
        
        return allocations
    
    def get_strategy_parameters(self, strategy_name: str, 
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimal parameters for a specific options strategy.
        
        Args:
            strategy_name: Name of the options strategy
            market_data: Current market data
            
        Returns:
            Dictionary with strategy parameters
        """
        # Get underlying price
        underlying_price = market_data.get('underlying_price', 100.0)
        
        # Default parameters
        params = {
            "underlying_symbol": market_data.get('symbol', 'SPY'),
            "underlying_price": underlying_price,
            "target_dte": 45  # Default to 45-day expiration
        }
        
        # Get IV percentile
        iv_percentile = market_data.get('iv_percentile', 50)
        
        # Adjust parameters based on strategy and market conditions
        if strategy_name == "long_put_spreads":
            params.update({
                "option_type": "put",
                "is_debit_spread": True,
                "delta_long": 0.30,
                "width": 5 if underlying_price > 100 else underlying_price * 0.05
            })
        elif strategy_name == "long_call_spreads":
            params.update({
                "option_type": "call",
                "is_debit_spread": True,
                "delta_long": 0.30,
                "width": 5 if underlying_price > 100 else underlying_price * 0.05
            })
        elif strategy_name == "short_iron_condors":
            # Adjust width based on IV
            if iv_percentile < 30:
                width_factor = 0.03  # Narrower wings in low IV
            else:
                width_factor = 0.04 + (iv_percentile / 100) * 0.03  # Wider wings in higher IV
                
            params.update({
                "delta_short_puts": 0.16,
                "delta_short_calls": 0.16,
                "width_puts": underlying_price * width_factor,
                "width_calls": underlying_price * width_factor
            })
        elif strategy_name == "long_straddles":
            params.update({
                "delta_target": 0.50,  # ATM options
                "target_dte": 30  # Shorter expiration for long volatility
            })
        elif strategy_name == "calendar_spreads":
            params.update({
                "delta_target": 0.30 if market_data.get('market_trend', 0) > 0 else 0.20,
                "front_month_dte": 30,
                "back_month_dte": 60,
                "option_type": "call" if market_data.get('market_trend', 0) > 0 else "put"
            })
        elif strategy_name == "butterflies":
            params.update({
                "center_strike_offset": 0,  # ATM
                "wing_width": 0.05 * underlying_price,
                "option_type": "call" if market_data.get('market_trend', 0) > 0 else "put"
            })
            
        return params
    
    def execute_strategy_rotation(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a full strategy rotation based on current market conditions.
        
        Args:
            market_data: Current market data
            
        Returns:
            Dict with rotation results
        """
        # Analyze market environment
        market_analysis = self.analyze_market_environment(market_data)
        
        # Get strategy allocations
        allocations = self.get_strategy_allocations(market_analysis)
        
        # Process trades for each selected strategy
        recommended_trades = []
        
        for strategy_name, allocation in allocations.items():
            if allocation > 0.01:  # Only consider strategies with meaningful allocation
                # Get strategy parameters
                params = self.get_strategy_parameters(strategy_name, market_data)
                
                # Create strategy using the advanced options manager
                strategy = self.options_manager.create_volatility_strategy(
                    strategy_type=self._map_to_manager_strategy(strategy_name),
                    underlying_symbol=params['underlying_symbol'],
                    underlying_price=params['underlying_price'],
                    target_dte=params['target_dte'],
                    **{k: v for k, v in params.items() if k not in ['underlying_symbol', 'underlying_price', 'target_dte']}
                )
                
                if strategy:
                    recommended_trades.append({
                        'strategy_name': strategy_name,
                        'allocation': allocation,
                        'trade_details': strategy,
                        'expected_metrics': self._calculate_expected_metrics(strategy, market_data)
                    })
        
        # Check for position adjustments
        adjustments = self.options_manager.check_positions_for_adjustments(datetime.now())
        
        # Create final rotation recommendation
        rotation_result = {
            'market_analysis': market_analysis,
            'allocations': allocations,
            'recommended_trades': recommended_trades,
            'position_adjustments': adjustments,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return rotation_result
    
    def _map_to_manager_strategy(self, strategy_name: str) -> str:
        """Map internal strategy names to the advanced manager's strategy types."""
        strategy_map = {
            "long_put_spreads": "vertical_spread",
            "long_call_spreads": "vertical_spread",
            "short_iron_condors": "iron_condor",
            "long_straddles": "straddle",
            "calendar_spreads": "calendar",
            "butterflies": "butterfly"
        }
        return strategy_map.get(strategy_name, "iron_condor")  # Default to iron condor
    
    def _calculate_expected_metrics(self, strategy: Dict[str, Any], 
                                  market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate expected metrics for a strategy."""
        # This would use real options pricing models in production
        # Simplified example here
        return {
            'expected_profit': 100.0,
            'max_loss': 500.0,
            'probability_of_profit': 0.68,
            'expected_holding_period': 30  # days
        }
    
    def update_positions(self, position_updates: List[Dict[str, Any]]) -> None:
        """
        Update the status of active positions.
        
        Args:
            position_updates: List of position updates
        """
        for update in position_updates:
            position_id = update.get('position_id')
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                
                # Update position data
                position['current_price'] = update.get('current_price', position['current_price'])
                position['unrealized_pnl'] = update.get('unrealized_pnl', position.get('unrealized_pnl', 0))
                
                # Check if position is closed
                if update.get('status') == 'closed':
                    position['close_date'] = update.get('close_date', datetime.now())
                    position['realized_pnl'] = update.get('realized_pnl', position.get('unrealized_pnl', 0))
                    
                    # Move to history
                    self.position_history.append(position)
                    del self.active_positions[position_id]
                    logger.info(f"Closed position {position_id} with P&L ${position['realized_pnl']:.2f}")


def main():
    """Example usage of the Options Strategy Rotator."""
    # Create rotator
    rotator = OptionsStrategyRotator()
    
    # Simulate market data
    market_data = {
        'symbol': 'SPY',
        'underlying_price': 420.0,
        'vix': 22.5,
        'market_trend': 0.3,  # Positive trend
        'sector_dispersion': 0.6,
        'iv_percentile': 65,
        'iv_term_structure': 0.05
    }
    
    # Execute rotation
    result = rotator.execute_strategy_rotation(market_data)
    
    # Print results
    print("\nOptions Strategy Rotation Results")
    print("================================")
    
    print(f"\nMarket Analysis:")
    print(f"Volatility Regime: {result['market_analysis']['vol_regime']}")
    print(f"Trend Regime: {result['market_analysis']['trend_regime']}")
    print(f"IV Environment: {result['market_analysis']['iv_environment']}")
    
    print("\nStrategy Allocations:")
    for strategy, allocation in result['allocations'].items():
        if allocation > 0.01:
            print(f"{strategy}: {allocation:.1%}")
    
    print("\nRecommended Trades:")
    for trade in result['recommended_trades']:
        print(f"\n{trade['strategy_name']} ({trade['allocation']:.1%}):")
        print(f"  Type: {trade['trade_details']['type']}")
        print(f"  Underlying: {trade['trade_details']['underlying']}")
        print(f"  Expiration: {trade['trade_details']['target_expiration'].date()}")
        print("  Legs:")
        for leg in trade['trade_details']['legs']:
            print(f"    {leg['position']} {leg['option_type']} @ ${leg['strike']}")
    
    print("\nStrategy rotation complete!")


if __name__ == "__main__":
    main() 