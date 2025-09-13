#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Options Strategies Example

This script demonstrates how to use the advanced options strategies module,
including:
1. Analyzing implied volatility surfaces
2. Selecting strikes based on volatility metrics
3. Managing and rolling options positions
4. Implementing gamma scalping for delta-neutral positions
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trading_bot.strategies.advanced_options_strategies import (
    ImpliedVolatilitySurface,
    OptionsPositionManager,
    VolatilityBasedStrikeSelector,
    GammaScalper,
    AdvancedOptionsStrategyManager
)

def generate_mock_options_chain(underlying_price=100.0, expiration_days=[30, 60, 90], 
                               strike_range=(0.7, 1.3), strike_step=0.05):
    """
    Generate a mock options chain for testing.
    
    Args:
        underlying_price: Price of the underlying asset
        expiration_days: List of days to expiration
        strike_range: Tuple with (min_moneyness, max_moneyness)
        strike_step: Step size for strikes
        
    Returns:
        DataFrame: Mock options chain
    """
    options_chain = []
    current_date = datetime.now()
    
    # Generate strikes
    min_moneyness, max_moneyness = strike_range
    moneyness_range = np.arange(min_moneyness, max_moneyness + strike_step, strike_step)
    strikes = [round(underlying_price * m, 2) for m in moneyness_range]
    
    for dte in expiration_days:
        expiration_date = current_date + timedelta(days=dte)
        
        # Base level of implied volatility (higher for longer maturities)
        base_iv = 0.20 + 0.05 * (dte / 30)
        
        for strike in strikes:
            moneyness = strike / underlying_price
            
            # Generate IV skew (higher IVs for OTM options)
            skew_factor = 1.0 + abs(moneyness - 1.0) * 0.5
            
            # Put skew - puts typically have higher IVs than calls
            put_skew = 1.1 if moneyness < 1.0 else 1.0
            call_skew = 1.0 if moneyness > 1.0 else 0.9
            
            # Calculate IVs
            call_iv = base_iv * skew_factor * call_skew
            put_iv = base_iv * skew_factor * put_skew
            
            # Add call option
            options_chain.append({
                'symbol': 'SPY',
                'expiration': expiration_date,
                'days_to_expiration': dte,
                'strike': strike,
                'option_type': 'call',
                'underlying_price': underlying_price,
                'bid': 0.0,  # These would be calculated from Black-Scholes in a real system
                'ask': 0.0,
                'implied_volatility': call_iv,
                'volume': np.random.randint(10, 1000)
            })
            
            # Add put option
            options_chain.append({
                'symbol': 'SPY',
                'expiration': expiration_date,
                'days_to_expiration': dte,
                'strike': strike,
                'option_type': 'put',
                'underlying_price': underlying_price,
                'bid': 0.0,  # These would be calculated from Black-Scholes in a real system
                'ask': 0.0,
                'implied_volatility': put_iv,
                'volume': np.random.randint(10, 1000)
            })
    
    return pd.DataFrame(options_chain)

def demo_iv_surface():
    """Demonstrate the implied volatility surface analysis."""
    print("\n=== Implied Volatility Surface Demo ===")
    
    # Generate mock options chain
    options_chain = generate_mock_options_chain()
    
    # Create and update IV surface
    iv_surface = ImpliedVolatilitySurface()
    iv_surface.update_surface(options_chain)
    
    # Get IV for specific moneyness and days to expiration
    moneyness = 1.0  # ATM
    days = 30
    atm_iv = iv_surface.get_iv(moneyness, days)
    print(f"ATM IV for {days} days: {atm_iv:.2%}")
    
    # Get IV skew metrics
    skew = iv_surface.get_iv_skew(days)
    print(f"IV Skew metrics for {days} days:")
    print(f"  Skew: {skew['skew']:.4f}")
    print(f"  Convexity: {skew['convexity']:.4f}")
    print(f"  Put-Call Skew: {skew['put_call_skew']:.4%}")
    
    # Visualize the surface
    print("\nGenerating IV surface visualization...")
    iv_surface.visualize_surface()

def demo_strike_selection():
    """Demonstrate volatility-based strike selection."""
    print("\n=== Volatility-Based Strike Selection Demo ===")
    
    # Generate mock options chain
    options_chain = generate_mock_options_chain()
    
    # Create IV surface and strike selector
    iv_surface = ImpliedVolatilitySurface()
    iv_surface.update_surface(options_chain)
    strike_selector = VolatilityBasedStrikeSelector(iv_surface)
    
    # Select strikes for an iron condor
    underlying_price = 100.0
    target_dte = 45
    
    print(f"Selecting iron condor strikes for ${underlying_price:.2f} underlying, {target_dte} DTE")
    
    strikes = strike_selector.select_iron_condor_strikes(
        underlying_price=underlying_price,
        target_dte=target_dte,
        delta_short_puts=0.16,
        delta_short_calls=0.16,
        width_puts=5,
        width_calls=5
    )
    
    print("\nSelected strikes:")
    print(f"  Short Put: ${strikes['short_put_strike']}")
    print(f"  Long Put: ${strikes['long_put_strike']}")
    print(f"  Short Call: ${strikes['short_call_strike']}")
    print(f"  Long Call: ${strikes['long_call_strike']}")
    print(f"\nExpected move: ${strikes['expected_move']}")
    print(f"Expected price range: ${strikes['expected_range'][0]} to ${strikes['expected_range'][1]}")
    
    # Calculate option greeks
    call_greeks = strike_selector.calculate_option_greeks(
        underlying_price=underlying_price,
        strike_price=strikes['short_call_strike'],
        days_to_expiration=target_dte,
        implied_volatility=iv_surface.get_iv(strikes['short_call_strike']/underlying_price, target_dte),
        option_type='call'
    )
    
    put_greeks = strike_selector.calculate_option_greeks(
        underlying_price=underlying_price,
        strike_price=strikes['short_put_strike'],
        days_to_expiration=target_dte,
        implied_volatility=iv_surface.get_iv(strikes['short_put_strike']/underlying_price, target_dte),
        option_type='put'
    )
    
    print("\nShort Call Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    
    print("\nShort Put Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")

def demo_position_management():
    """Demonstrate options position management and roll recommendations."""
    print("\n=== Options Position Management Demo ===")
    
    # Create position manager
    position_manager = OptionsPositionManager(roll_dte_threshold=7)
    
    # Create a sample position (iron condor)
    current_date = datetime.now()
    expiration_date = current_date + timedelta(days=30)
    position_data = {
        'strategy': 'theta_collection',
        'type': 'iron_condor',
        'legs': [
            {
                'option_type': 'put',
                'strike': 90.0,
                'expiration': expiration_date,
                'quantity': 1,
                'position': 'long'
            },
            {
                'option_type': 'put',
                'strike': 95.0,
                'expiration': expiration_date,
                'quantity': 1,
                'position': 'short'
            },
            {
                'option_type': 'call',
                'strike': 105.0,
                'expiration': expiration_date,
                'quantity': 1,
                'position': 'short'
            },
            {
                'option_type': 'call',
                'strike': 110.0,
                'expiration': expiration_date,
                'quantity': 1,
                'position': 'long'
            }
        ],
        'underlying': 'SPY',
        'entry_date': current_date,
        'entry_price': 1.20,  # Net credit received
        'max_profit': 1.20,   # Net credit received
        'max_loss': 3.80      # Width of spreads (5.0) minus credit (1.20)
    }
    
    # Add position
    position_id = 'IC_SPY_1'
    position_manager.add_position(position_id, position_data)
    print(f"Added position {position_id}: {position_data['type']} on {position_data['underlying']}")
    
    # Check if roll needed (should be no since we just created it)
    check_date = current_date
    positions_to_roll = position_manager.check_positions_for_rolls(check_date)
    print(f"Positions to roll on {check_date.date()}: {len(positions_to_roll)}")
    
    # Fast forward to near expiration
    check_date = expiration_date - timedelta(days=5)
    positions_to_roll = position_manager.check_positions_for_rolls(check_date)
    print(f"Positions to roll on {check_date.date()}: {len(positions_to_roll)}")
    
    if positions_to_roll:
        # Generate a roll recommendation
        print("\nGenerating roll recommendation:")
        iv_surface = ImpliedVolatilitySurface()
        iv_surface.update_surface(generate_mock_options_chain())
        
        recommendation = position_manager.generate_roll_recommendation(
            positions_to_roll[0], None, iv_surface)
        
        print(f"Recommendation: Roll to {recommendation['target_expiration'].date()}")
        print("New legs:")
        for leg in recommendation['legs']:
            print(f"  {leg['position']} {leg['option_type']} @ ${leg['strike']}")

def demo_gamma_scalping():
    """Demonstrate gamma scalping for delta-neutral positions."""
    print("\n=== Gamma Scalping Demo ===")
    
    # Create gamma scalper
    gamma_scalper = GammaScalper(rebalance_threshold=0.10)
    
    # Create a sample position (long straddle)
    position_id = "STRADDLE_SPY_1"
    underlying_price = 100.0
    strike = 100.0
    
    # Calculate option greeks
    strike_selector = VolatilityBasedStrikeSelector()
    call_greeks = strike_selector.calculate_option_greeks(
        underlying_price=underlying_price,
        strike_price=strike,
        days_to_expiration=30,
        implied_volatility=0.20,
        option_type='call'
    )
    
    put_greeks = strike_selector.calculate_option_greeks(
        underlying_price=underlying_price,
        strike_price=strike,
        days_to_expiration=30,
        implied_volatility=0.22,  # Slightly higher IV for puts
        option_type='put'
    )
    
    # Create straddle position
    options_legs = [
        {
            'option_type': 'call',
            'strike': strike,
            'expiration': datetime.now() + timedelta(days=30),
            'quantity': 1,
            'greeks': call_greeks,
            'symbol': 'SPY'
        },
        {
            'option_type': 'put',
            'strike': strike,
            'expiration': datetime.now() + timedelta(days=30),
            'quantity': 1,
            'greeks': put_greeks,
            'symbol': 'SPY'
        }
    ]
    
    # Add position
    position = gamma_scalper.add_position(position_id, options_legs, underlying_price)
    print(f"Added position {position_id} for gamma scalping")
    print(f"Position delta: {position['position_greeks']['delta']:.4f}")
    print(f"Position gamma: {position['position_greeks']['gamma']:.6f}")
    print(f"Initial hedge: {position['hedge_quantity']} shares")
    
    # Simulate price changes and see hedge adjustments
    price_changes = [101.5, 103.0, 101.0, 98.5, 97.0, 99.0]
    
    print("\nSimulating price changes:")
    for i, new_price in enumerate(price_changes):
        print(f"\nDay {i+1}: Underlying price ${new_price:.2f}")
        
        # Update greeks based on new price (simplified)
        for leg in options_legs:
            # Simple delta adjustment based on price change
            if leg['option_type'] == 'call':
                leg['greeks']['delta'] += (new_price - underlying_price) * leg['greeks']['gamma']
            else:  # put
                leg['greeks']['delta'] -= (new_price - underlying_price) * leg['greeks']['gamma']
                
            # Ensure delta is within bounds
            if leg['option_type'] == 'call':
                leg['greeks']['delta'] = min(1.0, max(0.0, leg['greeks']['delta']))
            else:  # put
                leg['greeks']['delta'] = max(-1.0, min(0.0, leg['greeks']['delta']))
                
        # Update position with new price and greeks
        result = gamma_scalper.update_position(
            position_id, new_price, [leg['greeks'] for leg in options_legs])
        
        # Display results
        if result['rebalance_needed']:
            print(f"Rebalance needed: Adjust hedge by {result['hedge_adjustment']} shares")
        else:
            print("No rebalance needed")
            
        print(f"Current delta: {result['position']['position_greeks']['delta']:.4f}")
        print(f"Current hedge: {result['position']['hedge_quantity']} shares")
        print(f"Realized P&L from gamma scalping: ${result['gamma_scalping_pnl']:.2f}")
        
        # Update underlying price for next iteration
        underlying_price = new_price
    
    # Get position analytics
    analytics = gamma_scalper.get_position_analytics(position_id)
    
    print("\nPosition analytics:")
    print(f"Realized P&L from gamma scalping: ${analytics['realized_pnl']:.2f}")
    
    # Plot P&L by price
    prices = sorted(analytics['pnl_by_price'].keys())
    pnls = [analytics['pnl_by_price'][p] for p in prices]
    
    plt.figure(figsize=(10, 6))
    plt.plot(prices, pnls)
    plt.xlabel('Underlying Price')
    plt.ylabel('Projected P&L')
    plt.title('Projected P&L by Underlying Price')
    plt.grid(True)
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.axvline(analytics['current_price'], color='red', linestyle='--', alpha=0.3, 
               label=f'Current price: ${analytics["current_price"]:.2f}')
    plt.legend()
    plt.show()

def main():
    """Run all demos."""
    print("Advanced Options Strategies Demo")
    print("================================")
    
    # Demo IV surface analysis
    demo_iv_surface()
    
    # Demo strike selection
    demo_strike_selection()
    
    # Demo position management
    demo_position_management()
    
    # Demo gamma scalping
    demo_gamma_scalping()
    
    print("\nAll demos completed!")

if __name__ == "__main__":
    main() 