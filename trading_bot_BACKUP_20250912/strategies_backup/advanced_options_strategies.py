#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Options Trading Strategies Module

This module extends the strategy rotation system with advanced options trading capabilities,
including dynamic implied volatility surface analysis, automatic roll management,
volatility-based strike selection, and gamma scalping for delta-neutral positions.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import scipy.stats as stats
from scipy.interpolate import griddata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("advanced_options_strategies")


class ImpliedVolatilitySurface:
    """Analyze and model the implied volatility surface."""
    
    def __init__(self, min_days=7, max_days=120):
        """
        Initialize the IV surface analyzer.
        
        Args:
            min_days: Minimum days to expiration to consider
            max_days: Maximum days to expiration to consider
        """
        self.min_days = min_days
        self.max_days = max_days
        self.surface_data = None
        self.last_update = None
    
    def update_surface(self, options_chain):
        """
        Update the volatility surface model from options chain data.
        
        Args:
            options_chain: DataFrame containing options data with columns:
                          [expiration, strike, bid, ask, type, underlying_price, implied_volatility]
        """
        # Filter options data
        filtered_data = options_chain[
            (options_chain['days_to_expiration'] >= self.min_days) &
            (options_chain['days_to_expiration'] <= self.max_days)
        ].copy()
        
        if filtered_data.empty:
            logger.warning("No valid options data for IV surface calculation")
            return False
        
        # Calculate moneyness (strike / underlying price)
        filtered_data['moneyness'] = filtered_data['strike'] / filtered_data['underlying_price']
        
        # Create points for the surface
        points = filtered_data[['moneyness', 'days_to_expiration']].values
        values = filtered_data['implied_volatility'].values
        
        # Store the raw data for interpolation
        self.surface_data = {
            'points': points,
            'values': values,
            'min_moneyness': filtered_data['moneyness'].min(),
            'max_moneyness': filtered_data['moneyness'].max(),
            'min_dte': filtered_data['days_to_expiration'].min(),
            'max_dte': filtered_data['days_to_expiration'].max()
        }
        
        self.last_update = datetime.now()
        logger.info(f"IV Surface updated with {len(filtered_data)} data points")
        return True
    
    def get_iv(self, moneyness, days_to_expiration):
        """
        Get the implied volatility for a specific moneyness and days to expiration.
        
        Args:
            moneyness: Strike price / underlying price
            days_to_expiration: Days to option expiration
            
        Returns:
            float: Interpolated implied volatility
        """
        if self.surface_data is None:
            logger.error("IV Surface not initialized")
            return None
        
        # Check if within bounds
        if (moneyness < self.surface_data['min_moneyness'] or 
            moneyness > self.surface_data['max_moneyness'] or
            days_to_expiration < self.surface_data['min_dte'] or
            days_to_expiration > self.surface_data['max_dte']):
            logger.warning(f"Request for IV outside bounds: {moneyness}, {days_to_expiration}")
            
            # Clip to bounds
            moneyness = max(min(moneyness, self.surface_data['max_moneyness']), 
                            self.surface_data['min_moneyness'])
            days_to_expiration = max(min(days_to_expiration, self.surface_data['max_dte']), 
                                    self.surface_data['min_dte'])
        
        # Interpolate to get IV
        point = np.array([[moneyness, days_to_expiration]])
        iv = griddata(
            self.surface_data['points'],
            self.surface_data['values'],
            point,
            method='cubic'
        )[0]
        
        return iv
    
    def get_iv_skew(self, days_to_expiration):
        """
        Calculate the volatility skew for a specific expiration.
        
        Args:
            days_to_expiration: Days to expiration
            
        Returns:
            dict: Volatility skew metrics
        """
        if self.surface_data is None:
            logger.error("IV Surface not initialized")
            return None
        
        # Generate moneyness range
        moneyness_range = np.linspace(0.9, 1.1, 21)
        
        # Get IV for each moneyness
        ivs = [self.get_iv(m, days_to_expiration) for m in moneyness_range]
        
        # Calculate skew (slope of IV curve)
        skew = np.polyfit(moneyness_range, ivs, 1)[0]
        
        # Calculate convexity (curvature of IV curve)
        convexity = np.polyfit(moneyness_range, ivs, 2)[0]
        
        # Calculate put-call skew (difference between 0.95 and 1.05 moneyness)
        put_call_skew = self.get_iv(0.95, days_to_expiration) - self.get_iv(1.05, days_to_expiration)
        
        return {
            'skew': skew,
            'convexity': convexity,
            'put_call_skew': put_call_skew,
            'days_to_expiration': days_to_expiration
        }
    
    def visualize_surface(self, save_path=None):
        """
        Generate a 3D visualization of the volatility surface.
        
        Args:
            save_path: Path to save the plot (if None, display only)
            
        Returns:
            bool: True if successful
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            if self.surface_data is None:
                logger.error("IV Surface not initialized")
                return False
            
            # Create a grid for the surface
            moneyness_range = np.linspace(
                self.surface_data['min_moneyness'],
                self.surface_data['max_moneyness'],
                50
            )
            dte_range = np.linspace(
                self.surface_data['min_dte'],
                self.surface_data['max_dte'],
                50
            )
            
            X, Y = np.meshgrid(moneyness_range, dte_range)
            Z = np.zeros_like(X)
            
            # Calculate IV at each grid point
            for i in range(len(dte_range)):
                for j in range(len(moneyness_range)):
                    Z[i, j] = self.get_iv(X[i, j], Y[i, j])
            
            # Create the 3D plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            
            # Add labels and colorbar
            ax.set_xlabel('Moneyness (Strike/Spot)')
            ax.set_ylabel('Days to Expiration')
            ax.set_zlabel('Implied Volatility')
            ax.set_title('Implied Volatility Surface')
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"IV Surface plot saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing IV surface: {str(e)}")
            return False


class OptionsPositionManager:
    """Manage options positions including rolls and adjustments."""
    
    def __init__(self, roll_dte_threshold=7, profit_target_pct=50, max_loss_pct=100):
        """
        Initialize the options position manager.
        
        Args:
            roll_dte_threshold: Days to expiration threshold for rolling positions
            profit_target_pct: Profit target as percentage of max profit
            max_loss_pct: Maximum loss threshold as percentage of max loss
        """
        self.roll_dte_threshold = roll_dte_threshold
        self.profit_target_pct = profit_target_pct
        self.max_loss_pct = max_loss_pct
        self.positions = {}
        
    def add_position(self, position_id, position_data):
        """
        Add a new options position to manage.
        
        Args:
            position_id: Unique identifier for the position
            position_data: Dictionary containing position details:
                          {
                              'strategy': Strategy name,
                              'type': Position type (e.g., 'iron_condor', 'vertical_spread'),
                              'legs': [
                                  {
                                      'option_type': 'call' or 'put',
                                      'strike': Strike price,
                                      'expiration': Expiration date,
                                      'quantity': Number of contracts,
                                      'position': 'long' or 'short'
                                  },
                                  ...
                              ],
                              'underlying': Underlying symbol,
                              'entry_date': Entry date,
                              'entry_price': Net debit/credit,
                              'max_profit': Maximum profit,
                              'max_loss': Maximum loss
                          }
        """
        self.positions[position_id] = position_data
        logger.info(f"Added position {position_id} to manager: {position_data['type']} on {position_data['underlying']}")
        
    def remove_position(self, position_id):
        """
        Remove a position from management.
        
        Args:
            position_id: Position identifier
            
        Returns:
            bool: True if position was found and removed
        """
        if position_id in self.positions:
            position = self.positions.pop(position_id)
            logger.info(f"Removed position {position_id}: {position['type']} on {position['underlying']}")
            return True
        return False
    
    def check_positions_for_rolls(self, current_date, options_chain=None):
        """
        Check all positions for potential rolls based on DTE and profit/loss.
        
        Args:
            current_date: Current date
            options_chain: Optional options chain data for new positions
            
        Returns:
            list: Positions that need to be rolled
        """
        positions_to_roll = []
        
        for position_id, position in self.positions.items():
            # Get the earliest expiration among all legs
            earliest_exp = min([leg['expiration'] for leg in position['legs']])
            
            # Calculate days to expiration
            dte = (earliest_exp - current_date).days
            
            # Check if position meets roll criteria
            if dte <= self.roll_dte_threshold:
                positions_to_roll.append({
                    'position_id': position_id,
                    'position': position,
                    'reason': 'dte',
                    'dte': dte
                })
                logger.info(f"Position {position_id} flagged for roll due to DTE: {dte}")
            
            # Add additional criteria for early rolling (e.g., profit target reached)
            # This would require current price data which is not provided in this example
        
        return positions_to_roll
    
    def generate_roll_recommendation(self, position_to_roll, options_chain, iv_surface):
        """
        Generate a recommendation for rolling an options position.
        
        Args:
            position_to_roll: Position information that needs to be rolled
            options_chain: Options chain data for new position
            iv_surface: Implied volatility surface
            
        Returns:
            dict: Roll recommendation
        """
        position = position_to_roll['position']
        position_type = position['type']
        
        # Default to rolling out 30 days
        target_dte = 30
        
        # Get current IV environment
        if iv_surface and 'days_to_expiration' in position_to_roll:
            current_dte = position_to_roll['dte']
            current_iv_skew = iv_surface.get_iv_skew(current_dte)
            target_iv_skew = iv_surface.get_iv_skew(target_dte)
        else:
            current_iv_skew = None
            target_iv_skew = None
        
        # Generate recommendations based on position type
        if position_type == 'iron_condor':
            recommendation = self._generate_iron_condor_roll(position, target_dte, options_chain, target_iv_skew)
        elif position_type == 'vertical_spread':
            recommendation = self._generate_vertical_spread_roll(position, target_dte, options_chain, target_iv_skew)
        elif position_type == 'calendar_spread':
            recommendation = self._generate_calendar_roll(position, target_dte, options_chain, target_iv_skew)
        else:
            recommendation = None
            logger.warning(f"Unsupported position type for rolling: {position_type}")
        
        if recommendation:
            recommendation['original_position'] = position
            recommendation['reason'] = position_to_roll['reason']
        
        return recommendation
    
    def _generate_iron_condor_roll(self, position, target_dte, options_chain, iv_skew):
        """
        Generate a roll recommendation for an iron condor.
        
        Args:
            position: Current position data
            target_dte: Target days to expiration
            options_chain: Options chain data
            iv_skew: IV skew data for target expiration
            
        Returns:
            dict: Roll recommendation
        """
        # Find the appropriate expiration date
        target_date = datetime.now() + timedelta(days=target_dte)
        
        # Filter options chain for the target expiration
        if options_chain is not None:
            # Find closest expiration
            expirations = sorted(options_chain['expiration'].unique())
            target_exp = min(expirations, key=lambda x: abs((x - target_date).days))
            filtered_chain = options_chain[options_chain['expiration'] == target_exp]
        else:
            # Mock data for example
            filtered_chain = None
        
        # Analyze the current position
        put_spreads = [leg for leg in position['legs'] if leg['option_type'] == 'put']
        call_spreads = [leg for leg in position['legs'] if leg['option_type'] == 'call']
        
        # Find current strikes
        put_short = next((leg['strike'] for leg in put_spreads if leg['position'] == 'short'), None)
        put_long = next((leg['strike'] for leg in put_spreads if leg['position'] == 'long'), None)
        call_short = next((leg['strike'] for leg in call_spreads if leg['position'] == 'short'), None)
        call_long = next((leg['strike'] for leg in call_spreads if leg['position'] == 'long'), None)
        
        # Calculate width of current spreads
        put_width = abs(put_long - put_short) if put_long and put_short else 0
        call_width = abs(call_long - call_short) if call_long and call_short else 0
        
        # Use the same width for the new position
        
        # If we have a real options chain, we could select optimal strikes
        # For now, use mock strikes for illustration
        underlying_price = 100  # Mock price
        
        # Adjust based on volatility skew if available
        if iv_skew:
            # If volatility is skewed to puts, widen the put wing
            skew_factor = 1.0
            if iv_skew['put_call_skew'] > 0.05:
                skew_factor = 1.2  # Move put strikes further OTM
            elif iv_skew['put_call_skew'] < -0.05:
                skew_factor = 0.8  # Move put strikes closer to ATM
                
            # Calculate new strikes based on skew
            new_put_short = underlying_price * (1 - 0.05 * skew_factor)
            new_put_long = new_put_short - put_width
            new_call_short = underlying_price * (1 + 0.05 / skew_factor)
            new_call_long = new_call_short + call_width
        else:
            # Default strikes if no IV data
            new_put_short = underlying_price * 0.95
            new_put_long = new_put_short - put_width
            new_call_short = underlying_price * 1.05
            new_call_long = new_call_short + call_width
        
        # Create the recommendation
        recommendation = {
            'type': 'iron_condor',
            'underlying': position['underlying'],
            'target_expiration': target_exp if 'target_exp' in locals() else target_date,
            'legs': [
                {'option_type': 'put', 'strike': new_put_long, 'position': 'long'},
                {'option_type': 'put', 'strike': new_put_short, 'position': 'short'},
                {'option_type': 'call', 'strike': new_call_short, 'position': 'short'},
                {'option_type': 'call', 'strike': new_call_long, 'position': 'long'}
            ],
            'estimated_credit': None,  # This would be calculated with real options data
            'width': {'put': put_width, 'call': call_width}
        }
        
        return recommendation
    
    def _generate_vertical_spread_roll(self, position, target_dte, options_chain, iv_skew):
        """
        Generate a roll recommendation for a vertical spread.
        
        Args:
            position: Current position data
            target_dte: Target days to expiration
            options_chain: Options chain data
            iv_skew: IV skew data for target expiration
            
        Returns:
            dict: Roll recommendation
        """
        # Similar implementation as iron condor but for vertical spreads
        # Simplified for this example
        
        # Find the appropriate expiration date
        target_date = datetime.now() + timedelta(days=target_dte)
        
        # Analyze the current position
        option_type = position['legs'][0]['option_type']  # 'call' or 'put'
        
        # Sort legs by strike
        legs_by_strike = sorted(position['legs'], key=lambda x: x['strike'])
        
        # For credit spreads, short leg is closer to ATM
        # For debit spreads, long leg is closer to ATM
        is_credit_spread = legs_by_strike[0]['position'] == 'short' if option_type == 'put' else legs_by_strike[1]['position'] == 'short'
        
        # Get current strikes
        low_strike = legs_by_strike[0]['strike']
        high_strike = legs_by_strike[1]['strike']
        spread_width = high_strike - low_strike
        
        # Mock underlying price
        underlying_price = 100
        
        # Create the recommendation
        recommendation = {
            'type': 'vertical_spread',
            'underlying': position['underlying'],
            'target_expiration': target_date,
            'option_type': option_type,
            'is_credit_spread': is_credit_spread,
            'spread_width': spread_width,
            'estimated_price': None  # Would be calculated with real options data
        }
        
        # Calculate new strikes based on the current underlying price
        # This is simplified and would normally use delta or other criteria
        if option_type == 'call':
            if is_credit_spread:
                # Credit call spread (bear call spread)
                short_strike = underlying_price * 1.05
                long_strike = short_strike + spread_width
            else:
                # Debit call spread (bull call spread)
                long_strike = underlying_price * 0.98
                short_strike = long_strike + spread_width
        else:  # put
            if is_credit_spread:
                # Credit put spread (bull put spread)
                short_strike = underlying_price * 0.95
                long_strike = short_strike - spread_width
            else:
                # Debit put spread (bear put spread)
                long_strike = underlying_price * 1.02
                short_strike = long_strike - spread_width
        
        recommendation['legs'] = [
            {'option_type': option_type, 'strike': min(long_strike, short_strike), 
             'position': 'long' if min(long_strike, short_strike) == long_strike else 'short'},
            {'option_type': option_type, 'strike': max(long_strike, short_strike), 
             'position': 'long' if max(long_strike, short_strike) == long_strike else 'short'}
        ]
        
        return recommendation
    
    def _generate_calendar_roll(self, position, target_dte, options_chain, iv_skew):
        """
        Generate a roll recommendation for a calendar spread.
        
        Args:
            position: Current position data
            target_dte: Target days to expiration
            options_chain: Options chain data
            iv_skew: IV skew data for target expiration
            
        Returns:
            dict: Roll recommendation
        """
        # Calendar spreads are typically rolled by closing the front month
        # and opening a new back month, keeping the long position in place
        
        # Find the appropriate expiration date for the new back month
        target_date = datetime.now() + timedelta(days=target_dte + 30)  # Add 30 days for back month
        
        # Get the option type and strike from the original position
        option_type = position['legs'][0]['option_type']
        strike = position['legs'][0]['strike']
        
        # Find the long (back month) leg
        long_leg = next((leg for leg in position['legs'] if leg['position'] == 'long'), None)
        
        if not long_leg:
            logger.error("No long leg found in calendar spread - invalid position")
            return None
        
        # Create the recommendation - keep the same strike for continuity
        recommendation = {
            'type': 'calendar_roll',
            'underlying': position['underlying'],
            'original_expiration': min([leg['expiration'] for leg in position['legs']]),
            'target_expiration': target_date,
            'strike': strike,
            'option_type': option_type,
            'estimated_debit': None  # Would be calculated with real options data
        }
        
        # We're closing the front month (short leg) and opening a new back month
        recommendation['legs'] = [
            # The existing long leg stays in place
            {'option_type': option_type, 'strike': strike, 'position': 'long', 
             'expiration': long_leg['expiration'], 'action': 'keep'},
            # Close the existing short leg
            {'option_type': option_type, 'strike': strike, 'position': 'short', 
             'expiration': min([leg['expiration'] for leg in position['legs']]), 'action': 'close'},
            # Open a new short leg
            {'option_type': option_type, 'strike': strike, 'position': 'short', 
             'expiration': target_date, 'action': 'open'}
        ]
        
        return recommendation 


class VolatilityBasedStrikeSelector:
    """Select option strikes based on volatility metrics."""
    
    def __init__(self, iv_surface=None):
        """
        Initialize the strike selector.
        
        Args:
            iv_surface: Optional implied volatility surface
        """
        self.iv_surface = iv_surface
        
    def select_iron_condor_strikes(self, underlying_price, target_dte,
                                   delta_short_puts=0.16, delta_short_calls=0.16,
                                   width_puts=5, width_calls=5):
        """
        Select optimal strikes for an iron condor based on delta and volatility.
        
        Args:
            underlying_price: Current price of the underlying
            target_dte: Target days to expiration
            delta_short_puts: Target delta for short puts
            delta_short_calls: Target delta for short calls
            width_puts: Width of put spread in points or percent
            width_calls: Width of call spread in points or percent
            
        Returns:
            dict: Selected strikes
        """
        # Calculate implied volatility for the target DTE if IV surface available
        if self.iv_surface:
            atm_iv = self.iv_surface.get_iv(1.0, target_dte)
            iv_skew = self.iv_surface.get_iv_skew(target_dte)
            
            # Adjust deltas based on IV skew
            if iv_skew and iv_skew['put_call_skew'] > 0.05:
                # Higher IV on puts - move put strikes further OTM
                delta_short_puts = max(0.1, delta_short_puts - 0.03)
            elif iv_skew and iv_skew['put_call_skew'] < -0.05:
                # Higher IV on calls - move call strikes further OTM
                delta_short_calls = max(0.1, delta_short_calls - 0.03)
        else:
            # Use a default IV if no surface available
            atm_iv = 0.20
        
        # Calculate annual volatility
        annual_vol = atm_iv
        
        # Calculate daily volatility
        daily_vol = annual_vol / np.sqrt(252)
        
        # Calculate expected move to expiration
        expected_move = underlying_price * daily_vol * np.sqrt(target_dte)
        
        # Calculate strikes based on delta targets
        short_put_strike = self._calculate_strike_from_delta(
            underlying_price, target_dte, delta_short_puts, atm_iv, 'put')
        short_call_strike = self._calculate_strike_from_delta(
            underlying_price, target_dte, delta_short_calls, atm_iv, 'call')
        
        # Calculate long strikes based on width
        # Check if width is percentage or points
        if isinstance(width_puts, float) and width_puts < 1.0:
            long_put_strike = short_put_strike * (1 - width_puts)
        else:
            long_put_strike = short_put_strike - width_puts
            
        if isinstance(width_calls, float) and width_calls < 1.0:
            long_call_strike = short_call_strike * (1 + width_calls)
        else:
            long_call_strike = short_call_strike + width_calls
        
        return {
            'short_put_strike': round(short_put_strike, 2),
            'long_put_strike': round(long_put_strike, 2),
            'short_call_strike': round(short_call_strike, 2),
            'long_call_strike': round(long_call_strike, 2),
            'expected_move': round(expected_move, 2),
            'expected_range': [
                round(underlying_price - expected_move, 2),
                round(underlying_price + expected_move, 2)
            ]
        }
    
    def _calculate_strike_from_delta(self, underlying_price, days_to_expiration, 
                                    target_delta, implied_volatility, option_type):
        """
        Calculate the strike price that corresponds to a target delta.
        
        Args:
            underlying_price: Current price of the underlying
            days_to_expiration: Days to expiration
            target_delta: Target delta
            implied_volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            float: Strike price
        """
        # We need to find the strike price where the delta equals the target
        # This requires an iterative process since delta depends on strike
        
        # Convert days to years
        t = days_to_expiration / 365.0
        
        # Adjust target delta for puts (puts have negative delta)
        if option_type == 'put':
            target_delta = -target_delta
        
        # Initial strike guess - use expected move
        expected_move = underlying_price * implied_volatility * np.sqrt(t)
        
        if option_type == 'call':
            strike_guess = underlying_price + expected_move * abs(target_delta) * 2
        else:
            strike_guess = underlying_price - expected_move * abs(target_delta) * 2
        
        # Iteratively find strike with target delta
        max_iterations = 20
        tolerance = 0.001
        
        for i in range(max_iterations):
            # Calculate delta at current strike guess
            current_delta = self._calculate_option_delta(
                underlying_price, strike_guess, days_to_expiration,
                implied_volatility, option_type
            )
            
            # Check if we're close enough
            if abs(current_delta - target_delta) < tolerance:
                break
            
            # Adjust strike guess based on delta difference
            delta_diff = current_delta - target_delta
            adjustment = delta_diff * expected_move * 2
            
            # Adjust strike in the appropriate direction
            if option_type == 'call':
                strike_guess += adjustment
            else:
                strike_guess -= adjustment
        
        return strike_guess
    
    def _calculate_option_delta(self, underlying_price, strike_price, days_to_expiration,
                               implied_volatility, option_type):
        """
        Calculate the delta of an option.
        
        Args:
            underlying_price: Current price of the underlying
            strike_price: Strike price of the option
            days_to_expiration: Days to expiration
            implied_volatility: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            float: Option delta
        """
        # Convert days to years
        t = days_to_expiration / 365.0
        
        # Early exit for very short-dated options
        if t < 0.001:
            if option_type == 'call':
                return 1.0 if underlying_price > strike_price else 0.0
            else:  # put
                return -1.0 if underlying_price < strike_price else 0.0
        
        # Black-Scholes formula parameters
        # For simplicity, using 0 rate and dividend
        r = 0.0
        q = 0.0
        
        sigma = implied_volatility
        S = underlying_price
        K = strike_price
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        
        # Calculate delta
        if option_type == 'call':
            delta = np.exp(-q * t) * stats.norm.cdf(d1)
        else:  # put
            delta = -np.exp(-q * t) * stats.norm.cdf(-d1)
        
        return delta
    
    def calculate_option_greeks(self, underlying_price, strike_price, days_to_expiration,
                               implied_volatility, option_type, option_price=None):
        """
        Calculate all Greeks for an option.
        
        Args:
            underlying_price: Current price of the underlying
            strike_price: Strike price of the option
            days_to_expiration: Days to expiration
            implied_volatility: Implied volatility
            option_type: 'call' or 'put'
            option_price: Optional actual option price for calculating implied volatility
            
        Returns:
            dict: Option Greeks
        """
        # Convert days to years
        t = days_to_expiration / 365.0
        
        # Black-Scholes formula parameters
        # For simplicity, using 0 rate and dividend
        r = 0.0
        q = 0.0
        
        sigma = implied_volatility
        S = underlying_price
        K = strike_price
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        
        # Standard normal PDF and CDF
        n_d1 = stats.norm.pdf(d1)
        n_d2 = stats.norm.pdf(d2)
        N_d1 = stats.norm.cdf(d1)
        N_d2 = stats.norm.cdf(d2)
        N_neg_d1 = stats.norm.cdf(-d1)
        N_neg_d2 = stats.norm.cdf(-d2)
        
        # Calculate option price
        if option_type == 'call':
            price = S * np.exp(-q * t) * N_d1 - K * np.exp(-r * t) * N_d2
            delta = np.exp(-q * t) * N_d1
            theta = (-S * np.exp(-q * t) * n_d1 * sigma / (2 * np.sqrt(t)) - 
                     r * K * np.exp(-r * t) * N_d2 + 
                     q * S * np.exp(-q * t) * N_d1) / 365.0
        else:  # put
            price = K * np.exp(-r * t) * N_neg_d2 - S * np.exp(-q * t) * N_neg_d1
            delta = -np.exp(-q * t) * N_neg_d1
            theta = (-S * np.exp(-q * t) * n_d1 * sigma / (2 * np.sqrt(t)) + 
                     r * K * np.exp(-r * t) * N_neg_d2 - 
                     q * S * np.exp(-q * t) * N_neg_d1) / 365.0
        
        # Greeks that are the same for calls and puts
        gamma = np.exp(-q * t) * n_d1 / (S * sigma * np.sqrt(t))
        vega = S * np.exp(-q * t) * n_d1 * np.sqrt(t) / 100.0  # Divide by 100 for 1% change
        rho = (K * t * np.exp(-r * t) * (N_d2 if option_type == 'call' else -N_neg_d2)) / 100.0
        
        # Calculate theta for 1 day change
        
        # Return all Greeks
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho,
            'iv': implied_volatility
        } 


class GammaScalper:
    """
    Implements gamma scalping strategies for delta-neutral options positions.
    
    Gamma scalping is a sophisticated options trading technique that involves:
    1. Establishing a long gamma, delta-neutral position (typically long straddles/strangles)
    2. Dynamically hedging the position as the underlying price moves to capture gamma profits
    3. Managing overall position risk with regard to time decay and volatility changes
    """
    
    def __init__(self, rebalance_threshold=0.10, underlying_unit=100):
        """
        Initialize the gamma scalper.
        
        Args:
            rebalance_threshold: Delta threshold to trigger rebalancing (e.g., 0.10 = 10% of position)
            underlying_unit: Number of shares in one standard position (e.g., 100 shares per option)
        """
        self.rebalance_threshold = rebalance_threshold
        self.underlying_unit = underlying_unit
        self.positions = {}
        self.hedge_history = {}
        
    def add_position(self, position_id, options_legs, underlying_price):
        """
        Add a new position to gamma scalp.
        
        Args:
            position_id: Unique identifier for the position
            options_legs: List of option legs with greeks
                         [
                             {
                                 'option_type': 'call' or 'put',
                                 'strike': Strike price,
                                 'expiration': Expiration date,
                                 'quantity': Number of contracts,
                                 'greeks': {'delta': X, 'gamma': Y, ...}
                             },
                             ...
                         ]
            underlying_price: Current price of the underlying
            
        Returns:
            dict: Position data with initial hedge recommendation
        """
        # Calculate position greeks
        position_greeks = self._calculate_position_greeks(options_legs)
        
        # Calculate initial hedge
        hedge_quantity = self._calculate_hedge_quantity(position_greeks['delta'])
        
        # Store position
        position = {
            'options_legs': options_legs,
            'entry_price': underlying_price,
            'current_price': underlying_price,
            'position_greeks': position_greeks,
            'hedge_quantity': hedge_quantity,
            'hedge_history': [
                {
                    'timestamp': datetime.now(),
                    'underlying_price': underlying_price,
                    'position_delta': position_greeks['delta'],
                    'hedge_quantity': hedge_quantity,
                    'action': 'initial_hedge'
                }
            ],
            'realized_pnl': 0.0
        }
        
        self.positions[position_id] = position
        self.hedge_history[position_id] = position['hedge_history']
        
        logger.info(f"Added position {position_id} for gamma scalping with {len(options_legs)} legs")
        logger.info(f"Initial hedge: {hedge_quantity} shares at ${underlying_price:.2f}")
        
        return position
    
    def update_position(self, position_id, new_underlying_price, new_greeks=None):
        """
        Update position with new price and optionally new greeks.
        
        Args:
            position_id: Position identifier
            new_underlying_price: Current price of the underlying
            new_greeks: Optional updated greeks for each leg
            
        Returns:
            dict: Updated position with rebalance recommendation if needed
        """
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        old_price = position['current_price']
        price_change = new_underlying_price - old_price
        
        # Update position
        position['current_price'] = new_underlying_price
        
        # Update greeks if provided
        if new_greeks:
            for i, leg_greeks in enumerate(new_greeks):
                if i < len(position['options_legs']):
                    position['options_legs'][i]['greeks'] = leg_greeks
            
            # Recalculate position greeks
            position['position_greeks'] = self._calculate_position_greeks(position['options_legs'])
        
        # Calculate P&L from hedge if price has changed
        if abs(price_change) > 0.001 and position['hedge_quantity'] != 0:
            hedge_pnl = position['hedge_quantity'] * price_change
            position['realized_pnl'] += hedge_pnl
            logger.info(f"Realized P&L from hedge: ${hedge_pnl:.2f}, total: ${position['realized_pnl']:.2f}")
        
        # Check if rebalance is needed
        current_delta = position['position_greeks']['delta']
        abs_delta = abs(current_delta * self.underlying_unit)
        
        rebalance_needed = abs_delta > self.rebalance_threshold
        
        if rebalance_needed:
            # Calculate new hedge quantity
            new_hedge_quantity = self._calculate_hedge_quantity(current_delta)
            hedge_adjustment = new_hedge_quantity - position['hedge_quantity']
            
            # Update position
            position['hedge_quantity'] = new_hedge_quantity
            
            # Record hedge action
            position['hedge_history'].append({
                'timestamp': datetime.now(),
                'underlying_price': new_underlying_price,
                'position_delta': current_delta,
                'hedge_quantity': new_hedge_quantity,
                'hedge_adjustment': hedge_adjustment,
                'action': 'rebalance'
            })
            
            logger.info(f"Rebalance needed: Adjust hedge by {hedge_adjustment} shares at ${new_underlying_price:.2f}")
            
            # Update global hedge history
            self.hedge_history[position_id] = position['hedge_history']
            
            return {
                'position': position,
                'rebalance_needed': True,
                'hedge_adjustment': hedge_adjustment,
                'gamma_scalping_pnl': position['realized_pnl']
            }
        else:
            return {
                'position': position,
                'rebalance_needed': False,
                'gamma_scalping_pnl': position['realized_pnl']
            }
    
    def _calculate_position_greeks(self, options_legs):
        """
        Calculate aggregate Greeks for the entire position.
        
        Args:
            options_legs: List of option legs with greeks
            
        Returns:
            dict: Aggregate position Greeks
        """
        # Initialize position greeks
        position_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }
        
        # Sum up greeks from all legs
        for leg in options_legs:
            quantity = leg['quantity']
            greeks = leg['greeks']
            
            position_greeks['delta'] += greeks['delta'] * quantity
            position_greeks['gamma'] += greeks['gamma'] * quantity
            position_greeks['theta'] += greeks['theta'] * quantity
            position_greeks['vega'] += greeks['vega'] * quantity
        
        return position_greeks
    
    def _calculate_hedge_quantity(self, position_delta):
        """
        Calculate the hedge quantity to neutralize the position delta.
        
        Args:
            position_delta: Current position delta
            
        Returns:
            int: Number of shares to hold as a hedge (rounded to nearest 100)
        """
        # Delta hedging requires taking the opposite position in the underlying
        raw_hedge_quantity = -position_delta * self.underlying_unit
        
        # Round to nearest unit of underlying (typically 100 shares for options)
        hedge_quantity = round(raw_hedge_quantity / self.underlying_unit) * self.underlying_unit
        
        return hedge_quantity
    
    def get_position_analytics(self, position_id, underlying_price=None):
        """
        Get analytics for a gamma scalping position.
        
        Args:
            position_id: Position identifier
            underlying_price: Optional current price to use for analytics
            
        Returns:
            dict: Position analytics
        """
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return None
        
        position = self.positions[position_id]
        
        # Use provided price or current price
        price = underlying_price if underlying_price is not None else position['current_price']
        
        # Calculate expected position P&L at different price levels
        price_range = np.linspace(price * 0.9, price * 1.1, 21)
        pnl_by_price = {}
        
        for test_price in price_range:
            # Calculate option P&L (simplified)
            # In a real system, this would use proper option pricing model
            pnl = 0.0
            for leg in position['options_legs']:
                greeks = leg['greeks']
                quantity = leg['quantity']
                
                # Simplified P&L calculation - using delta and gamma approximation
                price_diff = test_price - price
                delta_pnl = greeks['delta'] * price_diff * quantity
                gamma_pnl = 0.5 * greeks['gamma'] * price_diff ** 2 * quantity
                
                pnl += delta_pnl + gamma_pnl
            
            # Add hedge P&L
            hedge_pnl = position['hedge_quantity'] * (test_price - price)
            
            # Total P&L for this price level
            total_pnl = pnl + hedge_pnl + position['realized_pnl']
            
            pnl_by_price[float(test_price)] = total_pnl
        
        # Construct analytics
        analytics = {
            'position_id': position_id,
            'current_price': price,
            'position_greeks': position['position_greeks'],
            'hedge_quantity': position['hedge_quantity'],
            'realized_pnl': position['realized_pnl'],
            'pnl_by_price': pnl_by_price,
            'breakeven_prices': self._calculate_breakeven_prices(pnl_by_price),
            'hedge_history': position['hedge_history']
        }
        
        return analytics
    
    def _calculate_breakeven_prices(self, pnl_by_price):
        """
        Calculate breakeven prices from a P&L by price mapping.
        
        Args:
            pnl_by_price: Dictionary mapping prices to P&L values
            
        Returns:
            list: Estimated breakeven prices
        """
        # Convert to arrays for interpolation
        prices = np.array(list(pnl_by_price.keys()))
        pnls = np.array(list(pnl_by_price.values()))
        
        # Find where P&L is close to zero
        breakeven_prices = []
        
        for i in range(len(pnls)-1):
            if (pnls[i] <= 0 and pnls[i+1] >= 0) or (pnls[i] >= 0 and pnls[i+1] <= 0):
                # Simple linear interpolation to find more precise breakeven
                x1, y1 = prices[i], pnls[i]
                x2, y2 = prices[i+1], pnls[i+1]
                
                # y = 0 occurs at:
                # x = x1 - y1 * (x2 - x1) / (y2 - y1)
                if y2 - y1 != 0:
                    breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                    breakeven_prices.append(breakeven)
        
        return sorted(breakeven_prices)


class AdvancedOptionsStrategyManager:
    """
    Manages and coordinates advanced options trading strategies.
    
    This class integrates the various options strategy components including:
    1. Implied volatility surface analysis
    2. Position management and rolling
    3. Volatility-based strike selection 
    4. Gamma scalping for delta-neutral positions
    """
    
    def __init__(self):
        """Initialize the advanced options strategy manager."""
        self.iv_surface = ImpliedVolatilitySurface()
        self.position_manager = OptionsPositionManager()
        self.strike_selector = VolatilityBasedStrikeSelector(iv_surface=self.iv_surface)
        self.gamma_scalper = GammaScalper()
        
        # Strategy performance tracking
        self.strategy_performance = {}
        
    def update_market_data(self, options_chain, underlying_prices=None):
        """
        Update market data for all strategy components.
        
        Args:
            options_chain: DataFrame with options chain data
            underlying_prices: Optional dict mapping symbols to prices
        """
        # Update IV surface
        self.iv_surface.update_surface(options_chain)
        
        # Update strike selector
        self.strike_selector.iv_surface = self.iv_surface
        
        # If we have underlying prices, update gamma scalping positions
        if underlying_prices:
            for position_id, position in self.gamma_scalper.positions.items():
                symbol = position['options_legs'][0]['symbol']
                if symbol in underlying_prices:
                    self.gamma_scalper.update_position(position_id, underlying_prices[symbol])
                    
    def create_volatility_strategy(self, strategy_type, underlying_symbol, 
                                 underlying_price, target_dte, **kwargs):
        """
        Create a volatility-based options strategy.
        
        Args:
            strategy_type: Type of strategy ('iron_condor', 'strangle', etc.)
            underlying_symbol: Symbol of the underlying asset
            underlying_price: Current price of the underlying
            target_dte: Target days to expiration
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            dict: Strategy definition with legs
        """
        if strategy_type == 'iron_condor':
            return self._create_iron_condor(underlying_symbol, underlying_price, target_dte, **kwargs)
        elif strategy_type == 'strangle':
            return self._create_strangle(underlying_symbol, underlying_price, target_dte, **kwargs)
        elif strategy_type == 'straddle':
            return self._create_straddle(underlying_symbol, underlying_price, target_dte, **kwargs)
        else:
            logger.error(f"Unsupported strategy type: {strategy_type}")
            return None
            
    def _create_iron_condor(self, underlying_symbol, underlying_price, target_dte,
                         delta_short_puts=0.16, delta_short_calls=0.16,
                         width_puts=5, width_calls=5):
        """Create an iron condor strategy."""
        # Use strike selector to choose optimal strikes
        strikes = self.strike_selector.select_iron_condor_strikes(
            underlying_price, target_dte, delta_short_puts, delta_short_calls, width_puts, width_calls)
        
        # Create strategy definition
        strategy = {
            'type': 'iron_condor',
            'underlying': underlying_symbol,
            'target_expiration': datetime.now() + timedelta(days=target_dte),
            'legs': [
                {'option_type': 'put', 'strike': strikes['long_put_strike'], 'position': 'long'},
                {'option_type': 'put', 'strike': strikes['short_put_strike'], 'position': 'short'},
                {'option_type': 'call', 'strike': strikes['short_call_strike'], 'position': 'short'},
                {'option_type': 'call', 'strike': strikes['long_call_strike'], 'position': 'long'}
            ],
            'width': {'put': width_puts, 'call': width_calls},
            'expected_move': strikes['expected_move']
        }
        
        return strategy
    
    def _create_strangle(self, underlying_symbol, underlying_price, target_dte,
                      delta_puts=0.16, delta_calls=0.16):
        """Create a strangle strategy."""
        # Calculate strikes based on delta targets
        atm_iv = self.iv_surface.get_iv(1.0, target_dte) if self.iv_surface else 0.20
        
        put_strike = self.strike_selector._calculate_strike_from_delta(
            underlying_price, target_dte, delta_puts, atm_iv, 'put')
        call_strike = self.strike_selector._calculate_strike_from_delta(
            underlying_price, target_dte, delta_calls, atm_iv, 'call')
        
        # Create strategy definition
        strategy = {
            'type': 'strangle',
            'underlying': underlying_symbol,
            'target_expiration': datetime.now() + timedelta(days=target_dte),
            'legs': [
                {'option_type': 'put', 'strike': round(put_strike, 2), 'position': 'long'},
                {'option_type': 'call', 'strike': round(call_strike, 2), 'position': 'long'}
            ],
            'width': call_strike - put_strike
        }
        
        return strategy
    
    def _create_straddle(self, underlying_symbol, underlying_price, target_dte):
        """Create a straddle strategy."""
        # Find the strike closest to the current price
        atm_strike = round(underlying_price, 0)
        
        # Create strategy definition
        strategy = {
            'type': 'straddle',
            'underlying': underlying_symbol,
            'target_expiration': datetime.now() + timedelta(days=target_dte),
            'legs': [
                {'option_type': 'put', 'strike': atm_strike, 'position': 'long'},
                {'option_type': 'call', 'strike': atm_strike, 'position': 'long'}
            ]
        }
        
        return strategy
    
    def check_positions_for_adjustments(self, current_date):
        """
        Check all positions for potential adjustments.
        
        Args:
            current_date: Current date
            
        Returns:
            list: Positions that need adjustment with recommendations
        """
        # Check for rolls
        positions_to_roll = self.position_manager.check_positions_for_rolls(current_date)
        
        # Generate roll recommendations
        adjustments = []
        
        for position_to_roll in positions_to_roll:
            recommendation = self.position_manager.generate_roll_recommendation(
                position_to_roll, None, self.iv_surface)
            
            if recommendation:
                adjustments.append({
                    'position_id': position_to_roll['position_id'],
                    'adjustment_type': 'roll',
                    'recommendation': recommendation
                })
        
        return adjustments 