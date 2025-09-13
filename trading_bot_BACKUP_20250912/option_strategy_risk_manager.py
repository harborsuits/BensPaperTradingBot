"""
Options Strategy Risk Management Extension
This implementation provides options strategy analysis for multi-leg positions
including vertical spreads, iron condors, butterflies, calendar spreads, etc.
"""

import math
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

from trading_bot.options_risk_manager import (
    OptionsRiskManager,
    OptionPosition,
    OptionType,
    OptionStrategy,
    PositionSide
)


class OptionStrategyRiskManager(OptionsRiskManager):
    """
    Manages risk for multi-leg option strategies
    Extends the base OptionsRiskManager with strategy-specific functionality
    """
    
    def __init__(self, market_data, risk_manager, config=None):
        super().__init__(market_data, risk_manager, config)
        self.strategy_positions = {}  # Dictionary to store strategy legs
    
    def generate_strategy_id(
        self,
        underlying_symbol: str,
        strategy: OptionStrategy,
        legs: List[OptionPosition]
    ) -> str:
        """Create a unique strategy ID"""
        timestamp = datetime.now().timestamp()
        leg_info = "_".join([
            f"{leg.option_type.value}-{leg.strike_price}-{leg.expiration.isoformat()}"
            for leg in legs
        ])
        return f"{underlying_symbol}_{strategy.value}_{leg_info}_{int(timestamp)}"
    
    def add_strategy(
        self,
        underlying_symbol: str,
        strategy: OptionStrategy,
        legs: List[OptionPosition]
    ) -> Dict:
        """
        Add a multi-leg option strategy to the manager
        
        Args:
            underlying_symbol: Symbol of the underlying asset
            strategy: Type of option strategy
            legs: Array of option positions that make up the strategy
            
        Returns:
            Dictionary with strategy ID and risk analysis
        """
        # Generate strategy ID
        strategy_id = self.generate_strategy_id(underlying_symbol, strategy, legs)
        
        # Store strategy
        self.strategy_positions[strategy_id] = legs.copy()
        
        # Calculate strategy risk metrics
        return self.analyze_strategy(strategy_id)
    
    def analyze_strategy(self, strategy_id: str) -> Dict:
        """
        Analyze risk metrics for a multi-leg option strategy
        
        Args:
            strategy_id: ID of the strategy to analyze
            
        Returns:
            Complete risk analysis for the strategy
        """
        if strategy_id not in self.strategy_positions or not self.strategy_positions[strategy_id]:
            raise ValueError(f"Strategy with ID {strategy_id} not found")
        
        legs = self.strategy_positions[strategy_id]
        underlying_symbol = legs[0].underlying_symbol
        underlying_price = self.market_data.get_spot_price(underlying_symbol)
        
        # Initialize net Greeks
        net_greeks = self.OptionGreeks()
        
        # Calculate net Greeks across all legs
        for leg in legs:
            result = self.calculate_black_scholes(
                underlying_price,
                leg.strike_price,
                self.calculate_time_to_expiry(leg.expiration),
                leg.implied_volatility,
                self.risk_free_rate,
                leg.option_type
            )
            
            greeks = result["greeks"]
            
            position_multiplier = 1 if leg.side == PositionSide.LONG else -1
            contract_multiplier = leg.contract_size * leg.quantity * position_multiplier
            
            net_greeks.delta += greeks.delta * contract_multiplier
            net_greeks.gamma += greeks.gamma * contract_multiplier
            net_greeks.theta += greeks.theta * contract_multiplier
            net_greeks.vega += greeks.vega * contract_multiplier
            net_greeks.rho += greeks.rho * contract_multiplier
        
        # Calculate max gain, max loss, and breakevens based on strategy type
        max_loss = 0.0
        max_gain = 0.0
        breakevens = []
        probability_of_profit = 0.0
        
        # Get the strategy type from the legs
        strategy_type = self.get_strategy_type(legs)
        
        # Calculate risk metrics based on strategy type
        if strategy_type == OptionStrategy.VERTICAL_SPREAD:
            result = self.analyze_vertical_spread(legs, underlying_price)
            max_loss = result["max_loss"]
            max_gain = result["max_gain"]
            breakevens = result["breakevens"]
            probability_of_profit = result["pop"]
        
        elif strategy_type == OptionStrategy.IRON_CONDOR:
            result = self.analyze_iron_condor(legs, underlying_price)
            max_loss = result["max_loss"]
            max_gain = result["max_gain"]
            breakevens = result["breakevens"]
            probability_of_profit = result["pop"]
        
        elif strategy_type == OptionStrategy.BUTTERFLY:
            result = self.analyze_butterfly(legs, underlying_price)
            max_loss = result["max_loss"]
            max_gain = result["max_gain"]
            breakevens = result["breakevens"]
            probability_of_profit = result["pop"]
        
        elif strategy_type == OptionStrategy.CALENDAR_SPREAD:
            result = self.analyze_calendar_spread(legs, underlying_price)
            max_loss = result["max_loss"]
            max_gain = result["max_gain"]
            breakevens = result["breakevens"]
            probability_of_profit = result["pop"]
        
        elif strategy_type == OptionStrategy.STRADDLE:
            result = self.analyze_straddle(legs, underlying_price)
            max_loss = result["max_loss"]
            max_gain = result["max_gain"]
            breakevens = result["breakevens"]
            probability_of_profit = result["pop"]
        
        elif strategy_type == OptionStrategy.STRANGLE:
            result = self.analyze_strangle(legs, underlying_price)
            max_loss = result["max_loss"]
            max_gain = result["max_gain"]
            breakevens = result["breakevens"]
            probability_of_profit = result["pop"]
        
        else:
            # For single leg or custom strategies, calculate using simulation
            result = self.simulate_strategy_outcomes(legs, underlying_price)
            max_loss = result["max_loss"]
            max_gain = result["max_gain"]
            breakevens = result["breakevens"]
            probability_of_profit = result["probability_of_profit"]
        
        return {
            "strategy_id": strategy_id,
            "max_loss": max_loss,
            "max_gain": max_gain,
            "breakevens": breakevens,
            "probability_of_profit": probability_of_profit,
            "net_greeks": net_greeks
        }
    
    def get_strategy_type(self, legs: List[OptionPosition]) -> OptionStrategy:
        """Determine the type of option strategy from the legs"""
        if len(legs) == 1:
            return OptionStrategy.SINGLE_LEG
        
        if len(legs) == 2:
            leg1, leg2 = legs
            
            # Check for vertical spread
            if (leg1.option_type == leg2.option_type and
                leg1.side != leg2.side and
                leg1.strike_price != leg2.strike_price and
                self.is_same_expiration(leg1.expiration, leg2.expiration)):
                return OptionStrategy.VERTICAL_SPREAD
            
            # Check for calendar spread
            if (leg1.option_type == leg2.option_type and
                leg1.side != leg2.side and
                leg1.strike_price == leg2.strike_price and
                not self.is_same_expiration(leg1.expiration, leg2.expiration)):
                return OptionStrategy.CALENDAR_SPREAD
            
            # Check for straddle
            if (leg1.option_type != leg2.option_type and
                leg1.side == leg2.side and
                leg1.strike_price == leg2.strike_price and
                self.is_same_expiration(leg1.expiration, leg2.expiration)):
                return OptionStrategy.STRADDLE
            
            # Check for strangle
            if (leg1.option_type != leg2.option_type and
                leg1.side == leg2.side and
                leg1.strike_price != leg2.strike_price and
                self.is_same_expiration(leg1.expiration, leg2.expiration)):
                return OptionStrategy.STRANGLE
        
        if len(legs) == 3:
            # Check for butterfly
            strikes = sorted([leg.strike_price for leg in legs])
            unique_strikes = set(strikes)
            
            if (len(unique_strikes) == 3 and
                abs((strikes[1] - strikes[0]) - (strikes[2] - strikes[1])) < 0.01 and
                all(self.is_same_expiration(leg.expiration, legs[0].expiration) for leg in legs)):
                return OptionStrategy.BUTTERFLY
        
        if len(legs) == 4:
            # Check for iron condor
            call_legs = [leg for leg in legs if leg.option_type == OptionType.CALL]
            put_legs = [leg for leg in legs if leg.option_type == OptionType.PUT]
            
            if (len(call_legs) == 2 and
                len(put_legs) == 2 and
                len([leg for leg in call_legs if leg.side == PositionSide.LONG]) == 1 and
                len([leg for leg in call_legs if leg.side == PositionSide.SHORT]) == 1 and
                len([leg for leg in put_legs if leg.side == PositionSide.LONG]) == 1 and
                len([leg for leg in put_legs if leg.side == PositionSide.SHORT]) == 1 and
                all(self.is_same_expiration(leg.expiration, legs[0].expiration) for leg in legs)):
                return OptionStrategy.IRON_CONDOR
        
        # If we can't identify a specific strategy
        return OptionStrategy.SINGLE_LEG
    
    def is_same_expiration(self, date1: datetime, date2: datetime) -> bool:
        """Check if two expiration dates are for the same day"""
        return date1.date() == date2.date()
    
    def analyze_vertical_spread(
        self,
        legs: List[OptionPosition],
        underlying_price: float
    ) -> Dict:
        """Analyze risk metrics for a vertical spread strategy"""
        # Sort by strike price
        legs = sorted(legs, key=lambda leg: leg.strike_price)
        
        leg1, leg2 = legs
        strike1 = leg1.strike_price
        strike2 = leg2.strike_price
        
        # Calculate option prices
        result1 = self.calculate_black_scholes(
            underlying_price,
            strike1,
            self.calculate_time_to_expiry(leg1.expiration),
            leg1.implied_volatility,
            self.risk_free_rate,
            leg1.option_type
        )
        
        result2 = self.calculate_black_scholes(
            underlying_price,
            strike2,
            self.calculate_time_to_expiry(leg2.expiration),
            leg2.implied_volatility,
            self.risk_free_rate,
            leg2.option_type
        )
        
        price1 = result1["price"]
        price2 = result2["price"]
        
        multiplier1 = -1 if leg1.side == PositionSide.LONG else 1
        multiplier2 = -1 if leg2.side == PositionSide.LONG else 1
        
        # Calculate net premium
        net_premium = (price1 * multiplier1 + price2 * multiplier2) * leg1.contract_size
        
        max_loss = 0.0
        max_gain = 0.0
        breakeven = 0.0
        pop = 0.0
        
        width = abs(strike2 - strike1) * leg1.contract_size
        
        if leg1.option_type == OptionType.CALL:
            if leg1.side == PositionSide.LONG and leg2.side == PositionSide.SHORT:
                # Bull call spread
                max_loss = -net_premium
                max_gain = width + net_premium
                breakeven = strike1 - net_premium / leg1.contract_size
                
                # Calculate probability of profit
                std_dev = underlying_price * leg1.implied_volatility * math.sqrt(
                    self.calculate_time_to_expiry(leg1.expiration)
                )
                pop = 1.0 - self.normal_cdf((breakeven - underlying_price) / std_dev)
            else:
                # Bear call spread
                max_loss = width + net_premium
                max_gain = -net_premium
                breakeven = strike1 + (width + net_premium) / leg1.contract_size
                
                # Calculate probability of profit
                std_dev = underlying_price * leg1.implied_volatility * math.sqrt(
                    self.calculate_time_to_expiry(leg1.expiration)
                )
                pop = self.normal_cdf((breakeven - underlying_price) / std_dev)
        else:
            if leg1.side == PositionSide.LONG and leg2.side == PositionSide.SHORT:
                # Bear put spread
                max_loss = -net_premium
                max_gain = width + net_premium
                breakeven = strike2 + net_premium / leg1.contract_size
                
                # Calculate probability of profit
                std_dev = underlying_price * leg1.implied_volatility * math.sqrt(
                    self.calculate_time_to_expiry(leg1.expiration)
                )
                pop = self.normal_cdf((breakeven - underlying_price) / std_dev)
            else:
                # Bull put spread
                max_loss = width + net_premium
                max_gain = -net_premium
                breakeven = strike2 - (width + net_premium) / leg1.contract_size
                
                # Calculate probability of profit
                std_dev = underlying_price * leg1.implied_volatility * math.sqrt(
                    self.calculate_time_to_expiry(leg1.expiration)
                )
                pop = 1.0 - self.normal_cdf((breakeven - underlying_price) / std_dev)
        
        return {
            "max_loss": max_loss,
            "max_gain": max_gain,
            "breakevens": [breakeven],
            "pop": pop
        }
    
    def analyze_iron_condor(
        self,
        legs: List[OptionPosition],
        underlying_price: float
    ) -> Dict:
        """Analyze risk metrics for an iron condor strategy"""
        # Separate puts and calls
        calls = sorted(
            [leg for leg in legs if leg.option_type == OptionType.CALL],
            key=lambda leg: leg.strike_price
        )
        
        puts = sorted(
            [leg for leg in legs if leg.option_type == OptionType.PUT],
            key=lambda leg: leg.strike_price
        )
        
        # Organize call legs
        long_call = next((leg for leg in calls if leg.side == PositionSide.LONG), None)
        short_call = next((leg for leg in calls if leg.side == PositionSide.SHORT), None)
        
        # Organize put legs
        short_put = next((leg for leg in puts if leg.side == PositionSide.SHORT), None)
        long_put = next((leg for leg in puts if leg.side == PositionSide.LONG), None)
        
        # Calculate option prices and find net premium
        prices = []
        net_premium = 0.0
        contract_size = legs[0].contract_size
        
        for leg in legs:
            result = self.calculate_black_scholes(
                underlying_price,
                leg.strike_price,
                self.calculate_time_to_expiry(leg.expiration),
                leg.implied_volatility,
                self.risk_free_rate,
                leg.option_type
            )
            
            price = result["price"]
            prices.append(price)
            multiplier = -1 if leg.side == PositionSide.LONG else 1
            net_premium += price * multiplier * contract_size
        
        # Calculate max gain and max loss
        call_width = (long_call.strike_price - short_call.strike_price) * contract_size
        put_width = (short_put.strike_price - long_put.strike_price) * contract_size
        
        max_gain = net_premium
        max_loss = max(call_width, put_width) - net_premium
        
        # Calculate breakeven points
        lower_breakeven = short_put.strike_price - (net_premium / contract_size)
        upper_breakeven = short_call.strike_price + (net_premium / contract_size)
        
        # Calculate probability of profit
        time_to_expiry = self.calculate_time_to_expiry(legs[0].expiration)
        avg_iv = sum(leg.implied_volatility for leg in legs) / len(legs)
        std_dev = underlying_price * avg_iv * math.sqrt(time_to_expiry)
        
        # Probability that price stays between short strikes
        prob_below_upper_breakeven = self.normal_cdf((upper_breakeven - underlying_price) / std_dev)
        prob_above_lower_breakeven = 1.0 - self.normal_cdf((lower_breakeven - underlying_price) / std_dev)
        pop = prob_below_upper_breakeven - (1.0 - prob_above_lower_breakeven)
        
        return {
            "max_loss": max_loss,
            "max_gain": max_gain,
            "breakevens": [lower_breakeven, upper_breakeven],
            "pop": pop
        }
    
    def analyze_butterfly(
        self,
        legs: List[OptionPosition],
        underlying_price: float
    ) -> Dict:
        """Analyze risk metrics for a butterfly spread strategy"""
        # Sort legs by strike price
        legs = sorted(legs, key=lambda leg: leg.strike_price)
        
        leg1, leg2, leg3 = legs
        contract_size = leg1.contract_size
        
        # Calculate option prices
        prices = []
        for leg in legs:
            result = self.calculate_black_scholes(
                underlying_price,
                leg.strike_price,
                self.calculate_time_to_expiry(leg.expiration),
                leg.implied_volatility,
                self.risk_free_rate,
                leg.option_type
            )
            prices.append(result["price"])
        
        # Calculate net premium
        net_premium = 0.0
        for i, leg in enumerate(legs):
            multiplier = -1 if leg.side == PositionSide.LONG else 1
            net_premium += prices[i] * multiplier * contract_size * leg.quantity
        
        # Calculate width of the spread
        width = (leg3.strike_price - leg1.strike_price) * contract_size
        
        # For a typical long butterfly
        max_loss = -net_premium
        max_gain = (leg2.strike_price - leg1.strike_price) * contract_size - net_premium
        
        # Calculate breakeven points
        lower_breakeven = leg1.strike_price + (net_premium / contract_size)
        upper_breakeven = leg3.strike_price - (net_premium / contract_size)
        
        # Calculate probability of profit
        time_to_expiry = self.calculate_time_to_expiry(leg1.expiration)
        avg_iv = sum(leg.implied_volatility for leg in legs) / len(legs)
        std_dev = underlying_price * avg_iv * math.sqrt(time_to_expiry)
        
        # Probability price ends between breakevens
        prob_below_upper_breakeven = self.normal_cdf((upper_breakeven - underlying_price) / std_dev)
        prob_above_lower_breakeven = 1.0 - self.normal_cdf((lower_breakeven - underlying_price) / std_dev)
        pop = prob_below_upper_breakeven - (1.0 - prob_above_lower_breakeven)
        
        return {
            "max_loss": max_loss,
            "max_gain": max_gain,
            "breakevens": [lower_breakeven, upper_breakeven],
            "pop": pop
        }
    
    def analyze_calendar_spread(
        self,
        legs: List[OptionPosition],
        underlying_price: float
    ) -> Dict:
        """Analyze risk metrics for a calendar spread strategy"""
        # Sort legs by expiration date
        legs = sorted(legs, key=lambda leg: leg.expiration)
        
        near_leg, far_leg = legs
        contract_size = near_leg.contract_size
        
        # Calculate option prices
        near_result = self.calculate_black_scholes(
            underlying_price,
            near_leg.strike_price,
            self.calculate_time_to_expiry(near_leg.expiration),
            near_leg.implied_volatility,
            self.risk_free_rate,
            near_leg.option_type
        )
        
        far_result = self.calculate_black_scholes(
            underlying_price,
            far_leg.strike_price,
            self.calculate_time_to_expiry(far_leg.expiration),
            far_leg.implied_volatility,
            self.risk_free_rate,
            far_leg.option_type
        )
        
        near_price = near_result["price"]
        far_price = far_result["price"]
        
        # Calculate net premium (typically debit for calendar spread)
        near_multiplier = -1 if near_leg.side == PositionSide.LONG else 1
        far_multiplier = -1 if far_leg.side == PositionSide.LONG else 1
        
        net_premium = (near_price * near_multiplier + far_price * far_multiplier) * contract_size
        
        # For calendar spread, max loss is typically the net premium paid
        max_loss = min(0.0, net_premium)
        
        # Max gain is theoretically unlimited but difficult to calculate precisely
        # We'll estimate using a set of price points at expiration of near leg
        
        # Simulate price range to find maximum profit
        strike_price = near_leg.strike_price  # Typically same strike for both legs
        price_range = 0.3  # Simulate 30% price range
        steps = 100
        
        max_gain = float('-inf')
        
        min_price = underlying_price * (1 - price_range)
        max_price = underlying_price * (1 + price_range)
        step_size = (max_price - min_price) / steps
        
        upper_breakeven = 0.0
        lower_breakeven = 0.0
        
        # Find breakevens and max gain through simulation
        for price in np.arange(min_price, max_price, step_size):
            # At near expiration, near option value
            near_option_value = self.calculate_option_value_at_expiration(
                price,
                near_leg.strike_price,
                near_leg.option_type
            )
            
            # Remaining time value for far option
            remaining_time = (self.calculate_time_to_expiry(far_leg.expiration) -
                             self.calculate_time_to_expiry(near_leg.expiration))
            
            far_option_value = self.calculate_black_scholes(
                price,
                far_leg.strike_price,
                remaining_time,
                far_leg.implied_volatility,
                self.risk_free_rate,
                far_leg.option_type
            )["price"]
            
            # Calculate P/L at this price
            pnl = (near_option_value * near_multiplier * -1 + far_option_value * far_multiplier * -1) * \
                 contract_size - net_premium
            
            # Track max gain
            if pnl > max_gain:
                max_gain = pnl
            
            # Track breakevens (where P/L crosses zero)
            if pnl >= 0 and lower_breakeven == 0 and price < strike_price:
                lower_breakeven = price
            elif pnl >= 0 and upper_breakeven == 0 and price > strike_price and lower_breakeven != 0:
                upper_breakeven = price
        
        # Adjust for case where maxGain is less than zero
        max_gain = max(0.0, max_gain)
        
        # Estimate probability of profit
        # Simplified approach - probability price stays near the strike at near expiration
        time_to_expiry = self.calculate_time_to_expiry(near_leg.expiration)
        avg_iv = (near_leg.implied_volatility + far_leg.implied_volatility) / 2
        std_dev = underlying_price * avg_iv * math.sqrt(time_to_expiry)
        
        # If we found breakevens, use them for POP calculation
        pop = 0.5  # Default
        
        if lower_breakeven != 0 and upper_breakeven != 0:
            prob_below_upper_breakeven = self.normal_cdf((upper_breakeven - underlying_price) / std_dev)
            prob_above_lower_breakeven = 1.0 - self.normal_cdf((lower_breakeven - underlying_price) / std_dev)
            pop = prob_below_upper_breakeven - (1.0 - prob_above_lower_breakeven)
        
        # If breakevens were not found, estimate them
        if lower_breakeven == 0 or upper_breakeven == 0:
            lower_breakeven = strike_price * 0.9
            upper_breakeven = strike_price * 1.1
        
        return {
            "max_loss": max_loss,
            "max_gain": max_gain,
            "breakevens": [lower_breakeven, upper_breakeven],
            "pop": pop
        }
    
    def calculate_option_value_at_expiration(
        self,
        spot_price: float,
        strike_price: float,
        option_type: OptionType
    ) -> float:
        """Calculate option value at expiration"""
        if option_type == OptionType.CALL:
            return max(0, spot_price - strike_price)
        else:
            return max(0, strike_price - spot_price)
    
    def analyze_straddle(
        self,
        legs: List[OptionPosition],
        underlying_price: float
    ) -> Dict:
        """
        Analyze a straddle strategy (long or short)
        Straddle consists of a call and put at the same strike, same expiration
        """
        if len(legs) != 2:
            raise ValueError("Straddle must have exactly 2 legs")
        
        # Find call and put legs
        call_leg = next((leg for leg in legs if leg.option_type == OptionType.CALL), None)
        put_leg = next((leg for leg in legs if leg.option_type == OptionType.PUT), None)
        
        if not call_leg or not put_leg:
            raise ValueError("Straddle must have both call and put legs")
        
        contract_size = call_leg.contract_size
        strike = call_leg.strike_price  # Same strike for both legs
        
        # Calculate option prices
        call_result = self.calculate_black_scholes(
            underlying_price,
            strike,
            self.calculate_time_to_expiry(call_leg.expiration),
            call_leg.implied_volatility,
            self.risk_free_rate,
            OptionType.CALL
        )
        
        put_result = self.calculate_black_scholes(
            underlying_price,
            strike,
            self.calculate_time_to_expiry(put_leg.expiration),
            put_leg.implied_volatility,
            self.risk_free_rate,
            OptionType.PUT
        )
        
        call_price = call_result["price"]
        put_price = put_result["price"]
        
        # Total premium
        total_premium = (call_price + put_price) * contract_size
        
        if call_leg.side == PositionSide.LONG:
            # Long straddle
            max_loss = total_premium
            max_gain = float('inf')  # Theoretically unlimited on either side
            
            # Breakeven points
            upper_breakeven = strike + (total_premium / contract_size)
            lower_breakeven = strike - (total_premium / contract_size)
            
            # Probability of profit - price moves enough in either direction
            time_to_expiry = self.calculate_time_to_expiry(call_leg.expiration)
            avg_iv = (call_leg.implied_volatility + put_leg.implied_volatility) / 2
            std_dev = underlying_price * avg_iv * math.sqrt(time_to_expiry)
            
            # Probability price is outside breakeven range
            prob_above_upper_breakeven = 1.0 - self.normal_cdf((upper_breakeven - underlying_price) / std_dev)
            prob_below_lower_breakeven = self.normal_cdf((lower_breakeven - underlying_price) / std_dev)
            pop = prob_above_upper_breakeven + prob_below_lower_breakeven
            
            return {
                "max_loss": max_loss,
                "max_gain": max_gain,
                "breakevens": [lower_breakeven, upper_breakeven],
                "pop": pop
            }
        else:
            # Short straddle
            max_gain = total_premium
            max_loss = float('inf')  # Theoretically unlimited on either side
            
            # Breakeven points
            upper_breakeven = strike + (total_premium / contract_size)
            lower_breakeven = strike - (total_premium / contract_size)
            
            # Probability of profit - price stays between breakevens
            time_to_expiry = self.calculate_time_to_expiry(call_leg.expiration)
            avg_iv = (call_leg.implied_volatility + put_leg.implied_volatility) / 2
            std_dev = underlying_price * avg_iv * math.sqrt(time_to_expiry)
            
            # Probability price is between breakevens
            prob_below_upper_breakeven = self.normal_cdf((upper_breakeven - underlying_price) / std_dev)
            prob_above_lower_breakeven = 1.0 - self.normal_cdf((lower_breakeven - underlying_price) / std_dev)
            pop = prob_below_upper_breakeven - (1.0 - prob_above_lower_breakeven)
            
            return {
                "max_loss": max_loss,
                "max_gain": max_gain,
                "breakevens": [lower_breakeven, upper_breakeven],
                "pop": pop
            }
    
    def analyze_strangle(
        self,
        legs: List[OptionPosition],
        underlying_price: float
    ) -> Dict:
        """
        Analyze a strangle strategy (long or short)
        Strangle consists of OTM call and OTM put at different strikes, same expiration
        """
        if len(legs) != 2:
            raise ValueError("Strangle must have exactly 2 legs")
        
        # Find call and put legs
        call_leg = next((leg for leg in legs if leg.option_type == OptionType.CALL), None)
        put_leg = next((leg for leg in legs if leg.option_type == OptionType.PUT), None)
        
        if not call_leg or not put_leg:
            raise ValueError("Strangle must have both call and put legs")
        
        contract_size = call_leg.contract_size
        
        # Calculate option prices
        call_result = self.calculate_black_scholes(
            underlying_price,
            call_leg.strike_price,
            self.calculate_time_to_expiry(call_leg.expiration),
            call_leg.implied_volatility,
            self.risk_free_rate,
            OptionType.CALL
        )
        
        put_result = self.calculate_black_scholes(
            underlying_price,
            put_leg.strike_price,
            self.calculate_time_to_expiry(put_leg.expiration),
            put_leg.implied_volatility,
            self.risk_free_rate,
            OptionType.PUT
        )
        
        call_price = call_result["price"]
        put_price = put_result["price"]
        
        # Total premium
        total_premium = (call_price + put_price) * contract_size
        
        if call_leg.side == PositionSide.LONG:
            # Long strangle
            max_loss = total_premium
            max_gain = float('inf')  # Theoretically unlimited on either side
            
            # Breakeven points
            upper_breakeven = call_leg.strike_price + (total_premium / contract_size)
            lower_breakeven = put_leg.strike_price - (total_premium / contract_size)
            
            # Probability of profit - price moves enough in either direction
            time_to_expiry = self.calculate_time_to_expiry(call_leg.expiration)
            avg_iv = (call_leg.implied_volatility + put_leg.implied_volatility) / 2
            std_dev = underlying_price * avg_iv * math.sqrt(time_to_expiry)
            
            # Probability price is outside breakeven range
            prob_above_upper_breakeven = 1.0 - self.normal_cdf((upper_breakeven - underlying_price) / std_dev)
            prob_below_lower_breakeven = self.normal_cdf((lower_breakeven - underlying_price) / std_dev)
            pop = prob_above_upper_breakeven + prob_below_lower_breakeven
            
            return {
                "max_loss": max_loss,
                "max_gain": max_gain,
                "breakevens": [lower_breakeven, upper_breakeven],
                "pop": pop
            }
        else:
            # Short strangle
            max_gain = total_premium
            max_loss = float('inf')  # Theoretically unlimited on either side
            
            # Breakeven points
            upper_breakeven = call_leg.strike_price + (total_premium / contract_size)
            lower_breakeven = put_leg.strike_price - (total_premium / contract_size)
            
            # Probability of profit - price stays between breakevens
            time_to_expiry = self.calculate_time_to_expiry(call_leg.expiration)
            avg_iv = (call_leg.implied_volatility + put_leg.implied_volatility) / 2
            std_dev = underlying_price * avg_iv * math.sqrt(time_to_expiry)
            
            # Probability price is between breakevens
            prob_below_upper_breakeven = self.normal_cdf((upper_breakeven - underlying_price) / std_dev)
            prob_above_lower_breakeven = 1.0 - self.normal_cdf((lower_breakeven - underlying_price) / std_dev)
            pop = prob_below_upper_breakeven - (1.0 - prob_above_lower_breakeven)
            
            return {
                "max_loss": max_loss,
                "max_gain": max_gain,
                "breakevens": [lower_breakeven, upper_breakeven],
                "pop": pop
            }
    
    def simulate_strategy_outcomes(
        self,
        legs: List[OptionPosition],
        current_price: float
    ) -> Dict:
        """
        Simulate strategy outcomes across a range of prices
        Used for complex or custom strategies
        """
        # Define price range to simulate (±30% of current price)
        price_range = 0.3
        min_price = current_price * (1 - price_range)
        max_price = current_price * (1 + price_range)
        steps = 500
        step_size = (max_price - min_price) / steps
        
        # Calculate total initial premium
        initial_premium = 0.0
        for leg in legs:
            result = self.calculate_black_scholes(
                current_price,
                leg.strike_price,
                self.calculate_time_to_expiry(leg.expiration),
                leg.implied_volatility,
                self.risk_free_rate,
                leg.option_type
            )
            
            price = result["price"]
            multiplier = -1 if leg.side == PositionSide.LONG else 1
            initial_premium += price * multiplier * leg.contract_size * leg.quantity
        
        # Track key metrics
        max_loss = float('inf')
        max_gain = float('-inf')
        payoff_at_prices = []
        
        # Simulate at various expiration prices
        for price in np.arange(min_price, max_price, step_size):
            payoff = initial_premium
            
            # Calculate payoff for each leg at this price
            for leg in legs:
                expiration_value = self.calculate_option_value_at_expiration(
                    price,
                    leg.strike_price,
                    leg.option_type
                )
                
                multiplier = 1 if leg.side == PositionSide.LONG else -1
                payoff += expiration_value * multiplier * leg.contract_size * leg.quantity
            
            # Track this price point
            payoff_at_prices.append({"price": price, "payoff": payoff})
            
            # Update max loss and max gain
            if payoff < 0 and abs(payoff) < max_loss:
                max_loss = abs(payoff)
            
            if payoff > 0 and payoff > max_gain:
                max_gain = payoff
        
        # Find breakeven points (where payoff crosses zero)
        breakevens = []
        
        for i in range(1, len(payoff_at_prices)):
            prev = payoff_at_prices[i - 1]
            current = payoff_at_prices[i]
            
            # If sign changed, interpolate to find breakeven
            if (prev["payoff"] < 0 and current["payoff"] > 0) or (prev["payoff"] > 0 and current["payoff"] < 0):
                ratio = abs(prev["payoff"]) / (abs(prev["payoff"]) + abs(current["payoff"]))
                breakeven = prev["price"] + ratio * (current["price"] - prev["price"])
                breakevens.append(breakeven)
        
        # Calculate probability of profit
        # Use average implied volatility across all legs
        avg_iv = sum(leg.implied_volatility for leg in legs) / len(legs)
        time_to_expiry = self.calculate_time_to_expiry(legs[0].expiration)
        std_dev = current_price * avg_iv * math.sqrt(time_to_expiry)
        
        probability_of_profit = 0.0
        
        if len(breakevens) == 0:
            # No breakevens - either always profitable or always losing
            probability_of_profit = 1.0 if payoff_at_prices[0]["payoff"] > 0 else 0.0
        elif len(breakevens) == 1:
            # One breakeven - check if profitable above or below
            mid_point_index = len(payoff_at_prices) // 2
            is_lower_half_profitable = payoff_at_prices[0]["payoff"] > 0
            
            if is_lower_half_profitable:
                # Profitable below breakeven
                probability_of_profit = self.normal_cdf((breakevens[0] - current_price) / std_dev)
            else:
                # Profitable above breakeven
                probability_of_profit = 1.0 - self.normal_cdf((breakevens[0] - current_price) / std_dev)
        else:
            # Multiple breakevens - more complex calculation
            # Sort breakevens
            breakevens.sort()
            
            # Sample a point midway between each consecutive breakeven pair
            test_points = []
            for i in range(len(breakevens) - 1):
                test_points.append((breakevens[i] + breakevens[i + 1]) / 2)
            
            # Add test points below first and above last breakeven
            test_points.insert(0, min_price)
            test_points.append(max_price)
            
            # Test each region for profitability
            profitable_regions = []
            
            for i, test_price in enumerate(test_points):
                # Calculate payoff at this test price
                payoff = initial_premium
                for leg in legs:
                    expiration_value = self.calculate_option_value_at_expiration(
                        test_price,
                        leg.strike_price,
                        leg.option_type
                    )
                    
                    multiplier = 1 if leg.side == PositionSide.LONG else -1
                    payoff += expiration_value * multiplier * leg.contract_size * leg.quantity
                
                # If profitable, include this region
                if payoff > 0:
                    if i == 0:
                        # Below first breakeven to first breakeven
                        profitable_regions.append((min_price, breakevens[0]))
                    elif i == len(test_points) - 1:
                        # Last breakeven to above last breakeven
                        profitable_regions.append((breakevens[-1], max_price))
                    else:
                        # Between two breakevens
                        profitable_regions.append((breakevens[i - 1], breakevens[i]))
            
            # Calculate probability for each profitable region
            for lower, upper in profitable_regions:
                if lower == min_price:
                    # Region below first breakeven
                    probability_of_profit += self.normal_cdf((upper - current_price) / std_dev)
                elif upper == max_price:
                    # Region above last breakeven
                    probability_of_profit += 1.0 - self.normal_cdf((lower - current_price) / std_dev)
                else:
                    # Region between two breakevens
                    probability_of_profit += (self.normal_cdf((upper - current_price) / std_dev) -
                                             self.normal_cdf((lower - current_price) / std_dev))
        
        # Finalize results
        if max_loss == float('inf'):
            max_loss = -initial_premium  # Default to premium paid if no max loss found
        if max_gain == float('-inf'):
            max_gain = initial_premium  # Default to premium received if no max gain found
        
        return {
            "max_loss": max_loss,
            "max_gain": max_gain,
            "breakevens": breakevens,
            "probability_of_profit": probability_of_profit
        } 