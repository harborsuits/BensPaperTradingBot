"""
Options Risk Management Extension for MultiAssetAdapter
This implementation provides comprehensive options risk management capabilities
including Greeks calculations, position sizing, expiration management and IV risk handling
"""

import enum
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any


# Enums and Type Definitions
class AssetClass(enum.Enum):
    STOCKS = "STOCKS"
    OPTIONS = "OPTIONS"
    FUTURES = "FUTURES"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"


class PositionSide(enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OptionType(enum.Enum):
    CALL = "CALL"
    PUT = "PUT"


class OptionStrategy(enum.Enum):
    SINGLE_LEG = "SINGLE_LEG"
    VERTICAL_SPREAD = "VERTICAL_SPREAD"
    IRON_CONDOR = "IRON_CONDOR"
    BUTTERFLY = "BUTTERFLY"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    DIAGONAL_SPREAD = "DIAGONAL_SPREAD"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"


# Data Classes
class OptionGreeks:
    def __init__(
        self,
        delta: float = 0.0,
        gamma: float = 0.0,
        theta: float = 0.0,
        vega: float = 0.0,
        rho: float = 0.0,
        charm: float = None,
        vanna: float = None,
        volga: float = None
    ):
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho
        self.charm = charm
        self.vanna = vanna
        self.volga = volga

    def __repr__(self):
        return (
            f"OptionGreeks(delta={self.delta:.4f}, gamma={self.gamma:.4f}, "
            f"theta={self.theta:.4f}, vega={self.vega:.4f}, rho={self.rho:.4f})"
        )


class OptionDetails:
    def __init__(
        self,
        symbol: str,
        underlying_symbol: str,
        strike_price: float,
        expiration: datetime,
        option_type: OptionType,
        contract_size: int,
        implied_volatility: float,
        market_price: float,
        bid_price: float,
        ask_price: float,
        open_interest: int,
        volume: int,
        last_traded_price: float,
        delta: float = None,
        gamma: float = None,
        theta: float = None,
        vega: float = None,
        rho: float = None
    ):
        self.symbol = symbol
        self.underlying_symbol = underlying_symbol
        self.strike_price = strike_price
        self.expiration = expiration
        self.option_type = option_type
        self.contract_size = contract_size
        self.implied_volatility = implied_volatility
        self.market_price = market_price
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.open_interest = open_interest
        self.volume = volume
        self.last_traded_price = last_traded_price
        self.delta = delta
        self.gamma = gamma
        self.theta = theta
        self.vega = vega
        self.rho = rho


class OptionPosition:
    def __init__(
        self,
        id: str,
        symbol: str,
        underlying_symbol: str,
        side: PositionSide,
        quantity: int,
        entry_price: float,
        option_type: OptionType,
        strike_price: float,
        expiration: datetime,
        contract_size: int,
        implied_volatility: float
    ):
        self.id = id
        self.symbol = symbol
        self.underlying_symbol = underlying_symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price
        self.option_type = option_type
        self.strike_price = strike_price
        self.expiration = expiration
        self.contract_size = contract_size
        self.implied_volatility = implied_volatility


class OptionPositionRisk:
    def __init__(
        self,
        position: OptionPosition,
        greeks: OptionGreeks,
        days_to_expiration: int,
        time_value: float,
        intrinsic_value: float,
        iv_percentile: float,
        iv_rank: float,
        max_loss: float,
        max_gain: float,
        probability_of_profit: float,
        breakeven: List[float]
    ):
        self.position = position
        self.greeks = greeks
        self.days_to_expiration = days_to_expiration
        self.time_value = time_value
        self.intrinsic_value = intrinsic_value
        self.iv_percentile = iv_percentile
        self.iv_rank = iv_rank
        self.max_loss = max_loss
        self.max_gain = max_gain
        self.probability_of_profit = probability_of_profit
        self.breakeven = breakeven


# Main Options Risk Manager Class
class OptionsRiskManager:
    def __init__(
        self,
        market_data,
        risk_manager,
        config: Dict = None
    ):
        if config is None:
            config = {}
            
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.risk_factor = config.get("risk_factor", 0.02)
        self.iv_threshold = config.get("iv_threshold", 0.8)
        self.risk_free_rate = config.get("risk_free_rate", 0.05)
        self.iv_percentile_window = config.get("iv_percentile_window", 252)
        self.max_position_delta = config.get("max_position_delta", 500)
        self.max_position_gamma = config.get("max_position_gamma", 100)
        self.max_position_theta = config.get("max_position_theta", -500)
        self.max_position_vega = config.get("max_position_vega", 1000)
        self.dte_cutoff = config.get("dte_cutoff", 5)
        
        # Initialize portfolio Greeks tracker
        self.portfolio_greeks = OptionGreeks()
        
        # Map to track expiring options
        self.expiring_options = {}

    def calculate_black_scholes(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        iv: float,
        risk_free_rate: float = None,
        option_type: OptionType = None
    ) -> Dict:
        """
        Calculate Black-Scholes option pricing model and Greeks
        
        Args:
            spot_price: Current price of underlying asset
            strike_price: Strike price of the option
            time_to_expiry: Time to expiration (in years)
            iv: Implied volatility (decimal)
            risk_free_rate: Risk-free interest rate (decimal)
            option_type: Type of option (CALL or PUT)
            
        Returns:
            Dictionary with option price and Greeks
        """
        # Use default values if not provided
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        if option_type is None:
            option_type = OptionType.CALL
        
        # Handle expired/expiring options
        if time_to_expiry <= 0:
            if option_type == OptionType.CALL:
                intrinsic_value = max(0, spot_price - strike_price)
                return {
                    "price": intrinsic_value,
                    "greeks": OptionGreeks(
                        delta=1.0 if intrinsic_value > 0 else 0.0,
                        gamma=0.0,
                        theta=0.0,
                        vega=0.0,
                        rho=0.0
                    )
                }
            else:
                intrinsic_value = max(0, strike_price - spot_price)
                return {
                    "price": intrinsic_value,
                    "greeks": OptionGreeks(
                        delta=-1.0 if intrinsic_value > 0 else 0.0,
                        gamma=0.0,
                        theta=0.0,
                        vega=0.0,
                        rho=0.0
                    )
                }
        
        # Core Black-Scholes calculation
        sqrt_time = math.sqrt(time_to_expiry)
        iv_times_sqrt = iv * sqrt_time
        
        # Calculate d1 and d2
        d1 = (math.log(spot_price / strike_price) + 
              (risk_free_rate + 0.5 * iv * iv) * time_to_expiry) / iv_times_sqrt
        d2 = d1 - iv_times_sqrt
        
        # Standard normal CDF for d1 and d2
        if option_type == OptionType.CALL:
            nd1 = self.normal_cdf(d1)
            nd2 = self.normal_cdf(d2)
        else:
            nd1 = self.normal_cdf(-d1)
            nd2 = self.normal_cdf(-d2)
        
        # Standard normal PDF for d1
        pd1 = self.normal_pdf(d1)
        
        # Calculate option price
        if option_type == OptionType.CALL:
            price = spot_price * nd1 - strike_price * math.exp(-risk_free_rate * time_to_expiry) * nd2
        else:
            price = strike_price * math.exp(-risk_free_rate * time_to_expiry) * (1 - nd2) - spot_price * (1 - nd1)
        
        # Calculate Greeks
        # Delta - rate of change of option price with respect to underlying price
        delta = nd1 if option_type == OptionType.CALL else nd1 - 1
        
        # Gamma - rate of change of delta with respect to underlying price (same for calls and puts)
        gamma = pd1 / (spot_price * iv_times_sqrt)
        
        # Theta - rate of change of option price with respect to time (per day)
        theta1 = -spot_price * pd1 * iv / (2 * sqrt_time)
        theta2 = risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry)
        
        if option_type == OptionType.CALL:
            theta = -(theta1 + theta2 * nd2) / 365  # Convert to daily
        else:
            theta = -(theta1 - theta2 * (1 - nd2)) / 365
        
        # Vega - rate of change of option price with respect to volatility (per 1% change)
        vega = spot_price * sqrt_time * pd1 * 0.01  # Scale for 1% change
        
        # Rho - rate of change of option price with respect to interest rate (per 1% change)
        if option_type == OptionType.CALL:
            rho = strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * nd2 * 0.01
        else:
            rho = -strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * (1 - nd2) * 0.01
        
        # Advanced Greeks (second-order)
        # Charm - rate of change of delta with respect to time (delta decay)
        charm = -pd1 * (
            (risk_free_rate - ((iv * iv) / 2)) / (iv * sqrt_time) +
            (2 * risk_free_rate) / iv
        ) / 365  # Daily
        
        # Vanna - sensitivity of delta with respect to volatility change
        vanna = -pd1 * d2 / iv
        
        # Volga - sensitivity of vega with respect to volatility change
        volga = vega * (d1 * d2 / iv)
        
        # Create and return Greeks object
        greeks = OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            charm=charm,
            vanna=vanna,
            volga=volga
        )
        
        return {
            "price": price,
            "greeks": greeks
        }

    def normal_cdf(self, x: float) -> float:
        """Standard normal cumulative distribution function"""
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    def normal_pdf(self, x: float) -> float:
        """Standard normal probability density function"""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def calculate_position_size(
        self,
        option_details: OptionDetails,
        account_balance: float,
        risk_per_trade: float = None
    ) -> int:
        """
        Calculate option position size based on risk parameters
        
        Args:
            option_details: Details of the option
            account_balance: Current account balance
            risk_per_trade: Maximum risk percentage per trade
            
        Returns:
            Maximum number of contracts to trade
        """
        if risk_per_trade is None:
            risk_per_trade = self.risk_factor
            
        # Get current option data and Greeks
        result = self.calculate_black_scholes(
            self.market_data.get_spot_price(option_details.underlying_symbol),
            option_details.strike_price,
            self.calculate_time_to_expiry(option_details.expiration),
            option_details.implied_volatility,
            self.risk_free_rate,
            option_details.option_type
        )
        
        price = result["price"]
        greeks = result["greeks"]
        
        # Calculate maximum dollar risk for this trade
        max_risk_amount = account_balance * risk_per_trade
        
        # Calculate maximum loss based on strategy
        # For single options, max loss varies by type
        if option_details.option_type == OptionType.CALL or option_details.option_type == OptionType.PUT:
            # For long calls/puts, max loss is the premium paid
            max_loss_per_contract = price * option_details.contract_size
        
        # Calculate max position size based on risk
        max_contracts = math.floor(max_risk_amount / max_loss_per_contract)
        
        # Adjust for Greeks limits
        delta_per_contract = greeks.delta * option_details.contract_size
        gamma_per_contract = greeks.gamma * option_details.contract_size
        theta_per_contract = greeks.theta * option_details.contract_size
        vega_per_contract = greeks.vega * option_details.contract_size
        
        # Check if adding this position would exceed any Greeks limits
        available_delta = self.max_position_delta - abs(self.portfolio_greeks.delta)
        available_gamma = self.max_position_gamma - abs(self.portfolio_greeks.gamma)
        available_theta = abs(self.max_position_theta) - abs(self.portfolio_greeks.theta)
        available_vega = self.max_position_vega - abs(self.portfolio_greeks.vega)
        
        # Calculate max contracts based on each Greek limit
        max_contracts_by_delta = math.floor(available_delta / abs(delta_per_contract)) if abs(delta_per_contract) > 0 else float('inf')
        max_contracts_by_gamma = math.floor(available_gamma / abs(gamma_per_contract)) if abs(gamma_per_contract) > 0 else float('inf')
        max_contracts_by_theta = math.floor(available_theta / abs(theta_per_contract)) if abs(theta_per_contract) > 0 else float('inf')
        max_contracts_by_vega = math.floor(available_vega / abs(vega_per_contract)) if abs(vega_per_contract) > 0 else float('inf')
        
        # Take the minimum position size that doesn't violate any risk limit
        max_contracts_by_greeks = min(
            max_contracts_by_delta,
            max_contracts_by_gamma,
            max_contracts_by_theta,
            max_contracts_by_vega
        )
        
        # Final position size is minimum of risk-based and Greeks-based limits
        final_max_contracts = min(max_contracts, max_contracts_by_greeks)
        
        # Ensure at least 1 contract if all checks pass
        return max(1, final_max_contracts)

    def manage_expiring_options(self, positions: List[OptionPosition]) -> List[Dict]:
        """
        Manage options positions for approaching expiration dates
        
        Args:
            positions: All current option positions
            
        Returns:
            List of actions to take for expiring options
        """
        actions = []
        today = datetime.now()
        
        # Clear existing expiring options map
        self.expiring_options = {}
        
        # Group options by expiration date
        for position in positions:
            days_to_expiry = self.calculate_days_to_expiration(position.expiration)
            
            # Check if option is approaching expiration
            if days_to_expiry <= self.dte_cutoff:
                key = position.expiration.strftime("%Y-%m-%d")
                if key not in self.expiring_options:
                    self.expiring_options[key] = []
                self.expiring_options[key].append(position)
                
                # Determine action based on option state
                underlying_price = self.market_data.get_spot_price(position.underlying_symbol)
                result = self.calculate_black_scholes(
                    underlying_price,
                    position.strike_price,
                    self.calculate_time_to_expiry(position.expiration),
                    position.implied_volatility,
                    self.risk_free_rate,
                    position.option_type
                )
                
                greeks = result["greeks"]
                
                # Calculate intrinsic value
                if position.option_type == OptionType.CALL:
                    intrinsic_value = max(0, underlying_price - position.strike_price)
                else:
                    intrinsic_value = max(0, position.strike_price - underlying_price)
                
                # Calculate if ITM, ATM or OTM
                money_status = self.get_money_status(position, underlying_price)
                
                # Decision logic based on option state
                action = "HOLD"
                reason = ""
                
                if money_status == "ITM":
                    if abs(greeks.delta) > 0.9:
                        # Deep in the money
                        if position.side == PositionSide.LONG:
                            action = "CLOSE"
                            reason = "Deep ITM option approaching expiration - capture intrinsic value"
                        else:
                            action = "ROLL"
                            reason = "Short ITM option approaching expiration - roll to avoid assignment"
                    else:
                        if position.side == PositionSide.LONG:
                            action = "HOLD"
                            reason = "ITM option with remaining time value - monitor closely"
                        else:
                            action = "CLOSE"
                            reason = "Short ITM option approaching expiration - close to avoid assignment risk"
                elif money_status == "ATM":
                    if position.side == PositionSide.LONG:
                        action = "CLOSE"
                        reason = "ATM option approaching expiration - close to avoid gamma risk and theta decay"
                    else:
                        action = "CLOSE"
                        reason = "Short ATM option approaching expiration - close to avoid gamma risk"
                else:  # OTM
                    if position.side == PositionSide.LONG and abs(greeks.delta) < 0.1:
                        action = "CLOSE"
                        reason = "OTM long option with low delta - close to recover remaining premium"
                    elif position.side == PositionSide.SHORT:
                        action = "HOLD"
                        reason = "OTM short option approaching expiration - likely to expire worthless"
                    else:
                        action = "HOLD"
                        reason = "OTM option - monitor for changes in underlying"
                
                actions.append({
                    "option_id": position.id,
                    "action": action,
                    "reason": reason
                })
        
        return actions

    def get_money_status(self, position: OptionPosition, underlying_price: float) -> str:
        """
        Determine if option is In The Money (ITM), At The Money (ATM), or Out of The Money (OTM)
        """
        strike_price = position.strike_price
        buffer = underlying_price * 0.01  # 1% range for ATM consideration
        
        if position.option_type == OptionType.CALL:
            if underlying_price > strike_price + buffer:
                return "ITM"
            if underlying_price < strike_price - buffer:
                return "OTM"
            return "ATM"
        else:  # PUT
            if underlying_price < strike_price - buffer:
                return "ITM"
            if underlying_price > strike_price + buffer:
                return "OTM"
            return "ATM"

    def calculate_days_to_expiration(self, expiration_date: datetime) -> int:
        """Calculate days to expiration for an option"""
        today = datetime.now()
        diff_time = expiration_date - today
        return max(0, diff_time.days)

    def calculate_time_to_expiry(self, expiration_date: datetime) -> float:
        """Calculate time to expiry in years for Black-Scholes"""
        days_to_expiration = self.calculate_days_to_expiration(expiration_date)
        return days_to_expiration / 365.0

    def calculate_iv_metrics(
        self,
        symbol: str,
        current_iv: float
    ) -> Dict[str, float]:
        """
        Calculate implied volatility percentile for risk assessment
        
        Args:
            symbol: Symbol of the underlying
            current_iv: Current implied volatility
            
        Returns:
            Dictionary with IV percentile and rank
        """
        # Get historical IV data
        historical_iv = self.market_data.get_historical_iv(
            symbol,
            self.iv_percentile_window
        )
        
        if not historical_iv or len(historical_iv) == 0:
            return {"iv_percentile": 0.5, "iv_rank": 0.5}
        
        # Calculate IV Percentile
        sorted_iv = sorted(historical_iv)
        count_below = sum(1 for iv in sorted_iv if iv < current_iv)
        
        iv_percentile = count_below / len(sorted_iv)
        
        # Calculate IV Rank
        min_iv = sorted_iv[0]
        max_iv = sorted_iv[-1]
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv) if max_iv > min_iv else 0.5
        
        return {"iv_percentile": iv_percentile, "iv_rank": iv_rank}

    def calculate_probability_of_profit(
        self,
        position: OptionPosition,
        current_price: float
    ) -> float:
        """
        Calculate probability of profit for an option position
        Based on implied volatility and days to expiration
        """
        underlying_price = self.market_data.get_spot_price(position.underlying_symbol)
        time_to_expiry = self.calculate_time_to_expiry(position.expiration)
        iv = position.implied_volatility
        
        # Standard deviation calculation for expected price moves
        std_dev = underlying_price * iv * math.sqrt(time_to_expiry)
        
        probability = 0.0
        
        if position.side == PositionSide.LONG:
            if position.option_type == OptionType.CALL:
                # Long call: needs price to rise above strike + premium
                breakeven = position.strike_price + (current_price / position.contract_size)
                probability = 1.0 - self.normal_cdf((breakeven - underlying_price) / std_dev)
            else:
                # Long put: needs price to fall below strike - premium
                breakeven = position.strike_price - (current_price / position.contract_size)
                probability = self.normal_cdf((breakeven - underlying_price) / std_dev)
        else:  # SHORT positions
            if position.option_type == OptionType.CALL:
                # Short call: needs price to stay below strike + premium
                breakeven = position.strike_price + (current_price / position.contract_size)
                probability = self.normal_cdf((breakeven - underlying_price) / std_dev)
            else:
                # Short put: needs price to stay above strike - premium
                breakeven = position.strike_price - (current_price / position.contract_size)
                probability = 1.0 - self.normal_cdf((breakeven - underlying_price) / std_dev)
        
        return probability

    def handle_iv_risk(
        self,
        option_details: OptionDetails,
        suggested_position_size: int
    ) -> int:
        """
        Handle implied volatility risk
        Adjusts position sizing based on IV percentile and option vega
        """
        # Calculate IV metrics
        iv_metrics = self.calculate_iv_metrics(
            option_details.underlying_symbol,
            option_details.implied_volatility
        )
        iv_percentile = iv_metrics["iv_percentile"]
        
        # Calculate Black-Scholes to get vega
        result = self.calculate_black_scholes(
            self.market_data.get_spot_price(option_details.underlying_symbol),
            option_details.strike_price,
            self.calculate_time_to_expiry(option_details.expiration),
            option_details.implied_volatility,
            self.risk_free_rate,
            option_details.option_type
        )
        
        greeks = result["greeks"]
        adjusted_size = suggested_position_size
        
        # IV risk adjustment logic
        if option_details.option_type == OptionType.CALL:
            if iv_percentile > self.iv_threshold:
                # High IV environment for calls - reduce position size
                adjusted_size = math.floor(suggested_position_size * (1 - (iv_percentile - 0.5)))
            elif iv_percentile < (1 - self.iv_threshold):
                # Low IV environment for calls - can increase position size
                adjusted_size = math.ceil(suggested_position_size * (1 + (0.5 - iv_percentile)))
        else:  # PUTS
            if iv_percentile > self.iv_threshold:
                # High IV environment good for selling puts
                if greeks.vega < 0:  # If we're short vega
                    adjusted_size = math.ceil(suggested_position_size * (1 + (iv_percentile - 0.5)))
                else:
                    # Long puts in high IV - reduce size
                    adjusted_size = math.floor(suggested_position_size * (1 - (iv_percentile - 0.5)))
            elif iv_percentile < (1 - self.iv_threshold):
                # Low IV environment good for buying puts
                if greeks.vega > 0:  # If we're long vega
                    adjusted_size = math.ceil(suggested_position_size * (1 + (0.5 - iv_percentile)))
                else:
                    # Short puts in low IV - reduce size
                    adjusted_size = math.floor(suggested_position_size * (1 - (0.5 - iv_percentile)))
        
        # Ensure position size is at least 1
        return max(1, adjusted_size)

    def update_portfolio_greeks(
        self,
        position: OptionPosition,
        is_closing: bool = False
    ) -> OptionGreeks:
        """
        Update portfolio Greeks after adding a new position
        
        Args:
            position: The new option position
            is_closing: Whether this is closing an existing position
            
        Returns:
            Updated portfolio Greeks
        """
        underlying_price = self.market_data.get_spot_price(position.underlying_symbol)
        result = self.calculate_black_scholes(
            underlying_price,
            position.strike_price,
            self.calculate_time_to_expiry(position.expiration),
            position.implied_volatility,
            self.risk_free_rate,
            position.option_type
        )
        
        greeks = result["greeks"]
        
        multiplier = -1 if is_closing else 1
        position_multiplier = 1 if position.side == PositionSide.LONG else -1
        contract_multiplier = position.contract_size * position.quantity * multiplier * position_multiplier
        
        # Update portfolio Greeks
        self.portfolio_greeks.delta += greeks.delta * contract_multiplier
        self.portfolio_greeks.gamma += greeks.gamma * contract_multiplier
        self.portfolio_greeks.theta += greeks.theta * contract_multiplier
        self.portfolio_greeks.vega += greeks.vega * contract_multiplier
        self.portfolio_greeks.rho += greeks.rho * contract_multiplier
        
        return self.portfolio_greeks

    def get_option_position_risk(self, position: OptionPosition) -> OptionPositionRisk:
        """
        Get detailed risk analysis for an option position
        """
        underlying_price = self.market_data.get_spot_price(position.underlying_symbol)
        time_to_expiry = self.calculate_time_to_expiry(position.expiration)
        
        # Calculate Greeks
        result = self.calculate_black_scholes(
            underlying_price,
            position.strike_price,
            time_to_expiry,
            position.implied_volatility,
            self.risk_free_rate,
            position.option_type
        )
        
        price = result["price"]
        greeks = result["greeks"]
        
        # Calculate intrinsic and time value
        intrinsic_value = 0.0
        if position.option_type == OptionType.CALL:
            intrinsic_value = max(0, underlying_price - position.strike_price)
        else:
            intrinsic_value = max(0, position.strike_price - underlying_price)
        
        time_value = price - intrinsic_value
        
        # Get IV metrics
        iv_metrics = self.calculate_iv_metrics(
            position.underlying_symbol,
            position.implied_volatility
        )
        iv_percentile = iv_metrics["iv_percentile"]
        iv_rank = iv_metrics["iv_rank"]
        
        # Calculate max loss and max gain
        if position.side == PositionSide.LONG:
            # For long options
            max_loss = price * position.contract_size * position.quantity  # Premium paid
            if position.option_type == OptionType.CALL:
                max_gain = float('inf')  # Theoretically unlimited for calls
            else:
                max_gain = position.strike_price * position.contract_size * position.quantity - max_loss
        else:
            # For short options
            max_gain = price * position.contract_size * position.quantity  # Premium received
            if position.option_type == OptionType.CALL:
                max_loss = float('inf')  # Theoretically unlimited for short calls
            else:
                max_loss = position.strike_price * position.contract_size * position.quantity - max_gain
        
        # Calculate probability of profit
        prob_of_profit = self.calculate_probability_of_profit(position, price)
        
        # Calculate breakeven points
        breakeven_points = []
        if position.option_type == OptionType.CALL:
            breakeven_points.append(position.strike_price + (price / position.contract_size))
        else:
            breakeven_points.append(position.strike_price - (price / position.contract_size))
        
        return OptionPositionRisk(
            position=position,
            greeks=greeks,
            days_to_expiration=self.calculate_days_to_expiration(position.expiration),
            time_value=time_value,
            intrinsic_value=intrinsic_value,
            iv_percentile=iv_percentile,
            iv_rank=iv_rank,
            max_loss=max_loss,
            max_gain=max_gain,
            probability_of_profit=prob_of_profit,
            breakeven=breakeven_points
        )

    def get_portfolio_greeks(self) -> OptionGreeks:
        """Get current portfolio Greek exposures"""
        return self.portfolio_greeks

    def reset_portfolio_greeks(self) -> None:
        """Reset portfolio Greeks (typically used for recalculation)"""
        self.portfolio_greeks = OptionGreeks()

    def recalculate_portfolio_greeks(self, positions: List[OptionPosition]) -> OptionGreeks:
        """
        Recalculate all portfolio Greeks from a list of positions
        
        Args:
            positions: All current option positions
            
        Returns:
            Updated portfolio Greeks
        """
        self.reset_portfolio_greeks()
        
        for position in positions:
            underlying_price = self.market_data.get_spot_price(position.underlying_symbol)
            result = self.calculate_black_scholes(
                underlying_price,
                position.strike_price,
                self.calculate_time_to_expiry(position.expiration),
                position.implied_volatility,
                self.risk_free_rate,
                position.option_type
            )
            
            greeks = result["greeks"]
            
            position_multiplier = 1 if position.side == PositionSide.LONG else -1
            contract_multiplier = position.contract_size * position.quantity * position_multiplier
            
            self.portfolio_greeks.delta += greeks.delta * contract_multiplier
            self.portfolio_greeks.gamma += greeks.gamma * contract_multiplier
            self.portfolio_greeks.theta += greeks.theta * contract_multiplier
            self.portfolio_greeks.vega += greeks.vega * contract_multiplier
            self.portfolio_greeks.rho += greeks.rho * contract_multiplier
        
        return self.portfolio_greeks 