"""
Forex Arbitrage Strategy

This strategy identifies and exploits pricing inefficiencies between related currency pairs.
It uses triangular arbitrage and other techniques to capture risk-free (or low-risk) profit 
opportunities that arise due to market inefficiencies or latency.

The strategy monitors multiple currency pairs simultaneously and looks for:
1. Triangular arbitrage opportunities (e.g., EUR/USD, USD/JPY, EUR/JPY)
2. Cross-rate discrepancies
3. Broker pricing inconsistencies

Features:
- Real-time monitoring of price feeds from multiple sources
- Fast execution to capitalize on fleeting opportunities
- Risk management for imperfect execution
- Slippage and fee modeling
"""

import logging
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Set

from trading_bot.strategies_new.forex.base.forex_base_strategy import ForexBaseStrategy
from trading_bot.strategies_new.factory.registry import register_strategy
from trading_bot.models.signal import Signal


@register_strategy(
    asset_class="forex",
    strategy_type="arbitrage",
    name="ForexArbitrage",
    description="Exploits pricing inefficiencies between related currency pairs through triangular arbitrage and other techniques",
    parameters={
        "default": {
            # Arbitrage configuration
            "arbitrage_types": ["triangular", "cross_rate", "broker"],  # Types of arbitrage to seek
            "min_profit_threshold_pips": 0.5,  # Minimum profit threshold in pips to execute
            "min_profit_threshold_percent": 0.02,  # Minimum profit threshold as percentage (0.02 = 0.02%)
            
            # Execution parameters
            "max_execution_latency_ms": 100,  # Maximum expected execution latency in milliseconds
            "execution_priority": "speed",  # "speed" or "stealth"
            "use_limit_orders": False,  # Whether to use limit orders instead of market orders
            
            # Risk management
            "max_position_size": 1.0,  # Maximum position size in standard lots per arbitrage opportunity
            "max_total_exposure": 5.0,  # Maximum total position size across all arbitrage positions
            "risk_per_trade": 0.005,  # Percentage of account to risk per trade (0.5%)
            "use_hedging": True,  # Whether to use hedging to reduce directional exposure
            
            # Currency pairs configuration
            "currency_groups": {
                "major": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"],
                "crosses": ["EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD", "EURCHF", "GBPCHF"],
                "exotics": ["EURNOK", "EURSEK", "USDMXN", "USDZAR", "USDRUB", "USDTRY"]
            },
            
            # Data source configuration
            "primary_broker": "main_broker",
            "secondary_brokers": ["broker_2", "broker_3"],
            "price_feed_priority": ["direct_feed", "broker_api", "market_data_provider"],
            
            # Trade management
            "close_all_at_once": True,  # Close all legs of arbitrage simultaneously
            "max_holding_period_seconds": 300,  # Maximum holding period for arbitrage positions
            "hedge_timeout_seconds": 60,  # Maximum time to wait for hedging positions to be established
            
            # Filtering and validation
            "min_spread_quality": 0.7,  # Minimum spread quality score (0-1) to consider a pair
            "max_spread_for_arbitrage": {
                "major": 1.5,  # pips
                "crosses": 2.5,
                "exotics": 5.0
            },
            "quote_staleness_threshold_ms": 500,  # Maximum acceptable quote age in milliseconds
            
            # General parameters
            "timezone": pytz.UTC,
            "check_interval_ms": 100,  # Interval for checking arbitrage opportunities (milliseconds)
            "enable_cross_broker_arbitrage": False,  # Whether to enable arbitrage across different brokers
        },
        
        # For aggressive arbitrage hunting, more opportunities but higher risk
        "aggressive": {
            "min_profit_threshold_pips": 0.3,
            "min_profit_threshold_percent": 0.015,
            "max_execution_latency_ms": 150,
            "risk_per_trade": 0.008,
            "max_position_size": 1.5,
            "min_spread_quality": 0.6,
            "enable_cross_broker_arbitrage": True
        },
        
        # For conservative arbitrage, fewer but higher quality opportunities
        "conservative": {
            "min_profit_threshold_pips": 0.8,
            "min_profit_threshold_percent": 0.03,
            "max_execution_latency_ms": 80,
            "risk_per_trade": 0.003,
            "max_position_size": 0.5,
            "min_spread_quality": 0.8,
            "enable_cross_broker_arbitrage": False
        }
    }
)
class ForexArbitrageStrategy(ForexBaseStrategy):
    """
    A strategy that identifies and exploits pricing inefficiencies between related currency pairs.
    
    It uses triangular arbitrage and other techniques to capture risk-free (or low-risk) profit
    opportunities that arise due to market inefficiencies or latency.
    """
    
    def __init__(self, session=None):
        """
        Initialize the arbitrage strategy.
        
        Args:
            session: Trading session object with configuration
        """
        super().__init__(session)
        self.name = "ForexArbitrage"
        self.description = "Exploits pricing inefficiencies between related currency pairs"
        self.logger = logging.getLogger(__name__)
        
        # Track active arbitrage opportunities
        self.active_opportunities = {}  # id -> opportunity data
        
        # Track currency relationships for quick lookup
        self.currency_pairs = {}  # pair -> (base, quote) mapping
        self.currencies = set()  # Set of all currencies monitored
        self.triangles = []  # List of potential triangular arbitrage combinations
        
        # Performance statistics
        self.stats = {
            "opportunities_detected": 0,
            "opportunities_executed": 0,
            "total_profit_pips": 0,
            "total_profit_percent": 0,
            "execution_latency_ms": [],
            "opportunity_duration_ms": []
        }
        
        # Last check timestamp
        self.last_check_time = None
        
    def initialize(self) -> None:
        """Initialize strategy and load any required data."""
        super().initialize()
        
        # Initialize currency pairs
        self._initialize_currency_pairs()
        
        # Identify possible triangular arbitrage combinations
        self._identify_triangular_combinations()
        
        # Initialize arbitrage opportunity tracking
        self.active_opportunities = {}
        
        # Initialize last check time
        self.last_check_time = datetime.now(self.parameters["timezone"])
        
        self.logger.info(f"Initialized {self.name} strategy with {len(self.triangles)} potential triangles")
        
    def _initialize_currency_pairs(self) -> None:
        """Initialize the list of currency pairs to monitor and their relationships."""
        # Flatten the currency groups into a list of pairs to monitor
        monitored_pairs = []
        for group, pairs in self.parameters["currency_groups"].items():
            monitored_pairs.extend(pairs)
        
        # Create mappings and extract currencies
        for pair in monitored_pairs:
            if len(pair) >= 6:  # Standard currency pairs are 6 characters (EURUSD)
                base = pair[:3]
                quote = pair[3:6]
                self.currency_pairs[pair] = (base, quote)
                self.currencies.add(base)
                self.currencies.add(quote)
        
        self.logger.info(f"Monitoring {len(self.currency_pairs)} currency pairs across {len(self.currencies)} currencies")
    
    def _identify_triangular_combinations(self) -> None:
        """Identify all possible triangular arbitrage combinations."""
        # Reset triangles list
        self.triangles = []
        
        # For each currency (e.g., EUR), identify triangles where it can be the starting point
        for currency in self.currencies:
            # Find all pairs with this currency as base or quote
            pairs_with_currency = []
            for pair, (base, quote) in self.currency_pairs.items():
                if base == currency or quote == currency:
                    pairs_with_currency.append((pair, base, quote))
            
            # For each pair with this currency, find connected pairs
            for pair1, base1, quote1 in pairs_with_currency:
                # Determine the "other" currency in the pair (not the starting currency)
                other_currency = quote1 if base1 == currency else base1
                
                # Find pairs with the "other" currency that don't involve the starting currency
                for pair2, base2, quote2 in pairs_with_currency:
                    if pair2 == pair1:
                        continue  # Skip the same pair
                    
                    # Ensure this pair involves the starting currency but not the "other" currency
                    if (base2 == currency or quote2 == currency) and base2 != other_currency and quote2 != other_currency:
                        continue  # This pair doesn't connect with other_currency
                    
                    # Determine the third currency
                    third_currency = quote2 if base2 == other_currency else base2
                    
                    # Find the pair that completes the triangle (between third_currency and starting currency)
                    for pair3, base3, quote3 in pairs_with_currency:
                        if pair3 == pair1 or pair3 == pair2:
                            continue  # Skip already used pairs
                        
                        # Check if this pair connects the third currency back to the starting currency
                        if ((base3 == third_currency and quote3 == currency) or 
                            (base3 == currency and quote3 == third_currency)):
                            
                            # We found a triangle: currency -> other_currency -> third_currency -> currency
                            triangle = {
                                "currencies": [currency, other_currency, third_currency],
                                "pairs": [pair1, pair2, pair3],
                                "directions": []  # Will be filled during analysis
                            }
                            
                            # Check if this triangle (or equivalent) is already in the list
                            triangle_signature = ''.join(sorted([pair1, pair2, pair3]))
                            existing_signatures = [''.join(sorted(t["pairs"])) for t in self.triangles]
                            
                            if triangle_signature not in existing_signatures:
                                self.triangles.append(triangle)
        
        self.logger.info(f"Identified {len(self.triangles)} potential triangular arbitrage combinations")

    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Identify arbitrage opportunities and generate trading signals.
        
        Args:
            data_dict: Dictionary of market data for different pairs
            
        Returns:
            Dictionary of trading signals
        """
        # Early exit if we don't have enough data
        if not data_dict or len(data_dict) < 3:  # Need at least 3 pairs for triangular arbitrage
            return {}
        
        signals = {}
        current_time = datetime.now(self.parameters["timezone"])
        
        # Check if we should scan for new opportunities (based on check interval)
        time_since_last_check = (current_time - self.last_check_time).total_seconds() * 1000
        if time_since_last_check < self.parameters["check_interval_ms"]:
            return {}  # Too soon to check again
        
        self.last_check_time = current_time
        
        # Check for triangular arbitrage opportunities
        if "triangular" in self.parameters["arbitrage_types"]:
            triangular_signals = self._check_triangular_arbitrage(data_dict, current_time)
            signals.update(triangular_signals)
        
        # Check for cross-rate arbitrage opportunities
        if "cross_rate" in self.parameters["arbitrage_types"]:
            cross_rate_signals = self._check_cross_rate_arbitrage(data_dict, current_time)
            signals.update(cross_rate_signals)
        
        # Check for broker arbitrage opportunities
        if "broker" in self.parameters["arbitrage_types"] and self.parameters["enable_cross_broker_arbitrage"]:
            broker_signals = self._check_broker_arbitrage(data_dict, current_time)
            signals.update(broker_signals)
        
        # Log the number of opportunities found
        if signals:
            self.logger.info(f"Found {len(signals)} arbitrage opportunities")
        
        return signals
    
    def _check_triangular_arbitrage(self, data_dict: Dict[str, pd.DataFrame], 
                                  current_time: datetime) -> Dict[str, Signal]:
        """
        Check for triangular arbitrage opportunities.
        
        Args:
            data_dict: Dictionary of market data
            current_time: Current timestamp
            
        Returns:
            Dictionary of signals for triangular arbitrage
        """
        signals = {}
        
        # Check each potential triangle
        for triangle in self.triangles:
            # Ensure we have data for all pairs in the triangle
            pairs = triangle["pairs"]
            if not all(pair in data_dict for pair in pairs):
                continue
            
            # Get the latest prices for each pair
            prices = {}
            directions = []
            
            for pair in pairs:
                # Get the latest mid price
                if data_dict[pair].empty:
                    break
                
                latest_data = data_dict[pair].iloc[-1]
                bid = latest_data.get("bid", latest_data["close"] - 0.0001)  # Estimate if not available
                ask = latest_data.get("ask", latest_data["close"] + 0.0001)  # Estimate if not available
                mid = (bid + ask) / 2
                spread = ask - bid
                
                # Skip if spread is too wide (indicates poor liquidity)
                pair_type = self._get_pair_type(pair)
                max_spread = self.parameters["max_spread_for_arbitrage"].get(pair_type, 2.0)
                point_value = 0.0001 if "JPY" not in pair else 0.01
                spread_pips = spread / point_value
                
                if spread_pips > max_spread:
                    break
                
                # For each pair, determine if we would BUY or SELL based on the triangle
                base, quote = self.currency_pairs[pair]
                c1, c2, c3 = triangle["currencies"]
                
                # Logic to determine trade direction
                if (base == c1 and quote == c2) or (base == c2 and quote == c3) or (base == c3 and quote == c1):
                    direction = "buy"  # We're buying the base currency
                    exec_price = ask  # When buying, we pay the ask price
                else:
                    direction = "sell"  # We're selling the base currency
                    exec_price = bid  # When selling, we receive the bid price
                
                prices[pair] = {
                    "mid": mid,
                    "bid": bid,
                    "ask": ask,
                    "direction": direction,
                    "exec_price": exec_price,
                    "spread_pips": spread_pips
                }
                directions.append(direction)
            
            # Skip if we couldn't get prices for all pairs
            if len(prices) != len(pairs):
                continue
            
            # Calculate potential profit in the triangle
            profit_percent, profit_pips = self._calculate_triangular_profit(triangle, prices)
            
            # Check if profit exceeds threshold
            min_profit_pips = self.parameters["min_profit_threshold_pips"]
            min_profit_percent = self.parameters["min_profit_threshold_percent"]
            
            if profit_pips > min_profit_pips and profit_percent > min_profit_percent:
                # This is a valid opportunity - create signals
                
                # Update triangle with directions for future reference
                triangle["directions"] = directions
                
                # Create a unique ID for this arbitrage opportunity
                opportunity_id = f"tri_{triangle['currencies'][0]}_{triangle['currencies'][1]}_{triangle['currencies'][2]}_{current_time.timestamp()}"
                
                # Calculate position sizes
                position_sizes = self._calculate_arbitrage_position_sizes(triangle, prices)
                
                # Generate signals for each leg of the arbitrage
                arb_signals = {}
                for i, pair in enumerate(pairs):
                    direction = directions[i]
                    price_data = prices[pair]
                    
                    # Create signal for this leg
                    signal = Signal(
                        symbol=pair,
                        signal_type=direction,
                        entry_price=price_data["exec_price"],
                        stop_loss=None,  # Arbitrage typically doesn't use stops
                        take_profit=None,  # Arbitrage typically doesn't use take-profits
                        size=position_sizes[i],
                        timestamp=current_time,
                        timeframe="1m",  # Arbitrage operates on very short timeframes
                        strategy=self.name,
                        strength=profit_percent / 0.1,  # Normalize to 0-1 range (assuming 0.1% is strong)
                        metadata={
                            "arbitrage_type": "triangular",
                            "opportunity_id": opportunity_id,
                            "triangle_currencies": triangle["currencies"],
                            "profit_pips": profit_pips,
                            "profit_percent": profit_percent,
                            "leg_index": i,
                            "total_legs": len(pairs),
                            "spread_pips": price_data["spread_pips"],
                            "expiry_seconds": self.parameters["max_holding_period_seconds"]
                        }
                    )
                    
                    # Use the pair as the key in the signals dictionary
                    arb_signals[pair] = signal
                
                # Track this opportunity
                self.active_opportunities[opportunity_id] = {
                    "type": "triangular",
                    "triangle": triangle,
                    "prices": prices,
                    "profit_pips": profit_pips,
                    "profit_percent": profit_percent,
                    "position_sizes": position_sizes,
                    "start_time": current_time,
                    "pairs": pairs,
                    "signals": arb_signals
                }
                
                # Update statistics
                self.stats["opportunities_detected"] += 1
                
                # Add to the signals dictionary
                signals.update(arb_signals)
                
                # Log the opportunity
                self.logger.info(f"Triangular arbitrage opportunity: {triangle['currencies'][0]}->{triangle['currencies'][1]}->{triangle['currencies'][2]}->{triangle['currencies'][0]}, profit: {profit_pips:.1f} pips ({profit_percent:.4f}%)")
        
        return signals
    
    def _calculate_triangular_profit(self, triangle: Dict[str, Any], 
                                   prices: Dict[str, Dict[str, Any]]) -> Tuple[float, float]:
        """
        Calculate the potential profit for a triangular arbitrage opportunity.
        
        Args:
            triangle: Triangle information
            prices: Price data for each pair
            
        Returns:
            Tuple of (profit_percent, profit_pips)
        """
        # Starting with 1 unit of the first currency
        value = 1.0
        c1, c2, c3 = triangle["currencies"]
        pairs = triangle["pairs"]
        
        # Walk through the triangle
        for i, pair in enumerate(pairs):
            base, quote = self.currency_pairs[pair]
            direction = prices[pair]["direction"]
            exec_price = prices[pair]["exec_price"]
            
            if direction == "buy":
                # Buying base currency with quote currency
                if value > 0:  # If we have positive quote currency
                    # Convert to base currency
                    value = value / exec_price
                else:  # If we have negative quote currency
                    # Repay the loan
                    value = value / exec_price
            else:  # direction == "sell"
                # Selling base currency for quote currency
                if value > 0:  # If we have positive base currency
                    # Convert to quote currency
                    value = value * exec_price
                else:  # If we have negative base currency
                    # Repay the loan
                    value = value * exec_price
        
        # Calculate profit percentage
        profit_percent = (value - 1.0) * 100  # As a percentage
        
        # Estimate profit in pips
        # This is an approximation - would be more accurate with exact conversion
        avg_point_value = 0.0001  # Default
        for pair in pairs:
            if "JPY" in pair:
                avg_point_value = (avg_point_value + 0.01) / 2  # Adjust for JPY pairs
        
        profit_pips = profit_percent / 100 / avg_point_value
        
        return profit_percent, profit_pips
    
    def _calculate_arbitrage_position_sizes(self, triangle: Dict[str, Any], 
                                          prices: Dict[str, Dict[str, Any]]) -> List[float]:
        """
        Calculate appropriate position sizes for each leg of a triangular arbitrage.
        
        Args:
            triangle: Triangle information
            prices: Price data for each pair
            
        Returns:
            List of position sizes for each leg
        """
        # Get account balance
        account_balance = self.session.account_balance
        
        # Determine risk amount
        risk_amount = account_balance * self.parameters["risk_per_trade"]
        
        # Calculate base position size (in standard lots)
        base_size = min(risk_amount / account_balance, self.parameters["max_position_size"])
        
        # Adjust based on triangle relationship
        # For now, use the same size for all legs as a simplification
        # In a real implementation, these would need to be balanced based on pair relationships
        position_sizes = [base_size] * len(triangle["pairs"])
        
        return position_sizes
    
    def _check_cross_rate_arbitrage(self, data_dict: Dict[str, pd.DataFrame],
                                  current_time: datetime) -> Dict[str, Signal]:
        """
        Check for cross-rate arbitrage opportunities.
        
        Args:
            data_dict: Dictionary of market data
            current_time: Current timestamp
            
        Returns:
            Dictionary of signals for cross-rate arbitrage
        """
        # Cross-rate arbitrage is a type of triangular arbitrage where one leg
        # is a synthetic cross rate calculated from two other pairs
        
        # For a minimal implementation, we'll create a simplified version
        # that looks for common cross rate opportunities
        
        signals = {}
        
        # Find potential cross-rate opportunities
        for quote_currency in self.currencies:
            # Look for pairs that share the same quote currency
            matching_pairs = []
            for pair, (base, quote) in self.currency_pairs.items():
                if quote == quote_currency and pair in data_dict:
                    matching_pairs.append((pair, base))
            
            # Need at least 2 pairs with this quote to create crosses
            if len(matching_pairs) < 2:
                continue
            
            # Check all possible cross combinations
            for i in range(len(matching_pairs)):
                for j in range(i+1, len(matching_pairs)):
                    pair1, base1 = matching_pairs[i]
                    pair2, base2 = matching_pairs[j]
                    
                    # Look for the direct cross pair
                    cross_pair = None
                    for pair in self.currency_pairs:
                        pair_base, pair_quote = self.currency_pairs[pair]
                        if ((pair_base == base1 and pair_quote == base2) or 
                            (pair_base == base2 and pair_quote == base1)) and pair in data_dict:
                            cross_pair = pair
                            break
                    
                    if not cross_pair:
                        continue  # Cross pair not found or no data
                    
                    # Calculate synthetic cross rate
                    if data_dict[pair1].empty or data_dict[pair2].empty or data_dict[cross_pair].empty:
                        continue
                    
                    latest1 = data_dict[pair1].iloc[-1]
                    latest2 = data_dict[pair2].iloc[-1]
                    latest_cross = data_dict[cross_pair].iloc[-1]
                    
                    # Get mid prices
                    mid1 = (latest1.get("bid", latest1["close"] - 0.0001) + latest1.get("ask", latest1["close"] + 0.0001)) / 2
                    mid2 = (latest2.get("bid", latest2["close"] - 0.0001) + latest2.get("ask", latest2["close"] + 0.0001)) / 2
                    mid_cross = (latest_cross.get("bid", latest_cross["close"] - 0.0001) + latest_cross.get("ask", latest_cross["close"] + 0.0001)) / 2
                    
                    # Calculate synthetic cross rate
                    # If direct pair is BASE1/BASE2, synthetic should be MID1/MID2
                    # If direct pair is BASE2/BASE1, synthetic should be MID2/MID1
                    direct_base, direct_quote = self.currency_pairs[cross_pair]
                    
                    if direct_base == base1 and direct_quote == base2:
                        synthetic_rate = mid1 / mid2
                        profit_percent = ((mid_cross / synthetic_rate) - 1) * 100
                    else:  # direct_base == base2 and direct_quote == base1
                        synthetic_rate = mid2 / mid1
                        profit_percent = ((mid_cross / synthetic_rate) - 1) * 100
                    
                    # Calculate profit in pips
                    point_value = 0.0001 if "JPY" not in cross_pair else 0.01
                    profit_pips = abs(mid_cross - synthetic_rate) / point_value
                    
                    # Check if profit exceeds threshold
                    min_profit_pips = self.parameters["min_profit_threshold_pips"]
                    min_profit_percent = self.parameters["min_profit_threshold_percent"]
                    
                    if profit_pips > min_profit_pips and abs(profit_percent) > min_profit_percent:
                        # This is a valid opportunity
                        opportunity_id = f"cross_{base1}_{base2}_{quote_currency}_{current_time.timestamp()}"
                        
                        # Generate signals (simplified for this example)
                        # A full implementation would determine exact execution pathways
                        
                        # Skip implementation for now - would require more complex execution logic
                        # than we can reasonably implement in this example
                        
                        # Log the opportunity
                        self.logger.info(f"Cross-rate arbitrage opportunity: {base1}/{quote_currency}, {base2}/{quote_currency}, {cross_pair}, profit: {profit_pips:.1f} pips ({profit_percent:.4f}%)")
        
        return signals
    
    def _check_broker_arbitrage(self, data_dict: Dict[str, pd.DataFrame],
                              current_time: datetime) -> Dict[str, Signal]:
        """
        Check for broker arbitrage opportunities (price differences across brokers).
        
        Args:
            data_dict: Dictionary of market data
            current_time: Current timestamp
            
        Returns:
            Dictionary of signals for broker arbitrage
        """
        # Broker arbitrage requires multiple data sources, which would be outside
        # the scope of this implementation
        
        # This is a placeholder for a real implementation
        # In a production system, this would compare quotes from different brokers
        
        return {}  # No signals for this simplified implementation
    
    def _get_pair_type(self, pair: str) -> str:
        """
        Determine the type of a currency pair (major, cross, exotic).
        
        Args:
            pair: Currency pair symbol
            
        Returns:
            Pair type as string
        """
        for pair_type, pairs in self.parameters["currency_groups"].items():
            if pair in pairs:
                return pair_type
        
        # Default to 'cross' if not found
        return "cross"
    
    def check_exit_conditions(self, position: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Check if an arbitrage position should be exited.
        
        Args:
            position: Current position information
            data: Latest market data
            
        Returns:
            True if position should be exited, False otherwise
        """
        if data.empty:
            return False
            
        # Extract position metadata
        opportunity_id = position.get("metadata", {}).get("opportunity_id")
        symbol = position["symbol"]
        
        if not opportunity_id:
            return False  # Not an arbitrage position or missing metadata
        
        # Look up the opportunity
        opportunity = self.active_opportunities.get(opportunity_id)
        if not opportunity:
            # If we don't have the opportunity tracked, exit based on time
            entry_time_str = position.get("metadata", {}).get("entry_time")
            if not entry_time_str:
                return False
                
            try:
                entry_time = datetime.fromisoformat(entry_time_str)
                current_time = datetime.now(self.parameters["timezone"])
                expiry_seconds = position.get("metadata", {}).get("expiry_seconds", 
                                                           self.parameters["max_holding_period_seconds"])
                
                if (current_time - entry_time).total_seconds() > expiry_seconds:
                    self.logger.info(f"Exiting {symbol} due to expiry (no opportunity data)")
                    return True
            except (ValueError, TypeError):
                return False
                
            return False
        
        # Check time-based exit conditions
        current_time = datetime.now(self.parameters["timezone"])
        start_time = opportunity.get("start_time")
        
        if start_time and (current_time - start_time).total_seconds() > self.parameters["max_holding_period_seconds"]:
            self.logger.info(f"Exiting {symbol} due to maximum holding period for arbitrage")
            return True
        
        # If we're using a close-all-at-once approach, we would check
        # for a global exit signal for this opportunity
        if self.parameters["close_all_at_once"]:
            # In a real implementation, this would check if all legs are ready to exit
            # and potentially send a close signal to all related positions
            pass
        
        # Check if this opportunity has been marked for exit
        if opportunity.get("exit_signal", False):
            self.logger.info(f"Exiting {symbol} due to global exit signal for arbitrage {opportunity_id}")
            return True
        
        return False
