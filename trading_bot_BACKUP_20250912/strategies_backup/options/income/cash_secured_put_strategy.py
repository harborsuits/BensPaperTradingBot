#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cash-Secured Put Income Strategy Module

This module implements a cash-secured put income strategy that generates premium 
by selling puts against cash reserves, aiming to buy quality stocks at a discount 
if assigned, while collecting time decay.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
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

class CashSecuredPutStrategy(StrategyOptimizable):
    """
    Cash-Secured Put Income Strategy designed to generate premium income by selling puts
    against cash reserves, while being prepared to buy quality stocks at a discount if assigned.
    
    This strategy aims to collect time decay by selling out-of-the-money put options on
    quality underlyings with appropriate strike and expiration selection.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Cash-Secured Put strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters based on the strategy blueprint
        default_params = {
            # Section 3: Selection Criteria
            "trend_filter": {
                "use_sma": True,
                "sma_period": 50,
                "value_range_low": None,
                "value_range_high": None
            },
            "iv_rank_min": 40,  # Minimum IV rank percentile
            "liquidity": {
                "min_adv": 500000,  # Minimum average daily volume
                "min_option_open_interest": 1000,  # Minimum open interest
                "max_bid_ask_spread_pct": 0.15  # Maximum bid-ask spread as percent
            },
            
            # Section 4: Strike Selection
            "otm_buffer_pct": 3.0,  # Default OTM buffer percentage
            "delta_target": 0.25,  # Target delta (0.20-0.30 range)
            "iv_regime_adjustment": {
                "high_iv_threshold": 75,  # IV rank percentile for high IV
                "low_iv_threshold": 25,   # IV rank percentile for low IV
                "high_iv_buffer_pct": 5.0,  # Deeper OTM in high IV
                "low_iv_buffer_pct": 2.0    # Less OTM in low IV
            },
            
            # Section 5: Expiration Selection
            "dte_range": (21, 45),  # Target days to expiration range
            "min_dte": 10,  # Avoid options with less than this DTE
            "weekly_triggers": {
                "min_iv_rank_weekly": 60,  # Minimum IV rank to consider weeklies
                "catalysts": ["earnings", "fomc", "product_launch"]  # Catalyst events
            },
            
            # Section 7: Exit & Assignment Management
            "profit_take_pct": 40,  # Buy back at 40% of original premium
            "time_exit_dte": 7,     # Close or roll within this many days to expiry
            "roll_threshold_pct": 1.0,  # Roll when underlying within this % of strike
            "assignment_strategy": "covered_call",  # Strategy if assigned: 'covered_call' or 'sell'
            
            # Section 8: Position Sizing & Risk Controls
            "risk_per_trade_pct": 5.0,  # Maximum percent of equity per trade
            "max_allocation_pct": 20.0,  # Maximum allocation to cash-secured puts
            "cash_buffer_pct": 10.0,  # Additional cash reserve percentage
            "max_positions_per_sector": 2,  # Maximum positions per sector
            
            # Section 10: Continuous Optimization
            "optimization_frequency_days": 30,  # Re-optimize every 30 days
            "switch_to_spread_iv_threshold": 20  # Switch to spreads below this IV rank
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        # Initialize strategy-specific attributes
        self.active_positions = {}  # Track active put positions
        self.cash_reserved = 0.0    # Track reserved cash for puts
        self.assignments = []       # Track assignments
        self.recent_trades = []     # Track recent trades
        
        logger.info(f"Initialized Cash-Secured Put strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            # Selection criteria parameters
            "trend_filter.sma_period": [20, 50, 100, 200],
            "iv_rank_min": [30, 40, 50, 60],
            
            # Strike selection parameters
            "otm_buffer_pct": [2.0, 3.0, 4.0, 5.0],
            "delta_target": [0.15, 0.20, 0.25, 0.30, 0.35],
            
            # Expiration selection parameters
            "dte_range": [(15, 30), (21, 45), (30, 60)],
            
            # Exit management parameters
            "profit_take_pct": [30, 40, 50, 60],
            "time_exit_dte": [5, 7, 10],
            
            # Position sizing parameters
            "risk_per_trade_pct": [3.0, 5.0, 7.0],
            "max_allocation_pct": [15.0, 20.0, 25.0]
        }
    
    # Section 1: Strategy Philosophy
    def _strategy_philosophy(self) -> str:
        """
        Return the strategy philosophy and purpose.
        
        Returns:
            Description of strategy philosophy
        """
        # TODO: Document the premium generation approach
        # TODO: Define criteria for quality underlyings
        # TODO: Explain the balance between premium collection and assignment risk
        
        return """
        This strategy generates premium income by selling puts against cash reserves, 
        aiming to buy quality stocks at a discount if assigned, while collecting time decay.
        It focuses on disciplined strike selection, proper position sizing, and active
        management of options approaching expiration.
        """
    
    # Section 2: Underlying & Option Universe & Timeframe
    def _define_universe(self) -> Dict[str, Any]:
        """
        Define the universe of underlyings and options to trade.
        
        Returns:
            Dictionary of universe specifications
        """
        # TODO: Implement screening for liquid large-caps or ETFs
        # TODO: Create rules for filtering based on fundamentals
        # TODO: Define option chain selection logic
        
        universe = {
            "underlyings": {
                "type": "liquid_large_caps_etfs",
                "min_market_cap": 10e9,  # $10B minimum market cap
                "sectors": ["Technology", "Healthcare", "Consumer", "Financials", "Industrials", "ETFs"],
                "examples": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN"]
            },
            "options": {
                "primary_cycle": "monthly",
                "secondary_cycle": "weekly",
                "contract_types": ["put"]
            },
            "holding_period": {
                "typical_days": (21, 45),
                "max_days": 60
            }
        }
        
        return universe
    
    # Section 3: Selection Criteria
    def _check_selection_criteria(
        self, 
        underlying_data: pd.DataFrame,
        option_chain: pd.DataFrame,
        market_data: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Check if an underlying meets the selection criteria.
        
        Args:
            underlying_data: OHLCV data for the underlying
            option_chain: Option chain data
            market_data: Additional market data dictionary
            
        Returns:
            Dictionary of criteria check results
        """
        # TODO: Implement trend filter (above 50-day SMA or in value range)
        # TODO: Calculate IV rank and check against threshold
        # TODO: Verify liquidity requirements (volume, OI, spread)
        
        results = {
            "trend_filter_passed": False,
            "iv_rank_passed": False,
            "liquidity_passed": False,
            "all_criteria_passed": False
        }
        
        return results
    
    # Section 4: Strike Selection
    def _select_optimal_strike(
        self, 
        underlying_price: float,
        option_chain: pd.DataFrame,
        iv_rank: float
    ) -> Dict[str, Any]:
        """
        Select the optimal strike price for selling puts.
        
        Args:
            underlying_price: Current price of the underlying
            option_chain: Option chain data
            iv_rank: Current IV rank (0-100)
            
        Returns:
            Dictionary with selected strike information
        """
        # TODO: Implement OTM buffer calculation (2-5% below spot)
        # TODO: Find options closest to delta target (0.20-0.30)
        # TODO: Adjust buffer based on IV regime
        
        # Determine buffer based on IV regime
        iv_regime_adj = self.parameters["iv_regime_adjustment"]
        if iv_rank >= iv_regime_adj["high_iv_threshold"]:
            # High IV - use deeper OTM
            buffer_pct = iv_regime_adj["high_iv_buffer_pct"]
        elif iv_rank <= iv_regime_adj["low_iv_threshold"]:
            # Low IV - use less OTM
            buffer_pct = iv_regime_adj["low_iv_buffer_pct"]
        else:
            # Normal IV - use default buffer
            buffer_pct = self.parameters["otm_buffer_pct"]

        # Calculate target strike based on buffer
        target_strike = underlying_price * (1 - buffer_pct / 100)
        
        # Placeholder for actual strike selection logic
        strike_info = {
            "strike": target_strike,
            "delta": self.parameters["delta_target"],
            "otm_pct": buffer_pct,
            "iv": 0.0,  # To be filled
            "premium": 0.0  # To be filled
        }
        
        return strike_info
    
    # Section 5: Expiration Selection
    def _select_optimal_expiration(
        self, 
        option_chain: pd.DataFrame,
        iv_rank: float,
        upcoming_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select the optimal expiration date for the put option.
        
        Args:
            option_chain: Option chain data
            iv_rank: Current IV rank (0-100)
            upcoming_events: List of upcoming catalyst events
            
        Returns:
            Dictionary with selected expiration information
        """
        # TODO: Find expirations within optimal DTE range (21-45 days)
        # TODO: Avoid options with < 10 DTE
        # TODO: Check if weeklies should be used based on IV and catalysts
        
        min_dte, max_dte = self.parameters["dte_range"]
        
        # Check if we should consider weeklies based on IV rank and catalysts
        use_weekly = False
        if iv_rank >= self.parameters["weekly_triggers"]["min_iv_rank_weekly"]:
            # High IV, check for catalysts
            if any(event["type"] in self.parameters["weekly_triggers"]["catalysts"] 
                  for event in upcoming_events):
                use_weekly = True
        
        expiration_info = {
            "expiration_date": None,  # To be filled
            "dte": 0,  # To be filled
            "is_weekly": use_weekly,
            "cycle": "weekly" if use_weekly else "monthly",
            "catalyst_events": [e for e in upcoming_events if e["date"] < datetime.now() + timedelta(days=max_dte)]
        }
        
        return expiration_info
    
    # Section 6: Entry Execution
    def _prepare_entry_order(
        self, 
        underlying_symbol: str,
        strike_info: Dict[str, Any],
        expiration_info: Dict[str, Any],
        account_balance: float
    ) -> Dict[str, Any]:
        """
        Prepare the entry order for selling a cash-secured put.
        
        Args:
            underlying_symbol: Symbol of the underlying
            strike_info: Strike information from strike selection
            expiration_info: Expiration information
            account_balance: Current account balance
            
        Returns:
            Dictionary with order details
        """
        # TODO: Calculate required cash reserve (strike × contract size)
        # TODO: Prepare limit order at or near mid-quote
        # TODO: Include cash-secure checks
        
        # Standard contract size is 100 shares
        contract_size = 100
        
        # Calculate required cash reserve
        required_cash = strike_info["strike"] * contract_size
        
        # Check if we have enough cash for this position
        max_risk_amount = account_balance * (self.parameters["risk_per_trade_pct"] / 100)
        if required_cash > max_risk_amount:
            # Adjust position size
            contracts = int(max_risk_amount / required_cash)
            if contracts < 1:
                logger.warning(f"Insufficient cash to sell 1 put for {underlying_symbol}")
                return None
        else:
            contracts = 1  # Default to 1 contract
        
        order_details = {
            "action": "SELL",
            "symbol": underlying_symbol,
            "option_type": "PUT",
            "strike": strike_info["strike"],
            "expiration": expiration_info["expiration_date"],
            "contracts": contracts,
            "order_type": "LIMIT",
            "limit_price": strike_info["premium"],
            "cash_reserved": required_cash * contracts,
            "delta": strike_info["delta"],
            "iv": strike_info["iv"],
            "dte": expiration_info["dte"]
        }
        
        return order_details
    
    # Section 7: Exit & Assignment Management
    def _check_exit_conditions(
        self, 
        position: Dict[str, Any],
        current_price: float,
        current_option_price: float,
        dte: int
    ) -> Dict[str, Any]:
        """
        Check if exit conditions are met for a position.
        
        Args:
            position: Current position details
            current_price: Current underlying price
            current_option_price: Current option price
            dte: Days to expiration
            
        Returns:
            Dictionary with exit decision and reason
        """
        # TODO: Check if premium decayed to profit target level
        # TODO: Check if approaching expiration (time exit)
        # TODO: Check if underlying near strike for potential roll
        # TODO: Prepare assignment handling if needed
        
        exit_decision = {
            "exit": False,
            "reason": None,
            "action": None,
            "roll": False,
            "roll_details": None
        }
        
        # Check profit target
        entry_premium = position["entry_price"]
        profit_target = entry_premium * (self.parameters["profit_take_pct"] / 100)
        if current_option_price <= profit_target:
            exit_decision["exit"] = True
            exit_decision["reason"] = "profit_target"
            exit_decision["action"] = "BUY"  # Buy back the put
            return exit_decision
        
        # Check time exit
        if dte <= self.parameters["time_exit_dte"]:
            exit_decision["exit"] = True
            exit_decision["reason"] = "time_exit"
            exit_decision["action"] = "BUY"  # Buy back the put
            
            # Check if we should roll
            strike = position["strike"]
            roll_threshold = strike * (self.parameters["roll_threshold_pct"] / 100)
            if abs(current_price - strike) <= roll_threshold:
                exit_decision["roll"] = True
                # Roll details would be filled by a separate method
            
            return exit_decision
        
        return exit_decision
    
    def _handle_assignment(
        self, 
        position: Dict[str, Any],
        current_price: float,
        account_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle assignment of puts, either by holding shares or selling.
        
        Args:
            position: Position being assigned
            current_price: Current price of the underlying
            account_data: Account information
            
        Returns:
            Dictionary with assignment handling details
        """
        # TODO: Implement assignment handling logic
        # TODO: Create covered call if that's the chosen strategy
        # TODO: Prepare to sell shares if that's preferred
        
        assignment_strategy = self.parameters["assignment_strategy"]
        assignment_details = {
            "symbol": position["symbol"],
            "shares_assigned": position["contracts"] * 100,
            "cost_basis": position["strike"],
            "current_price": current_price,
            "unrealized_pl": (current_price - position["strike"]) * position["contracts"] * 100,
            "strategy": assignment_strategy
        }
        
        if assignment_strategy == "covered_call":
            # Prepare covered call details
            assignment_details["covered_call"] = {
                "action": "SELL",
                "option_type": "CALL",
                # Additional call details would be determined by a separate method
            }
        elif assignment_strategy == "sell":
            # Prepare to sell shares
            assignment_details["sell_order"] = {
                "action": "SELL",
                "order_type": "MARKET",
                "shares": position["contracts"] * 100
            }
        
        return assignment_details
    
    # Section 8: Position Sizing & Risk Controls
    def _calculate_position_size(
        self, 
        underlying_symbol: str,
        strike_price: float,
        premium: float,
        account_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters.
        
        Args:
            underlying_symbol: Symbol of the underlying
            strike_price: Selected strike price
            premium: Option premium
            account_data: Account information including balance and existing positions
            
        Returns:
            Dictionary with position sizing information
        """
        # TODO: Implement risk-based position sizing
        # TODO: Check against maximum allocation
        # TODO: Verify cash buffer is maintained
        # TODO: Apply concentration limits (sector/correlation)
        
        equity = account_data["equity"]
        cash_available = account_data["cash_available"]
        
        # Calculate maximum risk per trade
        max_risk_per_trade = equity * (self.parameters["risk_per_trade_pct"] / 100)
        
        # Calculate cash required per contract
        cash_per_contract = strike_price * 100  # 100 shares per contract
        
        # Calculate maximum contracts based on risk limit
        max_contracts_by_risk = int(max_risk_per_trade / cash_per_contract)
        
        # Calculate maximum contracts based on available cash
        max_contracts_by_cash = int(cash_available / cash_per_contract)
        
        # Check allocation limit
        current_csp_allocation = account_data.get("cash_secured_put_allocation", 0)
        max_allocation = equity * (self.parameters["max_allocation_pct"] / 100)
        remaining_allocation = max_allocation - current_csp_allocation
        max_contracts_by_allocation = int(remaining_allocation / cash_per_contract)
        
        # Take the minimum of all constraints
        max_contracts = min(max_contracts_by_risk, max_contracts_by_cash, max_contracts_by_allocation)
        
        # Check sector concentration
        sector = account_data.get("sectors", {}).get(underlying_symbol, "Unknown")
        sector_positions = sum(1 for pos in account_data.get("positions", []) 
                              if account_data.get("sectors", {}).get(pos["symbol"], "") == sector)
        
        if sector_positions >= self.parameters["max_positions_per_sector"]:
            logger.warning(f"Sector concentration limit reached for {sector}")
            max_contracts = 0
        
        position_size = {
            "max_contracts": max_contracts,
            "recommended_contracts": max(1, max_contracts),  # At least 1 contract if allowed
            "cash_required": cash_per_contract * max(1, max_contracts),
            "premium_expected": premium * 100 * max(1, max_contracts),
            "sector": sector,
            "sector_exposure": sector_positions
        }
        
        return position_size
    
    # Section 9: Backtesting & Performance Metrics
    def _calculate_performance_metrics(
        self, 
        trade_history: List[Dict[str, Any]],
        account_history: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            trade_history: List of completed trades
            account_history: DataFrame with account balance history
            
        Returns:
            Dictionary of performance metrics
        """
        # TODO: Calculate annualized yield
        # TODO: Compute assignment rate
        # TODO: Determine win rate
        # TODO: Calculate max drawdown
        # TODO: Calculate average return per cycle
        
        metrics = {
            "annualized_yield": 0.0,
            "assignment_rate": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "avg_return_per_cycle": 0.0,
            "sharpe_ratio": 0.0,
            "total_premium_collected": 0.0,
            "total_trades": 0
        }
        
        return metrics
    
    # Section 10: Continuous Optimization
    def _optimize_parameters(
        self, 
        trade_history: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        current_iv_rank: float
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on recent performance and market conditions.
        
        Args:
            trade_history: Recent trade history
            market_data: Current market data
            current_iv_rank: Current IV rank (0-100)
            
        Returns:
            Dictionary of optimized parameters
        """
        # TODO: Re-optimize OTM buffers based on realized returns
        # TODO: Adjust delta targets based on performance
        # TODO: Modify DTE windows based on results
        # TODO: Implement IV-adaptive strike selection
        # TODO: Add strategy switch logic for low IV environments
        
        optimized_params = self.parameters.copy()
        
        # IV-adaptive strike adjustments
        if current_iv_rank >= 75:
            # High IV environment - go deeper OTM
            optimized_params["otm_buffer_pct"] = self.parameters["iv_regime_adjustment"]["high_iv_buffer_pct"]
            logger.info(f"High IV environment ({current_iv_rank}), adjusting OTM buffer to {optimized_params['otm_buffer_pct']}%")
        elif current_iv_rank <= 25:
            # Low IV environment - less OTM
            optimized_params["otm_buffer_pct"] = self.parameters["iv_regime_adjustment"]["low_iv_buffer_pct"]
            logger.info(f"Low IV environment ({current_iv_rank}), adjusting OTM buffer to {optimized_params['otm_buffer_pct']}%")
        
        # Strategy switch for very low IV
        if current_iv_rank < self.parameters["switch_to_spread_iv_threshold"]:
            logger.info(f"Very low IV environment ({current_iv_rank}), recommending switch to bull put spreads")
            optimized_params["strategy_switch"] = "bull_put_spread"
        else:
            optimized_params["strategy_switch"] = None
        
        return optimized_params
    
    def calculate_iv_rank(
        self, 
        current_iv: float,
        historical_iv: pd.Series,
        lookback_days: int = 252
    ) -> float:
        """
        Calculate the IV rank (percentile of current IV relative to history).
        
        Args:
            current_iv: Current implied volatility
            historical_iv: Series of historical IV values
            lookback_days: Number of days to look back
            
        Returns:
            IV rank as a percentage (0-100)
        """
        # Use recent history based on lookback period
        recent_iv = historical_iv.tail(lookback_days)
        
        if len(recent_iv) < 10:  # Need enough data points
            logger.warning("Insufficient IV history for proper IV rank calculation")
            return 50.0  # Default to middle
        
        iv_min = recent_iv.min()
        iv_max = recent_iv.max()
        
        if iv_max == iv_min:  # Avoid division by zero
            return 50.0
        
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        
        return iv_rank
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate indicators needed for the cash-secured put strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        for symbol, df in data.items():
            try:
                # Calculate 50-day SMA for trend filter
                sma_period = self.parameters["trend_filter"]["sma_period"]
                sma = df['close'].rolling(window=sma_period).mean()
                
                # Trend status (above/below SMA)
                trend_status = df['close'] > sma
                
                # Placeholder for IV data (would come from options data in real implementation)
                # Here we'll just simulate IV for demonstration
                iv_series = pd.Series(np.random.uniform(0.2, 0.4, len(df)), index=df.index)
                iv_current = iv_series.iloc[-1]
                
                # Calculate IV rank
                iv_rank = self.calculate_iv_rank(iv_current, iv_series)
                
                # Store indicators
                indicators[symbol] = {
                    "sma": sma,
                    "trend_status": trend_status,
                    "iv_series": iv_series,
                    "iv_current": iv_current,
                    "iv_rank": iv_rank,
                    "above_sma": df['close'].iloc[-1] > sma.iloc[-1],
                    "price_to_sma_ratio": df['close'].iloc[-1] / sma.iloc[-1] if not pd.isna(sma.iloc[-1]) else 1.0
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def generate_signals(
        self, 
        data: Dict[str, pd.DataFrame],
        option_chains: Optional[Dict[str, pd.DataFrame]] = None,
        account_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Signal]:
        """
        Generate cash-secured put signals based on selection criteria.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data
            option_chains: Optional dictionary of option chains
            account_data: Optional account information
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        
        # Initialize signals dictionary
        signals = {}
        
        # Mock option chains and account data if not provided
        if option_chains is None:
            option_chains = self._mock_option_chains(data)
        
        if account_data is None:
            account_data = {
                "equity": 100000.0,
                "cash_available": 80000.0,
                "cash_secured_put_allocation": 0.0,
                "positions": [],
                "sectors": {symbol: "Technology" for symbol in data.keys()}
            }
        
        # Generate signals
        for symbol, symbol_indicators in indicators.items():
            try:
                # Get latest price
                latest_data = data[symbol].iloc[-1]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Check selection criteria
                selection_results = self._check_selection_criteria(
                    data[symbol], 
                    option_chains.get(symbol, pd.DataFrame()),
                    {"iv_rank": symbol_indicators["iv_rank"]}
                )
                
                # Skip if criteria not met
                if not selection_results.get("all_criteria_passed", False):
                    logger.debug(f"Selection criteria not met for {symbol}: {selection_results}")
                    continue
                
                # Select strike
                strike_info = self._select_optimal_strike(
                    latest_price,
                    option_chains.get(symbol, pd.DataFrame()),
                    symbol_indicators["iv_rank"]
                )
                
                # Select expiration
                expiration_info = self._select_optimal_expiration(
                    option_chains.get(symbol, pd.DataFrame()),
                    symbol_indicators["iv_rank"],
                    []  # Mock empty list of upcoming events
                )
                
                # Calculate position size
                position_size = self._calculate_position_size(
                    symbol,
                    strike_info["strike"],
                    strike_info["premium"],
                    account_data
                )
                
                # Skip if no position size available
                if position_size["max_contracts"] <= 0:
                    logger.debug(f"No valid position size for {symbol}")
                    continue
                
                # Prepare entry order
                entry_order = self._prepare_entry_order(
                    symbol,
                    strike_info,
                    expiration_info,
                    account_data["equity"]
                )
                
                if entry_order is None:
                    logger.debug(f"Could not prepare entry order for {symbol}")
                    continue
                
                # Generate signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=SignalType.OPTION_SELL_PUT,
                    price=latest_price,
                    timestamp=latest_timestamp,
                    confidence=min(symbol_indicators["iv_rank"] / 100, 0.9),  # Higher confidence with higher IV rank
                    stop_loss=None,  # Not applicable for cash-secured puts
                    take_profit=strike_info["premium"] * (1 - self.parameters["profit_take_pct"] / 100),
                    metadata={
                        "strategy_type": "cash_secured_put",
                        "iv_rank": symbol_indicators["iv_rank"],
                        "strike": strike_info["strike"],
                        "expiration": expiration_info.get("expiration_date"),
                        "dte": expiration_info.get("dte"),
                        "premium": strike_info["premium"],
                        "contracts": position_size["recommended_contracts"],
                        "cash_required": position_size["cash_required"],
                        "otm_pct": strike_info["otm_pct"],
                        "delta": strike_info["delta"],
                        "sma": symbol_indicators["sma"].iloc[-1],
                        "above_sma": symbol_indicators["above_sma"]
                    }
                )
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _mock_option_chains(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create mock option chains for testing/demo purposes.
        
        Args:
            data: Dictionary of price data
            
        Returns:
            Dictionary of mock option chains
        """
        option_chains = {}
        
        for symbol, df in data.items():
            latest_price = df['close'].iloc[-1]
            
            # Create a range of strikes
            strikes = [latest_price * (1 - i/100) for i in range(1, 10)]
            expirations = [
                datetime.now() + timedelta(days=days) for days in [7, 14, 21, 30, 45, 60, 90]
            ]
            
            # Create mock option data
            rows = []
            for strike in strikes:
                for exp in expirations:
                    dte = (exp - datetime.now()).days
                    delta = 0.5 - (latest_price - strike) / latest_price
                    iv = 0.3 + (0.1 * np.random.random())
                    bid = max(0.01, latest_price * 0.02 * (delta + 0.2) * (dte/30))
                    ask = bid * 1.1
                    
                    rows.append({
                        "strike": strike,
                        "expiration": exp,
                        "dte": dte,
                        "option_type": "put",
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                        "delta": max(0.01, min(0.99, delta)),
                        "gamma": 0.01,
                        "theta": -0.01,
                        "vega": 0.1,
                        "iv": iv,
                        "volume": int(100 * np.random.random()),
                        "open_interest": int(1000 * np.random.random())
                    })
            
            option_chains[symbol] = pd.DataFrame(rows)
        
        return option_chains 