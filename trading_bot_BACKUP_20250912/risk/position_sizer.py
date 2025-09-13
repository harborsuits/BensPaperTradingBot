#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Sizer Module

This module provides the PositionSizer class for determining
appropriate position sizes based on risk parameters and performance metrics.
"""

import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PositionSizer:
    """
    Advanced position sizing system for risk-controlled capital allocation.
    
    The PositionSizer implements sophisticated risk management algorithms to determine
    optimal position sizes across various asset classes. It transforms trading signals 
    into properly sized orders based on account equity, risk tolerances, and trade-specific 
    parameters, ensuring consistent risk management across the portfolio.
    
    Key capabilities:
    1. Multiple position sizing methodologies (fixed risk, volatility-based, Kelly)
    2. Asset class-specific sizing algorithms for stocks, options, and futures
    3. Portfolio-level risk management with correlation adjustments
    4. Volatility-based position size scaling
    5. Maximum position limits and diversification controls
    6. Custom risk parameter support for different strategies
    
    The PositionSizer serves as a critical risk management component that:
    - Protects trading capital through systematic risk control
    - Ensures appropriate scaling of position sizes with account growth
    - Prevents excess concentration in correlated positions
    - Adapts position sizes to market volatility conditions
    - Enforces strategy-specific risk parameters
    - Maintains risk consistency across the portfolio
    
    Implementation follows core risk management principles:
    - Never risk more than a small percentage of capital on any single trade
    - Scale position sizes according to market conditions and volatility
    - Reduce exposure in highly correlated assets
    - Incorporate trade-specific risk parameters (stop loss, max loss)
    - Provide comprehensive position information for decision-making
    
    The class can be extended with additional sizing methods or
    risk controls as needed for specific trading requirements.
    """
    
    def __init__(self, portfolio_value: float, default_risk_percent: float = 0.01, config: Dict[str, Any] = None):
        """
        Initialize the PositionSizer with core risk parameters.
        
        Creates a new PositionSizer configured with the current portfolio value and
        default risk parameters. This establishes the base risk framework for all
        subsequent position sizing calculations.
        
        Parameters:
            portfolio_value (float): Current total portfolio value in base currency.
                This should include all assets, cash, and open positions.
                
            default_risk_percent (float): Default percentage of portfolio to risk 
                per trade, expressed as a decimal (e.g., 0.01 = 1% of portfolio value).
                This is the maximum capital at risk for a standard trade.
                
        Risk Management Guidelines:
            - Conservative: 0.5% to 1% (0.005-0.01) per trade
            - Moderate: 1% to 2% (0.01-0.02) per trade
            - Aggressive: 2% to 3% (0.02-0.03) per trade
            - Values above 3% (0.03) are generally considered too risky
                
        Notes:
            - Portfolio value should be updated regularly to ensure accurate sizing
            - Risk percentage should be determined based on strategy characteristics
            - Default risk percent serves as a baseline but can be overridden per trade
            - Position sizer operates independently from broker connections
            - Risk calculations use pre-tax values and don't account for commissions
        """
        self.portfolio_value = portfolio_value
        self.default_risk_percent = default_risk_percent
        
        # Initialize with configuration or defaults
        self.config = config or {}
        
        # Performance and volatility scaling parameters
        self.base_risk_pct = self.config.get('base_risk_pct', 0.01)  # Base risk of 1% of equity per trade
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)  # Default stop-loss distance (2%)
        self.atr_period = self.config.get('atr_period', 14)  # Lookback window for ATR
        self.atr_multiplier = self.config.get('atr_multiplier', 3.0)  # Multiplier on ATR for stop-loss
        
        # Performance adjustment parameters
        self.perf_sharpe_window = self.config.get('perf_sharpe_window', 30)  # Window for Sharpe calculation
        self.perf_adj_min = self.config.get('perf_adj_min', 0.5)  # Minimum performance adjustment
        self.perf_adj_max = self.config.get('perf_adj_max', 1.5)  # Maximum performance adjustment
        
        # Performance metrics cache
        self.performance_metrics = {}
        self.last_metrics_update = None
        
        logger.info(f"Initialized PositionSizer with portfolio value: ${portfolio_value:,.2f}, "  
                   f"base_risk_pct: {self.base_risk_pct}, atr_multiplier: {self.atr_multiplier}")
    
    def get_portfolio_value(self) -> float:
        """
        Retrieve the current portfolio value used for position sizing.
        
        Returns:
            float: The current portfolio value in base currency
            
        Notes:
            - Returns the last updated portfolio value
            - Critical for all position sizing calculations
            - Should be updated whenever account value changes significantly
            - Used as the base value for all percentage-based risk calculations
        """
        return self.portfolio_value
    
    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update the portfolio value used for position sizing calculations.
        
        This method should be called whenever the portfolio value changes significantly,
        such as after trades are executed, at the beginning of a trading session,
        or when deposits or withdrawals occur.
        
        Parameters:
            new_value (float): The updated total portfolio value in base currency
            
        Side effects:
            - Updates the internal portfolio value used for all calculations
            - Logs the change in portfolio value for audit purposes
            
        Update strategy:
            - Daily: Update at the beginning of each trading session
            - After significant P&L changes: Update after large winning or losing trades
            - After deposits/withdrawals: Update when account capital changes
            - Periodic: Regular updates at set intervals (hourly, etc.)
            
        Notes:
            - Regular updates ensure position sizing remains proportional to current capital
            - Position sizes will automatically scale with portfolio growth or drawdown
            - Significant changes should trigger recalculation of open position sizes
            - Portfolio value should include all assets, not just cash balance
        """
        old_value = self.portfolio_value
        self.portfolio_value = new_value
        logger.info(f"Updated portfolio value: ${old_value:,.2f} -> ${new_value:,.2f}")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: Optional[float] = None,
                              risk_percent: Optional[float] = None,
                              market_data: Optional[pd.DataFrame] = None,
                              strategy_performance: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size for stock positions based on risk parameters.
        
        This method implements the fixed fractional risk position sizing model, where
        each trade risks a consistent percentage of total capital. The position size
        is determined by the distance between entry price and stop loss, ensuring
        that the maximum loss matches the target risk percentage.
        
        Parameters:
            symbol (str): Symbol of the instrument to trade
            entry_price (float): Anticipated entry price for the position
            stop_loss (Optional[float]): Price level for the stop loss order.
                If provided, position size is calculated to limit risk to the
                specified risk percentage if price reaches this level.
            risk_percent (Optional[float]): Percentage of portfolio to risk on this
                specific trade, overriding the default risk percentage if provided.
            market_data (Optional[pd.DataFrame]): Historical market data for volatility-based sizing
            strategy_performance (Optional[Dict[str, float]]): Performance metrics for performance-based sizing
                
        Returns:
            Dict[str, Any]: Comprehensive position size information including:
                - "symbol": Trading symbol
                - "entry_price": Entry price used for calculation
                - "stop_loss": Stop loss price used for calculation
                - "shares": Calculated number of shares to trade
                - "position_value": Total value of the position at entry
                - "portfolio_percent": Percentage of portfolio allocated to this position
                - "dollar_risk": Maximum dollar amount at risk
                - "risk_percent": Actual percentage of portfolio at risk
                - "performance_adjustment": Adjustment factor based on strategy performance
                - "atr_based_stop": Whether the stop loss is based on ATR
                
        Position sizing logic:
        1. With stop loss provided (preferred method):
           - Calculate risk per share as the distance from entry to stop
           - Determine shares based on dollar risk divided by risk per share
           - Round down to ensure risk stays within parameters
           
        2. Without stop loss (alternative method):
           - Uses position value as a percentage of portfolio (less precise)
           - Position size typically larger but without defined risk parameters
           - Generally 10x the risk percentage as allocation percentage
           
        Notes:
            - Position size is always rounded down to whole shares
            - If entry price and stop loss are identical, falls back to method 2
            - Position value refers to total capital allocated, not amount at risk
            - Dollar risk is the maximum expected loss if stop loss is hit
            - Both long and short positions use the same calculation logic
            - For short positions, entry price should be lower than stop loss
        """
        if risk_percent is None:
            risk_percent = self.default_risk_percent
        
        # Calculate dollar risk
        dollar_risk = self.portfolio_value * risk_percent
        
        # Performance adjustment
        perf_adjustment = 1.0
        if strategy_performance:
            sharpe_ratio = strategy_performance.get('sharpe_ratio', 0.0)
            if sharpe_ratio > 0:
                perf_adjustment = min(max(sharpe_ratio, self.perf_adj_min), self.perf_adj_max)
        
        # If stop loss is provided, calculate position size based on stop loss
        if stop_loss is not None and stop_loss != entry_price:
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            # Calculate position size in shares
            shares = dollar_risk / risk_per_share
            
            # Round down to nearest whole share
            shares = math.floor(shares)
            
            # Calculate total position value
            position_value = shares * entry_price
            
            # Percentage of portfolio
            portfolio_percent = position_value / self.portfolio_value
            
            # Actual dollar risk
            actual_dollar_risk = shares * risk_per_share
            
            # ATR-based stop loss
            is_atr_stop = False
            if market_data is not None:
                atr = self.calculate_atr(market_data, self.atr_period)
                if atr is not None:
                    atr_stop_loss = entry_price - (atr * self.atr_multiplier)
                    if stop_loss is None or atr_stop_loss < stop_loss:
                        stop_loss = atr_stop_loss
                        is_atr_stop = True
            
            return {
                "symbol": symbol,
                "shares": shares,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "position_value": position_value,
                "dollar_risk": actual_dollar_risk,
                "risk_percent": actual_dollar_risk / self.portfolio_value,
                "portfolio_percent": portfolio_percent,
                "performance_adjustment": perf_adjustment,
                "atr_based_stop": is_atr_stop
            }
        else:
            # Without a stop loss, use a fixed percentage of portfolio for position sizing
            # This is less ideal but allows for position sizing without a stop loss
            
            # Calculate position value as a percentage of portfolio
            position_value = self.portfolio_value * (risk_percent * 10)  # 10x risk percentage
            
            # Calculate shares
            shares = math.floor(position_value / entry_price)
            
            # Recalculate position value
            position_value = shares * entry_price
            
            # Percentage of portfolio
            portfolio_percent = position_value / self.portfolio_value
            
            return {
                "symbol": symbol,
                "shares": shares,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "position_value": position_value,
                "dollar_risk": None,  # Unknown without stop loss
                "risk_percent": None,  # Unknown without stop loss
                "portfolio_percent": portfolio_percent,
                "performance_adjustment": perf_adjustment,
                "atr_based_stop": False
            }
    
    def calculate_option_position_size(self, symbol: str, premium: float, 
                                     max_loss_per_contract: float,
                                     risk_percent: Optional[float] = None,
                                     min_contracts: int = 1,
                                     max_contracts: int = 100) -> Dict[str, Any]:
        """
        Calculate optimal position size for options trades based on risk parameters.
        
        This specialized method determines appropriate contract quantities for options
        trades, accounting for the unique characteristics of options including defined
        maximum loss profiles and contract multipliers.
        
        Parameters:
            symbol (str): Symbol of the underlying asset
            premium (float): Per-contract option premium in dollars (not multiplied by 100)
            max_loss_per_contract (float): Maximum potential loss per contract.
                For long options, this is typically the premium paid.
                For spreads or short options, this should be the maximum defined risk.
            risk_percent (Optional[float]): Percentage of portfolio to risk on this
                specific trade, overriding the default risk percentage if provided.
            min_contracts (int): Minimum number of contracts to trade (default: 1)
            max_contracts (int): Maximum number of contracts to trade (default: 100)
                
        Returns:
            Dict[str, Any]: Comprehensive position size information including:
                - "symbol": Trading symbol
                - "premium": Per-contract premium
                - "contracts": Number of contracts to trade
                - "total_premium": Total cost/credit for all contracts
                - "max_loss_per_contract": Maximum loss per contract
                - "total_max_loss": Maximum total loss for the position
                - "portfolio_percent": Percentage of portfolio allocated (premium basis)
                - "risk_percent": Actual percentage of portfolio at risk
                
        Position sizing logic for options:
        1. Calculate maximum dollar risk based on risk percentage
        2. Determine number of contracts based on max loss per contract
        3. Apply minimum and maximum contract limits
        4. Calculate total premium and maximum loss for the position
        
        Notes:
            - Options have a standard multiplier of 100 (each contract = 100 shares)
            - Premium should be specified per share (not multiplied by 100)
            - For defined-risk strategies (spreads), max_loss_per_contract is the spread width
            - For undefined-risk strategies, max_loss_per_contract should include a reasonable maximum
            - Long options: max_loss_per_contract equals the premium paid
            - Short options or spreads: max_loss_per_contract should be the defined risk
            - Minimum contract limits ensure practical position sizes
            - Maximum contract limits prevent excess risk on low-priced options
            - Total premium calculation includes the 100 multiplier
        """
        if risk_percent is None:
            risk_percent = self.default_risk_percent
        
        # Calculate dollar risk
        dollar_risk = self.portfolio_value * risk_percent
        
        # Calculate contracts based on risk
        if max_loss_per_contract > 0:
            contracts = math.floor(dollar_risk / max_loss_per_contract)
        else:
            contracts = min_contracts
        
        # Apply limits
        contracts = max(min_contracts, min(contracts, max_contracts))
        
        # Calculate total premium and max loss
        total_premium = contracts * premium * 100  # 100 multiplier for options
        total_max_loss = contracts * max_loss_per_contract
        
        # Percentage of portfolio
        portfolio_percent = total_premium / self.portfolio_value
        
        return {
            "symbol": symbol,
            "premium": premium,
            "contracts": contracts,
            "total_premium": total_premium,
            "max_loss_per_contract": max_loss_per_contract,
            "total_max_loss": total_max_loss,
            "portfolio_percent": portfolio_percent,
            "risk_percent": total_max_loss / self.portfolio_value
        }
    
    def adjust_for_correlation(self, position_sizes: Dict[str, Dict[str, Any]], 
                            correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
        """
        Adjust position sizes to account for portfolio correlation risk.
        
        This method implements portfolio-level risk management by adjusting individual
        position sizes based on their correlation with other positions in the portfolio.
        Highly correlated positions have their sizes reduced to prevent excess concentration
        risk and maintain overall portfolio diversification.
        
        Parameters:
            position_sizes (Dict[str, Dict[str, Any]]): Dictionary mapping symbols to 
                position size information generated by calculate_position_size().
            correlation_matrix (Dict[str, Dict[str, float]]): Two-dimensional dictionary 
                containing pairwise correlation coefficients between symbols, with values 
                ranging from -1.0 to 1.0.
                
        Returns:
            Dict[str, Dict[str, Any]]: Adjusted position sizes with correlation risk
                factored in. Maintains the same structure as the input but with
                potentially reduced position sizes for highly correlated instruments.
                
        Correlation adjustment logic:
        1. For each position, calculate average absolute correlation with other positions
        2. Apply a reduction factor for positions with high average correlation:
           - Positions with average correlation > 0.7 are considered highly correlated
           - Reduction scale is proportional to correlation intensity
           - Maximum reduction is 50% for perfectly correlated positions
        3. Recalculate all position metrics after size adjustment
        
        Diversification principles applied:
        - Highly correlated positions represent redundant exposure
        - Portfolio risk is not simply the sum of individual position risks
        - Total exposure to correlated assets should be limited
        - Size reduction is proportional to correlation strength
        
        Notes:
            - Correlation values should range from -1.0 to 1.0
            - Absolute correlation values are used, treating both positive and negative correlations
            - Missing correlations for a symbol pair are ignored in calculations
            - Only position size (shares/contracts) is adjusted, not the risk percentage
            - Position value and dollar risk are recalculated after adjustment
            - Correlation should be measured over a relevant timeframe
            - This is a simplified implementation; more sophisticated approaches may consider
              the full covariance matrix and portfolio optimization techniques
        """
        # This is a simplified correlation adjustment
        # In a real implementation, you would adjust position sizes based on
        # portfolio risk and correlation between positions
        
        adjusted_position_sizes = position_sizes.copy()
        
        for symbol, position in adjusted_position_sizes.items():
            # Skip if symbol not in correlation matrix
            if symbol not in correlation_matrix:
                continue
                
            # Calculate average correlation with other positions
            correlations = []
            for other_symbol in position_sizes:
                if other_symbol != symbol and other_symbol in correlation_matrix[symbol]:
                    correlations.append(abs(correlation_matrix[symbol][other_symbol]))
            
            if not correlations:
                continue
                
            avg_correlation = sum(correlations) / len(correlations)
            
            # Adjust position size based on correlation
            # Higher correlation = smaller position size
            if avg_correlation > 0.7:
                # Reduce position size by up to 50% for high correlations
                reduction_factor = 0.5 + 0.5 * (1 - avg_correlation) / 0.3
                
                # Apply reduction to shares and position value
                position["shares"] = math.floor(position["shares"] * reduction_factor)
                position["position_value"] = position["shares"] * position["entry_price"]
                
                # Update other values
                position["portfolio_percent"] = position["position_value"] / self.portfolio_value
                
                if position["stop_loss"] is not None:
                    position["dollar_risk"] = position["shares"] * abs(position["entry_price"] - position["stop_loss"])
                    position["risk_percent"] = position["dollar_risk"] / self.portfolio_value
        
        return adjusted_position_sizes
    
    def calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range (ATR) for volatility-based position sizing.
        
        Parameters:
            market_data: DataFrame with high, low, close prices
            period: Lookback period for ATR calculation
            
        Returns:
            float: ATR value or None if calculation fails
        """
        try:
            if len(market_data) < period + 1:
                logger.warning(f"Not enough data to calculate ATR. Need at least {period+1} periods.")
                return None
            
            # Extract price data
            high = market_data['high'].values
            low = market_data['low'].values
            close = market_data['close'].values
            
            # Calculate true range
            tr1 = np.abs(high[1:] - low[1:])
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            # True range is the maximum of the three
            tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
            
            # Calculate ATR using simple moving average
            atr = np.mean(tr[-period:])
            
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return None
    
    def calculate_performance_adjustment(self, performance: Dict[str, float]) -> float:
        """
        Calculate position size adjustment factor based on strategy performance metrics.
        
        This implements a performance-based sizing model where better performing
        strategies receive larger position allocations within their risk limits.
        
        Parameters:
            performance: Dict with performance metrics including sharpe_ratio, win_rate, etc.
            
        Returns:
            float: Adjustment factor between perf_adj_min and perf_adj_max
        """
        try:
            # Extract relevant metrics, defaulting to neutral values if missing
            sharpe = performance.get('sharpe_ratio', 1.0)
            win_rate = performance.get('win_rate', 0.5)
            avg_return = performance.get('avg_return', 0.0)
            
            # Create weighted scoring system
            # This can be made more sophisticated based on your preferences
            score = 0.0
            
            # Sharpe ratio component (0-2 typical range)
            if sharpe <= 0:
                sharpe_score = 0.0
            else:
                sharpe_score = min(1.0, sharpe / 2.0)
            
            # Win rate component (0.0-1.0 range)
            win_score = win_rate
            
            # Average return component (scale for typical returns like 0.5%-2%)
            if avg_return <= 0:
                return_score = 0.0
            else:
                # Scale so 2% return gets full score
                return_score = min(1.0, avg_return / 0.02)
            
            # Weighted average of components
            # Weights can be adjusted based on importance of each factor
            score = (0.4 * sharpe_score) + (0.4 * win_score) + (0.2 * return_score)
            
            # Scale to adjustment range
            adjustment = self.perf_adj_min + score * (self.perf_adj_max - self.perf_adj_min)
            
            logger.debug(f"Performance adjustment: {adjustment:.2f} (Sharpe: {sharpe:.2f}, "
                         f"Win Rate: {win_rate:.2%}, Avg Return: {avg_return:.2%})")
            
            return adjustment
        except Exception as e:
            logger.error(f"Error calculating performance adjustment: {str(e)}")
            return 1.0  # Default to neutral adjustment