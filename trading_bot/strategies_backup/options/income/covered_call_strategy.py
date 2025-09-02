#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Covered Call Strategy Module

This module implements a covered call strategy that generates recurring premium income
by selling call options against long stock or ETF positions while maintaining limited
upside participation.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
import math

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

class CoveredCallStrategy(StrategyOptimizable):
    """
    Covered Call Strategy designed to generate recurring premium income.
    
    This strategy involves selling call options against long stock or ETF positions,
    capturing time decay while participating in upside movement up to the call strike price.
    It aims to enhance returns on existing positions or establish new positions with reduced cost basis.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Covered Call strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy parameters
            metadata: Strategy metadata
        """
        # Default parameters
        default_params = {
            # 1. Strategy Philosophy parameters
            "strategy_name": "covered_call",
            "strategy_version": "1.0.0",
            
            # 2. Underlying & Option Universe parameters
            "universe": ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOG", "META", "NVDA", "BRK.B"],
            "data_granularity": "daily",
            "intraday_execution_granularity": "1min",
            "holding_period_days": 45,
            
            # 3. Selection Criteria parameters
            "min_adv": 500000,               # Minimum average daily volume (500K)
            "min_option_open_interest": 1000, # Minimum open interest for calls
            "max_bid_ask_spread_pct": 0.001,  # Maximum bid-ask spread as % (0.10%)
            "min_iv_rank": 30,                # Minimum IV rank (30%)
            "max_iv_rank": 60,                # Maximum IV rank (60%)
            
            # 4. Strike Selection parameters
            "otm_buffer_pct": 0.03,           # Default OTM buffer (3%)
            "delta_target": 0.25,             # Target delta (0.20-0.30)
            "strike_selection_method": "delta", # Options: "delta", "otm_percentage"
            "strong_trend_otm_pct": 0.02,     # OTM % for strong trends
            "weak_trend_otm_pct": 0.05,       # OTM % for weak trends
            
            # 5. Expiration Selection parameters
            "min_dte": 30,                    # Minimum days to expiration
            "max_dte": 45,                    # Maximum days to expiration
            "avoid_weeklies": True,           # Avoid weekly expirations
            "roll_dte_threshold": 7,          # Roll when DTE reaches this value
            
            # 6. Entry Execution parameters
            "use_limit_orders": True,         # Use limit orders
            "limit_order_buffer": 0.02,       # Limit order buffer (2%)
            "use_multi_leg_orders": True,     # Use multi-leg orders if available
            
            # 7. Exit & Management parameters
            "profit_take_pct": 0.50,          # Close at 50% of max profit
            "time_exit_dte": 7,               # Exit at 7 DTE
            "roll_when_tested": True,         # Roll when stock approaches strike
            "roll_threshold_pct": 0.02,       # Roll when within 2% of strike
            "underlying_stop_pct": 0.09,      # Exit if stock falls 9% below cost
            
            # 8. Position Sizing & Risk Control parameters
            "max_position_size_pct": 0.05,    # Max position size (5%)
            "max_covered_call_allocation": 0.25, # Max allocation to covered calls (25%)
            "max_sector_allocation": 0.10,    # Max allocation per sector (10%)
            
            # 9. Performance Metrics parameters
            "backtest_years": 3,              # Backtest window in years
            
            # 10. Optimization parameters
            "auto_adjust_strikes": True,      # Dynamically adjust strikes based on IV
            "high_iv_rank_threshold": 75,     # IV rank threshold for tighter strikes
            "low_iv_rank_threshold": 25,      # IV rank threshold for wider strikes
            "use_alternative_strategies": True, # Use alternative strategies in low IV
            "check_correlation": True,        # Check correlation between positions
            "correlation_threshold": 0.70     # Maximum correlation allowed
        }
        
        # Merge with provided parameters
        if parameters:
            default_params.update(parameters)
        
        super().__init__(name=name, parameters=default_params, metadata=metadata)
        
        logger.info(f"Initialized Covered Call strategy: {name}")
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Get parameter space for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return {
            "otm_buffer_pct": [0.02, 0.03, 0.04, 0.05],
            "delta_target": [0.20, 0.25, 0.30, 0.35],
            "min_dte": [25, 30, 35, 40],
            "max_dte": [40, 45, 50, 55],
            "profit_take_pct": [0.30, 0.40, 0.50, 0.60, 0.70],
            "time_exit_dte": [5, 7, 10, 14],
            "underlying_stop_pct": [0.07, 0.08, 0.09, 0.10]
        }
    
    # === 1. Strategy Philosophy Implementation ===
    def _get_market_regime(self, symbol: str, data: pd.DataFrame) -> str:
        """
        Determine current market regime to inform strategy decisions.
        
        Args:
            symbol: Ticker symbol
            data: Historical price data for the symbol
            
        Returns:
            Market regime classification ('bullish', 'neutral', 'bearish')
        """
        # Calculate 50-day SMA
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Calculate EMA slope over 20 days
        data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
        ema_slope = (data['ema_20'].iloc[-1] - data['ema_20'].iloc[-21]) / data['ema_20'].iloc[-21]
        
        # Determine regime
        latest_price = data['close'].iloc[-1]
        latest_sma = data['sma_50'].iloc[-1]
        
        if latest_price > latest_sma and ema_slope > 0.01:
            return 'bullish'
        elif latest_price < latest_sma and ema_slope < -0.01:
            return 'bearish'
        else:
            return 'neutral'
    
    # === 2. Underlying & Option Universe Selection ===
    def filter_universe(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter universe of symbols based on liquidity, trend, and option chain quality.
        
        Args:
            data: Dictionary of all available symbols with their data
            
        Returns:
            Filtered dictionary of symbols suitable for covered calls
        """
        filtered_data = {}
        
        for symbol, df in data.items():
            if symbol not in self.parameters.get("universe", []):
                continue
                
            # Check minimum data length
            if len(df) < 100:  # Need enough data to calculate indicators
                continue
                
            # Check average daily volume
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            latest_adv = df['volume_ma_20'].iloc[-1]
            
            if latest_adv < self.parameters.get("min_adv", 500000):
                logger.debug(f"{symbol} ADV {latest_adv:.0f} below minimum threshold")
                continue
            
            # Check trend conditions
            regime = self._get_market_regime(symbol, df)
            if regime == 'bearish':
                logger.debug(f"{symbol} in bearish trend, unsuitable for covered calls")
                continue
            
            # Include price data for symbols that pass initial filters
            filtered_data[symbol] = df
            
        logger.info(f"Filtered universe contains {len(filtered_data)} symbols for covered calls")
        return filtered_data
    
    # === 3. Selection & Filter Criteria ===
    def _calculate_iv_rank(self, data: pd.DataFrame, window: int = 252) -> float:
        """
        Calculate Implied Volatility Rank for a symbol.
        
        Args:
            data: DataFrame containing historical IV data
            window: Lookback window for IV rank calculation
            
        Returns:
            IV rank as percentage (0-100)
        """
        if 'implied_volatility' not in data.columns:
            return None
            
        # Get IV history for rank calculation
        iv_history = data['implied_volatility'].dropna().tail(window)
        
        if len(iv_history) < 30:  # Need enough history for reliable ranking
            return None
            
        current_iv = iv_history.iloc[-1]
        min_iv = iv_history.min()
        max_iv = iv_history.max()
        
        # Avoid division by zero
        if max_iv == min_iv:
            return 50.0
            
        # Calculate IV rank as percentage
        iv_rank = (current_iv - min_iv) / (max_iv - min_iv) * 100
        
        return iv_rank
    
    def _check_option_liquidity(self, option_chain: pd.DataFrame) -> bool:
        """
        Check if option chain meets liquidity requirements.
        
        Args:
            option_chain: DataFrame with option chain data
            
        Returns:
            Boolean indicating if liquidity conditions are met
        """
        if option_chain is None or option_chain.empty:
            return False
            
        # Filter for call options
        calls = option_chain[option_chain['option_type'] == 'call']
        
        if calls.empty:
            return False
            
        # Check open interest
        min_open_interest = self.parameters.get("min_option_open_interest", 1000)
        if calls['open_interest'].max() < min_open_interest:
            return False
            
        # Check bid-ask spreads
        max_bid_ask_pct = self.parameters.get("max_bid_ask_spread_pct", 0.001)
        
        # Calculate bid-ask spread as percentage of mid price
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
        calls['spread_pct'] = (calls['ask'] - calls['bid']) / calls['mid_price']
        
        # Check if any suitable options exist with acceptable spread
        suitable_options = calls[calls['spread_pct'] <= max_bid_ask_pct]
        
        return not suitable_options.empty
    
    # === 4. Strike Selection ===
    def _select_call_strike(
        self, 
        option_chain: pd.DataFrame, 
        current_price: float, 
        market_regime: str
    ) -> Dict[str, Any]:
        """
        Select appropriate call strike based on strategy parameters.
        
        Args:
            option_chain: DataFrame with option chain data
            current_price: Current price of underlying
            market_regime: Market regime ('bullish', 'neutral', 'bearish')
            
        Returns:
            Dictionary with selected call option details
        """
        if option_chain is None or option_chain.empty:
            return None
            
        # Filter for call options
        calls = option_chain[option_chain['option_type'] == 'call']
        
        if calls.empty:
            return None
            
        # Determine target OTM percentage based on market regime
        if market_regime == 'bullish':
            otm_pct = self.parameters.get("strong_trend_otm_pct", 0.02)
        elif market_regime == 'neutral':
            otm_pct = self.parameters.get("otm_buffer_pct", 0.03)
        else:
            otm_pct = self.parameters.get("weak_trend_otm_pct", 0.05)
            
        # Determine strike selection method
        selection_method = self.parameters.get("strike_selection_method", "delta")
        
        if selection_method == "delta":
            target_delta = self.parameters.get("delta_target", 0.25)
            
            # Find call option closest to target delta
            calls['delta_diff'] = abs(calls['delta'] - target_delta)
            selected_call = calls.loc[calls['delta_diff'].idxmin()]
            
        else:  # OTM percentage method
            target_strike = current_price * (1 + otm_pct)
            
            # Find call option closest to target strike
            calls['strike_diff'] = abs(calls['strike'] - target_strike)
            selected_call = calls.loc[calls['strike_diff'].idxmin()]
        
        # Calculate additional metrics
        premium_yield = selected_call['bid'] / current_price * 100  # Premium as % of stock price
        days_to_expiration = (pd.to_datetime(selected_call['expiration_date']) - pd.to_datetime(date.today())).days
        annualized_yield = premium_yield * (365 / days_to_expiration) if days_to_expiration > 0 else 0
        
        # Return selected call details
        return {
            'strike': selected_call['strike'],
            'expiration': selected_call['expiration_date'],
            'bid': selected_call['bid'],
            'ask': selected_call['ask'],
            'delta': selected_call['delta'],
            'dte': days_to_expiration,
            'premium_yield': premium_yield,
            'annualized_yield': annualized_yield,
            'option_symbol': selected_call['option_symbol'] if 'option_symbol' in selected_call else None
        }
    
    # === 5. Expiration Selection ===
    def _filter_expirations(self, option_chain: pd.DataFrame) -> pd.DataFrame:
        """
        Filter option chain for suitable expirations.
        
        Args:
            option_chain: DataFrame with option chain data
            
        Returns:
            Filtered option chain
        """
        if option_chain is None or option_chain.empty:
            return pd.DataFrame()
            
        min_dte = self.parameters.get("min_dte", 30)
        max_dte = self.parameters.get("max_dte", 45)
        avoid_weeklies = self.parameters.get("avoid_weeklies", True)
        
        # Convert expiration to datetime and calculate DTE
        option_chain['expiration_datetime'] = pd.to_datetime(option_chain['expiration_date'])
        option_chain['dte'] = (option_chain['expiration_datetime'] - pd.to_datetime(date.today())).dt.days
        
        # Filter by DTE
        filtered_chain = option_chain[(option_chain['dte'] >= min_dte) & (option_chain['dte'] <= max_dte)]
        
        if filtered_chain.empty:
            return pd.DataFrame()
            
        # Optionally filter out weeklies
        if avoid_weeklies:
            # Identify standard monthly expirations (typically 3rd Friday)
            filtered_chain['expiration_day'] = filtered_chain['expiration_datetime'].dt.day
            filtered_chain['expiration_weekday'] = filtered_chain['expiration_datetime'].dt.weekday
            
            # Keep only standard monthly expirations (approximation: day 15-21 and Friday)
            monthly_expirations = filtered_chain[
                (filtered_chain['expiration_day'] >= 15) & 
                (filtered_chain['expiration_day'] <= 21) & 
                (filtered_chain['expiration_weekday'] == 4)  # Friday
            ]
            
            if not monthly_expirations.empty:
                return monthly_expirations
        
        return filtered_chain
    
    def _check_roll_conditions(self, position: Dict[str, Any], current_price: float) -> bool:
        """
        Check if position should be rolled to the next expiration.
        
        Args:
            position: Current position details
            current_price: Current price of underlying
            
        Returns:
            Boolean indicating if position should be rolled
        """
        if not position:
            return False
            
        # Check DTE threshold
        call_dte = position.get('call_dte', 0)
        dte_threshold = self.parameters.get("roll_dte_threshold", 7)
        
        if call_dte <= dte_threshold:
            logger.info(f"Roll condition met: DTE {call_dte} <= threshold {dte_threshold}")
            return True
            
        # Check if stock approaches strike price
        call_strike = position.get('call_strike', 0)
        roll_threshold_pct = self.parameters.get("roll_threshold_pct", 0.02)
        
        if call_strike > 0 and current_price > call_strike * (1 - roll_threshold_pct):
            logger.info(f"Roll condition met: Price {current_price:.2f} approaching strike {call_strike:.2f}")
            return True
            
        return False
    
    # === 6. Entry Execution ===
    def _prepare_stock_order(self, symbol: str, current_price: float, account_value: float) -> Dict[str, Any]:
        """
        Prepare order for purchasing underlying stock.
        
        Args:
            symbol: Ticker symbol
            current_price: Current price of underlying
            account_value: Total account value
            
        Returns:
            Stock order details
        """
        max_position_pct = self.parameters.get("max_position_size_pct", 0.05)
        max_position_value = account_value * max_position_pct
        
        # Calculate number of shares to purchase
        shares = math.floor(max_position_value / current_price)
        
        if shares <= 0:
            return None
            
        # Prepare order details
        use_limit = self.parameters.get("use_limit_orders", True)
        limit_buffer = self.parameters.get("limit_order_buffer", 0.02)
        
        order_type = "LIMIT" if use_limit else "MARKET"
        limit_price = round(current_price * (1 + limit_buffer), 2) if use_limit else None
        
        return {
            "symbol": symbol,
            "order_type": order_type,
            "action": "BUY",
            "quantity": shares,
            "limit_price": limit_price,
            "order_details": {
                "strategy": "covered_call",
                "leg": "stock",
                "expected_cost": current_price * shares
            }
        }
    
    def _prepare_call_order(self, symbol: str, call_details: Dict[str, Any], shares: int) -> Dict[str, Any]:
        """
        Prepare order for selling call options.
        
        Args:
            symbol: Ticker symbol
            call_details: Selected call option details
            shares: Number of shares held/purchased
            
        Returns:
            Call option order details
        """
        if not call_details or shares <= 0:
            return None
            
        # Calculate number of contracts (1 contract = 100 shares)
        contracts = math.floor(shares / 100)
        
        if contracts <= 0:
            return None
            
        # Prepare order details
        limit_price = call_details['bid']  # Sell at the bid price
        
        return {
            "symbol": symbol,
            "option_symbol": call_details['option_symbol'],
            "order_type": "LIMIT",
            "action": "SELL",
            "quantity": contracts,
            "limit_price": limit_price,
            "order_details": {
                "strategy": "covered_call",
                "leg": "call",
                "strike": call_details['strike'],
                "expiration": call_details['expiration'],
                "dte": call_details['dte'],
                "premium": limit_price * contracts * 100,
                "yield": call_details['premium_yield'],
                "annualized_yield": call_details['annualized_yield']
            }
        }
    
    # === 7. Exit & Management Rules ===
    def _check_exit_conditions(self, position: Dict[str, Any], current_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if exit conditions are met for an existing position.
        
        Args:
            position: Current position details
            current_data: Current market data including prices
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if not position:
            return False, ""
            
        symbol = position.get('symbol')
        
        if not symbol or symbol not in current_data:
            return False, ""
            
        current_price = current_data[symbol].iloc[-1]['close']
        entry_price = position.get('stock_entry_price', current_price)
        current_call_price = position.get('current_call_price')
        initial_call_price = position.get('initial_call_price')
        call_dte = position.get('call_dte', 0)
        
        # Check profit target for call option
        if current_call_price is not None and initial_call_price is not None:
            profit_take_pct = self.parameters.get("profit_take_pct", 0.50)
            call_profit_pct = (initial_call_price - current_call_price) / initial_call_price
            
            if call_profit_pct >= profit_take_pct:
                return True, f"Call profit target reached: {call_profit_pct:.1%} >= {profit_take_pct:.1%}"
                
        # Check time exit
        time_exit_dte = self.parameters.get("time_exit_dte", 7)
        if call_dte <= time_exit_dte:
            return True, f"DTE threshold reached: {call_dte} <= {time_exit_dte}"
            
        # Check stop loss on underlying
        underlying_stop_pct = self.parameters.get("underlying_stop_pct", 0.09)
        price_drop_pct = (entry_price - current_price) / entry_price
        
        if price_drop_pct >= underlying_stop_pct:
            return True, f"Underlying stop loss triggered: {price_drop_pct:.1%} >= {underlying_stop_pct:.1%}"
            
        return False, ""
    
    def _prepare_roll_order(self, position: Dict[str, Any], new_call_details: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare orders for rolling a covered call to the next expiration.
        
        Args:
            position: Current position details
            new_call_details: Details of the new call option
            
        Returns:
            List of orders for roll operation
        """
        if not position or not new_call_details:
            return []
            
        symbol = position.get('symbol')
        current_call_symbol = position.get('call_option_symbol')
        current_call_price = position.get('current_call_price')
        contracts = position.get('call_contracts', 0)
        
        if not symbol or not current_call_symbol or contracts <= 0:
            return []
            
        orders = []
        
        # Buy back current call
        orders.append({
            "symbol": symbol,
            "option_symbol": current_call_symbol,
            "order_type": "LIMIT",
            "action": "BUY",
            "quantity": contracts,
            "limit_price": current_call_price,
            "order_details": {
                "strategy": "covered_call",
                "leg": "close_call",
                "roll": True
            }
        })
        
        # Sell new call
        orders.append({
            "symbol": symbol,
            "option_symbol": new_call_details['option_symbol'],
            "order_type": "LIMIT",
            "action": "SELL",
            "quantity": contracts,
            "limit_price": new_call_details['bid'],
            "order_details": {
                "strategy": "covered_call",
                "leg": "new_call",
                "strike": new_call_details['strike'],
                "expiration": new_call_details['expiration'],
                "dte": new_call_details['dte'],
                "premium": new_call_details['bid'] * contracts * 100,
                "yield": new_call_details['premium_yield'],
                "annualized_yield": new_call_details['annualized_yield'],
                "roll": True
            }
        })
        
        return orders
    
    # === 8. Position Sizing & Risk Controls ===
    def _calculate_position_allocation(self, symbol: str, portfolio: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate appropriate position allocation considering existing positions.
        
        Args:
            symbol: Ticker symbol
            portfolio: Current portfolio holdings
            
        Returns:
            Maximum allocation for this position as percentage
        """
        max_position_pct = self.parameters.get("max_position_size_pct", 0.05)
        max_covered_call_alloc = self.parameters.get("max_covered_call_allocation", 0.25)
        max_sector_alloc = self.parameters.get("max_sector_allocation", 0.10)
        
        # Calculate current covered call allocation
        current_covered_call_alloc = 0
        sector_allocation = 0
        symbol_sector = self._get_symbol_sector(symbol)
        
        for pos_symbol, position in portfolio.items():
            if position.get('strategy') == 'covered_call':
                current_covered_call_alloc += position.get('allocation', 0)
                
                # Track sector allocation
                if self._get_symbol_sector(pos_symbol) == symbol_sector:
                    sector_allocation += position.get('allocation', 0)
        
        # Calculate constraints
        remaining_strategy_alloc = max(0, max_covered_call_alloc - current_covered_call_alloc)
        remaining_sector_alloc = max(0, max_sector_alloc - sector_allocation)
        
        # Return the most restrictive constraint
        return min(max_position_pct, remaining_strategy_alloc, remaining_sector_alloc)
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """
        Get sector classification for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Sector classification string
        """
        # This would typically integrate with a data source to get sector info
        # For simplicity, using a basic mapping for common ETFs and stocks
        sector_map = {
            'SPY': 'Broad Market',
            'QQQ': 'Technology',
            'IWM': 'Small Cap',
            'DIA': 'Large Cap',
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLV': 'Healthcare',
            'XLI': 'Industrials',
            'XLP': 'Consumer Staples',
            'XLY': 'Consumer Discretionary',
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'GOOG': 'Communication Services',
            'META': 'Communication Services',
            'NVDA': 'Technology',
            'JPM': 'Financials',
            'JNJ': 'Healthcare',
            'PG': 'Consumer Staples'
        }
        
        return sector_map.get(symbol, 'Other')
    
    # === 9. Backtesting & Performance Metrics ===
    def calculate_performance_metrics(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate key performance metrics from backtest results.
        
        Args:
            backtest_results: DataFrame with backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        if backtest_results.empty:
            return {}
            
        metrics = {}
        
        # Calculate basic metrics
        total_trades = len(backtest_results)
        winning_trades = len(backtest_results[backtest_results['profit'] > 0])
        
        metrics['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
        
        # Premium metrics
        metrics['total_premium'] = backtest_results['premium'].sum()
        metrics['avg_premium_yield'] = backtest_results['premium_yield'].mean()
        metrics['annualized_premium_yield'] = backtest_results['annualized_yield'].mean()
        
        # Assignment metrics
        assignment_count = len(backtest_results[backtest_results['assigned'] == True])
        metrics['assignment_rate'] = assignment_count / total_trades if total_trades > 0 else 0
        
        # Return metrics
        metrics['avg_return_per_cycle'] = backtest_results['cycle_return'].mean()
        metrics['annualized_return'] = (1 + metrics['avg_return_per_cycle']) ** (12 / metrics['avg_holding_period']) - 1 if 'avg_holding_period' in metrics else 0
        
        # Drawdown metrics
        if 'equity_curve' in backtest_results:
            equity_curve = backtest_results['equity_curve']
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
        return metrics
    
    # === 10. Continuous Optimization ===
    def optimize_parameters(self, historical_performance: pd.DataFrame) -> Dict[str, Any]:
        """
        Optimize strategy parameters based on historical performance.
        
        Args:
            historical_performance: DataFrame with historical performance data
            
        Returns:
            Optimized parameters
        """
        optimized_params = {}
        
        # Quarterly review logic
        if not historical_performance.empty:
            # Optimize OTM buffer based on assignment rate
            assignment_rate = historical_performance['assigned'].mean()
            
            if assignment_rate > 0.3:  # Too many assignments
                optimized_params['otm_buffer_pct'] = min(0.05, self.parameters.get('otm_buffer_pct', 0.03) + 0.01)
            elif assignment_rate < 0.1:  # Too few assignments
                optimized_params['otm_buffer_pct'] = max(0.01, self.parameters.get('otm_buffer_pct', 0.03) - 0.01)
                
            # Optimize delta target based on returns
            if 'avg_return_per_cycle' in historical_performance.columns:
                avg_return = historical_performance['avg_return_per_cycle'].mean()
                
                if avg_return < 0.01:  # Low returns
                    optimized_params['delta_target'] = min(0.40, self.parameters.get('delta_target', 0.25) + 0.05)
                elif avg_return > 0.03 and assignment_rate > 0.3:  # High returns but too many assignments
                    optimized_params['delta_target'] = max(0.15, self.parameters.get('delta_target', 0.25) - 0.05)
        
        # IV-adaptive strike adjustments
        if self.parameters.get("auto_adjust_strikes", True):
            high_iv_threshold = self.parameters.get("high_iv_rank_threshold", 75)
            low_iv_threshold = self.parameters.get("low_iv_rank_threshold", 25)
            
            # Logic to adjust strikes based on IV rank would go here
            # We would need current IV rank data to implement this
            
        return optimized_params
    
    # === Main Strategy Methods ===
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Calculate indicators for the covered call strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with price and option data
            
        Returns:
            Dictionary of calculated indicators for each symbol
        """
        indicators = {}
        
        for symbol, df in data.items():
            try:
                # Calculate trend indicators
                df_with_indicators = df.copy()
                
                # SMA 50
                df_with_indicators['sma_50'] = df['close'].rolling(window=50).mean()
                
                # EMA 20
                df_with_indicators['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
                
                # EMA slope
                df_with_indicators['ema_20_prev'] = df_with_indicators['ema_20'].shift(20)
                df_with_indicators['ema_slope'] = (df_with_indicators['ema_20'] - df_with_indicators['ema_20_prev']) / df_with_indicators['ema_20_prev']
                
                # ADX for trend strength (if data available)
                if 'high' in df.columns and 'low' in df.columns:
                    df_with_indicators = self._calculate_adx(df_with_indicators)
                
                # IV Rank (if data available)
                if 'implied_volatility' in df.columns:
                    df_with_indicators['iv_rank'] = df.apply(lambda row: self._calculate_iv_rank(df.loc[:row.name]), axis=1)
                
                # Store indicators
                indicators[symbol] = {
                    "trend": pd.DataFrame({
                        "sma_50": df_with_indicators['sma_50'],
                        "ema_20": df_with_indicators['ema_20'],
                        "ema_slope": df_with_indicators['ema_slope'],
                        "price_to_sma50": df_with_indicators['close'] / df_with_indicators['sma_50']
                    }),
                    "volatility": pd.DataFrame({
                        "iv_rank": df_with_indicators['iv_rank'] if 'iv_rank' in df_with_indicators.columns else None,
                        "adx": df_with_indicators['adx'] if 'adx' in df_with_indicators.columns else None
                    })
                }
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {symbol}: {e}")
        
        return indicators
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) indicator.
        
        Args:
            df: DataFrame with price data
            period: ADX period
            
        Returns:
            DataFrame with ADX indicator
        """
        df = df.copy()
        
        # Calculate True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift())
        df['tr2'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate Directional Movement
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        df['plus_dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        
        df['minus_dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Calculate Directional Indicators
        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
        
        # Calculate Directional Index
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        
        # Calculate ADX
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Signal]:
        """
        Generate trading signals for covered call strategy.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with price and option data
            
        Returns:
            Dictionary mapping symbols to Signal objects
        """
        # Filter universe to suitable candidates
        filtered_data = self.filter_universe(data)
        
        # Calculate indicators
        indicators = self.calculate_indicators(filtered_data)
        
        # Generate signals
        signals = {}
        
        for symbol, symbol_data in filtered_data.items():
            try:
                # Get latest data
                latest_data = symbol_data.iloc[-1]
                latest_price = latest_data['close']
                latest_timestamp = latest_data.name if isinstance(latest_data.name, datetime) else datetime.now()
                
                # Get option chain data (assuming it's provided in the data dictionary)
                option_chain = self._get_option_chain(symbol, data)
                
                if option_chain is None or option_chain.empty:
                    logger.debug(f"No option chain data available for {symbol}")
                    continue
                
                # Check option liquidity
                if not self._check_option_liquidity(option_chain):
                    logger.debug(f"Option chain for {symbol} doesn't meet liquidity requirements")
                    continue
                
                # Determine market regime
                market_regime = self._get_market_regime(symbol, symbol_data)
                
                # Filter for suitable expirations
                filtered_chain = self._filter_expirations(option_chain)
                
                if filtered_chain.empty:
                    logger.debug(f"No suitable expirations for {symbol}")
                    continue
                
                # Select call strike
                call_details = self._select_call_strike(filtered_chain, latest_price, market_regime)
                
                if not call_details:
                    logger.debug(f"Could not select appropriate call strike for {symbol}")
                    continue
                
                # Calculate confidence based on indicators
                trend_confidence = self._calculate_trend_confidence(symbol, indicators)
                iv_confidence = self._calculate_iv_confidence(symbol, indicators)
                premium_confidence = min(1.0, call_details['annualized_yield'] / 25.0)  # Scale based on annualized yield
                
                # Combine confidence scores
                confidence = 0.4 * trend_confidence + 0.3 * iv_confidence + 0.3 * premium_confidence
                
                # Create signal
                signals[symbol] = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,  # For covered call, we're buying the underlying
                    price=latest_price,
                    timestamp=latest_timestamp,
                    confidence=confidence,
                    stop_loss=latest_price * (1 - self.parameters.get("underlying_stop_pct", 0.09)),
                    take_profit=call_details['strike'],  # Max profit if called away at strike
                    metadata={
                        "strategy_type": "covered_call",
                        "market_regime": market_regime,
                        "call_strike": call_details['strike'],
                        "call_expiration": call_details['expiration'],
                        "call_dte": call_details['dte'],
                        "call_premium": call_details['bid'],
                        "premium_yield": call_details['premium_yield'],
                        "annualized_yield": call_details['annualized_yield'],
                        "call_delta": call_details['delta']
                    }
                )
                
                logger.info(f"Generated covered call signal for {symbol}: {call_details['strike']} strike, {call_details['premium_yield']:.2f}% yield")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _get_option_chain(self, symbol: str, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Get option chain data for a symbol.
        
        Args:
            symbol: Ticker symbol
            data: Dictionary with market data
            
        Returns:
            DataFrame with option chain data or None if not available
        """
        # Check if option chain is directly included in the data dictionary
        option_key = f"{symbol}_options"
        if option_key in data:
            return data[option_key]
            
        # This method would typically interface with the system's option chain data provider
        # For now, return None as a placeholder
        return None
    
    def _calculate_trend_confidence(self, symbol: str, indicators: Dict[str, Dict[str, pd.DataFrame]]) -> float:
        """
        Calculate confidence level based on trend indicators.
        
        Args:
            symbol: Ticker symbol
            indicators: Dictionary of indicators
            
        Returns:
            Confidence score (0-1)
        """
        if symbol not in indicators or "trend" not in indicators[symbol]:
            return 0.5
            
        trend_data = indicators[symbol]["trend"].iloc[-1]
        
        # Price relative to SMA50
        price_to_sma = trend_data.get("price_to_sma50", 1.0)
        sma_confidence = min(1.0, max(0, (price_to_sma - 0.97) / 0.06))  # Scale: 0.97-1.03 → 0-1
        
        # EMA slope
        ema_slope = trend_data.get("ema_slope", 0)
        slope_confidence = min(1.0, max(0, (ema_slope + 0.01) / 0.03))  # Scale: -0.01-0.02 → 0-1
        
        # Combine trend confidence factors
        return 0.6 * sma_confidence + 0.4 * slope_confidence
    
    def _calculate_iv_confidence(self, symbol: str, indicators: Dict[str, Dict[str, pd.DataFrame]]) -> float:
        """
        Calculate confidence level based on volatility indicators.
        
        Args:
            symbol: Ticker symbol
            indicators: Dictionary of indicators
            
        Returns:
            Confidence score (0-1)
        """
        if symbol not in indicators or "volatility" not in indicators[symbol]:
            return 0.5
            
        vol_data = indicators[symbol]["volatility"].iloc[-1]
        
        # IV Rank confidence
        iv_rank = vol_data.get("iv_rank")
        
        if iv_rank is None:
            return 0.5
            
        # Optimal IV rank is between 30-60%
        min_iv = self.parameters.get("min_iv_rank", 30)
        max_iv = self.parameters.get("max_iv_rank", 60)
        
        if iv_rank < min_iv:
            iv_confidence = iv_rank / min_iv * 0.8  # Scale up to 0.8 as IV rank approaches min
        elif iv_rank > max_iv:
            iv_confidence = max(0, 1.0 - (iv_rank - max_iv) / (100 - max_iv))
        else:
            # Within optimal range
            iv_range_width = max_iv - min_iv
            position_in_range = (iv_rank - min_iv) / iv_range_width
            
            # Highest confidence at middle of range
            iv_confidence = 0.8 + 0.2 * (1 - abs(position_in_range - 0.5) * 2)
            
        return iv_confidence 