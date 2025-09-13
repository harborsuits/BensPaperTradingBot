#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Vantage Backtester

This module implements a backtesting system that uses Alpha Vantage data
for historical prices and technical indicators.
"""

import logging
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any, Callable
from datetime import datetime, timedelta

from trading_bot.backtest.base_backtester import BaseBacktester
from trading_bot.signals.alpha_vantage_signals import AlphaVantageTechnicalSignals

logger = logging.getLogger(__name__)

class AlphaVantageBacktester(BaseBacktester):
    """Backtester implementation that uses Alpha Vantage data"""
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        data_source: Optional[AlphaVantageTechnicalSignals] = None,
        commission: float = 0.005,  # 0.5% per trade
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        indicators_config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        data_dir: str = "backtest_data"
    ):
        """Initialize the Alpha Vantage backtester.
        
        Args:
            initial_capital: Initial capital for the backtest
            data_source: AlphaVantageTechnicalSignals instance
            commission: Commission rate per trade
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            indicators_config: Technical indicators configuration
            api_key: Alpha Vantage API key
            data_dir: Directory for caching data
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.data_source = data_source
        self.commission = commission
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.indicators_config = indicators_config or {}
        self.api_key = api_key
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Internal state
        self.positions = {}  # Current positions: {symbol: quantity}
        self.portfolio_value_history = []  # List of {date, value} dicts
        self.trades = []  # List of trade records
        self.data_cache = {}  # Cached data frames: {symbol: dataframe}
        
        # Trading strategies
        self.strategies = []
        
        logger.info("Initialized AlphaVantageBacktester")
    
    def add_strategy(self, name: str, strategy_func: Callable, params: Dict[str, Any] = None):
        """Add a trading strategy to the backtester.
        
        Args:
            name: Strategy name
            strategy_func: Strategy function that returns signals
            params: Strategy parameters
        """
        self.strategies.append({
            "name": name,
            "function": strategy_func,
            "params": params or {}
        })
        logger.info(f"Added strategy: {name}")
    
    def _load_or_fetch_data(self, symbol: str) -> pd.DataFrame:
        """Load cached data or fetch new data from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with historical price data
        """
        # Check cache
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        # Check if data is on disk
        cache_file = os.path.join(self.data_dir, f"{symbol}_daily.csv")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached data for {symbol}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            self.data_cache[symbol] = df
            return df
        
        # Fetch data from Alpha Vantage
        if not self.data_source:
            raise ValueError("Data source not provided and no cached data available")
        
        logger.info(f"Fetching new data for {symbol}")
        df = self.data_source.fetch_daily_data(symbol, full=True)
        
        if df is None or df.empty:
            raise ValueError(f"Failed to fetch data for {symbol}")
        
        # Cache to disk
        df.to_csv(cache_file)
        
        # Cache in memory
        self.data_cache[symbol] = df
        
        return df
    
    def _load_or_compute_indicators(self, symbol: str) -> Dict[str, pd.Series]:
        """Load cached indicators or compute them.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of indicator series
        """
        indicators = {}
        price_data = self._load_or_fetch_data(symbol)
        
        # Check for cached indicators
        cache_file = os.path.join(self.data_dir, f"{symbol}_indicators.json")
        
        if os.path.exists(cache_file):
            logger.info(f"Loading cached indicators for {symbol}")
            with open(cache_file, 'r') as f:
                indicator_data = json.load(f)
                
            # Convert to Series
            for name, values in indicator_data.items():
                indicators[name] = pd.Series(values, index=price_data.index)
                
            return indicators
        
        # Compute indicators using Alpha Vantage if available
        logger.info(f"Computing indicators for {symbol}")
        
        # Simple Moving Averages
        sma_periods = [
            self.indicators_config.get("sma_short", 20),
            self.indicators_config.get("sma_medium", 50),
            self.indicators_config.get("sma_long", 200)
        ]
        
        for period in sma_periods:
            indicators[f"SMA_{period}"] = price_data["Close"].rolling(window=period).mean()
        
        # Exponential Moving Averages
        ema_periods = [
            self.indicators_config.get("ema_short", 12),
            self.indicators_config.get("ema_medium", 26),
            self.indicators_config.get("ema_long", 50)
        ]
        
        for period in ema_periods:
            indicators[f"EMA_{period}"] = price_data["Close"].ewm(span=period, adjust=False).mean()
        
        # RSI
        rsi_period = self.indicators_config.get("rsi_period", 14)
        delta = price_data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        indicators["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        macd_fast = self.indicators_config.get("macd_fast", 12)
        macd_slow = self.indicators_config.get("macd_slow", 26)
        macd_signal = self.indicators_config.get("macd_signal", 9)
        
        ema_fast = price_data["Close"].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = price_data["Close"].ewm(span=macd_slow, adjust=False).mean()
        indicators["MACD"] = ema_fast - ema_slow
        indicators["MACD_Signal"] = indicators["MACD"].ewm(span=macd_signal, adjust=False).mean()
        indicators["MACD_Hist"] = indicators["MACD"] - indicators["MACD_Signal"]
        
        # Bollinger Bands
        bbands_period = self.indicators_config.get("bbands_period", 20)
        bbands_std = self.indicators_config.get("bbands_std", 2.0)
        
        indicators["BB_Middle"] = price_data["Close"].rolling(window=bbands_period).mean()
        rolling_std = price_data["Close"].rolling(window=bbands_period).std()
        indicators["BB_Upper"] = indicators["BB_Middle"] + (rolling_std * bbands_std)
        indicators["BB_Lower"] = indicators["BB_Middle"] - (rolling_std * bbands_std)
        
        # Cache indicators to disk
        indicator_data = {name: values.where(pd.notnull(values), None).to_dict() 
                         for name, values in indicators.items()}
        
        with open(cache_file, 'w') as f:
            json.dump(indicator_data, f)
        
        return indicators
    
    def _calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """Calculate portfolio value at a given date.
        
        Args:
            date: Date to calculate value for
            
        Returns:
            Total portfolio value
        """
        value = self.current_capital
        
        for symbol, quantity in self.positions.items():
            if quantity > 0:
                try:
                    price_data = self._load_or_fetch_data(symbol)
                    # Get closest date less than or equal to the given date
                    valid_dates = price_data.index[price_data.index <= date]
                    if not valid_dates.empty:
                        latest_date = valid_dates[-1]
                        price = price_data.loc[latest_date, "Close"]
                        value += price * quantity
                except Exception as e:
                    logger.error(f"Error calculating position value for {symbol}: {str(e)}")
        
        return value
    
    def _execute_buy(self, date: pd.Timestamp, symbol: str, amount: float, price: float):
        """Execute a buy order in the backtest.
        
        Args:
            date: Order date
            symbol: Stock symbol
            amount: Amount to invest (in dollars)
            price: Current price
        """
        if amount <= 0 or price <= 0:
            return
        
        # Check if we have enough capital
        if amount > self.current_capital:
            amount = self.current_capital  # Limit to available capital
        
        # Calculate quantity after commission
        effective_amount = amount * (1 - self.commission)
        quantity = effective_amount / price
        
        # Round down to whole shares
        quantity = int(quantity)
        
        if quantity <= 0:
            return
        
        # Update positions
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        
        # Update capital
        actual_cost = (quantity * price) * (1 + self.commission)
        self.current_capital -= actual_cost
        
        # Record trade
        self.trades.append({
            "date": date.strftime("%Y-%m-%d"),
            "type": "buy",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "value": quantity * price,
            "commission": actual_cost - (quantity * price),
            "capital_after": self.current_capital
        })
        
        logger.info(f"BUY: {quantity} shares of {symbol} at ${price:.2f} (${quantity * price:.2f})")
    
    def _execute_sell(self, date: pd.Timestamp, symbol: str, quantity: int, price: float):
        """Execute a sell order in the backtest.
        
        Args:
            date: Order date
            symbol: Stock symbol
            quantity: Quantity to sell
            price: Current price
        """
        if symbol not in self.positions or self.positions[symbol] <= 0:
            return
        
        # Limit to available shares
        quantity = min(quantity, self.positions[symbol])
        
        if quantity <= 0:
            return
        
        # Update positions
        self.positions[symbol] -= quantity
        
        # Update capital
        gross_sale = quantity * price
        net_sale = gross_sale * (1 - self.commission)
        self.current_capital += net_sale
        
        # Clean up if position is zero
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        # Record trade
        self.trades.append({
            "date": date.strftime("%Y-%m-%d"),
            "type": "sell",
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "value": gross_sale,
            "commission": gross_sale - net_sale,
            "capital_after": self.current_capital
        })
        
        logger.info(f"SELL: {quantity} shares of {symbol} at ${price:.2f} (${gross_sale:.2f})")
    
    def _process_signals(self, date: pd.Timestamp, signals: Dict[str, Any]):
        """Process trading signals for a given date.
        
        Args:
            date: Current date
            signals: Dictionary of trading signals
        """
        for symbol, signal in signals.items():
            # Skip if signal is neutral or missing
            if signal.get("signal") not in ["buy", "strong_buy", "sell", "strong_sell"]:
                continue
            
            try:
                price_data = self._load_or_fetch_data(symbol)
                
                # Get price for the current date
                if date not in price_data.index:
                    continue
                
                price = price_data.loc[date, "Close"]
                
                # Process buy signals
                if signal.get("signal") in ["buy", "strong_buy"]:
                    # Determine amount to invest
                    amount = self.current_capital * 0.1  # 10% of capital by default
                    
                    # Increase allocation for strong buy
                    if signal.get("signal") == "strong_buy":
                        amount = self.current_capital * 0.2  # 20% of capital
                    
                    self._execute_buy(date, symbol, amount, price)
                
                # Process sell signals
                elif signal.get("signal") in ["sell", "strong_sell"]:
                    if symbol in self.positions and self.positions[symbol] > 0:
                        quantity = self.positions[symbol]
                        
                        # Partial sell for regular sell signal
                        if signal.get("signal") == "sell":
                            quantity = int(quantity * 0.5)  # Sell 50%
                        
                        self._execute_sell(date, symbol, quantity, price)
            
            except Exception as e:
                logger.error(f"Error processing signal for {symbol} on {date}: {str(e)}")
    
    def run(self, symbols: List[str]) -> Dict[str, Any]:
        """Run the backtest on the given symbols.
        
        Args:
            symbols: List of stock symbols to backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with {len(symbols)} symbols")
        
        # Reset state
        self.current_capital = self.initial_capital
        self.positions = {}
        self.portfolio_value_history = []
        self.trades = []
        
        # Determine date range
        all_dates = set()
        
        for symbol in symbols:
            try:
                price_data = self._load_or_fetch_data(symbol)
                all_dates.update(price_data.index)
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
        
        all_dates = sorted(all_dates)
        
        if not all_dates:
            logger.error("No valid dates found in any price data")
            return {"error": "No valid dates"}
        
        # Apply date filters
        if self.start_date:
            all_dates = [d for d in all_dates if d >= self.start_date]
        
        if self.end_date:
            all_dates = [d for d in all_dates if d <= self.end_date]
        
        if not all_dates:
            logger.error("No dates within specified range")
            return {"error": "No dates in range"}
        
        logger.info(f"Backtest date range: {all_dates[0]} to {all_dates[-1]}")
        
        # Load indicators for all symbols
        logger.info("Loading/computing indicators for all symbols")
        indicators = {}
        
        for symbol in symbols:
            try:
                indicators[symbol] = self._load_or_compute_indicators(symbol)
            except Exception as e:
                logger.error(f"Error loading indicators for {symbol}: {str(e)}")
        
        # Run backtest day by day
        logger.info("Running day-by-day backtest simulation")
        
        for date in all_dates:
            # Calculate signals from all strategies
            daily_signals = {}
            
            for strategy in self.strategies:
                try:
                    # Prepare data for the strategy
                    strategy_data = {}
                    
                    for symbol in symbols:
                        if symbol in indicators:
                            price_data = self._load_or_fetch_data(symbol)
                            
                            # Filter data up to current date
                            symbol_data = {
                                "prices": price_data[price_data.index <= date],
                                "indicators": {
                                    name: series[series.index <= date]
                                    for name, series in indicators[symbol].items()
                                }
                            }
                            
                            strategy_data[symbol] = symbol_data
                    
                    # Execute strategy
                    strategy_signals = strategy["function"](
                        date=date,
                        data=strategy_data,
                        params=strategy["params"]
                    )
                    
                    # Merge signals
                    for symbol, signal in strategy_signals.items():
                        if symbol not in daily_signals:
                            daily_signals[symbol] = signal
                        else:
                            # Simple signal combination logic - more sophisticated logic could be implemented
                            current = daily_signals[symbol].get("signal", "neutral")
                            new = signal.get("signal", "neutral")
                            
                            signal_strength = {
                                "strong_buy": 2,
                                "buy": 1,
                                "neutral": 0,
                                "sell": -1,
                                "strong_sell": -2
                            }
                            
                            combined_strength = signal_strength.get(current, 0) + signal_strength.get(new, 0)
                            
                            if combined_strength >= 2:
                                daily_signals[symbol]["signal"] = "strong_buy"
                            elif combined_strength == 1:
                                daily_signals[symbol]["signal"] = "buy"
                            elif combined_strength == -1:
                                daily_signals[symbol]["signal"] = "sell"
                            elif combined_strength <= -2:
                                daily_signals[symbol]["signal"] = "strong_sell"
                            else:
                                daily_signals[symbol]["signal"] = "neutral"
                
                except Exception as e:
                    logger.error(f"Error executing strategy {strategy['name']} on {date}: {str(e)}")
            
            # Process signals for the day
            self._process_signals(date, daily_signals)
            
            # Record portfolio value
            portfolio_value = self._calculate_portfolio_value(date)
            
            self.portfolio_value_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": portfolio_value
            })
        
        # Calculate performance metrics
        return self._calculate_performance()
    
    def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.portfolio_value_history:
            return {"error": "No portfolio history"}
        
        # Extract values and dates
        dates = [pd.to_datetime(entry["date"]) for entry in self.portfolio_value_history]
        values = [entry["value"] for entry in self.portfolio_value_history]
        
        # Create DataFrame for calculations
        df = pd.DataFrame({"value": values}, index=dates)
        df["daily_return"] = df["value"].pct_change()
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = df["value"].iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        # Annualized return
        days = (dates[-1] - dates[0]).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (365 / days)) - 1
        else:
            annualized_return = 0
        
        # Risk metrics
        daily_returns = df["daily_return"].dropna()
        
        volatility = daily_returns.std() * (252 ** 0.5)  # Annualized
        
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = 0  # Default
        
        if volatility > 0:
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility
        
        # Drawdown analysis
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod()
        df["cumulative_max"] = df["cumulative_return"].cummax()
        df["drawdown"] = (df["cumulative_return"] / df["cumulative_max"]) - 1
        
        max_drawdown = df["drawdown"].min()
        
        # Compile results
        results = {
            "initial_capital": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": annualized_return,
            "annualized_return_pct": annualized_return * 100,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "number_of_trades": len(self.trades),
            "positions": self.positions,
            "portfolio_history": self.portfolio_value_history,
            "trades": self.trades
        }
        
        return results
    
    def plot_equity_curve(self, filename: Optional[str] = None) -> plt.Figure:
        """Plot equity curve from backtest results.
        
        Args:
            filename: Optional filename to save the plot
            
        Returns:
            Matplotlib figure object
        """
        if not self.portfolio_value_history:
            logger.error("No portfolio history to plot")
            return None
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract data
        dates = [pd.to_datetime(entry["date"]) for entry in self.portfolio_value_history]
        values = [entry["value"] for entry in self.portfolio_value_history]
        
        # Plot equity curve
        ax.plot(dates, values, label="Portfolio Value", linewidth=2)
        
        # Add reference line for initial capital
        ax.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, 
                   label=f"Initial Capital (${self.initial_capital:,.2f})")
        
        # Add trade markers
        for trade in self.trades:
            trade_date = pd.to_datetime(trade["date"])
            
            # Find portfolio value on that date
            for entry in self.portfolio_value_history:
                if entry["date"] == trade["date"]:
                    portfolio_value = entry["value"]
                    break
            else:
                continue
            
            # Plot marker
            if trade["type"] == "buy":
                ax.scatter(trade_date, portfolio_value, marker='^', color='green', s=100, alpha=0.7)
            else:  # sell
                ax.scatter(trade_date, portfolio_value, marker='v', color='red', s=100, alpha=0.7)
        
        # Add labels and title
        ax.set_title("Backtest Equity Curve", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as dollars
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.2f}"))
        
        # Rotate x-axis dates for better readability
        fig.autofmt_xdate()
        
        # Tight layout
        plt.tight_layout()
        
        # Save if filename is provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve plot to {filename}")
        
        return fig 