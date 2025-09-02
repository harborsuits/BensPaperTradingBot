#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stock Base Strategy Module

This module defines the base class for all stock trading strategies in the system.
It provides core functionality and interfaces that all stock-based strategies should implement,
ensuring consistent behavior and expected interfaces across the platform.

The base class handles common operations like universe filtering, data validation,
position sizing and risk management, allowing derived strategies to focus on their
specific trading logic and signals.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

from trading_bot.strategies.strategy_template import StrategyOptimizable, Signal, SignalType, TimeFrame, MarketRegime
from trading_bot.market.universe import Universe
from trading_bot.market.market_data import MarketData
from trading_bot.orders.order import Order
from trading_bot.risk.position_sizer import PositionSizer

logger = logging.getLogger(__name__)

class StockBaseStrategy(StrategyOptimizable):
    """
    Base Strategy for Stock Trading
    
    This abstract base class defines the foundational structure for all stock trading strategies.
    It implements common functionality and defines interfaces that specific stock strategies
    should implement to ensure platform consistency.
    
    Key responsibilities:
    - Defining universe filtering methods for stock selection
    - Providing core data validation and preprocessing
    - Implementing standard risk management approaches
    - Defining interfaces for entry/exit decisions
    - Managing order creation and execution
    
    Derived strategy classes should override the abstract methods to implement 
    their specific trading logic while adhering to the established framework.
    
    Attributes:
        params (Dict[str, Any]): Strategy parameters dictionary
        name (str): Strategy name
        version (str): Strategy version
    """
    
    DEFAULT_PARAMS = {
        # Strategy identification
        'strategy_name': 'stock_base',
        'strategy_version': '1.0.0',
        
        # Universe selection criteria
        'min_stock_price': 5.0,          # Minimum stock price to consider ($5)
        'max_stock_price': 1000.0,       # Maximum stock price to consider
        'min_market_cap': 300000000,     # Minimum market cap ($300M)
        'min_avg_volume': 500000,        # Minimum average daily volume (500k shares)
        
        # Data requirements
        'min_historical_days': 252,      # Minimum 1 year of trading data
        
        # Risk management parameters
        'max_position_size_percent': 0.05,  # Max 5% of portfolio per position
        'max_sector_exposure': 0.25,        # Max 25% exposure per sector
        'max_risk_per_trade': 0.01,         # Risk 1% of portfolio per trade
        
        # Position management
        'use_stop_loss': True,           # Whether to use stop losses
        'stop_loss_pct': 0.07,           # 7% stop loss
        'use_trailing_stop': False,      # Whether to use trailing stops
        'trailing_stop_pct': 0.05,       # 5% trailing stop
        'use_take_profit': True,         # Whether to use take profit targets
        'take_profit_pct': 0.15,         # 15% take profit target
    }
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the StockBaseStrategy with provided parameters.
        
        Parameters:
            params (Dict[str, Any], optional): Strategy parameters to override defaults
        """
        super().__init__(params)
        
    def define_universe(self, market_data: MarketData) -> Universe:
        """
        Define the universe of stocks to trade based on criteria.
        
        This base implementation filters stocks based on price range, market cap,
        and liquidity criteria defined in the strategy parameters.
        
        Parameters:
            market_data (MarketData): Market data provider containing price and reference data
            
        Returns:
            Universe: A Universe object containing the filtered symbols
            
        Notes:
            Derived strategies may override this method to add additional filtering criteria
            or implement strategy-specific universe selection logic.
        """
        universe = Universe()
        
        # Filter by price range
        price_df = market_data.get_latest_prices()
        if price_df is not None and not price_df.empty:
            filtered_symbols = price_df[(price_df['close'] >= self.params['min_stock_price']) & 
                                       (price_df['close'] <= self.params['max_stock_price'])].index.tolist()
            universe.add_symbols(filtered_symbols)
        
        # Filter by market cap if data available
        if hasattr(market_data, 'get_market_caps'):
            mkt_cap_df = market_data.get_market_caps()
            if mkt_cap_df is not None and not mkt_cap_df.empty:
                symbols_to_remove = []
                for symbol in universe.get_symbols():
                    if symbol in mkt_cap_df.index:
                        if mkt_cap_df.loc[symbol, 'market_cap'] < self.params['min_market_cap']:
                            symbols_to_remove.append(symbol)
                
                for symbol in symbols_to_remove:
                    universe.remove_symbol(symbol)
        
        # Filter by volume criteria
        if hasattr(market_data, 'get_average_volumes'):
            vol_df = market_data.get_average_volumes()
            if vol_df is not None and not vol_df.empty:
                symbols_to_remove = []
                for symbol in universe.get_symbols():
                    if symbol in vol_df.index:
                        if vol_df.loc[symbol, 'avg_volume'] < self.params['min_avg_volume']:
                            symbols_to_remove.append(symbol)
                
                for symbol in symbols_to_remove:
                    universe.remove_symbol(symbol)
        
        logger.info(f"Base stock universe contains {len(universe.get_symbols())} symbols after filtering")
        return universe
    
    def check_selection_criteria(self, symbol: str, market_data: MarketData) -> bool:
        """
        Check if a symbol meets the selection criteria for the strategy.
        
        This base implementation checks for sufficient historical data and any 
        other fundamental criteria that all stock strategies should verify.
        
        Parameters:
            symbol (str): Symbol to check
            market_data (MarketData): Market data provider
            
        Returns:
            bool: True if symbol meets all criteria, False otherwise
            
        Notes:
            Derived strategies should override this method to add strategy-specific
            selection criteria like technical indicators or fundamental requirements.
        """
        # Check if we have enough historical data
        if not market_data.has_min_history(symbol, self.params['min_historical_days']):
            logger.debug(f"{symbol} doesn't have enough historical data")
            return False
        
        # Additional base filtering criteria can be added here
        
        # All base criteria met
        return True
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                               position_sizer: PositionSizer) -> int:
        """
        Calculate the position size based on risk parameters.
        
        Determines the appropriate number of shares to trade based on:
        1. Maximum position size limits
        2. Per-trade risk limits
        3. Available capital
        
        Parameters:
            symbol (str): Trading symbol
            current_price (float): Current price of the asset
            position_sizer (PositionSizer): Position sizing service with portfolio information
            
        Returns:
            int: Number of shares to trade (0 if position should not be taken)
            
        Notes:
            This implementation focuses on standard position sizing techniques.
            Derived strategies may override this for custom position sizing logic.
        """
        if current_price <= 0:
            return 0
        
        # Get portfolio value
        portfolio_value = position_sizer.get_portfolio_value()
        
        # Calculate position size based on max percentage of portfolio
        max_position_value = portfolio_value * self.params['max_position_size_percent']
        position_size = int(max_position_value / current_price)
        
        # Adjust for risk per trade if stop loss is used
        if self.params['use_stop_loss']:
            stop_distance_pct = self.params['stop_loss_pct']
            risk_amount = portfolio_value * self.params['max_risk_per_trade']
            
            risk_based_size = int(risk_amount / (current_price * stop_distance_pct))
            position_size = min(position_size, risk_based_size)
        
        return position_size
    
    def prepare_entry_orders(self, symbol: str, quantity: int, entry_price: float = None) -> List[Order]:
        """
        Prepare orders for entering a position.
        
        Creates the necessary orders to enter a position based on the provided parameters.
        
        Parameters:
            symbol (str): Trading symbol
            quantity (int): Number of shares to trade
            entry_price (float, optional): Limit price for entry, uses market order if None
            
        Returns:
            List[Order]: List of orders to execute for position entry
            
        Notes:
            This implementation creates a basic entry order.
            Derived strategies should override this for strategy-specific order types
            or complex entry logic.
        """
        raise NotImplementedError("Derived stock strategies must implement prepare_entry_orders")
    
    def check_exit_conditions(self, position: Dict[str, Any], market_data: MarketData) -> bool:
        """
        Check if exit conditions are met for an existing position.
        
        This base implementation checks common exit conditions like:
        - Stop loss triggers
        - Take profit targets
        - Trailing stop activation
        
        Parameters:
            position (Dict[str, Any]): Current position information
            market_data (MarketData): Market data provider
            
        Returns:
            bool: True if any exit condition is met, False otherwise
            
        Notes:
            Derived strategies should override this method to add strategy-specific
            exit criteria while maintaining the base exit checks.
        """
        if not position:
            return False
            
        symbol = position.get('symbol')
        entry_price = position.get('entry_price', 0)
        current_price = market_data.get_latest_price(symbol)
        
        if not symbol or entry_price <= 0 or not current_price:
            return False
            
        # Check stop loss
        if self.params['use_stop_loss']:
            if position['direction'] == 'long':
                stop_price = entry_price * (1 - self.params['stop_loss_pct'])
                if current_price <= stop_price:
                    logger.info(f"Exiting {symbol}: Stop loss triggered at {current_price:.2f}")
                    return True
            else:  # Short position
                stop_price = entry_price * (1 + self.params['stop_loss_pct'])
                if current_price >= stop_price:
                    logger.info(f"Exiting {symbol}: Stop loss triggered at {current_price:.2f}")
                    return True
        
        # Check take profit
        if self.params['use_take_profit']:
            if position['direction'] == 'long':
                target_price = entry_price * (1 + self.params['take_profit_pct'])
                if current_price >= target_price:
                    logger.info(f"Exiting {symbol}: Take profit target reached at {current_price:.2f}")
                    return True
            else:  # Short position
                target_price = entry_price * (1 - self.params['take_profit_pct'])
                if current_price <= target_price:
                    logger.info(f"Exiting {symbol}: Take profit target reached at {current_price:.2f}")
                    return True
        
        # Check trailing stop if enabled
        if self.params['use_trailing_stop'] and 'highest_price' in position:
            trailing_stop_pct = self.params['trailing_stop_pct']
            
            if position['direction'] == 'long':
                highest_price = position.get('highest_price', entry_price)
                stop_price = highest_price * (1 - trailing_stop_pct)
                
                if current_price <= stop_price:
                    logger.info(f"Exiting {symbol}: Trailing stop triggered at {current_price:.2f}")
                    return True
            else:  # Short position
                lowest_price = position.get('lowest_price', entry_price)
                stop_price = lowest_price * (1 + trailing_stop_pct)
                
                if current_price >= stop_price:
                    logger.info(f"Exiting {symbol}: Trailing stop triggered at {current_price:.2f}")
                    return True
        
        # No exit condition met
        return False
    
    def prepare_exit_orders(self, position: Dict[str, Any]) -> List[Order]:
        """
        Prepare orders to close an existing position.
        
        Creates the necessary orders to exit a position based on current state.
        
        Parameters:
            position (Dict[str, Any]): Current position information
            
        Returns:
            List[Order]: List of orders to execute for position exit
            
        Notes:
            This method should be implemented by derived strategies to handle
            specific exit order requirements.
        """
        raise NotImplementedError("Derived stock strategies must implement prepare_exit_orders")
    
    def get_optimization_params(self) -> Dict[str, Any]:
        """
        Define parameters that can be optimized and their ranges.
        
        Specifies which parameters should be considered during strategy optimization
        and their valid ranges for testing.
        
        Returns:
            Dict[str, Any]: Dictionary of optimization parameter specifications
                Each entry contains parameter type, min/max values, and step size
                
        Notes:
            This base implementation includes common parameters that most stock
            strategies would optimize. Derived strategies should override to add
            strategy-specific parameters.
        """
        return {
            'min_stock_price': {'type': 'float', 'min': 1.0, 'max': 10.0, 'step': 1.0},
            'min_avg_volume': {'type': 'int', 'min': 100000, 'max': 1000000, 'step': 100000},
            'stop_loss_pct': {'type': 'float', 'min': 0.03, 'max': 0.15, 'step': 0.01},
            'take_profit_pct': {'type': 'float', 'min': 0.05, 'max': 0.30, 'step': 0.05},
            'max_position_size_percent': {'type': 'float', 'min': 0.01, 'max': 0.10, 'step': 0.01},
        }
        
    def evaluate_performance(self, backtest_results: Dict[str, Any]) -> float:
        """
        Evaluate strategy performance for optimization purposes.
        
        Calculates a performance score based on backtest results that can be used
        to compare different parameter combinations during optimization.
        
        Parameters:
            backtest_results (Dict[str, Any]): Results from strategy backtest
            
        Returns:
            float: Performance score (higher is better)
            
        Notes:
            This base implementation uses a combination of Sharpe ratio, drawdown,
            and win rate to evaluate performance. Derived strategies may use different
            or additional metrics based on their specific goals.
        """
        if 'sharpe_ratio' not in backtest_results or 'max_drawdown' not in backtest_results:
            return 0.0
            
        sharpe = backtest_results.get('sharpe_ratio', 0)
        max_dd = abs(backtest_results.get('max_drawdown', 0))
        win_rate = backtest_results.get('win_rate', 0)
        
        # Penalize high drawdowns
        if max_dd > 0.25:  # 25% drawdown
            sharpe = sharpe * (1 - (max_dd - 0.25))
            
        # Reward high win rates
        if win_rate > 0.5:
            sharpe = sharpe * (1 + (win_rate - 0.5))
            
        return max(0, sharpe)
    
    def filter_universe(self, universe: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter the universe based on stock-specific criteria.
        
        Args:
            universe: Dictionary mapping symbols to DataFrames with stock data
            
        Returns:
            Filtered universe
        """
        filtered_universe = {}
        
        for symbol, data in universe.items():
            # Skip if no data
            if data.empty:
                continue
            
            # Get latest data
            latest = data.iloc[-1]
            
            # Apply price filters
            if self.params['min_stock_price'] > 0 and latest['close'] < self.params['min_stock_price']:
                continue
                
            if self.params['max_stock_price'] > 0 and latest['close'] > self.params['max_stock_price']:
                continue
            
            # Apply volume filter
            if 'volume' in data.columns and self.params['min_avg_volume'] > 0:
                avg_volume = data['volume'].mean()
                if avg_volume < self.params['min_avg_volume']:
                    continue
            
            # Symbol passed all filters
            filtered_universe[symbol] = data
        
        logger.info(f"Filtered universe from {len(universe)} to {len(filtered_universe)} symbols")
        return filtered_universe
    
    def calculate_stock_indicators(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate stock-specific technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of calculated indicators
        """
        indicators = {}
        
        # Calculate Moving Averages
        for period in [20, 50, 200]:
            ma_key = f'ma_{period}'
            indicators[ma_key] = pd.DataFrame({
                ma_key: data['close'].rolling(window=period).mean()
            })
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        indicators['rsi'] = pd.DataFrame({'rsi': rsi})
        
        # Calculate MACD
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        indicators['macd'] = pd.DataFrame({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_hist': macd_hist
        })
        
        # Calculate Bollinger Bands
        ma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)
        
        indicators['bbands'] = pd.DataFrame({
            'middle_band': ma20,
            'upper_band': upper_band,
            'lower_band': lower_band
        })
        
        # Calculate Volume Profile if enabled
        if self.params['use_volume_profile'] and 'volume' in data.columns:
            # Simple volume distribution by price
            price_buckets = pd.cut(data['close'], bins=10)
            volume_profile = data.groupby(price_buckets)['volume'].sum()
            
            # Convert to DataFrame
            indicators['volume_profile'] = pd.DataFrame({
                'volume_by_price': volume_profile
            })
        
        return indicators
    
    def check_earnings_announcement(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Check if there's an upcoming earnings announcement.
        
        Args:
            symbol: Stock symbol
            data: Stock data including fundamental info
            
        Returns:
            True if earnings are upcoming within parameter threshold days
        """
        # Skip if fundamental data is not enabled
        if not self.params['use_fundamentals']:
            return False
            
        # Check if earnings data is available
        if 'earnings_date' not in data:
            return False
            
        # Get next earnings date
        next_earnings = data['earnings_date']
        
        # If it's not a datetime, try to convert
        if not isinstance(next_earnings, datetime):
            try:
                next_earnings = pd.to_datetime(next_earnings)
            except:
                return False
        
        # Check if earnings are upcoming within threshold
        days_to_earnings = (next_earnings - datetime.now()).days
        
        # Default threshold is 5 days
        earnings_threshold = self.params.get('earnings_announcement_threshold', 5)
        
        return 0 <= days_to_earnings <= earnings_threshold
    
    def adjust_for_market_regime(self, signals: Dict[str, Signal], 
                                market_regime: MarketRegime) -> Dict[str, Signal]:
        """
        Adjust signals based on overall market regime.
        
        Args:
            signals: Dictionary of generated signals
            market_regime: Current market regime
            
        Returns:
            Adjusted signals
        """
        adjusted_signals = signals.copy()
        
        # In bear market, reduce position sizes and increase stop distance
        if market_regime == MarketRegime.BEAR_TREND:
            for symbol, signal in adjusted_signals.items():
                # Reduce confidence
                signal.confidence = signal.confidence * 0.7
                
                # Adjust stop loss to be wider
                if signal.stop_loss is not None and signal.price is not None:
                    # For buy signals
                    if signal.signal_type == SignalType.BUY:
                        stop_distance = signal.price - signal.stop_loss
                        # Increase stop distance by 50%
                        adjusted_stop = signal.price - (stop_distance * 1.5)
                        signal.stop_loss = adjusted_stop
        
        # In high volatility regime, tighten profit targets
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            for symbol, signal in adjusted_signals.items():
                if signal.take_profit is not None and signal.price is not None:
                    # For buy signals
                    if signal.signal_type == SignalType.BUY:
                        profit_distance = signal.take_profit - signal.price
                        # Reduce profit target by 30%
                        adjusted_target = signal.price + (profit_distance * 0.7)
                        signal.take_profit = adjusted_target
        
        return adjusted_signals
    
    def check_sector_rotation(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Analyze sector rotation to identify strong/weak sectors.
        
        Args:
            data: Dictionary mapping symbols to DataFrames with market data
            
        Returns:
            Dictionary mapping sectors to strength scores (higher is stronger)
        """
        # Skip if sector data not available
        if not data or 'sector' not in next(iter(data.values())).columns:
            return {}
            
        sector_returns = {}
        
        # Calculate returns for each sector
        for symbol, df in data.items():
            if df.empty or 'sector' not in df.columns:
                continue
                
            sector = df['sector'].iloc[-1]
            
            # Calculate 1-month return
            if len(df) > 20:
                returns = df['close'].iloc[-1] / df['close'].iloc[-21] - 1
                
                if sector not in sector_returns:
                    sector_returns[sector] = []
                    
                sector_returns[sector].append(returns)
        
        # Average returns by sector
        sector_strength = {}
        for sector, returns in sector_returns.items():
            if returns:
                sector_strength[sector] = np.mean(returns)
        
        return sector_strength 