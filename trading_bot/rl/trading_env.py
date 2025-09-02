#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Environment Module for Reinforcement Learning

This module provides a custom OpenAI Gym environment for training
reinforcement learning agents on portfolio optimization tasks.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict


class TradingEnv(gym.Env):
    """
    A trading environment for reinforcement learning portfolio optimization.
    
    This environment follows the OpenAI Gym interface and simulates a trading
    agent that can allocate capital across multiple assets. It tracks
    portfolio performance, applies trading costs, and calculates realistic
    rewards based on risk-adjusted returns.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 df_dict: Dict[str, pd.DataFrame],
                 features_list: List[str],
                 initial_balance: float = 100000.0,
                 trading_cost: float = 0.001,
                 slippage: float = 0.0005, 
                 window_size: int = 50,
                 max_steps: int = None,
                 reward_type: str = 'sharpe',
                 reward_scale: float = 1.0,
                 allow_short: bool = False,
                 render_mode: str = None):
        """
        Initialize the trading environment.
        
        Args:
            df_dict: Dictionary mapping asset symbols to DataFrames with OHLCV and features
            features_list: List of feature columns to include in state
            initial_balance: Starting cash balance
            trading_cost: Trading cost as percentage of trade value
            slippage: Slippage as percentage of asset price
            window_size: Number of time steps to include in history for state
            max_steps: Maximum number of time steps per episode (None = full data)
            reward_type: Type of reward function ('returns', 'sharpe', 'sortino')
            reward_scale: Scale factor for rewards
            allow_short: Whether to allow short positions
            render_mode: Rendering mode
        """
        super(TradingEnv, self).__init__()
        
        self.df_dict = df_dict
        self.features_list = features_list
        self.asset_list = list(df_dict.keys())
        self.initial_balance = initial_balance
        self.trading_cost = trading_cost
        self.slippage = slippage
        self.window_size = window_size
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.allow_short = allow_short
        self.render_mode = render_mode
        
        # Determine the length of the environment based on the shortest DataFrame
        self.max_episode_steps = max_steps or min(len(df) for df in df_dict.values()) - window_size
        
        # Check that all DataFrames have the minimum required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for symbol, df in df_dict.items():
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"DataFrame for {symbol} is missing required columns: {missing_cols}")
            
            # Check that all features are in the DataFrame
            missing_features = [feat for feat in features_list if feat not in df.columns]
            if missing_features:
                raise ValueError(f"DataFrame for {symbol} is missing features: {missing_features}")
        
        # Define the action space: allocation percentage for each asset + cash
        # Values are bounded between -1 (full short) and 1 (full long) if shorts allowed
        # Or between 0 and 1 if shorts not allowed
        if allow_short:
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(len(self.asset_list),), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=0, high=1, shape=(len(self.asset_list),), dtype=np.float32
            )
        
        # Define the observation space
        # State includes:
        # 1. Features for each asset (window_size x num_features x num_assets)
        # 2. Current portfolio allocations (num_assets + 1 for cash)
        # 3. Current portfolio value
        num_features = len(features_list)
        num_assets = len(self.asset_list)
        
        # Calculate the dimension of the observation space
        feature_dim = window_size * num_features * num_assets  # Historical features
        portfolio_dim = num_assets + 1  # Current allocations (assets + cash)
        performance_dim = 3  # Portfolio value, returns, volatility
        
        obs_dim = feature_dim + portfolio_dim + performance_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state variables
        self.portfolio = None
        self.cash = None
        self.current_step = None
        self.nav_history = None
        self.returns_history = None
        self.portfolio_value = None
        self.last_trades = None
        self.history = None
        
        # Episode tracking
        self.episode_returns = None
        self.episode_volatility = None
        self.episode_sharpe = None
        self.episode_sortino = None
        self.episode_trades = None
        self.episode_costs = None
        
        # Reset the environment to initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Initial observation and info dictionary
        """
        super().reset(seed=seed)
        
        # Reset portfolio state
        self.portfolio = {symbol: 0.0 for symbol in self.asset_list}
        self.cash = self.initial_balance
        self.current_step = self.window_size
        self.nav_history = [self.initial_balance]
        self.returns_history = [0.0]
        self.portfolio_value = self.initial_balance
        self.last_trades = {symbol: 0.0 for symbol in self.asset_list}
        
        # Initialize history buffer for window state
        self.history = defaultdict(list)
        
        # Initialize episode metrics
        self.episode_returns = []
        self.episode_volatility = 0.0
        self.episode_sharpe = 0.0
        self.episode_sortino = 0.0
        self.episode_trades = 0
        self.episode_costs = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio': self.portfolio.copy(),
            'cash': self.cash,
            'step': self.current_step
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action in the environment.
        
        Args:
            action: Array of allocations for each asset
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Get current prices for all assets
        current_prices = {symbol: self.df_dict[symbol].iloc[self.current_step]['close'] 
                         for symbol in self.asset_list}
        
        # Normalize action to ensure sum is 1.0 (proper allocation vector)
        if not self.allow_short:
            # If shorts not allowed, ensure all values are positive then normalize
            action = np.maximum(action, 0.0)
            
        # Calculate the sum of absolute allocations (for normalization)
        action_sum = np.sum(np.abs(action))
        if action_sum > 0:
            action = action / action_sum
        else:
            # If all actions are zero, allocate everything to cash
            action = np.zeros_like(action)
        
        # Store the previous portfolio value for reward calculation
        prev_portfolio_value = self.portfolio_value
        
        # Calculate current portfolio value
        portfolio_value = self.cash
        for symbol, shares in self.portfolio.items():
            portfolio_value += shares * current_prices[symbol]
        
        # Track last portfolio value
        self.portfolio_value = portfolio_value
        
        # Execute trades based on target allocations
        self._execute_trades(action, current_prices)
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate the new portfolio value after trading
        new_portfolio_value = self.cash
        for symbol, shares in self.portfolio.items():
            new_portfolio_value += shares * current_prices[symbol]
        
        # Calculate return and update history
        returns = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.nav_history.append(new_portfolio_value)
        self.returns_history.append(returns)
        self.episode_returns.append(returns)
        
        # Update episode metrics
        if len(self.episode_returns) > 1:
            self.episode_volatility = np.std(self.episode_returns)
            if self.episode_volatility > 0:
                self.episode_sharpe = np.mean(self.episode_returns) / self.episode_volatility
                # Calculate Sortino ratio (downside deviation only)
                negative_returns = [r for r in self.episode_returns if r < 0]
                if negative_returns:
                    downside_deviation = np.std(negative_returns)
                    if downside_deviation > 0:
                        self.episode_sortino = np.mean(self.episode_returns) / downside_deviation
        
        # Calculate reward based on the specified type
        reward = self._calculate_reward(returns)
        
        # Check if the episode is complete
        done = self.current_step >= self.window_size + self.max_episode_steps - 1
        
        # Prepare the next observation
        observation = self._get_observation()
        
        # Prepare info dictionary
        info = {
            'portfolio_value': new_portfolio_value,
            'returns': returns,
            'portfolio': self.portfolio.copy(),
            'cash': self.cash,
            'sharpe_ratio': self.episode_sharpe,
            'sortino_ratio': self.episode_sortino,
            'volatility': self.episode_volatility,
            'trades': self.episode_trades,
            'trading_costs': self.episode_costs,
            'step': self.current_step
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector from current state.
        
        Returns:
            Numpy array with the observation
        """
        # 1. Get historical features for each asset
        historical_features = []
        
        for symbol in self.asset_list:
            # Get window of data for this asset
            start_idx = self.current_step - self.window_size
            end_idx = self.current_step
            asset_history = self.df_dict[symbol].iloc[start_idx:end_idx]
            
            # Extract the requested features
            asset_features = asset_history[self.features_list].values
            
            # Flatten and append
            historical_features.append(asset_features.flatten())
        
        # Concatenate all historical features
        all_features = np.concatenate(historical_features)
        
        # 2. Current portfolio allocations
        current_prices = {symbol: self.df_dict[symbol].iloc[self.current_step]['close'] 
                         for symbol in self.asset_list}
        
        # Calculate total portfolio value
        total_value = self.cash
        for symbol, shares in self.portfolio.items():
            total_value += shares * current_prices[symbol]
            
        # If the portfolio has no value, set allocations to zero
        if total_value <= 0:
            portfolio_allocations = np.zeros(len(self.asset_list) + 1)
        else:
            # Calculate allocation percentages
            portfolio_allocations = []
            for symbol in self.asset_list:
                value = self.portfolio[symbol] * current_prices[symbol]
                allocation = value / total_value
                portfolio_allocations.append(allocation)
            
            # Add cash allocation
            cash_allocation = self.cash / total_value
            portfolio_allocations.append(cash_allocation)
            
        # 3. Portfolio performance metrics
        # - Current portfolio value (normalized)
        # - Recent returns
        # - Recent volatility
        
        normalized_value = total_value / self.initial_balance
        
        # Calculate recent returns if we have enough history
        if len(self.returns_history) > 1:
            recent_returns = np.mean(self.returns_history[-min(10, len(self.returns_history)):])
            recent_volatility = np.std(self.returns_history[-min(20, len(self.returns_history)):])
        else:
            recent_returns = 0.0
            recent_volatility = 0.0
        
        performance_metrics = [normalized_value, recent_returns, recent_volatility]
        
        # Combine all parts of the observation
        observation = np.concatenate([
            all_features,
            portfolio_allocations,
            performance_metrics
        ]).astype(np.float32)
        
        return observation
    
    def _execute_trades(self, target_allocations: np.ndarray, current_prices: Dict[str, float]) -> None:
        """
        Execute trades based on target allocations.
        
        Args:
            target_allocations: Target allocation vector for assets (excluding cash)
            current_prices: Current prices for all assets
        """
        # Calculate total portfolio value
        total_value = self.cash
        for symbol, shares in self.portfolio.items():
            total_value += shares * current_prices[symbol]
        
        # Track total trading costs
        total_trading_cost = 0.0
        
        # Reset last trades
        self.last_trades = {symbol: 0.0 for symbol in self.asset_list}
        
        # Calculate target values for each asset
        target_values = {}
        for i, symbol in enumerate(self.asset_list):
            # Get target allocation for this asset
            target_allocation = target_allocations[i]
            
            # Calculate target value
            target_values[symbol] = target_allocation * total_value
        
        # Execute trades for each asset
        for symbol in self.asset_list:
            price = current_prices[symbol]
            current_value = self.portfolio[symbol] * price
            target_value = target_values[symbol]
            
            # Calculate trade value (positive = buy, negative = sell)
            trade_value = target_value - current_value
            
            if abs(trade_value) > 0.01:  # Only trade if the difference is significant
                # Calculate trading cost
                trading_cost = abs(trade_value) * self.trading_cost
                
                # Apply slippage - adjust price worse for the trader
                trade_price = price * (1 + np.sign(trade_value) * self.slippage)
                
                # Calculate shares to trade
                shares_to_trade = trade_value / trade_price
                
                # Update portfolio
                self.portfolio[symbol] += shares_to_trade
                
                # Update cash (subtract both the trade value and the trading cost)
                self.cash -= (trade_value + trading_cost)
                
                # Update trade tracking
                self.last_trades[symbol] = shares_to_trade
                self.episode_trades += 1
                total_trading_cost += trading_cost
        
        # Update total trading costs for this episode
        self.episode_costs += total_trading_cost
    
    def _calculate_reward(self, returns: float) -> float:
        """
        Calculate the reward based on the specified reward type.
        
        Args:
            returns: Current portfolio returns
            
        Returns:
            Calculated reward value
        """
        if self.reward_type == 'returns':
            # Simple returns-based reward
            reward = returns
            
        elif self.reward_type == 'sharpe':
            # Sharpe ratio based reward
            if len(self.episode_returns) > 1:
                vol = max(self.episode_volatility, 1e-8)  # Avoid division by zero
                sharpe = np.mean(self.episode_returns) / vol
                reward = sharpe
            else:
                reward = 0.0
                
        elif self.reward_type == 'sortino':
            # Sortino ratio based reward
            if len(self.episode_returns) > 1:
                negative_returns = [r for r in self.episode_returns if r < 0]
                if negative_returns:
                    downside_deviation = max(np.std(negative_returns), 1e-8)
                    sortino = np.mean(self.episode_returns) / downside_deviation
                    reward = sortino
                else:
                    # No negative returns, high reward
                    reward = np.mean(self.episode_returns) * 10
            else:
                reward = 0.0
        else:
            # Default to returns
            reward = returns
        
        # Apply reward scaling
        reward = reward * self.reward_scale
        
        return reward
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if self.render_mode == 'human':
            # Calculate portfolio value
            portfolio_value = self.cash
            current_prices = {symbol: self.df_dict[symbol].iloc[self.current_step]['close'] 
                            for symbol in self.asset_list}
            
            for symbol, shares in self.portfolio.items():
                portfolio_value += shares * current_prices[symbol]
            
            # Print portfolio state
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print("Allocations:")
            
            for symbol, shares in self.portfolio.items():
                asset_value = shares * current_prices[symbol]
                allocation = asset_value / portfolio_value if portfolio_value > 0 else 0
                print(f"  {symbol}: {shares:.4f} shares, ${asset_value:.2f} ({allocation:.2%})")
            
            print(f"Recent Return: {self.returns_history[-1]:.2%}")
            print(f"Sharpe Ratio: {self.episode_sharpe:.4f}")
            print(f"Trades: {self.episode_trades}")
            print(f"Trading Costs: ${self.episode_costs:.2f}")
            print("-" * 50)
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get the portfolio history as a DataFrame for analysis.
        
        Returns:
            DataFrame with portfolio history
        """
        if hasattr(self, 'nav_history') and len(self.nav_history) > 0:
            # Create a portfolio history DataFrame
            # Limit history to actual episode steps
            history_len = min(len(self.nav_history), self.current_step - self.window_size + 1)
            
            # Get the relevant dates
            start_idx = self.window_size
            end_idx = start_idx + history_len
            
            # Use dates from the first asset as reference
            first_asset = next(iter(self.df_dict))
            dates = self.df_dict[first_asset].index[start_idx:end_idx]
            
            # Calculate returns
            nav_series = pd.Series(self.nav_history[:history_len], index=dates)
            returns = nav_series.pct_change().fillna(0)
            
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod() - 1
            
            # Calculate drawdown
            rolling_max = nav_series.cummax()
            drawdown = (nav_series - rolling_max) / rolling_max
            
            # Create result DataFrame
            result = pd.DataFrame({
                'portfolio_value': nav_series,
                'returns': returns,
                'cumulative_returns': cum_returns,
                'drawdown': drawdown
            })
            
            return result
        
        return pd.DataFrame()
    
    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get summary statistics for the current episode.
        
        Returns:
            Dictionary with episode statistics
        """
        stats = {
            'final_value': self.portfolio_value,
            'total_return': (self.portfolio_value / self.initial_balance) - 1,
            'sharpe_ratio': self.episode_sharpe,
            'sortino_ratio': self.episode_sortino,
            'volatility': self.episode_volatility,
            'max_drawdown': self.get_portfolio_history()['drawdown'].min() if not self.get_portfolio_history().empty else 0,
            'total_trades': self.episode_trades,
            'total_costs': self.episode_costs,
            'average_return': np.mean(self.episode_returns) if self.episode_returns else 0,
        }
        
        return stats 