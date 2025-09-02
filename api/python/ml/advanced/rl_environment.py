#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning Environment Module for BensBot

This module defines the reinforcement learning trading environment
that simulates the market and provides observations, actions, and rewards
for training RL-based trading strategies.
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Union, Tuple, Optional, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """
    A gym-compatible environment for reinforcement learning trading.
    Simulates market conditions and portfolio management.
    """
    
    def __init__(self, 
                price_data: pd.DataFrame,
                features: List[str] = None,
                window_size: int = 30,
                commission: float = 0.001,
                initial_balance: float = 10000.0,
                max_position: float = 1.0,
                reward_function: str = "sharpe",
                risk_aversion: float = 1.0):
        """
        Initialize the trading environment.
        
        Args:
            price_data: DataFrame with price data (must have 'close' column)
            features: List of feature column names to include in observations
            window_size: Number of time steps to include in state representation
            commission: Trading commission as a fraction of trade value
            initial_balance: Initial portfolio balance
            max_position: Maximum allowed position size as a fraction of portfolio
            reward_function: Type of reward function ('returns', 'sharpe', 'sortino', 'calmar')
            risk_aversion: Risk aversion parameter for reward calculation
        """
        super(TradingEnvironment, self).__init__()
        
        # Store parameters
        self.price_data = price_data
        self.features = features or []
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.reward_function = reward_function
        self.risk_aversion = risk_aversion
        
        # Validate price data
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain a 'close' column")
        
        # Add required columns to features if not present
        required_cols = ['close']
        for col in required_cols:
            if col not in self.features and col in price_data.columns:
                self.features.append(col)
        
        # Validate features
        for feature in self.features:
            if feature not in price_data.columns:
                raise ValueError(f"Feature '{feature}' not found in price data")
        
        # Filter price data to include only relevant columns
        self.price_data = self.price_data[self.features].copy()
        
        # Normalize features for better learning
        self.feature_means = self.price_data.mean()
        self.feature_stds = self.price_data.std()
        self.normalized_data = (self.price_data - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Set up action and observation spaces
        # Action: continuous value between -1 (short max) and 1 (long max)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32
        )
        
        # Observation: window_size time steps of features + current position
        n_features = len(self.features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, n_features + 1), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.reset()
        
        logger.info(f"Initialized TradingEnvironment with {len(self.price_data)} time steps "
                   f"and {len(self.features)} features")
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            Initial observation
        """
        # Reset the current step
        self.current_step = self.window_size
        
        # Reset portfolio state
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.total_value = self.initial_balance
        
        # Reset history
        self.returns = []
        self.portfolio_values = [self.initial_balance]
        self.positions = [0.0]
        self.trades = []
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment based on the action.
        
        Args:
            action: The action to take (-1 to 1)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Ensure we haven't exceeded data bounds
        if self.current_step >= len(self.price_data) - 1:
            return self._get_observation(), 0, True, {'reason': 'data_end'}
        
        # Get the current price
        current_price = self.price_data.iloc[self.current_step]['close']
        
        # Convert action to target position (-max_position to max_position)
        target_position = float(action[0]) * self.max_position
        
        # Calculate trade amount
        trade_amount = target_position - self.position
        
        # Calculate trade cost
        trade_value = abs(trade_amount) * current_price
        trade_cost = trade_value * self.commission
        
        # Execute trade if possible
        if abs(trade_amount) > 1e-5:  # Ignore very small trades
            # Check if we have enough balance for the trade
            if trade_amount > 0 and self.balance < trade_value + trade_cost:
                # Not enough cash to buy, limit to available cash
                trade_amount = (self.balance - trade_cost) / current_price
                if trade_amount <= 0:
                    trade_amount = 0
            
            # Update position and balance
            old_position = self.position
            self.position += trade_amount
            self.position_value = self.position * current_price
            self.balance -= (trade_amount * current_price + trade_cost)
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'trade_amount': trade_amount,
                'old_position': old_position,
                'new_position': self.position,
                'cost': trade_cost
            })
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate portfolio value
        new_price = self.price_data.iloc[self.current_step]['close']
        self.position_value = self.position * new_price
        old_total_value = self.total_value
        self.total_value = self.balance + self.position_value
        
        # Calculate return
        step_return = (self.total_value / old_total_value) - 1
        self.returns.append(step_return)
        self.portfolio_values.append(self.total_value)
        self.positions.append(self.position)
        
        # Calculate reward based on selected reward function
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= len(self.price_data) - 1
        
        # Prepare info dictionary
        info = {
            'step': self.current_step,
            'price': new_price,
            'position': self.position,
            'balance': self.balance,
            'position_value': self.position_value,
            'total_value': self.total_value,
            'return': step_return,
            'portfolio_value': self.total_value / self.initial_balance
        }
        
        # Return step results
        return self._get_observation(), reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Returns:
            Observation array
        """
        # Get window of normalized data
        start = self.current_step - self.window_size
        end = self.current_step
        
        if start < 0:
            # Pad with zeros if we don't have enough history
            obs = np.zeros((self.window_size, len(self.features)))
            available_steps = min(self.current_step, self.window_size)
            obs[-available_steps:] = self.normalized_data.iloc[0:available_steps].values
        else:
            obs = self.normalized_data.iloc[start:end].values
        
        # Add position information
        # Normalize position to be between -1 and 1
        norm_position = np.array([self.position / self.max_position] * self.window_size).reshape(-1, 1)
        
        # Combine features and position
        full_obs = np.hstack([obs, norm_position])
        
        return full_obs
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on the selected reward function.
        
        Returns:
            Reward value
        """
        if len(self.returns) < 2:
            return 0.0
        
        if self.reward_function == 'returns':
            # Simple returns
            return self.returns[-1]
        
        elif self.reward_function == 'sharpe':
            # Sharpe ratio (approximated for the recent window)
            window_returns = self.returns[-min(len(self.returns), self.window_size):]
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns) + 1e-8  # Avoid division by zero
            return mean_return / std_return
        
        elif self.reward_function == 'sortino':
            # Sortino ratio (using only negative returns for risk)
            window_returns = self.returns[-min(len(self.returns), self.window_size):]
            mean_return = np.mean(window_returns)
            
            # Calculate downside deviation
            negative_returns = [r for r in window_returns if r < 0]
            if not negative_returns:
                downside_deviation = 1e-8  # Avoid division by zero
            else:
                downside_deviation = np.sqrt(np.mean(np.square(negative_returns))) + 1e-8
            
            return mean_return / downside_deviation
        
        elif self.reward_function == 'calmar':
            # Calmar ratio (return / maximum drawdown)
            window_values = self.portfolio_values[-min(len(self.portfolio_values), self.window_size*2):]
            
            # Calculate maximum drawdown
            peak = window_values[0]
            max_dd = 0.0
            
            for value in window_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            
            if max_dd < 1e-8:
                max_dd = 1e-8  # Avoid division by zero
            
            # Use the most recent return
            return self.returns[-1] / max_dd
        
        else:
            # Default: risk-adjusted return
            # Return minus risk aversion * variance
            window_returns = self.returns[-min(len(self.returns), self.window_size):]
            mean_return = np.mean(window_returns)
            var_return = np.var(window_returns)
            return mean_return - self.risk_aversion * var_return
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array')
        """
        if mode == 'human':
            # Print current state
            current_price = self.price_data.iloc[self.current_step]['close']
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Balance: {self.balance:.2f}")
            print(f"Position Value: {self.position_value:.2f}")
            print(f"Total Value: {self.total_value:.2f}")
            print(f"Return: {self.returns[-1] if len(self.returns) > 0 else 0.0:.4%}")
            print("-" * 50)
        elif mode == 'rgb_array':
            # Not implemented
            return None
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get a summary of strategy performance.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.returns) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'num_trades': 0
            }
        
        # Calculate total return
        total_return = (self.total_value / self.initial_balance) - 1
        
        # Calculate Sharpe ratio (annualized, assuming daily returns)
        mean_return = np.mean(self.returns)
        std_return = np.std(self.returns) + 1e-8
        sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
        
        # Calculate maximum drawdown
        peak = self.portfolio_values[0]
        max_dd = 0.0
        
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        # Calculate win rate
        num_trades = len(self.trades)
        if num_trades > 0:
            profitable_trades = sum(1 for i in range(1, len(self.trades)) 
                                if self.trades[i]['trade_amount'] * 
                                (self.trades[i]['price'] - self.trades[i-1]['price']) > 0)
            win_rate = profitable_trades / num_trades
        else:
            win_rate = 0.0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
