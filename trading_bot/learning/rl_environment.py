#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLTradingEnv - A custom Gym-style environment for training RL agents on strategy selection.
"""

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from trading_bot.backtesting.data_manager import DataManager
from trading_bot.backtesting.pattern_learner import PatternLearner

logger = logging.getLogger(__name__)

class RLTradingEnv(gym.Env):
    """
    A custom Gym environment for RL agents to learn strategy allocation and parameter tuning.
    
    This environment simulates a trading system where the agent must decide:
    1. Which strategies to allocate capital to
    2. How much to allocate to each strategy
    3. (Optional) Tuning of strategy parameters
    
    The state space includes market data, strategy performance metrics,
    and market regime indicators. The reward is based on risk-adjusted returns.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self, 
        strategies: List[str],
        data_path: str = "data/backtest_history.json",
        window_size: int = 30,
        episode_length: int = 252,  # Roughly 1 trading year
        reward_type: str = "sharpe",
        use_pattern_insights: bool = True,
        render_mode: Optional[str] = None,
        tunable_params: bool = False
    ):
        """
        Initialize the RL Trading Environment.
        
        Args:
            strategies: List of strategy names to allocate between
            data_path: Path to backtest history data
            window_size: Number of days of history to include in state
            episode_length: Number of steps per episode
            reward_type: Type of reward function ('sharpe', 'sortino', 'pnl', 'calmar')
            use_pattern_insights: Whether to include PatternLearner insights in state
            render_mode: How to render the environment ('human', 'rgb_array', or None)
            tunable_params: Whether to allow the agent to tune strategy parameters
        """
        super(RLTradingEnv, self).__init__()
        
        self.strategies = strategies
        self.data_path = data_path
        self.window_size = window_size
        self.episode_length = episode_length
        self.reward_type = reward_type
        self.use_pattern_insights = use_pattern_insights
        self.render_mode = render_mode
        self.tunable_params = tunable_params
        
        # Load data
        self.data_manager = DataManager(save_path=data_path)
        self.data = self.data_manager.load()
        
        # Convert to DataFrames
        self._prepare_data()
        
        # Create pattern learner if enabled
        if use_pattern_insights:
            self.pattern_learner = PatternLearner(data_manager=self.data_manager)
            self.insights = self.pattern_learner.analyze(save_results=False)
        else:
            self.pattern_learner = None
            self.insights = None
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Initialize state
        self.current_step = 0
        self.current_portfolio_value = 100000.0  # Initial portfolio value
        self.portfolio_history = []
        self.action_history = []
        self.current_allocations = np.ones(len(strategies)) / len(strategies)  # Start with equal allocation
        
        logger.info(f"Initialized RLTradingEnv with {len(strategies)} strategies")
    
    def _prepare_data(self):
        """Prepare data for RL environment"""
        # Get DataFrames from data manager
        self.portfolio_df = self.data_manager.get_data_as_dataframe('portfolio_snapshot')
        self.signal_df = self.data_manager.get_data_as_dataframe('signal')
        self.trade_df = self.data_manager.get_data_as_dataframe('trade')
        
        # Convert timestamps to datetime and sort
        for df in [self.portfolio_df, self.signal_df, self.trade_df]:
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.sort_values('timestamp', inplace=True)
        
        # Create a timeline of all dates in the data
        if not self.portfolio_df.empty:
            all_dates = sorted(self.portfolio_df['timestamp'].unique())
            self.timeline = pd.DataFrame({'timestamp': all_dates})
            self.total_steps = len(all_dates)
            logger.info(f"Timeline created with {self.total_steps} dates")
        else:
            logger.warning("No portfolio data available for timeline creation")
            # Create a default timeline
            start_date = datetime.now() - timedelta(days=365)
            dates = [start_date + timedelta(days=i) for i in range(365)]
            self.timeline = pd.DataFrame({'timestamp': dates})
            self.total_steps = len(dates)
    
    def _define_spaces(self):
        """Define action and observation spaces"""
        # Action space: Strategy allocations (continuous values between 0 and 1)
        # Sum of allocations will be normalized to 1.0 during step()
        if self.tunable_params:
            # Add space for strategy parameters (simplified to 1 parameter per strategy)
            # [strat1_alloc, strat2_alloc, ..., stratN_alloc, strat1_param, strat2_param, ..., stratN_param]
            self.action_space = spaces.Box(
                low=np.zeros(len(self.strategies) * 2),
                high=np.ones(len(self.strategies) * 2),
                dtype=np.float32
            )
        else:
            # Just strategy allocations
            self.action_space = spaces.Box(
                low=np.zeros(len(self.strategies)),
                high=np.ones(len(self.strategies)),
                dtype=np.float32
            )
        
        # Observation space: A combination of features including
        # 1. Market data metrics (volatility, trend, etc.)
        # 2. Strategy performance metrics
        # 3. Portfolio state (returns, drawdown)
        # 4. Pattern insights if enabled
        
        # Calculate the size of the observation space
        market_features = 10  # Price, volume, volatility, trend, etc.
        strategy_features = 5 * len(self.strategies)  # returns, win rate, sharpe, etc. per strategy
        portfolio_features = 5  # value, returns, drawdown, cash, etc.
        
        if self.use_pattern_insights:
            # Additional features from pattern insights
            pattern_features = 10  # recommended strategies, regime indicators, etc.
        else:
            pattern_features = 0
        
        obs_size = market_features + strategy_features + portfolio_features + pattern_features
        
        # Create observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space size: {obs_size}")
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Choose a random starting point in the data
        # Ensure we have enough history and future data for a full episode
        min_start = self.window_size
        max_start = max(min_start + 1, self.total_steps - self.episode_length - 1)
        
        if max_start <= min_start:
            logger.warning(f"Not enough data for episode. Using default starting point.")
            self.current_step = min_start
        else:
            self.current_step = self.np_random.integers(min_start, max_start)
        
        # Reset portfolio and allocations
        self.current_portfolio_value = 100000.0
        self.portfolio_history = [self.current_portfolio_value]
        self.action_history = []
        self.current_allocations = np.ones(len(self.strategies)) / len(self.strategies)
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'step': self.current_step,
            'date': self.timeline.iloc[self.current_step]['timestamp'].strftime('%Y-%m-%d'),
            'portfolio_value': self.current_portfolio_value,
            'allocations': dict(zip(self.strategies, self.current_allocations))
        }
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment based on the agent's action.
        
        Args:
            action: Strategy allocation weights
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Process action
        allocations = self._process_action(action)
        
        # Store action
        self.action_history.append(allocations)
        self.current_allocations = allocations
        
        # Advance to the next step
        self.current_step += 1
        
        # Calculate portfolio change based on allocations and strategy performance
        portfolio_return = self._calculate_portfolio_return(allocations)
        
        # Update portfolio value
        self.current_portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.current_portfolio_value)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return)
        
        # Check if episode is done
        terminated = (self.current_step >= min(self.total_steps - 1, 
                                          self.current_step + self.episode_length - 1))
        truncated = False
        
        # Compile info dictionary
        info = {
            'step': self.current_step,
            'date': self.timeline.iloc[self.current_step]['timestamp'].strftime('%Y-%m-%d'),
            'portfolio_value': self.current_portfolio_value,
            'portfolio_return': portfolio_return,
            'allocations': dict(zip(self.strategies, allocations)),
            'reward': reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action):
        """
        Process the agent's action and convert to normalized allocations.
        
        Args:
            action: Raw action from the agent
            
        Returns:
            Normalized allocation weights
        """
        if self.tunable_params:
            # Extract allocations (first half of action array)
            allocation_part = action[:len(self.strategies)]
            
            # Parameter part is ignored for now (would be used to tune strategy parameters)
            # param_part = action[len(self.strategies):]
            
            allocations = allocation_part
        else:
            allocations = action
        
        # Ensure allocations are non-negative
        allocations = np.maximum(allocations, 0)
        
        # Normalize to sum to 1.0
        allocation_sum = np.sum(allocations)
        if allocation_sum > 0:
            allocations = allocations / allocation_sum
        else:
            # If all allocations are 0, use equal allocation
            allocations = np.ones_like(allocations) / len(allocations)
        
        return allocations
    
    def _get_observation(self):
        """
        Construct the observation vector.
        
        Returns:
            Numpy array of observations
        """
        current_date = self.timeline.iloc[self.current_step]['timestamp']
        
        # 1. Market features
        market_features = self._get_market_features(current_date)
        
        # 2. Strategy performance metrics
        strategy_features = self._get_strategy_features(current_date)
        
        # 3. Portfolio state
        portfolio_features = self._get_portfolio_features()
        
        # 4. Pattern insights if enabled
        if self.use_pattern_insights:
            pattern_features = self._get_pattern_features()
            observation = np.concatenate([market_features, strategy_features, portfolio_features, pattern_features])
        else:
            observation = np.concatenate([market_features, strategy_features, portfolio_features])
        
        return observation.astype(np.float32)
    
    def _get_market_features(self, current_date):
        """
        Extract market-related features.
        
        Args:
            current_date: Current simulation date
            
        Returns:
            Numpy array of market features
        """
        # Find portfolio snapshots within window period
        start_date = current_date - timedelta(days=self.window_size)
        
        window_portfolio = self.portfolio_df[
            (self.portfolio_df['timestamp'] >= start_date) & 
            (self.portfolio_df['timestamp'] <= current_date)
        ] if not self.portfolio_df.empty else pd.DataFrame()
        
        # Default features if no data available
        if window_portfolio.empty:
            return np.zeros(10)
        
        # Extract and calculate features
        returns = np.zeros(self.window_size)
        volatility = 0.0
        trend = 0.0
        volume = 0.0
        
        if 'daily_return' in window_portfolio.columns:
            returns_series = window_portfolio['daily_return'].fillna(0).values
            returns = np.pad(returns_series, (0, max(0, self.window_size - len(returns_series))))
            volatility = np.std(returns_series) if len(returns_series) > 1 else 0.0
            trend = np.mean(returns_series) / (volatility + 1e-10)  # Normalized trend
        
        # Calculate additional market features
        # In a real implementation, these would be more sophisticated
        drawdown = np.min(returns) if len(returns) > 0 else 0
        up_days = np.sum(returns > 0) / max(1, len(returns))
        down_days = np.sum(returns < 0) / max(1, len(returns))
        
        # Combine into feature vector
        # Use more recent returns as they're more significant
        market_features = np.array([
            returns[-5:].mean(),  # Recent returns (last 5 days)
            returns[-10:].mean(),  # Medium-term returns (last 10 days)
            returns.mean(),        # Full window returns
            volatility,
            trend,
            volume,
            drawdown,
            up_days,
            down_days,
            self.current_step / self.total_steps  # Normalized time position
        ])
        
        return market_features
    
    def _get_strategy_features(self, current_date):
        """
        Extract strategy-specific performance features.
        
        Args:
            current_date: Current simulation date
            
        Returns:
            Numpy array of strategy features
        """
        # Default features if no data available
        if self.signal_df.empty:
            return np.zeros(5 * len(self.strategies))
        
        start_date = current_date - timedelta(days=self.window_size)
        
        strategy_features = []
        
        for strategy in self.strategies:
            # Get signals for this strategy in the window
            strategy_signals = self.signal_df[
                (self.signal_df['timestamp'] >= start_date) & 
                (self.signal_df['timestamp'] <= current_date) &
                (self.signal_df['strategy'] == strategy)
            ]
            
            # Get trades for this strategy in the window
            strategy_trades = self.trade_df[
                (self.trade_df['timestamp'] >= start_date) & 
                (self.trade_df['timestamp'] <= current_date) &
                (self.trade_df['strategy'] == strategy)
            ] if 'strategy' in self.trade_df.columns else pd.DataFrame()
            
            # Calculate features
            signal_count = len(strategy_signals)
            
            # Signal strength
            avg_strength = strategy_signals['strength'].mean() if 'strength' in strategy_signals.columns and signal_count > 0 else 0.5
            
            # Win rate
            if not strategy_trades.empty and 'pnl' in strategy_trades.columns:
                win_rate = (strategy_trades['pnl'] > 0).mean() if len(strategy_trades) > 0 else 0.5
                avg_pnl = strategy_trades['pnl'].mean() if len(strategy_trades) > 0 else 0.0
                max_drawdown = strategy_trades['pnl'].min() if len(strategy_trades) > 0 else 0.0
            else:
                win_rate = 0.5
                avg_pnl = 0.0
                max_drawdown = 0.0
            
            # Combine into feature vector
            strat_features = np.array([
                signal_count / max(1, self.window_size),  # Normalized signal frequency
                avg_strength,
                win_rate,
                avg_pnl,
                max_drawdown
            ])
            
            strategy_features.append(strat_features)
        
        return np.concatenate(strategy_features)
    
    def _get_portfolio_features(self):
        """
        Extract portfolio state features.
        
        Returns:
            Numpy array of portfolio features
        """
        # Calculate portfolio metrics
        if len(self.portfolio_history) >= 2:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            current_return = returns[-1] if returns.size > 0 else 0.0
            avg_return = np.mean(returns) if returns.size > 0 else 0.0
            volatility = np.std(returns) if returns.size > 1 else 0.0
            
            # Calculate drawdown
            peak = np.maximum.accumulate(self.portfolio_history)
            drawdown = (peak[-1] - self.portfolio_history[-1]) / peak[-1] if peak[-1] > 0 else 0.0
        else:
            current_return = 0.0
            avg_return = 0.0
            volatility = 0.0
            drawdown = 0.0
        
        # Combine into feature vector
        portfolio_features = np.array([
            self.current_portfolio_value / 100000.0,  # Normalized portfolio value
            current_return,
            avg_return,
            volatility,
            drawdown
        ])
        
        return portfolio_features
    
    def _get_pattern_features(self):
        """
        Extract features from pattern insights.
        
        Returns:
            Numpy array of pattern features
        """
        # Default features if no insights available
        if not self.insights or "error" in self.insights:
            return np.zeros(10)
        
        # Extract features from pattern insights
        features = []
        
        # 1. Best strategies based on win rates
        if "win_rates" in self.insights and "by_strategy" in self.insights["win_rates"]:
            # Convert to DataFrame for easier analysis
            win_rates = pd.DataFrame(self.insights["win_rates"]["by_strategy"])
            if not win_rates.empty and 'win_rate' in win_rates.columns and 'strategy' in win_rates.columns:
                for strategy in self.strategies:
                    # Find win rate for this strategy
                    strat_win_rate = win_rates.loc[win_rates['strategy'] == strategy, 'win_rate'].values
                    win_rate = strat_win_rate[0] if len(strat_win_rate) > 0 else 0.5
                    features.append(win_rate)
            else:
                features.extend([0.5] * len(self.strategies))
        else:
            features.extend([0.5] * len(self.strategies))
        
        # 2. Market regime indicators
        if "regime_performance" in self.insights and "regime_returns" in self.insights["regime_performance"]:
            regime_data = self.insights["regime_performance"]["regime_returns"]
            if regime_data:
                # Get best and worst regimes
                regime_df = pd.DataFrame(regime_data)
                if not regime_df.empty and 'sharpe_ratio' in regime_df.columns:
                    best_sharpe = regime_df['sharpe_ratio'].max()
                    worst_sharpe = regime_df['sharpe_ratio'].min()
                    features.extend([best_sharpe, worst_sharpe])
                else:
                    features.extend([1.0, 0.5])
            else:
                features.extend([1.0, 0.5])
        else:
            features.extend([1.0, 0.5])
        
        # 3. Time pattern indicators
        if "time_patterns" in self.insights and "by_hour" in self.insights["time_patterns"]:
            hour_data = self.insights["time_patterns"]["by_hour"]
            if hour_data:
                hour_df = pd.DataFrame(hour_data)
                if not hour_df.empty and 'win_rate' in hour_df.columns:
                    best_hour_win_rate = hour_df['win_rate'].max()
                    features.append(best_hour_win_rate)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
        else:
            features.append(0.5)
        
        # Add dummy features to fill out the space
        while len(features) < 10:
            features.append(0.5)
        
        return np.array(features[:10])  # Limit to 10 features
    
    def _calculate_portfolio_return(self, allocations):
        """
        Calculate portfolio return based on allocations and strategy performance.
        
        Args:
            allocations: Strategy allocation weights
            
        Returns:
            Portfolio return for the step
        """
        current_date = self.timeline.iloc[self.current_step]['timestamp']
        
        # Get all signals on the current date
        day_signals = self.signal_df[self.signal_df['timestamp'].dt.date == current_date.date()]
        
        # Calculate strategy returns based on signals
        strategy_returns = np.zeros(len(self.strategies))
        
        for i, strategy in enumerate(self.strategies):
            # Get signals for this strategy
            strategy_signals = day_signals[day_signals['strategy'] == strategy]
            
            if not strategy_signals.empty:
                # In a real implementation, we would calculate returns based on signal strength,
                # direction, and market movement
                # Here we use a simplified approach
                
                # Average signal strength or confidence
                if 'strength' in strategy_signals.columns:
                    avg_strength = strategy_signals['strength'].mean()
                elif 'confidence' in strategy_signals.columns:
                    avg_strength = strategy_signals['confidence'].mean()
                else:
                    avg_strength = 0.5
                
                # Simplified return calculation
                # In a real implementation, this would be based on actual signal outcomes
                # or historical performance in similar conditions
                
                # 1. Base return (random component simulating market noise)
                base_return = np.random.normal(0.0001, 0.01)  # Mean slightly positive, std=1%
                
                # 2. Strategy-specific return based on signals
                if 'direction' in strategy_signals.columns:
                    # Aggregate signal direction (-1, 0, 1)
                    directions = strategy_signals['direction'].map({'buy': 1, 'sell': -1, 'neutral': 0}).values
                    signal_direction = np.mean(directions) if len(directions) > 0 else 0
                    
                    # Adjust base return based on signal direction and strength
                    signal_return = signal_direction * avg_strength * 0.02  # Max 2% impact
                else:
                    signal_return = (avg_strength - 0.5) * 0.02  # Neutral = 0, Strong = +1%, Weak = -1%
                
                strategy_returns[i] = base_return + signal_return
            else:
                # No signals for this strategy today
                strategy_returns[i] = np.random.normal(0.0, 0.005)  # Mean 0, std=0.5%
        
        # Calculate portfolio return based on allocation weights
        portfolio_return = np.sum(allocations * strategy_returns)
        
        return portfolio_return
    
    def _calculate_reward(self, portfolio_return):
        """
        Calculate reward based on portfolio performance.
        
        Args:
            portfolio_return: Current step's portfolio return
            
        Returns:
            Reward value
        """
        if self.reward_type == 'pnl':
            # Simple PnL-based reward
            reward = portfolio_return * 100  # Scale up small returns
            
        elif self.reward_type == 'sharpe':
            # Sharpe ratio based reward
            if len(self.portfolio_history) > self.window_size:
                # Calculate returns
                returns = np.diff(self.portfolio_history[-self.window_size:]) / self.portfolio_history[-self.window_size:-1]
                
                # Calculate Sharpe ratio (annualized)
                mean_return = np.mean(returns)
                std_return = np.std(returns) + 1e-6  # Avoid division by zero
                sharpe = mean_return / std_return * np.sqrt(252)  # Annualized
                
                # Use Sharpe as reward
                reward = sharpe
            else:
                # Not enough history for Sharpe
                reward = portfolio_return * 100
                
        elif self.reward_type == 'sortino':
            # Sortino ratio based reward (only penalizes downside volatility)
            if len(self.portfolio_history) > self.window_size:
                # Calculate returns
                returns = np.diff(self.portfolio_history[-self.window_size:]) / self.portfolio_history[-self.window_size:-1]
                
                # Calculate Sortino ratio (annualized)
                mean_return = np.mean(returns)
                
                # Only negative returns for downside deviation
                downside_returns = returns[returns < 0]
                downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
                
                sortino = mean_return / downside_deviation * np.sqrt(252)  # Annualized
                
                # Use Sortino as reward
                reward = sortino
            else:
                # Not enough history for Sortino
                reward = portfolio_return * 100
                
        elif self.reward_type == 'calmar':
            # Calmar ratio (return / max drawdown)
            if len(self.portfolio_history) > self.window_size:
                # Calculate returns
                values = np.array(self.portfolio_history[-self.window_size:])
                returns = np.diff(values) / values[:-1]
                
                # Calculate max drawdown
                peak = np.maximum.accumulate(values)
                drawdowns = (peak - values) / peak
                max_drawdown = np.max(drawdowns) + 1e-6  # Avoid division by zero
                
                # Calculate Calmar ratio
                mean_annual_return = np.mean(returns) * 252  # Annualized
                calmar = mean_annual_return / max_drawdown
                
                # Use Calmar as reward, with penalty for large drawdowns
                reward = calmar - max_drawdown * 10  # Penalty for drawdowns
            else:
                # Not enough history for Calmar
                reward = portfolio_return * 100
        else:
            # Default: simple return
            reward = portfolio_return * 100
        
        return reward
    
    def render(self):
        """Render the environment visualization"""
        if self.render_mode == "human":
            # Display current state
            current_date = self.timeline.iloc[self.current_step]['timestamp'].strftime('%Y-%m-%d')
            print(f"Step: {self.current_step}, Date: {current_date}")
            print(f"Portfolio Value: ${self.current_portfolio_value:.2f}")
            print("Strategy Allocations:")
            for i, strategy in enumerate(self.strategies):
                print(f"  {strategy}: {self.current_allocations[i]:.2%}")
            
            if len(self.portfolio_history) > 1:
                returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                current_return = returns[-1] if returns.size > 0 else 0.0
                print(f"Current Return: {current_return:.2%}")
            
            print("-" * 40)
        
        elif self.render_mode == "rgb_array":
            # Would implement a matplotlib visualization here
            # Return RGB array for video rendering
            return np.zeros((400, 600, 3), dtype=np.uint8)
        
        return None
    
    def close(self):
        """Clean up resources"""
        pass 