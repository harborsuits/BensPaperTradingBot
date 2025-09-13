#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Position Sizer Environment

This module implements a custom Gym environment for training RL agents to 
optimize position sizing based on signal confidence and market conditions.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

from trading_bot.ml.signal_model import SignalModel
from trading_bot.ml.market_regime_autoencoder import MarketRegimeAutoencoder

logger = logging.getLogger(__name__)

class PositionSizerEnv(gym.Env):
    """
    Custom environment for position sizing using RL.
    
    This environment simulates a trading scenario where an agent needs to decide
    on position size based on signal confidence, market regime, and portfolio state.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, data: Dict[str, pd.DataFrame], config: Dict[str, Any] = None):
        """
        Initialize the position sizer environment
        
        Args:
            data: Dictionary of market data frames for training
            config: Configuration dictionary
        """
        super(PositionSizerEnv, self).__init__()
        
        self.data = data
        self.config = config or {}
        self._set_default_config()
        
        # Define action space (position size as percentage of capital)
        self.action_space = spaces.Box(
            low=0.0, 
            high=self.config["max_position_size"],
            shape=(1,),
            dtype=np.float32
        )
        
        # Define observation space (features for decision-making)
        # Structure: [signal_confidence, account_metrics, risk_metrics, market_regime]
        num_features = 6 + self.config["num_regimes"]  # Base features + one-hot regime
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0,
            shape=(num_features,),
            dtype=np.float32
        )
        
        # Set up the environment
        self._setup_environment()
        
        logger.info("Position Sizer Environment initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # Environment parameters
        self.config.setdefault("initial_balance", 100000.0)
        self.config.setdefault("max_position_size", 0.25)  # 25% of capital max
        self.config.setdefault("transaction_cost", 0.001)  # 0.1% per trade
        self.config.setdefault("slippage", 0.0005)         # 0.05% slippage
        self.config.setdefault("num_regimes", 5)           # Number of market regimes
        
        # Reward function parameters
        self.config.setdefault("reward_type", "sortino")   # Options: returns, sharpe, sortino
        self.config.setdefault("risk_free_rate", 0.02/252) # Daily risk-free rate
        self.config.setdefault("reward_scale", 10.0)       # Scale factor for rewards
        
        # Risk penalty parameters
        self.config.setdefault("drawdown_penalty", 2.0)    # Penalty for drawdowns
        self.config.setdefault("oversize_penalty", 1.0)    # Penalty for exceeding max size
        
        # Simulation parameters
        self.config.setdefault("episode_length", 63)       # ~ 3 months (trading days)
        self.config.setdefault("warmup_period", 20)        # Days for calculating metrics
    
    def _setup_environment(self):
        """Set up the environment for training"""
        # Get the symbol from the first data frame
        self.symbols = list(self.data.keys())
        if not self.symbols:
            raise ValueError("No data provided for environment")
        
        # Create data preprocessing pipeline
        self._preprocess_data()
        
        # Initialize environment state
        self.reset()
    
    def _preprocess_data(self):
        """Preprocess data for training"""
        # Calculate returns for each asset
        self.returns_data = {}
        for symbol, df in self.data.items():
            # Calculate daily returns
            returns = df['close'].pct_change().fillna(0)
            self.returns_data[symbol] = returns
        
        # Generate signals and regime detection
        self._generate_signals()
        self._detect_regimes()
        
        # Create feature dataframes
        self._prepare_features()
    
    def _generate_signals(self):
        """Generate trading signals from data"""
        # This would ideally use the SignalModel, but for now we'll create synthetic signals
        self.signals = {}
        
        for symbol, df in self.data.items():
            # Create a simple signal model (random for simulation, would use real model in production)
            # In a real implementation, load or create SignalModel instance:
            # signal_model = SignalModel(self.config.get("signal_model_config"))
            # signals = signal_model.predict(df)
            
            # For simulation, create synthetic signals
            np.random.seed(42)  # For reproducibility
            length = len(df)
            
            # Create random signal confidences
            signal_df = pd.DataFrame({
                'date': df.index,
                'confidence': np.random.uniform(0.4, 0.9, size=length),
                'direction': np.random.choice([-1, 1], size=length, p=[0.3, 0.7]),
                'returns': self.returns_data[symbol].values
            })
            
            # Make signals autocorrelated
            for i in range(1, length):
                if np.random.random() < 0.8:  # 80% chance to keep previous direction
                    signal_df.loc[i, 'direction'] = signal_df.loc[i-1, 'direction']
            
            # Signal confidence
            signal_df['signal'] = signal_df['confidence'] * signal_df['direction']
            
            # Align signal with future returns for training
            # Higher signal should correlate with higher returns for better learning
            corr_factor = 0.4  # Correlation factor
            sorted_signals = signal_df['signal'].sort_values()
            sorted_returns = signal_df['returns'].shift(-1).sort_values()
            signal_df['signal'] = pd.Series(
                np.interp(
                    signal_df['signal'],
                    sorted_signals,
                    corr_factor * sorted_returns + (1 - corr_factor) * sorted_signals
                ),
                index=signal_df.index
            )
            
            # Scale between -1 and 1
            signal_df['signal'] = signal_df['signal'].clip(-1, 1)
            
            # Make confidence 0 to 1
            signal_df['confidence'] = (signal_df['signal'].abs() * 0.5) + 0.5
            
            self.signals[symbol] = signal_df
    
    def _detect_regimes(self):
        """Detect market regimes from data"""
        # This would ideally use the MarketRegimeAutoencoder, but for now we'll create synthetic regimes
        self.regimes = {}
        
        for symbol, df in self.data.items():
            # In a real implementation, load or create MarketRegimeAutoencoder instance:
            # regime_detector = MarketRegimeAutoencoder(self.config.get("regime_detector_config"))
            # regimes = regime_detector.detect_market_regime(df)
            
            # For simulation, create synthetic regimes
            length = len(df)
            
            # Create random regime changes
            regime_changes = np.random.choice(
                [0, 1], 
                size=length, 
                p=[0.95, 0.05]  # 5% chance of regime change
            )
            
            # Initialize with regime 0
            regimes = np.zeros(length)
            current_regime = 0
            
            # Generate regimes
            for i in range(length):
                if regime_changes[i] == 1:
                    # Change to a new regime
                    current_regime = (current_regime + 1) % self.config["num_regimes"]
                regimes[i] = current_regime
            
            # Add to regime dataframe
            regime_df = pd.DataFrame({
                'date': df.index,
                'regime': regimes.astype(int),
                'volatility': df['close'].pct_change().rolling(20).std().fillna(0.01)
            })
            
            self.regimes[symbol] = regime_df
    
    def _prepare_features(self):
        """Prepare combined features for training"""
        self.features = {}
        
        for symbol in self.symbols:
            # Get dataframes
            signal_df = self.signals[symbol]
            regime_df = self.regimes[symbol]
            
            # Combine features
            feature_df = pd.DataFrame({
                'date': signal_df['date'],
                'signal': signal_df['signal'],
                'confidence': signal_df['confidence'],
                'regime': regime_df['regime'],
                'volatility': regime_df['volatility'],
                'returns': self.returns_data[symbol]
            })
            
            # Add rolling return metrics
            feature_df['return_5d'] = self.returns_data[symbol].rolling(5).mean().fillna(0)
            feature_df['return_20d'] = self.returns_data[symbol].rolling(20).mean().fillna(0)
            feature_df['volatility_5d'] = self.returns_data[symbol].rolling(5).std().fillna(0.01)
            feature_df['volatility_20d'] = self.returns_data[symbol].rolling(20).std().fillna(0.01)
            
            self.features[symbol] = feature_df
    
    def reset(self, seed: Optional[int] = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        # Reset random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Choose a random asset and starting point
        self.current_symbol = np.random.choice(self.symbols)
        features = self.features[self.current_symbol]
        
        # Choose a random starting point that allows for a full episode
        max_start = len(features) - self.config["episode_length"] - 1
        if max_start <= 0:
            raise ValueError(f"Data for {self.current_symbol} is too short for episode length")
        
        self.current_step = np.random.randint(self.config["warmup_period"], max_start)
        
        # Reset portfolio state
        self.balance = self.config["initial_balance"]
        self.equity = self.balance
        self.position_size = 0.0
        self.position_value = 0.0
        self.portfolio_value = self.balance
        self.portfolio_return = 0.0
        
        # Reset metrics tracking
        self.returns_history = []
        self.position_history = []
        self.value_history = [self.portfolio_value]
        self.reward_history = []
        
        # Initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and move to next state
        
        Args:
            action: Position size action (percentage of capital)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get position size from action
        new_position_size = float(action[0])
        
        # Apply constraints
        new_position_size = np.clip(
            new_position_size, 
            0.0, 
            self.config["max_position_size"]
        )
        
        # Check for episode termination
        done = False
        truncated = False
        
        # Execute trading step
        features = self.features[self.current_symbol]
        current_features = features.iloc[self.current_step]
        next_features = features.iloc[self.current_step + 1]
        
        # Get current signal direction
        signal_direction = np.sign(current_features['signal'])
        if signal_direction == 0:
            signal_direction = 1  # Default to long
        
        # Calculate transaction costs if position size changed
        old_position_size = self.position_size
        position_change = abs(new_position_size - old_position_size)
        transaction_cost = position_change * self.portfolio_value * self.config["transaction_cost"]
        
        # Apply transaction costs
        self.balance -= transaction_cost
        
        # Update position size
        self.position_size = new_position_size
        
        # Get returns for next day (simulate overnight holding)
        next_return = next_features['returns'] * signal_direction
        
        # Calculate position P&L
        position_pnl = self.position_size * self.portfolio_value * next_return
        
        # Apply slippage on P&L
        slippage_cost = abs(position_pnl) * self.config["slippage"]
        position_pnl -= slippage_cost
        
        # Update account state
        self.balance += position_pnl
        self.portfolio_value = self.balance
        self.position_value = self.position_size * self.portfolio_value
        
        # Calculate return for this step
        step_return = position_pnl / (self.portfolio_value - position_pnl)
        self.returns_history.append(step_return)
        self.position_history.append(self.position_size)
        self.value_history.append(self.portfolio_value)
        
        # Calculate reward
        reward = self._calculate_reward(step_return)
        self.reward_history.append(reward)
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        if self.current_step >= len(features) - 1:
            done = True
        elif (self.current_step - (self.config["warmup_period"] + 1)) >= self.config["episode_length"]:
            truncated = True
        elif self.portfolio_value <= 0:
            # Bankrupt
            done = True
            reward -= 10.0  # Big penalty for bankruptcy
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info()
        
        # Add portfolio return to info
        cumulative_return = (self.portfolio_value / self.config["initial_balance"]) - 1
        self.portfolio_return = cumulative_return
        info["portfolio_return"] = cumulative_return
        info["step_return"] = step_return
        
        return observation, reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct the observation vector
        
        Returns:
            Observation array
        """
        features = self.features[self.current_symbol]
        current_features = features.iloc[self.current_step]
        
        # Extract features
        signal_confidence = current_features['confidence']
        regime = int(current_features['regime'])
        volatility = current_features['volatility']
        
        # Get historical return metrics (if available)
        if len(self.returns_history) >= 5:
            recent_returns = np.mean(self.returns_history[-5:])
            return_volatility = np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.01
            max_drawdown = self._calculate_drawdown()
        else:
            recent_returns = 0
            return_volatility = 0.01
            max_drawdown = 0
        
        # Prepare account metrics
        margin_used_pct = self.position_size
        
        # One-hot encode market regime
        regime_one_hot = np.zeros(self.config["num_regimes"])
        if 0 <= regime < self.config["num_regimes"]:
            regime_one_hot[regime] = 1
        
        # Construct state vector
        state = np.array([
            signal_confidence,          # Signal confidence
            margin_used_pct,            # Position size (as % of capital)
            recent_returns,             # Recent performance
            volatility,                 # Market volatility
            return_volatility,          # Portfolio volatility
            max_drawdown,               # Maximum drawdown
            *regime_one_hot             # Market regime one-hot encoding
        ], dtype=np.float32)
        
        return state
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get information about current state
        
        Returns:
            Info dictionary
        """
        features = self.features[self.current_symbol]
        current_features = features.iloc[self.current_step]
        
        # Calculate metrics if enough history
        sharpe = self._calculate_sharpe() if len(self.returns_history) >= 20 else 0
        sortino = self._calculate_sortino() if len(self.returns_history) >= 20 else 0
        
        return {
            "symbol": self.current_symbol,
            "step": self.current_step,
            "date": current_features['date'],
            "portfolio_value": self.portfolio_value,
            "position_size": self.position_size,
            "position_value": self.position_value,
            "signal": current_features['signal'],
            "confidence": current_features['confidence'],
            "regime": int(current_features['regime']),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "drawdown": self._calculate_drawdown()
        }
    
    def _calculate_reward(self, step_return: float) -> float:
        """
        Calculate reward for the current step
        
        Args:
            step_return: Return for the current step
            
        Returns:
            Calculated reward
        """
        reward_type = self.config["reward_type"]
        reward_scale = self.config["reward_scale"]
        
        # Base reward on return
        if reward_type == "returns":
            # Simple return-based reward
            reward = step_return * 100  # Scale to make returns more meaningful
        
        elif reward_type == "sharpe":
            # Sharpe ratio component (if enough history)
            if len(self.returns_history) >= 20:
                sharpe = self._calculate_sharpe()
                reward = sharpe
            else:
                reward = step_return * 100
        
        elif reward_type == "sortino":
            # Sortino ratio component (if enough history)
            if len(self.returns_history) >= 20:
                sortino = self._calculate_sortino()
                reward = sortino
            else:
                reward = step_return * 100
        
        else:
            # Default to return
            reward = step_return * 100
        
        # Apply drawdown penalty
        drawdown = self._calculate_drawdown()
        drawdown_penalty = self.config["drawdown_penalty"] * (drawdown ** 2)
        reward -= drawdown_penalty
        
        # Apply oversize penalty if needed
        if self.position_size > self.config["max_position_size"]:
            oversize_amount = self.position_size - self.config["max_position_size"]
            oversize_penalty = self.config["oversize_penalty"] * (oversize_amount ** 2)
            reward -= oversize_penalty
        
        # Scale reward
        reward = reward * reward_scale
        
        return reward
    
    def _calculate_sharpe(self) -> float:
        """
        Calculate Sharpe ratio from returns history
        
        Returns:
            Sharpe ratio
        """
        if len(self.returns_history) < 2:
            return 0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - self.config["risk_free_rate"]
        
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return == 0:
            return 0
        
        # Annualize (assuming daily returns and 252 trading days)
        sharpe = mean_excess_return / std_excess_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino(self) -> float:
        """
        Calculate Sortino ratio from returns history
        
        Returns:
            Sortino ratio
        """
        if len(self.returns_history) < 2:
            return 0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - self.config["risk_free_rate"]
        
        mean_excess_return = np.mean(excess_returns)
        
        # Only consider downside deviation
        downside_returns = np.minimum(excess_returns, 0)
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_deviation == 0:
            return 0
        
        # Annualize (assuming daily returns and 252 trading days)
        sortino = mean_excess_return / downside_deviation * np.sqrt(252)
        
        return sortino
    
    def _calculate_drawdown(self) -> float:
        """
        Calculate current drawdown
        
        Returns:
            Maximum drawdown as a positive number (e.g., 0.1 for 10% drawdown)
        """
        if len(self.value_history) < 2:
            return 0
        
        values = np.array(self.value_history)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        
        return float(drawdown[-1])
    
    def render(self, mode: str = 'human'):
        """
        Render the environment
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            features = self.features[self.current_symbol]
            current_features = features.iloc[self.current_step]
            
            print(f"Step: {self.current_step}")
            print(f"Symbol: {self.current_symbol}")
            print(f"Date: {current_features['date']}")
            print(f"Signal: {current_features['signal']:.4f}")
            print(f"Confidence: {current_features['confidence']:.4f}")
            print(f"Regime: {int(current_features['regime'])}")
            print(f"Position Size: {self.position_size:.4f}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Return: {self.portfolio_return:.2%}")
            
            if len(self.returns_history) >= 20:
                print(f"Sharpe Ratio: {self._calculate_sharpe():.4f}")
                print(f"Sortino Ratio: {self._calculate_sortino():.4f}")
            
            print(f"Drawdown: {self._calculate_drawdown():.2%}")
            print("-" * 50)
    
    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get statistics for the current episode
        
        Returns:
            Dictionary of episode statistics
        """
        episode_return = self.portfolio_return
        sharpe = self._calculate_sharpe() if len(self.returns_history) >= 20 else 0
        sortino = self._calculate_sortino() if len(self.returns_history) >= 20 else 0
        
        return {
            "portfolio_return": episode_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": self._calculate_drawdown(),
            "final_value": self.portfolio_value,
            "avg_position_size": np.mean(self.position_history) if self.position_history else 0,
            "n_steps": len(self.returns_history)
        }
