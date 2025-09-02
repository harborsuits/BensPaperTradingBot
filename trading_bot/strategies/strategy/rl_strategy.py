#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RL-based Strategy - Reinforcement Learning strategies for trading
that integrate with the StrategyRotator system.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# Import base strategy class
from trading_bot.strategy.strategy_rotator import Strategy
from trading_bot.common.market_types import MarketRegime
from trading_bot.common.config_utils import setup_directories, load_config, save_state, load_state

# Setup logging
logger = logging.getLogger("RLStrategy")

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for RL training"""
    
    def __init__(self, capacity: int = 10000):
        """Initialize replay buffer with given capacity"""
        self.memory = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to memory"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences"""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)


class TradingEnvironment(gym.Env):
    """OpenAI Gym environment for trading simulation"""
    
    def __init__(self, market_data: pd.DataFrame, window_size: int = 30, 
                 commission: float = 0.001, initial_balance: float = 10000.0):
        """
        Initialize the trading environment.
        
        Args:
            market_data: DataFrame with market data (OHLCV)
            window_size: Number of time steps to include in state
            commission: Trading commission as a fraction
            initial_balance: Initial account balance
        """
        super(TradingEnvironment, self).__init__()
        
        self.market_data = market_data
        self.window_size = window_size
        self.commission = commission
        self.initial_balance = initial_balance
        
        # Action space: -1 (sell), 0 (hold), 1 (buy)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price data + account info
        self.feature_dim = 5  # OHLCV
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size * self.feature_dim + 2,)  # +2 for position and balance
        )
        
        # Episode state
        self.reset()
    
    def reset(self):
        """Reset the environment to the beginning of an episode"""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # No position
        self.done = False
        self.history = []
        
        return self._get_observation()
    
    def step(self, action):
        """Take a step in the environment with the given action"""
        # Map action (0, 1, 2) to (-1, 0, 1)
        action_map = {0: -1, 1: 0, 2: 1}
        action = action_map[action]
        
        # Get current price
        current_price = self.market_data.iloc[self.current_step]['close']
        
        # Calculate reward based on action and price change
        next_price = self.market_data.iloc[self.current_step + 1]['close']
        price_change = (next_price - current_price) / current_price
        
        # Apply position change
        old_position = self.position
        self.position = action  # Simplified to just taking the action as the new position
        
        # Calculate trade cost (commission)
        if old_position != self.position:
            trade_cost = abs(self.position - old_position) * current_price * self.commission
        else:
            trade_cost = 0
        
        # Update balance based on position and price change
        self.balance += old_position * current_price * price_change - trade_cost
        
        # Record history
        self.history.append({
            'step': self.current_step,
            'price': current_price,
            'action': action,
            'position': self.position,
            'balance': self.balance,
            'reward': self.position * price_change - trade_cost
        })
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        if self.current_step >= len(self.market_data) - 1:
            self.done = True
        
        # Calculate reward: PnL minus trading cost
        reward = self.position * price_change - trade_cost
        
        # Get observation for next state
        observation = self._get_observation()
        
        return observation, reward, self.done, {'balance': self.balance}
    
    def _get_observation(self):
        """Get the current observation (state)"""
        # Extract window of market data
        window_data = self.market_data.iloc[self.current_step - self.window_size:self.current_step]
        
        # Normalize features 
        features = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            # Simple normalization: divide by mean
            if col == 'volume':
                feature = window_data[col].values / window_data[col].mean()
            else:
                feature = window_data[col].values / window_data['close'].iloc[-1]
            features.extend(feature)
        
        # Add position and balance
        features.append(self.position)
        features.append(self.balance / self.initial_balance)
        
        return np.array(features)
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            if self.current_step > 0:
                history = self.history[-1]
                print(f"Step: {history['step']}, Price: {history['price']:.2f}, "
                      f"Position: {history['position']}, Balance: {history['balance']:.2f}, "
                      f"Reward: {history['reward']:.4f}")
        else:
            pass


class DQNModel(nn.Module):
    """Deep Q-Network model for trading"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [128, 64]):
        """Initialize the DQN model"""
        super(DQNModel, self).__init__()
        
        # Build layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)


class RLStrategy(Strategy):
    """Base class for reinforcement learning-based trading strategies"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the RL strategy"""
        super().__init__(name, config)
        
        # Get RL-specific config parameters
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)  # Discount factor
        self.eps_start = config.get("eps_start", 1.0)  # Epsilon-greedy start value
        self.eps_end = config.get("eps_end", 0.01)  # Epsilon-greedy end value
        self.eps_decay = config.get("eps_decay", 0.995)  # Epsilon decay rate
        self.learning_rate = config.get("learning_rate", 0.001)
        self.target_update = config.get("target_update", 10)  # Update target network every N episodes
        self.memory_capacity = config.get("memory_capacity", 10000)
        self.window_size = config.get("window_size", 30)
        
        # Setup training and inference directories
        self.paths = setup_directories(
            data_dir=config.get("data_dir"),
            component_name=f"rl_strategy_{name}"
        )
        
        # Initialize replay memory
        self.memory = ReplayBuffer(self.memory_capacity)
        
        # Initialize epsilon for exploration
        self.epsilon = self.eps_start
        
        # Initialize timestep (used for target network updates)
        self.timestep = 0
        
        # Feature dimension (state size)
        self.feature_dim = 5 * self.window_size + 2  # OHLCV + position + balance
        
        # Initialize models
        self._initialize_models()
        
        # Training/inference flags
        self.is_training = False
        
        # Latest action
        self.latest_action = 0  # neutral
    
    def _initialize_models(self):
        """Initialize Q-network and target network"""
        # Policy network (main Q-network)
        self.policy_net = DQNModel(self.feature_dim, 3)  # 3 actions: sell, hold, buy
        
        # Target network
        self.target_net = DQNModel(self.feature_dim, 3)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set to evaluation mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def train(self, market_data: pd.DataFrame, epochs: int = 100):
        """
        Train the RL strategy on historical market data.
        
        Args:
            market_data: DataFrame with market data
            epochs: Number of training epochs
        """
        self.is_training = True
        
        # Create environment
        env = TradingEnvironment(market_data, window_size=self.window_size)
        
        # Training loop
        total_rewards = []
        
        for epoch in range(epochs):
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            done = False
            total_reward = 0
            
            while not done:
                # Select action
                action = self._select_action(state)
                
                # Take step in environment
                next_state, reward, done, _ = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32)
                
                # Store transition in memory
                self.memory.add(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                
                # Update total reward
                total_reward += reward.item()
                
                # Optimize model
                self._optimize_model()
                
                # Update timestep
                self.timestep += 1
                
                # Update target network
                if self.timestep % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
            
            # Record total reward
            total_rewards.append(total_reward)
            
            # Log progress
            logger.info(f"Epoch {epoch+1}/{epochs}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        # Save model
        self._save_model()
        
        self.is_training = False
        return total_rewards
    
    def _select_action(self, state):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state tensor
        
        Returns:
            torch.Tensor: Selected action
        """
        if self.is_training and random.random() < self.epsilon:
            # Random action
            return torch.tensor([[random.randrange(3)]], dtype=torch.long)
        else:
            # Greedy action
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
    
    def _optimize_model(self):
        """Optimize the model using a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        
        # Unpack batch
        states = torch.cat([torch.tensor(exp.state, dtype=torch.float32) for exp in batch])
        actions = torch.cat([torch.tensor([[exp.action]], dtype=torch.long) for exp in batch])
        rewards = torch.cat([torch.tensor([exp.reward], dtype=torch.float32) for exp in batch])
        next_states = torch.cat([torch.tensor(exp.next_state, dtype=torch.float32) for exp in batch])
        dones = torch.cat([torch.tensor([exp.done], dtype=torch.bool) for exp in batch])
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the actions taken
        state_action_values = self.policy_net(states).gather(1, actions)
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)
        with torch.no_grad():
            next_state_values[~dones] = self.target_net(next_states[~dones]).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = rewards + (self.gamma * next_state_values)
        
        # Compute loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
    def _save_model(self):
        """Save the model to disk"""
        model_path = os.path.join(self.paths["data_dir"], "model.pt")
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'timestep': self.timestep
        }, model_path)
        logger.info(f"Model saved to {model_path}")
    
    def _load_model(self):
        """Load the model from disk"""
        model_path = os.path.join(self.paths["data_dir"], "model.pt")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.timestep = checkpoint['timestep']
            logger.info(f"Model loaded from {model_path}")
            return True
        else:
            logger.warning(f"No model found at {model_path}")
            return False
    
    def generate_signal(self, market_data: Dict[str, Any]) -> float:
        """
        Generate a trading signal using the RL model.
        
        Args:
            market_data: Current market data
            
        Returns:
            float: Signal between -1.0 (strong sell) and 1.0 (strong buy)
        """
        # Convert market data to state
        state = self._prepare_state(market_data)
        
        if state is None:
            return 0.0
        
        # Select action using the policy network
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.max(1)[1].item()
        
        # Map action (0, 1, 2) to signal (-1, 0, 1)
        action_map = {0: -1.0, 1: 0.0, 2: 1.0}
        signal = action_map[action_idx]
        
        # Update last signal and time
        self.last_signal = signal
        self.last_update_time = datetime.now()
        self.latest_action = action_idx
        
        return signal
    
    def _prepare_state(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Prepare state vector from market data.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            np.ndarray: State vector for the model
        """
        # Extract price and volume data
        prices = market_data.get("prices", [])
        volumes = market_data.get("volumes", [])
        
        # Check if we have enough data
        if len(prices) < self.window_size:
            logger.warning(f"Not enough price data: {len(prices)} < {self.window_size}")
            return None
        
        # Create OHLCV data (simplified - assuming all are close prices)
        # In a real implementation, you'd use actual OHLCV data
        ohlcv = []
        for i in range(self.window_size):
            idx = len(prices) - self.window_size + i
            price = prices[idx]
            volume = volumes[idx] if idx < len(volumes) else 0
            
            # Approximate OHLCV (this is simplified)
            open_price = price
            high_price = price * 1.005  # Approximate
            low_price = price * 0.995   # Approximate
            close_price = price
            
            # Normalize
            ref_price = prices[-1]  # Current price
            ohlcv.extend([
                open_price / ref_price,
                high_price / ref_price,
                low_price / ref_price,
                close_price / ref_price,
                volume / (sum(volumes[-self.window_size:]) / self.window_size) if volumes else 1.0
            ])
        
        # Add position and balance (placeholder values for inference)
        position = market_data.get("position", 0.0)
        balance = market_data.get("balance", 1.0)  # Normalized
        
        ohlcv.append(position)
        ohlcv.append(balance)
        
        return np.array(ohlcv)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "epsilon": self.epsilon,
            "timestep": self.timestep,
            "latest_action": self.latest_action
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLStrategy':
        """Create strategy from dictionary."""
        strategy = super().from_dict(data)
        strategy.epsilon = data.get("epsilon", strategy.eps_start)
        strategy.timestep = data.get("timestep", 0)
        strategy.latest_action = data.get("latest_action", 0)
        
        # Load model
        strategy._load_model()
        
        return strategy


class DQNStrategy(RLStrategy):
    """DQN-based trading strategy"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize DQN strategy"""
        super().__init__(name, config or {})
    
    # All core functionality inherited from RLStrategy


class PPOStrategy(RLStrategy):
    """PPO-based trading strategy (stub for future implementation)"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize PPO strategy"""
        super().__init__(name, config or {})
        logger.warning("PPO Strategy not fully implemented yet - using base RL implementation")
    
    # Future implementation would include PPO-specific components


class MetaLearningStrategy(RLStrategy):
    """
    Meta-learning strategy that can adapt quickly to new market regimes.
    This is a stub for future implementation.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize meta-learning strategy"""
        super().__init__(name, config or {})
        self.meta_models = {}  # Models for different market regimes
        logger.warning("Meta-Learning Strategy not fully implemented yet - using base RL implementation")
    
    def adapt_to_regime(self, regime: MarketRegime):
        """
        Adapt the strategy to a specific market regime.
        
        Args:
            regime: Market regime
        """
        regime_name = regime.name
        logger.info(f"Adapting to regime: {regime_name}")
        
        # In a full implementation, this would switch between pre-trained models
        # or adapt the existing model to the new regime
        if regime_name in self.meta_models:
            # Use existing model for this regime
            logger.info(f"Using existing model for regime: {regime_name}")
            # Placeholder for model switching logic
        else:
            # Create new model for this regime
            logger.info(f"Creating new model for regime: {regime_name}")
            # Placeholder for model creation logic
    
    # Future implementation would include meta-learning components


# Simple example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create test market data
    # In a real implementation, you would use actual market data
    prices = [100 + i * 0.1 + np.sin(i/5) * 2 for i in range(100)]
    volumes = [1000 + np.random.normal(0, 100) for _ in range(100)]
    
    market_data = {
        "prices": prices,
        "volumes": volumes
    }
    
    # Create RL strategy
    config = {
        "window_size": 20,
        "batch_size": 32,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 0.995,
        "learning_rate": 0.001
    }
    
    strategy = DQNStrategy("DQNStrategy", config)
    
    # Generate signal
    signal = strategy.generate_signal(market_data)
    print(f"Generated signal: {signal}") 