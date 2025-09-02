#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning Trainer Module for BensBot

This module handles the training and evaluation of RL agents
for trading strategies. It manages the interactions between
agents and environments, logging, checkpointing, and evaluation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import os
import datetime
import json
from typing import Dict, List, Union, Tuple, Optional, Any
import gym

# Import local modules
from trading_bot.ml.advanced.rl_environment import TradingEnvironment
from trading_bot.ml.advanced.rl_agent import DQNAgent, DDPGAgent

# Configure logging
logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Trainer for reinforcement learning trading agents.
    """
    
    def __init__(self, 
                env: TradingEnvironment,
                agent_type: str = "ddpg",
                checkpoint_dir: str = "checkpoints",
                log_dir: str = "logs",
                agent_params: Dict = None):
        """
        Initialize the RL trainer.
        
        Args:
            env: Trading environment instance
            agent_type: Type of agent to use ('dqn' or 'ddpg')
            checkpoint_dir: Directory to save model checkpoints
            log_dir: Directory to save training logs
            agent_params: Parameters for agent initialization
        """
        self.env = env
        self.agent_type = agent_type.lower()
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.agent_params = agent_params or {}
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize agent based on type
        self._initialize_agent()
        
        # Initialize training metrics
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_lengths = []
        self.evaluation_returns = []
        
        # TensorBoard summary writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(log_dir, f"{agent_type}_{current_time}")
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)
        
        logger.info(f"Initialized RL trainer with {agent_type} agent")
    
    def _initialize_agent(self):
        """
        Initialize the RL agent based on the specified type.
        """
        # Get state shape from environment
        state_shape = self.env.observation_space.shape
        
        if self.agent_type == "dqn":
            # For DQN, we need to discretize the action space
            n_actions = self.agent_params.get("n_actions", 5)  # Default: 5 discrete actions
            
            # Create agent
            self.agent = DQNAgent(
                state_shape=state_shape,
                n_actions=n_actions,
                learning_rate=self.agent_params.get("learning_rate", 0.001),
                gamma=self.agent_params.get("gamma", 0.99),
                epsilon_start=self.agent_params.get("epsilon_start", 1.0),
                epsilon_end=self.agent_params.get("epsilon_end", 0.01),
                epsilon_decay=self.agent_params.get("epsilon_decay", 0.995),
                batch_size=self.agent_params.get("batch_size", 64),
                target_update_freq=self.agent_params.get("target_update_freq", 1000),
                replay_buffer_size=self.agent_params.get("replay_buffer_size", 10000)
            )
            
        elif self.agent_type == "ddpg":
            # For DDPG, we use the continuous action space
            action_dim = self.env.action_space.shape[0]
            action_high = float(self.env.action_space.high[0])
            
            # Create agent
            self.agent = DDPGAgent(
                state_shape=state_shape,
                action_dim=action_dim,
                action_high=action_high,
                actor_learning_rate=self.agent_params.get("actor_learning_rate", 0.001),
                critic_learning_rate=self.agent_params.get("critic_learning_rate", 0.002),
                gamma=self.agent_params.get("gamma", 0.99),
                tau=self.agent_params.get("tau", 0.005),
                batch_size=self.agent_params.get("batch_size", 64),
                replay_buffer_size=self.agent_params.get("replay_buffer_size", 10000),
                exploration_noise=self.agent_params.get("exploration_noise", 0.1)
            )
            
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}. Must be 'dqn' or 'ddpg'.")
    
    def _convert_action_if_needed(self, agent_action) -> np.ndarray:
        """
        Convert the agent's action to the format expected by the environment.
        
        Args:
            agent_action: Action from the agent
            
        Returns:
            Action in the format expected by the environment
        """
        if self.agent_type == "dqn":
            # Convert discrete action to continuous
            # Map from [0, n_actions-1] to [-1, 1]
            n_actions = self.agent_params.get("n_actions", 5)
            action_value = 2 * (agent_action / (n_actions - 1)) - 1
            return np.array([action_value])
        else:
            # DDPG already outputs continuous actions
            return agent_action
    
    def train(self, 
            num_episodes: int = 1000, 
            max_steps_per_episode: int = None,
            eval_frequency: int = 10,
            early_stopping_patience: int = 20,
            checkpoint_frequency: int = 50,
            render: bool = False,
            verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the agent on the environment.
        
        Args:
            num_episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode (None to use env default)
            eval_frequency: Frequency of evaluation episodes
            early_stopping_patience: Patience for early stopping based on evaluation returns
            checkpoint_frequency: Frequency of model checkpoints
            render: Whether to render the environment during training
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting training for {num_episodes} episodes")
        
        best_eval_return = -np.inf
        patience_counter = 0
        
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            steps = 0
            done = False
            
            while not done:
                # Select action
                agent_action = self.agent.select_action(state, training=True)
                
                # Convert action if needed
                env_action = self._convert_action_if_needed(agent_action)
                
                # Take step in environment
                next_state, reward, done, info = self.env.step(env_action)
                
                # Store transition in replay buffer
                self.agent.replay_buffer.add(state, agent_action, reward, next_state, done)
                
                # Train agent
                train_metrics = self.agent.train()
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Render if requested
                if render:
                    self.env.render()
                
                # Check if max steps reached
                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    break
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(steps)
            
            # Get end of episode portfolio return
            portfolio_return = info.get('portfolio_value', 1.0) - 1.0
            self.episode_returns.append(portfolio_return)
            
            # Log to TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('reward', episode_reward, step=episode)
                tf.summary.scalar('return', portfolio_return, step=episode)
                tf.summary.scalar('episode_length', steps, step=episode)
                
                # Log training metrics
                for key, value in train_metrics.items():
                    tf.summary.scalar(key, value, step=episode)
            
            # Print progress
            if verbose and episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Return: {portfolio_return:.2%} | "
                      f"Length: {steps}")
            
            # Evaluate agent
            if episode % eval_frequency == 0:
                eval_return = self.evaluate(num_episodes=5, verbose=False)
                self.evaluation_returns.append(eval_return)
                
                # Log to TensorBoard
                with self.summary_writer.as_default():
                    tf.summary.scalar('eval_return', eval_return, step=episode)
                
                if verbose:
                    print(f"Evaluation at episode {episode}: {eval_return:.2%}")
                
                # Check for early stopping
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    patience_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model"))
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {episode} episodes")
                        break
            
            # Save checkpoint
            if episode % checkpoint_frequency == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f"model_ep{episode}"))
        
        # Final evaluation
        final_eval_return = self.evaluate(num_episodes=10, verbose=True)
        
        logger.info(f"Training completed. Final evaluation return: {final_eval_return:.2%}")
        
        # Return training metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'evaluation_returns': self.evaluation_returns
        }
        
        return metrics
    
    def evaluate(self, 
                num_episodes: int = 5, 
                render: bool = False,
                verbose: bool = True) -> float:
        """
        Evaluate the agent on the environment.
        
        Args:
            num_episodes: Number of episodes to evaluate for
            render: Whether to render the environment during evaluation
            verbose: Whether to print progress information
            
        Returns:
            Average portfolio return over evaluation episodes
        """
        returns = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # Select action without exploration
                agent_action = self.agent.select_action(state, training=False)
                env_action = self._convert_action_if_needed(agent_action)
                
                # Take step in environment
                next_state, _, done, info = self.env.step(env_action)
                
                # Update state
                state = next_state
                
                # Render if requested
                if render:
                    self.env.render()
            
            # Get portfolio return
            portfolio_return = info.get('portfolio_value', 1.0) - 1.0
            returns.append(portfolio_return)
            
            if verbose:
                print(f"Evaluation episode {episode+1}/{num_episodes}: {portfolio_return:.2%}")
        
        # Calculate average return
        avg_return = np.mean(returns)
        
        if verbose:
            print(f"Average evaluation return: {avg_return:.2%}")
        
        return avg_return
    
    def save_checkpoint(self, filepath: str):
        """
        Save a checkpoint of the agent.
        
        Args:
            filepath: Path to save the checkpoint
        """
        # Save agent model
        self.agent.save(filepath)
        
        # Save training metrics
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_returns': self.episode_returns,
            'episode_lengths': self.episode_lengths,
            'evaluation_returns': self.evaluation_returns
        }
        
        with open(filepath + "_metrics.json", 'w') as f:
            json.dump({
                k: [float(v) for v in vals] for k, vals in metrics.items()
            }, f)
        
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load a checkpoint of the agent.
        
        Args:
            filepath: Path to load the checkpoint from
        """
        # Load agent model
        self.agent.load(filepath)
        
        # Load training metrics if available
        metrics_path = filepath + "_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                
                self.episode_rewards = metrics.get('episode_rewards', [])
                self.episode_returns = metrics.get('episode_returns', [])
                self.episode_lengths = metrics.get('episode_lengths', [])
                self.evaluation_returns = metrics.get('evaluation_returns', [])
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def plot_training_metrics(self, figsize: Tuple[int, int] = (15, 10), 
                           smooth_window: int = 10) -> plt.Figure:
        """
        Plot training metrics.
        
        Args:
            figsize: Figure size
            smooth_window: Window size for smoothing
            
        Returns:
            Matplotlib Figure object
        """
        def smooth(y, window):
            """Apply simple moving average smoothing."""
            box = np.ones(window) / window
            y_smooth = np.convolve(y, box, mode='valid')
            return y_smooth
        
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Plot episode rewards
        ax = axs[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) > smooth_window:
            ax.plot(
                range(smooth_window-1, len(self.episode_rewards)),
                smooth(self.episode_rewards, smooth_window),
                label=f'Smoothed (window={smooth_window})'
            )
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot episode returns
        ax = axs[0, 1]
        ax.plot(self.episode_returns, alpha=0.3, label='Raw')
        if len(self.episode_returns) > smooth_window:
            ax.plot(
                range(smooth_window-1, len(self.episode_returns)),
                smooth(self.episode_returns, smooth_window),
                label=f'Smoothed (window={smooth_window})'
            )
        ax.set_title('Episode Returns')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Portfolio Return')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot episode lengths
        ax = axs[1, 0]
        ax.plot(self.episode_lengths)
        ax.set_title('Episode Lengths')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.grid(True, alpha=0.3)
        
        # Plot evaluation returns
        ax = axs[1, 1]
        if self.evaluation_returns:
            eval_episodes = range(0, len(self.episode_returns), len(self.episode_returns) // len(self.evaluation_returns))
            eval_episodes = list(eval_episodes)[:len(self.evaluation_returns)]
            
            ax.plot(eval_episodes, self.evaluation_returns, 'o-')
            ax.set_title('Evaluation Returns')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Return')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Reinforcement Learning Strategy Wrapper
class RLTradingStrategy:
    """
    Wrapper for using trained RL agents as trading strategies in the BensBot framework.
    """
    
    def __init__(self, 
                model_path: str,
                agent_type: str = "ddpg",
                lookback_window: int = 30,
                feature_columns: List[str] = None,
                price_column: str = "close",
                commission: float = 0.001):
        """
        Initialize the RL trading strategy.
        
        Args:
            model_path: Path to the trained model
            agent_type: Type of agent ('dqn' or 'ddpg')
            lookback_window: Number of time steps to include in state
            feature_columns: List of feature column names to use
            price_column: Name of the price column
            commission: Trading commission rate
        """
        self.model_path = model_path
        self.agent_type = agent_type.lower()
        self.lookback_window = lookback_window
        self.feature_columns = feature_columns or ["close", "volume", "open", "high", "low"]
        self.price_column = price_column
        self.commission = commission
        
        # Add price column to features if not already included
        if self.price_column not in self.feature_columns:
            self.feature_columns.append(self.price_column)
        
        # Will be set when loading data
        self.feature_means = None
        self.feature_stds = None
        self.current_position = 0.0
        self.data_buffer = []
        
        # Initialize agent
        self._initialize_agent()
        
        logger.info(f"Initialized RL trading strategy using {agent_type} model from {model_path}")
    
    def _initialize_agent(self):
        """
        Initialize the RL agent.
        """
        # We need to determine the state shape
        # It will be (lookback_window, len(feature_columns) + 1)
        state_shape = (self.lookback_window, len(self.feature_columns) + 1)
        
        if self.agent_type == "dqn":
            # For DQN, we need to know the number of discrete actions
            # We'll assume it's 5 by default but should be loaded from the model
            self.agent = DQNAgent(
                state_shape=state_shape,
                n_actions=5  # This will be overwritten when model is loaded
            )
        elif self.agent_type == "ddpg":
            # For DDPG, we use continuous actions
            self.agent = DDPGAgent(
                state_shape=state_shape,
                action_dim=1,
                action_high=1.0
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}. Must be 'dqn' or 'ddpg'.")
        
        # Load the trained model
        self.agent.load(self.model_path)
    
    def _convert_action_if_needed(self, agent_action) -> float:
        """
        Convert the agent's action to a position size (-1 to 1).
        
        Args:
            agent_action: Action from the agent
            
        Returns:
            Position size (-1 to 1)
        """
        if self.agent_type == "dqn":
            # Convert discrete action to continuous
            # Map from [0, n_actions-1] to [-1, 1]
            n_actions = 5  # Should match what was used during training
            position_size = 2 * (agent_action / (n_actions - 1)) - 1
            return float(position_size)
        else:
            # DDPG already outputs continuous actions
            return float(agent_action[0])
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using stored means and stds,
        or calculate them if not available.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Normalized features DataFrame
        """
        if self.feature_means is None or self.feature_stds is None:
            # Calculate means and stds
            self.feature_means = features_df.mean()
            self.feature_stds = features_df.std()
        
        # Normalize
        normalized = (features_df - self.feature_means) / (self.feature_stds + 1e-8)
        return normalized
    
    def _prepare_state(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare the state representation from features.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            State representation (observation)
        """
        # Ensure we have the right columns
        features_df = features_df[self.feature_columns].copy()
        
        # Normalize features
        normalized = self._normalize_features(features_df)
        
        # Get last lookback_window rows
        window_data = normalized.iloc[-self.lookback_window:].values
        
        # If we don't have enough history, pad with zeros
        if len(window_data) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(window_data), len(self.feature_columns)))
            window_data = np.vstack([padding, window_data])
        
        # Add position information
        position_col = np.ones((self.lookback_window, 1)) * self.current_position
        
        # Combine features and position
        state = np.hstack([window_data, position_col])
        
        return state
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a trading signal based on the features.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Dictionary with prediction details including position size
        """
        # Prepare state
        state = self._prepare_state(features_df)
        
        # Get action from agent (without exploration)
        agent_action = self.agent.select_action(state, training=False)
        
        # Convert to position size if needed
        position_size = self._convert_action_if_needed(agent_action)
        
        # Calculate trade amount
        old_position = self.current_position
        trade_amount = position_size - old_position
        
        # Update current position
        self.current_position = position_size
        
        # Get current price
        current_price = features_df[self.price_column].iloc[-1]
        
        # Calculate trade cost
        trade_value = abs(trade_amount) * current_price
        trade_cost = trade_value * self.commission
        
        # Prepare prediction result
        prediction = {
            'position_size': position_size,
            'trade_amount': trade_amount,
            'trade_value': trade_value,
            'trade_cost': trade_cost,
            'signal': 'buy' if trade_amount > 0 else 'sell' if trade_amount < 0 else 'hold',
            'confidence': abs(position_size),  # Use position size as confidence
            'price': current_price
        }
        
        return prediction
    
    def update(self, features_df: pd.DataFrame, trade_result: Dict[str, Any] = None):
        """
        Update the strategy with new data and trade results.
        
        Args:
            features_df: DataFrame with feature columns
            trade_result: Result of executing the trade (optional)
        """
        # Store features in buffer for future reference
        self.data_buffer.append(features_df.iloc[-1])
        
        # Limit buffer size
        if len(self.data_buffer) > self.lookback_window * 2:
            self.data_buffer.pop(0)
        
        # If we have trade results, we could potentially use them to adapt the strategy
        # This would require online learning, which is not implemented here
        pass
