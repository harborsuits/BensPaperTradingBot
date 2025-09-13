#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Trainer Module for Reinforcement Learning

This module provides functions for training, evaluating, and using
reinforcement learning agents for portfolio optimization.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
from pathlib import Path

# Import Stable-Baselines3 components
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Import our custom environment
from trading_bot.rl.trading_env import TradingEnv


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for tracking and saving training progress.
    """
    
    def __init__(self, 
                 eval_env: TradingEnv, 
                 eval_freq: int = 10000,
                 log_dir: str = "./logs",
                 verbose: int = 1):
        """
        Initialize the callback.
        
        Args:
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency (in timesteps)
            log_dir: Directory for saving logs
            verbose: Verbosity level
        """
        super(TrainingProgressCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.best_reward = -np.inf
        self.metrics_history = {
            'timesteps': [],
            'mean_reward': [],
            'portfolio_value': [],
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'max_drawdown': [],
            'volatility': [],
            'trades': []
        }
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        """Called at each step during training."""
        if self.n_calls % self.eval_freq == 0:
            # Evaluate policy
            rewards, portfolio_values = self._evaluate_policy()
            mean_reward = np.mean(rewards)
            
            # Get episode stats
            stats = self.eval_env.get_episode_stats()
            
            # Update history
            self.metrics_history['timesteps'].append(self.num_timesteps)
            self.metrics_history['mean_reward'].append(mean_reward)
            self.metrics_history['portfolio_value'].append(stats.get('final_value', 0))
            self.metrics_history['sharpe_ratio'].append(stats.get('sharpe_ratio', 0))
            self.metrics_history['sortino_ratio'].append(stats.get('sortino_ratio', 0))
            self.metrics_history['max_drawdown'].append(stats.get('max_drawdown', 0))
            self.metrics_history['volatility'].append(stats.get('volatility', 0))
            self.metrics_history['trades'].append(stats.get('total_trades', 0))
            
            # Log metrics
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/portfolio_value', stats.get('final_value', 0))
            self.logger.record('eval/sharpe_ratio', stats.get('sharpe_ratio', 0))
            self.logger.record('eval/sortino_ratio', stats.get('sortino_ratio', 0))
            self.logger.record('eval/max_drawdown', stats.get('max_drawdown', 0))
            self.logger.record('eval/volatility', stats.get('volatility', 0))
            self.logger.record('eval/trades', stats.get('total_trades', 0))
            
            # Save best model
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(os.path.join(self.log_dir, 'best_model'))
                
                # Save performance chart
                self._save_performance_chart(portfolio_values)
                
            # Save history to JSON
            with open(os.path.join(self.log_dir, 'metrics_history.json'), 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        
        return True
    
    def _evaluate_policy(self) -> Tuple[List[float], List[float]]:
        """
        Evaluate the current policy.
        
        Returns:
            Tuple of (reward_list, portfolio_value_list)
        """
        # Reset the environment
        obs, _ = self.eval_env.reset()
        
        # Lists to store rewards and portfolio values
        rewards = []
        portfolio_values = []
        
        # Run until done
        done = False
        while not done:
            # Get action from policy
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            done = terminated or truncated
            
            # Record reward and portfolio value
            rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
        
        return rewards, portfolio_values
    
    def _save_performance_chart(self, portfolio_values: List[float]) -> None:
        """
        Save a chart of portfolio performance.
        
        Args:
            portfolio_values: List of portfolio values
        """
        plt.figure(figsize=(12, 6))
        plt.plot(portfolio_values)
        plt.title('Portfolio Value Over Time (Best Model)')
        plt.xlabel('Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'portfolio_performance.png'))
        plt.close()


class AgentTrainer:
    """
    Trainer for reinforcement learning agents for portfolio optimization.
    """
    
    def __init__(self, 
                 train_env: TradingEnv,
                 eval_env: Optional[TradingEnv] = None,
                 agent_type: str = 'ppo',
                 model_params: Optional[Dict[str, Any]] = None,
                 output_dir: str = './rl_models'):
        """
        Initialize the agent trainer.
        
        Args:
            train_env: Training environment
            eval_env: Evaluation environment (if None, a copy of train_env will be used)
            agent_type: Type of RL agent ('ppo', 'a2c', 'sac', 'ddpg', 'td3')
            model_params: Parameters for the RL agent
            output_dir: Directory for saving models and logs
        """
        self.train_env = train_env
        self.eval_env = eval_env or train_env
        self.agent_type = agent_type.lower()
        self.model_params = model_params or {}
        self.output_dir = output_dir
        
        # Create output directories
        self.model_dir = os.path.join(output_dir, 'models')
        self.log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up the agent
        self.model = self._create_agent()
        self.trained = False
        
        # Generate a unique run ID
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _create_agent(self) -> Any:
        """
        Create a reinforcement learning agent.
        
        Returns:
            RL agent model
        """
        # Create and configure the agent based on type
        if self.agent_type == 'ppo':
            model = PPO('MlpPolicy', self.train_env, verbose=1, **self.model_params)
        elif self.agent_type == 'a2c':
            model = A2C('MlpPolicy', self.train_env, verbose=1, **self.model_params)
        elif self.agent_type == 'sac':
            model = SAC('MlpPolicy', self.train_env, verbose=1, **self.model_params)
        elif self.agent_type == 'ddpg':
            model = DDPG('MlpPolicy', self.train_env, verbose=1, **self.model_params)
        elif self.agent_type == 'td3':
            model = TD3('MlpPolicy', self.train_env, verbose=1, **self.model_params)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
        
        return model
    
    def train(self, 
              total_timesteps: int = 1000000,
              eval_freq: int = 10000,
              save_freq: int = 50000) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            eval_freq: Frequency of evaluation (in timesteps)
            save_freq: Frequency of saving checkpoints (in timesteps)
            
        Returns:
            Dictionary with training metrics
        """
        print(f"Starting training for {total_timesteps} timesteps")
        
        # Set up callbacks
        progress_callback = TrainingProgressCallback(
            eval_env=self.eval_env,
            eval_freq=eval_freq,
            log_dir=os.path.join(self.log_dir, self.run_id),
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=os.path.join(self.model_dir, self.run_id),
            name_prefix="agent",
            verbose=1
        )
        
        # Configure logger
        logger = configure(os.path.join(self.log_dir, self.run_id), ['stdout', 'csv', 'tensorboard'])
        self.model.set_logger(logger)
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[progress_callback, checkpoint_callback]
        )
        
        # Save the final model
        final_model_path = os.path.join(self.model_dir, self.run_id, 'final_model')
        self.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save environment configuration for reproducibility
        env_config = {
            'initial_balance': self.train_env.initial_balance,
            'trading_cost': self.train_env.trading_cost,
            'slippage': self.train_env.slippage,
            'window_size': self.train_env.window_size,
            'reward_type': self.train_env.reward_type,
            'reward_scale': self.train_env.reward_scale,
            'allow_short': self.train_env.allow_short,
            'assets': self.train_env.asset_list,
            'features': self.train_env.features_list
        }
        
        with open(os.path.join(self.model_dir, self.run_id, 'env_config.json'), 'w') as f:
            json.dump(env_config, f, indent=2)
        
        # Load and return training metrics
        try:
            with open(os.path.join(self.log_dir, self.run_id, 'metrics_history.json'), 'r') as f:
                metrics = json.load(f)
            return metrics
        except FileNotFoundError:
            return {}
    
    def evaluate(self, 
                model_path: Optional[str] = None, 
                num_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate a trained agent.
        
        Args:
            model_path: Path to the model to evaluate (if None, use the current model)
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Load model if path specified
        if model_path:
            if self.agent_type == 'ppo':
                model = PPO.load(model_path, env=self.eval_env)
            elif self.agent_type == 'a2c':
                model = A2C.load(model_path, env=self.eval_env)
            elif self.agent_type == 'sac':
                model = SAC.load(model_path, env=self.eval_env)
            elif self.agent_type == 'ddpg':
                model = DDPG.load(model_path, env=self.eval_env)
            elif self.agent_type == 'td3':
                model = TD3.load(model_path, env=self.eval_env)
        else:
            model = self.model
            
        # Lists to store episode results
        all_portfolio_values = []
        all_returns = []
        all_drawdowns = []
        all_trades = []
        all_sharpes = []
        all_sortinos = []
        
        # Evaluate for multiple episodes
        for episode in range(num_episodes):
            print(f"Evaluating episode {episode+1}/{num_episodes}")
            
            # Reset environment
            obs, _ = self.eval_env.reset()
            
            # Run until done
            done = False
            while not done:
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            
            # Get episode stats
            stats = self.eval_env.get_episode_stats()
            portfolio_history = self.eval_env.get_portfolio_history()
            
            # Record results
            all_portfolio_values.append(stats['final_value'])
            all_returns.append(stats['total_return'])
            all_drawdowns.append(stats['max_drawdown'])
            all_trades.append(stats['total_trades'])
            all_sharpes.append(stats['sharpe_ratio'])
            all_sortinos.append(stats['sortino_ratio'])
            
            # Save portfolio history for this episode
            if not portfolio_history.empty:
                portfolio_history.to_csv(
                    os.path.join(self.log_dir, f'eval_episode_{episode+1}_history.csv')
                )
                
                # Plot and save portfolio value
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_history['portfolio_value'])
                plt.title(f'Portfolio Value (Episode {episode+1})')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True)
                plt.savefig(os.path.join(self.log_dir, f'eval_episode_{episode+1}_portfolio.png'))
                plt.close()
                
                # Plot and save drawdown
                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_history['drawdown'], color='red')
                plt.title(f'Portfolio Drawdown (Episode {episode+1})')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                plt.savefig(os.path.join(self.log_dir, f'eval_episode_{episode+1}_drawdown.png'))
                plt.close()
        
        # Calculate aggregate metrics
        eval_metrics = {
            'mean_final_value': np.mean(all_portfolio_values),
            'mean_return': np.mean(all_returns),
            'mean_drawdown': np.mean(all_drawdowns),
            'mean_trades': np.mean(all_trades),
            'mean_sharpe': np.mean(all_sharpes),
            'mean_sortino': np.mean(all_sortinos),
            'std_return': np.std(all_returns),
            'max_return': np.max(all_returns),
            'min_return': np.min(all_returns),
            'max_drawdown': np.min(all_drawdowns),  # Drawdowns are negative
            'episodes': num_episodes
        }
        
        # Save evaluation metrics
        eval_dir = os.path.join(self.log_dir, 'evaluation')
        os.makedirs(eval_dir, exist_ok=True)
        
        with open(os.path.join(eval_dir, f'eval_metrics_{self.run_id}.json'), 'w') as f:
            json.dump(eval_metrics, f, indent=2)
            
        print(f"Evaluation complete. Mean return: {eval_metrics['mean_return']:.2%}")
        
        return eval_metrics
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained agent.
        
        Args:
            filepath: Path to save the model (if None, use default path)
            
        Returns:
            Path where the model was saved
        """
        if filepath is None:
            os.makedirs(self.model_dir, exist_ok=True)
            filepath = os.path.join(self.model_dir, f'{self.agent_type}_{self.run_id}')
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def load(self, filepath: str) -> None:
        """
        Load a trained agent.
        
        Args:
            filepath: Path to the saved model
        """
        if self.agent_type == 'ppo':
            self.model = PPO.load(filepath, env=self.train_env)
        elif self.agent_type == 'a2c':
            self.model = A2C.load(filepath, env=self.train_env)
        elif self.agent_type == 'sac':
            self.model = SAC.load(filepath, env=self.train_env)
        elif self.agent_type == 'ddpg':
            self.model = DDPG.load(filepath, env=self.train_env)
        elif self.agent_type == 'td3':
            self.model = TD3.load(filepath, env=self.train_env)
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
            
        print(f"Model loaded from {filepath}")
        self.trained = True
    
    def predict(self, 
               observation: np.ndarray, 
               deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make a prediction with the trained agent.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (action, state)
        """
        if not self.trained and not hasattr(self.model, 'policy'):
            raise ValueError("Agent not trained. Call train() or load() first.")
            
        return self.model.predict(observation, deterministic=deterministic)
    
    def get_optimized_portfolio(self, 
                               market_data: Dict[str, pd.DataFrame],
                               features_list: List[str],
                               current_portfolio: Optional[Dict[str, float]] = None,
                               cash: float = 0.0) -> Dict[str, float]:
        """
        Get optimized portfolio allocations for the current market state.
        
        Args:
            market_data: Dictionary of market data for each asset
            features_list: List of feature columns to use
            current_portfolio: Current portfolio positions (if None, assume all cash)
            cash: Current cash amount
            
        Returns:
            Dictionary with optimized portfolio allocations
        """
        if not self.trained and not hasattr(self.model, 'policy'):
            raise ValueError("Agent not trained. Call train() or load() first.")
        
        # Create a temporary environment for prediction
        env = TradingEnv(
            df_dict=market_data,
            features_list=features_list,
            initial_balance=100000.0 if cash <= 0 else cash,
            trading_cost=self.train_env.trading_cost,
            slippage=self.train_env.slippage,
            window_size=self.train_env.window_size,
            max_steps=1,
            reward_type=self.train_env.reward_type,
            reward_scale=self.train_env.reward_scale,
            allow_short=self.train_env.allow_short
        )
        
        # Reset environment
        obs, _ = env.reset()
        
        # If we have a current portfolio, set it in the environment
        if current_portfolio:
            # Get prices for calculating values
            current_prices = {}
            for symbol in env.asset_list:
                if symbol in market_data:
                    df = market_data[symbol]
                    if not df.empty:
                        current_prices[symbol] = df.iloc[-1]['close']
                    else:
                        current_prices[symbol] = 0.0
                else:
                    current_prices[symbol] = 0.0
            
            # Set portfolio positions
            for symbol, shares in current_portfolio.items():
                if symbol in env.portfolio:
                    env.portfolio[symbol] = shares
            
            # Recalculate portfolio value
            portfolio_value = cash
            for symbol, shares in env.portfolio.items():
                if symbol in current_prices:
                    portfolio_value += shares * current_prices[symbol]
            
            env.portfolio_value = portfolio_value
            env.cash = cash
            
            # Update observation based on current portfolio
            obs = env._get_observation()
        
        # Get action from policy
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Ensure action is properly normalized
        if not env.allow_short:
            action = np.maximum(action, 0.0)
        
        # Normalize to sum to 1.0
        action_sum = np.sum(np.abs(action))
        if action_sum > 0:
            action = action / action_sum
            
        # Convert action to portfolio allocations
        allocation = {}
        for i, symbol in enumerate(env.asset_list):
            allocation[symbol] = float(action[i])
            
        # Calculate cash allocation (not directly controlled by agent)
        total_asset_allocation = sum(abs(alloc) for alloc in allocation.values())
        cash_allocation = max(0.0, 1.0 - total_asset_allocation)
        allocation['cash'] = cash_allocation
        
        return allocation 