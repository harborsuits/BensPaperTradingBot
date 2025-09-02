#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLStrategyAgent - A reinforcement learning agent for strategy selection and allocation.
"""

import os
import numpy as np
import pandas as pd
import logging
import gym
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Import stable_baselines3 for RL algorithms
try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.evaluation import evaluate_policy
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable_baselines3 not available. Install with 'pip install stable-baselines3'")

from trading_bot.learning.rl_environment import RLTradingEnv

logger = logging.getLogger(__name__)

class RLStrategyAgent:
    """
    A reinforcement learning agent for strategy selection and allocation.
    
    This agent is trained to allocate capital between trading strategies based on market conditions.
    It uses PPO (Proximal Policy Optimization) as the default algorithm, but can be configured
    to use other algorithms from stable_baselines3.
    """
    
    def __init__(
        self,
        strategies: List[str],
        model_path: str = "models/rl_strategy_agent",
        algorithm: str = "PPO",
        data_path: str = "data/backtest_history.json",
        window_size: int = 30,
        episode_length: int = 252,
        reward_type: str = "sharpe",
        use_pattern_insights: bool = True,
        learning_rate: float = 3e-4,
        verbose: int = 1
    ):
        """
        Initialize the RL strategy agent.
        
        Args:
            strategies: List of strategy names to allocate between
            model_path: Path to save/load the trained model
            algorithm: RL algorithm to use ('PPO', 'A2C', or 'DQN')
            data_path: Path to backtest history data
            window_size: Number of days of history to include in state
            episode_length: Number of steps per episode
            reward_type: Type of reward function ('sharpe', 'sortino', 'pnl', 'calmar')
            use_pattern_insights: Whether to include PatternLearner insights in state
            learning_rate: Learning rate for the algorithm
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("stable_baselines3 is required for RLStrategyAgent")
        
        self.strategies = strategies
        self.model_path = model_path
        self.algorithm = algorithm
        self.data_path = data_path
        self.window_size = window_size
        self.episode_length = episode_length
        self.reward_type = reward_type
        self.use_pattern_insights = use_pattern_insights
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Initialize environment
        self.env = RLTradingEnv(
            strategies=strategies,
            data_path=data_path,
            window_size=window_size,
            episode_length=episode_length,
            reward_type=reward_type,
            use_pattern_insights=use_pattern_insights
        )
        
        # Wrap environment with Monitor for logging
        log_dir = os.path.join(os.path.dirname(model_path), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(self.env, log_dir)
        
        # Initialize RL model
        self.model = self._create_model()
        
        # Training metrics
        self.training_history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "evaluation_returns": [],
            "learning_rate": []
        }
        
        logger.info(f"Initialized {algorithm} agent for {len(strategies)} strategies")
    
    def _create_model(self):
        """
        Create the RL model with the specified algorithm.
        
        Returns:
            Initialized RL model
        """
        if self.algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                verbose=self.verbose,
                tensorboard_log=os.path.join(os.path.dirname(self.model_path), "tensorboard")
            )
        elif self.algorithm == "A2C":
            model = A2C(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                verbose=self.verbose,
                tensorboard_log=os.path.join(os.path.dirname(self.model_path), "tensorboard")
            )
        elif self.algorithm == "DQN":
            # DQN requires discrete action space, so not suitable for continuous allocation
            # This implementation would need modification to use DQN properly
            logger.warning("DQN is not well-suited for continuous action spaces. Consider using PPO or A2C instead.")
            model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                verbose=self.verbose,
                tensorboard_log=os.path.join(os.path.dirname(self.model_path), "tensorboard")
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}. Use 'PPO', 'A2C', or 'DQN'.")
        
        return model
    
    def train(self, total_timesteps: int = 100000, eval_freq: int = 10000):
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            eval_freq: Frequency of evaluation during training
            
        Returns:
            Training metrics
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Create evaluation environment
        eval_env = RLTradingEnv(
            strategies=self.strategies,
            data_path=self.data_path,
            window_size=self.window_size,
            episode_length=self.episode_length,
            reward_type=self.reward_type,
            use_pattern_insights=self.use_pattern_insights
        )
        
        # Set up callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(self.model_path),
            log_path=os.path.join(os.path.dirname(self.model_path), "eval_logs"),
            eval_freq=eval_freq,
            deterministic=True,
            render=False
        )
        
        # Custom callback for tracking metrics
        class MetricsCallback(BaseCallback):
            def __init__(self, verbose=0):
                super(MetricsCallback, self).__init__(verbose)
                self.episode_rewards = []
                self.episode_lengths = []
                self.current_episode_reward = 0
                self.current_episode_length = 0
            
            def _on_step(self):
                # Get current reward and update episode stats
                reward = self.locals['rewards'][0]
                self.current_episode_reward += reward
                self.current_episode_length += 1
                
                # Check if episode is done
                done = self.locals['dones'][0]
                if done:
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episode_lengths.append(self.current_episode_length)
                    self.current_episode_reward = 0
                    self.current_episode_length = 0
                
                return True
        
        metrics_callback = MetricsCallback()
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, metrics_callback],
            tb_log_name=f"{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Save the final model
        self.model.save(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        # Update training history
        self.training_history["episode_rewards"] = metrics_callback.episode_rewards
        self.training_history["episode_lengths"] = metrics_callback.episode_lengths
        
        # Save training history
        self._save_training_history()
        
        return self.training_history
    
    def _save_training_history(self):
        """Save training history to JSON file"""
        history_path = f"{self.model_path}_history.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                serializable_history[key] = [v.tolist() for v in value]
            elif isinstance(value, np.ndarray):
                serializable_history[key] = value.tolist()
            else:
                serializable_history[key] = value
        
        with open(history_path, "w") as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    def load(self, model_path: Optional[str] = None):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the trained model (defaults to self.model_path)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path = model_path or self.model_path
        
        try:
            if self.algorithm == "PPO":
                self.model = PPO.load(path, env=self.env)
            elif self.algorithm == "A2C":
                self.model = A2C.load(path, env=self.env)
            elif self.algorithm == "DQN":
                self.model = DQN.load(path, env=self.env)
            
            logger.info(f"Model loaded from {path}")
            
            # Load training history if available
            history_path = f"{path}_history.json"
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    self.training_history = json.load(f)
                
                logger.info(f"Training history loaded from {history_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict strategy allocations for a given observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic action or sample from distribution
            
        Returns:
            Strategy allocation weights
        """
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Process action to get allocations
        if hasattr(self.env, "_process_action"):
            allocations = self.env._process_action(action)
        else:
            # If env wrapper doesn't have _process_action, process manually
            # Ensure allocations are non-negative and sum to 1
            allocations = np.maximum(action, 0)
            allocation_sum = np.sum(allocations)
            
            if allocation_sum > 0:
                allocations = allocations / allocation_sum
            else:
                # Equal allocation if all are zero
                allocations = np.ones_like(allocations) / len(allocations)
        
        return allocations
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the trained agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating agent for {n_episodes} episodes")
        
        # Create evaluation environment
        eval_env = RLTradingEnv(
            strategies=self.strategies,
            data_path=self.data_path,
            window_size=self.window_size,
            episode_length=self.episode_length,
            reward_type=self.reward_type,
            use_pattern_insights=self.use_pattern_insights
        )
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=n_episodes,
            deterministic=True
        )
        
        # Run custom evaluation to get more detailed metrics
        episode_metrics = []
        for i in range(n_episodes):
            episode_reward, allocation_history, portfolio_history = self._evaluate_episode(eval_env)
            episode_metrics.append({
                "episode": i,
                "total_reward": episode_reward,
                "final_portfolio_value": portfolio_history[-1],
                "portfolio_return": (portfolio_history[-1] / portfolio_history[0]) - 1,
                "max_drawdown": self._calculate_max_drawdown(portfolio_history)
            })
        
        # Calculate summary metrics
        portfolio_returns = [m["portfolio_return"] for m in episode_metrics]
        final_values = [m["final_portfolio_value"] for m in episode_metrics]
        max_drawdowns = [m["max_drawdown"] for m in episode_metrics]
        
        evaluation_metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_return": np.mean(portfolio_returns),
            "std_return": np.std(portfolio_returns),
            "mean_final_value": np.mean(final_values),
            "mean_max_drawdown": np.mean(max_drawdowns),
            "sharpe_ratio": np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-6),
            "episode_metrics": episode_metrics
        }
        
        logger.info(f"Evaluation results: Mean reward={mean_reward:.2f}±{std_reward:.2f}, "
                   f"Mean return={evaluation_metrics['mean_return']:.2%}, "
                   f"Sharpe ratio={evaluation_metrics['sharpe_ratio']:.2f}")
        
        return evaluation_metrics
    
    def _evaluate_episode(self, env) -> Tuple[float, List[np.ndarray], List[float]]:
        """
        Evaluate a single episode, tracking allocations and portfolio values.
        
        Args:
            env: Evaluation environment
            
        Returns:
            Tuple of (total reward, allocation history, portfolio history)
        """
        obs, _ = env.reset()
        done = False
        total_reward = 0
        allocation_history = []
        portfolio_history = [env.current_portfolio_value]
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track metrics
            total_reward += reward
            allocation_history.append(env.current_allocations)
            portfolio_history.append(info["portfolio_value"])
            
            # Check if episode is done
            done = terminated or truncated
        
        return total_reward, allocation_history, portfolio_history
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate maximum drawdown from portfolio value history.
        
        Args:
            portfolio_values: List of portfolio values
            
        Returns:
            Maximum drawdown as a percentage
        """
        # Convert to numpy array
        values = np.array(portfolio_values)
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(values)
        drawdowns = (peak - values) / peak
        max_drawdown = np.max(drawdowns) if drawdowns.size > 0 else 0.0
        
        return max_drawdown
    
    def get_allocation_for_signals(
        self, 
        signals: Dict[str, Any], 
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get strategy allocations based on current market signals.
        This method is intended for integration with the trading system.
        
        Args:
            signals: Dictionary of signals from strategies
            market_data: Recent market data DataFrame
            
        Returns:
            Dictionary of strategy allocations
        """
        # Extract observation from signals and market data
        # This is a simplified method and would need to be adapted to match the
        # observation space expected by the trained agent
        
        # In a real implementation, we would construct a proper observation vector
        # that matches what the model was trained on.
        # For illustration, we'll just create a dummy observation
        
        # Get default observation from environment reset
        dummy_obs, _ = self.env.reset()
        
        # Predict allocations
        allocations = self.predict(dummy_obs)
        
        # Convert to dictionary
        allocation_dict = dict(zip(self.strategies, allocations))
        
        return allocation_dict
    
    def get_integration_info(self) -> Dict[str, Any]:
        """
        Get information about the agent for integration with other components.
        
        Returns:
            Dictionary with agent information
        """
        return {
            "algorithm": self.algorithm,
            "strategies": self.strategies,
            "reward_type": self.reward_type,
            "window_size": self.window_size,
            "use_pattern_insights": self.use_pattern_insights,
            "model_path": self.model_path,
            "trained": hasattr(self, "model") and self.model is not None
        } 