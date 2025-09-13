#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO-based Position Sizer Agent

This module implements a reinforcement learning agent for dynamic position sizing
based on trade signal confidence, market regime, and account metrics.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime

# Import RL libraries
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Import our environment
from trading_bot.rl.position_sizer_env import PositionSizerEnv

logger = logging.getLogger(__name__)

class PositionSizerAgent:
    """RL agent for dynamic position sizing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the position sizer agent
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or {}
        self._set_default_config()
        
        self.model = None
        self.vec_env = None
        self.trained = False
        
        logger.info("Position Sizer Agent initialized")
    
    def _set_default_config(self):
        """Set default configuration parameters"""
        # Model parameters
        self.config.setdefault("model_type", "PPO")  # Only PPO supported for now
        self.config.setdefault("policy_type", "MlpPolicy")
        
        # PPO hyperparameters
        self.config.setdefault("ppo_params", {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.01,  # Encourage exploration
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "sde_sample_freq": -1,
            "target_kl": None,
            "verbose": 1
        })
        
        # Position sizing constraints
        self.config.setdefault("max_position_size", 0.25)  # Max allocation per position
        self.config.setdefault("risk_per_trade", 0.01)     # Default risk per trade (1%)
        
        # Training parameters
        self.config.setdefault("total_timesteps", 1_000_000)
        self.config.setdefault("eval_freq", 10000)
        self.config.setdefault("save_freq", 50000)
        
        # Model persistence
        self.config.setdefault("model_dir", "models/position_sizer")
    
    def setup_environment(self, 
                         env: Optional[PositionSizerEnv] = None,
                         train_data: Optional[Dict[str, pd.DataFrame]] = None,
                         eval_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Set up environment for training and evaluation
        
        Args:
            env: Pre-configured environment (optional)
            train_data: Training data (required if env not provided)
            eval_data: Evaluation data (optional)
        """
        if env is not None:
            # Use provided environment
            self.env = env
        elif train_data is not None:
            # Create environment from training data
            self.env = PositionSizerEnv(train_data, self.config)
        else:
            raise ValueError("Either env or train_data must be provided")
        
        # Create vectorized environment for training
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Apply normalization for stable training
        self.vec_env = VecNormalize(
            self.vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=self.config["ppo_params"]["gamma"]
        )
        
        # Create evaluation environment if eval_data provided
        if eval_data is not None:
            self.eval_env = PositionSizerEnv(eval_data, self.config)
        else:
            self.eval_env = None
        
        logger.info("Environment setup complete")
    
    def build_model(self):
        """Build the PPO model for position sizing"""
        if self.vec_env is None:
            raise ValueError("Environment not set up. Call setup_environment() first.")
        
        # Create PPO model
        self.model = PPO(
            policy=self.config["policy_type"],
            env=self.vec_env,
            **self.config["ppo_params"]
        )
        
        logger.info(f"Built PPO model for position sizing")
    
    def train(self, callback: Optional[BaseCallback] = None) -> Dict[str, Any]:
        """
        Train the position sizer agent
        
        Args:
            callback: Optional custom callback for training
            
        Returns:
            Dictionary with training metrics
        """
        if self.model is None:
            self.build_model()
        
        total_timesteps = self.config["total_timesteps"]
        logger.info(f"Starting training for {total_timesteps} timesteps")
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        self.trained = True
        logger.info("Training completed")
        
        # Basic evaluation if eval_env exists
        if self.eval_env is not None:
            eval_metrics = self.evaluate()
            return {
                "training_completed": True,
                "total_timesteps": total_timesteps,
                "evaluation": eval_metrics
            }
        
        return {
            "training_completed": True,
            "total_timesteps": total_timesteps
        }
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, Any]:
        """
        Evaluate the trained agent
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.eval_env is None:
            raise ValueError("No evaluation environment. Set up with eval_data.")
        
        env = self.eval_env
        
        # Run evaluation episodes
        episode_rewards = []
        episode_sizes = []
        episode_returns = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            position_sizes = []
            
            while not done:
                # Predict action
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Record position size
                position_sizes.append(float(action[0]))
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            episode_sizes.append(np.mean(position_sizes))
            episode_returns.append(env.portfolio_return)
        
        # Calculate aggregated metrics
        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_position_size": float(np.mean(episode_sizes)),
            "mean_return": float(np.mean(episode_returns)),
            "num_episodes": num_episodes
        }
        
        logger.info(f"Evaluation results: mean reward = {metrics['mean_reward']:.4f}, " 
                   f"mean return = {metrics['mean_return']:.4f}")
        
        return metrics
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained agent
        
        Args:
            filepath: Path to save the model (if None, use default path)
            
        Returns:
            Path where the model was saved
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.config["model_dir"], exist_ok=True)
        
        # Use default path if not specified
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.config["model_dir"], f"position_sizer_{timestamp}")
        
        # Save model and VecNormalize stats
        self.model.save(filepath)
        self.vec_env.save(f"{filepath}_vecnorm.pkl")
        
        # Save config
        with open(f"{filepath}_config.json", "w") as f:
            import json
            json.dump(self.config, f, indent=4)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load(self, filepath: str, vec_normalize_path: Optional[str] = None) -> bool:
        """
        Load a trained agent
        
        Args:
            filepath: Path to the saved model
            vec_normalize_path: Path to the saved VecNormalize stats
            
        Returns:
            Success flag
        """
        try:
            # Load the model
            self.model = PPO.load(filepath)
            self.trained = True
            
            # Try to load VecNormalize stats if path not provided
            if vec_normalize_path is None:
                vec_normalize_path = f"{filepath}_vecnorm.pkl"
            
            # If VecNormalize stats exist, load them
            if os.path.exists(vec_normalize_path) and self.vec_env is not None:
                self.vec_env = VecNormalize.load(vec_normalize_path, self.vec_env)
                # Don't update normalization statistics during prediction
                self.vec_env.training = False
                # Don't normalize rewards when predicting
                self.vec_env.norm_reward = False
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, state: np.ndarray, deterministic: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Predict position size for a given state
        
        Args:
            state: Current state vector 
            deterministic: Whether to use deterministic prediction
            
        Returns:
            Tuple of (position_size, info_dict)
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get normalized state if VecNormalize is used
        if self.vec_env is not None and hasattr(self.vec_env, "normalize_obs"):
            state = self.vec_env.normalize_obs(state)
        
        # Get raw action from model
        action, _ = self.model.predict(state, deterministic=deterministic)
        
        # Extract position size from action
        position_size = float(action[0])
        
        # Clip to config limits for safety
        max_size = self.config["max_position_size"]
        position_size = max(0.0, min(position_size, max_size))
        
        # Additional info
        info = {
            "raw_action": float(action[0]),
            "clipped_size": position_size,
            "max_allowed": max_size
        }
        
        return position_size, info
    
    def get_position_size(self, 
                         signal_confidence: float, 
                         market_regime: int,
                         account_state: Dict[str, float],
                         risk_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get recommended position size based on signal confidence and current state
        
        Args:
            signal_confidence: Confidence score from ML signal (0.0-1.0)
            market_regime: Market regime category (integer)
            account_state: Account metrics (balance, equity, etc.)
            risk_metrics: Optional risk metrics (volatility, VaR, etc.)
            
        Returns:
            Dictionary with position size and related information
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Construct state vector for prediction
        state = self._construct_state_vector(
            signal_confidence, market_regime, account_state, risk_metrics
        )
        
        # Get position size
        position_size, info = self.predict(state)
        
        # Calculate actual position size in account currency
        account_size = account_state.get("equity", account_state.get("balance", 0))
        currency_size = account_size * position_size
        
        # Add to info dictionary
        result = {
            "position_size_pct": position_size,
            "position_size_currency": currency_size,
            "signal_confidence": signal_confidence,
            "market_regime": market_regime,
            "account_equity": account_size,
            **info
        }
        
        return result
    
    def _construct_state_vector(self,
                              signal_confidence: float,
                              market_regime: int,
                              account_state: Dict[str, float],
                              risk_metrics: Dict[str, float] = None) -> np.ndarray:
        """
        Construct state vector for model input
        
        Args:
            signal_confidence: Confidence score from ML signal (0.0-1.0)
            market_regime: Market regime category
            account_state: Account metrics
            risk_metrics: Risk metrics
            
        Returns:
            State vector as numpy array
        """
        # Extract account metrics
        equity = account_state.get("equity", 0)
        balance = account_state.get("balance", equity)
        margin_used = account_state.get("margin_used", 0)
        margin_level = account_state.get("margin_level", 100)
        
        # Normalize account metrics
        if equity > 0:
            margin_used_pct = margin_used / equity
        else:
            margin_used_pct = 0
            
        # Handle risk metrics
        volatility = risk_metrics.get("volatility", 0.01) if risk_metrics else 0.01
        var = risk_metrics.get("value_at_risk", 0.02) if risk_metrics else 0.02
        max_drawdown = risk_metrics.get("max_drawdown", 0.05) if risk_metrics else 0.05
        
        # One-hot encode market regime
        num_regimes = 5  # default number of regime categories
        regime_one_hot = np.zeros(num_regimes)
        if 0 <= market_regime < num_regimes:
            regime_one_hot[market_regime] = 1
        
        # Construct state vector
        state = np.array([
            signal_confidence,          # Signal confidence
            margin_used_pct,            # Margin used percentage
            min(margin_level, 1000)/1000, # Normalized margin level
            volatility,                 # Market volatility
            var,                        # Value at Risk
            max_drawdown,               # Maximum drawdown
            *regime_one_hot             # Market regime one-hot encoding
        ], dtype=np.float32)
        
        return state.reshape(1, -1)  # Return as 2D array with batch dim


# Utility function to create a position sizer agent with default configuration
def create_position_sizer_agent(config: Dict[str, Any] = None) -> PositionSizerAgent:
    """
    Create a position sizer agent with default or custom configuration
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized PositionSizerAgent instance
    """
    return PositionSizerAgent(config)
