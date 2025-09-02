#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RLTrainer - Manages the training and integration of RL agents for strategy rotator.
"""

import os
import numpy as np
import pandas as pd
import json
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from trading_bot.backtesting.data_manager import DataManager
from trading_bot.backtesting.pattern_learner import PatternLearner
from trading_bot.learning.rl_agent import RLStrategyAgent
from trading_bot.learning.rl_environment import RLTradingEnv

logger = logging.getLogger(__name__)

class RLTrainer:
    """
    Manages the training, evaluation, and integration of RL agents for trading strategy allocation.
    
    This class coordinates the data pipeline, training process, and production integration
    of reinforcement learning models for trading strategy selection.
    """
    
    def __init__(
        self,
        strategies: List[str],
        model_dir: str = "models/rl_agents",
        data_path: str = "data/backtest_history.json",
        results_dir: str = "results/rl_agents",
        algorithm: str = "PPO",
        reward_type: str = "sharpe",
        use_pattern_insights: bool = True,
        retrain_frequency: int = 7,  # days
        eval_frequency: int = 1,  # days
        window_size: int = 30,
        episode_length: int = 252,
        verbose: int = 1
    ):
        """
        Initialize the RL Trainer.
        
        Args:
            strategies: List of strategy names to allocate between
            model_dir: Directory to save/load models
            data_path: Path to backtest history data
            results_dir: Directory to save results
            algorithm: RL algorithm to use ('PPO', 'A2C', or 'DQN')
            reward_type: Type of reward function ('sharpe', 'sortino', 'pnl', 'calmar')
            use_pattern_insights: Whether to include PatternLearner insights in state
            retrain_frequency: Frequency of retraining in days
            eval_frequency: Frequency of evaluation in days
            window_size: Number of days of history to include in state
            episode_length: Number of steps per episode
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
        """
        self.strategies = strategies
        self.model_dir = model_dir
        self.data_path = data_path
        self.results_dir = results_dir
        self.algorithm = algorithm
        self.reward_type = reward_type
        self.use_pattern_insights = use_pattern_insights
        self.retrain_frequency = retrain_frequency
        self.eval_frequency = eval_frequency
        self.window_size = window_size
        self.episode_length = episode_length
        self.verbose = verbose
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.data_manager = DataManager(save_path=data_path)
        self.pattern_learner = PatternLearner(data_manager=self.data_manager) if use_pattern_insights else None
        
        # Initialize agent
        self.agent = RLStrategyAgent(
            strategies=strategies,
            model_path=os.path.join(model_dir, f"agent_{algorithm}_{reward_type}"),
            algorithm=algorithm,
            data_path=data_path,
            window_size=window_size,
            episode_length=episode_length,
            reward_type=reward_type,
            use_pattern_insights=use_pattern_insights,
            verbose=verbose
        )
        
        # Training info
        self.last_train_time = None
        self.last_eval_time = None
        self.training_thread = None
        self.is_training = False
        self.is_evaluating = False
        self.training_history = []
        self.eval_history = []
        
        # Best model info
        self.best_model_path = None
        self.best_model_performance = -float('inf')
        
        logger.info(f"Initialized RLTrainer for {len(strategies)} strategies: {', '.join(strategies)}")
    
    def train(
        self, 
        total_timesteps: int = 100000, 
        eval_freq: int = 10000,
        save_plots: bool = True,
        save_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            eval_freq: Frequency of evaluation during training
            save_plots: Whether to save training plots
            save_metrics: Whether to save training metrics
            
        Returns:
            Training results
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.is_training = True
        
        # Update data if needed
        if self.use_pattern_insights and self.pattern_learner:
            self.pattern_learner.analyze(save_results=True)
        
        # Train the agent
        train_start_time = datetime.now()
        training_history = self.agent.train(total_timesteps=total_timesteps, eval_freq=eval_freq)
        train_end_time = datetime.now()
        
        # Update training info
        self.last_train_time = train_end_time
        
        # Extract key metrics
        training_results = {
            "start_time": train_start_time.isoformat(),
            "end_time": train_end_time.isoformat(),
            "duration_seconds": (train_end_time - train_start_time).total_seconds(),
            "total_timesteps": total_timesteps,
            "algorithm": self.algorithm,
            "reward_type": self.reward_type,
            "window_size": self.window_size,
            "episode_length": self.episode_length,
            "use_pattern_insights": self.use_pattern_insights,
            "final_episode_rewards": training_history.get("episode_rewards", [])[-10:],
            "mean_episode_reward": np.mean(training_history.get("episode_rewards", [0])),
            "n_episodes": len(training_history.get("episode_rewards", [])),
            "model_path": self.agent.model_path
        }
        
        # Save training metrics
        if save_metrics:
            metrics_path = os.path.join(self.results_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(metrics_path, "w") as f:
                json.dump(training_results, f, indent=2)
            logger.info(f"Training metrics saved to {metrics_path}")
        
        # Generate and save plots
        if save_plots:
            self._generate_training_plots(training_history)
        
        # Append to history
        self.training_history.append(training_results)
        
        # Evaluate the trained model
        logger.info("Evaluating trained model")
        eval_results = self.evaluate(n_episodes=10, save_plots=save_plots, save_metrics=save_metrics)
        
        # Update best model if this one is better
        eval_metric = eval_results.get("mean_reward", -float('inf'))
        if eval_metric > self.best_model_performance:
            self.best_model_performance = eval_metric
            self.best_model_path = self.agent.model_path
            
            # Save best model separately
            best_path = os.path.join(self.model_dir, f"best_agent_{self.algorithm}_{self.reward_type}")
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            
            # Copy model files
            import shutil
            if os.path.exists(self.agent.model_path):
                shutil.copy2(self.agent.model_path, best_path)
                logger.info(f"New best model saved to {best_path}")
                
                # Copy history file if it exists
                history_path = f"{self.agent.model_path}_history.json"
                best_history_path = f"{best_path}_history.json"
                if os.path.exists(history_path):
                    shutil.copy2(history_path, best_history_path)
        
        self.is_training = False
        logger.info("Training completed")
        
        return {**training_results, "evaluation": eval_results}
    
    def _generate_training_plots(self, training_history: Dict[str, Any]):
        """
        Generate and save training plots.
        
        Args:
            training_history: Training history data
        """
        try:
            # Create plot directory
            plot_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Episode rewards
            if "episode_rewards" in training_history and training_history["episode_rewards"]:
                plt.figure(figsize=(10, 6))
                rewards = training_history["episode_rewards"]
                plt.plot(rewards)
                plt.title(f"{self.algorithm} Training: Episode Rewards")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.grid(True, alpha=0.3)
                
                # Add smoothed line
                window_size = min(50, len(rewards) // 10 + 1)
                if window_size > 1:
                    smoothed = pd.Series(rewards).rolling(window=window_size).mean()
                    plt.plot(smoothed, color='red', linewidth=2)
                    plt.legend(['Rewards', f'{window_size}-Episode Moving Average'])
                
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"episode_rewards_{timestamp}.png"))
                plt.close()
            
            # Episode lengths
            if "episode_lengths" in training_history and training_history["episode_lengths"]:
                plt.figure(figsize=(10, 6))
                lengths = training_history["episode_lengths"]
                plt.plot(lengths)
                plt.title(f"{self.algorithm} Training: Episode Lengths")
                plt.xlabel("Episode")
                plt.ylabel("Length (steps)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"episode_lengths_{timestamp}.png"))
                plt.close()
            
            logger.info(f"Training plots saved to {plot_dir}")
        except Exception as e:
            logger.error(f"Error generating training plots: {str(e)}")
    
    def train_in_background(self, total_timesteps: int = 100000):
        """
        Start training in a background thread.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            
        Returns:
            True if training started, False if already training
        """
        if self.is_training:
            logger.warning("Training already in progress")
            return False
        
        # Create and start training thread
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(total_timesteps,),
            daemon=True
        )
        self.training_thread.start()
        logger.info(f"Background training started with {total_timesteps} timesteps")
        
        return True
    
    def _training_worker(self, total_timesteps: int):
        """
        Worker function for background training.
        
        Args:
            total_timesteps: Total number of timesteps to train for
        """
        try:
            self.train(total_timesteps=total_timesteps)
        except Exception as e:
            logger.error(f"Error in background training: {str(e)}")
            self.is_training = False
    
    def load_agent(self, model_path: Optional[str] = None):
        """
        Load a trained agent.
        
        Args:
            model_path: Path to the trained model (defaults to best model)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # Use best model path if available and no path provided
        if model_path is None and self.best_model_path is not None:
            model_path = self.best_model_path
        
        # If still None, use the default path
        if model_path is None:
            model_path = os.path.join(self.model_dir, f"agent_{self.algorithm}_{self.reward_type}")
        
        return self.agent.load(model_path)
    
    def evaluate(
        self, 
        n_episodes: int = 10, 
        save_plots: bool = True,
        save_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate the agent's performance.
        
        Args:
            n_episodes: Number of episodes to evaluate
            save_plots: Whether to save evaluation plots
            save_metrics: Whether to save evaluation metrics
            
        Returns:
            Evaluation results
        """
        logger.info(f"Starting evaluation for {n_episodes} episodes")
        self.is_evaluating = True
        
        # Evaluate the agent
        eval_start_time = datetime.now()
        eval_metrics = self.agent.evaluate(n_episodes=n_episodes)
        eval_end_time = datetime.now()
        
        # Update evaluation info
        self.last_eval_time = eval_end_time
        
        # Add metadata to results
        eval_results = {
            **eval_metrics,
            "start_time": eval_start_time.isoformat(),
            "end_time": eval_end_time.isoformat(),
            "duration_seconds": (eval_end_time - eval_start_time).total_seconds(),
            "n_episodes": n_episodes,
            "algorithm": self.algorithm,
            "reward_type": self.reward_type,
            "model_path": self.agent.model_path
        }
        
        # Save evaluation metrics
        if save_metrics:
            metrics_path = os.path.join(self.results_dir, f"eval_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            # Convert numpy values to Python types for JSON serialization
            json_metrics = eval_results.copy()
            for key, value in json_metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, np.float32) or isinstance(value, np.float64):
                    json_metrics[key] = float(value)
                elif isinstance(value, np.int32) or isinstance(value, np.int64):
                    json_metrics[key] = int(value)
            
            with open(metrics_path, "w") as f:
                json.dump(json_metrics, f, indent=2)
            logger.info(f"Evaluation metrics saved to {metrics_path}")
        
        # Generate and save plots
        if save_plots:
            self._generate_evaluation_plots(eval_results)
        
        # Append to history
        self.eval_history.append(eval_results)
        
        self.is_evaluating = False
        logger.info("Evaluation completed")
        
        return eval_results
    
    def _generate_evaluation_plots(self, eval_results: Dict[str, Any]):
        """
        Generate and save evaluation plots.
        
        Args:
            eval_results: Evaluation results
        """
        try:
            # Create plot directory
            plot_dir = os.path.join(self.results_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Episode metrics
            if "episode_metrics" in eval_results and eval_results["episode_metrics"]:
                # Portfolio returns
                plt.figure(figsize=(10, 6))
                returns = [m["portfolio_return"] * 100 for m in eval_results["episode_metrics"]]
                episodes = list(range(1, len(returns) + 1))
                
                plt.bar(episodes, returns, alpha=0.7)
                plt.axhline(y=np.mean(returns), color='r', linestyle='-', label=f'Mean: {np.mean(returns):.2f}%')
                plt.title(f"{self.algorithm} Evaluation: Portfolio Returns by Episode")
                plt.xlabel("Episode")
                plt.ylabel("Return (%)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"eval_returns_{timestamp}.png"))
                plt.close()
                
                # Drawdowns
                plt.figure(figsize=(10, 6))
                drawdowns = [m["max_drawdown"] * 100 for m in eval_results["episode_metrics"]]
                plt.bar(episodes, drawdowns, alpha=0.7, color='orange')
                plt.axhline(y=np.mean(drawdowns), color='r', linestyle='-', label=f'Mean: {np.mean(drawdowns):.2f}%')
                plt.title(f"{self.algorithm} Evaluation: Maximum Drawdowns by Episode")
                plt.xlabel("Episode")
                plt.ylabel("Max Drawdown (%)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f"eval_drawdowns_{timestamp}.png"))
                plt.close()
            
            logger.info(f"Evaluation plots saved to {plot_dir}")
        except Exception as e:
            logger.error(f"Error generating evaluation plots: {str(e)}")
    
    def get_recommended_allocations(self, market_data: pd.DataFrame, signals: Dict[str, Any]) -> Dict[str, float]:
        """
        Get recommended strategy allocations based on current market conditions.
        
        Args:
            market_data: Recent market data
            signals: Dictionary of recent strategy signals
            
        Returns:
            Dictionary mapping strategy names to allocation weights
        """
        # Ensure agent is loaded
        if not hasattr(self.agent, 'model') or self.agent.model is None:
            self.load_agent()
        
        # Get allocations from agent
        allocations = self.agent.get_allocation_for_signals(signals, market_data)
        
        return allocations
    
    def should_retrain(self) -> bool:
        """
        Check if the agent should be retrained based on time since last training.
        
        Returns:
            True if retraining is needed, False otherwise
        """
        if self.last_train_time is None:
            return True
        
        time_since_last_train = datetime.now() - self.last_train_time
        return time_since_last_train.days >= self.retrain_frequency
    
    def should_evaluate(self) -> bool:
        """
        Check if the agent should be evaluated based on time since last evaluation.
        
        Returns:
            True if evaluation is needed, False otherwise
        """
        if self.last_eval_time is None:
            return True
        
        time_since_last_eval = datetime.now() - self.last_eval_time
        return time_since_last_eval.days >= self.eval_frequency
    
    def start_continuous_improvement(self, train_timesteps: int = 100000, check_interval: int = 3600):
        """
        Start a background thread for continuous improvement of the agent.
        This will periodically check if retraining or evaluation is needed.
        
        Args:
            train_timesteps: Number of timesteps to train for when retraining
            check_interval: Time between checks in seconds (default: 1 hour)
            
        Returns:
            Thread object
        """
        thread = threading.Thread(
            target=self._continuous_improvement_worker,
            args=(train_timesteps, check_interval),
            daemon=True
        )
        thread.start()
        logger.info(f"Continuous improvement started (check interval: {check_interval}s)")
        
        return thread
    
    def _continuous_improvement_worker(self, train_timesteps: int, check_interval: int):
        """
        Worker function for continuous improvement thread.
        
        Args:
            train_timesteps: Number of timesteps to train for when retraining
            check_interval: Time between checks in seconds
        """
        try:
            while True:
                # Check if retraining is needed
                if self.should_retrain() and not self.is_training:
                    logger.info("Starting scheduled retraining")
                    self.train(total_timesteps=train_timesteps)
                
                # Check if evaluation is needed
                elif self.should_evaluate() and not self.is_evaluating and not self.is_training:
                    logger.info("Starting scheduled evaluation")
                    self.evaluate()
                
                # Wait for next check
                time.sleep(check_interval)
                
        except Exception as e:
            logger.error(f"Error in continuous improvement thread: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current status of the RL system.
        
        Returns:
            Dictionary with system status
        """
        return {
            "is_training": self.is_training,
            "is_evaluating": self.is_evaluating,
            "last_train_time": self.last_train_time.isoformat() if self.last_train_time else None,
            "last_eval_time": self.last_eval_time.isoformat() if self.last_eval_time else None,
            "model_path": self.agent.model_path,
            "best_model_path": self.best_model_path,
            "best_model_performance": self.best_model_performance,
            "algorithm": self.algorithm,
            "reward_type": self.reward_type,
            "strategies": self.strategies,
            "training_runs": len(self.training_history),
            "evaluation_runs": len(self.eval_history),
            "latest_performance": self.eval_history[-1].get("mean_reward") if self.eval_history else None
        } 