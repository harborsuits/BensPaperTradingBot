#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reinforcement Learning Agent Module for BensBot

This module defines the RL agents that can learn and execute trading strategies
in the trading environment. It implements various agent architectures
including DQN, Actor-Critic, and PPO.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import logging
from typing import Dict, List, Union, Tuple, Optional, Any
import gym
import os
import datetime
from collections import deque
import random

# Configure logging
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """
        Get the current size of the buffer.
        
        Returns:
            Number of transitions in the buffer
        """
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for discrete action trading.
    """
    
    def __init__(self, 
                state_shape: Tuple[int, ...],
                n_actions: int,
                learning_rate: float = 0.001,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                batch_size: int = 64,
                target_update_freq: int = 1000,
                replay_buffer_size: int = 10000):
        """
        Initialize the DQN agent.
        
        Args:
            state_shape: Shape of the state observations
            n_actions: Number of discrete actions
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate at which to decay epsilon
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            replay_buffer_size: Size of the replay buffer
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # Create Q-networks
        self.q_network = self._build_q_network()
        self.target_q_network = self._build_q_network()
        
        # Initialize target network weights
        self.target_q_network.set_weights(self.q_network.get_weights())
        
        # Training step counter
        self.train_step_counter = 0
        
        logger.info(f"Initialized DQN agent with {n_actions} actions")
    
    def _build_q_network(self) -> keras.Model:
        """
        Build the Q-network model.
        
        Returns:
            Keras Model instance
        """
        inputs = layers.Input(shape=self.state_shape)
        
        # Use LSTM for temporal features (if state has time dimension)
        if len(self.state_shape) > 1 and self.state_shape[0] > 1:
            x = layers.LSTM(64, return_sequences=True)(inputs)
            x = layers.LSTM(64)(x)
        else:
            x = layers.Flatten()(inputs)
            x = layers.Dense(64, activation='relu')(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        outputs = layers.Dense(self.n_actions, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        
        return model
    
    def select_action(self, state, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether we're in training mode
            
        Returns:
            Selected action
        """
        # Expand dimensions to match batch format expected by the model
        state_batch = np.expand_dims(state, axis=0)
        
        # Epsilon-greedy action selection
        if training and np.random.random() < self.epsilon:
            # Choose random action
            return np.random.randint(0, self.n_actions)
        else:
            # Choose greedy action
            q_values = self.q_network.predict(state_batch)[0]
            return np.argmax(q_values)
    
    def train(self) -> Dict[str, float]:
        """
        Train the agent on a batch from the replay buffer.
        
        Returns:
            Dictionary with training metrics
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {'loss': 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Compute target Q values
        next_q_values = self.target_q_network.predict(next_states)
        max_next_q = np.max(next_q_values, axis=1)
        
        # Q-learning update rule
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Get current Q-values and update with targets
        current_q = self.q_network.predict(states)
        
        # Update only the Q-values for the actions that were taken
        for i in range(self.batch_size):
            current_q[i, actions[i]] = target_q_values[i]
        
        # Train the network
        history = self.q_network.fit(
            states, current_q, verbose=0, batch_size=self.batch_size
        )
        
        # Update target network if needed
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            'loss': history.history['loss'][0],
            'epsilon': self.epsilon
        }
    
    def save(self, filepath: str):
        """
        Save the agent's models.
        
        Args:
            filepath: Path to save the models
        """
        self.q_network.save(filepath + "_q_network.h5")
        
        # Save other parameters
        params = {
            'epsilon': self.epsilon,
            'train_step_counter': self.train_step_counter
        }
        np.save(filepath + "_params.npy", params)
        
        logger.info(f"Saved DQN agent to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the agent's models.
        
        Args:
            filepath: Path to load the models from
        """
        self.q_network = keras.models.load_model(filepath + "_q_network.h5")
        self.target_q_network = keras.models.load_model(filepath + "_q_network.h5")
        
        # Load other parameters
        params = np.load(filepath + "_params.npy", allow_pickle=True).item()
        self.epsilon = params['epsilon']
        self.train_step_counter = params['train_step_counter']
        
        logger.info(f"Loaded DQN agent from {filepath}")


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent for continuous action trading.
    """
    
    def __init__(self, 
                state_shape: Tuple[int, ...],
                action_dim: int,
                action_high: float = 1.0,
                actor_learning_rate: float = 0.001,
                critic_learning_rate: float = 0.002,
                gamma: float = 0.99,
                tau: float = 0.005,
                batch_size: int = 64,
                replay_buffer_size: int = 10000,
                exploration_noise: float = 0.1):
        """
        Initialize the DDPG agent.
        
        Args:
            state_shape: Shape of the state observations
            action_dim: Dimension of the action space
            action_high: Maximum value of actions
            actor_learning_rate: Learning rate for the actor
            critic_learning_rate: Learning rate for the critic
            gamma: Discount factor for future rewards
            tau: Soft update coefficient for target networks
            batch_size: Batch size for training
            replay_buffer_size: Size of the replay buffer
            exploration_noise: Standard deviation of exploration noise
        """
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.action_high = action_high
        self.actor_lr = actor_learning_rate
        self.critic_lr = critic_learning_rate
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_size)
        
        # Create actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        # Create target networks
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Initialize target network weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        logger.info(f"Initialized DDPG agent with {action_dim} dimensional actions")
    
    def _build_actor(self) -> keras.Model:
        """
        Build the actor model.
        
        Returns:
            Keras Model instance
        """
        inputs = layers.Input(shape=self.state_shape)
        
        # Use LSTM for temporal features (if state has time dimension)
        if len(self.state_shape) > 1 and self.state_shape[0] > 1:
            x = layers.LSTM(64, return_sequences=True)(inputs)
            x = layers.LSTM(64)(x)
        else:
            x = layers.Flatten()(inputs)
            x = layers.Dense(64, activation='relu')(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output with tanh activation to bound actions
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)
        outputs = outputs * self.action_high
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.actor_lr))
        
        return model
    
    def _build_critic(self) -> keras.Model:
        """
        Build the critic model.
        
        Returns:
            Keras Model instance
        """
        # State input
        state_input = layers.Input(shape=self.state_shape)
        
        # Process state input
        if len(self.state_shape) > 1 and self.state_shape[0] > 1:
            state_x = layers.LSTM(64, return_sequences=True)(state_input)
            state_x = layers.LSTM(64)(state_x)
        else:
            state_x = layers.Flatten()(state_input)
            state_x = layers.Dense(64, activation='relu')(state_x)
        
        # Action input
        action_input = layers.Input(shape=(self.action_dim,))
        
        # Combine state and action
        x = layers.Concatenate()([state_x, action_input])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output Q-value
        outputs = layers.Dense(1)(x)
        
        model = keras.Model(inputs=[state_input, action_input], outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.critic_lr))
        
        return model
    
    def select_action(self, state, training: bool = True) -> np.ndarray:
        """
        Select an action using the actor network with optional exploration noise.
        
        Args:
            state: Current state
            training: Whether we're in training mode
            
        Returns:
            Selected action
        """
        # Expand dimensions to match batch format expected by the model
        state_batch = np.expand_dims(state, axis=0)
        
        # Get action from actor
        action = self.actor.predict(state_batch)[0]
        
        # Add exploration noise if in training mode
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action += noise
            
            # Clip action to valid range
            action = np.clip(action, -self.action_high, self.action_high)
        
        return action
    
    def train(self) -> Dict[str, float]:
        """
        Train the agent on a batch from the replay buffer.
        
        Returns:
            Dictionary with training metrics
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # --------------- Train Critic ---------------
        # Get actions for next states using target actor
        next_actions = self.target_actor.predict(next_states)
        
        # Get Q-values for next states and actions using target critic
        next_q_values = self.target_critic.predict([next_states, next_actions])
        
        # Compute target Q-values using Bellman equation
        target_q = rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * self.gamma * next_q_values
        
        # Train critic
        critic_history = self.critic.fit(
            [states, actions], target_q,
            verbose=0, batch_size=self.batch_size
        )
        critic_loss = critic_history.history['loss'][0]
        
        # --------------- Train Actor ---------------
        # Custom training step for actor using gradients from critic
        with tf.GradientTape() as tape:
            # Get actions from actor
            actions_pred = self.actor(states)
            
            # Get Q-values from critic
            q_values = self.critic([states, actions_pred])
            
            # Actor loss is negative mean Q-value (we want to maximize Q)
            actor_loss = -tf.reduce_mean(q_values)
        
        # Apply gradients to actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # --------------- Update Target Networks ---------------
        # Soft update target networks
        self._update_target_networks()
        
        return {
            'actor_loss': actor_loss.numpy(),
            'critic_loss': critic_loss
        }
    
    def _update_target_networks(self):
        """
        Soft update target networks.
        """
        # Update target actor
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        
        self.target_actor.set_weights(target_actor_weights)
        
        # Update target critic
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
        
        self.target_critic.set_weights(target_critic_weights)
    
    def save(self, filepath: str):
        """
        Save the agent's models.
        
        Args:
            filepath: Path to save the models
        """
        self.actor.save(filepath + "_actor.h5")
        self.critic.save(filepath + "_critic.h5")
        
        logger.info(f"Saved DDPG agent to {filepath}")
    
    def load(self, filepath: str):
        """
        Load the agent's models.
        
        Args:
            filepath: Path to load the models from
        """
        self.actor = keras.models.load_model(filepath + "_actor.h5")
        self.critic = keras.models.load_model(filepath + "_critic.h5")
        
        # Load target networks
        self.target_actor = keras.models.load_model(filepath + "_actor.h5")
        self.target_critic = keras.models.load_model(filepath + "_critic.h5")
        
        logger.info(f"Loaded DDPG agent from {filepath}")
