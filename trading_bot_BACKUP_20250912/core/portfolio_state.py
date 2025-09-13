#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PortfolioStateManager - Central brain that maintains real-time awareness of the entire trading system
"""

import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from threading import Lock

logger = logging.getLogger(__name__)

class PortfolioStateManager:
    """
    PortfolioStateManager maintains a comprehensive representation of the trading system's state.
    
    This includes:
    - Portfolio positions and values
    - Strategy allocations and performance
    - Trading activity and signals
    - System status and learning progress
    
    This class serves as the central state repository that other components can query
    for current system state, enabling context-aware decisions and responses.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the PortfolioStateManager with optional initial capital.
        
        Args:
            initial_capital: The starting capital amount (default: 100,000)
        """
        self.logger = logging.getLogger(__name__)
        self.lock = Lock()
        
        # Initialize portfolio state
        self._state = {
            "portfolio": {
                "total_value": initial_capital,
                "cash": initial_capital,
                "positions": {},
                "daily_pnl": 0.0,
                "daily_pnl_percent": 0.0,
                "overall_pnl": 0.0,
                "overall_pnl_percent": 0.0,
                "initial_capital": initial_capital
            },
            "allocations": {},
            "trades": [],
            "signals": [],
            "metrics": {
                "daily_return": 0.0,
                "weekly_return": 0.0,
                "monthly_return": 0.0,
                "yearly_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
                "win_rate": 0.0
            },
            "risk": {
                "total_exposure": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "var_99": 0.0
            },
            "learning": {
                "rl_training": False,
                "pattern_learning": False,
                "last_model_update": None,
                "current_episode": 0,
                "total_episodes": 0,
                "current_reward": 0.0,
                "best_reward": 0.0
            },
            "system": {
                "status": "initialized",
                "last_update": datetime.now().isoformat(),
                "market_hours": "closed"
            },
            "history": {
                "portfolio_values": [],
                "returns": []
            }
        }
        
        self.logger.info("PortfolioStateManager initialized with capital: ${:.2f}".format(initial_capital))
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current comprehensive state of the portfolio and system.
        
        Returns:
            Dict containing the complete state
        """
        with self.lock:
            # Create a deep copy to prevent modification of internal state
            return json.loads(json.dumps(self._state))
    
    def update_portfolio(self, 
                         total_value: Optional[float] = None,
                         cash: Optional[float] = None,
                         positions: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Update the portfolio component of the state.
        
        Args:
            total_value: Total portfolio value (optional)
            cash: Available cash balance (optional)
            positions: Dict of current positions (optional)
        """
        with self.lock:
            portfolio = self._state["portfolio"]
            
            if total_value is not None:
                old_value = portfolio["total_value"]
                portfolio["total_value"] = total_value
                
                # Update PnL values
                portfolio["daily_pnl"] = total_value - old_value
                portfolio["daily_pnl_percent"] = (portfolio["daily_pnl"] / old_value) * 100 if old_value > 0 else 0
                portfolio["overall_pnl"] = total_value - portfolio["initial_capital"]
                portfolio["overall_pnl_percent"] = (portfolio["overall_pnl"] / portfolio["initial_capital"]) * 100
                
                # Update history
                self._update_portfolio_history(total_value)
            
            if cash is not None:
                portfolio["cash"] = cash
                
            if positions is not None:
                portfolio["positions"] = positions
                
                # Calculate exposure
                exposure = sum(p.get("market_value", 0) for p in positions.values())
                self._state["risk"]["total_exposure"] = exposure
                
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.debug("Portfolio state updated")
    
    def update_strategy_allocation(self, strategy_name: str, allocation: float, performance: Optional[Dict[str, Any]] = None):
        """
        Update the allocation and performance data for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            allocation: Percentage allocation (0-100)
            performance: Dict containing performance metrics (optional)
        """
        with self.lock:
            allocations = self._state["allocations"]
            
            if strategy_name not in allocations:
                allocations[strategy_name] = {
                    "allocation": allocation,
                    "performance": performance or {}
                }
            else:
                allocations[strategy_name]["allocation"] = allocation
                if performance:
                    allocations[strategy_name]["performance"] = performance
                    
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.debug(f"Strategy allocation updated: {strategy_name} = {allocation}%")
    
    def record_trade(self, 
                    symbol: str, 
                    action: str, 
                    quantity: float,
                    price: float,
                    timestamp: Optional[str] = None,
                    strategy: Optional[str] = None,
                    status: str = "executed"):
        """
        Record a new trade in the system.
        
        Args:
            symbol: The trading symbol
            action: Buy or sell
            quantity: Number of shares/contracts
            price: Execution price
            timestamp: ISO format timestamp (default: current time)
            strategy: Name of the strategy that generated the trade
            status: Trade status
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        trade = {
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "value": price * quantity,
            "timestamp": timestamp,
            "strategy": strategy,
            "status": status
        }
        
        with self.lock:
            # Add to trades list
            self._state["trades"].append(trade)
            
            # Limit the size of the trades list
            if len(self._state["trades"]) > 1000:
                self._state["trades"] = self._state["trades"][-1000:]
                
            # Update portfolio positions and cash
            portfolio = self._state["portfolio"]
            positions = portfolio["positions"]
            
            if action.lower() == "buy":
                # Reduce cash
                portfolio["cash"] -= (price * quantity)
                
                # Update position
                if symbol in positions:
                    # Update existing position
                    pos = positions[symbol]
                    new_quantity = pos["quantity"] + quantity
                    new_cost_basis = ((pos["quantity"] * pos["cost_basis"]) + (quantity * price)) / new_quantity
                    
                    pos["quantity"] = new_quantity
                    pos["cost_basis"] = new_cost_basis
                    pos["market_value"] = new_quantity * price
                    pos["unrealized_pnl"] = (price - new_cost_basis) * new_quantity
                else:
                    # Create new position
                    positions[symbol] = {
                        "quantity": quantity,
                        "cost_basis": price,
                        "market_value": price * quantity,
                        "unrealized_pnl": 0,
                        "last_price": price
                    }
            
            elif action.lower() == "sell":
                # Increase cash
                portfolio["cash"] += (price * quantity)
                
                # Update position
                if symbol in positions:
                    pos = positions[symbol]
                    new_quantity = pos["quantity"] - quantity
                    
                    # Calculate realized PnL
                    realized_pnl = (price - pos["cost_basis"]) * quantity
                    
                    if new_quantity <= 0:
                        # Position closed
                        del positions[symbol]
                    else:
                        # Position reduced
                        pos["quantity"] = new_quantity
                        pos["market_value"] = new_quantity * price
                        pos["unrealized_pnl"] = (price - pos["cost_basis"]) * new_quantity
                        pos["last_price"] = price
            
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.info(f"Trade recorded: {action} {quantity} {symbol} @ ${price:.2f}")
    
    def record_signal(self, 
                     symbol: str, 
                     signal_type: str, 
                     strength: float,
                     timestamp: Optional[str] = None,
                     strategy: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a new trading signal in the system.
        
        Args:
            symbol: The trading symbol
            signal_type: Type of signal (buy, sell, hold)
            strength: Signal strength (0-1)
            timestamp: ISO format timestamp (default: current time)
            strategy: Name of the strategy that generated the signal
            metadata: Additional signal metadata (optional)
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        signal = {
            "symbol": symbol,
            "signal_type": signal_type,
            "strength": strength,
            "timestamp": timestamp,
            "strategy": strategy,
            "metadata": metadata or {}
        }
        
        with self.lock:
            # Add to signals list
            self._state["signals"].append(signal)
            
            # Limit the size of the signals list
            if len(self._state["signals"]) > 1000:
                self._state["signals"] = self._state["signals"][-1000:]
                
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.debug(f"Signal recorded: {signal_type} {symbol} (strength: {strength:.2f})")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """
        Update the performance metrics of the portfolio.
        
        Args:
            metrics: Dict of metric names and values to update
        """
        with self.lock:
            for key, value in metrics.items():
                if key in self._state["metrics"]:
                    self._state["metrics"][key] = value
                    
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.debug("Performance metrics updated")
    
    def update_risk_metrics(self, risk_metrics: Dict[str, float]):
        """
        Update the risk metrics of the portfolio.
        
        Args:
            risk_metrics: Dict of risk metric names and values to update
        """
        with self.lock:
            for key, value in risk_metrics.items():
                if key in self._state["risk"]:
                    self._state["risk"][key] = value
                    
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.debug("Risk metrics updated")
    
    def update_learning_status(self, 
                              rl_training: Optional[bool] = None,
                              pattern_learning: Optional[bool] = None,
                              current_episode: Optional[int] = None,
                              total_episodes: Optional[int] = None,
                              current_reward: Optional[float] = None,
                              best_reward: Optional[float] = None):
        """
        Update the learning system status.
        
        Args:
            rl_training: Whether RL training is active
            pattern_learning: Whether pattern learning is active
            current_episode: Current training episode
            total_episodes: Total training episodes
            current_reward: Current episode reward
            best_reward: Best reward achieved so far
        """
        with self.lock:
            learning = self._state["learning"]
            
            if rl_training is not None:
                learning["rl_training"] = rl_training
                
            if pattern_learning is not None:
                learning["pattern_learning"] = pattern_learning
                
            if current_episode is not None:
                learning["current_episode"] = current_episode
                
            if total_episodes is not None:
                learning["total_episodes"] = total_episodes
                
            if current_reward is not None:
                learning["current_reward"] = current_reward
                
            if best_reward is not None:
                learning["best_reward"] = best_reward
                
            # Update model timestamp if training is happening
            if rl_training or pattern_learning:
                learning["last_model_update"] = datetime.now().isoformat()
                
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.debug("Learning status updated")
    
    def update_system_status(self, status: str, market_hours: Optional[str] = None):
        """
        Update the system status.
        
        Args:
            status: System status string
            market_hours: Market hours status (open, closed)
        """
        with self.lock:
            self._state["system"]["status"] = status
            
            if market_hours is not None:
                self._state["system"]["market_hours"] = market_hours
                
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.info(f"System status updated: {status}")
    
    def get_portfolio_value_history(self) -> List[Dict[str, Any]]:
        """
        Get the historical portfolio values.
        
        Returns:
            List of portfolio value history entries
        """
        with self.lock:
            return self._state["history"]["portfolio_values"]
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current portfolio positions.
        
        Returns:
            Dict of positions by symbol
        """
        with self.lock:
            return self._state["portfolio"]["positions"]
    
    def get_cash(self) -> float:
        """
        Get the current cash balance.
        
        Returns:
            Current cash balance
        """
        with self.lock:
            return self._state["portfolio"]["cash"]
    
    def get_total_value(self) -> float:
        """
        Get the current total portfolio value.
        
        Returns:
            Current total portfolio value
        """
        with self.lock:
            return self._state["portfolio"]["total_value"]
    
    def get_strategy_allocations(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the current strategy allocations.
        
        Returns:
            Dict of strategy allocations
        """
        with self.lock:
            return self._state["allocations"]
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent trades.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of recent trades
        """
        with self.lock:
            return self._state["trades"][-limit:]
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent signals.
        
        Args:
            limit: Maximum number of signals to return
            
        Returns:
            List of recent signals
        """
        with self.lock:
            return self._state["signals"][-limit:]
    
    def to_json(self) -> str:
        """
        Convert the current state to a JSON string.
        
        Returns:
            JSON string representation of the state
        """
        with self.lock:
            return json.dumps(self._state, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """
        Load state from a JSON string.
        
        Args:
            json_str: JSON string representation of the state
        """
        with self.lock:
            try:
                new_state = json.loads(json_str)
                self._state = new_state
                self.logger.info("State loaded from JSON")
            except Exception as e:
                self.logger.error(f"Error loading state from JSON: {e}")
                raise
    
    def _update_portfolio_history(self, current_value: float) -> None:
        """
        Update the portfolio value history.
        
        Args:
            current_value: The current portfolio value
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "value": current_value
        }
        
        history = self._state["history"]
        history["portfolio_values"].append(entry)
        
        # Limit the size of the history list
        if len(history["portfolio_values"]) > 1000:
            history["portfolio_values"] = history["portfolio_values"][-1000:]
        
        # If we have more than one entry, calculate return
        if len(history["portfolio_values"]) > 1:
            prev_value = history["portfolio_values"][-2]["value"]
            if prev_value > 0:
                return_pct = ((current_value - prev_value) / prev_value) * 100
                return_entry = {
                    "timestamp": entry["timestamp"],
                    "return_pct": return_pct
                }
                history["returns"].append(return_entry)
                
                # Limit the size of the returns list
                if len(history["returns"]) > 1000:
                    history["returns"] = history["returns"][-1000:]
    
    def calculate_performance_metrics(self) -> None:
        """
        Calculate and update performance metrics based on history.
        """
        with self.lock:
            history = self._state["history"]
            
            if len(history["returns"]) < 2:
                return
            
            # Convert history to pandas dataframe
            returns_df = pd.DataFrame(history["returns"])
            returns_df["timestamp"] = pd.to_datetime(returns_df["timestamp"])
            returns_df.set_index("timestamp", inplace=True)
            
            # Convert percentage returns to decimals
            returns_df["return"] = returns_df["return_pct"] / 100
            
            # Calculate daily return (average)
            daily_return = returns_df["return"].mean()
            
            # Calculate metrics
            metrics = self._state["metrics"]
            
            # Simple returns
            metrics["daily_return"] = daily_return * 100  # back to percentage
            metrics["weekly_return"] = daily_return * 5 * 100
            metrics["monthly_return"] = daily_return * 21 * 100
            metrics["yearly_return"] = daily_return * 252 * 100
            
            # Volatility (standard deviation of returns)
            metrics["volatility"] = returns_df["return"].std() * np.sqrt(252) * 100
            
            # Sharpe ratio (assuming risk-free rate of 0)
            if metrics["volatility"] > 0:
                metrics["sharpe_ratio"] = (metrics["yearly_return"] / 100) / (metrics["volatility"] / 100)
            
            # Max drawdown
            cum_returns = (1 + returns_df["return"]).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns / running_max) - 1
            metrics["max_drawdown"] = abs(drawdown.min()) * 100
            
            # Win rate
            positive_days = (returns_df["return"] > 0).sum()
            total_days = len(returns_df)
            metrics["win_rate"] = (positive_days / total_days) * 100 if total_days > 0 else 0
            
            # Update risk metrics
            risk = self._state["risk"]
            risk["max_drawdown"] = metrics["max_drawdown"]
            
            # Calculate Value at Risk (VaR)
            risk["var_95"] = abs(returns_df["return"].quantile(0.05)) * 100
            risk["var_99"] = abs(returns_df["return"].quantile(0.01)) * 100
            
            # Update timestamp
            self._state["system"]["last_update"] = datetime.now().isoformat()
            
        self.logger.debug("Performance metrics calculated")
    
    def reset(self, initial_capital: Optional[float] = None) -> None:
        """
        Reset the portfolio state to initial values.
        
        Args:
            initial_capital: New initial capital amount (optional)
        """
        with self.lock:
            if initial_capital is None:
                initial_capital = self._state["portfolio"]["initial_capital"]
                
            # Re-initialize portfolio state
            self._state = {
                "portfolio": {
                    "total_value": initial_capital,
                    "cash": initial_capital,
                    "positions": {},
                    "daily_pnl": 0.0,
                    "daily_pnl_percent": 0.0,
                    "overall_pnl": 0.0,
                    "overall_pnl_percent": 0.0,
                    "initial_capital": initial_capital
                },
                "allocations": {},
                "trades": [],
                "signals": [],
                "metrics": {
                    "daily_return": 0.0,
                    "weekly_return": 0.0,
                    "monthly_return": 0.0,
                    "yearly_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "volatility": 0.0,
                    "win_rate": 0.0
                },
                "risk": {
                    "total_exposure": 0.0,
                    "max_drawdown": 0.0,
                    "var_95": 0.0,
                    "var_99": 0.0
                },
                "learning": {
                    "rl_training": False,
                    "pattern_learning": False,
                    "last_model_update": None,
                    "current_episode": 0,
                    "total_episodes": 0,
                    "current_reward": 0.0,
                    "best_reward": 0.0
                },
                "system": {
                    "status": "reset",
                    "last_update": datetime.now().isoformat(),
                    "market_hours": "closed"
                },
                "history": {
                    "portfolio_values": [],
                    "returns": []
                }
            }
            
        self.logger.info(f"Portfolio state reset with capital: ${initial_capital:.2f}") 