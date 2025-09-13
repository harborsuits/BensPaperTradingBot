#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Journal module for tracking strategy allocations, performance, and market context.
"""

import os
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class TradeJournal:
    """
    Track and store information about strategy allocations, performance metrics, and market context.
    
    The TradeJournal maintains historical records of trading activity, strategy rotations,
    and performance metrics to facilitate analysis, reporting, and strategy optimization.
    """
    
    def __init__(self, journal_dir="data/journal"):
        """
        Initialize the TradeJournal.
        
        Args:
            journal_dir (str): Directory path to store journal files
        """
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), journal_dir)
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        
        self.strategy_history_file = os.path.join(self.base_dir, "strategy_allocations.csv")
        self.performance_history_file = os.path.join(self.base_dir, "performance_metrics.csv")
        self.rotation_logs_file = os.path.join(self.base_dir, "rotation_logs.json")
        
        self._init_journal_files()
        logger.info(f"Trade Journal initialized at {self.base_dir}")
    
    def _init_journal_files(self):
        """Initialize journal files if they don't exist."""
        # Create strategy allocations history file
        if not os.path.exists(self.strategy_history_file):
            df = pd.DataFrame(columns=["timestamp", "strategy", "allocation", "capital_allocated"])
            df.to_csv(self.strategy_history_file, index=False)
        
        # Create performance metrics history file
        if not os.path.exists(self.performance_history_file):
            df = pd.DataFrame(columns=["timestamp", "strategy", "return", "sharpe", "max_drawdown", "win_rate"])
            df.to_csv(self.performance_history_file, index=False)
        
        # Create rotation logs file
        if not os.path.exists(self.rotation_logs_file):
            with open(self.rotation_logs_file, "w") as f:
                json.dump([], f)
    
    def log_allocations(self, allocations, total_capital):
        """
        Log strategy allocations.
        
        Args:
            allocations (dict): Strategy allocations as {strategy_name: allocation_percentage}
            total_capital (float): Total capital in the account
        """
        now = datetime.now().isoformat()
        new_rows = []
        
        for strategy, allocation in allocations.items():
            capital_allocated = total_capital * (allocation / 100)
            new_rows.append({
                "timestamp": now,
                "strategy": strategy,
                "allocation": allocation,
                "capital_allocated": capital_allocated
            })
        
        df = pd.read_csv(self.strategy_history_file)
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(self.strategy_history_file, index=False)
        
        logger.debug(f"Logged allocations for {len(allocations)} strategies")
        
    def log_performance(self, performance_metrics):
        """
        Log performance metrics for strategies.
        
        Args:
            performance_metrics (dict): Performance metrics as 
                {strategy_name: {metric_name: value}}
        """
        now = datetime.now().isoformat()
        new_rows = []
        
        for strategy, metrics in performance_metrics.items():
            row = {
                "timestamp": now,
                "strategy": strategy,
                **metrics
            }
            new_rows.append(row)
        
        df = pd.read_csv(self.performance_history_file)
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(self.performance_history_file, index=False)
        
        logger.debug(f"Logged performance metrics for {len(performance_metrics)} strategies")
    
    def log_rotation(self, rotation_data):
        """
        Log a strategy rotation event.
        
        Args:
            rotation_data (dict): Data about the rotation, including:
                - timestamp: When the rotation occurred
                - market_context: Current market context
                - old_allocations: Previous strategy allocations
                - new_allocations: New strategy allocations
                - reasoning: Reasoning behind the rotation
        """
        rotation_data["timestamp"] = datetime.now().isoformat()
        
        with open(self.rotation_logs_file, "r") as f:
            logs = json.load(f)
        
        logs.append(rotation_data)
        
        with open(self.rotation_logs_file, "w") as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Logged rotation event at {rotation_data['timestamp']}")
    
    def get_strategy_history(self, strategy=None, days=None):
        """
        Get historical allocation data for strategies.
        
        Args:
            strategy (str, optional): Filter by strategy name
            days (int, optional): Number of days to look back
            
        Returns:
            DataFrame: Historical allocation data
        """
        df = pd.read_csv(self.strategy_history_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if strategy:
            df = df[df["strategy"] == strategy]
        
        if days:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]
        
        return df
    
    def get_performance_history(self, strategy=None, days=None):
        """
        Get historical performance data for strategies.
        
        Args:
            strategy (str, optional): Filter by strategy name
            days (int, optional): Number of days to look back
            
        Returns:
            DataFrame: Historical performance data
        """
        df = pd.read_csv(self.performance_history_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if strategy:
            df = df[df["strategy"] == strategy]
        
        if days:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df["timestamp"] >= cutoff]
        
        return df
    
    def get_rotation_logs(self, count=None):
        """
        Get rotation log events.
        
        Args:
            count (int, optional): Number of most recent events to return
            
        Returns:
            list: Rotation log events
        """
        with open(self.rotation_logs_file, "r") as f:
            logs = json.load(f)
        
        if count:
            return logs[-count:]
        
        return logs 