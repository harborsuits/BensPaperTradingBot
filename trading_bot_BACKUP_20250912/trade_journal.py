"""
Trade Journal System

This module provides comprehensive trade journaling capabilities with advanced analytics,
performance tracking, and feedback mechanisms to continuously improve trading outcomes.
"""

import os
import json
import uuid
import datetime
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class TradeResult(str, Enum):
    """Trade outcome classifications"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PARTIAL = "partial"
    SCRATCH = "scratch"
    OPEN = "open"

class MarketRegime(str, Enum):
    """Market environment classifications"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"
    UNCLEAR = "unclear"

class TradeEntry:
    """
    Comprehensive trade entry model with detailed metadata for analysis.
    """
    
    def __init__(self, 
                trade_id: Optional[str] = None,
                timestamp: Optional[str] = None,
                symbol: str = "",
                strategy_name: str = "",
                strategy_type: str = "",
                market_regime: Optional[str] = None,
                sector: Optional[str] = None,
                trade_direction: str = "",
                position_type: str = "equity",
                position_size: float = 0.0,
                entry_price: float = 0.0,
                exit_price: Optional[float] = None,
                risk_pct: float = 1.0,
                capital_allocation_pct: float = 0.0,
                duration: Optional[str] = None,
                entry_timeframe: str = "daily",
                exit_timeframe: Optional[str] = None,
                risk_reward_ratio: Optional[float] = None,
                pnl_pct: Optional[float] = None,
                pnl_dollars: Optional[float] = None,
                result: Optional[str] = None,
                entry_tags: Optional[List[str]] = None,
                exit_tags: Optional[List[str]] = None,
                setup_strength_score: Optional[float] = None,
                confidence_level: Optional[str] = None,
                volatility_env: Optional[str] = None,
                emotional_flag: Optional[str] = None,
                journal_comment: Optional[str] = None,
                exit_reason: Optional[str] = None,
                modifications: Optional[List[Dict[str, Any]]] = None,
                linked_trades: Optional[List[Dict[str, Any]]] = None,
                **additional_fields):
        """
        Initialize a comprehensive trade entry.
        
        Args:
            trade_id: Unique identifier for the trade
            timestamp: ISO-8601 formatted entry timestamp 
            symbol: Trading symbol
            strategy_name: Name of the strategy used
            strategy_type: Category of strategy (swing, day trade, etc.)
            market_regime: Overall market environment during trade
            sector: Market sector of the instrument
            trade_direction: Long or short
            position_type: Position instrument type (equity, option, etc.)
            position_size: Number of shares/contracts
            entry_price: Entry price per share/contract
            exit_price: Exit price per share/contract
            risk_pct: Risk percentage of account
            capital_allocation_pct: Percentage of capital allocated
            duration: Trade duration as string (e.g., "3d" for 3 days)
            entry_timeframe: Timeframe used for entry decision
            exit_timeframe: Timeframe used for exit decision
            risk_reward_ratio: Planned risk/reward ratio
            pnl_pct: Profit/loss percentage
            pnl_dollars: Profit/loss in currency
            result: Trade outcome (win, loss, breakeven, etc.)
            entry_tags: List of entry pattern tags
            exit_tags: List of exit pattern tags
            setup_strength_score: Quality score for the setup (0-10)
            confidence_level: Subjective confidence level
            volatility_env: Volatility environment description
            emotional_flag: Any emotional issues noted
            journal_comment: Detailed trade notes
            exit_reason: Reason for exiting the trade
            modifications: List of trade modifications
            linked_trades: Related trades
            **additional_fields: Any additional custom fields
        """
        # Generate UUID if not provided
        self.trade_id = trade_id or str(uuid.uuid4())
        
        # Use current time if not provided
        if timestamp is None:
            self.timestamp = datetime.datetime.now().isoformat()
        else:
            self.timestamp = timestamp
            
        # Basic trade information
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.market_regime = market_regime
        self.sector = sector
        self.trade_direction = trade_direction
        self.position_type = position_type
        
        # Position details
        self.position_size = position_size
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.risk_pct = risk_pct
        self.capital_allocation_pct = capital_allocation_pct
        
        # Timing information
        self.duration = duration
        self.entry_timeframe = entry_timeframe
        self.exit_timeframe = exit_timeframe
        
        # Performance metrics
        self.risk_reward_ratio = risk_reward_ratio
        self.pnl_pct = pnl_pct
        self.pnl_dollars = pnl_dollars
        self.result = result
        
        # Analysis tags
        self.entry_tags = entry_tags or []
        self.exit_tags = exit_tags or []
        self.setup_strength_score = setup_strength_score
        self.confidence_level = confidence_level
        self.volatility_env = volatility_env
        self.emotional_flag = emotional_flag
        
        # Notes and commentary
        self.journal_comment = journal_comment
        self.exit_reason = exit_reason
        
        # Advanced tracking
        self.modifications = modifications or []
        self.linked_trades = linked_trades or []
        
        # Additional custom fields
        self.additional_fields = additional_fields
        
        # Calculated fields
        self._calculate_derived_fields()
    
    def _calculate_derived_fields(self) -> None:
        """Calculate any fields that can be derived from other fields."""
        # Calculate P&L if we have entry and exit prices
        if self.exit_price is not None and self.entry_price > 0:
            # Calculate P&L percentage
            if self.trade_direction.lower() == "long":
                self.pnl_pct = ((self.exit_price - self.entry_price) / self.entry_price) * 100
            else:  # short
                self.pnl_pct = ((self.entry_price - self.exit_price) / self.entry_price) * 100
            
            # Calculate P&L dollars
            price_diff = abs(self.exit_price - self.entry_price)
            self.pnl_dollars = price_diff * self.position_size
            
            # Determine result if not explicitly set
            if self.result is None:
                if self.pnl_dollars > 0:
                    self.result = TradeResult.WIN
                elif self.pnl_dollars < 0:
                    self.result = TradeResult.LOSS
                else:
                    self.result = TradeResult.BREAKEVEN
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade entry to dictionary."""
        base_dict = {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "market_regime": self.market_regime,
            "sector": self.sector,
            "trade_direction": self.trade_direction,
            "position_type": self.position_type,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "risk_pct": self.risk_pct,
            "capital_allocation_pct": self.capital_allocation_pct,
            "duration": self.duration,
            "entry_timeframe": self.entry_timeframe,
            "exit_timeframe": self.exit_timeframe,
            "risk_reward_ratio": self.risk_reward_ratio,
            "pnl_pct": self.pnl_pct,
            "pnl_dollars": self.pnl_dollars,
            "result": self.result,
            "entry_tags": self.entry_tags,
            "exit_tags": self.exit_tags,
            "setup_strength_score": self.setup_strength_score,
            "confidence_level": self.confidence_level,
            "volatility_env": self.volatility_env,
            "emotional_flag": self.emotional_flag,
            "journal_comment": self.journal_comment,
            "exit_reason": self.exit_reason,
            "modifications": self.modifications,
            "linked_trades": self.linked_trades
        }
        
        # Add any additional fields
        base_dict.update(self.additional_fields)
        
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeEntry':
        """Create TradeEntry instance from dictionary."""
        # Extract known fields
        known_fields = {
            'trade_id', 'timestamp', 'symbol', 'strategy_name', 'strategy_type',
            'market_regime', 'sector', 'trade_direction', 'position_type',
            'position_size', 'entry_price', 'exit_price', 'risk_pct',
            'capital_allocation_pct', 'duration', 'entry_timeframe', 'exit_timeframe',
            'risk_reward_ratio', 'pnl_pct', 'pnl_dollars', 'result',
            'entry_tags', 'exit_tags', 'setup_strength_score', 'confidence_level',
            'volatility_env', 'emotional_flag', 'journal_comment', 'exit_reason',
            'modifications', 'linked_trades'
        }
        
        # Separate known fields from additional fields
        standard_fields = {k: v for k, v in data.items() if k in known_fields}
        additional_fields = {k: v for k, v in data.items() if k not in known_fields}
        
        # Create instance with both standard and additional fields
        return cls(**standard_fields, **additional_fields)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update trade entry with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_fields[key] = value
        
        # Recalculate derived fields after updates
        self._calculate_derived_fields()

class TradeJournal:
    """
    Comprehensive trade journaling system with advanced analytics.
    """
    
    def __init__(self, journal_dir: str = "journal"):
        """
        Initialize the trade journal.
        
        Args:
            journal_dir: Directory to store journal files
        """
        self.journal_dir = journal_dir
        self.trades: Dict[str, TradeEntry] = {}
        self.metrics_cache = {}
        
        # Create journal directory if it doesn't exist
        os.makedirs(journal_dir, exist_ok=True)
        
        # Load existing trades
        self._load_trades()
        
        logger.info(f"Trade journal initialized with {len(self.trades)} existing trades")
    
    def _load_trades(self) -> None:
        """Load existing trades from journal directory."""
        try:
            journal_file = os.path.join(self.journal_dir, "trade_journal.json")
            if os.path.exists(journal_file):
                with open(journal_file, 'r') as f:
                    trades_dict = json.load(f)
                    
                    for trade_id, trade_data in trades_dict.items():
                        self.trades[trade_id] = TradeEntry.from_dict(trade_data)
                        
                logger.info(f"Loaded {len(self.trades)} trades from journal")
            else:
                logger.info("No existing journal file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading trades: {str(e)}")
    
    def _save_trades(self) -> None:
        """Save all trades to journal file."""
        try:
            journal_file = os.path.join(self.journal_dir, "trade_journal.json")
            
            # Convert trades to dictionary
            trades_dict = {trade_id: trade.to_dict() for trade_id, trade in self.trades.items()}
            
            with open(journal_file, 'w') as f:
                json.dump(trades_dict, f, indent=2)
                
            logger.info(f"Saved {len(self.trades)} trades to journal")
        except Exception as e:
            logger.error(f"Error saving trades: {str(e)}")
    
    def add_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        Add a new trade to the journal.
        
        Args:
            trade_data: Dictionary with trade details
            
        Returns:
            Trade ID of the added trade
        """
        # Create TradeEntry from data
        trade = TradeEntry.from_dict(trade_data)
        
        # Store in trades dictionary
        self.trades[trade.trade_id] = trade
        
        # Save to persistent storage
        self._save_trades()
        
        # Invalidate metrics cache
        self.metrics_cache = {}
        
        logger.info(f"Added trade {trade.trade_id} to journal")
        
        return trade.trade_id
    
    def update_trade(self, trade_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing trade with new information.
        
        Args:
            trade_id: ID of trade to update
            updates: Dictionary with updated fields
            
        Returns:
            Updated trade dictionary or None if trade not found
        """
        if trade_id not in self.trades:
            logger.error(f"Trade {trade_id} not found")
            return None
        
        # Update the trade
        self.trades[trade_id].update(updates)
        
        # Save changes
        self._save_trades()
        
        # Invalidate metrics cache
        self.metrics_cache = {}
        
        logger.info(f"Updated trade {trade_id}")
        
        return self.trades[trade_id].to_dict()
    
    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific trade by ID.
        
        Args:
            trade_id: ID of trade to retrieve
            
        Returns:
            Trade dictionary or None if not found
        """
        if trade_id not in self.trades:
            return None
            
        return self.trades[trade_id].to_dict()
    
    def get_all_trades(self) -> List[Dict[str, Any]]:
        """
        Get all trades in the journal.
        
        Returns:
            List of trade dictionaries
        """
        return [trade.to_dict() for trade in self.trades.values()]
    
    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Get all trades as a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with all trades
        """
        trades_list = self.get_all_trades()
        
        if not trades_list:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'trade_id', 'timestamp', 'symbol', 'strategy_name', 'result',
                'pnl_dollars', 'pnl_pct', 'duration'
            ])
        
        return pd.DataFrame(trades_list)
    
    def delete_trade(self, trade_id: str) -> bool:
        """
        Delete a trade from the journal.
        
        Args:
            trade_id: ID of trade to delete
            
        Returns:
            Boolean indicating success
        """
        if trade_id not in self.trades:
            logger.error(f"Trade {trade_id} not found")
            return False
        
        # Remove from trades dictionary
        del self.trades[trade_id]
        
        # Save changes
        self._save_trades()
        
        # Invalidate metrics cache
        self.metrics_cache = {}
        
        logger.info(f"Deleted trade {trade_id}")
        
        return True
    
    def get_performance_metrics(self, strategy: Optional[str] = None, 
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate performance metrics for trades, optionally filtered.
        
        Args:
            strategy: Filter by strategy name
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            
        Returns:
            Dictionary with performance metrics
        """
        # Create cache key
        cache_key = f"{strategy}_{start_date}_{end_date}"
        
        # Return cached results if available
        if cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]
        
        # Filter trades based on criteria
        filtered_trades = self._filter_trades(strategy, start_date, end_date)
        
        if not filtered_trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "net_profit": 0,
                "win_loss_ratio": 0,
                "expectancy": 0,
                "average_duration": "0d"
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([t.to_dict() for t in filtered_trades])
        
        # Separate wins and losses
        wins = df[df['result'] == TradeResult.WIN]
        losses = df[df['result'] == TradeResult.LOSS]
        
        # Calculate metrics
        total_trades = len(filtered_trades)
        win_count = len(wins)
        loss_count = len(losses)
        
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        total_profit = wins['pnl_dollars'].sum() if win_count > 0 else 0
        total_loss = abs(losses['pnl_dollars'].sum()) if loss_count > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        average_win = wins['pnl_dollars'].mean() if win_count > 0 else 0
        average_loss = losses['pnl_dollars'].mean() if loss_count > 0 else 0
        
        largest_win = wins['pnl_dollars'].max() if win_count > 0 else 0
        largest_loss = losses['pnl_dollars'].min() if loss_count > 0 else 0
        
        net_profit = df['pnl_dollars'].sum()
        
        win_loss_ratio = abs(average_win / average_loss) if average_loss != 0 else float('inf')
        
        expectancy = (win_rate * average_win) - ((1 - win_rate) * abs(average_loss))
        
        # Calculate average duration (this assumes duration is stored in a parseable format)
        # For simplicity, we'll just report it as a string for now
        average_duration = "N/A"
        
        # Assemble metrics dictionary
        metrics = {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "net_profit": net_profit,
            "win_loss_ratio": win_loss_ratio,
            "expectancy": expectancy,
            "average_duration": average_duration
        }
        
        # Cache results
        self.metrics_cache[cache_key] = metrics
        
        return metrics
    
    def get_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        """
        Compare performance across different strategies.
        
        Returns:
            Dictionary with strategy-level metrics
        """
        # Get all strategies
        all_trades = self.get_all_trades()
        strategies = set(trade['strategy_name'] for trade in all_trades if trade['strategy_name'])
        
        # Calculate metrics for each strategy
        comparison = {}
        for strategy in strategies:
            metrics = self.get_performance_metrics(strategy=strategy)
            comparison[strategy] = metrics
        
        return comparison
    
    def get_equity_curve(self, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate equity curve data points.
        
        Args:
            strategy: Optional strategy to filter by
            
        Returns:
            List of equity data points
        """
        # Filter trades
        filtered_trades = self._filter_trades(strategy=strategy)
        
        if not filtered_trades:
            return []
        
        # Sort by timestamp
        sorted_trades = sorted(filtered_trades, key=lambda t: t.timestamp)
        
        # Generate equity curve
        starting_equity = 10000  # Example starting equity
        current_equity = starting_equity
        equity_points = [{"timestamp": sorted_trades[0].timestamp, "equity": starting_equity}]
        
        for trade in sorted_trades:
            if trade.pnl_dollars is not None:
                current_equity += trade.pnl_dollars
                equity_points.append({
                    "timestamp": trade.timestamp,
                    "equity": current_equity,
                    "trade_id": trade.trade_id
                })
        
        return equity_points
    
    def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect trading patterns and generate insights.
        
        Returns:
            Dictionary with detected patterns and recommendations
        """
        df = self.get_trades_dataframe()
        
        if len(df) < 5:  # Need minimum trades for meaningful analysis
            return {"message": "Not enough trades for pattern detection"}
        
        patterns = {}
        
        # Check for consecutive losses
        if 'result' in df.columns:
            results = df['result'].tolist()
            max_consecutive_losses = self._find_max_consecutive(results, TradeResult.LOSS)
            patterns["max_consecutive_losses"] = max_consecutive_losses
            
            if max_consecutive_losses >= 3:
                patterns["warning"] = f"Detected {max_consecutive_losses} consecutive losses. Consider reducing position size."
        
        # Check win rate by day of week
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'])
            df['day_of_week'] = df['date'].dt.day_name()
            
            day_win_rates = {}
            for day, group in df.groupby('day_of_week'):
                wins = len(group[group['result'] == TradeResult.WIN])
                total = len(group)
                win_rate = wins / total if total > 0 else 0
                day_win_rates[day] = win_rate
            
            patterns["day_of_week_win_rates"] = day_win_rates
            
            # Find best and worst days
            if day_win_rates:
                best_day = max(day_win_rates.items(), key=lambda x: x[1])
                worst_day = min(day_win_rates.items(), key=lambda x: x[1])
                
                patterns["best_day"] = {"day": best_day[0], "win_rate": best_day[1]}
                patterns["worst_day"] = {"day": worst_day[0], "win_rate": worst_day[1]}
                
                # Add insight if significant difference
                if best_day[1] > 0 and worst_day[1] > 0 and best_day[1] / worst_day[1] > 1.5:
                    patterns["insight"] = f"Consider focusing on {best_day[0]} trades where win rate is {best_day[1]:.2%} vs {worst_day[0]} at {worst_day[1]:.2%}"
        
        # Check strategy performance
        if 'strategy_name' in df.columns and len(df['strategy_name'].unique()) > 1:
            strategy_comparison = self.get_strategy_comparison()
            
            # Find best and worst strategies
            if strategy_comparison:
                best_strategy = max(strategy_comparison.items(), key=lambda x: x[1].get('expectancy', 0))
                worst_strategy = min(strategy_comparison.items(), key=lambda x: x[1].get('expectancy', 0))
                
                patterns["best_strategy"] = {
                    "name": best_strategy[0],
                    "expectancy": best_strategy[1].get('expectancy', 0)
                }
                
                patterns["worst_strategy"] = {
                    "name": worst_strategy[0],
                    "expectancy": worst_strategy[1].get('expectancy', 0)
                }
        
        return patterns
    
    def get_trade_recommendations(self) -> Dict[str, Any]:
        """
        Generate recommendations based on trade history.
        
        Returns:
            Dictionary with recommendations
        """
        patterns = self.detect_patterns()
        metrics = self.get_performance_metrics()
        
        recommendations = []
        
        # Check win rate
        if metrics["win_rate"] < 0.4:
            recommendations.append({
                "type": "warning",
                "message": f"Low win rate of {metrics['win_rate']:.2%}. Consider reviewing entry criteria."
            })
        
        # Check profit factor
        if 0 < metrics["profit_factor"] < 1.5:
            recommendations.append({
                "type": "suggestion",
                "message": f"Profit factor of {metrics['profit_factor']:.2f} is below optimal. Focus on increasing average win size or decreasing average loss."
            })
        
        # Check for day of week patterns
        if "best_day" in patterns and "worst_day" in patterns:
            if patterns["best_day"]["win_rate"] > 0.6 and patterns["worst_day"]["win_rate"] < 0.4:
                recommendations.append({
                    "type": "opportunity",
                    "message": f"Consider trading more on {patterns['best_day']['day']} ({patterns['best_day']['win_rate']:.2%} win rate) and less on {patterns['worst_day']['day']} ({patterns['worst_day']['win_rate']:.2%} win rate)."
                })
        
        # Strategy recommendations
        if "best_strategy" in patterns and "worst_strategy" in patterns:
            best_expectancy = patterns["best_strategy"]["expectancy"]
            worst_expectancy = patterns["worst_strategy"]["expectancy"]
            
            if best_expectancy > 0 and worst_expectancy < 0:
                recommendations.append({
                    "type": "allocation",
                    "message": f"Consider allocating more capital to {patterns['best_strategy']['name']} and less to {patterns['worst_strategy']['name']}."
                })
        
        # Check consecutive losses
        if patterns.get("max_consecutive_losses", 0) >= 3:
            recommendations.append({
                "type": "risk_management",
                "message": f"Detected {patterns['max_consecutive_losses']} consecutive losses. Consider implementing a drawdown protection rule."
            })
        
        return {
            "recommendations": recommendations,
            "metrics_summary": metrics,
            "patterns": patterns
        }
    
    def _filter_trades(self, strategy: Optional[str] = None, 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> List[TradeEntry]:
        """
        Filter trades based on criteria.
        
        Args:
            strategy: Filter by strategy name
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            List of filtered TradeEntry objects
        """
        filtered = list(self.trades.values())
        
        # Filter by strategy
        if strategy:
            filtered = [t for t in filtered if t.strategy_name == strategy]
        
        # Filter by date range
        if start_date:
            start_dt = datetime.datetime.fromisoformat(start_date)
            filtered = [t for t in filtered if datetime.datetime.fromisoformat(t.timestamp) >= start_dt]
        
        if end_date:
            end_dt = datetime.datetime.fromisoformat(end_date)
            filtered = [t for t in filtered if datetime.datetime.fromisoformat(t.timestamp) <= end_dt]
        
        return filtered
    
    def _find_max_consecutive(self, results: List[str], target_result: str) -> int:
        """
        Find maximum consecutive occurrences of a result.
        
        Args:
            results: List of trade results
            target_result: Result to look for
            
        Returns:
            Maximum consecutive occurrences
        """
        max_consecutive = 0
        current_consecutive = 0
        
        for result in results:
            if result == target_result:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
        
    def export_to_csv(self, filepath: str) -> bool:
        """
        Export trade journal to CSV file.
        
        Args:
            filepath: Path to output CSV file
            
        Returns:
            Boolean indicating success
        """
        try:
            df = self.get_trades_dataframe()
            df.to_csv(filepath, index=False)
            logger.info(f"Exported trade journal to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return False
    
    def import_from_csv(self, filepath: str) -> int:
        """
        Import trades from CSV file.
        
        Args:
            filepath: Path to input CSV file
            
        Returns:
            Number of trades imported
        """
        try:
            df = pd.read_csv(filepath)
            
            # Convert DataFrame rows to trade entries
            count = 0
            for _, row in df.iterrows():
                trade_data = row.to_dict()
                
                # Clean NaN values
                trade_data = {k: v for k, v in trade_data.items() if pd.notna(v)}
                
                # Add trade
                self.add_trade(trade_data)
                count += 1
            
            logger.info(f"Imported {count} trades from {filepath}")
            return count
            
        except Exception as e:
            logger.error(f"Error importing from CSV: {str(e)}")
            return 0 