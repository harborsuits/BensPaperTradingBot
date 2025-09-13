#!/usr/bin/env python3
"""
AI Bot Competition System

This module implements an AI bot competition system where:
1. Evolved strategies compete with micro-capital allocations
2. Winners get larger allocations (snowball effect)
3. Losers get smaller allocations or are eliminated
4. Real-time performance tracking and rankings
5. Integration with EvoTester for continuous strategy evolution
"""

import logging
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import random
import numpy as np

from trading_bot.risk.dynamic_capital_allocator import DynamicCapitalAllocator
from trading_bot.event_system import EventBus, Event, EventType

logger = logging.getLogger(__name__)


@dataclass
class CompetitionBot:
    """Represents an AI bot competing in the competition"""
    bot_id: str
    strategy_id: str
    strategy_name: str
    symbol: str
    initial_capital: float
    current_capital: float
    allocated_capital: float
    fitness_score: float
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    trades_count: int
    start_time: datetime
    last_update: datetime
    status: str = "active"  # active, paused, eliminated
    rank: int = 0
    generation: int = 0
    evolution_source: str = "evotester"  # evotester, manual, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "allocated_capital": self.allocated_capital,
            "fitness_score": self.fitness_score,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "trades_count": self.trades_count,
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "status": self.status,
            "rank": self.rank,
            "generation": self.generation,
            "evolution_source": self.evolution_source
        }


@dataclass
class CompetitionRound:
    """Represents a competition round"""
    round_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_days: int = 7
    total_capital: float = 10000.0
    min_bot_capital: float = 100.0
    max_bot_capital: float = 1000.0
    bots: List[CompetitionBot] = field(default_factory=list)
    status: str = "pending"  # pending, active, completed
    winner_bot_id: Optional[str] = None
    total_trades: int = 0
    total_return: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id": self.round_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_days": self.duration_days,
            "total_capital": self.total_capital,
            "min_bot_capital": self.min_bot_capital,
            "max_bot_capital": self.max_bot_capital,
            "bots": [bot.to_dict() for bot in self.bots],
            "status": self.status,
            "winner_bot_id": self.winner_bot_id,
            "total_trades": self.total_trades,
            "total_return": self.total_return
        }


class AIBotCompetition:
    """
    AI Bot Competition System

    Manages competitions where evolved strategies compete with micro-capital allocations.
    Winners get larger allocations, losers get smaller or are eliminated.
    """

    def __init__(self,
                 event_bus: Optional[EventBus] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.event_bus = event_bus
        self.config = config or {}

        # Competition settings
        self.round_duration_days = self.config.get("round_duration_days", 7)
        self.total_competition_capital = self.config.get("total_competition_capital", 10000.0)
        self.min_bot_capital = self.config.get("min_bot_capital", 100.0)
        self.max_bot_capital = self.config.get("max_bot_capital", 1000.0)
        self.max_bots_per_round = self.config.get("max_bots_per_round", 20)
        self.elimination_threshold = self.config.get("elimination_threshold", -0.05)  # -5%

        # Capital allocation
        self.capital_allocator = DynamicCapitalAllocator(
            config={
                "initial_capital": self.total_competition_capital,
                "min_strategy_allocation": self.min_bot_capital / self.total_competition_capital,
                "max_strategy_allocation": self.max_bot_capital / self.total_competition_capital,
                "use_snowball": True,
                "snowball_allocation_factor": 0.6  # 60% of profits reinvested
            }
        )

        # Competition state
        self.current_round: Optional[CompetitionRound] = None
        self.completed_rounds: List[CompetitionRound] = []
        self.active_bots: Dict[str, CompetitionBot] = {}

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []

        logger.info("AI Bot Competition initialized")

    async def start_new_round(self,
                            evo_strategies: List[Dict[str, Any]],
                            symbols: List[str]) -> str:
        """
        Start a new competition round with evolved strategies

        Args:
            evo_strategies: List of evolved strategies from EvoTester
            symbols: List of symbols to trade

        Returns:
            Round ID
        """
        round_id = f"round_{int(time.time())}_{random.randint(1000, 9999)}"

        # Create competition round
        round_obj = CompetitionRound(
            round_id=round_id,
            start_time=datetime.now(),
            duration_days=self.round_duration_days,
            total_capital=self.total_competition_capital,
            min_bot_capital=self.min_bot_capital,
            max_bot_capital=self.max_bot_capital
        )

        # Create bots from evolved strategies
        bots = []
        available_capital = self.total_competition_capital

        for i, strategy in enumerate(evo_strategies[:self.max_bots_per_round]):
            # Allocate capital based on fitness score
            fitness_weight = strategy.get('fitness', 0.5)
            base_allocation = self.min_bot_capital + (
                (self.max_bot_capital - self.min_bot_capital) * fitness_weight
            )

            # Ensure we don't exceed total capital
            allocation = min(base_allocation, available_capital / (len(evo_strategies) - i))
            available_capital -= allocation

            bot = CompetitionBot(
                bot_id=f"bot_{round_id}_{i}",
                strategy_id=strategy.get('id', f"strategy_{i}"),
                strategy_name=strategy.get('name', f"Strategy {i}"),
                symbol=random.choice(symbols),
                initial_capital=allocation,
                current_capital=allocation,
                allocated_capital=allocation,
                fitness_score=strategy.get('fitness', 0.5),
                total_return=0.0,
                sharpe_ratio=strategy.get('sharpeRatio', 0.0),
                win_rate=strategy.get('winRate', 0.5),
                max_drawdown=strategy.get('maxDrawdown', 0.1),
                trades_count=0,
                start_time=datetime.now(),
                last_update=datetime.now(),
                generation=strategy.get('generation', 0),
                evolution_source="evotester"
            )

            bots.append(bot)
            self.active_bots[bot.bot_id] = bot

        round_obj.bots = bots
        round_obj.status = "active"
        self.current_round = round_obj

        logger.info(f"Started competition round {round_id} with {len(bots)} bots")

        # Emit event
        if self.event_bus:
            await self.event_bus.emit(Event(
                type=EventType.STRATEGY_DEPLOYMENT,
                data={
                    "action": "competition_started",
                    "round_id": round_id,
                    "bot_count": len(bots),
                    "total_capital": self.total_competition_capital
                }
            ))

        return round_id

    async def update_bot_performance(self,
                                   bot_id: str,
                                   performance_data: Dict[str, Any]) -> None:
        """
        Update a bot's performance during competition

        Args:
            bot_id: Bot identifier
            performance_data: Updated performance metrics
        """
        if bot_id not in self.active_bots:
            logger.warning(f"Bot {bot_id} not found in active bots")
            return

        bot = self.active_bots[bot_id]

        # Update performance metrics
        bot.total_return = performance_data.get('total_return', bot.total_return)
        bot.sharpe_ratio = performance_data.get('sharpe_ratio', bot.sharpe_ratio)
        bot.win_rate = performance_data.get('win_rate', bot.win_rate)
        bot.max_drawdown = performance_data.get('max_drawdown', bot.max_drawdown)
        bot.trades_count = performance_data.get('trades_count', bot.trades_count)
        bot.last_update = datetime.now()

        # Update capital based on performance
        pnl = performance_data.get('pnl', 0.0)
        bot.current_capital += pnl

        # Check for elimination
        if bot.total_return <= self.elimination_threshold:
            bot.status = "eliminated"
            logger.info(f"Bot {bot_id} eliminated due to poor performance")

        # Update round totals
        if self.current_round:
            self.current_round.total_trades += performance_data.get('new_trades', 0)

    async def reallocate_capital(self) -> None:
        """Reallocate capital based on current performance"""
        if not self.current_round:
            return

        # Calculate performance-based allocations
        total_performance_score = sum(
            max(0, bot.total_return) for bot in self.current_round.bots
            if bot.status == "active"
        )

        if total_performance_score == 0:
            return

        # Reallocate capital
        for bot in self.current_round.bots:
            if bot.status != "active":
                continue

            # Performance-based allocation
            performance_weight = max(0, bot.total_return) / total_performance_score
            new_allocation = self.total_competition_capital * performance_weight

            # Apply limits
            new_allocation = max(self.min_bot_capital,
                               min(self.max_bot_capital, new_allocation))

            bot.allocated_capital = new_allocation

        logger.info("Capital reallocation completed")

    async def end_round(self) -> Optional[str]:
        """
        End the current competition round

        Returns:
            Winner bot ID or None
        """
        if not self.current_round:
            return None

        # Find winner (best performing bot)
        active_bots = [bot for bot in self.current_round.bots if bot.status == "active"]

        if not active_bots:
            logger.warning("No active bots remaining in round")
            return None

        winner = max(active_bots, key=lambda b: b.total_return)
        winner_bot_id = winner.bot_id

        # Update round
        self.current_round.end_time = datetime.now()
        self.current_round.status = "completed"
        self.current_round.winner_bot_id = winner_bot_id
        self.current_round.total_return = sum(bot.total_return for bot in active_bots)

        # Move to completed rounds
        self.completed_rounds.append(self.current_round)
        self.current_round = None

        # Clean up active bots
        self.active_bots.clear()

        logger.info(f"Competition round ended. Winner: {winner_bot_id}")

        # Emit event
        if self.event_bus:
            await self.event_bus.emit(Event(
                type=EventType.STRATEGY_DEPLOYMENT,
                data={
                    "action": "competition_ended",
                    "round_id": self.completed_rounds[-1].round_id,
                    "winner_bot_id": winner_bot_id,
                    "total_return": self.completed_rounds[-1].total_return
                }
            ))

        return winner_bot_id

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current competition leaderboard"""
        if not self.current_round:
            return []

        # Sort bots by total return
        sorted_bots = sorted(
            self.current_round.bots,
            key=lambda b: b.total_return,
            reverse=True
        )

        # Assign ranks
        for i, bot in enumerate(sorted_bots, 1):
            bot.rank = i

        return [bot.to_dict() for bot in sorted_bots]

    def get_round_stats(self) -> Dict[str, Any]:
        """Get current round statistics"""
        if not self.current_round:
            return {}

        active_bots = len([b for b in self.current_round.bots if b.status == "active"])
        total_return = sum(bot.total_return for bot in self.current_round.bots)
        avg_sharpe = np.mean([bot.sharpe_ratio for bot in self.current_round.bots])
        avg_win_rate = np.mean([bot.win_rate for bot in self.current_round.bots])

        return {
            "round_id": self.current_round.round_id,
            "active_bots": active_bots,
            "total_bots": len(self.current_round.bots),
            "total_capital": self.current_round.total_capital,
            "total_return": total_return,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_win_rate,
            "total_trades": self.current_round.total_trades,
            "days_remaining": self.round_duration_days - (
                datetime.now() - self.current_round.start_time
            ).days
        }

    def get_competition_history(self) -> List[Dict[str, Any]]:
        """Get competition history"""
        return [round_obj.to_dict() for round_obj in self.completed_rounds]
