"""
Strategy Rotation Automation Module.

This module uses LLM evaluations to automatically adjust capital allocation
across different trading strategies based on market conditions.
"""

import os
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import strategy prioritizer
from trading_bot.ai_scoring.strategy_prioritizer import StrategyPrioritizer
from trading_bot.journal.llm_trade_journal import LLMTradeJournal
from trading_bot.notification_manager.telegram_notifier import TelegramNotifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StrategyRotator:
    """
    Automates capital allocation across trading strategies based on
    LLM evaluations of market conditions and strategy performance.
    """
    
    def __init__(
        self,
        strategies: List[str] = None,
        initial_allocations: Dict[str, float] = None,
        portfolio_value: float = 100000.0,
        strategy_prioritizer: Optional[StrategyPrioritizer] = None,
        journal: Optional[LLMTradeJournal] = None,
        notifier: Optional[TelegramNotifier] = None,
        max_allocation_change: float = 15.0,  # Maximum percentage change in a single rotation
        min_allocation_pct: float = 5.0,  # Minimum allocation percentage for any active strategy
        rotation_frequency_days: int = 30,  # How often to perform strategy rotation
        config_path: Optional[str] = None,
        enable_guardrails: bool = True,  # Enable allocation guardrails
        market_context_fetcher = None,
        use_mock: bool = False
    ):
        """
        Initialize the strategy rotator.
        
        Args:
            strategies: List of available strategy names
            initial_allocations: Initial capital allocations by strategy (percentages)
            portfolio_value: Total portfolio value
            strategy_prioritizer: Strategy prioritizer for market analysis
            journal: Trade journal for performance tracking
            notifier: Notification system for allocation changes
            max_allocation_change: Maximum percentage a strategy allocation can change in one rotation
            min_allocation_pct: Minimum allocation percentage for any active strategy
            rotation_frequency_days: How often to perform strategy rotation
            config_path: Path to configuration file
            enable_guardrails: Whether to enable allocation guardrails based on risk assessment
            market_context_fetcher: MarketContextFetcher instance
            use_mock: Whether to use mock data for testing
        """
        self.logger = logging.getLogger("StrategyRotator")
        
        # Set available strategies
        self.strategies = strategies or [
            'trend_following', 'momentum', 'mean_reversion', 
            'breakout_swing', 'volatility_breakout', 'option_spreads'
        ]
        
        # Set portfolio value
        self.portfolio_value = portfolio_value
        
        # Load or set initial allocations
        if initial_allocations:
            self.current_allocations = initial_allocations
        else:
            # Equal allocation across all strategies
            equal_allocation = 100.0 / len(self.strategies)
            self.current_allocations = {s: equal_allocation for s in self.strategies}
        
        # Normalize allocations to ensure they sum to 100%
        self._normalize_allocations()
        
        # Set strategy prioritizer
        self.strategy_prioritizer = strategy_prioritizer
        
        # Set journal and notifier
        self.journal = journal
        self.notifier = notifier
        
        # Set configuration
        self.max_allocation_change = max_allocation_change
        self.min_allocation_pct = min_allocation_pct
        self.rotation_frequency_days = rotation_frequency_days
        self.enable_guardrails = enable_guardrails
        
        # Initialize rotation history
        self.rotation_history = []
        
        # Set last rotation date
        self.last_rotation_date = None
        
        # Initialize risk metrics
        self.strategy_drawdowns = {s: 0.0 for s in self.strategies}
        self.portfolio_drawdown = 0.0
        self.highest_portfolio_value = portfolio_value
        self.risk_levels = {
            "low": {"max_allocation": 35.0, "aggressive_strategies_cap": 60.0},
            "medium": {"max_allocation": 25.0, "aggressive_strategies_cap": 45.0},
            "high": {"max_allocation": 20.0, "aggressive_strategies_cap": 30.0}
        }
        
        # Strategy risk classifications
        self.strategy_risk_levels = {
            "option_spreads": "high",
            "volatility_breakout": "high",
            "momentum": "medium",
            "breakout_swing": "medium",
            "trend_following": "medium",
            "mean_reversion": "low"
        }
        
        # Set default risk level
        self.current_risk_level = "low"
        
        # Load config if provided
        if config_path:
            self._load_config(config_path)
            
        # Load state if available
        self._load_state()
        
        # Store instances of services
        self.market_context_fetcher = market_context_fetcher
        
        self.logger.info(f"Strategy Rotator initialized with {len(self.strategies)} strategies")
        self.logger.info(f"Current allocations: {json.dumps(self.current_allocations)}")
        if self.enable_guardrails:
            self.logger.info("Allocation guardrails enabled for risk management")
        
        # Set use_mock
        self.use_mock = use_mock
    
    def _normalize_allocations(self) -> None:
        """Normalize allocations to ensure they sum to 100%."""
        total = sum(self.current_allocations.values())
        if total != 100.0:
            scaling_factor = 100.0 / total
            self.current_allocations = {
                s: allocation * scaling_factor 
                for s, allocation in self.current_allocations.items()
            }
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update configuration
                if 'max_allocation_change' in config:
                    self.max_allocation_change = config['max_allocation_change']
                if 'min_allocation_pct' in config:
                    self.min_allocation_pct = config['min_allocation_pct']
                if 'rotation_frequency_days' in config:
                    self.rotation_frequency_days = config['rotation_frequency_days']
                
                # Load guardrail configuration if available
                if 'enable_guardrails' in config:
                    self.enable_guardrails = config['enable_guardrails']
                    
                if 'risk_levels' in config:
                    self.risk_levels.update(config['risk_levels'])
                    
                if 'strategy_risk_levels' in config:
                    self.strategy_risk_levels.update(config['strategy_risk_levels'])
                
                self.logger.info(f"Configuration loaded from {config_path}")
            else:
                self.logger.warning(f"Config file {config_path} not found, using defaults")
        
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {str(e)}")
    
    def _load_state(self) -> None:
        """Load the previous state of allocations and rotation history."""
        state_path = os.path.join(
            os.path.dirname(__file__),
            "data",
            "strategy_rotator_state.json"
        )
        
        try:
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                # Load current allocations
                if 'current_allocations' in state:
                    self.current_allocations = state['current_allocations']
                
                # Load rotation history
                if 'rotation_history' in state:
                    self.rotation_history = state['rotation_history']
                
                # Load last rotation date
                if 'last_rotation_date' in state:
                    self.last_rotation_date = datetime.fromisoformat(state['last_rotation_date'])
                else:
                    self.last_rotation_date = datetime.now() - timedelta(days=self.rotation_frequency_days + 1)
                
                # Load risk metrics if available
                if 'highest_portfolio_value' in state:
                    self.highest_portfolio_value = state['highest_portfolio_value']
                
                if 'portfolio_drawdown' in state:
                    self.portfolio_drawdown = state['portfolio_drawdown']
                    
                if 'current_risk_level' in state:
                    self.current_risk_level = state['current_risk_level']
                    
                if 'strategy_drawdowns' in state:
                    self.strategy_drawdowns = state['strategy_drawdowns']
                
                self.logger.info(f"State loaded from {state_path}")
            else:
                self.logger.info("No previous state found, using initial allocations")
                self.last_rotation_date = datetime.now() - timedelta(days=self.rotation_frequency_days + 1)
        
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            self.last_rotation_date = datetime.now() - timedelta(days=self.rotation_frequency_days + 1)
    
    def _save_state(self) -> None:
        """Save the current state of allocations and rotation history."""
        # Create directory if it doesn't exist
        state_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(state_dir, exist_ok=True)
        
        state_path = os.path.join(state_dir, "strategy_rotator_state.json")
        
        try:
            state = {
                'current_allocations': self.current_allocations,
                'rotation_history': self.rotation_history,
                'last_rotation_date': self.last_rotation_date.isoformat() if self.last_rotation_date else None,
                'portfolio_value': self.portfolio_value,
                'highest_portfolio_value': self.highest_portfolio_value,
                'portfolio_drawdown': self.portfolio_drawdown,
                'current_risk_level': self.current_risk_level,
                'strategy_drawdowns': self.strategy_drawdowns
            }
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"State saved to {state_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
    
    def get_strategy_performance(self, lookback_days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Get recent performance metrics for each strategy.
        
        Args:
            lookback_days: Number of days to look back for performance data
            
        Returns:
            Dictionary with performance metrics by strategy
        """
        performance = {}
        
        # If no journal, return empty performance metrics
        if not self.journal:
            return {s: {'win_rate': 0.5, 'trades_count': 0} for s in self.strategies}
        
        try:
            # Calculate date range
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            # Get performance for each strategy
            for strategy in self.strategies:
                # Get trades for this strategy
                trades = self.journal.search_trades(
                    strategy=strategy,
                    start_date=start_date,
                    status='closed'
                )
                
                # Calculate performance metrics
                if trades:
                    wins = [t for t in trades if t.get('pnl', 0) > 0]
                    win_rate = len(wins) / len(trades) if trades else 0.5
                    
                    total_pnl = sum(t.get('pnl', 0) for t in trades)
                    avg_pnl = total_pnl / len(trades) if trades else 0
                    
                    performance[strategy] = {
                        'win_rate': win_rate,
                        'trades_count': len(trades),
                        'total_pnl': total_pnl,
                        'avg_pnl': avg_pnl
                    }
                else:
                    # No trades found, use default metrics
                    performance[strategy] = {
                        'win_rate': 0.5,
                        'trades_count': 0,
                        'total_pnl': 0,
                        'avg_pnl': 0
                    }
        
        except Exception as e:
            self.logger.error(f"Error calculating strategy performance: {str(e)}")
            # Return default performance metrics
            performance = {s: {'win_rate': 0.5, 'trades_count': 0} for s in self.strategies}
        
        return performance
    
    def is_rotation_due(self) -> bool:
        """
        Check if it's time to perform a strategy rotation.
        
        Returns:
            True if rotation is due, False otherwise
        """
        # If no previous rotation, it's due
        if not self.last_rotation_date:
            return True
        
        # Calculate days since last rotation
        days_since_rotation = (datetime.now() - self.last_rotation_date).days
        
        # Check if it's time for rotation
        return days_since_rotation >= self.rotation_frequency_days
    
    def rotate_strategies(
        self, 
        market_context: Optional[Dict[str, Any]] = None,
        current_allocations: Optional[Dict[str, float]] = None,
        force_rotation: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Perform strategy rotation based on market conditions.
        
        Args:
            market_context: Market context data (optional)
            current_allocations: Current allocations (uses internal state if None)
            force_rotation: Force rotation even if not due
            
        Returns:
            Tuple of (new_allocations, rotation_result)
        """
        # Use provided current allocations or internal state
        if current_allocations is not None:
            self.current_allocations = current_allocations.copy()
            self._normalize_allocations()
        
        # Check if rotation is due
        if not force_rotation and not self.is_rotation_due():
            self.logger.info("Strategy rotation not due yet")
            return self.current_allocations, {
                'status': 'skipped',
                'message': 'Rotation not due yet',
                'days_to_next_rotation': self.rotation_frequency_days - (datetime.now() - self.last_rotation_date).days
                if self.last_rotation_date else 0
            }
        
        # Check if we have a strategy prioritizer
        if not self.strategy_prioritizer:
            self.logger.error("Strategy prioritizer not available, cannot perform rotation")
            return self.current_allocations, {
                'status': 'error',
                'message': 'Strategy prioritizer not available'
            }
        
        self.logger.info("Performing strategy rotation")
        
        try:
            # Get strategy allocations from prioritizer
            target_allocations = self.strategy_prioritizer.get_strategy_allocation(market_context)
            
            # Ensure all strategies are included
            for strategy in self.strategies:
                if strategy not in target_allocations:
                    target_allocations[strategy] = 0.0
            
            # Calculate allocation changes
            allocation_changes = {}
            for strategy in self.strategies:
                current = self.current_allocations.get(strategy, 0.0)
                target = float(target_allocations.get(strategy, 0.0))  # Ensure it's a float
                change = target - current
                allocation_changes[strategy] = change
            
            # Assess risk level and update current risk level if guardrails are enabled
            if self.enable_guardrails and market_context:
                self._assess_risk_level(market_context)
                
                # Apply guardrails to target allocations based on risk assessment
                target_allocations = self._apply_allocation_guardrails(target_allocations)
                
                # Recalculate allocation changes with guardrail adjustments
                for strategy in self.strategies:
                    current = self.current_allocations.get(strategy, 0.0)
                    target = float(target_allocations.get(strategy, 0.0))
                    change = target - current
                    allocation_changes[strategy] = change
            
            # Apply constraints to allocation changes
            constrained_allocations = self._apply_allocation_constraints(target_allocations)
            
            # Calculate dollar values
            dollar_values = {}
            for strategy, allocation in constrained_allocations.items():
                dollar_values[strategy] = (allocation / 100.0) * self.portfolio_value
            
            # Create rotation record
            rotation_record = {
                'timestamp': datetime.now().isoformat(),
                'previous_allocations': self.current_allocations.copy(),
                'target_allocations': target_allocations,
                'constrained_allocations': constrained_allocations,
                'allocation_changes': allocation_changes,
                'dollar_values': dollar_values,
                'portfolio_value': self.portfolio_value,
                'market_context_summary': self._get_market_context_summary(market_context),
                'risk_level': self.current_risk_level if self.enable_guardrails else 'not_assessed'
            }
            
            # Update current allocations
            self.current_allocations = constrained_allocations
            
            # Update last rotation date
            self.last_rotation_date = datetime.now()
            
            # Add rotation to history
            self.rotation_history.append(rotation_record)
            
            # Trim history if needed
            if len(self.rotation_history) > 50:
                self.rotation_history = self.rotation_history[-50:]
            
            # Save state
            self._save_state()
            
            # Send notification if available
            if self.notifier:
                self._send_rotation_notification(rotation_record)
            
            self.logger.info(f"Strategy rotation completed: {json.dumps(constrained_allocations)}")
            
            return constrained_allocations, {
                'status': 'success',
                'message': 'Rotation completed successfully',
                'results': rotation_record
            }
            
        except Exception as e:
            self.logger.error(f"Error rotating strategies: {str(e)}")
            return self.current_allocations, {
                'status': 'error',
                'message': f'Error during rotation: {str(e)}'
            }
    
    def _apply_allocation_constraints(
        self, 
        target_allocations: Dict[str, float], 
        allocation_changes: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Apply constraints to the target allocations to ensure smooth transitions.
        
        Args:
            target_allocations: Target allocation percentages
            allocation_changes: Optional changes in allocation percentages
            
        Returns:
            Constrained allocation percentages
        """
        constrained_allocations = {}
        
        # Calculate allocation changes if not provided
        if allocation_changes is None:
            allocation_changes = {
                strategy: target_allocations.get(strategy, 0.0) - self.current_allocations.get(strategy, 0.0)
                for strategy in self.strategies
            }
        
        # Convert target_allocations values to float
        target_allocations = {k: float(v) for k, v in target_allocations.items()}
        
        # Apply maximum change constraint
        for strategy, change in allocation_changes.items():
            # Limit change to max_allocation_change
            if abs(change) > self.max_allocation_change:
                # Apply constrained change while respecting direction
                if change > 0:
                    constrained_change = self.max_allocation_change
                else:
                    constrained_change = -self.max_allocation_change
                
                # Update constrained allocations
                constrained_allocations[strategy] = self.current_allocations.get(strategy, 0.0) + constrained_change
            else:
                # Change is within limits, apply directly
                constrained_allocations[strategy] = target_allocations[strategy]
        
        # Apply minimum allocation constraint
        # First identify active strategies (targeted above 0%)
        active_strategies = [s for s, a in target_allocations.items() if float(a) > 0]
        
        # Ensure active strategies meet minimum allocation
        for strategy in active_strategies:
            if 0 < constrained_allocations[strategy] < self.min_allocation_pct:
                # Boost to minimum allocation
                constrained_allocations[strategy] = self.min_allocation_pct
        
        # Normalize to ensure allocations sum to 100%
        total = sum(constrained_allocations.values())
        if total != 100.0:
            scaling_factor = 100.0 / total
            constrained_allocations = {
                s: allocation * scaling_factor 
                for s, allocation in constrained_allocations.items()
            }
        
        return constrained_allocations
    
    def _get_market_context_summary(self, market_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary of the market context for recording in rotation history.
        
        Args:
            market_context: Full market context data
            
        Returns:
            Simplified market context summary
        """
        if not market_context:
            return {'market_regime': 'unknown'}
        
        summary = {
            'market_regime': str(market_context.get('market_regime', 'unknown')),
            'volatility_index': float(market_context.get('volatility_index', 0))
        }
        
        # Add sector performance summary if available
        if 'sector_performance' in market_context:
            sector_perf = market_context['sector_performance']
            if isinstance(sector_perf, dict) and sector_perf:
                try:
                    # Convert values to float for comparison
                    float_sector_perf = {k: float(v) for k, v in sector_perf.items()}
                    # Get top and bottom sectors
                    sorted_sectors = sorted(float_sector_perf.items(), key=lambda x: x[1], reverse=True)
                    if sorted_sectors:
                        summary['top_sector'] = sorted_sectors[0][0]
                        summary['bottom_sector'] = sorted_sectors[-1][0]
                except (ValueError, TypeError):
                    # Handle case where values can't be converted to float
                    self.logger.warning("Could not sort sector performance, values not numeric")
        
        # Add recent news summary if available
        if 'recent_news' in market_context and market_context['recent_news']:
            recent_news = market_context['recent_news']
            if isinstance(recent_news, list) and recent_news:
                # Get the most recent news item
                latest_news = recent_news[0]
                if isinstance(latest_news, dict) and 'headline' in latest_news:
                    summary['latest_news'] = latest_news['headline']
        
        return summary
    
    def _send_rotation_notification(self, rotation_record: Dict[str, Any]) -> None:
        """
        Send a notification about strategy rotation.
        
        Args:
            rotation_record: Record of the latest rotation
        """
        try:
            # Build message
            message = f"""📊 <b>Strategy Rotation Completed</b>

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>Market Regime:</b> {rotation_record['market_context_summary'].get('market_regime', 'Unknown')}

<b>New Allocations:</b>
"""
            
            # Sort strategies by allocation
            sorted_strategies = sorted(
                rotation_record['constrained_allocations'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Add allocation details
            for strategy, allocation in sorted_strategies:
                if allocation > 0:
                    prev_allocation = rotation_record['previous_allocations'].get(strategy, 0.0)
                    change = allocation - prev_allocation
                    change_emoji = "🔼" if change > 0 else "🔽" if change < 0 else "◼️"
                    
                    message += f"• {strategy}: {allocation:.1f}% ({change_emoji} {change:+.1f}%)\n"
            
            # Add portfolio value
            message += f"\n<b>Portfolio Value:</b> ${rotation_record['portfolio_value']:,.2f}"
            
            # Send message
            self.notifier.send_message(message, parse_mode="HTML")
            
            self.logger.info("Rotation notification sent")
            
        except Exception as e:
            self.logger.error(f"Error sending rotation notification: {str(e)}")
    
    def get_allocations(self) -> Dict[str, float]:
        """
        Get current strategy allocations.
        
        Returns:
            Dictionary with current allocations
        """
        return self.current_allocations
    
    def get_dollar_allocations(self) -> Dict[str, float]:
        """
        Get current strategy allocations in dollar amounts.
        
        Returns:
            Dictionary with current dollar allocations
        """
        return {
            strategy: (allocation / 100.0) * self.portfolio_value
            for strategy, allocation in self.current_allocations.items()
        }
    
    def update_portfolio_value(self, new_value: float) -> None:
        """
        Update the total portfolio value.
        
        Args:
            new_value: New portfolio value
        """
        # Track drawdown if guardrails enabled
        if self.enable_guardrails:
            # Update highest portfolio value if current value is higher
            if new_value > self.highest_portfolio_value:
                self.highest_portfolio_value = new_value
            
            # Calculate current drawdown percentage
            if self.highest_portfolio_value > 0:
                self.portfolio_drawdown = (self.highest_portfolio_value - new_value) / self.highest_portfolio_value * 100
                self.logger.info(f"Current portfolio drawdown: {self.portfolio_drawdown:.2f}%")
        
        self.portfolio_value = new_value
        self._save_state()
        
        self.logger.info(f"Portfolio value updated to ${new_value:,.2f}")
    
    def manual_adjust_allocation(
        self, 
        strategy: str, 
        new_allocation: float,
        normalize: bool = True
    ) -> Dict[str, Any]:
        """
        Manually adjust the allocation for a specific strategy.
        
        Args:
            strategy: Strategy name
            new_allocation: New allocation percentage
            normalize: Whether to normalize all allocations after adjustment
            
        Returns:
            Dictionary with adjustment results
        """
        # Check if strategy exists
        if strategy not in self.strategies:
            self.logger.error(f"Strategy {strategy} not found in available strategies")
            return {
                'status': 'error',
                'message': f'Strategy {strategy} not found'
            }
        
        # Store previous allocation
        previous_allocation = self.current_allocations.get(strategy, 0.0)
        
        # Update allocation
        self.current_allocations[strategy] = new_allocation
        
        # Normalize if requested
        if normalize:
            self._normalize_allocations()
        
        # Create adjustment record
        adjustment_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'previous_allocation': previous_allocation,
            'new_allocation': self.current_allocations[strategy],
            'adjustment_type': 'manual',
            'normalize_applied': normalize
        }
        
        # Add to rotation history
        self.rotation_history.append(adjustment_record)
        
        # Save state
        self._save_state()
        
        # Send notification if available
        if self.notifier:
            try:
                message = f"""🔧 <b>Manual Allocation Adjustment</b>

<b>Strategy:</b> {strategy}
<b>Previous:</b> {previous_allocation:.1f}%
<b>New:</b> {self.current_allocations[strategy]:.1f}%
<b>Change:</b> {self.current_allocations[strategy] - previous_allocation:+.1f}%

<i>Adjusted manually at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
                self.notifier.send_message(message, parse_mode="HTML")
            except Exception as e:
                self.logger.error(f"Error sending adjustment notification: {str(e)}")
        
        self.logger.info(
            f"Manual allocation adjustment for {strategy}: "
            f"{previous_allocation:.1f}% -> {self.current_allocations[strategy]:.1f}%"
        )
        
        return {
            'status': 'success',
            'message': f'Allocation for {strategy} adjusted successfully',
            'results': adjustment_record
        }
    
    def get_rotation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of strategy rotations.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of rotation records, most recent first
        """
        # Return the most recent items
        return self.rotation_history[-limit:][::-1]
    
    def reset_to_equal_allocation(self) -> Dict[str, Any]:
        """
        Reset allocations to equal distribution across all strategies.
        
        Returns:
            Dictionary with reset results
        """
        # Store previous allocations
        previous_allocations = self.current_allocations.copy()
        
        # Calculate equal allocation
        equal_allocation = 100.0 / len(self.strategies)
        
        # Set new allocations
        self.current_allocations = {s: equal_allocation for s in self.strategies}
        
        # Create reset record
        reset_record = {
            'timestamp': datetime.now().isoformat(),
            'previous_allocations': previous_allocations,
            'new_allocations': self.current_allocations.copy(),
            'adjustment_type': 'reset_to_equal'
        }
        
        # Add to rotation history
        self.rotation_history.append(reset_record)
        
        # Save state
        self._save_state()
        
        # Send notification if available
        if self.notifier:
            try:
                message = f"""🔄 <b>Strategy Allocations Reset</b>

Allocations have been reset to equal distribution across all strategies.
<b>Each strategy:</b> {equal_allocation:.1f}%

<i>Reset performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
                self.notifier.send_message(message, parse_mode="HTML")
            except Exception as e:
                self.logger.error(f"Error sending reset notification: {str(e)}")
        
        self.logger.info("Strategy allocations reset to equal distribution")
        
        return {
            'status': 'success',
            'message': 'Allocations reset to equal distribution',
            'results': reset_record
        }

    def get_market_summary(self, market_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Get LLM-generated market summary based on current conditions.
        
        Args:
            market_context: Market context information (fetched automatically if None)
            
        Returns:
            String with market summary
        """
        # Get prioritization result
        result = self.strategy_prioritizer.get_strategy_allocation(market_context)
        
        # Extract market summary
        if "market_summary" in result:
            return result["market_summary"]
        
        # Fallback to basic summary
        if market_context:
            return market_context.get("market_summary", "No market summary available")
        else:
            return "No market summary available"

    def _apply_allocation_guardrails(self, target_allocations: Dict[str, float]) -> Dict[str, float]:
        """
        Apply guardrails to strategy allocations based on risk assessment.
        
        This method applies risk-based guardrails to the target allocations,
        ensuring that allocations respect maximum concentration limits and
        risk exposure constraints based on current market conditions and
        portfolio drawdown levels.
        
        Args:
            target_allocations: Target allocations from strategy prioritizer
            
        Returns:
            Risk-adjusted target allocations
        """
        if not self.enable_guardrails:
            return target_allocations
            
        self.logger.info(f"Applying allocation guardrails (risk level: {self.current_risk_level})")
        
        # Create a copy of target allocations
        adjusted_allocations = {s: float(a) for s, a in target_allocations.items()}
        
        # Get risk level parameters
        risk_params = self.risk_levels[self.current_risk_level]
        max_allocation = risk_params["max_allocation"]
        aggressive_cap = risk_params["aggressive_strategies_cap"]
        
        # Identify high and medium risk strategies
        aggressive_strategies = [s for s in self.strategies 
                              if self.strategy_risk_levels.get(s, "medium") in ["high", "medium"]]
        high_risk_strategies = [s for s in self.strategies 
                             if self.strategy_risk_levels.get(s, "medium") == "high"]
        
        # Limit individual strategy allocations
        for strategy, allocation in adjusted_allocations.items():
            strategy_risk = self.strategy_risk_levels.get(strategy, "medium")
            
            # Adjust max allocation based on strategy risk level
            strategy_max = max_allocation
            if strategy_risk == "high":
                strategy_max = max(max_allocation * 0.8, 15.0)  # 20% lower cap for high risk strategies
            elif strategy_risk == "low":
                strategy_max = max_allocation * 1.2  # 20% higher cap for low risk strategies
                
            # Apply individual strategy cap
            if allocation > strategy_max:
                self.logger.info(f"Guardrail: {strategy} allocation reduced from {allocation:.1f}% to {strategy_max:.1f}%")
                adjusted_allocations[strategy] = strategy_max
        
        # Calculate total allocation for aggressive strategies
        aggressive_total = sum(adjusted_allocations.get(s, 0) for s in aggressive_strategies)
        
        # If aggressive allocation exceeds cap, reduce proportionally
        if aggressive_total > aggressive_cap:
            reduction_factor = aggressive_cap / aggressive_total
            
            for strategy in aggressive_strategies:
                original = adjusted_allocations[strategy]
                adjusted_allocations[strategy] = original * reduction_factor
                
            self.logger.info(f"Guardrail: Aggressive strategies reduced from {aggressive_total:.1f}% to {aggressive_cap:.1f}%")
            
        # Special handling for high drawdown situations (>10%)
        if self.portfolio_drawdown > 10.0:
            # Reduce high risk strategies further
            for strategy in high_risk_strategies:
                reduction_factor = 0.7  # 30% reduction
                original = adjusted_allocations[strategy]
                adjusted_allocations[strategy] = original * reduction_factor
                
            self.logger.info(f"Guardrail: High drawdown ({self.portfolio_drawdown:.1f}%) - reduced high-risk strategy allocations")
        
        # Normalize the allocations to ensure they sum to 100%
        total = sum(adjusted_allocations.values())
        if total > 0:  # Avoid division by zero
            normalized_allocations = {s: (a / total) * 100.0 for s, a in adjusted_allocations.items()}
        else:
            # Fallback to equal allocation
            normalized_allocations = {s: 100.0 / len(self.strategies) for s in self.strategies}
        
        return normalized_allocations
    
    def _assess_risk_level(self, market_context: Dict[str, Any]) -> str:
        """
        Assess the current risk level based on market conditions and portfolio state.
        
        This method evaluates various risk factors to determine appropriate
        guardrail settings for the current market environment.
        
        Args:
            market_context: Current market context data
            
        Returns:
            Current risk level (low, medium, high)
        """
        # Default to medium risk level
        risk_level = "medium"
        
        # Extract relevant market metrics
        volatility_index = float(market_context.get('volatility_index', 15.0))
        market_regime = str(market_context.get('market_regime', 'unknown')).lower()
        
        # Assess based on VIX
        if volatility_index >= 25.0:
            risk_level = "high"
        elif volatility_index <= 15.0:
            risk_level = "low"
            
        # Adjust based on market regime
        if market_regime == "volatile":
            risk_level = "high"
        elif market_regime == "bullish" and risk_level != "high":
            risk_level = "low"
            
        # Consider portfolio drawdown
        if self.portfolio_drawdown >= 10.0:
            risk_level = "high"
        elif self.portfolio_drawdown >= 5.0 and risk_level == "low":
            risk_level = "medium"
            
        # Log risk assessment
        self.logger.info(f"Risk assessment: level={risk_level}, VIX={volatility_index:.1f}, " +
                       f"regime={market_regime}, drawdown={self.portfolio_drawdown:.1f}%")
        
        # Update current risk level
        self.current_risk_level = risk_level
        
        return risk_level


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Available strategies
    strategies = [
        "breakout_swing",
        "momentum",
        "mean_reversion",
        "trend_following",
        "volatility_breakout",
        "option_spreads"
    ]
    
    # Create a strategy prioritizer
    prioritizer = StrategyPrioritizer(available_strategies=strategies)
    
    # Create a strategy rotator
    rotator = StrategyRotator(
        strategies=strategies,
        strategy_prioritizer=prioritizer,
        portfolio_value=100000.0
    )
    
    # Example market context (high volatility)
    market_context = {
        "market_regime": "volatile",
        "volatility_index": 28.5,
        "trend_strength": 0.3,
        "market_indices": {
            "SPY": {"daily_change_pct": -1.5, "above_200ma": True},
            "QQQ": {"daily_change_pct": -2.1, "above_200ma": False}
        },
        "sector_performance": {
            "technology": -2.3,
            "utilities": 0.8,
            "energy": -1.2
        },
        "recent_news": [
            {
                "headline": "Fed signals potential rate hike due to inflation concerns",
                "sentiment": "negative",
                "relevance": 0.9
            },
            {
                "headline": "Market volatility reaches 6-month high",
                "sentiment": "negative",
                "relevance": 0.85
            }
        ]
    }
    
    # Perform rotation
    rotation_result = rotator.rotate_strategies(market_context, force_rotation=True)
    
    # Print new allocations
    print("\nNew Strategy Allocations:")
    for strategy, allocation in rotator.get_allocations().items():
        dollar_allocation = (allocation / 100.0) * rotator.portfolio_value
        print(f"{strategy}: {allocation:.1f}% (${dollar_allocation:,.2f})")
    
    # Print rotation result
    print(f"\nRotation Status: {rotation_result['status']}")
    if rotation_result['status'] == 'success':
        print(f"Market Regime: {rotation_result['results']['market_context_summary'].get('market_regime', 'unknown')}")
        
        # Print biggest changes
        changes = rotation_result['results']['allocation_changes']
        sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)
        print("\nBiggest Allocation Changes:")
        for strategy, change in sorted_changes[:3]:
            direction = "Increased" if change > 0 else "Decreased"
            print(f"{strategy}: {direction} by {abs(change):.1f}%") 