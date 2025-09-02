"""
Market Regime Integration Module

This module integrates the advanced market regime analysis and strategy selection system
with the trade execution engine and journaling system, creating a complete end-to-end
trading platform with sophisticated market awareness.
"""

import logging
import datetime
import json
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from enum import Enum

# Import trade execution components
from trade_executor import TradeExecutor, TradeResult, TradeType, OrderSide, OrderType
from trade_executor_journal_integration import JournaledTradeExecutor, apply_journal_to_executor
from strategy_loader import StrategyLoader

# Import market regime components
from market_regime.market_context import MarketContext, MarketRegime, FilterType
from market_regime.regime_analyzer import RegimeAnalyzer
from market_regime.strategy_selector import StrategySelector
from market_regime.signal_generator import SignalGenerator


class MarketAwareTrader:
    """
    Comprehensive trading system that combines market regime analysis,
    strategy selection, trade execution, and trade journaling.
    """
    
    def __init__(self, 
                loader: StrategyLoader, 
                executor: TradeExecutor,
                journal_dir: str = "journal",
                config: Dict[str, Any] = None):
        """
        Initialize the market-aware trading system
        
        Args:
            loader: Strategy loader for accessing strategies and configurations
            executor: Trade executor for executing trades
            journal_dir: Directory for journal data
            config: Optional configuration dictionary
        """
        self.loader = loader
        self.raw_executor = executor
        
        # Initialize journaled executor for enhanced trade journaling
        self.executor = apply_journal_to_executor(executor, journal_dir)
        
        # Configuration
        self.config = config or {}
        
        # Initialize market regime components
        self.regime_analyzer = RegimeAnalyzer(loader)
        self.strategy_selector = StrategySelector(loader)
        self.signal_generator = SignalGenerator(loader)
        
        # Set up logging
        self.logger = logging.getLogger("MarketAwareTrader")
        
        # Universe of tradable symbols
        self.universe = self.config.get("universe", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN"])
        
        # Position limits
        self.max_positions = self.config.get("max_positions", 5)
        self.max_trades_per_day = self.config.get("max_trades_per_day", 10)
        
        # Performance tracking
        self.daily_trades_count = 0
        self.regime_history = []
        self.last_analysis_time = None
        self.current_regime = None
        
        self.logger.info("Market-aware trading system initialized")
    
    def create_market_context(self, data: Dict[str, Any]) -> MarketContext:
        """
        Create a rich market context from input data
        
        Args:
            data: Raw input data
            
        Returns:
            MarketContext object
        """
        # Create context with default values
        context = MarketContext()
        
        # Update with provided data
        for key, value in data.items():
            if hasattr(context, key):
                setattr(context, key, value)
        
        # Set current date if not provided
        if not data.get("current_date"):
            context.current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
        # Parse date to extract month, quarter, etc. if not already set
        if not data.get("current_month"):
            try:
                date_obj = datetime.datetime.strptime(context.current_date, "%Y-%m-%d")
                context.current_month = date_obj.month
                context.current_quarter = (date_obj.month - 1) // 3 + 1
                context.day_of_week = date_obj.weekday()
            except:
                # If parsing fails, use current date
                now = datetime.datetime.now()
                context.current_month = now.month
                context.current_quarter = (now.month - 1) // 3 + 1
                context.day_of_week = now.weekday()
        
        return context
    
    def analyze_market_regime(self, context_data: Dict[str, Any]) -> str:
        """
        Analyze the current market regime
        
        Args:
            context_data: Market context data
            
        Returns:
            String representing the current market regime
        """
        # Convert raw data to rich context
        context = self.create_market_context(context_data)
        
        # Analyze the current regime
        regime = self.regime_analyzer.analyze_regime(context)
        
        # Record regime to history
        timestamp = datetime.datetime.now().isoformat()
        self.regime_history.append((timestamp, regime))
        if len(self.regime_history) > 60:  # Keep last 60 data points
            self.regime_history.pop(0)
        
        # Update current regime
        self.current_regime = regime
        self.last_analysis_time = timestamp
        
        self.logger.info(f"Current market regime: {regime}")
        return regime
    
    def select_strategies(self, regime: str, context: MarketContext) -> List[str]:
        """
        Select appropriate strategies for the current regime
        
        Args:
            regime: Current market regime
            context: Market context
            
        Returns:
            List of selected strategy names
        """
        # Get strategy suite for the regime
        suite = self.strategy_selector.select_strategy_suite(regime, context)
        
        # Get active strategies and their allocations
        active_strategies = suite.get("active_strategies", [])
        allocations = suite.get("allocation", {})
        
        # Rank strategies based on current conditions
        ranked_strategies = self.strategy_selector.rank_strategies(
            strategies=active_strategies,
            context=context,
            allocation=allocations
        )
        
        # Filter to strategies with positive scores
        viable_strategies = [name for name, score in ranked_strategies if score > 0]
        
        # Limit to maximum number of strategies
        max_strategies = min(len(viable_strategies), 3)  # Limit to top 3
        selected_strategies = viable_strategies[:max_strategies]
        
        self.logger.info(f"Selected strategies: {selected_strategies}")
        return selected_strategies
    
    def select_symbols(self, 
                     strategies: List[str], 
                     context: MarketContext) -> List[str]:
        """
        Select appropriate symbols for the selected strategies
        
        Args:
            strategies: List of selected strategy names
            context: Market context
            
        Returns:
            List of selected symbols
        """
        # Get strategies with symbol preferences
        symbol_preferences = {}
        for strategy_name in strategies:
            strategy = self.loader.get_strategy(strategy_name)
            if strategy and "symbols" in strategy:
                symbol_preferences[strategy_name] = strategy["symbols"]
        
        # If strategies have explicit symbol preferences, use those
        if symbol_preferences:
            # Flatten and deduplicate
            preferred_symbols = []
            for symbols in symbol_preferences.values():
                preferred_symbols.extend(symbols)
            
            # Filter to symbols in our universe
            selected = list(set(s for s in preferred_symbols if s in self.universe))
            
            # If we have preferences, return those (up to 5)
            if selected:
                return selected[:5]
        
        # Otherwise use volatility-based selection
        if hasattr(context, "volatility_regime") and callable(context.volatility_regime):
            vol_regime = context.volatility_regime()
            if vol_regime == "high":
                # In high volatility, focus on liquid ETFs
                return [s for s in self.universe if s in ["SPY", "QQQ", "IWM", "DIA"]]
            else:
                # In normal/low volatility, include individual stocks
                return self.universe[:min(len(self.universe), 5)]
        
        # Fallback to default universe (up to 5 symbols)
        return self.universe[:min(len(self.universe), 5)]
    
    def generate_signals(self, 
                      strategy_names: List[str], 
                      context: MarketContext,
                      price_data: Dict[str, float],
                      symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Generate trade signals for the selected strategies and symbols
        
        Args:
            strategy_names: List of strategy names
            context: Market context
            price_data: Dictionary of symbol to price data
            symbols: List of target symbols
            
        Returns:
            List of trade signal dictionaries
        """
        return self.signal_generator.generate_signals(
            strategy_names=strategy_names,
            context=context,
            price_data=price_data,
            symbols=symbols
        )
    
    def execute_signals(self, signals: List[Dict[str, Any]]) -> List[TradeResult]:
        """
        Execute generated trade signals
        
        Args:
            signals: List of trade signal dictionaries
            
        Returns:
            List of TradeResult objects
        """
        results = []
        
        # Execute each signal
        for signal in signals:
            # Add market regime information to signal
            if self.current_regime:
                signal["market_regime"] = self.current_regime
            
            # Execute trade using the journaled executor
            result = self.raw_executor.route_trade(signal)
            results.append(result)
            
            # Update daily trade count
            if result.success:
                self.daily_trades_count += 1
                
                # Add regime information to journal
                trade_id = result.trade_id
                self._add_regime_to_journal(trade_id)
        
        return results
    
    def _add_regime_to_journal(self, trade_id: str):
        """
        Add market regime information to journal entry
        
        Args:
            trade_id: ID of the trade to enhance with regime data
        """
        if not hasattr(self.executor, "journal") or not self.current_regime:
            return
            
        # Create market regime context data
        regime_data = {
            "market_regime": {
                "primary_regime": self.current_regime,
                "analysis_timestamp": self.last_analysis_time or datetime.datetime.now().isoformat(),
                "regime_characteristics": self._get_regime_characteristics(self.current_regime)
            }
        }
        
        # Add to journal using the journal integration
        if hasattr(self.executor, "journal"):
            try:
                self.executor.journal.add_market_context(trade_id, regime_data)
                self.logger.debug(f"Added regime data to journal for trade {trade_id}")
            except Exception as e:
                self.logger.error(f"Error adding regime data to journal: {e}")
    
    def _get_regime_characteristics(self, regime: str) -> Dict[str, Any]:
        """
        Get characteristics of a market regime
        
        Args:
            regime: Market regime name
            
        Returns:
            Dictionary of regime characteristics
        """
        # Default characteristics
        characteristics = {
            "volatility_expectation": "normal",
            "directional_bias": "neutral",
            "sector_rotation": "moderate",
            "typical_duration": "unknown"
        }
        
        # Customize based on regime
        if regime == MarketRegime.BULLISH.value:
            characteristics.update({
                "volatility_expectation": "low to moderate",
                "directional_bias": "positive",
                "sector_rotation": "growth-oriented",
                "typical_duration": "months to years"
            })
        elif regime == MarketRegime.BEARISH.value:
            characteristics.update({
                "volatility_expectation": "high",
                "directional_bias": "negative",
                "sector_rotation": "defensive",
                "typical_duration": "weeks to months"
            })
        elif regime == MarketRegime.SIDEWAYS.value:
            characteristics.update({
                "volatility_expectation": "low",
                "directional_bias": "neutral",
                "sector_rotation": "minimal",
                "typical_duration": "weeks to months"
            })
        elif regime == MarketRegime.HIGH_VOLATILITY.value:
            characteristics.update({
                "volatility_expectation": "very high",
                "directional_bias": "variable",
                "sector_rotation": "rapid",
                "typical_duration": "days to weeks"
            })
        
        return characteristics
    
    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get information about the current market regime
        
        Returns:
            Dictionary with current regime information
        """
        if not self.current_regime:
            return {
                "regime": "unknown",
                "analysis_time": None
            }
            
        return {
            "regime": self.current_regime,
            "analysis_time": self.last_analysis_time,
            "characteristics": self._get_regime_characteristics(self.current_regime),
            "recent_history": self.regime_history[-5:] if len(self.regime_history) >= 5 else self.regime_history
        }
    
    def get_regime_history(self, days: int = 30) -> List[Tuple[str, str]]:
        """
        Get recent regime history
        
        Args:
            days: Number of days of history to return
            
        Returns:
            List of (timestamp, regime) tuples
        """
        return self.regime_history[-days:]
    
    def reset_daily_counters(self):
        """Reset daily counters for trades"""
        self.daily_trades_count = 0
        self.logger.info("Daily trade counters reset")
    
    def run_full_cycle(self, context_data: Dict[str, Any], price_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a complete market analysis and trading cycle
        
        Args:
            context_data: Market context data
            price_data: Dictionary of symbol to price data
            
        Returns:
            Dictionary with results of the trading cycle
        """
        # Convert raw data to rich context
        context = self.create_market_context(context_data)
        
        # 1. Analyze market regime
        regime = self.analyze_market_regime(context_data)
        
        # 2. Select strategies based on regime
        selected_strategies = self.select_strategies(regime, context)
        
        if not selected_strategies:
            self.logger.info("No strategies selected for current regime")
            return {
                "regime": regime,
                "strategies_selected": 0,
                "signals_generated": 0,
                "trades_executed": 0,
                "execution_results": []
            }
        
        # 3. Select symbols to trade
        selected_symbols = self.select_symbols(selected_strategies, context)
        
        if not selected_symbols:
            self.logger.info("No symbols selected for trading")
            return {
                "regime": regime,
                "strategies_selected": len(selected_strategies),
                "selected_strategies": selected_strategies,
                "symbols_selected": 0,
                "signals_generated": 0,
                "trades_executed": 0,
                "execution_results": []
            }
        
        # 4. Generate trade signals
        signals = self.generate_signals(
            strategy_names=selected_strategies,
            context=context,
            price_data=price_data,
            symbols=selected_symbols
        )
        
        if not signals:
            self.logger.info("No trade signals generated")
            return {
                "regime": regime,
                "strategies_selected": len(selected_strategies),
                "selected_strategies": selected_strategies,
                "symbols_selected": len(selected_symbols),
                "selected_symbols": selected_symbols,
                "signals_generated": 0,
                "trades_executed": 0,
                "execution_results": []
            }
        
        # 5. Execute trade signals
        results = self.execute_signals(signals)
        
        # 6. Compile results
        successful_trades = [r for r in results if r.success]
        
        return {
            "regime": regime,
            "strategies_selected": len(selected_strategies),
            "selected_strategies": selected_strategies,
            "symbols_selected": len(selected_symbols),
            "selected_symbols": selected_symbols,
            "signals_generated": len(signals),
            "trades_executed": len(successful_trades),
            "execution_results": [r.to_dict() for r in results]
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on system activity
        
        Returns:
            Dictionary with detailed report data
        """
        # Get current market regime data
        regime_data = self.get_current_regime()
        
        # Get trade journal analytics via executor
        journal_analytics = {}
        if hasattr(self.executor, "get_performance_report"):
            try:
                journal_analytics = self.executor.get_performance_report()
            except Exception as e:
                self.logger.error(f"Error getting journal analytics: {e}")
        
        # Get executor performance metrics
        execution_metrics = self.raw_executor.get_performance_metrics()
        
        # Get strategy performance history
        strategy_performance = {}
        if hasattr(self.strategy_selector, "strategy_performance"):
            strategy_performance = {
                name: perf[-5:] for name, perf in self.strategy_selector.strategy_performance.items()
                if perf
            }
        
        # Compile report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "market_regime": regime_data,
            "regime_history": self.regime_history[-30:],
            "execution_metrics": execution_metrics,
            "journal_analytics": journal_analytics,
            "strategy_performance": strategy_performance,
            "daily_trades": self.daily_trades_count
        }
        
        return report


def create_market_aware_trader(loader=None, executor=None, journal_dir="journal", config=None):
    """
    Factory function to create a market-aware trader with appropriate defaults
    
    Args:
        loader: StrategyLoader instance (created if None)
        executor: TradeExecutor instance (created if None)
        journal_dir: Directory for journal data
        config: Configuration dictionary
        
    Returns:
        MarketAwareTrader instance
    """
    # Create loader if not provided
    if loader is None:
        loader = StrategyLoader()
        loader.load_all()
    
    # Create executor if not provided
    if executor is None:
        executor = TradeExecutor(
            loader=loader,
            account_balance=100000,
            risk_mode="balanced"
        )
    
    # Create and return trader
    return MarketAwareTrader(
        loader=loader,
        executor=executor,
        journal_dir=journal_dir,
        config=config
    ) 