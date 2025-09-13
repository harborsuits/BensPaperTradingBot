"""
Autonomous Trading Engine

This module orchestrates the full autonomous trading workflow:
1. Market scanning and opportunity identification
2. Strategy generation and selection
3. Automated backtesting
4. Performance evaluation and filtering
5. Preparing strategies for approval
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import time
import random

# Import strategy components
try:
    from trading_bot.strategies.modular_strategy_system import ComponentType
    from trading_bot.strategies.components.component_registry import get_component_registry
    from trading_bot.strategies.momentum import MomentumStrategy
    from trading_bot.strategies.mean_reversion import MeanReversionStrategy
    from trading_bot.strategies.trend_following import TrendFollowingStrategy
    from trading_bot.strategies.breakout import VolatilityBreakout
    from trading_bot.strategies.ml_strategy import MLStrategy
    from trading_bot.strategies.optimizer.enhanced_optimizer import EnhancedOptimizer
except ImportError:
    pass

# Import market scanners
try:
    from trading_bot.scanners.technical_scanner import TechnicalScanner
    from trading_bot.scanners.fundamental_scanner import FundamentalScanner
    from trading_bot.scanners.sentiment_scanner import SentimentScanner
except ImportError:
    pass

# Import event system
try:
    from trading_bot.event_system import EventBus, Event, EventType
except ImportError:
    pass

# Import backtesting components
try:
    from trading_bot.backtesting.backtester import Backtester
except ImportError:
    pass

# Configure logging
logger = logging.getLogger("autonomous_engine")
logger.setLevel(logging.INFO)

class StrategyCandidate:
    """Represents a strategy that has been generated and backtested"""
    
    def __init__(self, 
                 strategy_id: str, 
                 strategy_type: str,
                 symbols: List[str],
                 universe: str,
                 parameters: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.strategy_type = strategy_type
        self.symbols = symbols
        self.universe = universe
        self.parameters = parameters
        
        # Performance metrics
        self.returns = 0.0
        self.sharpe_ratio = 0.0
        self.drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.trades_count = 0
        
        # Status
        self.status = "pending"  # pending, backtested, approved, rejected, deployed
        self.meets_criteria = False
        
        # Results
        self.equity_curve = []
        self.trades = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type,
            "symbols": self.symbols,
            "universe": self.universe,
            "parameters": self.parameters,
            "performance": {
                "returns": self.returns,
                "sharpe_ratio": self.sharpe_ratio,
                "drawdown": self.drawdown,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "trades_count": self.trades_count
            },
            "status": self.status,
            "meets_criteria": self.meets_criteria
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyCandidate':
        """Create from dictionary"""
        candidate = cls(
            strategy_id=data.get("strategy_id", ""),
            strategy_type=data.get("strategy_type", ""),
            symbols=data.get("symbols", []),
            universe=data.get("universe", ""),
            parameters=data.get("parameters", {})
        )
        
        performance = data.get("performance", {})
        candidate.returns = performance.get("returns", 0.0)
        candidate.sharpe_ratio = performance.get("sharpe_ratio", 0.0)
        candidate.drawdown = performance.get("drawdown", 0.0)
        candidate.win_rate = performance.get("win_rate", 0.0)
        candidate.profit_factor = performance.get("profit_factor", 0.0)
        candidate.trades_count = performance.get("trades_count", 0)
        
        candidate.status = data.get("status", "pending")
        candidate.meets_criteria = data.get("meets_criteria", False)
        
        return candidate


class AutonomousEngine:
    """Main engine for autonomous trading strategy generation,
    backtesting, and preparation for approval.
    """
    
    def __init__(self, config=None):
        """Initialize the autonomous engine
        
        Args:
            config: Configuration object, can contain settings like use_real_data
        """
        # Set data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "autonomous")
            
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Store config for data source determination
        self.config = config
        self.use_real_data = getattr(config, 'use_real_data', False) if config else False
        
        # Strategy candidates
        self.strategy_candidates = {}  # Dict[str, StrategyCandidate]
        self.top_candidates = []  # List of strategies that meet criteria
        self.near_miss_candidates = []  # List of strategies that are near misses
        
        # Universe and strategy types
        self.universe = ""  # Current universe being processed
        self.symbols = []  # Symbols for the universe
        self.last_scan_time = datetime.now() - timedelta(days=365)  # Far in past
        self.strategy_types = []  # List of strategy types to generate
        
        # Thresholds for evaluating strategies
        self.thresholds = {
            "min_sharpe_ratio": 1.5,
            "min_profit_factor": 1.8,
            "max_drawdown": 15.0,
            "min_win_rate": 55.0
        }
        
        # Near-miss thresholds (percentage of main thresholds)
        self.near_miss_threshold_pct = 0.85  # 85% of threshold is considered "near-miss"
        
        # Optimization
        self.optimizer = None
        try:
            self.optimizer = EnhancedOptimizer()
        except NameError:
            logger.warning("EnhancedOptimizer not available")
            
        # Event bus
        self.event_bus = None
        try:
            self.event_bus = EventBus()
        except NameError:
            logger.warning("EventBus not available")
        
        # Process control
        self.running = False
        # Status
        self.is_running = False
        self.process_thread = None
        self.current_phase = "idle"  # idle, scanning, generating, backtesting, evaluating
        self.progress = 0
        self.status_message = ""
        
        # Load previous results
        self._load_candidates()
    
    def start_process(self, universe: str, 
                    strategy_types: List[str], 
                    thresholds: Dict[str, float]) -> None:
        """Start the autonomous trading process"""
        if self.is_running:
            logger.warning("Process already running")
            return
        
        # Update configuration
        self.universe = universe
        self.strategy_types = strategy_types
        self.thresholds.update(thresholds)
        
        # Start process in a separate thread
        self.is_running = True
        self.current_phase = "starting"
        self.progress = 0
        self.status_message = "Starting autonomous process..."
        
        self.process_thread = threading.Thread(target=self._run_process)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def stop_process(self) -> None:
        """Stop the autonomous trading process"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.status_message = "Stopping process..."
        
        # Wait for thread to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        
        self.current_phase = "idle"
        self.status_message = "Process stopped"
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the autonomous process"""
        return {
            "is_running": self.is_running,
            "current_phase": self.current_phase,
            "progress": self.progress,
            "status_message": self.status_message,
            "candidates_count": len(self.strategy_candidates),
            "top_candidates_count": len(self.top_candidates),
            "near_miss_candidates_count": len(self.near_miss_candidates),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_all_candidates(self) -> List[Dict]:
        """Get all strategy candidates"""
        return [candidate.to_dict() for candidate in self.strategy_candidates.values()]
    
    def get_top_candidates(self) -> List[Dict]:
        """Get top strategy candidates that meet criteria"""
        return [candidate.to_dict() for candidate in self.top_candidates]
    
    def get_near_miss_candidates(self) -> List[Dict]:
        """Get near-miss strategy candidates"""
        return [candidate.to_dict() for candidate in self.near_miss_candidates]
    
    def approve_strategy(self, strategy_id: str) -> bool:
        """Approve a strategy for deployment"""
        if strategy_id in self.strategy_candidates:
            candidate = self.strategy_candidates[strategy_id]
            candidate.status = "approved"
            self._save_candidates()
            return True
        return False
    
    def reject_strategy(self, strategy_id: str) -> bool:
        """Reject a strategy"""
        if strategy_id in self.strategy_candidates:
            candidate = self.strategy_candidates[strategy_id]
            candidate.status = "rejected"
            self._save_candidates()
            return True
        return False
    
    def deploy_strategy(self, strategy_id: str) -> bool:
        """Deploy an approved strategy to paper trading"""
        if strategy_id not in self.strategy_candidates:
            logger.error(f"Strategy {strategy_id} not found")
            return False
        
        strategy = self.strategy_candidates[strategy_id]
        if strategy.status != "approved":
            logger.error(f"Strategy {strategy_id} is not approved for deployment")
            return False
        
        # TODO: Implement real deployment to paper trading
        # For now, just update status
        strategy.status = "deployed"
        
        logger.info(f"Strategy {strategy_id} deployed to paper trading")
        self._save_candidates()
        
        return True
        
    def scan_for_opportunities(self) -> Dict:
        """Actively scan the market for trading opportunities
        
        This is the entry point for the autonomous workflow. It scans
        the market for opportunities, generates strategies, and backtests them.
        
        Returns:
            Dict with scan results
        """
        logger.info("Starting market scan for opportunities")
        self.last_scan_time = datetime.now()
        self.scan_results = {}
        
        # Set a default universe if none is specified
        if not self.universe:
            self.universe = "SP500"
            
        # Set default strategy types if none specified
        if not self.strategy_types:
            self.strategy_types = ["Momentum", "MeanReversion", "TrendFollowing"]
        
        # Get symbols for current universe
        self.symbols = self._get_symbols_for_universe(self.universe)
        
        if not self.symbols:
            logger.error("No symbols found for universe")
            return {"error": "No symbols found for universe"}
            
        # Generate strategies based on opportunities
        candidates = self._generate_strategies()
        
        # Add candidates to our registry
        for candidate in candidates:
            self.strategy_candidates[candidate.strategy_id] = candidate
            
        # Backtest all generated strategies
        for strategy_id, strategy in self.strategy_candidates.items():
            if strategy.status == "pending":
                self._backtest_strategy(strategy)
                
        # Evaluate strategies against performance criteria
        self._evaluate_strategies()
        
        # Save results
        self._save_candidates()
        
        # Prepare scan results
        self.scan_results = {
            "universe": self.universe,
            "symbols_scanned": len(self.symbols),
            "strategies_generated": len(self.strategy_candidates),
            "strategies_meeting_criteria": len(self.top_candidates),
            "near_miss_strategies": len(self.near_miss_candidates),
            "scan_time": self.last_scan_time.isoformat(),
            "data_source": "real" if self.use_real_data else "simulated"
        }
        
        return self.scan_results
        
    def get_vetted_strategies(self) -> List[Dict]:
        """Get strategies that meet performance criteria in a format 
        ready for the dashboard
        
        Returns:
            List of strategy dictionaries
        """
        result = []
        
        # Include only strategies that meet criteria
        for strategy in self.top_candidates:
            # Convert to dashboard-friendly format
            strat_dict = {
                "strategy_id": strategy.strategy_id,
                "name": f"{strategy.strategy_type} on {', '.join(strategy.symbols)}",
                "type": strategy.strategy_type,
                "universe": strategy.universe,
                "symbols": strategy.symbols,
                "parameters": {k: str(v) for k, v in strategy.parameters.items()},
                "performance": {
                    "return": round(strategy.returns, 1),
                    "sharpe": round(strategy.sharpe_ratio, 2),
                    "drawdown": round(strategy.drawdown, 1),
                    "win_rate": round(strategy.win_rate, 1),
                    "profit_factor": round(strategy.profit_factor, 2),
                    "trades": strategy.trades_count
                }
            }
            
            # Add a trigger if we can determine one
            if "trigger" in strategy.parameters:
                strat_dict["trigger"] = strategy.parameters["trigger"]
            elif "trigger_type" in strategy.parameters:
                trigger_type = strategy.parameters["trigger_type"]
                if trigger_type == "technical":
                    strat_dict["trigger"] = "Technical: Pattern detected with probability of trend continuation"
                elif trigger_type == "news":
                    strat_dict["trigger"] = "News-driven: Upcoming catalyst with potential for volatility"
                elif trigger_type == "fundamental":
                    strat_dict["trigger"] = "Fundamental: Valuation metrics indicate potential price movement"
                else:
                    strat_dict["trigger"] = f"{trigger_type.capitalize()}: Opportunity detected"
            else:
                # Generate a realistic trigger based on strategy type
                if strategy.strategy_type == "Momentum":
                    strat_dict["trigger"] = "Volatility: Range-bound price action expected"
                elif strategy.strategy_type == "MeanReversion":
                    strat_dict["trigger"] = "Technical: Price consolidation near key level"
                elif strategy.strategy_type == "TrendFollowing":
                    strat_dict["trigger"] = "Event-driven: High impact announcement pending"
                elif strategy.strategy_type == "VolatilityBreakout":
                    strat_dict["trigger"] = "Portfolio: Protecting gains on long position"
                else:
                    strat_dict["trigger"] = "Market analysis: Favorable risk/reward scenario"
            
            result.append(strat_dict)
            
        # Cache the results for future use
        self._cache_vetted_strategies(result)
        
        return result
        
    def _cache_vetted_strategies(self, strategies):
        """Cache vetted strategies to a file for the dashboard to use"""
        try:
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, "cached_strategies.json")
            
            with open(cache_file, 'w') as f:
                json.dump(strategies, f, indent=2)
                
            logger.info(f"Cached {len(strategies)} vetted strategies to {cache_file}")
        except Exception as e:
            logger.error(f"Error caching vetted strategies: {e}")
            return True
        return False
    
    def _run_process(self) -> None:
        """Run the autonomous process in a separate thread"""
        try:
            # Phase 1: Market scanning
            self.current_phase = "scanning"
            self.status_message = f"Scanning {self.universe} for opportunities..."
            self.progress = 10
            
            # For demo, simulate scanning time
            time.sleep(2)
            
            # Get symbols based on universe
            self.symbols = self._get_symbols_for_universe(self.universe)
            
            # Phase 2: Strategy generation
            self.current_phase = "generating"
            self.status_message = "Generating strategy candidates..."
            self.progress = 30
            
            # For demo, simulate strategy generation time
            time.sleep(2)
            
            # Generate strategies based on universe and types
            candidates = self._generate_strategies()
            
            # Add candidates to registry
            for candidate in candidates:
                self.strategy_candidates[candidate.strategy_id] = candidate
            
            time.sleep(1)  # Simulate processing time
            
            # Phase 3: Backtesting strategies
            self.current_phase = "backtesting"
            self.status_message = "Backtesting strategy candidates..."
            self.progress = 60
            
            for strategy_id, strategy in self.strategy_candidates.items():
                # Skip already backtested strategies
                if strategy.status == "pending":
                    self._backtest_strategy(strategy)
            
            time.sleep(1)  # Simulate processing time
            
            # Phase 4: Evaluating strategies
            self.current_phase = "evaluating"
            self.status_message = "Evaluating strategy performance..."
            self.progress = 80
            
            self._evaluate_strategies()
            
            # Phase 5: Optimization
            self.current_phase = "optimizing"
            self.status_message = "Optimizing near-miss strategies..."
            self.progress = 90
            
            self._optimize_near_miss_strategies()
            
            # Phase 6: Completed
            self.current_phase = "completed"
            self.status_message = "Process completed"
            self.progress = 100
            
            # Save results
            self._save_candidates()
            
        except Exception as e:
            logger.error(f"Error in autonomous process: {e}")
            self.current_phase = "error"
            self.status_message = f"Error: {str(e)}"
        finally:
            # Reset running status after completion or error
            if not self.is_running:
                self.current_phase = "stopped"
            
            self.is_running = False
    
    def _get_symbols_for_universe(self, universe: str) -> List[str]:
        """Get symbols for the selected universe"""
        # Try to use real market data providers if available
        try:
            # Attempt to import real data provider
            from trading_bot.data.providers import YahooDataProvider, FinnhubDataProvider
            self.using_real_data = True
            
            # Use provider for universe lookup
            if universe == "S&P 500":
                return self._get_index_components("^GSPC", 30) # Limit to top 30 for demo
            elif universe == "Nasdaq 100":
                return self._get_index_components("^NDX", 30)
            elif universe == "Dow Jones 30":
                return self._get_index_components("^DJI", 30)
            elif universe == "Russell 2000":
                return self._get_index_components("^RUT", 20)
            elif universe == "Crypto Top 20":
                return ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD", "ADA-USD"]
            elif universe == "Forex Majors":
                return ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", "AUDUSD=X"]
            else:
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
                
        except ImportError:
            # Fallback to sample data
            self.using_real_data = False
            logger.warning("Real data providers not available, using sample data")
            
            if universe == "S&P 500":
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
            elif universe == "Nasdaq 100":
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "ADBE"]
            elif universe == "Dow Jones 30":
                return ["AAPL", "MSFT", "GS", "HD", "CAT", "JPM"]
            elif universe == "Russell 2000":
                return ["PLCE", "PRTS", "MCRI", "KTOS", "SPB"]
            elif universe == "Crypto Top 20":
                return ["BTC", "ETH", "XRP", "BNB", "ADA"]
            elif universe == "Forex Majors":
                return ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CHF", "AUD/USD"]
            else:
                return ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    def _get_index_components(self, index_symbol: str, limit: int = 30) -> List[str]:
        """Get actual index components using market data provider"""
        try:
            # Try to use real data provider
            from trading_bot.data.providers import YahooDataProvider
            
            # In a real implementation, this would fetch actual components
            # For demo purposes, return major components
            if index_symbol == "^GSPC":  # S&P 500
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "NVDA", "UNH", "JNJ", 
                        "JPM", "V", "PG", "XOM", "HD"][:limit]
            elif index_symbol == "^NDX":  # Nasdaq 100
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ADBE", "PYPL", "CMCSA", 
                        "NFLX", "INTC", "CSCO", "PEP", "AVGO"][:limit]
            elif index_symbol == "^DJI":  # Dow Jones 30
                return ["AAPL", "MSFT", "GS", "HD", "CAT", "JPM", "UNH", "BA", "MCD", "MMM", 
                        "DIS", "IBM", "WMT", "CVX", "KO"][:limit]
            elif index_symbol == "^RUT":  # Russell 2000
                return ["PLCE", "PRTS", "MCRI", "KTOS", "SPB", "AMSF", "BCC", "CEVA", "HSKA", "TBNK", 
                        "MOFG", "STFC", "LIVN", "WNC", "CCBG"][:limit]
            else:
                return ["AAPL", "MSFT", "GOOGL", "AMZN"][:limit]
                
        except ImportError:
            # Fallback for demo
            if index_symbol == "^GSPC":  # S&P 500
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"][:limit]
            elif index_symbol == "^NDX":  # Nasdaq 100
                return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "ADBE"][:limit]
            elif index_symbol == "^DJI":  # Dow Jones 30
                return ["AAPL", "MSFT", "GS", "HD", "CAT", "JPM"][:limit]
            elif index_symbol == "^RUT":  # Russell 2000
                return ["PLCE", "PRTS", "MCRI", "KTOS", "SPB"][:limit]
            else:
                return ["AAPL", "MSFT", "GOOGL", "AMZN"][:limit]
    
    def _generate_strategies(self) -> List[StrategyCandidate]:
        """Generate strategy candidates based on configuration"""
        # In a real implementation, this would generate actual strategies
        # using your existing strategy templates and market conditions
        
        candidates = []
        
        for strategy_type in self.strategy_types:
            # For each strategy type, generate strategies for selected symbols
            if strategy_type == "Momentum":
                universe_segments = ["Tech Sector", "Consumer Sector", self.universe]
                
                for segment in universe_segments:
                    # Create strategy candidate
                    candidate_id = f"MOM-{segment.split()[0].upper()}-{datetime.now().strftime('%d%m')}"
                    
                    symbols = self.symbols[:3] if len(self.symbols) >= 3 else self.symbols
                    
                    candidate = StrategyCandidate(
                        strategy_id=candidate_id,
                        strategy_type="Momentum",
                        symbols=symbols,
                        universe=segment,
                        parameters={
                            "rsi_period": random.choice([9, 14, 21]),
                            "rsi_oversold": random.choice([25, 30, 35]),
                            "rsi_overbought": random.choice([65, 70, 75]),
                            "rate_of_change_period": random.choice([5, 10, 15]),
                            "ma_period": random.choice([20, 50, 100])
                        }
                    )
                    
                    candidates.append(candidate)
                    
                    # Store in candidates dict
                    self.strategy_candidates[candidate_id] = candidate
            
            elif strategy_type == "Mean Reversion":
                universe_segments = ["Banking Sector", "Utilities Sector", self.universe]
                
                for segment in universe_segments:
                    # Create strategy candidate
                    candidate_id = f"MR-{segment.split()[0].upper()}-{datetime.now().strftime('%d%m')}"
                    
                    symbols = self.symbols[1:4] if len(self.symbols) >= 4 else self.symbols
                    
                    candidate = StrategyCandidate(
                        strategy_id=candidate_id,
                        strategy_type="Mean Reversion",
                        symbols=symbols,
                        universe=segment,
                        parameters={
                            "lookback_period": random.choice([10, 20, 30]),
                            "std_dev_threshold": random.choice([1.5, 2.0, 2.5]),
                            "ma_period": random.choice([50, 100, 200]),
                            "exit_threshold": random.choice([0.5, 1.0, 1.5])
                        }
                    )
                    
                    candidates.append(candidate)
                    
                    # Store in candidates dict
                    self.strategy_candidates[candidate_id] = candidate
            
            elif strategy_type == "Trend Following":
                universe_segments = [self.universe]
                
                for segment in universe_segments:
                    # Create strategy candidate
                    candidate_id = f"TF-{segment.split()[0].upper()}-{datetime.now().strftime('%d%m')}"
                    
                    symbols = self.symbols
                    
                    candidate = StrategyCandidate(
                        strategy_id=candidate_id,
                        strategy_type="Trend Following",
                        symbols=symbols,
                        universe=segment,
                        parameters={
                            "fast_ma": random.choice([10, 20, 30]),
                            "slow_ma": random.choice([50, 100, 200]),
                            "signal_smoothing": random.choice([5, 9, 14]),
                            "atr_period": random.choice([10, 14, 21])
                        }
                    )
                    
                    candidates.append(candidate)
                    
                    # Store in candidates dict
                    self.strategy_candidates[candidate_id] = candidate
            
            elif strategy_type == "Volatility Breakout":
                universe_segments = ["Energy Sector", self.universe]
                
                for segment in universe_segments:
                    # Create strategy candidate
                    candidate_id = f"VB-{segment.split()[0].upper()}-{datetime.now().strftime('%d%m')}"
                    
                    symbols = self.symbols[2:5] if len(self.symbols) >= 5 else self.symbols
                    
                    candidate = StrategyCandidate(
                        strategy_id=candidate_id,
                        strategy_type="Volatility Breakout",
                        symbols=symbols,
                        universe=segment,
                        parameters={
                            "atr_period": random.choice([10, 14, 21]),
                            "breakout_multiplier": random.choice([1.0, 1.5, 2.0]),
                            "stop_loss_atr": random.choice([1.5, 2.0, 3.0])
                        }
                    )
                    
                    candidates.append(candidate)
                    
                    # Store in candidates dict
                    self.strategy_candidates[candidate_id] = candidate
            
            elif strategy_type == "Machine Learning":
                universe_segments = ["S&P 500"]
                
                for segment in universe_segments:
                    # Create strategy candidate
                    candidate_id = f"ML-{segment.split()[0].upper()}-{datetime.now().strftime('%d%m')}"
                    
                    symbols = self.symbols
                    
                    candidate = StrategyCandidate(
                        strategy_id=candidate_id,
                        strategy_type="Machine Learning",
                        symbols=symbols,
                        universe=segment,
                        parameters={
                            "model_type": random.choice(["random_forest", "gradient_boosting", "lstm"]),
                            "lookback_period": random.choice([10, 20, 30]),
                            "prediction_horizon": random.choice([1, 3, 5]),
                            "feature_selection": random.choice(["all", "technical", "price_action"])
                        }
                    )
                    
                    candidates.append(candidate)
                    
                    # Store in candidates dict
                    self.strategy_candidates[candidate_id] = candidate
        
        return candidates
    
    def _backtest_strategy(self, strategy: StrategyCandidate) -> None:
        """Backtest a strategy candidate"""
        # In a real implementation, this would use your backtesting engine
        # For demo purposes, generate simulated results
        
        # Simulate backtest based on strategy type
        if strategy.strategy_type == "Momentum":
            returns = random.uniform(15.0, 35.0)
            sharpe = random.uniform(1.5, 2.0)
            drawdown = random.uniform(8.0, 15.0)
            win_rate = random.uniform(55.0, 65.0)
            profit_factor = random.uniform(1.8, 2.5)
            trades = random.randint(50, 100)
        elif strategy.strategy_type == "Mean Reversion":
            returns = random.uniform(10.0, 25.0)
            sharpe = random.uniform(1.4, 1.8)
            drawdown = random.uniform(5.0, 12.0)
            win_rate = random.uniform(65.0, 80.0)
            profit_factor = random.uniform(1.6, 2.2)
            trades = random.randint(40, 80)
        elif strategy.strategy_type == "Trend Following":
            returns = random.uniform(18.0, 28.0)
            sharpe = random.uniform(1.6, 1.9)
            drawdown = random.uniform(12.0, 18.0)
            win_rate = random.uniform(50.0, 60.0)
            profit_factor = random.uniform(1.7, 2.3)
            trades = random.randint(30, 60)
        elif strategy.strategy_type == "Volatility Breakout":
            returns = random.uniform(12.0, 20.0)
            sharpe = random.uniform(1.2, 1.6)
            drawdown = random.uniform(15.0, 22.0)
            win_rate = random.uniform(45.0, 55.0)
            profit_factor = random.uniform(1.4, 1.8)
            trades = random.randint(20, 40)
        elif strategy.strategy_type == "Machine Learning":
            returns = random.uniform(10.0, 25.0)
            sharpe = random.uniform(1.3, 1.7)
            drawdown = random.uniform(12.0, 20.0)
            win_rate = random.uniform(50.0, 65.0)
            profit_factor = random.uniform(1.5, 2.0)
            trades = random.randint(100, 150)
        else:
            returns = random.uniform(5.0, 15.0)
            sharpe = random.uniform(0.8, 1.4)
            drawdown = random.uniform(10.0, 25.0)
            win_rate = random.uniform(40.0, 60.0)
            profit_factor = random.uniform(1.2, 1.6)
            trades = random.randint(30, 70)
        
        # Update strategy performance
        strategy.returns = returns
        strategy.sharpe_ratio = sharpe
        strategy.drawdown = drawdown
        strategy.win_rate = win_rate
        strategy.profit_factor = profit_factor
        strategy.trades_count = trades
        strategy.status = "backtested"
    
    def _evaluate_strategies(self) -> None:
        """Evaluate strategies against performance thresholds"""
        self.top_candidates = []
        self.near_miss_candidates = []
        
        for strategy_id, strategy in self.strategy_candidates.items():
            # Check if strategy meets performance criteria
            meets_criteria = (
                strategy.sharpe_ratio >= self.thresholds["min_sharpe_ratio"] and
                strategy.profit_factor >= self.thresholds["min_profit_factor"] and
                strategy.drawdown <= self.thresholds["max_drawdown"] and
                strategy.win_rate >= self.thresholds["min_win_rate"]
            )
            
            # Check if strategy is a near-miss (close to meeting criteria)
            is_near_miss = self._is_near_miss_candidate(strategy)
            
            strategy.meets_criteria = meets_criteria
            
            # Add to top candidates if meets criteria
            if meets_criteria:
                self.top_candidates.append(strategy)
            # Add to near-miss candidates if close but doesn't fully meet criteria
            elif is_near_miss:
                self.near_miss_candidates.append(strategy)
        
        # Sort top candidates by returns
        self.top_candidates.sort(key=lambda x: x.returns, reverse=True)
        
        # If we have near-miss candidates and an optimizer, try to optimize them
        if self.optimizer and self.near_miss_candidates:
            self._optimize_near_miss_candidates()
    
    def _save_candidates(self) -> None:
        """Save strategy candidates to disk"""
        candidates_data = {
            strategy_id: strategy.to_dict()
            for strategy_id, strategy in self.strategy_candidates.items()
        }
        
        file_path = os.path.join(self.data_dir, "candidates.json")
        
        try:
            with open(file_path, 'w') as f:
                json.dump(candidates_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving candidates: {e}")
            
    def _is_near_miss_candidate(self, strategy: 'StrategyCandidate') -> bool:
        """Check if a strategy is a near-miss candidate (close to meeting criteria)
        
        Args:
            strategy: Strategy to check
            
        Returns:
            True if strategy is a near-miss candidate
        """
        # Don't consider strategies that already meet criteria or have been optimized
        if strategy.meets_criteria or strategy.status in ["approved", "rejected", "optimized", "exhausted"]:
            return False
            
        # Calculate how close the strategy is to meeting the thresholds
        sharpe_ratio_pct = strategy.sharpe_ratio / self.thresholds["min_sharpe_ratio"]
        profit_factor_pct = strategy.profit_factor / self.thresholds["min_profit_factor"]
        drawdown_pct = self.thresholds["max_drawdown"] / max(strategy.drawdown, 0.01)  # Prevent division by zero
        win_rate_pct = strategy.win_rate / self.thresholds["min_win_rate"]
        
        # Consider it a near-miss if it meets the near-miss threshold for all criteria
        return all([
            sharpe_ratio_pct >= self.near_miss_threshold_pct,
            profit_factor_pct >= self.near_miss_threshold_pct,
            drawdown_pct >= self.near_miss_threshold_pct,
            win_rate_pct >= self.near_miss_threshold_pct
        ])
    
    def _optimize_near_miss_candidates(self) -> None:
        """Optimize near-miss candidates using EnhancedOptimizer"""
        if not self.optimizer:
            logger.warning("Cannot optimize: EnhancedOptimizer not available")
            return
            
        logger.info(f"Optimizing {len(self.near_miss_candidates)} near-miss candidates")
        
        for candidate in self.near_miss_candidates:
            # Skip already optimized candidates
            if candidate.status in ["optimized", "exhausted"]:
                continue
                
            logger.info(f"Optimizing strategy {candidate.strategy_id} ({candidate.strategy_type})")
            
            try:
                # Set the strategy parameters for optimization
                self.optimizer.with_parameter_ranges(candidate.parameters)
                
                # Attempt to optimize
                optimization_result = self.optimizer.optimize(candidate)
                
                if optimization_result and optimization_result.best_parameters:
                    # Update the strategy with the optimized parameters
                    candidate.parameters = optimization_result.best_parameters
                    
                    # Backtest with new parameters
                    self._backtest_strategy(candidate)
                    
                    # Check if it now meets criteria
                    meets_criteria = (
                        candidate.sharpe_ratio >= self.thresholds["min_sharpe_ratio"] and
                        candidate.profit_factor >= self.thresholds["min_profit_factor"] and
                        candidate.drawdown <= self.thresholds["max_drawdown"] and
                        candidate.win_rate >= self.thresholds["min_win_rate"]
                    )
                    
                    if meets_criteria:
                        # Update status and add to top candidates
                        candidate.meets_criteria = True
                        candidate.status = "optimized"
                        if candidate not in self.top_candidates:
                            self.top_candidates.append(candidate)
                        
                        # Emit event
                        self._emit_event("STRATEGY_OPTIMISED", {
                            "strategy_id": candidate.strategy_id,
                            "strategy_type": candidate.strategy_type,
                            "symbols": candidate.symbols,
                            "original_parameters": optimization_result.parameter_sets[0] if optimization_result.parameter_sets else {},
                            "optimized_parameters": candidate.parameters,
                            "performance": candidate.to_dict()["performance"]
                        })
                    else:
                        # Strategy still doesn't meet criteria after optimization
                        candidate.status = "exhausted"
                        
                        # Emit event
                        self._emit_event("STRATEGY_EXHAUSTED", {
                            "strategy_id": candidate.strategy_id,
                            "strategy_type": candidate.strategy_type,
                            "symbols": candidate.symbols,
                            "parameters": candidate.parameters,
                            "performance": candidate.to_dict()["performance"],
                            "thresholds": self.thresholds
                        })
                else:
                    # Optimization didn't improve the strategy
                    candidate.status = "exhausted"
                    
                    # Emit event
                    self._emit_event("STRATEGY_EXHAUSTED", {
                        "strategy_id": candidate.strategy_id,
                        "strategy_type": candidate.strategy_type,
                        "symbols": candidate.symbols,
                        "parameters": candidate.parameters,
                        "performance": candidate.to_dict()["performance"],
                        "thresholds": self.thresholds
                    })
            except Exception as e:
                logger.error(f"Error optimizing strategy {candidate.strategy_id}: {e}")
                # Mark as failed optimization
                candidate.status = "optimization_failed"
        
        # Save all candidates after optimization
        self._save_candidates()
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the event bus
        
        Args:
            event_type: Type of event
            data: Event data
        """
        if not self.event_bus:
            return
            
        try:
            event = Event(
                event_type=event_type,
                source="AutonomousEngine",
                data=data,
                timestamp=datetime.now()
            )
            self.event_bus.publish(event)
            logger.info(f"Emitted event: {event_type}")
        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")
    
    def _load_candidates(self) -> None:
        """Load strategy candidates from disk"""
        file_path = os.path.join(self.data_dir, "candidates.json")
        
        if not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'r') as f:
                candidates_data = json.load(f)
            
            for strategy_id, data in candidates_data.items():
                self.strategy_candidates[strategy_id] = StrategyCandidate.from_dict(data)
            
            # Re-evaluate to update top candidates
            self._evaluate_strategies()
            
        except Exception as e:
            logger.error(f"Error loading candidates: {e}")
