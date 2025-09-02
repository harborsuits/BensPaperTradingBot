#!/usr/bin/env python3
"""
Forex EvoTrader - Integration Module

This is the main integration component that brings together all Forex-specific modules:
- Forex News Guard
- Forex Pair Manager
- Forex Pip Logger
- Session Performance Tracker

It coordinates the evolution, evaluation, and deployment of Forex strategies
while integrating with BenBot as the master decision maker.
"""

import os
import sys
import yaml
import logging
import json
import datetime
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Tuple, Union, Optional, Any
import uuid

# Import all Forex-specific components
from forex_news_guard import NewsGuard
from forex_pair_manager import ForexPairManager, ForexPair
from forex_pip_logger import ForexPipLogger
from forex_session_manager import ForexSessionManager
from session_performance_tracker import SessionPerformanceTracker
from session_performance_db import SessionPerformanceDB

# Import evolution and backtest modules
from forex_evotrader_evolution import ForexStrategyEvolution
from forex_evotrader_backtest import ForexStrategyBacktest

from forex_smart_session import SmartSessionAnalyzer
from forex_smart_pips import SmartPipAnalyzer
from forex_smart_news import SmartNewsAnalyzer
from forex_smart_compliance import SmartComplianceMonitor
from forex_smart_benbot import SmartBenBotConnector
from forex_smart_integration import ForexSmartIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_evotrader')


class ForexEvoTrader:
    """
    Main integration class for Forex-optimized evolutionary trading.
    Coordinates all components and manages interaction with BenBot.
    """
    
    def __init__(self, config_path: str = 'forex_evotrader_config.yaml'):
        """
        Initialize the Forex EvoTrader integration component.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.benbot_endpoint = self.config.get('benbot', {}).get('endpoint', None)
        
        # Initialize components
        self.pair_manager = ForexPairManager(self.config.get('pair_config', 'forex_pairs.yaml'))
        self.news_guard = NewsGuard(self.config.get('news_config', None))
        self.session_manager = ForexSessionManager()
        self.pip_logger = ForexPipLogger(self.pair_manager)
        self.performance_tracker = SessionPerformanceTracker(
            self.config.get('session_db', 'session_performance.db'),
            self.benbot_endpoint
        )
        
        # Setup BenBot integration
        if self.benbot_endpoint:
            self.pip_logger.enable_benbot(self.benbot_endpoint)
            self.benbot_available = self._check_benbot_connection()
            logger.info(f"BenBot integration {'enabled' if self.benbot_available else 'unavailable'}")
        else:
            self.benbot_available = False
        
        # Create directories if they don't exist
        for dir_key in ['data_dir', 'results_dir', 'reports_dir', 'logs_dir']:
            if dir_key in self.config:
                os.makedirs(self.config[dir_key], exist_ok=True)
        
        logger.info("Forex EvoTrader integration module initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from yaml file."""
        default_config = {
            'benbot': {
                'endpoint': None,
                'priority_override': True,
                'report_frequency_minutes': 30
            },
            'pair_config': 'forex_pairs.yaml',
            'news_config': None,
            'session_db': 'session_performance.db',
            'data_dir': 'data',
            'results_dir': 'results',
            'reports_dir': 'reports',
            'logs_dir': 'logs',
            'evolution': {
                'population_size': 50,
                'generations': 20,
                'crossover_rate': 0.7,
                'mutation_rate': 0.3,
                'elite_size': 5,
                'session_fitness_weight': 0.3,
                'pip_fitness_weight': 0.7
            },
            'strategy_templates': 'forex_strategy_templates.yaml',
            'prop_risk_profile': 'prop_risk_profile.yaml'
        }
        
        # Load user config if it exists
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    
                # Merge configs, user config takes precedence
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def _check_benbot_connection(self) -> bool:
        """Check if BenBot is available."""
        if not self.benbot_endpoint:
            return False
            
        try:
            import requests
            response = requests.get(f"{self.benbot_endpoint.rstrip('/')}/ping", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"BenBot connection failed: {e}")
            return False
    
    def _consult_benbot(self, action: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consult BenBot for decisions or to report data.
        
        Args:
            action: Action type (e.g., 'evolve', 'backtest', 'trade')
            data: Data to send to BenBot
            
        Returns:
            BenBot response or default response if unavailable
        """
        if not self.benbot_available:
            return {'status': 'unavailable', 'proceed': True, 'message': 'BenBot unavailable'}
            
        try:
            import requests
            
            payload = {
                'source': 'EvoTrader',
                'module': 'ForexEvoTrader',
                'action': action,
                'data': data,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            endpoint = f"{self.benbot_endpoint.rstrip('/')}/decision"
            response = requests.post(endpoint, json=payload, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"BenBot consultation failed: {response.status_code}")
                return {'status': 'error', 'proceed': True, 'message': f"BenBot error {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error consulting BenBot: {e}")
            return {'status': 'error', 'proceed': True, 'message': f"BenBot error: {str(e)}"}
    
    def load_strategy_template(self, template_name: str) -> Dict[str, Any]:
        """
        Load a strategy template from the templates file.
        
        Args:
            template_name: Name of the template to load
            
        Returns:
            Strategy template configuration
        """
        template_path = self.config.get('strategy_templates', 'forex_strategy_templates.yaml')
        
        try:
            with open(template_path, 'r') as f:
                templates = yaml.safe_load(f)
                
            if template_name in templates:
                return templates[template_name]
            else:
                logger.error(f"Strategy template not found: {template_name}")
                available = list(templates.keys())
                logger.info(f"Available templates: {available}")
                return {}
                
        except Exception as e:
            logger.error(f"Error loading strategy template: {e}")
            return {}
    
    def check_news_safe(self, pair: str, timestamp: Optional[datetime.datetime] = None) -> Tuple[bool, str]:
        """
        Check if it's safe to trade a pair based on news events.
        
        Args:
            pair: Currency pair
            timestamp: Time to check (default: now)
            
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check news guard
        can_trade, reason = self.news_guard.can_trade(pair, timestamp)
        
        # If news guard says no, check with BenBot for possible override
        if not can_trade and self.benbot_available and self.config.get('news_filtering', {}).get('benbot_can_override', False):
            benbot_response = self._consult_benbot('news_override', {
                'pair': pair,
                'timestamp': timestamp.isoformat() if timestamp else datetime.datetime.now().isoformat(),
                'reason': reason
            })
            
            # BenBot can override the news guard
            if benbot_response.get('override', False):
                logger.info(f"BenBot overrode news restriction for {pair}: {benbot_response.get('message', 'No reason')}")
                return True, f"BenBot override: {benbot_response.get('message', 'No reason')}"
        
        return can_trade, reason
    
    def check_session_optimal(self, pair: str, strategy_id: Optional[str] = None, timestamp: Optional[datetime.datetime] = None) -> Tuple[bool, str]:
        """
        Check if current session is optimal for trading.
        
        Args:
            pair: Currency pair
            strategy_id: Strategy ID (optional)
            timestamp: Time to check (default: now)
            
        Returns:
            Tuple of (is_optimal, reason)
        """
        if timestamp is None:
            timestamp = datetime.datetime.now()
            
        # Get active sessions
        active_sessions = self.session_manager.get_active_sessions(timestamp)
        if not active_sessions:
            return False, "No active trading session"
            
        # Check if pair has optimal sessions defined
        pair_obj = self.pair_manager.get_pair(pair)
        if pair_obj and hasattr(pair_obj, 'optimal_sessions') and pair_obj.optimal_sessions:
            # Check if any active session is optimal for this pair
            is_optimal = any(session in pair_obj.optimal_sessions for session in active_sessions)
            if not is_optimal:
                optimal_str = ', '.join(pair_obj.optimal_sessions)
                active_str = ', '.join(active_sessions)
                return False, f"Current session ({active_str}) not optimal for {pair}. Optimal: {optimal_str}"
        
        # If we have a strategy ID, check if the current session is optimal for this strategy
        if strategy_id:
            optimal_session = self.performance_tracker.get_optimal_session(strategy_id)
            if optimal_session and optimal_session not in active_sessions:
                return False, f"Current session not optimal for strategy. Optimal: {optimal_session}"
        
        # Check with BenBot if applicable
        if self.benbot_available and strategy_id:
            directive = self.performance_tracker.check_benbot_session_directive(strategy_id, active_sessions[0])
            if not directive.get('trade_allowed', True):
                return False, directive.get('reason', 'BenBot directive: no trade')
        
        return True, f"Optimal session active: {active_sessions[0]}"

    def get_strategy_id(self, strategy_name: str, template: str, pair: str, timeframe: str) -> str:
        """
        Generate a unique strategy ID.
        
        Args:
            strategy_name: Strategy name
            template: Template name
            pair: Currency pair
            timeframe: Timeframe
            
        Returns:
            Unique strategy ID
        """
        # Create a unique ID based on parameters
        unique_id = str(uuid.uuid4())[:8]
        return f"{template}_{pair}_{timeframe}_{unique_id}"
    
    def save_evolution_results(self, results: Dict[str, Any], strategy_id: str) -> str:
        """
        Save evolution results to file.
        
        Args:
            results: Evolution results dictionary
            strategy_id: Strategy ID
            
        Returns:
            Path to saved file
        """
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{results_dir}/{strategy_id}_{timestamp}.json"
        
        try:
            # Prepare results for serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    serializable_results[key] = value
                else:
                    # Try to convert to dict
                    try:
                        serializable_results[key] = dict(value)
                    except:
                        serializable_results[key] = str(value)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            logger.info(f"Saved evolution results to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving evolution results: {e}")
            return ""


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex EvoTrader - Integration Module")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize module
    parser.add_argument(
        "--config", 
        type=str,
        default="forex_evotrader_config.yaml",
        help="Path to configuration file"
    )
    
    # Evolve strategy command
    evolve_parser = subparsers.add_parser("evolve-strategy", help="Evolve a Forex trading strategy")
    evolve_parser.add_argument("--pair", type=str, required=True, help="Currency pair (e.g., EURUSD)")
    evolve_parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (e.g., 1h, 4h)")
    evolve_parser.add_argument("--template", type=str, required=True, help="Strategy template name")
    evolve_parser.add_argument("--session", type=str, help="Session focus (e.g., London, NewYork)")
    evolve_parser.add_argument("--generations", type=int, help="Maximum generations to run")
    evolve_parser.add_argument("--name", type=str, help="Custom strategy name")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtests with session awareness")
    backtest_parser.add_argument("--strategy-id", type=str, help="Strategy ID to backtest")
    backtest_parser.add_argument("--pair", type=str, help="Currency pair (required if strategy-id not provided)")
    backtest_parser.add_argument("--timeframe", type=str, help="Timeframe (required if strategy-id not provided)")
    backtest_parser.add_argument("--template", type=str, help="Strategy template (required if strategy-id not provided)")
    backtest_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--session-filter", type=str, help="Filter to specific session")
    
    # Forward test command
    forward_parser = subparsers.add_parser("forward-test", help="Run session-aware forward tests")
    forward_parser.add_argument("--strategy-id", type=str, required=True, help="Strategy ID to forward test")
    forward_parser.add_argument("--days", type=int, default=30, help="Number of days to forward test")
    forward_parser.add_argument("--session-only", action="store_true", help="Only trade during optimal sessions")
    forward_parser.add_argument("--enable-news-guard", action="store_true", help="Enable news event filtering")
    
    # Check news command
    news_parser = subparsers.add_parser("check-news", help="Check if trading is allowed based on news events")
    news_parser.add_argument("--pair", type=str, required=True, help="Currency pair to check")
    news_parser.add_argument("--time", type=str, help="Time to check (YYYY-MM-DD HH:MM:SS), default: now")
    
    # Check session command
    session_parser = subparsers.add_parser("check-session", help="Check if current session is optimal for trading")
    session_parser.add_argument("--pair", type=str, required=True, help="Currency pair to check")
    session_parser.add_argument("--strategy-id", type=str, help="Strategy ID (for session performance lookup)")
    session_parser.add_argument("--time", type=str, help="Time to check (YYYY-MM-DD HH:MM:SS), default: now")
    
    # Help message for available operations
    if len(sys.argv) == 1:
        print("Forex EvoTrader - Integration Module")
        print("===================================")
        print("This is the main integration component for Forex-optimized evolutionary trading.")
        print("It coordinates all Forex-specific modules and integrates with BenBot.")
        print("\nAvailable operations:")
        print("  - evolve-strategy: Evolve a Forex trading strategy")
        print("  - backtest: Run backtests with session awareness")
        print("  - forward-test: Run session-aware forward tests")
        print("  - check-news: Check if trading is allowed based on news events")
        print("  - check-session: Check if current session is optimal for trading")
        print("\nUse --help for more information")
        sys.exit(0)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize module
    forex_evotrader = ForexEvoTrader(args.config)
    
    # Execute command
    if args.command == "evolve-strategy":
        # Initialize evolution module
        evolution = ForexStrategyEvolution(forex_evotrader)
        
        # Run evolution
        results = evolution.evolve_strategy(
            pair=args.pair,
            timeframe=args.timeframe,
            template_name=args.template,
            session_focus=args.session,
            max_generations=args.generations,
            strategy_name=args.name
        )
        
        # Print summary
        if 'error' in results:
            print(f"Evolution error: {results['error']}")
        else:
            print("\nEvolution Results Summary:")
            print(f"Strategy ID: {results['strategy_id']}")
            print(f"Generations ran: {results['generations_ran']}")
            print(f"Best fitness: {results['best_fitness_history'][-1]:.4f}")
            print(f"Results saved to: {forex_evotrader.save_evolution_results(results, results['strategy_id'])}")
    
    elif args.command == "backtest":
        # Validate arguments
        if not args.strategy_id and not (args.pair and args.timeframe and args.template):
            print("Error: Either --strategy-id or --pair, --timeframe, and --template must be provided")
            sys.exit(1)
        
        # Initialize backtest module
        backtest = ForexStrategyBacktest(forex_evotrader)
        
        # Prepare parameters
        if args.strategy_id:
            # Run backtest with strategy ID
            results = backtest.backtest_strategy(
                strategy_id=args.strategy_id,
                start_date=args.start_date,
                end_date=args.end_date,
                session_filter=args.session_filter
            )
        else:
            # Load strategy template
            template = forex_evotrader.load_strategy_template(args.template)
            if not template:
                print(f"Error: Template not found: {args.template}")
                sys.exit(1)
            
            # Run backtest with parameters
            results = backtest.backtest_strategy(
                strategy_params=template.get('parameters', {}),
                strategy_type=args.template,
                pair=args.pair,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                session_filter=args.session_filter
            )
        
        # Print summary
        if 'error' in results:
            print(f"Backtest error: {results['error']}")
        else:
            print("\nBacktest Results Summary:")
            print(f"Pair: {results['pair']} {results['timeframe']}")
            print(f"Total trades: {results['total_trades']}")
            print(f"Win rate: {results['win_rate']:.2%}")
            print(f"Total pips: {results['total_pips']:.1f}")
            print(f"Profit factor: {results['profit_factor']:.2f}")
            print(f"Max drawdown: {results['max_drawdown']:.2%}")
            
            # Print compliance results
            if 'compliance' in results:
                print("\nProp Firm Compliance:")
                compliance = results['compliance']
                print(f"Compliant: {'✓' if compliance['compliant'] else '✗'}")
                print(f"Drawdown: {compliance['max_drawdown_percent']:.2f}% (limit: {compliance['max_drawdown_limit']}%)")
                if 'worst_daily_loss' in compliance and compliance['worst_daily_loss'] is not None:
                    print(f"Worst daily loss: {compliance['worst_daily_loss']:.2f}% (limit: {compliance['daily_loss_limit']}%)")
                print(f"Profit target reached: {'✓' if compliance['profit_target_reached'] else '✗'} ({compliance['total_return']:.2f}% vs {compliance['profit_target']}%)")
            
            # Print session performance
            if 'session_performance' in results:
                print("\nSession Performance:")
                for session, metrics in results['session_performance'].items():
                    print(f"  {session}:")
                    print(f"    Trades: {metrics['total_trades']}")
                    print(f"    Win rate: {metrics['win_rate']:.2%}")
                    print(f"    Total pips: {metrics['total_pips']:.1f}")
                    print(f"    Profit factor: {metrics['profit_factor']:.2f}")
    
    elif args.command == "forward-test":
        print("Forward test functionality requires the forex_evotrader_forward_test.py module.")
        print("This will be implemented in the next development phase.")
        
        # In a real implementation, this would:
        # 1. Import ForexStrategyForwardTest from forex_evotrader_forward_test
        # 2. Initialize the forward test module
        # 3. Run the forward test with the specified parameters
        # 4. Print a summary of the results
    
    elif args.command == "check-news":
        # Parse time if provided
        timestamp = None
        if args.time:
            try:
                timestamp = datetime.datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"Error: Invalid time format. Use YYYY-MM-DD HH:MM:SS")
                sys.exit(1)
        
        # Check news
        is_safe, reason = forex_evotrader.check_news_safe(args.pair, timestamp)
        
        # Print result
        if is_safe:
            print(f"✓ Safe to trade {args.pair}: {reason}")
        else:
            print(f"✗ Not safe to trade {args.pair}: {reason}")
    
    elif args.command == "check-session":
        # Parse time if provided
        timestamp = None
        if args.time:
            try:
                timestamp = datetime.datetime.strptime(args.time, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                print(f"Error: Invalid time format. Use YYYY-MM-DD HH:MM:SS")
                sys.exit(1)
        
        # Check session
        is_optimal, reason = forex_evotrader.check_session_optimal(
            args.pair, args.strategy_id, timestamp
        )
        
        # Print result
        if is_optimal:
            print(f"✓ Optimal session for trading {args.pair}: {reason}")
        
    elif args.command == "smart-analysis":
        evo_trader = ForexEvoTrader(args.config)
        
        # Ensure smart methods are enhanced
        evo_trader.enhance_with_smart_methods()
        
        # Run smart analysis
        results = evo_trader.smart_integration.run_smart_analysis(
            args.pair, args.analysis_type, args.strategy_id)
        
        # Print results
        import json
        print(json.dumps(results, indent=2, default=str))
    else:
            print(f"✗ Not optimal session for trading {args.pair}: {reason}")
    
    else:
        parser.print_help()

    # Smart analysis parser
    smart_parser = subparsers.add_parser(
        "smart-analysis", 
        help="Run smart analysis on a pair or strategy")
    
    smart_parser.add_argument(
        "--pair", 
        type=str, 
        required=True, 
        help="Currency pair to analyze")
        
    smart_parser.add_argument(
        "--strategy-id", 
        type=str, 
        help="Strategy ID to analyze")
        
    smart_parser.add_argument(
        "--analysis-type", 
        type=str,
        choices=["session", "pips", "news", "compliance", "all"],
        default="all",
        help="Type of smart analysis to run")
    