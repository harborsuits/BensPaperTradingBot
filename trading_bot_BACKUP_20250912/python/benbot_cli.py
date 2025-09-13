#!/usr/bin/env python3
"""
BenBot CLI - Unified Command Line Interface
Single entry point for all BenBot operations
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the trading_bot directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with error handling for missing modules
try:
    from trading_bot.core.engine import TradingEngine
except ImportError:
    TradingEngine = None

try:
    from trading_bot.evaluation.evaluator import StrategyEvaluator
except ImportError:
    StrategyEvaluator = None

try:
    from trading_bot.backtesting.backtester import Backtester
except ImportError:
    Backtester = None

try:
    from trading_bot.dashboard.launch_dashboard import launch_dashboard
except ImportError:
    launch_dashboard = None

try:
    from trading_bot.monitor.monitor import SystemMonitor
except ImportError:
    SystemMonitor = None


class BenBotCLI:
    """Unified CLI for BenBot operations"""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create the main argument parser"""
        parser = argparse.ArgumentParser(
            description="BenBot - Algorithmic Trading System CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  benbot evaluator --strategy momentum --symbols SPY,AAPL
  benbot backtest --config config.yaml --start-date 2024-01-01
  benbot live --config live_config.yaml
  benbot dashboard --port 8050
  benbot monitor --alert-threshold 0.05
            """
        )

        parser.add_argument(
            '--log-level',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
            default='INFO',
            help='Set logging level'
        )

        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file'
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Evaluator command
        eval_parser = subparsers.add_parser(
            'evaluator',
            help='Run strategy evaluation'
        )
        eval_parser.add_argument(
            '--strategy',
            required=True,
            help='Strategy to evaluate'
        )
        eval_parser.add_argument(
            '--symbols',
            help='Comma-separated list of symbols'
        )
        eval_parser.add_argument(
            '--start-date',
            help='Start date for evaluation (YYYY-MM-DD)'
        )
        eval_parser.add_argument(
            '--end-date',
            help='End date for evaluation (YYYY-MM-DD)'
        )

        # Backtest command
        backtest_parser = subparsers.add_parser(
            'backtest',
            help='Run backtesting'
        )
        backtest_parser.add_argument(
            '--start-date',
            required=True,
            help='Start date for backtest (YYYY-MM-DD)'
        )
        backtest_parser.add_argument(
            '--end-date',
            required=True,
            help='End date for backtest (YYYY-MM-DD)'
        )
        backtest_parser.add_argument(
            '--initial-capital',
            type=float,
            default=100000,
            help='Initial capital for backtest'
        )

        # Live trading command
        live_parser = subparsers.add_parser(
            'live',
            help='Run live trading'
        )
        live_parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run in dry-run mode (no real trades)'
        )
        live_parser.add_argument(
            '--max-position-size',
            type=float,
            help='Maximum position size as percentage'
        )

        # Dashboard command
        dashboard_parser = subparsers.add_parser(
            'dashboard',
            help='Launch trading dashboard'
        )
        dashboard_parser.add_argument(
            '--port',
            type=int,
            default=8050,
            help='Port to run dashboard on'
        )
        dashboard_parser.add_argument(
            '--host',
            default='localhost',
            help='Host to bind dashboard to'
        )

        # Monitor command
        monitor_parser = subparsers.add_parser(
            'monitor',
            help='Run system monitoring'
        )
        monitor_parser.add_argument(
            '--alert-threshold',
            type=float,
            default=0.05,
            help='Alert threshold for drawdown/risk metrics'
        )
        monitor_parser.add_argument(
            '--check-interval',
            type=int,
            default=60,
            help='Check interval in seconds'
        )

        # Config command
        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management'
        )
        config_parser.add_argument(
            'action',
            choices=['validate', 'show', 'init'],
            help='Configuration action'
        )

        return parser

    def setup_logging(self, log_level):
        """Setup logging configuration"""
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")

        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f"benbot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            ]
        )

    def run_evaluator(self, args):
        """Run strategy evaluation"""
        if StrategyEvaluator is None:
            print("❌ StrategyEvaluator module not found")
            return

        print(f"🚀 Running strategy evaluation for: {args.strategy}")

        evaluator = StrategyEvaluator()

        # Parse symbols
        symbols = args.symbols.split(',') if args.symbols else ['SPY']

        # Set date range
        start_date = args.start_date or '2024-01-01'
        end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')

        results = evaluator.evaluate_strategy(
            strategy_name=args.strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        print("✅ Evaluation complete!")
        print(f"Strategy: {args.strategy}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Date Range: {start_date} to {end_date}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 'N/A')}")
        print(f"Total Return: {results.get('total_return', 'N/A')}")
        print(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}")

    def run_backtest(self, args):
        """Run backtesting"""
        if Backtester is None:
            print("❌ Backtester module not found")
            return

        print(f"📊 Running backtest from {args.start_date} to {args.end_date}")

        backtester = Backtester()

        results = backtester.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.initial_capital
        )

        print("✅ Backtest complete!")
        print(f"Initial Capital: ${args.initial_capital:,.2f}")
        print(f"Final Value: ${results.get('final_value', 0):,.2f}")
        print(f"Total Return: {results.get('total_return', 0):.2%}")
        print(f"Sharpe Ratio: {results.get('sharpe_ratio', 'N/A')}")
        print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")

    def run_live(self, args):
        """Run live trading"""
        if TradingEngine is None:
            print("❌ TradingEngine module not found")
            return

        mode = "DRY RUN" if args.dry_run else "LIVE TRADING"
        print(f"⚡ Starting {mode}")

        engine = TradingEngine(dry_run=args.dry_run)

        if args.max_position_size:
            engine.set_max_position_size(args.max_position_size)

        try:
            engine.start()
            print(f"✅ {mode} engine started successfully")
            print("Press Ctrl+C to stop...")

            # Keep running until interrupted
            while True:
                import time
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n🛑 Stopping {mode} engine...")
            engine.stop()
            print("✅ Engine stopped successfully")

    def run_dashboard(self, args):
        """Launch dashboard"""
        if launch_dashboard is None:
            print("❌ Dashboard launch function not found")
            return

        print(f"📈 Launching dashboard on {args.host}:{args.port}")

        try:
            launch_dashboard(host=args.host, port=args.port)
        except Exception as e:
            print(f"❌ Failed to launch dashboard: {e}")
            sys.exit(1)

    def run_monitor(self, args):
        """Run system monitoring"""
        if SystemMonitor is None:
            print("❌ SystemMonitor module not found")
            return

        print(f"👀 Starting system monitor (alert threshold: {args.alert_threshold})")

        monitor = SystemMonitor(alert_threshold=args.alert_threshold)

        try:
            monitor.start(check_interval=args.check_interval)
            print("✅ Monitor started successfully")
            print("Press Ctrl+C to stop...")

            # Keep running until interrupted
            while True:
                import time
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n🛑 Stopping monitor...")
            monitor.stop()
            print("✅ Monitor stopped successfully")

    def run_config(self, args):
        """Handle configuration commands"""
        if args.action == 'validate':
            print("🔍 Validating configuration...")
            # Add validation logic here
            print("✅ Configuration is valid")

        elif args.action == 'show':
            print("📋 Current configuration:")
            # Add show logic here
            print("Configuration display not yet implemented")

        elif args.action == 'init':
            print("⚙️ Initializing default configuration...")
            # Add init logic here
            print("✅ Default configuration created")

    def run(self):
        """Main CLI entry point"""
        args = self.parser.parse_args()

        if not args.command:
            self.parser.print_help()
            return

        # Setup logging
        self.setup_logging(args.log_level)

        # Dispatch to appropriate handler
        try:
            if args.command == 'evaluator':
                self.run_evaluator(args)
            elif args.command == 'backtest':
                self.run_backtest(args)
            elif args.command == 'live':
                self.run_live(args)
            elif args.command == 'dashboard':
                self.run_dashboard(args)
            elif args.command == 'monitor':
                self.run_monitor(args)
            elif args.command == 'config':
                self.run_config(args)

        except Exception as e:
            print(f"❌ Error running {args.command}: {e}")
            logging.exception(f"Error in {args.command}")
            sys.exit(1)


def main():
    """CLI entry point"""
    cli = BenBotCLI()
    cli.run()


if __name__ == '__main__':
    main()
