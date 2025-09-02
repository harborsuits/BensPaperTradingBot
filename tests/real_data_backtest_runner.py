#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Data Backtesting Runner

This script runs a complete backtest using real market data, our enhanced
contextual trading strategy, and advanced trade simulation features.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
from typing import Dict, List, Any, Tuple, Optional

# Import our custom components
from real_data_backtest import YahooFinanceDataProvider, MarketRegimeDetector, EventBus, Event, EventType
from enhanced_contextual_strategy import EnhancedContextualStrategy
from trade_simulator import TradeSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BacktestRunner")

class BacktestRunner:
    """
    Main backtest runner that orchestrates the entire backtesting process.
    """
    
    def __init__(self, symbols, start_date, end_date, interval='1d', initial_balance=5000.0):
        """
        Initialize the backtest runner.
        
        Args:
            symbols: List of symbols to include in backtest
            start_date: Start date for backtest data (YYYY-MM-DD)
            end_date: End date for backtest data (YYYY-MM-DD)
            interval: Data interval ('1d', '1h', etc.)
            initial_balance: Starting account balance for simulations
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.initial_balance = initial_balance
        
        # Create event bus for communication
        self.event_bus = EventBus()
        
        # Initialize components
        self.data_provider = YahooFinanceDataProvider(self.event_bus)
        self.regime_detector = MarketRegimeDetector(self.event_bus)
        
        # Initialize strategies
        self.contextual_strategy = EnhancedContextualStrategy(self.event_bus, initial_balance)
        
        # Initialize simulators
        self.contextual_simulator = TradeSimulator({}, initial_balance, "Contextual Strategy")
        self.static_simulator = TradeSimulator({}, initial_balance, "Static Strategy")
        
        # Create output directories
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)
        os.makedirs("results/reports", exist_ok=True)
    
    def load_data(self, force_download=False):
        """
        Load or download market data for all symbols.
        
        Args:
            force_download: Whether to force download even if cached data exists
        """
        logger.info(f"Loading market data for {self.symbols} from {self.start_date} to {self.end_date}")
        
        try:
            # Download data
            self.market_data = self.data_provider.download_data(
                self.symbols, 
                self.start_date, 
                self.end_date, 
                self.interval,
                force_download
            )
            
            if not self.market_data:
                # Try loading from cache
                logger.info("Download failed, trying to load from cache")
                self.market_data = self.data_provider.load_cached_data(self.symbols, self.interval)
                
                if not self.market_data:
                    logger.error("No data available for backtest")
                    return False
            
            # Update simulators with market data
            self.contextual_simulator.data = self.market_data
            self.static_simulator.data = self.market_data
            
            logger.info(f"Loaded data for {len(self.market_data)} symbols")
            for symbol, data in self.market_data.items():
                logger.info(f"  {symbol}: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def detect_regimes(self):
        """
        Detect market regimes for all symbols and all time periods.
        """
        logger.info("Detecting market regimes for all symbols")
        
        self.regimes = {}
        for symbol, data in self.market_data.items():
            logger.info(f"Detecting regimes for {symbol}")
            
            # Create a copy of data to avoid modifying original
            df = data.copy()
            
            # Detect regime for each point
            regime_series = []
            
            # Start from 100th bar to ensure enough history for indicators
            for i in range(100, len(df)):
                current_data = df.iloc[:i+1]
                regime_info = self.regime_detector.detect_regime(symbol, current_data)
                
                regime_series.append({
                    'time': df.index[i],
                    'regime': regime_info['regime'],
                    'confidence': regime_info['confidence']
                })
            
            self.regimes[symbol] = regime_series
            
            # Log distribution of regimes
            regime_counts = {}
            for r in regime_series:
                regime = r['regime']
                if regime not in regime_counts:
                    regime_counts[regime] = 0
                regime_counts[regime] += 1
                
            total = len(regime_series)
            logger.info(f"Regime distribution for {symbol}:")
            for regime, count in regime_counts.items():
                logger.info(f"  {regime}: {count} bars ({count/total*100:.1f}%)")
    
    def calculate_volatility(self):
        """
        Calculate volatility states for all symbols.
        """
        logger.info("Calculating volatility for all symbols")
        
        self.volatility = self.data_provider.calculate_volatility(self.symbols)
        
        # Log volatility states
        for symbol, state in self.volatility.items():
            logger.info(f"Volatility for {symbol}: {state}")
    
    def calculate_correlations(self):
        """
        Calculate correlation matrix between symbols.
        """
        if len(self.symbols) > 1:
            logger.info("Calculating correlations between symbols")
            
            self.correlations = self.data_provider.calculate_correlations(self.symbols)
            
            # Log strong correlations
            if self.correlations is not None:
                for i in range(len(self.symbols)):
                    for j in range(i+1, len(self.symbols)):
                        sym1, sym2 = self.symbols[i], self.symbols[j]
                        if sym1 in self.correlations.index and sym2 in self.correlations.columns:
                            corr = self.correlations.loc[sym1, sym2]
                            if abs(corr) > 0.7:
                                logger.info(f"Strong correlation between {sym1} and {sym2}: {corr:.2f}")
    
    def run_contextual_backtest(self):
        """
        Run backtest using contextual strategy.
        """
        logger.info("Running contextual strategy backtest")
        
        # Reset simulator
        self.contextual_simulator.reset()
        
        # Iterate through each symbol
        for symbol, data in self.market_data.items():
            logger.info(f"Trading {symbol} with contextual strategy")
            
            # Reset strategy balance to match simulator
            self.contextual_strategy.update_balance(self.contextual_simulator.balance)
            
            # Create a copy of data to avoid modifying original
            df = data.copy()
            
            # Start from 100th bar to ensure enough history for indicators
            for i in range(100, len(df) - 1):
                current_time = df.index[i]
                current_bar = df.iloc[i]
                
                # Skip if missing data
                if pd.isna(current_bar['Close']):
                    continue
                
                # Get current regime
                if symbol in self.regimes and i - 100 < len(self.regimes[symbol]):
                    regime_info = self.regimes[symbol][i - 100]  # Adjust index
                    current_regime = regime_info['regime']
                else:
                    current_regime = 'unknown'
                
                # Publish regime change event if needed
                if i > 100:
                    prev_regime = self.regimes[symbol][i - 101]['regime'] if symbol in self.regimes and i - 101 < len(self.regimes[symbol]) else 'unknown'
                    if current_regime != prev_regime:
                        self.event_bus.publish(Event(
                            EventType.MARKET_REGIME_CHANGE,
                            {
                                'symbol': symbol,
                                'regime': current_regime,
                                'previous_regime': prev_regime,
                                'confidence': regime_info['confidence'] if 'confidence' in regime_info else 0.5
                            }
                        ))
                
                # Convert current bar to DataFrame for strategy
                current_data = df.iloc[:i+1]
                
                # Select strategy
                strategy = self.contextual_strategy.select_strategy(symbol, current_data)
                
                # Skip if strategy says to skip trading
                if strategy.get('skip_trading', False):
                    continue
                
                # Check if we have an entry signal
                signal = strategy.get('signal', 'none')
                
                if signal != 'none':
                    # Calculate position size
                    stop_loss_pips = 20  # Default stop loss in pips
                    position = self.contextual_strategy.calculate_position_size(
                        symbol, 
                        current_bar['Close'], 
                        stop_loss_pips,
                        self.contextual_simulator.balance
                    )
                    
                    # Execute trade
                    trade = self.contextual_simulator.execute_trade(
                        symbol=symbol,
                        entry_time=current_time,
                        direction=strategy['direction'],
                        position_size=position['position_size'],
                        risk_amount=position['risk_amount'],
                        stop_loss_pips=position['stop_loss_pips'],
                        use_trailing_stop=strategy.get('use_trailing_stop', False),
                        trailing_activation=strategy.get('trailing_activation', 0.5),
                        trailing_distance=strategy.get('trailing_distance', 0.5),
                        tp_sl_ratio=strategy.get('tp_sl_ratio', 2.0),
                        strategy_id=strategy.get('id', 'unknown'),
                        strategy_name=strategy.get('name', 'Unknown Strategy'),
                        entry_conditions=strategy.get('entry_conditions', {}),
                        market_regime=current_regime
                    )
                    
                    if trade:
                        # Update strategy balance
                        self.contextual_strategy.update_balance(self.contextual_simulator.balance)
                        
                        # Record that we made a trade at this bar
                        self.contextual_strategy.last_trade_bar[symbol] = self.contextual_strategy.current_context.get('bars_since_regime_change', 0)
                        
                        # Publish trade closed event
                        if trade['pnl'] != 0:
                            self.event_bus.publish(Event(
                                EventType.TRADE_CLOSED,
                                {
                                    'symbol': symbol,
                                    'entry_time': trade['entry_time'],
                                    'exit_time': trade['exit_time'],
                                    'direction': trade['direction'],
                                    'pnl': trade['pnl'],
                                    'pips': trade['pips'],
                                    'exit_reason': trade['exit_reason'],
                                    'regime': current_regime,
                                    'strategy_id': trade['strategy_id'],
                                    'strategy_name': trade['strategy_name'],
                                    'entry_conditions': trade['entry_conditions'],
                                    'balance_after': trade['balance_after']
                                }
                            ))
        
        # Get final results
        contextual_results = self.contextual_simulator.get_results()
        
        # Log summary
        logger.info(f"Contextual strategy backtest completed:")
        logger.info(f"  Initial balance: ${contextual_results['initial_balance']:.2f}")
        logger.info(f"  Final balance: ${contextual_results['final_balance']:.2f}")
        logger.info(f"  Total return: {contextual_results['total_return']:.2f}%")
        logger.info(f"  Win rate: {contextual_results['win_rate']:.2f}%")
        logger.info(f"  Profit factor: {contextual_results['profit_factor']:.2f}")
        logger.info(f"  Max drawdown: {contextual_results['max_drawdown']:.2f}%")
        logger.info(f"  Total trades: {contextual_results['total_trades']}")
        
        return contextual_results
    
    def run_static_backtest(self):
        """
        Run backtest using a static strategy for comparison.
        """
        logger.info("Running static strategy backtest for comparison")
        
        # Reset simulator
        self.static_simulator.reset()
        
        # Simple strategy parameters
        static_strategy = {
            'id': 'static',
            'name': 'Static Strategy',
            'risk_percentage': 0.02,  # Fixed 2% risk
            'tp_sl_ratio': 2.0       # Fixed 2:1 reward:risk
        }
        
        # Iterate through each symbol
        for symbol, data in self.market_data.items():
            logger.info(f"Trading {symbol} with static strategy")
            
            # Create a copy of data to avoid modifying original
            df = data.copy()
            
            # Add simple indicators for entry signals
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            
            # Start from 100th bar to ensure enough history for indicators
            for i in range(100, len(df) - 1, 5):  # Trade every 5 bars for static strategy
                current_time = df.index[i]
                current_bar = df.iloc[i]
                
                # Skip if missing data
                if pd.isna(current_bar['Close']) or pd.isna(current_bar['SMA20']) or pd.isna(current_bar['SMA50']):
                    continue
                
                # Simple signal: SMA20 crosses above/below SMA50
                signal = 'none'
                if i > 0:
                    prev_bar = df.iloc[i-1]
                    if pd.notna(prev_bar['SMA20']) and pd.notna(prev_bar['SMA50']):
                        # Buy if SMA20 crosses above SMA50
                        if prev_bar['SMA20'] <= prev_bar['SMA50'] and current_bar['SMA20'] > current_bar['SMA50']:
                            signal = 'buy'
                        # Sell if SMA20 crosses below SMA50
                        elif prev_bar['SMA20'] >= prev_bar['SMA50'] and current_bar['SMA20'] < current_bar['SMA50']:
                            signal = 'sell'
                
                if signal != 'none':
                    # Calculate position size
                    risk_amount = self.static_simulator.balance * static_strategy['risk_percentage']
                    stop_loss_pips = 20  # Default stop loss in pips
                    
                    # Calculate pip value (simplified for forex)
                    pip_value = 0.0001
                    if symbol.endswith('JPY'):
                        pip_value = 0.01
                    
                    position_size = risk_amount / (stop_loss_pips * pip_value * 10000)
                    
                    # Execute trade
                    trade = self.static_simulator.execute_trade(
                        symbol=symbol,
                        entry_time=current_time,
                        direction=signal,
                        position_size=position_size,
                        risk_amount=risk_amount,
                        stop_loss_pips=stop_loss_pips,
                        tp_sl_ratio=static_strategy['tp_sl_ratio'],
                        strategy_id=static_strategy['id'],
                        strategy_name=static_strategy['name']
                    )
        
        # Get final results
        static_results = self.static_simulator.get_results()
        
        # Log summary
        logger.info(f"Static strategy backtest completed:")
        logger.info(f"  Initial balance: ${static_results['initial_balance']:.2f}")
        logger.info(f"  Final balance: ${static_results['final_balance']:.2f}")
        logger.info(f"  Total return: {static_results['total_return']:.2f}%")
        logger.info(f"  Win rate: {static_results['win_rate']:.2f}%")
        logger.info(f"  Profit factor: {static_results['profit_factor']:.2f}")
        logger.info(f"  Max drawdown: {static_results['max_drawdown']:.2f}%")
        logger.info(f"  Total trades: {static_results['total_trades']}")
        
        return static_results
    
    def generate_reports(self):
        """
        Generate performance reports and plots.
        """
        logger.info("Generating performance reports and plots")
        
        # Generate plots
        try:
            self.contextual_simulator.plot_results("results/plots/contextual_equity.png")
            self.static_simulator.plot_results("results/plots/static_equity.png")
            
            # Generate HTML reports
            self.contextual_simulator.generate_report("results/reports/contextual_report.html")
            self.static_simulator.generate_report("results/reports/static_report.html")
            
            # Generate comparison report
            self._generate_comparison_report(
                self.contextual_simulator.get_results(),
                self.static_simulator.get_results()
            )
            
            logger.info("Reports and plots generated successfully")
            logger.info("See 'results/reports' and 'results/plots' directories for outputs")
            
        except Exception as e:
            logger.error(f"Error generating reports: {str(e)}")
    
    def _generate_comparison_report(self, contextual_results, static_results):
        """Generate a comparison report between the two strategies."""
        
        # Calculate outperformance
        outperformance = contextual_results['total_return'] - static_results['total_return']
        
        html = f"""
        <html>
        <head>
            <title>Strategy Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .comparison {{ display: flex; justify-content: space-between; }}
                .strategy {{ width: 48%; background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .winner {{ border: 2px solid green; }}
                .summary {{ margin-top: 30px; padding: 15px; background: #e9f7ef; border-radius: 5px; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                table, th, td {{ border: 1px solid #ddd; }}
                th, td {{ padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart {{ margin-top: 30px; text-align: center; }}
                .outperformance {{ font-size: 18px; font-weight: bold; text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Trading Strategy Comparison Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="outperformance">
                    Contextual Strategy Outperformance: <span class="{('good' if outperformance > 0 else 'bad')}">{outperformance:.2f}%</span>
                </div>
                
                <div class="comparison">
                    <div class="strategy {('winner' if contextual_results['final_balance'] > static_results['final_balance'] else '')}">
                        <h2>{contextual_results['name']}</h2>
                        <p>Initial Balance: ${contextual_results['initial_balance']:.2f}</p>
                        <p>Final Balance: <span class="{('good' if contextual_results['final_balance'] >= contextual_results['initial_balance'] else 'bad')}">${contextual_results['final_balance']:.2f}</span></p>
                        <p>Total Return: <span class="{('good' if contextual_results['total_return'] >= 0 else 'bad')}">{contextual_results['total_return']:.2f}%</span></p>
                        <p>Win Rate: {contextual_results['win_rate']:.2f}%</p>
                        <p>Profit Factor: {contextual_results['profit_factor']:.2f}</p>
                        <p>Max Drawdown: {contextual_results['max_drawdown']:.2f}%</p>
                        <p>Total Trades: {contextual_results['total_trades']}</p>
                        <p>Sharpe Ratio: {contextual_results.get('sharpe_ratio', 0):.2f}</p>
                    </div>
                    
                    <div class="strategy {('winner' if static_results['final_balance'] > contextual_results['final_balance'] else '')}">
                        <h2>{static_results['name']}</h2>
                        <p>Initial Balance: ${static_results['initial_balance']:.2f}</p>
                        <p>Final Balance: <span class="{('good' if static_results['final_balance'] >= static_results['initial_balance'] else 'bad')}">${static_results['final_balance']:.2f}</span></p>
                        <p>Total Return: <span class="{('good' if static_results['total_return'] >= 0 else 'bad')}">{static_results['total_return']:.2f}%</span></p>
                        <p>Win Rate: {static_results['win_rate']:.2f}%</p>
                        <p>Profit Factor: {static_results['profit_factor']:.2f}</p>
                        <p>Max Drawdown: {static_results['max_drawdown']:.2f}%</p>
                        <p>Total Trades: {static_results['total_trades']}</p>
                        <p>Sharpe Ratio: {static_results.get('sharpe_ratio', 0):.2f}</p>
                    </div>
                </div>
                
                <div class="summary">
                    <h2>Key Findings</h2>
                    <ul>
                        <li>The Contextual Strategy {('outperformed' if outperformance > 0 else 'underperformed')} the Static Strategy by <strong>{abs(outperformance):.2f}%</strong></li>
                        <li>Contextual Strategy win rate was {contextual_results['win_rate'] - static_results['win_rate']:.2f}% {('higher' if contextual_results['win_rate'] > static_results['win_rate'] else 'lower')} than Static Strategy</li>
                        <li>Contextual Strategy had {('better' if contextual_results['max_drawdown'] < static_results['max_drawdown'] else 'worse')} maximum drawdown: {contextual_results['max_drawdown']:.2f}% vs {static_results['max_drawdown']:.2f}%</li>
                        <li>Contextual Strategy generated {contextual_results['total_trades']} trades vs {static_results['total_trades']} trades for Static Strategy</li>
                    </ul>
                </div>
                
                <div class="chart">
                    <h2>Equity Curves</h2>
                    <p>See equity curve plots in the results/plots directory.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open("results/reports/comparison_report.html", 'w') as f:
            f.write(html)


def run_backtest():
    """
    Main function to run the backtest.
    """
    # Define backtest parameters
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']  # Major forex pairs
    start_date = '2020-01-01'
    end_date = '2022-12-31'
    initial_balance = 5000.0
    
    logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
    
    # Create backtest runner
    runner = BacktestRunner(symbols, start_date, end_date, '1d', initial_balance)
    
    # Load data
    if not runner.load_data():
        logger.error("Failed to load market data, aborting backtest")
        return
    
    # Detect regimes
    runner.detect_regimes()
    
    # Calculate volatility
    runner.calculate_volatility()
    
    # Calculate correlations
    runner.calculate_correlations()
    
    # Run contextual strategy backtest
    contextual_results = runner.run_contextual_backtest()
    
    # Run static strategy backtest
    static_results = runner.run_static_backtest()
    
    # Generate reports
    runner.generate_reports()
    
    # Print comparison
    outperformance = contextual_results['total_return'] - static_results['total_return']
    logger.info("\n=== STRATEGY COMPARISON RESULTS ===")
    logger.info(f"Contextual Strategy: {contextual_results['total_return']:.2f}% return")
    logger.info(f"Static Strategy: {static_results['total_return']:.2f}% return")
    logger.info(f"Outperformance: {outperformance:.2f}%")
    logger.info("See 'results' directory for detailed reports and charts")


if __name__ == "__main__":
    run_backtest()
