#!/usr/bin/env python3
"""
Historical Market Data Validation

Tests evolved strategies against real historical market data to validate performance.
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import importlib
import yfinance as yf

# Add project root to path if needed
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_historical_data(symbols: List[str], 
                         start_date: str, 
                         end_date: str, 
                         interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Fetch historical market data for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        interval: Data interval (1d, 1h, etc.)
        
    Returns:
        Dictionary of DataFrames with market data for each symbol
    """
    data = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Add symbol column
            df['symbol'] = symbol
            
            data[symbol] = df
            logger.info(f"Fetched {len(df)} data points for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return data

def backtest_strategy(strategy, market_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Backtest a strategy on historical market data.
    
    Args:
        strategy: Strategy object
        market_data: DataFrame with market data
        
    Returns:
        Dictionary with backtest results
    """
    logger.info(f"Backtesting strategy {strategy.__class__.__name__} on {market_data['symbol'].iloc[0]}")
    
    # Initialize results
    trades = []
    signals = []
    position = 0
    entry_price = 0
    entry_date = None
    capital = 10000.0
    cash = capital
    equity = [capital]
    dates = [market_data.index[0]]
    
    # Process each day
    for i in range(len(market_data) - 1):
        current_date = market_data.index[i]
        current_data = market_data.iloc[:i+1]
        next_day = market_data.iloc[i+1]
        next_date = market_data.index[i+1]
        
        # Get signal
        if hasattr(strategy, "calculate_signal"):
            signal = strategy.calculate_signal(current_data)
        elif hasattr(strategy, "generate_signals"):
            signal = strategy.generate_signals(current_data)
        else:
            signal = {"signal": "none", "confidence": 0}
        
        # Add metadata to signal
        signal["date"] = current_date
        signal["price"] = current_data["close"].iloc[-1]
        signals.append(signal)
        
        # Process signal for next day
        trade_executed = False
        exit_price = None
        
        # Calculate position size (either from signal or default to 1.0)
        position_size = signal.get("position_size", 1.0)
        
        if signal["signal"] == "buy" and position <= 0:
            # Close any existing short position
            if position < 0:
                exit_price = next_day["open"]
                pnl = (entry_price - exit_price) * abs(position)
                cash += entry_price * abs(position) + pnl
                
                # Record the closed trade
                trades.append({
                    "type": "exit_short",
                    "entry_date": entry_date,
                    "exit_date": next_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / (entry_price * abs(position)),
                    "size": abs(position)
                })
            
            # Enter long position
            entry_price = next_day["open"]
            entry_date = next_date
            
            # Calculate position size
            affordable_shares = cash / entry_price * position_size
            position = affordable_shares
            
            # Update cash
            cash -= entry_price * position
            
            # Record the new trade
            trades.append({
                "type": "entry_long",
                "date": next_date,
                "price": entry_price,
                "size": position,
                "cash": cash
            })
            
            trade_executed = True
            
        elif signal["signal"] == "sell" and position >= 0:
            # Close any existing long position
            if position > 0:
                exit_price = next_day["open"]
                pnl = (exit_price - entry_price) * position
                cash += exit_price * position
                
                # Record the closed trade
                trades.append({
                    "type": "exit_long",
                    "entry_date": entry_date,
                    "exit_date": next_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "pnl_pct": pnl / (entry_price * position),
                    "size": position
                })
            
            # Enter short position (if shorting is allowed)
            entry_price = next_day["open"]
            entry_date = next_date
            
            # Calculate position size (negative for short)
            affordable_shares = cash / entry_price * position_size
            position = -affordable_shares
            
            # Update cash (add the borrowed shares value)
            cash += entry_price * abs(position)
            
            # Record the new trade
            trades.append({
                "type": "entry_short",
                "date": next_date,
                "price": entry_price,
                "size": position,
                "cash": cash
            })
            
            trade_executed = True
            
        # Update equity
        current_equity = cash
        if position != 0:
            # Add position value
            current_price = next_day["close"]
            if position > 0:
                # Long position
                position_value = position * current_price
                current_equity += position_value
            else:
                # Short position
                pnl = (entry_price - current_price) * abs(position)
                current_equity += pnl
        
        equity.append(current_equity)
        dates.append(next_date)
        
        # Update strategy with position info
        if hasattr(strategy, "update_position"):
            strategy.update_position(position, entry_price)
    
    # Close final position at the last price
    if position != 0:
        last_price = market_data["close"].iloc[-1]
        last_date = market_data.index[-1]
        
        if position > 0:
            # Close long position
            pnl = (last_price - entry_price) * position
            
            trades.append({
                "type": "exit_long",
                "entry_date": entry_date,
                "exit_date": last_date,
                "entry_price": entry_price,
                "exit_price": last_price,
                "pnl": pnl,
                "pnl_pct": pnl / (entry_price * position),
                "size": position
            })
            
        else:
            # Close short position
            pnl = (entry_price - last_price) * abs(position)
            
            trades.append({
                "type": "exit_short",
                "entry_date": entry_date,
                "exit_date": last_date,
                "entry_price": entry_price,
                "exit_price": last_price,
                "pnl": pnl,
                "pnl_pct": pnl / (entry_price * abs(position)),
                "size": abs(position)
            })
    
    # Create equity curve DataFrame
    equity_df = pd.DataFrame({
        "date": dates,
        "equity": equity
    })
    equity_df.set_index("date", inplace=True)
    
    # Calculate trade metrics
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    losses = sum(1 for t in trades if t.get("pnl", 0) < 0)
    total_trades = wins + losses
    
    if total_trades > 0:
        win_rate = wins / total_trades
    else:
        win_rate = 0
    
    # Calculate returns
    returns = np.diff(equity) / equity[:-1]
    
    # Calculate metrics
    if len(returns) > 0:
        total_return = (equity[-1] - equity[0]) / equity[0]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = equity[0]
        for e in equity:
            if e > peak:
                peak = e
            drawdown = (peak - e) / peak
            max_drawdown = max(max_drawdown, drawdown)
    else:
        total_return = 0
        annual_return = 0
        sharpe = 0
        max_drawdown = 0
    
    # Compile results
    results = {
        "equity_curve": equity_df.to_dict(),
        "trades": [t for t in trades if "pnl" in t],  # Only include completed trades
        "signals": signals,
        "metrics": {
            "initial_capital": capital,
            "final_equity": equity[-1],
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "winning_trades": wins,
            "losing_trades": losses
        }
    }
    
    logger.info(f"Backtest completed with {total_trades} trades and {total_return*100:.2f}% return")
    
    return results

def load_strategy(strategy_path: str, parameters: Optional[Dict[str, Any]] = None) -> Any:
    """
    Load a strategy class from a module path.
    
    Args:
        strategy_path: Path to strategy module and class (module.ClassName)
        parameters: Optional parameters to initialize the strategy with
        
    Returns:
        Initialized strategy object
    """
    try:
        module_path, class_name = strategy_path.rsplit(".", 1)
        
        # Add current directory to path if needed
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        
        # Initialize with parameters if provided
        if parameters:
            strategy = strategy_class(parameters)
        else:
            strategy = strategy_class()
        
        logger.info(f"Loaded strategy: {strategy_class.__name__}")
        return strategy
    
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to load strategy: {e}")
        raise

def generate_backtest_report(results: Dict[str, Any], 
                           strategy_name: str,
                           symbol: str,
                           output_dir: str) -> str:
    """
    Generate a backtest report from results.
    
    Args:
        results: Backtest results dictionary
        strategy_name: Name of the strategy
        symbol: Symbol that was tested
        output_dir: Directory to save the report
        
    Returns:
        Path to the report file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output paths
    report_file = os.path.join(output_dir, f"backtest_report_{symbol.replace('/', '_')}.md")
    equity_chart_file = os.path.join(output_dir, f"equity_curve_{symbol.replace('/', '_')}.png")
    
    # Create equity chart
    equity_df = pd.DataFrame.from_dict(results["equity_curve"])
    
    # Only create chart if we have data
    if len(equity_df) > 0:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=equity_df, x=equity_df.index, y="equity")
        plt.title(f"Equity Curve: {strategy_name} on {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(equity_chart_file, dpi=100, bbox_inches="tight")
        plt.close()
    
    # Extract metrics
    metrics = results["metrics"]
    trades = results["trades"]
    
    # Write report
    with open(report_file, "w") as f:
        f.write(f"# Backtest Report: {strategy_name} on {symbol}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write(f"- **Initial Capital:** ${metrics['initial_capital']:.2f}\n")
        f.write(f"- **Final Equity:** ${metrics['final_equity']:.2f}\n")
        f.write(f"- **Total Return:** {metrics['total_return']*100:.2f}%\n")
        f.write(f"- **Annual Return:** {metrics['annual_return']*100:.2f}%\n")
        f.write(f"- **Sharpe Ratio:** {metrics['sharpe_ratio']:.2f}\n")
        f.write(f"- **Max Drawdown:** {metrics['max_drawdown']*100:.2f}%\n")
        f.write(f"- **Win Rate:** {metrics['win_rate']*100:.2f}%\n")
        f.write(f"- **Total Trades:** {metrics['total_trades']}\n")
        f.write("\n")
        
        # Include equity chart
        if len(equity_df) > 0:
            f.write("## Equity Curve\n\n")
            f.write(f"![Equity Curve]({os.path.basename(equity_chart_file)})\n\n")
        
        # Trade analysis
        f.write("## Trade Analysis\n\n")
        
        # Calculate trade stats
        if trades:
            wins = [t for t in trades if t.get("pnl", 0) > 0]
            losses = [t for t in trades if t.get("pnl", 0) < 0]
            
            # Calculate profit metrics
            total_profit = sum(t["pnl"] for t in wins) if wins else 0
            total_loss = sum(t["pnl"] for t in losses) if losses else 0
            net_profit = total_profit + total_loss
            
            # Calculate average trade metrics
            avg_win = total_profit / len(wins) if wins else 0
            avg_loss = total_loss / len(losses) if losses else 0
            avg_trade = net_profit / len(trades) if trades else 0
            
            # Calculate profit factor
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
            
            # Write trade stats
            f.write("### Trade Statistics\n\n")
            f.write(f"- **Total Net Profit:** ${net_profit:.2f}\n")
            f.write(f"- **Gross Profit:** ${total_profit:.2f}\n")
            f.write(f"- **Gross Loss:** ${total_loss:.2f}\n")
            f.write(f"- **Profit Factor:** {profit_factor:.2f}\n")
            f.write(f"- **Average Trade:** ${avg_trade:.2f}\n")
            f.write(f"- **Average Win:** ${avg_win:.2f}\n")
            f.write(f"- **Average Loss:** ${avg_loss:.2f}\n")
            
            if avg_loss != 0:
                f.write(f"- **Win/Loss Ratio:** {abs(avg_win/avg_loss):.2f}\n")
            
            f.write("\n")
            
            # List largest trades
            f.write("### Largest Winning Trades\n\n")
            f.write("| Entry Date | Exit Date | Type | Entry | Exit | Profit | Return |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            # Sort by profit
            sorted_wins = sorted(wins, key=lambda t: t["pnl"], reverse=True)
            for trade in sorted_wins[:5]:  # Show top 5
                f.write(f"| {trade['entry_date'].strftime('%Y-%m-%d')} | ")
                f.write(f"{trade['exit_date'].strftime('%Y-%m-%d')} | ")
                
                trade_type = "Long" if "exit_long" in trade["type"] else "Short"
                f.write(f"{trade_type} | ")
                
                f.write(f"${trade['entry_price']:.2f} | ")
                f.write(f"${trade['exit_price']:.2f} | ")
                f.write(f"${trade['pnl']:.2f} | ")
                f.write(f"{trade['pnl_pct']*100:.2f}% |\n")
            
            f.write("\n")
            
            # List largest losing trades
            f.write("### Largest Losing Trades\n\n")
            f.write("| Entry Date | Exit Date | Type | Entry | Exit | Loss | Return |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            
            # Sort by loss (ascending)
            sorted_losses = sorted(losses, key=lambda t: t["pnl"])
            for trade in sorted_losses[:5]:  # Show top 5
                f.write(f"| {trade['entry_date'].strftime('%Y-%m-%d')} | ")
                f.write(f"{trade['exit_date'].strftime('%Y-%m-%d')} | ")
                
                trade_type = "Long" if "exit_long" in trade["type"] else "Short"
                f.write(f"{trade_type} | ")
                
                f.write(f"${trade['entry_price']:.2f} | ")
                f.write(f"${trade['exit_price']:.2f} | ")
                f.write(f"${trade['pnl']:.2f} | ")
                f.write(f"{trade['pnl_pct']*100:.2f}% |\n")
        else:
            f.write("No trades were executed during the backtest period.\n")
        
        f.write("\n## Conclusion and Recommendations\n\n")
        
        # Generate simple conclusions based on results
        if metrics['total_return'] > 0:
            if metrics['sharpe_ratio'] > 1.0:
                f.write("The strategy demonstrates **positive performance** with good risk-adjusted returns. ")
                
                if metrics['win_rate'] > 0.5:
                    f.write("With a high win rate and positive risk/reward profile, ")
                    f.write("this strategy appears suitable for live trading.\n\n")
                else:
                    f.write("Despite a lower win rate, the positive expectancy indicates ")
                    f.write("the strategy's winning trades outsize the losing ones.\n\n")
            else:
                f.write("The strategy shows **positive returns** but with suboptimal risk-adjusted metrics. ")
                f.write("Consider adjusting risk parameters to improve the Sharpe ratio.\n\n")
        else:
            f.write("The strategy generated **negative returns** during the test period. ")
            f.write("Further optimization is recommended before deployment.\n\n")
        
        # Add specific recommendations
        f.write("### Recommendations\n\n")
        
        if metrics['max_drawdown'] > 0.2:
            f.write("- **Reduce risk exposure**: The strategy experiences significant drawdowns. ")
            f.write("Consider implementing tighter stop-losses or reducing position sizes.\n")
        
        if metrics['total_trades'] < 10:
            f.write("- **Increase sample size**: The strategy generated too few trades for statistical significance. ")
            f.write("Test over a longer period or adjust parameters to increase trading frequency.\n")
        
        if metrics['win_rate'] < 0.4:
            f.write("- **Improve entry conditions**: The low win rate suggests entry signals may be suboptimal. ")
            f.write("Consider adding filters or adjusting entry thresholds.\n")
    
    logger.info(f"Backtest report saved to: {report_file}")
    return report_file

def main():
    """Run the historical validation with command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate strategies against historical market data")
    parser.add_argument("--strategy", type=str, required=True, help="Path to strategy module and class (module.ClassName)")
    parser.add_argument("--symbols", type=str, nargs="+", required=True, help="Symbols to test")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, help="Output directory (default: timestamped directory)")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        timestamp = int(time.time())
        args.output_dir = f"test_results/historical_validation_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Install required packages if not available
    try:
        import yfinance
    except ImportError:
        import subprocess
        logger.info("Installing required packages: yfinance, seaborn, matplotlib")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "seaborn", "matplotlib"])
    
    # Load strategy
    strategy = load_strategy(args.strategy)
    strategy_name = strategy.__class__.__name__
    
    # Fetch historical data
    data = fetch_historical_data(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Run backtests
    results = {}
    for symbol, market_data in data.items():
        if len(market_data) > 0:
            # Run backtest
            backtest_result = backtest_strategy(strategy, market_data)
            results[symbol] = backtest_result
            
            # Generate report
            report_file = generate_backtest_report(
                results=backtest_result,
                strategy_name=strategy_name,
                symbol=symbol,
                output_dir=args.output_dir
            )
            
            print(f"\nBacktest for {symbol} completed!")
            print(f"Report: {report_file}")
            
            # Print summary
            metrics = backtest_result["metrics"]
            print(f"Total Return: {metrics['total_return']*100:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
            print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    
    # Save combined results
    results_file = os.path.join(args.output_dir, "combined_results.json")
    
    # Extract just the metrics for the JSON file (avoid large data structures)
    combined_metrics = {
        symbol: results[symbol]["metrics"]
        for symbol in results
    }
    
    with open(results_file, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    
    print(f"\nHistorical validation completed!")
    print(f"Results saved to: {results_file}")
    
    # Calculate average metrics across all symbols
    total_returns = [results[symbol]["metrics"]["total_return"] for symbol in results]
    sharpe_ratios = [results[symbol]["metrics"]["sharpe_ratio"] for symbol in results]
    win_rates = [results[symbol]["metrics"]["win_rate"] for symbol in results]
    
    if total_returns:
        print(f"\nAVERAGE METRICS ACROSS ALL SYMBOLS:")
        print(f"Average Total Return: {np.mean(total_returns)*100:.2f}%")
        print(f"Average Sharpe Ratio: {np.mean(sharpe_ratios):.2f}")
        print(f"Average Win Rate: {np.mean(win_rates)*100:.2f}%")

if __name__ == "__main__":
    main()
