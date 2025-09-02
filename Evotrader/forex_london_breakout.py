#!/usr/bin/env python3
"""
London Breakout Forex Strategy

This strategy capitalizes on the volatility surge that occurs during the London 
session opening hours. It defines a range based on the Asian session's price action,
then trades breakouts of this range after the London session begins.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
import datetime
from dateutil import parser
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_london_breakout')


class LondonBreakoutStrategy:
    """
    London Breakout Strategy for Forex Trading
    
    This strategy:
    1. Measures the high-low range during the Asian session
    2. Places buy orders above and sell orders below this range
    3. Executes when price breaks out during London session open
    4. Uses ATR for dynamic position sizing and risk management
    """
    
    def __init__(self,
                asian_session_start: str = "21:00",    # Start of Asian session (UTC)
                asian_session_end: str = "07:00",      # End of Asian session (UTC)
                london_session_start: str = "07:00",   # Start of London session (UTC)
                breakout_buffer: int = 15,             # Minutes after London open
                range_buffer: float = 5.0,             # Buffer percentage for range
                atr_filter: int = 14,                  # ATR period for volatility filter
                min_range_atr: float = 0.8,            # Min range as ATR multiple
                tp_range_multiple: float = 1.0,        # Take profit as range multiple
                sl_range_multiple: float = 0.5):       # Stop loss as range multiple
        """
        Initialize the London Breakout strategy.
        
        Args:
            asian_session_start: Asian session start time (UTC)
            asian_session_end: Asian session end time (UTC)
            london_session_start: London session start time (UTC)
            breakout_buffer: Minutes after London session to confirm breakout
            range_buffer: Percentage buffer to add to range
            atr_filter: Period for ATR calculation
            min_range_atr: Minimum range size as ATR multiple
            tp_range_multiple: Take profit as multiple of range
            sl_range_multiple: Stop loss as multiple of range
        """
        self.asian_session_start = self._parse_time(asian_session_start)
        self.asian_session_end = self._parse_time(asian_session_end)
        self.london_session_start = self._parse_time(london_session_start)
        self.breakout_buffer = datetime.timedelta(minutes=breakout_buffer)
        self.range_buffer = range_buffer / 100.0  # Convert to decimal
        self.atr_filter = atr_filter
        self.min_range_atr = min_range_atr
        self.tp_range_multiple = tp_range_multiple
        self.sl_range_multiple = sl_range_multiple
        
        # Strategy type for classification
        self.strategy_type = "LondonBreakout"
        
        # Store parameters for saving/loading
        self.parameters = {
            'asian_session_start': asian_session_start,
            'asian_session_end': asian_session_end,
            'london_session_start': london_session_start,
            'breakout_buffer': breakout_buffer,
            'range_buffer': range_buffer,
            'atr_filter': atr_filter,
            'min_range_atr': min_range_atr,
            'tp_range_multiple': tp_range_multiple,
            'sl_range_multiple': sl_range_multiple
        }
        
        logger.info(f"London Breakout Strategy initialized with parameters: {self.parameters}")
    
    def _parse_time(self, time_str: str) -> datetime.time:
        """Parse time string to datetime.time object."""
        try:
            return datetime.time(int(time_str.split(':')[0]), int(time_str.split(':')[1]))
        except Exception as e:
            logger.error(f"Error parsing time '{time_str}': {e}")
            # Default to midnight
            return datetime.time(0, 0)
    
    def _is_in_asian_session(self, dt: Union[datetime.datetime, pd.Timestamp]) -> bool:
        """Check if datetime is in Asian session."""
        time_of_day = dt.time()
        
        # Handle sessions that cross midnight
        if self.asian_session_start > self.asian_session_end:
            return time_of_day >= self.asian_session_start or time_of_day < self.asian_session_end
        else:
            return self.asian_session_start <= time_of_day < self.asian_session_end
    
    def _is_london_breakout_window(self, dt: Union[datetime.datetime, pd.Timestamp]) -> bool:
        """Check if datetime is in London breakout window."""
        time_of_day = dt.time()
        session_buffer_end = (
            datetime.datetime.combine(datetime.date.today(), self.london_session_start) + 
            self.breakout_buffer
        ).time()
        
        return self.london_session_start <= time_of_day < session_buffer_end
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on London breakout strategy.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
            
        Returns:
            DataFrame with signals (1 for long, -1 for short, 0 for no position)
        """
        if len(data) < 30:  # Ensure enough data
            return pd.DataFrame(index=data.index, data={'position': 0})
        
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Calculate ATR for volatility filtering
        data['atr'] = self._calculate_atr(data, self.atr_filter)
        
        # Initialize signal series
        signals = pd.Series(0, index=data.index)
        
        # Track active trades
        active_long = False
        active_short = False
        entry_price_long = 0.0
        entry_price_short = 0.0
        
        # Initialize range variables
        asian_high = None
        asian_low = None
        range_start_time = None
        
        # Process each candle
        for i in range(1, len(data)):
            current_datetime = data.index[i]
            prev_datetime = data.index[i-1]
            
            # Skip if NaN values
            if np.isnan(data.iloc[i]['open']) or np.isnan(data.iloc[i]['high']) or np.isnan(data.iloc[i]['low']):
                continue
                
            # Check for Asian session start
            if (not self._is_in_asian_session(prev_datetime) and 
                self._is_in_asian_session(current_datetime)):
                # Reset range for new Asian session
                asian_high = data.iloc[i]['high']
                asian_low = data.iloc[i]['low']
                range_start_time = current_datetime
                continue
            
            # Update Asian session range
            if self._is_in_asian_session(current_datetime) and range_start_time is not None:
                asian_high = max(asian_high, data.iloc[i]['high'])
                asian_low = min(asian_low, data.iloc[i]['low'])
                continue
            
            # Check for London session start
            if (self._is_london_breakout_window(current_datetime) and 
                asian_high is not None and asian_low is not None):
                
                # Calculate range and check if it meets minimum volatility
                asian_range = asian_high - asian_low
                current_atr = data.iloc[i]['atr']
                
                if not np.isnan(current_atr) and asian_range >= current_atr * self.min_range_atr:
                    # Add buffer to range
                    breakout_high = asian_high * (1 + self.range_buffer)
                    breakout_low = asian_low * (1 - self.range_buffer)
                    
                    # Check for breakouts
                    current_high = data.iloc[i]['high']
                    current_low = data.iloc[i]['low']
                    current_close = data.iloc[i]['close']
                    
                    # Long signal on break of Asian high
                    if current_close > breakout_high and not active_long:
                        signals.iloc[i] = 1
                        active_long = True
                        entry_price_long = current_close
                        
                        # Calculate take profit and stop loss
                        tp_long = entry_price_long + (asian_range * self.tp_range_multiple)
                        sl_long = entry_price_long - (asian_range * self.sl_range_multiple)
                        logger.info(f"LONG signal at {current_datetime}: Entry={entry_price_long}, TP={tp_long}, SL={sl_long}")
                    
                    # Short signal on break of Asian low
                    elif current_close < breakout_low and not active_short:
                        signals.iloc[i] = -1
                        active_short = True
                        entry_price_short = current_close
                        
                        # Calculate take profit and stop loss
                        tp_short = entry_price_short - (asian_range * self.tp_range_multiple)
                        sl_short = entry_price_short + (asian_range * self.sl_range_multiple)
                        logger.info(f"SHORT signal at {current_datetime}: Entry={entry_price_short}, TP={tp_short}, SL={sl_short}")
            
            # Exit logic - simplified for this example
            # In a real implementation, would check for TP/SL hits or end of London session
            # Here we just exit at the end of the current day or if position has been open for 24 hours
            
            # Exit long position
            if active_long:
                # Simple exit after holding for some time (implement your own exit logic)
                if (current_datetime - data.index[signals.iloc[:i].last_valid_index()]).total_seconds() > 24*3600:
                    signals.iloc[i] = 0
                    active_long = False
                    logger.info(f"Exit LONG at {current_datetime}")
            
            # Exit short position
            if active_short:
                # Simple exit after holding for some time (implement your own exit logic)
                if (current_datetime - data.index[signals.iloc[:i].last_valid_index()]).total_seconds() > 24*3600:
                    signals.iloc[i] = 0
                    active_short = False
                    logger.info(f"Exit SHORT at {current_datetime}")
        
        # Create output dataframe
        output = pd.DataFrame(index=data.index)
        output['position'] = signals
        
        return output
    
    def backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run backtest for the strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with backtest results
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Initialize results
        equity_curve = [100.0]  # Start with $100
        trades = []
        position = 0
        entry_price = 0.0
        entry_time = None
        
        # Simulate trading
        for i in range(1, len(data)):
            current_signal = signals['position'].iloc[i]
            prev_position = position
            
            # Process position changes
            if current_signal != prev_position:
                # Exit position
                if prev_position != 0:
                    exit_price = data['close'].iloc[i]
                    exit_time = data.index[i]
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 * prev_position
                    
                    # Calculate pip value for forex
                    pips = abs(exit_price - entry_price) * 10000  # For 4-digit pairs
                    if "JPY" in data.columns.name if isinstance(data.columns, pd.MultiIndex) else "":
                        pips = abs(exit_price - entry_price) * 100  # For JPY pairs
                    
                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': prev_position,
                        'pnl': pnl_pct,
                        'pips': pips * prev_position,  # Positive for winning trades
                        'symbol': data.columns.name if isinstance(data.columns, pd.MultiIndex) else "FOREX"
                    }
                    trades.append(trade)
                    
                    # Update equity
                    equity_curve.append(equity_curve[-1] * (1 + pnl_pct / 100))
                
                # Enter new position
                if current_signal != 0:
                    position = current_signal
                    entry_price = data['close'].iloc[i]
                    entry_time = data.index[i]
                else:
                    position = 0
            
            # Update equity curve even when position doesn't change
            if i > 0 and position != 0:
                current_price = data['close'].iloc[i]
                unrealized_pnl_pct = (current_price - entry_price) / entry_price * 100 * position
                equity_curve.append(equity_curve[-1] * (1 + unrealized_pnl_pct / 100))
            elif i > 0:
                equity_curve.append(equity_curve[-1])
        
        # Calculate performance metrics
        if len(equity_curve) > 1:
            total_return_pct = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
            
            # Calculate maximum drawdown
            peak = equity_curve[0]
            max_drawdown = 0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate win rate
            winning_trades = sum(1 for trade in trades if trade['pnl'] > 0)
            win_rate = winning_trades / len(trades) * 100 if trades else 0
            
            # Calculate profit factor
            gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate daily returns
            daily_returns = []
            prev_day = None
            prev_equity = equity_curve[0]
            
            for i, dt in enumerate(data.index):
                day = dt.date()
                if prev_day is None or day != prev_day:
                    if prev_day is not None:
                        daily_return = (equity_curve[i] - prev_equity) / prev_equity * 100
                        daily_returns.append(daily_return)
                    prev_day = day
                    prev_equity = equity_curve[i]
            
            # Calculate worst daily loss
            worst_daily_loss = min(daily_returns) if daily_returns else 0
            
            # Prepare metrics
            metrics = {
                'total_return_pct': total_return_pct,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': len(trades),
                'worst_daily_loss': worst_daily_loss,
                'sharpe_ratio': np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 1 else 0
            }
            
            # Prepare daily P&L for reporting
            daily_pnl = {}
            for dt, ret in zip([d.date() for d in data.index[1:] if d.date() != data.index[0].date()], daily_returns):
                daily_pnl[dt.isoformat()] = ret
            
            # Prepare results
            results = {
                'metrics': metrics,
                'trades': trades,
                'equity_curve': {str(data.index[i]): equity_curve[i] for i in range(len(equity_curve))},
                'daily_returns': daily_returns,
                'daily_pnl': daily_pnl,
                'parameters': self.parameters
            }
            
            return results
        
        else:
            # No trades executed
            return {
                'metrics': {
                    'total_return_pct': 0,
                    'max_drawdown': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'total_trades': 0,
                    'worst_daily_loss': 0,
                    'sharpe_ratio': 0
                },
                'trades': [],
                'equity_curve': {str(data.index[0]): 100.0},
                'daily_returns': [],
                'daily_pnl': {},
                'parameters': self.parameters
            }


# Module execution
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Test London Breakout strategy")
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="Path to OHLCV data CSV file"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file for backtest results"
    )
    
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Generate equity curve plot"
    )
    
    args = parser.parse_args()
    
    # Load data
    try:
        data = pd.read_csv(args.data, index_col=0, parse_dates=True)
        
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]
        
        print(f"Loaded data from {args.data}: {len(data)} rows")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import sys
        sys.exit(1)
    
    # Create strategy and run backtest
    strategy = LondonBreakoutStrategy()
    results = strategy.backtest(data)
    
    # Print metrics
    print("\n=== London Breakout Strategy Backtest Results ===")
    print(f"Total Return: {results['metrics']['total_return_pct']:.2f}%")
    print(f"Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
    print(f"Win Rate: {results['metrics']['win_rate']:.2f}%")
    print(f"Profit Factor: {results['metrics']['profit_factor']:.2f}")
    print(f"Total Trades: {results['metrics']['total_trades']}")
    print(f"Worst Daily Loss: {results['metrics']['worst_daily_loss']:.2f}%")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            # Convert equity curve keys to strings
            serializable_results = results.copy()
            
            # Make sure all dates are strings
            serializable_results['equity_curve'] = {
                str(k): v for k, v in results['equity_curve'].items()
            }
            
            # Convert daily_pnl keys to strings if they aren't already
            serializable_results['daily_pnl'] = {
                str(k): v for k, v in results['daily_pnl'].items()
            }
            
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    # Generate plot
    if args.plot:
        plt.figure(figsize=(12, 6))
        
        # Convert equity curve to list
        equity_values = list(results['equity_curve'].values())
        dates = [parser.parse(d) for d in results['equity_curve'].keys()]
        
        plt.plot(dates, equity_values)
        plt.title('London Breakout Strategy Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        
        plt.tight_layout()
        
        plot_file = 'london_breakout_equity.png'
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Equity curve plot saved to {plot_file}")
