import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pytest
import os
import sys
import json
from datetime import datetime, timedelta

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# These imports will work once the API clients and database are implemented
try:
    from api_clients import alpaca_client, tradier_client
    from database import db
except ImportError:
    print("Warning: Could not import API clients or database. Some tests will be skipped.")

class BacktestEngine:
    """Engine for running backtests against historical data"""
    
    def __init__(self, symbol, start_date, end_date, initial_capital=100000):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = None
        self.trades = []
        self.equity_curve = []
    
    def fetch_historical_data(self):
        """Fetch historical price data from Alpaca"""
        try:
            # Convert dates to string format if they're datetime objects
            start_str = self.start_date.strftime("%Y-%m-%d") if isinstance(self.start_date, datetime) else self.start_date
            end_str = self.end_date.strftime("%Y-%m-%d") if isinstance(self.end_date, datetime) else self.end_date
            
            # First try to get data from database if available
            try:
                price_data = list(db.find_many("historical_prices", {
                    "symbol": self.symbol,
                    "date": {"$gte": start_str, "$lte": end_str}
                }, sort_key="date", sort_direction=1))
            except (NameError, AttributeError):
                # Database not available, use API directly
                price_data = []
            
            if not price_data:
                # If no data in DB, fetch from Alpaca
                try:
                    bars = alpaca_client.get_bars(self.symbol, "1D", 1000)
                    price_data = [
                        {
                            "symbol": self.symbol,
                            "date": bar["timestamp"],
                            "open": bar["open"],
                            "high": bar["high"],
                            "low": bar["low"],
                            "close": bar["close"],
                            "volume": bar["volume"]
                        }
                        for bar in bars
                    ]
                    
                    # Save to database if available
                    try:
                        for bar in price_data:
                            db.update_one("historical_prices", 
                                      {"symbol": bar["symbol"], "date": bar["date"]},
                                      {"$set": bar}, upsert=True)
                    except (NameError, AttributeError):
                        pass  # Database not available
                except (NameError, AttributeError):
                    # API client not available, use demo data
                    print("API client not available, using demo data")
                    price_data = self._generate_demo_data()
            
            # Convert to DataFrame
            self.data = pd.DataFrame(price_data)
            return True
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return False
    
    def _generate_demo_data(self):
        """Generate demo price data for testing"""
        # Generate 100 days of demo data
        start_date = datetime.now() - timedelta(days=100)
        dates = [start_date + timedelta(days=i) for i in range(100)]
        
        # Generate a demo price series with some randomness
        price = 100.0
        price_data = []
        
        for date in dates:
            # Add some random walk to price
            change = np.random.normal(0, 1) / 100  # 1% standard deviation
            price = price * (1 + change)
            
            # Generate OHLC data
            high = price * (1 + abs(np.random.normal(0, 0.5) / 100))
            low = price * (1 - abs(np.random.normal(0, 0.5) / 100))
            open_price = low + (high - low) * np.random.random()
            close = low + (high - low) * np.random.random()
            volume = int(np.random.normal(1000000, 200000))
            
            price_data.append({
                "symbol": self.symbol,
                "date": date.strftime("%Y-%m-%d"),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
        
        return price_data
    
    def apply_strategy(self, strategy_function, strategy_params):
        """Apply a strategy function to the data"""
        if self.data is None:
            if not self.fetch_historical_data():
                return False
        
        # Apply strategy to generate signals
        self.data = strategy_function(self.data, strategy_params)
        
        # Reset trades and equity
        self.trades = []
        self.equity_curve = [self.initial_capital]
        current_equity = self.initial_capital
        
        # Simulate trades
        position = 0
        entry_price = 0
        entry_date = None
        
        for i, row in self.data.iterrows():
            if i == 0:
                continue  # Skip first row as we need previous signal
                
            signal = self.data.iloc[i-1]["signal"]
            
            # Entry logic
            if signal == 1 and position == 0:
                position = 1
                entry_price = row["open"]
                entry_date = row["date"]
                entry_index = i
            
            # Exit logic
            elif (signal == -1 or signal == 0) and position == 1:
                exit_price = row["open"]
                exit_date = row["date"]
                
                # Calculate P&L
                pnl = (exit_price - entry_price) / entry_price
                position_size = current_equity * 0.95  # Use 95% of equity for position
                
                # Update equity
                trade_pnl = position_size * pnl
                current_equity += trade_pnl
                
                # Record trade
                self.trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "pnl_pct": pnl * 100,
                    "pnl_dollars": trade_pnl,
                    "equity_after": current_equity
                })
                
                # Reset position
                position = 0
            
            # Record equity curve point
            self.equity_curve.append(current_equity)
        
        return True
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "total_return": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        
        # Convert equity curve to Series
        equity_series = pd.Series(self.equity_curve)
        
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        # Calculate drawdowns
        drawdowns = equity_series / equity_series.cummax() - 1
        
        # Calculate win/loss metrics
        winning_trades = [t for t in self.trades if t["pnl_dollars"] > 0]
        losing_trades = [t for t in self.trades if t["pnl_dollars"] <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        total_profits = sum(t["pnl_dollars"] for t in winning_trades)
        total_losses = abs(sum(t["pnl_dollars"] for t in losing_trades)) if losing_trades else 1
        
        profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        
        metrics = {
            "total_trades": len(self.trades),
            "total_return": (equity_series.iloc[-1] / equity_series.iloc[0] - 1) * 100,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "max_drawdown": drawdowns.min() * 100,
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            "final_equity": equity_series.iloc[-1]
        }
        
        return metrics
    
    def plot_results(self, filename=None):
        """Plot backtest results"""
        if not self.trades:
            print("No trades to plot")
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve)
        plt.title(f"Equity Curve - {self.symbol}")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        
        # Plot trades
        plt.subplot(2, 1, 2)
        for trade in self.trades:
            color = 'green' if trade["pnl_dollars"] > 0 else 'red'
            plt.plot([trade["entry_date"], trade["exit_date"]], 
                    [trade["entry_price"], trade["exit_price"]], 
                    color=color, linewidth=2)
        
        plt.title(f"Trades - {self.symbol}")
        plt.ylabel("Price ($)")
        plt.grid(True)
        
        # Save or show
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

# Example strategy functions
def moving_average_crossover(data, params):
    """Simple moving average crossover strategy"""
    # Add moving averages
    data['short_ma'] = data['close'].rolling(window=params['short_window']).mean()
    data['long_ma'] = data['close'].rolling(window=params['long_window']).mean()
    
    # Generate signals (1: buy, -1: sell, 0: hold)
    data['signal'] = 0
    data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
    data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
    
    return data

def test_backtest_functionality():
    """Test the backtest engine functionality"""
    # Use a known stock and period
    engine = BacktestEngine(
        symbol="AAPL",
        start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        end_date=datetime.now().strftime("%Y-%m-%d"),
        initial_capital=100000
    )
    
    # Run a simple moving average crossover strategy
    strategy_params = {
        'short_window': 20,
        'long_window': 50
    }
    
    # Apply strategy
    result = engine.apply_strategy(moving_average_crossover, strategy_params)
    assert result, "Strategy application failed"
    
    # Calculate metrics
    metrics = engine.calculate_metrics()
    
    # Basic assertions
    assert metrics["total_trades"] >= 0
    assert isinstance(metrics["total_return"], float)
    assert 0 <= metrics["win_rate"] <= 100
    assert metrics["profit_factor"] >= 0
    assert metrics["max_drawdown"] <= 0
    
    # Print results
    print(f"Backtest Results for {engine.symbol}:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save plot for review
    try:
        engine.plot_results("backtest_results.png")
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    return metrics

if __name__ == "__main__":
    # Run the backtest test function
    test_backtest_functionality() 