#!/usr/bin/env python3
"""
Advanced Strategy Types - Additional strategy types for evolutionary optimization
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)


class MeanReversionStrategy:
    """
    Mean reversion strategy that trades on price deviations from a moving average.
    
    Mean reversion assumes that prices tend to return to their historical average over time.
    This strategy enters positions when prices move significantly away from the average
    and closes them when prices return to the average.
    """
    
    def __init__(self, lookback_period: int = 20, 
                 entry_std: float = 2.0, 
                 exit_std: float = 0.5, 
                 smoothing: int = 3):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for calculating the moving average
            entry_std: Number of standard deviations for entry signals
            exit_std: Number of standard deviations for exit signals
            smoothing: Period for smoothing the z-score
        """
        self.strategy_name = "MeanReversion"
        self.parameters = {
            "lookback_period": lookback_period,
            "entry_std": entry_std,
            "exit_std": exit_std,
            "smoothing": smoothing
        }
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on mean reversion.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if len(market_data) < self.parameters["lookback_period"] + 5:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate moving average and standard deviation
        ma = close_prices.rolling(window=self.parameters["lookback_period"]).mean()
        std = close_prices.rolling(window=self.parameters["lookback_period"]).std()
        
        # Calculate z-score
        z_score = (close_prices - ma) / std
        
        # Smooth z-score
        smooth_z = z_score.rolling(window=self.parameters["smoothing"]).mean()
        
        # Get latest values
        current_z = smooth_z.iloc[-1]
        current_price = close_prices.iloc[-1]
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Generate signal based on z-score
        if np.isnan(current_z):
            return {"signal": "none", "confidence": 0}
        
        # Entry signals
        if current_z <= -self.parameters["entry_std"]:
            # Price is significantly below average -> buy signal
            signal = "buy"
            confidence = min(1.0, abs(current_z) / self.parameters["entry_std"])
        elif current_z >= self.parameters["entry_std"]:
            # Price is significantly above average -> sell signal
            signal = "sell"
            confidence = min(1.0, abs(current_z) / self.parameters["entry_std"])
        
        # Exit signals (only provide if already in a position)
        # This information can be used by a portfolio manager to exit positions
        # when prices return to the average
        exit_long = current_z >= -self.parameters["exit_std"]
        exit_short = current_z <= self.parameters["exit_std"]
        
        return {
            "signal": signal,
            "confidence": confidence,
            "z_score": current_z,
            "exit_long": exit_long,
            "exit_short": exit_short
        }


class MomentumStrategy:
    """
    Momentum strategy that trades on price trend strength and continuation.
    
    Momentum assumes that assets that have performed well will continue to perform well
    in the short to medium term, and vice versa for poorly performing assets.
    This strategy measures price momentum over different timeframes.
    """
    
    def __init__(self, 
                short_period: int = 14, 
                medium_period: int = 30, 
                long_period: int = 90,
                threshold: float = 0.02,
                smoothing: int = 3):
        """
        Initialize momentum strategy.
        
        Args:
            short_period: Period for short-term momentum calculation
            medium_period: Period for medium-term momentum calculation
            long_period: Period for long-term momentum calculation
            threshold: Signal threshold as percentage change
            smoothing: Period for smoothing the momentum
        """
        self.strategy_name = "Momentum"
        self.parameters = {
            "short_period": short_period,
            "medium_period": medium_period,
            "long_period": long_period,
            "threshold": threshold,
            "smoothing": smoothing
        }
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on momentum.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if len(market_data) < self.parameters["long_period"] + 5:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate price changes over different periods
        short_momentum = close_prices.pct_change(self.parameters["short_period"])
        medium_momentum = close_prices.pct_change(self.parameters["medium_period"])
        long_momentum = close_prices.pct_change(self.parameters["long_period"])
        
        # Smooth momentum
        short_momentum_smooth = short_momentum.rolling(window=self.parameters["smoothing"]).mean()
        medium_momentum_smooth = medium_momentum.rolling(window=self.parameters["smoothing"]).mean()
        long_momentum_smooth = long_momentum.rolling(window=self.parameters["smoothing"]).mean()
        
        # Get latest values
        current_short = short_momentum_smooth.iloc[-1]
        current_medium = medium_momentum_smooth.iloc[-1]
        current_long = long_momentum_smooth.iloc[-1]
        
        # Check for NaN values
        if np.isnan(current_short) or np.isnan(current_medium) or np.isnan(current_long):
            return {"signal": "none", "confidence": 0}
        
        # Calculate weighted momentum score
        momentum_score = (
            0.5 * current_short +
            0.3 * current_medium +
            0.2 * current_long
        )
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Generate signal based on momentum score
        if momentum_score > self.parameters["threshold"]:
            # Strong positive momentum -> buy signal
            signal = "buy"
            confidence = min(1.0, momentum_score / self.parameters["threshold"])
        elif momentum_score < -self.parameters["threshold"]:
            # Strong negative momentum -> sell signal
            signal = "sell"
            confidence = min(1.0, abs(momentum_score) / self.parameters["threshold"])
        
        return {
            "signal": signal,
            "confidence": confidence,
            "momentum_score": momentum_score,
            "short_momentum": current_short,
            "medium_momentum": current_medium,
            "long_momentum": current_long
        }


class VolumeProfileStrategy:
    """
    Volume Profile strategy that trades based on volume-price relationships.
    
    Volume Profile analyzes the amount of volume traded at specific price levels
    to identify support and resistance areas based on where most trading occurs.
    """
    
    def __init__(self, 
                lookback_period: int = 20, 
                volume_threshold: float = 1.5,
                price_levels: int = 20,
                smoothing: int = 3):
        """
        Initialize volume profile strategy.
        
        Args:
            lookback_period: Period for volume profile calculation
            volume_threshold: Volume multiple to identify significant levels
            price_levels: Number of price levels to divide the range into
            smoothing: Period for smoothing signals
        """
        self.strategy_name = "VolumeProfile"
        self.parameters = {
            "lookback_period": lookback_period,
            "volume_threshold": volume_threshold,
            "price_levels": price_levels,
            "smoothing": smoothing
        }
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on volume profile.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if len(market_data) < self.parameters["lookback_period"] + 5:
            return {"signal": "none", "confidence": 0}
        
        # Extract last N periods for analysis
        lookback_data = market_data.iloc[-self.parameters["lookback_period"]:]
        
        # Get price and volume data
        close_prices = lookback_data['Close'] if 'Close' in lookback_data.columns else lookback_data['close']
        volumes = lookback_data['Volume'] if 'Volume' in lookback_data.columns else lookback_data['volume']
        
        # Calculate price range
        price_min = close_prices.min()
        price_max = close_prices.max()
        price_range = price_max - price_min
        
        # Calculate volume-weighted price levels
        if price_range == 0:  # Avoid division by zero
            return {"signal": "none", "confidence": 0}
        
        # Create price bins
        bin_size = price_range / self.parameters["price_levels"]
        price_bins = [price_min + i*bin_size for i in range(self.parameters["price_levels"]+1)]
        
        # Initialize volume profile
        volume_profile = np.zeros(self.parameters["price_levels"])
        
        # Build volume profile
        for i in range(len(lookback_data)):
            price = close_prices.iloc[i]
            volume = volumes.iloc[i]
            
            # Determine which bin this price falls into
            bin_idx = min(int((price - price_min) / bin_size), self.parameters["price_levels"]-1)
            volume_profile[bin_idx] += volume
        
        # Normalize volume profile
        avg_volume = np.mean(volume_profile[volume_profile > 0])
        if avg_volume == 0:
            return {"signal": "none", "confidence": 0}
        
        normalized_profile = volume_profile / avg_volume
        
        # Identify high volume areas (potential support/resistance)
        high_volume_areas = [
            (price_bins[i], price_bins[i+1], normalized_profile[i])
            for i in range(len(normalized_profile))
            if normalized_profile[i] >= self.parameters["volume_threshold"]
        ]
        
        # Current price
        current_price = close_prices.iloc[-1]
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        nearest_level = None
        nearest_distance = float('inf')
        nearest_volume = 0
        
        # Generate signal based on relationship to high volume areas
        for low, high, volume in high_volume_areas:
            # Calculate distance as percentage of price
            if current_price >= low and current_price <= high:
                # Price is within a high volume area
                # No signal as this is likely a consolidation area
                distance = 0
            elif current_price < low:
                # Price is below high volume area (potential resistance)
                distance = (low - current_price) / current_price
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = low
                    nearest_volume = volume
                    
                    # Only generate signal if we're very close to the level
                    if distance < 0.02:  # Within 2% of level
                        signal = "sell"
                        confidence = min(1.0, volume / self.parameters["volume_threshold"])
            else:  # current_price > high
                # Price is above high volume area (potential support)
                distance = (current_price - high) / current_price
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_level = high
                    nearest_volume = volume
                    
                    # Only generate signal if we're very close to the level
                    if distance < 0.02:  # Within 2% of level
                        signal = "buy"
                        confidence = min(1.0, volume / self.parameters["volume_threshold"])
        
        return {
            "signal": signal,
            "confidence": confidence,
            "nearest_level": nearest_level,
            "nearest_distance": nearest_distance,
            "nearest_volume": nearest_volume,
            "high_volume_areas": high_volume_areas
        }


class VolatilityBreakoutStrategy:
    """
    Volatility Breakout strategy that trades on price movements beyond recent volatility ranges.
    
    This strategy measures the average true range (ATR) to quantify volatility and 
    generates signals when price moves beyond its expected volatility range.
    """
    
    def __init__(self, 
                atr_period: int = 14,
                breakout_multiple: float = 1.5,
                lookback_period: int = 5,
                filter_threshold: float = 0.2):
        """
        Initialize volatility breakout strategy.
        
        Args:
            atr_period: Period for ATR calculation
            breakout_multiple: ATR multiple for breakout threshold
            lookback_period: Period to confirm breakout
            filter_threshold: Threshold for trend filter
        """
        self.strategy_name = "VolatilityBreakout"
        self.parameters = {
            "atr_period": atr_period,
            "breakout_multiple": breakout_multiple,
            "lookback_period": lookback_period,
            "filter_threshold": filter_threshold
        }
    
    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading signal based on volatility breakout.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        """
        if len(market_data) < self.parameters["atr_period"] + 10:
            return {"signal": "none", "confidence": 0}
        
        # Get price data
        high_prices = market_data['High'] if 'High' in market_data.columns else market_data['high']
        low_prices = market_data['Low'] if 'Low' in market_data.columns else market_data['low']
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate ATR
        # First, calculate True Range
        tr1 = high_prices - low_prices
        tr2 = abs(high_prices - close_prices.shift(1))
        tr3 = abs(low_prices - close_prices.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=self.parameters["atr_period"]).mean()
        
        # Calculate price channels
        lookback = self.parameters["lookback_period"]
        upper_channel = close_prices.rolling(window=lookback).max()
        lower_channel = close_prices.rolling(window=lookback).min()
        
        # Get latest values
        current_price = close_prices.iloc[-1]
        current_atr = atr.iloc[-1]
        current_upper = upper_channel.iloc[-1]
        current_lower = lower_channel.iloc[-1]
        
        # Calculate breakout thresholds
        breakout_range = current_atr * self.parameters["breakout_multiple"]
        upper_breakout = current_upper + breakout_range
        lower_breakout = current_lower - breakout_range
        
        # Calculate short-term trend
        short_trend = close_prices.pct_change(5).iloc[-1]
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Generate signal based on breakouts
        if np.isnan(current_atr) or np.isnan(current_upper) or np.isnan(current_lower):
            return {"signal": "none", "confidence": 0}
        
        if current_price > upper_breakout and short_trend > self.parameters["filter_threshold"]:
            # Upward breakout with confirming trend
            signal = "buy"
            confidence = min(1.0, (current_price - current_upper) / breakout_range)
        elif current_price < lower_breakout and short_trend < -self.parameters["filter_threshold"]:
            # Downward breakout with confirming trend
            signal = "sell"
            confidence = min(1.0, (current_lower - current_price) / breakout_range)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "atr": current_atr,
            "upper_breakout": upper_breakout,
            "lower_breakout": lower_breakout,
            "short_trend": short_trend
        }


def backtest_strategy(strategy, market_data: pd.DataFrame, 
                     initial_capital: float = 10000.0,
                     position_size_pct: float = 0.95,
                     commission_pct: float = 0.001) -> Dict[str, Any]:
    """
    Backtest a strategy on historical market data.
    
    Args:
        strategy: Strategy object with calculate_signal method
        market_data: DataFrame with OHLCV data
        initial_capital: Starting capital
        position_size_pct: Percentage of capital to use per trade
        commission_pct: Commission percentage per trade
        
    Returns:
        Dictionary with backtest results
    """
    print(f"Backtesting {strategy.strategy_name} on {len(market_data)} data points")
    
    # Initialize backtest variables
    equity = [initial_capital]
    position = 0
    entry_price = 0
    entry_idx = 0
    trades = []
    
    # Process each day
    for i in range(100, len(market_data)-1):  # Start after warmup period
        # Get data up to current day
        current_data = market_data.iloc[:i+1]
        next_day = market_data.iloc[i+1]
        
        # Calculate signal
        signal = strategy.calculate_signal(current_data)
        
        # Process signal for next day
        if signal["signal"] == "buy" and position <= 0:
            # Close any existing short position
            if position < 0:
                exit_price = next_day["open"]
                profit = (entry_price - exit_price) * abs(position)
                profit -= abs(position) * exit_price * commission_pct  # Subtract commission
                
                trade = {
                    "type": "exit_short",
                    "entry_date": market_data.index[entry_idx],
                    "exit_date": next_day.name,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit": profit,
                    "profit_pct": profit / (entry_price * abs(position)) * 100
                }
                trades.append(trade)
                
                # Update equity
                equity.append(equity[-1] + profit)
            else:
                # No position to close, just copy last equity value
                equity.append(equity[-1])
            
            # Enter long position
            position = (equity[-1] * position_size_pct) / next_day["open"]
            position -= position * commission_pct  # Subtract commission
            entry_price = next_day["open"]
            entry_idx = i+1
            
            trades.append({
                "type": "entry_long",
                "date": next_day.name,
                "price": entry_price,
                "position": position,
                "confidence": signal["confidence"]
            })
        
        elif signal["signal"] == "sell" and position >= 0:
            # Close any existing long position
            if position > 0:
                exit_price = next_day["open"]
                profit = (exit_price - entry_price) * position
                profit -= position * exit_price * commission_pct  # Subtract commission
                
                trade = {
                    "type": "exit_long",
                    "entry_date": market_data.index[entry_idx],
                    "exit_date": next_day.name,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "profit": profit,
                    "profit_pct": profit / (entry_price * position) * 100
                }
                trades.append(trade)
                
                # Update equity
                equity.append(equity[-1] + profit)
            else:
                # No position to close, just copy last equity value
                equity.append(equity[-1])
            
            # Enter short position
            position = -(equity[-1] * position_size_pct) / next_day["open"]
            position -= abs(position) * commission_pct  # Subtract commission
            entry_price = next_day["open"]
            entry_idx = i+1
            
            trades.append({
                "type": "entry_short",
                "date": next_day.name,
                "price": entry_price,
                "position": position,
                "confidence": signal["confidence"]
            })
        
        else:
            # No action, just copy last equity value
            equity.append(equity[-1])
    
    # Close final position
    if position != 0:
        last_price = market_data["close"].iloc[-1]
        
        if position > 0:
            profit = (last_price - entry_price) * position
            profit -= position * last_price * commission_pct  # Subtract commission
            
            trades.append({
                "type": "exit_long",
                "entry_date": market_data.index[entry_idx],
                "exit_date": market_data.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "profit": profit,
                "profit_pct": profit / (entry_price * position) * 100
            })
        else:
            profit = (entry_price - last_price) * abs(position)
            profit -= abs(position) * last_price * commission_pct  # Subtract commission
            
            trades.append({
                "type": "exit_short",
                "entry_date": market_data.index[entry_idx],
                "exit_date": market_data.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "profit": profit,
                "profit_pct": profit / (entry_price * abs(position)) * 100
            })
        
        # Update final equity
        equity.append(equity[-1] + profit)
    
    # Calculate performance metrics
    if len(trades) > 0:
        # Filter completed trades (entry and exit pairs)
        completed_trades = [t for t in trades if t["type"].startswith("exit")]
        
        if completed_trades:
            # Calculate returns
            total_return_pct = (equity[-1] - initial_capital) / initial_capital * 100
            
            # Calculate winning trades
            winning_trades = [t for t in completed_trades if t["profit"] > 0]
            win_rate = len(winning_trades) / len(completed_trades) * 100 if completed_trades else 0
            
            # Calculate max drawdown
            peak = equity[0]
            max_drawdown = 0
            
            for eq in equity:
                if eq > peak:
                    peak = eq
                drawdown = (peak - eq) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate additional metrics
            avg_profit_pct = sum(t["profit_pct"] for t in completed_trades) / len(completed_trades) if completed_trades else 0
            
            profit_trades = [t["profit_pct"] for t in completed_trades if t["profit"] > 0]
            loss_trades = [t["profit_pct"] for t in completed_trades if t["profit"] <= 0]
            
            avg_win = sum(profit_trades) / len(profit_trades) if profit_trades else 0
            avg_loss = sum(loss_trades) / len(loss_trades) if loss_trades else 0
            
            profit_factor = abs(sum(profit_trades) / sum(loss_trades)) if sum(loss_trades) != 0 else float('inf')
        else:
            # No completed trades
            total_return_pct = 0
            win_rate = 0
            max_drawdown = 0
            avg_profit_pct = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            completed_trades = []
    else:
        # No trades executed
        total_return_pct = 0
        win_rate = 0
        max_drawdown = 0
        avg_profit_pct = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        completed_trades = []
    
    print(f"{strategy.strategy_name} Results:")
    print(f"Total Return: {total_return_pct:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Trades: {len(completed_trades)}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    return {
        "equity_curve": equity,
        "trades": trades,
        "total_return_pct": total_return_pct,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "trade_count": len(completed_trades),
        "avg_profit_pct": avg_profit_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "final_equity": equity[-1]
    }


if __name__ == "__main__":
    import argparse
    import yfinance as yf
    
    parser = argparse.ArgumentParser(description="Test advanced trading strategies")
    
    parser.add_argument(
        "--strategy", 
        type=str, 
        default="all",
        choices=["mean_reversion", "momentum", "volume_profile", "volatility_breakout", "all"],
        help="Strategy to test"
    )
    
    parser.add_argument(
        "--symbol", 
        type=str, 
        default="SPY",
        help="Symbol to backtest on"
    )
    
    parser.add_argument(
        "--start", 
        type=str, 
        default="2022-01-01",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end", 
        type=str, 
        default=None,
        help="End date for backtest (YYYY-MM-DD), default is today"
    )
    
    args = parser.parse_args()
    
    # Download market data
    print(f"Downloading data for {args.symbol} from {args.start} to {args.end or 'today'}")
    data = yf.download(args.symbol, start=args.start, end=args.end)
    
    if len(data) == 0:
        print(f"No data found for {args.symbol}")
        sys.exit(1)
        
    print(f"Downloaded {len(data)} data points")
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_results/advanced_strategies_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run tests
    if args.strategy == "mean_reversion" or args.strategy == "all":
        strategy = MeanReversionStrategy()
        results = backtest_strategy(strategy, data)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title(f"Mean Reversion Strategy - {args.symbol}")
        plt.xlabel("Trading Days")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{args.symbol}_mean_reversion.png"))
    
    if args.strategy == "momentum" or args.strategy == "all":
        strategy = MomentumStrategy()
        results = backtest_strategy(strategy, data)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title(f"Momentum Strategy - {args.symbol}")
        plt.xlabel("Trading Days")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{args.symbol}_momentum.png"))
    
    if args.strategy == "volume_profile" or args.strategy == "all":
        strategy = VolumeProfileStrategy()
        results = backtest_strategy(strategy, data)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title(f"Volume Profile Strategy - {args.symbol}")
        plt.xlabel("Trading Days")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{args.symbol}_volume_profile.png"))
    
    if args.strategy == "volatility_breakout" or args.strategy == "all":
        strategy = VolatilityBreakoutStrategy()
        results = backtest_strategy(strategy, data)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(results['equity_curve'])
        plt.title(f"Volatility Breakout Strategy - {args.symbol}")
        plt.xlabel("Trading Days")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{args.symbol}_volatility_breakout.png"))
    
    print(f"All results saved to {output_dir}")
