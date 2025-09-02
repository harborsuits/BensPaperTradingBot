#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Recognition Test Script

This script demonstrates the pattern recognition system's capabilities by:
1. Loading or generating test market data
2. Detecting patterns across different market regimes
3. Integrating pattern signals with the contextual trading system
4. Showing how pattern detection improves trade entry/exit decisions
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Tuple, Optional

# Import our pattern recognition components
from trading_bot.analysis.pattern_structure import PatternRegistry, MarketContext
from trading_bot.analysis.pattern_analyzer import PatternAnalyzer
from trading_bot.core.pattern_integration import PatternIntegrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PatternTest")

class PatternTester:
    """
    Tests the pattern recognition system with different market data and regimes.
    """
    
    def __init__(self, data_dir="data/test_data"):
        """Initialize the pattern tester"""
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize the pattern integration system
        self.pattern_manager = PatternIntegrationManager(
            pattern_registry_file="data/patterns/test_registry.json"
        )
        
        # Market data by symbol
        self.market_data = {}
        
        # Detected patterns storage
        self.detected_patterns = {}
        self.trade_signals = {}
        
        # Test statistics
        self.stats = {
            "total_bars": 0,
            "patterns_detected": 0,
            "signals_generated": 0,
            "by_regime": {},
            "by_pattern_type": {}
        }
    
    def generate_test_data(self, symbol="TEST", days=120, pattern_types=None):
        """
        Generate synthetic test data with embedded patterns.
        
        Args:
            symbol: Symbol name for the test data
            days: Number of days to generate
            pattern_types: List of pattern types to embed
            
        Returns:
            DataFrame with test data
        """
        logger.info(f"Generating {days} days of test data for {symbol}")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initial price
        price = 100.0
        
        # Data lists
        data = []
        
        # Regimes we'll cycle through
        regimes = [
            "trending_up", "volatile", "ranging", 
            "trending_down", "breakout", "ranging"
        ]
        
        # Generate price series with different regimes
        regime_idx = 0
        regime_duration = days // len(regimes)
        regime_days_left = regime_duration
        current_regime = regimes[regime_idx]
        
        # Trend direction and volatility state
        trend_direction = 1  # 1 for up, -1 for down
        volatility = 0.01    # Base volatility
        
        for i, date in enumerate(dates):
            # Check if we need to switch regimes
            if regime_days_left <= 0:
                regime_idx = (regime_idx + 1) % len(regimes)
                current_regime = regimes[regime_idx]
                regime_days_left = regime_duration
                
                logger.info(f"Switching to {current_regime} regime at day {i}")
                
                # Adjust parameters for the new regime
                if current_regime == "trending_up":
                    trend_direction = 1
                    volatility = 0.01
                elif current_regime == "trending_down":
                    trend_direction = -1
                    volatility = 0.01
                elif current_regime == "ranging":
                    trend_direction = 0
                    volatility = 0.005
                elif current_regime == "volatile":
                    trend_direction = trend_direction  # Keep current trend
                    volatility = 0.025
                elif current_regime == "breakout":
                    # Reverse trend direction for breakout
                    trend_direction = -trend_direction
                    volatility = 0.02
            
            # Generate price move based on regime
            base_move = 0
            
            # Add trend component
            if trend_direction != 0:
                base_move += trend_direction * 0.0005
            
            # Add random component
            random_move = np.random.normal(0, volatility)
            daily_return = base_move + random_move
            
            # Update price
            price *= (1 + daily_return)
            
            # Calculate high, low, open prices
            daily_range = price * volatility * 2
            high = price + daily_range / 2
            low = price - daily_range / 2
            open_price = low + np.random.uniform(0, daily_range)
            
            # Occasionally insert known patterns
            has_pattern = False
            pattern_name = None
            
            # Every ~10 days, insert a pattern
            if i % 10 == 0 and pattern_types:
                pattern_idx = (i // 10) % len(pattern_types)
                pattern_type = pattern_types[pattern_idx]
                
                if pattern_type == "pin_bar" and current_regime in ["trending_up", "trending_down", "reversal"]:
                    # Create a pin bar
                    has_pattern = True
                    pattern_name = "pin_bar"
                    
                    if current_regime == "trending_down":
                        # Bullish pin bar (long lower wick)
                        low = price - daily_range * 2
                        high = price + daily_range * 0.3
                        close_price = price + daily_range * 0.1
                        open_price = price
                    else:
                        # Bearish pin bar (long upper wick)
                        high = price + daily_range * 2
                        low = price - daily_range * 0.3
                        close_price = price - daily_range * 0.1
                        open_price = price
                
                elif pattern_type == "engulfing" and current_regime in ["trending_up", "trending_down"]:
                    # Create an engulfing pattern (need previous bar)
                    if i > 0:
                        has_pattern = True
                        pattern_name = "engulfing"
                        
                        prev_data = data[-1]
                        
                        if current_regime == "trending_down":
                            # Bullish engulfing
                            open_price = prev_data["Close"] * 0.99
                            close_price = prev_data["Open"] * 1.02
                            low = open_price * 0.99
                            high = close_price * 1.01
                        else:
                            # Bearish engulfing
                            open_price = prev_data["Close"] * 1.01
                            close_price = prev_data["Open"] * 0.98
                            low = close_price * 0.99
                            high = open_price * 1.01
                
                elif pattern_type == "double_bottom" and current_regime == "ranging":
                    # Simply adjust current bar, pattern would be detected over multiple bars
                    has_pattern = True
                    pattern_name = "double_bottom"
                    
                    # Make sure we have a low point
                    if i % 20 >= 18:  # At the end of ranging periods
                        low = price * 0.97
                        high = price * 1.02
                        close_price = price * 1.01
                        open_price = price * 0.99
            
            # Add to data list
            data.append({
                "Date": date,
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": price,  # Default close to price, unless overridden
                "Volume": np.random.randint(1000, 10000),
                "Regime": current_regime,
                "HasPattern": has_pattern,
                "PatternName": pattern_name
            })
            
            # Override close if set above
            if 'close_price' in locals():
                data[-1]["Close"] = close_price
                del close_price
            
            # Decrement regime counter
            regime_days_left -= 1
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # Calculate technical indicators
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Store data
        self.market_data[symbol] = df
        
        # Save to CSV
        csv_path = os.path.join(self.data_dir, f"{symbol}_test.csv")
        df.to_csv(csv_path)
        logger.info(f"Saved test data to {csv_path}")
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = -losses  # Make losses positive
        
        # Calculate average gain and loss
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def detect_patterns_in_data(self, symbol="TEST", min_confidence=0.5):
        """
        Detect patterns in the market data.
        
        Args:
            symbol: Symbol to analyze
            min_confidence: Minimum confidence threshold for pattern detection
            
        Returns:
            Dictionary with detection results by bar
        """
        if symbol not in self.market_data:
            logger.error(f"No data found for {symbol}")
            return {}
        
        df = self.market_data[symbol]
        
        # Ensure we have enough data for pattern detection
        if len(df) < 50:
            logger.warning(f"Not enough data for {symbol} ({len(df)} bars)")
            return {}
        
        logger.info(f"Detecting patterns in {len(df)} bars of {symbol} data")
        
        # Results storage
        results = {}
        
        # Minimum window size for pattern detection
        window_size = 20
        
        # Analyze each bar with a rolling window
        for i in range(window_size, len(df)):
            # Get current window and regime
            current_window = df.iloc[i-window_size:i+1]
            current_regime = current_window.iloc[-1].get('Regime', 'unknown')
            
            # Analyze patterns in this window
            analysis = self.pattern_manager.analyze_patterns(
                symbol=symbol,
                data=current_window,
                market_regime=current_regime
            )
            
            # Get a trade signal if patterns were found
            if analysis['has_patterns']:
                signal = self.pattern_manager.get_trade_signal(
                    symbol=symbol,
                    data=current_window,
                    market_regime=current_regime,
                    min_confidence=min_confidence
                )
                
                # Store signal
                if signal['signal'] != 'none':
                    self.trade_signals[i] = signal
                    
                    # Update statistics
                    self.stats['signals_generated'] += 1
                    
                    # Track by regime
                    if current_regime not in self.stats['by_regime']:
                        self.stats['by_regime'][current_regime] = {
                            'signals': 0, 'patterns': 0
                        }
                    self.stats['by_regime'][current_regime]['signals'] += 1
            
            # Store analysis results
            results[i] = {
                'date': df.index[i],
                'regime': current_regime,
                'has_patterns': analysis['has_patterns'],
                'patterns': analysis['patterns'],
                'signal': self.trade_signals.get(i, {'signal': 'none'})
            }
            
            # Update statistics
            if analysis['has_patterns']:
                self.stats['patterns_detected'] += len(analysis['patterns'])
                
                # Track by regime
                if current_regime not in self.stats['by_regime']:
                    self.stats['by_regime'][current_regime] = {
                        'signals': 0, 'patterns': 0
                    }
                self.stats['by_regime'][current_regime]['patterns'] += len(analysis['patterns'])
                
                # Track by pattern type
                for pattern in analysis['patterns']:
                    pattern_type = pattern.get('pattern_type', 'unknown')
                    
                    if pattern_type not in self.stats['by_pattern_type']:
                        self.stats['by_pattern_type'][pattern_type] = 0
                    
                    self.stats['by_pattern_type'][pattern_type] += 1
        
        # Store detected patterns
        self.detected_patterns[symbol] = results
        
        # Update total bars
        self.stats['total_bars'] += len(df)
        
        return results
    
    def simulate_pattern_based_trades(self, symbol="TEST"):
        """
        Simulate trades based on pattern signals.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with trade results
        """
        if symbol not in self.market_data or symbol not in self.detected_patterns:
            logger.error(f"No data or patterns for {symbol}")
            return {}
        
        df = self.market_data[symbol]
        patterns = self.detected_patterns[symbol]
        
        logger.info(f"Simulating pattern-based trades for {symbol}")
        
        # Trade tracking
        trades = []
        active_trade = None
        
        # Settings
        stop_loss_pct = 0.01  # 1% stop loss
        take_profit_pct = 0.02  # 2% take profit
        
        # Go through each bar where we detected patterns
        for i in sorted(patterns.keys()):
            if i >= len(df) - 1:  # Skip the last bar
                continue
                
            current_bar = df.iloc[i]
            next_bar = df.iloc[i+1]
            current_patterns = patterns[i]
            
            # If we have an active trade, check if it's closed
            if active_trade:
                entry_price = active_trade['entry_price']
                direction = active_trade['direction']
                
                # Calculate profit/loss based on direction
                if direction == 'buy':
                    stop_level = entry_price * (1 - stop_loss_pct)
                    target_level = entry_price * (1 + take_profit_pct)
                    
                    # Check if stopped out
                    if next_bar['Low'] <= stop_level:
                        pnl = (stop_level / entry_price - 1) * 100
                        active_trade['exit_price'] = stop_level
                        active_trade['exit_date'] = next_bar.name
                        active_trade['pnl'] = pnl
                        active_trade['exit_reason'] = 'stop_loss'
                        active_trade['pips'] = pnl * 100  # Convert to pips (approx)
                        
                        # Update pattern performance
                        self.pattern_manager.update_trade_outcome(
                            symbol=symbol,
                            success=False,
                            profit_pips=active_trade['pips'],
                            pattern_name=active_trade['pattern'],
                            market_regime=active_trade['regime']
                        )
                        
                        trades.append(active_trade)
                        active_trade = None
                    
                    # Check if target hit
                    elif next_bar['High'] >= target_level:
                        pnl = (target_level / entry_price - 1) * 100
                        active_trade['exit_price'] = target_level
                        active_trade['exit_date'] = next_bar.name
                        active_trade['pnl'] = pnl
                        active_trade['exit_reason'] = 'take_profit'
                        active_trade['pips'] = pnl * 100  # Convert to pips (approx)
                        
                        # Update pattern performance
                        self.pattern_manager.update_trade_outcome(
                            symbol=symbol,
                            success=True,
                            profit_pips=active_trade['pips'],
                            pattern_name=active_trade['pattern'],
                            market_regime=active_trade['regime']
                        )
                        
                        trades.append(active_trade)
                        active_trade = None
                
                elif direction == 'sell':
                    stop_level = entry_price * (1 + stop_loss_pct)
                    target_level = entry_price * (1 - take_profit_pct)
                    
                    # Check if stopped out
                    if next_bar['High'] >= stop_level:
                        pnl = (entry_price / stop_level - 1) * 100
                        active_trade['exit_price'] = stop_level
                        active_trade['exit_date'] = next_bar.name
                        active_trade['pnl'] = pnl
                        active_trade['exit_reason'] = 'stop_loss'
                        active_trade['pips'] = pnl * 100  # Convert to pips (approx)
                        
                        # Update pattern performance
                        self.pattern_manager.update_trade_outcome(
                            symbol=symbol,
                            success=False,
                            profit_pips=active_trade['pips'],
                            pattern_name=active_trade['pattern'],
                            market_regime=active_trade['regime']
                        )
                        
                        trades.append(active_trade)
                        active_trade = None
                    
                    # Check if target hit
                    elif next_bar['Low'] <= target_level:
                        pnl = (entry_price / target_level - 1) * 100
                        active_trade['exit_price'] = target_level
                        active_trade['exit_date'] = next_bar.name
                        active_trade['pnl'] = pnl
                        active_trade['exit_reason'] = 'take_profit'
                        active_trade['pips'] = pnl * 100  # Convert to pips (approx)
                        
                        # Update pattern performance
                        self.pattern_manager.update_trade_outcome(
                            symbol=symbol,
                            success=True,
                            profit_pips=active_trade['pips'],
                            pattern_name=active_trade['pattern'],
                            market_regime=active_trade['regime']
                        )
                        
                        trades.append(active_trade)
                        active_trade = None
            
            # Check for a new trade signal
            if not active_trade and i in self.trade_signals:
                signal = self.trade_signals[i]
                
                if signal['signal'] in ['buy', 'sell']:
                    # Create a new trade
                    active_trade = {
                        'entry_date': next_bar.name,
                        'entry_price': next_bar['Open'],
                        'direction': signal['signal'],
                        'pattern': signal['pattern'],
                        'confidence': signal['confidence'],
                        'adjusted_confidence': signal.get('adjusted_confidence', signal['confidence']),
                        'regime': current_patterns['regime']
                    }
                    
                    logger.info(f"New {signal['signal']} trade at {active_trade['entry_date']} "
                               f"based on {signal['pattern']} pattern in {current_patterns['regime']} regime")
        
        # Close any open trade at the end
        if active_trade:
            last_bar = df.iloc[-1]
            
            direction = active_trade['direction']
            entry_price = active_trade['entry_price']
            
            if direction == 'buy':
                pnl = (last_bar['Close'] / entry_price - 1) * 100
            else:
                pnl = (entry_price / last_bar['Close'] - 1) * 100
            
            active_trade['exit_price'] = last_bar['Close']
            active_trade['exit_date'] = last_bar.name
            active_trade['pnl'] = pnl
            active_trade['exit_reason'] = 'end_of_data'
            active_trade['pips'] = pnl * 100  # Convert to pips (approx)
            
            # Update pattern performance
            self.pattern_manager.update_trade_outcome(
                symbol=symbol,
                success=pnl > 0,
                profit_pips=active_trade['pips'],
                pattern_name=active_trade['pattern'],
                market_regime=active_trade['regime']
            )
            
            trades.append(active_trade)
        
        # Calculate trade statistics
        if trades:
            win_trades = [t for t in trades if t['pnl'] > 0]
            loss_trades = [t for t in trades if t['pnl'] <= 0]
            
            win_rate = len(win_trades) / len(trades) if trades else 0
            avg_win = sum(t['pnl'] for t in win_trades) / len(win_trades) if win_trades else 0
            avg_loss = sum(t['pnl'] for t in loss_trades) / len(loss_trades) if loss_trades else 0
            
            logger.info(f"Trade statistics for {symbol}:")
            logger.info(f"  Total trades: {len(trades)}")
            logger.info(f"  Win rate: {win_rate:.2%}")
            logger.info(f"  Average win: {avg_win:.2f}%")
            logger.info(f"  Average loss: {avg_loss:.2f}%")
            
            if win_trades and loss_trades:
                profit_factor = (sum(t['pnl'] for t in win_trades) / 
                               -sum(t['pnl'] for t in loss_trades))
                logger.info(f"  Profit factor: {profit_factor:.2f}")
        
        return {
            'trades': trades,
            'win_rate': win_rate if trades else 0,
            'avg_win': avg_win if win_trades else 0,
            'avg_loss': avg_loss if loss_trades else 0,
            'profit_factor': profit_factor if trades and win_trades and loss_trades else 0,
            'total_pnl': sum(t['pnl'] for t in trades)
        }
    
    def run_regime_specific_tests(self):
        """
        Run and compare pattern detection in different market regimes.
        
        Returns:
            Dictionary with test results by regime
        """
        # Generate test data with clear regime changes
        symbol = "REGIME_TEST"
        self.generate_test_data(symbol=symbol, days=120)
        
        # Detect patterns
        self.detect_patterns_in_data(symbol=symbol)
        
        # Simulate trades
        trade_results = self.simulate_pattern_based_trades(symbol=symbol)
        
        # Get patterns by regime
        regime_patterns = {}
        for idx, data in self.detected_patterns[symbol].items():
            regime = data['regime']
            
            if regime not in regime_patterns:
                regime_patterns[regime] = []
            
            if data['has_patterns']:
                regime_patterns[regime].extend(data['patterns'])
        
        # Calculate regime-specific pattern detection stats
        regime_stats = {}
        for regime, patterns in regime_patterns.items():
            # Group by pattern name
            pattern_counts = {}
            for p in patterns:
                name = p['pattern_name']
                if name not in pattern_counts:
                    pattern_counts[name] = 0
                pattern_counts[name] += 1
            
            # Sort by frequency
            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Store stats
            regime_stats[regime] = {
                'total_patterns': len(patterns),
                'unique_patterns': len(pattern_counts),
                'most_common': sorted_patterns[:3] if sorted_patterns else [],
                'pattern_counts': pattern_counts
            }
        
        # Get regime-specific trade performance
        if trade_results.get('trades'):
            # Group trades by regime
            regime_trades = {}
            for trade in trade_results['trades']:
                regime = trade['regime']
                
                if regime not in regime_trades:
                    regime_trades[regime] = []
                
                regime_trades[regime].append(trade)
            
            # Calculate performance by regime
            for regime, trades in regime_trades.items():
                win_trades = [t for t in trades if t['pnl'] > 0]
                
                win_rate = len(win_trades) / len(trades) if trades else 0
                avg_pnl = sum(t['pnl'] for t in trades) / len(trades) if trades else 0
                
                if regime in regime_stats:
                    regime_stats[regime]['trades'] = len(trades)
                    regime_stats[regime]['win_rate'] = win_rate
                    regime_stats[regime]['avg_pnl'] = avg_pnl
        
        # Generate best pattern report by regime
        best_patterns = {}
        for regime in regime_patterns.keys():
            # Convert to MarketContext format if needed
            if regime == "trending_up":
                context = "TRENDING_UP"
            elif regime == "trending_down":
                context = "TRENDING_DOWN"
            elif regime == "ranging":
                context = "RANGING"
            elif regime == "breakout":
                context = "BREAKOUT"
            elif regime == "volatile":
                context = "VOLATILE"
            else:
                context = regime.upper()
                
            best_patterns[regime] = self.pattern_manager.get_best_patterns_for_regime(context)
                
        return {
            'regime_patterns': regime_patterns,
            'regime_stats': regime_stats,
            'best_patterns': best_patterns,
            'trades': trade_results.get('trades', [])
        }
    
    def print_pattern_summary(self):
        """Print a summary of pattern detection performance"""
        # Overall statistics
        logger.info("\n=== PATTERN DETECTION SUMMARY ===")
        logger.info(f"Total bars analyzed: {self.stats['total_bars']}")
        logger.info(f"Total patterns detected: {self.stats['patterns_detected']}")
        logger.info(f"Total trade signals generated: {self.stats['signals_generated']}")
        
        # Pattern detection rate
        if self.stats['total_bars'] > 0:
            detection_rate = self.stats['patterns_detected'] / self.stats['total_bars']
            logger.info(f"Pattern detection rate: {detection_rate:.2%}")
        
        # By regime
        logger.info("\nPattern detection by regime:")
        for regime, stats in self.stats['by_regime'].items():
            logger.info(f"  {regime}: {stats['patterns']} patterns, {stats['signals']} signals")
        
        # By pattern type
        logger.info("\nPattern detection by type:")
        for pattern_type, count in self.stats['by_pattern_type'].items():
            logger.info(f"  {pattern_type}: {count} patterns")
        
        # Top patterns
        logger.info("\nTop patterns by regime:")
        for symbol in self.detected_patterns:
            # Count patterns by name for each regime
            regime_pattern_counts = {}
            
            for idx, data in self.detected_patterns[symbol].items():
                regime = data['regime']
                
                if regime not in regime_pattern_counts:
                    regime_pattern_counts[regime] = {}
                
                for pattern in data.get('patterns', []):
                    name = pattern['pattern_name']
                    
                    if name not in regime_pattern_counts[regime]:
                        regime_pattern_counts[regime][name] = 0
                    
                    regime_pattern_counts[regime][name] += 1
            
            # Print top patterns for each regime
            for regime, pattern_counts in regime_pattern_counts.items():
                if not pattern_counts:
                    continue
                    
                top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                logger.info(f"  {regime}: {', '.join([f'{name} ({count})' for name, count in top_patterns])}")
        
def main():
    """Main function to run pattern recognition tests"""
    tester = PatternTester()
    
    # Test with known patterns
    patterns_to_test = ["pin_bar", "engulfing", "double_bottom"]
    
    # Generate test data with these patterns
    tester.generate_test_data(
        symbol="TEST",
        days=120,
        pattern_types=patterns_to_test
    )
    
    # Detect patterns
    detected_patterns = tester.detect_patterns_in_data(symbol="TEST")
    
    # Simulate pattern-based trades
    trade_results = tester.simulate_pattern_based_trades(symbol="TEST")
    
    # Run regime-specific tests
    regime_tests = tester.run_regime_specific_tests()
    
    # Print summary
    tester.print_pattern_summary()
    
    # Return results
    return {
        'detected_patterns': detected_patterns,
        'trade_results': trade_results,
        'regime_tests': regime_tests
    }

if __name__ == "__main__":
    main()
