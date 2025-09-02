#!/usr/bin/env python3
"""
Strategy Deployment Tool - Implements the best evolved strategies in production-ready format
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
from typing import Dict, List, Any, Optional, Union

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Import registry module
from strategy_registry import StrategyRegistry


class StrategyDeployer:
    """
    Deploys evolved strategies for backtesting and live trading.
    
    Features:
    - Loads best strategies from registry
    - Creates production-ready implementations
    - Backtests on historical data
    - Prepares strategies for live trading
    """
    
    def __init__(self, registry_dir: Optional[str] = None):
        """
        Initialize the strategy deployer.
        
        Args:
            registry_dir: Directory containing strategy registry
                         (if None, will search for latest in test_results)
        """
        if registry_dir is None:
            # Find the latest registry directory
            registry_dir = self._find_latest_registry()
        
        print(f"Loading strategy registry from: {registry_dir}")
        self.registry = StrategyRegistry(registry_dir=registry_dir)
        
        # Directory for saving deployment files
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.deploy_dir = f"deployments/evolved_strategies_{timestamp}"
        
        if not os.path.exists(self.deploy_dir):
            os.makedirs(self.deploy_dir)
            
        print(f"Deployment files will be saved to: {self.deploy_dir}")
    
    def _find_latest_registry(self) -> str:
        """Find the latest strategy registry directory"""
        search_dir = "test_results"
        
        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"Cannot find test_results directory")
        
        # Find all enhanced_evolution directories
        evolution_dirs = []
        for entry in os.scandir(search_dir):
            if entry.is_dir() and entry.name.startswith("enhanced_evolution_"):
                registry_path = os.path.join(entry.path, "strategy_registry")
                if os.path.exists(registry_path):
                    evolution_dirs.append((entry.name, registry_path))
        
        if not evolution_dirs:
            raise FileNotFoundError(f"Cannot find any strategy registry in {search_dir}")
        
        # Sort by timestamp (most recent first)
        evolution_dirs.sort(reverse=True)
        
        return evolution_dirs[0][1]
    
    def get_top_strategies(self, 
                           top_n: int = 3, 
                           by_type: bool = True) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top N strategies overall or by type.
        
        Args:
            top_n: Number of top strategies to return
            by_type: Whether to get top N for each strategy type
            
        Returns:
            Dictionary mapping strategy type to list of strategy data
        """
        # Collect strategies with performance data
        strategies = []
        
        for strategy_id, strategy_data in self.registry.strategies.items():
            if strategy_id in self.registry.strategy_performance:
                perf = self.registry.strategy_performance[strategy_id]
                
                if 'overall' in perf:
                    overall = perf['overall']
                    
                    strategies.append({
                        'id': strategy_id,
                        'type': strategy_data['type'],
                        'parameters': strategy_data['parameters'],
                        'avg_return': overall.get('avg_return', 0),
                        'win_rate': overall.get('avg_win_rate', 0),
                        'max_drawdown': overall.get('avg_drawdown', 0),
                        'sharpe': overall.get('avg_sharpe', 0),
                        'robustness': overall.get('robustness_score', 0)
                    })
        
        if not strategies:
            print("No strategies found with performance data")
            return {}
            
        # Group by type if required
        result = {}
        
        if by_type:
            # Group strategies by type
            by_type_dict = {}
            for strategy in strategies:
                strategy_type = strategy['type']
                if strategy_type not in by_type_dict:
                    by_type_dict[strategy_type] = []
                by_type_dict[strategy_type].append(strategy)
            
            # Get top N for each type
            for strategy_type, strats in by_type_dict.items():
                # Sort by robustness
                strats.sort(key=lambda x: x['robustness'], reverse=True)
                result[strategy_type] = strats[:top_n]
        else:
            # Sort all strategies by robustness
            strategies.sort(key=lambda x: x['robustness'], reverse=True)
            result['overall'] = strategies[:top_n]
        
        return result
    
    def generate_strategy_code(self, strategy_data: Dict[str, Any]) -> str:
        """
        Generate production-ready code for a strategy.
        
        Args:
            strategy_data: Strategy data dictionary
            
        Returns:
            Python code string
        """
        strategy_type = strategy_data['type']
        parameters = strategy_data['parameters']
        
        code = f"""#!/usr/bin/env python3
\"\"\"
Evolved Trading Strategy: {strategy_type}
Strategy ID: {strategy_data['id']}
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Average Return: {strategy_data.get('avg_return', 0):.2f}%
- Win Rate: {strategy_data.get('win_rate', 0):.1f}%
- Max Drawdown: {strategy_data.get('max_drawdown', 0):.2f}%
- Robustness Score: {strategy_data.get('robustness', 0):.3f}
\"\"\"

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class Evolved{strategy_type.replace("Strategy", "")}:
    \"\"\"
    Evolved trading strategy based on {strategy_type}.
    This strategy was optimized through evolutionary algorithms against diverse market conditions.
    \"\"\"
    
    def __init__(self):
        \"\"\"Initialize the evolved strategy with optimized parameters\"\"\"
        self.name = "Evolved{strategy_type.replace("Strategy", "")}"
        self.parameters = {json.dumps(parameters, indent=8)}
        
"""
        
        # Generate strategy-specific code
        if "MovingAverageCrossover" in strategy_type:
            code += self._generate_ma_crossover_code()
        elif "BollingerBands" in strategy_type:
            code += self._generate_bollinger_bands_code()
        elif "RSI" in strategy_type:
            code += self._generate_rsi_code()
        else:
            # Generic placeholder
            code += self._generate_generic_code()
            
        # Add example usage
        code += """

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt
    
    # Get some test data
    symbol = "SPY"
    data = yf.download(symbol, start="2022-01-01")
    
    # Initialize strategy
    strategy = Evolved{0}()
    
    # Apply strategy to data
    signals = []
    for i in range(100, len(data)):
        chunk = data.iloc[:i+1]
        signal = strategy.calculate_signal(chunk)
        signals.append(signal["signal"])
    
    # Show results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(signals):], data['Close'][-len(signals):])
    
    # Mark buy signals
    buy_days = [day for i, day in enumerate(data.index[-len(signals):]) if signals[i] == "buy"]
    buy_prices = [data.loc[day, 'Close'] for day in buy_days]
    plt.scatter(buy_days, buy_prices, marker='^', color='green', s=100)
    
    # Mark sell signals
    sell_days = [day for i, day in enumerate(data.index[-len(signals):]) if signals[i] == "sell"]
    sell_prices = [data.loc[day, 'Close'] for day in sell_days]
    plt.scatter(sell_days, sell_prices, marker='v', color='red', s=100)
    
    plt.title(f"{{strategy.name}} Signals for {{symbol}}")
    plt.tight_layout()
    plt.show()
""".format(strategy_type.replace("Strategy", ""))
        
        return code
    
    def _generate_ma_crossover_code(self) -> str:
        """Generate code for MA Crossover strategy"""
        return """    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"
        Calculate trading signal based on moving average crossover.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        \"\"\"
        if len(market_data) < self.parameters["slow_period"]:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate fast and slow moving averages
        fast_ma = close_prices.rolling(window=self.parameters["fast_period"]).mean()
        slow_ma = close_prices.rolling(window=self.parameters["slow_period"]).mean()
        
        # Check for NaN values at the end
        if pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]):
            return {"signal": "none", "confidence": 0}
        
        # Get current values
        current_fast = float(fast_ma.iloc[-1])
        current_slow = float(slow_ma.iloc[-1])
        
        # Get previous values (for crossover detection)
        if len(fast_ma) > 1 and len(slow_ma) > 1:
            prev_fast = float(fast_ma.iloc[-2])
            prev_slow = float(slow_ma.iloc[-2])
        else:
            prev_fast, prev_slow = current_fast, current_slow
        
        # Calculate percentage difference
        diff_pct = abs(current_fast - current_slow) / current_slow
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Detect crossover and check threshold
        if diff_pct >= self.parameters["signal_threshold"]:
            if current_fast > current_slow and prev_fast <= prev_slow:
                # Bullish crossover
                signal = "buy"
                confidence = min(1.0, diff_pct * 10)
            elif current_fast < current_slow and prev_fast >= prev_slow:
                # Bearish crossover
                signal = "sell"
                confidence = min(1.0, diff_pct * 10)
        
        # Check for strong trends (not just crossover)
        elif current_fast > current_slow and diff_pct >= self.parameters["signal_threshold"] / 2:
            signal = "buy"
            confidence = min(0.7, diff_pct * 5)  # Lower confidence for trend following
        elif current_fast < current_slow and diff_pct >= self.parameters["signal_threshold"] / 2:
            signal = "sell"
            confidence = min(0.7, diff_pct * 5)  # Lower confidence for trend following
        
        return {
            "signal": signal,
            "confidence": confidence,
            "fast_ma": current_fast,
            "slow_ma": current_slow,
            "diff_pct": diff_pct
        }
"""
    
    def _generate_bollinger_bands_code(self) -> str:
        """Generate code for Bollinger Bands strategy"""
        return """    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"
        Calculate trading signal based on Bollinger Bands.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        \"\"\"
        if len(market_data) < self.parameters["period"]:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate Bollinger Bands
        rolling_mean = close_prices.rolling(window=self.parameters["period"]).mean()
        rolling_std = close_prices.rolling(window=self.parameters["period"]).std()
        
        upper_band = rolling_mean + (rolling_std * self.parameters["std_dev"])
        lower_band = rolling_mean - (rolling_std * self.parameters["std_dev"])
        
        # Check for NaN values at the end
        if pd.isna(rolling_mean.iloc[-1]) or pd.isna(rolling_std.iloc[-1]):
            return {"signal": "none", "confidence": 0}
        
        # Get current values
        current_price = float(close_prices.iloc[-1])
        current_upper = float(upper_band.iloc[-1])
        current_lower = float(lower_band.iloc[-1])
        current_middle = float(rolling_mean.iloc[-1])
        
        # Calculate band width (volatility measure)
        band_width = (current_upper - current_lower) / current_middle
        
        # Calculate how far price is into the band (normalized position)
        if current_upper != current_lower:  # Avoid division by zero
            normalized_position = (current_price - current_lower) / (current_upper - current_lower)
        else:
            normalized_position = 0.5
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Calculate thresholds
        lower_threshold = current_lower + (band_width * self.parameters["signal_threshold"])
        upper_threshold = current_upper - (band_width * self.parameters["signal_threshold"])
        
        # Generate signal logic
        if current_price <= lower_threshold:
            # Price at or below lower band
            signal = "buy"
            # Higher confidence when price is further below the band
            confidence = min(1.0, (lower_threshold - current_price) / current_lower * 5 + 0.5)
            if confidence < 0:
                confidence = 0.5  # Ensure positive confidence
        elif current_price >= upper_threshold:
            # Price at or above upper band
            signal = "sell"
            # Higher confidence when price is further above the band
            confidence = min(1.0, (current_price - upper_threshold) / current_upper * 5 + 0.5)
            if confidence < 0:
                confidence = 0.5  # Ensure positive confidence
                
        # Add trend filter
        if signal == "buy" and len(rolling_mean) >= 3:
            # Check if moving average is falling
            if rolling_mean.iloc[-1] < rolling_mean.iloc[-3]:
                confidence *= 0.7  # Reduce confidence in counter-trend signal
        elif signal == "sell" and len(rolling_mean) >= 3:
            # Check if moving average is rising
            if rolling_mean.iloc[-1] > rolling_mean.iloc[-3]:
                confidence *= 0.7  # Reduce confidence in counter-trend signal
        
        return {
            "signal": signal,
            "confidence": confidence,
            "current_price": current_price,
            "upper_band": current_upper,
            "lower_band": current_lower,
            "middle_band": current_middle,
            "band_width": band_width,
            "normalized_position": normalized_position
        }
"""
    
    def _generate_rsi_code(self) -> str:
        """Generate code for RSI strategy"""
        return """    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"
        Calculate trading signal based on Relative Strength Index (RSI).
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        \"\"\"
        if len(market_data) < self.parameters["period"] + 1:
            return {"signal": "none", "confidence": 0}
        
        # Get closing prices
        close_prices = market_data['Close'] if 'Close' in market_data.columns else market_data['close']
        
        # Calculate price changes
        delta = close_prices.diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Convert to positive values
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.parameters["period"]).mean()
        avg_loss = loss.rolling(window=self.parameters["period"]).mean()
        
        # Check for NaN values at the end
        if pd.isna(avg_gain.iloc[-1]) or pd.isna(avg_loss.iloc[-1]):
            return {"signal": "none", "confidence": 0}
        
        # Calculate RSI
        if float(avg_loss.iloc[-1]) == 0:
            rsi = 100
        else:
            rs = float(avg_gain.iloc[-1]) / float(avg_loss.iloc[-1])
            rsi = 100 - (100 / (1 + rs))
        
        # Initialize signal values
        signal = "none"
        confidence = 0
        
        # Generate signal logic
        if rsi <= self.parameters["oversold"]:
            # RSI in oversold territory
            signal = "buy"
            # Higher confidence when RSI is lower
            confidence = min(1.0, (self.parameters["oversold"] - rsi) / self.parameters["oversold"] * 2)
        elif rsi >= self.parameters["overbought"]:
            # RSI in overbought territory
            signal = "sell"
            # Higher confidence when RSI is higher
            confidence = min(1.0, (rsi - self.parameters["overbought"]) / (100 - self.parameters["overbought"]) * 2)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "rsi": rsi,
            "overbought_level": self.parameters["overbought"],
            "oversold_level": self.parameters["oversold"]
        }
"""
    
    def _generate_generic_code(self) -> str:
        """Generate generic strategy code"""
        return """    def calculate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"
        Calculate trading signal based on the evolved strategy parameters.
        
        Args:
            market_data: DataFrame with OHLCV data
            
        Returns:
            Signal dictionary
        \"\"\"
        # Implement strategy-specific logic here
        # Using self.parameters for the optimized parameters
        
        # For now, return a placeholder signal
        return {
            "signal": "none",
            "confidence": 0
        }
"""
    
    def deploy_top_strategies(self, top_n: int = 3) -> None:
        """
        Generate production code for top strategies.
        
        Args:
            top_n: Number of top strategies to deploy per type
        """
        # Get top strategies by type
        top_strategies = self.get_top_strategies(top_n=top_n, by_type=True)
        
        if not top_strategies:
            print("No strategies to deploy")
            return
        
        deployed_count = 0
        
        # Generate code files for each strategy
        for strategy_type, strategies in top_strategies.items():
            for i, strategy in enumerate(strategies):
                strategy_name = f"evolved_{strategy_type.lower().replace('strategy', '')}_{i+1}.py"
                file_path = os.path.join(self.deploy_dir, strategy_name)
                
                # Generate code
                code = self.generate_strategy_code(strategy)
                
                # Write to file
                with open(file_path, 'w') as f:
                    f.write(code)
                
                print(f"Deployed {strategy_type} to {file_path}")
                deployed_count += 1
        
        # Create README file
        readme_path = os.path.join(self.deploy_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write("# Evolved Trading Strategies\n\n")
            f.write(f"Deployed on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total strategies: {deployed_count}\n\n")
            
            f.write("## Strategy Files\n\n")
            for strategy_type, strategies in top_strategies.items():
                f.write(f"### {strategy_type}\n\n")
                for i, strategy in enumerate(strategies):
                    strategy_name = f"evolved_{strategy_type.lower().replace('strategy', '')}_{i+1}.py"
                    f.write(f"- [{strategy_name}]({strategy_name}): ")
                    f.write(f"Robustness: {strategy.get('robustness', 0):.3f}, ")
                    f.write(f"Return: {strategy.get('avg_return', 0):.2f}%, ")
                    f.write(f"Win Rate: {strategy.get('win_rate', 0):.1f}%\n")
                f.write("\n")
            
            f.write("## Usage Instructions\n\n")
            f.write("Each strategy file is self-contained and can be used independently:\n\n")
            f.write("```python\n")
            f.write("from evolved_strategy_file import EvolvedStrategyClass\n\n")
            f.write("# Initialize strategy\n")
            f.write("strategy = EvolvedStrategyClass()\n\n")
            f.write("# Calculate signal from market data\n")
            f.write("signal = strategy.calculate_signal(market_data)\n\n")
            f.write("# Use signal information\n")
            f.write("if signal['signal'] == 'buy':\n")
            f.write("    # Execute buy order\n")
            f.write("    pass\n")
            f.write("elif signal['signal'] == 'sell':\n")
            f.write("    # Execute sell order\n")
            f.write("    pass\n")
            f.write("```\n\n")
            
            f.write("## Integration\n\n")
            f.write("These strategies can be integrated into the Evotrader platform by:\n\n")
            f.write("1. Copying the strategy files to your project directory\n")
            f.write("2. Importing the strategy classes where needed\n")
            f.write("3. Using the `calculate_signal` method to generate trading signals\n")
        
        print(f"\nDeployed {deployed_count} strategies to {self.deploy_dir}")
        print(f"Deployment documentation: {readme_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Deploy evolved trading strategies")
    
    parser.add_argument(
        "--registry", 
        type=str, 
        default=None,
        help="Path to strategy registry directory (default: auto-detect latest)"
    )
    
    parser.add_argument(
        "--top", 
        type=int, 
        default=3,
        help="Number of top strategies to deploy per type"
    )
    
    args = parser.parse_args()
    
    try:
        deployer = StrategyDeployer(registry_dir=args.registry)
        deployer.deploy_top_strategies(top_n=args.top)
    except Exception as e:
        print(f"Error deploying strategies: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
