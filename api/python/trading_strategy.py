import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from trading_bot.tradier_integration import TradierIntegration

# Set up logging
logger = logging.getLogger(__name__)

class TradingStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    Concrete strategy implementations should inherit from this class and
    override the generate_signals method.
    """
    
    def __init__(self, symbols: List[str], **kwargs):
        """
        Initialize the trading strategy.
        
        Args:
            symbols: List of symbols the strategy will trade
            **kwargs: Additional strategy-specific parameters
        """
        self.symbols = symbols
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Set up any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @abstractmethod
    def generate_signals(self, historical_data: Dict[str, pd.DataFrame], current_date) -> Dict[str, Dict]:
        """
        Generate trading signals for the given date based on historical data.
        
        Args:
            historical_data: Dictionary mapping symbols to their historical data DataFrames
            current_date: The date for which to generate signals
            
        Returns:
            Dictionary mapping symbols to signal dictionaries
            
            Each signal dictionary should contain at minimum:
            {
                'action': str ('buy', 'sell', 'short', 'cover', or 'hold'),
                'risk_percent': float (percentage of capital to risk),
                'metadata': dict (optional additional signal information)
            }
        """
        pass
    
    def should_exit(self, symbol: str, current_price: float, trade: object, current_date) -> bool:
        """
        Determine if a position should be exited based on the current price and trade information.
        This method can be overridden by subclasses to implement custom exit logic,
        such as stop losses, take profits, or time-based exits.
        
        Args:
            symbol: The symbol being traded
            current_price: The current price of the asset
            trade: The Trade object representing the open position
            current_date: The current date
            
        Returns:
            True if the position should be exited, False otherwise
        """
        return False
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given DataFrame.
        This is a utility method that concrete strategies can use.
        
        Args:
            data: DataFrame containing price data
            
        Returns:
            DataFrame with added indicator columns
        """
        pass


class MovingAverageCrossover(TradingStrategy):
    """
    Simple moving average crossover strategy.
    
    Generates buy signals when the fast moving average crosses above the slow moving average,
    and sell signals when the fast moving average crosses below the slow moving average.
    """
    
    def __init__(self, symbols: List[str], fast_period: int = 10, slow_period: int = 30, **kwargs):
        """
        Initialize the moving average crossover strategy.
        
        Args:
            symbols: List of symbols to trade
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            **kwargs: Additional parameters
        """
        super().__init__(symbols, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        self.logger.info(f"Initialized MovingAverageCrossover strategy with fast_period={fast_period}, slow_period={slow_period}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate fast and slow moving averages
        df[f'ma_{self.fast_period}'] = df['close'].rolling(window=self.fast_period).mean()
        df[f'ma_{self.slow_period}'] = df['close'].rolling(window=self.slow_period).mean()
        
        # Calculate moving average crossover signals
        df['signal'] = 0
        df['signal'] = np.where(
            df[f'ma_{self.fast_period}'] > df[f'ma_{self.slow_period}'], 1, 0)
        
        # Generate crossover signal (signal changes from 0 to 1 or from 1 to 0)
        df['crossover'] = df['signal'].diff()
        
        return df
    
    def generate_signals(self, historical_data: Dict[str, pd.DataFrame], current_date) -> Dict[str, Dict]:
        """Generate trading signals based on moving average crossovers"""
        signals = {}
        
        for symbol, data in historical_data.items():
            # Skip if not enough data
            if len(data) < self.slow_period + 1:
                continue
                
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            # Get the current row
            current_data = df.loc[current_date]
            
            # Check for crossover signals
            if current_data['crossover'] == 1:  # Fast MA crosses above slow MA
                signals[symbol] = {
                    'action': 'buy',
                    'risk_percent': 1.0,
                    'metadata': {
                        'fast_ma': current_data[f'ma_{self.fast_period}'],
                        'slow_ma': current_data[f'ma_{self.slow_period}'],
                        'reason': 'ma_crossover_bullish'
                    }
                }
            elif current_data['crossover'] == -1:  # Fast MA crosses below slow MA
                signals[symbol] = {
                    'action': 'sell',
                    'risk_percent': 1.0,
                    'metadata': {
                        'fast_ma': current_data[f'ma_{self.fast_period}'],
                        'slow_ma': current_data[f'ma_{self.slow_period}'],
                        'reason': 'ma_crossover_bearish'
                    }
                }
        
        return signals
    
    def should_exit(self, symbol: str, current_price: float, trade: object, current_date) -> bool:
        """
        Check if we should exit the position based on stop loss or take profit.
        This example uses simple fixed percentage stops.
        """
        # Example stop loss at 2% and take profit at 5%
        stop_loss_pct = getattr(self, 'stop_loss_pct', 0.02)
        take_profit_pct = getattr(self, 'take_profit_pct', 0.05)
        
        if trade.direction == 'long':
            # Check stop loss
            if current_price <= trade.entry_price * (1 - stop_loss_pct):
                self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                return True
                
            # Check take profit
            if current_price >= trade.entry_price * (1 + take_profit_pct):
                self.logger.info(f"Take profit triggered for {symbol} at {current_price}")
                return True
        
        elif trade.direction == 'short':
            # Check stop loss (price goes up)
            if current_price >= trade.entry_price * (1 + stop_loss_pct):
                self.logger.info(f"Stop loss triggered for short {symbol} at {current_price}")
                return True
                
            # Check take profit (price goes down)
            if current_price <= trade.entry_price * (1 - take_profit_pct):
                self.logger.info(f"Take profit triggered for short {symbol} at {current_price}")
                return True
        
        return False


class RSIStrategy(TradingStrategy):
    """
    Relative Strength Index (RSI) based trading strategy.
    
    Generates buy signals when RSI crosses below the oversold threshold and then back above it,
    and sell signals when RSI crosses above the overbought threshold and then back below it.
    """
    
    def __init__(self, symbols: List[str], rsi_period: int = 14, 
                 oversold: int = 30, overbought: int = 70, **kwargs):
        """
        Initialize the RSI strategy.
        
        Args:
            symbols: List of symbols to trade
            rsi_period: Period for calculating RSI
            oversold: RSI threshold for oversold condition
            overbought: RSI threshold for overbought condition
            **kwargs: Additional parameters
        """
        super().__init__(symbols, **kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        
        # Track oversold/overbought conditions
        self.conditions = {symbol: {'was_oversold': False, 'was_overbought': False} 
                          for symbol in symbols}
        
        self.logger.info(f"Initialized RSI strategy with period={rsi_period}, "
                        f"oversold={oversold}, overbought={overbought}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator"""
        df = data.copy()
        
        # Calculate daily price changes
        delta = df['close'].diff()
        
        # Separate gains (up) and losses (down)
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        # Calculate exponential moving average (EMA) of gains and losses
        avg_gain = up.rolling(window=self.rsi_period).mean()
        avg_loss = down.rolling(window=self.rsi_period).mean()
        
        # Calculate relative strength (RS)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def generate_signals(self, historical_data: Dict[str, pd.DataFrame], current_date) -> Dict[str, Dict]:
        """Generate trading signals based on RSI values"""
        signals = {}
        
        for symbol, data in historical_data.items():
            # Skip if not enough data
            if len(data) < self.rsi_period + 1:
                continue
                
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            # Get the current row
            try:
                current_data = df.loc[current_date]
                current_rsi = current_data['rsi']
            except (KeyError, ValueError):
                continue
                
            # Skip if RSI is NaN
            if pd.isna(current_rsi):
                continue
                
            # Check for buy signal: RSI was below oversold and now crossing back above
            if current_rsi > self.oversold and self.conditions[symbol]['was_oversold']:
                signals[symbol] = {
                    'action': 'buy',
                    'risk_percent': 1.0,
                    'metadata': {
                        'rsi': current_rsi,
                        'reason': 'rsi_oversold_reversal'
                    }
                }
                self.conditions[symbol]['was_oversold'] = False
            
            # Check for sell signal: RSI was above overbought and now crossing back below
            elif current_rsi < self.overbought and self.conditions[symbol]['was_overbought']:
                signals[symbol] = {
                    'action': 'sell',
                    'risk_percent': 1.0,
                    'metadata': {
                        'rsi': current_rsi,
                        'reason': 'rsi_overbought_reversal'
                    }
                }
                self.conditions[symbol]['was_overbought'] = False
            
            # Update conditions
            if current_rsi <= self.oversold:
                self.conditions[symbol]['was_oversold'] = True
            
            if current_rsi >= self.overbought:
                self.conditions[symbol]['was_overbought'] = True
        
        return signals


class BollingerBandsStrategy(TradingStrategy):
    """
    Bollinger Bands trading strategy.
    
    Generates buy signals when price touches the lower band and sell signals
    when price touches the upper band.
    """
    
    def __init__(self, symbols: List[str], period: int = 20, std_dev: float = 2.0, **kwargs):
        """
        Initialize the Bollinger Bands strategy.
        
        Args:
            symbols: List of symbols to trade
            period: Period for calculating the moving average
            std_dev: Number of standard deviations for the bands
            **kwargs: Additional parameters
        """
        super().__init__(symbols, **kwargs)
        self.period = period
        self.std_dev = std_dev
        
        self.logger.info(f"Initialized Bollinger Bands strategy with period={period}, std_dev={std_dev}")
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands indicators"""
        df = data.copy()
        
        # Calculate middle band (SMA)
        df['middle_band'] = df['close'].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        df['std_dev'] = df['close'].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * self.std_dev)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * self.std_dev)
        
        # Calculate band width
        df['band_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        # Calculate % B (where price is relative to the bands)
        df['%b'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
        
        return df
    
    def generate_signals(self, historical_data: Dict[str, pd.DataFrame], current_date) -> Dict[str, Dict]:
        """Generate trading signals based on Bollinger Bands"""
        signals = {}
        
        for symbol, data in historical_data.items():
            # Skip if not enough data
            if len(data) < self.period + 1:
                continue
                
            # Calculate indicators
            df = self.calculate_indicators(data)
            
            # Get the current row
            try:
                current_data = df.loc[current_date]
                previous_day = df.loc[:current_date].iloc[-2]
            except (KeyError, ValueError, IndexError):
                continue
                
            # Skip if indicators are NaN
            if pd.isna(current_data['%b']):
                continue
                
            # Price crosses below lower band (buy signal)
            if (current_data['close'] <= current_data['lower_band'] and 
                previous_day['close'] > previous_day['lower_band']):
                signals[symbol] = {
                    'action': 'buy',
                    'risk_percent': 1.0,
                    'metadata': {
                        'percent_b': current_data['%b'],
                        'band_width': current_data['band_width'],
                        'reason': 'bollinger_lower_band_touch'
                    }
                }
            
            # Price crosses above upper band (sell signal if we're already in a position)
            elif (current_data['close'] >= current_data['upper_band'] and 
                  previous_day['close'] < previous_day['upper_band']):
                signals[symbol] = {
                    'action': 'sell',
                    'risk_percent': 1.0,
                    'metadata': {
                        'percent_b': current_data['%b'],
                        'band_width': current_data['band_width'],
                        'reason': 'bollinger_upper_band_touch'
                    }
                }
        
        return signals
    
    def should_exit(self, symbol: str, current_price: float, trade: object, current_date) -> bool:
        """
        Check if we should exit based on mean reversion (price returning to middle band)
        """
        # Only implement this logic if we have historical data available in the strategy
        # In a real implementation, you would need to pass historical_data to this method
        # or store it in the strategy instance
        
        return False  # Simplified for this example


class CombinedStrategy(TradingStrategy):
    """
    A strategy that combines signals from multiple sub-strategies.
    
    This strategy aggregates signals from multiple strategies and generates a final
    signal based on a consensus or weighted approach.
    """
    
    def __init__(self, symbols: List[str], strategies: List[TradingStrategy], **kwargs):
        """
        Initialize the combined strategy.
        
        Args:
            symbols: List of symbols to trade
            strategies: List of strategy instances to combine
            **kwargs: Additional parameters
        """
        super().__init__(symbols, **kwargs)
        self.strategies = strategies
        
        # Ensure all strategies have the same symbols
        for strategy in self.strategies:
            strategy.symbols = symbols
        
        self.logger.info(f"Initialized Combined strategy with {len(strategies)} sub-strategies")
    
    def generate_signals(self, historical_data: Dict[str, pd.DataFrame], current_date) -> Dict[str, Dict]:
        """Generate trading signals based on the consensus of sub-strategies"""
        all_signals = {}
        final_signals = {}
        
        # Collect signals from all sub-strategies
        for strategy in self.strategies:
            strategy_signals = strategy.generate_signals(historical_data, current_date)
            
            for symbol, signal in strategy_signals.items():
                if symbol not in all_signals:
                    all_signals[symbol] = []
                all_signals[symbol].append(signal)
        
        # Process collected signals to generate final signals
        for symbol, signals in all_signals.items():
            if not signals:
                continue
                
            # Count actions
            actions = [signal['action'] for signal in signals]
            buy_count = actions.count('buy')
            sell_count = actions.count('sell')
            short_count = actions.count('short')
            cover_count = actions.count('cover')
            
            # Simple majority vote
            total_strategies = len(self.strategies)
            
            # Generate final signal based on majority consensus
            if buy_count > total_strategies / 2:
                final_signals[symbol] = {
                    'action': 'buy',
                    'risk_percent': 1.0,
                    'metadata': {
                        'consensus': f"{buy_count}/{total_strategies}",
                        'reason': 'combined_majority_buy'
                    }
                }
            elif sell_count > total_strategies / 2:
                final_signals[symbol] = {
                    'action': 'sell',
                    'risk_percent': 1.0,
                    'metadata': {
                        'consensus': f"{sell_count}/{total_strategies}",
                        'reason': 'combined_majority_sell'
                    }
                }
            elif short_count > total_strategies / 2:
                final_signals[symbol] = {
                    'action': 'short',
                    'risk_percent': 1.0,
                    'metadata': {
                        'consensus': f"{short_count}/{total_strategies}",
                        'reason': 'combined_majority_short'
                    }
                }
            elif cover_count > total_strategies / 2:
                final_signals[symbol] = {
                    'action': 'cover',
                    'risk_percent': 1.0,
                    'metadata': {
                        'consensus': f"{cover_count}/{total_strategies}",
                        'reason': 'combined_majority_cover'
                    }
                }
        
        return final_signals
    
    def should_exit(self, symbol: str, current_price: float, trade: object, current_date) -> bool:
        """
        Check if we should exit based on the consensus of sub-strategies
        """
        # Check if any sub-strategy suggests exiting
        exit_votes = 0
        for strategy in self.strategies:
            if strategy.should_exit(symbol, current_price, trade, current_date):
                exit_votes += 1
        
        # Exit if at least one strategy suggests exiting
        return exit_votes > 0


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example: Initialize broker and strategy
    try:
        # Initialize Tradier integration
        tradier = TradierIntegration(
            api_key=os.environ.get("TRADIER_API_KEY", "demo_key"),
            account_id=os.environ.get("TRADIER_ACCOUNT_ID", "demo_account"),
            use_sandbox=True
        )
        
        # Define symbols to trade
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        # Create MA Crossover strategy
        strategy = MovingAverageCrossover(
            broker=tradier,
            symbols=symbols,
            config={
                "fast_period": 9,
                "slow_period": 21,
                "signal_period": 9,
                "ma_type": "ema",
                "lookback_days": 60,
                "min_volume": 1000000,
                "max_positions": 3,
                "position_size_percent": 5.0
            }
        )
        
        # Run strategy
        result = strategy.run()
        
        # Print results
        print(f"Strategy: {result['strategy']}")
        print(f"Run time: {result['run_time_seconds']:.2f} seconds")
        print(f"Signals: {result['signals_count']}")
        print(f"Trades executed: {result['trades_executed']}")
        
        # Print signals
        if result['signals']:
            print("\nSignals:")
            for symbol, signal in result['signals'].items():
                if signal['action'] != 'none':
                    print(f"{symbol}: {signal['action'].upper()} (confidence: {signal.get('confidence', 0):.2f})")
        
    except Exception as e:
        print(f"Error: {str(e)}") 