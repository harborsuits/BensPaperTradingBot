import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple, Optional
import matplotlib.pyplot as plt

class TechnicalStrategy:
    """Base class for all technical indicator strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = {}
        self.positions = {}
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Generate trading signals for each symbol
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary with symbols as keys and price data as DataFrames
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with symbols as keys and signal DataFrames
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def calculate_positions(self, signals: Dict[str, pd.DataFrame], 
                           initial_capital: float) -> Dict[str, pd.DataFrame]:
        """
        Calculate positions and portfolio value based on signals
        
        Parameters:
        -----------
        signals : Dict[str, pd.DataFrame]
            Dictionary with symbols as keys and signal DataFrames
        initial_capital : float
            Initial capital to start trading with
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with symbols as keys and position DataFrames
        """
        self.positions = {}
        total_symbols = len(signals)
        capital_per_symbol = initial_capital / total_symbols
        
        for symbol, signal_df in signals.items():
            # Make a copy of the DataFrame to avoid modifying the original
            positions = signal_df.copy()
            
            # Add position column (1 for long, -1 for short, 0 for no position)
            positions['position'] = positions['signal'].diff()
            
            # Add holdings and cash columns
            positions['holdings'] = positions['close'] * positions['position'] * capital_per_symbol
            positions['cash'] = capital_per_symbol - positions['holdings'].cumsum()
            
            # Add total column (holdings + cash)
            positions['total'] = positions['cash'] + positions['holdings']
            
            self.positions[symbol] = positions
            
        return self.positions
    
    def calculate_returns(self) -> Dict[str, float]:
        """
        Calculate returns for each symbol
        
        Returns:
        --------
        Dict[str, float]
            Dictionary with symbols as keys and returns as values
        """
        returns = {}
        
        for symbol, position_df in self.positions.items():
            # Calculate returns
            initial_value = position_df['total'].iloc[0]
            final_value = position_df['total'].iloc[-1]
            returns[symbol] = (final_value - initial_value) / initial_value * 100
            
        return returns
    
    def plot_signals(self, symbol: str, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot trading signals and positions for a symbol
        
        Parameters:
        -----------
        symbol : str
            Symbol to plot signals for
        figsize : Tuple[int, int]
            Figure size
        """
        if symbol not in self.signals or symbol not in self.positions:
            raise ValueError(f"No signals or positions for symbol {symbol}")
            
        signals_df = self.signals[symbol]
        positions_df = self.positions[symbol]
        
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot price and signals on first axis
        ax1.plot(signals_df.index, signals_df['close'], label='Price')
        
        # Plot buy signals
        buy_signals = signals_df[signals_df['signal'] == 1]
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   marker='^', color='green', s=100, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = signals_df[signals_df['signal'] == -1]
        ax1.scatter(sell_signals.index, sell_signals['close'], 
                   marker='v', color='red', s=100, label='Sell Signal')
        
        # Add title and legend
        ax1.set_title(f'{self.name} Strategy - {symbol}')
        ax1.set_ylabel('Price')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # Plot portfolio value on second axis
        ax2.plot(positions_df.index, positions_df['total'], label='Portfolio Value')
        
        # Add title and legend
        ax2.set_title(f'Portfolio Value - {symbol}')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Value')
        ax2.legend(loc='best')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


class SMACrossover(TechnicalStrategy):
    """
    Simple Moving Average Crossover strategy
    
    Buy when short-term SMA crosses above long-term SMA
    Sell when short-term SMA crosses below long-term SMA
    """
    
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(name=f"SMA Crossover ({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals based on SMA crossover"""
        signals = {}
        
        for symbol, df in data.items():
            # Make a copy of the DataFrame to avoid modifying the original
            signals_df = df.copy()
            
            # Calculate short and long moving averages
            signals_df[f'sma_{self.short_window}'] = signals_df['close'].rolling(window=self.short_window).mean()
            signals_df[f'sma_{self.long_window}'] = signals_df['close'].rolling(window=self.long_window).mean()
            
            # Initialize signal column
            signals_df['signal'] = 0
            
            # Generate signals
            signals_df['signal'] = np.where(
                signals_df[f'sma_{self.short_window}'] > signals_df[f'sma_{self.long_window}'], 1, 0
            )
            
            # Remove rows with NaN values
            signals_df = signals_df.dropna()
            
            signals[symbol] = signals_df
            
        self.signals = signals
        return signals


class MACDStrategy(TechnicalStrategy):
    """
    Moving Average Convergence Divergence (MACD) strategy
    
    Buy when MACD line crosses above signal line
    Sell when MACD line crosses below signal line
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(name=f"MACD ({fast_period}/{slow_period}/{signal_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals based on MACD crossover"""
        signals = {}
        
        for symbol, df in data.items():
            # Make a copy of the DataFrame to avoid modifying the original
            signals_df = df.copy()
            
            # Calculate MACD components
            signals_df['ema_fast'] = signals_df['close'].ewm(span=self.fast_period, adjust=False).mean()
            signals_df['ema_slow'] = signals_df['close'].ewm(span=self.slow_period, adjust=False).mean()
            
            # Calculate MACD line and signal line
            signals_df['macd'] = signals_df['ema_fast'] - signals_df['ema_slow']
            signals_df['signal_line'] = signals_df['macd'].ewm(span=self.signal_period, adjust=False).mean()
            signals_df['histogram'] = signals_df['macd'] - signals_df['signal_line']
            
            # Initialize signal column
            signals_df['signal'] = 0
            
            # Generate signals (1 for long, -1 for short, 0 for no position)
            signals_df['signal'] = np.where(
                signals_df['macd'] > signals_df['signal_line'], 1, 0
            )
            
            # Remove rows with NaN values
            signals_df = signals_df.dropna()
            
            signals[symbol] = signals_df
            
        self.signals = signals
        return signals


class RSIStrategy(TechnicalStrategy):
    """
    Relative Strength Index (RSI) strategy
    
    Buy when RSI crosses above oversold threshold from below
    Sell when RSI crosses below overbought threshold from above
    """
    
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__(name=f"RSI ({period}, {oversold}/{overbought})")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals based on RSI thresholds"""
        signals = {}
        
        for symbol, df in data.items():
            # Make a copy of the DataFrame to avoid modifying the original
            signals_df = df.copy()
            
            # Calculate daily price changes
            signals_df['price_change'] = signals_df['close'].diff()
            
            # Calculate gains and losses
            signals_df['gain'] = np.where(signals_df['price_change'] > 0, signals_df['price_change'], 0)
            signals_df['loss'] = np.where(signals_df['price_change'] < 0, -signals_df['price_change'], 0)
            
            # Calculate average gains and losses
            signals_df['avg_gain'] = signals_df['gain'].rolling(window=self.period).mean()
            signals_df['avg_loss'] = signals_df['loss'].rolling(window=self.period).mean()
            
            # Calculate RS and RSI
            signals_df['rs'] = signals_df['avg_gain'] / signals_df['avg_loss']
            signals_df['rsi'] = 100 - (100 / (1 + signals_df['rs']))
            
            # Initialize signal column
            signals_df['signal'] = 0
            
            # Generate signals
            # Buy when RSI crosses above oversold threshold from below
            # Sell when RSI crosses below overbought threshold from above
            for i in range(1, len(signals_df)):
                if signals_df['rsi'].iloc[i-1] < self.oversold and signals_df['rsi'].iloc[i] >= self.oversold:
                    signals_df['signal'].iloc[i] = 1  # Buy signal
                elif signals_df['rsi'].iloc[i-1] > self.overbought and signals_df['rsi'].iloc[i] <= self.overbought:
                    signals_df['signal'].iloc[i] = -1  # Sell signal
            
            # Remove rows with NaN values
            signals_df = signals_df.dropna()
            
            signals[symbol] = signals_df
            
        self.signals = signals
        return signals


class BollingerBandsStrategy(TechnicalStrategy):
    """
    Bollinger Bands strategy
    
    Buy when price crosses above lower band from below
    Sell when price crosses below upper band from above
    """
    
    def __init__(self, period: int = 20, num_std: int = 2):
        super().__init__(name=f"Bollinger Bands ({period}, {num_std}σ)")
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals based on Bollinger Bands"""
        signals = {}
        
        for symbol, df in data.items():
            # Make a copy of the DataFrame to avoid modifying the original
            signals_df = df.copy()
            
            # Calculate middle band (SMA)
            signals_df['middle_band'] = signals_df['close'].rolling(window=self.period).mean()
            
            # Calculate standard deviation
            signals_df['std'] = signals_df['close'].rolling(window=self.period).std()
            
            # Calculate upper and lower bands
            signals_df['upper_band'] = signals_df['middle_band'] + (signals_df['std'] * self.num_std)
            signals_df['lower_band'] = signals_df['middle_band'] - (signals_df['std'] * self.num_std)
            
            # Initialize signal column
            signals_df['signal'] = 0
            
            # Generate signals
            # Buy when price crosses above lower band from below
            # Sell when price crosses below upper band from above
            for i in range(1, len(signals_df)):
                if signals_df['close'].iloc[i-1] < signals_df['lower_band'].iloc[i-1] and \
                   signals_df['close'].iloc[i] >= signals_df['lower_band'].iloc[i]:
                    signals_df['signal'].iloc[i] = 1  # Buy signal
                elif signals_df['close'].iloc[i-1] > signals_df['upper_band'].iloc[i-1] and \
                     signals_df['close'].iloc[i] <= signals_df['upper_band'].iloc[i]:
                    signals_df['signal'].iloc[i] = -1  # Sell signal
            
            # Remove rows with NaN values
            signals_df = signals_df.dropna()
            
            signals[symbol] = signals_df
            
        self.signals = signals
        return signals


class MultiIndicatorStrategy(TechnicalStrategy):
    """
    Multi-indicator strategy that combines SMA, MACD, RSI, and Bollinger Bands
    
    A buy signal is generated when at least threshold_buy indicators give buy signals
    A sell signal is generated when at least threshold_sell indicators give sell signals
    """
    
    def __init__(self, threshold_buy: int = 2, threshold_sell: int = 2):
        super().__init__(name=f"Multi-Indicator (Buy:{threshold_buy}, Sell:{threshold_sell})")
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        
        # Initialize individual strategies
        self.sma_strategy = SMACrossover()
        self.macd_strategy = MACDStrategy()
        self.rsi_strategy = RSIStrategy()
        self.bb_strategy = BollingerBandsStrategy()
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals by combining multiple indicators"""
        # Generate signals for each individual strategy
        sma_signals = self.sma_strategy.generate_signals(data)
        macd_signals = self.macd_strategy.generate_signals(data)
        rsi_signals = self.rsi_strategy.generate_signals(data)
        bb_signals = self.bb_strategy.generate_signals(data)
        
        signals = {}
        
        for symbol in data.keys():
            # Get signals from individual strategies
            sma_df = sma_signals[symbol]
            macd_df = macd_signals[symbol]
            rsi_df = rsi_signals[symbol]
            bb_df = bb_signals[symbol]
            
            # Create a new DataFrame with common index
            common_index = sma_df.index.intersection(macd_df.index).intersection(rsi_df.index).intersection(bb_df.index)
            signals_df = pd.DataFrame(index=common_index)
            
            # Add close price
            signals_df['close'] = data[symbol].loc[common_index]['close']
            
            # Add indicator signals
            signals_df['sma_signal'] = sma_df.loc[common_index]['signal']
            signals_df['macd_signal'] = macd_df.loc[common_index]['signal']
            signals_df['rsi_signal'] = rsi_df.loc[common_index]['signal']
            signals_df['bb_signal'] = bb_df.loc[common_index]['signal']
            
            # Count buy and sell signals
            signals_df['buy_count'] = (signals_df['sma_signal'] == 1).astype(int) + \
                                    (signals_df['macd_signal'] == 1).astype(int) + \
                                    (signals_df['rsi_signal'] == 1).astype(int) + \
                                    (signals_df['bb_signal'] == 1).astype(int)
                                    
            signals_df['sell_count'] = (signals_df['sma_signal'] == -1).astype(int) + \
                                     (signals_df['macd_signal'] == -1).astype(int) + \
                                     (signals_df['rsi_signal'] == -1).astype(int) + \
                                     (signals_df['bb_signal'] == -1).astype(int)
            
            # Generate signals based on thresholds
            signals_df['signal'] = 0
            signals_df.loc[signals_df['buy_count'] >= self.threshold_buy, 'signal'] = 1
            signals_df.loc[signals_df['sell_count'] >= self.threshold_sell, 'signal'] = -1
            
            signals[symbol] = signals_df
            
        self.signals = signals
        return signals 