import numpy as np
from collections import deque, defaultdict
from datetime import datetime 
import time 
import logging 

class SignalGeneration:
    def __init__(self, buffer_size=30):
        """
        Initialize the SignalGeneration class with buffers for VWAP and signals.

        Parameters:
            buffer_size (int): Number of VWAP entries to keep for signal calculations.
        """
        self.vwap_buffers = defaultdict(lambda: deque(maxlen=buffer_size))  # VWAP buffer per ticker
        self.signals = {}  # Store the latest signals per ticker

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("finnhub_websocket.log"),
            ],
        )
    
    def add_vwap(self, latest_vwap):
        """
        Add the latest VWAP value to the buffer for the given ticker.

        Parameters:
            latest_vwap (dict): VWAP values dict for the tickers.
        """
        for ticker,vwap in latest_vwap.items():
            self.vwap_buffers[ticker].append(vwap)

    def collect_vwap(self,stock_pipeline):
        """
        Continuously fetch VWAP data from stock_pipeline and update SignalGeneration.
        """
        while True:
            # Wait for 5 seconds into the next minute
            current_time = datetime.now()
            sleep_seconds = (60 - current_time.second) + 5
            time.sleep(sleep_seconds)

            # Fetch the latest VWAP data
            latest_vwap = stock_pipeline.latest_vwap

            if latest_vwap:
                # Update SignalGeneration with the latest VWAP data
                self.add_vwap(latest_vwap)
                logging.info(f"VWAP updated for signal calculation at {datetime.now()}")
            
            if not latest_vwap:
                logging.warning(f"No VWAP data received at {datetime.now()}")
                        

    def sma_signal(self, vwap_array, fast_window=5, slow_window=30):
        """
        Generates SMA crossover signals using VWAP numpy array.

        Parameters:
            vwap_array (np.ndarray): 1D array containing VWAP prices.
            fast_window (int): Window size for the fast SMA.
            slow_window (int): Window size for the slow SMA.

        Returns:
            int: Signal value (1 for Buy, -1 for Sell, 0 for Hold).
        """
        if len(vwap_array) < slow_window:
            return 0  # Not enough data for signal generation

        fast_sma = np.mean(vwap_array[-fast_window:])
        slow_sma = np.mean(vwap_array[-slow_window:])

        if fast_sma > slow_sma and vwap_array[-fast_window - 1:].mean() <= vwap_array[-slow_window - 1:].mean():
            return 1  # Buy signal
        elif fast_sma < slow_sma and vwap_array[-fast_window - 1:].mean() >= vwap_array[-slow_window - 1:].mean():
            return -1  # Sell signal
        return 0  # Hold

    def calculate_rsi_signal(self, vwap_array, period=15, overbought=70, oversold=30):
        """
        Calculate RSI and generate trading signals using VWAP numpy array.

        Parameters:
            vwap_array (np.ndarray): 1D array containing VWAP prices.
            period (int): RSI period.
            overbought (int): Overbought threshold.
            oversold (int): Oversold threshold.

        Returns:
            int: Signal value (1 for Buy, -1 for Sell, 0 for Hold).
        """
        if len(vwap_array) < period:
            return 0  # Not enough data for RSI calculation

        changes = np.diff(vwap_array)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            rsi = 100  # Max RSI
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        if rsi > overbought:
            return -1  # Sell
        elif rsi < oversold:
            return 1  # Buy
        return 0  # Hold

    def stochastic_signal(self, vwap_array, lookback_period=14, overbought=80, oversold=20):
        """
        Calculates the Stochastic Oscillator and generates signals using VWAP numpy array.

        Parameters:
            vwap_array (np.ndarray): 1D array containing VWAP prices.
            lookback_period (int): Lookback period for the oscillator.
            overbought (int): Overbought threshold.
            oversold (int): Oversold threshold.

        Returns:
            int: Signal value (1 for Buy, -1 for Sell, 0 for Hold).
        """
        if len(vwap_array) < lookback_period:
            return 0  # Not enough data for Stochastic Oscillator

        lowest_low = np.min(vwap_array[-lookback_period:])
        highest_high = np.max(vwap_array[-lookback_period:])
        current_price = vwap_array[-1]
        k_percent = (current_price - lowest_low) / (highest_high - lowest_low) * 100

        if k_percent > overbought:
            return -1  # Sell
        elif k_percent < oversold:
            return 1  # Buy
        return 0  # Hold
    
    def breakout_signal(self, price_array, lookback_period=20, breakout_threshold=1.02, breakdown_threshold=0.98):
        """
        Calculates a Breakout Signal based on recent price highs and lows.

        Parameters:
            price_array (np.ndarray): 1D array containing price data.
            lookback_period (int): Lookback period to determine breakout levels.
            breakout_threshold (float): Multiplier for breakout level.
            breakdown_threshold (float): Multiplier for breakdown level.

        Returns:
            int: Signal value (1 for Buy, -1 for Sell, 0 for Hold).
        """
        if len(price_array) < lookback_period:
            return 0  # Not enough data for Breakout Signal

        highest_high = np.max(price_array[-lookback_period:])
        lowest_low = np.min(price_array[-lookback_period:])
        current_price = price_array[-1]

        if current_price > highest_high * breakout_threshold:
            return 1  # Buy
        elif current_price < lowest_low * breakdown_threshold:
            return -1  # Sell
        return 0  # Hold

    def calculate_signals(self):
        """
        Calculate all signals for the given ticker.

        Returns:
            dict: Dictionary of signals (SMA, RSI, Stochastic).
        """
        for ticker,vwap in self.vwap_buffers.items():

            vwap_array = np.array(vwap)
            signals = {
                "SMA": self.sma_signal(vwap_array),
                "RSI": self.calculate_rsi_signal(vwap_array),
                "Stochastic": self.stochastic_signal(vwap_array),
                'Breakout':self.breakout_signal(vwap_array)
            }
            self.signals[ticker] = signals

    def get_signals(self):
        """
        Retrieve all current signals.

        Returns:
            dict: Dictionary of signals for all tickers.
        """
        #print(self.vwap_buffers)
        self.calculate_signals()
        return self.signals
