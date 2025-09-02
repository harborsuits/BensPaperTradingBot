#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alpha Vantage Enhanced Backtester

This module extends the UnifiedBacktester to use Alpha Vantage technical indicators
for improved backtesting accuracy and signal generation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import json
import os

from trading_bot.backtesting.unified_backtester import UnifiedBacktester
from trading_bot.signals.alpha_vantage_signals import AlphaVantageTechnicalSignals
from trading_bot.market.market_data import MarketData

logger = logging.getLogger(__name__)

class AlphaVantageBacktester(UnifiedBacktester):
    """
    Enhanced backtester that uses Alpha Vantage data for more accurate technical indicators
    and backtesting results.
    """
    
    def __init__(self, 
                initial_capital: float = 10000, 
                strategies: List[Dict] = None,
                start_date: str = None, 
                end_date: str = None,
                data_source: str = "alpha_vantage",
                api_key: Optional[str] = None,
                indicators_config: Dict = None,
                **kwargs):
        """
        Initialize the AlphaVantageBacktester.
        
        Args:
            initial_capital: Initial capital for backtesting
            strategies: List of strategy configurations
            start_date: Start date for backtesting in "YYYY-MM-DD" format
            end_date: End date for backtesting in "YYYY-MM-DD" format
            data_source: Data source to use (alpha_vantage, yahoo_finance, etc.)
            api_key: Alpha Vantage API key
            indicators_config: Configuration for technical indicators
            **kwargs: Additional arguments passed to the UnifiedBacktester
        """
        # Initialize parent class
        super().__init__(
            initial_capital=initial_capital,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            data_source=data_source,
            **kwargs
        )
        
        self.api_key = api_key
        self.indicators_config = indicators_config or {}
        
        # Initialize the Alpha Vantage technical signals
        self.av_signals = None
        
        logger.info("Initialized AlphaVantageBacktester")
    
    def _initialize_signals(self):
        """Initialize the AlphaVantageTechnicalSignals"""
        if self.av_signals is None and hasattr(self, 'market_data'):
            self.av_signals = AlphaVantageTechnicalSignals(
                market_data=self.market_data,
                api_key=self.api_key
            )
            logger.info("Initialized Alpha Vantage technical signals")
    
    def load_market_data(self, mock_data=None):
        """
        Load market data with enhanced Alpha Vantage data if available.
        
        Args:
            mock_data: Optional mock data for testing
        """
        # Call parent implementation first
        super().load_market_data(mock_data)
        
        # Initialize Alpha Vantage signals
        self._initialize_signals()
    
    def _enrich_market_data_with_indicators(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich market data with additional technical indicators from Alpha Vantage.
        
        Args:
            symbol: Stock symbol
            df: Market data DataFrame
            
        Returns:
            DataFrame with added technical indicators
        """
        if self.av_signals is None:
            self._initialize_signals()
            
        if self.av_signals is None:
            logger.warning("Alpha Vantage signals not initialized, skipping data enrichment")
            return df
        
        try:
            # Create a copy to avoid modifying the original
            enriched_df = df.copy()
            
            # Get the indicators to add based on configuration
            indicators_to_add = self.indicators_config.get("indicators", [
                {"name": "SMA", "period": 20},
                {"name": "SMA", "period": 50},
                {"name": "SMA", "period": 200},
                {"name": "RSI", "period": 14},
                {"name": "MACD", "fast_period": 12, "slow_period": 26, "signal_period": 9},
                {"name": "BBANDS", "period": 20, "std_dev": 2.0}
            ])
            
            # Add each indicator
            for indicator in indicators_to_add:
                name = indicator.get("name", "").lower()
                
                if name == "sma":
                    period = indicator.get("period", 20)
                    sma = self.av_signals.calculate_sma(symbol, period=period)
                    if sma is not None:
                        # Resample to match market data dates
                        sma = sma.reindex(enriched_df.index, method='ffill')
                        enriched_df[f'sma_{period}'] = sma
                
                elif name == "ema":
                    period = indicator.get("period", 20)
                    ema = self.av_signals.calculate_ema(symbol, period=period)
                    if ema is not None:
                        # Resample to match market data dates
                        ema = ema.reindex(enriched_df.index, method='ffill')
                        enriched_df[f'ema_{period}'] = ema
                
                elif name == "rsi":
                    period = indicator.get("period", 14)
                    rsi = self.av_signals.calculate_rsi(symbol, period=period)
                    if rsi is not None:
                        # Resample to match market data dates
                        rsi = rsi.reindex(enriched_df.index, method='ffill')
                        enriched_df[f'rsi_{period}'] = rsi
                
                elif name == "macd":
                    fast_period = indicator.get("fast_period", 12)
                    slow_period = indicator.get("slow_period", 26)
                    signal_period = indicator.get("signal_period", 9)
                    
                    macd_line, signal_line, histogram = self.av_signals.calculate_macd(
                        symbol, 
                        fast_period=fast_period, 
                        slow_period=slow_period, 
                        signal_period=signal_period
                    )
                    
                    if macd_line is not None and signal_line is not None and histogram is not None:
                        # Resample to match market data dates
                        macd_line = macd_line.reindex(enriched_df.index, method='ffill')
                        signal_line = signal_line.reindex(enriched_df.index, method='ffill')
                        histogram = histogram.reindex(enriched_df.index, method='ffill')
                        
                        enriched_df['macd_line'] = macd_line
                        enriched_df['macd_signal'] = signal_line
                        enriched_df['macd_histogram'] = histogram
                
                elif name == "bbands":
                    period = indicator.get("period", 20)
                    std_dev = indicator.get("std_dev", 2.0)
                    
                    upper_band, middle_band, lower_band = self.av_signals.calculate_bollinger_bands(
                        symbol, 
                        period=period, 
                        std_dev=std_dev
                    )
                    
                    if upper_band is not None and middle_band is not None and lower_band is not None:
                        # Resample to match market data dates
                        upper_band = upper_band.reindex(enriched_df.index, method='ffill')
                        middle_band = middle_band.reindex(enriched_df.index, method='ffill')
                        lower_band = lower_band.reindex(enriched_df.index, method='ffill')
                        
                        enriched_df['bb_upper'] = upper_band
                        enriched_df['bb_middle'] = middle_band
                        enriched_df['bb_lower'] = lower_band
                
                elif name == "adx":
                    period = indicator.get("period", 14)
                    adx = self.av_signals.calculate_adx(symbol, period=period)
                    if adx is not None:
                        # Resample to match market data dates
                        adx = adx.reindex(enriched_df.index, method='ffill')
                        enriched_df[f'adx_{period}'] = adx
                
                elif name == "atr":
                    period = indicator.get("period", 14)
                    atr = self.av_signals.calculate_atr(symbol, period=period)
                    if atr is not None:
                        # Resample to match market data dates
                        atr = atr.reindex(enriched_df.index, method='ffill')
                        enriched_df[f'atr_{period}'] = atr
            
            logger.info(f"Successfully enriched market data for {symbol} with Alpha Vantage indicators")
            return enriched_df
            
        except Exception as e:
            logger.error(f"Error enriching market data for {symbol}: {str(e)}")
            return df
    
    def get_data_for_symbol(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get market data for a symbol with enhanced technical indicators.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with market data and technical indicators
        """
        # Get base market data
        df = super().get_data_for_symbol(symbol, start_date, end_date)
        
        # Enrich with Alpha Vantage indicators
        if not df.empty and self.indicators_config.get("auto_enrich", True):
            df = self._enrich_market_data_with_indicators(symbol, df)
        
        return df
    
    def get_technical_summaries(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get technical analysis summaries for a list of symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary of technical summaries by symbol
        """
        if self.av_signals is None:
            self._initialize_signals()
            
        if self.av_signals is None:
            logger.error("Alpha Vantage signals not initialized")
            return {}
        
        summaries = {}
        for symbol in symbols:
            try:
                summary = self.av_signals.get_technical_summary(symbol)
                summaries[symbol] = summary
            except Exception as e:
                logger.error(f"Error getting technical summary for {symbol}: {str(e)}")
                summaries[symbol] = {"error": str(e)}
        
        return summaries
    
    def save_technical_summaries(self, symbols: List[str], output_dir: str = "data/technical_summaries"):
        """
        Save technical analysis summaries to JSON files.
        
        Args:
            symbols: List of stock symbols
            output_dir: Directory to save summaries
        """
        # Get technical summaries
        summaries = self.get_technical_summaries(symbols)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each summary to a file
        for symbol, summary in summaries.items():
            try:
                filename = os.path.join(output_dir, f"{symbol}_summary.json")
                with open(filename, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Saved technical summary for {symbol} to {filename}")
            except Exception as e:
                logger.error(f"Error saving technical summary for {symbol}: {str(e)}")
    
    def backtest_with_av_signals(self, strategy_name: str, symbols: List[str]) -> Dict:
        """
        Run a backtest using Alpha Vantage enhanced signals.
        
        Args:
            strategy_name: Name of the strategy to backtest
            symbols: List of symbols to include in the backtest
            
        Returns:
            Dictionary with backtest results
        """
        # Get technical summaries for pre-trade analysis
        summaries = self.get_technical_summaries(symbols)
        
        # Filter symbols based on technical analysis if needed
        if self.indicators_config.get("filter_symbols", False):
            filtered_symbols = []
            for symbol, summary in summaries.items():
                # Skip symbols with errors
                if "error" in summary:
                    continue
                
                # Apply filtering based on configuration
                filter_config = self.indicators_config.get("filter_criteria", {})
                
                if filter_config.get("require_signal", None) == "bullish":
                    if summary.get("overall_signal") != "bullish":
                        continue
                
                if filter_config.get("min_adx", 0) > 0:
                    adx = summary.get("indicators", {}).get("adx", 0)
                    if adx < filter_config["min_adx"]:
                        continue
                
                if filter_config.get("rsi_range", None):
                    rsi = summary.get("indicators", {}).get("rsi", 0)
                    rsi_min, rsi_max = filter_config["rsi_range"]
                    if rsi < rsi_min or rsi > rsi_max:
                        continue
                
                filtered_symbols.append(symbol)
            
            logger.info(f"Filtered symbols from {len(symbols)} to {len(filtered_symbols)} based on technical criteria")
            symbols = filtered_symbols
        
        # Run the backtest
        results = self.backtest(strategy_name=strategy_name, symbols=symbols)
        
        # Enhance results with technical summaries
        results["technical_summaries"] = summaries
        
        return results 