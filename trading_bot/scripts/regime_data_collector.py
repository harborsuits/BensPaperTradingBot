#!/usr/bin/env python
"""
Regime Data Collector

Script for downloading and analyzing historical data to populate
initial regime classifications and prepare datasets for ML training.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
import time
import yfinance as yf

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import regime components
from trading_bot.analytics.market_regime.detector import MarketRegimeDetector, MarketRegimeType
from trading_bot.core.event_bus import EventBus
from trading_bot.analytics.market_regime.features import RegimeFeaturesCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("regime_data_collector")

class HistoricalDataCollector:
    """Collects and analyzes historical data for regime classification."""
    
    def __init__(self, output_dir, config=None):
        """
        Initialize the historical data collector.
        
        Args:
            output_dir: Directory to store collected data
            config: Configuration parameters
        """
        self.output_dir = output_dir
        self.config = config or {}
        
        # Create output directories
        self.data_dir = os.path.join(output_dir, "historical_data")
        self.regime_dir = os.path.join(output_dir, "regime_data")
        self.features_dir = os.path.join(output_dir, "feature_data")
        self.plots_dir = os.path.join(output_dir, "regime_plots")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.regime_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Create event bus (not actually used for events, just required for detector)
        self.event_bus = EventBus()
        
        # Create detector with minimal config
        detector_config = {
            "auto_update": False,
            "min_data_points": 100,
            "emit_events": False
        }
        self.detector = MarketRegimeDetector(self.event_bus, detector_config)
        
        # Disable broker manager
        self.detector.broker_manager = None
        
        # Configure download settings
        self.periods = {
            "1d": "5y",   # 5 years for daily data
            "1h": "60d",  # 60 days for hourly data
            "15m": "10d"  # 10 days for 15-minute data
        }
        
        # Features to track
        self.feature_columns = [
            'trend_strength', 'volatility', 'momentum', 'trading_range',
            'rsi', 'macd', 'bollinger_width', 'atr_percent', 'volume_change'
        ]
        
        logger.info("Historical Data Collector initialized")
    
    def download_data(self, symbols, timeframes):
        """
        Download historical data for the specified symbols and timeframes.
        
        Args:
            symbols: List of symbols to download data for
            timeframes: List of timeframes to download
            
        Returns:
            Dict mapping symbols to timeframes to DataFrames
        """
        logger.info(f"Downloading data for {len(symbols)} symbols and {len(timeframes)} timeframes")
        
        # Convert timeframes to yfinance intervals
        interval_map = {
            "1d": "1d",
            "1h": "1h",
            "15m": "15m"
        }
        
        # Check for unsupported timeframes
        for tf in timeframes:
            if tf not in interval_map:
                logger.warning(f"Unsupported timeframe: {tf}, skipping")
        
        # Filter to supported timeframes
        supported_timeframes = [tf for tf in timeframes if tf in interval_map]
        
        # Download data for each symbol and timeframe
        data = {}
        
        for symbol in tqdm(symbols, desc="Downloading symbols"):
            data[symbol] = {}
            
            for tf in supported_timeframes:
                try:
                    # Download data
                    interval = interval_map[tf]
                    period = self.periods.get(tf, "1mo")
                    
                    # Download with yfinance
                    df = yf.download(
                        symbol, 
                        period=period, 
                        interval=interval, 
                        auto_adjust=True,
                        progress=False
                    )
                    
                    # Skip if no data
                    if df.empty:
                        logger.warning(f"No data for {symbol} {tf}")
                        continue
                    
                    # Rename columns to lowercase
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Ensure OHLCV columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if not all(col in df.columns for col in required_cols):
                        logger.warning(f"Missing required columns for {symbol} {tf}")
                        continue
                    
                    # Store in result
                    data[symbol][tf] = df
                    
                    # Save to file
                    output_file = os.path.join(self.data_dir, f"{symbol}_{tf}.csv")
                    df.to_csv(output_file)
                    
                    logger.debug(f"Downloaded {len(df)} bars for {symbol} {tf}")
                    
                    # Avoid rate limits
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Error downloading {symbol} {tf}: {str(e)}")
        
        logger.info(f"Downloaded data for {len(data)} symbols")
        return data
    
    def analyze_regimes(self, data, window_size=100, step_size=20):
        """
        Analyze historical data to classify market regimes.
        
        Args:
            data: Dict mapping symbols to timeframes to DataFrames
            window_size: Number of bars to use for regime detection
            step_size: Number of bars to step between detections
            
        Returns:
            Dict mapping symbols to timeframes to regime DataFrames
        """
        logger.info(f"Analyzing regimes for {len(data)} symbols")
        
        # Prepare detector
        features_calculator = self.detector.features_calculator
        classifier = self.detector.classifier
        
        # Store results
        regime_data = {}
        feature_data = {}
        
        # Process each symbol and timeframe
        for symbol in tqdm(data.keys(), desc="Analyzing regimes"):
            regime_data[symbol] = {}
            feature_data[symbol] = {}
            
            for tf, df in data[symbol].items():
                # Skip if not enough data
                if len(df) < window_size:
                    logger.warning(f"Not enough data for {symbol} {tf}, need at least {window_size} bars")
                    continue
                
                # Prepare results array
                results = []
                features_list = []
                
                # Add symbol to detector
                self.detector.add_symbol(symbol, [tf])
                
                # Mock the _get_ohlcv_data method to use our historical data
                def get_ohlcv_mock(s, t, count=None):
                    return df.copy()
                
                self.detector._get_ohlcv_data = get_ohlcv_mock
                
                # Analyze regimes in windows
                for i in range(window_size, len(df), step_size):
                    try:
                        window = df.iloc[i-window_size:i]
                        
                        # Calculate features
                        features = features_calculator.calculate_features(window)
                        
                        # Classify regime
                        regime, confidence = classifier.classify_regime(features)
                        
                        # Store result
                        results.append({
                            'timestamp': df.index[i-1],
                            'regime': regime.value,
                            'confidence': confidence,
                            'close': df.iloc[i-1]['close']
                        })
                        
                        # Store features (selected subset)
                        feature_row = {
                            'timestamp': df.index[i-1],
                            'regime': regime.value,
                            'confidence': confidence,
                            'close': df.iloc[i-1]['close']
                        }
                        
                        # Add features
                        for feature in self.feature_columns:
                            if feature in features:
                                feature_row[feature] = features[feature]
                        
                        features_list.append(feature_row)
                        
                    except Exception as e:
                        logger.error(f"Error analyzing window for {symbol} {tf} at index {i}: {str(e)}")
                
                # Convert to DataFrame
                if results:
                    results_df = pd.DataFrame(results)
                    features_df = pd.DataFrame(features_list)
                    
                    # Set timestamp as index
                    results_df.set_index('timestamp', inplace=True)
                    features_df.set_index('timestamp', inplace=True)
                    
                    # Store results
                    regime_data[symbol][tf] = results_df
                    feature_data[symbol][tf] = features_df
                    
                    # Save to file
                    regime_file = os.path.join(self.regime_dir, f"{symbol}_{tf}_regimes.csv")
                    feature_file = os.path.join(self.features_dir, f"{symbol}_{tf}_features.csv")
                    
                    results_df.to_csv(regime_file)
                    features_df.to_csv(feature_file)
                    
                    logger.debug(f"Analyzed {len(results_df)} regime points for {symbol} {tf}")
                    
                    # Generate plot
                    self._generate_regime_plot(symbol, tf, df, results_df)
                
                # Clear from detector
                if symbol in self.detector.tracked_symbols:
                    self.detector.remove_symbol(symbol)
        
        logger.info(f"Completed regime analysis for {len(regime_data)} symbols")
        return regime_data, feature_data
    
    def _generate_regime_plot(self, symbol, timeframe, price_df, regime_df):
        """Generate a plot of price and regime changes."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot price
            ax1.plot(price_df.index[-len(regime_df)*3:], price_df['close'][-len(regime_df)*3:])
            ax1.set_title(f"{symbol} {timeframe} Price")
            ax1.grid(True)
            
            # Plot regimes
            # Map regime strings to numeric values for plotting
            regime_map = {regime.value: i for i, regime in enumerate(MarketRegimeType)}
            regime_values = [regime_map.get(r, 0) for r in regime_df['regime']]
            
            # Create scatter plot
            scatter = ax2.scatter(
                regime_df.index, 
                regime_values, 
                c=regime_df['confidence'], 
                cmap='viridis', 
                alpha=0.7,
                s=30
            )
            
            # Add colorbar for confidence
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Confidence')
            
            # Set y-axis labels
            ax2.set_yticks(list(regime_map.values()))
            ax2.set_yticklabels(list(regime_map.keys()))
            ax2.set_title("Market Regimes")
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.plots_dir, f"{symbol}_{timeframe}_regime_plot.png")
            plt.savefig(plot_file, dpi=150)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error generating plot for {symbol} {timeframe}: {str(e)}")
    
    def prepare_training_dataset(self, feature_data):
        """
        Prepare a dataset for ML training.
        
        Args:
            feature_data: Dict mapping symbols to timeframes to feature DataFrames
            
        Returns:
            DataFrame with combined features for ML training
        """
        logger.info("Preparing training dataset")
        
        # Combine all feature data
        all_features = []
        
        for symbol, timeframes in feature_data.items():
            for tf, df in timeframes.items():
                # Add symbol and timeframe columns
                df = df.copy()
                df['symbol'] = symbol
                df['timeframe'] = tf
                
                # Add to list
                all_features.append(df)
        
        # Combine into single DataFrame
        if not all_features:
            logger.warning("No feature data available")
            return None
        
        combined_df = pd.concat(all_features)
        
        # Save combined dataset
        output_file = os.path.join(self.output_dir, "ml_training_data.csv")
        combined_df.to_csv(output_file)
        
        logger.info(f"Created training dataset with {len(combined_df)} samples")
        return combined_df
    
    def generate_data_report(self, data, regime_data):
        """
        Generate a report of the collected and analyzed data.
        
        Args:
            data: Dict mapping symbols to timeframes to price DataFrames
            regime_data: Dict mapping symbols to timeframes to regime DataFrames
        """
        logger.info("Generating data report")
        
        report = {
            "collection_date": datetime.now().isoformat(),
            "symbols": {},
            "regime_distribution": {},
            "total_samples": 0
        }
        
        # Initialize regime distribution counters
        regime_distribution = {regime.value: 0 for regime in MarketRegimeType}
        
        # Process each symbol
        for symbol in data.keys():
            symbol_report = {
                "timeframes": {}
            }
            
            for tf in data[symbol].keys():
                # Skip if no regime data
                if symbol not in regime_data or tf not in regime_data[symbol]:
                    continue
                
                # Get dataframes
                price_df = data[symbol][tf]
                regime_df = regime_data[symbol][tf]
                
                # Add to report
                symbol_report["timeframes"][tf] = {
                    "price_data_points": len(price_df),
                    "regime_data_points": len(regime_df),
                    "start_date": price_df.index[0].isoformat(),
                    "end_date": price_df.index[-1].isoformat()
                }
                
                # Count regime distribution
                regime_counts = regime_df['regime'].value_counts().to_dict()
                symbol_report["timeframes"][tf]["regime_distribution"] = regime_counts
                
                # Add to global counts
                for regime, count in regime_counts.items():
                    regime_distribution[regime] = regime_distribution.get(regime, 0) + count
                
                report["total_samples"] += len(regime_df)
            
            report["symbols"][symbol] = symbol_report
        
        # Calculate percentages
        if report["total_samples"] > 0:
            report["regime_distribution"] = {
                regime: {
                    "count": count,
                    "percentage": round(count / report["total_samples"] * 100, 2)
                }
                for regime, count in regime_distribution.items()
                if count > 0
            }
        
        # Save report
        report_file = os.path.join(self.output_dir, "data_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\nData Collection Summary:")
        print(f"Total samples collected: {report['total_samples']}")
        print("\nRegime Distribution:")
        for regime, info in report["regime_distribution"].items():
            print(f"{regime}: {info['count']} ({info['percentage']}%)")
        
        logger.info("Data report generated")
        return report

def main():
    parser = argparse.ArgumentParser(description='Regime Data Collector')
    parser.add_argument('--symbols', '-s', required=True, help='Comma-separated list of symbols')
    parser.add_argument('--timeframes', '-t', default="1d,1h,15m", help='Comma-separated list of timeframes')
    parser.add_argument('--output-dir', '-o', default="data/market_regime", help='Output directory')
    parser.add_argument('--window-size', '-w', type=int, default=100, help='Window size for regime detection')
    parser.add_argument('--step-size', '-p', type=int, default=20, help='Step size between detections')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download and use existing files')
    
    args = parser.parse_args()
    
    # Parse arguments
    symbols = args.symbols.split(',')
    timeframes = args.timeframes.split(',')
    
    # Create collector
    collector = HistoricalDataCollector(args.output_dir)
    
    # Download or load data
    if args.skip_download:
        logger.info("Skipping download, loading existing data")
        data = {}
        
        for symbol in symbols:
            data[symbol] = {}
            
            for tf in timeframes:
                data_file = os.path.join(collector.data_dir, f"{symbol}_{tf}.csv")
                
                if os.path.exists(data_file):
                    try:
                        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                        data[symbol][tf] = df
                        logger.info(f"Loaded {len(df)} bars for {symbol} {tf}")
                    except Exception as e:
                        logger.error(f"Error loading {data_file}: {str(e)}")
    else:
        # Download data
        data = collector.download_data(symbols, timeframes)
    
    # Analyze regimes
    regime_data, feature_data = collector.analyze_regimes(
        data, window_size=args.window_size, step_size=args.step_size
    )
    
    # Prepare training dataset
    training_data = collector.prepare_training_dataset(feature_data)
    
    # Generate report
    collector.generate_data_report(data, regime_data)
    
    print("\nData collection and analysis complete!")
    print(f"Results saved to {args.output_dir}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
