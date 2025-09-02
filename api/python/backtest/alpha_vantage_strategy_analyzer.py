import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns

# Import Alpha Vantage data source
from trading_bot.data.sources.alpha_vantage import AlphaVantageDataSource
from trading_bot.data.models import TimeFrame, MarketData
from trading_bot.backtest.backtest_visualizer import BacktestVisualizer

class AlphaVantageStrategyAnalyzer:
    """
    A class for analyzing trading strategies using Alpha Vantage data.
    Integrates with the backtesting system to enhance strategy evaluation
    with additional technical indicators and market data.
    """
    
    def __init__(self, api_key: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy analyzer with Alpha Vantage API key.
        
        Args:
            api_key: Alpha Vantage API key
            logger: Logger instance
        """
        self.api_key = api_key
        self.alpha_vantage = AlphaVantageDataSource(api_key=api_key)
        self.logger = logger or logging.getLogger(__name__)
        
        # Set default style for matplotlib
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        self.logger.info("Initialized Alpha Vantage Strategy Analyzer")
    
    def analyze_backtest_results(self, results_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze backtest results and enhance with Alpha Vantage data.
        
        Args:
            results_dir: Directory containing backtest results
            output_dir: Directory to save analysis results (defaults to results_dir/analysis)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Set up output directory
            if output_dir is None:
                output_dir = os.path.join(results_dir, 'analysis')
            os.makedirs(output_dir, exist_ok=True)
            
            # Initialize backtest visualizer and load results
            visualizer = BacktestVisualizer(results_dir, logger=self.logger, alpha_vantage_api_key=self.api_key)
            results = visualizer.load_backtest_results()
            
            if not results:
                self.logger.error("No backtest results to analyze")
                return {}
                
            summary = results['summary']
            equity_curve = results['equity_curve']
            trades = results.get('trades', pd.DataFrame())
            
            # Analyze traded symbols
            if not trades.empty and 'symbol' in trades.columns:
                symbols = trades['symbol'].unique()
                self.logger.info(f"Analyzing {len(symbols)} symbols from backtest results")
                
                # Get market data for each symbol
                market_data = self._get_market_data_for_symbols(
                    symbols=symbols,
                    start_date=equity_curve.index.min(),
                    end_date=equity_curve.index.max()
                )
                
                # Create enhanced analysis
                analysis_results = self._create_enhanced_analysis(
                    results=results,
                    market_data=market_data,
                    output_dir=output_dir
                )
                
                # Save analysis to JSON
                with open(os.path.join(output_dir, 'analysis_summary.json'), 'w') as f:
                    json.dump(analysis_results, f, indent=4, default=str)
                
                self.logger.info(f"Analysis completed and saved to {output_dir}")
                return analysis_results
            else:
                self.logger.warning("No trades found in backtest results")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error analyzing backtest results: {str(e)}")
            return {}
    
    def _get_market_data_for_symbols(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Get market data with technical indicators for multiple symbols.
        
        Args:
            symbols: List of symbols to get data for
            start_date: Start date for data range
            end_date: End date for data range
            
        Returns:
            Dictionary mapping symbols to DataFrames with market data and indicators
        """
        market_data = {}
        
        for symbol in symbols:
            try:
                # Get raw data from Alpha Vantage
                data = self.alpha_vantage.get_data(
                    symbol=symbol,
                    start_date=start_date - timedelta(days=100),  # Get extra data for indicators
                    end_date=end_date,
                    timeframe=TimeFrame.DAY_1
                )
                
                if not data:
                    self.logger.warning(f"No data available for {symbol}")
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame([d.to_dict() for d in data])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Calculate technical indicators
                df = self._calculate_technical_indicators(df)
                
                # Filter to requested date range
                df = df[df.index >= start_date]
                df = df[df.index <= end_date]
                
                market_data[symbol] = df
                self.logger.info(f"Loaded market data for {symbol} with {len(df)} rows")
                
            except Exception as e:
                self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
        
        return market_data
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for market data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        
        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Trend indicators
        df['price_pct_change'] = df['close'].pct_change()
        df['is_uptrend'] = np.where((df['sma_50'] > df['sma_200']) &
                                   (df['close'] > df['sma_50']), 1, 0)
        
        # Calculate support and resistance levels
        df['local_high'] = df['high'].rolling(window=20).max()
        df['local_low'] = df['low'].rolling(window=20).min()
        
        return df
    
    def _create_enhanced_analysis(self, results: Dict[str, Any], market_data: Dict[str, pd.DataFrame], output_dir: str) -> Dict[str, Any]:
        """
        Create enhanced analysis using market data and technical indicators.
        
        Args:
            results: Dictionary with backtest results
            market_data: Dictionary with market data for symbols
            output_dir: Directory to save analysis
            
        Returns:
            Dictionary with analysis results
        """
        summary = results['summary']
        trades = results.get('trades', pd.DataFrame())
        
        analysis = {
            'strategy_name': summary['strategy_name'],
            'analysis_date': datetime.now().isoformat(),
            'total_symbols_analyzed': len(market_data),
            'market_conditions': {},
            'trade_analysis': {},
            'performance_by_indicator': {}
        }
        
        if trades.empty:
            return analysis
            
        # Analyze market conditions during backtest period
        analysis['market_conditions'] = self._analyze_market_conditions(market_data)
        
        # Analyze trades with technical indicators
        if 'symbol' in trades.columns and 'date' in trades.columns and 'side' in trades.columns:
            enhanced_trades = self._enhance_trades_with_indicators(trades, market_data)
            
            # Save enhanced trades to CSV
            if not enhanced_trades.empty:
                enhanced_trades.to_csv(os.path.join(output_dir, 'enhanced_trades.csv'), index=False)
                
                # Analyze trades by technical indicators
                analysis['trade_analysis'] = self._analyze_trades_by_indicators(enhanced_trades)
                
                # Analyze performance by indicator conditions
                analysis['performance_by_indicator'] = self._analyze_performance_by_indicator(enhanced_trades)
                
                # Create indicator effectiveness plots
                self._create_indicator_effectiveness_plots(enhanced_trades, output_dir)
        
        return analysis
    
    def _analyze_market_conditions(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze market conditions during the backtest period.
        
        Args:
            market_data: Dictionary with market data for symbols
            
        Returns:
            Dictionary with market condition analysis
        """
        if not market_data:
            return {}
            
        # Combine data across symbols for overall market analysis
        all_data = []
        for symbol, df in market_data.items():
            temp_df = df.copy()
            temp_df['symbol'] = symbol
            all_data.append(temp_df)
            
        combined_df = pd.concat(all_data)
        
        market_conditions = {
            'avg_volatility': combined_df['bb_std'].mean(),
            'avg_volume': combined_df['volume'].mean(),
            'trending_days_pct': combined_df['is_uptrend'].mean() * 100,
            'high_rsi_days_pct': (combined_df['rsi'] > 70).mean() * 100,
            'low_rsi_days_pct': (combined_df['rsi'] < 30).mean() * 100,
            'symbols_in_uptrend_pct': 0,
        }
        
        # Calculate percentage of symbols in uptrend
        symbol_trends = {}
        for symbol, df in market_data.items():
            last_idx = df.index.max()
            if last_idx in df.index:
                symbol_trends[symbol] = df.loc[last_idx, 'is_uptrend'] == 1
                
        if symbol_trends:
            market_conditions['symbols_in_uptrend_pct'] = sum(symbol_trends.values()) / len(symbol_trends) * 100
        
        return market_conditions
    
    def _enhance_trades_with_indicators(self, trades: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Enhance trade data with technical indicators at time of trade.
        
        Args:
            trades: DataFrame with trade data
            market_data: Dictionary with market data and indicators
            
        Returns:
            DataFrame with enhanced trade data
        """
        enhanced_trades = []
        
        for _, trade in trades.iterrows():
            symbol = trade['symbol']
            trade_date = pd.to_datetime(trade['date'])
            trade_side = trade['side']
            
            if symbol not in market_data:
                continue
                
            symbol_data = market_data[symbol]
            
            # Find closest data point to trade date (same day or before)
            closest_idx = symbol_data.index.asof(trade_date)
            if closest_idx is not pd.NaT:
                indicators = symbol_data.loc[closest_idx].to_dict()
                
                # Combine trade data with indicators
                trade_dict = trade.to_dict()
                
                # Remove some duplicated columns
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in trade_dict and col in indicators:
                        indicators.pop(col)
                
                enhanced_trade = {**trade_dict, **indicators}
                enhanced_trades.append(enhanced_trade)
        
        if enhanced_trades:
            return pd.DataFrame(enhanced_trades)
        else:
            return pd.DataFrame()
    
    def _analyze_trades_by_indicators(self, enhanced_trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trades grouped by technical indicator conditions.
        
        Args:
            enhanced_trades: DataFrame with enhanced trade data
            
        Returns:
            Dictionary with analysis by indicator
        """
        if enhanced_trades.empty:
            return {}
            
        analysis = {}
        
        # Create profitable flag
        enhanced_trades['profitable'] = False
        if 'pnl' in enhanced_trades.columns:
            enhanced_trades['profitable'] = enhanced_trades['pnl'] > 0
        
        # Analyze by trend
        analysis['by_trend'] = {
            'uptrend': {
                'count': len(enhanced_trades[enhanced_trades['is_uptrend'] == 1]),
                'win_rate': enhanced_trades[enhanced_trades['is_uptrend'] == 1]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['is_uptrend'] == 1]) > 0 else 0
            },
            'downtrend': {
                'count': len(enhanced_trades[enhanced_trades['is_uptrend'] == 0]),
                'win_rate': enhanced_trades[enhanced_trades['is_uptrend'] == 0]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['is_uptrend'] == 0]) > 0 else 0
            }
        }
        
        # Analyze by RSI
        analysis['by_rsi'] = {
            'overbought': {
                'count': len(enhanced_trades[enhanced_trades['rsi'] > 70]),
                'win_rate': enhanced_trades[enhanced_trades['rsi'] > 70]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['rsi'] > 70]) > 0 else 0
            },
            'neutral': {
                'count': len(enhanced_trades[(enhanced_trades['rsi'] >= 30) & (enhanced_trades['rsi'] <= 70)]),
                'win_rate': enhanced_trades[(enhanced_trades['rsi'] >= 30) & (enhanced_trades['rsi'] <= 70)]['profitable'].mean() * 100
                if len(enhanced_trades[(enhanced_trades['rsi'] >= 30) & (enhanced_trades['rsi'] <= 70)]) > 0 else 0
            },
            'oversold': {
                'count': len(enhanced_trades[enhanced_trades['rsi'] < 30]),
                'win_rate': enhanced_trades[enhanced_trades['rsi'] < 30]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['rsi'] < 30]) > 0 else 0
            }
        }
        
        # Analyze by Bollinger Bands
        analysis['by_bollinger'] = {
            'above_upper': {
                'count': len(enhanced_trades[enhanced_trades['close'] > enhanced_trades['bb_upper']]),
                'win_rate': enhanced_trades[enhanced_trades['close'] > enhanced_trades['bb_upper']]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['close'] > enhanced_trades['bb_upper']]) > 0 else 0
            },
            'within_bands': {
                'count': len(enhanced_trades[(enhanced_trades['close'] <= enhanced_trades['bb_upper']) & 
                                           (enhanced_trades['close'] >= enhanced_trades['bb_lower'])]),
                'win_rate': enhanced_trades[(enhanced_trades['close'] <= enhanced_trades['bb_upper']) & 
                                          (enhanced_trades['close'] >= enhanced_trades['bb_lower'])]['profitable'].mean() * 100
                if len(enhanced_trades[(enhanced_trades['close'] <= enhanced_trades['bb_upper']) & 
                                     (enhanced_trades['close'] >= enhanced_trades['bb_lower'])]) > 0 else 0
            },
            'below_lower': {
                'count': len(enhanced_trades[enhanced_trades['close'] < enhanced_trades['bb_lower']]),
                'win_rate': enhanced_trades[enhanced_trades['close'] < enhanced_trades['bb_lower']]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['close'] < enhanced_trades['bb_lower']]) > 0 else 0
            }
        }
        
        # Analyze by MACD
        analysis['by_macd'] = {
            'positive_histogram': {
                'count': len(enhanced_trades[enhanced_trades['macd_hist'] > 0]),
                'win_rate': enhanced_trades[enhanced_trades['macd_hist'] > 0]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['macd_hist'] > 0]) > 0 else 0
            },
            'negative_histogram': {
                'count': len(enhanced_trades[enhanced_trades['macd_hist'] <= 0]),
                'win_rate': enhanced_trades[enhanced_trades['macd_hist'] <= 0]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['macd_hist'] <= 0]) > 0 else 0
            }
        }
        
        # Analyze by Volume
        analysis['by_volume'] = {
            'above_average': {
                'count': len(enhanced_trades[enhanced_trades['volume_ratio'] > 1]),
                'win_rate': enhanced_trades[enhanced_trades['volume_ratio'] > 1]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['volume_ratio'] > 1]) > 0 else 0
            },
            'below_average': {
                'count': len(enhanced_trades[enhanced_trades['volume_ratio'] <= 1]),
                'win_rate': enhanced_trades[enhanced_trades['volume_ratio'] <= 1]['profitable'].mean() * 100
                if len(enhanced_trades[enhanced_trades['volume_ratio'] <= 1]) > 0 else 0
            }
        }
        
        return analysis
    
    def _analyze_performance_by_indicator(self, enhanced_trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze performance metrics grouped by indicator combinations.
        
        Args:
            enhanced_trades: DataFrame with enhanced trade data
            
        Returns:
            Dictionary with performance by indicator combinations
        """
        if enhanced_trades.empty or 'pnl' not in enhanced_trades.columns:
            return {}
            
        analysis = {}
        
        # Define indicator conditions to test
        conditions = {
            'rsi_oversold_uptrend': (enhanced_trades['rsi'] < 30) & (enhanced_trades['is_uptrend'] == 1),
            'rsi_overbought_downtrend': (enhanced_trades['rsi'] > 70) & (enhanced_trades['is_uptrend'] == 0),
            'price_below_bb_lower': enhanced_trades['close'] < enhanced_trades['bb_lower'],
            'price_above_bb_upper': enhanced_trades['close'] > enhanced_trades['bb_upper'],
            'macd_positive_cross': (enhanced_trades['macd'] > enhanced_trades['macd_signal']) & (enhanced_trades['macd_hist'] > 0),
            'macd_negative_cross': (enhanced_trades['macd'] < enhanced_trades['macd_signal']) & (enhanced_trades['macd_hist'] < 0),
            'high_volume_breakout': (enhanced_trades['volume_ratio'] > 1.5) & (enhanced_trades['price_pct_change'] > 0.01),
            'stoch_oversold': enhanced_trades['stoch_k'] < 20,
            'stoch_overbought': enhanced_trades['stoch_k'] > 80,
        }
        
        # Calculate performance metrics for each condition
        for name, condition in conditions.items():
            filtered_trades = enhanced_trades[condition]
            if len(filtered_trades) > 0:
                analysis[name] = {
                    'count': len(filtered_trades),
                    'avg_pnl': filtered_trades['pnl'].mean(),
                    'win_rate': (filtered_trades['pnl'] > 0).mean() * 100,
                    'profit_factor': abs(filtered_trades[filtered_trades['pnl'] > 0]['pnl'].sum()) / 
                                   abs(filtered_trades[filtered_trades['pnl'] < 0]['pnl'].sum())
                                   if abs(filtered_trades[filtered_trades['pnl'] < 0]['pnl'].sum()) > 0 else float('inf')
                }
        
        return analysis
    
    def _create_indicator_effectiveness_plots(self, enhanced_trades: pd.DataFrame, output_dir: str) -> None:
        """
        Create plots showing effectiveness of various technical indicators.
        
        Args:
            enhanced_trades: DataFrame with enhanced trade data
            output_dir: Directory to save plots
        """
        if enhanced_trades.empty or 'pnl' not in enhanced_trades.columns:
            return
            
        # Create win rate by RSI plot
        plt.figure(figsize=(10, 6))
        
        # Define RSI bins
        rsi_bins = [0, 20, 30, 40, 50, 60, 70, 80, 100]
        rsi_labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-100']
        
        enhanced_trades['rsi_bin'] = pd.cut(enhanced_trades['rsi'], bins=rsi_bins, labels=rsi_labels)
        win_rate_by_rsi = enhanced_trades.groupby('rsi_bin')['profitable'].mean() * 100
        trades_count_by_rsi = enhanced_trades.groupby('rsi_bin').size()
        
        ax = win_rate_by_rsi.plot(kind='bar', color='skyblue')
        
        # Add trade count as text on bars
        for i, v in enumerate(win_rate_by_rsi):
            if not np.isnan(v):
                count = trades_count_by_rsi.iloc[i]
                ax.text(i, v + 2, f"n={count}", ha='center')
        
        plt.title('Win Rate by RSI Range')
        plt.xlabel('RSI Range')
        plt.ylabel('Win Rate (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'win_rate_by_rsi.png'), dpi=300)
        plt.close()
        
        # Create win rate by BB position plot
        plt.figure(figsize=(10, 6))
        
        # Define BB position categories
        enhanced_trades['bb_position'] = 'Within Bands'
        enhanced_trades.loc[enhanced_trades['close'] > enhanced_trades['bb_upper'], 'bb_position'] = 'Above Upper'
        enhanced_trades.loc[enhanced_trades['close'] < enhanced_trades['bb_lower'], 'bb_position'] = 'Below Lower'
        
        win_rate_by_bb = enhanced_trades.groupby('bb_position')['profitable'].mean() * 100
        trades_count_by_bb = enhanced_trades.groupby('bb_position').size()
        
        ax = win_rate_by_bb.plot(kind='bar', color='lightgreen')
        
        # Add trade count as text on bars
        for i, v in enumerate(win_rate_by_bb):
            if not np.isnan(v):
                count = trades_count_by_bb.iloc[i]
                ax.text(i, v + 2, f"n={count}", ha='center')
        
        plt.title('Win Rate by Bollinger Band Position')
        plt.xlabel('Position Relative to Bollinger Bands')
        plt.ylabel('Win Rate (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'win_rate_by_bb.png'), dpi=300)
        plt.close()
        
        # Create win rate by volume ratio plot
        plt.figure(figsize=(10, 6))
        
        # Define volume ratio bins
        vol_bins = [0, 0.5, 0.8, 1.0, 1.5, 2.0, 10.0]
        vol_labels = ['0-0.5', '0.5-0.8', '0.8-1.0', '1.0-1.5', '1.5-2.0', '2.0+']
        
        enhanced_trades['volume_ratio_bin'] = pd.cut(enhanced_trades['volume_ratio'], bins=vol_bins, labels=vol_labels)
        win_rate_by_vol = enhanced_trades.groupby('volume_ratio_bin')['profitable'].mean() * 100
        trades_count_by_vol = enhanced_trades.groupby('volume_ratio_bin').size()
        
        ax = win_rate_by_vol.plot(kind='bar', color='salmon')
        
        # Add trade count as text on bars
        for i, v in enumerate(win_rate_by_vol):
            if not np.isnan(v):
                count = trades_count_by_vol.iloc[i]
                ax.text(i, v + 2, f"n={count}", ha='center')
        
        plt.title('Win Rate by Volume Ratio')
        plt.xlabel('Volume Ratio (vs 20-day Average)')
        plt.ylabel('Win Rate (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'win_rate_by_volume.png'), dpi=300)
        plt.close()
        
        # Create scatter plot of RSI vs PnL
        plt.figure(figsize=(10, 6))
        plt.scatter(enhanced_trades['rsi'], enhanced_trades['pnl'], alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('PnL vs RSI at Entry')
        plt.xlabel('RSI')
        plt.ylabel('Profit/Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pnl_vs_rsi.png'), dpi=300)
        plt.close()
        
        # Create combo indicator effectiveness plot
        plt.figure(figsize=(12, 8))
        
        # Define combinations to test
        combo_results = []
        
        # RSI + Trend
        rsi_uptrend = enhanced_trades[(enhanced_trades['rsi'] < 30) & (enhanced_trades['is_uptrend'] == 1)]
        if len(rsi_uptrend) > 0:
            combo_results.append({
                'combo': 'RSI<30 + Uptrend',
                'win_rate': rsi_uptrend['profitable'].mean() * 100,
                'count': len(rsi_uptrend)
            })
            
        rsi_downtrend = enhanced_trades[(enhanced_trades['rsi'] > 70) & (enhanced_trades['is_uptrend'] == 0)]
        if len(rsi_downtrend) > 0:
            combo_results.append({
                'combo': 'RSI>70 + Downtrend',
                'win_rate': rsi_downtrend['profitable'].mean() * 100,
                'count': len(rsi_downtrend)
            })
        
        # BB + Volume
        bb_lower_vol = enhanced_trades[(enhanced_trades['close'] < enhanced_trades['bb_lower']) & 
                                      (enhanced_trades['volume_ratio'] > 1.5)]
        if len(bb_lower_vol) > 0:
            combo_results.append({
                'combo': 'Below BB + High Vol',
                'win_rate': bb_lower_vol['profitable'].mean() * 100,
                'count': len(bb_lower_vol)
            })
            
        bb_upper_vol = enhanced_trades[(enhanced_trades['close'] > enhanced_trades['bb_upper']) & 
                                      (enhanced_trades['volume_ratio'] > 1.5)]
        if len(bb_upper_vol) > 0:
            combo_results.append({
                'combo': 'Above BB + High Vol',
                'win_rate': bb_upper_vol['profitable'].mean() * 100,
                'count': len(bb_upper_vol)
            })
        
        # MACD + Stoch
        macd_stoch_over = enhanced_trades[(enhanced_trades['macd_hist'] > 0) & (enhanced_trades['stoch_k'] > 80)]
        if len(macd_stoch_over) > 0:
            combo_results.append({
                'combo': 'MACD+ + Stoch>80',
                'win_rate': macd_stoch_over['profitable'].mean() * 100,
                'count': len(macd_stoch_over)
            })
            
        macd_stoch_under = enhanced_trades[(enhanced_trades['macd_hist'] < 0) & (enhanced_trades['stoch_k'] < 20)]
        if len(macd_stoch_under) > 0:
            combo_results.append({
                'combo': 'MACD- + Stoch<20',
                'win_rate': macd_stoch_under['profitable'].mean() * 100,
                'count': len(macd_stoch_under)
            })
        
        if combo_results:
            combo_df = pd.DataFrame(combo_results)
            combo_df = combo_df.sort_values('win_rate', ascending=False)
            
            # Plot combo results
            bars = plt.bar(combo_df['combo'], combo_df['win_rate'], color='purple')
            
            # Add count labels
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 2,
                    f"n={combo_df.iloc[i]['count']}",
                    ha='center'
                )
            
            plt.title('Win Rate by Technical Indicator Combinations')
            plt.xlabel('Indicator Combination')
            plt.ylabel('Win Rate (%)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'indicator_combinations.png'), dpi=300)
            plt.close()
            
        # Create PnL distribution by indicator
        if 'side' in enhanced_trades.columns:
            for side in ['buy', 'sell']:
                side_trades = enhanced_trades[enhanced_trades['side'] == side]
                if not side_trades.empty:
                    plt.figure(figsize=(10, 6))
                    
                    # Categorize trades
                    side_trades['condition'] = 'No Condition'
                    
                    if side == 'buy':
                        # Conditions appropriate for buy trades
                        side_trades.loc[(side_trades['rsi'] < 30) & (side_trades['is_uptrend'] == 1), 'condition'] = 'RSI<30 + Uptrend'
                        side_trades.loc[(side_trades['close'] < side_trades['bb_lower']), 'condition'] = 'Below BB Lower'
                        side_trades.loc[(side_trades['macd_hist'] > 0) & (side_trades['macd'] > 0), 'condition'] = 'MACD+ Crossover'
                    else:
                        # Conditions appropriate for sell trades
                        side_trades.loc[(side_trades['rsi'] > 70) & (side_trades['is_uptrend'] == 0), 'condition'] = 'RSI>70 + Downtrend'
                        side_trades.loc[(side_trades['close'] > side_trades['bb_upper']), 'condition'] = 'Above BB Upper'
                        side_trades.loc[(side_trades['macd_hist'] < 0) & (side_trades['macd'] < 0), 'condition'] = 'MACD- Crossover'
                    
                    # Plot PnL by condition
                    sns.boxplot(x='condition', y='pnl', data=side_trades)
                    plt.title(f'PnL Distribution by Indicator Condition ({side.capitalize()} Trades)')
                    plt.xlabel('Condition')
                    plt.ylabel('Profit/Loss')
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.grid(True, axis='y')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'pnl_distribution_{side}.png'), dpi=300)
                    plt.close()

    def schedule_strategy_monitoring(self, symbols: List[str], time_interval: int = 60, output_file: str = None) -> None:
        """
        Schedule regular monitoring of market conditions for a set of symbols.
        
        Args:
            symbols: List of symbols to monitor
            time_interval: Time interval in minutes between updates
            output_file: File to save the monitoring results (default: None)
        """
        try:
            import schedule
            import time
            import threading
            
            # Function to run analysis
            def run_analysis():
                try:
                    now = datetime.now()
                    end_date = now
                    start_date = end_date - timedelta(days=30)  # Last 30 days
                    
                    self.logger.info(f"Running scheduled analysis for {len(symbols)} symbols")
                    
                    # Get market data
                    market_data = self._get_market_data_for_symbols(
                        symbols=symbols,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    # Analyze market conditions
                    market_conditions = self._analyze_market_conditions(market_data)
                    
                    # Generate signals
                    signals = self._generate_trading_signals(market_data)
                    
                    # Combine results
                    results = {
                        'timestamp': now.isoformat(),
                        'market_conditions': market_conditions,
                        'signals': signals
                    }
                    
                    # Save results if specified
                    if output_file:
                        if os.path.exists(output_file):
                            with open(output_file, 'r') as f:
                                existing_data = json.load(f)
                        else:
                            existing_data = {'history': []}
                            
                        existing_data['current'] = results
                        existing_data['history'].append(results)
                        
                        # Keep only last 100 entries
                        if len(existing_data['history']) > 100:
                            existing_data['history'] = existing_data['history'][-100:]
                            
                        with open(output_file, 'w') as f:
                            json.dump(existing_data, f, indent=4, default=str)
                    
                    self.logger.info(f"Scheduled analysis completed at {now}")
                    
                except Exception as e:
                    self.logger.error(f"Error in scheduled analysis: {str(e)}")
            
            # Schedule the job
            schedule.every(time_interval).minutes.do(run_analysis)
            
            # Run initial analysis
            run_analysis()
            
            # Create a thread to run the scheduler
            def run_scheduler():
                while True:
                    schedule.run_pending()
                    time.sleep(1)
            
            scheduler_thread = threading.Thread(target=run_scheduler)
            scheduler_thread.daemon = True
            scheduler_thread.start()
            
            self.logger.info(f"Scheduled monitoring every {time_interval} minutes for {len(symbols)} symbols")
            
        except ImportError:
            self.logger.error("Schedule package not installed. Cannot schedule monitoring.")
    
    def _generate_trading_signals(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            market_data: Dictionary with market data for symbols
            
        Returns:
            Dictionary with trading signals by symbol
        """
        signals = {}
        
        for symbol, df in market_data.items():
            symbol_signals = []
            if len(df) < 2:
                continue
                
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # RSI signals
            if latest['rsi'] < 30 and previous['rsi'] >= 30:
                symbol_signals.append({
                    'indicator': 'RSI',
                    'signal': 'Oversold',
                    'strength': 'Strong' if latest['rsi'] < 20 else 'Medium',
                    'value': float(latest['rsi'])
                })
            elif latest['rsi'] > 70 and previous['rsi'] <= 70:
                symbol_signals.append({
                    'indicator': 'RSI',
                    'signal': 'Overbought',
                    'strength': 'Strong' if latest['rsi'] > 80 else 'Medium',
                    'value': float(latest['rsi'])
                })
            
            # MACD signals
            if latest['macd_hist'] > 0 and previous['macd_hist'] <= 0:
                symbol_signals.append({
                    'indicator': 'MACD',
                    'signal': 'Bullish Crossover',
                    'strength': 'Strong' if latest['macd'] > 0 else 'Medium',
                    'value': float(latest['macd'])
                })
            elif latest['macd_hist'] < 0 and previous['macd_hist'] >= 0:
                symbol_signals.append({
                    'indicator': 'MACD',
                    'signal': 'Bearish Crossover',
                    'strength': 'Strong' if latest['macd'] < 0 else 'Medium',
                    'value': float(latest['macd'])
                })
            
            # Bollinger Band signals
            if latest['close'] < latest['bb_lower']:
                symbol_signals.append({
                    'indicator': 'Bollinger Bands',
                    'signal': 'Below Lower Band',
                    'strength': 'Strong',
                    'value': float(latest['close'] / latest['bb_lower'])
                })
            elif latest['close'] > latest['bb_upper']:
                symbol_signals.append({
                    'indicator': 'Bollinger Bands',
                    'signal': 'Above Upper Band',
                    'strength': 'Strong',
                    'value': float(latest['close'] / latest['bb_upper'])
                })
            
            # Moving Average signals
            if latest['sma_50'] > latest['sma_200'] and previous['sma_50'] <= previous['sma_200']:
                symbol_signals.append({
                    'indicator': 'Moving Averages',
                    'signal': 'Golden Cross',
                    'strength': 'Strong',
                    'value': None
                })
            elif latest['sma_50'] < latest['sma_200'] and previous['sma_50'] >= previous['sma_200']:
                symbol_signals.append({
                    'indicator': 'Moving Averages',
                    'signal': 'Death Cross',
                    'strength': 'Strong',
                    'value': None
                })
            
            if symbol_signals:
                signals[symbol] = symbol_signals
        
        return signals 