#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PatternLearner - Analyzes backtest data to identify profitable patterns and trading behaviors.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class PatternLearner:
    """
    Analyzes backtest data to identify profitable patterns and trading behaviors.
    """
    
    def __init__(self, data_manager=None, data_path=None):
        """
        Initialize the PatternLearner.
        
        Args:
            data_manager: DataManager instance for fetching data
            data_path: Path to load backtest history from if no data_manager is provided
        """
        self.data_manager = data_manager
        self.data_path = data_path
        self.data = None
        self.trade_df = None
        self.signal_df = None
        self.portfolio_df = None
        self.results_dir = "data/pattern_analysis"
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("Initialized PatternLearner")
    
    def load_data(self):
        """Load data from data_manager or file"""
        if self.data_manager:
            logger.info("Loading data from DataManager")
            self.data = self.data_manager.load()
            self.trade_df = self.data_manager.get_data_as_dataframe('trade')
            self.signal_df = self.data_manager.get_data_as_dataframe('signal')
            self.portfolio_df = self.data_manager.get_data_as_dataframe('portfolio_snapshot')
        elif self.data_path and os.path.exists(self.data_path):
            logger.info(f"Loading data from file: {self.data_path}")
            with open(self.data_path, 'r') as f:
                self.data = json.load(f)
            
            # Convert to DataFrames
            trade_data = [entry['data'] for entry in self.data if entry['type'] == 'trade']
            signal_data = [entry['data'] for entry in self.data if entry['type'] == 'signal']
            portfolio_data = [entry['data'] for entry in self.data if entry['type'] == 'portfolio_snapshot']
            
            self.trade_df = pd.DataFrame(trade_data) if trade_data else pd.DataFrame()
            self.signal_df = pd.DataFrame(signal_data) if signal_data else pd.DataFrame()
            self.portfolio_df = pd.DataFrame(portfolio_data) if portfolio_data else pd.DataFrame()
        else:
            logger.error("No data source available")
            return False
        
        # Convert timestamp columns to datetime
        for df in [self.trade_df, self.signal_df, self.portfolio_df]:
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Loaded data: {len(self.trade_df)} trades, {len(self.signal_df)} signals, {len(self.portfolio_df)} portfolio snapshots")
        return True
    
    def analyze(self, save_results=True):
        """
        Analyze the backtest data to identify patterns.
        
        Args:
            save_results: Whether to save analysis results
            
        Returns:
            Dictionary with analysis results
        """
        if self.data is None and not self.load_data():
            logger.error("No data available for analysis")
            return {"error": "No data available"}
        
        logger.info("Starting data analysis")
        
        results = {}
        
        # 1. Analyze win rates by strategy
        win_rates = self._analyze_win_rates()
        results["win_rates"] = win_rates
        
        # 2. Analyze performance by market regime
        regime_performance = self._analyze_regime_performance()
        results["regime_performance"] = regime_performance
        
        # 3. Analyze time patterns
        time_patterns = self._analyze_time_patterns()
        results["time_patterns"] = time_patterns
        
        # 4. Cluster trades by features
        trade_clusters = self._cluster_trades()
        results["trade_clusters"] = trade_clusters
        
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.results_dir, f"analysis_results_{timestamp}.json")
            
            # Convert numpy values to Python types for JSON serialization
            json_results = self._convert_to_json_serializable(results)
            
            with open(save_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Saved analysis results to {save_path}")
        
        return results
    
    def _convert_to_json_serializable(self, obj):
        """Convert numpy values to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_json_serializable(obj.tolist())
        else:
            return obj
    
    def _analyze_win_rates(self):
        """
        Analyze win rates by strategy, symbol, and signal type.
        
        Returns:
            Dictionary with win rate analysis
        """
        if self.trade_df.empty:
            return {"error": "No trade data available"}
        
        # Create a copy with winning trade flag
        trades = self.trade_df.copy()
        if 'pnl' in trades.columns:
            trades['is_win'] = trades['pnl'] > 0
        else:
            logger.warning("No PnL data available for win rate analysis")
            return {"error": "No PnL data available"}
        
        results = {}
        
        # 1. Overall win rate
        overall_win_rate = trades['is_win'].mean()
        results["overall"] = overall_win_rate
        
        # 2. Win rate by strategy
        if 'strategy' in trades.columns:
            strategy_wins = trades.groupby('strategy')['is_win'].agg(['mean', 'count']).reset_index()
            strategy_wins.columns = ['strategy', 'win_rate', 'trade_count']
            results["by_strategy"] = strategy_wins.to_dict(orient='records')
        
        # 3. Win rate by symbol
        if 'symbol' in trades.columns:
            symbol_wins = trades.groupby('symbol')['is_win'].agg(['mean', 'count']).reset_index()
            symbol_wins.columns = ['symbol', 'win_rate', 'trade_count']
            symbol_wins = symbol_wins.sort_values('trade_count', ascending=False)
            results["by_symbol"] = symbol_wins.to_dict(orient='records')
        
        # 4. Win rate by trade type
        if 'type' in trades.columns:
            type_wins = trades.groupby('type')['is_win'].agg(['mean', 'count']).reset_index()
            type_wins.columns = ['type', 'win_rate', 'trade_count']
            results["by_type"] = type_wins.to_dict(orient='records')
        
        # 5. Win rate by market regime
        if 'market_context' in trades.columns:
            # This requires parsing the market_context JSON
            try:
                # Try to extract regime from market_context
                if isinstance(trades['market_context'].iloc[0], str):
                    trades['regime'] = trades['market_context'].apply(
                        lambda x: json.loads(x).get('regime', 'unknown') if isinstance(x, str) else 'unknown'
                    )
                else:
                    trades['regime'] = trades['market_context'].apply(
                        lambda x: x.get('regime', 'unknown') if isinstance(x, dict) else 'unknown'
                    )
                
                regime_wins = trades.groupby('regime')['is_win'].agg(['mean', 'count']).reset_index()
                regime_wins.columns = ['regime', 'win_rate', 'trade_count']
                results["by_regime"] = regime_wins.to_dict(orient='records')
            except Exception as e:
                logger.error(f"Error analyzing win rates by regime: {e}")
        
        return results
    
    def _analyze_regime_performance(self):
        """
        Analyze performance by market regime.
        
        Returns:
            Dictionary with regime performance analysis
        """
        if self.portfolio_df.empty or self.signal_df.empty:
            return {"error": "Insufficient data for regime analysis"}
        
        results = {}
        
        # Extract market regimes from signals
        if 'market_context' in self.signal_df.columns:
            try:
                # Extract regime from market_context
                if isinstance(self.signal_df['market_context'].iloc[0], str):
                    self.signal_df['regime'] = self.signal_df['market_context'].apply(
                        lambda x: json.loads(x).get('regime', 'unknown') if isinstance(x, str) else 'unknown'
                    )
                else:
                    self.signal_df['regime'] = self.signal_df['market_context'].apply(
                        lambda x: x.get('regime', 'unknown') if isinstance(x, dict) else 'unknown'
                    )
                
                # Group signals by regime and strategy
                regime_signals = self.signal_df.groupby(['regime', 'strategy']).size().reset_index(name='signal_count')
                results["signal_distribution"] = regime_signals.to_dict(orient='records')
                
                # Get unique regimes
                regimes = self.signal_df['regime'].unique()
                
                # Analyze portfolio performance by regime
                if 'daily_return' in self.portfolio_df.columns:
                    regime_returns = []
                    
                    for regime in regimes:
                        # Get signals in this regime
                        regime_dates = self.signal_df[self.signal_df['regime'] == regime]['timestamp']
                        
                        if len(regime_dates) > 0:
                            # Find portfolio snapshots in this date range
                            min_date = min(regime_dates)
                            max_date = max(regime_dates)
                            
                            regime_portfolio = self.portfolio_df[
                                (self.portfolio_df['timestamp'] >= min_date) & 
                                (self.portfolio_df['timestamp'] <= max_date)
                            ]
                            
                            if len(regime_portfolio) > 0:
                                avg_return = regime_portfolio['daily_return'].mean()
                                volatility = regime_portfolio['daily_return'].std()
                                sharpe = avg_return / volatility if volatility > 0 else 0
                                
                                regime_returns.append({
                                    'regime': regime,
                                    'avg_daily_return': avg_return,
                                    'volatility': volatility,
                                    'sharpe_ratio': sharpe,
                                    'days': len(regime_portfolio)
                                })
                    
                    results["regime_returns"] = regime_returns
            except Exception as e:
                logger.error(f"Error analyzing regime performance: {e}")
                
        return results
    
    def _analyze_time_patterns(self):
        """
        Analyze performance by time patterns (hour of day, day of week, etc.).
        
        Returns:
            Dictionary with time pattern analysis
        """
        if self.trade_df.empty:
            return {"error": "No trade data available"}
        
        results = {}
        
        # Create a copy with datetime parsed
        trades = self.trade_df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in trades.columns:
            if not pd.api.types.is_datetime64_any_dtype(trades['timestamp']):
                trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            
            # Extract time components
            trades['hour'] = trades['timestamp'].dt.hour
            trades['day_of_week'] = trades['timestamp'].dt.day_name()
            trades['month'] = trades['timestamp'].dt.month_name()
            
            # Mark winning trades
            if 'pnl' in trades.columns:
                trades['is_win'] = trades['pnl'] > 0
                
                # 1. Win rate by hour of day
                hour_wins = trades.groupby('hour')['is_win'].agg(['mean', 'count']).reset_index()
                hour_wins.columns = ['hour', 'win_rate', 'trade_count']
                results["by_hour"] = hour_wins.to_dict(orient='records')
                
                # 2. Win rate by day of week
                dow_wins = trades.groupby('day_of_week')['is_win'].agg(['mean', 'count']).reset_index()
                dow_wins.columns = ['day_of_week', 'win_rate', 'trade_count']
                results["by_day_of_week"] = dow_wins.to_dict(orient='records')
                
                # 3. Win rate by month
                month_wins = trades.groupby('month')['is_win'].agg(['mean', 'count']).reset_index()
                month_wins.columns = ['month', 'win_rate', 'trade_count']
                results["by_month"] = month_wins.to_dict(orient='records')
                
                # 4. Average PnL by hour
                hour_pnl = trades.groupby('hour')['pnl'].mean().reset_index()
                hour_pnl.columns = ['hour', 'avg_pnl']
                results["avg_pnl_by_hour"] = hour_pnl.to_dict(orient='records')
        else:
            logger.warning("No timestamp data available for time pattern analysis")
            return {"error": "No timestamp data available"}
        
        return results
    
    def _cluster_trades(self, n_clusters=3):
        """
        Cluster trades by features to identify common patterns.
        
        Args:
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with clustering results
        """
        if self.trade_df.empty:
            return {"error": "No trade data available"}
        
        # Create a copy for clustering
        trades = self.trade_df.copy()
        
        # Only keep numeric columns for clustering
        numeric_cols = trades.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['id']]
        
        if len(numeric_cols) < 2:
            logger.warning("Insufficient numeric features for clustering")
            return {"error": "Insufficient numeric features"}
        
        try:
            # Extract features for clustering
            features = trades[numeric_cols].copy()
            
            # Handle missing values
            features = features.fillna(0)
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            trades['cluster'] = kmeans.fit_predict(scaled_features)
            
            # Analyze clusters
            cluster_stats = []
            
            for cluster in range(n_clusters):
                cluster_trades = trades[trades['cluster'] == cluster]
                
                # Calculate cluster statistics
                stats = {
                    'cluster': cluster,
                    'trade_count': len(cluster_trades),
                    'avg_pnl': cluster_trades['pnl'].mean() if 'pnl' in cluster_trades else None,
                    'win_rate': cluster_trades['is_win'].mean() if 'is_win' in cluster_trades else None
                }
                
                # Add feature means
                for col in numeric_cols:
                    stats[f'avg_{col}'] = cluster_trades[col].mean()
                
                cluster_stats.append(stats)
            
            return {
                "cluster_stats": cluster_stats,
                "feature_importance": dict(zip(numeric_cols, kmeans.cluster_centers_.std(axis=0)))
            }
            
        except Exception as e:
            logger.error(f"Error clustering trades: {e}")
            return {"error": f"Clustering error: {str(e)}"}
    
    def plot_analysis_results(self, results, save_path=None):
        """
        Plot analysis results.
        
        Args:
            results: Dictionary with analysis results
            save_path: Path to save the plots
            
        Returns:
            Boolean indicating success
        """
        if not results:
            return False
        
        try:
            # 1. Win rates by strategy
            if "win_rates" in results and "by_strategy" in results["win_rates"]:
                plt.figure(figsize=(10, 6))
                strategy_data = pd.DataFrame(results["win_rates"]["by_strategy"])
                strategy_data = strategy_data.sort_values('win_rate', ascending=False)
                
                plt.bar(strategy_data['strategy'], strategy_data['win_rate'] * 100)
                plt.xlabel('Strategy')
                plt.ylabel('Win Rate (%)')
                plt.title('Win Rate by Strategy')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if save_path:
                    plot_path = os.path.join(save_path, "win_rate_by_strategy.png")
                    plt.savefig(plot_path)
                    logger.info(f"Saved plot to {plot_path}")
                else:
                    plt.show()
                
                plt.close()
            
            # 2. Win rates by time of day
            if "time_patterns" in results and "by_hour" in results["time_patterns"]:
                plt.figure(figsize=(12, 6))
                hour_data = pd.DataFrame(results["time_patterns"]["by_hour"])
                
                plt.bar(hour_data['hour'], hour_data['win_rate'] * 100)
                plt.xlabel('Hour of Day')
                plt.ylabel('Win Rate (%)')
                plt.title('Win Rate by Hour of Day')
                plt.xticks(range(24))
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                if save_path:
                    plot_path = os.path.join(save_path, "win_rate_by_hour.png")
                    plt.savefig(plot_path)
                    logger.info(f"Saved plot to {plot_path}")
                else:
                    plt.show()
                
                plt.close()
            
            # 3. Regime performance comparison
            if "regime_performance" in results and "regime_returns" in results["regime_performance"]:
                plt.figure(figsize=(10, 6))
                regime_data = pd.DataFrame(results["regime_performance"]["regime_returns"])
                
                x = regime_data['regime']
                y1 = regime_data['avg_daily_return'] * 100  # Convert to percentage
                y2 = regime_data['sharpe_ratio']
                
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                color = 'tab:blue'
                ax1.set_xlabel('Market Regime')
                ax1.set_ylabel('Avg Daily Return (%)', color=color)
                ax1.bar(x, y1, color=color, alpha=0.7)
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.set_xticklabels(x, rotation=45)
                
                ax2 = ax1.twinx()
                color = 'tab:red'
                ax2.set_ylabel('Sharpe Ratio', color=color)
                ax2.plot(x, y2, 'o-', color=color, linewidth=2)
                ax2.tick_params(axis='y', labelcolor=color)
                
                plt.title('Performance by Market Regime')
                plt.tight_layout()
                
                if save_path:
                    plot_path = os.path.join(save_path, "regime_performance.png")
                    plt.savefig(plot_path)
                    logger.info(f"Saved plot to {plot_path}")
                else:
                    plt.show()
                
                plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting analysis results: {e}")
            return False
    
    def get_recommendations(self, results=None):
        """
        Generate trading recommendations based on analysis.
        
        Args:
            results: Analysis results (will analyze data if not provided)
            
        Returns:
            List of recommendations
        """
        if results is None:
            results = self.analyze(save_results=False)
        
        recommendations = []
        
        try:
            # 1. Strategy recommendations based on win rates
            if "win_rates" in results and "by_strategy" in results["win_rates"]:
                strategy_data = pd.DataFrame(results["win_rates"]["by_strategy"])
                if not strategy_data.empty:
                    best_strategy = strategy_data.loc[strategy_data['win_rate'].idxmax()]
                    worst_strategy = strategy_data.loc[strategy_data['win_rate'].idxmin()]
                    
                    recommendations.append({
                        'type': 'strategy_allocation',
                        'description': f"Increase allocation to '{best_strategy['strategy']}' strategy with {best_strategy['win_rate']*100:.1f}% win rate",
                        'confidence': min(best_strategy['win_rate'] * 2, 0.95)
                    })
                    
                    if worst_strategy['win_rate'] < 0.4:  # Less than 40% win rate
                        recommendations.append({
                            'type': 'strategy_allocation',
                            'description': f"Decrease allocation to '{worst_strategy['strategy']}' strategy with low {worst_strategy['win_rate']*100:.1f}% win rate",
                            'confidence': min((0.5 - worst_strategy['win_rate']) * 2, 0.9)
                        })
            
            # 2. Market regime recommendations
            if "regime_performance" in results and "regime_returns" in results["regime_performance"]:
                regime_data = pd.DataFrame(results["regime_performance"]["regime_returns"])
                if not regime_data.empty:
                    best_regime = regime_data.loc[regime_data['sharpe_ratio'].idxmax()]
                    worst_regime = regime_data.loc[regime_data['sharpe_ratio'].idxmin()]
                    
                    recommendations.append({
                        'type': 'regime_adaptation',
                        'description': f"Optimize for '{best_regime['regime']}' regime with Sharpe ratio of {best_regime['sharpe_ratio']:.2f}",
                        'confidence': min(0.5 + best_regime['sharpe_ratio'] / 4, 0.95)
                    })
                    
                    if worst_regime['sharpe_ratio'] < 0.5:
                        recommendations.append({
                            'type': 'risk_management',
                            'description': f"Reduce exposure during '{worst_regime['regime']}' regime (Sharpe ratio: {worst_regime['sharpe_ratio']:.2f})",
                            'confidence': min(0.5 + (0.5 - worst_regime['sharpe_ratio']), 0.9)
                        })
            
            # 3. Time-based recommendations
            if "time_patterns" in results and "by_hour" in results["time_patterns"]:
                hour_data = pd.DataFrame(results["time_patterns"]["by_hour"])
                if not hour_data.empty:
                    best_hours = hour_data[hour_data['win_rate'] > 0.6]
                    worst_hours = hour_data[hour_data['win_rate'] < 0.4]
                    
                    if not best_hours.empty:
                        best_hours_list = best_hours['hour'].tolist()
                        recommendations.append({
                            'type': 'time_filter',
                            'description': f"Focus trading during hours with high win rates: {best_hours_list}",
                            'confidence': min(0.5 + best_hours['win_rate'].mean() / 2, 0.9)
                        })
                    
                    if not worst_hours.empty:
                        worst_hours_list = worst_hours['hour'].tolist()
                        recommendations.append({
                            'type': 'time_filter',
                            'description': f"Avoid trading during hours with low win rates: {worst_hours_list}",
                            'confidence': min(0.5 + (0.5 - worst_hours['win_rate'].mean()), 0.9)
                        })
            
            # 4. Cluster-based recommendations
            if "trade_clusters" in results and "cluster_stats" in results["trade_clusters"]:
                cluster_data = pd.DataFrame(results["trade_clusters"]["cluster_stats"])
                if not cluster_data.empty and 'avg_pnl' in cluster_data.columns:
                    best_cluster = cluster_data.loc[cluster_data['avg_pnl'].idxmax()]
                    
                    # Get top features of the best cluster
                    feature_cols = [col for col in best_cluster.index if col.startswith('avg_') and col not in ['avg_pnl']]
                    
                    if feature_cols:
                        feature_values = {col: best_cluster[col] for col in feature_cols}
                        top_features = sorted(feature_values.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                        
                        feature_desc = ", ".join([f"{feat.replace('avg_', '')}: {val:.2f}" for feat, val in top_features])
                        
                        recommendations.append({
                            'type': 'trade_characteristics',
                            'description': f"Prioritize trades with these characteristics: {feature_desc}",
                            'confidence': min(0.5 + best_cluster['win_rate'] / 2 if 'win_rate' in best_cluster else 0.7, 0.9)
                        })
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations 