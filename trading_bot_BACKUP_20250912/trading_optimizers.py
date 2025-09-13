"""
Trading Optimizers Module

This module provides advanced optimization tools for trading systems including:
1. Strategy Rotation - Automatically rotate capital between strategies based on performance
2. Confidence Scoring - Calculate confidence scores for trades to adjust position sizes
3. Report Generation - Generate daily/weekly PDF/HTML reports with metrics and insights
4. Trade Replay - Reconstruct and visualize historical trades for analysis
"""

import logging
import pandas as pd
import numpy as np
import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import os
import jinja2
import base64
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyRotator:
    """
    Automatically rotates capital between trading strategies based on performance metrics.
    Uses a dynamic allocation algorithm to weight strategies according to recent performance,
    risk-adjusted returns, and market regime compatibility.
    """
    
    def __init__(self, 
                 strategies: List[str], 
                 initial_capital: float, 
                 lookback_periods: List[int] = [7, 30, 90],
                 metrics: List[str] = ['sharpe', 'win_rate', 'profit_factor', 'expectancy'],
                 minimum_allocation: float = 0.05,
                 regime_compatibility: Dict[str, Dict[str, float]] = None):
        """
        Initialize the strategy rotator.
        
        Args:
            strategies: List of strategy names to rotate between
            initial_capital: Total capital to allocate across strategies
            lookback_periods: List of periods (in days) to calculate performance metrics
            metrics: Performance metrics to consider for rotation
            minimum_allocation: Minimum allocation percentage for any active strategy
            regime_compatibility: Dict mapping market regimes to strategy compatibility scores
        """
        self.strategies = strategies
        self.total_capital = initial_capital
        self.lookback_periods = lookback_periods
        self.metrics = metrics
        self.minimum_allocation = minimum_allocation
        self.regime_compatibility = regime_compatibility or {}
        
        # Initialize allocations with equal weight
        self.allocations = {strategy: self.total_capital / len(strategies) for strategy in strategies}
        self.performance_history = {}
        self.allocation_history = []
        
        logger.info(f"Initialized StrategyRotator with {len(strategies)} strategies")
    
    def update_performance_data(self, performance_data: Dict[str, Dict[str, Any]]):
        """
        Update performance metrics for all strategies.
        
        Args:
            performance_data: Dict mapping strategy names to their performance metrics
        """
        self.performance_history = performance_data
        logger.info(f"Updated performance data for {len(performance_data)} strategies")
    
    def update_market_regime(self, current_regime: str):
        """
        Update the current market regime to adjust strategy compatibility.
        
        Args:
            current_regime: Current market regime classification
        """
        self.current_regime = current_regime
        logger.info(f"Updated market regime to: {current_regime}")
    
    def calculate_allocations(self) -> Dict[str, float]:
        """
        Calculate optimal capital allocations across strategies based on performance metrics,
        strategy correlations, and market regime compatibility.
        
        Returns:
            Dict mapping strategy names to capital allocations
        """
        if not self.performance_history:
            logger.warning("Performance history is empty, using equal allocations")
            return self.allocations
        
        # Calculate performance scores
        scores = {}
        for strategy in self.strategies:
            if strategy not in self.performance_history:
                scores[strategy] = 0
                continue
                
            # Calculate performance score based on multiple metrics and lookback periods
            score = 0
            metrics_count = 0
            
            for metric in self.metrics:
                if metric in self.performance_history[strategy]:
                    # Weight recent performance more heavily
                    short_term = self.performance_history[strategy].get(f"{metric}_7d", 0)
                    medium_term = self.performance_history[strategy].get(f"{metric}_30d", 0)
                    long_term = self.performance_history[strategy].get(f"{metric}_90d", 0)
                    
                    # Weighted average with higher weights for more recent periods
                    weighted_score = (short_term * 0.5) + (medium_term * 0.3) + (long_term * 0.2)
                    score += weighted_score
                    metrics_count += 1
            
            # Avoid division by zero
            if metrics_count > 0:
                score /= metrics_count
            
            # Apply regime compatibility adjustment if available
            if (self.current_regime in self.regime_compatibility and 
                strategy in self.regime_compatibility[self.current_regime]):
                regime_factor = self.regime_compatibility[self.current_regime][strategy]
                score *= regime_factor
                logger.debug(f"Applied regime factor {regime_factor} to {strategy}")
            
            scores[strategy] = max(score, 0.001)  # Ensure non-negative scores
        
        # Normalize scores to get allocation percentages
        total_score = sum(scores.values())
        if total_score > 0:
            raw_allocations = {s: (scores[s] / total_score) for s in self.strategies}
        else:
            # Equal allocations if total score is zero
            raw_allocations = {s: (1.0 / len(self.strategies)) for s in self.strategies}
        
        # Apply minimum allocation constraint
        active_strategies = [s for s in self.strategies if raw_allocations[s] >= self.minimum_allocation]
        
        if not active_strategies:
            # If no strategies meet minimum threshold, use highest scoring one
            best_strategy = max(scores, key=scores.get)
            active_strategies = [best_strategy]
        
        # Recalculate allocations for active strategies
        active_total_score = sum(scores[s] for s in active_strategies)
        final_allocations = {s: (scores[s] / active_total_score if s in active_strategies else 0) 
                            for s in self.strategies}
        
        # Convert percentages to actual capital amounts
        self.allocations = {s: final_allocations[s] * self.total_capital for s in self.strategies}
        
        # Log allocation changes
        self.allocation_history.append({
            'date': datetime.datetime.now().isoformat(),
            'allocations': self.allocations.copy(),
            'regime': getattr(self, 'current_regime', 'unknown')
        })
        
        logger.info(f"Calculated new allocations across {len(active_strategies)} active strategies")
        return self.allocations
    
    def get_allocation_changes(self) -> Dict[str, float]:
        """
        Calculate changes in allocation compared to previous state.
        
        Returns:
            Dict mapping strategy names to changes in allocation (positive or negative)
        """
        if len(self.allocation_history) < 2:
            return {s: 0 for s in self.strategies}
        
        prev_allocations = self.allocation_history[-2]['allocations']
        changes = {s: self.allocations[s] - prev_allocations.get(s, 0) for s in self.strategies}
        
        return changes
    
    def generate_rotation_report(self) -> Dict[str, Any]:
        """
        Generate a report with rotation decisions and reasoning.
        
        Returns:
            Dict containing rotation report data
        """
        changes = self.get_allocation_changes()
        significant_changes = {s: changes[s] for s in changes if abs(changes[s]) > 0.05 * self.total_capital}
        
        report = {
            'date': datetime.datetime.now().isoformat(),
            'total_capital': self.total_capital,
            'current_allocations': self.allocations,
            'allocation_changes': changes,
            'significant_changes': significant_changes,
            'market_regime': getattr(self, 'current_regime', 'unknown'),
            'active_strategies': [s for s in self.strategies if self.allocations[s] > 0],
            'inactive_strategies': [s for s in self.strategies if self.allocations[s] == 0]
        }
        
        return report
            

class ConfidenceScorer:
    """
    Calculates confidence scores for trades based on multiple factors.
    Used to adjust position sizes dynamically based on conviction level.
    """
    
    class ConfidenceFactors(Enum):
        STRATEGY_PERFORMANCE = "strategy_performance"
        SETUP_QUALITY = "setup_quality"
        MARKET_ALIGNMENT = "market_alignment"
        VOLATILITY_CONTEXT = "volatility_context"
        TREND_STRENGTH = "trend_strength"
        SUPPORT_RESISTANCE = "support_resistance"
        VOLUME_CONFIRMATION = "volume_confirmation"
        NEWS_SENTIMENT = "news_sentiment"
        SECTOR_STRENGTH = "sector_strength"
        
    def __init__(self, 
                 default_weights: Dict[str, float] = None,
                 min_confidence: float = 0.3,
                 max_confidence: float = 1.0):
        """
        Initialize the confidence scorer.
        
        Args:
            default_weights: Default weights for each confidence factor
            min_confidence: Minimum confidence score to return
            max_confidence: Maximum confidence score to return
        """
        # Default factor weights if not provided
        self.weights = default_weights or {
            self.ConfidenceFactors.STRATEGY_PERFORMANCE.value: 0.15,
            self.ConfidenceFactors.SETUP_QUALITY.value: 0.20,
            self.ConfidenceFactors.MARKET_ALIGNMENT.value: 0.15,
            self.ConfidenceFactors.VOLATILITY_CONTEXT.value: 0.10,
            self.ConfidenceFactors.TREND_STRENGTH.value: 0.10,
            self.ConfidenceFactors.SUPPORT_RESISTANCE.value: 0.10,
            self.ConfidenceFactors.VOLUME_CONFIRMATION.value: 0.05,
            self.ConfidenceFactors.NEWS_SENTIMENT.value: 0.05,
            self.ConfidenceFactors.SECTOR_STRENGTH.value: 0.10
        }
        
        # Ensure weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
            
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.confidence_history = []
        
        logger.info("Initialized ConfidenceScorer with %d factors", len(self.weights))
        
    def calculate_confidence(self, 
                            factor_scores: Dict[str, float], 
                            strategy_name: str = None) -> Dict[str, Any]:
        """
        Calculate overall confidence score based on individual factor scores.
        
        Args:
            factor_scores: Dict mapping factor names to scores (0.0 to 1.0)
            strategy_name: Name of the strategy being scored (optional)
            
        Returns:
            Dict containing overall confidence score and component scores
        """
        # Fill missing factors with neutral values
        complete_scores = {factor: factor_scores.get(factor, 0.5) for factor in self.weights}
        
        # Calculate weighted average
        weighted_scores = {factor: score * self.weights.get(factor, 0) 
                          for factor, score in complete_scores.items()}
        
        overall_score = sum(weighted_scores.values())
        
        # Apply bounds
        bounded_score = max(min(overall_score, self.max_confidence), self.min_confidence)
        
        # Scale score to be between min and max confidence
        range_size = self.max_confidence - self.min_confidence
        if range_size > 0 and overall_score > self.min_confidence:
            normalized_score = self.min_confidence + (
                (overall_score - self.min_confidence) / 
                (1.0 - self.min_confidence) * range_size
            )
        else:
            normalized_score = bounded_score
        
        result = {
            'timestamp': datetime.datetime.now().isoformat(),
            'strategy': strategy_name,
            'overall_confidence': bounded_score,
            'normalized_confidence': normalized_score,
            'component_scores': complete_scores,
            'weighted_scores': weighted_scores
        }
        
        # Store in history
        self.confidence_history.append(result)
        
        logger.info(f"Calculated confidence score: {bounded_score:.2f} for strategy: {strategy_name}")
        return result
    
    def adjust_position_size(self, base_size: float, confidence_result: Dict[str, Any]) -> float:
        """
        Adjust position size based on confidence score.
        
        Args:
            base_size: Base position size before adjustment
            confidence_result: Result from calculate_confidence method
            
        Returns:
            Adjusted position size
        """
        confidence = confidence_result['normalized_confidence']
        
        # Linear adjustment based on confidence
        adjusted_size = base_size * confidence
        
        logger.info(f"Adjusted position size from {base_size:.2f} to {adjusted_size:.2f} " +
                   f"based on confidence {confidence:.2f}")
        
        return adjusted_size
    
    def get_confidence_threshold_signals(self, threshold: float = 0.7) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get high and low confidence signals based on threshold.
        
        Args:
            threshold: Confidence threshold for high/low categorization
            
        Returns:
            Dict with high and low confidence signals
        """
        if not self.confidence_history:
            return {'high': [], 'low': []}
            
        recent_scores = self.confidence_history[-50:]  # Last 50 signals
        
        high_confidence = [score for score in recent_scores 
                          if score['overall_confidence'] >= threshold]
        low_confidence = [score for score in recent_scores 
                         if score['overall_confidence'] < threshold]
        
        return {
            'high': high_confidence,
            'low': low_confidence
        }


class TradeReportGenerator:
    """
    Generates comprehensive trade reports in HTML or PDF format.
    Provides daily and weekly summaries with metrics, visualizations, and recommendations.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = output_dir
        self.template_dir = os.path.join(os.path.dirname(__file__), "templates")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup Jinja2 environment for HTML templates
        try:
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.template_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
            logger.info(f"Initialized Jinja2 environment with template dir: {self.template_dir}")
        except Exception as e:
            logger.warning(f"Could not initialize Jinja2 environment: {e}")
            self.jinja_env = None
            
        self.report_history = []
        logger.info(f"Initialized TradeReportGenerator with output directory: {self.output_dir}")
        
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for HTML embedding"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        return base64.b64encode(image_png).decode('utf-8')
        
    def generate_daily_report(self, 
                             trade_data: List[Dict[str, Any]],
                             performance_metrics: Dict[str, Any],
                             strategy_allocations: Dict[str, float] = None,
                             market_context: Dict[str, Any] = None) -> str:
        """
        Generate a daily trading report in HTML format.
        
        Args:
            trade_data: List of trade dictionaries with trade details
            performance_metrics: Dictionary of performance metrics
            strategy_allocations: Current strategy allocations
            market_context: Current market context information
            
        Returns:
            Path to the generated report file
        """
        # Create a report date (today's date)
        report_date = datetime.datetime.now().strftime("%Y-%m-%d")
        report_filename = f"daily_report_{report_date}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Generate trade summary
        trades_df = pd.DataFrame(trade_data) if trade_data else pd.DataFrame()
        
        # Create performance visualizations
        performance_fig = None
        if trades_df.shape[0] > 0 and 'pnl' in trades_df.columns:
            plt.figure(figsize=(10, 6))
            
            # Daily PnL
            if 'entry_time' in trades_df.columns:
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
                daily_pnl = trades_df.groupby('entry_date')['pnl'].sum()
                
                plt.subplot(2, 1, 1)
                daily_pnl.plot(kind='bar', color=['g' if x >= 0 else 'r' for x in daily_pnl])
                plt.title('Daily P&L')
                plt.xlabel('Date')
                plt.ylabel('P&L')
                plt.grid(True, alpha=0.3)
                
                # Cumulative P&L
                plt.subplot(2, 1, 2)
                daily_pnl.cumsum().plot()
                plt.title('Cumulative P&L')
                plt.xlabel('Date')
                plt.ylabel('Cumulative P&L')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            performance_fig = plt.gcf()
            performance_img = self._fig_to_base64(performance_fig)
            plt.close()
        else:
            performance_img = None
            
        # Strategy allocation chart
        allocation_img = None
        if strategy_allocations:
            plt.figure(figsize=(8, 8))
            allocations_series = pd.Series(strategy_allocations)
            allocations_series = allocations_series[allocations_series > 0]  # Filter out zero allocations
            allocations_series.plot.pie(autopct='%1.1f%%', startangle=90)
            plt.title('Strategy Allocations')
            plt.ylabel('')
            allocation_img = self._fig_to_base64(plt.gcf())
            plt.close()
        
        # Trade type breakdown
        trade_type_img = None
        if trades_df.shape[0] > 0 and 'side' in trades_df.columns:
            plt.figure(figsize=(8, 6))
            trades_df['side'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
            plt.title('Trade Sides')
            plt.ylabel('')
            trade_type_img = self._fig_to_base64(plt.gcf())
            plt.close()
        
        # Prepare report context
        context = {
            'report_date': report_date,
            'trades': trade_data,
            'performance_metrics': performance_metrics,
            'strategy_allocations': strategy_allocations,
            'market_context': market_context,
            'performance_img': performance_img,
            'allocation_img': allocation_img,
            'trade_type_img': trade_type_img
        }
        
        # Generate HTML report using template if available
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template('daily_report.html')
                report_html = template.render(**context)
                
                with open(report_path, 'w') as f:
                    f.write(report_html)
                    
                logger.info(f"Generated daily report at: {report_path}")
            except Exception as e:
                logger.error(f"Error generating report with template: {e}")
                # Fallback to basic HTML
                self._generate_basic_html_report(context, report_path)
        else:
            # Generate basic HTML report without Jinja
            self._generate_basic_html_report(context, report_path)
        
        # Add to report history
        self.report_history.append({
            'date': report_date,
            'type': 'daily',
            'path': report_path
        })
        
        return report_path
    
    def _generate_basic_html_report(self, context: Dict[str, Any], report_path: str):
        """Generate a basic HTML report without using Jinja templates"""
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Trading Report {context['report_date']}</title></head><body>")
            f.write(f"<h1>Daily Trading Report - {context['report_date']}</h1>")
            
            # Performance metrics section
            f.write("<h2>Performance Metrics</h2>")
            if context.get('performance_metrics'):
                f.write("<table border='1'><tr><th>Metric</th><th>Value</th></tr>")
                for key, value in context['performance_metrics'].items():
                    f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
                f.write("</table>")
            else:
                f.write("<p>No performance metrics available.</p>")
            
            # Add performance image if available
            if context.get('performance_img'):
                f.write("<h2>Performance Chart</h2>")
                f.write(f"<img src='data:image/png;base64,{context['performance_img']}' />")
            
            # Strategy allocations section
            if context.get('strategy_allocations'):
                f.write("<h2>Strategy Allocations</h2>")
                f.write("<table border='1'><tr><th>Strategy</th><th>Allocation</th></tr>")
                for strategy, allocation in context['strategy_allocations'].items():
                    f.write(f"<tr><td>{strategy}</td><td>${allocation:.2f}</td></tr>")
                f.write("</table>")
                
                # Add allocation chart if available
                if context.get('allocation_img'):
                    f.write(f"<img src='data:image/png;base64,{context['allocation_img']}' />")
            
            # Trade list section
            f.write("<h2>Today's Trades</h2>")
            if context.get('trades'):
                f.write("<table border='1'><tr>")
                # Use first trade to get columns
                columns = list(context['trades'][0].keys())
                for col in columns:
                    f.write(f"<th>{col}</th>")
                f.write("</tr>")
                
                for trade in context['trades']:
                    f.write("<tr>")
                    for col in columns:
                        value = trade.get(col, '')
                        f.write(f"<td>{value}</td>")
                    f.write("</tr>")
                f.write("</table>")
            else:
                f.write("<p>No trades executed today.</p>")
            
            # Market context section
            if context.get('market_context'):
                f.write("<h2>Market Context</h2>")
                f.write("<table border='1'><tr><th>Factor</th><th>Value</th></tr>")
                for key, value in context['market_context'].items():
                    f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
                f.write("</table>")
            
            f.write("</body></html>")
            
        logger.info(f"Generated basic HTML report at: {report_path}")
    
    def generate_weekly_report(self,
                              weekly_trade_data: List[Dict[str, Any]],
                              weekly_performance: Dict[str, Any],
                              strategy_performance: Dict[str, Dict[str, Any]] = None,
                              recommendations: List[str] = None) -> str:
        """
        Generate a weekly trading report with more comprehensive analysis.
        
        Args:
            weekly_trade_data: List of trades for the week
            weekly_performance: Weekly performance metrics
            strategy_performance: Performance metrics by strategy
            recommendations: List of recommendations for the next week
            
        Returns:
            Path to the generated report file
        """
        # Create a report for the past week
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=7)
        
        report_filename = f"weekly_report_{start_date.strftime('%Y-%m-%d')}_to_{end_date.strftime('%Y-%m-%d')}.html"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Additional analysis for weekly report
        weekly_df = pd.DataFrame(weekly_trade_data) if weekly_trade_data else pd.DataFrame()
        
        # Generate more charts for weekly report
        charts = {}
        
        if weekly_df.shape[0] > 0:
            # Win/loss ratio chart
            if 'is_win' in weekly_df.columns:
                plt.figure(figsize=(8, 6))
                win_loss = weekly_df['is_win'].value_counts()
                labels = ['Win', 'Loss']
                plt.pie([win_loss.get(True, 0), win_loss.get(False, 0)], 
                      labels=labels, 
                      autopct='%1.1f%%',
                      colors=['green', 'red'])
                plt.title('Win/Loss Ratio')
                charts['win_loss_chart'] = self._fig_to_base64(plt.gcf())
                plt.close()
            
            # Profit by strategy chart
            if 'strategy' in weekly_df.columns and 'pnl' in weekly_df.columns:
                plt.figure(figsize=(10, 6))
                strategy_pnl = weekly_df.groupby('strategy')['pnl'].sum().sort_values(ascending=False)
                strategy_pnl.plot(kind='bar', color=['g' if x >= 0 else 'r' for x in strategy_pnl])
                plt.title('Profit by Strategy')
                plt.grid(True, alpha=0.3)
                charts['strategy_pnl_chart'] = self._fig_to_base64(plt.gcf())
                plt.close()
            
            # Trade duration analysis
            if 'duration_minutes' in weekly_df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(weekly_df['duration_minutes'], bins=20)
                plt.title('Trade Duration Distribution')
                plt.xlabel('Duration (minutes)')
                plt.grid(True, alpha=0.3)
                charts['duration_chart'] = self._fig_to_base64(plt.gcf())
                plt.close()
            
            # Profit by instrument chart
            if 'symbol' in weekly_df.columns and 'pnl' in weekly_df.columns:
                plt.figure(figsize=(12, 6))
                symbol_pnl = weekly_df.groupby('symbol')['pnl'].sum().sort_values(ascending=False).head(10)
                symbol_pnl.plot(kind='bar', color=['g' if x >= 0 else 'r' for x in symbol_pnl])
                plt.title('Top 10 Instruments by Profit')
                plt.grid(True, alpha=0.3)
                charts['symbol_pnl_chart'] = self._fig_to_base64(plt.gcf())
                plt.close()
        
        # Prepare context for weekly report
        context = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'trade_count': len(weekly_trade_data) if weekly_trade_data else 0,
            'weekly_performance': weekly_performance,
            'strategy_performance': strategy_performance,
            'recommendations': recommendations,
            'charts': charts
        }
        
        # Generate HTML report using template if available
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template('weekly_report.html')
                report_html = template.render(**context)
                
                with open(report_path, 'w') as f:
                    f.write(report_html)
                    
                logger.info(f"Generated weekly report at: {report_path}")
            except Exception as e:
                logger.error(f"Error generating weekly report with template: {e}")
                # Fallback to basic HTML
                self._generate_basic_weekly_html_report(context, report_path)
        else:
            # Generate basic HTML report without Jinja
            self._generate_basic_weekly_html_report(context, report_path)
        
        # Add to report history
        self.report_history.append({
            'date': end_date.strftime('%Y-%m-%d'),
            'type': 'weekly',
            'path': report_path
        })
        
        return report_path
    
    def _generate_basic_weekly_html_report(self, context: Dict[str, Any], report_path: str):
        """Generate a basic HTML weekly report without using Jinja templates"""
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Weekly Trading Report</title></head><body>")
            
            f.write(f"<h1>Weekly Trading Report - {context['start_date']} to {context['end_date']}</h1>")
            f.write(f"<p>Total Trades: {context['trade_count']}</p>")
            
            # Weekly performance metrics
            f.write("<h2>Weekly Performance</h2>")
            if context.get('weekly_performance'):
                f.write("<table border='1'><tr><th>Metric</th><th>Value</th></tr>")
                for key, value in context['weekly_performance'].items():
                    f.write(f"<tr><td>{key}</td><td>{value}</td></tr>")
                f.write("</table>")
            
            # Strategy performance
            if context.get('strategy_performance'):
                f.write("<h2>Strategy Performance</h2>")
                f.write("<table border='1'><tr><th>Strategy</th><th>Win Rate</th><th>Profit Factor</th><th>Total P&L</th></tr>")
                for strategy, metrics in context['strategy_performance'].items():
                    f.write(f"<tr><td>{strategy}</td><td>{metrics.get('win_rate', 'N/A')}</td>")
                    f.write(f"<td>{metrics.get('profit_factor', 'N/A')}</td><td>{metrics.get('total_pnl', 'N/A')}</td></tr>")
                f.write("</table>")
            
            # Charts
            if context.get('charts'):
                f.write("<h2>Performance Analysis</h2>")
                for chart_name, chart_data in context['charts'].items():
                    f.write(f"<div><img src='data:image/png;base64,{chart_data}' /></div>")
            
            # Recommendations
            if context.get('recommendations'):
                f.write("<h2>Recommendations for Next Week</h2>")
                f.write("<ul>")
                for rec in context['recommendations']:
                    f.write(f"<li>{rec}</li>")
                f.write("</ul>")
            
            f.write("</body></html>")
            
        logger.info(f"Generated basic weekly HTML report at: {report_path}")
    
    def get_latest_report(self, report_type: str = 'daily') -> Optional[str]:
        """
        Get the path to the latest report of specified type.
        
        Args:
            report_type: Type of report ('daily' or 'weekly')
            
        Returns:
            Path to the latest report file or None if not found
        """
        matching_reports = [r for r in self.report_history if r['type'] == report_type]
        if matching_reports:
            return sorted(matching_reports, key=lambda x: x['date'], reverse=True)[0]['path']
        return None


class ReplayEngine:
    """
    Reconstructs and visualizes historical trades for analysis.
    Enables stepping through trades in sequence to analyze decision making.
    """
    
    def __init__(self):
        """Initialize the replay engine."""
        self.trade_history = []
        self.current_position = 0
        self.replay_state = {}
        
        logger.info("Initialized ReplayEngine")
    
    def load_trades(self, trades: List[Dict[str, Any]]):
        """
        Load trade history for replay.
        
        Args:
            trades: List of trade dictionaries with trade details
        """
        # Sort trades by entry time
        if trades and 'entry_time' in trades[0]:
            self.trade_history = sorted(trades, key=lambda x: x.get('entry_time', ''))
        else:
            self.trade_history = trades
            
        self.current_position = 0
        logger.info(f"Loaded {len(trades)} trades for replay")
    
    def get_current_trade(self) -> Optional[Dict[str, Any]]:
        """
        Get the current trade in the replay sequence.
        
        Returns:
            Current trade dictionary or None if at the end
        """
        if not self.trade_history or self.current_position >= len(self.trade_history):
            return None
        return self.trade_history[self.current_position]
    
    def next_trade(self) -> Optional[Dict[str, Any]]:
        """
        Move to the next trade in the sequence.
        
        Returns:
            Next trade dictionary or None if at the end
        """
        if self.current_position < len(self.trade_history) - 1:
            self.current_position += 1
            return self.get_current_trade()
        return None
    
    def previous_trade(self) -> Optional[Dict[str, Any]]:
        """
        Move to the previous trade in the sequence.
        
        Returns:
            Previous trade dictionary or None if at the beginning
        """
        if self.current_position > 0:
            self.current_position -= 1
            return self.get_current_trade()
        return None
    
    def jump_to_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Jump to a specific trade by ID.
        
        Args:
            trade_id: ID of the trade to jump to
            
        Returns:
            The trade dictionary or None if not found
        """
        for i, trade in enumerate(self.trade_history):
            if trade.get('id') == trade_id:
                self.current_position = i
                return trade
        logger.warning(f"Trade with ID {trade_id} not found in history")
        return None
    
    def get_trade_sequence(self, window_size: int = 5) -> List[Dict[str, Any]]:
        """
        Get a window of trades around the current position.
        
        Args:
            window_size: Number of trades to include before and after current
            
        Returns:
            List of trade dictionaries in the sequence window
        """
        start_idx = max(0, self.current_position - window_size)
        end_idx = min(len(self.trade_history), self.current_position + window_size + 1)
        
        return self.trade_history[start_idx:end_idx]
    
    def get_trade_context(self, trade_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed context for the current or specified trade.
        
        Args:
            trade_id: Optional ID of the trade to get context for
            
        Returns:
            Dictionary with trade details and surrounding context
        """
        if trade_id:
            self.jump_to_trade(trade_id)
            
        current_trade = self.get_current_trade()
        if not current_trade:
            return {"error": "No trade available"}
            
        # Get surrounding trades
        surrounding_trades = self.get_trade_sequence(3)
        
        # Classify if this was a good trade
        is_good_trade = current_trade.get('pnl', 0) > 0
        
        # Calculate statistics
        trade_metrics = {
            'total_trades_so_far': self.current_position + 1,
            'win_rate_so_far': sum(1 for t in self.trade_history[:self.current_position+1] 
                                  if t.get('pnl', 0) > 0) / (self.current_position + 1),
            'avg_win_so_far': np.mean([t.get('pnl', 0) for t in self.trade_history[:self.current_position+1] 
                                     if t.get('pnl', 0) > 0]) if any(t.get('pnl', 0) > 0 for t in self.trade_history[:self.current_position+1]) else 0,
            'avg_loss_so_far': np.mean([abs(t.get('pnl', 0)) for t in self.trade_history[:self.current_position+1] 
                                      if t.get('pnl', 0) < 0]) if any(t.get('pnl', 0) < 0 for t in self.trade_history[:self.current_position+1]) else 0
        }
        
        return {
            'current_trade': current_trade,
            'surrounding_trades': surrounding_trades,
            'position_in_sequence': self.current_position,
            'total_trades': len(self.trade_history),
            'is_good_trade': is_good_trade,
            'trade_metrics': trade_metrics
        }
    
    def visualize_trade(self, trade_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create visualization data for the current or specified trade.
        
        Args:
            trade_id: Optional ID of the trade to visualize
            
        Returns:
            Dictionary with visualization data
        """
        if trade_id:
            self.jump_to_trade(trade_id)
            
        current_trade = self.get_current_trade()
        if not current_trade:
            return {"error": "No trade available to visualize"}
        
        # Create basic visualization data
        visualization = {
            'trade_id': current_trade.get('id'),
            'symbol': current_trade.get('symbol'),
            'entry_time': current_trade.get('entry_time'),
            'exit_time': current_trade.get('exit_time'),
            'entry_price': current_trade.get('entry_price'),
            'exit_price': current_trade.get('exit_price'),
            'side': current_trade.get('side'),
            'pnl': current_trade.get('pnl', 0),
            'strategy': current_trade.get('strategy'),
            'setup_type': current_trade.get('setup_type'),
            'position_in_sequence': self.current_position,
            'total_trades': len(self.trade_history)
        }
        
        return visualization
    
    def search_similar_trades(self, trade_id: Optional[str] = None, 
                             criteria: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar trades to the current or specified trade.
        
        Args:
            trade_id: Optional ID of the trade to find similar trades for
            criteria: List of criteria to match (e.g., 'symbol', 'strategy', 'setup_type')
            
        Returns:
            List of similar trade dictionaries
        """
        if trade_id:
            self.jump_to_trade(trade_id)
            
        current_trade = self.get_current_trade()
        if not current_trade:
            return []
            
        # Default criteria if none specified
        if not criteria:
            criteria = ['symbol', 'strategy', 'setup_type']
            
        # Filter similar trades
        similar_trades = []
        for trade in self.trade_history:
            if trade.get('id') == current_trade.get('id'):
                continue  # Skip the current trade
                
            # Check if trade matches all criteria
            is_match = True
            for criterion in criteria:
                if criterion in current_trade and criterion in trade:
                    if current_trade[criterion] != trade[criterion]:
                        is_match = False
                        break
                        
            if is_match:
                similar_trades.append(trade)
                
        return similar_trades 