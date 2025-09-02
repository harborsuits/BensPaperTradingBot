#!/usr/bin/env python3
"""
Autonomous Backtester for ML-Powered Trading Strategies

This module automatically runs backtests for ML-generated strategies
and catalogs the results for future learning.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import uuid

logger = logging.getLogger(__name__)

class BacktestResultAnalyzer:
    """Analyzes backtest results to generate insights"""
    
    def __init__(self):
        """Initialize the backtest result analyzer"""
        logger.info("BacktestResultAnalyzer initialized")
        
    def analyze_results(self, winning_strategies, losing_strategies):
        """
        Analyze backtest results to generate insights
        
        Args:
            winning_strategies: List of successful strategies
            losing_strategies: List of unsuccessful strategies
            
        Returns:
            dict: Insights and patterns from results
        """
        logger.info(f"Analyzing {len(winning_strategies)} winning and {len(losing_strategies)} losing strategies")
        
        # Extract features for analysis
        winning_features = self._extract_strategy_features(winning_strategies)
        losing_features = self._extract_strategy_features(losing_strategies)
        
        # Generate insights
        return {
            "winning_patterns": self._identify_patterns(winning_features, "winning"),
            "losing_patterns": self._identify_patterns(losing_features, "losing"),
            "strategy_comparisons": self._compare_strategies(winning_features, losing_features),
            "market_condition_analysis": self._analyze_market_conditions(winning_features, losing_features),
            "parameter_insights": self._analyze_parameters(winning_features, losing_features),
            "timestamp": datetime.now()
        }
        
    def _extract_strategy_features(self, strategies):
        """Extract features from strategies for analysis"""
        features = []
        
        for strategy_result in strategies:
            strategy = strategy_result.get("strategy", {})
            performance = strategy_result.get("aggregate_performance", {})
            
            feature = {
                "id": strategy.get("id", ""),
                "template": strategy.get("template", ""),
                "params": strategy.get("params", {}),
                "risk_params": strategy.get("risk_params", {}),
                "return": performance.get("return", 0),
                "sharpe_ratio": performance.get("sharpe_ratio", 0),
                "max_drawdown": performance.get("max_drawdown", 0),
                "win_rate": performance.get("win_rate", 0)
            }
            
            features.append(feature)
            
        return features
        
    def _identify_patterns(self, features, category):
        """Identify patterns in strategy features"""
        if not features:
            return []
            
        patterns = []
        
        # Group by template
        templates = {}
        for feature in features:
            template = feature.get("template", "unknown")
            if template not in templates:
                templates[template] = []
            templates[template].append(feature)
            
        # Analyze patterns by template
        for template, template_features in templates.items():
            if len(template_features) >= 2:
                avg_return = np.mean([f.get("return", 0) for f in template_features])
                avg_sharpe = np.mean([f.get("sharpe_ratio", 0) for f in template_features])
                
                if category == "winning" and avg_return > 0:
                    patterns.append(f"{template.replace('_', ' ').title()} strategies show consistent positive returns (avg {avg_return:.2f}%)")
                    
                    if avg_sharpe > 1.5:
                        patterns.append(f"{template.replace('_', ' ').title()} strategies show strong risk-adjusted returns (avg Sharpe {avg_sharpe:.2f})")
                
                # Analyze parameter patterns
                param_patterns = self._analyze_template_parameters(template_features, template)
                patterns.extend(param_patterns)
        
        return patterns
        
    def _analyze_template_parameters(self, features, template):
        """Analyze patterns in parameters for a specific template"""
        patterns = []
        
        # Don't analyze if too few samples
        if len(features) < 2:
            return patterns
            
        # Extract parameter values
        param_values = {}
        for feature in features:
            params = feature.get("params", {})
            for param, value in params.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(value)
                
        # Look for patterns in parameters
        for param, values in param_values.items():
            if len(values) >= 2:
                avg_value = np.mean(values)
                std_value = np.std(values)
                
                # Check if values are consistent
                if std_value / avg_value < 0.2:  # Low coefficient of variation
                    patterns.append(f"Consistent {param.replace('_', ' ')} values around {avg_value:.1f} work well for {template.replace('_', ' ')} strategies")
                elif std_value / avg_value > 0.5:  # High variation
                    patterns.append(f"Wide range of {param.replace('_', ' ')} values work for {template.replace('_', ' ')} strategies, suggesting parameter is less critical")
                    
        return patterns
        
    def _compare_strategies(self, winning_features, losing_features):
        """Compare winning and losing strategies"""
        comparisons = []
        
        # Don't compare if either set is empty
        if not winning_features or not losing_features:
            return comparisons
            
        # Compare average returns and Sharpe ratios
        avg_winning_return = np.mean([f.get("return", 0) for f in winning_features])
        avg_losing_return = np.mean([f.get("return", 0) for f in losing_features])
        avg_winning_sharpe = np.mean([f.get("sharpe_ratio", 0) for f in winning_features])
        avg_losing_sharpe = np.mean([f.get("sharpe_ratio", 0) for f in losing_features])
        
        return_diff = avg_winning_return - avg_losing_return
        sharpe_diff = avg_winning_sharpe - avg_losing_sharpe
        
        comparisons.append(f"Winning strategies outperform losing strategies by {return_diff:.2f}% on average")
        comparisons.append(f"Winning strategies have {sharpe_diff:.2f} higher Sharpe ratios on average")
        
        # Compare strategy types
        winning_templates = {}
        losing_templates = {}
        
        for feature in winning_features:
            template = feature.get("template", "unknown")
            winning_templates[template] = winning_templates.get(template, 0) + 1
            
        for feature in losing_features:
            template = feature.get("template", "unknown")
            losing_templates[template] = losing_templates.get(template, 0) + 1
            
        # Find templates that appear more in winning than losing
        for template, count in winning_templates.items():
            losing_count = losing_templates.get(template, 0)
            if count > losing_count and count >= 2:
                template_name = template.replace('_', ' ').title()
                comparisons.append(f"{template_name} strategies appear {count} times in winning vs. {losing_count} times in losing")
                
        return comparisons
        
    def _analyze_market_conditions(self, winning_features, losing_features):
        """Analyze which strategies work in different market conditions"""
        # This would normally analyze real market data
        # For the mock implementation, we'll return generic insights
        return [
            "Momentum strategies perform better in trending markets with higher volatility",
            "Mean reversion strategies perform better in range-bound markets with lower volatility",
            "News sentiment strategies show strongest performance during earnings seasons"
        ]
        
    def _analyze_parameters(self, winning_features, losing_features):
        """Analyze parameter differences between winning and losing strategies"""
        insights = []
        
        # Group by template
        winning_by_template = {}
        losing_by_template = {}
        
        for feature in winning_features:
            template = feature.get("template", "unknown")
            if template not in winning_by_template:
                winning_by_template[template] = []
            winning_by_template[template].append(feature)
            
        for feature in losing_features:
            template = feature.get("template", "unknown")
            if template not in losing_by_template:
                losing_by_template[template] = []
            losing_by_template[template].append(feature)
            
        # Find templates that appear in both winning and losing
        common_templates = set(winning_by_template.keys()) & set(losing_by_template.keys())
        
        for template in common_templates:
            winning_params = {}
            losing_params = {}
            
            # Collect parameter values
            for feature in winning_by_template[template]:
                params = feature.get("params", {})
                for param, value in params.items():
                    if param not in winning_params:
                        winning_params[param] = []
                    winning_params[param].append(value)
                    
            for feature in losing_by_template[template]:
                params = feature.get("params", {})
                for param, value in params.items():
                    if param not in losing_params:
                        losing_params[param] = []
                    losing_params[param].append(value)
                    
            # Compare parameters
            common_params = set(winning_params.keys()) & set(losing_params.keys())
            
            for param in common_params:
                avg_winning = np.mean(winning_params[param])
                avg_losing = np.mean(losing_params[param])
                
                if abs(avg_winning - avg_losing) / max(abs(avg_winning), abs(avg_losing)) > 0.2:
                    # Significant difference
                    template_name = template.replace('_', ' ').title()
                    param_name = param.replace('_', ' ').title()
                    
                    if avg_winning > avg_losing:
                        insights.append(f"Higher {param_name} ({avg_winning:.2f} vs {avg_losing:.2f}) is associated with better performance in {template_name} strategies")
                    else:
                        insights.append(f"Lower {param_name} ({avg_winning:.2f} vs {avg_losing:.2f}) is associated with better performance in {template_name} strategies")
                        
        return insights


class AutonomousBacktester:
    """Runs backtests for ML-generated strategies autonomously"""
    
    def __init__(self, data_layer, strategy_generator, result_analyzer=None):
        """
        Initialize the autonomous backtester
        
        Parameters:
            data_layer: DataIntegrationLayer instance
            strategy_generator: StrategyGenerator instance
            result_analyzer: Component to analyze backtest results
        """
        self.data_layer = data_layer
        self.strategy_generator = strategy_generator
        self.result_analyzer = result_analyzer if result_analyzer else BacktestResultAnalyzer()
        self.database = {}  # Mock database for storing results
        logger.info("AutonomousBacktester initialized")
        
    def run_full_autonomous_cycle(self, tickers=None, timeframes=None, sectors=None):
        """
        Run a complete autonomous cycle of generating and testing strategies
        
        Args:
            tickers: List of tickers to test or None to select automatically
            timeframes: List of timeframes to test
            sectors: List of sectors to filter by
            
        Returns:
            dict: Complete results with winning and losing strategies
        """
        if timeframes is None:
            timeframes = ["1m", "3m", "6m", "1y"]
            
        logger.info(f"Running full autonomous cycle for {len(tickers) if tickers else 'auto-selected'} tickers across {len(timeframes)} timeframes")
        
        results = {
            "winning_strategies": [],
            "losing_strategies": [],
            "ml_insights": {},
            "timestamp": datetime.now()
        }
        
        # If no tickers provided, use ML to select tickers
        if not tickers:
            tickers = self._select_tickers_with_ml(sectors)
            
        for ticker in tickers:
            logger.info(f"Processing ticker: {ticker}")
            
            # Get comprehensive data
            ticker_data = self.data_layer.get_comprehensive_data(ticker=ticker)
            
            # Generate strategies
            strategies = self.strategy_generator.generate_strategies(ticker_data)
            
            # Test each strategy across timeframes
            for strategy in strategies:
                strategy_results = []
                
                for timeframe in timeframes:
                    # Get historical data for this timeframe
                    historical_data = self.data_layer.get_comprehensive_data(
                        ticker=ticker, 
                        timeframe=timeframe
                    ).get("price_data")
                    
                    # Run backtest
                    backtest_result = self._backtest_strategy(
                        strategy, 
                        historical_data, 
                        ticker, 
                        timeframe
                    )
                    
                    strategy_results.append(backtest_result)
                
                # Calculate aggregate results
                aggregate_result = self._aggregate_results(strategy_results)
                
                # Save detailed results
                detailed_result = {
                    "strategy": strategy,
                    "ticker": ticker,
                    "aggregate_performance": aggregate_result,
                    "detailed_results": strategy_results
                }
                
                # Categorize as winning or losing
                if aggregate_result["return"] > 0:
                    results["winning_strategies"].append(detailed_result)
                else:
                    results["losing_strategies"].append(detailed_result)
                
                # Save to database
                self._save_backtest_result(detailed_result)
                
        # Sort strategies by performance
        results["winning_strategies"].sort(
            key=lambda x: x["aggregate_performance"]["return"], 
            reverse=True
        )
        results["losing_strategies"].sort(
            key=lambda x: x["aggregate_performance"]["return"]
        )
        
        # Generate ML insights
        results["ml_insights"] = self.result_analyzer.analyze_results(
            results["winning_strategies"],
            results["losing_strategies"]
        )
        
        logger.info(f"Autonomous cycle complete. Found {len(results['winning_strategies'])} winning and {len(results['losing_strategies'])} losing strategies")
        
        return results
        
    def _select_tickers_with_ml(self, sectors=None):
        """
        Use ML to select promising tickers to test
        
        Args:
            sectors: List of sectors to filter by
            
        Returns:
            list: Selected tickers
        """
        # In a real implementation, this would use ML to select tickers
        # For now, we'll return some popular tech stocks
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
    def _backtest_strategy(self, strategy, historical_data, ticker, timeframe):
        """
        Run a backtest for a single strategy on historical data
        
        Args:
            strategy: Strategy to test
            historical_data: Historical price data
            ticker: Ticker symbol
            timeframe: Timeframe of the data
            
        Returns:
            dict: Backtest results with performance metrics
        """
        logger.info(f"Backtesting {strategy['name']} on {ticker} ({timeframe})")
        
        # Initialize performance metrics
        initial_capital = 100000  # $100k
        
        # Apply strategy rules to generate entry/exit signals
        signals = self._generate_trading_signals(strategy, historical_data)
        
        # Simulate trades
        trades, portfolio_value = self._simulate_trades(
            signals, 
            historical_data, 
            initial_capital,
            strategy["risk_params"]
        )
        
        # Calculate performance metrics
        total_return = (portfolio_value[-1] / initial_capital - 1) * 100 if len(portfolio_value) > 0 else 0
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_value)
        max_drawdown = self._calculate_max_drawdown(portfolio_value)
        win_rate = self._calculate_win_rate(trades)
        
        logger.info(f"Backtest complete. Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}, Win Rate: {win_rate:.2f}%")
        
        # Return comprehensive results
        return {
            "strategy_name": strategy["name"],
            "ticker": ticker,
            "timeframe": timeframe,
            "return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "trades": trades,
            "portfolio_value": portfolio_value
        }
        
    def _generate_trading_signals(self, strategy, historical_data):
        """
        Generate trading signals based on strategy rules
        
        Args:
            strategy: Strategy configuration
            historical_data: Historical price data
            
        Returns:
            DataFrame: Price data with signal column
        """
        if historical_data is None or len(historical_data) == 0:
            return pd.DataFrame()
            
        # Copy data to avoid modifying original
        data = historical_data.copy()
        template = strategy.get("template")
        params = strategy.get("params", {})
        
        # Generate signals based on strategy template
        if template == "moving_average_crossover":
            fast_period = params.get("fast_period", 20)
            slow_period = params.get("slow_period", 50)
            
            # Calculate moving averages
            data['fast_ma'] = data['close'].rolling(window=fast_period).mean()
            data['slow_ma'] = data['close'].rolling(window=slow_period).mean()
            
            # Generate signals (1 for buy, -1 for sell, 0 for hold)
            data['signal'] = 0
            # Buy signal: fast MA crosses above slow MA
            data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
            # Sell signal: fast MA crosses below slow MA
            data.loc[data['fast_ma'] <= data['slow_ma'], 'signal'] = -1
            
        elif template == "rsi_reversal":
            period = params.get("rsi_period", 14)
            oversold = params.get("oversold_threshold", 30)
            overbought = params.get("overbought_threshold", 70)
            
            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Generate signals
            data['signal'] = 0
            # Buy signal: RSI crosses below oversold and then back above
            data.loc[data['rsi'] < oversold, 'signal'] = 1
            # Sell signal: RSI crosses above overbought and then back below
            data.loc[data['rsi'] > overbought, 'signal'] = -1
            
        elif template == "breakout_momentum":
            period = params.get("breakout_period", 20)
            volume_factor = params.get("volume_factor", 1.5)
            
            # Calculate price channels
            data['upper_channel'] = data['high'].rolling(window=period).max()
            data['lower_channel'] = data['low'].rolling(window=period).min()
            data['avg_volume'] = data['volume'].rolling(window=period).mean()
            
            # Generate signals
            data['signal'] = 0
            # Buy signal: price breaks above upper channel with high volume
            breakout_condition = (
                (data['close'] > data['upper_channel'].shift(1)) & 
                (data['volume'] > data['avg_volume'] * volume_factor)
            )
            data.loc[breakout_condition, 'signal'] = 1
            
            # Sell signal: price breaks below lower channel with high volume
            breakdown_condition = (
                (data['close'] < data['lower_channel'].shift(1)) & 
                (data['volume'] > data['avg_volume'] * volume_factor)
            )
            data.loc[breakdown_condition, 'signal'] = -1
            
        elif template == "news_sentiment_momentum":
            # This would normally use real sentiment data
            # For now, we'll create a simple momentum strategy as a placeholder
            momentum_period = params.get("momentum_period", 5)
            
            # Calculate momentum
            data['momentum'] = data['close'].pct_change(periods=momentum_period)
            
            # Generate signals
            data['signal'] = 0
            data.loc[data['momentum'] > 0.02, 'signal'] = 1  # 2% momentum threshold for buy
            data.loc[data['momentum'] < -0.02, 'signal'] = -1  # -2% momentum threshold for sell
            
        else:
            # Default to a simple moving average strategy for unknown templates
            data['ma'] = data['close'].rolling(window=50).mean()
            data['signal'] = 0
            data.loc[data['close'] > data['ma'], 'signal'] = 1
            data.loc[data['close'] <= data['ma'], 'signal'] = -1
            
        # Clean up and return
        data.dropna(inplace=True)
        return data
        
    def _simulate_trades(self, signals, price_data, initial_capital, risk_params):
        """
        Simulate trades based on signals
        
        Args:
            signals: DataFrame with signal column
            price_data: Historical price data
            initial_capital: Initial capital amount
            risk_params: Risk parameters
            
        Returns:
            tuple: (trades, portfolio_value)
        """
        if signals.empty or 'signal' not in signals.columns:
            return [], [initial_capital]
            
        position = 0
        capital = initial_capital
        portfolio_value = [capital]
        trades = []
        
        # Get risk parameters
        position_size = risk_params.get("position_size", 0.1)  # Default to 10% of portfolio
        stop_loss_pct = risk_params.get("stop_loss_percentage", 2.0)
        take_profit_pct = risk_params.get("take_profit_percentage", 4.0)
        trailing_stop = risk_params.get("trailing_stop_enabled", True)
        
        entry_price = 0
        entry_date = None
        highest_since_entry = 0
        stop_loss_price = 0
        take_profit_price = 0
        
        for i, row in signals.iterrows():
            current_price = row['close']
            current_date = i
            signal = row['signal']
            
            # Portfolio value based on current position
            if position != 0:
                # Update portfolio value
                pnl = position * (current_price - entry_price)
                portfolio_value.append(capital + pnl)
                
                # Check for stop loss and take profit
                if position > 0:  # Long position
                    # Update trailing stop if enabled
                    if trailing_stop and current_price > highest_since_entry:
                        highest_since_entry = current_price
                        stop_loss_price = highest_since_entry * (1 - stop_loss_pct/100)
                    
                    # Check if stop loss hit
                    if current_price <= stop_loss_price:
                        # Close position with stop loss
                        pnl = position * (stop_loss_price - entry_price)
                        capital += pnl
                        
                        trades.append({
                            "entry_date": entry_date,
                            "exit_date": current_date,
                            "entry_price": entry_price,
                            "exit_price": stop_loss_price,
                            "position": position,
                            "pnl": pnl,
                            "pnl_pct": (stop_loss_price / entry_price - 1) * 100,
                            "exit_reason": "stop_loss"
                        })
                        
                        position = 0
                        
                    # Check if take profit hit
                    elif current_price >= take_profit_price:
                        # Close position with take profit
                        pnl = position * (take_profit_price - entry_price)
                        capital += pnl
                        
                        trades.append({
                            "entry_date": entry_date,
                            "exit_date": current_date,
                            "entry_price": entry_price,
                            "exit_price": take_profit_price,
                            "position": position,
                            "pnl": pnl,
                            "pnl_pct": (take_profit_price / entry_price - 1) * 100,
                            "exit_reason": "take_profit"
                        })
                        
                        position = 0
                        
                elif position < 0:  # Short position
                    # Update trailing stop if enabled
                    if trailing_stop and current_price < highest_since_entry:
                        highest_since_entry = current_price
                        stop_loss_price = highest_since_entry * (1 + stop_loss_pct/100)
                    
                    # Check if stop loss hit
                    if current_price >= stop_loss_price:
                        # Close position with stop loss
                        pnl = position * (entry_price - stop_loss_price)
                        capital += pnl
                        
                        trades.append({
                            "entry_date": entry_date,
                            "exit_date": current_date,
                            "entry_price": entry_price,
                            "exit_price": stop_loss_price,
                            "position": position,
                            "pnl": pnl,
                            "pnl_pct": (entry_price / stop_loss_price - 1) * 100,
                            "exit_reason": "stop_loss"
                        })
                        
                        position = 0
                        
                    # Check if take profit hit
                    elif current_price <= take_profit_price:
                        # Close position with take profit
                        pnl = position * (entry_price - take_profit_price)
                        capital += pnl
                        
                        trades.append({
                            "entry_date": entry_date,
                            "exit_date": current_date,
                            "entry_price": entry_price,
                            "exit_price": take_profit_price,
                            "position": position,
                            "pnl": pnl,
                            "pnl_pct": (entry_price / take_profit_price - 1) * 100,
                            "exit_reason": "take_profit"
                        })
                        
                        position = 0
            else:
                # No position, just update portfolio value
                portfolio_value.append(capital)
            
            # Signal processing
            if position == 0 and signal != 0:
                # Enter new position
                if signal > 0:  # Buy signal
                    # Calculate position size
                    position = (capital * position_size) / current_price
                    entry_price = current_price
                    entry_date = current_date
                    highest_since_entry = current_price
                    
                    # Set stop loss and take profit levels
                    stop_loss_price = entry_price * (1 - stop_loss_pct/100)
                    take_profit_price = entry_price * (1 + take_profit_pct/100)
                    
                elif signal < 0:  # Sell signal
                    # Calculate position size (negative for short)
                    position = -(capital * position_size) / current_price
                    entry_price = current_price
                    entry_date = current_date
                    highest_since_entry = current_price
                    
                    # Set stop loss and take profit levels
                    stop_loss_price = entry_price * (1 + stop_loss_pct/100)
                    take_profit_price = entry_price * (1 - take_profit_pct/100)
                    
            elif position != 0 and signal != 0 and np.sign(position) != np.sign(signal):
                # Signal to close current position and open new one
                # First close current position
                pnl = position * (current_price - entry_price)
                capital += pnl
                
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": current_date,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "position": position,
                    "pnl": pnl,
                    "pnl_pct": (current_price / entry_price - 1) * 100 if position > 0 else (entry_price / current_price - 1) * 100,
                    "exit_reason": "signal_change"
                })
                
                # Then open new position
                if signal > 0:  # Buy signal
                    position = (capital * position_size) / current_price
                    entry_price = current_price
                    entry_date = current_date
                    highest_since_entry = current_price
                    
                    # Set stop loss and take profit levels
                    stop_loss_price = entry_price * (1 - stop_loss_pct/100)
                    take_profit_price = entry_price * (1 + take_profit_pct/100)
                    
                elif signal < 0:  # Sell signal
                    position = -(capital * position_size) / current_price
                    entry_price = current_price
                    entry_date = current_date
                    highest_since_entry = current_price
                    
                    # Set stop loss and take profit levels
                    stop_loss_price = entry_price * (1 + stop_loss_pct/100)
                    take_profit_price = entry_price * (1 - take_profit_pct/100)
        
        # Close any open position at the end
        if position != 0:
            last_price = signals['close'].iloc[-1]
            pnl = position * (last_price - entry_price)
            capital += pnl
            
            trades.append({
                "entry_date": entry_date,
                "exit_date": signals.index[-1],
                "entry_price": entry_price,
                "exit_price": last_price,
                "position": position,
                "pnl": pnl,
                "pnl_pct": (last_price / entry_price - 1) * 100 if position > 0 else (entry_price / last_price - 1) * 100,
                "exit_reason": "end_of_period"
            })
        
        return trades, portfolio_value
        
    def _calculate_sharpe_ratio(self, portfolio_value, risk_free_rate=0.02, trading_days=252):
        """Calculate Sharpe ratio for a portfolio"""
        if len(portfolio_value) < 2:
            return 0
            
        # Calculate daily returns
        returns = np.array([portfolio_value[i] / portfolio_value[i-1] - 1 for i in range(1, len(portfolio_value))])
        
        # Calculate annualized Sharpe ratio
        excess_returns = returns - (risk_free_rate / trading_days)
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0
            
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(trading_days)
        return sharpe_ratio
        
    def _calculate_max_drawdown(self, portfolio_value):
        """Calculate maximum drawdown for a portfolio"""
        if len(portfolio_value) < 2:
            return 0
            
        # Calculate the running maximum
        running_max = np.maximum.accumulate(portfolio_value)
        
        # Calculate drawdown
        drawdown = (np.array(portfolio_value) - running_max) / running_max * 100
        
        # Return the maximum drawdown
        return float(np.min(drawdown))
        
    def _calculate_win_rate(self, trades):
        """Calculate win rate from trades"""
        if not trades:
            return 0
            
        wins = sum(1 for trade in trades if trade['pnl'] > 0)
        return (wins / len(trades)) * 100
        
    def _aggregate_results(self, strategy_results):
        """
        Aggregate results from multiple timeframes
        
        Args:
            strategy_results: List of backtest results for different timeframes
            
        Returns:
            dict: Aggregated performance metrics
        """
        if not strategy_results:
            return {
                "return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "trades_count": 0
            }
            
        # Calculate average metrics
        avg_return = np.mean([result['return'] for result in strategy_results])
        avg_sharpe = np.mean([result['sharpe_ratio'] for result in strategy_results])
        avg_drawdown = np.mean([result['max_drawdown'] for result in strategy_results])
        avg_win_rate = np.mean([result['win_rate'] for result in strategy_results])
        total_trades = sum(len(result['trades']) for result in strategy_results)
        
        return {
            "return": avg_return,
            "sharpe_ratio": avg_sharpe,
            "max_drawdown": avg_drawdown,
            "win_rate": avg_win_rate,
            "trades_count": total_trades
        }
        
    def _save_backtest_result(self, result):
        """
        Save backtest result to database
        
        Args:
            result: Backtest result to save
        """
        # In a real implementation, this would save to a database
        # For now, we'll just store in memory
        result_id = str(uuid.uuid4())
        self.database[result_id] = result
        logger.info(f"Saved backtest result with ID: {result_id}") 