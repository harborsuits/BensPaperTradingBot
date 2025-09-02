"""
Autonomous Integration System

This module integrates the ML pipeline with the existing trading bot
to create a fully autonomous trading system.
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

try:
    from trading_bot.strategies.strategy_factory import StrategyFactory
    from trading_bot.ml_pipeline.feature_engineering import FeatureEngineeringFramework
    from trading_bot.ml_pipeline.model_trainer import ModelTrainer
    from trading_bot.ml_pipeline.signal_generator import SignalGenerator
    from trading_bot.backtesting.autonomous_backtester import AutonomousBacktester, BacktestResultAnalyzer
    from trading_bot.data.market_data_provider import create_data_provider
    from trading_bot.brokers.trade_executor import TradeExecutor
    from trading_bot.portfolio_state import PortfolioStateManager
    from trading_bot.triggers.notification_connector import NotificationManager

    ML_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing ML pipeline components: {e}")
    ML_COMPONENTS_AVAILABLE = False


class AutonomousIntegration:
    """
    Integrates ML models with traditional strategies for autonomous trading
    
    This class serves as the central coordinator for the autonomous trading system,
    combining ML predictions with traditional strategy signals for enhanced decision making.
    """
    
    def __init__(self, config=None):
        """
        Initialize the autonomous integration system
        
        Args:
            config: Configuration dictionary for the system
        """
        self.config = config or {}
        self.ml_enabled = self.config.get('ml_enabled', True) and ML_COMPONENTS_AVAILABLE
        
        # Initialize components based on availability
        if not ML_COMPONENTS_AVAILABLE:
            logger.warning("ML components not available, using only traditional strategies")
            
        # Initialize feature engineering framework
        self.feature_engineering = FeatureEngineeringFramework(
            config=self.config.get('feature_engineering', {})
        ) if self.ml_enabled else None
        
        # Initialize model trainer (for offline training)
        self.model_trainer = ModelTrainer(
            config=self.config.get('model_trainer', {})
        ) if self.ml_enabled else None
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator(
            config=self.config.get('signal_generator', {}),
            confidence_threshold=self.config.get('confidence_threshold', 0.65)
        ) if self.ml_enabled else None
        
        # Initialize traditional strategy factory
        self.strategy_factory = StrategyFactory()
        
        # Initialize data provider
        self.data_provider = create_data_provider(
            provider_type=self.config.get('data_provider', 'alpha_vantage'),
            api_key=self.config.get('api_keys', {}).get('alpha_vantage', '')
        )
        
        # Initialize portfolio state manager
        self.portfolio_state = PortfolioStateManager()
        
        # Initialize trade executor
        try:
            self.trade_executor = TradeExecutor(
                broker=self.config.get('broker', 'tradier'),
                config=self.config.get('broker_config', {})
            )
        except Exception as e:
            logger.error(f"Error initializing trade executor: {e}")
            self.trade_executor = None
        
        # Initialize notification manager
        try:
            self.notification_manager = NotificationManager(
                telegram_token=self.config.get('telegram_token', ''),
                telegram_chat_id=self.config.get('telegram_chat_id', '')
            )
        except Exception as e:
            logger.error(f"Error initializing notification manager: {e}")
            self.notification_manager = None
            
        # Initialize backtester for strategy optimization
        try:
            self.result_analyzer = BacktestResultAnalyzer()
            self.backtester = AutonomousBacktester(
                data_layer=self.data_provider,
                strategy_generator=None,  # Will be set in run_backtest_cycle
                result_analyzer=self.result_analyzer
            )
        except Exception as e:
            logger.error(f"Error initializing backtester: {e}")
            self.backtester = None
            
        # Load ML models if available
        self.models = {}
        if self.ml_enabled:
            self._load_models()
            
        logger.info(f"Autonomous Integration initialized with ML {'' if self.ml_enabled else 'disabled'}")
    
    def _load_models(self):
        """Load trained ML models from the models directory"""
        if not self.ml_enabled or not self.model_trainer:
            return
            
        models_dir = self.model_trainer.models_dir
        if not os.path.exists(models_dir):
            logger.warning(f"Models directory not found: {models_dir}")
            return
            
        # Find model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        # Load each model
        for model_file in model_files:
            try:
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(models_dir, model_file)
                
                # Check for metadata file
                metadata_file = f"{model_name}_metadata.json"
                metadata_path = os.path.join(models_dir, metadata_file)
                metadata = {}
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                # Load model
                model = self.model_trainer.load_model(model_path)
                
                # Add to signal generator
                if self.signal_generator:
                    self.signal_generator.add_model(model_name, model, metadata)
                    
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {e}")
    
    def run_autonomous_cycle(self, tickers=None, timeframe='1d'):
        """
        Run a complete autonomous trading cycle
        
        This includes:
        1. Retrieving market data
        2. Generating features
        3. Getting ML signals
        4. Getting traditional strategy signals
        5. Combining signals
        6. Executing trades based on signals
        7. Sending notifications
        
        Args:
            tickers: List of ticker symbols to analyze
            timeframe: Data timeframe ('1d', '1h', etc.)
            
        Returns:
            Dictionary with results of the trading cycle
        """
        if not tickers:
            tickers = self.config.get('tickers', ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN'])
        
        cycle_start = datetime.now()
        logger.info(f"Starting autonomous trading cycle for {len(tickers)} tickers")
        
        # Initialize results dictionary
        results = {
            'cycle_start': cycle_start,
            'tickers': tickers,
            'timeframe': timeframe,
            'signals': {},
            'trades_executed': [],
            'errors': []
        }
        
        # Process each ticker
        for ticker in tickers:
            try:
                # Get market data
                market_data = self._get_market_data(ticker, timeframe)
                
                if market_data is None or market_data.empty:
                    results['errors'].append(f"No data for {ticker}")
                    continue
                
                # Generate features if ML is enabled
                if self.ml_enabled and self.feature_engineering:
                    market_data = self.feature_engineering.generate_features(market_data)
                
                # Get signals
                signals = self._generate_combined_signals(ticker, market_data, timeframe)
                results['signals'][ticker] = signals
                
                # Execute trades based on signals
                if signals.get('signal') in ['buy', 'sell'] and self.trade_executor:
                    trade_result = self._execute_trade(ticker, signals)
                    if trade_result:
                        results['trades_executed'].append(trade_result)
                        
                        # Send notification
                        if self.notification_manager:
                            self._send_signal_notification(ticker, signals, trade_result)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                results['errors'].append(f"Error processing {ticker}: {str(e)}")
        
        # Calculate cycle statistics
        cycle_end = datetime.now()
        results['cycle_end'] = cycle_end
        results['cycle_duration'] = (cycle_end - cycle_start).total_seconds()
        results['signals_generated'] = len([s for s in results['signals'].values() 
                                          if s.get('signal') in ['buy', 'sell']])
        results['trades_executed_count'] = len(results['trades_executed'])
        
        logger.info(f"Completed autonomous cycle in {results['cycle_duration']:.2f} seconds")
        logger.info(f"Generated {results['signals_generated']} signals, executed {results['trades_executed_count']} trades")
        
        return results
    
    def _get_market_data(self, ticker, timeframe='1d', days=100):
        """Get market data for a ticker"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = self.data_provider.get_historical_data(
                symbol=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {ticker}: {e}")
            return None
    
    def _generate_combined_signals(self, ticker, market_data, timeframe):
        """
        Generate combined signals from ML and traditional strategies
        
        This creates a weighted combination of signals from both approaches
        """
        # Initialize signal containers
        ml_signal = {
            'signal': 'neutral',
            'confidence': 0.0,
            'weight': 0.0
        }
        
        traditional_signal = {
            'signal': 'neutral',
            'confidence': 0.0,
            'weight': 0.0
        }
        
        # Get ML signal if enabled
        if self.ml_enabled and self.signal_generator:
            try:
                ml_signal = self.signal_generator.generate_signals(market_data, ticker, timeframe)
                ml_signal['weight'] = self.config.get('ml_weight', 0.6)  # Default 60% weight to ML
            except Exception as e:
                logger.error(f"Error generating ML signal for {ticker}: {e}")
        
        # Get traditional strategy signals
        try:
            # Use strategy factory to get signals from traditional strategies
            strategy_type = self.config.get('default_strategy', 'momentum')
            strategy = self.strategy_factory.create_strategy(strategy_type)
            
            # Generate signals
            trad_signal_data = strategy.generate_signals(market_data)
            
            # Convert to standard format
            action = trad_signal_data.get('action', 'hold')
            confidence = trad_signal_data.get('confidence', 0.5)
            
            traditional_signal = {
                'signal': 'buy' if action == 'buy' else 'sell' if action == 'sell' else 'neutral',
                'confidence': confidence,
                'weight': 1.0 - ml_signal['weight']  # Remaining weight
            }
            
        except Exception as e:
            logger.error(f"Error generating traditional signal for {ticker}: {e}")
        
        # Combine signals
        combined_signal = self._combine_signals(ml_signal, traditional_signal)
        
        # Add additional context
        combined_signal['ticker'] = ticker
        combined_signal['timestamp'] = datetime.now()
        combined_signal['timeframe'] = timeframe
        combined_signal['ml_signal'] = ml_signal.get('signal')
        combined_signal['ml_confidence'] = ml_signal.get('confidence')
        combined_signal['traditional_signal'] = traditional_signal.get('signal')
        combined_signal['traditional_confidence'] = traditional_signal.get('confidence')
        
        # Add risk parameters from the ML signal (if available)
        combined_signal['risk_params'] = ml_signal.get('risk_params', {})
        
        return combined_signal
    
    def _combine_signals(self, ml_signal, traditional_signal):
        """
        Combine ML and traditional signals using weighted approach
        
        Args:
            ml_signal: Signal from ML models
            traditional_signal: Signal from traditional strategies
            
        Returns:
            Combined signal
        """
        ml_direction = 1 if ml_signal.get('signal') == 'buy' else \
                     -1 if ml_signal.get('signal') == 'sell' else 0
                     
        trad_direction = 1 if traditional_signal.get('signal') == 'buy' else \
                        -1 if traditional_signal.get('signal') == 'sell' else 0
        
        # Calculate weighted direction
        ml_weight = ml_signal.get('weight', 0.6)
        trad_weight = traditional_signal.get('weight', 0.4)
        
        weighted_direction = (ml_direction * ml_weight * ml_signal.get('confidence', 0.5)) + \
                            (trad_direction * trad_weight * traditional_signal.get('confidence', 0.5))
        
        # Determine final signal
        if weighted_direction > 0.2:  # Positive threshold
            signal = 'buy'
            confidence = min(abs(weighted_direction), 1.0)
        elif weighted_direction < -0.2:  # Negative threshold
            signal = 'sell'
            confidence = min(abs(weighted_direction), 1.0)
        else:
            signal = 'neutral'
            confidence = 0.5
        
        # Calculate final position size based on confidence
        position_size = ml_signal.get('position_size', 0.0) * ml_weight + \
                       traditional_signal.get('position_size', 0.0) * trad_weight
            
        return {
            'signal': signal,
            'confidence': confidence,
            'position_size': position_size,
            'weighted_direction': weighted_direction
        }
    
    def _execute_trade(self, ticker, signal_data):
        """
        Execute a trade based on the signal
        
        Args:
            ticker: Ticker symbol
            signal_data: Signal data dictionary
            
        Returns:
            Trade execution result or None if no trade executed
        """
        if not self.trade_executor:
            logger.warning("Trade executor not available, skipping trade execution")
            return None
            
        # Extract signal information
        signal = signal_data.get('signal', 'neutral')
        confidence = signal_data.get('confidence', 0.0)
        position_size = signal_data.get('position_size', 0.0)
        
        # Skip if neutral or low confidence
        if signal == 'neutral' or confidence < 0.5:
            return None
            
        # Determine order parameters
        action = 'buy' if signal == 'buy' else 'sell'
        
        # Get risk parameters
        risk_params = signal_data.get('risk_params', {})
        stop_loss_pct = risk_params.get('stop_loss_pct', 0.0)
        take_profit_pct = risk_params.get('take_profit_pct', 0.0)
        
        # Get portfolio state
        portfolio = self.portfolio_state.get_portfolio_state()
        current_cash = portfolio.get('cash', 0.0)
        
        # Calculate position size in dollars
        if action == 'buy':
            # For buys, calculate based on position_size (% of portfolio)
            portfolio_value = portfolio.get('total_value', current_cash)
            dollar_amount = portfolio_value * position_size
            
            # Ensure we have enough cash
            if dollar_amount > current_cash:
                dollar_amount = current_cash * 0.95  # Leave some buffer
                
            # Minimum check
            if dollar_amount < 100:  # Minimum $100 per trade
                logger.info(f"Trade amount {dollar_amount:.2f} below minimum, skipping")
                return None
        else:
            # For sells, check if we have the position
            positions = portfolio.get('positions', {})
            position_value = positions.get(ticker, {}).get('value', 0.0)
            
            # Use position size percentage of existing position
            dollar_amount = position_value * position_size
            
            # Skip if no position or too small
            if position_value == 0 or dollar_amount < 100:
                logger.info(f"No position or too small for {ticker}, skipping sell")
                return None
        
        try:
            # Execute the trade
            result = self.trade_executor.execute_trade(
                symbol=ticker,
                action=action,
                quantity=None,  # Will be calculated from dollar_amount
                order_type='market',
                dollar_amount=dollar_amount,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct
            )
            
            # Log result
            if result.get('success'):
                logger.info(f"Executed {action} for {ticker}: {result}")
            else:
                logger.warning(f"Trade execution failed for {ticker}: {result}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error executing trade for {ticker}: {e}")
            return {
                'success': False,
                'error': str(e),
                'ticker': ticker,
                'action': action,
                'attempted_amount': dollar_amount
            }
    
    def _send_signal_notification(self, ticker, signal_data, trade_result=None):
        """Send notification about signal and trade execution"""
        if not self.notification_manager:
            return
            
        signal = signal_data.get('signal', 'neutral')
        confidence = signal_data.get('confidence', 0.0)
        
        # Skip notification for neutral signals
        if signal == 'neutral':
            return
            
        # Build message
        if signal == 'buy':
            icon = "🟢"
            action = "BUY"
        else:
            icon = "🔴"
            action = "SELL"
            
        message = f"{icon} {action}: {ticker}\n"
        message += f"Confidence: {confidence:.2f}\n"
        
        # Add price info if available
        if 'last_price' in signal_data:
            message += f"Price: ${signal_data['last_price']:.2f}\n"
            
        # Add signal source information
        message += "\nSignal sources:\n"
        message += f"ML: {signal_data.get('ml_signal', 'N/A')} ({signal_data.get('ml_confidence', 0):.2f})\n"
        message += f"Traditional: {signal_data.get('traditional_signal', 'N/A')} ({signal_data.get('traditional_confidence', 0):.2f})\n"
        
        # Add risk parameters
        risk_params = signal_data.get('risk_params', {})
        if risk_params:
            message += "\nRisk parameters:\n"
            message += f"Stop Loss: {risk_params.get('stop_loss_pct', 0):.2f}%\n"
            message += f"Take Profit: {risk_params.get('take_profit_pct', 0):.2f}%\n"
            message += f"Risk-Reward: {risk_params.get('risk_reward_ratio', 0):.2f}\n"
            
        # Add trade execution info if available
        if trade_result:
            message += "\nTrade executed:\n"
            message += f"Status: {'Success' if trade_result.get('success') else 'Failed'}\n"
            if 'filled_quantity' in trade_result:
                message += f"Quantity: {trade_result.get('filled_quantity')}\n"
            if 'filled_price' in trade_result:
                message += f"Price: ${trade_result.get('filled_price'):.2f}\n"
            if 'total_amount' in trade_result:
                message += f"Amount: ${trade_result.get('total_amount'):.2f}\n"
                
        # Send notification
        try:
            self.notification_manager.send_notification(message)
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def run_backtest_cycle(self, tickers=None, timeframes=None, optimize_models=False):
        """
        Run a backtesting cycle to evaluate and improve strategies
        
        Args:
            tickers: List of tickers to backtest (or None to auto-select)
            timeframes: List of timeframes to test
            optimize_models: Whether to optimize ML models
            
        Returns:
            Dictionary with backtest results and insights
        """
        if not self.backtester:
            logger.warning("Backtester not available")
            return {'error': 'Backtester not available'}
            
        logger.info("Starting autonomous backtest cycle")
        
        if not tickers:
            tickers = self.config.get('backtest_tickers', 
                                    ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN'])
            
        if not timeframes:
            timeframes = self.config.get('backtest_timeframes', ['1d'])
            
        try:
            # Run backtest cycle
            results = self.backtester.run_full_autonomous_cycle(
                tickers=tickers,
                timeframes=timeframes
            )
            
            # Analyze results to improve strategies
            insights = self.result_analyzer.analyze_results(
                results.get('winning_strategies', []),
                results.get('losing_strategies', [])
            )
            
            # Optimize ML models if requested
            if optimize_models and self.ml_enabled and self.model_trainer:
                self._optimize_ml_models(tickers, insights)
                
            logger.info(f"Completed backtest cycle with {len(results.get('winning_strategies', []))} winning strategies")
            
            return {
                'results': results,
                'insights': insights,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error running backtest cycle: {e}")
            return {'error': str(e)}
    
    def _optimize_ml_models(self, tickers, insights):
        """
        Optimize ML models based on backtest insights
        
        Args:
            tickers: Tickers used in backtest
            insights: Insights from backtest results
        """
        if not self.ml_enabled or not self.model_trainer:
            return
            
        logger.info("Starting ML model optimization")
        
        try:
            # Get market data for all tickers
            all_data = {}
            for ticker in tickers:
                data = self._get_market_data(ticker, '1d', days=365)
                if data is not None and not data.empty:
                    # Generate features
                    data = self.feature_engineering.generate_features(data)
                    all_data[ticker] = data
            
            # Generate training labels
            for ticker, data in all_data.items():
                # Generate directional labels
                data['label_direction'] = self.model_trainer.generate_labels(
                    data, label_type='directional', horizon=5, threshold=0.0
                )
                
                # Generate volatility labels
                data['label_volatility'] = self.model_trainer.generate_labels(
                    data, label_type='volatility', horizon=5
                )
                
            # Combine data from all tickers
            combined_data = pd.concat(all_data.values())
            combined_data = combined_data.dropna()
            
            # Train directional model
            features = [col for col in combined_data.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'date', 
                                     'label_direction', 'label_volatility']]
            
            X_train, X_test, y_train, y_test, _ = self.model_trainer.prepare_training_data(
                combined_data, 'label_direction', features
            )
            
            # Optimize hyperparameters
            best_params, best_model = self.model_trainer.optimize_hyperparameters(
                X_train, y_train, model_type='gradient_boosting'
            )
            
            # Evaluate model
            metrics = self.model_trainer.evaluate_model(best_model, X_test, y_test)
            
            # Save model if performance is good
            if metrics['accuracy'] > 0.55:  # Better than random
                model_name = 'directional_model'
                self.model_trainer.save_model(
                    best_model, 
                    model_name, 
                    {
                        'features': features,
                        'metrics': metrics,
                        'params': best_params,
                        'backtest_insights': insights,
                        'training_date': datetime.now().isoformat()
                    }
                )
                
                # Add to signal generator
                if self.signal_generator:
                    self.signal_generator.add_model(
                        model_name, 
                        best_model, 
                        {
                            'features': features,
                            'metrics': metrics,
                            'weight': 1.0  # Full weight for newest model
                        }
                    )
                
                logger.info(f"Optimized and saved new directional model with accuracy {metrics['accuracy']:.4f}")
            else:
                logger.info(f"Model performance below threshold: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error optimizing ML models: {e}")
    
    def generate_dashboard_insights(self):
        """
        Generate insights for the dashboard
        
        Returns:
            Dictionary with insights data
        """
        insights = {
            'ml_enabled': self.ml_enabled,
            'timestamp': datetime.now(),
            'trading_status': 'active' if self.trade_executor else 'inactive',
            'models': [],
            'recent_signals': [],
            'signal_performance': {},
            'portfolio_impact': {}
        }
        
        # Add model information
        if self.ml_enabled and self.signal_generator:
            for model_name, model_info in self.signal_generator.models.items():
                model_metadata = model_info.get('metadata', {})
                
                insights['models'].append({
                    'name': model_name,
                    'type': model_name.split('_')[0] if '_' in model_name else 'unknown',
                    'accuracy': model_metadata.get('metrics', {}).get('accuracy', 0.0),
                    'feature_count': len(model_metadata.get('features', [])),
                    'training_date': model_metadata.get('training_date', 'unknown')
                })
                
            # Get recent signals
            insights['recent_signals'] = self.signal_generator.get_signal_history(days_back=7)
            
            # Get signal performance metrics
            insights['signal_performance'] = self.signal_generator.get_signal_performance(lookback_days=30)
        
        # Get portfolio impact of ML signals
        if self.portfolio_state:
            portfolio = self.portfolio_state.get_portfolio_state()
            ml_positions = [pos for pos in portfolio.get('positions', {}).values()
                          if pos.get('source') == 'ml']
            
            # Calculate performance metrics for ML positions
            if ml_positions:
                total_value = sum(pos.get('value', 0) for pos in ml_positions)
                total_gain_loss = sum(pos.get('unrealized_gain_loss', 0) for pos in ml_positions)
                
                insights['portfolio_impact'] = {
                    'position_count': len(ml_positions),
                    'total_value': total_value,
                    'avg_gain_loss_pct': total_gain_loss / total_value if total_value > 0 else 0.0,
                    'best_position': max(ml_positions, key=lambda p: p.get('unrealized_gain_loss_pct', 0))
                                    if ml_positions else {},
                    'worst_position': min(ml_positions, key=lambda p: p.get('unrealized_gain_loss_pct', 0))
                                    if ml_positions else {}
                }
        
        return insights
