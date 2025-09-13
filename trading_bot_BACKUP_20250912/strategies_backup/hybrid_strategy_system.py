"""
Hybrid Strategy System

This module implements a strategy hybridization system that combines signals
from multiple strategies using weighted voting for more robust decision making.
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Import our strategy implementations
try:
    from trading_bot.strategies.strategy_factory import StrategyFactory
    from trading_bot.strategies.weighted_avg_peak import WeightedAvgPeakStrategy
    from trading_bot.ml_pipeline.autonomous_integration import AutonomousIntegration
    ML_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Error importing ML pipeline: {e}")
    ML_AVAILABLE = False


class HybridStrategySystem:
    """
    Hybrid Strategy System
    
    Combines signals from multiple strategy types:
    1. Traditional technical strategies
    2. ML-based strategies (if available)
    3. WeightedAvgPeak strategy
    
    Using weighted voting for more robust trading decisions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the hybrid strategy system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.strategy_factory = StrategyFactory()
        
        # Initialize strategy weights
        self.strategy_weights = self.config.get('strategy_weights', {
            'technical': 0.4,
            'ml': 0.4,
            'weighted_avg_peak': 0.2
        })
        
        # Initialize market awareness flag
        self.market_hours_aware = self.config.get('market_hours_aware', True)
        
        # Initialize ML integration if available
        self.ml_integration = None
        if ML_AVAILABLE:
            try:
                self.ml_integration = AutonomousIntegration(config=self.config.get('ml_config', {}))
                logger.info("ML integration initialized for hybrid strategy system")
            except Exception as e:
                logger.error(f"Error initializing ML integration: {e}")
        
        # Initialize WeightedAvgPeak strategy
        self.weighted_avg_peak = WeightedAvgPeakStrategy(
            parameters=self.config.get('weighted_avg_peak_params', {})
        )
        
        # Initialize traditional strategies
        self.technical_strategies = {}
        self._initialize_technical_strategies()
        
        # Initialize strategy history for tracking
        self.strategy_history = []
        
        logger.info("Hybrid strategy system initialized")
    
    def _initialize_technical_strategies(self):
        """Initialize traditional technical strategies"""
        strategy_types = self.config.get('technical_strategies', [
            'momentum', 'mean_reversion', 'trend_following', 'volatility_breakout'
        ])
        
        for strategy_type in strategy_types:
            try:
                self.technical_strategies[strategy_type] = self.strategy_factory.create_strategy(
                    strategy_type,
                    config=self.config.get(f'{strategy_type}_params', {})
                )
                logger.info(f"Initialized {strategy_type} strategy")
            except Exception as e:
                logger.error(f"Error initializing {strategy_type} strategy: {e}")
    
    def generate_signals(self, data, ticker=None, timeframe='1d', **kwargs) -> Dict[str, Any]:
        """
        Generate trading signals using the hybrid approach
        
        Args:
            data: DataFrame with price data
            ticker: Symbol being analyzed
            timeframe: Data timeframe
            **kwargs: Additional parameters
            
        Returns:
            Dict: Combined signal with action, confidence and supporting data
        """
        start_time = datetime.now()
        
        # Track signal generation for analysis
        signal_tracking = {
            'ticker': ticker,
            'timeframe': timeframe,
            'timestamp': start_time,
            'signals': {},
            'combined_signal': None
        }
        
        # Generate signals from each strategy type
        technical_signals = self._generate_technical_signals(data)
        ml_signals = self._generate_ml_signals(data, ticker, timeframe) if self.ml_integration else {}
        weighted_avg_signal = self._generate_weighted_avg_signal(data)
        
        # Record all signals for tracking
        signal_tracking['signals'] = {
            'technical': technical_signals,
            'ml': ml_signals,
            'weighted_avg_peak': weighted_avg_signal
        }
        
        # Combine signals using weighted voting
        combined_signal = self._combine_signals(
            technical_signals, ml_signals, weighted_avg_signal
        )
        
        # Add execution metadata
        combined_signal['timestamp'] = datetime.now()
        combined_signal['execution_time_ms'] = (datetime.now() - start_time).total_seconds() * 1000
        combined_signal['ticker'] = ticker
        combined_signal['timeframe'] = timeframe
        
        # Record the combined signal
        signal_tracking['combined_signal'] = combined_signal
        
        # Save to history
        self.strategy_history.append(signal_tracking)
        if len(self.strategy_history) > 100:
            self.strategy_history = self.strategy_history[-100:]
        
        logger.info(f"Generated {combined_signal['action']} signal for {ticker} with confidence {combined_signal['confidence']:.2f}")
        return combined_signal
    
    def _generate_technical_signals(self, data) -> Dict[str, Dict[str, Any]]:
        """
        Generate signals from traditional technical strategies
        
        Args:
            data: Price DataFrame
            
        Returns:
            Dict: Signals from each technical strategy
        """
        signals = {}
        
        for strategy_name, strategy in self.technical_strategies.items():
            try:
                signal = strategy.generate_signals(data)
                signals[strategy_name] = signal
            except Exception as e:
                logger.error(f"Error generating signal from {strategy_name}: {e}")
                signals[strategy_name] = {'action': 'hold', 'confidence': 0.0, 'error': str(e)}
        
        return signals
    
    def _generate_ml_signals(self, data, ticker, timeframe) -> Dict[str, Any]:
        """
        Generate signals from ML-based strategies
        
        Args:
            data: Price DataFrame
            ticker: Symbol
            timeframe: Data timeframe
            
        Returns:
            Dict: ML signal
        """
        if not self.ml_integration:
            return {'action': 'hold', 'confidence': 0.0, 'reason': 'ML not available'}
        
        try:
            # Generate ML signal
            ml_signal = self.ml_integration._generate_combined_signals(ticker, data, timeframe)
            
            # Convert from buy/sell/neutral to buy/sell/hold for consistency
            action = ml_signal.get('signal', 'neutral')
            if action == 'neutral':
                action = 'hold'
                
            return {
                'action': action,
                'confidence': ml_signal.get('confidence', 0.0),
                'reason': ml_signal.get('reason', ''),
                'risk_params': ml_signal.get('risk_params', {})
            }
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'error': str(e)}
    
    def _generate_weighted_avg_signal(self, data) -> Dict[str, Any]:
        """
        Generate signal from WeightedAvgPeak strategy
        
        Args:
            data: Price DataFrame
            
        Returns:
            Dict: WeightedAvgPeak signal
        """
        try:
            return self.weighted_avg_peak.generate_signals(data)
        except Exception as e:
            logger.error(f"Error generating WeightedAvgPeak signal: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'error': str(e)}
    
    def _combine_signals(self, technical_signals, ml_signal, weighted_avg_signal) -> Dict[str, Any]:
        """
        Combine signals from all strategies using weighted voting
        
        Args:
            technical_signals: Signals from technical strategies
            ml_signal: Signal from ML-based strategy
            weighted_avg_signal: Signal from WeightedAvgPeak strategy
            
        Returns:
            Dict: Combined signal
        """
        # Initialize vote counters
        buy_votes = 0.0
        sell_votes = 0.0
        hold_votes = 0.0
        total_votes = 0.0
        
        # Calculate votes from technical strategies
        tech_weight = self.strategy_weights.get('technical', 0.4)
        tech_count = len(technical_signals) if technical_signals else 1
        individual_tech_weight = tech_weight / tech_count
        
        for strategy_name, signal in technical_signals.items():
            action = signal.get('action', 'hold').lower()
            confidence = signal.get('confidence', 0.5)
            
            # Weight by confidence and strategy importance
            weighted_vote = confidence * individual_tech_weight
            
            if action == 'buy':
                buy_votes += weighted_vote
            elif action == 'sell':
                sell_votes += weighted_vote
            else:
                hold_votes += weighted_vote
                
            total_votes += individual_tech_weight
        
        # Add votes from ML strategy
        if ml_signal:
            ml_weight = self.strategy_weights.get('ml', 0.4)
            ml_action = ml_signal.get('action', 'hold').lower()
            ml_confidence = ml_signal.get('confidence', 0.5)
            
            weighted_vote = ml_confidence * ml_weight
            
            if ml_action == 'buy':
                buy_votes += weighted_vote
            elif ml_action == 'sell':
                sell_votes += weighted_vote
            else:
                hold_votes += weighted_vote
                
            total_votes += ml_weight
        
        # Add votes from WeightedAvgPeak strategy
        if weighted_avg_signal:
            wap_weight = self.strategy_weights.get('weighted_avg_peak', 0.2)
            wap_action = weighted_avg_signal.get('action', 'hold').lower()
            wap_confidence = weighted_avg_signal.get('confidence', 0.5)
            
            weighted_vote = wap_confidence * wap_weight
            
            if wap_action == 'buy':
                buy_votes += weighted_vote
            elif wap_action == 'sell':
                sell_votes += weighted_vote
            else:
                hold_votes += weighted_vote
                
            total_votes += wap_weight
        
        # Normalize votes
        if total_votes > 0:
            buy_votes = buy_votes / total_votes
            sell_votes = sell_votes / total_votes
            hold_votes = hold_votes / total_votes
        
        # Determine final action
        action = 'hold'
        confidence = hold_votes
        
        if buy_votes > sell_votes and buy_votes > hold_votes:
            action = 'buy'
            confidence = buy_votes
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            action = 'sell'
            confidence = sell_votes
        
        # Generate explanations
        explanation = []
        if technical_signals:
            tech_explanations = [
                f"{name.capitalize()}: {signal.get('action')} ({signal.get('confidence', 0):.2f})"
                for name, signal in technical_signals.items()
            ]
            explanation.append(f"Technical strategies: {', '.join(tech_explanations)}")
            
        if ml_signal:
            explanation.append(
                f"ML strategy: {ml_signal.get('action')} ({ml_signal.get('confidence', 0):.2f})"
            )
            
        if weighted_avg_signal:
            explanation.append(
                f"WeightedAvgPeak: {weighted_avg_signal.get('action')} ({weighted_avg_signal.get('confidence', 0):.2f})"
            )
        
        # Combine risk parameters from strategies
        risk_params = {}
        
        # WeightedAvgPeak risk parameters
        if weighted_avg_signal and weighted_avg_signal.get('action') != 'hold':
            risk_params['stop'] = weighted_avg_signal.get('stop')
            risk_params['target'] = weighted_avg_signal.get('target')
        
        # ML risk parameters (take precedence if available)
        if ml_signal and ml_signal.get('risk_params'):
            ml_risk = ml_signal.get('risk_params', {})
            if 'stop_loss_pct' in ml_risk:
                risk_params['stop_loss_pct'] = ml_risk['stop_loss_pct']
            if 'take_profit_pct' in ml_risk:
                risk_params['take_profit_pct'] = ml_risk['take_profit_pct']
        
        # Create the final combined signal
        combined_signal = {
            'action': action,
            'confidence': confidence,
            'votes': {
                'buy': buy_votes,
                'sell': sell_votes,
                'hold': hold_votes
            },
            'explanation': " • ".join(explanation),
            'risk_params': risk_params
        }
        
        return combined_signal
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get performance metrics for each strategy
        
        Returns:
            Dict: Performance metrics for each strategy
        """
        if not self.strategy_history:
            return {}
            
        performance = {
            'technical': {},
            'ml': {},
            'weighted_avg_peak': {},
            'combined': {}
        }
        
        # Calculate accuracy and other metrics
        for strategy_type in performance:
            correct_calls = 0
            total_calls = 0
            buy_signals = 0
            sell_signals = 0
            hold_signals = 0
            
            for entry in self.strategy_history:
                if strategy_type == 'combined':
                    signal = entry.get('combined_signal', {})
                    if signal:
                        action = signal.get('action', 'hold').lower()
                        
                        if action == 'buy':
                            buy_signals += 1
                        elif action == 'sell':
                            sell_signals += 1
                        else:
                            hold_signals += 1
                            
                        total_calls += 1
                else:
                    signals = entry.get('signals', {}).get(strategy_type, {})
                    if not signals:
                        continue
                        
                    if isinstance(signals, dict) and 'action' in signals:
                        # Single signal
                        action = signals.get('action', 'hold').lower()
                        
                        if action == 'buy':
                            buy_signals += 1
                        elif action == 'sell':
                            sell_signals += 1
                        else:
                            hold_signals += 1
                            
                        total_calls += 1
                    elif isinstance(signals, dict):
                        # Multiple signals
                        for name, signal in signals.items():
                            if isinstance(signal, dict) and 'action' in signal:
                                action = signal.get('action', 'hold').lower()
                                
                                if action == 'buy':
                                    buy_signals += 1
                                elif action == 'sell':
                                    sell_signals += 1
                                else:
                                    hold_signals += 1
                                    
                                total_calls += 1
            
            performance[strategy_type] = {
                'total_signals': total_calls,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'buy_pct': buy_signals / total_calls if total_calls > 0 else 0,
                'sell_pct': sell_signals / total_calls if total_calls > 0 else 0,
                'hold_pct': hold_signals / total_calls if total_calls > 0 else 0
            }
        
        return performance
    
    def export_strategy_history(self, filepath: str):
        """
        Export strategy history to file
        
        Args:
            filepath: Path to export file
        """
        # Clean up datetime objects for JSON serialization
        export_data = []
        
        for entry in self.strategy_history:
            export_entry = {
                'ticker': entry.get('ticker'),
                'timeframe': entry.get('timeframe'),
                'timestamp': entry.get('timestamp').isoformat() if entry.get('timestamp') else None,
                'signals': entry.get('signals'),
                'combined_signal': entry.get('combined_signal')
            }
            
            # Clean up timestamp in combined signal
            if export_entry['combined_signal'] and 'timestamp' in export_entry['combined_signal']:
                export_entry['combined_signal']['timestamp'] = (
                    export_entry['combined_signal']['timestamp'].isoformat()
                    if export_entry['combined_signal']['timestamp'] else None
                )
            
            export_data.append(export_entry)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Exported strategy history to {filepath}")
    
    def import_strategy_history(self, filepath: str):
        """
        Import strategy history from file
        
        Args:
            filepath: Path to import file
        """
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Convert string timestamps back to datetime
            for entry in import_data:
                if entry.get('timestamp'):
                    entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                
                if entry.get('combined_signal') and entry['combined_signal'].get('timestamp'):
                    entry['combined_signal']['timestamp'] = (
                        datetime.fromisoformat(entry['combined_signal']['timestamp'])
                    )
            
            self.strategy_history = import_data
            logger.info(f"Imported {len(import_data)} strategy history entries")
            
        except Exception as e:
            logger.error(f"Error importing strategy history: {e}")


# Factory function to create hybrid strategy system
def create_hybrid_strategy_system(config=None) -> HybridStrategySystem:
    """
    Create a hybrid strategy system with the provided configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HybridStrategySystem: Initialized hybrid strategy system
    """
    return HybridStrategySystem(config=config)
