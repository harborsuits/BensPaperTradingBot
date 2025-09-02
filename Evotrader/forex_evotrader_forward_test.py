#!/usr/bin/env python3
"""
Forex EvoTrader Forward Test - Session-Aware Forward Testing Module

This module handles paper trading forward testing of forex strategies with:
- Session awareness (London, New York, Asia, Sydney)
- News event filtering
- Prop firm compliance monitoring
- Pip-based performance tracking
- BenBot integration
"""

import os
import sys
import yaml
import json
import logging
import datetime
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('forex_evotrader_forward_test')


class ForexStrategyForwardTest:
    """
    Handles forward testing of forex strategies with session awareness and prop firm compliance.
    """
    
    def __init__(self, forex_evotrader, config: Dict[str, Any] = None):
        """
        Initialize the forex strategy forward test module.
        
        Args:
            forex_evotrader: Parent ForexEvoTrader instance
            config: Forward test configuration (if None, uses parent config)
        """
        self.forex_evotrader = forex_evotrader
        
        # Use parent config if none provided
        if config is None and hasattr(forex_evotrader, 'config'):
            config = forex_evotrader.config
        
        self.config = config or {}
        self.forward_test_config = self.config.get('forward_test', {})
        
        # Access parent components
        self.pair_manager = getattr(forex_evotrader, 'pair_manager', None)
        self.news_guard = getattr(forex_evotrader, 'news_guard', None)
        self.session_manager = getattr(forex_evotrader, 'session_manager', None)
        self.pip_logger = getattr(forex_evotrader, 'pip_logger', None)
        self.performance_tracker = getattr(forex_evotrader, 'performance_tracker', None)
        
        # Load prop firm compliance rules
        self.prop_compliance_rules = self._load_prop_compliance_rules()
        
        # Initialize test state variables
        self.test_state = {
            'active': False,
            'strategy_id': None,
            'pair': None,
            'timeframe': None,
            'strategy_params': None,
            'strategy_type': None,
            'start_time': None,
            'end_time': None,
            'current_time': None,
            'equity': 100000.0,  # Default starting equity
            'initial_equity': 100000.0,
            'max_equity': 100000.0,
            'position': None,
            'trade_history': [],
            'daily_pnl': {},
            'compliance_status': {
                'compliant': True,
                'max_drawdown_reached': False,
                'daily_loss_limit_reached': False,
                'profit_target_reached': False
            },
            'session_trades': {},
            'last_benbot_report_time': None
        }
        
        # Set update interval (in seconds)
        self.update_interval = self.forward_test_config.get('update_interval_seconds', 60)
        self.benbot_report_interval = self.forward_test_config.get('benbot_report_interval_minutes', 30) * 60
        
        logger.info("Forex Strategy Forward Test module initialized")
    
    def _load_prop_compliance_rules(self) -> Dict[str, Any]:
        """Load prop firm compliance rules from configuration."""
        prop_config = self.config.get('prop_compliance', {})
        
        # Default rules
        default_rules = {
            'max_drawdown_percent': 5.0,
            'daily_loss_limit_percent': 3.0,
            'target_profit_percent': 8.0,
            'stop_trading_at_threshold': True,
            'drawdown_warning_threshold': 4.0,
            'daily_loss_warning_threshold': 2.0
        }
        
        # Merge with config
        rules = {**default_rules, **prop_config}
        
        # Load from prop risk profile if available
        risk_profile_path = self.config.get('prop_risk_profile', 'prop_risk_profile.yaml')
        if os.path.exists(risk_profile_path):
            try:
                with open(risk_profile_path, 'r') as f:
                    risk_profile = yaml.safe_load(f)
                
                # Update rules from risk profile
                if 'compliance' in risk_profile:
                    for key, value in risk_profile['compliance'].items():
                        rules[key] = value
                        
                logger.info(f"Loaded compliance rules from {risk_profile_path}")
            except Exception as e:
                logger.error(f"Error loading prop risk profile: {e}")
        
        return rules
    
    def load_strategy(self, strategy_id: str) -> bool:
        """
        Load strategy parameters by ID.
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            True if strategy loaded successfully, False otherwise
        """
        strategy_params, strategy_type, pair, timeframe = None, None, None, None
        
        # Check if strategy exists in results directory
        results_dir = self.config.get('results_dir', 'results')
        
        # Try to find matching files
        strategy_files = []
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if strategy_id in filename and filename.endswith('.json'):
                    strategy_files.append(os.path.join(results_dir, filename))
        
        if strategy_files:
            # Sort by modification time (newest first)
            strategy_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            try:
                with open(strategy_files[0], 'r') as f:
                    strategy_data = json.load(f)
                
                if 'best_strategy' in strategy_data and 'parameters' in strategy_data['best_strategy']:
                    logger.info(f"Loaded strategy {strategy_id} from {strategy_files[0]}")
                    strategy_params = strategy_data['best_strategy']['parameters']
                    strategy_type = strategy_data.get('template', 'unknown')
                    pair = strategy_data.get('pair', 'EURUSD')
                    timeframe = strategy_data.get('timeframe', '1h')
                    
            except Exception as e:
                logger.error(f"Error loading strategy file: {e}")
        
        # If not found in files, check performance tracker
        if not strategy_params and self.performance_tracker:
            metadata = self.performance_tracker.db.get_strategy_metadata(strategy_id)
            if metadata and 'params' in metadata:
                logger.info(f"Loaded strategy {strategy_id} from performance tracker")
                strategy_params = metadata['params']
                strategy_type = metadata.get('strategy_type', 'unknown')
                pair = metadata.get('pair', 'EURUSD')
                timeframe = metadata.get('timeframe', '1h')
        
        if strategy_params:
            # Update test state
            self.test_state['strategy_id'] = strategy_id
            self.test_state['strategy_params'] = strategy_params
            self.test_state['strategy_type'] = strategy_type
            self.test_state['pair'] = pair
            self.test_state['timeframe'] = timeframe
            return True
        
        logger.error(f"Strategy not found: {strategy_id}")
        return False
    
    def start_forward_test(self, 
                         strategy_id: str, 
                         days: int = 30, 
                         session_only: bool = False,
                         enable_news_guard: bool = True,
                         start_equity: float = 100000.0) -> bool:
        """
        Start forward testing a forex strategy.
        
        Args:
            strategy_id: Strategy ID to forward test
            days: Number of days to forward test
            session_only: Only trade during optimal sessions
            enable_news_guard: Enable news event filtering
            start_equity: Starting equity amount
            
        Returns:
            True if test started successfully, False otherwise
        """
        # Load strategy
        if not self.load_strategy(strategy_id):
            return False
        
        # Set test parameters
        self.test_state['active'] = True
        self.test_state['start_time'] = datetime.datetime.now()
        self.test_state['end_time'] = self.test_state['start_time'] + datetime.timedelta(days=days)
        self.test_state['current_time'] = self.test_state['start_time']
        self.test_state['equity'] = start_equity
        self.test_state['initial_equity'] = start_equity
        self.test_state['max_equity'] = start_equity
        self.test_state['session_only'] = session_only
        self.test_state['enable_news_guard'] = enable_news_guard
        self.test_state['trade_history'] = []
        self.test_state['daily_pnl'] = {}
        self.test_state['last_benbot_report_time'] = self.test_state['start_time']
        
        # Reset compliance status
        self.test_state['compliance_status'] = {
            'compliant': True,
            'max_drawdown_reached': False,
            'daily_loss_limit_reached': False,
            'profit_target_reached': False
        }
        
        # Initialize session trades tracking
        self.test_state['session_trades'] = {
            'London': [],
            'NewYork': [],
            'Asia': [],
            'Sydney': []
        }
        
        # Log start
        logger.info(f"Starting forward test for strategy {strategy_id}")
        logger.info(f"Pair: {self.test_state['pair']}, Timeframe: {self.test_state['timeframe']}")
        logger.info(f"Test period: {self.test_state['start_time']} to {self.test_state['end_time']}")
        logger.info(f"Session only: {session_only}, News guard: {enable_news_guard}")
        
        # Consult BenBot if available
        if hasattr(self.forex_evotrader, 'benbot_available') and self.forex_evotrader.benbot_available:
            benbot_feedback = self.forex_evotrader._consult_benbot('forward_test_start', {
                'strategy_id': strategy_id,
                'pair': self.test_state['pair'],
                'timeframe': self.test_state['timeframe'],
                'start_time': self.test_state['start_time'].isoformat(),
                'end_time': self.test_state['end_time'].isoformat(),
                'session_only': session_only,
                'enable_news_guard': enable_news_guard
            })
            
            if not benbot_feedback.get('proceed', True):
                logger.warning(f"BenBot rejected forward test: {benbot_feedback.get('reason', 'No reason')}")
                self.test_state['active'] = False
                return False
            
            logger.info(f"BenBot approved forward test: {benbot_feedback.get('message', 'No message')}")
        
        return True
    
    def run_forward_test(self) -> Dict[str, Any]:
        """
        Run the forward test from start to finish.
        
        Returns:
            Dictionary with test results
        """
        if not self.test_state['active']:
            logger.error("No active forward test. Call start_forward_test first.")
            return {'error': "No active forward test"}
        
        # Main test loop
        try:
            while self.test_state['active'] and self.test_state['current_time'] <= self.test_state['end_time']:
                # Process current time step
                self._process_time_step()
                
                # Check if we should stop due to compliance issues
                if self._check_stop_conditions():
                    break
                
                # Advance time
                self.test_state['current_time'] += datetime.timedelta(seconds=self.update_interval)
                
                # Report to BenBot periodically
                if (self.test_state['current_time'] - self.test_state['last_benbot_report_time']).total_seconds() >= self.benbot_report_interval:
                    self._report_to_benbot()
                    self.test_state['last_benbot_report_time'] = self.test_state['current_time']
        
        except KeyboardInterrupt:
            logger.info("Forward test interrupted by user")
        except Exception as e:
            logger.error(f"Error in forward test: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Compile results
        results = self._compile_test_results()
        
        # Final report to BenBot
        if hasattr(self.forex_evotrader, 'benbot_available') and self.forex_evotrader.benbot_available:
            self.forex_evotrader._consult_benbot('forward_test_complete', results)
        
        # Cleanup
        self.test_state['active'] = False
        
        return results
        
    def _process_time_step(self) -> None:
        """
        Process the current time step in the forward test.
        """
        current_time = self.test_state['current_time']
        pair = self.test_state['pair']
        
        # Get the current session
        active_sessions = []
        if self.session_manager:
            active_sessions = self.session_manager.get_active_sessions(current_time)
        
        # Check if we should trade based on session
        session_allowed = True
        if self.test_state.get('session_only', False) and active_sessions:
            # Check if current session is optimal for this strategy/pair
            session_allowed, reason = self.forex_evotrader.check_session_optimal(
                pair, self.test_state['strategy_id'], current_time
            )
            
            if not session_allowed:
                logger.info(f"Skipping time step due to session: {reason}")
                return
        
        # Check news events if enabled
        news_allowed = True
        if self.test_state.get('enable_news_guard', True) and self.news_guard:
            news_allowed, reason = self.forex_evotrader.check_news_safe(pair, current_time)
            
            if not news_allowed:
                logger.info(f"Skipping time step due to news: {reason}")
                return
        
        # Check prop firm compliance
        if not self.test_state['compliance_status']['compliant']:
            logger.info("Skipping time step due to compliance issues")
            return
        
        # Get current market data (simulated for forward test)
        market_data = self._get_simulated_market_data()
        
        # Generate trading signal
        signal = self._generate_trading_signal(market_data)
        
        # Consult BenBot for signal override if available
        if hasattr(self.forex_evotrader, 'benbot_available') and self.forex_evotrader.benbot_available:
            benbot_response = self.forex_evotrader._consult_benbot('trade_signal', {
                'strategy_id': self.test_state['strategy_id'],
                'pair': pair,
                'time': current_time.isoformat(),
                'signal': signal,
                'active_sessions': active_sessions,
                'current_position': self.test_state['position'],
                'equity': self.test_state['equity'],
                'drawdown': self._calculate_current_drawdown()
            })
            
            # Apply BenBot override if provided
            if 'override_signal' in benbot_response:
                logger.info(f"BenBot override: {signal} -> {benbot_response['override_signal']}")
                signal = benbot_response['override_signal']
        
        # Execute the signal
        if signal != 0:  # Non-zero signal means take action
            self._execute_trade(signal, market_data, active_sessions)
        
        # Update equity with open position P&L
        if self.test_state['position']:
            self._update_position_pnl(market_data)
        
        # Update daily P&L tracking
        self._update_daily_pnl()
        
        # Check compliance after trading
        self._check_compliance()
    
    def _get_simulated_market_data(self) -> Dict[str, Any]:
        """
        Generate simulated market data for the current time step.
        
        In a real implementation, this would fetch actual market data.
        For the forward test, we simulate price movement with randomness.
        
        Returns:
            Dictionary with market data
        """
        # If we have a position, use its entry price as reference
        if self.test_state['position']:
            last_price = self.test_state['position']['entry_price']
        else:
            # Default starting price (simulated for major pairs)
            if 'last_price' not in self.test_state:
                if 'JPY' in self.test_state['pair']:
                    self.test_state['last_price'] = 130.0  # Typical for JPY pairs
                else:
                    self.test_state['last_price'] = 1.1  # Typical for EUR/USD, GBP/USD, etc.
            
            last_price = self.test_state['last_price']
        
        # Generate random price movement (simulated)
        # More sophisticated simulation would account for volatility, spread, etc.
        if 'JPY' in self.test_state['pair']:
            # JPY pairs move in larger nominal amounts
            price_change = np.random.normal(0, 0.05)  # ~5 pips for JPY pairs
        else:
            price_change = np.random.normal(0, 0.0005)  # ~5 pips for 4-decimal pairs
        
        # Calculate new price
        new_price = last_price + price_change
        
        # Calculate simulated OHLC data
        high = max(last_price, new_price) + abs(price_change) * 0.5
        low = min(last_price, new_price) - abs(price_change) * 0.5
        
        # Store last price for next time step
        self.test_state['last_price'] = new_price
        
        # Return simulated market data
        return {
            'pair': self.test_state['pair'],
            'time': self.test_state['current_time'],
            'open': last_price,
            'high': high,
            'low': low,
            'close': new_price,
            'volume': np.random.randint(10, 100),  # Simulated volume
            'spread': self._get_simulated_spread()
        }
    
    def _get_simulated_spread(self) -> float:
        """
        Get simulated spread for the current pair.
        
        Returns:
            Spread in pips
        """
        # Use pair manager if available
        if self.pair_manager:
            pair_obj = self.pair_manager.get_pair(self.test_state['pair'])
            if pair_obj:
                spread_range = pair_obj.get_spread_range()
                if spread_range and len(spread_range) == 2:
                    return np.random.uniform(spread_range[0], spread_range[1])
        
        # Default spreads if pair manager not available
        if 'JPY' in self.test_state['pair']:
            return np.random.uniform(1.0, 3.0)  # Typical for JPY pairs
        elif self.test_state['pair'] in ['EURUSD', 'GBPUSD', 'USDJPY']:
            return np.random.uniform(0.5, 1.5)  # Major pairs
        else:
            return np.random.uniform(1.0, 4.0)  # Cross pairs
    
    def _generate_trading_signal(self, market_data: Dict[str, Any]) -> int:
        """
        Generate trading signal based on the strategy and market data.
        
        Args:
            market_data: Current market data
            
        Returns:
            Signal: 1 for buy, -1 for sell, 0 for no action
        """
        # In a real implementation, this would instantiate the strategy class
        # and call its signal generation method with the market data.
        # For now, we'll simulate signals with randomness
        
        # If we have an open position, consider closing it
        if self.test_state['position']:
            position = self.test_state['position']
            direction = position['direction']
            entry_price = position['entry_price']
            current_price = market_data['close']
            position_age = (self.test_state['current_time'] - position['entry_time']).total_seconds() / 3600.0
            
            # Calculate profit/loss in pips
            pip_multiplier = 10000  # Default for 4-decimal pairs
            if 'JPY' in self.test_state['pair']:
                pip_multiplier = 100  # For JPY pairs
                
            if self.pair_manager:
                pair_obj = self.pair_manager.get_pair(self.test_state['pair'])
                if pair_obj:
                    pip_multiplier = pair_obj.get_pip_multiplier()
            
            price_diff = (current_price - entry_price) * direction
            pips = price_diff * pip_multiplier
            
            # Random exit logic (simulated)
            # More sophisticated implementation would use actual strategy logic
            if position_age > 12:  # Hold for max 12 hours
                return -direction  # Close position
            elif pips > 20:  # Take profit at 20 pips
                return -direction  # Close position
            elif pips < -10:  # Stop loss at 10 pips
                return -direction  # Close position
            else:
                return 0  # Hold position
        
        # If no position, consider opening one
        else:
            # Frequency of new trades (adjust as needed)
            # On average, open a new position every ~4 hours, more sophisticated
            # implementation would use actual strategy logic
            if np.random.random() < 0.25:
                # 50% chance of buy, 50% chance of sell
                return np.random.choice([1, -1])
            else:
                return 0  # No new position
    
    def _execute_trade(self, signal: int, market_data: Dict[str, Any], active_sessions: List[str]) -> None:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell, 0 for no action)
            market_data: Current market data
            active_sessions: List of active trading sessions
        """
        if signal == 0:
            return
            
        pair = self.test_state['pair']
        current_time = self.test_state['current_time']
        
        # If we have an open position and signal is opposite, close it
        if self.test_state['position'] and signal == -self.test_state['position']['direction']:
            position = self.test_state['position']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            direction = position['direction']
            lot_size = position['lot_size']
            spread = market_data['spread']
            
            # Calculate exit price with spread
            exit_price = market_data['close']
            if direction == 1:  # Buy position, exit at bid
                exit_price -= spread / self._get_pip_divisor()
            else:  # Sell position, exit at ask
                exit_price += spread / self._get_pip_divisor()
            
            # Calculate profit/loss
            pip_multiplier = self._get_pip_multiplier()
            price_diff = (exit_price - entry_price) * direction
            pips = price_diff * pip_multiplier
            pip_value = self._get_pip_value(lot_size)
            profit_loss = pips * pip_value
            
            # Update equity
            self.test_state['equity'] += profit_loss
            
            # Update max equity if needed
            if self.test_state['equity'] > self.test_state['max_equity']:
                self.test_state['max_equity'] = self.test_state['equity']
            
            # Record the trade
            trade = {
                'pair': pair,
                'direction': direction,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_time': current_time,
                'exit_price': exit_price,
                'lot_size': lot_size,
                'spread_pips': spread,
                'pips': pips,
                'profit_loss': profit_loss,
                'session': active_sessions[0] if active_sessions else 'Unknown'
            }
            
            self.test_state['trade_history'].append(trade)
            
            # Track session performance
            if active_sessions:
                session = active_sessions[0]
                if session in self.test_state['session_trades']:
                    self.test_state['session_trades'][session].append(trade)
            
            # Log trade to pip logger if available
            if self.pip_logger:
                try:
                    self.pip_logger.log_trade(
                        pair=pair,
                        entry_time=entry_time,
                        exit_time=current_time,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        lot_size=lot_size,
                        entry_spread_pips=position.get('entry_spread', spread / 2),
                        exit_spread_pips=spread / 2,
                        session=active_sessions[0] if active_sessions else None,
                        strategy_name=self.test_state['strategy_id']
                    )
                except Exception as e:
                    logger.error(f"Error logging trade to pip logger: {e}")
            
            # Reset position
            self.test_state['position'] = None
            
            logger.info(f"Closed {pair} position: {pips:.1f} pips, {profit_loss:.2f} profit/loss")
            
        # If we don't have a position and signal is non-zero, open one
        elif not self.test_state['position'] and signal != 0:
            spread = market_data['spread']
            
            # Calculate entry price with spread
            entry_price = market_data['close']
            if signal == 1:  # Buy at ask
                entry_price += spread / self._get_pip_divisor() / 2
            else:  # Sell at bid
                entry_price -= spread / self._get_pip_divisor() / 2
            
            # Standard lot size (could be made dynamic or risk-based)
            lot_size = 0.1
            
            # Create position
            position = {
                'pair': pair,
                'direction': signal,
                'entry_time': current_time,
                'entry_price': entry_price,
                'lot_size': lot_size,
                'entry_spread': spread
            }
            
            self.test_state['position'] = position
            
            logger.info(f"Opened {pair} position: {'Buy' if signal == 1 else 'Sell'} at {entry_price}")
    
    def _update_position_pnl(self, market_data: Dict[str, Any]) -> None:
        """
        Update the unrealized profit/loss of the current position.
        
        Args:
            market_data: Current market data
        """
        if not self.test_state['position']:
            return
            
        position = self.test_state['position']
        entry_price = position['entry_price']
        direction = position['direction']
        lot_size = position['lot_size']
        current_price = market_data['close']
        
        # Calculate unrealized P&L
        pip_multiplier = self._get_pip_multiplier()
        price_diff = (current_price - entry_price) * direction
        pips = price_diff * pip_multiplier
        pip_value = self._get_pip_value(lot_size)
        unrealized_pnl = pips * pip_value
        
        # Store in position
        position['unrealized_pnl'] = unrealized_pnl
        position['unrealized_pips'] = pips
    
    def _calculate_current_drawdown(self) -> float:
        """
        Calculate the current drawdown as a percentage.
        
        Returns:
            Drawdown percentage
        """
        if self.test_state['equity'] >= self.test_state['max_equity']:
            return 0.0
            
        drawdown = (self.test_state['max_equity'] - self.test_state['equity']) / self.test_state['max_equity'] * 100.0
        return drawdown
    
    def _update_daily_pnl(self) -> None:
        """
        Update the daily profit/loss tracking.
        """
        current_date = self.test_state['current_time'].date().isoformat()
        
        # Initialize daily P&L record if needed
        if current_date not in self.test_state['daily_pnl']:
            self.test_state['daily_pnl'][current_date] = {
                'start_equity': self.test_state['equity'],
                'current_equity': self.test_state['equity'],
                'trades': 0,
                'pips': 0.0
            }
        
        # Update current equity
        self.test_state['daily_pnl'][current_date]['current_equity'] = self.test_state['equity']
        
        # Update trade count and pips from today's completed trades
        today_trades = [t for t in self.test_state['trade_history'] 
                       if t['exit_time'].date().isoformat() == current_date]
        
        self.test_state['daily_pnl'][current_date]['trades'] = len(today_trades)
        self.test_state['daily_pnl'][current_date]['pips'] = sum(t['pips'] for t in today_trades)
    
    def _check_compliance(self) -> None:
        """
        Check prop firm compliance status.
        """
        # Calculate current drawdown
        drawdown = self._calculate_current_drawdown()
        max_drawdown = self.prop_compliance_rules['max_drawdown_percent']
        
        # Check drawdown compliance
        if drawdown > max_drawdown:
            self.test_state['compliance_status']['compliant'] = False
            self.test_state['compliance_status']['max_drawdown_reached'] = True
            logger.warning(f"Max drawdown exceeded: {drawdown:.2f}% > {max_drawdown}%")
        
        # Check daily loss compliance
        current_date = self.test_state['current_time'].date().isoformat()
        if current_date in self.test_state['daily_pnl']:
            daily_data = self.test_state['daily_pnl'][current_date]
            daily_pnl_pct = (daily_data['current_equity'] / daily_data['start_equity'] - 1) * 100
            daily_loss_limit = -self.prop_compliance_rules['daily_loss_limit_percent']
            
            if daily_pnl_pct < daily_loss_limit:
                self.test_state['compliance_status']['compliant'] = False
                self.test_state['compliance_status']['daily_loss_limit_reached'] = True
                logger.warning(f"Daily loss limit exceeded: {daily_pnl_pct:.2f}% < {daily_loss_limit}%")
        
        # Check profit target (if reached, we don't stop trading but mark it as achieved)
        total_return_pct = (self.test_state['equity'] / self.test_state['initial_equity'] - 1) * 100
        profit_target = self.prop_compliance_rules['target_profit_percent']
        
        if total_return_pct >= profit_target and not self.test_state['compliance_status']['profit_target_reached']:
            self.test_state['compliance_status']['profit_target_reached'] = True
            logger.info(f"Profit target reached: {total_return_pct:.2f}% >= {profit_target}%")
    
    def _check_stop_conditions(self) -> bool:
        """
        Check if we should stop the forward test.
        
        Returns:
            True if we should stop, False otherwise
        """
        # Check if stop trading at threshold is enabled
        if not self.prop_compliance_rules.get('stop_trading_at_threshold', True):
            return False
            
        # Check max drawdown
        if self.test_state['compliance_status']['max_drawdown_reached']:
            logger.warning("Stopping forward test due to max drawdown reached")
            return True
            
        # Check daily loss limit
        if self.test_state['compliance_status']['daily_loss_limit_reached']:
            logger.warning("Stopping forward test due to daily loss limit reached")
            return True
            
        return False
    
    def _get_pip_multiplier(self) -> float:
        """
        Get the pip multiplier for the current pair.
        
        Returns:
            Pip multiplier (e.g., 10000 for 4-decimal pairs, 100 for JPY pairs)
        """
        pair = self.test_state['pair']
        
        # Use pair manager if available
        if self.pair_manager:
            pair_obj = self.pair_manager.get_pair(pair)
            if pair_obj:
                return pair_obj.get_pip_multiplier()
        
        # Default multipliers
        if 'JPY' in pair:
            return 100  # JPY pairs have 2 decimal places
        else:
            return 10000  # Most pairs have 4 decimal places
    
    def _get_pip_divisor(self) -> float:
        """
        Get the pip divisor for the current pair (reciprocal of multiplier).
        
        Returns:
            Pip divisor
        """
        return 1.0 / self._get_pip_multiplier()
    
    def _get_pip_value(self, lot_size: float) -> float:
        """
        Get the pip value for the current pair and lot size.
        
        Args:
            lot_size: Lot size (e.g., 0.1 for mini lot)
            
        Returns:
            Pip value in account currency
        """
        pair = self.test_state['pair']
        
        # Use pair manager if available
        if self.pair_manager:
            pair_obj = self.pair_manager.get_pair(pair)
            if pair_obj:
                return pair_obj.calculate_pip_value(lot_size)
        
        # Default pip value (approx. $10 per pip per standard lot)
        return 10.0 * lot_size
    
    def _report_to_benbot(self) -> None:
        """
        Report current forward test status to BenBot.
        """
        if not hasattr(self.forex_evotrader, 'benbot_available') or not self.forex_evotrader.benbot_available:
            return
            
        # Calculate key metrics
        drawdown = self._calculate_current_drawdown()
        total_return_pct = (self.test_state['equity'] / self.test_state['initial_equity'] - 1) * 100
        
        # Calculate win rate and trading stats
        total_trades = len(self.test_state['trade_history'])
        winning_trades = len([t for t in self.test_state['trade_history'] if t['pips'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pips = sum(t['pips'] for t in self.test_state['trade_history'])
        
        # Compile session stats
        session_stats = {}
        for session, trades in self.test_state['session_trades'].items():
            if trades:
                session_winning = len([t for t in trades if t['pips'] > 0])
                session_win_rate = session_winning / len(trades) if trades else 0
                session_stats[session] = {
                    'trades': len(trades),
                    'win_rate': session_win_rate,
                    'total_pips': sum(t['pips'] for t in trades),
                    'avg_pips_per_trade': sum(t['pips'] for t in trades) / len(trades) if trades else 0
                }
        
        # Create report
        report = {
            'strategy_id': self.test_state['strategy_id'],
            'pair': self.test_state['pair'],
            'timeframe': self.test_state['timeframe'],
            'current_time': self.test_state['current_time'].isoformat(),
            'elapsed_time': (self.test_state['current_time'] - self.test_state['start_time']).total_seconds() / 3600.0,
            'equity': self.test_state['equity'],
            'drawdown': drawdown,
            'total_return_percent': total_return_pct,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'current_position': self.test_state['position'],
            'compliance_status': self.test_state['compliance_status'],
            'session_stats': session_stats,
            'daily_pnl': self.test_state['daily_pnl']
        }
        
        # Send to BenBot
        self.forex_evotrader._consult_benbot('forward_test_update', report)
        
        logger.info(f"Reported to BenBot: {total_trades} trades, {win_rate:.2%} win rate, {total_pips:.1f} pips, {drawdown:.2f}% drawdown")
    
    def _compile_test_results(self) -> Dict[str, Any]:
        """
        Compile the results of the forward test.
        
        Returns:
            Dictionary with test results
        """
        # Calculate overall metrics
        total_trades = len(self.test_state['trade_history'])
        winning_trades = len([t for t in self.test_state['trade_history'] if t['pips'] > 0])
        losing_trades = len([t for t in self.test_state['trade_history'] if t['pips'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pips = sum(t['pips'] for t in self.test_state['trade_history'])
        avg_pips_per_trade = total_pips / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_win_pips = sum(t['pips'] for t in self.test_state['trade_history'] if t['pips'] > 0)
        gross_loss_pips = abs(sum(t['pips'] for t in self.test_state['trade_history'] if t['pips'] <= 0))
        profit_factor = gross_win_pips / gross_loss_pips if gross_loss_pips > 0 else float('inf')
        
        # Calculate returns and drawdown
        total_return_pct = (self.test_state['equity'] / self.test_state['initial_equity'] - 1) * 100
        max_drawdown = self._calculate_current_drawdown()
        
        # Calculate session performance
        session_performance = {}
        for session, trades in self.test_state['session_trades'].items():
            if trades:
                session_winning = len([t for t in trades if t['pips'] > 0])
                session_win_rate = session_winning / len(trades) if trades else 0
                session_performance[session] = {
                    'trades': len(trades),
                    'winning_trades': session_winning,
                    'losing_trades': len(trades) - session_winning,
                    'win_rate': session_win_rate,
                    'total_pips': sum(t['pips'] for t in trades),
                    'avg_pips_per_trade': sum(t['pips'] for t in trades) / len(trades) if trades else 0
                }
        
        # Compile results
        results = {
            'strategy_id': self.test_state['strategy_id'],
            'pair': self.test_state['pair'],
            'timeframe': self.test_state['timeframe'],
            'start_time': self.test_state['start_time'].isoformat(),
            'end_time': self.test_state['current_time'].isoformat(),
            'duration_hours': (self.test_state['current_time'] - self.test_state['start_time']).total_seconds() / 3600.0,
            'initial_equity': self.test_state['initial_equity'],
            'final_equity': self.test_state['equity'],
            'max_equity': self.test_state['max_equity'],
            'total_return_percent': total_return_pct,
            'max_drawdown_percent': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_pips_per_trade': avg_pips_per_trade,
            'profit_factor': profit_factor,
            'session_performance': session_performance,
            'compliance_status': self.test_state['compliance_status'],
            'session_only': self.test_state.get('session_only', False),
            'news_guard_enabled': self.test_state.get('enable_news_guard', True),
            'daily_pnl': self.test_state['daily_pnl'],
            'trade_history': [self._trade_to_dict(t) for t in self.test_state['trade_history']]
        }
        
        # Log summary
        logger.info(f"Forward test completed: {results['strategy_id']}")
        logger.info(f"Period: {results['start_time']} to {results['end_time']} ({results['duration_hours']:.1f} hours)")
        logger.info(f"Trades: {total_trades}, Win rate: {win_rate:.2%}, Total pips: {total_pips:.1f}")
        logger.info(f"Return: {total_return_pct:.2f}%, Max drawdown: {max_drawdown:.2f}%")
        logger.info(f"Compliance: {results['compliance_status']}")
        
        # Save results to file
        self._save_results_to_file(results)
        
        return results
    
    def _trade_to_dict(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert trade to serializable dictionary.
        
        Args:
            trade: Trade dictionary
            
        Returns:
            Serializable dictionary
        """
        # Convert datetime to ISO format strings
        result = trade.copy()
        if 'entry_time' in result and isinstance(result['entry_time'], datetime.datetime):
            result['entry_time'] = result['entry_time'].isoformat()
        if 'exit_time' in result and isinstance(result['exit_time'], datetime.datetime):
            result['exit_time'] = result['exit_time'].isoformat()
        return result
    
    def _save_results_to_file(self, results: Dict[str, Any]) -> str:
        """
        Save forward test results to file.
        
        Args:
            results: Test results dictionary
            
        Returns:
            Path to saved file
        """
        results_dir = self.config.get('results_dir', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{results_dir}/forward_test_{results['strategy_id']}_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Saved forward test results to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving forward test results: {e}")
            return ""


# Main execution
if __name__ == "__main__":
    import argparse
    import sys
    from forex_evotrader import ForexEvoTrader
    
    parser = argparse.ArgumentParser(description="Forex Strategy Forward Test")
    
    parser.add_argument(
        "--strategy-id",
        type=str,
        required=True,
        help="Strategy ID to forward test"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to forward test"
    )
    
    parser.add_argument(
        "--session-only",
        action="store_true",
        help="Only trade during optimal sessions"
    )
    
    parser.add_argument(
        "--enable-news-guard",
        action="store_true",
        help="Enable news event filtering"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="forex_evotrader_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--accelerated",
        action="store_true",
        help="Run in accelerated mode (faster simulation)"
    )
    
    args = parser.parse_args()
    
    # Initialize EvoTrader
    try:
        forex_evotrader = ForexEvoTrader(args.config)
    except Exception as e:
        print(f"Error initializing ForexEvoTrader: {e}")
        sys.exit(1)
    
    # Initialize forward test
    forward_test = ForexStrategyForwardTest(forex_evotrader)
    
    # Set accelerated mode if requested
    if args.accelerated:
        forward_test.update_interval = 1  # 1 second instead of default
    
    # Start forward test
    print(f"Starting forward test for strategy {args.strategy_id}...")
    success = forward_test.start_forward_test(
        strategy_id=args.strategy_id,
        days=args.days,
        session_only=args.session_only,
        enable_news_guard=args.enable_news_guard
    )
    
    if not success:
        print("Failed to start forward test. See logs for details.")
        sys.exit(1)
    
    # Run forward test
    try:
        print("Running forward test... Press Ctrl+C to stop")
        results = forward_test.run_forward_test()
        
        # Print summary
        if 'error' in results:
            print(f"Forward test error: {results['error']}")
        else:
            print("\nForward Test Results Summary:")
            print(f"Strategy: {results['strategy_id']} ({results['pair']} {results['timeframe']})")
            print(f"Period: {results['start_time']} to {results['end_time']} ({results['duration_hours']:.1f} hours)")
            print(f"Trades: {results['total_trades']}, Win rate: {results['win_rate']:.2%}")
            print(f"Total pips: {results['total_pips']:.1f}, Avg pips/trade: {results['avg_pips_per_trade']:.1f}")
            print(f"Return: {results['total_return_percent']:.2f}%, Max drawdown: {results['max_drawdown_percent']:.2f}%")
            
            # Print compliance status
            print("\nProp Firm Compliance:")
            compliance = results['compliance_status']
            print(f"Compliant: {'✓' if compliance['compliant'] else '✗'}")
            print(f"Max drawdown reached: {'✓' if compliance['max_drawdown_reached'] else '✗'}")
            print(f"Daily loss limit reached: {'✓' if compliance['daily_loss_limit_reached'] else '✗'}")
            print(f"Profit target reached: {'✓' if compliance['profit_target_reached'] else '✗'}")
            
            # Print session performance
            print("\nSession Performance:")
            for session, metrics in results['session_performance'].items():
                if metrics['trades'] > 0:
                    print(f"  {session}:")
                    print(f"    Trades: {metrics['trades']}")
                    print(f"    Win rate: {metrics['win_rate']:.2%}")
                    print(f"    Total pips: {metrics['total_pips']:.1f}")
                    print(f"    Avg pips/trade: {metrics['avg_pips_per_trade']:.1f}")
    
    except KeyboardInterrupt:
        print("\nForward test interrupted by user")
    except Exception as e:
        print(f"\nError in forward test: {e}")
        import traceback
        print(traceback.format_exc())
