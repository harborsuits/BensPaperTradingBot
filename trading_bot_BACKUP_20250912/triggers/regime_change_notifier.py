"""
Market Regime Change Notification System

Monitors and detects market regime changes, then sends notifications
through various channels (Telegram, email, SMS) to alert users.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import requests

# Import regime detector
from trading_bot.ml_pipeline.ml_regime_detector import MLRegimeDetector

# Import notification system
from trading_bot.triggers.notification_connector import NotificationConnector

logger = logging.getLogger(__name__)

class RegimeChangeNotifier:
    """
    Regime change detection and notification system
    
    Monitors market conditions, detects regime changes, and
    sends notifications through configured channels.
    """
    
    def __init__(self, config=None):
        """
        Initialize the regime change notifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Notification settings
        self.notification_channels = self.config.get('notification_channels', ['telegram', 'email'])
        self.check_interval_minutes = self.config.get('check_interval_minutes', 60)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.market_symbols = self.config.get('market_symbols', ['SPY', 'QQQ', 'IWM'])
        self.lookback_days = self.config.get('lookback_days', 90)
        
        # Initialize components
        self.regime_detector = MLRegimeDetector(config=self.config.get('regime_detector', {}))
        self.notifier = NotificationConnector(config=self.config.get('notification', {}))
        
        # State tracking
        self.last_detected_regime = None
        self.last_check_time = None
        self.regime_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Load historical regimes
        self._load_regime_history()
        
        logger.info("RegimeChangeNotifier initialized")
    
    def _load_regime_history(self):
        """Load historical regime changes from disk"""
        try:
            history_file = os.path.join('data', 'regime_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # Convert to list of dicts with proper timestamps
                self.regime_history = []
                for item in history:
                    try:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        self.regime_history.append(item)
                    except (ValueError, KeyError):
                        pass
                
                if self.regime_history:
                    self.last_detected_regime = self.regime_history[-1]['regime']
                    logger.info(f"Loaded {len(self.regime_history)} historical regime changes")
        except Exception as e:
            logger.error(f"Error loading regime history: {e}")
    
    def _save_regime_history(self):
        """Save regime change history to disk"""
        try:
            os.makedirs('data', exist_ok=True)
            history_file = os.path.join('data', 'regime_history.json')
            
            # Convert datetime objects to strings
            serializable_history = []
            for item in self.regime_history:
                serialized_item = item.copy()
                if isinstance(serialized_item.get('timestamp'), datetime):
                    serialized_item['timestamp'] = serialized_item['timestamp'].isoformat()
                serializable_history.append(serialized_item)
            
            with open(history_file, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            
            logger.debug(f"Saved regime history with {len(self.regime_history)} entries")
        except Exception as e:
            logger.error(f"Error saving regime history: {e}")
    
    def check_for_regime_change(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Check if there has been a market regime change
        
        Args:
            market_data: Dictionary of symbol -> DataFrame with market data
            
        Returns:
            Dict with regime change information
        """
        try:
            # Update last check time
            self.last_check_time = datetime.now()
            
            # Detect regime for each symbol
            regime_votes = {}
            confidence_sum = 0
            
            for symbol, data in market_data.items():
                if len(data) > 20:  # Need sufficient data
                    regime_info = self.regime_detector.detect_regime(data)
                    
                    # Count votes for each regime
                    regime = regime_info['regime']
                    confidence = regime_info['confidence']
                    
                    if regime not in regime_votes:
                        regime_votes[regime] = 0
                    
                    # Weight vote by confidence
                    regime_votes[regime] += confidence
                    confidence_sum += confidence
            
            # No data or detection failed
            if not regime_votes or confidence_sum == 0:
                return {
                    'regime_changed': False,
                    'error': 'Insufficient data for regime detection'
                }
            
            # Find majority regime
            current_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
            current_confidence = regime_votes[current_regime] / confidence_sum
            
            # Check if regime has changed with sufficient confidence
            regime_changed = (
                self.last_detected_regime is not None and
                current_regime != self.last_detected_regime and
                current_confidence >= self.min_confidence
            )
            
            # If new regime or first detection
            if self.last_detected_regime is None or regime_changed:
                # Record the regime change
                change_info = {
                    'regime': current_regime,
                    'confidence': current_confidence,
                    'previous_regime': self.last_detected_regime,
                    'timestamp': datetime.now(),
                    'detected_symbols': list(market_data.keys())
                }
                
                # Update state
                self.last_detected_regime = current_regime
                self.regime_history.append(change_info)
                self._save_regime_history()
                
                logger.info(f"Regime change detected: {self.last_detected_regime} -> {current_regime} "
                          f"(confidence: {current_confidence:.4f})")
                
                # Return change information
                return {
                    'regime_changed': True,
                    'from_regime': change_info['previous_regime'],
                    'to_regime': current_regime,
                    'confidence': current_confidence,
                    'description': self.regime_detector._get_regime_description(current_regime),
                    'timestamp': change_info['timestamp']
                }
            
            # No regime change
            return {
                'regime_changed': False,
                'current_regime': current_regime,
                'confidence': current_confidence
            }
            
        except Exception as e:
            logger.error(f"Error checking for regime change: {e}")
            return {
                'regime_changed': False,
                'error': str(e)
            }
    
    def send_regime_change_notification(self, change_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send notification about regime change
        
        Args:
            change_info: Regime change information
            
        Returns:
            Dict with notification results
        """
        try:
            # Create notification message
            from_regime = change_info.get('from_regime', 'Unknown')
            to_regime = change_info.get('to_regime', 'Unknown')
            confidence = change_info.get('confidence', 0.0)
            description = change_info.get('description', '')
            
            title = f"🚨 Market Regime Change Detected: {to_regime.title()}"
            
            message = f"""
Market Regime Change Alert

Changed From: {from_regime.title()} → To: {to_regime.title()}
Confidence: {confidence:.1%}

{description}

Detected at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Recommended Actions:
"""
            # Add recommended actions based on new regime
            if to_regime == 'bullish_trend':
                message += """
• Increase allocation to trend-following strategies
• Consider reducing hedges
• Focus on momentum stocks
• Adjust trailing stops to capture upside
"""
            elif to_regime == 'bearish_trend':
                message += """
• Increase hedging positions
• Reduce position sizes
• Focus on defensive sectors
• Tighten stop losses
• Consider inverse ETFs
"""
            elif to_regime == 'volatile':
                message += """
• Reduce overall exposure
• Implement volatility-based position sizing
• Consider short-term mean reversion strategies
• Use options strategies to benefit from volatility
• Widen stop losses to avoid whipsaws
"""
            elif to_regime == 'ranging':
                message += """
• Focus on mean-reversion strategies
• Look for range-bound trading opportunities
• Reduce trend-following allocations
• Consider pairs trading strategies
• Be patient with entries and exits
"""
            else:
                message += """
• Review current strategy allocations
• Rebalance portfolio based on new regime
• Monitor closely for confirmation of regime change
"""
            
            # Send notification through configured channels
            results = {}
            
            for channel in self.notification_channels:
                if channel == 'telegram':
                    result = self.notifier.send_telegram_message(message)
                    results['telegram'] = result.get('success', False)
                
                elif channel == 'email':
                    result = self.notifier.send_email(title, message)
                    results['email'] = result.get('success', False)
                
                elif channel == 'sms':
                    # Create a shorter message for SMS
                    sms_message = f"Market Regime Change: {from_regime} → {to_regime} ({confidence:.1%}). Review strategy allocations."
                    result = self.notifier.send_sms(sms_message)
                    results['sms'] = result.get('success', False)
            
            # Log notification
            logger.info(f"Regime change notifications sent via {', '.join(results.keys())}")
            
            return {
                'success': any(results.values()),
                'channels': results
            }
            
        except Exception as e:
            logger.error(f"Error sending regime change notification: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def start_monitoring(self, data_provider=None):
        """
        Start monitoring for regime changes in a background thread
        
        Args:
            data_provider: Function to get market data
        """
        if self.monitoring_active:
            logger.warning("Regime monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Create monitoring thread
        def monitoring_loop():
            logger.info("Starting regime change monitoring loop")
            
            while self.monitoring_active:
                try:
                    # Get market data
                    market_data = {}
                    
                    if data_provider is not None:
                        # Use provided data function
                        market_data = data_provider(self.market_symbols, self.lookback_days)
                    else:
                        # Use default data provider
                        self._fetch_market_data()
                    
                    # Check for regime change
                    if market_data:
                        change_info = self.check_for_regime_change(market_data)
                        
                        # If regime changed, send notification
                        if change_info.get('regime_changed', False):
                            self.send_regime_change_notification(change_info)
                    
                    # Sleep until next check
                    time.sleep(self.check_interval_minutes * 60)
                
                except Exception as e:
                    logger.error(f"Error in regime monitoring loop: {e}")
                    time.sleep(300)  # Sleep 5 minutes on error
        
        # Start thread
        self.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Regime change monitoring started, checking every {self.check_interval_minutes} minutes")
    
    def stop_monitoring(self):
        """Stop monitoring for regime changes"""
        if not self.monitoring_active:
            logger.warning("Regime monitoring not active")
            return
        
        self.monitoring_active = False
        
        if self.monitor_thread:
            # Wait for thread to finish
            if self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
            
            self.monitor_thread = None
        
        logger.info("Regime change monitoring stopped")
    
    def _fetch_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for regime detection
        
        Returns:
            Dictionary of symbol -> DataFrame with market data
        """
        # Import data provider
        from trading_bot.data.market_data_provider import create_data_provider
        
        try:
            # Create data provider
            data_provider = create_data_provider()
            
            # Fetch data for each symbol
            market_data = {}
            
            for symbol in self.market_symbols:
                # Get daily data
                data = data_provider.get_historical_data(
                    symbol=symbol,
                    interval='1day',
                    days=self.lookback_days
                )
                
                if data is not None and not data.empty:
                    market_data[symbol] = data
            
            logger.debug(f"Fetched market data for {len(market_data)} symbols")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def get_regime_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get history of regime changes
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of regime change events
        """
        if not self.regime_history:
            return []
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter history by date
        recent_history = [
            change for change in self.regime_history
            if isinstance(change.get('timestamp'), datetime) and change['timestamp'] >= cutoff_date
        ]
        
        # Convert datetime objects to strings for JSON serialization
        serializable_history = []
        for item in recent_history:
            serialized_item = item.copy()
            if isinstance(serialized_item.get('timestamp'), datetime):
                serialized_item['timestamp'] = serialized_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            serializable_history.append(serialized_item)
        
        return serializable_history
    
    def get_current_regime(self) -> Dict[str, Any]:
        """
        Get information about the current market regime
        
        Returns:
            Dict with current regime information
        """
        if self.last_detected_regime is None:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'description': 'No regime detected yet'
            }
        
        # Get the most recent regime
        if self.regime_history:
            latest = self.regime_history[-1]
            
            return {
                'regime': latest['regime'],
                'confidence': latest.get('confidence', 0.0),
                'description': self.regime_detector._get_regime_description(latest['regime']),
                'timestamp': latest.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(latest.get('timestamp'), datetime) else str(latest.get('timestamp', ''))
            }
        
        return {
            'regime': self.last_detected_regime,
            'confidence': 0.0,
            'description': self.regime_detector._get_regime_description(self.last_detected_regime)
        }
