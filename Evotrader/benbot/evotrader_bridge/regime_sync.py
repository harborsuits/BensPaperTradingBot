#!/usr/bin/env python3
"""
Market Regime Synchronization Component

Synchronizes market regime detection between EvoTrader and BensBot without modifying either system.
Ensures both systems have a consistent view of market conditions for better strategy selection.
"""

import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import threading

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import EvoTrader components
from market_regime_detector import MarketRegimeDetector
from benbot_api import BenBotAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('regime_sync')

class MarketRegimeSync:
    """
    Synchronizes market regime information between EvoTrader and BensBot.
    
    This component:
    1. Retrieves regime information from EvoTrader's sophisticated regime detector
    2. Pushes this information to BensBot for strategy selection
    3. Optionally pulls regime information from BensBot if it has its own detection
    4. Maintains a consistent regime view across both systems
    """
    
    def __init__(self, 
                benbot_api_url: str = None,
                update_interval: int = 3600,  # Default 1 hour
                mock_mode: bool = False):
        """
        Initialize regime synchronization component.
        
        Args:
            benbot_api_url: URL for BensBot API
            update_interval: Update interval in seconds
            mock_mode: Use mock data for testing
        """
        logger.info("Initializing market regime synchronization")
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector()
        self.benbot_api = BenBotAPI(base_url=benbot_api_url, mock=mock_mode)
        
        # Configuration
        self.update_interval = update_interval
        self.mock_mode = mock_mode
        
        # State
        self.last_regime = None
        self.last_update_time = None
        self.running = False
        self.sync_thread = None
        self.regime_history = []
        
        # Maximum history to keep
        self.max_history = 100
        
        logger.info(f"Regime sync initialized with update interval: {update_interval}s")
    
    def detect_and_push_regime(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Detect current market regime and push to BensBot.
        
        Args:
            force_update: Force update regardless of interval
            
        Returns:
            Current regime information
        """
        try:
            # Check if update is needed
            current_time = datetime.now()
            
            if not force_update and self.last_update_time:
                elapsed = (current_time - self.last_update_time).total_seconds()
                if elapsed < self.update_interval:
                    logger.debug(f"Skipping update, {elapsed}s elapsed since last update (interval: {self.update_interval}s)")
                    return self.last_regime or {}
            
            logger.info("Detecting current market regime")
            
            # Detect regime
            regime_info = self.regime_detector.detect_current_regime()
            
            if not regime_info:
                logger.warning("Failed to detect market regime")
                return self.last_regime or {}
            
            # Add timestamp
            regime_info['timestamp'] = current_time.isoformat()
            
            # Push to BensBot
            try:
                logger.info(f"Pushing regime to BensBot: {regime_info['regime']}")
                result = self.benbot_api.update_market_regime(regime_info)
                
                if result:
                    logger.info("Successfully pushed regime to BensBot")
                else:
                    logger.warning("Failed to push regime to BensBot")
            except Exception as e:
                logger.error(f"Error pushing regime to BensBot: {e}")
            
            # Update state
            self.last_regime = regime_info
            self.last_update_time = current_time
            
            # Add to history
            self.regime_history.append(regime_info)
            
            # Trim history if needed
            if len(self.regime_history) > self.max_history:
                self.regime_history = self.regime_history[-self.max_history:]
            
            return regime_info
            
        except Exception as e:
            logger.error(f"Error in detect_and_push_regime: {e}")
            return self.last_regime or {}
    
    def pull_regime_from_benbot(self) -> Dict[str, Any]:
        """
        Pull current market regime from BensBot.
        
        Returns:
            Current regime information from BensBot
        """
        try:
            logger.info("Pulling market regime from BensBot")
            
            # Get regime from BensBot
            regime_info = self.benbot_api.get_market_regime()
            
            if regime_info:
                logger.info(f"Retrieved regime from BensBot: {regime_info.get('regime', 'unknown')}")
                
                # Update local state
                self.last_regime = regime_info
                self.last_update_time = datetime.now()
            else:
                logger.warning("Failed to retrieve regime from BensBot")
            
            return regime_info or {}
            
        except Exception as e:
            logger.error(f"Error in pull_regime_from_benbot: {e}")
            return {}
    
    def sync_regimes_bidirectional(self) -> Dict[str, Any]:
        """
        Synchronize regimes bidirectionally between EvoTrader and BensBot.
        
        - First tries to get regime from BensBot
        - If unavailable or older than threshold, detects and pushes from EvoTrader
        
        Returns:
            Synchronized regime information
        """
        try:
            logger.info("Performing bidirectional regime synchronization")
            
            # Try to get regime from BensBot
            benbot_regime = self.pull_regime_from_benbot()
            
            # Check if valid regime was received
            if benbot_regime and 'regime' in benbot_regime and 'timestamp' in benbot_regime:
                # Check if regime is recent enough
                try:
                    benbot_time = datetime.fromisoformat(benbot_regime['timestamp'])
                    current_time = datetime.now()
                    
                    # If regime is older than half the update interval, update it
                    if (current_time - benbot_time).total_seconds() > (self.update_interval / 2):
                        logger.info("BensBot regime is outdated, detecting new regime")
                        return self.detect_and_push_regime(force_update=True)
                    else:
                        logger.info("Using recent regime from BensBot")
                        self.last_regime = benbot_regime
                        self.last_update_time = current_time
                        return benbot_regime
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse BensBot regime timestamp: {e}")
                    # Fall back to detecting new regime
                    return self.detect_and_push_regime(force_update=True)
            else:
                # No valid regime from BensBot, detect and push
                logger.info("No valid regime from BensBot, detecting new regime")
                return self.detect_and_push_regime(force_update=True)
                
        except Exception as e:
            logger.error(f"Error in sync_regimes_bidirectional: {e}")
            
            # Fall back to local detection if synchronization fails
            try:
                return self.detect_and_push_regime(force_update=True)
            except Exception as inner_e:
                logger.error(f"Fallback detection also failed: {inner_e}")
                return self.last_regime or {}
    
    def get_regime_history(self, 
                         days: int = 30, 
                         include_indicators: bool = False) -> List[Dict[str, Any]]:
        """
        Get historical regime information.
        
        Args:
            days: Number of days of history to retrieve
            include_indicators: Whether to include indicator values
            
        Returns:
            List of historical regime information
        """
        try:
            logger.info(f"Getting regime history for the past {days} days")
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter history
            filtered_history = []
            
            for regime in self.regime_history:
                try:
                    timestamp = datetime.fromisoformat(regime['timestamp'])
                    if timestamp >= cutoff_date:
                        # Create a copy
                        regime_copy = regime.copy()
                        
                        # Remove indicator values if not requested
                        if not include_indicators and 'indicators' in regime_copy:
                            del regime_copy['indicators']
                        
                        filtered_history.append(regime_copy)
                except (ValueError, KeyError):
                    # Skip invalid entries
                    continue
            
            # If history is insufficient, try to augment with local detection
            if len(filtered_history) < 5 and not self.mock_mode:
                logger.info("Insufficient regime history, augmenting with local detection")
                
                # Try to get more history
                try:
                    additional_history = self.regime_detector.get_historical_regimes(days=days)
                    
                    if additional_history:
                        # Merge histories
                        merged_history = filtered_history.copy()
                        
                        # Add only entries not already in the history
                        existing_timestamps = {regime.get('timestamp') for regime in filtered_history}
                        
                        for regime in additional_history:
                            if regime.get('timestamp') not in existing_timestamps:
                                # Remove indicator values if not requested
                                if not include_indicators and 'indicators' in regime:
                                    regime_copy = regime.copy()
                                    del regime_copy['indicators']
                                    merged_history.append(regime_copy)
                                else:
                                    merged_history.append(regime)
                        
                        # Sort by timestamp
                        filtered_history = sorted(merged_history, 
                                               key=lambda x: x.get('timestamp', '0'))
                except Exception as e:
                    logger.warning(f"Failed to augment regime history: {e}")
            
            logger.info(f"Retrieved {len(filtered_history)} regime history entries")
            return filtered_history
            
        except Exception as e:
            logger.error(f"Error in get_regime_history: {e}")
            return []
    
    def start_sync_thread(self):
        """Start background synchronization thread."""
        if self.running:
            logger.warning("Sync already running, not starting new thread")
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        logger.info("Started regime synchronization thread")
    
    def stop_sync_thread(self):
        """Stop background synchronization thread."""
        self.running = False
        
        if self.sync_thread:
            # Wait for thread to finish (with timeout)
            self.sync_thread.join(timeout=5)
            self.sync_thread = None
        
        logger.info("Stopped regime synchronization thread")
    
    def _sync_loop(self):
        """Background synchronization loop."""
        logger.info("Regime sync loop started")
        
        while self.running:
            try:
                # Perform sync
                self.sync_regimes_bidirectional()
                
                # Sleep for interval
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in regime sync loop: {e}")
                
                # Sleep for a shorter interval on error
                time.sleep(min(300, self.update_interval / 2))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Regime Synchronization")
    parser.add_argument("--benbot-api", type=str, help="BensBot API URL")
    parser.add_argument("--interval", type=int, default=3600, help="Update interval in seconds")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for testing")
    parser.add_argument("--daemon", action="store_true", help="Run as a background daemon")
    
    args = parser.parse_args()
    
    # Create sync component
    sync = MarketRegimeSync(
        benbot_api_url=args.benbot_api,
        update_interval=args.interval,
        mock_mode=args.mock
    )
    
    if args.daemon:
        # Run as background thread
        sync.start_sync_thread()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            sync.stop_sync_thread()
            print("Stopped regime synchronization")
    else:
        # Run single sync
        result = sync.sync_regimes_bidirectional()
        print(f"Current market regime: {result.get('regime', 'unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        
        if 'indicators' in result:
            print("\nKey indicators:")
            for indicator, value in result['indicators'].items():
                print(f"  {indicator}: {value}")
        
        print(f"\nTimestamp: {result.get('timestamp', 'unknown')}")
