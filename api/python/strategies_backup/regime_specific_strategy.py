"""
Regime-Specific Strategy Module

This module provides a strategy adapter that dynamically selects optimal
strategy parameters based on the current market regime.
"""

import logging
import os
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime

from trading_bot.ml_pipeline.market_regime_detector import MarketRegimeDetector
from trading_bot.strategies.hybrid_strategy_adapter import HybridStrategyAdapter
from trading_bot.strategies.strategy_factory import StrategyFactory
from trading_bot.data_handlers.data_loader import DataLoader

logger = logging.getLogger(__name__)

class RegimeSpecificStrategy:
    """
    Strategy adapter that dynamically adjusts parameters based on market regime
    
    This class detects the current market regime and adjusts strategy parameters
    to use the optimal configuration for the detected regime.
    """
    
    def __init__(self, config=None):
        """
        Initialize the regime-specific strategy
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.regime_detector = MarketRegimeDetector(config=self.config.get('market_regime', {}))
        self.data_loader = DataLoader(config=self.config.get('data_loader', {}))
        
        # Load optimization results
        self.regime_params = {}
        self.default_params = {}
        self.last_regime_update = None
        self.current_regime = None
        self.current_confidence = 0.0
        
        # Load regime-specific parameters
        self._load_regime_params()
        
        # Default strategy type is hybrid
        self.strategy_type = self.config.get('strategy_type', 'hybrid')
        self.strategy = None
        
        # Time period for checking regime (hours)
        self.regime_check_period = self.config.get('regime_check_period', 4)
        
        # Initialize strategy with default parameters
        self._initialize_strategy()
        
        logger.info(f"Regime-Specific Strategy initialized with {len(self.regime_params)} regime configurations")
    
    def _load_regime_params(self):
        """Load regime-specific parameters from optimization results"""
        results_dir = self.config.get('results_dir', 'optimization_results')
        
        # Find latest regime-specific weights file
        weights_files = []
        if os.path.exists(results_dir):
            weights_files = [f for f in os.listdir(results_dir) 
                           if f.startswith('regime_specific_weights_') and f.endswith('.json')]
        
        if not weights_files:
            logger.warning("No regime-specific weights found, using default parameters")
            self.regime_params = {}
            return
        
        # Sort by timestamp (newest first)
        weights_files.sort(reverse=True)
        latest_file = os.path.join(results_dir, weights_files[0])
        
        try:
            with open(latest_file, 'r') as f:
                regime_weights = json.load(f)
            
            logger.info(f"Loaded regime-specific weights from {latest_file}")
            
            # Extract parameters for each regime
            for regime, data in regime_weights.items():
                if 'weights' in data:
                    self.regime_params[regime] = data['weights']
                    
                    # Use most conservative regime as default (ranging)
                    if regime == 'ranging':
                        self.default_params = data['weights']
            
            # If no ranging regime found, use the first one as default
            if not self.default_params and regime_weights:
                first_regime = next(iter(regime_weights))
                self.default_params = regime_weights[first_regime].get('weights', {})
                
        except Exception as e:
            logger.error(f"Error loading regime-specific weights: {e}")
            self.regime_params = {}
    
    def _initialize_strategy(self):
        """Initialize strategy with current parameters"""
        try:
            # Use default parameters initially
            params = self.default_params.copy()
            
            # Create strategy of specified type with parameters
            self.strategy = StrategyFactory.create_strategy(
                self.strategy_type,
                config=params
            )
            
            logger.info(f"Initialized {self.strategy_type} strategy with default parameters")
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            self.strategy = None
    
    def detect_regime(self, data: pd.DataFrame = None, symbol: str = "SPY") -> Dict[str, Any]:
        """
        Detect current market regime
        
        Args:
            data: Optional DataFrame with market data
            symbol: Symbol to use for regime detection if data not provided
            
        Returns:
            Dictionary with regime information
        """
        # If data not provided, load it
        if data is None:
            try:
                # Load data for specified symbol (default to SPY)
                data = self.data_loader.load_historical_data(
                    symbol=symbol,
                    timeframe="1d",  # Daily data for regime detection
                    days=120         # Use 120 days of history for regime detection
                )
            except Exception as e:
                logger.error(f"Error loading data for regime detection: {e}")
                return {
                    'regime': 'unknown',
                    'confidence': 0.0,
                    'description': f"Error loading data: {e}"
                }
        
        # Detect regime
        try:
            regime_info = self.regime_detector.detect_regime(data)
            
            # Update current regime
            self.current_regime = regime_info['regime']
            self.current_confidence = regime_info['confidence']
            self.last_regime_update = datetime.now()
            
            logger.info(f"Detected market regime: {self.current_regime} "
                      f"(confidence: {self.current_confidence:.2f})")
            
            return regime_info
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'description': f"Error detecting regime: {e}"
            }
    
    def update_strategy_parameters(self, force_update: bool = False) -> bool:
        """
        Update strategy parameters based on current regime
        
        Args:
            force_update: Whether to force parameter update regardless of time
            
        Returns:
            True if parameters were updated, False otherwise
        """
        # Check if we need to update regime detection
        current_time = datetime.now()
        needs_update = force_update
        
        if self.last_regime_update is None:
            needs_update = True
        elif (current_time - self.last_regime_update).total_seconds() > self.regime_check_period * 3600:
            needs_update = True
        
        # Detect regime if needed
        if needs_update:
            regime_info = self.detect_regime()
            
            # If regime detection failed, don't update parameters
            if regime_info['regime'] == 'unknown':
                logger.warning("Failed to detect market regime, keeping current parameters")
                return False
        
        # Get parameters for current regime
        if self.current_regime in self.regime_params:
            # Use parameters for current regime
            new_params = self.regime_params[self.current_regime].copy()
            
            # Only update if confidence is high enough
            if self.current_confidence >= 0.6:
                try:
                    # Update strategy with new parameters
                    if hasattr(self.strategy, 'update_parameters'):
                        self.strategy.update_parameters(new_params)
                    else:
                        # Recreate strategy with new parameters
                        self.strategy = StrategyFactory.create_strategy(
                            self.strategy_type,
                            config=new_params
                        )
                    
                    logger.info(f"Updated strategy parameters for {self.current_regime} regime")
                    return True
                except Exception as e:
                    logger.error(f"Error updating strategy parameters: {e}")
                    return False
            else:
                logger.info(f"Regime confidence too low ({self.current_confidence:.2f}), "
                          f"keeping current parameters")
                return False
        else:
            logger.warning(f"No parameters available for {self.current_regime} regime, "
                         f"using default parameters")
            
            # Use default parameters
            if self.default_params:
                try:
                    # Update strategy with default parameters
                    if hasattr(self.strategy, 'update_parameters'):
                        self.strategy.update_parameters(self.default_params)
                    else:
                        # Recreate strategy with default parameters
                        self.strategy = StrategyFactory.create_strategy(
                            self.strategy_type,
                            config=self.default_params
                        )
                    
                    logger.info("Updated strategy with default parameters")
                    return True
                except Exception as e:
                    logger.error(f"Error updating strategy with default parameters: {e}")
                    return False
            
            return False
    
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signals using the regime-appropriate strategy
        
        Args:
            data: DataFrame with market data
            
        Returns:
            Dictionary with trading signals
        """
        # Update strategy parameters based on current regime
        self.update_strategy_parameters()
        
        # Generate signals using current strategy
        if self.strategy is not None:
            try:
                signals = self.strategy.generate_signals(data)
                
                # Add regime information to signals
                if signals:
                    signals['regime'] = self.current_regime
                    signals['regime_confidence'] = self.current_confidence
                
                return signals
            except Exception as e:
                logger.error(f"Error generating signals: {e}")
                return {}
        
        logger.error("No strategy available for generating signals")
        return {}
    
    def get_regime_info(self) -> Dict[str, Any]:
        """
        Get information about current market regime and strategy parameters
        
        Returns:
            Dictionary with regime and strategy information
        """
        return {
            'regime': self.current_regime,
            'confidence': self.current_confidence,
            'last_update': self.last_regime_update,
            'current_parameters': self.strategy.get_parameters() if hasattr(self.strategy, 'get_parameters') else {},
            'available_regimes': list(self.regime_params.keys()),
            'strategy_type': self.strategy_type
        }
