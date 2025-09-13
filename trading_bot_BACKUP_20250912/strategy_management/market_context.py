import json
import logging
import os
import time
from typing import Dict, Any, Optional

from trading_bot.strategy_management.interfaces import MarketContext

logger = logging.getLogger(__name__)

class TradingMarketContext(MarketContext):
    """Implementation of the MarketContext interface for trading applications"""
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """
        Initialize market context with optional initial data
        
        Args:
            initial_data: Dictionary of initial key-value pairs
        """
        self._data = {}
        self._last_updated = {}
        
        if initial_data:
            for key, value in initial_data.items():
                self.set_value(key, value)
                
        # Set default market regime if not provided
        if 'market_regime' not in self._data:
            self._data['market_regime'] = 'unknown'
            self._last_updated['market_regime'] = time.time()
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get value by key
        
        Args:
            key: The key to look up
            default: Default value to return if key not found
            
        Returns:
            The value associated with the key, or default if not found
        """
        return self._data.get(key, default)
    
    def set_value(self, key: str, value: Any) -> None:
        """
        Set value by key
        
        Args:
            key: The key to set
            value: The value to store
        """
        self._data[key] = value
        self._last_updated[key] = time.time()
        logger.debug(f"Market context updated: {key}")
    
    def get_all_values(self) -> Dict[str, Any]:
        """
        Get all values as dictionary
        
        Returns:
            A dictionary containing all key-value pairs
        """
        return self._data.copy()
    
    def get_last_updated(self, key: str) -> Optional[float]:
        """
        Get timestamp when key was last updated
        
        Args:
            key: The key to check
            
        Returns:
            Timestamp as float, or None if key not found
        """
        return self._last_updated.get(key)
    
    def get_regime(self) -> str:
        """
        Get current market regime classification
        
        Returns:
            String identifying current market regime
        """
        return self._data.get('market_regime', 'unknown')
    
    def get_market_state(self) -> Dict[str, Any]:
        """
        Get comprehensive market state data
        
        Returns:
            Dictionary with key market indicators
        """
        # Extract and organize key market indicators
        state = {
            'regime': self.get_regime(),
            'volatility': self._data.get('volatility', None),
            'trend': self._data.get('trend', None),
            'sentiment': self._data.get('market_sentiment', None),
            'economic_indicators': self._data.get('economic_indicators', {}),
            'technical_indicators': self._data.get('technical_indicators', {}),
        }
        
        # Add timestamp
        state['timestamp'] = time.time()
        
        return state
    
    def export_data(self) -> Dict[str, Any]:
        """
        Export all data for serialization
        
        Returns:
            Dictionary with all context data and metadata
        """
        export = {
            'data': self._data,
            'last_updated': self._last_updated,
            'export_timestamp': time.time()
        }
        return export
    
    def import_data(self, data: Dict[str, Any]) -> None:
        """
        Import data from serialized format
        
        Args:
            data: Dictionary with context data and metadata
        """
        if 'data' in data:
            self._data = data['data']
        
        if 'last_updated' in data:
            self._last_updated = data['last_updated']
            
        logger.info("Market context data imported")
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save context data to a JSON file
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.export_data(), f, indent=2)
            logger.info(f"Market context saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save market context: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """
        Load context data from a JSON file
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(filepath):
            logger.warning(f"Market context file not found: {filepath}")
            return False
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.import_data(data)
            logger.info(f"Market context loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load market context: {e}")
            return False 