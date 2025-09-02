"""
Pattern Data Connector Module

This module provides connectors for importing pattern data from various sources:
1. Third-party pattern libraries (MetaTrader, TradingView, etc.)
2. Custom/proprietary pattern databases
3. Historical backtest results for pattern discovery

It serves as the bridge between external pattern knowledge and our internal
pattern discovery system, allowing the AI to leverage existing patterns while
developing its own specialized strategies.
"""

import os
import json
import logging
import pandas as pd
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from trading_bot.core.event_bus import EventBus, Event, EventType

logger = logging.getLogger(__name__)

class PatternDataConnector:
    """
    Connects to external pattern data sources and imports pattern definitions
    for use by the pattern discovery system.
    """
    
    def __init__(self, 
                 event_bus: EventBus,
                 data_dir: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Pattern Data Connector.
        
        Args:
            event_bus: Event bus for system communication
            data_dir: Directory for storing pattern data
            config: Configuration for data connectors
        """
        self.event_bus = event_bus
        
        # Set up data directory
        if not data_dir:
            data_dir = os.path.join(os.path.dirname(__file__), '../../data/patterns')
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        
        # Load configuration
        self.config = config or {}
        
        # Available connectors
        self.connectors = {
            'tradingview': self._import_from_tradingview,
            'metatrader': self._import_from_metatrader,
            'custom': self._import_from_custom_source,
            'backtest': self._import_from_backtest_results
        }
        
        # Track import history
        self.import_history = []
        
        logger.info("Pattern Data Connector initialized")
        
    def register_event_handlers(self):
        """Register event handlers for data import events."""
        self.event_bus.subscribe(EventType.PATTERN_IMPORT_REQUESTED, self.handle_import_request)
        
        # Schedule automatic imports if configured
        if self.config.get('auto_import_enabled', False):
            import_interval = self.config.get('auto_import_interval_hours', 24) * 3600
            self.event_bus.register_interval(self.run_scheduled_imports, import_interval)
            
        logger.info("Event handlers registered for Pattern Data Connector")
        
    def handle_import_request(self, event: Event):
        """
        Handle requests to import pattern data.
        """
        data = event.data
        source = data.get('source')
        parameters = data.get('parameters', {})
        
        if not source:
            logger.error("Import request missing source")
            return
            
        # Execute import from specified source
        result = self.import_patterns_from_source(source, parameters)
        
        # Publish import result
        self.event_bus.publish(
            EventType.PATTERN_IMPORT_COMPLETED,
            {
                'source': source,
                'success': result.get('success', False),
                'patterns_imported': result.get('patterns_imported', 0),
                'errors': result.get('errors', []),
                'timestamp': datetime.now()
            }
        )
        
    def import_patterns_from_source(self, source: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import patterns from a specified source.
        
        Args:
            source: Name of the source to import from
            parameters: Parameters for the import
            
        Returns:
            Dictionary with import results
        """
        if source not in self.connectors:
            return {
                'success': False,
                'errors': [f"Unknown source: {source}"],
                'patterns_imported': 0
            }
            
        try:
            # Call the appropriate connector
            connector_func = self.connectors[source]
            result = connector_func(parameters)
            
            # Track import history
            self.import_history.append({
                'source': source,
                'timestamp': datetime.now(),
                'patterns_imported': result.get('patterns_imported', 0),
                'success': result.get('success', False)
            })
            
            return result
        except Exception as e:
            logger.error(f"Error importing patterns from {source}: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'patterns_imported': 0
            }
            
    def run_scheduled_imports(self):
        """Run scheduled imports based on configuration."""
        scheduled_imports = self.config.get('scheduled_imports', [])
        
        for import_config in scheduled_imports:
            source = import_config.get('source')
            parameters = import_config.get('parameters', {})
            
            if source:
                logger.info(f"Running scheduled import from {source}")
                self.import_patterns_from_source(source, parameters)
                
    def get_available_pattern_libraries(self) -> List[Dict[str, Any]]:
        """
        Get list of available pattern libraries that can be imported.
        
        Returns:
            List of available libraries with metadata
        """
        libraries = []
        
        # Check TradingView libraries
        tv_dir = os.path.join(self.data_dir, 'tradingview')
        if os.path.exists(tv_dir):
            for file in os.listdir(tv_dir):
                if file.endswith('.json'):
                    lib_path = os.path.join(tv_dir, file)
                    try:
                        with open(lib_path, 'r') as f:
                            metadata = json.load(f).get('metadata', {})
                            
                        libraries.append({
                            'source': 'tradingview',
                            'name': metadata.get('name', file),
                            'patterns': metadata.get('pattern_count', 0),
                            'last_updated': metadata.get('last_updated')
                        })
                    except Exception as e:
                        logger.error(f"Error reading library metadata: {e}")
        
        # Check MetaTrader libraries
        mt_dir = os.path.join(self.data_dir, 'metatrader')
        if os.path.exists(mt_dir):
            for file in os.listdir(mt_dir):
                if file.endswith('.json'):
                    lib_path = os.path.join(mt_dir, file)
                    try:
                        with open(lib_path, 'r') as f:
                            metadata = json.load(f).get('metadata', {})
                            
                        libraries.append({
                            'source': 'metatrader',
                            'name': metadata.get('name', file),
                            'patterns': metadata.get('pattern_count', 0),
                            'last_updated': metadata.get('last_updated')
                        })
                    except Exception as e:
                        logger.error(f"Error reading library metadata: {e}")
                        
        # Check custom libraries
        custom_dir = os.path.join(self.data_dir, 'custom')
        if os.path.exists(custom_dir):
            for file in os.listdir(custom_dir):
                if file.endswith('.json'):
                    lib_path = os.path.join(custom_dir, file)
                    try:
                        with open(lib_path, 'r') as f:
                            metadata = json.load(f).get('metadata', {})
                            
                        libraries.append({
                            'source': 'custom',
                            'name': metadata.get('name', file),
                            'patterns': metadata.get('pattern_count', 0),
                            'last_updated': metadata.get('last_updated')
                        })
                    except Exception as e:
                        logger.error(f"Error reading library metadata: {e}")
                        
        return libraries
        
    def _import_from_tradingview(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import patterns from TradingView.
        
        Args:
            parameters: Import parameters
            
        Returns:
            Dictionary with import results
        """
        # TradingView import would require API access or parsing exported files
        # This is a placeholder for the actual implementation
        
        # For now, let's simulate importing from a local file
        file_path = parameters.get('file_path')
        if not file_path:
            return {
                'success': False,
                'errors': ["Missing file_path parameter"],
                'patterns_imported': 0
            }
            
        if not os.path.exists(file_path):
            return {
                'success': False,
                'errors': [f"File not found: {file_path}"],
                'patterns_imported': 0
            }
            
        try:
            with open(file_path, 'r') as f:
                pattern_data = json.load(f)
                
            patterns = pattern_data.get('patterns', [])
            if not patterns:
                return {
                    'success': True,
                    'patterns_imported': 0,
                    'message': "No patterns found in file"
                }
                
            # Process and publish each pattern
            imported_count = 0
            for pattern in patterns:
                # Add source information
                pattern['source'] = 'tradingview'
                pattern['import_time'] = datetime.now().isoformat()
                
                # Publish pattern for discovery system to process
                self.event_bus.publish(
                    EventType.PATTERN_DISCOVERED,
                    {
                        'pattern': pattern,
                        'source': 'tradingview',
                        'confidence': pattern.get('confidence', 0.5),
                        'timestamp': datetime.now()
                    }
                )
                
                imported_count += 1
                
            # Save to our repository
            output_dir = os.path.join(self.data_dir, 'tradingview')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, 
                                      f"tv_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(output_file, 'w') as f:
                json.dump(pattern_data, f, indent=2)
                
            return {
                'success': True,
                'patterns_imported': imported_count,
                'output_file': output_file
            }
                
        except Exception as e:
            logger.error(f"Error importing from TradingView: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'patterns_imported': 0
            }
            
    def _import_from_metatrader(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import patterns from MetaTrader.
        
        Args:
            parameters: Import parameters
            
        Returns:
            Dictionary with import results
        """
        # This would implement MetaTrader pattern import
        # Similar implementation to TradingView but with MT-specific format handling
        # For now, return a placeholder
        
        return {
            'success': False,
            'errors': ["MetaTrader import not yet implemented"],
            'patterns_imported': 0
        }
        
    def _import_from_custom_source(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import patterns from a custom source.
        
        Args:
            parameters: Import parameters
            
        Returns:
            Dictionary with import results
        """
        # This would handle custom format imports
        # The format would be defined by our own system
        
        file_path = parameters.get('file_path')
        if not file_path:
            return {
                'success': False,
                'errors': ["Missing file_path parameter"],
                'patterns_imported': 0
            }
            
        if not os.path.exists(file_path):
            return {
                'success': False,
                'errors': [f"File not found: {file_path}"],
                'patterns_imported': 0
            }
            
        try:
            with open(file_path, 'r') as f:
                pattern_data = json.load(f)
                
            patterns = pattern_data.get('patterns', [])
            if not patterns:
                return {
                    'success': True,
                    'patterns_imported': 0,
                    'message': "No patterns found in file"
                }
                
            # Process and publish each pattern
            imported_count = 0
            for pattern in patterns:
                # Add source information
                pattern['source'] = 'custom'
                pattern['import_time'] = datetime.now().isoformat()
                
                # Publish pattern for discovery system to process
                self.event_bus.publish(
                    EventType.PATTERN_DISCOVERED,
                    {
                        'pattern': pattern,
                        'source': 'custom',
                        'confidence': pattern.get('confidence', 0.5),
                        'timestamp': datetime.now()
                    }
                )
                
                imported_count += 1
                
            # Save to our repository
            output_dir = os.path.join(self.data_dir, 'custom')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, 
                                      f"custom_patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            with open(output_file, 'w') as f:
                json.dump(pattern_data, f, indent=2)
                
            return {
                'success': True,
                'patterns_imported': imported_count,
                'output_file': output_file
            }
                
        except Exception as e:
            logger.error(f"Error importing from custom source: {e}")
            return {
                'success': False,
                'errors': [str(e)],
                'patterns_imported': 0
            }
            
    def _import_from_backtest_results(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Import patterns discovered from backtest results.
        
        Args:
            parameters: Import parameters
            
        Returns:
            Dictionary with import results
        """
        # This would analyze backtest results to extract patterns
        # Potentially using ML to find patterns in successful trades
        
        backtest_id = parameters.get('backtest_id')
        min_occurrences = parameters.get('min_occurrences', 3)
        min_win_rate = parameters.get('min_win_rate', 0.65)
        
        if not backtest_id:
            return {
                'success': False,
                'errors': ["Missing backtest_id parameter"],
                'patterns_imported': 0
            }
            
        # This would retrieve and analyze backtest results
        # For now, return a placeholder
        
        return {
            'success': False,
            'errors': ["Backtest pattern analysis not yet implemented"],
            'patterns_imported': 0
        }
        
    def convert_to_system_format(self, pattern: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Convert an external pattern to our system's format.
        
        Args:
            pattern: External pattern definition
            source: Source of the pattern
            
        Returns:
            Converted pattern
        """
        # Different sources have different formats that need to be normalized
        # This would implement the conversion logic
        
        if source == 'tradingview':
            return self._convert_tradingview_pattern(pattern)
        elif source == 'metatrader':
            return self._convert_metatrader_pattern(pattern)
        else:
            # Assume custom source is already in our format
            return pattern
            
    def _convert_tradingview_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a TradingView pattern to system format."""
        # This would implement TradingView-specific conversion
        # For now, return a placeholder
        
        converted = {
            'id': f"tv_{int(datetime.now().timestamp())}",
            'name': pattern.get('name', 'Unknown TradingView Pattern'),
            'description': pattern.get('description', ''),
            'pattern_type': self._determine_pattern_type(pattern),
            'source': 'tradingview',
            'created_at': datetime.now().isoformat(),
            'created_by': 'import',
            'confidence': pattern.get('accuracy', 0.5),
            'parameters': {}
        }
        
        # Map pattern conditions from TradingView format
        # This would be customized based on actual TV format
        
        return converted
        
    def _convert_metatrader_pattern(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a MetaTrader pattern to system format."""
        # This would implement MetaTrader-specific conversion
        # For now, return a placeholder
        
        converted = {
            'id': f"mt_{int(datetime.now().timestamp())}",
            'name': pattern.get('name', 'Unknown MetaTrader Pattern'),
            'description': pattern.get('description', ''),
            'pattern_type': self._determine_pattern_type(pattern),
            'source': 'metatrader',
            'created_at': datetime.now().isoformat(),
            'created_by': 'import',
            'confidence': pattern.get('win_rate', 0.5),
            'parameters': {}
        }
        
        # Map pattern conditions from MetaTrader format
        # This would be customized based on actual MT format
        
        return converted
        
    def _determine_pattern_type(self, pattern: Dict[str, Any]) -> str:
        """Determine the type of pattern based on its properties."""
        # This would analyze pattern properties to categorize it
        
        if 'symbol' in pattern and pattern.get('symbol'):
            return 'symbol_specific'
        elif any(key in pattern for key in ['month', 'day', 'weekday', 'season']):
            return 'seasonal'
        elif any(key in pattern for key in ['event', 'earnings', 'economic']):
            return 'event_driven'
        else:
            return 'market_condition'
