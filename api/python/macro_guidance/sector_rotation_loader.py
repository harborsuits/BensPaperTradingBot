"""
Sector Rotation Framework Loader

This module handles loading and validating the sector rotation framework data
that's used by the MacroGuidanceEngine for sector rotation recommendations.
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SectorRotationLoader:
    """
    Handles loading and validating sector rotation framework data.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the sector rotation loader.
        
        Args:
            config_path: Optional path to sector rotation configuration file
        """
        self.config_path = config_path
        self.framework_data = {}
        
        # Attempt to load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load sector rotation framework data from file.
        
        Args:
            file_path: Path to config file (JSON or YAML)
            
        Returns:
            Dictionary containing the framework data
        """
        try:
            # Determine file type from extension
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() in ['.json']:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            elif ext.lower() in ['.yaml', '.yml']:
                with open(file_path, 'r') as f:
                    data = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported file format: {ext}")
                raise ValueError(f"Unsupported file format: {ext}. Must be JSON or YAML.")
            
            # Validate the data
            self.validate_framework(data)
            
            # Store the data
            self.framework_data = data
            
            logger.info(f"Successfully loaded sector rotation framework from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading sector rotation framework: {str(e)}")
            raise
    
    def load_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load sector rotation framework from dictionary data.
        
        Args:
            data: Dictionary containing the framework data
            
        Returns:
            The validated framework data
        """
        try:
            # Validate the data
            self.validate_framework(data)
            
            # Store the data
            self.framework_data = data
            
            logger.info("Successfully loaded sector rotation framework from data")
            return data
            
        except Exception as e:
            logger.error(f"Error loading sector rotation framework: {str(e)}")
            raise
    
    def validate_framework(self, data: Dict[str, Any]) -> bool:
        """
        Validate the sector rotation framework data.
        
        Args:
            data: Framework data to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check basic structure
        if not isinstance(data, dict):
            raise ValueError("Framework data must be a dictionary")
        
        # Check for required metadata
        if "framework_version" not in data:
            logger.warning("Framework data missing 'framework_version' metadata")
        
        if "last_updated" not in data:
            logger.warning("Framework data missing 'last_updated' metadata")
        
        # Check for at least one cycle phase
        cycle_phases = [key for key in data.keys() if key not in ["framework_version", "last_updated", "meta_data", "advanced_identification_framework", "multi_timeframe_application", "implementation_framework", "bot_specific_implementation"]]
        
        if not cycle_phases:
            raise ValueError("Framework must contain at least one economic cycle phase")
        
        # Validate each cycle phase
        for phase in cycle_phases:
            self._validate_cycle_phase(phase, data[phase])
        
        return True
    
    def _validate_cycle_phase(self, phase_name: str, phase_data: Dict[str, Any]) -> bool:
        """
        Validate an individual cycle phase.
        
        Args:
            phase_name: Name of the cycle phase
            phase_data: Data for the cycle phase
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check for required sections
        required_sections = ["description", "macro_signals", "favored_sectors", "strategies"]
        missing = [section for section in required_sections if section not in phase_data]
        
        if missing:
            raise ValueError(f"Cycle phase '{phase_name}' missing required sections: {missing}")
        
        # Validate favored sectors structure
        self._validate_favored_sectors(phase_data.get("favored_sectors", {}))
        
        # Validate strategies structure
        self._validate_strategies(phase_data.get("strategies", {}))
        
        return True
    
    def _validate_favored_sectors(self, favored_sectors: Dict[str, Any]) -> bool:
        """
        Validate the favored sectors section.
        
        Args:
            favored_sectors: Favored sectors data
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check for required sections
        required_sections = ["primary_sectors", "secondary_sectors", "sectors_to_avoid"]
        missing = [section for section in required_sections if section not in favored_sectors]
        
        if missing:
            logger.warning(f"Favored sectors missing recommended sections: {missing}")
        
        # Validate primary sectors
        for sector in favored_sectors.get("primary_sectors", []):
            if not isinstance(sector, dict) or "sector" not in sector:
                raise ValueError("Each primary sector must be a dictionary with at least a 'sector' key")
        
        # Validate secondary sectors
        for sector in favored_sectors.get("secondary_sectors", []):
            if not isinstance(sector, dict) or "sector" not in sector:
                raise ValueError("Each secondary sector must be a dictionary with at least a 'sector' key")
        
        # Validate sectors to avoid
        for sector in favored_sectors.get("sectors_to_avoid", []):
            if not isinstance(sector, dict) or "sector" not in sector:
                raise ValueError("Each sector to avoid must be a dictionary with at least a 'sector' key")
        
        return True
    
    def _validate_strategies(self, strategies: Dict[str, Any]) -> bool:
        """
        Validate the strategies section.
        
        Args:
            strategies: Strategies data
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check for recommended sections
        recommended_sections = ["equity_strategies", "options_strategies"]
        missing = [section for section in recommended_sections if section not in strategies]
        
        if missing:
            logger.warning(f"Strategies missing recommended sections: {missing}")
        
        # Check that strategies is a dictionary
        if not isinstance(strategies, dict):
            raise ValueError("Strategies must be a dictionary")
        
        # Validate equity strategies
        for strategy in strategies.get("equity_strategies", []):
            if not isinstance(strategy, dict) or "strategy" not in strategy or "implementation" not in strategy:
                raise ValueError("Each equity strategy must be a dictionary with at least 'strategy' and 'implementation' keys")
        
        # Validate options strategies
        for strategy in strategies.get("options_strategies", []):
            if not isinstance(strategy, dict) or "strategy" not in strategy or "implementation" not in strategy:
                raise ValueError("Each options strategy must be a dictionary with at least 'strategy' and 'implementation' keys")
        
        return True
    
    def get_framework(self) -> Dict[str, Any]:
        """
        Get the currently loaded framework data.
        
        Returns:
            Current framework data or empty dict if none loaded
        """
        return self.framework_data
    
    def save_to_file(self, file_path: str = None) -> bool:
        """
        Save the current framework data to file.
        
        Args:
            file_path: Path to save to (default: self.config_path)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.framework_data:
            logger.error("No framework data to save")
            return False
        
        file_path = file_path or self.config_path
        
        if not file_path:
            logger.error("No file path provided for saving framework")
            return False
        
        try:
            # Determine file type from extension
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() in ['.json']:
                with open(file_path, 'w') as f:
                    json.dump(self.framework_data, f, indent=2)
            elif ext.lower() in ['.yaml', '.yml']:
                with open(file_path, 'w') as f:
                    yaml.safe_dump(self.framework_data, f)
            else:
                logger.error(f"Unsupported file format: {ext}")
                return False
            
            logger.info(f"Successfully saved sector rotation framework to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving sector rotation framework: {str(e)}")
            return False 