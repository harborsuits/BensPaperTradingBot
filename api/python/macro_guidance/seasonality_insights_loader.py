"""
Seasonality Insights Framework Loader

This module handles loading and validating the seasonality insights framework data
that's used by the MacroGuidanceEngine for seasonality-based trading recommendations.
"""

import os
import json
import logging
import yaml
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SeasonalityInsightsLoader:
    """
    Handles loading and validating seasonality insights framework data.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the seasonality insights loader.
        
        Args:
            config_path: Optional path to seasonality configuration file
        """
        self.config_path = config_path
        self.framework_data = {}
        
        # Attempt to load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load seasonality insights framework data from file.
        
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
            
            logger.info(f"Successfully loaded seasonality insights framework from {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading seasonality insights framework: {str(e)}")
            raise
    
    def load_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load seasonality insights framework from dictionary data.
        
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
            
            logger.info("Successfully loaded seasonality insights framework from data")
            return data
            
        except Exception as e:
            logger.error(f"Error loading seasonality insights framework: {str(e)}")
            raise
    
    def validate_framework(self, data: Dict[str, Any]) -> bool:
        """
        Validate the seasonality insights framework data.
        
        Args:
            data: Framework data to validate
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check basic structure
        if not isinstance(data, dict):
            raise ValueError("Framework data must be a dictionary")
        
        if "seasonality_insights" not in data:
            raise ValueError("Framework data must contain a 'seasonality_insights' key")
        
        insights = data["seasonality_insights"]
        
        # Check for required metadata
        if "framework_version" not in insights:
            logger.warning("Framework data missing 'framework_version' metadata")
        
        if "last_updated" not in insights:
            logger.warning("Framework data missing 'last_updated' metadata")
        
        # Check for monthly patterns
        if "monthly_patterns" not in insights:
            raise ValueError("Framework must contain 'monthly_patterns'")
        
        monthly_patterns = insights["monthly_patterns"]
        if not isinstance(monthly_patterns, list):
            raise ValueError("monthly_patterns must be a list")
        
        # Validate each monthly pattern
        for pattern in monthly_patterns:
            self._validate_monthly_pattern(pattern)
        
        # Check for recurring patterns if present
        if "recurring_patterns" in insights:
            recurring_patterns = insights["recurring_patterns"]
            if not isinstance(recurring_patterns, list):
                raise ValueError("recurring_patterns must be a list")
            
            # Validate each recurring pattern
            for pattern in recurring_patterns:
                self._validate_recurring_pattern(pattern)
        
        return True
    
    def _validate_monthly_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Validate an individual monthly pattern.
        
        Args:
            pattern: Data for the monthly pattern
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check for required fields
        required_fields = ["month", "primary_asset_classes"]
        missing = [field for field in required_fields if field not in pattern]
        
        if missing:
            raise ValueError(f"Monthly pattern missing required fields: {missing}")
        
        # Validate primary asset classes
        if not isinstance(pattern["primary_asset_classes"], list):
            raise ValueError("primary_asset_classes must be a list")
        
        for asset_class in pattern["primary_asset_classes"]:
            if not isinstance(asset_class, dict) or "asset_class" not in asset_class:
                raise ValueError("Each asset class must be a dictionary with at least an 'asset_class' key")
        
        # Validate trading strategies if present
        if "trading_strategies" in pattern:
            strategies = pattern["trading_strategies"]
            if not isinstance(strategies, dict):
                raise ValueError("trading_strategies must be a dictionary")
            
            # Check for equity_strategies and options_strategies
            for strategy_type in ["equity_strategies", "options_strategies"]:
                if strategy_type in strategies:
                    if not isinstance(strategies[strategy_type], list):
                        raise ValueError(f"{strategy_type} must be a list")
                    
                    # Validate each strategy
                    for strategy in strategies[strategy_type]:
                        if not isinstance(strategy, dict) or "strategy" not in strategy or "implementation" not in strategy:
                            raise ValueError(f"Each {strategy_type} item must be a dictionary with 'strategy' and 'implementation' keys")
        
        return True
    
    def _validate_recurring_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Validate an individual recurring pattern.
        
        Args:
            pattern: Data for the recurring pattern
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check for required fields
        required_fields = ["pattern", "frequency", "primary_asset_classes"]
        missing = [field for field in required_fields if field not in pattern]
        
        if missing:
            raise ValueError(f"Recurring pattern missing required fields: {missing}")
        
        # Validate primary asset classes
        if not isinstance(pattern["primary_asset_classes"], list):
            raise ValueError("primary_asset_classes must be a list")
        
        for asset_class in pattern["primary_asset_classes"]:
            if not isinstance(asset_class, dict) or "asset_class" not in asset_class:
                raise ValueError("Each asset class must be a dictionary with at least an 'asset_class' key")
        
        # Validate trading strategies if present
        if "trading_strategies" in pattern:
            strategies = pattern["trading_strategies"]
            if not isinstance(strategies, dict):
                raise ValueError("trading_strategies must be a dictionary")
            
            # Check for equity_strategies and options_strategies
            for strategy_type in ["equity_strategies", "options_strategies"]:
                if strategy_type in strategies:
                    if not isinstance(strategies[strategy_type], list):
                        raise ValueError(f"{strategy_type} must be a list")
                    
                    # Validate each strategy
                    for strategy in strategies[strategy_type]:
                        if not isinstance(strategy, dict) or "strategy" not in strategy or "implementation" not in strategy:
                            raise ValueError(f"Each {strategy_type} item must be a dictionary with 'strategy' and 'implementation' keys")
        
        return True
    
    def get_framework(self) -> Dict[str, Any]:
        """
        Get the currently loaded framework data.
        
        Returns:
            Current framework data or empty dict if none loaded
        """
        return self.framework_data
    
    def get_monthly_pattern(self, month: str) -> Dict[str, Any]:
        """
        Get the pattern data for a specific month.
        
        Args:
            month: Month name (e.g., "January")
            
        Returns:
            Monthly pattern data or empty dict if not found
        """
        if not self.framework_data:
            return {}
        
        insights = self.framework_data.get("seasonality_insights", {})
        monthly_patterns = insights.get("monthly_patterns", [])
        
        for pattern in monthly_patterns:
            if pattern.get("month", "").lower() == month.lower():
                return pattern
        
        return {}
    
    def get_current_monthly_pattern(self) -> Dict[str, Any]:
        """
        Get the pattern data for the current month.
        
        Returns:
            Current month's pattern data or empty dict if not found
        """
        import datetime
        current_month = datetime.datetime.now().strftime("%B")
        return self.get_monthly_pattern(current_month)
    
    def get_recurring_pattern(self, pattern_name: str) -> Dict[str, Any]:
        """
        Get the data for a specific recurring pattern.
        
        Args:
            pattern_name: Name of the recurring pattern
            
        Returns:
            Recurring pattern data or empty dict if not found
        """
        if not self.framework_data:
            return {}
        
        insights = self.framework_data.get("seasonality_insights", {})
        recurring_patterns = insights.get("recurring_patterns", [])
        
        for pattern in recurring_patterns:
            if pattern.get("pattern", "").lower() == pattern_name.lower():
                return pattern
        
        return {}
    
    def get_active_recurring_patterns(self) -> List[Dict[str, Any]]:
        """
        Get all active recurring patterns based on current date and conditions.
        This is a simplified implementation and would be more sophisticated in practice.
        
        Returns:
            List of active recurring patterns
        """
        import datetime
        
        if not self.framework_data:
            return []
        
        insights = self.framework_data.get("seasonality_insights", {})
        recurring_patterns = insights.get("recurring_patterns", [])
        active_patterns = []
        
        now = datetime.datetime.now()
        current_month = now.strftime("%B")
        current_day = now.day
        current_weekday = now.weekday()  # 0-6 (Monday-Sunday)
        
        # Check each pattern for potential activation
        for pattern in recurring_patterns:
            pattern_name = pattern.get("pattern", "").lower()
            frequency = pattern.get("frequency", "").lower()
            
            # Simple checks - these would be more sophisticated in practice
            if "end-of-month" in pattern_name and current_day >= 25:
                active_patterns.append(pattern)
            elif "quad witching" in pattern_name and current_month in ["March", "June", "September", "December"] and 13 <= current_day <= 21:
                # Check if we're in the week of the third Friday
                active_patterns.append(pattern)
            elif "options expiration" in pattern_name and 13 <= current_day <= 21:
                # Check if we're near monthly options expiration (third Friday)
                active_patterns.append(pattern)
            elif "fed meeting" in pattern_name and current_month in ["January", "March", "May", "June", "July", "September", "November", "December"]:
                # Very simplified - would need to know actual Fed calendar
                active_patterns.append(pattern)
            elif "jobs report" in pattern_name and current_day <= 7 and current_weekday == 4:  # Friday
                # First Friday of the month
                active_patterns.append(pattern)
            elif "cpi release" in pattern_name and 10 <= current_day <= 20:
                # CPI typically released around middle of month
                active_patterns.append(pattern)
        
        return active_patterns
    
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
            
            logger.info(f"Successfully saved seasonality insights framework to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving seasonality insights framework: {str(e)}")
            return False 