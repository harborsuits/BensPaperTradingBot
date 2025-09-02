"""
Advanced Feature Generator Framework

This module provides a configurable, extensible feature generation system with:
- Feature sets defined via configuration
- Multi-source data merging capabilities
- Automated feature creation and selection
- Concurrent processing for high-throughput
"""

import pandas as pd
import numpy as np
import logging
import json
import importlib
import concurrent.futures
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
import joblib
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import existing feature engineering framework for base capabilities
from trading_bot.ml_pipeline.feature_engineering import FeatureEngineeringFramework

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    """Configuration for a set of related features"""
    name: str
    source: str  # Data source identifier
    features: List[str]  # Feature names to include
    transforms: List[Dict[str, Any]] = field(default_factory=list)  # Transformations to apply
    dependencies: List[str] = field(default_factory=list)  # Other feature sets this depends on
    importance_threshold: float = 0.01  # Minimum feature importance to keep
    enabled: bool = True

@dataclass
class DataSourceConfig:
    """Configuration for a data source"""
    name: str
    type: str  # Type of data source (ohlcv, alternative, fundamental, etc)
    params: Dict[str, Any] = field(default_factory=dict)  # Parameters for data fetching
    features: List[str] = field(default_factory=list)  # Base features to extract
    preprocessing: List[Dict[str, Any]] = field(default_factory=list)  # Preprocessing steps

class AdvancedFeatureGenerator:
    """
    Advanced feature generation framework that supports configuration-based
    feature creation, multi-source data merging, and automated feature selection.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced feature generator
        
        Args:
            config_path: Path to configuration file
            config: Configuration dictionary (overrides config_path)
        """
        self.config = config or {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
        # Initialize base feature engineering framework
        self.base_framework = FeatureEngineeringFramework(self.config.get('base_config', {}))
        
        # Initialize feature sets from config
        self.feature_sets = {}
        self.data_sources = {}
        self._parse_config()
        
        # Cache for feature importance
        self.feature_importance = {}
        
        # Feature selection models
        self.feature_selectors = {}
        
        logger.info("Advanced Feature Generator initialized")
    
    def _parse_config(self):
        """Parse configuration to create feature sets and data sources"""
        # Parse data sources
        sources_config = self.config.get('data_sources', [])
        for source in sources_config:
            source_obj = DataSourceConfig(
                name=source['name'],
                type=source['type'],
                params=source.get('params', {}),
                features=source.get('features', []),
                preprocessing=source.get('preprocessing', [])
            )
            self.data_sources[source['name']] = source_obj
            logger.debug(f"Registered data source: {source['name']}")
        
        # Parse feature sets
        feature_sets = self.config.get('feature_sets', [])
        for fs in feature_sets:
            feature_set = FeatureSet(
                name=fs['name'],
                source=fs['source'],
                features=fs.get('features', []),
                transforms=fs.get('transforms', []),
                dependencies=fs.get('dependencies', []),
                importance_threshold=fs.get('importance_threshold', 0.01),
                enabled=fs.get('enabled', True)
            )
            self.feature_sets[fs['name']] = feature_set
            logger.debug(f"Registered feature set: {fs['name']}")
    
    def generate_features(self, data_dict: Dict[str, pd.DataFrame], target_variable: Optional[str] = None) -> pd.DataFrame:
        """
        Generate features from multiple data sources
        
        Args:
            data_dict: Dictionary of DataFrames with source name as key
            target_variable: Optional target variable for supervised feature selection
            
        Returns:
            DataFrame with all generated features
        """
        # Process each data source
        processed_data = {}
        for source_name, source_config in self.data_sources.items():
            if source_name in data_dict:
                df = data_dict[source_name].copy()
                # Apply preprocessing
                for preproc in source_config.preprocessing:
                    method = preproc['method']
                    params = preproc.get('params', {})
                    df = self._apply_preprocessing(df, method, params)
                processed_data[source_name] = df
        
        # Generate feature sets in dependency order
        feature_generation_order = self._get_feature_generation_order()
        all_features = pd.DataFrame(index=data_dict[list(data_dict.keys())[0]].index)
        
        for fs_name in feature_generation_order:
            feature_set = self.feature_sets[fs_name]
            if not feature_set.enabled:
                continue
                
            source_df = processed_data.get(feature_set.source)
            if source_df is None:
                logger.warning(f"Data source {feature_set.source} not available for feature set {fs_name}")
                continue
            
            # Generate base features using existing framework
            base_features = self.base_framework.generate_features(source_df)
            
            # Extract requested features
            selected_features = base_features[feature_set.features] if feature_set.features else base_features
            
            # Apply transformations
            for transform in feature_set.transforms:
                method = transform['method']
                params = transform.get('params', {})
                selected_features = self._apply_transformation(selected_features, method, params)
            
            # Add to master feature set
            prefix = f"{fs_name}_" if self.config.get('use_prefixes', True) else ""
            for col in selected_features.columns:
                all_features[f"{prefix}{col}"] = selected_features[col]
        
        # Apply feature selection if target is provided
        if target_variable and target_variable in all_features.columns:
            all_features = self._select_features(all_features, target_variable)
        
        return all_features
    
    def _get_feature_generation_order(self) -> List[str]:
        """
        Determine the order to generate feature sets based on dependencies
        
        Returns:
            List of feature set names in dependency order
        """
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(fs_name):
            if fs_name in visited:
                return
            if fs_name in temp_visited:
                logger.warning(f"Circular dependency detected for feature set {fs_name}")
                return
            
            temp_visited.add(fs_name)
            
            fs = self.feature_sets.get(fs_name)
            if fs:
                for dep in fs.dependencies:
                    visit(dep)
            
            temp_visited.remove(fs_name)
            visited.add(fs_name)
            result.append(fs_name)
        
        for fs_name in self.feature_sets:
            visit(fs_name)
            
        return result
    
    def _apply_preprocessing(self, df: pd.DataFrame, method: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply preprocessing method to DataFrame"""
        if method == 'fillna':
            return df.fillna(**params)
        elif method == 'normalize':
            for col in params.get('columns', df.columns):
                if col in df.columns:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
            return df
        elif method == 'resample':
            return df.resample(params.get('rule', '1D')).agg(params.get('agg', 'last'))
        elif method == 'custom':
            # Import and apply custom preprocessing function
            module_path = params.get('module', '')
            function_name = params.get('function', '')
            if module_path and function_name:
                try:
                    module = importlib.import_module(module_path)
                    custom_func = getattr(module, function_name)
                    return custom_func(df, **params.get('kwargs', {}))
                except Exception as e:
                    logger.error(f"Error applying custom preprocessing: {e}")
        return df
    
    def _apply_transformation(self, df: pd.DataFrame, method: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Apply transformation method to DataFrame"""
        result = df.copy()
        
        if method == 'rolling':
            window = params.get('window', 5)
            function = params.get('function', 'mean')
            cols = params.get('columns', df.columns)
            for col in cols:
                if col in df.columns:
                    result[f"{col}_{function}_{window}"] = getattr(df[col].rolling(window=window), function)()
        
        elif method == 'lag':
            periods = params.get('periods', [1, 5, 10])
            cols = params.get('columns', df.columns)
            for col in cols:
                if col in df.columns:
                    for period in periods:
                        result[f"{col}_lag_{period}"] = df[col].shift(period)
        
        elif method == 'diff':
            periods = params.get('periods', [1])
            cols = params.get('columns', df.columns)
            for col in cols:
                if col in df.columns:
                    for period in periods:
                        result[f"{col}_diff_{period}"] = df[col].diff(period)
        
        elif method == 'pct_change':
            periods = params.get('periods', [1])
            cols = params.get('columns', df.columns)
            for col in cols:
                if col in df.columns:
                    for period in periods:
                        result[f"{col}_pct_{period}"] = df[col].pct_change(period)
        
        elif method == 'ta':
            # Apply technical indicators from base framework
            # This is just a reference to the underlying indicator
            indicator = params.get('indicator', '')
            args = params.get('args', {})
            if hasattr(self.base_framework, f"_calculate_{indicator}"):
                func = getattr(self.base_framework, f"_calculate_{indicator}")
                result[indicator] = func(df, **args)
        
        elif method == 'custom':
            # Import and apply custom transformation
            module_path = params.get('module', '')
            function_name = params.get('function', '')
            if module_path and function_name:
                try:
                    module = importlib.import_module(module_path)
                    custom_func = getattr(module, function_name)
                    transformed = custom_func(df, **params.get('kwargs', {}))
                    # Merge with result if it's a DataFrame, otherwise assume it's a Series
                    if isinstance(transformed, pd.DataFrame):
                        result = pd.concat([result, transformed], axis=1)
                    else:
                        result[params.get('name', function_name)] = transformed
                except Exception as e:
                    logger.error(f"Error applying custom transformation: {e}")
        
        return result
    
    def _select_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Select relevant features using various feature selection methods
        
        Args:
            df: DataFrame with features
            target_column: Target variable for supervised selection
            
        Returns:
            DataFrame with selected features
        """
        selection_config = self.config.get('feature_selection', {})
        method = selection_config.get('method', 'importance')
        n_features = selection_config.get('n_features', None)
        threshold = selection_config.get('threshold', 0.01)
        
        if method == 'none' or not target_column:
            return df
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        if method == 'importance':
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Determine if classification or regression
            is_classification = selection_config.get('is_classification', False)
            
            # Train model
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            model.fit(X.fillna(0), y)
            
            # Get feature importance
            importances = pd.Series(model.feature_importances_, index=X.columns)
            self.feature_importance = importances.sort_values(ascending=False)
            
            # Select features
            if n_features:
                selected_features = self.feature_importance.nlargest(n_features).index
            else:
                selected_features = self.feature_importance[self.feature_importance >= threshold].index
                
            # Keep target and selected features
            return df[[target_column] + list(selected_features)]
        
        elif method == 'kbest':
            if n_features is None:
                n_features = min(50, X.shape[1])
                
            # Choose score function
            is_classification = selection_config.get('is_classification', False)
            score_func = f_classif if is_classification else mutual_info_regression
            
            # Create and fit selector
            selector = SelectKBest(score_func=score_func, k=n_features)
            self.feature_selectors['kbest'] = selector
            selector.fit(X.fillna(0), y)
            
            # Get selected feature mask
            feature_mask = selector.get_support()
            selected_features = X.columns[feature_mask]
            
            # Keep target and selected features
            return df[[target_column] + list(selected_features)]
        
        elif method == 'pca':
            if n_features is None:
                n_features = min(20, X.shape[1])
                
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.fillna(0))
            
            # Apply PCA
            pca = PCA(n_components=n_features)
            self.feature_selectors['pca'] = pca
            X_pca = pca.fit_transform(X_scaled)
            
            # Create new DataFrame with PCA components
            pca_df = pd.DataFrame(
                X_pca, 
                columns=[f'PC{i+1}' for i in range(n_features)],
                index=df.index
            )
            
            # Add target column
            pca_df[target_column] = y
            
            return pca_df
        
        return df
    
    def save_feature_config(self, path: str):
        """Save feature configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def save_feature_selectors(self, path: str):
        """Save feature selectors to file"""
        joblib.dump(self.feature_selectors, path)
    
    def load_feature_selectors(self, path: str):
        """Load feature selectors from file"""
        self.feature_selectors = joblib.load(path)
    
    def parallel_generate_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate features in parallel using ThreadPoolExecutor
        
        Args:
            data_dict: Dictionary of DataFrames with source name as key
            
        Returns:
            DataFrame with all generated features
        """
        # Extract feature sets by source
        source_feature_sets = {}
        for fs_name, fs in self.feature_sets.items():
            if fs.source not in source_feature_sets:
                source_feature_sets[fs.source] = []
            source_feature_sets[fs.source].append(fs_name)
        
        results = {}
        
        # Process each source in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for source, fs_names in source_feature_sets.items():
                if source in data_dict:
                    futures[executor.submit(self._process_source, data_dict[source], fs_names)] = source
            
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                try:
                    source_features = future.result()
                    results.update(source_features)
                except Exception as e:
                    logger.error(f"Error processing source {source}: {e}")
        
        # Combine results into single DataFrame
        if not results:
            return pd.DataFrame()
            
        # Find common index
        common_index = None
        for df in results.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        # Combine features
        all_features = pd.DataFrame(index=common_index)
        for feature_name, series in results.items():
            all_features[feature_name] = series.reindex(common_index)
            
        return all_features
    
    def _process_source(self, source_df: pd.DataFrame, feature_set_names: List[str]) -> Dict[str, pd.Series]:
        """
        Process a single data source for multiple feature sets
        
        Args:
            source_df: DataFrame for the source
            feature_set_names: List of feature set names to process
            
        Returns:
            Dictionary of features as Series
        """
        result = {}
        
        # Get base features using existing framework
        base_features = self.base_framework.generate_features(source_df)
        
        for fs_name in feature_set_names:
            feature_set = self.feature_sets[fs_name]
            if not feature_set.enabled:
                continue
                
            # Extract requested features
            if feature_set.features:
                available_features = [f for f in feature_set.features if f in base_features.columns]
                selected_features = base_features[available_features].copy() if available_features else pd.DataFrame(index=base_features.index)
            else:
                selected_features = base_features.copy()
            
            # Apply transformations
            for transform in feature_set.transforms:
                method = transform['method']
                params = transform.get('params', {})
                selected_features = self._apply_transformation(selected_features, method, params)
            
            # Add to result with prefix
            prefix = f"{fs_name}_" if self.config.get('use_prefixes', True) else ""
            for col in selected_features.columns:
                result[f"{prefix}{col}"] = selected_features[col]
                
        return result
