"""
Model Registry

Handles registration, versioning, and management of prediction models for the
multi-model prediction pipeline.
"""

import os
import json
import logging
import joblib
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
import importlib.util

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for a registered model"""
    name: str
    type: str
    version: str
    created_at: datetime
    updated_at: datetime
    path: str
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class ModelRegistry:
    """
    Model Registry for managing trading prediction models
    
    Handles:
    - Model registration, versioning, and management
    - Model metadata storage and retrieval
    - Model performance tracking
    """
    
    def __init__(self, registry_path: str = "models"):
        """
        Initialize the model registry
        
        Args:
            registry_path: Base path for model storage
        """
        self.registry_path = registry_path
        self.models_metadata = {}
        self.active_models = {}
        
        # Create the registry directory if it doesn't exist
        os.makedirs(registry_path, exist_ok=True)
        os.makedirs(os.path.join(registry_path, "metadata"), exist_ok=True)
        
        # Load existing models metadata
        self._load_registry()
        
        logger.info(f"Model registry initialized at {registry_path}")
    
    def _load_registry(self):
        """Load model metadata from the registry"""
        metadata_path = os.path.join(self.registry_path, "metadata")
        if os.path.exists(metadata_path):
            for filename in os.listdir(metadata_path):
                if filename.endswith(".json"):
                    try:
                        with open(os.path.join(metadata_path, filename), 'r') as f:
                            metadata_dict = json.load(f)
                            
                            # Convert timestamps
                            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                            metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                            
                            # Create metadata object
                            metadata = ModelMetadata(**metadata_dict)
                            self.models_metadata[metadata.name] = metadata
                            
                            logger.debug(f"Loaded model metadata: {metadata.name} (v{metadata.version})")
                    except Exception as e:
                        logger.error(f"Error loading model metadata from {filename}: {e}")
    
    def register_model(self, 
                      name: str, 
                      model: Any, 
                      model_type: str, 
                      description: str = "", 
                      parameters: Dict[str, Any] = None, 
                      tags: List[str] = None, 
                      metrics: Dict[str, float] = None,
                      version: Optional[str] = None) -> ModelMetadata:
        """
        Register a new model or update an existing one
        
        Args:
            name: Model name
            model: Model object
            model_type: Type of model (ml, statistical, rule_based, hybrid, ensemble)
            description: Description of the model
            parameters: Parameters used to train the model
            tags: Tags for categorizing the model
            metrics: Performance metrics
            version: Optional version string (default: auto-increment)
            
        Returns:
            ModelMetadata object
        """
        # Determine version
        if version is None:
            if name in self.models_metadata:
                # Auto-increment version
                prev_version = self.models_metadata[name].version
                try:
                    # Try to increment numeric version
                    version_num = int(prev_version)
                    version = str(version_num + 1)
                except ValueError:
                    # If not numeric, just append timestamp
                    version = f"{prev_version}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            else:
                # First version
                version = "1"
        
        # Save the model
        model_path = self._save_model(name, model, model_type, version)
        
        # Create metadata
        now = datetime.now()
        metadata = ModelMetadata(
            name=name,
            type=model_type,
            version=version,
            created_at=now,
            updated_at=now,
            path=model_path,
            description=description,
            metrics=metrics or {},
            parameters=parameters or {},
            tags=tags or []
        )
        
        # Save metadata
        self._save_metadata(metadata)
        
        # Update registry
        self.models_metadata[name] = metadata
        
        logger.info(f"Registered model: {name} (v{version})")
        
        return metadata
    
    def _save_model(self, name: str, model: Any, model_type: str, version: str) -> str:
        """
        Save a model to the registry
        
        Args:
            name: Model name
            model: Model object
            model_type: Type of model
            version: Version string
            
        Returns:
            Path to the saved model
        """
        # Create model directory
        model_dir = os.path.join(self.registry_path, name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine save path and method based on model type
        if model_type == 'ml':
            # Save scikit-learn or similar model
            model_path = os.path.join(model_dir, f"{name}_v{version}.joblib")
            joblib.dump(model, model_path)
        elif model_type in ['rule_based', 'statistical', 'hybrid']:
            # Save Python module
            if isinstance(model, str) and os.path.exists(model):
                # If model is a path to a Python file, just copy it
                model_path = os.path.join(model_dir, f"{name}_v{version}.py")
                shutil.copy(model, model_path)
            else:
                # Serialize parameters to recreate the model
                model_path = os.path.join(model_dir, f"{name}_v{version}.json")
                with open(model_path, 'w') as f:
                    json.dump({
                        'class_name': model.__class__.__name__,
                        'module_path': model.__class__.__module__,
                        'parameters': getattr(model, 'get_params', lambda: {})()
                    }, f, indent=2)
        else:
            # Save as pickle for unknown types
            model_path = os.path.join(model_dir, f"{name}_v{version}.joblib")
            joblib.dump(model, model_path)
        
        return model_path
    
    def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to disk"""
        metadata_path = os.path.join(self.registry_path, "metadata", f"{metadata.name}.json")
        
        # Convert datetime to ISO format for serialization
        metadata_dict = metadata.__dict__.copy()
        metadata_dict['created_at'] = metadata_dict['created_at'].isoformat()
        metadata_dict['updated_at'] = metadata_dict['updated_at'].isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    def load_model(self, name: str, version: Optional[str] = None) -> Any:
        """
        Load a model from the registry
        
        Args:
            name: Model name
            version: Optional version (default: latest)
            
        Returns:
            Loaded model object
        """
        if name not in self.models_metadata:
            raise ValueError(f"Model {name} not found in registry")
        
        # Get metadata
        metadata = self.models_metadata[name]
        
        # Use specific version if provided
        if version is not None:
            # Find version in registry
            model_dir = os.path.join(self.registry_path, name)
            for filename in os.listdir(model_dir):
                if filename.endswith(f"_v{version}.joblib") or filename.endswith(f"_v{version}.py") or filename.endswith(f"_v{version}.json"):
                    metadata.path = os.path.join(model_dir, filename)
                    break
            
            if not os.path.exists(metadata.path):
                raise ValueError(f"Version {version} of model {name} not found")
        
        # Check if already loaded
        if name in self.active_models:
            return self.active_models[name]
        
        # Load model based on type
        if metadata.path.endswith('.joblib'):
            model = joblib.load(metadata.path)
        elif metadata.path.endswith('.py'):
            # Load Python module
            module_name = f"{name}_model_v{metadata.version}"
            spec = importlib.util.spec_from_file_location(module_name, metadata.path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find model class
            model_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    # Simple heuristic: check if it's a class with apply or predict method
                    if hasattr(attr, 'apply') or hasattr(attr, 'predict') or hasattr(attr, 'analyze'):
                        model_class = attr
                        break
            
            if model_class is None:
                raise ValueError(f"No model class found in {metadata.path}")
                
            # Instantiate model
            model = model_class(**metadata.parameters)
        elif metadata.path.endswith('.json'):
            # Load from parameters
            with open(metadata.path, 'r') as f:
                model_info = json.load(f)
            
            # Import module
            module = importlib.import_module(model_info['module_path'])
            model_class = getattr(module, model_info['class_name'])
            
            # Instantiate model
            model = model_class(**model_info['parameters'])
        else:
            raise ValueError(f"Unsupported model format: {metadata.path}")
        
        # Cache model
        self.active_models[name] = model
        
        logger.info(f"Loaded model: {name} (v{metadata.version})")
        
        return model
    
    def get_model_info(self, name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a model
        
        Args:
            name: Model name
            
        Returns:
            ModelMetadata object or None if not found
        """
        return self.models_metadata.get(name)
    
    def list_models(self, model_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """
        List all registered models, optionally filtered by type or tags
        
        Args:
            model_type: Optional model type filter
            tags: Optional list of tags to filter by (models must have ALL tags)
            
        Returns:
            List of ModelMetadata objects
        """
        models = list(self.models_metadata.values())
        
        # Filter by type
        if model_type:
            models = [m for m in models if m.type == model_type]
        
        # Filter by tags
        if tags:
            models = [m for m in models if all(tag in m.tags for tag in tags)]
        
        return models
    
    def delete_model(self, name: str, version: Optional[str] = None):
        """
        Delete a model from the registry
        
        Args:
            name: Model name
            version: Optional version (default: all versions)
        """
        if name not in self.models_metadata:
            raise ValueError(f"Model {name} not found in registry")
        
        model_dir = os.path.join(self.registry_path, name)
        
        if version:
            # Delete specific version
            for filename in os.listdir(model_dir):
                if filename.endswith(f"_v{version}.joblib") or filename.endswith(f"_v{version}.py") or filename.endswith(f"_v{version}.json"):
                    os.remove(os.path.join(model_dir, filename))
                    break
        else:
            # Delete all versions
            shutil.rmtree(model_dir)
            
            # Remove metadata
            metadata_path = os.path.join(self.registry_path, "metadata", f"{name}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Remove from cache
            if name in self.active_models:
                del self.active_models[name]
            
            # Remove from registry
            if name in self.models_metadata:
                del self.models_metadata[name]
        
        logger.info(f"Deleted model: {name}")
    
    def update_model_metrics(self, name: str, metrics: Dict[str, float]):
        """
        Update performance metrics for a model
        
        Args:
            name: Model name
            metrics: Dictionary of metric names and values
        """
        if name not in self.models_metadata:
            raise ValueError(f"Model {name} not found in registry")
        
        # Update in-memory metadata
        self.models_metadata[name].metrics.update(metrics)
        self.models_metadata[name].updated_at = datetime.now()
        
        # Save updated metadata
        self._save_metadata(self.models_metadata[name])
        
        logger.info(f"Updated metrics for model: {name}")
    
    def get_model_versions(self, name: str) -> List[str]:
        """
        Get all versions of a model
        
        Args:
            name: Model name
            
        Returns:
            List of version strings
        """
        if name not in self.models_metadata:
            return []
        
        model_dir = os.path.join(self.registry_path, name)
        if not os.path.exists(model_dir):
            return []
        
        versions = []
        for filename in os.listdir(model_dir):
            if "_v" in filename:
                try:
                    version = filename.split("_v")[1].split(".")[0]
                    versions.append(version)
                except Exception:
                    pass
        
        return sorted(versions)
    
    def export_model(self, name: str, export_path: str, version: Optional[str] = None):
        """
        Export a model to a specified path
        
        Args:
            name: Model name
            export_path: Path to export to
            version: Optional version (default: latest)
        """
        if name not in self.models_metadata:
            raise ValueError(f"Model {name} not found in registry")
        
        # Get metadata
        metadata = self.models_metadata[name]
        
        # Use specific version if provided
        if version is not None:
            # Find version in registry
            model_dir = os.path.join(self.registry_path, name)
            for filename in os.listdir(model_dir):
                if filename.endswith(f"_v{version}.joblib") or filename.endswith(f"_v{version}.py") or filename.endswith(f"_v{version}.json"):
                    model_path = os.path.join(model_dir, filename)
                    break
            
            if not os.path.exists(model_path):
                raise ValueError(f"Version {version} of model {name} not found")
        else:
            model_path = metadata.path
        
        # Copy model file
        shutil.copy(model_path, export_path)
        
        # Save metadata
        metadata_path = os.path.join(os.path.dirname(export_path), f"{name}_metadata.json")
        
        # Convert datetime to ISO format for serialization
        metadata_dict = metadata.__dict__.copy()
        metadata_dict['created_at'] = metadata_dict['created_at'].isoformat()
        metadata_dict['updated_at'] = metadata_dict['updated_at'].isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        logger.info(f"Exported model: {name} to {export_path}")
        
    def import_model(self, model_path: str, metadata_path: Optional[str] = None) -> ModelMetadata:
        """
        Import a model from a specified path
        
        Args:
            model_path: Path to model file
            metadata_path: Optional path to metadata file
            
        Returns:
            ModelMetadata object
        """
        # Extract model info from filename
        filename = os.path.basename(model_path)
        name = filename.split('_v')[0] if '_v' in filename else os.path.splitext(filename)[0]
        
        # Load metadata if provided
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                
                # Convert timestamps
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                
                # Create metadata object
                metadata = ModelMetadata(**metadata_dict)
        else:
            # Create basic metadata
            extension = os.path.splitext(filename)[1]
            if extension == '.joblib':
                model_type = 'ml'
            elif extension == '.py':
                model_type = 'rule_based'  # Assume rule-based for Python files
            elif extension == '.json':
                model_type = 'hybrid'      # Assume hybrid for JSON files
            else:
                model_type = 'unknown'
            
            version = "1"
            if '_v' in filename:
                try:
                    version = filename.split('_v')[1].split('.')[0]
                except Exception:
                    pass
            
            # Create metadata
            now = datetime.now()
            metadata = ModelMetadata(
                name=name,
                type=model_type,
                version=version,
                created_at=now,
                updated_at=now,
                path=model_path,
                description=f"Imported model: {name}",
                metrics={},
                parameters={},
                tags=["imported"]
            )
        
        # Copy model to registry
        model_dir = os.path.join(self.registry_path, name)
        os.makedirs(model_dir, exist_ok=True)
        
        new_model_path = os.path.join(model_dir, filename)
        shutil.copy(model_path, new_model_path)
        
        # Update path in metadata
        metadata.path = new_model_path
        
        # Save metadata
        self._save_metadata(metadata)
        
        # Update registry
        self.models_metadata[name] = metadata
        
        logger.info(f"Imported model: {name} (v{metadata.version})")
        
        return metadata
    
    def clear_cache(self):
        """Clear the model cache"""
        self.active_models = {}
        logger.info("Cleared model cache")
