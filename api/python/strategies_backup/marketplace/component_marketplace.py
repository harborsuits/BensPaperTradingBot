"""
Component Marketplace

This module provides functionality for sharing and importing strategy components.
It supports local and remote component repositories, versioning, and security checks.
"""

import os
import json
import hashlib
import logging
import requests
import zipfile
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import uuid

from trading_bot.strategies.modular_strategy_system import ComponentType
from trading_bot.strategies.components.component_registry import get_component_registry

logger = logging.getLogger(__name__)

# Default locations
DEFAULT_LOCAL_REPO = os.path.join(os.path.dirname(__file__), '../../../config/component_repo')
DEFAULT_REMOTE_URL = "https://api.your-trading-platform.com/component-marketplace"  # Change to your actual API

class ComponentMetadata:
    """Metadata for marketplace components."""
    
    def __init__(self, 
                component_id: str,
                name: str,
                description: str,
                component_type: str,
                author: str,
                version: str,
                tags: List[str] = None,
                created_at: Optional[str] = None,
                updated_at: Optional[str] = None,
                dependencies: List[str] = None,
                compatibility: Dict[str, str] = None,
                source_code: Optional[str] = None,
                hash_value: Optional[str] = None):
        """
        Initialize component metadata
        
        Args:
            component_id: Unique identifier
            name: Component name
            description: Component description
            component_type: Component type
            author: Component author
            version: Component version
            tags: Component tags
            created_at: Creation date (ISO format)
            updated_at: Last update date (ISO format)
            dependencies: List of dependencies
            compatibility: Compatibility information
            source_code: Component source code
            hash_value: Hash of the component code
        """
        self.component_id = component_id
        self.name = name
        self.description = description
        self.component_type = component_type
        self.author = author
        self.version = version
        self.tags = tags or []
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or self.created_at
        self.dependencies = dependencies or []
        self.compatibility = compatibility or {}
        self.source_code = source_code
        self.hash_value = hash_value or self._generate_hash()
    
    def _generate_hash(self) -> str:
        """
        Generate hash for the component
        
        Returns:
            Hash string
        """
        if not self.source_code:
            return ""
        
        # Create hash from source code and metadata
        hash_input = (
            self.source_code + 
            self.name + 
            self.description + 
            self.version + 
            self.author
        )
        
        return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            'component_id': self.component_id,
            'name': self.name,
            'description': self.description,
            'component_type': self.component_type,
            'author': self.author,
            'version': self.version,
            'tags': self.tags,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'dependencies': self.dependencies,
            'compatibility': self.compatibility,
            'hash': self.hash_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentMetadata':
        """
        Create metadata from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            ComponentMetadata instance
        """
        return cls(
            component_id=data.get('component_id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            component_type=data.get('component_type', ''),
            author=data.get('author', ''),
            version=data.get('version', ''),
            tags=data.get('tags', []),
            created_at=data.get('created_at', ''),
            updated_at=data.get('updated_at', ''),
            dependencies=data.get('dependencies', []),
            compatibility=data.get('compatibility', {}),
            hash_value=data.get('hash', '')
        )

class ComponentPackage:
    """Component package for marketplace."""
    
    def __init__(self, 
                metadata: ComponentMetadata,
                source_code: str,
                dependencies: Dict[str, str] = None,
                examples: List[Dict[str, Any]] = None):
        """
        Initialize component package
        
        Args:
            metadata: Component metadata
            source_code: Component source code
            dependencies: Dependencies versions
            examples: Example configurations
        """
        self.metadata = metadata
        self.source_code = source_code
        self.dependencies = dependencies or {}
        self.examples = examples or []
        
        # Update metadata with source code
        self.metadata.source_code = source_code
        self.metadata.hash_value = self.metadata._generate_hash()
    
    def package(self) -> Dict[str, Any]:
        """
        Package component for distribution
        
        Returns:
            Package dictionary
        """
        return {
            'metadata': self.metadata.to_dict(),
            'source_code': self.source_code,
            'dependencies': self.dependencies,
            'examples': self.examples
        }
    
    def save(self, file_path: str) -> bool:
        """
        Save package to file
        
        Args:
            file_path: Path to save package
            
        Returns:
            Success flag
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save package to file
            with open(file_path, 'w') as f:
                json.dump(self.package(), f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving package: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> Optional['ComponentPackage']:
        """
        Load package from file
        
        Args:
            file_path: Path to load package from
            
        Returns:
            ComponentPackage instance or None if loading failed
        """
        try:
            # Load package from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create component package
            metadata = ComponentMetadata.from_dict(data.get('metadata', {}))
            source_code = data.get('source_code', '')
            dependencies = data.get('dependencies', {})
            examples = data.get('examples', [])
            
            return cls(metadata, source_code, dependencies, examples)
        except Exception as e:
            logger.error(f"Error loading package: {e}")
            return None

class ComponentMarketplace:
    """Marketplace for trading strategy components."""
    
    def __init__(self, 
                local_repo: str = DEFAULT_LOCAL_REPO,
                remote_url: str = DEFAULT_REMOTE_URL,
                api_key: Optional[str] = None):
        """
        Initialize component marketplace
        
        Args:
            local_repo: Path to local repository
            remote_url: URL of remote repository
            api_key: API key for remote repository
        """
        self.local_repo = local_repo
        self.remote_url = remote_url
        self.api_key = api_key
        self.registry = get_component_registry()
        
        # Ensure local repository exists
        os.makedirs(self.local_repo, exist_ok=True)
        
        # Component type directories
        for comp_type in ComponentType:
            os.makedirs(os.path.join(self.local_repo, comp_type.name), exist_ok=True)
    
    def list_local_components(self, component_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List components in local repository
        
        Args:
            component_type: Component type filter
            
        Returns:
            List of component metadata
        """
        components = []
        
        # Determine directories to search
        if component_type:
            type_dirs = [os.path.join(self.local_repo, component_type)]
        else:
            type_dirs = [
                os.path.join(self.local_repo, comp_type.name) 
                for comp_type in ComponentType
            ]
        
        # Search directories
        for type_dir in type_dirs:
            if not os.path.exists(type_dir):
                continue
            
            for filename in os.listdir(type_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(type_dir, filename)
                    
                    try:
                        # Load package from file
                        package = ComponentPackage.load(file_path)
                        if package:
                            components.append(package.metadata.to_dict())
                    except Exception as e:
                        logger.warning(f"Error loading {file_path}: {e}")
        
        return components
    
    def search_local_components(self, query: str, 
                              component_type: Optional[str] = None,
                              tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search components in local repository
        
        Args:
            query: Search query
            component_type: Component type filter
            tags: Tags filter
            
        Returns:
            List of matching component metadata
        """
        components = self.list_local_components(component_type)
        results = []
        
        query = query.lower()
        
        for component in components:
            # Check if component matches query
            name_match = query in component.get('name', '').lower()
            desc_match = query in component.get('description', '').lower()
            
            # Check if component matches tags
            tag_match = True
            if tags:
                component_tags = component.get('tags', [])
                tag_match = all(tag.lower() in [t.lower() for t in component_tags] for tag in tags)
            
            if (name_match or desc_match) and tag_match:
                results.append(component)
        
        return results
    
    def get_local_component(self, component_id: str) -> Optional[ComponentPackage]:
        """
        Get component from local repository
        
        Args:
            component_id: Component ID
            
        Returns:
            ComponentPackage instance or None if not found
        """
        # Search in all component type directories
        for comp_type in ComponentType:
            type_dir = os.path.join(self.local_repo, comp_type.name)
            if not os.path.exists(type_dir):
                continue
            
            # Check for component file
            file_path = os.path.join(type_dir, f"{component_id}.json")
            if os.path.exists(file_path):
                return ComponentPackage.load(file_path)
        
        return None
    
    def publish_component(self, package: ComponentPackage, overwrite: bool = False) -> bool:
        """
        Publish component to local repository
        
        Args:
            package: Component package
            overwrite: Whether to overwrite existing component
            
        Returns:
            Success flag
        """
        # Check if component already exists
        component_id = package.metadata.component_id
        existing = self.get_local_component(component_id)
        
        if existing and not overwrite:
            logger.warning(f"Component {component_id} already exists. Use overwrite=True to replace it.")
            return False
        
        # Determine component type directory
        type_dir = os.path.join(self.local_repo, package.metadata.component_type)
        
        # Save package to file
        file_path = os.path.join(type_dir, f"{component_id}.json")
        return package.save(file_path)
    
    def import_component_from_code(self, 
                                 source_code: str,
                                 name: str,
                                 description: str,
                                 component_type: str,
                                 author: str,
                                 version: str = "1.0.0",
                                 tags: List[str] = None) -> Optional[str]:
        """
        Import component from source code
        
        Args:
            source_code: Component source code
            name: Component name
            description: Component description
            component_type: Component type
            author: Component author
            version: Component version
            tags: Component tags
            
        Returns:
            Component ID or None if import failed
        """
        # Create component ID
        component_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = ComponentMetadata(
            component_id=component_id,
            name=name,
            description=description,
            component_type=component_type,
            author=author,
            version=version,
            tags=tags or []
        )
        
        # Create package
        package = ComponentPackage(
            metadata=metadata,
            source_code=source_code
        )
        
        # Publish component
        if self.publish_component(package):
            return component_id
        
        return None
    
    def delete_local_component(self, component_id: str) -> bool:
        """
        Delete component from local repository
        
        Args:
            component_id: Component ID
            
        Returns:
            Success flag
        """
        # Search in all component type directories
        for comp_type in ComponentType:
            type_dir = os.path.join(self.local_repo, comp_type.name)
            if not os.path.exists(type_dir):
                continue
            
            # Check for component file
            file_path = os.path.join(type_dir, f"{component_id}.json")
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    return True
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
                    return False
        
        return False
    
    def search_remote_components(self, query: str, 
                               component_type: Optional[str] = None,
                               tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search components in remote repository
        
        Args:
            query: Search query
            component_type: Component type filter
            tags: Tags filter
            
        Returns:
            List of matching component metadata
        """
        try:
            # Prepare request parameters
            params = {'query': query}
            
            if component_type:
                params['component_type'] = component_type
            
            if tags:
                params['tags'] = ','.join(tags)
            
            # Add API key if available
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send request
            response = requests.get(
                f"{self.remote_url}/search",
                params=params,
                headers=headers
            )
            
            # Check response
            if response.status_code == 200:
                return response.json().get('components', [])
            else:
                logger.error(f"Error searching remote components: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error searching remote components: {e}")
            return []
    
    def download_remote_component(self, component_id: str) -> Optional[ComponentPackage]:
        """
        Download component from remote repository
        
        Args:
            component_id: Component ID
            
        Returns:
            ComponentPackage instance or None if download failed
        """
        try:
            # Add API key if available
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send request
            response = requests.get(
                f"{self.remote_url}/download/{component_id}",
                headers=headers
            )
            
            # Check response
            if response.status_code == 200:
                # Create temporary file
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                
                # Load package from temporary file
                package = ComponentPackage.load(temp_file_path)
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
                return package
            else:
                logger.error(f"Error downloading component: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error downloading component: {e}")
            return None
    
    def import_remote_component(self, component_id: str, overwrite: bool = False) -> bool:
        """
        Import component from remote repository to local repository
        
        Args:
            component_id: Component ID
            overwrite: Whether to overwrite existing component
            
        Returns:
            Success flag
        """
        # Download component
        package = self.download_remote_component(component_id)
        if not package:
            return False
        
        # Publish to local repository
        return self.publish_component(package, overwrite)
    
    def upload_component(self, component_id: str) -> bool:
        """
        Upload component to remote repository
        
        Args:
            component_id: Component ID
            
        Returns:
            Success flag
        """
        try:
            # Get component
            package = self.get_local_component(component_id)
            if not package:
                logger.error(f"Component {component_id} not found in local repository")
                return False
            
            # Add API key if available
            headers = {}
            if self.api_key:
                headers['Authorization'] = f"Bearer {self.api_key}"
            
            # Send request
            response = requests.post(
                f"{self.remote_url}/upload",
                json=package.package(),
                headers=headers
            )
            
            # Check response
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Error uploading component: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error uploading component: {e}")
            return False
    
    def install_component(self, component_id: str) -> bool:
        """
        Install component into component registry
        
        Args:
            component_id: Component ID
            
        Returns:
            Success flag
        """
        # Get component
        package = self.get_local_component(component_id)
        if not package:
            logger.error(f"Component {component_id} not found in local repository")
            return False
        
        try:
            # Create temporary file for source code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                temp_file.write(package.source_code.encode('utf-8'))
                temp_file_path = temp_file.name
            
            # Import module dynamically
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"component_{component_id}", 
                temp_file_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find component class
            component_classes = []
            for name, obj in module.__dict__.items():
                if (isinstance(obj, type) and 
                    hasattr(obj, 'component_type') and 
                    obj.__module__ == module.__name__):
                    component_classes.append((name, obj))
            
            # Register component classes
            for class_name, class_obj in component_classes:
                self.registry.register_component_class(class_name, class_obj)
            
            logger.info(f"Installed {len(component_classes)} component classes from {component_id}")
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return len(component_classes) > 0
        except Exception as e:
            logger.error(f"Error installing component: {e}")
            return False
    
    def export_component_bundle(self, component_ids: List[str], output_file: str) -> bool:
        """
        Export components as a bundle (ZIP file)
        
        Args:
            component_ids: List of component IDs
            output_file: Output file path
            
        Returns:
            Success flag
        """
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy component files to temporary directory
                for component_id in component_ids:
                    # Get component
                    package = self.get_local_component(component_id)
                    if not package:
                        logger.warning(f"Component {component_id} not found, skipping")
                        continue
                    
                    # Save to temporary directory
                    file_path = os.path.join(temp_dir, f"{component_id}.json")
                    package.save(file_path)
                
                # Create ZIP file
                with zipfile.ZipFile(output_file, 'w') as zip_file:
                    for file in os.listdir(temp_dir):
                        zip_file.write(
                            os.path.join(temp_dir, file),
                            arcname=file
                        )
            
            return True
        except Exception as e:
            logger.error(f"Error exporting components: {e}")
            return False
    
    def import_component_bundle(self, input_file: str, overwrite: bool = False) -> List[str]:
        """
        Import components from a bundle (ZIP file)
        
        Args:
            input_file: Input file path
            overwrite: Whether to overwrite existing components
            
        Returns:
            List of imported component IDs
        """
        imported_ids = []
        
        try:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract ZIP file
                with zipfile.ZipFile(input_file, 'r') as zip_file:
                    zip_file.extractall(temp_dir)
                
                # Import components
                for file in os.listdir(temp_dir):
                    if file.endswith('.json'):
                        file_path = os.path.join(temp_dir, file)
                        
                        # Load package
                        package = ComponentPackage.load(file_path)
                        if not package:
                            continue
                        
                        # Publish to local repository
                        component_id = package.metadata.component_id
                        if self.publish_component(package, overwrite):
                            imported_ids.append(component_id)
            
            return imported_ids
        except Exception as e:
            logger.error(f"Error importing components: {e}")
            return imported_ids
