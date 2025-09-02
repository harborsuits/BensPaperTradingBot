"""
Component store for the marketplace API.

Provides:
- Local storage and management of components
- Metadata tracking
- Search and discovery
"""

import os
import json
import shutil
import logging
import glob
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class ComponentStore:
    """
    Manages local storage of marketplace components.
    
    Responsible for:
    - Storing components and metadata
    - Component retrieval
    - Local search and discovery
    """
    
    def __init__(self, components_path: str):
        """
        Initialize the component store.
        
        Args:
            components_path: Path to store components
        """
        self.components_path = components_path
        self.metadata_path = os.path.join(components_path, "metadata")
        
        # Create directories
        os.makedirs(components_path, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        
        # Cache for component metadata
        self.metadata_cache = {}
        
        logger.info(f"Component store initialized at {components_path}")
    
    def add_component(self, component_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a component to the store.
        
        Args:
            component_path: Path to the component file
            metadata: Component metadata
        
        Returns:
            Updated metadata
        """
        if not os.path.exists(component_path):
            raise FileNotFoundError(f"Component file not found: {component_path}")
        
        # Extract component ID and version
        component_id = metadata.get("component_id")
        version = metadata.get("version")
        
        if not component_id:
            component_id = os.path.basename(component_path).split(".")[0]
            metadata["component_id"] = component_id
        
        if not version:
            version = "1.0.0"
            metadata["version"] = version
        
        # Create component directory
        component_dir = os.path.join(self.components_path, component_id)
        os.makedirs(component_dir, exist_ok=True)
        
        # Create version directory
        version_dir = os.path.join(component_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy component file
        dest_path = os.path.join(version_dir, os.path.basename(component_path))
        shutil.copy2(component_path, dest_path)
        
        # Save metadata
        metadata["local_path"] = dest_path
        metadata["added_date"] = datetime.now().isoformat()
        
        metadata_file = os.path.join(self.metadata_path, f"{component_id}_{version}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update cache
        cache_key = f"{component_id}_{version}"
        self.metadata_cache[cache_key] = metadata
        
        logger.info(f"Added component {component_id} v{version} to store")
        return metadata
    
    def save_component(self, component_id: str, component_data: bytes, metadata: Dict[str, Any]) -> str:
        """
        Save component data to the store.
        
        Args:
            component_id: Component ID
            component_data: Component file data
            metadata: Component metadata
        
        Returns:
            Path to the saved component
        """
        # Extract version
        version = metadata.get("version", "1.0.0")
        
        # Create component directory
        component_dir = os.path.join(self.components_path, component_id)
        os.makedirs(component_dir, exist_ok=True)
        
        # Create version directory
        version_dir = os.path.join(component_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Determine file extension based on component type
        component_type = metadata.get("component_type", "")
        if component_type in ["SIGNAL_GENERATOR", "FILTER", "POSITION_SIZER", "EXIT_MANAGER", "STRATEGY"]:
            extension = ".py"
        else:
            extension = ".py"  # Default to Python
        
        # Save component file
        dest_path = os.path.join(version_dir, f"{component_id}{extension}")
        with open(dest_path, 'wb') as f:
            f.write(component_data)
        
        # Update metadata
        metadata["local_path"] = dest_path
        metadata["added_date"] = datetime.now().isoformat()
        
        # Save metadata
        metadata_file = os.path.join(self.metadata_path, f"{component_id}_{version}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update cache
        cache_key = f"{component_id}_{version}"
        self.metadata_cache[cache_key] = metadata
        
        logger.info(f"Saved component {component_id} v{version} to store")
        return dest_path
    
    def get_component_path(self, component_id: str, version: Optional[str] = None) -> Optional[str]:
        """
        Get the path to a component file.
        
        Args:
            component_id: Component ID
            version: Component version (None for latest)
        
        Returns:
            Path to the component file, or None if not found
        """
        # Get metadata
        metadata = self.get_component_metadata_by_id(component_id, version)
        
        if metadata and "local_path" in metadata:
            return metadata["local_path"]
        
        # If metadata doesn't have local_path, try to find the component manually
        component_dir = os.path.join(self.components_path, component_id)
        
        if not os.path.exists(component_dir):
            return None
        
        if version:
            # Check specific version
            version_dir = os.path.join(component_dir, version)
            
            if not os.path.exists(version_dir):
                return None
            
            # Find component file
            for ext in [".py", ".json"]:
                component_path = os.path.join(version_dir, f"{component_id}{ext}")
                if os.path.exists(component_path):
                    return component_path
            
            return None
        else:
            # Find latest version
            versions = self.get_component_versions(component_id)
            
            if not versions:
                return None
            
            # Sort versions and get latest
            sorted_versions = sorted(versions, key=lambda v: [int(x) for x in v.split(".")])
            latest = sorted_versions[-1]
            
            # Check latest version
            version_dir = os.path.join(component_dir, latest)
            
            # Find component file
            for ext in [".py", ".json"]:
                component_path = os.path.join(version_dir, f"{component_id}{ext}")
                if os.path.exists(component_path):
                    return component_path
            
            return None
    
    def get_component_metadata(self, component_path: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a component by path.
        
        Args:
            component_path: Path to the component file
        
        Returns:
            Component metadata, or None if not found
        """
        try:
            # Parse component ID and version from path
            path_parts = component_path.split(os.path.sep)
            
            if len(path_parts) < 3:
                return None
            
            # Extract component ID and version from path
            for i in range(len(path_parts) - 2):
                if path_parts[i] == "components":
                    component_id = path_parts[i + 1]
                    version = path_parts[i + 2]
                    break
            else:
                return None
            
            # Get metadata by ID and version
            return self.get_component_metadata_by_id(component_id, version)
            
        except Exception as e:
            logger.error(f"Failed to get component metadata: {e}")
            return None
    
    def get_component_metadata_by_id(self, component_id: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a component by ID and version.
        
        Args:
            component_id: Component ID
            version: Component version (None for latest)
        
        Returns:
            Component metadata, or None if not found
        """
        try:
            if version:
                # Check cache first
                cache_key = f"{component_id}_{version}"
                if cache_key in self.metadata_cache:
                    return self.metadata_cache[cache_key]
                
                # Check metadata file
                metadata_file = os.path.join(self.metadata_path, f"{component_id}_{version}.json")
                
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        self.metadata_cache[cache_key] = metadata
                        return metadata
                
                return None
            else:
                # Get all versions and find latest
                versions = self.get_component_versions(component_id)
                
                if not versions:
                    return None
                
                # Sort versions and get latest
                sorted_versions = sorted(versions, key=lambda v: [int(x) for x in v.split(".")])
                latest = sorted_versions[-1]
                
                # Get metadata for latest version
                return self.get_component_metadata_by_id(component_id, latest)
            
        except Exception as e:
            logger.error(f"Failed to get component metadata: {e}")
            return None
    
    def get_component_versions(self, component_id: str) -> List[str]:
        """
        Get all versions of a component.
        
        Args:
            component_id: Component ID
        
        Returns:
            List of version strings
        """
        # Check component directory
        component_dir = os.path.join(self.components_path, component_id)
        
        if not os.path.exists(component_dir):
            return []
        
        # Get all version directories
        versions = []
        for item in os.listdir(component_dir):
            if os.path.isdir(os.path.join(component_dir, item)):
                # Verify it's a valid version (basic check)
                if "." in item:
                    versions.append(item)
        
        return versions
    
    def update_metadata(self, component_id: str, metadata: Dict[str, Any], version: Optional[str] = None) -> bool:
        """
        Update metadata for a component.
        
        Args:
            component_id: Component ID
            metadata: Updated metadata
            version: Component version (None for latest)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get existing metadata
            existing = self.get_component_metadata_by_id(component_id, version)
            
            if not existing:
                return False
            
            # Use the version from existing metadata
            version = version or existing.get("version")
            
            # Merge metadata
            updated = {**existing, **metadata}
            
            # Save updated metadata
            metadata_file = os.path.join(self.metadata_path, f"{component_id}_{version}.json")
            with open(metadata_file, 'w') as f:
                json.dump(updated, f, indent=2)
            
            # Update cache
            cache_key = f"{component_id}_{version}"
            self.metadata_cache[cache_key] = updated
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update component metadata: {e}")
            return False
    
    def update_rating(self, component_id: str, average_rating: float, ratings_count: int, version: Optional[str] = None) -> bool:
        """
        Update rating for a component.
        
        Args:
            component_id: Component ID
            average_rating: New average rating
            ratings_count: New ratings count
            version: Component version (None for latest)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update metadata
            return self.update_metadata(
                component_id=component_id,
                metadata={
                    "average_rating": average_rating,
                    "ratings_count": ratings_count
                },
                version=version
            )
            
        except Exception as e:
            logger.error(f"Failed to update component rating: {e}")
            return False
    
    def list_components(self, component_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available components.
        
        Args:
            component_type: Optional filter by component type
        
        Returns:
            List of component metadata
        """
        results = []
        
        # Get all metadata files
        metadata_files = glob.glob(os.path.join(self.metadata_path, "*.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Filter by component type if specified
                if component_type and metadata.get("component_type") != component_type:
                    continue
                
                results.append(metadata)
                
            except Exception as e:
                logger.error(f"Failed to read metadata file {metadata_file}: {e}")
        
        return results
    
    def search_components(self, 
                         query: Optional[str] = None,
                         component_type: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         min_rating: Optional[float] = None,
                         sort_by: str = "downloads") -> List[Dict[str, Any]]:
        """
        Search for components.
        
        Args:
            query: Search query for component name or description
            component_type: Filter by component type
            tags: Filter by tags
            min_rating: Minimum average rating
            sort_by: Sort field ('downloads', 'rating', 'published_date')
        
        Returns:
            List of matching component metadata
        """
        # Get all components
        components = self.list_components(component_type)
        
        # Apply filters
        results = []
        
        for component in components:
            # Filter by query
            if query:
                # Check if query matches component ID or description
                query_lower = query.lower()
                component_id = component.get("component_id", "").lower()
                description = component.get("description", "").lower()
                
                if query_lower not in component_id and query_lower not in description:
                    continue
            
            # Filter by tags
            if tags:
                component_tags = component.get("tags", [])
                # Check if any of the search tags match component tags
                if not any(tag in component_tags for tag in tags):
                    continue
            
            # Filter by rating
            if min_rating is not None:
                component_rating = component.get("average_rating", 0.0)
                if component_rating < min_rating:
                    continue
            
            results.append(component)
        
        # Sort results
        if sort_by == "downloads":
            results.sort(key=lambda c: c.get("downloads", 0), reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda c: c.get("average_rating", 0.0), reverse=True)
        elif sort_by == "published_date":
            results.sort(key=lambda c: c.get("published_date", ""), reverse=True)
        
        return results
    
    def delete_component(self, component_id: str, version: Optional[str] = None) -> bool:
        """
        Delete a component from the store.
        
        Args:
            component_id: Component ID
            version: Component version (None for all versions)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            component_dir = os.path.join(self.components_path, component_id)
            
            if not os.path.exists(component_dir):
                logger.warning(f"Component {component_id} not found")
                return False
            
            if version:
                # Delete specific version
                version_dir = os.path.join(component_dir, version)
                
                if not os.path.exists(version_dir):
                    logger.warning(f"Component {component_id} version {version} not found")
                    return False
                
                # Delete version directory
                shutil.rmtree(version_dir)
                
                # Delete metadata file
                metadata_file = os.path.join(self.metadata_path, f"{component_id}_{version}.json")
                if os.path.exists(metadata_file):
                    os.unlink(metadata_file)
                
                # Remove from cache
                cache_key = f"{component_id}_{version}"
                if cache_key in self.metadata_cache:
                    del self.metadata_cache[cache_key]
                
                # Check if any versions remain
                versions = self.get_component_versions(component_id)
                if not versions:
                    # No versions remain, delete component directory
                    shutil.rmtree(component_dir)
                
                logger.info(f"Deleted component {component_id} version {version}")
                return True
            else:
                # Delete all versions
                shutil.rmtree(component_dir)
                
                # Delete all metadata files
                for metadata_file in glob.glob(os.path.join(self.metadata_path, f"{component_id}_*.json")):
                    os.unlink(metadata_file)
                
                # Remove from cache
                cache_keys = [k for k in self.metadata_cache if k.startswith(f"{component_id}_")]
                for key in cache_keys:
                    del self.metadata_cache[key]
                
                logger.info(f"Deleted all versions of component {component_id}")
                return True
            
        except Exception as e:
            logger.error(f"Failed to delete component: {e}")
            return False
