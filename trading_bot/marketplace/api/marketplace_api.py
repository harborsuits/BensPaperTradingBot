"""
Main marketplace API module providing secure endpoints for component sharing and management.

This API enables:
- Secure component publishing and downloading
- Version management and dependency resolution
- Component rating and reviews
- Code signing and verification
- Sandboxed execution of marketplace components
"""

import os
import json
import time
import uuid
import logging
import threading
import requests
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
from pathlib import Path

from .security import SecurityManager, CodeVerifier, CodeSigner, Sandbox
from .version_manager import VersionManager, DependencyResolver
from .component_store import ComponentStore
from .rating_system import RatingSystem, ReviewManager

# Configure logging
logger = logging.getLogger(__name__)

class ApiEndpoint(Enum):
    """API endpoints for the marketplace"""
    PUBLISH = "publish"
    DOWNLOAD = "download"
    SEARCH = "search"
    LIST = "list"
    RATE = "rate"
    REVIEW = "review"
    VERSION = "version"
    VERIFY = "verify"
    USER = "user"
    STATS = "stats"
    DELETE = "delete"


class MarketplaceAPI:
    """
    The main API for interacting with the component marketplace.
    
    Provides secure endpoints for:
    - Publishing components
    - Downloading components
    - Searching and discovering components
    - Version management
    - Rating and reviews
    - Security verification
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 server_url: Optional[str] = None,
                 local_mode: bool = False):
        """
        Initialize the marketplace API.
        
        Args:
            api_key: Optional API key for authenticated requests
            server_url: URL of the marketplace server
            local_mode: If True, operate in local mode only (no external server)
        """
        self.api_key = api_key
        self.server_url = server_url or "https://api.tradingbotmarketplace.com/v1"
        self.local_mode = local_mode
        
        # Default paths
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.components_path = os.path.join(self.base_path, "marketplace", "components")
        self.keys_path = os.path.join(self.base_path, "marketplace", "keys")
        
        # Create necessary directories
        os.makedirs(self.components_path, exist_ok=True)
        os.makedirs(self.keys_path, exist_ok=True)
        
        # Initialize sub-components
        self.security_manager = SecurityManager(keys_path=self.keys_path)
        self.version_manager = VersionManager()
        self.component_store = ComponentStore(components_path=self.components_path)
        self.rating_system = RatingSystem()
        self.review_manager = ReviewManager()
        
        # Local cache for component metadata
        self.component_cache = {}
        self.last_cache_update = 0
        self.cache_lock = threading.Lock()
        
        logger.info(f"Marketplace API initialized. Local mode: {local_mode}")
    
    def publish_component(self, 
                         component_path: str, 
                         component_type: str,
                         version: str,
                         description: str,
                         tags: List[str] = None,
                         dependencies: Dict[str, str] = None,
                         sign: bool = True) -> Dict[str, Any]:
        """
        Publish a component to the marketplace.
        
        Args:
            component_path: Path to the component file
            component_type: Type of component (SIGNAL_GENERATOR, FILTER, etc.)
            version: Version string (semver format)
            description: Component description
            tags: List of tags for categorization
            dependencies: Dictionary of dependencies {name: version}
            sign: Whether to sign the component with the user's key
        
        Returns:
            Dict containing the result of the publish operation
        """
        if not os.path.exists(component_path):
            raise FileNotFoundError(f"Component file not found: {component_path}")
        
        # Prepare metadata
        component_id = os.path.basename(component_path).split(".")[0]
        metadata = {
            "component_id": component_id,
            "component_type": component_type,
            "version": version,
            "description": description,
            "tags": tags or [],
            "dependencies": dependencies or {},
            "published_date": datetime.now().isoformat(),
            "publisher": self._get_user_id(),
            "downloads": 0,
            "average_rating": 0.0,
            "ratings_count": 0,
            "verified": False
        }
        
        # Sign the component if requested
        if sign:
            try:
                signature = self.security_manager.sign_component(component_path)
                metadata["signature"] = signature
                metadata["verified"] = True
            except Exception as e:
                logger.error(f"Failed to sign component: {e}")
                raise RuntimeError(f"Component signing failed: {e}")
        
        # Verify dependencies can be resolved
        if dependencies:
            try:
                self.version_manager.resolve_dependencies(dependencies)
            except Exception as e:
                logger.error(f"Dependency resolution failed: {e}")
                raise ValueError(f"Dependency resolution failed: {e}")
        
        # In local mode, store locally
        if self.local_mode:
            result = self.component_store.add_component(
                component_path=component_path,
                metadata=metadata
            )
            logger.info(f"Component {component_id} v{version} published locally")
            return result
        
        # Otherwise, publish to server
        try:
            with open(component_path, 'rb') as file:
                component_data = file.read()
                
            response = self._api_request(
                endpoint=ApiEndpoint.PUBLISH,
                method="POST",
                files={"component": component_data},
                data={"metadata": json.dumps(metadata)}
            )
            
            # Also store locally
            self.component_store.add_component(
                component_path=component_path,
                metadata=metadata
            )
            
            logger.info(f"Component {component_id} v{version} published to marketplace")
            return response
            
        except Exception as e:
            logger.error(f"Failed to publish component to marketplace: {e}")
            raise RuntimeError(f"Publishing to marketplace failed: {e}")
    
    def download_component(self, 
                          component_id: str, 
                          version: Optional[str] = None,
                          verify: bool = True) -> str:
        """
        Download a component from the marketplace.
        
        Args:
            component_id: ID of the component to download
            version: Specific version to download (None for latest)
            verify: Whether to verify the component signature
        
        Returns:
            Path to the downloaded component file
        """
        # Check if component is available locally first
        local_path = self.component_store.get_component_path(component_id, version)
        
        if local_path and os.path.exists(local_path):
            logger.info(f"Component {component_id} found locally")
            
            # Verify if requested
            if verify:
                is_verified = self.verify_component(local_path)
                if not is_verified:
                    logger.warning(f"Local component {component_id} failed verification")
                    # Continue to download from server if verification fails
                else:
                    return local_path
        
        # If in local mode and not found locally, fail
        if self.local_mode and not local_path:
            raise FileNotFoundError(f"Component {component_id} not found locally and local mode is enabled")
        
        # Download from server
        try:
            params = {"component_id": component_id}
            if version:
                params["version"] = version
                
            response = self._api_request(
                endpoint=ApiEndpoint.DOWNLOAD,
                method="GET",
                params=params
            )
            
            if "component_data" not in response or "metadata" not in response:
                raise ValueError("Invalid response from server")
            
            # Save component to local store
            component_data = response["component_data"]
            metadata = response["metadata"]
            
            # Verify if requested
            if verify:
                is_verified = self.security_manager.verify_component_data(
                    component_data, 
                    metadata.get("signature"), 
                    metadata.get("publisher")
                )
                
                if not is_verified:
                    raise SecurityError(f"Component {component_id} failed verification")
            
            # Save to local store
            local_path = self.component_store.save_component(
                component_id=component_id,
                component_data=component_data,
                metadata=metadata
            )
            
            logger.info(f"Component {component_id} downloaded from marketplace")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download component {component_id}: {e}")
            raise RuntimeError(f"Download failed: {e}")
    
    def search_components(self, 
                         query: Optional[str] = None,
                         component_type: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         min_rating: Optional[float] = None,
                         sort_by: str = "downloads",
                         include_dependencies: bool = False) -> List[Dict[str, Any]]:
        """
        Search for components in the marketplace.
        
        Args:
            query: Search query for component name or description
            component_type: Filter by component type
            tags: Filter by tags
            min_rating: Minimum average rating
            sort_by: Sort field ('downloads', 'rating', 'published_date')
            include_dependencies: Whether to include dependency info
        
        Returns:
            List of component metadata dictionaries
        """
        # In local mode, search local store only
        if self.local_mode:
            components = self.component_store.search_components(
                query=query,
                component_type=component_type,
                tags=tags,
                min_rating=min_rating,
                sort_by=sort_by
            )
            return components
        
        # Otherwise, search on server
        try:
            params = {}
            if query:
                params["query"] = query
            if component_type:
                params["component_type"] = component_type
            if tags:
                params["tags"] = ",".join(tags)
            if min_rating is not None:
                params["min_rating"] = min_rating
                
            params["sort_by"] = sort_by
            params["include_dependencies"] = "true" if include_dependencies else "false"
                
            response = self._api_request(
                endpoint=ApiEndpoint.SEARCH,
                method="GET",
                params=params
            )
            
            return response.get("components", [])
            
        except Exception as e:
            logger.error(f"Failed to search marketplace: {e}")
            # Fall back to local search
            logger.info("Falling back to local component search")
            return self.component_store.search_components(
                query=query,
                component_type=component_type,
                tags=tags,
                min_rating=min_rating,
                sort_by=sort_by
            )
    
    def list_components(self, 
                       component_type: Optional[str] = None,
                       include_local: bool = True) -> List[Dict[str, Any]]:
        """
        List components available in the marketplace.
        
        Args:
            component_type: Optional filter by component type
            include_local: Whether to include locally stored components
        
        Returns:
            List of component metadata dictionaries
        """
        # Start with local components if requested
        components = []
        if include_local:
            local_components = self.component_store.list_components(component_type)
            components.extend(local_components)
        
        # If in local mode, return only local components
        if self.local_mode:
            return components
        
        # Otherwise, get components from server
        try:
            params = {}
            if component_type:
                params["component_type"] = component_type
                
            response = self._api_request(
                endpoint=ApiEndpoint.LIST,
                method="GET",
                params=params
            )
            
            server_components = response.get("components", [])
            
            # Merge with local components, avoiding duplicates
            if components:
                local_ids = set((c["component_id"], c.get("version")) for c in components)
                for comp in server_components:
                    if (comp["component_id"], comp.get("version")) not in local_ids:
                        components.append(comp)
            else:
                components = server_components
                
            return components
            
        except Exception as e:
            logger.error(f"Failed to list marketplace components: {e}")
            # Return local components only if server request fails
            return self.component_store.list_components(component_type)
    
    def rate_component(self, 
                      component_id: str, 
                      rating: float, 
                      version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rate a component in the marketplace.
        
        Args:
            component_id: ID of the component to rate
            rating: Rating value (1-5)
            version: Specific version to rate (None for latest)
        
        Returns:
            Dict containing the updated rating information
        """
        if rating < 1 or rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        # In local mode, store rating locally
        if self.local_mode:
            result = self.rating_system.add_rating(
                component_id=component_id,
                rating=rating,
                user_id=self._get_user_id(),
                version=version
            )
            
            # Update component metadata
            self.component_store.update_rating(
                component_id=component_id,
                average_rating=result["average_rating"],
                ratings_count=result["ratings_count"],
                version=version
            )
            
            return result
        
        # Otherwise, submit to server
        try:
            params = {
                "component_id": component_id,
                "rating": rating,
                "user_id": self._get_user_id()
            }
            
            if version:
                params["version"] = version
                
            response = self._api_request(
                endpoint=ApiEndpoint.RATE,
                method="POST",
                data=params
            )
            
            # Update local cache
            if "average_rating" in response and "ratings_count" in response:
                self.component_store.update_rating(
                    component_id=component_id,
                    average_rating=response["average_rating"],
                    ratings_count=response["ratings_count"],
                    version=version
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to rate component {component_id}: {e}")
            raise RuntimeError(f"Rating submission failed: {e}")
    
    def add_review(self, 
                  component_id: str, 
                  review_text: str,
                  version: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a review for a component.
        
        Args:
            component_id: ID of the component to review
            review_text: Text of the review
            version: Specific version to review (None for latest)
        
        Returns:
            Dict containing the review information
        """
        if not review_text or len(review_text.strip()) < 5:
            raise ValueError("Review text must be at least 5 characters")
        
        # Generate review ID
        review_id = str(uuid.uuid4())
        
        # In local mode, store review locally
        if self.local_mode:
            result = self.review_manager.add_review(
                component_id=component_id,
                review_id=review_id,
                review_text=review_text,
                user_id=self._get_user_id(),
                version=version
            )
            return result
        
        # Otherwise, submit to server
        try:
            params = {
                "component_id": component_id,
                "review_text": review_text,
                "user_id": self._get_user_id(),
                "review_id": review_id
            }
            
            if version:
                params["version"] = version
                
            response = self._api_request(
                endpoint=ApiEndpoint.REVIEW,
                method="POST",
                data=params
            )
            
            # Store locally as well
            self.review_manager.add_review(
                component_id=component_id,
                review_id=review_id,
                review_text=review_text,
                user_id=self._get_user_id(),
                version=version
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to submit review for component {component_id}: {e}")
            raise RuntimeError(f"Review submission failed: {e}")
    
    def get_reviews(self, 
                   component_id: str,
                   version: Optional[str] = None,
                   limit: int = 10,
                   offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get reviews for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
            limit: Maximum number of reviews to return
            offset: Offset for pagination
        
        Returns:
            List of review dictionaries
        """
        # In local mode, get reviews from local store
        if self.local_mode:
            reviews = self.review_manager.get_reviews(
                component_id=component_id,
                version=version,
                limit=limit,
                offset=offset
            )
            return reviews
        
        # Otherwise, get from server
        try:
            params = {
                "component_id": component_id,
                "limit": limit,
                "offset": offset
            }
            
            if version:
                params["version"] = version
                
            response = self._api_request(
                endpoint=ApiEndpoint.REVIEW,
                method="GET",
                params=params
            )
            
            return response.get("reviews", [])
            
        except Exception as e:
            logger.error(f"Failed to get reviews for component {component_id}: {e}")
            # Fall back to local reviews
            return self.review_manager.get_reviews(
                component_id=component_id,
                version=version,
                limit=limit,
                offset=offset
            )
    
    def verify_component(self, component_path: str) -> bool:
        """
        Verify a component's signature.
        
        Args:
            component_path: Path to the component file
        
        Returns:
            True if verification succeeds, False otherwise
        """
        try:
            # Get component metadata
            metadata = self.component_store.get_component_metadata(component_path)
            
            if not metadata or "signature" not in metadata:
                logger.warning(f"Component {component_path} has no signature")
                return False
            
            # Verify signature
            is_verified = self.security_manager.verify_component(
                component_path=component_path,
                signature=metadata["signature"],
                publisher_id=metadata.get("publisher")
            )
            
            return is_verified
            
        except Exception as e:
            logger.error(f"Component verification failed: {e}")
            return False
    
    def get_component_info(self, 
                          component_id: str,
                          version: Optional[str] = None,
                          refresh: bool = False) -> Dict[str, Any]:
        """
        Get detailed information about a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
            refresh: Whether to refresh info from server
        
        Returns:
            Component metadata dictionary
        """
        # Check cache first unless refresh is requested
        cache_key = f"{component_id}_{version or 'latest'}"
        
        with self.cache_lock:
            if not refresh and cache_key in self.component_cache:
                return self.component_cache[cache_key]
        
        # Try local store first
        metadata = self.component_store.get_component_metadata_by_id(
            component_id=component_id,
            version=version
        )
        
        if metadata and not refresh:
            with self.cache_lock:
                self.component_cache[cache_key] = metadata
            return metadata
        
        # If in local mode, return local info or None
        if self.local_mode:
            return metadata
        
        # Otherwise, get from server
        try:
            params = {"component_id": component_id}
            if version:
                params["version"] = version
                
            response = self._api_request(
                endpoint=ApiEndpoint.VERSION,
                method="GET",
                params=params
            )
            
            if "metadata" in response:
                metadata = response["metadata"]
                
                # Update local cache
                with self.cache_lock:
                    self.component_cache[cache_key] = metadata
                
                # Update local store
                self.component_store.update_metadata(
                    component_id=component_id,
                    metadata=metadata,
                    version=version
                )
                
                return metadata
            else:
                return None
                
        except Exception as e:
            logger.error(f"Failed to get component info for {component_id}: {e}")
            return metadata  # Return local info if server request fails
    
    def get_available_versions(self, component_id: str) -> List[str]:
        """
        Get all available versions of a component.
        
        Args:
            component_id: ID of the component
        
        Returns:
            List of version strings
        """
        # Check local store first
        local_versions = self.component_store.get_component_versions(component_id)
        
        # If in local mode, return local versions
        if self.local_mode:
            return local_versions
        
        # Otherwise, get from server
        try:
            params = {"component_id": component_id}
                
            response = self._api_request(
                endpoint=ApiEndpoint.VERSION,
                method="GET",
                params=params
            )
            
            server_versions = response.get("versions", [])
            
            # Merge with local versions
            all_versions = set(local_versions)
            all_versions.update(server_versions)
            
            # Sort versions
            sorted_versions = self.version_manager.sort_versions(list(all_versions))
            
            return sorted_versions
            
        except Exception as e:
            logger.error(f"Failed to get versions for component {component_id}: {e}")
            return local_versions  # Return local versions if server request fails
    
    def resolve_dependencies(self, 
                            component_id: str,
                            version: Optional[str] = None,
                            download: bool = True) -> Dict[str, str]:
        """
        Resolve and optionally download dependencies for a component.
        
        Args:
            component_id: ID of the component
            version: Specific version (None for latest)
            download: Whether to download missing dependencies
        
        Returns:
            Dictionary of resolved dependencies {name: version}
        """
        # Get component metadata
        metadata = self.get_component_info(component_id, version)
        
        if not metadata:
            raise ValueError(f"Component {component_id} not found")
        
        if "dependencies" not in metadata or not metadata["dependencies"]:
            return {}
        
        dependencies = metadata["dependencies"]
        
        # Resolve dependencies
        resolved = self.version_manager.resolve_dependencies(dependencies)
        
        # Download missing dependencies if requested
        if download:
            for dep_id, dep_version in resolved.items():
                try:
                    # Check if already available locally
                    local_path = self.component_store.get_component_path(dep_id, dep_version)
                    
                    if not local_path or not os.path.exists(local_path):
                        self.download_component(dep_id, dep_version)
                        logger.info(f"Downloaded dependency {dep_id} v{dep_version}")
                except Exception as e:
                    logger.error(f"Failed to download dependency {dep_id}: {e}")
                    raise RuntimeError(f"Dependency resolution failed: {e}")
        
        return resolved
    
    def execute_in_sandbox(self, 
                          component_path: str,
                          input_data: Dict[str, Any],
                          timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a component in a secure sandbox environment.
        
        Args:
            component_path: Path to the component file
            input_data: Input data for the component
            timeout: Execution timeout in seconds
        
        Returns:
            Dictionary with execution results
        """
        # Verify component first
        is_verified = self.verify_component(component_path)
        
        if not is_verified:
            logger.warning(f"Executing unverified component {component_path}")
        
        # Create sandbox and execute
        try:
            sandbox = Sandbox()
            result = sandbox.execute(
                component_path=component_path,
                input_data=input_data,
                timeout=timeout
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            raise RuntimeError(f"Sandbox execution failed: {e}")
    
    def generate_api_key(self) -> str:
        """
        Generate a new API key.
        
        Returns:
            New API key string
        """
        # Generate key using security manager
        api_key = self.security_manager.generate_api_key()
        self.api_key = api_key
        
        return api_key
    
    def _api_request(self, 
                    endpoint: ApiEndpoint, 
                    method: str = "GET",
                    params: Dict[str, Any] = None,
                    data: Dict[str, Any] = None,
                    files: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an API request to the marketplace server.
        
        Args:
            endpoint: API endpoint to request
            method: HTTP method (GET, POST, etc.)
            params: URL parameters
            data: Request data
            files: Files to upload
        
        Returns:
            Response data as dictionary
        """
        if self.local_mode:
            raise RuntimeError("Cannot make API requests in local mode")
        
        url = f"{self.server_url}/{endpoint.value}"
        
        # Add API key to headers if available
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, data=data, files=files, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for errors
            response.raise_for_status()
            
            # Parse JSON response
            return response.json()
            
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise RuntimeError(f"API request failed: {e}")
    
    def _get_user_id(self) -> str:
        """
        Get the current user ID.
        
        Returns:
            User ID string
        """
        # In a real implementation, this would get the authenticated user ID
        # For now, use a placeholder
        return self.security_manager.get_user_id()


class SecurityError(Exception):
    """Exception raised for security verification failures"""
    pass
