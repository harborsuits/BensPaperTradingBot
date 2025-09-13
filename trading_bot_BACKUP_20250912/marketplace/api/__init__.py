"""
Component Marketplace API
Provides secure endpoints for sharing, versioning, and managing trading components
"""

from .marketplace_api import MarketplaceAPI
from .security import SecurityManager, CodeVerifier, CodeSigner, Sandbox
from .version_manager import VersionManager, DependencyResolver
from .component_store import ComponentStore
from .rating_system import RatingSystem, ReviewManager

# Version
__version__ = '0.1.0'
