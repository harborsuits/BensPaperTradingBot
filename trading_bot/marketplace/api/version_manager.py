"""
Version management system for marketplace components.

Provides:
- Semantic versioning utilities
- Version comparison and sorting
- Dependency resolution
- Compatibility checking
"""

import re
import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Set

# Configure logging
logger = logging.getLogger(__name__)

class VersionManager:
    """
    Manages component versioning and compatibility.
    
    Implements semantic versioning (SemVer) for components.
    """
    
    def __init__(self):
        """Initialize the version manager"""
        logger.info("Version manager initialized")
    
    def parse_version(self, version_str: str) -> Tuple[int, int, int, str]:
        """
        Parse a version string into its components.
        
        Args:
            version_str: Version string in format 'x.y.z[-prerelease]'
        
        Returns:
            Tuple of (major, minor, patch, prerelease)
        
        Raises:
            ValueError: If version string is invalid
        """
        # Regular expression for semantic versioning
        semver_pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        match = re.match(semver_pattern, version_str)
        
        if not match:
            raise ValueError(f"Invalid version string: {version_str}")
        
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3))
        prerelease = match.group(4) or ""
        
        return major, minor, patch, prerelease
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """
        Compare two version strings.
        
        Args:
            version1: First version string
            version2: Second version string
        
        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        try:
            # Parse versions
            v1_major, v1_minor, v1_patch, v1_pre = self.parse_version(version1)
            v2_major, v2_minor, v2_patch, v2_pre = self.parse_version(version2)
            
            # Compare major, minor, patch versions
            if v1_major != v2_major:
                return -1 if v1_major < v2_major else 1
            
            if v1_minor != v2_minor:
                return -1 if v1_minor < v2_minor else 1
            
            if v1_patch != v2_patch:
                return -1 if v1_patch < v2_patch else 1
            
            # Both versions have same major.minor.patch
            
            # If both have prerelease or neither has prerelease
            if bool(v1_pre) == bool(v2_pre):
                if not v1_pre:
                    # Both are stable releases with same version
                    return 0
                
                # Compare prerelease strings
                if v1_pre < v2_pre:
                    return -1
                elif v1_pre > v2_pre:
                    return 1
                else:
                    return 0
            
            # One has prerelease, one doesn't
            # The one without prerelease is considered newer
            return -1 if v1_pre else 1
            
        except ValueError as e:
            logger.error(f"Version comparison failed: {e}")
            # Default to string comparison if parsing fails
            if version1 == version2:
                return 0
            return -1 if version1 < version2 else 1
    
    def sort_versions(self, versions: List[str]) -> List[str]:
        """
        Sort a list of version strings in ascending order.
        
        Args:
            versions: List of version strings
        
        Returns:
            Sorted list of versions
        """
        try:
            return sorted(versions, key=lambda v: self._version_key(v))
        except Exception as e:
            logger.error(f"Version sorting failed: {e}")
            # Fall back to simple string sorting
            return sorted(versions)
    
    def _version_key(self, version: str) -> Tuple:
        """
        Create a sort key for a version string.
        
        Args:
            version: Version string
        
        Returns:
            Tuple for sorting
        """
        try:
            major, minor, patch, prerelease = self.parse_version(version)
            # Stable releases come after prereleases
            prerelease_flag = 0 if prerelease else 1
            return (major, minor, patch, prerelease_flag, prerelease)
        except ValueError:
            # For invalid versions, use a minimal sort key
            return (-1, -1, -1, -1, version)
    
    def get_latest_version(self, versions: List[str]) -> Optional[str]:
        """
        Get the latest version from a list of versions.
        
        Args:
            versions: List of version strings
        
        Returns:
            Latest version string, or None if list is empty
        """
        if not versions:
            return None
        
        sorted_versions = self.sort_versions(versions)
        return sorted_versions[-1]
    
    def get_latest_compatible_version(self, versions: List[str], target_version: str) -> Optional[str]:
        """
        Get the latest version compatible with a target version.
        
        Compatibility rules:
        - Major version must match
        - Minor and patch versions must be greater than or equal to target
        
        Args:
            versions: List of available versions
            target_version: Target version to be compatible with
        
        Returns:
            Latest compatible version, or None if none found
        """
        try:
            target_major, target_minor, target_patch, _ = self.parse_version(target_version)
            
            # Filter compatible versions
            compatible_versions = []
            
            for version in versions:
                try:
                    major, minor, patch, _ = self.parse_version(version)
                    
                    # Must match major version
                    if major != target_major:
                        continue
                    
                    # Must be >= target minor & patch
                    if minor < target_minor:
                        continue
                    
                    if minor == target_minor and patch < target_patch:
                        continue
                    
                    compatible_versions.append(version)
                except ValueError:
                    # Skip invalid versions
                    continue
            
            # Get latest from compatible versions
            return self.get_latest_version(compatible_versions)
            
        except ValueError as e:
            logger.error(f"Compatible version lookup failed: {e}")
            return None
    
    def is_version_in_range(self, version: str, version_range: str) -> bool:
        """
        Check if a version is in a specified range.
        
        Supported range syntax:
        - Exact: "1.2.3"
        - Greater than or equal: ">=1.2.3"
        - Less than or equal: "<=1.2.3"
        - Greater than: ">1.2.3"
        - Less than: "<1.2.3"
        - Range: ">=1.2.3 <2.0.0"
        - Tilde: "~1.2.3" (>=1.2.3 <1.3.0)
        - Caret: "^1.2.3" (>=1.2.3 <2.0.0)
        
        Args:
            version: Version to check
            version_range: Version range specification
        
        Returns:
            True if version is in range, False otherwise
        """
        try:
            # Exact match
            if re.match(r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+)?$', version_range):
                return self.compare_versions(version, version_range) == 0
            
            # Tilde range (~1.2.3)
            tilde_match = re.match(r'^~(\d+\.\d+\.\d+)$', version_range)
            if tilde_match:
                base_version = tilde_match.group(1)
                major, minor, _, _ = self.parse_version(base_version)
                lower_bound = base_version
                upper_bound = f"{major}.{minor+1}.0"
                
                return (self.compare_versions(version, lower_bound) >= 0 and
                        self.compare_versions(version, upper_bound) < 0)
            
            # Caret range (^1.2.3)
            caret_match = re.match(r'^\^(\d+\.\d+\.\d+)$', version_range)
            if caret_match:
                base_version = caret_match.group(1)
                major, _, _, _ = self.parse_version(base_version)
                lower_bound = base_version
                upper_bound = f"{major+1}.0.0"
                
                return (self.compare_versions(version, lower_bound) >= 0 and
                        self.compare_versions(version, upper_bound) < 0)
            
            # Inequality operators
            if version_range.startswith(">="):
                return self.compare_versions(version, version_range[2:]) >= 0
            if version_range.startswith("<="):
                return self.compare_versions(version, version_range[2:]) <= 0
            if version_range.startswith(">"):
                return self.compare_versions(version, version_range[1:]) > 0
            if version_range.startswith("<"):
                return self.compare_versions(version, version_range[1:]) < 0
            
            # Range with multiple conditions
            if " " in version_range:
                conditions = version_range.split()
                for condition in conditions:
                    if not self.is_version_in_range(version, condition):
                        return False
                return True
            
            # Unknown range syntax
            logger.warning(f"Unknown version range syntax: {version_range}")
            return False
            
        except Exception as e:
            logger.error(f"Version range check failed: {e}")
            return False


class DependencyResolver:
    """
    Resolves dependencies between components.
    
    Handles dependency resolution and compatibility checking.
    """
    
    def __init__(self, version_manager: Optional[VersionManager] = None):
        """
        Initialize the dependency resolver.
        
        Args:
            version_manager: Optional VersionManager instance
        """
        self.version_manager = version_manager or VersionManager()
        self.dependency_graph = {}
        self.resolved = set()
        self.unresolved = set()
        
        logger.info("Dependency resolver initialized")
    
    def resolve_dependencies(self, dependencies: Dict[str, str], available_versions: Optional[Dict[str, List[str]]] = None) -> Dict[str, str]:
        """
        Resolve dependencies to specific versions.
        
        Args:
            dependencies: Dictionary of {component_id: version_range}
            available_versions: Optional dictionary of {component_id: [versions]}
        
        Returns:
            Dictionary of resolved dependencies {component_id: specific_version}
        
        Raises:
            ValueError: If dependencies cannot be resolved
        """
        if not dependencies:
            return {}
        
        resolved_deps = {}
        available_versions = available_versions or {}
        
        # Build dependency graph
        self.dependency_graph = {}
        self.resolved = set()
        self.unresolved = set()
        
        # First pass: resolve all dependencies to specific versions
        for dep_id, version_range in dependencies.items():
            # Get available versions for this dependency
            versions = available_versions.get(dep_id, [])
            
            if not versions:
                # No versions available, use the specified range as a specific version
                if self.version_manager.parse_version(version_range):
                    # If the range is a valid specific version, use it
                    resolved_deps[dep_id] = version_range
                else:
                    raise ValueError(f"Dependency {dep_id} with range {version_range} cannot be resolved: no available versions")
            else:
                # Find best matching version
                for version in self.version_manager.sort_versions(versions):
                    if self.version_manager.is_version_in_range(version, version_range):
                        resolved_deps[dep_id] = version
                        break
                else:
                    raise ValueError(f"No version of {dep_id} matches range {version_range}")
        
        # Second pass: resolve transitive dependencies
        # In a real implementation, this would recursively resolve dependencies of dependencies
        
        return resolved_deps
    
    def check_conflicts(self, dependencies: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Check for conflicts in resolved dependencies.
        
        Args:
            dependencies: Dictionary of resolved dependencies
        
        Returns:
            Dictionary of conflicts {component_id: [conflicting_versions]}
        """
        # In a real implementation, this would check for conflicting requirements
        # Simplified version just returns empty dict (no conflicts)
        return {}
    
    def generate_dependency_graph(self, dependencies: Dict[str, Dict[str, str]]) -> Dict[str, Set[str]]:
        """
        Generate a dependency graph.
        
        Args:
            dependencies: Dictionary of {component_id: {dependency_id: version}}
        
        Returns:
            Dictionary of {component_id: {dependencies}}
        """
        graph = {}
        
        for component_id, deps in dependencies.items():
            graph[component_id] = set(deps.keys())
        
        return graph
    
    def get_dependency_order(self, dependencies: Dict[str, Dict[str, str]]) -> List[str]:
        """
        Get the dependency resolution order.
        
        Args:
            dependencies: Dictionary of {component_id: {dependency_id: version}}
        
        Returns:
            List of component IDs in dependency order
        
        Raises:
            ValueError: If there is a circular dependency
        """
        # Build dependency graph
        self.dependency_graph = self.generate_dependency_graph(dependencies)
        self.resolved = set()
        self.unresolved = set()
        
        result = []
        
        # Process each component
        for component_id in self.dependency_graph:
            if component_id not in self.resolved:
                self._resolve_dependencies(component_id, result)
        
        return result
    
    def _resolve_dependencies(self, component_id: str, result: List[str]) -> None:
        """
        Recursively resolve dependencies for a component.
        
        Args:
            component_id: Component ID to resolve
            result: List to append resolved components to
        
        Raises:
            ValueError: If there is a circular dependency
        """
        self.unresolved.add(component_id)
        
        # Process dependencies
        for dependency in self.dependency_graph.get(component_id, set()):
            if dependency not in self.resolved:
                if dependency in self.unresolved:
                    # Circular dependency detected
                    raise ValueError(f"Circular dependency detected: {component_id} -> {dependency}")
                
                self._resolve_dependencies(dependency, result)
        
        self.resolved.add(component_id)
        self.unresolved.remove(component_id)
        result.append(component_id)
