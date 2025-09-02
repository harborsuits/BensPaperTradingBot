#!/usr/bin/env python3
"""
Feature Flag Dependencies Visualizer

This module provides visualization tools for feature flag dependencies to help
identify potential issues when enabling or disabling flags. It generates interactive
graphs showing relationships between flags and their impact on system components.
"""

import logging
import json
import os
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime
import networkx as nx
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

# Setup logging
logger = logging.getLogger(__name__)

class DependencyType(Enum):
    """Types of dependencies between feature flags"""
    REQUIRES = "requires"  # Flag A requires Flag B to be enabled
    CONFLICTS = "conflicts"  # Flag A conflicts with Flag B (cannot be enabled together)
    ENHANCES = "enhances"  # Flag A enhances Flag B (optional but beneficial)
    ALTERNATIVE = "alternative"  # Flag A is an alternative to Flag B

class SystemComponent(Enum):
    """Major system components that feature flags may affect"""
    CORE = "core"
    UI = "ui"
    API = "api"
    DATABASE = "database"
    TRADING = "trading"
    ANALYTICS = "analytics"
    BACKTESTING = "backtesting"
    MONITORING = "monitoring"
    SECURITY = "security"

class DependencyGraph:
    """
    Manages a graph of feature flag dependencies.
    
    This class builds and maintains a directed graph representing
    relationships between feature flags, allowing analysis of
    potential conflicts or requirements when enabling/disabling flags.
    """
    
    def __init__(self, storage_path: str = "data/feature_flags"):
        """
        Initialize the dependency graph.
        
        Args:
            storage_path: Path to store dependency data
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize the graph
        self.graph = nx.DiGraph()
        
        # Track components affected by each flag
        self.flag_components: Dict[str, Set[SystemComponent]] = {}
        
        # Load existing dependencies
        self._load_dependencies()
        
        logger.info(f"Initialized dependency graph with {len(self.graph.nodes)} flags")
    
    def _load_dependencies(self):
        """Load existing dependencies from storage"""
        try:
            deps_file = os.path.join(self.storage_path, "dependencies.json")
            if not os.path.exists(deps_file):
                return
                
            with open(deps_file, 'r') as f:
                data = json.load(f)
                
            # Add nodes for flags
            for flag_data in data.get("flags", []):
                flag_name = flag_data.get("name")
                if not flag_name:
                    continue
                    
                self.graph.add_node(
                    flag_name,
                    description=flag_data.get("description", ""),
                    owner=flag_data.get("owner", ""),
                    created_at=flag_data.get("created_at", "")
                )
                
                # Add components
                components = flag_data.get("components", [])
                self.flag_components[flag_name] = set(
                    SystemComponent(c) for c in components if c in [e.value for e in SystemComponent]
                )
                
            # Add edges for dependencies
            for dep_data in data.get("dependencies", []):
                source = dep_data.get("source")
                target = dep_data.get("target")
                dep_type = dep_data.get("type")
                
                if not (source and target and dep_type):
                    continue
                    
                if source not in self.graph.nodes or target not in self.graph.nodes:
                    continue
                    
                self.graph.add_edge(
                    source,
                    target,
                    type=dep_type,
                    description=dep_data.get("description", "")
                )
                
            logger.info(f"Loaded {len(self.graph.nodes)} flags and {len(self.graph.edges)} dependencies")
                
        except Exception as e:
            logger.error(f"Error loading dependencies: {str(e)}")
    
    def _save_dependencies(self):
        """Save dependencies to storage"""
        try:
            deps_file = os.path.join(self.storage_path, "dependencies.json")
            
            # Prepare flags data
            flags_data = []
            for flag_name in self.graph.nodes:
                flag_data = {
                    "name": flag_name,
                    **self.graph.nodes[flag_name]
                }
                
                # Add components
                components = self.flag_components.get(flag_name, set())
                flag_data["components"] = [c.value for c in components]
                
                flags_data.append(flag_data)
                
            # Prepare dependencies data
            deps_data = []
            for source, target, data in self.graph.edges(data=True):
                deps_data.append({
                    "source": source,
                    "target": target,
                    **data
                })
                
            # Save to file
            with open(deps_file, 'w') as f:
                json.dump({
                    "last_updated": datetime.now().isoformat(),
                    "flags": flags_data,
                    "dependencies": deps_data
                }, f, indent=2)
                
            logger.debug(f"Saved {len(self.graph.nodes)} flags and {len(self.graph.edges)} dependencies")
            
        except Exception as e:
            logger.error(f"Error saving dependencies: {str(e)}")
    
    def add_flag(self, 
                flag_name: str, 
                description: str = "", 
                owner: str = "",
                components: Optional[List[SystemComponent]] = None) -> bool:
        """
        Add a new feature flag to the graph.
        
        Args:
            flag_name: Name of the feature flag
            description: Description of the flag
            owner: Owner of the flag
            components: List of system components affected by the flag
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        if flag_name in self.graph.nodes:
            logger.warning(f"Flag '{flag_name}' already exists")
            return False
            
        self.graph.add_node(
            flag_name,
            description=description,
            owner=owner,
            created_at=datetime.now().isoformat()
        )
        
        # Add components
        if components:
            self.flag_components[flag_name] = set(components)
        else:
            self.flag_components[flag_name] = set()
            
        self._save_dependencies()
        
        logger.info(f"Added flag '{flag_name}'")
        return True
    
    def update_flag(self, 
                   flag_name: str, 
                   description: Optional[str] = None, 
                   owner: Optional[str] = None,
                   components: Optional[List[SystemComponent]] = None) -> bool:
        """
        Update an existing feature flag in the graph.
        
        Args:
            flag_name: Name of the feature flag
            description: New description (if provided)
            owner: New owner (if provided)
            components: New list of system components (if provided)
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        if flag_name not in self.graph.nodes:
            logger.warning(f"Flag '{flag_name}' does not exist")
            return False
            
        if description is not None:
            self.graph.nodes[flag_name]["description"] = description
            
        if owner is not None:
            self.graph.nodes[flag_name]["owner"] = owner
            
        if components is not None:
            self.flag_components[flag_name] = set(components)
            
        self._save_dependencies()
        
        logger.info(f"Updated flag '{flag_name}'")
        return True
    
    def delete_flag(self, flag_name: str) -> bool:
        """
        Delete a feature flag from the graph.
        
        Args:
            flag_name: Name of the feature flag
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if flag_name not in self.graph.nodes:
            logger.warning(f"Flag '{flag_name}' does not exist")
            return False
            
        self.graph.remove_node(flag_name)
        
        if flag_name in self.flag_components:
            del self.flag_components[flag_name]
            
        self._save_dependencies()
        
        logger.info(f"Deleted flag '{flag_name}'")
        return True
    
    def add_dependency(self, 
                      source_flag: str, 
                      target_flag: str, 
                      dependency_type: DependencyType,
                      description: str = "") -> bool:
        """
        Add a dependency between two feature flags.
        
        Args:
            source_flag: Name of the source flag
            target_flag: Name of the target flag
            dependency_type: Type of dependency
            description: Description of the dependency
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        if source_flag not in self.graph.nodes:
            logger.warning(f"Source flag '{source_flag}' does not exist")
            return False
            
        if target_flag not in self.graph.nodes:
            logger.warning(f"Target flag '{target_flag}' does not exist")
            return False
            
        self.graph.add_edge(
            source_flag,
            target_flag,
            type=dependency_type.value,
            description=description
        )
        
        self._save_dependencies()
        
        logger.info(f"Added {dependency_type.value} dependency from '{source_flag}' to '{target_flag}'")
        return True
    
    def delete_dependency(self, source_flag: str, target_flag: str) -> bool:
        """
        Delete a dependency between two feature flags.
        
        Args:
            source_flag: Name of the source flag
            target_flag: Name of the target flag
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        if not self.graph.has_edge(source_flag, target_flag):
            logger.warning(f"No dependency from '{source_flag}' to '{target_flag}'")
            return False
            
        self.graph.remove_edge(source_flag, target_flag)
        
        self._save_dependencies()
        
        logger.info(f"Deleted dependency from '{source_flag}' to '{target_flag}'")
        return True
    
    def get_flag_dependencies(self, flag_name: str) -> Tuple[List[str], List[str]]:
        """
        Get dependencies for a specific flag.
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            Tuple[List[str], List[str]]: Lists of flags this flag depends on and flags that depend on this flag
        """
        if flag_name not in self.graph.nodes:
            logger.warning(f"Flag '{flag_name}' does not exist")
            return [], []
            
        # Get dependencies (outgoing edges)
        dependencies = list(self.graph.successors(flag_name))
        
        # Get dependents (incoming edges)
        dependents = list(self.graph.predecessors(flag_name))
        
        return dependencies, dependents
    
    def check_conflicts(self, enabled_flags: List[str]) -> List[Dict[str, Any]]:
        """
        Check for conflicts in the set of enabled flags.
        
        Args:
            enabled_flags: List of currently enabled flags
            
        Returns:
            List[Dict[str, Any]]: List of detected conflicts
        """
        conflicts = []
        
        # Check all edges for conflicts
        for source, target, data in self.graph.edges(data=True):
            dep_type = data.get("type")
            
            if dep_type == DependencyType.CONFLICTS.value:
                # If both conflicting flags are enabled, report conflict
                if source in enabled_flags and target in enabled_flags:
                    conflicts.append({
                        "type": "conflict",
                        "flag1": source,
                        "flag2": target,
                        "description": data.get("description", "")
                    })
            elif dep_type == DependencyType.REQUIRES.value:
                # If source is enabled but target is not, report missing requirement
                if source in enabled_flags and target not in enabled_flags:
                    conflicts.append({
                        "type": "missing_requirement",
                        "flag": source,
                        "requires": target,
                        "description": data.get("description", "")
                    })
                    
        return conflicts
    
    def get_flag_impact(self, flag_name: str) -> Dict[str, Any]:
        """
        Get the impact of enabling/disabling a flag.
        
        Args:
            flag_name: Name of the flag
            
        Returns:
            Dict[str, Any]: Information about the flag's impact
        """
        if flag_name not in self.graph.nodes:
            logger.warning(f"Flag '{flag_name}' does not exist")
            return {}
            
        # Get dependencies and dependents
        dependencies, dependents = self.get_flag_dependencies(flag_name)
        
        # Get components affected
        components = self.flag_components.get(flag_name, set())
        
        # Determine potential impact
        impact = {
            "flag": flag_name,
            "description": self.graph.nodes[flag_name].get("description", ""),
            "components": [c.value for c in components],
            "dependencies": dependencies,
            "dependents": dependents,
            "required_flags": [],
            "conflicting_flags": [],
        }
        
        # Identify required and conflicting flags
        for target in dependencies:
            edge_data = self.graph.get_edge_data(flag_name, target)
            dep_type = edge_data.get("type")
            
            if dep_type == DependencyType.REQUIRES.value:
                impact["required_flags"].append(target)
            elif dep_type == DependencyType.CONFLICTS.value:
                impact["conflicting_flags"].append(target)
                
        return impact
    
    def visualize_graph(self, output_file: str = None) -> Optional[plt.Figure]:
        """
        Visualize the dependency graph using matplotlib.
        
        Args:
            output_file: Optional path to save the visualization
            
        Returns:
            Optional[plt.Figure]: The generated figure, or None if visualization failed
        """
        try:
            if len(self.graph.nodes) == 0:
                logger.warning("No flags to visualize")
                return None
                
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create a layout
            pos = nx.spring_layout(self.graph)
            
            # Define colors for different dependency types
            edge_colors = {
                DependencyType.REQUIRES.value: 'red',
                DependencyType.CONFLICTS.value: 'orange',
                DependencyType.ENHANCES.value: 'green',
                DependencyType.ALTERNATIVE.value: 'blue'
            }
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color='lightblue')
            
            # Draw edges for each dependency type
            for dep_type, color in edge_colors.items():
                edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                         if d.get('type') == dep_type]
                nx.draw_networkx_edges(self.graph, pos, edgelist=edges, 
                                      width=2, edge_color=color, arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos, font_size=10)
            
            # Create legend
            legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=dep_type)
                              for dep_type, color in edge_colors.items()]
            plt.legend(handles=legend_elements)
            
            plt.title('Feature Flag Dependencies')
            plt.axis('off')
            
            # Save if output file is provided
            if output_file:
                plt.savefig(output_file, bbox_inches='tight')
                logger.info(f"Saved visualization to {output_file}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")
            return None
    
    def create_interactive_graph(self) -> Optional[go.Figure]:
        """
        Create an interactive visualization using Plotly.
        
        Returns:
            Optional[go.Figure]: Plotly figure or None if creation failed
        """
        try:
            if len(self.graph.nodes) == 0:
                logger.warning("No flags to visualize")
                return None
                
            # Create a layout
            pos = nx.spring_layout(self.graph, seed=42)
            
            # Create edges for each dependency type
            edge_traces = []
            
            # Define colors for different dependency types
            edge_colors = {
                DependencyType.REQUIRES.value: 'red',
                DependencyType.CONFLICTS.value: 'orange',
                DependencyType.ENHANCES.value: 'green',
                DependencyType.ALTERNATIVE.value: 'blue'
            }
            
            # Group edges by type
            edges_by_type = {}
            for dep_type in edge_colors:
                edges_by_type[dep_type] = []
                
            for u, v, data in self.graph.edges(data=True):
                dep_type = data.get('type')
                if dep_type in edges_by_type:
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    edges_by_type[dep_type].append((x0, y0, x1, y1, u, v, data.get('description', '')))
            
            # Create a trace for each dependency type
            for dep_type, edges in edges_by_type.items():
                if not edges:
                    continue
                    
                x_edges = []
                y_edges = []
                hover_texts = []
                
                for x0, y0, x1, y1, u, v, desc in edges:
                    x_edges.extend([x0, x1, None])
                    y_edges.extend([y0, y1, None])
                    hover_texts.append(f"{u} -> {v} ({dep_type})<br>{desc}")
                
                edge_trace = go.Scatter(
                    x=x_edges,
                    y=y_edges,
                    line=dict(width=2, color=edge_colors[dep_type]),
                    hoverinfo='text',
                    mode='lines',
                    name=dep_type,
                    text=hover_texts
                )
                
                edge_traces.append(edge_trace)
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_components = []
            
            for node in self.graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Get components for color
                comps = self.flag_components.get(node, set())
                node_components.append(list(c.value for c in comps))
                
                # Create hover text
                hover_text = f"<b>{node}</b><br>"
                hover_text += f"Description: {self.graph.nodes[node].get('description', '')}<br>"
                hover_text += f"Owner: {self.graph.nodes[node].get('owner', '')}<br>"
                hover_text += f"Components: {', '.join(c.value for c in comps)}"
                node_text.append(hover_text)
            
            # Determine node colors based on components
            node_colors = []
            for comps in node_components:
                if 'core' in comps:
                    node_colors.append('red')
                elif 'trading' in comps:
                    node_colors.append('green')
                elif 'ui' in comps:
                    node_colors.append('blue')
                else:
                    node_colors.append('gray')
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    showscale=False,
                    color=node_colors,
                    size=15,
                    line=dict(width=2, color='black')
                ),
                text=node_text,
                name='Flags'
            )
            
            # Create figure
            fig = go.Figure(
                data=edge_traces + [node_trace],
                layout=go.Layout(
                    title='Feature Flag Dependencies (Interactive)',
                    titlefont=dict(size=16),
                    showlegend=True,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive graph: {str(e)}")
            return None
    
    def generate_dot_file(self, output_file: str) -> bool:
        """
        Generate a DOT file for visualization in tools like Graphviz.
        
        Args:
            output_file: Path to save the DOT file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert graph to DOT format
            dot_data = nx.drawing.nx_pydot.to_pydot(self.graph)
            
            # Enhance the DOT representation
            for i, node in enumerate(dot_data.get_nodes()):
                node_name = node.get_name().strip('"')
                
                # Skip any non-existent nodes
                if node_name not in self.graph.nodes:
                    continue
                
                # Add label with description
                label = f"{node_name}\n"
                desc = self.graph.nodes[node_name].get("description", "")
                if desc:
                    # Truncate long descriptions
                    if len(desc) > 30:
                        desc = desc[:27] + "..."
                    label += f"{desc}\n"
                
                # Add components
                comps = self.flag_components.get(node_name, set())
                if comps:
                    comp_str = ", ".join(c.value for c in comps)
                    label += f"[{comp_str}]"
                
                node.set_label(f'"{label}"')
                
                # Set color based on components
                if any(c == SystemComponent.CORE for c in comps):
                    node.set_fillcolor('"#ffcccc"')  # Light red
                elif any(c == SystemComponent.TRADING for c in comps):
                    node.set_fillcolor('"#ccffcc"')  # Light green
                elif any(c == SystemComponent.UI for c in comps):
                    node.set_fillcolor('"#ccccff"')  # Light blue
                else:
                    node.set_fillcolor('"#eeeeee"')  # Light gray
                
                node.set_style('"filled"')
            
            # Enhance edges with colors for dependency types
            for edge in dot_data.get_edges():
                source = edge.get_source().strip('"')
                target = edge.get_destination().strip('"')
                
                # Get edge data
                if self.graph.has_edge(source, target):
                    data = self.graph.get_edge_data(source, target)
                    dep_type = data.get('type', '')
                    
                    # Set color and label based on dependency type
                    if dep_type == DependencyType.REQUIRES.value:
                        edge.set_color('"red"')
                        edge.set_label('"requires"')
                    elif dep_type == DependencyType.CONFLICTS.value:
                        edge.set_color('"orange"')
                        edge.set_label('"conflicts"')
                    elif dep_type == DependencyType.ENHANCES.value:
                        edge.set_color('"green"')
                        edge.set_label('"enhances"')
                    elif dep_type == DependencyType.ALTERNATIVE.value:
                        edge.set_color('"blue"')
                        edge.set_label('"alternative"')
            
            # Write to file
            with open(output_file, 'w') as f:
                f.write(dot_data.to_string())
            
            logger.info(f"Generated DOT file at {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating DOT file: {str(e)}")
            return False
    
    def export_to_json(self, output_file: str) -> bool:
        """
        Export the dependency graph to a JSON file.
        
        Args:
            output_file: Path to save the JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert graph to dictionary
            data = {
                "generated_at": datetime.now().isoformat(),
                "flags": [],
                "dependencies": [],
                "components": {}
            }
            
            # Add flags
            for flag_name in self.graph.nodes:
                flag_data = {
                    "name": flag_name,
                    **self.graph.nodes[flag_name]
                }
                data["flags"].append(flag_data)
                
                # Add components
                components = self.flag_components.get(flag_name, set())
                data["components"][flag_name] = [c.value for c in components]
                
            # Add dependencies
            for source, target, edge_data in self.graph.edges(data=True):
                data["dependencies"].append({
                    "source": source,
                    "target": target,
                    **edge_data
                })
                
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Exported dependency graph to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting to JSON: {str(e)}")
            return False

class FlagDependencyAnalyzer:
    """
    Analyzes feature flag dependencies to identify potential issues.
    
    This class uses the dependency graph to identify patterns,
    potential issues, and optimization opportunities in feature flags.
    """
    
    def __init__(self, dependency_graph: DependencyGraph):
        """
        Initialize the analyzer.
        
        Args:
            dependency_graph: The graph containing flag dependencies
        """
        self.graph = dependency_graph
        
    def find_circular_dependencies(self) -> List[List[str]]:
        """
        Find circular dependencies in the graph.
        
        Returns:
            List[List[str]]: List of cycles in the graph
        """
        try:
            cycles = list(nx.simple_cycles(self.graph.graph))
            return cycles
        except Exception as e:
            logger.error(f"Error finding circular dependencies: {str(e)}")
            return []
    
    def find_isolated_flags(self) -> List[str]:
        """
        Find flags that have no dependencies.
        
        Returns:
            List[str]: List of isolated flags
        """
        isolated = []
        for node in self.graph.graph.nodes():
            if self.graph.graph.degree(node) == 0:
                isolated.append(node)
        return isolated
    
    def find_bottleneck_flags(self) -> List[Tuple[str, int]]:
        """
        Find flags that are bottlenecks (many flags depend on them).
        
        Returns:
            List[Tuple[str, int]]: List of flags and their dependent count
        """
        bottlenecks = []
        for node in self.graph.graph.nodes():
            # Count incoming edges (dependents)
            dependent_count = len(list(self.graph.graph.predecessors(node)))
            if dependent_count > 2:  # Threshold for being a bottleneck
                bottlenecks.append((node, dependent_count))
        
        # Sort by dependent count (descending)
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks
    
    def get_component_impact(self, component: SystemComponent) -> Dict[str, Any]:
        """
        Get analysis of flags affecting a specific component.
        
        Args:
            component: The system component to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        # Find flags affecting this component
        affected_flags = []
        for flag_name, components in self.graph.flag_components.items():
            if component in components:
                affected_flags.append(flag_name)
        
        # Find dependencies between these flags
        dependencies = []
        for source in affected_flags:
            for target in affected_flags:
                if source != target and self.graph.graph.has_edge(source, target):
                    edge_data = self.graph.graph.get_edge_data(source, target)
                    dependencies.append({
                        "source": source,
                        "target": target,
                        "type": edge_data.get("type", ""),
                        "description": edge_data.get("description", "")
                    })
        
        # Find external dependencies (flags outside this component)
        external_deps = []
        for flag in affected_flags:
            for dep in self.graph.graph.successors(flag):
                if dep not in affected_flags:
                    edge_data = self.graph.graph.get_edge_data(flag, dep)
                    external_deps.append({
                        "from_component": flag,
                        "to_external": dep,
                        "type": edge_data.get("type", ""),
                        "description": edge_data.get("description", "")
                    })
        
        return {
            "component": component.value,
            "flags_count": len(affected_flags),
            "flags": affected_flags,
            "internal_dependencies": dependencies,
            "external_dependencies": external_deps
        }
    
    def suggest_flag_grouping(self) -> List[Dict[str, Any]]:
        """
        Suggest logical groupings of related flags.
        
        Returns:
            List[Dict[str, Any]]: Suggested flag groups
        """
        # Use community detection to find groups
        communities = list(nx.algorithms.community.greedy_modularity_communities(
            nx.Graph(self.graph.graph)
        ))
        
        results = []
        for i, community in enumerate(communities):
            flags = list(community)
            
            # Find common components
            common_components = set()
            for flag in flags:
                if flag in self.graph.flag_components:
                    if not common_components:
                        common_components = set(self.graph.flag_components[flag])
                    else:
                        common_components &= set(self.graph.flag_components[flag])
            
            # Find internal dependencies
            internal_deps = []
            for source in flags:
                for target in flags:
                    if source != target and self.graph.graph.has_edge(source, target):
                        edge_data = self.graph.graph.get_edge_data(source, target)
                        internal_deps.append({
                            "source": source,
                            "target": target,
                            "type": edge_data.get("type", "")
                        })
            
            # Suggest a name based on components or members
            suggested_name = ""
            if common_components:
                suggested_name = f"group_{next(iter(common_components)).value}"
            else:
                # Use the name of the flag with most connections
                most_connected = max(flags, key=lambda f: self.graph.graph.degree(f), default=None)
                if most_connected:
                    suggested_name = f"group_{most_connected.split('_')[0]}"
            
            if not suggested_name:
                suggested_name = f"flag_group_{i+1}"
            
            results.append({
                "suggested_name": suggested_name,
                "flags": flags,
                "common_components": [c.value for c in common_components] if common_components else [],
                "internal_dependencies_count": len(internal_deps),
                "internal_dependencies": internal_deps
            })
        
        return results
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Dict[str, Any]: Analysis report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "flag_count": len(self.graph.graph.nodes),
            "dependency_count": len(self.graph.graph.edges),
            "circular_dependencies": self.find_circular_dependencies(),
            "isolated_flags": self.find_isolated_flags(),
            "bottleneck_flags": self.find_bottleneck_flags(),
            "component_analysis": {},
            "flag_grouping_suggestions": self.suggest_flag_grouping()
        }
        
        # Add component analysis
        for component in SystemComponent:
            report["component_analysis"][component.value] = self.get_component_impact(component)
        
        return report

def render_interactive_dashboard(dependency_graph: DependencyGraph, output_file: str) -> bool:
    """
    Create an interactive HTML dashboard for exploring the dependency graph.
    
    Args:
        dependency_graph: The graph to visualize
        output_file: Path to save the HTML dashboard
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        from plotly.offline import plot
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create interactive graph
        graph_fig = dependency_graph.create_interactive_graph()
        if not graph_fig:
            return False
        
        # Generate analyzer report
        analyzer = FlagDependencyAnalyzer(dependency_graph)
        report = analyzer.generate_analysis_report()
        
        # Create the dashboard subplot layout
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter", "colspan": 2}, None],
                   [{"type": "table"}, {"type": "table"}]],
            subplot_titles=("Feature Flag Dependency Graph", 
                           "Key Metrics", "Potential Issues")
        )
        
        # Add the dependency graph
        for trace in graph_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # Create key metrics table
        metrics_headers = ["Metric", "Value"]
        metrics_values = [
            ["Total Flags", len(dependency_graph.graph.nodes)],
            ["Total Dependencies", len(dependency_graph.graph.edges)],
            ["Isolated Flags", len(report["isolated_flags"])],
            ["Circular Dependencies", len(report["circular_dependencies"])],
            ["Flag Groups", len(report["flag_grouping_suggestions"])]
        ]
        
        metrics_table = go.Table(
            header=dict(values=metrics_headers,
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=list(zip(*metrics_values)),
                      fill_color='lavender',
                      align='left'),
            domain=dict(row=1, column=0)
        )
        
        # Create issues table
        issues_headers = ["Issue Type", "Details"]
        issues_values = []
        
        # Add circular dependencies
        for cycle in report["circular_dependencies"]:
            cycle_str = " -> ".join(cycle) + " -> " + cycle[0]
            issues_values.append(["Circular Dependency", cycle_str])
        
        # Add bottleneck flags
        for flag, count in report["bottleneck_flags"]:
            issues_values.append(["Bottleneck Flag", f"{flag} - {count} dependents"])
        
        if not issues_values:
            issues_values.append(["No Issues", "No critical issues detected"])
        
        issues_table = go.Table(
            header=dict(values=issues_headers,
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=list(zip(*issues_values)),
                      fill_color='lavender',
                      align='left'),
            domain=dict(row=1, column=1)
        )
        
        # Add tables to the figure
        fig.add_trace(metrics_table, row=2, col=1)
        fig.add_trace(issues_table, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title_text="Feature Flag Dependencies Dashboard",
            showlegend=True,
            height=900,
            width=1200
        )
        
        # Save the dashboard
        plot(fig, filename=output_file, auto_open=False)
        
        logger.info(f"Generated interactive dashboard at {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return False

# Helper functions for external use

def create_dependency_graph(storage_path: str = "data/feature_flags") -> DependencyGraph:
    """
    Create and initialize a dependency graph.
    
    Args:
        storage_path: Path to store dependency data
        
    Returns:
        DependencyGraph: Initialized graph
    """
    return DependencyGraph(storage_path)

def visualize_dependencies(graph: DependencyGraph, 
                         output_path: str = "data/feature_flags/visualizations",
                         file_format: str = "all") -> Dict[str, str]:
    """
    Generate visualizations of the dependency graph in different formats.
    
    Args:
        graph: The dependency graph to visualize
        output_path: Path to save visualizations
        file_format: Format to generate (png, html, dot, json, or all)
        
    Returns:
        Dict[str, str]: Paths to generated files
    """
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {}
    
    if file_format in ["png", "all"]:
        png_file = os.path.join(output_path, f"dependencies_{timestamp}.png")
        if graph.visualize_graph(png_file):
            results["png"] = png_file
    
    if file_format in ["html", "all"]:
        html_file = os.path.join(output_path, f"dependencies_{timestamp}.html")
        if render_interactive_dashboard(graph, html_file):
            results["html"] = html_file
    
    if file_format in ["dot", "all"]:
        dot_file = os.path.join(output_path, f"dependencies_{timestamp}.dot")
        if graph.generate_dot_file(dot_file):
            results["dot"] = dot_file
    
    if file_format in ["json", "all"]:
        json_file = os.path.join(output_path, f"dependencies_{timestamp}.json")
        if graph.export_to_json(json_file):
            results["json"] = json_file
    
    return results

def analyze_dependencies(graph: DependencyGraph, 
                       output_file: str = None) -> Dict[str, Any]:
    """
    Analyze feature flag dependencies and generate a report.
    
    Args:
        graph: The dependency graph to analyze
        output_file: Optional path to save the report
        
    Returns:
        Dict[str, Any]: Analysis report
    """
    analyzer = FlagDependencyAnalyzer(graph)
    report = analyzer.generate_analysis_report()
    
    if output_file:
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved dependency analysis report to {output_file}")
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
    
    return report

def create_sample_dependencies() -> DependencyGraph:
    """
    Create a sample dependency graph for testing and demonstration.
    
    Returns:
        DependencyGraph: Sample graph with test data
    """
    graph = DependencyGraph()
    
    # Add some flags
    graph.add_flag("core_metrics", "Core metrics collection", "system", 
                 [SystemComponent.CORE, SystemComponent.MONITORING])
    
    graph.add_flag("advanced_charting", "Advanced charting features", "ui_team", 
                 [SystemComponent.UI])
    
    graph.add_flag("market_data_cache", "Cache market data for faster access", "data_team", 
                 [SystemComponent.CORE, SystemComponent.TRADING])
    
    graph.add_flag("trade_simulation", "Simulated trading for testing", "trading_team", 
                 [SystemComponent.TRADING, SystemComponent.BACKTESTING])
    
    graph.add_flag("portfolio_analytics", "Advanced portfolio analytics", "analytics_team", 
                 [SystemComponent.ANALYTICS, SystemComponent.UI])
    
    graph.add_flag("risk_alerts", "Real-time risk alerts", "risk_team", 
                 [SystemComponent.TRADING, SystemComponent.MONITORING])
    
    graph.add_flag("dark_mode", "Dark mode UI", "ui_team", 
                 [SystemComponent.UI])
    
    graph.add_flag("trade_confirmation", "Enhanced trade confirmation UI", "ui_team", 
                 [SystemComponent.UI, SystemComponent.TRADING])
    
    # Add dependencies
    graph.add_dependency("advanced_charting", "core_metrics", 
                       DependencyType.REQUIRES, "Needs metrics data")
    
    graph.add_dependency("portfolio_analytics", "core_metrics", 
                       DependencyType.REQUIRES, "Needs metrics data")
    
    graph.add_dependency("risk_alerts", "core_metrics", 
                       DependencyType.REQUIRES, "Needs metrics data")
    
    graph.add_dependency("trade_simulation", "market_data_cache", 
                       DependencyType.ENHANCES, "Performs better with cache")
    
    graph.add_dependency("dark_mode", "advanced_charting", 
                       DependencyType.CONFLICTS, "Has theme compatibility issues")
    
    graph.add_dependency("trade_confirmation", "trade_simulation", 
                       DependencyType.CONFLICTS, "Cannot confirm simulated trades")
    
    graph.add_dependency("portfolio_analytics", "risk_alerts", 
                       DependencyType.ENHANCES, "Provides better context for alerts")
    
    return graph 