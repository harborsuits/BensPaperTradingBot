"""
Feature Flag Dependencies Visualizer

This module provides visualization tools for feature flag dependencies,
helping understand the relationships between different flags and their impact.
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .service import get_feature_flag_service, FeatureFlag

logger = logging.getLogger(__name__)

class FlagDependencyVisualizer:
    """Visualizes feature flag dependencies as a graph."""
    
    def __init__(
        self,
        output_dir: str = "data/feature_flags/visualizations",
        figsize: Tuple[int, int] = (12, 10)
    ):
        """Initialize the flag dependency visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            figsize: Figure size (width, height) in inches
        """
        self.output_dir = output_dir
        self.figsize = figsize
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Flag dependency visualizer initialized")
    
    def build_dependency_graph(self) -> nx.DiGraph:
        """Build a directed graph of flag dependencies.
        
        Returns:
            nx.DiGraph: Directed graph of flag dependencies
        """
        # Get feature flag service
        service = get_feature_flag_service()
        
        # Get all flags
        flags = service.list_flags()
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes (flags)
        for flag in flags:
            G.add_node(
                flag.id,
                name=flag.name,
                category=flag.category.name,
                enabled=flag.enabled,
                dependent_flags=list(flag.dependent_flags)
            )
        
        # Add edges (dependencies)
        for flag in flags:
            for dep_id in flag.dependent_flags:
                # Check if dependency exists
                if dep_id in G:
                    G.add_edge(dep_id, flag.id)
        
        return G
    
    def visualize_dependencies(
        self,
        filename: Optional[str] = None,
        show: bool = False,
        highlight_flags: Optional[List[str]] = None,
        metric_values: Optional[Dict[str, float]] = None
    ) -> str:
        """Visualize flag dependencies.
        
        Args:
            filename: Output filename (without extension)
            show: Whether to show the plot interactively
            highlight_flags: List of flags to highlight
            metric_values: Metric values to use for node sizing
            
        Returns:
            str: Path to saved visualization file
        """
        # Build graph
        G = self.build_dependency_graph()
        
        # Set up filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flag_dependencies_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Set up layout
        if len(G) <= 10:
            # For small graphs, use a more structured layout
            pos = nx.spring_layout(G, seed=42, k=0.8)
        else:
            # For larger graphs, use hierarchical layout
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
        
        # Calculate node sizes based on dependencies
        if metric_values:
            # Use provided metric values
            node_sizes = []
            for node in G.nodes():
                if node in metric_values:
                    # Scale between 300 and 1500
                    size = 300 + min(1200, abs(metric_values[node]) * 100)
                    node_sizes.append(size)
                else:
                    node_sizes.append(500)  # Default size
        else:
            # Use number of dependencies as size
            node_sizes = [300 + 200 * (G.out_degree(node) + G.in_degree(node)) for node in G.nodes()]
        
        # Generate node colors based on enabled status and category
        node_colors = []
        for node in G.nodes():
            data = G.nodes[node]
            enabled = data.get("enabled", False)
            category = data.get("category", "")
            
            # Base color on category
            if category == "RISK":
                base_color = "red" if enabled else "lightcoral"
            elif category == "STRATEGY":
                base_color = "green" if enabled else "lightgreen"
            elif category == "EXPERIMENTAL":
                base_color = "purple" if enabled else "plum"
            else:
                base_color = "blue" if enabled else "lightblue"
            
            node_colors.append(base_color)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            node_shape="o"
        )
        
        # Highlight specific nodes if requested
        if highlight_flags:
            highlight_nodes = [node for node in G.nodes() if node in highlight_flags]
            if highlight_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=highlight_nodes,
                    node_size=[node_sizes[list(G.nodes()).index(node)] for node in highlight_nodes],
                    node_color="yellow",
                    node_shape="*",
                    alpha=1.0,
                    linewidths=2
                )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            arrows=True,
            arrowsize=15,
            width=1.5,
            alpha=0.7,
            connectionstyle="arc3,rad=0.1"  # Curved edges
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_family="sans-serif",
            font_weight="bold"
        )
        
        # Add title and adjust layout
        plt.title("Feature Flag Dependencies", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        
        # Save and show
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved visualization to {filepath}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return filepath
    
    def visualize_impact_heatmap(
        self,
        metric_impacts: Dict[str, Dict[str, float]],
        filename: Optional[str] = None,
        show: bool = False
    ) -> str:
        """Visualize metric impacts as a heatmap.
        
        Args:
            metric_impacts: Dict mapping flag IDs to dicts of metric impacts
            filename: Output filename (without extension)
            show: Whether to show the plot interactively
            
        Returns:
            str: Path to saved visualization file
        """
        # Set up filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flag_impact_heatmap_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        
        # Get all flags and metrics
        flags = sorted(list(metric_impacts.keys()))
        all_metrics = set()
        for impacts in metric_impacts.values():
            all_metrics.update(impacts.keys())
        metrics = sorted(list(all_metrics))
        
        # Create data matrix
        data = np.zeros((len(flags), len(metrics)))
        for i, flag in enumerate(flags):
            for j, metric in enumerate(metrics):
                if flag in metric_impacts and metric in metric_impacts[flag]:
                    data[i, j] = metric_impacts[flag][metric]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create custom colormap for positive/negative values
        colors = ["red", "white", "green"]
        cmap = LinearSegmentedColormap.from_list("RdWtGn", colors, N=100)
        
        # Normalize data for colormap
        vmax = np.max(np.abs(data))
        norm = plt.Normalize(-vmax, vmax)
        
        # Create heatmap
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Impact (%)")
        
        # Add labels
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(flags)))
        ax.set_xticklabels(metrics, rotation=45, ha="right")
        ax.set_yticklabels(flags)
        
        # Add grid
        ax.set_xticks(np.arange(-.5, len(metrics), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(flags), 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
        
        # Add title
        plt.title("Feature Flag Impact on Metrics", fontsize=16)
        
        # Add text annotations
        for i in range(len(flags)):
            for j in range(len(metrics)):
                value = data[i, j]
                if abs(value) > 0.5:  # Only show significant values
                    text_color = "black" if abs(value) < 10 else "white"
                    ax.text(j, i, f"{value:.1f}%", 
                            ha="center", va="center", 
                            color=text_color, fontweight="bold")
        
        plt.tight_layout()
        
        # Save and show
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved impact heatmap to {filepath}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return filepath
    
    def visualize_flag_status_timeline(
        self,
        flag_history: Dict[str, List[Dict[str, Any]]],
        filename: Optional[str] = None,
        show: bool = False,
        days: int = 30
    ) -> str:
        """Visualize flag status changes over time.
        
        Args:
            flag_history: Dict mapping flag IDs to lists of status changes
            filename: Output filename (without extension)
            show: Whether to show the plot interactively
            days: Number of days to show in the timeline
            
        Returns:
            str: Path to saved visualization file
        """
        # Set up filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flag_status_timeline_{timestamp}"
        
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        
        # Sort flags by importance/category
        flags = sorted(flag_history.keys())
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create timeline
        now = datetime.now()
        start_date = now - datetime.timedelta(days=days)
        
        # Set up y-axis (flags)
        y_positions = range(len(flags))
        
        # Draw background grid
        ax.set_yticks(y_positions)
        ax.set_yticklabels(flags)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Plot status changes for each flag
        for i, flag_id in enumerate(flags):
            history = flag_history[flag_id]
            
            # Filter to changes within the time window
            relevant_history = [
                change for change in history
                if datetime.fromisoformat(change["timestamp"]) >= start_date
            ]
            
            # Sort by timestamp
            relevant_history.sort(key=lambda x: x["timestamp"])
            
            # Add current status if no recent changes
            if not relevant_history:
                service = get_feature_flag_service()
                flag = service.get_flag(flag_id)
                if flag:
                    relevant_history.append({
                        "timestamp": start_date.isoformat(),
                        "enabled": flag.enabled
                    })
            
            # Plot status segments
            for j in range(len(relevant_history)):
                change = relevant_history[j]
                start = datetime.fromisoformat(change["timestamp"])
                
                # Determine end time (next change or now)
                if j < len(relevant_history) - 1:
                    end = datetime.fromisoformat(relevant_history[j+1]["timestamp"])
                else:
                    end = now
                
                # Handle changes before the time window
                if start < start_date:
                    start = start_date
                
                # Skip if outside time window
                if end < start_date or start > now:
                    continue
                
                # Plot segment
                color = "green" if change.get("enabled", False) else "red"
                ax.plot(
                    [start, end], [i, i],
                    color=color, linewidth=6, solid_capstyle="butt",
                    alpha=0.8
                )
        
        # Set up x-axis
        ax.set_xlim(start_date, now)
        
        # Format dates on x-axis
        plt.gcf().autofmt_xdate()
        
        # Add title and labels
        plt.title("Feature Flag Status Timeline", fontsize=16)
        plt.xlabel("Date/Time")
        plt.ylabel("Feature Flag")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="green", lw=4, label="Enabled"),
            Line2D([0], [0], color="red", lw=4, label="Disabled")
        ]
        ax.legend(handles=legend_elements, loc="upper right")
        
        plt.tight_layout()
        
        # Save and show
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        logger.info(f"Saved status timeline to {filepath}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return filepath
    
    def generate_comprehensive_report(
        self,
        metric_impacts: Optional[Dict[str, Dict[str, float]]] = None,
        flag_history: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        highlight_flags: Optional[List[str]] = None,
        filename_prefix: Optional[str] = None,
        show: bool = False
    ) -> List[str]:
        """Generate a comprehensive visual report with multiple visualizations.
        
        Args:
            metric_impacts: Dict mapping flag IDs to dicts of metric impacts
            flag_history: Dict mapping flag IDs to lists of status changes
            highlight_flags: List of flags to highlight in dependency graph
            filename_prefix: Prefix for output filenames
            show: Whether to show the plots interactively
            
        Returns:
            List[str]: Paths to saved visualization files
        """
        # Generate timestamp for filenames
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            filename_prefix = f"flag_report_{timestamp}"
        
        # List to store generated file paths
        files = []
        
        # Generate dependency graph
        dep_file = self.visualize_dependencies(
            filename=f"{filename_prefix}_dependencies",
            show=show,
            highlight_flags=highlight_flags
        )
        files.append(dep_file)
        
        # Generate impact heatmap if data provided
        if metric_impacts:
            heatmap_file = self.visualize_impact_heatmap(
                metric_impacts=metric_impacts,
                filename=f"{filename_prefix}_impact_heatmap",
                show=show
            )
            files.append(heatmap_file)
        
        # Generate status timeline if data provided
        if flag_history:
            timeline_file = self.visualize_flag_status_timeline(
                flag_history=flag_history,
                filename=f"{filename_prefix}_status_timeline",
                show=show
            )
            files.append(timeline_file)
        
        return files


def get_visualizer() -> FlagDependencyVisualizer:
    """Get a flag dependency visualizer instance.
    
    Returns:
        FlagDependencyVisualizer: Visualizer instance
    """
    return FlagDependencyVisualizer() 