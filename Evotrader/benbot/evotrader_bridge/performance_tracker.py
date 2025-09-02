"""
Performance Tracker for BensBot-EvoTrader Integration

This module tracks and stores performance metrics for strategies, enabling
evaluation, comparison, and promotion decisions for the evolutionary system.
"""

# Add EvoTrader to Python path
import evotrader_path

import os
import json
import time
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceMetric:
    """Defines a performance metric with metadata."""
    
    def __init__(
        self, 
        name: str,
        description: str,
        higher_is_better: bool = True,
        weight: float = 1.0,
        normalization_func = None
    ):
        self.name = name
        self.description = description
        self.higher_is_better = higher_is_better
        self.weight = weight
        self.normalization_func = normalization_func or (lambda x: x)


class PerformanceTracker:
    """Tracks and stores performance metrics for trading strategies."""
    
    # Standard performance metrics
    STANDARD_METRICS = [
        PerformanceMetric(
            name="profit",
            description="Total profit percentage",
            higher_is_better=True,
            weight=0.5
        ),
        PerformanceMetric(
            name="win_rate",
            description="Percentage of winning trades",
            higher_is_better=True,
            weight=0.3
        ),
        PerformanceMetric(
            name="max_drawdown",
            description="Maximum drawdown percentage",
            higher_is_better=False,
            weight=0.2,
            normalization_func=lambda x: max(0, 1 - (x / 100))  # Convert to 0-1 score where 1 is best
        ),
        PerformanceMetric(
            name="sharpe_ratio",
            description="Sharpe ratio (risk-adjusted return)",
            higher_is_better=True,
            weight=0.4
        ),
        PerformanceMetric(
            name="trade_count", 
            description="Number of trades executed",
            higher_is_better=True,
            weight=0.1
        )
    ]
    
    def __init__(self, db_path: str = None):
        """
        Initialize the performance tracker.
        
        Args:
            db_path: Path to SQLite database for storing metrics
        """
        # Set up database
        self.db_path = db_path or os.path.join("strategy_evolution", "performance.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Set up metrics
        self.metrics = {m.name: m for m in self.STANDARD_METRICS}
        
        # Configure logging
        self.logger = logging.getLogger(f"{__name__}.performance_tracker")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Performance tracker initialized with database at {self.db_path}")
    
    def _init_database(self):
        """Initialize the SQLite database for storing metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id TEXT PRIMARY KEY,
            strategy_type TEXT,
            generation INTEGER,
            parent_ids TEXT,
            creation_timestamp REAL,
            created_at TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id TEXT,
            timestamp REAL,
            recorded_at TEXT,
            test_id TEXT,
            generation INTEGER,
            metrics TEXT,
            fitness_score REAL,
            FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_strategy(self, strategy_id: str, metadata: Dict[str, Any]):
        """
        Register a strategy in the performance database.
        
        Args:
            strategy_id: Unique identifier for the strategy
            metadata: Strategy metadata (type, generation, parents, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if strategy already exists
        cursor.execute("SELECT strategy_id FROM strategies WHERE strategy_id = ?", (strategy_id,))
        if cursor.fetchone():
            conn.close()
            self.logger.warning(f"Strategy {strategy_id} already registered")
            return
            
        # Extract metadata
        strategy_type = metadata.get("strategy_type", "unknown")
        generation = metadata.get("generation", 0)
        parent_ids = json.dumps(metadata.get("parent_ids", []))
        creation_timestamp = metadata.get("creation_timestamp", time.time())
        created_at = datetime.fromtimestamp(creation_timestamp).isoformat()
        
        # Insert strategy record
        cursor.execute('''
        INSERT INTO strategies 
        (strategy_id, strategy_type, generation, parent_ids, creation_timestamp, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (strategy_id, strategy_type, generation, parent_ids, creation_timestamp, created_at))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Registered strategy {strategy_id} (type: {strategy_type}, gen: {generation})")
    
    def record_performance(
        self, 
        strategy_id: str, 
        metrics: Dict[str, Any], 
        test_id: str = None,
        generation: int = None
    ) -> float:
        """
        Record performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            metrics: Dictionary of performance metrics
            test_id: Optional identifier for the test/evaluation
            generation: Optional generation number (defaults to strategy's registered generation)
            
        Returns:
            fitness_score: Calculated fitness score based on metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Verify strategy exists
        cursor.execute("SELECT generation FROM strategies WHERE strategy_id = ?", (strategy_id,))
        result = cursor.fetchone()
        
        if not result:
            # Auto-register the strategy if it doesn't exist
            self.register_strategy(strategy_id, {
                "strategy_type": "unknown",
                "generation": generation or 0,
                "parent_ids": [],
                "creation_timestamp": time.time()
            })
            # Default to generation 0 if not specified
            strategy_generation = generation or 0
        else:
            # Use strategy's registered generation if not overridden
            strategy_generation = generation if generation is not None else result[0]
        
        # Calculate fitness score
        fitness_score = self.calculate_fitness(metrics)
        
        # Prepare record
        timestamp = time.time()
        recorded_at = datetime.fromtimestamp(timestamp).isoformat()
        metrics_json = json.dumps(metrics)
        test_id = test_id or f"test_{int(timestamp)}"
        
        # Insert performance record
        cursor.execute('''
        INSERT INTO performance_records
        (strategy_id, timestamp, recorded_at, test_id, generation, metrics, fitness_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (strategy_id, timestamp, recorded_at, test_id, strategy_generation, 
             metrics_json, fitness_score))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Recorded performance for strategy {strategy_id} with fitness {fitness_score:.4f}")
        
        return fitness_score
    
    def calculate_fitness(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate fitness score from performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            fitness_score: Normalized fitness score (0-1)
        """
        fitness_components = []
        total_weight = 0
        
        # Process each metric that we have a definition for
        for metric_name, metric_def in self.metrics.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Apply normalization function
                norm_value = metric_def.normalization_func(value)
                
                # Flip value if lower is better
                if not metric_def.higher_is_better:
                    norm_value = 1 - norm_value
                
                # Cap to 0-1 range
                norm_value = max(0, min(1, norm_value))
                
                # Add to weighted sum
                fitness_components.append(norm_value * metric_def.weight)
                total_weight += metric_def.weight
        
        # Calculate weighted average
        if total_weight > 0:
            fitness_score = sum(fitness_components) / total_weight
        else:
            fitness_score = 0
            
        return fitness_score
    
    def get_strategy_metrics(
        self, 
        strategy_id: str,
        limit: int = 10,
        include_metrics: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
            limit: Maximum number of records to return
            include_metrics: Whether to include full metrics
            
        Returns:
            List of performance records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT record_id, timestamp, recorded_at, test_id, generation, metrics, fitness_score
        FROM performance_records
        WHERE strategy_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (strategy_id, limit))
        
        records = []
        for row in cursor.fetchall():
            record = {
                "record_id": row[0],
                "timestamp": row[1],
                "recorded_at": row[2],
                "test_id": row[3],
                "generation": row[4],
                "fitness_score": row[6]
            }
            
            if include_metrics:
                record["metrics"] = json.loads(row[5])
                
            records.append(record)
        
        conn.close()
        return records
    
    def get_best_strategies(
        self, 
        generation: int = None,
        limit: int = 10,
        test_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get the best performing strategies.
        
        Args:
            generation: Filter by generation (None for all)
            limit: Maximum number of strategies to return
            test_id: Filter by specific test ID (None for most recent)
            
        Returns:
            List of strategies with performance data, sorted by fitness
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Use Row for dict-like access
        cursor = conn.cursor()
        
        query = '''
        SELECT s.strategy_id, s.strategy_type, s.generation, s.parent_ids,
               p.fitness_score, p.metrics, p.test_id, p.recorded_at
        FROM strategies s
        JOIN performance_records p ON s.strategy_id = p.strategy_id
        '''
        
        params = []
        where_clauses = []
        
        if generation is not None:
            where_clauses.append("s.generation = ?")
            params.append(generation)
            
        if test_id:
            where_clauses.append("p.test_id = ?")
            params.append(test_id)
        else:
            # Get the most recent record for each strategy
            query = '''
            SELECT s.strategy_id, s.strategy_type, s.generation, s.parent_ids,
                   p.fitness_score, p.metrics, p.test_id, p.recorded_at
            FROM strategies s
            JOIN (
                SELECT strategy_id, MAX(timestamp) as max_time
                FROM performance_records
                GROUP BY strategy_id
            ) latest ON s.strategy_id = latest.strategy_id
            JOIN performance_records p ON p.strategy_id = latest.strategy_id 
                                      AND p.timestamp = latest.max_time
            '''
        
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
            
        query += " ORDER BY p.fitness_score DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        results = []
        for row in cursor:
            strategy_data = {
                "strategy_id": row["strategy_id"],
                "strategy_type": row["strategy_type"],
                "generation": row["generation"],
                "parent_ids": json.loads(row["parent_ids"]),
                "fitness_score": row["fitness_score"],
                "test_id": row["test_id"],
                "recorded_at": row["recorded_at"],
                "metrics": json.loads(row["metrics"])
            }
            results.append(strategy_data)
        
        conn.close()
        return results
    
    def get_generation_performance(self, generation: int) -> Dict[str, Any]:
        """
        Get aggregated performance metrics for a generation.
        
        Args:
            generation: Generation number
            
        Returns:
            Dictionary with aggregated statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Count strategies in generation
        cursor.execute('''
        SELECT COUNT(*) FROM strategies WHERE generation = ?
        ''', (generation,))
        strategy_count = cursor.fetchone()[0]
        
        # Get all performance records for generation (most recent per strategy)
        cursor.execute('''
        SELECT s.strategy_id, p.fitness_score, p.metrics
        FROM strategies s
        JOIN (
            SELECT strategy_id, MAX(timestamp) as max_time
            FROM performance_records
            WHERE generation = ?
            GROUP BY strategy_id
        ) latest ON s.strategy_id = latest.strategy_id
        JOIN performance_records p ON p.strategy_id = latest.strategy_id 
                                  AND p.timestamp = latest.max_time
        WHERE s.generation = ?
        ''', (generation, generation))
        
        results = cursor.fetchall()
        
        # Initialize statistics
        stats = {
            "generation": generation,
            "strategy_count": strategy_count,
            "recorded_strategies": len(results),
            "avg_fitness": 0,
            "max_fitness": 0,
            "min_fitness": 1.0 if results else 0,
            "metric_averages": {},
            "top_strategy_id": None
        }
        
        if not results:
            return stats
            
        # Process results
        all_metrics = {}
        total_fitness = 0
        
        for row in results:
            strategy_id, fitness_score, metrics_json = row
            
            # Update fitness stats
            total_fitness += fitness_score
            stats["max_fitness"] = max(stats["max_fitness"], fitness_score)
            stats["min_fitness"] = min(stats["min_fitness"], fitness_score)
            
            if fitness_score == stats["max_fitness"]:
                stats["top_strategy_id"] = strategy_id
                
            # Extract metrics for averaging
            metrics = json.loads(metrics_json)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
        
        # Calculate averages
        stats["avg_fitness"] = total_fitness / len(results)
        
        for key, values in all_metrics.items():
            stats["metric_averages"][key] = sum(values) / len(values)
        
        conn.close()
        return stats
    
    def generate_performance_report(
        self, 
        generations: List[int] = None,
        output_file: str = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report across generations.
        
        Args:
            generations: List of generations to include (None for all)
            output_file: Path to save JSON report (None to skip saving)
            
        Returns:
            Report dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all generations if not specified
        if generations is None:
            cursor.execute("SELECT DISTINCT generation FROM strategies ORDER BY generation")
            generations = [row[0] for row in cursor.fetchall()]
            
        # Initialize report
        report = {
            "timestamp": datetime.now().isoformat(),
            "generations_analyzed": generations,
            "total_strategies": 0,
            "generation_stats": {},
            "top_strategies": {},
            "performance_trends": {
                "avg_fitness": [],
                "max_fitness": [],
                "metric_trends": {}
            }
        }
        
        # Generate stats for each generation
        for gen in generations:
            gen_stats = self.get_generation_performance(gen)
            report["generation_stats"][gen] = gen_stats
            report["total_strategies"] += gen_stats["strategy_count"]
            
            # Track performance trends
            report["performance_trends"]["avg_fitness"].append(gen_stats["avg_fitness"])
            report["performance_trends"]["max_fitness"].append(gen_stats["max_fitness"])
            
            # Track metric trends
            for metric, value in gen_stats.get("metric_averages", {}).items():
                if metric not in report["performance_trends"]["metric_trends"]:
                    report["performance_trends"]["metric_trends"][metric] = []
                report["performance_trends"]["metric_trends"][metric].append(value)
            
            # Get top strategies for this generation
            top_strategies = self.get_best_strategies(generation=gen, limit=3)
            report["top_strategies"][gen] = top_strategies
        
        # Add overall best strategies
        report["overall_best_strategies"] = self.get_best_strategies(limit=5)
        
        conn.close()
        
        # Save report if requested
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f"Saved performance report to {output_file}")
            
        return report
    
    def export_to_pandas(self, query: str = None) -> pd.DataFrame:
        """
        Export performance data to pandas DataFrame for analysis.
        
        Args:
            query: Custom SQL query (None for all performance records)
            
        Returns:
            DataFrame containing performance data
        """
        conn = sqlite3.connect(self.db_path)
        
        if query is None:
            query = '''
            SELECT s.strategy_id, s.strategy_type, s.generation, s.parent_ids,
                   p.timestamp, p.test_id, p.fitness_score, p.metrics
            FROM strategies s
            JOIN performance_records p ON s.strategy_id = p.strategy_id
            ORDER BY p.timestamp DESC
            '''
            
        # Load data
        df = pd.read_sql_query(query, conn)
        
        # Expand metrics JSON into separate columns
        if 'metrics' in df.columns:
            # Convert metrics JSON to dict
            df['metrics'] = df['metrics'].apply(json.loads)
            
            # Extract frequently used metrics as separate columns
            common_metrics = ['profit', 'win_rate', 'max_drawdown', 'sharpe_ratio', 'trade_count']
            for metric in common_metrics:
                df[metric] = df['metrics'].apply(lambda x: x.get(metric, None))
        
        conn.close()
        return df
    
    def plot_fitness_evolution(self, output_file: str = None):
        """
        Generate plot showing fitness evolution across generations.
        
        Args:
            output_file: Path to save plot image (None to display)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Get generation stats
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT DISTINCT generation FROM strategies ORDER BY generation
            ''')
            generations = [row[0] for row in cursor.fetchall()]
            
            avg_fitness = []
            max_fitness = []
            min_fitness = []
            
            for gen in generations:
                stats = self.get_generation_performance(gen)
                avg_fitness.append(stats["avg_fitness"])
                max_fitness.append(stats["max_fitness"])
                min_fitness.append(stats["min_fitness"])
            
            conn.close()
            
            # Generate plot
            plt.figure(figsize=(12, 6))
            plt.plot(generations, avg_fitness, 'b-', label='Average Fitness')
            plt.plot(generations, max_fitness, 'g-', label='Max Fitness')
            plt.plot(generations, min_fitness, 'r-', label='Min Fitness')
            
            plt.xlabel('Generation')
            plt.ylabel('Fitness Score')
            plt.title('Evolution of Strategy Fitness Across Generations')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved fitness evolution plot to {output_file}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.error("Matplotlib not available for plotting")
    
    def compare_strategies(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """
        Generate detailed comparison between multiple strategies.
        
        Args:
            strategy_ids: List of strategy IDs to compare
            
        Returns:
            Comparison data dictionary
        """
        comparison = {
            "strategies": {},
            "metrics_comparison": {},
            "common_ancestors": set(),
            "generation_spread": {
                "min": float('inf'),
                "max": 0,
                "range": 0
            }
        }
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get basic info for each strategy
        for strategy_id in strategy_ids:
            cursor.execute('''
            SELECT s.strategy_type, s.generation, s.parent_ids,
                   p.fitness_score, p.metrics
            FROM strategies s
            JOIN (
                SELECT strategy_id, MAX(timestamp) as max_time
                FROM performance_records
                WHERE strategy_id = ?
                GROUP BY strategy_id
            ) latest ON s.strategy_id = latest.strategy_id
            JOIN performance_records p ON p.strategy_id = latest.strategy_id 
                                      AND p.timestamp = latest.max_time
            WHERE s.strategy_id = ?
            ''', (strategy_id, strategy_id))
            
            row = cursor.fetchone()
            if not row:
                continue
                
            strategy_type, generation, parent_ids_json, fitness_score, metrics_json = row
            
            # Parse JSON fields
            parent_ids = json.loads(parent_ids_json)
            metrics = json.loads(metrics_json)
            
            # Store strategy data
            comparison["strategies"][strategy_id] = {
                "type": strategy_type,
                "generation": generation,
                "parent_ids": parent_ids,
                "fitness": fitness_score,
                "metrics": metrics
            }
            
            # Update generation spread
            comparison["generation_spread"]["min"] = min(comparison["generation_spread"]["min"], generation)
            comparison["generation_spread"]["max"] = max(comparison["generation_spread"]["max"], generation)
            
            # Collect metrics for comparison
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    if metric not in comparison["metrics_comparison"]:
                        comparison["metrics_comparison"][metric] = {}
                    comparison["metrics_comparison"][metric][strategy_id] = value
        
        # Calculate generation range
        gen_min = comparison["generation_spread"]["min"]
        gen_max = comparison["generation_spread"]["max"]
        comparison["generation_spread"]["range"] = gen_max - gen_min
        
        # Adjust min value if no strategies were found
        if gen_min == float('inf'):
            comparison["generation_spread"]["min"] = 0
            
        # Find common ancestors
        common_ancestors = None
        for strategy_id, data in comparison["strategies"].items():
            ancestors = set(data["parent_ids"])
            if common_ancestors is None:
                common_ancestors = ancestors
            else:
                common_ancestors &= ancestors
                
        comparison["common_ancestors"] = list(common_ancestors) if common_ancestors else []
        
        conn.close()
        return comparison
