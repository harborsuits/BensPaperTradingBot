#!/usr/bin/env python3
"""
Optimization Jobs

This module provides the core job data structures and priority calculations for the
optimization scheduler. It defines the OptimizationJob class and related functionality
for managing strategy optimization tasks.

Classes:
    OptimizationJob: Represents a scheduled optimization task with priority and metadata
    OptimizationStatus: Enum for job status values
    OptimizationMethod: Enum for optimization method types
    PriorityCalculator: Calculates job priorities based on strategy performance

Functions:
    create_optimization_job: Factory function to create a new optimization job
    get_job_store: Returns the singleton instance of the job store
"""

import os
import json
import uuid
import logging
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import heapq
import time
import numpy as np
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStatus(str, Enum):
    """Enum for possible optimization job statuses."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    

class OptimizationMethod(str, Enum):
    """Enum for optimization methods."""
    BAYESIAN = "bayesian"
    GRID = "grid"
    RANDOM = "random"
    GENETIC = "genetic"
    GRADIENT = "gradient"
    

class OptimizationPriority:
    """Constants for standard priority levels."""
    LOW = 10
    NORMAL = 50
    HIGH = 75
    CRITICAL = 100
    
    @staticmethod
    def validate(priority: float) -> float:
        """Ensure priority is within valid range (0-100)."""
        return max(0, min(100, priority))


class OptimizationJob:
    """
    Represents a scheduled optimization task with priority and metadata.
    
    Attributes:
        job_id (str): Unique identifier for the job
        strategy_id (str): Strategy to optimize
        version_id (str): Current version being optimized
        scheduled_time (datetime): When the job is scheduled to run
        priority (float): Job priority (0-100), higher is more important
        parameters (Dict): Optimization parameters
        status (OptimizationStatus): Current job status
        reason (str): Reason the job was scheduled
        creation_time (datetime): When the job was created
        last_updated (datetime): When the job was last updated
    """
    
    def __init__(
        self,
        strategy_id: str,
        version_id: str,
        scheduled_time: datetime,
        priority: float = OptimizationPriority.NORMAL,
        parameters: Optional[Dict[str, Any]] = None,
        reason: str = "manual",
        job_id: Optional[str] = None,
    ):
        """
        Initialize a new optimization job.
        
        Args:
            strategy_id: ID of the strategy to optimize
            version_id: Version ID of the strategy
            scheduled_time: When to run the optimization
            priority: Job priority (0-100)
            parameters: Optimization parameters
            reason: Why this job was scheduled
            job_id: Optional job ID, generated if not provided
        """
        self.job_id = job_id or str(uuid.uuid4())
        self.strategy_id = strategy_id
        self.version_id = version_id
        self.scheduled_time = scheduled_time
        self.priority = OptimizationPriority.validate(priority)
        self.parameters = parameters or {
            "method": OptimizationMethod.BAYESIAN.value,
            "iterations": 100,
            "target_metric": "sharpe_ratio"
        }
        self.status = OptimizationStatus.PENDING
        self.reason = reason
        self.creation_time = datetime.now()
        self.last_updated = self.creation_time
        self.results = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "strategy_id": self.strategy_id,
            "version_id": self.version_id,
            "scheduled_time": self.scheduled_time.isoformat(),
            "priority": self.priority,
            "parameters": self.parameters,
            "status": self.status.value,
            "reason": self.reason,
            "creation_time": self.creation_time.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "results": self.results,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationJob':
        """Create job from dictionary representation."""
        job = cls(
            strategy_id=data["strategy_id"],
            version_id=data["version_id"],
            scheduled_time=datetime.fromisoformat(data["scheduled_time"]),
            priority=data["priority"],
            parameters=data["parameters"],
            reason=data["reason"],
            job_id=data["job_id"]
        )
        job.status = OptimizationStatus(data["status"])
        job.creation_time = datetime.fromisoformat(data["creation_time"])
        job.last_updated = datetime.fromisoformat(data["last_updated"])
        job.results = data.get("results")
        job.error = data.get("error")
        return job
    
    def update_status(self, status: OptimizationStatus) -> None:
        """
        Update job status and last_updated timestamp.
        
        Args:
            status: New status to set
        """
        self.status = status
        self.last_updated = datetime.now()
    
    def set_results(self, results: Dict[str, Any]) -> None:
        """
        Set job results and update status to completed.
        
        Args:
            results: Optimization results
        """
        self.results = results
        self.update_status(OptimizationStatus.COMPLETED)
    
    def set_error(self, error: str) -> None:
        """
        Set job error and update status to failed.
        
        Args:
            error: Error message
        """
        self.error = error
        self.update_status(OptimizationStatus.FAILED)
    
    def is_due(self) -> bool:
        """Check if job is due for execution."""
        return (
            self.status == OptimizationStatus.SCHEDULED 
            and datetime.now() >= self.scheduled_time
        )
    
    def __lt__(self, other: 'OptimizationJob') -> bool:
        """
        Compare jobs for priority queue ordering.
        
        Jobs are ordered by:
        1. Priority (higher first)
        2. Scheduled time (earlier first)
        3. Creation time (earlier first)
        """
        if self.priority != other.priority:
            return self.priority > other.priority  # Higher priority first
        
        if self.scheduled_time != other.scheduled_time:
            return self.scheduled_time < other.scheduled_time  # Earlier time first
            
        return self.creation_time < other.creation_time  # Earlier creation first


class PriorityCalculator:
    """
    Calculates job priorities based on strategy performance metrics.
    """
    
    @staticmethod
    def calculate_priority(
        strategy_id: str,
        strategy_type: str,
        universe: str,
        performance_metrics: Dict[str, float],
        optimization_history: List[Dict[str, Any]],
        last_optimization: Optional[datetime] = None
    ) -> float:
        """
        Calculate optimization priority based on various factors.
        
        Args:
            strategy_id: Strategy identifier
            strategy_type: Type of strategy
            universe: Trading universe
            performance_metrics: Current performance metrics
            optimization_history: Previous optimization attempts
            last_optimization: When strategy was last optimized
            
        Returns:
            Priority score (0-100)
        """
        # Start with baseline priority
        priority = OptimizationPriority.NORMAL
        
        # Factor 1: Performance relative to targets
        if "sharpe_ratio" in performance_metrics:
            sharpe = performance_metrics["sharpe_ratio"]
            if sharpe < 0.5:
                priority += 20  # Poor performance, high priority
            elif sharpe < 1.0:
                priority += 10  # Mediocre performance
            elif sharpe > 2.0:
                priority -= 10  # Good performance, lower priority
        
        # Factor 2: Drawdown concerns
        if "max_drawdown" in performance_metrics:
            drawdown = abs(performance_metrics["max_drawdown"])
            if drawdown > 0.2:  # 20% drawdown
                priority += 15  # Significant drawdown, higher priority
        
        # Factor 3: Time since last optimization
        if last_optimization:
            days_since = (datetime.now() - last_optimization).days
            if days_since > 30:
                priority += min(20, days_since // 15)  # Max +20 for old optimizations
        else:
            # Never optimized - higher priority
            priority += 15
        
        # Factor 4: Previous optimization results
        if optimization_history:
            # Check if previous optimizations were effective
            recent_history = optimization_history[-3:]  # Last 3 optimizations
            effectiveness = sum(1 for h in recent_history if h.get("effective", False))
            
            if effectiveness == 0:
                priority -= 10  # Past optimizations not effective, reduce priority
            elif effectiveness >= 2:
                priority += 5  # Past optimizations effective, increase priority
        
        # Factor 5: Strategy volatility (if available)
        if "volatility" in performance_metrics:
            volatility = performance_metrics["volatility"]
            if volatility > 0.15:  # High volatility
                priority += 10
        
        # Factor 6: Strategy type adjustments
        strategy_type_bonuses = {
            "iron_condor": 5,    # More complex, benefit from optimization
            "strangle": 3,       # Sensitive to volatility
            "butterfly": 5,      # Parameter sensitive
            "calendar_spread": 8  # Very parameter sensitive
        }
        priority += strategy_type_bonuses.get(strategy_type, 0)
        
        # Factor 7: Universe-based adjustments
        universe_bonuses = {
            "options": 5,       # Options benefit more from optimization
            "crypto": 3,        # Higher volatility, more optimization benefit
            "futures": 2,       # Moderate benefit
            "equities": 0       # Baseline
        }
        priority += universe_bonuses.get(universe, 0)
        
        # Ensure priority is within valid range
        return OptimizationPriority.validate(priority)


class OptimizationJobStore:
    """
    Persistent storage and management for optimization jobs.
    
    This class handles creating, retrieving, updating, and persisting 
    optimization jobs to disk, ensuring job state is preserved across restarts.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the job store.
        
        Args:
            storage_path: Path to store job data, defaults to ~/.trading_bot/optimization
        """
        self.storage_path = storage_path or os.path.join(
            os.path.expanduser("~"), ".trading_bot", "optimization"
        )
        self.jobs_file = os.path.join(self.storage_path, "optimization_jobs.json")
        self.jobs: Dict[str, OptimizationJob] = {}
        self.lock = threading.RLock()
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing jobs from disk
        self._load_jobs()
    
    def _load_jobs(self) -> None:
        """Load jobs from disk."""
        if os.path.exists(self.jobs_file):
            try:
                with open(self.jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                
                with self.lock:
                    for job_data in jobs_data:
                        job = OptimizationJob.from_dict(job_data)
                        self.jobs[job.job_id] = job
                
                logger.info(f"Loaded {len(self.jobs)} optimization jobs from {self.jobs_file}")
            except Exception as e:
                logger.error(f"Error loading optimization jobs: {str(e)}")
                # Create backup of corrupted file
                if os.path.exists(self.jobs_file):
                    backup_file = f"{self.jobs_file}.bak.{int(time.time())}"
                    try:
                        os.rename(self.jobs_file, backup_file)
                        logger.info(f"Created backup of jobs file at {backup_file}")
                    except Exception as be:
                        logger.error(f"Error creating backup: {str(be)}")
    
    def _save_jobs(self) -> None:
        """Save jobs to disk."""
        try:
            with self.lock:
                jobs_data = [job.to_dict() for job in self.jobs.values()]
            
            # Write to temporary file first, then rename for atomic update
            temp_file = f"{self.jobs_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
            
            # Atomic rename
            os.replace(temp_file, self.jobs_file)
            logger.debug(f"Saved {len(self.jobs)} optimization jobs to {self.jobs_file}")
        except Exception as e:
            logger.error(f"Error saving optimization jobs: {str(e)}")
    
    def add_job(self, job: OptimizationJob) -> None:
        """
        Add a new job to the store.
        
        Args:
            job: Job to add
        """
        with self.lock:
            self.jobs[job.job_id] = job
        self._save_jobs()
    
    def get_job(self, job_id: str) -> Optional[OptimizationJob]:
        """
        Retrieve a job by ID.
        
        Args:
            job_id: ID of job to retrieve
            
        Returns:
            Job if found, None otherwise
        """
        with self.lock:
            return self.jobs.get(job_id)
    
    def update_job(self, job: OptimizationJob) -> None:
        """
        Update an existing job.
        
        Args:
            job: Updated job
        """
        with self.lock:
            if job.job_id in self.jobs:
                self.jobs[job.job_id] = job
                job.last_updated = datetime.now()
        self._save_jobs()
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the store.
        
        Args:
            job_id: ID of job to remove
            
        Returns:
            True if job was removed, False otherwise
        """
        with self.lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                self._save_jobs()
                return True
        return False
    
    def get_all_jobs(self) -> List[OptimizationJob]:
        """
        Get all jobs in the store.
        
        Returns:
            List of all jobs
        """
        with self.lock:
            return list(self.jobs.values())
    
    def get_jobs_by_status(self, status: OptimizationStatus) -> List[OptimizationJob]:
        """
        Get jobs with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of matching jobs
        """
        with self.lock:
            return [job for job in self.jobs.values() if job.status == status]
    
    def get_jobs_by_strategy(self, strategy_id: str) -> List[OptimizationJob]:
        """
        Get jobs for a specific strategy.
        
        Args:
            strategy_id: Strategy ID to filter by
            
        Returns:
            List of matching jobs
        """
        with self.lock:
            return [job for job in self.jobs.values() if job.strategy_id == strategy_id]
    
    def get_scheduled_jobs(self) -> List[OptimizationJob]:
        """
        Get all scheduled jobs sorted by priority.
        
        Returns:
            List of scheduled jobs in priority order
        """
        with self.lock:
            scheduled_jobs = [
                job for job in self.jobs.values() 
                if job.status == OptimizationStatus.SCHEDULED
            ]
        
        # Sort by priority (using job comparison)
        return sorted(scheduled_jobs)
    
    def get_pending_jobs(self) -> List[OptimizationJob]:
        """
        Get all pending jobs (not yet scheduled).
        
        Returns:
            List of pending jobs
        """
        with self.lock:
            return [
                job for job in self.jobs.values() 
                if job.status == OptimizationStatus.PENDING
            ]
    
    def get_due_jobs(self) -> List[OptimizationJob]:
        """
        Get jobs that are due for execution.
        
        Returns:
            List of due jobs in priority order
        """
        with self.lock:
            due_jobs = [job for job in self.jobs.values() if job.is_due()]
        
        # Sort by priority
        return sorted(due_jobs)
    
    def clean_old_jobs(self, days: int = 30) -> int:
        """
        Remove completed or failed jobs older than specified days.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of jobs removed
        """
        threshold = datetime.now() - timedelta(days=days)
        removed = 0
        
        with self.lock:
            job_ids_to_remove = []
            
            for job_id, job in self.jobs.items():
                if (job.status in (OptimizationStatus.COMPLETED, OptimizationStatus.FAILED) 
                    and job.last_updated < threshold):
                    job_ids_to_remove.append(job_id)
            
            for job_id in job_ids_to_remove:
                del self.jobs[job_id]
                removed += 1
        
        if removed > 0:
            self._save_jobs()
            logger.info(f"Cleaned {removed} old optimization jobs")
        
        return removed


# Singleton instance
_job_store = None


def get_job_store() -> OptimizationJobStore:
    """
    Get the singleton instance of OptimizationJobStore.
    
    Returns:
        OptimizationJobStore instance
    """
    global _job_store
    
    if _job_store is None:
        _job_store = OptimizationJobStore()
    
    return _job_store


def create_optimization_job(
    strategy_id: str,
    strategy_type: str,
    universe: str,
    version_id: str,
    performance_metrics: Dict[str, float],
    optimization_history: Optional[List[Dict[str, Any]]] = None,
    last_optimization: Optional[datetime] = None,
    scheduled_time: Optional[datetime] = None,
    parameters: Optional[Dict[str, Any]] = None,
    reason: str = "auto",
    priority: Optional[float] = None
) -> OptimizationJob:
    """
    Factory function to create a new optimization job with calculated priority.
    
    Args:
        strategy_id: Strategy identifier
        strategy_type: Type of strategy
        universe: Trading universe
        version_id: Version ID of the strategy
        performance_metrics: Current performance metrics
        optimization_history: Previous optimization attempts
        last_optimization: When strategy was last optimized
        scheduled_time: When to run the optimization (default: now + 1 hour)
        parameters: Optimization parameters
        reason: Why this job was scheduled
        priority: Override calculated priority if provided
        
    Returns:
        New OptimizationJob instance
    """
    # Calculate priority if not provided
    if priority is None:
        optimization_history = optimization_history or []
        priority = PriorityCalculator.calculate_priority(
            strategy_id,
            strategy_type,
            universe,
            performance_metrics,
            optimization_history,
            last_optimization
        )
    
    # Set default scheduled time if not provided
    if scheduled_time is None:
        scheduled_time = datetime.now() + timedelta(hours=1)
    
    # Create job
    job = OptimizationJob(
        strategy_id=strategy_id,
        version_id=version_id,
        scheduled_time=scheduled_time,
        priority=priority,
        parameters=parameters,
        reason=reason
    )
    
    # Add to store
    job_store = get_job_store()
    job_store.add_job(job)
    
    logger.info(f"Created optimization job {job.job_id} for {strategy_id} with priority {priority}")
    return job


if __name__ == "__main__":
    # Example usage
    job = create_optimization_job(
        strategy_id="iron_condor_spy",
        strategy_type="iron_condor",
        universe="options",
        version_id="v1.2.3",
        performance_metrics={
            "sharpe_ratio": 0.8,
            "max_drawdown": -0.12,
            "win_rate": 0.65
        }
    )
    
    print(f"Created job with ID {job.job_id} and priority {job.priority}")
    print(f"Job will run at {job.scheduled_time}")
