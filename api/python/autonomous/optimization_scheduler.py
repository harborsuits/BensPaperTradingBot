#!/usr/bin/env python3
"""
Optimization Scheduler

This module provides the main scheduler functionality for strategy optimization.
It handles scheduling, prioritizing, and executing optimization jobs based on
strategy performance and system resources.

Classes:
    OptimizationScheduler: Main class that manages and schedules optimization jobs
"""

import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
import heapq

# Import event system
from trading_bot.event_system import EventBus, EventManager, Event, EventType

# Import job management
from trading_bot.autonomous.optimization_jobs import (
    OptimizationJob, OptimizationStatus, get_job_store, create_optimization_job
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define optimization-related event types
class OptimizationEventType(str, Enum):
    """Event types related to optimization scheduling."""
    OPTIMIZATION_SCHEDULED = "optimization_scheduled"
    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    OPTIMIZATION_FAILED = "optimization_failed"
    OPTIMIZATION_CANCELLED = "optimization_cancelled"


class ResourceMonitor:
    """
    Monitors system resources to determine optimization capacity.
    """
    
    def __init__(self, max_concurrent_jobs: int = 2):
        """
        Initialize the resource monitor.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent optimization jobs
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.active_jobs: Set[str] = set()
        self.lock = threading.RLock()
    
    def can_start_job(self) -> bool:
        """
        Check if system has capacity to start a new optimization job.
        
        Returns:
            True if a new job can be started, False otherwise
        """
        with self.lock:
            return len(self.active_jobs) < self.max_concurrent_jobs
    
    def register_job(self, job_id: str) -> bool:
        """
        Register a job as active.
        
        Args:
            job_id: ID of job to register
            
        Returns:
            True if job was registered, False if at capacity
        """
        with self.lock:
            if len(self.active_jobs) < self.max_concurrent_jobs:
                self.active_jobs.add(job_id)
                return True
            return False
    
    def unregister_job(self, job_id: str) -> None:
        """
        Unregister a job from active set.
        
        Args:
            job_id: ID of job to unregister
        """
        with self.lock:
            self.active_jobs.discard(job_id)
    
    def get_active_job_count(self) -> int:
        """
        Get the number of active jobs.
        
        Returns:
            Count of active jobs
        """
        with self.lock:
            return len(self.active_jobs)


class OptimizationScheduler:
    """
    Manages and schedules strategy optimization jobs.
    
    This class is responsible for:
    - Scheduling optimization jobs based on strategy performance
    - Prioritizing jobs based on various metrics
    - Executing jobs when resources are available
    - Tracking optimization history and effectiveness
    - Emitting events for optimization lifecycle
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the optimization scheduler.
        
        Args:
            config: Configuration settings
        """
        self.config = config or {
            "max_concurrent_jobs": 2,
            "check_interval_seconds": 60,
            "retry_limit": 3,
            "job_timeout_hours": 4,
            "auto_schedule": True
        }
        
        # Initialize components
        self.job_store = get_job_store()
        self.resource_monitor = ResourceMonitor(
            max_concurrent_jobs=self.config["max_concurrent_jobs"]
        )
        self.event_bus = EventBus()
        
        # Thread control
        self.running = False
        self.scheduler_thread = None
        self.lock = threading.RLock()
        
        # Job execution callbacks
        self.execution_callbacks: Dict[str, Callable[[OptimizationJob], None]] = {}
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events."""
        # Strategy performance update events
        self.event_bus.register(
            EventType.STRATEGY_PERFORMANCE_UPDATED,
            self._handle_performance_updated
        )
        
        # Strategy deployment events
        self.event_bus.register(
            EventType.STRATEGY_DEPLOYED, 
            self._handle_strategy_deployed
        )
        
        # Optimization result events
        self.event_bus.register(
            OptimizationEventType.OPTIMIZATION_COMPLETED,
            self._handle_optimization_completed
        )
        
        self.event_bus.register(
            OptimizationEventType.OPTIMIZATION_FAILED,
            self._handle_optimization_failed
        )
    
    def _handle_performance_updated(self, event: Event) -> None:
        """
        Handle strategy performance update events.
        
        Args:
            event: Performance update event
        """
        if not self.config["auto_schedule"]:
            return
            
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        performance = data.get("performance", {})
        
        if not strategy_id or not performance:
            return
            
        # Check if strategy needs optimization
        if self._should_schedule_optimization(strategy_id, performance):
            version_id = data.get("version_id", "current")
            universe = data.get("universe", "unknown")
            strategy_type = data.get("strategy_type", "unknown")
            
            # Schedule optimization
            self.schedule_optimization(
                strategy_id=strategy_id,
                strategy_type=strategy_type,
                universe=universe,
                version_id=version_id,
                performance_metrics=performance,
                reason="performance_update"
            )
    
    def _handle_strategy_deployed(self, event: Event) -> None:
        """
        Handle strategy deployment events.
        
        Args:
            event: Strategy deployment event
        """
        data = event.data
        if not data:
            return
            
        strategy_id = data.get("strategy_id")
        version_id = data.get("version_id")
        
        if not strategy_id or not version_id:
            return
            
        # Schedule initial optimization job for the future
        # This helps establish a baseline for the newly deployed strategy
        scheduled_time = datetime.now() + timedelta(days=1)
        
        self.schedule_optimization(
            strategy_id=strategy_id,
            strategy_type=data.get("strategy_type", "unknown"),
            universe=data.get("universe", "unknown"),
            version_id=version_id,
            performance_metrics=data.get("performance", {}),
            reason="initial_deployment",
            scheduled_time=scheduled_time
        )
    
    def _handle_optimization_completed(self, event: Event) -> None:
        """
        Handle optimization completion events.
        
        Args:
            event: Optimization complete event
        """
        data = event.data
        if not data:
            return
            
        job_id = data.get("job_id")
        if not job_id:
            return
            
        # Update job status
        job = self.job_store.get_job(job_id)
        if job:
            job.set_results(data.get("results", {}))
            self.job_store.update_job(job)
            
        # Free up resources
        self.resource_monitor.unregister_job(job_id)
    
    def _handle_optimization_failed(self, event: Event) -> None:
        """
        Handle optimization failure events.
        
        Args:
            event: Optimization failed event
        """
        data = event.data
        if not data:
            return
            
        job_id = data.get("job_id")
        if not job_id:
            return
            
        # Update job status
        job = self.job_store.get_job(job_id)
        if job:
            job.set_error(data.get("error", "Unknown error"))
            self.job_store.update_job(job)
            
        # Free up resources
        self.resource_monitor.unregister_job(job_id)
    
    def _should_schedule_optimization(
        self, 
        strategy_id: str, 
        performance: Dict[str, Any]
    ) -> bool:
        """
        Determine if a strategy should be scheduled for optimization.
        
        Args:
            strategy_id: Strategy identifier
            performance: Performance metrics
            
        Returns:
            True if optimization should be scheduled
        """
        # Check if strategy already has pending or scheduled jobs
        existing_jobs = self.job_store.get_jobs_by_strategy(strategy_id)
        active_jobs = [
            j for j in existing_jobs 
            if j.status in (OptimizationStatus.PENDING, OptimizationStatus.SCHEDULED, OptimizationStatus.RUNNING)
        ]
        
        if active_jobs:
            return False
            
        # Check performance thresholds
        if "sharpe_ratio" in performance and performance["sharpe_ratio"] < 0.8:
            return True
            
        if "max_drawdown" in performance and abs(performance["max_drawdown"]) > 0.15:
            return True
            
        if "win_rate" in performance and performance["win_rate"] < 0.55:
            return True
        
        # Check time since last optimization
        completed_jobs = [
            j for j in existing_jobs 
            if j.status == OptimizationStatus.COMPLETED
        ]
        
        if not completed_jobs:
            # Never optimized
            return True
            
        last_optimization = max(j.last_updated for j in completed_jobs)
        days_since = (datetime.now() - last_optimization).days
        
        # Schedule optimization if it's been more than 14 days
        return days_since > 14
    
    def schedule_optimization(
        self,
        strategy_id: str,
        strategy_type: str,
        universe: str,
        version_id: str,
        performance_metrics: Dict[str, float],
        reason: str = "manual",
        scheduled_time: Optional[datetime] = None,
        parameters: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None
    ) -> OptimizationJob:
        """
        Schedule a strategy for optimization.
        
        Args:
            strategy_id: Strategy identifier
            strategy_type: Type of strategy
            universe: Trading universe
            version_id: Version ID of the strategy
            performance_metrics: Current performance metrics
            reason: Why this job was scheduled
            scheduled_time: When to run the optimization
            parameters: Optimization parameters
            priority: Override calculated priority
            
        Returns:
            The created optimization job
        """
        # Get optimization history
        optimization_history = self._get_optimization_history(strategy_id)
        
        # Get last optimization time
        last_optimization = None
        if optimization_history:
            last_opt = optimization_history[-1]
            last_optimization = datetime.fromisoformat(last_opt.get("timestamp", ""))
        
        # Create job
        job = create_optimization_job(
            strategy_id=strategy_id,
            strategy_type=strategy_type,
            universe=universe,
            version_id=version_id,
            performance_metrics=performance_metrics,
            optimization_history=optimization_history,
            last_optimization=last_optimization,
            scheduled_time=scheduled_time,
            parameters=parameters,
            reason=reason,
            priority=priority
        )
        
        # Update job status
        job.update_status(OptimizationStatus.SCHEDULED)
        self.job_store.update_job(job)
        
        # Emit event
        self.event_bus.emit(
            OptimizationEventType.OPTIMIZATION_SCHEDULED,
            {
                "job_id": job.job_id,
                "strategy_id": strategy_id,
                "version_id": version_id,
                "scheduled_time": job.scheduled_time.isoformat(),
                "priority": job.priority,
                "reason": reason
            }
        )
        
        logger.info(f"Scheduled optimization for {strategy_id} with priority {job.priority}")
        return job
    
    def _get_optimization_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        Get optimization history for a strategy.
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            List of optimization history entries
        """
        # This will be implemented in optimization_history.py
        # For now, return an empty list
        return []
    
    def start(self) -> None:
        """Start the scheduler."""
        with self.lock:
            if self.running:
                logger.warning("Scheduler is already running")
                return
                
            self.running = True
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                daemon=True,
                name="OptimizationScheduler"
            )
            self.scheduler_thread.start()
            
        logger.info("Optimization scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        with self.lock:
            self.running = False
            
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5.0)
            
        logger.info("Optimization scheduler stopped")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop that processes jobs."""
        while self.running:
            try:
                # Process due jobs
                self._process_due_jobs()
                
                # Check for timed out jobs
                self._check_job_timeouts()
                
                # Clean up old jobs periodically
                if datetime.now().hour == 2 and datetime.now().minute < 5:
                    self.job_store.clean_old_jobs()
                    
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                
            # Sleep until next check
            time.sleep(self.config["check_interval_seconds"])
    
    def _process_due_jobs(self) -> None:
        """Process jobs that are due for execution."""
        # Get jobs that are due
        due_jobs = self.job_store.get_due_jobs()
        
        if not due_jobs:
            return
            
        # Try to start jobs if resources are available
        for job in due_jobs:
            if not self.resource_monitor.can_start_job():
                # No resources available, try again later
                break
                
            if self._start_job(job):
                logger.info(f"Started optimization job {job.job_id} for {job.strategy_id}")
    
    def _start_job(self, job: OptimizationJob) -> bool:
        """
        Start executing an optimization job.
        
        Args:
            job: Job to start
            
        Returns:
            True if job was started, False otherwise
        """
        # Try to register job with resource monitor
        if not self.resource_monitor.register_job(job.job_id):
            return False
            
        # Update job status
        job.update_status(OptimizationStatus.RUNNING)
        self.job_store.update_job(job)
        
        # Emit event
        self.event_bus.emit(
            OptimizationEventType.OPTIMIZATION_STARTED,
            {
                "job_id": job.job_id,
                "strategy_id": job.strategy_id,
                "version_id": job.version_id,
                "parameters": job.parameters
            }
        )
        
        # Execute job
        self._execute_job(job)
        return True
    
    def _execute_job(self, job: OptimizationJob) -> None:
        """
        Execute an optimization job.
        
        Args:
            job: Job to execute
        """
        # Execute in a separate thread to avoid blocking scheduler
        thread = threading.Thread(
            target=self._job_execution_thread,
            args=(job,),
            daemon=True,
            name=f"OptJob-{job.job_id[:8]}"
        )
        thread.start()
    
    def _job_execution_thread(self, job: OptimizationJob) -> None:
        """
        Thread function for job execution.
        
        Args:
            job: Job to execute
        """
        try:
            # Get the appropriate callback
            callback = self.execution_callbacks.get("default")
            
            if callback:
                callback(job)
            else:
                # No callback registered, mark as failed
                self.event_bus.emit(
                    OptimizationEventType.OPTIMIZATION_FAILED,
                    {
                        "job_id": job.job_id,
                        "strategy_id": job.strategy_id,
                        "error": "No execution callback registered"
                    }
                )
        except Exception as e:
            # Handle execution error
            logger.error(f"Error executing job {job.job_id}: {str(e)}")
            
            self.event_bus.emit(
                OptimizationEventType.OPTIMIZATION_FAILED,
                {
                    "job_id": job.job_id,
                    "strategy_id": job.strategy_id,
                    "error": str(e)
                }
            )
    
    def _check_job_timeouts(self) -> None:
        """Check for and handle timed out jobs."""
        timeout_threshold = datetime.now() - timedelta(hours=self.config["job_timeout_hours"])
        
        # Get running jobs
        running_jobs = self.job_store.get_jobs_by_status(OptimizationStatus.RUNNING)
        
        for job in running_jobs:
            if job.last_updated < timeout_threshold:
                # Job has timed out
                logger.warning(f"Job {job.job_id} timed out after {self.config['job_timeout_hours']} hours")
                
                # Mark as failed
                job.set_error(f"Timed out after {self.config['job_timeout_hours']} hours")
                self.job_store.update_job(job)
                
                # Free up resources
                self.resource_monitor.unregister_job(job.job_id)
                
                # Emit event
                self.event_bus.emit(
                    OptimizationEventType.OPTIMIZATION_FAILED,
                    {
                        "job_id": job.job_id,
                        "strategy_id": job.strategy_id,
                        "error": f"Timed out after {self.config['job_timeout_hours']} hours"
                    }
                )
    
    def register_execution_callback(
        self, 
        callback: Callable[[OptimizationJob], None],
        name: str = "default"
    ) -> None:
        """
        Register a callback for job execution.
        
        Args:
            callback: Function to call for executing jobs
            name: Name of the callback
        """
        self.execution_callbacks[name] = callback
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        Get the current status of the scheduler.
        
        Returns:
            Status information
        """
        job_store = get_job_store()
        
        return {
            "running": self.running,
            "active_jobs": self.resource_monitor.get_active_job_count(),
            "max_concurrent_jobs": self.config["max_concurrent_jobs"],
            "pending_jobs": len(job_store.get_pending_jobs()),
            "scheduled_jobs": len(job_store.get_scheduled_jobs()),
            "running_jobs": len(job_store.get_jobs_by_status(OptimizationStatus.RUNNING)),
            "completed_jobs": len(job_store.get_jobs_by_status(OptimizationStatus.COMPLETED)),
            "failed_jobs": len(job_store.get_jobs_by_status(OptimizationStatus.FAILED))
        }


# Singleton instance
_scheduler = None


def get_optimization_scheduler() -> OptimizationScheduler:
    """
    Get the singleton instance of OptimizationScheduler.
    
    Returns:
        OptimizationScheduler instance
    """
    global _scheduler
    
    if _scheduler is None:
        _scheduler = OptimizationScheduler()
    
    return _scheduler


if __name__ == "__main__":
    # Start scheduler when run directly
    scheduler = get_optimization_scheduler()
    scheduler.start()
    
    try:
        while True:
            time.sleep(60)
            status = scheduler.get_scheduler_status()
            print(f"Scheduler status: {status}")
    except KeyboardInterrupt:
        print("Stopping scheduler...")
        scheduler.stop()
